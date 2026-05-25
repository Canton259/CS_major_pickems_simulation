"""Map-specific win probability adjustments and Major veto simulation."""

from __future__ import annotations

import random
from math import exp

from config import (
    DEFAULT_MAP_ADJUSTMENT_WEIGHT,
    DEFAULT_MAP_POOL as CONFIG_DEFAULT_MAP_POOL,
    DEFAULT_VETO_TEMPERATURE,
    ModelParams,
    PROBABILITY_CEILING,
    PROBABILITY_FLOOR,
    Team,
    clamp_probability,
    win_probability,
)


DEFAULT_MAP_POOL = CONFIG_DEFAULT_MAP_POOL
MAP_ADJUSTMENT_WEIGHT = DEFAULT_MAP_ADJUSTMENT_WEIGHT
MAX_MAP_ADJUSTMENT = 0.08
VETO_HISTORY_WEIGHT = 0.80
VETO_ADVANTAGE_WEIGHT = 0.20


def map_strength(team: Team, map_name: str) -> float:
    """Return a team's configured strength on one map, defaulting to neutral."""
    return team.map_strengths.get(map_name, 0.5)


def map_pick_rate(team: Team, map_name: str) -> float:
    """Return a team's historical HLTV pick rate on one map."""
    return team.map_pick_rates.get(map_name, 0.0)


def map_ban_rate(team: Team, map_name: str) -> float:
    """Return a team's historical HLTV ban rate on one map."""
    return team.map_ban_rates.get(map_name, 0.0)


def map_win_probability(
    team_a: Team,
    team_b: Team,
    map_name: str,
    sigma: tuple[float, ...],
    weights: tuple[float, ...] | None,
    model_params: ModelParams | None = None,
) -> float:
    """Calculate team_a's map win probability with a capped map-pool adjustment."""
    params = model_params or ModelParams()
    base_prob = win_probability(team_a, team_b, sigma, weights=weights)
    map_diff = map_strength(team_a, map_name) - map_strength(team_b, map_name)
    adjustment = map_diff * params.map_adjustment_weight
    adjustment = min(max(adjustment, -MAX_MAP_ADJUSTMENT), MAX_MAP_ADJUSTMENT)
    return clamp_probability(
        base_prob + adjustment,
        (PROBABILITY_FLOOR, PROBABILITY_CEILING),
    )


def choose_team_roles(team_a: Team, team_b: Team) -> tuple[Team, Team]:
    """Return the heuristic veto Team A/Team B roles, with higher seed choosing Team A."""
    ordered = sorted((team_a, team_b), key=lambda team: (team.seed, team.id))
    return ordered[0], ordered[1]


def _validate_veto_map_pool(map_pool: tuple[str, ...]) -> list[str]:
    maps = list(map_pool)
    if len(maps) != 7:
        raise ValueError(f"Major veto simulation requires exactly 7 maps, got {len(maps)}")
    if len(set(maps)) != len(maps):
        raise ValueError("Major veto simulation requires unique maps in map_pool")
    return maps


def _clamp_unit(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _advantage_score(strength_diff: float) -> float:
    return _clamp_unit(0.5 + strength_diff / 2)


def _ban_score(team: Team, opponent: Team, map_name: str) -> float:
    history = 0.5 * map_ban_rate(team, map_name) + 0.5 * map_pick_rate(opponent, map_name)
    advantage = _advantage_score(map_strength(opponent, map_name) - map_strength(team, map_name))
    return VETO_HISTORY_WEIGHT * history + VETO_ADVANTAGE_WEIGHT * advantage


def _pick_score(team: Team, opponent: Team, map_name: str) -> float:
    history = map_pick_rate(team, map_name)
    advantage = _advantage_score(map_strength(team, map_name) - map_strength(opponent, map_name))
    return VETO_HISTORY_WEIGHT * history + VETO_ADVANTAGE_WEIGHT * advantage


def _lower_seed_team(team_a: Team, team_b: Team) -> Team:
    return max((team_a, team_b), key=lambda team: (team.seed, team.id))


def _deterministic_choice(candidates: list[str], score_fn) -> str:
    return min(candidates, key=lambda map_name: (-score_fn(map_name), map_name))


def _softmax_choice(
    candidates: list[str],
    score_fn,
    rng: random.Random,
    temperature: float,
) -> str:
    if not candidates:
        raise ValueError("Cannot choose from an empty map candidate list")
    if temperature <= 0:
        return _deterministic_choice(candidates, score_fn)

    ordered_candidates = sorted(candidates)
    scores = [score_fn(map_name) for map_name in ordered_candidates]
    max_score = max(scores)
    weights = [exp((score - max_score) / temperature) for score in scores]
    total_weight = sum(weights)
    threshold = rng.random() * total_weight

    cumulative = 0.0
    for map_name, weight in zip(ordered_candidates, weights):
        cumulative += weight
        if threshold <= cumulative:
            return map_name
    return ordered_candidates[-1]


def _first_ban_candidates(team: Team, available_maps: list[str]) -> list[str]:
    if not available_maps:
        return []
    highest_ban_rate = max(map_ban_rate(team, map_name) for map_name in available_maps)
    if highest_ban_rate <= 0:
        return []
    return [
        map_name
        for map_name in available_maps
        if map_ban_rate(team, map_name) == highest_ban_rate
    ]


def _first_ban_map(team: Team, opponent: Team, available_maps: list[str]) -> str | None:
    """Return the first-ban priority inferred from the highest configured ban rate."""
    candidates = _first_ban_candidates(team, available_maps)
    if not candidates:
        return None
    return _deterministic_choice(candidates, lambda map_name: _ban_score(team, opponent, map_name))


def _shared_first_ban_map(team_a: Team, team_b: Team, available_maps: list[str]) -> str | None:
    team_a_first_ban = _first_ban_map(team_a, team_b, available_maps)
    team_b_first_ban = _first_ban_map(team_b, team_a, available_maps)
    if team_a_first_ban and team_a_first_ban == team_b_first_ban:
        return team_a_first_ban
    return None


def _best_ban_by_score(
    available_maps: list[str],
    team: Team,
    opponent: Team,
    protected_maps: set[str],
) -> str:
    candidates = [map_name for map_name in available_maps if map_name not in protected_maps]
    if not candidates:
        candidates = available_maps

    return min(
        candidates,
        key=lambda map_name: (-_ban_score(team, opponent, map_name), map_name),
    )


def _remove_bans(
    available_maps: list[str],
    team: Team,
    opponent: Team,
    count: int,
    rng: random.Random | None = None,
    veto_temperature: float = DEFAULT_VETO_TEMPERATURE,
    use_first_ban: bool = False,
    shared_first_ban: str | None = None,
) -> None:
    chooser_rng = rng or random.Random(0)
    protected_maps = set()
    if shared_first_ban and team != _lower_seed_team(team, opponent):
        protected_maps.add(shared_first_ban)

    for index in range(count):
        selected = None
        if use_first_ban and index == 0:
            first_ban_candidates = [
                map_name
                for map_name in _first_ban_candidates(team, available_maps)
                if map_name not in protected_maps
            ]
            if first_ban_candidates:
                selected = _softmax_choice(
                    first_ban_candidates,
                    lambda map_name: _ban_score(team, opponent, map_name),
                    chooser_rng,
                    veto_temperature,
                )

        if selected is None:
            candidates = [map_name for map_name in available_maps if map_name not in protected_maps]
            if not candidates:
                candidates = available_maps
            selected = _softmax_choice(
                candidates,
                lambda map_name: _ban_score(team, opponent, map_name),
                chooser_rng,
                veto_temperature,
            )

        available_maps.remove(selected)


def _pick_map(
    available_maps: list[str],
    team: Team,
    opponent: Team,
    rng: random.Random | None = None,
    veto_temperature: float = DEFAULT_VETO_TEMPERATURE,
) -> str:
    selected = _softmax_choice(
        available_maps,
        lambda map_name: _pick_score(team, opponent, map_name),
        rng or random.Random(0),
        veto_temperature,
    )
    available_maps.remove(selected)
    return selected


def bo1_veto_maps(
    team_a: Team,
    team_b: Team,
    map_pool: tuple[str, ...],
    rng: random.Random | None = None,
    model_params: ModelParams | None = None,
) -> str:
    """Simulate the Major BO1 veto flow and return the remaining map."""
    params = model_params or ModelParams()
    chooser_rng = rng or random.Random(0)
    role_a, role_b = choose_team_roles(team_a, team_b)
    available_maps = _validate_veto_map_pool(map_pool)
    shared_first_ban = _shared_first_ban_map(role_a, role_b, available_maps)

    _remove_bans(
        available_maps,
        role_a,
        role_b,
        2,
        chooser_rng,
        params.veto_temperature,
        use_first_ban=True,
        shared_first_ban=shared_first_ban,
    )
    _remove_bans(
        available_maps,
        role_b,
        role_a,
        3,
        chooser_rng,
        params.veto_temperature,
        use_first_ban=True,
        shared_first_ban=shared_first_ban,
    )
    _remove_bans(available_maps, role_a, role_b, 1, chooser_rng, params.veto_temperature)

    if len(available_maps) != 1:
        raise ValueError("BO1 veto did not resolve to exactly one map")
    return available_maps[0]


def bo3_veto_maps(
    team_a: Team,
    team_b: Team,
    map_pool: tuple[str, ...],
    rng: random.Random | None = None,
    model_params: ModelParams | None = None,
) -> tuple[str, str, str]:
    """Simulate the Major BO3 veto flow and return map1, map2, and decider."""
    params = model_params or ModelParams()
    chooser_rng = rng or random.Random(0)
    role_a, role_b = choose_team_roles(team_a, team_b)
    available_maps = _validate_veto_map_pool(map_pool)
    shared_first_ban = _shared_first_ban_map(role_a, role_b, available_maps)

    _remove_bans(
        available_maps,
        role_a,
        role_b,
        1,
        chooser_rng,
        params.veto_temperature,
        use_first_ban=True,
        shared_first_ban=shared_first_ban,
    )
    _remove_bans(
        available_maps,
        role_b,
        role_a,
        1,
        chooser_rng,
        params.veto_temperature,
        use_first_ban=True,
        shared_first_ban=shared_first_ban,
    )
    map1 = _pick_map(available_maps, role_a, role_b, chooser_rng, params.veto_temperature)
    map2 = _pick_map(available_maps, role_b, role_a, chooser_rng, params.veto_temperature)
    _remove_bans(available_maps, role_b, role_a, 1, chooser_rng, params.veto_temperature)
    _remove_bans(available_maps, role_a, role_b, 1, chooser_rng, params.veto_temperature)

    if len(available_maps) != 1:
        raise ValueError("BO3 veto did not resolve to exactly one decider map")
    return (map1, map2, available_maps[0])


def simulate_bo1_with_veto(
    team_a: Team,
    team_b: Team,
    map_pool: tuple[str, ...],
    sigma: tuple[float, ...],
    weights: tuple[float, ...] | None,
    rng: random.Random,
    model_params: ModelParams | None = None,
) -> bool:
    """Simulate a BO1 after veto and return whether the input team_a wins."""
    map_name = bo1_veto_maps(team_a, team_b, map_pool, rng, model_params)
    probability = map_win_probability(team_a, team_b, map_name, sigma, weights, model_params)
    return probability > rng.random()


def simulate_bo3_with_veto(
    team_a: Team,
    team_b: Team,
    map_pool: tuple[str, ...],
    sigma: tuple[float, ...],
    weights: tuple[float, ...] | None,
    rng: random.Random,
    model_params: ModelParams | None = None,
) -> bool:
    """Simulate a BO3 over the vetoed maps and return whether the input team_a wins."""
    team_a_maps_won = 0
    team_b_maps_won = 0

    for map_name in bo3_veto_maps(team_a, team_b, map_pool, rng, model_params):
        probability = map_win_probability(team_a, team_b, map_name, sigma, weights, model_params)
        if probability > rng.random():
            team_a_maps_won += 1
        else:
            team_b_maps_won += 1

        if team_a_maps_won == 2:
            return True
        if team_b_maps_won == 2:
            return False

    return team_a_maps_won > team_b_maps_won
