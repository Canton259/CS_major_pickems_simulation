"""Map-specific win probability adjustments and Major veto simulation."""

from __future__ import annotations

import random

from config import (
    DEFAULT_MAP_POOL as CONFIG_DEFAULT_MAP_POOL,
    PROBABILITY_CEILING,
    PROBABILITY_FLOOR,
    Team,
    clamp_probability,
    win_probability,
)


DEFAULT_MAP_POOL = CONFIG_DEFAULT_MAP_POOL
MAP_ADJUSTMENT_WEIGHT = 0.20
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
) -> float:
    """Calculate team_a's map win probability with a capped map-pool adjustment."""
    base_prob = win_probability(team_a, team_b, sigma, weights=weights)
    map_diff = map_strength(team_a, map_name) - map_strength(team_b, map_name)
    adjustment = map_diff * MAP_ADJUSTMENT_WEIGHT
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


def _first_ban_map(team: Team, opponent: Team, available_maps: list[str]) -> str | None:
    """Return the first-ban priority inferred from the highest configured ban rate."""
    if not available_maps:
        return None

    selected = min(
        available_maps,
        key=lambda map_name: (
            -map_ban_rate(team, map_name),
            -_ban_score(team, opponent, map_name),
            map_name,
        ),
    )
    return selected if map_ban_rate(team, selected) > 0 else None


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
    use_first_ban: bool = False,
    shared_first_ban: str | None = None,
) -> None:
    protected_maps = set()
    if shared_first_ban and team != _lower_seed_team(team, opponent):
        protected_maps.add(shared_first_ban)

    for index in range(count):
        selected = None
        if use_first_ban and index == 0:
            first_ban = _first_ban_map(team, opponent, available_maps)
            if first_ban and first_ban not in protected_maps:
                selected = first_ban

        if selected is None:
            selected = _best_ban_by_score(available_maps, team, opponent, protected_maps)

        available_maps.remove(selected)


def _pick_map(
    available_maps: list[str],
    team: Team,
    opponent: Team,
) -> str:
    selected = min(
        available_maps,
        key=lambda map_name: (-_pick_score(team, opponent, map_name), map_name),
    )
    available_maps.remove(selected)
    return selected


def bo1_veto_maps(team_a: Team, team_b: Team, map_pool: tuple[str, ...]) -> str:
    """Simulate the Major BO1 veto flow and return the remaining map."""
    role_a, role_b = choose_team_roles(team_a, team_b)
    available_maps = _validate_veto_map_pool(map_pool)
    shared_first_ban = _shared_first_ban_map(role_a, role_b, available_maps)

    _remove_bans(
        available_maps,
        role_a,
        role_b,
        2,
        use_first_ban=True,
        shared_first_ban=shared_first_ban,
    )
    _remove_bans(
        available_maps,
        role_b,
        role_a,
        3,
        use_first_ban=True,
        shared_first_ban=shared_first_ban,
    )
    _remove_bans(available_maps, role_a, role_b, 1)

    if len(available_maps) != 1:
        raise ValueError("BO1 veto did not resolve to exactly one map")
    return available_maps[0]


def bo3_veto_maps(team_a: Team, team_b: Team, map_pool: tuple[str, ...]) -> tuple[str, str, str]:
    """Simulate the Major BO3 veto flow and return map1, map2, and decider."""
    role_a, role_b = choose_team_roles(team_a, team_b)
    available_maps = _validate_veto_map_pool(map_pool)
    shared_first_ban = _shared_first_ban_map(role_a, role_b, available_maps)

    _remove_bans(
        available_maps,
        role_a,
        role_b,
        1,
        use_first_ban=True,
        shared_first_ban=shared_first_ban,
    )
    _remove_bans(
        available_maps,
        role_b,
        role_a,
        1,
        use_first_ban=True,
        shared_first_ban=shared_first_ban,
    )
    map1 = _pick_map(available_maps, role_a, role_b)
    map2 = _pick_map(available_maps, role_b, role_a)
    _remove_bans(available_maps, role_b, role_a, 1)
    _remove_bans(available_maps, role_a, role_b, 1)

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
) -> bool:
    """Simulate a BO1 after veto and return whether the input team_a wins."""
    map_name = bo1_veto_maps(team_a, team_b, map_pool)
    probability = map_win_probability(team_a, team_b, map_name, sigma, weights)
    return probability > rng.random()


def simulate_bo3_with_veto(
    team_a: Team,
    team_b: Team,
    map_pool: tuple[str, ...],
    sigma: tuple[float, ...],
    weights: tuple[float, ...] | None,
    rng: random.Random,
) -> bool:
    """Simulate a BO3 over the vetoed maps and return whether the input team_a wins."""
    team_a_maps_won = 0
    team_b_maps_won = 0

    for map_name in bo3_veto_maps(team_a, team_b, map_pool):
        probability = map_win_probability(team_a, team_b, map_name, sigma, weights)
        if probability > rng.random():
            team_a_maps_won += 1
        else:
            team_b_maps_won += 1

        if team_a_maps_won == 2:
            return True
        if team_b_maps_won == 2:
            return False

    return team_a_maps_won > team_b_maps_won
