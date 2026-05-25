import random
import unittest

from config import Team, win_probability
from maps import (
    DEFAULT_MAP_POOL,
    MAP_ADJUSTMENT_WEIGHT,
    MAX_MAP_ADJUSTMENT,
    _ban_score,
    _pick_score,
    _pick_map,
    _remove_bans,
    _shared_first_ban_map,
    bo1_veto_maps,
    bo3_veto_maps,
    map_win_probability,
    simulate_bo3_with_veto,
)


def make_team(
    team_id: int,
    name: str,
    seed: int,
    maps: dict[str, float] | None = None,
    picks: dict[str, float] | None = None,
    bans: dict[str, float] | None = None,
) -> Team:
    return Team(
        id=team_id,
        name=name,
        seed=seed,
        rating=(1500.0, 500.0),
        map_strengths=maps or {},
        map_pick_rates=picks or {},
        map_ban_rates=bans or {},
    )


class MapProbabilityTests(unittest.TestCase):
    def test_map_strength_advantage_increases_base_probability(self) -> None:
        team_a = make_team(1, "A", 1, {"Mirage": 0.65})
        team_b = make_team(2, "B", 2, {"Mirage": 0.45})
        base = win_probability(team_a, team_b, (600.0, 1600.0), weights=(0.5, 0.5))

        probability = map_win_probability(
            team_a,
            team_b,
            "Mirage",
            (600.0, 1600.0),
            (0.5, 0.5),
        )

        self.assertGreater(probability, base)

    def test_map_adjustment_is_capped(self) -> None:
        team_a = make_team(1, "A", 1, {"Nuke": 1.0})
        team_b = make_team(2, "B", 2, {"Nuke": 0.0})
        base = win_probability(team_a, team_b, (600.0, 1600.0), weights=(0.5, 0.5))

        probability = map_win_probability(
            team_a,
            team_b,
            "Nuke",
            (600.0, 1600.0),
            (0.5, 0.5),
        )

        self.assertGreater(1.0 * MAP_ADJUSTMENT_WEIGHT, MAX_MAP_ADJUSTMENT)
        self.assertAlmostEqual(probability - base, MAX_MAP_ADJUSTMENT)


class VetoTests(unittest.TestCase):
    def test_first_ban_uses_highest_ban_rate_before_ban_score(self) -> None:
        team_a = make_team(1, "A", 1, bans={"Ancient": 0.7})
        team_b = make_team(2, "B", 2, picks={"Dust2": 0.9})
        available_maps = list(DEFAULT_MAP_POOL)

        _remove_bans(available_maps, team_a, team_b, 1, use_first_ban=True)

        self.assertNotIn("Ancient", available_maps)
        self.assertIn("Dust2", available_maps)

    def test_shared_first_ban_is_reserved_for_lower_seed_team(self) -> None:
        high_seed = make_team(1, "High", 1, bans={"Mirage": 0.8})
        low_seed = make_team(2, "Low", 2, bans={"Mirage": 0.7})
        available_maps = list(DEFAULT_MAP_POOL)
        shared_first_ban = _shared_first_ban_map(high_seed, low_seed, available_maps)

        _remove_bans(
            available_maps,
            high_seed,
            low_seed,
            1,
            use_first_ban=True,
            shared_first_ban=shared_first_ban,
        )
        self.assertIn("Mirage", available_maps)

        _remove_bans(
            available_maps,
            low_seed,
            high_seed,
            1,
            use_first_ban=True,
            shared_first_ban=shared_first_ban,
        )
        self.assertNotIn("Mirage", available_maps)

    def test_zero_first_ban_falls_back_to_ban_score(self) -> None:
        team_a = make_team(1, "A", 1, bans={"Ancient": 0.0})
        team_b = make_team(2, "B", 2, picks={"Nuke": 0.9})
        available_maps = list(DEFAULT_MAP_POOL)

        _remove_bans(available_maps, team_a, team_b, 1, use_first_ban=True)

        self.assertNotIn("Nuke", available_maps)
        self.assertIn("Ancient", available_maps)

    def test_equal_opening_ban_rate_uses_ban_score_before_map_name(self) -> None:
        team_a = make_team(1, "A", 1, bans={"Dust2": 0.8, "Nuke": 0.8})
        team_b = make_team(2, "B", 2, picks={"Nuke": 0.9})
        available_maps = ["Dust2", "Nuke", "Mirage"]

        _remove_bans(available_maps, team_a, team_b, 1, use_first_ban=True)

        self.assertNotIn("Nuke", available_maps)
        self.assertIn("Dust2", available_maps)

    def test_equal_opening_ban_rate_and_score_uses_map_name(self) -> None:
        team_a = make_team(1, "A", 1, bans={"Dust2": 0.8, "Mirage": 0.8})
        team_b = make_team(2, "B", 2)
        available_maps = ["Mirage", "Dust2", "Inferno"]

        _remove_bans(available_maps, team_a, team_b, 1, use_first_ban=True)

        self.assertNotIn("Dust2", available_maps)
        self.assertIn("Mirage", available_maps)

    def test_first_pick_is_not_special_cased(self) -> None:
        team_a = make_team(1, "A", 1, picks={"Dust2": 0.9})
        team_b = make_team(2, "B", 2)
        available_maps = ["Dust2", "Ancient"]

        selected = _pick_map(available_maps, team_a, team_b)

        self.assertEqual(selected, "Dust2")

    def test_ban_score_uses_own_ban_rate(self) -> None:
        team_a = make_team(1, "A", 1, bans={"Dust2": 0.9})
        team_b = make_team(2, "B", 2)

        self.assertGreater(
            _ban_score(team_a, team_b, "Dust2"),
            _ban_score(team_a, team_b, "Nuke"),
        )

    def test_ban_score_uses_opponent_pick_rate(self) -> None:
        team_a = make_team(1, "A", 1)
        team_b = make_team(2, "B", 2, picks={"Mirage": 0.9})

        self.assertGreater(
            _ban_score(team_a, team_b, "Mirage"),
            _ban_score(team_a, team_b, "Nuke"),
        )

    def test_pick_score_uses_own_pick_rate(self) -> None:
        team_a = make_team(1, "A", 1, picks={"Mirage": 0.9})
        team_b = make_team(2, "B", 2)

        self.assertGreater(
            _pick_score(team_a, team_b, "Mirage"),
            _pick_score(team_a, team_b, "Nuke"),
        )

    def test_bo1_veto_returns_one_map_from_pool(self) -> None:
        team_a = make_team(1, "A", 1, {"Mirage": 0.7, "Nuke": 0.2})
        team_b = make_team(2, "B", 2, {"Mirage": 0.2, "Nuke": 0.7})

        selected_map = bo1_veto_maps(team_a, team_b, DEFAULT_MAP_POOL)

        self.assertIn(selected_map, DEFAULT_MAP_POOL)
        self.assertIsInstance(selected_map, str)

    def test_bo3_veto_returns_three_distinct_maps(self) -> None:
        team_a = make_team(1, "A", 1, {"Mirage": 0.7, "Nuke": 0.2, "Anubis": 0.6})
        team_b = make_team(2, "B", 2, {"Mirage": 0.2, "Nuke": 0.7, "Ancient": 0.6})

        maps = bo3_veto_maps(team_a, team_b, DEFAULT_MAP_POOL)

        self.assertEqual(len(maps), 3)
        self.assertEqual(len(set(maps)), 3)
        self.assertTrue(set(maps).issubset(DEFAULT_MAP_POOL))

    def test_bo3_veto_ties_are_resolved_by_map_name(self) -> None:
        team_a = make_team(1, "A", 1)
        team_b = make_team(2, "B", 2)

        maps = bo3_veto_maps(team_a, team_b, DEFAULT_MAP_POOL)

        self.assertEqual(maps, ("Dust2", "Inferno", "Overpass"))

    def test_simulation_is_reproducible_with_same_seed(self) -> None:
        team_a = make_team(1, "A", 1, {"Dust2": 0.7, "Mirage": 0.6})
        team_b = make_team(2, "B", 2, {"Dust2": 0.3, "Mirage": 0.4})

        first = simulate_bo3_with_veto(
            team_a,
            team_b,
            DEFAULT_MAP_POOL,
            (600.0, 1600.0),
            (0.5, 0.5),
            random.Random(42),
        )
        second = simulate_bo3_with_veto(
            team_a,
            team_b,
            DEFAULT_MAP_POOL,
            (600.0, 1600.0),
            (0.5, 0.5),
            random.Random(42),
        )

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
