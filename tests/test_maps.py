import random
import unittest

from config import Team, win_probability
from maps import (
    DEFAULT_MAP_POOL,
    MAP_ADJUSTMENT_WEIGHT,
    MAX_MAP_ADJUSTMENT,
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
) -> Team:
    return Team(
        id=team_id,
        name=name,
        seed=seed,
        rating=(1500.0, 500.0),
        map_strengths=maps or {},
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
