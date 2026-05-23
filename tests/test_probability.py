import random
import unittest

from config import Team, win_probability
from simulate import Simulation, bo3_match_probability


class WinProbabilityTests(unittest.TestCase):
    def test_equal_ratings_are_even(self) -> None:
        team_a = Team(id=1, name="A", seed=1, rating=(1500.0, 500.0))
        team_b = Team(id=2, name="B", seed=2, rating=(1500.0, 500.0))

        probability = win_probability(team_a, team_b, (600.0, 1600.0), weights=(0.5, 0.5))

        self.assertAlmostEqual(probability, 0.5)

    def test_higher_rated_team_is_favored(self) -> None:
        strong = Team(id=1, name="Strong", seed=1, rating=(1700.0, 700.0))
        weak = Team(id=2, name="Weak", seed=2, rating=(1500.0, 500.0))

        probability = win_probability(strong, weak, (600.0, 1600.0), weights=(0.5, 0.5))

        self.assertGreater(probability, 0.5)

    def test_pair_probabilities_are_symmetric(self) -> None:
        team_a = Team(id=1, name="A", seed=1, rating=(1700.0, 700.0))
        team_b = Team(id=2, name="B", seed=2, rating=(1500.0, 500.0))

        probability_a = win_probability(team_a, team_b, (600.0, 1600.0), weights=(0.5, 0.5))
        probability_b = win_probability(team_b, team_a, (600.0, 1600.0), weights=(0.5, 0.5))

        self.assertAlmostEqual(probability_a + probability_b, 1.0)


class Bo3ProbabilityTests(unittest.TestCase):
    def test_even_map_probability_stays_even(self) -> None:
        self.assertEqual(bo3_match_probability(0.5), 0.5)

    def test_bo3_amplifies_favorite(self) -> None:
        self.assertGreater(bo3_match_probability(0.6), 0.6)

    def test_bo3_penalizes_underdog(self) -> None:
        self.assertLess(bo3_match_probability(0.4), 0.4)

    def test_invalid_map_probability_raises(self) -> None:
        with self.assertRaises(ValueError):
            bo3_match_probability(1.1)


class RandomnessTests(unittest.TestCase):
    def test_simulation_uses_local_rng(self) -> None:
        # 模拟器应使用自己的 Random 实例，不污染调用方的全局 random 状态。
        random.seed(12345)
        before = random.getstate()

        Simulation("major_stage.json").run(5, 1, seed=42)

        self.assertEqual(random.getstate(), before)

    def test_fixed_seed_single_worker_is_reproducible(self) -> None:
        simulation = Simulation("major_stage.json")

        first = simulation.run(20, 1, seed=42)
        second = simulation.run(20, 1, seed=42)

        self.assertEqual(first.combination_counts, second.combination_counts)
        self.assertEqual(
            {
                team.name: (
                    result.three_zero,
                    result.advanced,
                    result.zero_three,
                )
                for team, result in first.team_results.items()
            },
            {
                team.name: (
                    result.three_zero,
                    result.advanced,
                    result.zero_three,
                )
                for team, result in second.team_results.items()
            },
        )


if __name__ == "__main__":
    unittest.main()
