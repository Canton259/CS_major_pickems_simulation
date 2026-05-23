import csv
from pathlib import Path
import tempfile
import unittest

from simulate import Simulation, bo3_match_probability, write_team_summary


class Bo3ProbabilityTests(unittest.TestCase):
    def test_even_map_probability_stays_even(self) -> None:
        self.assertEqual(bo3_match_probability(0.5), 0.5)

    def test_bo3_amplifies_favorite(self) -> None:
        self.assertGreater(bo3_match_probability(0.6), 0.6)

    def test_bo3_penalizes_underdog(self) -> None:
        self.assertLess(bo3_match_probability(0.4), 0.4)

    def test_invalid_map_probability_raises(self) -> None:
        for probability in (-0.1, 1.1):
            with self.subTest(probability=probability):
                with self.assertRaises(ValueError):
                    bo3_match_probability(probability)


class SimulationTests(unittest.TestCase):
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

    def test_write_team_summary_outputs_qualified_fields(self) -> None:
        simulation = Simulation("major_stage.json")
        summary = simulation.run(10, 1, seed=42)

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "team_summary.csv"
            write_team_summary(summary, simulation.teams, 10, csv_path)

            with csv_path.open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(
            list(rows[0].keys()),
            [
                "team",
                "three_zero_count",
                "advanced_count",
                "qualified_count",
                "zero_three_count",
                "three_zero_probability",
                "advanced_probability",
                "qualified_probability",
                "zero_three_probability",
                "total",
            ],
        )
        self.assertEqual([row["team"] for row in rows], [team.name for team in simulation.teams])
        self.assertTrue(all(row["total"] == "10" for row in rows))

        for row in rows:
            qualified_count = int(row["three_zero_count"]) + int(row["advanced_count"])
            self.assertEqual(int(row["qualified_count"]), qualified_count)
            self.assertAlmostEqual(float(row["qualified_probability"]), qualified_count / 10)


if __name__ == "__main__":
    unittest.main()
