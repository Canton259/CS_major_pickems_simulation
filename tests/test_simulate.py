import csv
from pathlib import Path
import random
import tempfile
import unittest

from config import Team
from simulate import Record, Simulation, SwissSystem, bo3_match_probability, write_team_summary


def make_test_teams(count: int) -> list[Team]:
    return [
        Team(id=index, name=f"Team{index}", seed=index, rating=(1000.0, 1000.0))
        for index in range(1, count + 1)
    ]


def make_swiss(teams: list[Team]) -> SwissSystem:
    return SwissSystem(
        sigma=(600.0, 1600.0),
        weights=(0.5, 0.5),
        records={team: Record.new() for team in teams},
        remaining=set(teams),
        rng=random.Random(1),
    )


def mark_played(swiss: SwissSystem, team_a: Team, team_b: Team) -> None:
    swiss.records[team_a].teams_faced.add(team_b)
    swiss.records[team_b].teams_faced.add(team_a)


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


class ValvePairingTests(unittest.TestCase):
    def test_initial_round_uses_valve_seed_pairings(self) -> None:
        teams = make_test_teams(16)
        swiss = make_swiss(teams)

        pairs = swiss.pair_initial_round(list(reversed(teams)))

        self.assertEqual(
            [(team_a.seed, team_b.seed) for team_a, team_b in pairs],
            [(1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15), (8, 16)],
        )

    def test_round_two_three_pairing_uses_high_seed_vs_low_seed(self) -> None:
        teams = make_test_teams(8)
        swiss = make_swiss(teams)

        pairs = swiss.pair_group(list(reversed(teams)), round_number=2)

        self.assertEqual(
            [(team_a.seed, team_b.seed) for team_a, team_b in pairs],
            [(1, 8), (2, 7), (3, 6), (4, 5)],
        )

    def test_round_two_three_pairing_avoids_rematch(self) -> None:
        teams = make_test_teams(8)
        swiss = make_swiss(teams)
        mark_played(swiss, teams[0], teams[7])

        pairs = swiss.pair_group(teams, round_number=3)

        self.assertEqual(
            [(team_a.seed, team_b.seed) for team_a, team_b in pairs],
            [(1, 7), (2, 8), (3, 6), (4, 5)],
        )

    def test_round_four_five_uses_first_valve_priority_row_without_rematches(self) -> None:
        teams = make_test_teams(6)
        swiss = make_swiss(teams)
        mark_played(swiss, teams[1], teams[4])

        pairs = swiss.pair_group(teams, round_number=4)

        self.assertEqual(
            [(team_a.seed, team_b.seed) for team_a, team_b in pairs],
            [(1, 6), (2, 4), (3, 5)],
        )

    def test_pairing_fallback_chooses_fewest_rematches_then_priority_order(self) -> None:
        teams = make_test_teams(4)
        swiss = make_swiss(teams)
        mark_played(swiss, teams[0], teams[3])
        mark_played(swiss, teams[1], teams[2])
        mark_played(swiss, teams[0], teams[2])
        mark_played(swiss, teams[0], teams[1])

        pairs = swiss.pair_group(teams, round_number=2)

        self.assertEqual(
            [(team_a.seed, team_b.seed) for team_a, team_b in pairs],
            [(1, 3), (2, 4)],
        )

    def test_round_four_five_rejects_unexpected_group_size(self) -> None:
        teams = make_test_teams(4)
        swiss = make_swiss(teams)

        with self.assertRaisesRegex(ValueError, "6 支队伍"):
            swiss.pair_group(teams, round_number=4)


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
