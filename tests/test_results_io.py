import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from greedy import parse_simulation_results
from simulate import Simulation, write_combination_results, write_team_summary


class ResultsIoTests(unittest.TestCase):
    def test_text_and_jsonl_results_are_parseable(self) -> None:
        simulation = Simulation("major_stage.json")
        summary = simulation.run(10, 1, seed=42)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            text_path = temp_path / "results.txt"
            jsonl_path = temp_path / "results.jsonl"

            write_combination_results(summary, 10, text_path)
            write_combination_results(summary, 10, jsonl_path)

            text_results, text_total = parse_simulation_results(text_path)
            jsonl_results, jsonl_total = parse_simulation_results(jsonl_path)

            self.assertEqual(text_total, 10)
            self.assertEqual(jsonl_total, 10)
            self.assertEqual(text_results, jsonl_results)

    def test_greedy_cli_accepts_text_and_jsonl_results(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        simulation = Simulation("major_stage.json")
        summary = simulation.run(10, 1, seed=42)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            text_path = temp_path / "results.txt"
            jsonl_path = temp_path / "results.jsonl"

            write_combination_results(summary, 10, text_path)
            write_combination_results(summary, 10, jsonl_path)

            for result_path in (text_path, jsonl_path):
                completed = subprocess.run(
                    [
                        sys.executable,
                        "greedy.py",
                        "--results",
                        str(result_path),
                        "--top",
                        "1",
                    ],
                    cwd=project_root,
                    text=True,
                    capture_output=True,
                    check=False,
                )

                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("候选组合排名", completed.stdout)

    def test_jsonl_output_contains_structured_records(self) -> None:
        simulation = Simulation("major_stage.json")
        summary = simulation.run(1, 1, seed=42)

        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = Path(temp_dir) / "results.jsonl"
            write_combination_results(summary, 1, jsonl_path)

            record = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])

            self.assertIn("three_zero", record)
            self.assertIn("advanced", record)
            self.assertIn("zero_three", record)
            self.assertEqual(record["count"], 1)
            self.assertEqual(record["total"], 1)

    def test_team_summary_csv_uses_config_order(self) -> None:
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
                "zero_three_count",
                "three_zero_probability",
                "advanced_probability",
                "zero_three_probability",
                "total",
            ],
        )
        self.assertEqual([row["team"] for row in rows], [team.name for team in simulation.teams])
        self.assertTrue(all(row["total"] == "10" for row in rows))


if __name__ == "__main__":
    unittest.main()
