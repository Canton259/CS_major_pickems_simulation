import json
from pathlib import Path
import tempfile
import unittest

from greedy import parse_simulation_results
from simulate import Simulation, write_combination_results


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


if __name__ == "__main__":
    unittest.main()
