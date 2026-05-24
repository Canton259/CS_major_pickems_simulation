import json
from pathlib import Path
import tempfile
import unittest

from greedy import (
    evaluate_candidate,
    evaluate_candidate_by_inclusion_exclusion,
    parse_simulation_results,
    validate_candidate,
)


TEXT_RESULTS = (
    "3-0: A, B | 3-1/3-2: C, D, E, F, G, H | 0-3: I, J: "
    "3/5 (60.0000%)\n"
    "3-0: A, C | 3-1/3-2: B, D, E, F, G, H | 0-3: I, J: "
    "2/5 (40.0000%)\n"
)

JSONL_RECORDS = [
    {
        "three_zero": ["A", "B"],
        "advanced": ["C", "D", "E", "F", "G", "H"],
        "zero_three": ["I", "J"],
        "count": 3,
        "total": 5,
    },
    {
        "three_zero": ["A", "C"],
        "advanced": ["B", "D", "E", "F", "G", "H"],
        "zero_three": ["I", "J"],
        "count": 2,
        "total": 5,
    },
]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


class GreedyParsingTests(unittest.TestCase):
    def test_parse_text_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            text_path = Path(temp_dir) / "result.txt"
            text_path.write_text(TEXT_RESULTS, encoding="utf-8")

            results, total = parse_simulation_results(text_path)

        self.assertEqual(total, 5)
        self.assertEqual(sum(results.values()), 5)

    def test_parse_jsonl_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = Path(temp_dir) / "result.jsonl"
            write_jsonl(jsonl_path, JSONL_RECORDS)

            results, total = parse_simulation_results(jsonl_path)

        self.assertEqual(total, 5)
        self.assertEqual(sum(results.values()), 5)

    def test_text_and_jsonl_parse_to_same_structure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            text_path = temp_path / "result.txt"
            jsonl_path = temp_path / "result.jsonl"
            text_path.write_text(TEXT_RESULTS, encoding="utf-8")
            write_jsonl(jsonl_path, JSONL_RECORDS)

            text_results, text_total = parse_simulation_results(text_path)
            jsonl_results, jsonl_total = parse_simulation_results(jsonl_path)

        self.assertEqual(text_total, jsonl_total)
        self.assertEqual(text_results, jsonl_results)

    def test_jsonl_missing_required_field_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = Path(temp_dir) / "bad.jsonl"
            write_jsonl(
                jsonl_path,
                [{"three_zero": [], "advanced": [], "count": 1}],
            )

            with self.assertRaisesRegex(ValueError, r"bad\.jsonl.*第 1 行.*zero_three"):
                parse_simulation_results(jsonl_path)

    def test_jsonl_negative_or_non_integer_count_raises_clear_error(self) -> None:
        invalid_records = [
            ({"three_zero": [], "advanced": [], "zero_three": [], "count": -1}, "非负整数"),
            ({"three_zero": [], "advanced": [], "zero_three": [], "count": 1.5}, "整数"),
        ]

        for record, message in invalid_records:
            with self.subTest(record=record):
                with tempfile.TemporaryDirectory() as temp_dir:
                    jsonl_path = Path(temp_dir) / "bad.jsonl"
                    write_jsonl(jsonl_path, [record])

                    with self.assertRaisesRegex(ValueError, rf"bad\.jsonl.*第 1 行.*count.*{message}"):
                        parse_simulation_results(jsonl_path)


class CandidateEvaluationTests(unittest.TestCase):
    def test_evaluate_candidate_counts_predictions_ge_five(self) -> None:
        candidate = {
            "3-0": {"A", "B"},
            "3-1/3-2": {"C", "D", "E", "F", "G", "H"},
            "0-3": {"I", "J"},
        }
        results = {
            (
                frozenset({"A", "B"}),
                frozenset({"C", "D", "E", "F", "G", "H"}),
                frozenset({"I", "J"}),
            ): 3,
            (
                frozenset({"K", "L"}),
                frozenset({"M", "N", "O", "P", "Q", "R"}),
                frozenset({"S", "T"}),
            ): 2,
        }

        self.assertEqual(evaluate_candidate(candidate, results), 3 / 5)

    def test_inclusion_exclusion_score_matches_reference_evaluator(self) -> None:
        candidate = {
            "3-0": {"A", "B"},
            "3-1/3-2": {"C", "D", "E", "F", "G", "H"},
            "0-3": {"I", "J"},
        }
        results = {
            (
                frozenset({"A", "B"}),
                frozenset({"C", "D", "E", "F", "G", "H"}),
                frozenset({"I", "J"}),
            ): 3,
            (
                frozenset({"A", "C"}),
                frozenset({"B", "D", "E", "F", "G", "H"}),
                frozenset({"I", "J"}),
            ): 2,
        }

        self.assertEqual(
            evaluate_candidate_by_inclusion_exclusion(candidate, results),
            evaluate_candidate(candidate, results),
        )

    def test_invalid_candidate_groups_are_rejected(self) -> None:
        invalid_candidate = {
            "3-0": {"A", "B"},
            "3-1/3-2": {"B", "C", "D", "E", "F", "G"},
            "0-3": {"I", "J"},
        }

        with self.assertRaisesRegex(ValueError, "不能重叠"):
            validate_candidate(invalid_candidate)


if __name__ == "__main__":
    unittest.main()
