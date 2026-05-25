import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from backtest import (
    BacktestParams,
    evaluate_history,
    fit_parameters,
    load_matches,
    parse_match_html,
    parse_results_html,
    rolling_validation_windows,
)
from config import ModelParams


def team(name: str, seed: int, valve: float, hltv: float) -> dict:
    return {
        "name": name,
        "seed": seed,
        "valve": valve,
        "hltv": hltv,
        "map_stats": {
            "Dust2": {
                "win_rate": 0.5,
                "pick_rate": 0.0,
                "ban_rate": 0.0,
                "maps_played": 10,
            }
        },
    }


def sample_matches() -> list[dict]:
    return [
        {
            "id": "m1",
            "event": "Fixture",
            "date": "2026-01-01",
            "format": "bo1",
            "team_a": team("A", 1, 2000, 1000),
            "team_b": team("B", 2, 1000, 2000),
            "winner": "A",
        },
        {
            "id": "m2",
            "event": "Fixture",
            "date": "2026-01-02",
            "format": "bo3",
            "team_a": team("C", 1, 1900, 1000),
            "team_b": team("D", 2, 1000, 1900),
            "winner": "C",
        },
    ]


def write_history(path: Path, matches: list[dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "matches.jsonl").open("w", encoding="utf-8") as file:
        for match in matches:
            file.write(json.dumps(match) + "\n")


class BacktestParserTests(unittest.TestCase):
    def test_parse_results_html_extracts_unique_match_links(self) -> None:
        html = (
            '<a href="/matches/1/team-a-vs-team-b">one</a>'
            '<a href="/matches/1/team-a-vs-team-b">duplicate</a>'
            '<a href="/matches/2/team-c-vs-team-d">two</a>'
        )

        matches = parse_results_html(html)

        self.assertEqual([match["id"] for match in matches], ["1", "2"])

    def test_parse_match_html_extracts_minimal_match(self) -> None:
        html = """
        <html><head><title>A vs B at Event</title></head>
        <body>
          <div class="teamName">A</div>
          <div class="teamName">B</div>
        </body></html>
        """

        match = parse_match_html(html, "1", "https://example.test")

        self.assertIsNotNone(match)
        self.assertEqual(match["team_a"]["name"], "A")
        self.assertEqual(match["team_b"]["name"], "B")


class BacktestEvaluationTests(unittest.TestCase):
    def test_evaluate_history_returns_match_metrics(self) -> None:
        params = BacktestParams(model_params=ModelParams(veto_temperature=0))

        result = evaluate_history(sample_matches(), params, veto_samples=1, swiss_iterations=1)

        self.assertEqual(result["match"]["matches"], 2)
        self.assertGreater(result["match"]["accuracy"], 0.5)

    def test_fit_parameters_prefers_valve_signal_on_synthetic_history(self) -> None:
        result = fit_parameters(sample_matches(), veto_samples=1)

        self.assertGreaterEqual(result["params"]["weights"]["valve"], 0.5)
        self.assertLess(result["best_log_loss"], 0.7)
        self.assertEqual(result["validation"]["mode"], "all_matches")

    def test_rolling_validation_windows_use_90_30_30_schedule(self) -> None:
        matches = []
        for index, match_date in enumerate(("2026-01-01", "2026-01-10", "2026-04-02", "2026-04-10")):
            match = sample_matches()[0].copy()
            match["id"] = f"m{index}"
            match["date"] = match_date
            matches.append(match)

        windows = rolling_validation_windows(matches)

        self.assertEqual(len(windows), 1)
        self.assertEqual(len(windows[0][0]), 2)
        self.assertEqual(len(windows[0][1]), 2)

    def test_backtest_evaluate_cli_reads_fixture_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            history_dir = Path(temp_dir) / "history"
            write_history(history_dir, sample_matches())

            completed = subprocess.run(
                [
                    sys.executable,
                    "backtest.py",
                    "evaluate",
                    "--history",
                    str(history_dir),
                    "--veto-samples",
                    "1",
                    "--swiss-iterations",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        output = json.loads(completed.stdout)
        self.assertEqual(output["match"]["matches"], 2)

    def test_load_matches_accepts_jsonl_file_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = Path(temp_dir) / "matches.jsonl"
            write_history(Path(temp_dir), sample_matches())

            matches = load_matches(history_path)

        self.assertEqual(len(matches), 2)


if __name__ == "__main__":
    unittest.main()
