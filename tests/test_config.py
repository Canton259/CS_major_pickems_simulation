import json
from pathlib import Path
import tempfile
import unittest

from config import load_tournament_config


class ConfigLoadingTests(unittest.TestCase):
    def test_rating_order_follows_tournament_systems_not_json_order(self) -> None:
        config_data = {
            "systems": {
                "hltv": "lambda x: x",
                "valve": "lambda x: x",
            },
            "sigma": {
                "hltv": 1600,
                "valve": 600,
            },
            "weights": {
                "hltv": 0.5,
                "valve": 0.5,
            },
            "teams": {
                "Example": {
                    "seed": 1,
                    "hltv": 123,
                    "valve": 456,
                }
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "stage.json"
            config_path.write_text(json.dumps(config_data), encoding="utf-8")

            tournament = load_tournament_config(config_path)

        self.assertEqual(tournament.systems[:2], ("valve", "hltv"))
        self.assertEqual(tournament.sigma[:2], (600.0, 1600.0))
        self.assertEqual(tournament.weights[:2], (0.5, 0.5))
        self.assertEqual(tournament.teams[0].rating[:2], (456.0, 123.0))


if __name__ == "__main__":
    unittest.main()
