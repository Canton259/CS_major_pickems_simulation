import json
from pathlib import Path
import tempfile
import unittest

from config import HLTV_WEIGHT, Team, VRS_WEIGHT, load_tournament_config, win_probability
from maps import DEFAULT_MAP_POOL


def make_config() -> dict:
    teams = {}
    for index in range(1, 17):
        teams[f"Team{index}"] = {
            "seed": index,
            "hltv": 100 + index,
            "valve": 1000 + index,
        }

    return {
        "systems": {
            "hltv": "lambda x: x",
            "valve": "lambda x: x",
        },
        "sigma": {
            "hltv": 1600,
            "valve": 600,
        },
        "teams": teams,
    }


def load_from_temp(config_data: dict):
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "stage.json"
        config_path.write_text(json.dumps(config_data), encoding="utf-8")
        return load_tournament_config(config_path)


class ConfigLoadingTests(unittest.TestCase):
    def test_rating_order_follows_tournament_systems_not_json_order(self) -> None:
        config_data = make_config()
        config_data["teams"]["Team1"]["hltv"] = 123
        config_data["teams"]["Team1"]["valve"] = 456

        tournament = load_from_temp(config_data)

        self.assertEqual(tournament.systems[:2], ("valve", "hltv"))
        self.assertEqual(tournament.sigma[:2], (600.0, 1600.0))
        self.assertEqual(tournament.teams[0].rating[:2], (456.0, 123.0))

    def test_missing_weights_fall_back_to_config_defaults(self) -> None:
        tournament = load_from_temp(make_config())

        self.assertEqual(tournament.weights[:2], (VRS_WEIGHT, HLTV_WEIGHT))

    def test_missing_map_pool_uses_default_pool(self) -> None:
        tournament = load_from_temp(make_config())

        self.assertEqual(tournament.map_pool, DEFAULT_MAP_POOL)

    def test_missing_team_maps_default_to_neutral_strengths(self) -> None:
        tournament = load_from_temp(make_config())

        self.assertEqual(tournament.teams[0].map_strengths, {map_name: 0.5 for map_name in DEFAULT_MAP_POOL})

    def test_missing_single_map_strength_defaults_to_neutral(self) -> None:
        config_data = make_config()
        config_data["map_pool"] = ["Dust2", "Mirage", "Inferno"]
        config_data["teams"]["Team1"]["maps"] = {
            "Dust2": 0.61,
            "Mirage": 0.43,
        }

        tournament = load_from_temp(config_data)

        self.assertEqual(tournament.map_pool, ("Dust2", "Mirage", "Inferno"))
        self.assertEqual(tournament.teams[0].map_strengths["Dust2"], 0.61)
        self.assertEqual(tournament.teams[0].map_strengths["Mirage"], 0.43)
        self.assertEqual(tournament.teams[0].map_strengths["Inferno"], 0.5)

    def test_invalid_map_strength_raises_clear_error(self) -> None:
        for value in (-0.01, 1.01):
            with self.subTest(value=value):
                config_data = make_config()
                config_data["teams"]["Team1"]["maps"] = {"Mirage": value}

                with self.assertRaisesRegex(ValueError, "Team1.*Mirage.*\\[0, 1\\]"):
                    load_from_temp(config_data)

    def test_weights_cannot_both_be_zero(self) -> None:
        config_data = make_config()
        config_data["weights"] = {"valve": 0, "hltv": 0}

        with self.assertRaisesRegex(ValueError, "权重.*0"):
            load_from_temp(config_data)

    def test_sigma_must_be_positive(self) -> None:
        config_data = make_config()
        config_data["sigma"]["hltv"] = 0

        with self.assertRaisesRegex(ValueError, "sigma.*大于 0"):
            load_from_temp(config_data)

    def test_missing_team_rating_field_raises_clear_error(self) -> None:
        config_data = make_config()
        del config_data["teams"]["Team1"]["valve"]

        with self.assertRaisesRegex(ValueError, "Team1.*valve"):
            load_from_temp(config_data)

    def test_missing_seed_raises_clear_error(self) -> None:
        config_data = make_config()
        del config_data["teams"]["Team1"]["seed"]

        with self.assertRaisesRegex(ValueError, "Team1.*seed"):
            load_from_temp(config_data)

    def test_non_integer_seed_raises_clear_error(self) -> None:
        config_data = make_config()
        config_data["teams"]["Team1"]["seed"] = "1"

        with self.assertRaisesRegex(ValueError, "Team1.*seed.*整数"):
            load_from_temp(config_data)

    def test_missing_required_system_raises_clear_error(self) -> None:
        config_data = make_config()
        config_data["systems"] = {"hltv": "lambda x: x"}

        with self.assertRaisesRegex(ValueError, "valve"):
            load_from_temp(config_data)

    def test_only_16_team_swiss_is_supported(self) -> None:
        config_data = make_config()
        del config_data["teams"]["Team16"]

        with self.assertRaisesRegex(ValueError, "16 队瑞士轮.*15"):
            load_from_temp(config_data)


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


if __name__ == "__main__":
    unittest.main()
