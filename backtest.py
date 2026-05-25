"""Historical backtesting and parameter fitting for the Pick'Em simulator."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
from html import unescape
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Iterable

from config import (
    DEFAULT_MAP_POOL,
    HLTV_WEIGHT,
    ModelParams,
    Team,
    VRS_WEIGHT,
    map_strength_from_stats,
)
from maps import bo1_veto_maps, bo3_veto_maps, map_win_probability
from simulate import Record, SwissSystem


DEFAULT_HISTORY_DIR = Path(".cache/hltv_history")
MATCHES_FILE = "matches.jsonl"
RAW_DIR = "raw"
HLTV_BASE_URL = "https://www.hltv.org"
MATCH_LINK_RE = re.compile(r'href="(?P<path>/matches/(?P<id>\d+)/[^"]+)"')
TEAM_NAME_RE = re.compile(r'class="[^"]*teamName[^"]*"[^>]*>(?P<name>[^<]+)<')


@dataclass(frozen=True)
class BacktestParams:
    weights: tuple[float, float] = (VRS_WEIGHT, HLTV_WEIGHT)
    sigma: tuple[float, float] = (600.0, 1600.0)
    model_params: ModelParams = ModelParams()


def _clamp_probability(value: float) -> float:
    return min(max(value, 1e-6), 1 - 1e-6)


def log_loss(probability: float, actual: bool) -> float:
    probability = _clamp_probability(probability)
    return -math.log(probability if actual else 1 - probability)


def brier_score(probability: float, actual: bool) -> float:
    target = 1.0 if actual else 0.0
    return (probability - target) ** 2


def parse_results_html(html: str) -> list[dict[str, str]]:
    matches = []
    seen_ids = set()
    for match in MATCH_LINK_RE.finditer(html):
        match_id = match.group("id")
        if match_id in seen_ids:
            continue
        seen_ids.add(match_id)
        matches.append({"id": match_id, "url": HLTV_BASE_URL + match.group("path")})
    return matches


def parse_match_html(html: str, match_id: str | None = None, url: str | None = None) -> dict[str, Any] | None:
    team_names = [unescape(match.group("name")).strip() for match in TEAM_NAME_RE.finditer(html)]
    team_names = [name for name in team_names if name]
    if len(team_names) < 2:
        return None

    title_match = re.search(r"<title>(?P<title>.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title = unescape(title_match.group("title")).strip() if title_match else ""
    return {
        "id": match_id or "",
        "url": url or "",
        "event": title,
        "date": "",
        "format": "bo1",
        "team_a": {"name": team_names[0], "seed": 1, "valve": 1000.0, "hltv": 1000.0},
        "team_b": {"name": team_names[1], "seed": 2, "valve": 1000.0, "hltv": 1000.0},
        "winner": team_names[0],
    }


def _history_matches_path(history: str | Path) -> Path:
    history_path = Path(history)
    if history_path.is_dir():
        return history_path / MATCHES_FILE
    return history_path


def load_matches(history: str | Path) -> list[dict[str, Any]]:
    matches_path = _history_matches_path(history)
    if not matches_path.exists():
        raise FileNotFoundError(f"History matches file not found: {matches_path}")

    matches = []
    with matches_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                matches.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {matches_path}:{line_number}") from exc
    return matches


def _team_from_history(team_data: dict[str, Any], team_id: int, map_pool: tuple[str, ...], params: BacktestParams) -> Team:
    map_strengths = {map_name: 0.5 for map_name in map_pool}
    map_win_rates = {map_name: 0.0 for map_name in map_pool}
    map_pick_rates = {map_name: 0.0 for map_name in map_pool}
    map_ban_rates = {map_name: 0.0 for map_name in map_pool}
    map_played_counts = {map_name: 0 for map_name in map_pool}

    for map_name, raw_stats in (team_data.get("map_stats") or {}).items():
        if map_name not in map_strengths:
            continue
        win_rate = float(raw_stats.get("win_rate", 0.0))
        pick_rate = float(raw_stats.get("pick_rate", 0.0))
        ban_rate = float(raw_stats.get("ban_rate", 0.0))
        maps_played = int(raw_stats.get("maps_played", 0))
        missing = bool(raw_stats.get("missing")) or (
            maps_played == 0 and win_rate == 0.0 and pick_rate == 0.0 and ban_rate == 1.0
        )

        map_win_rates[map_name] = win_rate
        map_pick_rates[map_name] = pick_rate
        map_ban_rates[map_name] = ban_rate
        map_played_counts[map_name] = maps_played
        if "win_rate" in raw_stats and "maps_played" in raw_stats:
            map_strengths[map_name] = map_strength_from_stats(
                win_rate,
                maps_played,
                pick_rate,
                ban_rate,
                params.model_params.bayes_prior_maps,
                missing=missing,
            )
        else:
            map_strengths[map_name] = float(raw_stats.get("strength", 0.5))

    for map_name, strength in (team_data.get("maps") or {}).items():
        if map_name in map_strengths:
            map_strengths[map_name] = float(strength)

    return Team(
        id=team_id,
        name=str(team_data["name"]),
        seed=int(team_data.get("seed", team_id + 1)),
        rating=(float(team_data.get("valve", 1000.0)), float(team_data.get("hltv", 1000.0))),
        map_strengths=map_strengths,
        map_win_rates=map_win_rates,
        map_pick_rates=map_pick_rates,
        map_ban_rates=map_ban_rates,
        map_played_counts=map_played_counts,
    )


def _stable_seed(match_id: str, sample_index: int) -> int:
    digest = hashlib.sha256(f"{match_id}:{sample_index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _bo3_probability(map_probabilities: tuple[float, float, float]) -> float:
    p1, p2, p3 = map_probabilities
    return p1 * p2 + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3


def _match_format(match: dict[str, Any]) -> str:
    if "best_of" in match:
        return "bo3" if int(match["best_of"]) == 3 else "bo1"
    return str(match.get("format", "bo1")).lower()


def _winner_is_team_a(match: dict[str, Any], team_a: Team) -> bool:
    winner = str(match.get("winner", "")).strip()
    return winner.lower() in {"a", "team_a", team_a.name.lower()}


def _match_date(match: dict[str, Any]) -> date | None:
    raw_date = str(match.get("date", "")).strip()
    if not raw_date:
        return None
    try:
        return date.fromisoformat(raw_date[:10])
    except ValueError:
        return None


def rolling_validation_windows(
    matches: list[dict[str, Any]],
    train_days: int = 90,
    validation_days: int = 30,
    step_days: int = 30,
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    dated_matches = [(match_date, match) for match in matches if (match_date := _match_date(match)) is not None]
    if not dated_matches:
        return []

    dated_matches.sort(key=lambda item: item[0])
    first_date = dated_matches[0][0]
    last_date = dated_matches[-1][0]
    windows = []
    window_start = first_date

    while window_start + timedelta(days=train_days) <= last_date:
        train_end = window_start + timedelta(days=train_days)
        validation_end = train_end + timedelta(days=validation_days)
        train_matches = [
            match for match_date, match in dated_matches if window_start <= match_date < train_end
        ]
        validation_matches = [
            match for match_date, match in dated_matches if train_end <= match_date < validation_end
        ]
        if train_matches and validation_matches:
            windows.append((train_matches, validation_matches))
        window_start += timedelta(days=step_days)

    return windows


def predict_match_probability(
    match: dict[str, Any],
    params: BacktestParams,
    veto_samples: int = 32,
) -> float:
    map_pool = tuple(match.get("map_pool") or DEFAULT_MAP_POOL)
    team_a = _team_from_history(match["team_a"], 1, map_pool, params)
    team_b = _team_from_history(match["team_b"], 2, map_pool, params)
    match_id = str(match.get("id") or f"{team_a.name}-{team_b.name}-{match.get('date', '')}")
    samples = 1 if params.model_params.veto_temperature <= 0 else max(1, veto_samples)

    probabilities = []
    for sample_index in range(samples):
        rng = random.Random(_stable_seed(match_id, sample_index))
        if _match_format(match) == "bo3":
            veto_maps = bo3_veto_maps(team_a, team_b, map_pool, rng, params.model_params)
            probabilities.append(
                _bo3_probability(
                    tuple(
                        map_win_probability(
                            team_a,
                            team_b,
                            map_name,
                            params.sigma,
                            params.weights,
                            params.model_params,
                        )
                        for map_name in veto_maps
                    )
                )
            )
        else:
            map_name = bo1_veto_maps(team_a, team_b, map_pool, rng, params.model_params)
            probabilities.append(
                map_win_probability(
                    team_a,
                    team_b,
                    map_name,
                    params.sigma,
                    params.weights,
                    params.model_params,
                )
            )

    return sum(probabilities) / len(probabilities)


def evaluate_match_metrics(
    matches: Iterable[dict[str, Any]],
    params: BacktestParams,
    veto_samples: int = 32,
) -> dict[str, float | int]:
    losses = []
    briers = []
    correct = 0
    count = 0

    for match in matches:
        team_a = _team_from_history(match["team_a"], 1, tuple(match.get("map_pool") or DEFAULT_MAP_POOL), params)
        actual = _winner_is_team_a(match, team_a)
        probability = predict_match_probability(match, params, veto_samples)
        losses.append(log_loss(probability, actual))
        briers.append(brier_score(probability, actual))
        correct += int((probability >= 0.5) == actual)
        count += 1

    if count == 0:
        return {"matches": 0, "log_loss": 0.0, "brier": 0.0, "accuracy": 0.0}
    return {
        "matches": count,
        "log_loss": sum(losses) / count,
        "brier": sum(briers) / count,
        "accuracy": correct / count,
    }


def _event_key(match: dict[str, Any]) -> str:
    return str(match.get("event_id") or match.get("event") or "unknown")


def _teams_for_event(matches: list[dict[str, Any]], params: BacktestParams) -> list[Team]:
    snapshots: dict[str, dict[str, Any]] = {}
    map_pool = tuple(matches[0].get("map_pool") or DEFAULT_MAP_POOL)
    for match in matches:
        for side in ("team_a", "team_b"):
            team_data = match[side]
            snapshots.setdefault(str(team_data["name"]), team_data)
    return [
        _team_from_history(team_data, index + 1, map_pool, params)
        for index, team_data in enumerate(sorted(snapshots.values(), key=lambda data: int(data.get("seed", 999))))
    ]


def evaluate_swiss_metrics(
    matches: list[dict[str, Any]],
    params: BacktestParams,
    iterations: int = 200,
    seed: int = 42,
) -> dict[str, float | int]:
    event_matches: dict[str, list[dict[str, Any]]] = {}
    for match in matches:
        if str(match.get("stage", "")).lower() == "swiss":
            event_matches.setdefault(_event_key(match), []).append(match)

    event_count = 0
    log_losses = []
    briers = []

    for matches_in_event in event_matches.values():
        teams = _teams_for_event(matches_in_event, params)
        if len(teams) != 16:
            continue

        actual_records: dict[str, list[int]] = {team.name: [0, 0] for team in teams}
        for match in matches_in_event:
            team_a_name = str(match["team_a"]["name"])
            team_b_name = str(match["team_b"]["name"])
            team_a_won = str(match.get("winner", "")).lower() in {"a", "team_a", team_a_name.lower()}
            winner_name, loser_name = (team_a_name, team_b_name) if team_a_won else (team_b_name, team_a_name)
            if winner_name in actual_records and loser_name in actual_records:
                actual_records[winner_name][0] += 1
                actual_records[loser_name][1] += 1

        advanced_counts = {team.name: 0 for team in teams}
        zero_three_counts = {team.name: 0 for team in teams}
        for index in range(iterations):
            rng = random.Random(seed + index)
            swiss = SwissSystem(
                sigma=params.sigma,
                weights=params.weights,
                records={team: Record.new() for team in teams},
                remaining=set(teams),
                rng=rng,
                map_pool=tuple(matches_in_event[0].get("map_pool") or DEFAULT_MAP_POOL),
                model_params=params.model_params,
            )
            swiss.simulate_tournament()
            for team, record in swiss.records.items():
                advanced_counts[team.name] += int(record.wins == 3)
                zero_three_counts[team.name] += int(record.losses == 3 and record.wins == 0)

        for team in teams:
            wins, losses = actual_records[team.name]
            actual_advanced = wins >= 3
            probability = advanced_counts[team.name] / iterations
            log_losses.append(log_loss(probability, actual_advanced))
            briers.append(brier_score(probability, actual_advanced))

            actual_zero_three = losses >= 3 and wins == 0
            zero_probability = zero_three_counts[team.name] / iterations
            log_losses.append(log_loss(zero_probability, actual_zero_three))
            briers.append(brier_score(zero_probability, actual_zero_three))

        event_count += 1

    if not log_losses:
        return {"swiss_events": event_count, "log_loss": 0.0, "brier": 0.0}
    return {
        "swiss_events": event_count,
        "log_loss": sum(log_losses) / len(log_losses),
        "brier": sum(briers) / len(briers),
    }


def evaluate_history(
    matches: list[dict[str, Any]],
    params: BacktestParams,
    veto_samples: int = 32,
    swiss_iterations: int = 200,
) -> dict[str, Any]:
    return {
        "match": evaluate_match_metrics(matches, params, veto_samples),
        "swiss": evaluate_swiss_metrics(matches, params, swiss_iterations),
    }


def _parameter_grid(best: BacktestParams | None = None) -> Iterable[BacktestParams]:
    if best is None:
        shares = (0.5, 0.7, 0.9)
        valve_sigmas = (450.0, 600.0, 750.0)
        hltv_sigmas = (1200.0, 1600.0, 2000.0)
        map_adjustments = (0.12, 0.20, 0.28)
        temperatures = (0.0, 0.15, 0.30)
    else:
        share = best.weights[0] / sum(best.weights)
        shares = tuple(sorted({min(max(share + delta, 0.05), 0.95) for delta in (-0.1, -0.05, 0, 0.05, 0.1)}))
        valve_sigmas = tuple(sorted({max(100.0, best.sigma[0] + delta) for delta in (-100, -50, 0, 50, 100)}))
        hltv_sigmas = tuple(sorted({max(100.0, best.sigma[1] + delta) for delta in (-250, -125, 0, 125, 250)}))
        map_adjustments = tuple(sorted({max(0.0, best.model_params.map_adjustment_weight + delta) for delta in (-0.06, -0.03, 0, 0.03, 0.06)}))
        temperatures = tuple(sorted({max(0.0, best.model_params.veto_temperature + delta) for delta in (-0.08, -0.04, 0, 0.04, 0.08)}))

    for share in shares:
        for valve_sigma in valve_sigmas:
            for hltv_sigma in hltv_sigmas:
                for map_adjustment in map_adjustments:
                    for temperature in temperatures:
                        yield BacktestParams(
                            weights=(share, 1.0 - share),
                            sigma=(valve_sigma, hltv_sigma),
                            model_params=ModelParams(
                                map_adjustment_weight=map_adjustment,
                                veto_temperature=temperature,
                                bayes_prior_maps=5.0,
                            ),
                        )


def _fit_objective(
    matches: list[dict[str, Any]],
    params: BacktestParams,
    veto_samples: int,
    windows: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]],
) -> float | None:
    if not windows:
        metrics = evaluate_match_metrics(matches, params, veto_samples)
        return float(metrics["log_loss"]) if metrics["matches"] else None

    weighted_loss = 0.0
    total_matches = 0
    for _train_matches, validation_matches in windows:
        metrics = evaluate_match_metrics(validation_matches, params, veto_samples)
        match_count = int(metrics["matches"])
        if match_count == 0:
            continue
        weighted_loss += float(metrics["log_loss"]) * match_count
        total_matches += match_count

    if total_matches == 0:
        return None
    return weighted_loss / total_matches


def fit_parameters(matches: list[dict[str, Any]], veto_samples: int = 16) -> dict[str, Any]:
    best_params = None
    best_loss = math.inf
    evaluated = 0
    validation_windows = rolling_validation_windows(matches)

    for round_index in range(2):
        grid = _parameter_grid(best_params)
        for params in grid:
            objective = _fit_objective(matches, params, veto_samples, validation_windows)
            evaluated += 1
            if objective is not None and objective < best_loss:
                best_loss = objective
                best_params = params

    if best_params is None:
        raise ValueError("Cannot fit parameters without historical matches")

    evaluation = evaluate_history(matches, best_params, veto_samples, swiss_iterations=50)
    return {
        "evaluated_candidates": evaluated,
        "best_log_loss": best_loss,
        "validation": {
            "mode": "rolling" if validation_windows else "all_matches",
            "windows": len(validation_windows),
            "train_days": 90,
            "validation_days": 30,
            "step_days": 30,
        },
        "metrics": evaluation,
        "params": params_to_dict(best_params),
    }


def params_to_dict(params: BacktestParams) -> dict[str, Any]:
    return {
        "weights": {"valve": params.weights[0], "hltv": params.weights[1]},
        "sigma": {"valve": params.sigma[0], "hltv": params.sigma[1]},
        "model_params": {
            "map_adjustment_weight": params.model_params.map_adjustment_weight,
            "veto_temperature": params.model_params.veto_temperature,
            "bayes_prior_maps": params.model_params.bayes_prior_maps,
        },
    }


def params_from_dict(data: dict[str, Any]) -> BacktestParams:
    params_data = data.get("params", data)
    weights = params_data.get("weights", {})
    sigma = params_data.get("sigma", {})
    model_params = params_data.get("model_params", {})
    return BacktestParams(
        weights=(float(weights.get("valve", VRS_WEIGHT)), float(weights.get("hltv", HLTV_WEIGHT))),
        sigma=(float(sigma.get("valve", 600.0)), float(sigma.get("hltv", 1600.0))),
        model_params=ModelParams(
            map_adjustment_weight=float(model_params.get("map_adjustment_weight", 0.20)),
            veto_temperature=float(model_params.get("veto_temperature", 0.15)),
            bayes_prior_maps=float(model_params.get("bayes_prior_maps", 5.0)),
        ),
    )


def apply_params(config_path: Path, params_path: Path, output_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fitted = json.loads(params_path.read_text(encoding="utf-8"))
    params = params_to_dict(params_from_dict(fitted))
    config["weights"] = params["weights"]
    config["sigma"] = params["sigma"]
    config["model_params"] = params["model_params"]
    output_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fetch_history(months: int, cache_dir: Path, max_matches: int) -> dict[str, Any]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is required for fetch. Install it and run: python -m playwright install chromium") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = cache_dir / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    end_date = date.today()
    start_date = end_date - timedelta(days=months * 30)
    results_url = f"{HLTV_BASE_URL}/results?startDate={start_date.isoformat()}&endDate={end_date.isoformat()}"
    parsed_matches = []
    skipped_matches = 0

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(user_agent="Mozilla/5.0")
        page.goto(results_url, wait_until="domcontentloaded", timeout=60_000)
        results_html = page.content()
        (raw_dir / "results.html").write_text(results_html, encoding="utf-8")

        match_links = parse_results_html(results_html)[:max_matches]
        for match_link in match_links:
            match_url = match_link["url"]
            page.goto(match_url, wait_until="domcontentloaded", timeout=60_000)
            match_html = page.content()
            (raw_dir / f"match_{match_link['id']}.html").write_text(match_html, encoding="utf-8")
            parsed = parse_match_html(match_html, match_link["id"], match_url)
            if parsed is None:
                skipped_matches += 1
                continue
            parsed_matches.append(parsed)
        browser.close()

    matches_path = cache_dir / MATCHES_FILE
    with matches_path.open("w", encoding="utf-8") as file:
        for match in parsed_matches:
            file.write(json.dumps(match, ensure_ascii=False) + "\n")

    manifest = {
        "source": "hltv",
        "results_url": results_url,
        "months": months,
        "raw_matches": len(parsed_matches) + skipped_matches,
        "parsed_matches": len(parsed_matches),
        "skipped_matches": skipped_matches,
        "matches_path": str(matches_path),
    }
    (cache_dir / "fetch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Historical backtest and parameter fitting tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch")
    fetch_parser.add_argument("--months", type=int, default=6)
    fetch_parser.add_argument("--cache", default=str(DEFAULT_HISTORY_DIR))
    fetch_parser.add_argument("--max-matches", type=int, default=100)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--history", default=str(DEFAULT_HISTORY_DIR))
    evaluate_parser.add_argument("--veto-samples", type=int, default=32)
    evaluate_parser.add_argument("--swiss-iterations", type=int, default=200)

    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--history", default=str(DEFAULT_HISTORY_DIR))
    fit_parser.add_argument("--output", default="fitted_params.json")
    fit_parser.add_argument("--veto-samples", type=int, default=16)

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--config", default="major_stage.json")
    apply_parser.add_argument("--params", default="fitted_params.json")
    apply_parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "fetch":
        result = fetch_history(args.months, Path(args.cache), args.max_matches)
    elif args.command == "evaluate":
        matches = load_matches(args.history)
        result = evaluate_history(matches, BacktestParams(), args.veto_samples, args.swiss_iterations)
    elif args.command == "fit":
        matches = load_matches(args.history)
        result = fit_parameters(matches, args.veto_samples)
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif args.command == "apply":
        apply_params(Path(args.config), Path(args.params), Path(args.output))
        result = {"output": args.output}
    else:
        raise ValueError(f"Unsupported command: {args.command}")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
