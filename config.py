"""
CS2 Major Pick'Em 模拟器配置文件。

这个文件负责三件事：
1. 定义队伍和赛事配置的数据结构；
2. 从 JSON 安全加载队伍、评分系统、sigma 和权重参数；
3. 计算任意两支队伍之间的单图胜率。
"""

from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


# 评分系统权重配置。这里仍保留为全局常量，方便快速调参。
VRS_WEIGHT = 0.7  # Valve 评分系统权重
HLTV_WEIGHT = 0.3  # HLTV 评分系统权重

# 兼容旧调用的默认 sigma。真实模拟时优先使用 major_stage.json 中的 sigma。
SIGMA = 349.2

# 概率裁剪区间，避免模型给出过于极端的单图胜率。
PROBABILITY_FLOOR = 0.03
PROBABILITY_CEILING = 0.97

# 当前胜率模型明确使用前两个评分系统：valve 和 hltv。
REQUIRED_SYSTEMS = ("valve", "hltv")
SUPPORTED_TEAM_COUNT = 16
DEFAULT_MAP_POOL = (
    "Dust2",
    "Mirage",
    "Inferno",
    "Nuke",
    "Overpass",
    "Ancient",
    "Anubis",
)


@dataclass(frozen=True)
class Team:
    """
    队伍类，存储队伍的基本信息和评分。

    Attributes:
        id: 队伍唯一标识符，用于稳定哈希。
        name: 队伍名称。
        seed: 初始种子排名，数字越小种子越高。
        rating: 按 TournamentConfig.systems 顺序排列的评分元组。
    """

    id: int
    name: str
    seed: int
    rating: Tuple[float, ...]
    map_strengths: Dict[str, float] = field(default_factory=dict)
    map_win_rates: Dict[str, float] = field(default_factory=dict)
    map_pick_rates: Dict[str, float] = field(default_factory=dict)
    map_ban_rates: Dict[str, float] = field(default_factory=dict)
    map_played_counts: Dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return self.id


@dataclass(frozen=True)
class TournamentConfig:
    """
    赛事配置，确保模拟器和竞猜评估使用同一份参数。

    Attributes:
        systems: 评分系统名称，例如 ("valve", "hltv")。
        sigma: 与 systems 顺序一致的 sigma 元组。
        weights: 与 systems 顺序一致的权重元组。
        teams: 按 JSON 文件顺序加载的队伍列表。
    """

    systems: Tuple[str, ...]
    sigma: Tuple[float, ...]
    weights: Tuple[float, ...]
    map_pool: Tuple[str, ...]
    teams: Tuple[Team, ...]


def _identity(value: Any) -> Any:
    """默认转换函数：原样返回 JSON 中的评分值。"""
    return value


# JSON 中只允许引用这些转换函数，避免 eval 执行任意代码。
SYSTEM_TRANSFORMS: Dict[str, Callable[[Any], Any]] = {
    "identity": _identity,
    "lambda x: x": _identity,  # 兼容旧版 major_stage.json 的写法
    "int": int,
    "float": float,
}


def _apply_system_transform(system_name: str, transform_name: str, value: Any) -> float:
    """
    应用评分转换函数。

    使用白名单而不是 eval：配置文件只能选择已登记的转换函数，不能执行代码。
    """
    if transform_name not in SYSTEM_TRANSFORMS:
        allowed = ", ".join(sorted(SYSTEM_TRANSFORMS))
        raise ValueError(
            f"评分系统 {system_name!r} 使用了未知转换函数 {transform_name!r}，"
            f"允许值：{allowed}"
        )

    return float(SYSTEM_TRANSFORMS[transform_name](value))


def _default_weight_for_system(system_name: str) -> float:
    """
    返回评分系统的默认权重。

    旧配置文件没有 weights 字段时，valve/vrs 回退到 VRS_WEIGHT，
    hltv 回退到 HLTV_WEIGHT；未知评分系统给 1.0，避免破坏扩展配置。
    """
    normalized = system_name.lower()
    if normalized in {"valve", "vrs"}:
        return VRS_WEIGHT
    if normalized == "hltv":
        return HLTV_WEIGHT
    return 1.0


def _ordered_system_names(systems_data: Dict[str, str]) -> Tuple[str, ...]:
    """
    固定模型依赖的评分系统顺序。

    win_probability 使用 rating[0] 作为 valve，rating[1] 作为 hltv；
    因此这里必须把这两个系统排在最前，避免 JSON 字段顺序改变后结果失真。
    """
    missing = [system_name for system_name in REQUIRED_SYSTEMS if system_name not in systems_data]
    if missing:
        raise ValueError(f"配置文件缺少必要评分系统：{', '.join(missing)}")

    extra_systems = tuple(
        system_name for system_name in systems_data.keys() if system_name not in REQUIRED_SYSTEMS
    )
    return REQUIRED_SYSTEMS + extra_systems


def _validate_positive_values(values: Tuple[float, ...], label: str) -> None:
    """校验一组参数必须大于 0。"""
    for index, value in enumerate(values):
        if value <= 0:
            raise ValueError(f"{label}[{index}] 必须大于 0，当前值为 {value}")


def _validate_weights(weights: Tuple[float, ...]) -> None:
    """校验权重必须非负，且模型实际使用的前两个权重不能同时为 0。"""
    for index, weight in enumerate(weights):
        if weight < 0:
            raise ValueError(f"weights[{index}] 必须大于等于 0，当前值为 {weight}")

    active_weight_sum = sum(weights[: len(REQUIRED_SYSTEMS)])
    if active_weight_sum <= 0:
        raise ValueError("valve 和 hltv 的权重不能同时为 0")


def _read_team_seed(team_name: str, team_data: Dict[str, Any]) -> int:
    """读取并校验队伍 seed，避免配置错误变成难懂的 KeyError。"""
    if "seed" not in team_data:
        raise ValueError(f"队伍 {team_name!r} 缺少 seed 字段")

    seed = team_data["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"队伍 {team_name!r} 的 seed 必须是整数")

    return seed


def _read_team_rating(
    team_name: str,
    team_data: Dict[str, Any],
    system_name: str,
    transform_name: str,
) -> float:
    """按 TournamentConfig.systems 顺序读取评分，并给缺字段配置提供清晰错误。"""
    if system_name not in team_data:
        raise ValueError(f"队伍 {team_name!r} 缺少评分字段 {system_name!r}")

    return _apply_system_transform(system_name, transform_name, team_data[system_name])


def _read_map_pool(data: Dict[str, Any]) -> Tuple[str, ...]:
    """Read the configured map pool, falling back to the current CS2 default pool."""
    raw_map_pool = data.get("map_pool", DEFAULT_MAP_POOL)
    if not isinstance(raw_map_pool, (list, tuple)):
        raise ValueError("map_pool must be a list of map names")

    map_pool = tuple(raw_map_pool)
    if not map_pool:
        raise ValueError("map_pool must contain at least one map")

    seen_maps = set()
    for map_name in map_pool:
        if not isinstance(map_name, str) or not map_name.strip():
            raise ValueError(f"map_pool contains an invalid map name: {map_name!r}")
        if map_name in seen_maps:
            raise ValueError(f"map_pool contains duplicate map {map_name!r}")
        seen_maps.add(map_name)

    return map_pool


def _validate_map_stat(team_name: str, map_name: str, field_name: str, value: Any) -> float:
    """Validate one team/map stat value from JSON and return it as a float."""
    if isinstance(value, bool):
        raise ValueError(
            f"Team {team_name!r} map {map_name!r} {field_name} must be a number in [0, 1]"
        )

    try:
        stat_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Team {team_name!r} map {map_name!r} {field_name} must be a number in [0, 1]"
        ) from exc

    if not 0 <= stat_value <= 1:
        raise ValueError(
            f"Team {team_name!r} map {map_name!r} {field_name} must be in [0, 1], got {stat_value}"
        )

    return stat_value


def _validate_maps_played(team_name: str, map_name: str, value: Any) -> int:
    """Validate a non-negative integer map sample count."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"Team {team_name!r} map {map_name!r} maps_played must be a non-negative integer"
        )
    if value < 0:
        raise ValueError(
            f"Team {team_name!r} map {map_name!r} maps_played must be a non-negative integer"
        )
    return value


def _validate_team_map_name(team_name: str, map_name: Any) -> str:
    if not isinstance(map_name, str) or not map_name.strip():
        raise ValueError(f"Team {team_name!r} has an invalid map name: {map_name!r}")
    return map_name


def _read_team_map_stats(
    team_name: str,
    team_data: Dict[str, Any],
    map_pool: Tuple[str, ...],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, int]]:
    """Read optional per-map stats, defaulting missing pool maps to neutral history."""
    map_strengths = {map_name: 0.5 for map_name in map_pool}
    map_win_rates = {map_name: 0.0 for map_name in map_pool}
    map_pick_rates = {map_name: 0.0 for map_name in map_pool}
    map_ban_rates = {map_name: 0.0 for map_name in map_pool}
    map_played_counts = {map_name: 0 for map_name in map_pool}

    raw_map_stats = team_data.get("map_stats")
    if raw_map_stats is not None:
        if not isinstance(raw_map_stats, dict):
            raise ValueError(f"Team {team_name!r} map_stats must be an object of map stat objects")

        for raw_map_name, raw_stats in raw_map_stats.items():
            map_name = _validate_team_map_name(team_name, raw_map_name)
            if not isinstance(raw_stats, dict):
                raise ValueError(
                    f"Team {team_name!r} map {map_name!r} map_stats entry must be an object"
                )
            map_strengths[map_name] = _validate_map_stat(
                team_name,
                map_name,
                "strength",
                raw_stats.get("strength", 0.5),
            )
            map_win_rates[map_name] = _validate_map_stat(
                team_name,
                map_name,
                "win_rate",
                raw_stats.get("win_rate", 0.0),
            )
            map_pick_rates[map_name] = _validate_map_stat(
                team_name,
                map_name,
                "pick_rate",
                raw_stats.get("pick_rate", 0.0),
            )
            map_ban_rates[map_name] = _validate_map_stat(
                team_name,
                map_name,
                "ban_rate",
                raw_stats.get("ban_rate", 0.0),
            )
            map_played_counts[map_name] = _validate_maps_played(
                team_name,
                map_name,
                raw_stats.get("maps_played", 0),
            )

        return map_strengths, map_win_rates, map_pick_rates, map_ban_rates, map_played_counts

    raw_maps = team_data.get("maps")
    if raw_maps is None:
        return map_strengths, map_win_rates, map_pick_rates, map_ban_rates, map_played_counts

    if not isinstance(raw_maps, dict):
        raise ValueError(f"Team {team_name!r} maps must be an object of map strengths")

    for raw_map_name, value in raw_maps.items():
        map_name = _validate_team_map_name(team_name, raw_map_name)
        map_strengths[map_name] = _validate_map_stat(team_name, map_name, "strength", value)

    return map_strengths, map_win_rates, map_pick_rates, map_ban_rates, map_played_counts


def _validate_supported_team_count(teams: List[Team]) -> None:
    """当前瑞士轮实现只覆盖 CS Major Pick'Em 的 16 队阶段。"""
    if len(teams) != SUPPORTED_TEAM_COUNT:
        raise ValueError(
            f"当前模拟器只支持 16 队瑞士轮，当前配置包含 {len(teams)} 支队伍"
        )


def load_tournament_config(file_path: str | Path) -> TournamentConfig:
    """
    从 JSON 文件加载完整赛事配置。

    Args:
        file_path: JSON 文件路径。

    Returns:
        TournamentConfig: 包含评分系统、sigma、权重和队伍数据的配置对象。
    """
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    systems_data = data.get("systems")
    if not systems_data:
        raise ValueError("配置文件缺少 systems 字段，无法判断评分系统顺序")

    system_names = _ordered_system_names(systems_data)
    map_pool = _read_map_pool(data)

    # sigma 按 systems 顺序读取；缺失时回退到兼容旧代码的默认 SIGMA。
    sigma_data = data.get("sigma", {})
    sigma = tuple(float(sigma_data.get(system_name, SIGMA)) for system_name in system_names)
    _validate_positive_values(sigma, "sigma")

    # weights 同样按 systems 顺序读取；缺失时回退到 config.py 中的全局常量。
    weights_data = data.get("weights") or {}
    weights = tuple(
        float(weights_data.get(system_name, _default_weight_for_system(system_name)))
        for system_name in system_names
    )
    _validate_weights(weights)

    teams = []
    for team_id, (team_name, team_data) in enumerate(data["teams"].items()):
        # rating 必须严格跟 TournamentConfig.systems 保持同序：
        # rating[0] 是 valve，rating[1] 是 hltv。不能使用 JSON 原始字段顺序。
        rating = tuple(
            _read_team_rating(
                team_name,
                team_data,
                system_name,
                systems_data[system_name],
            )
            for system_name in system_names
        )
        (
            map_strengths,
            map_win_rates,
            map_pick_rates,
            map_ban_rates,
            map_played_counts,
        ) = _read_team_map_stats(
            team_name,
            team_data,
            map_pool,
        )
        teams.append(
            Team(
                id=team_id,
                name=team_name,
                seed=_read_team_seed(team_name, team_data),
                rating=rating,
                map_strengths=map_strengths,
                map_win_rates=map_win_rates,
                map_pick_rates=map_pick_rates,
                map_ban_rates=map_ban_rates,
                map_played_counts=map_played_counts,
            )
        )

    _validate_supported_team_count(teams)

    return TournamentConfig(
        systems=system_names,
        sigma=sigma,
        weights=weights,
        map_pool=map_pool,
        teams=tuple(teams),
    )


def load_teams(file_path: str | Path) -> List[Team]:
    """
    从 JSON 文件加载队伍数据。

    这个函数保留给旧代码使用；新代码建议直接调用 load_tournament_config，
    这样可以同时拿到队伍和 sigma。
    """
    return list(load_tournament_config(file_path).teams)


@lru_cache(maxsize=None)
def win_probability(
    a: Team,
    b: Team,
    sigma: Tuple[float, ...] = (SIGMA, SIGMA),
    clamp: Tuple[float, float] = (PROBABILITY_FLOOR, PROBABILITY_CEILING),
    weights: Tuple[float, ...] | None = None,
) -> float:
    """
    计算队伍 a 在单图中战胜队伍 b 的概率。

    当前模型使用两部分信息：
    1. Valve/VRS 评分差，走 Elo 形式；
    2. HLTV 评分差，也走 Elo/logistic 形式。

    Args:
        a: 队伍 a。
        b: 队伍 b。
        sigma: 与评分系统顺序一致的 sigma 元组；sigma[0] 给 VRS，sigma[1] 给 HLTV。
        clamp: 概率裁剪区间，默认限制在 3% 到 97%。
        weights: 与评分系统顺序一致的权重元组；缺失时回退到全局常量。
    """
    if len(a.rating) < 2 or len(b.rating) < 2:
        raise ValueError("胜率模型至少需要 valve 和 hltv 两个评分系统")

    v1, h1 = a.rating[0], a.rating[1]
    v2, h2 = b.rating[0], b.rating[1]

    # VRS 使用 Elo 公式；sigma 越大，评分差对胜率的影响越平缓。
    vrs_sigma = sigma[0] if sigma else SIGMA
    if vrs_sigma <= 0:
        raise ValueError(f"vrs_sigma 必须大于 0，当前值为 {vrs_sigma}")
    p_vrs = 1 / (1 + 10 ** ((v2 - v1) / vrs_sigma))

    # HLTV 同样使用 Elo/logistic 公式；缺少 sigma.hltv 时回退到 SIGMA。
    hltv_sigma = sigma[1] if len(sigma) > 1 else SIGMA
    if hltv_sigma <= 0:
        raise ValueError(f"hltv_sigma 必须大于 0，当前值为 {hltv_sigma}")
    p_hltv = 1 / (1 + 10 ** ((h2 - h1) / hltv_sigma))

    # 加权融合两路胜率。若没有传入 TournamentConfig.weights，则兼容旧全局常量。
    actual_weights = weights if weights is not None else (VRS_WEIGHT, HLTV_WEIGHT)
    vrs_weight = actual_weights[0] if len(actual_weights) > 0 else VRS_WEIGHT
    hltv_weight = actual_weights[1] if len(actual_weights) > 1 else HLTV_WEIGHT
    _validate_weights((vrs_weight, hltv_weight))
    weight_sum = vrs_weight + hltv_weight

    raw_probability = (vrs_weight * p_vrs + hltv_weight * p_hltv) / weight_sum
    return clamp_probability(raw_probability, clamp)


def clamp_probability(probability: float, clamp: Tuple[float, float]) -> float:
    """
    将胜率限制在指定区间内。

    默认区间 [0.03, 0.97] 是对称的，因此不会破坏 p(a,b) + p(b,a) ≈ 1 的性质。
    """
    lower, upper = clamp
    if lower < 0 or upper > 1 or lower > upper:
        raise ValueError("概率裁剪区间必须满足 0 <= lower <= upper <= 1")
    return min(max(probability, lower), upper)


def calculate_win_matrix(
    teams: List[Team],
    sigma: Tuple[float, ...] = (SIGMA, SIGMA),
    weights: Tuple[float, ...] | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    计算所有队伍之间的胜率矩阵。

    Returns:
        Dict[str, Dict[str, float]]: {队伍A: {队伍B: A 胜 B 的概率}}
    """
    win_matrix = {}

    for team1 in teams:
        win_matrix[team1.name] = {}
        for team2 in teams:
            if team1 != team2:
                win_matrix[team1.name][team2.name] = win_probability(
                    team1,
                    team2,
                    sigma,
                    weights=weights,
                )

    return win_matrix


def print_win_matrix(win_matrix: Dict[str, Dict[str, float]], teams: List[Team]) -> None:
    """打印胜率矩阵，便于人工检查参数是否合理。"""
    print("胜率矩阵（行队名 vs. 列队名 -> 行队名获胜概率）:")

    column_width = 10

    header = "队伍".center(column_width)
    for team in teams:
        team_name = team.name
        if len(team_name) > column_width - 2:
            team_name = team_name[: column_width - 3] + "..."
        header += team_name.center(column_width)
    print(header)

    print("-" * column_width * (len(teams) + 1))

    for team1 in teams:
        team1_name = team1.name
        if len(team1_name) > column_width - 2:
            team1_name = team1_name[: column_width - 3] + "..."
        row = team1_name.center(column_width)

        for team2 in teams:
            if team1 == team2:
                row += "-".center(column_width)
            else:
                win_rate = win_matrix[team1.name][team2.name]
                row += f"{win_rate:.2f}".center(column_width)
        print(row)


def main() -> None:
    """打印当前配置下的胜率矩阵，用作配置文件自检入口。"""
    file_path = "major_stage.json"
    try:
        tournament = load_tournament_config(file_path)
        teams = list(tournament.teams)
        win_matrix = calculate_win_matrix(teams, tournament.sigma, tournament.weights)
        print_win_matrix(win_matrix, teams)
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
    except json.JSONDecodeError:
        print(f"错误：{file_path} 不是有效的 JSON 文件")
    except Exception as exc:
        print(f"发生错误：{exc}")


if __name__ == "__main__":
    main()
