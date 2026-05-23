"""
CS2 Major Pick'Em 模拟器配置文件。

这个文件负责三件事：
1. 定义队伍和赛事配置的数据结构；
2. 从 JSON 安全加载队伍、评分系统和 sigma 参数；
3. 计算任意两支队伍之间的单图胜率。
"""

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


# 评分系统权重配置。这里仍保留为全局常量，方便快速调参。
VRS_WEIGHT = 0.5  # Valve 评分系统权重
HLTV_WEIGHT = 0.5  # HLTV 评分系统权重

# 兼容旧调用的默认 sigma。真实模拟时优先使用 major_stage.json 中的 sigma。
SIGMA = 349.2


@dataclass(frozen=True)
class Team:
    """
    队伍类，存储队伍的基本信息和评分。

    Attributes:
        id: 队伍唯一标识符，用于稳定哈希。
        name: 队伍名称。
        seed: 初始种子排名，数字越小种子越高。
        rating: 按 JSON 中 systems 顺序排列的评分元组。
    """

    id: int
    name: str
    seed: int
    rating: Tuple[float, ...]

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
        teams: 按 JSON 文件顺序加载的队伍列表。
    """

    systems: Tuple[str, ...]
    sigma: Tuple[float, ...]
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


def load_tournament_config(file_path: str | Path) -> TournamentConfig:
    """
    从 JSON 文件加载完整赛事配置。

    Args:
        file_path: JSON 文件路径。

    Returns:
        TournamentConfig: 包含评分系统、sigma 和队伍数据的配置对象。
    """
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    systems_data = data.get("systems")
    if not systems_data:
        raise ValueError("配置文件缺少 systems 字段，无法判断评分系统顺序")

    system_names = tuple(systems_data.keys())

    # sigma 按 systems 顺序读取；缺失时回退到兼容旧代码的默认 SIGMA。
    sigma_data = data.get("sigma", {})
    sigma = tuple(float(sigma_data.get(system_name, SIGMA)) for system_name in system_names)

    teams = []
    for team_id, (team_name, team_data) in enumerate(data["teams"].items()):
        rating = tuple(
            _apply_system_transform(system_name, transform_name, team_data[system_name])
            for system_name, transform_name in systems_data.items()
        )
        teams.append(
            Team(
                id=team_id,
                name=team_name,
                seed=int(team_data["seed"]),
                rating=rating,
            )
        )

    return TournamentConfig(
        systems=system_names,
        sigma=sigma,
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
def win_probability(a: Team, b: Team, sigma: Tuple[float, ...] = (SIGMA, SIGMA)) -> float:
    """
    计算队伍 a 在单图中战胜队伍 b 的概率。

    当前模型使用两部分信息：
    1. Valve/VRS 评分差，走 Elo 形式；
    2. HLTV 分数比例，作为另一路经验信号。
    """
    if len(a.rating) < 2 or len(b.rating) < 2:
        raise ValueError("胜率模型至少需要 valve 和 hltv 两个评分系统")

    v1, h1 = a.rating[0], a.rating[1]
    v2, h2 = b.rating[0], b.rating[1]

    # VRS 使用 Elo 公式；sigma 越大，评分差对胜率的影响越平缓。
    vrs_sigma = sigma[0] if sigma else SIGMA
    p_vrs = 1 / (1 + 10 ** ((v2 - v1) / vrs_sigma))

    # HLTV 使用比例模型；极端情况下避免 0 除。
    hltv_total = h1 + h2
    p_hltv = 0.5 if hltv_total == 0 else h1 / hltv_total

    # 加权融合两路胜率。若权重都为 0，则退回五五开。
    weight_sum = VRS_WEIGHT + HLTV_WEIGHT
    if weight_sum <= 0:
        return 0.5

    return (VRS_WEIGHT * p_vrs + HLTV_WEIGHT * p_hltv) / weight_sum


def calculate_win_matrix(
    teams: List[Team],
    sigma: Tuple[float, ...] = (SIGMA, SIGMA),
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
                win_matrix[team1.name][team2.name] = win_probability(team1, team2, sigma)

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
        win_matrix = calculate_win_matrix(teams, tournament.sigma)
        print_win_matrix(win_matrix, teams)
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
    except json.JSONDecodeError:
        print(f"错误：{file_path} 不是有效的 JSON 文件")
    except Exception as exc:
        print(f"发生错误：{exc}")


if __name__ == "__main__":
    main()
