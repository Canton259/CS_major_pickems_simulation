"""
CS2 Major Pick'Em 模拟器主程序。

核心职责：
1. 按瑞士轮规则模拟比赛阶段；
2. 统计每支队伍的 3-0、3-1/3-2、0-3 次数；
3. 统计每一种 Pick'Em 结果组合的出现频率；
4. 通过命令行参数控制模拟次数、进程数、随机种子和输出文件。
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
import csv
from dataclasses import dataclass
import json
from multiprocessing import Pool
from os import cpu_count
from pathlib import Path
import random
from time import perf_counter_ns

from config import (
    Team,
    TournamentConfig,
    load_tournament_config,
    win_probability,
)


VALVE_ROUND_FOUR_FIVE_PAIRING_PRIORITY: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 5), (1, 4), (2, 3)),
    ((0, 5), (1, 3), (2, 4)),
    ((0, 4), (1, 5), (2, 3)),
    ((0, 4), (1, 3), (2, 5)),
    ((0, 3), (1, 5), (2, 4)),
    ((0, 3), (1, 4), (2, 5)),
    ((0, 5), (1, 2), (3, 4)),
    ((0, 4), (1, 2), (3, 5)),
    ((0, 2), (1, 5), (3, 4)),
    ((0, 2), (1, 4), (3, 5)),
    ((0, 3), (1, 2), (4, 5)),
    ((0, 2), (1, 3), (4, 5)),
    ((0, 1), (2, 5), (3, 4)),
    ((0, 1), (2, 4), (3, 5)),
    ((0, 1), (2, 3), (4, 5)),
)


@dataclass
class Record:
    """
    单支队伍在一次模拟中的即时战绩。

    teams_faced 用于计算 Buchholz 难度，也避免排序时只看初始种子。
    """

    wins: int
    losses: int
    teams_faced: set[Team]

    @staticmethod
    def new() -> Record:
        """创建一份空战绩。"""
        return Record(wins=0, losses=0, teams_faced=set())

    @property
    def diff(self) -> int:
        """胜负场差，瑞士轮分组和排序时会用到。"""
        return self.wins - self.losses


@dataclass
class Result:
    """单支队伍在多次模拟后的累计结果。"""

    three_zero: int
    advanced: int
    zero_three: int

    @staticmethod
    def new() -> Result:
        """创建一份空累计结果。"""
        return Result(three_zero=0, advanced=0, zero_three=0)

    def __add__(self, other: Result) -> Result:
        """合并两个进程返回的队伍统计。"""
        return Result(
            three_zero=self.three_zero + other.three_zero,
            advanced=self.advanced + other.advanced,
            zero_three=self.zero_three + other.zero_three,
        )


@dataclass
class SimulationSummary:
    """
    一批模拟的完整汇总。

    team_results 和 combination_counts 分开保存，避免把组合统计塞到某个队伍上。
    """

    team_results: dict[Team, Result]
    combination_counts: dict[str, int]

    @staticmethod
    def new(teams: tuple[Team, ...]) -> SimulationSummary:
        """按队伍列表初始化一份空汇总。"""
        return SimulationSummary(
            team_results={team: Result.new() for team in teams},
            combination_counts={},
        )

    def __add__(self, other: SimulationSummary) -> SimulationSummary:
        """合并两个进程的模拟汇总。"""
        team_results = {}
        for team in set(self.team_results) | set(other.team_results):
            team_results[team] = self.team_results.get(team, Result.new()) + other.team_results.get(
                team,
                Result.new(),
            )

        combination_counts = dict(self.combination_counts)
        for combination, count in other.combination_counts.items():
            combination_counts[combination] = combination_counts.get(combination, 0) + count

        return SimulationSummary(
            team_results=team_results,
            combination_counts=combination_counts,
        )


def bo3_match_probability(p_map: float) -> float:
    """
    根据单图胜率计算 BO3（三局两胜）胜率。

    推导结果为：赢前两图，或前两图一胜一负后赢决胜图。
    化简后是 p_map ** 2 * (3 - 2 * p_map)。
    """
    if not 0 <= p_map <= 1:
        raise ValueError(f"p_map 必须在 [0, 1] 范围内，当前值为 {p_map}")
    return p_map**2 * (3 - 2 * p_map)


@dataclass
class SwissSystem:
    """
    单次瑞士轮模拟器。

    remaining 只保留还未 3 胜晋级、也未 3 负淘汰的队伍。
    """

    sigma: tuple[float, ...]
    weights: tuple[float, ...]
    records: dict[Team, Record]
    remaining: set[Team]
    rng: random.Random

    def seeding(self, team: Team) -> tuple[int, int, int]:
        """
        计算当前轮排序键。

        排序优先级：
        1. 当前胜负场差；
        2. Buchholz 难度，即已交手对手的胜负场差总和；
        3. 初始种子排名。
        """
        return (
            -self.records[team].diff,
            -sum(self.records[opp].diff for opp in self.records[team].teams_faced),
            team.seed,
        )

    def simulate_match(self, team_a: Team, team_b: Team) -> None:
        """
        模拟一场比赛。

        2 胜晋级局和 2 负淘汰局按 BO3 处理，其余轮次按 BO1 处理。
        """
        record_a = self.records[team_a]
        record_b = self.records[team_b]
        is_bo3 = (
            record_a.wins == 2
            or record_a.losses == 2
            or record_b.wins == 2
            or record_b.losses == 2
        )

        # BO1 直接使用单图胜率；BO3 先折算成三局两胜的比赛胜率。
        p_map = win_probability(team_a, team_b, self.sigma, weights=self.weights)
        p_match = bo3_match_probability(p_map) if is_bo3 else p_map
        team_a_win = p_match > self.rng.random()

        if team_a_win:
            record_a.wins += 1
            record_b.losses += 1
        else:
            record_a.losses += 1
            record_b.wins += 1

        record_a.teams_faced.add(team_b)
        record_b.teams_faced.add(team_a)

        # BO3 结束后才会产生 3 胜或 3 负队伍，将其移出后续轮次。
        if is_bo3:
            for team in (team_a, team_b):
                record = self.records[team]
                if record.wins == 3 or record.losses == 3:
                    self.remaining.discard(team)

    def has_played(self, team_a: Team, team_b: Team) -> bool:
        """判断两支队伍此前是否已经交手。"""
        return team_b in self.records[team_a].teams_faced

    def current_round(self) -> int:
        """根据剩余队伍已完成的比赛数推断当前瑞士轮轮次。"""
        if not self.remaining:
            return 0
        return max(self.records[team].wins + self.records[team].losses for team in self.remaining) + 1

    def pairing_repeat_count(self, pairs: list[tuple[Team, Team]]) -> int:
        """统计一组候选配对中会产生多少次重复交手。"""
        return sum(int(self.has_played(team_a, team_b)) for team_a, team_b in pairs)

    def select_pairing(self, candidates: list[list[tuple[Team, Team]]]) -> list[tuple[Team, Team]]:
        """
        按 Valve 优先顺序选择配对。

        优先返回第一组完全不重赛的配对；若所有候选都会重赛，则选择重赛次数最少的一组。
        重赛次数相同时保留候选列表中的 Valve 优先顺序。
        """
        best_pairs: list[tuple[Team, Team]] | None = None
        best_repeat_count: int | None = None

        for pairs in candidates:
            repeat_count = self.pairing_repeat_count(pairs)
            if repeat_count == 0:
                return pairs
            if best_repeat_count is None or repeat_count < best_repeat_count:
                best_repeat_count = repeat_count
                best_pairs = pairs

        return best_pairs or []

    def high_low_pairing_candidates(self, group: list[Team]) -> list[list[tuple[Team, Team]]]:
        """生成第 2/3 轮高种子对低种子的候选配对，顺序即 Valve 优先级。"""
        if not group:
            return [[]]

        team_a = group[0]
        candidates: list[list[tuple[Team, Team]]] = []
        for candidate_index in range(len(group) - 1, 0, -1):
            team_b = group[candidate_index]
            remaining = group[1:candidate_index] + group[candidate_index + 1 :]
            for rest_pairs in self.high_low_pairing_candidates(remaining):
                candidates.append([(team_a, team_b), *rest_pairs])
        return candidates

    def priority_table_pairing_candidates(self, group: list[Team]) -> list[list[tuple[Team, Team]]]:
        """按 Valve 第 4/5 轮六队优先表生成候选配对。"""
        return [
            [(group[index_a], group[index_b]) for index_a, index_b in row]
            for row in VALVE_ROUND_FOUR_FIVE_PAIRING_PRIORITY
        ]

    def pair_initial_round(self, group: list[Team]) -> list[tuple[Team, Team]]:
        """第一轮固定为 1v9、2v10、...、8v16。"""
        ordered = sorted(group, key=self.seeding)
        midpoint = len(ordered) // 2
        return list(zip(ordered[:midpoint], ordered[midpoint:]))

    def pair_group(self, group: list[Team], round_number: int) -> list[tuple[Team, Team]]:
        """
        为同一战绩组生成 Valve Major 瑞士轮配对。

        第 2/3 轮按高种子对最低可用种子；第 4/5 轮按 Valve 六队优先表。
        """
        if len(group) < 2:
            return []
        if len(group) % 2 != 0:
            raise ValueError(f"瑞士轮同战绩组队伍数必须为偶数，当前为 {len(group)}")

        ordered = sorted(group, key=self.seeding)
        if round_number in (2, 3):
            return self.select_pairing(self.high_low_pairing_candidates(ordered))

        if round_number in (4, 5):
            if len(ordered) != 6:
                raise ValueError(
                    f"Valve 第 {round_number} 轮配对要求每个非空同战绩组有 6 支队伍，"
                    f"当前为 {len(ordered)} 支"
                )
            return self.select_pairing(self.priority_table_pairing_candidates(ordered))

        raise ValueError(f"不支持第 {round_number} 轮瑞士轮配对")

    def pair_round(self) -> list[tuple[Team, Team]]:
        """根据本轮开始时的战绩和种子快照生成整轮对阵。"""
        round_number = self.current_round()
        ordered_remaining = sorted(self.remaining, key=self.seeding)

        if round_number == 1:
            return self.pair_initial_round(ordered_remaining)

        groups: dict[tuple[int, int], list[Team]] = {}
        for team in ordered_remaining:
            record = self.records[team]
            groups.setdefault((record.wins, record.losses), []).append(team)

        round_pairs: list[tuple[Team, Team]] = []
        for group in groups.values():
            round_pairs.extend(self.pair_group(group, round_number))
        return round_pairs

    def simulate_round(self) -> None:
        """根据当前战绩分组并模拟一轮对阵。"""
        for team_a, team_b in self.pair_round():
            self.simulate_match(team_a, team_b)

    def simulate_tournament(self) -> None:
        """持续模拟轮次，直到所有队伍晋级或淘汰。"""
        while self.remaining:
            self.simulate_round()


def format_combination_key(
    three_zero_teams: list[str],
    advanced_teams: list[str],
    zero_three_teams: list[str],
) -> str:
    """把一次模拟的 Pick'Em 结果格式化为稳定文本键。"""
    return (
        f"3-0: {', '.join(sorted(three_zero_teams))} | "
        f"3-1/3-2: {', '.join(sorted(advanced_teams))} | "
        f"0-3: {', '.join(sorted(zero_three_teams))}"
    )


def parse_team_names(text: str) -> list[str]:
    """解析组合文本中的队伍列表，自动忽略空项。"""
    return [team.strip() for team in text.split(",") if team.strip()]


def parse_combination_key(combination: str) -> dict[str, list[str]]:
    """
    将旧版文本组合键解析成结构化字段。

    这个函数只解析 format_combination_key 生成的内部格式，用于 JSONL 输出。
    """
    if not combination.startswith("3-0: "):
        raise ValueError(f"无法解析组合键：{combination}")

    three_zero_text, rest = combination.removeprefix("3-0: ").split(" | 3-1/3-2: ", 1)
    advanced_text, zero_three_text = rest.split(" | 0-3: ", 1)
    return {
        "three_zero": parse_team_names(three_zero_text),
        "advanced": parse_team_names(advanced_text),
        "zero_three": parse_team_names(zero_three_text),
    }


class Simulation:
    """赛事模拟器，负责加载配置并调度单进程/多进程模拟。"""

    def __init__(self, filepath: str | Path) -> None:
        self.tournament = load_tournament_config(filepath)
        self.sigma = self.tournament.sigma
        self.weights = self.tournament.weights
        self.teams = self.tournament.teams

    def batch(self, n: int, seed: int | None = None) -> SimulationSummary:
        """
        运行 n 次模拟。

        seed 只作用于当前批次；多进程时每个进程会拿到不同 seed。
        """
        rng = random.Random(seed)
        summary = SimulationSummary.new(self.teams)

        for _ in range(n):
            swiss = SwissSystem(
                sigma=self.sigma,
                weights=self.weights,
                records={team: Record.new() for team in self.teams},
                remaining=set(self.teams),
                rng=rng,
            )
            swiss.simulate_tournament()

            three_zero_teams = []
            advanced_teams = []
            zero_three_teams = []

            for team, record in swiss.records.items():
                if record.wins == 3 and record.losses == 0:
                    summary.team_results[team].three_zero += 1
                    three_zero_teams.append(team.name)
                elif record.wins == 3:
                    summary.team_results[team].advanced += 1
                    advanced_teams.append(team.name)
                elif record.losses == 3 and record.wins == 0:
                    summary.team_results[team].zero_three += 1
                    zero_three_teams.append(team.name)

            combination_key = format_combination_key(
                three_zero_teams,
                advanced_teams,
                zero_three_teams,
            )
            summary.combination_counts[combination_key] = (
                summary.combination_counts.get(combination_key, 0) + 1
            )

        return summary

    def run(self, n: int, workers: int, seed: int | None = None) -> SimulationSummary:
        """
        使用 workers 个进程运行 n 次模拟。

        注意：固定 seed 可以保证同样的 workers 数下结果可复现；改变 workers 数会改变分批方式。
        """
        if n <= 0:
            raise ValueError("模拟次数必须大于 0")

        workers = max(1, min(workers, n))
        iterations = [n // workers for _ in range(workers)]
        for index in range(n % workers):
            iterations[index] += 1

        if workers == 1:
            return self.batch(iterations[0], seed)

        with Pool(workers) as pool:
            futures = []
            for index, batch_size in enumerate(iterations):
                worker_seed = None if seed is None else seed + index
                futures.append(pool.apply_async(self.batch, [batch_size, worker_seed]))

            summaries = [future.get() for future in futures]

        merged = summaries[0]
        for summary in summaries[1:]:
            merged = merged + summary
        return merged


def default_worker_count() -> int:
    """默认保留一个 CPU 核心给系统，避免本机完全跑满。"""
    available = cpu_count() or 1
    return max(1, available - 1)


def default_output_path(tournament: TournamentConfig) -> Path:
    """
    根据真实参数生成输出文件名。

    文件名包含实际权重，以及 JSON 中每个评分系统的 sigma。
    """
    weight_text = "_".join(f"{value:.4f}" for value in tournament.weights)
    sigma_text = "_".join(f"{value:.4f}" for value in tournament.sigma)
    return Path(f"{weight_text}_{sigma_text}.txt")


def write_combination_results(
    summary: SimulationSummary,
    n: int,
    output_path: str | Path,
) -> Path:
    """
    将组合频率按出现次数降序写入结果文件。

    默认写旧版文本格式；当输出文件扩展名为 .jsonl 时，写结构化 JSON Lines。
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sorted_combinations = sorted(
        summary.combination_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    if path.suffix.lower() == ".jsonl":
        with path.open("w", encoding="utf-8") as file:
            for combination, count in sorted_combinations:
                record = parse_combination_key(combination)
                record.update(
                    {
                        "count": count,
                        "total": n,
                        "probability": count / n,
                    }
                )
                file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    else:
        with path.open("w", encoding="utf-8") as file:
            for combination, count in sorted_combinations:
                file.write(f"{combination}: {count}/{n} ({count / n * 100:.4f}%)\n")

    return path


def write_team_summary(
    summary: SimulationSummary,
    teams: tuple[Team, ...],
    total: int,
    output_path: str | Path,
) -> Path:
    """按配置文件中的队伍顺序输出队伍单项概率 CSV。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
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
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for team in teams:
            result = summary.team_results[team]
            qualified_count = result.three_zero + result.advanced
            writer.writerow(
                {
                    "team": team.name,
                    "three_zero_count": result.three_zero,
                    "advanced_count": result.advanced,
                    "qualified_count": qualified_count,
                    "zero_three_count": result.zero_three,
                    "three_zero_probability": result.three_zero / total,
                    "advanced_probability": result.advanced / total,
                    "qualified_probability": qualified_count / total,
                    "zero_three_probability": result.zero_three / total,
                    "total": total,
                }
            )

    return path


def format_results(
    summary: SimulationSummary,
    n: int,
    run_time: float,
    output_path: str | Path,
    team_summary_path: str | Path | None = None,
    teams: tuple[Team, ...] | None = None,
) -> list[str]:
    """格式化命令行输出，并写入组合统计文件。"""
    path = write_combination_results(summary, n, output_path)
    out = [
        f"已进行 {n:,} 次瑞士轮模拟",
        f"共出现 {len(summary.combination_counts):,} 种 Pick'Em 结果组合",
        f"组合统计已写入: {path}",
        f"运行耗时: {run_time:.4f} 秒",
    ]

    if team_summary_path is not None:
        if teams is None:
            raise ValueError("输出队伍汇总 CSV 时必须传入 teams")
        summary_path = write_team_summary(summary, teams, n, team_summary_path)
        out.insert(3, f"队伍单项概率已写入: {summary_path}")

    return out


def parse_args() -> Namespace:
    """解析命令行参数。"""
    parser = ArgumentParser(description="CS2 Major Pick'Em 瑞士轮模拟器")
    parser.add_argument(
        "-i",
        "--input",
        default="major_stage.json",
        help="赛事配置 JSON 文件路径，默认 major_stage.json",
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=100_000,
        help="模拟次数，默认 100000；如需大样本可传 10000000",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=default_worker_count(),
        help="进程数，默认使用 CPU 核心数减一",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子；设置后同样的进程数可复现结果",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="输出结果文件；默认根据权重和真实 sigma 自动命名",
    )
    parser.add_argument(
        "--team-summary",
        default=None,
        help="输出队伍单项概率 CSV 的路径",
    )
    return parser.parse_args()


def main() -> None:
    """脚本入口。"""
    args = parse_args()
    simulation = Simulation(args.input)
    output_path = Path(args.output) if args.output else default_output_path(simulation.tournament)

    start = perf_counter_ns()
    summary = simulation.run(args.iterations, args.workers, args.seed)
    run_time = (perf_counter_ns() - start) / 1_000_000_000

    print(
        "\n".join(
            format_results(
                summary,
                args.iterations,
                run_time,
                output_path,
                team_summary_path=args.team_summary,
                teams=simulation.teams,
            )
        )
    )


if __name__ == "__main__":
    main()
