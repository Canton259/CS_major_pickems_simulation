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
from dataclasses import dataclass
from multiprocessing import Pool
from os import cpu_count
from pathlib import Path
import random
from time import perf_counter_ns

from config import (
    HLTV_WEIGHT,
    VRS_WEIGHT,
    Team,
    TournamentConfig,
    load_tournament_config,
    win_probability,
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


@dataclass
class SwissSystem:
    """
    单次瑞士轮模拟器。

    remaining 只保留还未 3 胜晋级、也未 3 负淘汰的队伍。
    """

    sigma: tuple[float, ...]
    records: dict[Team, Record]
    remaining: set[Team]

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

        # p 是 team_a 的单图胜率；BO3 用三张图独立抽样近似。
        p = win_probability(team_a, team_b, self.sigma)
        if is_bo3:
            first_map = p > random.random()
            second_map = p > random.random()
            team_a_win = p > random.random() if first_map != second_map else first_map
        else:
            team_a_win = p > random.random()

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

    def simulate_round(self) -> None:
        """根据当前战绩分组并模拟一轮对阵。"""
        even_teams, pos_teams, neg_teams = [], [], []

        for team in sorted(self.remaining, key=self.seeding):
            if self.records[team].diff > 0:
                pos_teams.append(team)
            elif self.records[team].diff < 0:
                neg_teams.append(team)
            else:
                even_teams.append(team)

        # 第一轮固定为上半区对下半区：1-9、2-10、3-11 ...
        if len(even_teams) == len(self.records):
            for team_a, team_b in zip(even_teams, even_teams[len(even_teams) // 2 :]):
                self.simulate_match(team_a, team_b)
            return

        # 后续轮次每个战绩组内部配对。当前实现沿用原项目的种子/Buchholz排序策略。
        for group in (pos_teams, even_teams, neg_teams):
            second_half = reversed(group[len(group) // 2 :])
            for team_a, team_b in zip(group, second_half):
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


class Simulation:
    """赛事模拟器，负责加载配置并调度单进程/多进程模拟。"""

    def __init__(self, filepath: str | Path) -> None:
        self.tournament = load_tournament_config(filepath)
        self.sigma = self.tournament.sigma
        self.teams = self.tournament.teams

    def batch(self, n: int, seed: int | None = None) -> SimulationSummary:
        """
        运行 n 次模拟。

        seed 只作用于当前批次；多进程时每个进程会拿到不同 seed。
        """
        if seed is not None:
            random.seed(seed)

        summary = SimulationSummary.new(self.teams)

        for _ in range(n):
            swiss = SwissSystem(
                sigma=self.sigma,
                records={team: Record.new() for team in self.teams},
                remaining=set(self.teams),
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

    文件名包含 VRS 权重、HLTV 权重，以及 JSON 中每个评分系统的 sigma。
    """
    sigma_text = "_".join(f"{value:.4f}" for value in tournament.sigma)
    return Path(f"{VRS_WEIGHT:.4f}_{HLTV_WEIGHT:.4f}_{sigma_text}.txt")


def write_combination_results(
    summary: SimulationSummary,
    n: int,
    output_path: str | Path,
) -> Path:
    """将组合频率按出现次数降序写入结果文件。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sorted_combinations = sorted(
        summary.combination_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    with path.open("w", encoding="utf-8") as file:
        for combination, count in sorted_combinations:
            file.write(f"{combination}: {count}/{n} ({count / n * 100:.4f}%)\n")

    return path


def format_results(
    summary: SimulationSummary,
    n: int,
    run_time: float,
    output_path: str | Path,
) -> list[str]:
    """格式化命令行输出，并写入组合统计文件。"""
    path = write_combination_results(summary, n, output_path)
    return [
        f"已进行 {n:,} 次瑞士轮模拟",
        f"共出现 {len(summary.combination_counts):,} 种 Pick'Em 结果组合",
        f"组合统计已写入: {path}",
        f"运行耗时: {run_time:.4f} 秒",
    ]


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
    return parser.parse_args()


def main() -> None:
    """脚本入口。"""
    args = parse_args()
    simulation = Simulation(args.input)
    output_path = Path(args.output) if args.output else default_output_path(simulation.tournament)

    start = perf_counter_ns()
    summary = simulation.run(args.iterations, args.workers, args.seed)
    run_time = (perf_counter_ns() - start) / 1_000_000_000

    print("\n".join(format_results(summary, args.iterations, run_time, output_path)))


if __name__ == "__main__":
    main()
