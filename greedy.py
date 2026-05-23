"""
CS2 Major Pick'Em 候选组合评估器。

说明：
这个脚本使用启发式方法生成候选 Pick'Em 组合，然后基于模拟结果文件评估
“预测正确数 >= 5”的概率。它不是全组合穷举，因此输出含义是“候选集内最佳”。
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections import defaultdict
import itertools
import re
from pathlib import Path
from typing import Dict, List, Tuple

from config import Team, load_tournament_config, win_probability


DEFAULT_RESULTS_FILE = "0.5000_0.5000_600.0000_1600.0000.txt"


def parse_team_group(text: str) -> frozenset[str]:
    """把结果文件中的队伍列表文本解析成集合，自动忽略空白项。"""
    return frozenset(team.strip() for team in text.split(",") if team.strip())


def parse_simulation_results(file_path: str | Path) -> tuple[dict, int]:
    """
    解析模拟结果文件。

    文件行格式：
    3-0: A, B | 3-1/3-2: C, D ... | 0-3: X, Y: 123/100000 (0.1230%)
    """
    results = defaultdict(int)
    total_simulations = 0
    pattern = re.compile(r"3-0: (.*?) \| 3-1/3-2: (.*?) \| 0-3: (.*?): (\d+)/\d+")

    with Path(file_path).open("r", encoding="utf-8") as file:
        for line in file:
            match = pattern.match(line)
            if not match:
                continue

            three_zero = parse_team_group(match.group(1))
            three_one_two = parse_team_group(match.group(2))
            zero_three = parse_team_group(match.group(3))
            count = int(match.group(4))

            key = (three_zero, three_one_two, zero_three)
            results[key] += count
            total_simulations += count

    if total_simulations == 0:
        raise ValueError(f"结果文件 {file_path} 中没有解析到有效模拟结果")

    return dict(results), total_simulations


def calculate_team_probabilities(
    teams: List[Team],
    sigma: Tuple[float, ...],
) -> Dict[str, Dict[str, float]]:
    """
    估算每个队伍获得 3-0、3-1/3-2、0-3 的启发式概率。

    注意：这里不是重新跑瑞士轮，而是用队伍间胜率快速估计，用于生成候选组合。
    最终排序仍以模拟结果文件中的真实频次为准。
    """
    probabilities = {}

    for team in teams:
        three_zero_prob = 1.0
        zero_three_prob = 1.0

        for other in teams:
            if other == team:
                continue

            win_prob = win_probability(team, other, sigma)
            three_zero_prob *= win_prob
            zero_three_prob *= 1 - win_prob

        advanced_prob = max(0.0, 1 - three_zero_prob - zero_three_prob)
        probabilities[team.name] = {
            "3-0": three_zero_prob,
            "3-1/3-2": advanced_prob,
            "0-3": zero_three_prob,
        }

    return probabilities


def generate_candidate_combinations(
    team_stats: dict,
    results: dict,
    top_n: int = 5,
) -> list[dict[str, set[str]]]:
    """
    生成候选预测组合。

    候选来源包括：
    1. 直接按单项概率贪心；
    2. 3-0、0-3、晋级区的局部替换；
    3. 模拟结果文件中最高频的实际组合。
    """
    candidates = []
    teams = list(team_stats.keys())

    greedy_3_0 = sorted(teams, key=lambda name: team_stats[name]["3-0"], reverse=True)[:2]
    greedy_0_3 = sorted(teams, key=lambda name: team_stats[name]["0-3"], reverse=True)[:2]
    greedy_pool = [team for team in teams if team not in greedy_3_0 and team not in greedy_0_3]
    greedy_adv = sorted(
        greedy_pool,
        key=lambda name: team_stats[name]["3-1/3-2"],
        reverse=True,
    )[:6]

    candidates.append(
        {
            "3-0": set(greedy_3_0),
            "3-1/3-2": set(greedy_adv),
            "0-3": set(greedy_0_3),
        }
    )

    # 3-0 变种：在最高概率的 top_n 支队伍里尝试不同二人组。
    top_3_0 = sorted(teams, key=lambda name: team_stats[name]["3-0"], reverse=True)[:top_n]
    for combo in itertools.combinations(top_3_0, 2):
        if set(combo) == set(greedy_3_0):
            continue
        remaining = [team for team in teams if team not in combo and team not in greedy_0_3]
        advanced = sorted(
            remaining,
            key=lambda name: team_stats[name]["3-1/3-2"],
            reverse=True,
        )[:6]
        candidates.append(
            {
                "3-0": set(combo),
                "3-1/3-2": set(advanced),
                "0-3": set(greedy_0_3),
            }
        )

    # 0-3 变种：在最高概率的 top_n 支队伍里尝试不同二人组。
    top_0_3 = sorted(teams, key=lambda name: team_stats[name]["0-3"], reverse=True)[:top_n]
    for combo in itertools.combinations(top_0_3, 2):
        if set(combo) == set(greedy_0_3):
            continue
        remaining = [team for team in teams if team not in greedy_3_0 and team not in combo]
        advanced = sorted(
            remaining,
            key=lambda name: team_stats[name]["3-1/3-2"],
            reverse=True,
        )[:6]
        candidates.append(
            {
                "3-0": set(greedy_3_0),
                "3-1/3-2": set(advanced),
                "0-3": set(combo),
            }
        )

    # 晋级区变种：替换贪心晋级名单中末尾 1-2 支队伍。
    top_adv = sorted(
        greedy_pool,
        key=lambda name: team_stats[name]["3-1/3-2"],
        reverse=True,
    )
    if len(top_adv) > 6:
        for to_remove in range(1, 3):
            replacements = top_adv[len(greedy_adv) : len(greedy_adv) + to_remove]
            if len(replacements) < to_remove:
                continue
            new_adv = greedy_adv[:-to_remove] + replacements
            candidates.append(
                {
                    "3-0": set(greedy_3_0),
                    "3-1/3-2": set(new_adv),
                    "0-3": set(greedy_0_3),
                }
            )

    # 高频实际组合：直接把模拟结果里的高频组合加入候选池。
    sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
    for (three_zero, three_one_two, zero_three), _count in sorted_results[:50]:
        candidates.append(
            {
                "3-0": set(three_zero),
                "3-1/3-2": set(three_one_two),
                "0-3": set(zero_three),
            }
        )

    # 去重，避免同一个组合被不同策略重复评估。
    unique_candidates = []
    seen = set()
    for candidate in candidates:
        key = (
            frozenset(candidate["3-0"]),
            frozenset(candidate["3-1/3-2"]),
            frozenset(candidate["0-3"]),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)

    return unique_candidates


def evaluate_candidate(candidate: dict, results: dict) -> float:
    """评估候选组合在模拟结果中预测正确数 >= 5 的概率。"""
    correct_ge5_count = 0
    total_simulations = sum(results.values())
    if total_simulations <= 0:
        raise ValueError("模拟结果为空，无法评估候选组合")

    for (three_zero, three_one_two, zero_three), count in results.items():
        correct = 0

        for team in candidate["3-0"]:
            if team in three_zero:
                correct += 1

        for team in candidate["3-1/3-2"]:
            if team in three_one_two:
                correct += 1

        for team in candidate["0-3"]:
            if team in zero_three:
                correct += 1

        if correct >= 5:
            correct_ge5_count += count

    return correct_ge5_count / total_simulations


def find_best_candidate(
    file_path: str | Path,
    teams: List[Team],
    sigma: Tuple[float, ...],
    top_n: int = 5,
) -> tuple[dict | None, float, list[tuple[dict, float]], int]:
    """
    寻找候选集内预测正确数 >= 5 概率最高的组合。

    Returns:
        tuple: (最佳候选, 最佳概率, 所有候选评估结果, 模拟总次数)
    """
    results, total_simulations = parse_simulation_results(file_path)
    team_stats = calculate_team_probabilities(teams, sigma)
    candidates = generate_candidate_combinations(team_stats, results, top_n)

    best_combination = None
    best_probability = -1.0
    evaluation_results = []

    for candidate in candidates:
        probability = evaluate_candidate(candidate, results)
        evaluation_results.append((candidate, probability))

        if probability > best_probability:
            best_probability = probability
            best_combination = candidate

    evaluation_results.sort(key=lambda item: item[1], reverse=True)
    return best_combination, best_probability, evaluation_results, total_simulations


def find_optimal_combination(file_path: str | Path, teams: List[Team]) -> tuple:
    """
    兼容旧代码的函数名。

    旧名称叫 optimal，但实际算法是候选集评估；新代码请优先使用 find_best_candidate。
    """
    # 旧入口不知道 sigma，只能使用配置默认值；保留它是为了不让外部导入直接失效。
    tournament = load_tournament_config("major_stage.json")
    best, probability, results, _total = find_best_candidate(file_path, teams, tournament.sigma)
    return best, probability, results


def parse_args() -> Namespace:
    """解析命令行参数。"""
    parser = ArgumentParser(description="CS2 Major Pick'Em 候选组合评估器")
    parser.add_argument(
        "-r",
        "--results",
        default=DEFAULT_RESULTS_FILE,
        help="simulate.py 生成的结果文件路径",
    )
    parser.add_argument(
        "-i",
        "--input",
        default="major_stage.json",
        help="赛事配置 JSON 文件路径，默认 major_stage.json",
    )
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=5,
        help="生成候选时每个分区参考的高概率队伍数量，默认 5",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="打印前多少个候选组合，默认 10",
    )
    return parser.parse_args()


def main() -> None:
    """脚本入口。"""
    args = parse_args()
    tournament = load_tournament_config(args.input)
    teams = list(tournament.teams)

    _best_combination, _best_probability, all_results, total_simulations = find_best_candidate(
        args.results,
        teams,
        tournament.sigma,
        args.candidate_pool,
    )

    print(f"候选组合排名（基于 {total_simulations:,} 次模拟结果）:")
    for index, (candidate, probability) in enumerate(all_results[: args.top], 1):
        print(f"\nNO.{index}: 预测正确数 >= 5 的概率 = {probability:.4f}")
        print(f"  3-0 晋级: {', '.join(sorted(candidate['3-0']))}")
        print(f"  3-1/3-2 晋级: {', '.join(sorted(candidate['3-1/3-2']))}")
        print(f"  0-3 淘汰: {', '.join(sorted(candidate['0-3']))}")


if __name__ == "__main__":
    main()
