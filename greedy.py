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
import json
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

from config import Team, load_tournament_config, win_probability


DEFAULT_RESULTS_FILE = "0.7000_0.3000_600.0000_1600.0000.txt"
PICK_KEYS = ("3-0", "3-1/3-2", "0-3")
PICK_GROUP_SIZES = {
    "3-0": 2,
    "3-1/3-2": 6,
    "0-3": 2,
}
PICKEM_SUCCESS_THRESHOLD = 5
TEAM_COUNT = 16
INCLUSION_EXCLUSION_COEFFICIENTS = {
    size: (-1) ** (size - PICKEM_SUCCESS_THRESHOLD)
    * math.comb(size - 1, PICKEM_SUCCESS_THRESHOLD - 1)
    for size in range(PICKEM_SUCCESS_THRESHOLD, 11)
}
_EXHAUSTIVE_KERNELS = None


def parse_team_group(text: str) -> frozenset[str]:
    """把结果文件中的队伍列表文本解析成集合，自动忽略空白项。"""
    return frozenset(team.strip() for team in text.split(",") if team.strip())


def normalize_team_group(value: object) -> frozenset[str]:
    """把 JSONL 或文本中的队伍列表规范化为 frozenset。"""
    if isinstance(value, str):
        return parse_team_group(value)

    if isinstance(value, list):
        normalized = []
        for team in value:
            if not isinstance(team, str):
                raise ValueError(f"队伍列表元素必须是字符串：{team!r}")
            team_name = team.strip()
            if team_name:
                normalized.append(team_name)
        return frozenset(normalized)

    raise ValueError(f"无法解析队伍列表：{value!r}")


def candidate_key(candidate: dict) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """生成候选组合去重键，保持 evaluate_candidate 使用的返回结构不变。"""
    validate_candidate(candidate)
    return (
        frozenset(candidate["3-0"]),
        frozenset(candidate["3-1/3-2"]),
        frozenset(candidate["0-3"]),
    )


def validate_candidate(candidate: dict) -> None:
    """Validate Pick'Em group sizes and make sure groups do not overlap."""
    missing = [key for key in PICK_KEYS if key not in candidate]
    if missing:
        raise ValueError(f"候选组合缺少字段: {', '.join(missing)}")

    groups = {key: set(candidate[key]) for key in PICK_KEYS}
    for key, expected_size in PICK_GROUP_SIZES.items():
        actual_size = len(groups[key])
        if actual_size != expected_size:
            raise ValueError(f"{key} 必须包含 {expected_size} 支队伍，当前为 {actual_size}")

    total_size = sum(len(group) for group in groups.values())
    union_size = len(set().union(*groups.values()))
    if union_size != total_size:
        raise ValueError("候选组合中的 3-0、晋级和 0-3 队伍不能重叠")


def validate_jsonl_integer(
    value: object,
    field_name: str,
    file_path: str | Path,
    line_number: int,
    *,
    positive: bool,
) -> int:
    """校验 JSONL 中的整数统计字段，并把错误定位到具体文件行号。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{file_path} 第 {line_number} 行字段 {field_name!r} 必须是整数")

    if positive and value <= 0:
        raise ValueError(f"{file_path} 第 {line_number} 行字段 {field_name!r} 必须是正整数")
    if not positive and value < 0:
        raise ValueError(f"{file_path} 第 {line_number} 行字段 {field_name!r} 必须是非负整数")

    return value


def parse_simulation_results(file_path: str | Path) -> tuple[dict, int]:
    """
    解析模拟结果文件。

    支持两种格式：
    1. 旧版文本行：
    3-0: A, B | 3-1/3-2: C, D ... | 0-3: X, Y: 123/100000 (0.1230%)
    2. JSON Lines：
    {"three_zero": [...], "advanced": [...], "zero_three": [...], "count": 123}
    """
    path = Path(file_path)
    if path.suffix.lower() == ".jsonl":
        return parse_simulation_results_jsonl(path)

    return parse_simulation_results_text(path)


def parse_simulation_results_text(file_path: str | Path) -> tuple[dict, int]:
    """解析旧版文本模拟结果文件。"""
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


def parse_simulation_results_jsonl(file_path: str | Path) -> tuple[dict, int]:
    """解析 JSON Lines 结构化模拟结果文件。"""
    results = defaultdict(int)
    total_simulations = 0
    required_fields = ("three_zero", "advanced", "zero_three", "count")

    with Path(file_path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{file_path} 第 {line_number} 行不是有效 JSON") from exc

            if not isinstance(record, dict):
                raise ValueError(f"{file_path} 第 {line_number} 行必须是 JSON 对象")

            for field_name in required_fields:
                if field_name not in record:
                    raise ValueError(
                        f"{file_path} 第 {line_number} 行缺少字段 {field_name!r}"
                    )

            try:
                three_zero = normalize_team_group(record["three_zero"])
                three_one_two = normalize_team_group(record["advanced"])
                zero_three = normalize_team_group(record["zero_three"])
            except ValueError as exc:
                raise ValueError(f"{file_path} 第 {line_number} 行：{exc}") from exc

            count = validate_jsonl_integer(
                record["count"],
                "count",
                file_path,
                line_number,
                positive=False,
            )
            if "total" in record:
                validate_jsonl_integer(
                    record["total"],
                    "total",
                    file_path,
                    line_number,
                    positive=True,
                )

            key = (three_zero, three_one_two, zero_three)
            results[key] += count
            total_simulations += count

    if total_simulations == 0:
        raise ValueError(f"结果文件 {file_path} 中没有解析到有效模拟结果")

    return dict(results), total_simulations


def calculate_team_probabilities(
    teams: List[Team],
    sigma: Tuple[float, ...],
    weights: Tuple[float, ...] | None = None,
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

            win_prob = win_probability(team, other, sigma, weights=weights)
            three_zero_prob *= win_prob
            zero_three_prob *= 1 - win_prob

        advanced_prob = max(0.0, 1 - three_zero_prob - zero_three_prob)
        probabilities[team.name] = {
            "3-0": three_zero_prob,
            "3-1/3-2": advanced_prob,
            "0-3": zero_three_prob,
        }

    return probabilities


def calculate_team_probabilities_from_results(
    teams: List[Team],
    results: dict,
) -> Dict[str, Dict[str, float]]:
    """
    从模拟结果文件统计每支队伍的边际概率。

    这比基于 pairwise 胜率的快速估算更贴近真实瑞士轮路径；
    因为它已经包含了配对、BO1/BO3、晋级和淘汰路径的影响。
    """
    total_simulations = sum(results.values())
    if total_simulations <= 0:
        raise ValueError("模拟结果为空，无法统计队伍边际概率")

    probabilities = {
        team.name: {
            "3-0": 0.0,
            "3-1/3-2": 0.0,
            "0-3": 0.0,
        }
        for team in teams
    }

    for (three_zero, three_one_two, zero_three), count in results.items():
        for team_name in three_zero:
            if team_name in probabilities:
                probabilities[team_name]["3-0"] += count

        for team_name in three_one_two:
            if team_name in probabilities:
                probabilities[team_name]["3-1/3-2"] += count

        for team_name in zero_three:
            if team_name in probabilities:
                probabilities[team_name]["0-3"] += count

    for team_stats in probabilities.values():
        for result_key in team_stats:
            team_stats[result_key] /= total_simulations

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
    zero_three_pool = [team for team in teams if team not in greedy_3_0]
    greedy_0_3 = sorted(
        zero_three_pool,
        key=lambda name: team_stats[name]["0-3"],
        reverse=True,
    )[:2]
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
    top_3_0 = sorted(
        [team for team in teams if team not in greedy_0_3],
        key=lambda name: team_stats[name]["3-0"],
        reverse=True,
    )[:top_n]
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
    top_0_3 = sorted(
        [team for team in teams if team not in greedy_3_0],
        key=lambda name: team_stats[name]["0-3"],
        reverse=True,
    )[:top_n]
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
        key = candidate_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)

    return unique_candidates


def generate_random_candidate_combinations(
    teams: List[Team],
    candidate_count: int,
    seed: int | None = None,
) -> list[dict[str, set[str]]]:
    """随机生成合法 Pick'Em 组合，作为启发式候选集之外的补充探索。"""
    if candidate_count <= 0:
        raise ValueError("--random-candidates 必须大于 0")

    team_names = [team.name for team in teams]
    if len(team_names) < 10:
        raise ValueError("随机模式至少需要 10 支队伍才能生成合法 Pick'Em 组合")

    rng = random.Random(seed)
    candidates = []
    seen = set()
    attempts = 0
    max_attempts = max(candidate_count * 10, candidate_count + 1000)

    while len(candidates) < candidate_count and attempts < max_attempts:
        attempts += 1
        shuffled = team_names[:]
        rng.shuffle(shuffled)
        candidate = {
            "3-0": set(shuffled[:2]),
            "3-1/3-2": set(shuffled[2:8]),
            "0-3": set(shuffled[8:10]),
        }
        key = candidate_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)

    return candidates


def evaluate_candidate(candidate: dict, results: dict) -> float:
    """评估候选组合在模拟结果中预测正确数 >= 5 的概率。"""
    validate_candidate(candidate)
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


def evaluate_candidate_by_inclusion_exclusion(candidate: dict, results: dict) -> float:
    """
    Score a candidate through the same subset formula used by exhaustive mode.

    This intentionally favors clarity over speed and is used as a small test
    oracle for the optimized exhaustive implementation.
    """
    validate_candidate(candidate)
    total_simulations = sum(results.values())
    if total_simulations <= 0:
        raise ValueError("模拟结果为空，无法评估候选组合")

    occurrence_counts: dict[frozenset[tuple[str, str]], int] = defaultdict(int)
    for (three_zero, advanced, zero_three), count in results.items():
        labeled_teams = [
            *((team_name, "3-0") for team_name in three_zero),
            *((team_name, "3-1/3-2") for team_name in advanced),
            *((team_name, "0-3") for team_name in zero_three),
        ]
        for subset_size in INCLUSION_EXCLUSION_COEFFICIENTS:
            for subset in itertools.combinations(labeled_teams, subset_size):
                occurrence_counts[frozenset(subset)] += count

    candidate_labels = [
        *((team_name, "3-0") for team_name in candidate["3-0"]),
        *((team_name, "3-1/3-2") for team_name in candidate["3-1/3-2"]),
        *((team_name, "0-3") for team_name in candidate["0-3"]),
    ]
    success_count = 0
    for subset_size, coefficient in INCLUSION_EXCLUSION_COEFFICIENTS.items():
        for subset in itertools.combinations(candidate_labels, subset_size):
            success_count += coefficient * occurrence_counts.get(frozenset(subset), 0)

    return success_count / total_simulations


def team_names(teams: List[Team]) -> list[str]:
    """Return team names in config order."""
    return [team.name for team in teams]


def mask_to_team_set(mask: int, names: list[str]) -> set[str]:
    """Convert a 16-bit team mask back to the public candidate dict format."""
    return {name for index, name in enumerate(names) if mask & (1 << index)}


def masks_to_candidate(
    three_zero_mask: int,
    advanced_mask: int,
    zero_three_mask: int,
    names: list[str],
) -> dict[str, set[str]]:
    """Convert internal masks back to a Pick'Em candidate dictionary."""
    return {
        "3-0": mask_to_team_set(three_zero_mask, names),
        "3-1/3-2": mask_to_team_set(advanced_mask, names),
        "0-3": mask_to_team_set(zero_three_mask, names),
    }


def group_to_mask(group: frozenset[str], team_index: dict[str, int], label: str) -> int:
    """Convert one parsed result group to a bit mask with clear validation."""
    expected_size = PICK_GROUP_SIZES[label]
    if len(group) != expected_size:
        raise ValueError(f"{label} 结果必须包含 {expected_size} 支队伍，当前为 {len(group)}")

    mask = 0
    for team_name in group:
        if team_name not in team_index:
            raise ValueError(f"结果文件包含未知队伍: {team_name}")
        bit = 1 << team_index[team_name]
        if mask & bit:
            raise ValueError(f"{label} 结果包含重复队伍: {team_name}")
        mask |= bit
    return mask


def load_numeric_dependencies():
    """Lazy-load numeric dependencies so heuristic/random modes stay lightweight."""
    try:
        import numpy as np
        import numba
    except ImportError as exc:
        raise ImportError(
            "exhaustive 搜索需要安装 numpy 和 numba，请先运行 pip install -r requirements.txt"
        ) from exc

    return np, numba


def result_masks_from_results(
    results: dict,
    teams: List[Team],
):
    """Convert parsed result records into compact NumPy arrays for exhaustive search."""
    np, _numba = load_numeric_dependencies()
    names = team_names(teams)
    if len(names) != TEAM_COUNT:
        raise ValueError(f"exhaustive 搜索当前只支持 {TEAM_COUNT} 支队伍")

    team_index = {name: index for index, name in enumerate(names)}
    three_zero_masks = np.empty(len(results), dtype=np.uint16)
    advanced_masks = np.empty(len(results), dtype=np.uint16)
    zero_three_masks = np.empty(len(results), dtype=np.uint16)
    counts = np.empty(len(results), dtype=np.int32)

    for row_index, ((three_zero, advanced, zero_three), count) in enumerate(results.items()):
        if count < 0:
            raise ValueError("模拟结果次数必须为非负数")
        three_zero_mask = group_to_mask(three_zero, team_index, "3-0")
        advanced_mask = group_to_mask(advanced, team_index, "3-1/3-2")
        zero_three_mask = group_to_mask(zero_three, team_index, "0-3")
        if (three_zero_mask | advanced_mask | zero_three_mask).bit_count() != 10:
            raise ValueError("结果文件中的 3-0、晋级和 0-3 队伍不能重叠")

        three_zero_masks[row_index] = three_zero_mask
        advanced_masks[row_index] = advanced_mask
        zero_three_masks[row_index] = zero_three_mask
        counts[row_index] = count

    return three_zero_masks, advanced_masks, zero_three_masks, counts


def build_rank_data(np):
    """Build mask rank tables and fixed-size mask lists for the 16-team search space."""
    rank_by_size = np.full((TEAM_COUNT + 1, 1 << TEAM_COUNT), -1, dtype=np.int32)
    masks_by_size: list[list[int]] = [[] for _ in range(TEAM_COUNT + 1)]
    bit_count = np.empty(1 << TEAM_COUNT, dtype=np.uint8)

    for mask in range(1 << TEAM_COUNT):
        size = mask.bit_count()
        bit_count[mask] = size
        rank_by_size[size, mask] = len(masks_by_size[size])
        masks_by_size[size].append(mask)

    masks2 = np.array(masks_by_size[2], dtype=np.uint16)
    masks6 = np.array(masks_by_size[6], dtype=np.uint16)
    return rank_by_size, bit_count, masks2, masks6


def build_exhaustive_type_tables(np):
    """Build compact offsets for all labeled subsets used by inclusion-exclusion."""
    type_offset = np.full((3, 7, 3), -1, dtype=np.int64)
    type_dim_b = np.zeros((3, 7, 3), dtype=np.int64)
    type_dim_c = np.zeros((3, 7, 3), dtype=np.int64)
    total_cells = 0

    for a_size in range(3):
        for b_size in range(7):
            for c_size in range(3):
                total_size = a_size + b_size + c_size
                if not PICKEM_SUCCESS_THRESHOLD <= total_size <= 10:
                    continue

                dim_a = math.comb(TEAM_COUNT, a_size)
                dim_b = math.comb(TEAM_COUNT, b_size)
                dim_c = math.comb(TEAM_COUNT, c_size)
                type_offset[a_size, b_size, c_size] = total_cells
                type_dim_b[a_size, b_size, c_size] = dim_b
                type_dim_c[a_size, b_size, c_size] = dim_c
                total_cells += dim_a * dim_b * dim_c

    coefficients = np.zeros(11, dtype=np.int64)
    for subset_size, coefficient in INCLUSION_EXCLUSION_COEFFICIENTS.items():
        coefficients[subset_size] = coefficient

    return type_offset, type_dim_b, type_dim_c, coefficients, total_cells


def compile_exhaustive_kernels():
    """Compile and cache Numba kernels for exact exhaustive scoring."""
    global _EXHAUSTIVE_KERNELS
    if _EXHAUSTIVE_KERNELS is not None:
        return _EXHAUSTIVE_KERNELS

    np, numba = load_numeric_dependencies()
    njit = numba.njit
    prange = numba.prange
    get_num_threads = numba.get_num_threads
    get_thread_id = numba.get_thread_id

    @njit
    def build_occurrence_kernel(
        result_three_zero_masks,
        result_advanced_masks,
        result_zero_three_masks,
        result_counts,
        rank_by_size,
        bit_count,
        type_offset,
        type_dim_b,
        type_dim_c,
        occurrence_counts,
    ):
        for row_index in range(result_counts.shape[0]):
            full_a = int(result_three_zero_masks[row_index])
            full_b = int(result_advanced_masks[row_index])
            full_c = int(result_zero_three_masks[row_index])
            count = int(result_counts[row_index])

            sub_a = full_a
            while True:
                a_size = int(bit_count[sub_a])
                rank_a = int(rank_by_size[a_size, sub_a])
                sub_b = full_b
                while True:
                    b_size = int(bit_count[sub_b])
                    rank_b = int(rank_by_size[b_size, sub_b])
                    sub_c = full_c
                    while True:
                        c_size = int(bit_count[sub_c])
                        total_size = a_size + b_size + c_size
                        if total_size >= PICKEM_SUCCESS_THRESHOLD:
                            offset = int(type_offset[a_size, b_size, c_size])
                            dim_b = int(type_dim_b[a_size, b_size, c_size])
                            dim_c = int(type_dim_c[a_size, b_size, c_size])
                            rank_c = int(rank_by_size[c_size, sub_c])
                            table_index = offset + (rank_a * dim_b + rank_b) * dim_c + rank_c
                            occurrence_counts[table_index] += count
                        if sub_c == 0:
                            break
                        sub_c = (sub_c - 1) & full_c
                    if sub_b == 0:
                        break
                    sub_b = (sub_b - 1) & full_b
                if sub_a == 0:
                    break
                sub_a = (sub_a - 1) & full_a

    @njit
    def score_candidate_kernel(
        full_a,
        full_b,
        full_c,
        rank_by_size,
        bit_count,
        type_offset,
        type_dim_b,
        type_dim_c,
        occurrence_counts,
        coefficients,
    ):
        score = 0
        sub_a = full_a
        while True:
            a_size = int(bit_count[sub_a])
            rank_a = int(rank_by_size[a_size, sub_a])
            sub_b = full_b
            while True:
                b_size = int(bit_count[sub_b])
                rank_b = int(rank_by_size[b_size, sub_b])
                sub_c = full_c
                while True:
                    c_size = int(bit_count[sub_c])
                    total_size = a_size + b_size + c_size
                    if total_size >= PICKEM_SUCCESS_THRESHOLD:
                        offset = int(type_offset[a_size, b_size, c_size])
                        dim_b = int(type_dim_b[a_size, b_size, c_size])
                        dim_c = int(type_dim_c[a_size, b_size, c_size])
                        rank_c = int(rank_by_size[c_size, sub_c])
                        table_index = offset + (rank_a * dim_b + rank_b) * dim_c + rank_c
                        score += int(coefficients[total_size]) * int(occurrence_counts[table_index])
                    if sub_c == 0:
                        break
                    sub_c = (sub_c - 1) & full_c
                if sub_b == 0:
                    break
                sub_b = (sub_b - 1) & full_b
            if sub_a == 0:
                break
            sub_a = (sub_a - 1) & full_a
        return score

    @njit
    def insert_thread_top(score, full_a, full_b, full_c, top_scores, top_a, top_b, top_c, thread_id):
        top_n = top_scores.shape[1]
        if score <= top_scores[thread_id, top_n - 1]:
            return

        insert_at = top_n - 1
        while insert_at > 0 and score > top_scores[thread_id, insert_at - 1]:
            top_scores[thread_id, insert_at] = top_scores[thread_id, insert_at - 1]
            top_a[thread_id, insert_at] = top_a[thread_id, insert_at - 1]
            top_b[thread_id, insert_at] = top_b[thread_id, insert_at - 1]
            top_c[thread_id, insert_at] = top_c[thread_id, insert_at - 1]
            insert_at -= 1

        top_scores[thread_id, insert_at] = score
        top_a[thread_id, insert_at] = full_a
        top_b[thread_id, insert_at] = full_b
        top_c[thread_id, insert_at] = full_c

    @njit(parallel=True)
    def exhaustive_top_kernel(
        masks2,
        masks6,
        rank_by_size,
        bit_count,
        type_offset,
        type_dim_b,
        type_dim_c,
        occurrence_counts,
        coefficients,
        top_n,
    ):
        thread_count = get_num_threads()
        top_scores = np.full((thread_count, top_n), -1, dtype=np.int64)
        top_a = np.zeros((thread_count, top_n), dtype=np.uint16)
        top_b = np.zeros((thread_count, top_n), dtype=np.uint16)
        top_c = np.zeros((thread_count, top_n), dtype=np.uint16)

        for a_index in prange(masks2.shape[0]):
            full_a = int(masks2[a_index])
            thread_id = get_thread_id()
            for c_index in range(masks2.shape[0]):
                full_c = int(masks2[c_index])
                blocked = full_a | full_c
                if full_a & full_c:
                    continue

                for b_index in range(masks6.shape[0]):
                    full_b = int(masks6[b_index])
                    if full_b & blocked:
                        continue

                    score = score_candidate_kernel(
                        full_a,
                        full_b,
                        full_c,
                        rank_by_size,
                        bit_count,
                        type_offset,
                        type_dim_b,
                        type_dim_c,
                        occurrence_counts,
                        coefficients,
                    )
                    insert_thread_top(
                        score,
                        full_a,
                        full_b,
                        full_c,
                        top_scores,
                        top_a,
                        top_b,
                        top_c,
                        thread_id,
                    )

        return top_scores, top_a, top_b, top_c

    _EXHAUSTIVE_KERNELS = build_occurrence_kernel, exhaustive_top_kernel
    return _EXHAUSTIVE_KERNELS


def find_best_candidate_exhaustive(
    results: dict,
    total_simulations: int,
    teams: List[Team],
    result_limit: int,
) -> tuple[dict | None, float, list[tuple[dict, float]], int]:
    """Find the global best Pick'Em candidates across the full legal search space."""
    if result_limit <= 0:
        raise ValueError("--top 必须大于 0")

    np, _numba = load_numeric_dependencies()
    result_arrays = result_masks_from_results(results, teams)
    rank_by_size, bit_count, masks2, masks6 = build_rank_data(np)
    type_offset, type_dim_b, type_dim_c, coefficients, total_cells = build_exhaustive_type_tables(np)
    if total_cells > np.iinfo(np.int32).max:
        raise MemoryError("exhaustive 搜索表过大，无法在当前 int32 索引下构建")

    occurrence_counts = np.zeros(total_cells, dtype=np.int32)
    build_occurrence_kernel, exhaustive_top_kernel = compile_exhaustive_kernels()
    build_occurrence_kernel(
        *result_arrays,
        rank_by_size,
        bit_count,
        type_offset,
        type_dim_b,
        type_dim_c,
        occurrence_counts,
    )

    top_scores, top_a, top_b, top_c = exhaustive_top_kernel(
        masks2,
        masks6,
        rank_by_size,
        bit_count,
        type_offset,
        type_dim_b,
        type_dim_c,
        occurrence_counts,
        coefficients,
        result_limit,
    )

    names = team_names(teams)
    scored_masks = []
    for thread_index in range(top_scores.shape[0]):
        for top_index in range(top_scores.shape[1]):
            score = int(top_scores[thread_index, top_index])
            if score < 0:
                continue
            scored_masks.append(
                (
                    score,
                    int(top_a[thread_index, top_index]),
                    int(top_b[thread_index, top_index]),
                    int(top_c[thread_index, top_index]),
                )
            )

    scored_masks.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    evaluation_results = [
        (
            masks_to_candidate(three_zero_mask, advanced_mask, zero_three_mask, names),
            score / total_simulations,
        )
        for score, three_zero_mask, advanced_mask, zero_three_mask in scored_masks[:result_limit]
    ]

    best_candidate, best_probability = (
        evaluation_results[0] if evaluation_results else (None, -1.0)
    )
    return best_candidate, best_probability, evaluation_results, total_simulations


def find_best_candidate(
    file_path: str | Path,
    teams: List[Team],
    sigma: Tuple[float, ...],
    top_n: int = 5,
    weights: Tuple[float, ...] | None = None,
    search_mode: str = "heuristic",
    random_candidates: int = 10_000,
    seed: int | None = None,
    result_limit: int = 10,
) -> tuple[dict | None, float, list[tuple[dict, float]], int]:
    """
    寻找候选集内预测正确数 >= 5 概率最高的组合。

    Returns:
        tuple: (最佳候选, 最佳概率, 所有候选评估结果, 模拟总次数)
    """
    results, total_simulations = parse_simulation_results(file_path)
    if search_mode == "heuristic":
        team_stats = calculate_team_probabilities_from_results(teams, results)
        candidates = generate_candidate_combinations(team_stats, results, top_n)
    elif search_mode == "random":
        candidates = generate_random_candidate_combinations(teams, random_candidates, seed)
    elif search_mode == "exhaustive":
        return find_best_candidate_exhaustive(results, total_simulations, teams, result_limit)
    else:
        raise ValueError(f"未知 search mode：{search_mode}")

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
    best, probability, results, _total = find_best_candidate(
        file_path,
        teams,
        tournament.sigma,
        weights=tournament.weights,
    )
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
        "--search-mode",
        choices=("heuristic", "random", "exhaustive"),
        default="heuristic",
        help=(
            "搜索模式：heuristic 使用快速候选集，random 随机生成合法组合，"
            "exhaustive 全局搜索所有合法组合"
        ),
    )
    parser.add_argument(
        "--random-candidates",
        type=int,
        default=10_000,
        help="random 模式生成的随机候选组合数量，默认 10000",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random 搜索模式的随机种子",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="打印前多少个组合；exhaustive 模式下也是全局 Top N，默认 10",
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
        weights=tournament.weights,
        search_mode=args.search_mode,
        random_candidates=args.random_candidates,
        seed=args.seed,
        result_limit=args.top,
    )

    print(f"候选组合排名（基于 {total_simulations:,} 次模拟结果）:")
    for index, (candidate, probability) in enumerate(all_results[: args.top], 1):
        print(f"\nNO.{index}: 预测正确数 >= 5 的概率 = {probability:.4f}")
        print(f"  3-0 晋级: {', '.join(sorted(candidate['3-0']))}")
        print(f"  3-1/3-2 晋级: {', '.join(sorted(candidate['3-1/3-2']))}")
        print(f"  0-3 淘汰: {', '.join(sorted(candidate['0-3']))}")


if __name__ == "__main__":
    main()
