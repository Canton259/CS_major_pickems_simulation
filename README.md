# CS Major Swiss Pick'Em Simulator
# CS Major 瑞士轮竞猜模拟器

This is a Python program for simulating CS2 Major Swiss stages. The current default data file is configured for IEM Cologne Major 2026 Stage 1.

这是一个用于模拟 CS2 Major 瑞士轮阶段的 Python 程序。当前默认数据文件已配置为 IEM Cologne Major 2026 Stage 1。

## Version
## 版本说明

Current release: **v0.2.1**

当前版本：**v0.2.1**

v0.2.1 updates the IEM Cologne Major 2026 Stage 1 data model and default inputs:

v0.2.1 更新了 IEM Cologne Major 2026 Stage 1 的默认数据和地图模型：

- Updated team `valve` values to HLTV Valve Global Ranking points from 2026-05-25 / 将队伍 `valve` 值更新为 HLTV Valve Global Ranking 2026-05-25 points
- Rechecked `hltv` values against the latest available HLTV World Ranking, which remains 2026-05-18 at update time / 重新核对 `hltv` 值；更新时最新可用 HLTV World Ranking 仍为 2026-05-18
- Uses `VRS_WEIGHT = 0.7` and `HLTV_WEIGHT = 0.3` for the default rating blend / 默认评分融合使用 `VRS_WEIGHT = 0.7` 和 `HLTV_WEIGHT = 0.3`
- Replaced per-map First Pick / First Ban fields with `win_rate`, `pick_rate`, `ban_rate`, and `maps_played` from the user-provided HLTV past-3-month map data / 使用用户提供的 HLTV 近 3 个月地图数据替代 First Pick / First Ban 字段
- Infers each team's opening ban from the highest `ban_rate`; maps with no recent sample are stored with `ban_rate = 1.0` / 每队首 ban 由最高 `ban_rate` 推导；无近期样本地图写入 `ban_rate = 1.0`
- Keeps txt/jsonl output formats and `greedy.py` parsing compatibility unchanged / 保持 txt/jsonl 输出格式和 `greedy.py` 解析兼容性不变

## Project References
## 项目参考

This project is developed based on the following open-source projects:

本项目基于以下开源项目开发:

- [Major Pick'ems Simulator](https://github.com/ndunnett/major-pickems-sim) - Reference Project / 参考项目
- [CS2 Major Rules](https://github.com/ValveSoftware/counter-strike_rules_and_regs) - Swiss System Rules / 瑞士轮规则
- [CS2 Regional Standings](https://github.com/ValveSoftware/counter-strike_regional_standings) - VRS System / VRS系统

## Features
## 功能特点

- Monte Carlo Simulation / 蒙特卡洛模拟
  - Large-scale tournament simulation / 大规模比赛模拟
  - Probability-based match outcomes / 基于概率的比赛结果
  - Statistical analysis of results / 结果统计分析

- Greedy Algorithm / 贪心算法
  - Fast candidate Pick'Em combination search / 快速候选竞猜组合搜索
  - Heuristic, random, and exhaustive search modes / 支持启发式、随机和全局穷举搜索模式
  - Probability-based team selection / 基于概率的队伍选择
  - Top 10 combinations ranking / Top 10 组合排名

- Swiss System Tournament / 瑞士轮赛制
  - Buchholz system implementation / Buchholz系统实现
  - Seeding-based matchmaking / 基于种子的对阵
  - Round-by-round progression / 逐轮晋级机制
  - Currently supports 16-team Swiss stages / 当前支持 16 队瑞士轮阶段

- Map Veto Model / 地图 veto 模型
  - Heuristic BO1 and BO3 Major map veto simulation / 启发式模拟 Major BO1 和 BO3 地图 ban/pick
  - Configurable global `map_pool` in `major_stage.json` / 可在 `major_stage.json` 中配置全局地图池
  - Per-team `map_stats` with recalculated map strength, WR, pick/ban rate, and maps played / 每支队伍可配置 `map_stats` 地图强度、WR、pick/ban 率和地图样本数

- Multi-process Computing / 多进程计算
  - Parallel simulation execution / 并行模拟执行
  - CPU core utilization / CPU核心利用
  - Performance optimization / 性能优化

- Customizable Parameters / 可自定义参数
  - Default VRS/HLTV rating weights in `config.py` / `config.py` 中的默认 VRS/HLTV评分权重
  - Sigma value adjustment / Sigma值调整
  - Team data configuration / 队伍数据配置

## Installation
## 安装说明

Recommended Python version / 推荐 Python 版本: **Python >= 3.10**

The base simulator uses the Python standard library. Exhaustive Pick'Em search in `greedy.py` uses NumPy and Numba.

基础模拟器使用 Python 标准库。`greedy.py` 的全局穷举竞猜搜索使用 NumPy 和 Numba。

```bash
pip install -r requirements.txt
```

## Usage
## 使用方法

1. Configure Parameters / 配置参数:
   - Set team information and stage sigma in `major_stage.json` / 在 `major_stage.json` 中设置队伍信息和阶段 sigma
   - Rating weights use `VRS_WEIGHT` and `HLTV_WEIGHT` in `config.py` by default / 评分权重默认使用 `config.py` 中的 `VRS_WEIGHT` 和 `HLTV_WEIGHT`
   - `map_pool` configures the seven-map Major pool; if missing, the simulator uses Dust2, Mirage, Inferno, Nuke, Overpass, Ancient, and Anubis / `map_pool` 配置七图 Major 地图池；缺失时使用默认地图池
   - Team `map_stats` can set `strength`, `win_rate`, `pick_rate`, `ban_rate`, and `maps_played`; missing strength defaults to `0.5`, while missing rates and sample counts default to `0.0` / 每队 `map_stats` 可配置地图强度、WR、pick/ban 率和地图样本数

2. Run Simulation / 运行模拟:
```bash
python simulate.py
```

You can also specify simulation parameters / 也可以指定模拟参数:
```bash
python simulate.py --input major_stage.json --iterations 1000000 --workers 8 --seed 42
```

3. View Results / 查看结果:
   - Default file naming format / 默认文件名格式: `VALVE_WEIGHT_HLTV_WEIGHT_VALVE_SIGMA_HLTV_SIGMA.txt`
   - Example / 示例: `0.7000_0.3000_600.0000_1600.0000.txt`
   - You can override the output path with `--output` / 可以通过 `--output` 指定输出路径
   - Default output is txt / 默认输出为 txt 文本格式
   - Use a `.jsonl` output path for structured JSON Lines results / 输出路径使用 `.jsonl` 后缀时会生成结构化 JSON Lines 结果
   - `greedy.py` supports both txt and jsonl result files / `greedy.py` 同时支持 txt 和 jsonl 结果文件
   - Use `--team-summary` to output per-team probability CSV / 使用 `--team-summary` 输出队伍单项概率 CSV

4. Solve Pick'Em Combinations / 竞猜组合求解:
```bash
python greedy.py --results 0.7000_0.3000_600.0000_1600.0000.txt
```

Heuristic search is the default. Random search can be used as an additional exploration mode.

启发式搜索是默认模式。随机搜索可作为补充探索模式使用。

```bash
python greedy.py --results result.txt --search-mode heuristic
python greedy.py --results result.txt --search-mode random --random-candidates 10000 --seed 42
python greedy.py --results result.txt --search-mode exhaustive --top 10
```

5. Run Tests / 运行测试:
```bash
python -m unittest discover
```

GitHub Actions runs the unit tests and smoke checks for txt/jsonl simulation outputs on every push and pull request.

GitHub Actions 会在每次 push 和 pull request 时运行单元测试，并检查 txt/jsonl 模拟输出流程。

The workflow is defined in `.github/workflows/test.yml` / 工作流定义在 `.github/workflows/test.yml`。

## Parameters
## 参数说明

- `VRS_WEIGHT`: Default Valve/VRS rating weight in `config.py` (Default: 0.7) / `config.py` 中的默认 Valve/VRS 评分权重（默认: 0.7）
- `HLTV_WEIGHT`: Default HLTV rating weight in `config.py` (Default: 0.3) / `config.py` 中的默认 HLTV 评分权重（默认: 0.3）
- `sigma.valve`: Valve Elo sigma in `major_stage.json` (Current default: 600) / `major_stage.json` 中的 Valve Elo 标准差参数（当前默认: 600）
- `sigma.hltv`: HLTV Elo sigma in `major_stage.json` (Current default: 1600) / `major_stage.json` 中的 HLTV Elo 标准差参数（当前默认: 1600）
- `map_pool`: Global seven-map pool in `major_stage.json`; missing field uses the built-in default pool / `major_stage.json` 中的全局七图地图池；缺失时使用内置默认地图池
- `teams.*.map_stats`: Optional per-team map stats. `strength` adjusts map win probability; `win_rate`, `pick_rate`, and `ban_rate` are decimals from 0.0 to 1.0; `maps_played` is a non-negative integer / 每队可选地图统计，`strength` 修正地图胜率，其余字段引导 veto 和记录样本量
- `SIGMA`: Compatibility fallback in `config.py` (Default: 349.2). Normal simulations use `major_stage.json` first. / `config.py` 中的兼容回退值（默认: 349.2）。正常模拟优先使用 `major_stage.json`。
- Team count: the current simulator only supports 16-team Swiss stages / 队伍数量：当前模拟器只支持 16 队瑞士轮
- `--iterations`: Number of Monte Carlo simulations (Default: 1000000) / 蒙特卡洛模拟次数（默认: 1000000）
- `--workers`: Number of worker processes (Default: CPU cores minus one) / 并行进程数（默认: CPU 核心数减一）
- `--seed`: Random seed for reproducible runs with the same worker count / 随机种子；在相同进程数下可复现实验
- `--output`: Simulation result output path / 模拟结果输出路径
- `--team-summary`: Per-team probability CSV output path / 队伍单项概率 CSV 输出路径
- `greedy.py --search-mode`: Candidate search mode. `heuristic` is default; `random` generates legal random Pick'Em combinations; `exhaustive` searches all legal combinations globally / 候选搜索模式。`heuristic` 为默认；`random` 会生成合法随机竞猜组合；`exhaustive` 会全局搜索所有合法组合
- `greedy.py --random-candidates`: Number of random candidates generated in random mode (Default: 10000) / random 模式生成的随机候选数量（默认: 10000）
- `greedy.py --seed`: Random seed for greedy random search / greedy 随机搜索模式的随机种子

## Probability Model
## 胜率模型

- Valve/VRS uses an Elo/logistic formula / Valve/VRS 使用 Elo/logistic 公式
- HLTV uses an Elo/logistic formula / HLTV 使用 Elo/logistic 公式
- `VRS_WEIGHT` and `HLTV_WEIGHT` control how much each rating system contributes / `VRS_WEIGHT` 和 `HLTV_WEIGHT` 控制各评分系统在最终胜率中的权重
- `sigma` controls how strongly rating differences affect win probability / `sigma` 控制评分差对胜率的影响强度
- Match probabilities are clamped to avoid overconfidence / 概率会被裁剪，避免模型过度自信

## Map Veto
## 地图 veto

- The simulator supports heuristic Major map veto for BO1 and BO3 matches / 模拟器支持 BO1 和 BO3 的启发式 Major 地图 veto。
- BO1 follows: higher seed chooses Team A by default, Team A removes 2 maps, Team B removes 3 maps, Team A removes 1 map, and the remaining map is played / BO1 流程为高种子默认选择 Team A，Team A ban 2 图，Team B ban 3 图，Team A ban 1 图，剩余 1 图进行比赛。
- BO3 follows: Team A removes 1, Team B removes 1, Team A picks map 1, Team B picks map 2, Team B removes 1, Team A removes 1, and the remaining map is the decider / BO3 流程为 A ban、B ban、A pick、B pick、B ban、A ban，剩余图作为决胜图。
- Each team's first ban round prioritizes the currently available map with its highest `ban_rate`; if all available ban rates are `0`, it falls back to the 80/20 score / 每队首个 ban 回合优先移除当前可用图中 `ban_rate` 最高的地图，若全为 0 则回退到 80/20 评分。
- If both teams share the same inferred first-ban map, that map is reserved for the lower seed team's first ban / 如果双方推导出的首 ban 地图相同，该图保留给低种子队伍执行首 ban。
- Veto choices use an 80/20 deterministic score: 80% HLTV historical pick/ban tendency and 20% relative map-strength advantage, with ties resolved by map name / veto 使用 80% HLTV 历史 pick/ban 倾向 + 20% 地图相对优势的确定性评分，同分按地图名排序。
- Current map data uses the user-provided HLTV past-3-month txt: WR, Pick %, Ban %, and W-L. `maps_played` is wins plus losses / 当前地图数据使用用户提供的 HLTV 近 3 个月 txt：WR、Pick%、Ban% 和 W-L，`maps_played` 为胜场加负场。
- `strength` is recalculated as `(WR * 0.55 + Pick% * 0.25 - Ban% * 0.20 + sample adjustment) / 100`, clamped to `[0, 1]` / `strength` 按用户公式重算并裁剪到 `[0, 1]`。
- Maps with no past-3-month sample or not listed in the source txt are stored as `strength=0.0`, `win_rate=0.0`, `pick_rate=0.0`, `ban_rate=1.0`, and `maps_played=0` / 无近 3 个月样本或未列出的地图写入 `ban_rate=1.0` 等缺失值。
- First Pick is no longer handled as a special veto priority / First Pick 不再作为特殊 veto 优先级处理。
- This still does not model psychology, anti-stratting, side choice, or roster-specific historical context / 这仍不模拟心理博弈、针对性准备、side choice 或阵容变化。
- Map strength gently adjusts the base VRS/HLTV win probability and does not replace the rating model / 地图强度只是对基础 VRS/HLTV 胜率做温和修正，不会替代原有评分模型。

## Output Formats
## 输出格式

- Combination results default to txt format / 组合结果默认输出 txt 格式
- If `--output` ends with `.jsonl`, results are written as JSON Lines / 如果 `--output` 以 `.jsonl` 结尾，则输出 JSON Lines
- `greedy.py` can parse both txt and jsonl result files / `greedy.py` 可解析 txt 和 jsonl 两种结果文件
- `--team-summary PATH` writes a UTF-8 CSV with per-team 3-0, advanced, qualified, and 0-3 counts/probabilities / `--team-summary PATH` 会输出 UTF-8 CSV，包含每队 3-0、晋级、总晋级、0-3 的次数和概率

Team summary CSV columns / 队伍汇总 CSV 字段:

```text
team,three_zero_count,advanced_count,qualified_count,zero_three_count,three_zero_probability,advanced_probability,qualified_probability,zero_three_probability,total
```

## Limitations
## 模型局限性

- The current simulator only supports 16-team Swiss stages / 当前模拟器只支持 16 队瑞士轮阶段
- Map veto uses historical HLTV pick/ban rates but remains heuristic and does not model deeper team-specific veto strategy / 地图 veto 使用 HLTV 历史 pick/ban 率，但仍是启发式模拟
- It does not fully account for recent form, roster changes, patch/meta updates, travel, or LAN conditions / 当前未完整考虑近期状态、阵容变动、版本更新、旅行和 LAN 状态
- Monte Carlo results depend heavily on the input win-probability model / 蒙特卡洛结果高度依赖输入胜率模型

## License
## 许可证

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。 
