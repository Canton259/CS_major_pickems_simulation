# CS Major Swiss Pick'Em Simulator
# CS Major 瑞士轮竞猜模拟器

This is a Python program for simulating CS2 Major Swiss stages. The current default data file is configured for IEM Cologne Major 2026 Stage 1.

这是一个用于模拟 CS2 Major 瑞士轮阶段的 Python 程序。当前默认数据文件已配置为 IEM Cologne Major 2026 Stage 1。

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
  - Best candidate Pick'Em combination search / 候选集内最佳竞猜组合搜索
  - Probability-based team selection / 基于概率的队伍选择
  - Top 10 combinations ranking / Top 10 组合排名

- Swiss System Tournament / 瑞士轮赛制
  - Buchholz system implementation / Buchholz系统实现
  - Seeding-based matchmaking / 基于种子的对阵
  - Round-by-round progression / 逐轮晋级机制

- Multi-process Computing / 多进程计算
  - Parallel simulation execution / 并行模拟执行
  - CPU core utilization / CPU核心利用
  - Performance optimization / 性能优化

- Customizable Parameters / 可自定义参数
  - VRS/HLTV rating weights / VRS/HLTV评分权重
  - Sigma value adjustment / Sigma值调整
  - Team data configuration / 队伍数据配置

## Installation
## 安装说明

```bash
pip install -r requirements.txt
```

## Usage
## 使用方法

1. Configure Parameters / 配置参数:
   - Set team information and stage sigma in `major_stage.json` / 在 `major_stage.json` 中设置队伍信息和阶段 sigma
   - Adjust `VRS_WEIGHT` and `HLTV_WEIGHT` in `config.py` / 在 `config.py` 中调整 `VRS_WEIGHT` 和 `HLTV_WEIGHT`

2. Run Simulation / 运行模拟:
```bash
python simulate.py
```

You can also specify simulation parameters / 也可以指定模拟参数:
```bash
python simulate.py --input major_stage.json --iterations 100000 --workers 8 --seed 42
```

3. View Results / 查看结果:
   - Default file naming format / 默认文件名格式: `VRS_WEIGHT_HLTV_WEIGHT_VALVE_SIGMA_HLTV_SIGMA.txt`
   - Example / 示例: `0.5000_0.5000_600.0000_1600.0000.txt`
   - You can override the output path with `--output` / 可以通过 `--output` 指定输出路径

4. Solve Pick'Em Combinations / 竞猜组合求解:
```bash
python greedy.py --results 0.5000_0.5000_600.0000_1600.0000.txt
```

## Parameters
## 参数说明

- `VRS_WEIGHT`: VRS/Valve rating weight in `config.py` (Default: 0.5) / `config.py` 中的 VRS/Valve 评分权重（默认: 0.5）
- `HLTV_WEIGHT`: HLTV rating weight in `config.py` (Default: 0.5) / `config.py` 中的 HLTV 评分权重（默认: 0.5）
- `sigma.valve`: Valve Elo sigma in `major_stage.json` (Current default: 600) / `major_stage.json` 中的 Valve Elo 标准差参数（当前默认: 600）
- `sigma.hltv`: HLTV sigma reserved for stage configuration (Current default: 1600) / `major_stage.json` 中预留的 HLTV 标准差参数（当前默认: 1600）
- `SIGMA`: Compatibility fallback in `config.py` (Default: 349.2). Normal simulations use `major_stage.json` first. / `config.py` 中的兼容回退值（默认: 349.2）。正常模拟优先使用 `major_stage.json`。
- `--iterations`: Number of Monte Carlo simulations (Default: 100000) / 蒙特卡洛模拟次数（默认: 100000）
- `--workers`: Number of worker processes (Default: CPU cores minus one) / 并行进程数（默认: CPU 核心数减一）
- `--seed`: Random seed for reproducible runs with the same worker count / 随机种子；在相同进程数下可复现实验
- `--output`: Simulation result output path / 模拟结果输出路径

## License
## 许可证

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。 
