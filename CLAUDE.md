# quant-btc

## 项目概述

BTC/USDT 永续合约量化交易系统，采用**符号主义规则引擎**架构：策略由可读的规则与参数组合表达，避免黑盒模型；指标 → 信号 → 仓位 → 风控 → 可视化 全链路可解释、可回溯。

**当前阶段：Phase 5 完成 — MVP 可用**：数据管道 / 指标层 / 规则引擎 / 回测引擎 / 可视化报告全部打通；可一键从配置文件描述策略到 HTML 报告输出。

## 技术栈

- **语言**：Python 3.12
- **包管理**：uv
- **数据处理**：Polars（主）+ pandas-ta（指标库）
- **行情接入**：Binance Vision CDN（历史 ZIP，`data.binance.vision`）+ Binance Futures REST（增量，`fapi.binance.com`，CN 网络受限）
- **存储**：Parquet 月度分区
- **可视化**：matplotlib（深色主题，CJK 字体回退链）
- **配置**：YAML（数据 / 策略 / 回测 三套独立 config）

## 目录结构

```
quant-btc/
├── src/
│   ├── data/          # 数据下载（OHLCV/资金费率/OI/FNG）+ Parquet 读写
│   ├── indicators/    # IndicatorEngine（pandas-ta，输入输出 Polars）
│   ├── engine/        # RuleEngine（YAML 策略 → Signal）
│   ├── backtest/      # Backtester + BacktestVisualizer
│   └── utils/         # DataConfig 等通用工具
├── scripts/
│   ├── download_all.py   # 一键全量下载（--test 冒烟）
│   ├── run_backtest.py   # 一键回测 + 可视化（--no-plot 跳过图）
│   └── inspect_data.py   # 数据完整性检查
├── config/
│   ├── data_config.yaml      # 数据下载参数
│   ├── strategies.yaml       # 策略规则定义
│   └── backtest_config.yaml  # 回测引擎参数
├── data/parquet/BTCUSDT/     # 数据存储（gitignore）
├── output/backtest_*/        # 回测产物（gitignore）
└── pyproject.toml
```

## 模块职责

| 模块 | 职责 |
|---|---|
| `data.downloader` | OHLCV 月度 ZIP（vision）+ REST 增量（fapi，CN 不可用） |
| `data.market_data` | 资金费率（vision/REST 双实现）、OI 月度聚合、FNG 全量 |
| `indicators.IndicatorEngine` | 15 个指标（趋势/动量/波动率/成交量四类）+ `compute_all` + `crossover/crossunder` |
| `engine.RuleEngine` | YAML 加载、阈值/cross/状态记忆/嵌套 AND-OR、跨周期对齐、信号方向冲突仲裁 |
| `backtest.Backtester` | 杠杆永续逐 K 线推进；同向加仓/反向先平后开；止损止盈/强平/最大回撤熔断/日内最大亏损/资金费率结算 |
| `backtest.BacktestVisualizer` | 净值+回撤、买卖点、月度热力图、指标摘要；输出 PNG + HTML 报告 |

## 使用方法

### 1. 安装依赖

```bash
uv sync
```

### 2. 数据下载

冒烟测试（仅 1d 2024-01..03 + FNG）：
```bash
uv run python scripts/download_all.py --test
```

全量下载（约 5–10 分钟，写入 `data/parquet/BTCUSDT/`）：
```bash
uv run python scripts/download_all.py
```

数据完整性检查：
```bash
uv run python scripts/inspect_data.py
```

### 3. 运行回测

默认配置（`config/strategies.yaml` + `config/backtest_config.yaml`）：
```bash
uv run python scripts/run_backtest.py
```

常用选项：
```bash
uv run python scripts/run_backtest.py \
    --strategy config/strategies.yaml \
    --backtest config/backtest_config.yaml \
    --output-dir output/my_run \
    --no-plot
```

### 4. 查看结果

每次回测在 `output/backtest_{YYYYmmdd_HHMMSS}/` 下生成：
- `report.html` — 入口页面，浏览器打开
- `equity_curve.png` — 净值曲线 + BTC 价格 + 回撤
- `trades_all.png` / `trades_recent90d.png` — 买卖点全程 / 近 90 天
- `monthly_returns.png` — 月度收益率热力图
- `metrics_summary.png` — 指标表格 + PnL 分布 + 持仓时长分布
- `trades.csv` — 全部交易记录

## 已知问题

- **`fapi.binance.com` 在中国大陆 DNS 级被墙**（`getaddrinfo failed`）：
  - 影响 1：OHLCV REST 增量（当月未结束的部分）拉不到，只能等 vision 月度归档（每月初）
  - 影响 2：资金费率 REST 拉不到 → 已切换到 vision 月度 ZIP（`download_funding_rate_vision`）
  - 解决：需访问实时数据时挂代理，或使用境外节点
- **OI 早期数据缺失**：vision daily metrics 起点约 2020-09，2020-01..2020-08 数据不存在
- **FNG 偶发历史 gap**：alternative.me 早期一处 4 天缺口（2018-04），不影响近期数据

## 开发规范

- **commit message 用中文**，按功能/修复独立提交，避免一次提交混合多个改动
- **参数集中到 YAML（`config/`）**：阈值、周期、仓位、风控全部走配置；策略迭代只动 config，不动引擎代码
- **数据存储用 Parquet 月度分区**：按 `symbol/{tf}_{YYYY}_{MM}.parquet` 命名，幂等跳过已有月份
- **类型严格**：所有公共函数标注类型；`from __future__ import annotations` 启用延迟求值
- **不入库**：数据 / 模型 / 回测产物 / `.env` 凭证（见 `.gitignore`）

## 后续方向

- 参数搜索：网格 / 贝叶斯优化策略阈值
- 多策略组合管理：跨策略风险预算、相关性控制
- 实盘信号桥接：RuleEngine 输出 → Binance API 下单
- 更多衍生指标：跨市场套利、链上数据、宏观因子
