# quant-btc

## 项目概述

BTC/USDT 永续合约量化交易系统，采用**符号主义规则引擎**架构：策略由可读的规则与参数组合表达，避免黑盒模型；指标 → 信号 → 仓位 → 风控 全链路可解释、可回溯。

## 技术栈

- **语言**：Python 3.12
- **包管理**：uv
- **数据处理**：Polars（主）+ pandas-ta（指标库）
- **行情接入**：ccxt（Binance 永续）
- **存储**：Parquet 分区（按交易对 / 周期 / 日期）

## 开发规范

- **commit message 用中文**，按功能/修复独立提交，避免一次提交混合多个改动
- **参数集中到 YAML（`config/`），代码不硬编码**：阈值、周期、仓位比例等全部走配置；策略迭代只动 config，不动引擎
- **数据存储用 Parquet 分区**：按 `symbol/timeframe/date` 三级分区，支持增量追加；原始数据与衍生指标分目录
- **类型严格**：所有公共函数标注类型；优先 `pyright` / `mypy --strict` 通过
- **不入库**：原始 K 线数据、模型权重、`.env` 凭证（见 `.gitignore`）

## 目录结构

```
quant-btc/
├── src/
│   ├── data/          # 行情下载、增量同步、Parquet 读写
│   ├── indicators/    # 技术指标计算（封装 pandas-ta，统一 Polars 输出）
│   ├── engine/        # 规则引擎：信号合成、仓位决策
│   ├── backtest/      # 回测引擎：撮合、滑点、手续费、绩效
│   └── utils/         # 通用工具：时间、日志、配置加载
├── config/            # YAML 策略参数与运行配置
├── data/              # 本地数据存储（gitignore）
│   └── parquet/
│       └── BTCUSDT/
├── tests/
└── pyproject.toml
```

## 当前阶段

**Phase 4 已完成 — 指标/规则/回测引擎已搭建**：

- **Phase 1 数据管道**：vision 月度 ZIP 拉 OHLCV + REST 增量；fapi REST 拉资金费率（CN 网络受限可能为空）；vision daily metrics 拉 OI 按月聚合；alternative.me 拉贪婪恐慌指数。
- **Phase 2 指标层** `src/indicators/`：`IndicatorEngine` 基于 pandas-ta，输入输出统一 Polars。覆盖趋势/动量/波动率/成交量四大类共 15 个指标；`compute_all(config)` 批量计算；`crossover/crossunder` 工具函数。
- **Phase 3 规则引擎** `src/engine/`：YAML 描述策略，支持阈值比较、cross above/below、from_above/from_below 状态记忆、AND/OR + 嵌套条件组、跨周期对齐、信号方向冲突仲裁（按 YAML 顺序优先级）。
- **Phase 4 回测引擎** `src/backtest/`：杠杆永续合约逐 K 线推进；同向加仓/反向先平后开；止损止盈、强平、最大回撤熔断、日内最大亏损、资金费率结算；输出净值曲线 + 总收益/年化/夏普/最大回撤/胜率/盈亏比/平均持仓时间。

入口：
- `uv run python scripts/download_all.py [--test]` 全量数据下载
- `uv run python scripts/run_backtest.py [--strategy ... --backtest ... --csv ...]` 一键回测

下阶段方向：参数搜索 / 多策略组合管理 / 实盘信号桥接。
