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

**Phase 1 — 数据管道建设**：
- ccxt 拉取 BTCUSDT 永续历史 K 线（多周期：1m/5m/15m/1h/4h/1d）
- Parquet 分区落盘 + 增量追加机制
- 数据完整性校验（缺失检测、时间戳对齐）

后续阶段（不在 Phase 1 范围）：指标层 → 规则引擎 → 回测 → 实盘信号。
