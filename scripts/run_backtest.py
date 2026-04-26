"""一键回测入口。

流程：
1. 从 data/parquet/{symbol}/ 读取所有 timeframe 的 parquet（合并 monthlies + current）
2. 按策略 YAML 中引用的指标列名，反推需计算的指标 → IndicatorEngine.compute_all
3. 加载策略 + 回测配置 → Backtester.run
4. 打印摘要并把交易记录导出 CSV

用法：
    uv run python scripts/run_backtest.py
    uv run python scripts/run_backtest.py --strategy config/strategies.yaml \
        --backtest config/backtest_config.yaml --csv out/trades.csv
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

import polars as pl
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from backtest import Backtester  # noqa: E402
from indicators import IndicatorEngine  # noqa: E402
from utils.config import DataConfig  # noqa: E402


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_ohlcv(cfg: DataConfig, timeframe: str) -> pl.DataFrame | None:
    """读取某 timeframe 下所有 parquet 并按 timestamp 去重排序。"""
    pat_monthly = re.compile(rf"^{re.escape(timeframe)}_\d{{4}}_\d{{2}}\.parquet$")
    files = [
        f for f in cfg.symbol_dir.glob(f"{timeframe}_*.parquet")
        if pat_monthly.match(f.name) or f.name == f"{timeframe}_current.parquet"
    ]
    if not files:
        return None
    parts = [pl.read_parquet(f) for f in files]
    return pl.concat(parts).unique(subset=["timestamp"]).sort("timestamp")


# 策略 YAML 列名 → IndicatorEngine 配置的解析表
_INDICATOR_PATTERNS: list[tuple[re.Pattern[str], str, callable]] = [
    (re.compile(r"^sma_(\d+)$"), "sma", lambda m: {"period": int(m[1])}),
    (re.compile(r"^ema_(\d+)$"), "ema", lambda m: {"period": int(m[1])}),
    (re.compile(r"^rsi_(\d+)$"), "rsi", lambda m: {"period": int(m[1])}),
    (re.compile(r"^atr_(\d+)$"), "atr", lambda m: {"period": int(m[1])}),
    (re.compile(r"^cci_(\d+)$"), "cci", lambda m: {"period": int(m[1])}),
    (re.compile(r"^williams_r_(\d+)$"), "williams_r", lambda m: {"period": int(m[1])}),
    (re.compile(r"^mfi_(\d+)$"), "mfi", lambda m: {"period": int(m[1])}),
    (re.compile(r"^cmf_(\d+)$"), "cmf", lambda m: {"period": int(m[1])}),
    (re.compile(r"^adx_(\d+)$"), "adx", lambda m: {"period": int(m[1])}),
    (re.compile(r"^dmp_(\d+)$"), "adx", lambda m: {"period": int(m[1])}),
    (re.compile(r"^dmn_(\d+)$"), "adx", lambda m: {"period": int(m[1])}),
    (
        re.compile(r"^macd_(line|signal|histogram)_(\d+)_(\d+)_(\d+)$"),
        "macd",
        lambda m: {"fast": int(m[2]), "slow": int(m[3]), "signal": int(m[4])},
    ),
    (
        re.compile(r"^bb_(upper|middle|lower)_(\d+)_([0-9.]+)$"),
        "bollinger",
        lambda m: {"period": int(m[2]), "std_dev": float(m[3])},
    ),
    (
        re.compile(r"^stoch_[kd]_(\d+)_(\d+)_(\d+)$"),
        "stoch",
        lambda m: {"k_period": int(m[1]), "d_period": int(m[2]), "smooth_k": int(m[3])},
    ),
    (
        re.compile(r"^kc_(upper|middle|lower)_(\d+)_(\d+)_([0-9.]+)$"),
        "keltner",
        lambda m: {"period": int(m[2]), "atr_period": int(m[3]), "multiplier": float(m[4])},
    ),
    (re.compile(r"^obv$"), "obv", lambda m: {}),
    (re.compile(r"^vwap$"), "vwap", lambda m: {}),
]


def _parse_indicator_name(name: str) -> tuple[str, dict[str, Any]] | None:
    for pat, key, builder in _INDICATOR_PATTERNS:
        m = pat.match(name)
        if m:
            return key, builder(m)
    return None


def _collect_required_indicators(strategies: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """递归扫描策略条件树，收集所有引用到的指标列名 → IndicatorEngine 配置。"""
    cfg: dict[str, dict[str, Any]] = {}

    def walk(cond: dict[str, Any]) -> None:
        if "conditions" in cond:
            for sub in cond["conditions"]:
                walk(sub)
            return
        for field in ("indicator", "reference"):
            v = cond.get(field)
            if not isinstance(v, str):
                continue
            parsed = _parse_indicator_name(v)
            if parsed is None:
                continue
            key, params = parsed
            cfg.setdefault(key, params)

    for s in strategies:
        for c in s.get("conditions", []):
            walk(c)
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="BTC 量化回测入口")
    parser.add_argument("--strategy", default=str(PROJECT_ROOT / "config" / "strategies.yaml"))
    parser.add_argument("--backtest", default=str(PROJECT_ROOT / "config" / "backtest_config.yaml"))
    parser.add_argument("--data-config", default=str(PROJECT_ROOT / "config" / "data_config.yaml"))
    parser.add_argument("--csv", default=str(PROJECT_ROOT / "out" / "trades.csv"))
    args = parser.parse_args()

    _setup_logging()
    log = logging.getLogger("run_backtest")

    data_cfg = DataConfig.from_yaml(args.data_config)
    bt = Backtester.from_yaml(args.backtest)

    with open(args.strategy, "r", encoding="utf-8") as f:
        strat_yaml = yaml.safe_load(f) or {}
    strategies = strat_yaml.get("strategies", [])
    if not strategies:
        log.error("策略文件为空：%s", args.strategy)
        return 1

    # 收集所有策略涉及的 timeframe
    used_tfs: set[str] = {bt.primary_tf}
    for s in strategies:
        for c in s.get("conditions", []):
            stack = [c]
            while stack:
                cur = stack.pop()
                if "conditions" in cur:
                    stack.extend(cur["conditions"])
                tf = cur.get("timeframe")
                if tf:
                    used_tfs.add(tf)
    log.info("策略涉及周期：%s（主周期=%s）", sorted(used_tfs), bt.primary_tf)

    # 计算每个 TF 的指标列
    ind_cfg = _collect_required_indicators(strategies)
    log.info("将计算指标：%s", ind_cfg)

    data_dict: dict[str, pl.DataFrame] = {}
    for tf in used_tfs:
        raw = _load_ohlcv(data_cfg, tf)
        if raw is None or raw.height == 0:
            log.error("缺少 %s 周期 parquet 数据，无法回测", tf)
            return 2
        log.info("%s 行数=%d 范围=%s → %s", tf, raw.height, raw["timestamp"].min(), raw["timestamp"].max())
        engine = IndicatorEngine(raw)
        with_ind = engine.compute_all(ind_cfg) if ind_cfg else raw
        data_dict[tf] = with_ind

    # 资金费率（可选）
    fr_path = data_cfg.symbol_dir / "funding_rate.parquet"
    fr_df: pl.DataFrame | None = None
    if fr_path.exists():
        fr_df = pl.read_parquet(fr_path)
        if fr_df.height == 0:
            fr_df = None
            log.warning("资金费率 parquet 为空，回测将忽略资金费率")

    log.info("开始回测：bars=%d", data_dict[bt.primary_tf].height)
    result = bt.run(data_dict, args.strategy, funding_rate_df=fr_df)
    result.print_summary()

    out_csv = Path(args.csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
