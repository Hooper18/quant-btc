"""一键回测入口。

流程：
1. 从 data/parquet/{symbol}/ 读取所有 timeframe 的 parquet（合并 monthlies + current）
2. 按策略 YAML 中引用的指标列名，反推需计算的指标 → IndicatorEngine.compute_all
3. 加载策略 + 回测配置 → Backtester.run
4. 打印摘要、导出交易 CSV
5. 默认生成可视化报告到 output/backtest_{YYYYmmdd_HHMMSS}/（--no-plot 跳过）

用法：
    uv run python scripts/run_backtest.py
    uv run python scripts/run_backtest.py --strategy config/strategies.yaml \
        --backtest config/backtest_config.yaml --no-plot
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from backtest import Backtester, BacktestVisualizer  # noqa: E402
from data import merge_market_data  # noqa: E402
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
    # 不同月份的 parquet 可能有不同列（旧 6 列 vs 新 7 列），用 diagonal_relaxed 对齐
    return (
        pl.concat(parts, how="diagonal_relaxed")
        .unique(subset=["timestamp"])
        .sort("timestamp")
    )


def _load_oi(cfg: DataConfig) -> pl.DataFrame | None:
    """合并所有月度 open_interest_*.parquet 为单一 DataFrame。"""
    files = sorted(cfg.symbol_dir.glob("open_interest_*.parquet"))
    if not files:
        return None
    return (
        pl.concat([pl.read_parquet(f) for f in files])
        .unique(subset=["timestamp"])
        .sort("timestamp")
    )


def _load_single(cfg: DataConfig, name: str) -> pl.DataFrame | None:
    p = cfg.symbol_dir / name
    if not p.exists():
        return None
    df = pl.read_parquet(p)
    return df if df.height > 0 else None


def load_aux_data(cfg: DataConfig) -> dict[str, pl.DataFrame | None]:
    """一次性加载所有辅助数据源；缺失返回 None。供 run_backtest / compare / walk_forward 共用。"""
    return {
        "funding": _load_single(cfg, "funding_rate.parquet"),
        "oi": _load_oi(cfg),
        "fgi": _load_single(cfg, "fear_greed_index.parquet"),
        "ls": _load_single(cfg, "long_short_ratio.parquet"),
        "tt": _load_single(cfg, "top_trader_ratio.parquet"),
    }


def build_data_dict(
    cfg: DataConfig,
    timeframes: list[str] | set[str],
    ind_cfg: list[tuple[str, dict[str, Any]]],
    aux: dict[str, pl.DataFrame | None] | None = None,
) -> dict[str, pl.DataFrame]:
    """对每个 TF：加载 OHLCV → merge 辅助数据 → 计算指标。返回 data_dict。

    `aux` 缺省时自动加载（每次调用都会读盘）。多次调用建议传入预加载的 aux。
    缺失列的策略自然忽略；老 OHLCV（无 taker_buy_volume）也能正常 merge。
    """
    if aux is None:
        aux = load_aux_data(cfg)
    out: dict[str, pl.DataFrame] = {}
    for tf in sorted(set(timeframes)):
        raw = _load_ohlcv(cfg, tf)
        if raw is None or raw.height == 0:
            raise FileNotFoundError(f"缺 {tf} 周期 OHLCV 数据")
        merged = merge_market_data(
            raw,
            funding_df=aux.get("funding"),
            oi_df=aux.get("oi"),
            fgi_df=aux.get("fgi"),
            long_short_df=aux.get("ls"),
            top_trader_df=aux.get("tt"),
        )
        out[tf] = (
            IndicatorEngine(merged).compute_all(ind_cfg) if ind_cfg else merged
        )
    return out


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
    # Phase11/12 衍生指标
    (re.compile(r"^taker_buy_ratio$"), "taker_buy_ratio", lambda m: {}),
    (re.compile(r"^oi_change_(\d+)$"), "oi_change", lambda m: {"period": int(m[1])}),
    (re.compile(r"^fear_greed_ma_(\d+)$"), "fear_greed_ma", lambda m: {"period": int(m[1])}),
    (re.compile(r"^rolling_max_(\d+)$"), "rolling_max", lambda m: {"period": int(m[1])}),
    (re.compile(r"^rolling_min_(\d+)$"), "rolling_min", lambda m: {"period": int(m[1])}),
]


def _parse_indicator_name(name: str) -> tuple[str, dict[str, Any]] | None:
    for pat, key, builder in _INDICATOR_PATTERNS:
        m = pat.match(name)
        if m:
            return key, builder(m)
    return None


def _collect_required_indicators(strategies: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """递归扫描策略条件树，收集所有引用到的指标列名 → IndicatorEngine 配置。

    返回 list[(name, params)]，允许同一指标有多组参数（如 ema_12 + ema_26）。
    按出现顺序去重相同 (name, frozenset(params)) 组合。
    """
    seen: set[tuple[str, frozenset]] = set()
    out: list[tuple[str, dict[str, Any]]] = []

    def walk(cond: dict[str, Any]) -> None:
        if "conditions" in cond:
            for sub in cond["conditions"]:
                walk(sub)
            return
        for field in ("indicator", "reference", "value"):
            v = cond.get(field)
            if not isinstance(v, str):
                continue
            parsed = _parse_indicator_name(v)
            if parsed is None:
                continue
            key, params = parsed
            sig = (key, frozenset((k, params[k]) for k in sorted(params)))
            if sig in seen:
                continue
            seen.add(sig)
            out.append((key, params))

    for s in strategies:
        for c in s.get("conditions", []):
            walk(c)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="BTC 量化回测入口")
    parser.add_argument("--strategy", default=str(PROJECT_ROOT / "config" / "strategies.yaml"))
    parser.add_argument("--backtest", default=str(PROJECT_ROOT / "config" / "backtest_config.yaml"))
    parser.add_argument("--data-config", default=str(PROJECT_ROOT / "config" / "data_config.yaml"))
    parser.add_argument(
        "--output-dir", default=None,
        help="可视化报告输出目录；默认 output/backtest_{YYYYmmdd_HHMMSS}/",
    )
    parser.add_argument("--no-plot", action="store_true", help="跳过可视化")
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

    # 一次性加载辅助数据 + 合并 + 计算指标
    aux = load_aux_data(data_cfg)
    log.info(
        "辅助数据：funding=%s OI=%s FGI=%s LS=%s TT=%s",
        "OK" if aux["funding"] is not None else "缺失",
        "OK" if aux["oi"] is not None else "缺失",
        "OK" if aux["fgi"] is not None else "缺失",
        "OK" if aux["ls"] is not None else "缺失",
        "OK" if aux["tt"] is not None else "缺失",
    )
    try:
        data_dict = build_data_dict(data_cfg, used_tfs, ind_cfg, aux=aux)
    except FileNotFoundError as e:
        log.error(str(e))
        return 2
    for tf, df in data_dict.items():
        log.info("%s: 行数=%d 列=%d", tf, df.height, len(df.columns))

    fr_df = aux.get("funding")

    log.info("开始回测：bars=%d", data_dict[bt.primary_tf].height)
    result = bt.run(data_dict, args.strategy, funding_rate_df=fr_df)
    result.print_summary()

    # 输出目录：默认 output/backtest_{ts}/
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = PROJECT_ROOT / "output" / f"backtest_{ts_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    result.to_csv(out_dir / "trades.csv")

    if not args.no_plot:
        log.info("生成可视化报告 → %s", out_dir)
        viz = BacktestVisualizer(result, data_dict[bt.primary_tf])
        report_path = viz.save_report(out_dir)
        print(f"\n报告已保存到 {out_dir}")
        print(f"  - HTML: {report_path}")
        print(f"  - CSV : {out_dir / 'trades.csv'}")
    else:
        print(f"\n交易记录已保存到 {out_dir / 'trades.csv'}（已跳过可视化）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
