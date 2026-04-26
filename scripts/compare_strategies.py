"""多策略对比器。

用法：
    uv run python scripts/compare_strategies.py \
        config/strategies.yaml \
        config/strategies_rsi_divergence.yaml \
        config/strategies_bollinger_squeeze.yaml \
        config/strategies_multi_signal.yaml \
        config/strategies_mean_reversion.yaml

对每个策略：跑回测、收集指标。最后输出：
- output/comparison_{ts}/comparison.csv  指标对比表
- output/comparison_{ts}/equity_overlay.png  净值曲线叠加图
- 控制台对齐打印对比表
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import polars as pl
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from backtest import Backtester, BacktestResult  # noqa: E402
from backtest.visualizer import _DARK_RC, _COLOR_BTC  # noqa: E402
from indicators import IndicatorEngine  # noqa: E402
from utils.config import DataConfig  # noqa: E402

# 复用 run_backtest 的指标解析 / OHLCV 加载逻辑
from run_backtest import (  # noqa: E402
    _collect_required_indicators,
    _load_ohlcv,
)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _used_timeframes(strategies: list[dict[str, Any]], primary: str) -> list[str]:
    used: set[str] = {primary}
    for s in strategies:
        for c in s.get("conditions", []):
            stack = [c]
            while stack:
                cur = stack.pop()
                if "conditions" in cur:
                    stack.extend(cur["conditions"])
                tf = cur.get("timeframe")
                if tf:
                    used.add(tf)
    return sorted(used)


def _run_one(strategy_path: Path, data_cfg: DataConfig, bt: Backtester,
             fr_df: pl.DataFrame | None, log: logging.Logger) -> tuple[str, BacktestResult] | None:
    with strategy_path.open(encoding="utf-8") as f:
        strat_yaml = yaml.safe_load(f) or {}
    strategies = strat_yaml.get("strategies", [])
    if not strategies:
        log.warning("空策略：%s", strategy_path)
        return None

    used_tfs = _used_timeframes(strategies, bt.primary_tf)
    ind_cfg = _collect_required_indicators(strategies)

    data_dict: dict[str, pl.DataFrame] = {}
    for tf in used_tfs:
        raw = _load_ohlcv(data_cfg, tf)
        if raw is None or raw.height == 0:
            log.error("[%s] 缺少 %s 周期数据", strategy_path.name, tf)
            return None
        data_dict[tf] = (
            IndicatorEngine(raw).compute_all(ind_cfg) if ind_cfg else raw
        )

    name = strategy_path.stem
    log.info("[%s] 开始回测：bars=%d", name, data_dict[bt.primary_tf].height)
    result = bt.run(data_dict, str(strategy_path), funding_rate_df=fr_df)
    return name, result


def _format_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        ("策略", "name", 30),
        ("总收益率%", "total_return_pct", 11),
        ("年化%", "annualized_return_pct", 9),
        ("夏普", "sharpe_ratio", 7),
        ("最大回撤%", "max_drawdown_pct", 11),
        ("胜率%", "win_rate_pct", 8),
        ("盈亏比", "profit_loss_ratio", 8),
        ("交易数", "total_trades", 7),
        ("熔断", "circuit_breaker", 5),
    ]
    sep = "  ".join("-" * w for _, _, w in headers)
    lines: list[str] = [
        "  ".join(f"{h:<{w}s}" for h, _, w in headers),
        sep,
    ]
    for row in rows:
        cells: list[str] = []
        for _, key, w in headers:
            v = row.get(key, "")
            if key == "name":
                cells.append(f"{str(v):<{w}.{w}s}")
            elif key == "circuit_breaker":
                cells.append(f"{'是' if v else '否':<{w}s}")
            elif isinstance(v, (int, float)):
                fmt = f"{v:>{w}.2f}" if isinstance(v, float) else f"{v:>{w}d}"
                cells.append(fmt)
            else:
                cells.append(f"{str(v):<{w}s}")
        lines.append("  ".join(cells))
    return "\n".join(lines)


def _save_overlay(results: list[tuple[str, BacktestResult]], out_path: Path) -> None:
    plt.rcParams.update(_DARK_RC)
    fig, ax = plt.subplots(figsize=(14, 7))
    for name, res in results:
        if not res.equity_curve:
            continue
        ax.plot(res.timestamps, res.equity_curve, linewidth=1.2, label=name)
    ax.axhline(
        results[0][1].metrics.get("initial_balance", 100) if results else 100,
        color="#888", linestyle="--", linewidth=0.6, alpha=0.6, label="初始资金",
    )
    ax.set_yscale("log")
    ax.set_title("策略净值对比（对数轴）")
    ax.set_xlabel("时间")
    ax.set_ylabel("净值 (USDT)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="多策略回测对比")
    parser.add_argument("strategies", nargs="+", help="策略 YAML 路径列表")
    parser.add_argument("--data-config", default=str(PROJECT_ROOT / "config" / "data_config.yaml"))
    parser.add_argument("--backtest", default=str(PROJECT_ROOT / "config" / "backtest_config.yaml"))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    _setup_logging()
    log = logging.getLogger("compare")

    data_cfg = DataConfig.from_yaml(args.data_config)
    bt = Backtester.from_yaml(args.backtest)

    fr_df: pl.DataFrame | None = None
    fr_path = data_cfg.symbol_dir / "funding_rate.parquet"
    if fr_path.exists():
        fr_df = pl.read_parquet(fr_path)
        if fr_df.height == 0:
            fr_df = None

    results: list[tuple[str, BacktestResult]] = []
    for sp in args.strategies:
        path = Path(sp)
        if not path.exists():
            log.error("文件不存在：%s", sp)
            continue
        out = _run_one(path, data_cfg, bt, fr_df, log)
        if out is not None:
            results.append(out)

    if not results:
        log.error("无成功回测结果")
        return 1

    # 输出目录
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = PROJECT_ROOT / "output" / f"comparison_{ts_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 汇总表
    rows: list[dict[str, Any]] = []
    for name, res in results:
        m = res.metrics
        rows.append({
            "name": name,
            "total_return_pct": m.get("total_return_pct", 0),
            "annualized_return_pct": m.get("annualized_return_pct", 0),
            "sharpe_ratio": m.get("sharpe_ratio", 0),
            "max_drawdown_pct": m.get("max_drawdown_pct", 0),
            "win_rate_pct": m.get("win_rate_pct", 0),
            "profit_loss_ratio": m.get("profit_loss_ratio", 0),
            "total_trades": int(m.get("total_trades", 0)),
            "circuit_breaker": int(m.get("circuit_breaker", 0)),
        })
    table = _format_table(rows)
    print("\n" + "=" * 110)
    print("策略对比")
    print("=" * 110)
    print(table)
    print("=" * 110)

    # CSV
    pl.DataFrame(rows).write_csv(out_dir / "comparison.csv")

    # 净值叠加图
    _save_overlay(results, out_dir / "equity_overlay.png")

    print(f"\n对比结果保存到 {out_dir}")
    print(f"  - 表格: {out_dir / 'comparison.csv'}")
    print(f"  - 叠加图: {out_dir / 'equity_overlay.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
