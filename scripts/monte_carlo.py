"""蒙特卡洛风险模拟入口。

用法：
    uv run python scripts/monte_carlo.py --strategy output/optimize_*/best_strategy.yaml
    uv run python scripts/monte_carlo.py --strategy ... --n 5000 --seed 7

流程：
1. 加载数据 + 跑一次回测 → 拿到所有已平仓交易的 PnL
2. Bootstrap N 次随机重排
3. 输出三联图（净值意大利面 + 收益率分布 + 回撤分布）+ 控制台百分位摘要
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from backtest import Backtester, MonteCarloSimulator  # noqa: E402
from backtest.visualizer import _COLOR_DD, _COLOR_LONG, _COLOR_NAV, _DARK_RC  # noqa: E402
from utils.config import DataConfig  # noqa: E402

from run_backtest import (  # noqa: E402
    _collect_required_indicators,
    build_data_dict,
    load_aux_data,
)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("monte_carlo").setLevel(logging.INFO)


def _used_timeframes(strategies: list[dict], primary: str) -> set[str]:
    used = {primary}
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
    return used


def _plot(mc, out_path: Path) -> None:
    plt.rcParams.update(_DARK_RC)
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 1, hspace=0.4, height_ratios=[1.6, 1, 1])

    ax_eq = fig.add_subplot(gs[0, 0])
    ax_ret = fig.add_subplot(gs[1, 0])
    ax_dd = fig.add_subplot(gs[2, 0])

    n_sim = mc.n_simulations
    n_pts = mc.equity_curves.shape[1]
    x = np.arange(n_pts)

    # 1) 净值意大利面（subset 显示，避免 1000 条全画太重）
    show = min(300, n_sim)
    rng = np.random.default_rng(0)
    idx = rng.choice(n_sim, size=show, replace=False)
    for i in idx:
        ax_eq.plot(x, mc.equity_curves[i], color=_COLOR_NAV, alpha=0.04, linewidth=0.6)
    median_curve = np.median(mc.equity_curves, axis=0)
    p5 = np.percentile(mc.equity_curves, 5, axis=0)
    p95 = np.percentile(mc.equity_curves, 95, axis=0)
    ax_eq.fill_between(x, p5, p95, color=_COLOR_NAV, alpha=0.2, label="5–95 百分位带")
    ax_eq.plot(x, median_curve, color="#fefefe", linewidth=2.0, label="中位数")
    ax_eq.axhline(mc.initial_balance, color="#888", linestyle="--", linewidth=0.6, label=f"初始 {mc.initial_balance:.0f}")
    ax_eq.axhline(mc.ruin_threshold, color=_COLOR_DD, linestyle=":", linewidth=0.8, label=f"破产线 {mc.ruin_threshold:.0f}")
    ax_eq.set_yscale("log")
    ax_eq.set_xlabel("交易序号")
    ax_eq.set_ylabel("净值 (USDT, 对数轴)")
    ax_eq.set_title(f"Monte Carlo 净值曲线（{n_sim} 次模拟，显示 {show} 条 + 中位数 + 5/95 带）")
    ax_eq.legend(loc="upper left", fontsize=8)
    ax_eq.grid(True, alpha=0.3)

    # 2) 最终收益率分布
    ret_pct = mc.final_returns_pct
    ret_qs = mc.percentiles(ret_pct, (5, 25, 50, 75, 95))
    ax_ret.hist(ret_pct, bins=60, color=_COLOR_LONG, edgecolor="#0e1117", alpha=0.85)
    for q, color in [(5, _COLOR_DD), (50, "#fefefe"), (95, _COLOR_LONG)]:
        ax_ret.axvline(ret_qs[q], color=color, linestyle="--", linewidth=1.0,
                       label=f"P{q} = {ret_qs[q]:+.1f}%")
    ax_ret.axvline(0, color="#888", linewidth=0.5)
    ax_ret.set_xlabel("最终收益率 (%)")
    ax_ret.set_ylabel("模拟次数")
    ax_ret.set_title(
        f"最终收益率分布  均值={ret_pct.mean():+.1f}%  中位数={ret_qs[50]:+.1f}%  "
        f"破产概率={mc.ruin_probability*100:.2f}%"
    )
    ax_ret.legend(loc="upper right", fontsize=8)
    ax_ret.grid(True, alpha=0.3)

    # 3) 最大回撤分布
    dd_pct = mc.max_drawdowns_pct
    dd_qs = mc.percentiles(dd_pct, (5, 25, 50, 75, 95))
    ax_dd.hist(dd_pct, bins=60, color=_COLOR_DD, edgecolor="#0e1117", alpha=0.75)
    for q, color in [(5, _COLOR_LONG), (50, "#fefefe"), (95, _COLOR_DD)]:
        ax_dd.axvline(dd_qs[q], color=color, linestyle="--", linewidth=1.0,
                      label=f"P{q} = {dd_qs[q]:.1f}%")
    ax_dd.set_xlabel("最大回撤 (%)")
    ax_dd.set_ylabel("模拟次数")
    ax_dd.set_title(
        f"最大回撤分布  均值={dd_pct.mean():.1f}%  中位数={dd_qs[50]:.1f}%"
    )
    ax_dd.legend(loc="upper right", fontsize=8)
    ax_dd.grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="蒙特卡洛风险模拟")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--backtest", default=str(PROJECT_ROOT / "config" / "backtest_config.yaml"))
    parser.add_argument("--data-config", default=str(PROJECT_ROOT / "config" / "data_config.yaml"))
    parser.add_argument("--n", type=int, default=1000, help="模拟次数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ruin-fraction", type=float, default=0.20,
                        help="破产阈值 = 初始资金 × 此比例")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    _setup_logging()
    log = logging.getLogger("monte_carlo")

    # 1) 跑一次回测拿交易
    with open(args.strategy, encoding="utf-8") as f:
        strat_yaml = yaml.safe_load(f) or {}
    strategies = strat_yaml.get("strategies", [])
    if not strategies:
        log.error("策略为空：%s", args.strategy)
        return 1

    data_cfg = DataConfig.from_yaml(args.data_config)
    bt = Backtester.from_yaml(args.backtest)
    used_tfs = _used_timeframes(strategies, bt.primary_tf)
    ind_cfg = _collect_required_indicators(strategies)
    aux = load_aux_data(data_cfg)
    data_dict = build_data_dict(data_cfg, used_tfs, ind_cfg, aux=aux)
    fr_df = aux.get("funding")

    log.info("先跑一次回测获取已平仓交易…")
    result = bt.run(data_dict, args.strategy, funding_rate_df=fr_df)
    n_closed = sum(
        1 for t in result.trades
        if t.side.endswith("_close") or t.side == "liquidate"
    )
    log.info("已平仓交易笔数：%d", n_closed)
    if n_closed < 5:
        log.error("已平仓交易过少（<5），无法 Monte Carlo")
        return 2

    # 2) Monte Carlo
    log.info("开始 %d 次 bootstrap 模拟…", args.n)
    mc = MonteCarloSimulator(result)
    mc_result = mc.run(n_simulations=args.n, ruin_fraction=args.ruin_fraction, seed=args.seed)

    # 3) 输出
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = PROJECT_ROOT / "output" / f"monte_carlo_{ts_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4) 控制台摘要
    ret_qs = mc_result.percentiles(mc_result.final_returns_pct)
    dd_qs = mc_result.percentiles(mc_result.max_drawdowns_pct)
    print()
    print("=" * 80)
    print(f"Monte Carlo 风险模拟  ({args.n} 次, 每次 {n_closed} 笔随机重排)")
    print("=" * 80)
    print(f"初始资金:          {mc_result.initial_balance:.2f} USDT")
    print(f"破产阈值:          {mc_result.ruin_threshold:.2f} USDT  (={args.ruin_fraction:.0%} × 初始)")
    print(f"破产概率:          {mc_result.ruin_probability * 100:.2f}%   "
          f"({int(mc_result.ruin_probability * args.n)} / {args.n})")
    print(f"原始回测收益率:    {result.metrics.get('total_return_pct', 0):+.2f}%")
    print(f"原始回测最大回撤:  {result.metrics.get('max_drawdown_pct', 0):.2f}%")
    print()
    print("最终收益率分布 (%):")
    print(f"  P5  = {ret_qs[5]:+8.2f}     P25 = {ret_qs[25]:+8.2f}     P50 = {ret_qs[50]:+8.2f}")
    print(f"  P75 = {ret_qs[75]:+8.2f}     P95 = {ret_qs[95]:+8.2f}     mean = {mc_result.final_returns_pct.mean():+.2f}")
    print(f"  → 95% 置信区间: [{ret_qs[5]:+.2f}%, {ret_qs[95]:+.2f}%]")
    print()
    print("最大回撤分布 (%):")
    print(f"  P5  = {dd_qs[5]:8.2f}     P25 = {dd_qs[25]:8.2f}     P50 = {dd_qs[50]:8.2f}")
    print(f"  P75 = {dd_qs[75]:8.2f}     P95 = {dd_qs[95]:8.2f}     mean = {mc_result.max_drawdowns_pct.mean():.2f}")
    print(f"  → 95% 置信区间: [{dd_qs[5]:.2f}%, {dd_qs[95]:.2f}%]")
    print("=" * 80)

    # 5) 图表
    img_path = out_dir / "monte_carlo.png"
    _plot(mc_result, img_path)
    print(f"\n图表已保存：{img_path}")

    # 6) 数值文件
    (out_dir / "summary.txt").write_text(
        f"n_simulations={args.n}\n"
        f"n_trades_per_sim={n_closed}\n"
        f"initial_balance={mc_result.initial_balance}\n"
        f"ruin_threshold={mc_result.ruin_threshold}\n"
        f"ruin_probability={mc_result.ruin_probability}\n"
        f"return_pct_p5={ret_qs[5]}\n"
        f"return_pct_p50={ret_qs[50]}\n"
        f"return_pct_p95={ret_qs[95]}\n"
        f"return_pct_mean={float(mc_result.final_returns_pct.mean())}\n"
        f"max_dd_pct_p5={dd_qs[5]}\n"
        f"max_dd_pct_p50={dd_qs[50]}\n"
        f"max_dd_pct_p95={dd_qs[95]}\n"
        f"max_dd_pct_mean={float(mc_result.max_drawdowns_pct.mean())}\n",
        encoding="utf-8",
    )

    print(f"\n输出目录：{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
