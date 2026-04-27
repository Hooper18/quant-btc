"""杠杆 → 破产概率扫描。

Phase 14 蒙特卡洛在 10× 杠杆下显示破产概率 25.8% — 不可接受。
本脚本扫描 [2, 3, 4, 5, 7, 10] 这 6 个杠杆，每个跑一次回测 + 1000 次 MC，
输出"杠杆 vs 收益/夏普/回撤/破产概率"对比表，并指出破产概率 < 5% 的最大杠杆。

输出：output/leverage_scan/{table.csv, summary.png, report.txt}
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from backtest import Backtester, MonteCarloSimulator  # noqa: E402
from backtest.visualizer import (  # noqa: E402
    _COLOR_DD, _COLOR_LONG, _COLOR_NAV, _COLOR_SHORT, _DARK_RC,
)
from utils.config import DataConfig  # noqa: E402

from run_backtest import (  # noqa: E402
    _collect_required_indicators,
    build_data_dict,
    load_aux_data,
)


LEVERAGES = [2, 3, 4, 5, 7, 10]
DEFAULT_RUIN_FRAC = 0.20
DEFAULT_N_SIM = 1000


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("leverage_scan").setLevel(logging.INFO)


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


def _make_bt(bt_cfg: dict, leverage: float) -> Backtester:
    """从 backtest_config dict 构造 Backtester，覆盖 leverage。"""
    return Backtester(
        initial_balance=float(bt_cfg["initial_balance"]),
        leverage=float(leverage),
        fee_rate=float(bt_cfg["fee_rate"]),
        slippage=float(bt_cfg["slippage"]),
        maintenance_margin_rate=float(bt_cfg["maintenance_margin_rate"]),
        max_drawdown_pct=float(bt_cfg["max_drawdown_pct"]),
        funding_rate_epochs_utc=list(bt_cfg.get("funding_rate_epochs_utc", [0, 8, 16])),
        primary_timeframe=str(bt_cfg["primary_timeframe"]),
        daily_max_loss_pct=bt_cfg.get("daily_max_loss_pct"),
    )


def _plot_summary(rows: list[dict], out_path: Path) -> None:
    plt.rcParams.update(_DARK_RC)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax_ret, ax_sharpe, ax_dd, ax_ruin = axes.flatten()

    levs = [r["leverage"] for r in rows]
    annual = [r["annualized_return_pct"] for r in rows]
    sharpe = [r["sharpe_ratio"] for r in rows]
    dd = [r["max_drawdown_pct"] for r in rows]
    ruin = [r["ruin_probability"] * 100 for r in rows]

    bar_colors = [_COLOR_LONG if v >= 0 else _COLOR_SHORT for v in annual]
    ax_ret.bar(range(len(levs)), annual, color=bar_colors, edgecolor="#0e1117")
    ax_ret.set_xticks(range(len(levs)))
    ax_ret.set_xticklabels([f"{v}×" for v in levs])
    ax_ret.set_title("年化收益率 (%)")
    ax_ret.axhline(0, color="#888", lw=0.5)
    ax_ret.grid(True, axis="y", alpha=0.3)

    bar_colors2 = [_COLOR_LONG if v > 0 else _COLOR_SHORT for v in sharpe]
    ax_sharpe.bar(range(len(levs)), sharpe, color=bar_colors2, edgecolor="#0e1117")
    ax_sharpe.set_xticks(range(len(levs)))
    ax_sharpe.set_xticklabels([f"{v}×" for v in levs])
    ax_sharpe.set_title("夏普比率")
    ax_sharpe.axhline(1.0, color="#9ba1a8", linestyle="--", lw=0.7)
    ax_sharpe.grid(True, axis="y", alpha=0.3)

    ax_dd.bar(range(len(levs)), dd, color=_COLOR_DD, edgecolor="#0e1117")
    ax_dd.set_xticks(range(len(levs)))
    ax_dd.set_xticklabels([f"{v}×" for v in levs])
    ax_dd.set_title("最大回撤 (%)")
    ax_dd.grid(True, axis="y", alpha=0.3)

    bar_colors3 = [_COLOR_LONG if v < 5 else (_COLOR_BTC_OK := "#f0a868") if v < 20 else _COLOR_SHORT
                   for v in ruin]
    ax_ruin.bar(range(len(levs)), ruin, color=bar_colors3, edgecolor="#0e1117")
    ax_ruin.set_xticks(range(len(levs)))
    ax_ruin.set_xticklabels([f"{v}×" for v in levs])
    ax_ruin.axhline(5.0, color=_COLOR_LONG, linestyle="--", lw=1.0, label="5% 安全阈")
    ax_ruin.axhline(20.0, color=_COLOR_SHORT, linestyle="--", lw=0.6, alpha=0.5, label="20% 红线")
    ax_ruin.set_title("破产概率 (%, MC=1000)")
    ax_ruin.legend(fontsize=8, loc="upper left")
    ax_ruin.grid(True, axis="y", alpha=0.3)

    fig.suptitle("杠杆扫描：收益 vs 风险", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="杠杆 → 破产概率扫描")
    parser.add_argument("--strategy", default=str(
        PROJECT_ROOT / "output" / "optimize_20260426_230903" / "best_strategy.yaml"
    ))
    parser.add_argument("--backtest", default=str(PROJECT_ROOT / "config" / "backtest_config.yaml"))
    parser.add_argument("--data-config", default=str(PROJECT_ROOT / "config" / "data_config.yaml"))
    parser.add_argument("--n-sim", type=int, default=DEFAULT_N_SIM)
    parser.add_argument("--ruin-fraction", type=float, default=DEFAULT_RUIN_FRAC)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "output" / "leverage_scan"))
    args = parser.parse_args()

    _setup_logging()
    log = logging.getLogger("leverage_scan")

    # 加载策略 + 推算 TF/指标
    with open(args.strategy, encoding="utf-8") as f:
        strat_yaml = yaml.safe_load(f) or {}
    strategies = strat_yaml.get("strategies", [])
    if not strategies:
        log.error("策略为空"); return 1

    with open(args.backtest, encoding="utf-8") as f:
        bt_cfg_dict = yaml.safe_load(f) or {}

    data_cfg = DataConfig.from_yaml(args.data_config)
    primary_tf = str(bt_cfg_dict["primary_timeframe"])
    used_tfs = _used_timeframes(strategies, primary_tf)
    ind_cfg = _collect_required_indicators(strategies)
    aux = load_aux_data(data_cfg)
    log.info("加载数据：TF=%s 指标=%d", sorted(used_tfs), len(ind_cfg))
    data_dict = build_data_dict(data_cfg, used_tfs, ind_cfg, aux=aux)
    fr_df = aux.get("funding")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    print()
    print(f"扫描杠杆 {LEVERAGES}（每个跑回测 + {args.n_sim} 次 MC）…\n")
    for lev in LEVERAGES:
        bt = _make_bt(bt_cfg_dict, leverage=lev)
        result = bt.run(data_dict, args.strategy, funding_rate_df=fr_df)
        n_closed = sum(
            1 for t in result.trades
            if t.side.endswith("_close") or t.side == "liquidate"
        )
        if n_closed < 5:
            log.warning("杠杆 %s× 已平仓交易 < 5，跳过 MC", lev)
            ruin_p = float("nan")
        else:
            mc = MonteCarloSimulator(result)
            mc_r = mc.run(n_simulations=args.n_sim, ruin_fraction=args.ruin_fraction, seed=42)
            ruin_p = mc_r.ruin_probability

        m = result.metrics
        row = {
            "leverage": lev,
            "annualized_return_pct": float(m.get("annualized_return_pct", 0)),
            "sharpe_ratio": float(m.get("sharpe_ratio", 0)),
            "max_drawdown_pct": float(m.get("max_drawdown_pct", 0)),
            "win_rate_pct": float(m.get("win_rate_pct", 0)),
            "total_trades": int(m.get("total_trades", 0)),
            "circuit_breaker": int(m.get("circuit_breaker", 0)),
            "ruin_probability": ruin_p,
        }
        rows.append(row)
        print(
            f"  杠杆 {lev:2d}×: 年化={row['annualized_return_pct']:+7.2f}%  "
            f"夏普={row['sharpe_ratio']:+.3f}  "
            f"回撤={row['max_drawdown_pct']:5.2f}%  "
            f"破产={ruin_p * 100 if ruin_p == ruin_p else float('nan'):5.2f}%  "
            f"交易={row['total_trades']:3d}  "
            f"熔断={'是' if row['circuit_breaker'] else '否'}"
        )

    # 汇总表
    print()
    print("=" * 100)
    print(f"{'杠杆':<6s}{'年化%':>10s}{'夏普':>8s}{'回撤%':>9s}{'胜率%':>9s}"
          f"{'交易':>6s}{'熔断':>5s}{'破产%':>10s}")
    print("-" * 100)
    for r in rows:
        print(
            f"{r['leverage']:<4d}× {r['annualized_return_pct']:>10.2f}"
            f" {r['sharpe_ratio']:>7.3f} {r['max_drawdown_pct']:>8.2f}"
            f" {r['win_rate_pct']:>8.2f} {r['total_trades']:>5d}"
            f" {('是' if r['circuit_breaker'] else '否'):>4s}"
            f" {r['ruin_probability'] * 100:>9.2f}"
        )
    print("=" * 100)

    # 找破产概率 < 5% 的最大杠杆
    safe = [r for r in rows if r["ruin_probability"] < 0.05]
    safe.sort(key=lambda x: -x["leverage"])
    if safe:
        best = safe[0]
        print(
            f"\n✓ 破产概率 < 5% 的最大杠杆: {best['leverage']}×  "
            f"(破产 {best['ruin_probability']*100:.2f}%, "
            f"年化 {best['annualized_return_pct']:+.2f}%, "
            f"夏普 {best['sharpe_ratio']:.3f})"
        )
        verdict = (
            f"建议生产用 {best['leverage']}× 杠杆\n"
            f"  - 破产概率 {best['ruin_probability']*100:.2f}% (< 5% 安全阈)\n"
            f"  - 年化收益 {best['annualized_return_pct']:+.2f}% (vs 10× 时 {rows[-1]['annualized_return_pct']:+.2f}%)\n"
            f"  - 最大回撤 {best['max_drawdown_pct']:.2f}%\n"
        )
    else:
        print("\n⚠️  所有杠杆破产概率均 ≥ 5%，建议进一步减小仓位（size_pct）或放宽止损")
        verdict = "无杠杆达 < 5% 破产标准；策略本身风险过高，需调整 size_pct 或止损"

    # 输出文件
    pl.DataFrame(rows).write_csv(out_dir / "table.csv")
    _plot_summary(rows, out_dir / "summary.png")
    (out_dir / "report.txt").write_text(verdict + "\n\n详细数据见 table.csv\n", encoding="utf-8")

    print(f"\n输出目录：{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
