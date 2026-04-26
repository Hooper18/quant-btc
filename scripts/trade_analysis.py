"""交易深度分析报告。

跑一次 best_strategy 回测后，对成交记录做 6 个维度切片：
1. 按小时（UTC）：开仓数 / 胜率 24 时段柱状图
2. 按星期（Mon-Sun）：胜率 + 平均 PnL
3. 按月份（年-月聚合）：每月 PnL 柱状图
4. 按持仓时长：<24h / 1-7d / >7d 三段对比
5. 连胜连亏：最长连胜/连亏 + 累计盈亏
6. 按 FGI 市场情绪：极度恐慌(<25) / 中性(25-75) / 极度贪婪(>75) 三段对比

输出：output/trade_analysis_{ts}/{*.png, report.html, summary.txt}
"""
from __future__ import annotations

import argparse
import calendar
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from backtest import Backtester, Trade  # noqa: E402
from backtest.visualizer import (  # noqa: E402
    _COLOR_BTC, _COLOR_DD, _COLOR_LONG, _COLOR_NAV, _COLOR_SHORT, _DARK_RC,
)
from utils.config import DataConfig  # noqa: E402

from run_backtest import (  # noqa: E402
    _collect_required_indicators,
    build_data_dict,
    load_aux_data,
)


@dataclass
class TradePair:
    """配对的开仓 + 平仓。"""
    open_ts: datetime
    close_ts: datetime
    side: str            # "long" | "short"
    pnl: float
    duration_h: float
    open_price: float
    close_price: float
    strategy: str

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("trade_analysis").setLevel(logging.INFO)


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


def _pair_trades(trades: list[Trade]) -> list[TradePair]:
    pairs: list[TradePair] = []
    open_stack: list[Trade] = []
    for t in trades:
        if t.side.endswith("_open"):
            open_stack.append(t)
        elif (t.side.endswith("_close") or t.side == "liquidate") and open_stack:
            opener = open_stack.pop(0)
            side = "long" if opener.side.startswith("long") else "short"
            duration_h = (t.timestamp - opener.timestamp).total_seconds() / 3600.0
            pairs.append(TradePair(
                open_ts=opener.timestamp, close_ts=t.timestamp,
                side=side, pnl=t.pnl, duration_h=duration_h,
                open_price=opener.price, close_price=t.price,
                strategy=opener.strategy,
            ))
    return pairs


# ---------- 各维度切片 ----------
def _hour_stats(pairs: list[TradePair]) -> dict[int, dict[str, float]]:
    """按开仓小时聚合：count, win_count, total_pnl。"""
    out: dict[int, dict[str, float]] = {h: {"count": 0, "wins": 0, "pnl": 0.0} for h in range(24)}
    for p in pairs:
        h = p.open_ts.hour
        out[h]["count"] += 1
        out[h]["wins"] += int(p.is_win)
        out[h]["pnl"] += p.pnl
    return out


def _weekday_stats(pairs: list[TradePair]) -> dict[int, dict[str, float]]:
    """按开仓 weekday（0=Mon ... 6=Sun）聚合。"""
    out = {d: {"count": 0, "wins": 0, "pnl": 0.0} for d in range(7)}
    for p in pairs:
        d = p.open_ts.weekday()
        out[d]["count"] += 1
        out[d]["wins"] += int(p.is_win)
        out[d]["pnl"] += p.pnl
    return out


def _month_stats(pairs: list[TradePair]) -> dict[str, float]:
    """按开仓 YYYY-MM 聚合 PnL。"""
    out: dict[str, float] = defaultdict(float)
    for p in pairs:
        key = f"{p.open_ts.year:04d}-{p.open_ts.month:02d}"
        out[key] += p.pnl
    return dict(sorted(out.items()))


def _holding_buckets(pairs: list[TradePair]) -> dict[str, dict[str, float]]:
    """三档：<24h / 24-168h(1-7d) / >168h(>7d)。"""
    buckets = {
        "短(<24h)": {"count": 0, "wins": 0, "pnl": 0.0, "pnls": []},
        "中(1-7d)": {"count": 0, "wins": 0, "pnl": 0.0, "pnls": []},
        "长(>7d)":  {"count": 0, "wins": 0, "pnl": 0.0, "pnls": []},
    }
    for p in pairs:
        if p.duration_h < 24:
            k = "短(<24h)"
        elif p.duration_h <= 168:
            k = "中(1-7d)"
        else:
            k = "长(>7d)"
        buckets[k]["count"] += 1
        buckets[k]["wins"] += int(p.is_win)
        buckets[k]["pnl"] += p.pnl
        buckets[k]["pnls"].append(p.pnl)
    return buckets


def _streaks(pairs: list[TradePair]) -> dict[str, Any]:
    """连胜/连亏统计。"""
    if not pairs:
        return {}
    longest_win, longest_loss = 0, 0
    cur_win, cur_loss = 0, 0
    win_streak_pnl, loss_streak_pnl = 0.0, 0.0
    cur_win_pnl, cur_loss_pnl = 0.0, 0.0
    # 连续亏损恢复时间：最大连亏期间起始 → 净值回到亏损前峰值的小时数
    in_streak = False
    streak_kind = None
    streak_start_idx = 0
    drawdown_periods = []  # (start_idx, length, recovery_idx, max_loss)

    cum = 0.0
    peaks = []
    cums = []
    ts_list = []
    for p in pairs:
        cum += p.pnl
        cums.append(cum)
        ts_list.append(p.close_ts)
        peaks.append(max(peaks[-1], cum) if peaks else cum)

    cur_win, cur_loss = 0, 0
    cur_win_pnl, cur_loss_pnl = 0.0, 0.0
    for p in pairs:
        if p.is_win:
            cur_win += 1
            cur_win_pnl += p.pnl
            if cur_loss > longest_loss:
                longest_loss = cur_loss
                loss_streak_pnl = cur_loss_pnl
            cur_loss = 0
            cur_loss_pnl = 0.0
        elif p.pnl < 0:
            cur_loss += 1
            cur_loss_pnl += p.pnl
            if cur_win > longest_win:
                longest_win = cur_win
                win_streak_pnl = cur_win_pnl
            cur_win = 0
            cur_win_pnl = 0.0
    # 末尾收尾
    if cur_win > longest_win:
        longest_win = cur_win
        win_streak_pnl = cur_win_pnl
    if cur_loss > longest_loss:
        longest_loss = cur_loss
        loss_streak_pnl = cur_loss_pnl

    # 最大累计回撤（基于交易序列，单位 USDT）+ 恢复时间
    max_dd_value = 0.0
    max_dd_recovery_h = 0.0
    peak_idx = 0
    trough_idx = 0
    for i, c in enumerate(cums):
        if c > cums[peak_idx]:
            peak_idx = i
        dd_value = peaks[i] - cums[i]
        if dd_value > max_dd_value:
            max_dd_value = dd_value
            trough_idx = i
    # 找回撤起点（trough 之前的 peak）
    if trough_idx > 0:
        recovery_peak = peaks[trough_idx]
        recovery_idx = None
        for j in range(trough_idx + 1, len(cums)):
            if cums[j] >= recovery_peak:
                recovery_idx = j
                break
        if recovery_idx is not None:
            max_dd_recovery_h = (
                ts_list[recovery_idx] - ts_list[trough_idx]
            ).total_seconds() / 3600.0

    return {
        "longest_win_streak": longest_win,
        "longest_win_streak_pnl": win_streak_pnl,
        "longest_loss_streak": longest_loss,
        "longest_loss_streak_pnl": loss_streak_pnl,
        "max_consecutive_dd_usdt": max_dd_value,
        "max_consecutive_dd_recovery_h": max_dd_recovery_h,
    }


def _fgi_regime(pairs: list[TradePair], fgi_df: pl.DataFrame | None) -> dict[str, dict[str, float]]:
    """按 FGI 区间分组：<25 极恐 / 25-75 中性 / >75 极贪。"""
    if fgi_df is None or fgi_df.height == 0:
        return {}
    fgi_sorted = fgi_df.sort("timestamp")
    fgi_ts = fgi_sorted["timestamp"].to_list()
    fgi_val = fgi_sorted["value"].to_list()

    def lookup(ts: datetime) -> float | None:
        # 二分找 ≤ ts 的最大
        lo, hi = 0, len(fgi_ts)
        while lo < hi:
            mid = (lo + hi) // 2
            if fgi_ts[mid] <= ts:
                lo = mid + 1
            else:
                hi = mid
        idx = lo - 1
        if idx < 0:
            return None
        return float(fgi_val[idx])

    out = {
        "极度恐慌(<25)": {"count": 0, "wins": 0, "pnl": 0.0, "pnls": []},
        "中性(25-75)":   {"count": 0, "wins": 0, "pnl": 0.0, "pnls": []},
        "极度贪婪(>75)": {"count": 0, "wins": 0, "pnl": 0.0, "pnls": []},
    }
    for p in pairs:
        v = lookup(p.open_ts)
        if v is None:
            continue
        if v < 25:
            k = "极度恐慌(<25)"
        elif v <= 75:
            k = "中性(25-75)"
        else:
            k = "极度贪婪(>75)"
        out[k]["count"] += 1
        out[k]["wins"] += int(p.is_win)
        out[k]["pnl"] += p.pnl
        out[k]["pnls"].append(p.pnl)
    return out


# ---------- 绘图 ----------
def _plot_hour(stats: dict[int, dict[str, float]], out_path: Path) -> dict:
    plt.rcParams.update(_DARK_RC)
    hours = list(range(24))
    counts = [stats[h]["count"] for h in hours]
    win_rates = [
        (stats[h]["wins"] / stats[h]["count"] * 100) if stats[h]["count"] > 0 else 0
        for h in hours
    ]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax1.bar(hours, counts, color=_COLOR_NAV, edgecolor="#0e1117")
    ax1.set_ylabel("开仓数")
    ax1.set_title("按 UTC 小时：开仓数")
    ax1.grid(True, axis="y", alpha=0.3)
    bar_colors = [_COLOR_LONG if w >= 50 else _COLOR_SHORT for w in win_rates]
    ax2.bar(hours, win_rates, color=bar_colors, edgecolor="#0e1117")
    ax2.axhline(50, color="#9ba1a8", linestyle="--", linewidth=0.7)
    ax2.set_xlabel("UTC 小时")
    ax2.set_ylabel("胜率 %")
    ax2.set_title("按 UTC 小时：胜率")
    ax2.set_xticks(hours)
    ax2.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    best_hour = max(hours, key=lambda h: (stats[h]["wins"] / stats[h]["count"]) if stats[h]["count"] > 5 else 0)
    most_active = max(hours, key=lambda h: stats[h]["count"])
    return {
        "best_hour_winrate": best_hour,
        "best_hour_winrate_value": (
            stats[best_hour]["wins"] / stats[best_hour]["count"] * 100
            if stats[best_hour]["count"] > 0 else 0
        ),
        "most_active_hour": most_active,
        "most_active_hour_count": stats[most_active]["count"],
    }


_WEEKDAY_ZH = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def _plot_weekday(stats: dict[int, dict[str, float]], out_path: Path) -> dict:
    plt.rcParams.update(_DARK_RC)
    days = list(range(7))
    counts = [stats[d]["count"] for d in days]
    pnls = [stats[d]["pnl"] for d in days]
    win_rates = [
        (stats[d]["wins"] / stats[d]["count"] * 100) if stats[d]["count"] > 0 else 0
        for d in days
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bar_colors = [_COLOR_LONG if p >= 0 else _COLOR_SHORT for p in pnls]
    axes[0].bar(_WEEKDAY_ZH, pnls, color=bar_colors, edgecolor="#0e1117")
    axes[0].axhline(0, color="#888", linewidth=0.5)
    axes[0].set_ylabel("总 PnL (USDT)")
    axes[0].set_title("按星期：累计 PnL")
    axes[0].grid(True, axis="y", alpha=0.3)
    bar_colors2 = [_COLOR_LONG if w >= 50 else _COLOR_SHORT for w in win_rates]
    axes[1].bar(_WEEKDAY_ZH, win_rates, color=bar_colors2, edgecolor="#0e1117")
    axes[1].axhline(50, color="#9ba1a8", linestyle="--", linewidth=0.7)
    axes[1].set_ylabel("胜率 %")
    axes[1].set_title(f"按星期：胜率（柱上数字=交易数）")
    for i, (d, c) in enumerate(zip(days, counts)):
        axes[1].text(i, win_rates[i] + 1, str(c), ha="center", fontsize=8, color="#9ba1a8")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    best_d = max(days, key=lambda d: stats[d]["pnl"])
    worst_d = min(days, key=lambda d: stats[d]["pnl"])
    return {
        "best_weekday": _WEEKDAY_ZH[best_d],
        "best_weekday_pnl": stats[best_d]["pnl"],
        "worst_weekday": _WEEKDAY_ZH[worst_d],
        "worst_weekday_pnl": stats[worst_d]["pnl"],
    }


def _plot_month(stats: dict[str, float], out_path: Path) -> dict:
    plt.rcParams.update(_DARK_RC)
    keys = list(stats.keys())
    pnls = [stats[k] for k in keys]
    fig, ax = plt.subplots(figsize=(15, 5))
    bar_colors = [_COLOR_LONG if p >= 0 else _COLOR_SHORT for p in pnls]
    ax.bar(keys, pnls, color=bar_colors, edgecolor="#0e1117")
    ax.axhline(0, color="#888", linewidth=0.5)
    ax.set_ylabel("月度 PnL (USDT)")
    ax.set_title(f"按月份：贡献 PnL（共 {len(keys)} 个月）")
    ax.grid(True, axis="y", alpha=0.3)
    for tick in ax.get_xticklabels():
        tick.set_rotation(60); tick.set_ha("right"); tick.set_fontsize(7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    if not keys:
        return {}
    best_m = max(keys, key=lambda k: stats[k])
    worst_m = min(keys, key=lambda k: stats[k])
    return {
        "best_month": best_m, "best_month_pnl": stats[best_m],
        "worst_month": worst_m, "worst_month_pnl": stats[worst_m],
        "positive_months": sum(1 for p in pnls if p > 0),
        "negative_months": sum(1 for p in pnls if p < 0),
        "total_months": len(keys),
    }


def _plot_holding(buckets: dict[str, dict], pairs: list[TradePair], out_path: Path) -> dict:
    plt.rcParams.update(_DARK_RC)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = list(buckets.keys())
    win_rates = [
        (b["wins"] / b["count"] * 100) if b["count"] > 0 else 0
        for b in buckets.values()
    ]
    avg_pnls = [
        (b["pnl"] / b["count"]) if b["count"] > 0 else 0
        for b in buckets.values()
    ]
    counts = [b["count"] for b in buckets.values()]
    bar_colors = [_COLOR_LONG if w >= 50 else _COLOR_SHORT for w in win_rates]
    axes[0].bar(labels, win_rates, color=bar_colors, edgecolor="#0e1117")
    axes[0].axhline(50, color="#9ba1a8", linestyle="--", linewidth=0.7)
    axes[0].set_ylabel("胜率 %")
    axes[0].set_title("按持仓时长：胜率（柱上=交易数）")
    for i, (l, c) in enumerate(zip(labels, counts)):
        axes[0].text(i, win_rates[i] + 1, str(c), ha="center", fontsize=9, color="#9ba1a8")
    axes[0].grid(True, axis="y", alpha=0.3)
    bar_colors2 = [_COLOR_LONG if p >= 0 else _COLOR_SHORT for p in avg_pnls]
    axes[1].bar(labels, avg_pnls, color=bar_colors2, edgecolor="#0e1117")
    axes[1].axhline(0, color="#888", linewidth=0.5)
    axes[1].set_ylabel("平均 PnL (USDT)")
    axes[1].set_title("按持仓时长：单笔平均 PnL")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 散点：x=duration, y=pnl
    fig2, ax2 = plt.subplots(figsize=(13, 5))
    durs = [p.duration_h for p in pairs]
    pnls = [p.pnl for p in pairs]
    colors = [_COLOR_LONG if p >= 0 else _COLOR_SHORT for p in pnls]
    ax2.scatter(durs, pnls, c=colors, alpha=0.55, s=18, edgecolors="none")
    ax2.axhline(0, color="#888", linewidth=0.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("持仓时长 (小时, 对数轴)")
    ax2.set_ylabel("单笔 PnL (USDT)")
    ax2.set_title(f"持仓时长 vs PnL 散点（共 {len(pairs)} 笔）")
    ax2.grid(True, alpha=0.3)
    scatter_path = out_path.parent / "holding_scatter.png"
    fig2.savefig(scatter_path, dpi=130, bbox_inches="tight")
    plt.close(fig2)

    return {label: {
        "count": b["count"],
        "win_rate_pct": (b["wins"] / b["count"] * 100) if b["count"] > 0 else 0,
        "avg_pnl": (b["pnl"] / b["count"]) if b["count"] > 0 else 0,
        "total_pnl": b["pnl"],
    } for label, b in buckets.items()}


def _plot_fgi(stats: dict[str, dict], out_path: Path) -> dict:
    if not stats:
        return {}
    plt.rcParams.update(_DARK_RC)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = list(stats.keys())
    win_rates = [
        (s["wins"] / s["count"] * 100) if s["count"] > 0 else 0
        for s in stats.values()
    ]
    pnls = [s["pnl"] for s in stats.values()]
    counts = [s["count"] for s in stats.values()]
    bar_colors = [_COLOR_LONG if w >= 50 else _COLOR_SHORT for w in win_rates]
    axes[0].bar(labels, win_rates, color=bar_colors, edgecolor="#0e1117")
    axes[0].axhline(50, color="#9ba1a8", linestyle="--", linewidth=0.7)
    axes[0].set_ylabel("胜率 %")
    axes[0].set_title("按 FGI 市场情绪：胜率")
    for i, (l, c) in enumerate(zip(labels, counts)):
        axes[0].text(i, win_rates[i] + 1, str(c), ha="center", fontsize=9, color="#9ba1a8")
    axes[0].grid(True, axis="y", alpha=0.3)
    bar_colors2 = [_COLOR_LONG if p >= 0 else _COLOR_SHORT for p in pnls]
    axes[1].bar(labels, pnls, color=bar_colors2, edgecolor="#0e1117")
    axes[1].axhline(0, color="#888", linewidth=0.5)
    axes[1].set_ylabel("总 PnL (USDT)")
    axes[1].set_title("按 FGI 市场情绪：累计 PnL")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    return {label: {
        "count": s["count"],
        "win_rate_pct": (s["wins"] / s["count"] * 100) if s["count"] > 0 else 0,
        "total_pnl": s["pnl"],
        "avg_pnl": (s["pnl"] / s["count"]) if s["count"] > 0 else 0,
    } for label, s in stats.items()}


# ---------- HTML 报告 ----------
_HTML_TPL = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>交易深度分析报告</title>
<style>
  body {{ background:#0e1117; color:#e6edf3; font-family:-apple-system,system-ui,sans-serif; margin:24px; }}
  h1 {{ color:#4cd2ff; border-bottom:1px solid #30363d; padding-bottom:8px; }}
  h2 {{ color:#f0a868; margin-top:30px; }}
  pre {{ background:#161b22; padding:14px; border:1px solid #30363d; border-radius:6px; overflow-x:auto; }}
  img {{ max-width:100%; border:1px solid #30363d; border-radius:6px; background:#0e1117; margin:8px 0; }}
  table {{ border-collapse:collapse; margin:12px 0; }}
  th, td {{ border:1px solid #30363d; padding:6px 14px; text-align:left; }}
  tr:nth-child(even) {{ background:#161b22; }}
</style></head>
<body>
<h1>BTC 量化交易深度分析</h1>

<h2>1. 按 UTC 小时分布</h2>
<img src="hour.png">

<h2>2. 按星期分布</h2>
<img src="weekday.png">

<h2>3. 按月份贡献</h2>
<img src="month.png">

<h2>4. 按持仓时长</h2>
<img src="holding.png">
<img src="holding_scatter.png">

<h2>5. 按 FGI 市场情绪</h2>
<img src="fgi.png">

<h2>6. 关键摘要</h2>
<pre>{summary}</pre>
</body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="交易深度分析")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--backtest", default=str(PROJECT_ROOT / "config" / "backtest_config.yaml"))
    parser.add_argument("--data-config", default=str(PROJECT_ROOT / "config" / "data_config.yaml"))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    _setup_logging()
    log = logging.getLogger("trade_analysis")

    with open(args.strategy, encoding="utf-8") as f:
        strat_yaml = yaml.safe_load(f) or {}
    strategies = strat_yaml.get("strategies", [])
    if not strategies:
        log.error("策略为空"); return 1

    data_cfg = DataConfig.from_yaml(args.data_config)
    bt = Backtester.from_yaml(args.backtest)
    used_tfs = _used_timeframes(strategies, bt.primary_tf)
    ind_cfg = _collect_required_indicators(strategies)
    aux = load_aux_data(data_cfg)
    data_dict = build_data_dict(data_cfg, used_tfs, ind_cfg, aux=aux)
    fgi_df = aux.get("fgi")

    log.info("跑回测获取交易…")
    result = bt.run(data_dict, args.strategy, funding_rate_df=aux.get("funding"))
    pairs = _pair_trades(result.trades)
    log.info("配对交易数：%d", len(pairs))
    if not pairs:
        log.error("无配对交易，无法分析"); return 2

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = PROJECT_ROOT / "output" / f"trade_analysis_{ts_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 6 个维度
    log.info("计算各维度统计…")
    h_stats = _hour_stats(pairs)
    w_stats = _weekday_stats(pairs)
    m_stats = _month_stats(pairs)
    hold_buckets = _holding_buckets(pairs)
    streak_stats = _streaks(pairs)
    fgi_stats = _fgi_regime(pairs, fgi_df)

    h_summary = _plot_hour(h_stats, out_dir / "hour.png")
    w_summary = _plot_weekday(w_stats, out_dir / "weekday.png")
    m_summary = _plot_month(m_stats, out_dir / "month.png")
    hold_summary = _plot_holding(hold_buckets, pairs, out_dir / "holding.png")
    fgi_summary = _plot_fgi(fgi_stats, out_dir / "fgi.png")

    # 摘要
    lines: list[str] = []
    lines.append(f"=== 交易分析摘要（共 {len(pairs)} 笔配对交易）===\n")
    lines.append(f"策略文件: {args.strategy}\n")
    lines.append(f"原始回测：收益率 {result.metrics.get('total_return_pct', 0):+.2f}%, "
                 f"夏普 {result.metrics.get('sharpe_ratio', 0):.3f}, "
                 f"胜率 {result.metrics.get('win_rate_pct', 0):.2f}%\n\n")

    lines.append("--- 时段 ---\n")
    lines.append(f"开仓最活跃 UTC 小时: {h_summary['most_active_hour']:02d}:00 "
                 f"({h_summary['most_active_hour_count']} 笔)\n")
    lines.append(f"胜率最高 UTC 小时:   {h_summary['best_hour_winrate']:02d}:00 "
                 f"({h_summary['best_hour_winrate_value']:.1f}%)\n\n")

    lines.append("--- 星期 ---\n")
    lines.append(f"表现最佳: {w_summary['best_weekday']} (PnL {w_summary['best_weekday_pnl']:+.2f})\n")
    lines.append(f"表现最差: {w_summary['worst_weekday']} (PnL {w_summary['worst_weekday_pnl']:+.2f})\n\n")

    lines.append("--- 月份 ---\n")
    lines.append(f"最赚钱月: {m_summary.get('best_month', '-')} "
                 f"(PnL {m_summary.get('best_month_pnl', 0):+.2f})\n")
    lines.append(f"最亏损月: {m_summary.get('worst_month', '-')} "
                 f"(PnL {m_summary.get('worst_month_pnl', 0):+.2f})\n")
    lines.append(f"正收益月: {m_summary.get('positive_months', 0)} / "
                 f"{m_summary.get('total_months', 0)}\n\n")

    lines.append("--- 持仓时长 ---\n")
    for label, s in hold_summary.items():
        lines.append(
            f"{label:12s}: 笔数={s['count']:3d}  胜率={s['win_rate_pct']:5.2f}%  "
            f"平均={s['avg_pnl']:+7.2f}  总={s['total_pnl']:+8.2f}\n"
        )
    lines.append("\n")

    lines.append("--- 连胜 / 连亏 ---\n")
    lines.append(f"最长连胜: {streak_stats.get('longest_win_streak', 0)} 笔, "
                 f"累计 {streak_stats.get('longest_win_streak_pnl', 0):+.2f}\n")
    lines.append(f"最长连亏: {streak_stats.get('longest_loss_streak', 0)} 笔, "
                 f"累计 {streak_stats.get('longest_loss_streak_pnl', 0):+.2f}\n")
    lines.append(f"最大连续亏损金额: {streak_stats.get('max_consecutive_dd_usdt', 0):.2f} USDT\n")
    lines.append(f"恢复时间: {streak_stats.get('max_consecutive_dd_recovery_h', 0):.1f} 小时 "
                 f"({streak_stats.get('max_consecutive_dd_recovery_h', 0)/24:.1f} 天)\n\n")

    if fgi_summary:
        lines.append("--- FGI 市场情绪 ---\n")
        for label, s in fgi_summary.items():
            lines.append(
                f"{label:14s}: 笔数={s['count']:3d}  胜率={s['win_rate_pct']:5.2f}%  "
                f"平均={s['avg_pnl']:+7.2f}  总={s['total_pnl']:+8.2f}\n"
            )

    summary_text = "".join(lines)
    print("\n" + summary_text)
    (out_dir / "summary.txt").write_text(summary_text, encoding="utf-8")

    # HTML
    (out_dir / "report.html").write_text(
        _HTML_TPL.format(summary=summary_text),
        encoding="utf-8",
    )

    print(f"\n输出目录：{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
