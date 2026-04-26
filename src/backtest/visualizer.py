"""回测结果可视化（matplotlib，深色主题）。

四张图：
- equity_curve: 上=净值+BTC价格(双 Y 轴)，下=回撤；最大回撤区间高亮
- trades:      BTC 价格曲线 + 买卖点叠加；可指定时间窗口
- monthly_returns: 月度收益率热力图（年×月）
- metrics_summary: 指标表格 + 收益分布直方图 + 持仓时间分布

save_report 把上面 4 图存 PNG + 拼成 report.html。
"""
from __future__ import annotations

import base64
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from .backtester import BacktestResult, Trade

logger = logging.getLogger(__name__)


# 深色主题
_DARK_RC = {
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#e6edf3",
    "axes.titlecolor": "#e6edf3",
    "axes.titleweight": "bold",
    "xtick.color": "#9ba1a8",
    "ytick.color": "#9ba1a8",
    "grid.color": "#30363d",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "text.color": "#e6edf3",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "savefig.facecolor": "#0e1117",
    "savefig.edgecolor": "none",
    "font.family": "sans-serif",
    # CJK 字体回退链：Windows 中文系统优先 Microsoft YaHei；其它平台依赖各自 Unicode 字体
    "font.sans-serif": [
        "Microsoft YaHei", "SimHei", "PingFang SC", "Hiragino Sans GB",
        "Noto Sans CJK SC", "Source Han Sans SC", "Arial Unicode MS", "DejaVu Sans",
    ],
    "axes.unicode_minus": False,  # 防止负号显示为方框
    "font.size": 9,
}

_COLOR_NAV = "#4cd2ff"
_COLOR_BTC = "#f0a868"
_COLOR_DD = "#ff6b6b"
_COLOR_LONG = "#3fb950"
_COLOR_SHORT = "#f85149"
_COLOR_CLOSE = "#9ba1a8"


def _apply_style() -> None:
    plt.rcParams.update(_DARK_RC)


class BacktestVisualizer:
    def __init__(self, result: BacktestResult, price_df: pl.DataFrame):
        if not result.equity_curve:
            raise ValueError("BacktestResult.equity_curve 为空，无法可视化")
        self.result = result
        self.price_df = price_df.sort("timestamp")
        self._eq = np.array(result.equity_curve, dtype=float)
        self._ts = list(result.timestamps)
        # 净值峰值 / 回撤
        self._peak = np.maximum.accumulate(self._eq)
        with np.errstate(divide="ignore", invalid="ignore"):
            self._dd = np.where(self._peak > 0, (self._peak - self._eq) / self._peak, 0.0)
        # 最大回撤区间
        if len(self._dd):
            self._max_dd_end = int(np.argmax(self._dd))
            self._max_dd_start = int(np.argmax(self._eq[: self._max_dd_end + 1])) if self._max_dd_end > 0 else 0
        else:
            self._max_dd_start = self._max_dd_end = 0

    # ---------- 净值 + 回撤 ----------
    def plot_equity_curve(self, ax_eq: plt.Axes, ax_dd: plt.Axes) -> None:
        ts = self._ts
        ax_eq.plot(ts, self._eq, color=_COLOR_NAV, linewidth=1.2, label="净值 NAV")
        ax_eq.set_ylabel("净值 (USDT)", color=_COLOR_NAV)
        ax_eq.tick_params(axis="y", labelcolor=_COLOR_NAV)
        ax_eq.grid(True, alpha=0.3)

        # 双 Y 轴：BTC 收盘
        ax_btc = ax_eq.twinx()
        ax_btc.plot(
            self.price_df["timestamp"].to_list(),
            self.price_df["close"].to_list(),
            color=_COLOR_BTC, linewidth=0.7, alpha=0.55, label="BTC 收盘",
        )
        ax_btc.set_ylabel("BTC 价格 (USDT)", color=_COLOR_BTC)
        ax_btc.tick_params(axis="y", labelcolor=_COLOR_BTC)

        # 标注最大回撤区间
        if self._max_dd_end > self._max_dd_start:
            ax_eq.axvspan(
                ts[self._max_dd_start], ts[self._max_dd_end],
                color=_COLOR_DD, alpha=0.15, label=f"最大回撤 {self._dd[self._max_dd_end]*100:.1f}%",
            )

        # 合并双轴 legend
        h1, l1 = ax_eq.get_legend_handles_labels()
        h2, l2 = ax_btc.get_legend_handles_labels()
        ax_eq.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)

        ax_eq.set_title("净值曲线 + BTC 价格")

        # 回撤
        ax_dd.fill_between(ts, self._dd * 100, 0, color=_COLOR_DD, alpha=0.5)
        ax_dd.plot(ts, self._dd * 100, color=_COLOR_DD, linewidth=0.8)
        ax_dd.set_ylabel("回撤 (%)")
        ax_dd.set_xlabel("时间")
        ax_dd.set_ylim(self._dd.max() * 100 * 1.05 if self._dd.max() > 0 else 1, 0)
        ax_dd.grid(True, alpha=0.3)
        ax_dd.set_title(f"回撤曲线 (最大 {self._dd.max()*100:.2f}%)")

        for ax in (ax_eq, ax_dd):
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            for label in ax.get_xticklabels():
                label.set_rotation(30)
                label.set_ha("right")

    # ---------- 交易点 ----------
    def plot_trades(
        self,
        ax: plt.Axes,
        start_date: date | datetime | None = None,
        end_date: date | datetime | None = None,
    ) -> None:
        df = self.price_df
        if start_date is not None or end_date is not None:
            mask = pl.lit(True)
            if start_date is not None:
                mask = mask & (pl.col("timestamp") >= pl.lit(start_date))
            if end_date is not None:
                mask = mask & (pl.col("timestamp") <= pl.lit(end_date))
            df = df.filter(mask)

        if df.height == 0:
            ax.set_title("交易点（无数据）")
            return

        ts_p = df["timestamp"].to_list()
        close_p = df["close"].to_list()
        ax.plot(ts_p, close_p, color=_COLOR_BTC, linewidth=0.8, alpha=0.85, label="BTC 收盘")

        # 按时间窗口过滤交易
        ts_min, ts_max = ts_p[0], ts_p[-1]
        trades_in = [
            t for t in self.result.trades
            if ts_min <= t.timestamp <= ts_max
        ]
        long_open = [t for t in trades_in if t.side == "long_open"]
        short_open = [t for t in trades_in if t.side == "short_open"]
        closes = [t for t in trades_in if t.side.endswith("_close") or t.side == "liquidate"]

        if long_open:
            ax.scatter(
                [t.timestamp for t in long_open], [t.price for t in long_open],
                marker="^", s=55, color=_COLOR_LONG, edgecolor="white", linewidth=0.4,
                label=f"做多入场 ({len(long_open)})", zorder=3,
            )
        if short_open:
            ax.scatter(
                [t.timestamp for t in short_open], [t.price for t in short_open],
                marker="v", s=55, color=_COLOR_SHORT, edgecolor="white", linewidth=0.4,
                label=f"做空入场 ({len(short_open)})", zorder=3,
            )
        if closes:
            ax.scatter(
                [t.timestamp for t in closes], [t.price for t in closes],
                marker="x", s=40, color=_COLOR_CLOSE, linewidth=1.2,
                label=f"平仓 ({len(closes)})", zorder=3,
            )

        ax.set_ylabel("BTC 价格 (USDT)")
        ax.set_xlabel("时间")
        title = "买卖点标注"
        if start_date or end_date:
            title += f" [{start_date or ''} → {end_date or ''}]"
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")

    # ---------- 月度收益热力图 ----------
    def _monthly_returns(self) -> tuple[list[int], np.ndarray]:
        """返回 (years, matrix[year_idx, month-1]) 月度收益率（小数）。"""
        if not self._ts:
            return [], np.zeros((0, 12))
        df = pl.DataFrame({"ts": self._ts, "eq": self._eq.tolist()}).sort("ts").with_columns(
            pl.col("ts").dt.year().alias("y"),
            pl.col("ts").dt.month().alias("m"),
        )
        # 每月最后一根 K 线的净值
        monthly_last = (
            df.group_by(["y", "m"])
            .agg(pl.col("eq").last().alias("eq_end"))
            .sort(["y", "m"])
        )
        years_sorted = sorted(set(monthly_last["y"].to_list()))
        ymin = min(years_sorted)
        ymax = max(years_sorted)
        years = list(range(ymin, ymax + 1))
        mat = np.full((len(years), 12), np.nan, dtype=float)

        # 用 initial_balance 做"上月末"占位起点
        prev_eq = float(self.result.metrics.get("initial_balance", self._eq[0]))
        rows = monthly_last.iter_rows(named=True)
        for r in rows:
            y, m, eq_end = int(r["y"]), int(r["m"]), float(r["eq_end"])
            ret = (eq_end - prev_eq) / prev_eq if prev_eq > 0 else 0.0
            mat[years.index(y), m - 1] = ret
            prev_eq = eq_end
        return years, mat

    def plot_monthly_returns(self, ax: plt.Axes) -> None:
        years, mat = self._monthly_returns()
        if not years:
            ax.set_title("月度收益率（无数据）")
            return
        bound = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)), 1e-6)
        im = ax.imshow(mat * 100, cmap="RdYlGn", aspect="auto", vmin=-bound * 100, vmax=bound * 100)
        ax.set_xticks(range(12))
        ax.set_xticklabels(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels(years)
        ax.set_xlabel("月")
        ax.set_ylabel("年")
        ax.set_title("月度收益率热力图 (%)")
        # 标注数值
        for i, _ in enumerate(years):
            for j in range(12):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                ax.text(
                    j, i, f"{v*100:.1f}",
                    ha="center", va="center",
                    fontsize=7,
                    color="white" if abs(v) > bound * 0.5 else "#202020",
                )
        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.ax.tick_params(colors="#9ba1a8")

    # ---------- 指标摘要 + 分布 ----------
    def plot_metrics_summary(self, ax_table: plt.Axes, ax_hist: plt.Axes, ax_hold: plt.Axes) -> None:
        m = self.result.metrics
        rows = [
            ("初始资金 (USDT)", f"{m.get('initial_balance', 0):.2f}"),
            ("最终资产 (USDT)", f"{m.get('final_equity', 0):.2f}"),
            ("总收益率 (%)", f"{m.get('total_return_pct', 0):.2f}"),
            ("年化收益率 (%)", f"{m.get('annualized_return_pct', 0):.2f}"),
            ("夏普比率", f"{m.get('sharpe_ratio', 0):.3f}"),
            ("最大回撤 (%)", f"{m.get('max_drawdown_pct', 0):.2f}"),
            ("总交易笔数", f"{int(m.get('total_trades', 0))}"),
            ("胜率 (%)", f"{m.get('win_rate_pct', 0):.2f}"),
            ("盈亏比", f"{m.get('profit_loss_ratio', 0):.3f}"),
            ("平均持仓 (h)", f"{m.get('avg_holding_hours', 0):.2f}"),
            ("是否熔断", "是" if m.get("circuit_breaker") else "否"),
        ]
        ax_table.axis("off")
        ax_table.set_title("关键指标")
        table = ax_table.table(
            cellText=rows, colLabels=["指标", "值"], loc="center", cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#30363d")
            cell.set_facecolor("#1c2128" if r % 2 else "#161b22")
            cell.get_text().set_color("#e6edf3")
            if r == 0:
                cell.set_facecolor("#22272e")
                cell.get_text().set_weight("bold")

        # 收益分布：每笔已平仓 trade 的 pnl
        closed = [t for t in self.result.trades
                  if t.side.endswith("_close") or t.side == "liquidate"]
        pnls = [t.pnl for t in closed if t.pnl != 0]
        if pnls:
            ax_hist.hist(pnls, bins=40, color=_COLOR_NAV, edgecolor="#0e1117", alpha=0.85)
            ax_hist.axvline(0, color="white", linewidth=0.7, linestyle="--", alpha=0.6)
            ax_hist.set_title("单笔已实现 PnL 分布 (USDT)")
            ax_hist.set_xlabel("PnL")
            ax_hist.set_ylabel("笔数")
            ax_hist.grid(True, alpha=0.3)
        else:
            ax_hist.set_title("PnL 分布（无数据）")

        # 持仓时长分布：配对 open→close
        opens: list[Trade] = []
        durations_h: list[float] = []
        for t in self.result.trades:
            if t.side.endswith("_open"):
                opens.append(t)
            elif (t.side.endswith("_close") or t.side == "liquidate") and opens:
                opener = opens.pop(0)
                durations_h.append((t.timestamp - opener.timestamp).total_seconds() / 3600.0)
        if durations_h:
            ax_hold.hist(durations_h, bins=40, color=_COLOR_LONG, edgecolor="#0e1117", alpha=0.85)
            ax_hold.set_title(f"持仓时长分布（中位 {np.median(durations_h):.1f}h）")
            ax_hold.set_xlabel("小时")
            ax_hold.set_ylabel("笔数")
            ax_hold.grid(True, alpha=0.3)
        else:
            ax_hold.set_title("持仓时长（无数据）")

    # ---------- 一键导出 ----------
    def save_report(self, output_dir: str | Path) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _apply_style()

        # 1) equity + drawdown
        fig1 = plt.figure(figsize=(14, 7))
        gs1 = GridSpec(3, 1, hspace=0.35, height_ratios=[2.2, 0.05, 1])
        ax_eq = fig1.add_subplot(gs1[0, 0])
        ax_dd = fig1.add_subplot(gs1[2, 0], sharex=ax_eq)
        self.plot_equity_curve(ax_eq, ax_dd)
        p1 = out / "equity_curve.png"
        fig1.savefig(p1, dpi=130, bbox_inches="tight")
        plt.close(fig1)

        # 2) trades 全程 + 最近 90 天
        fig2 = plt.figure(figsize=(14, 6))
        ax_tr = fig2.add_subplot(1, 1, 1)
        self.plot_trades(ax_tr)
        p2 = out / "trades_all.png"
        fig2.savefig(p2, dpi=130, bbox_inches="tight")
        plt.close(fig2)

        fig2b = plt.figure(figsize=(14, 6))
        ax_tr2 = fig2b.add_subplot(1, 1, 1)
        if self._ts:
            from datetime import timedelta
            zoom_end = self._ts[-1]
            zoom_start = zoom_end - timedelta(days=90)
            self.plot_trades(ax_tr2, start_date=zoom_start, end_date=zoom_end)
        p2b = out / "trades_recent90d.png"
        fig2b.savefig(p2b, dpi=130, bbox_inches="tight")
        plt.close(fig2b)

        # 3) 月度热力图
        fig3 = plt.figure(figsize=(11, max(3, 0.55 * (self._ts[-1].year - self._ts[0].year + 2))))
        ax_mh = fig3.add_subplot(1, 1, 1)
        self.plot_monthly_returns(ax_mh)
        p3 = out / "monthly_returns.png"
        fig3.savefig(p3, dpi=130, bbox_inches="tight")
        plt.close(fig3)

        # 4) 指标摘要
        fig4 = plt.figure(figsize=(15, 8))
        gs4 = GridSpec(2, 2, hspace=0.35, wspace=0.25)
        ax_t = fig4.add_subplot(gs4[:, 0])
        ax_h = fig4.add_subplot(gs4[0, 1])
        ax_d = fig4.add_subplot(gs4[1, 1])
        self.plot_metrics_summary(ax_t, ax_h, ax_d)
        p4 = out / "metrics_summary.png"
        fig4.savefig(p4, dpi=130, bbox_inches="tight")
        plt.close(fig4)

        # HTML 拼接
        html_path = out / "report.html"
        m = self.result.metrics
        html = self._render_html([p1, p2, p2b, p3, p4], m)
        html_path.write_text(html, encoding="utf-8")
        logger.info("报告已生成：%s", html_path)
        return html_path

    @staticmethod
    def _render_html(image_paths: list[Path], metrics: dict[str, Any]) -> str:
        sections = "\n".join(
            f'<section><h2>{p.stem}</h2><img src="{p.name}" alt="{p.stem}"></section>'
            for p in image_paths
        )
        kv = "\n".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in metrics.items()
        )
        return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>BTC 量化回测报告</title>
<style>
  body {{ background: #0e1117; color: #e6edf3; font-family: -apple-system, system-ui, sans-serif; margin: 24px; }}
  h1 {{ color: #4cd2ff; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
  h2 {{ color: #f0a868; margin-top: 28px; }}
  img {{ max-width: 100%; border: 1px solid #30363d; border-radius: 6px; background: #0e1117; }}
  table {{ border-collapse: collapse; margin: 12px 0 24px; }}
  td, th {{ border: 1px solid #30363d; padding: 6px 14px; text-align: left; }}
  tr:nth-child(even) {{ background: #161b22; }}
</style>
</head>
<body>
<h1>BTC 量化回测报告</h1>
<table><tr><th>指标</th><th>值</th></tr>{kv}</table>
{sections}
</body>
</html>
"""
