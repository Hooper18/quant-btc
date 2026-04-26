"""Walk-Forward 滚动前向验证。

工作流：
- 数据切成滚动窗口：训练 train_months + 测试 test_months，步进 step_months（默认 = test_months）
- 例：train=12, test=3, step=3 → 21 个窗口（覆盖 2020-01..2026-03 数据）
- 每窗口可选窗口内参数优化（param_grid 不为 None 时）
- 拼接所有测试期净值 → 计算汇总年化/夏普/最大回撤/胜率

设计要点：
- 指标已在外层 data_dict 上预计算；validator 只做切片 + 回测
- 各窗口测试集 backtester 都从 initial_balance 起跑；汇总时通过 multiplier 链式拼接

Walk-Forward 是比 Phase 9 单次 train/test 更严苛的样本外检验：
- 模拟"实盘中每季度调一次参数"的真实使用方式
- 每个测试窗口的参数都来自截止当时的"过去"，不偷看未来
- 汇总夏普 < 0.5 通常意味着策略不可靠
"""
from __future__ import annotations

import calendar
import logging
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from .backtester import Backtester
from .optimizer import StrategyOptimizer

logger = logging.getLogger(__name__)


def _add_months(d: date, months: int) -> date:
    total = d.month - 1 + months
    new_y = d.year + total // 12
    new_m = total % 12 + 1
    new_d = min(d.day, calendar.monthrange(new_y, new_m)[1])
    return date(new_y, new_m, new_d)


def _to_dt_utc(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


@dataclass
class WindowResult:
    index: int                              # 1-based 窗口编号
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    test_equity_curve: list[float] = field(default_factory=list)
    test_timestamps: list[datetime] = field(default_factory=list)
    best_params: dict[str, Any] | None = None   # 仅 --optimize 模式有值


@dataclass
class WalkForwardResult:
    windows: list[WindowResult]
    combined_equity: list[float]
    combined_timestamps: list[datetime]
    summary: dict[str, float]
    primary_tf: str
    optimized: bool

    def passes_sharpe_check(self, threshold: float = 0.5) -> bool:
        return self.summary.get("sharpe_ratio", 0.0) >= threshold

    # ---------- 报告 ----------
    def print_report(self) -> None:
        n = len(self.windows)
        print()
        print("=" * 130)
        print(f"Walk-Forward 报告（{'窗口内优化' if self.optimized else '固定参数'}，{n} 个窗口）")
        print("=" * 130)
        headers = ["#", "训练期", "测试期", "训练收益%", "训练夏普", "测试收益%", "测试夏普",
                   "测试回撤%", "测试胜率%", "测试交易"]
        widths = [3, 21, 21, 10, 9, 10, 9, 10, 10, 10]
        print("  ".join(f"{h:>{w}s}" for h, w in zip(headers, widths)))
        print("-" * 130)
        for w in self.windows:
            train_lbl = f"{w.train_start.date()}→{w.train_end.date()}"
            test_lbl = f"{w.test_start.date()}→{w.test_end.date()}"
            tr = w.train_metrics
            te = w.test_metrics
            cells = [
                f"{w.index:>3d}",
                f"{train_lbl:>21s}",
                f"{test_lbl:>21s}",
                f"{tr.get('total_return_pct', 0):>10.2f}",
                f"{tr.get('sharpe_ratio', 0):>9.3f}",
                f"{te.get('total_return_pct', 0):>10.2f}",
                f"{te.get('sharpe_ratio', 0):>9.3f}",
                f"{te.get('max_drawdown_pct', 0):>10.2f}",
                f"{te.get('win_rate_pct', 0):>10.2f}",
                f"{int(te.get('total_trades', 0)):>10d}",
            ]
            print("  ".join(cells))
        print("=" * 130)

        # 汇总
        s = self.summary
        print("\n========== 汇总（所有测试期拼接）==========")
        print(f"窗口数:        {int(s.get('n_windows', 0))}")
        print(f"测试期累计 bars: {int(s.get('combined_bars', 0))}")
        print(f"总收益率:      {s.get('total_return_pct', 0):.2f}%")
        print(f"年化收益率:    {s.get('annualized_return_pct', 0):.2f}%")
        print(f"夏普比率:      {s.get('sharpe_ratio', 0):.3f}")
        print(f"最大回撤:      {s.get('max_drawdown_pct', 0):.2f}%")
        print(f"胜率（加权）:  {s.get('win_rate_pct', 0):.2f}%")
        print(f"交易笔数:      {int(s.get('total_trades', 0))}")
        print("==========================================")

        if not self.passes_sharpe_check(0.5):
            print("\n⚠️  策略在 Walk-Forward 验证中表现不佳（汇总夏普 < 0.5），不建议实盘")
        else:
            print("\n✓  Walk-Forward 验证通过（汇总夏普 ≥ 0.5）")


class WalkForwardValidator:
    def __init__(
        self,
        strategy_path: str | Path,
        data_dict: dict[str, pl.DataFrame],
        backtest_config_path: str | Path = "config/backtest_config.yaml",
    ):
        self.strategy_path = Path(strategy_path)
        with self.strategy_path.open(encoding="utf-8") as f:
            self.base_yaml: dict[str, Any] = yaml.safe_load(f) or {}
        self.data_dict = data_dict
        self.backtester = Backtester.from_yaml(backtest_config_path)
        self.bt_config_path = backtest_config_path
        self.primary_tf = self.backtester.primary_tf
        if self.primary_tf not in data_dict:
            raise ValueError(
                f"data_dict 缺主 TF={self.primary_tf}；已有 {list(data_dict)}"
            )

    # ---------- 窗口生成 ----------
    def _windows(
        self, train_months: int, test_months: int, step_months: int,
    ) -> list[tuple[datetime, datetime, datetime]]:
        primary = self.data_dict[self.primary_tf]
        first_ts: datetime = primary["timestamp"].min()
        last_ts: datetime = primary["timestamp"].max()
        # train_start 取首月第 1 天（向上取整到下月第一天若不在月初）
        if first_ts.day != 1 or first_ts.hour != 0:
            first_month = _add_months(date(first_ts.year, first_ts.month, 1), 1)
        else:
            first_month = date(first_ts.year, first_ts.month, 1)

        out: list[tuple[datetime, datetime, datetime]] = []
        train_start = first_month
        # last_ts.date() 之后的下一天作为可包含上界
        last_inclusive = last_ts.date()
        while True:
            train_end = _add_months(train_start, train_months)
            test_end = _add_months(train_end, test_months)
            if test_end > last_inclusive:
                break
            out.append((_to_dt_utc(train_start), _to_dt_utc(train_end), _to_dt_utc(test_end)))
            train_start = _add_months(train_start, step_months)
        return out

    # ---------- 切片 ----------
    def _slice(self, ts_start: datetime, ts_end: datetime) -> dict[str, pl.DataFrame]:
        return {
            tf: df.filter(
                (pl.col("timestamp") >= ts_start) & (pl.col("timestamp") < ts_end)
            )
            for tf, df in self.data_dict.items()
        }

    # ---------- 回测 ----------
    def _run_one(
        self,
        strategy_yaml_dict: dict[str, Any],
        data_dict: dict[str, pl.DataFrame],
        fr_df: pl.DataFrame | None,
    ):
        tmp_path = Path(tempfile.gettempdir()) / f"wf_strat_{uuid.uuid4().hex}.yaml"
        tmp_path.write_text(
            yaml.safe_dump(strategy_yaml_dict, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        try:
            return self.backtester.run(
                data_dict, str(tmp_path), funding_rate_df=fr_df,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    # ---------- 主入口 ----------
    def run(
        self,
        train_months: int = 12,
        test_months: int = 3,
        step_months: int | None = None,
        param_grid: dict[str, list[Any]] | None = None,
        metric: str = "sharpe",
        funding_rate_df: pl.DataFrame | None = None,
    ) -> WalkForwardResult:
        if step_months is None:
            step_months = test_months
        windows_def = self._windows(train_months, test_months, step_months)
        if not windows_def:
            raise RuntimeError("数据不足以构造任何 train+test 窗口")
        logger.info(
            "构造 %d 个窗口（train=%d 月, test=%d 月, step=%d 月）",
            len(windows_def), train_months, test_months, step_months,
        )

        optimizer: StrategyOptimizer | None = None
        if param_grid is not None:
            optimizer = StrategyOptimizer(self.strategy_path, self.data_dict, self.bt_config_path)

        results: list[WindowResult] = []
        for i, (train_start, train_end, test_end) in enumerate(windows_def, 1):
            test_start = train_end
            train_data = self._slice(train_start, train_end)
            test_data = self._slice(train_end, test_end)
            primary_train_n = train_data[self.primary_tf].height
            primary_test_n = test_data[self.primary_tf].height
            if primary_train_n == 0 or primary_test_n == 0:
                logger.warning("窗口 %d 切出空集，跳过", i)
                continue

            # 1) 决定本窗口的 strategy YAML
            best_params: dict[str, Any] | None = None
            if optimizer is not None and param_grid is not None:
                # 窗口内网格搜索；进度前缀让多窗口日志可读
                best_params, _ = optimizer.grid_search(
                    train_data, param_grid, metric=metric,
                    funding_rate_df=funding_rate_df,
                    progress=False,   # 多窗口下静默；只打窗口级摘要
                )
                strat_yaml = optimizer.make_strategy_yaml(best_params)
            else:
                strat_yaml = self.base_yaml

            # 2) train 期回测（用于报告对比）
            train_result = self._run_one(strat_yaml, train_data, funding_rate_df)
            # 3) test 期回测（核心样本外评估）
            test_result = self._run_one(strat_yaml, test_data, funding_rate_df)

            results.append(WindowResult(
                index=i,
                train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end,
                train_metrics=train_result.metrics,
                test_metrics=test_result.metrics,
                test_equity_curve=list(test_result.equity_curve),
                test_timestamps=list(test_result.timestamps),
                best_params=best_params,
            ))
            params_msg = f"  best_params={best_params}" if best_params else ""
            print(
                f"[Window {i}/{len(windows_def)}] "
                f"train {train_start.date()}→{train_end.date()} ({primary_train_n}b) "
                f"| test {test_start.date()}→{test_end.date()} ({primary_test_n}b) "
                f"| test_sharpe={test_result.metrics.get('sharpe_ratio', 0):+.3f}"
                f" return={test_result.metrics.get('total_return_pct', 0):+.2f}%{params_msg}"
            )

        return self._build(results)

    # ---------- 拼接 + 汇总 ----------
    def _build(self, windows: list[WindowResult]) -> WalkForwardResult:
        initial = float(self.backtester.initial_balance)
        combined_eq: list[float] = []
        combined_ts: list[datetime] = []
        running = initial
        for w in windows:
            w_init = float(w.test_metrics.get("initial_balance", initial))
            if w_init <= 0 or not w.test_equity_curve:
                continue
            for ts, eq in zip(w.test_timestamps, w.test_equity_curve):
                combined_eq.append(eq / w_init * running)
                combined_ts.append(ts)
            running = w.test_equity_curve[-1] / w_init * running

        summary = self._summary(combined_eq, combined_ts, windows, initial)
        optimized = any(w.best_params is not None for w in windows)
        return WalkForwardResult(
            windows=windows,
            combined_equity=combined_eq,
            combined_timestamps=combined_ts,
            summary=summary,
            primary_tf=self.primary_tf,
            optimized=optimized,
        )

    def _summary(
        self,
        eq: list[float],
        ts: list[datetime],
        windows: list[WindowResult],
        initial: float,
    ) -> dict[str, float]:
        if not eq or not ts:
            return {"n_windows": len(windows), "combined_bars": 0}

        final_eq = eq[-1]
        total_return = (final_eq - initial) / initial
        days = max((ts[-1] - ts[0]).total_seconds() / 86400.0, 1.0)
        annual = (1 + total_return) ** (365.0 / days) - 1.0 if (1 + total_return) > 0 else -1.0

        # bar-级对数收益的夏普
        rets = [
            (eq[i] - eq[i - 1]) / eq[i - 1]
            for i in range(1, len(eq))
            if eq[i - 1] > 0
        ]
        if rets:
            mean_r = sum(rets) / len(rets)
            var = sum((r - mean_r) ** 2 for r in rets) / max(len(rets) - 1, 1)
            std = var ** 0.5
            tf_per_year = {
                "1m": 525600, "5m": 105120, "15m": 35040,
                "1h": 8760, "4h": 2190, "1d": 365,
            }.get(self.primary_tf, 8760)
            sharpe = (mean_r / std * (tf_per_year ** 0.5)) if std > 0 else 0.0
        else:
            sharpe = 0.0

        # 最大回撤
        peak = -float("inf")
        max_dd = 0.0
        for v in eq:
            peak = max(peak, v)
            if peak > 0:
                max_dd = max(max_dd, (peak - v) / peak)

        # 加权胜率
        total_trades = sum(int(w.test_metrics.get("total_trades", 0)) for w in windows)
        if total_trades > 0:
            weighted_wins = sum(
                int(w.test_metrics.get("total_trades", 0))
                * float(w.test_metrics.get("win_rate_pct", 0)) / 100.0
                for w in windows
            )
            win_rate = weighted_wins / total_trades * 100
        else:
            win_rate = 0.0

        return {
            "n_windows": len(windows),
            "combined_bars": len(eq),
            "final_equity": final_eq,
            "total_return_pct": total_return * 100,
            "annualized_return_pct": annual * 100,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd * 100,
            "win_rate_pct": win_rate,
            "total_trades": total_trades,
        }
