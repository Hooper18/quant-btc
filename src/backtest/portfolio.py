"""策略组合回测：多个 (策略, 币种, 资金分配) 子账户独立运行后汇总组合净值。

设计：
- 每个 sleeve = 一个 (strategy_yaml, symbol, allocation_pct)
- 各 sleeve 独立 Backtester.run，初始资金 = 总本金 × allocation
- 汇总：按时间戳对齐各 sleeve 的 equity_curve（1h primary 默认对齐），求和得到组合净值
- 风险指标基于汇总后的组合净值（不是各 sleeve 平均）
- 各币种贡献度 = sleeve 期末权益 - sleeve 初始资金
"""
from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from backtest.backtester import Backtester, BacktestResult
from data import merge_market_data
from indicators import IndicatorEngine
from utils.config import DataConfig

logger = logging.getLogger(__name__)


@dataclass
class SleeveConfig:
    strategy_path: Path
    symbol: str
    allocation: float  # 0..1


@dataclass
class SleeveRun:
    cfg: SleeveConfig
    initial_capital: float
    result: BacktestResult


@dataclass
class PortfolioResult:
    sleeves: list[SleeveRun]
    timestamps: list[datetime]
    equity_curve: list[float]
    metrics: dict[str, float]
    contribution: dict[str, float] = field(default_factory=dict)

    def print_summary(self) -> None:
        print("\n========== 组合回测摘要 ==========")
        print(f"初始本金:      {self.metrics.get('initial_balance', 0):.2f} USDT")
        print(f"组合期末:      {self.metrics.get('final_equity', 0):.2f} USDT")
        print(f"组合总收益率:  {self.metrics.get('total_return_pct', 0):.2f}%")
        print(f"年化:          {self.metrics.get('annualized_return_pct', 0):.2f}%")
        print(f"夏普:          {self.metrics.get('sharpe_ratio', 0):.3f}")
        print(f"最大回撤:      {self.metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"对齐 bar 数:    {self.metrics.get('aligned_bars', 0):.0f}")
        print("---------- 各子账户 ----------")
        for sl in self.sleeves:
            r = sl.result.metrics
            sym = sl.cfg.symbol
            print(
                f"  {sym:<8} alloc={sl.cfg.allocation:.0%} "
                f"start={sl.initial_capital:.2f} → end={r.get('final_equity', 0):.2f} "
                f"({r.get('total_return_pct', 0):+.1f}%) "
                f"sharpe={r.get('sharpe_ratio', 0):.2f} "
                f"trades={int(r.get('total_trades', 0))}"
            )
        print("---------- 贡献度 ----------")
        total_pnl = sum(self.contribution.values()) or 1.0
        for sym, pnl in self.contribution.items():
            pct = pnl / total_pnl * 100 if total_pnl else 0
            print(f"  {sym:<8} {pnl:+.2f} USDT ({pct:+.1f}% of组合PnL)")
        print("==================================")


# 复用 run_backtest.py 的指标解析（避免重复实现）
def _import_run_backtest_helpers():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from run_backtest import (  # type: ignore
        _collect_required_indicators, _load_ohlcv, load_aux_data,
    )
    return _collect_required_indicators, _load_ohlcv, load_aux_data


def _build_data_dict_for_symbol(
    data_cfg: DataConfig,
    sym: str,
    used_tfs: set[str],
    ind_cfg: list[tuple[str, dict[str, Any]]],
) -> tuple[dict[str, pl.DataFrame], pl.DataFrame | None]:
    _, _load_ohlcv, load_aux_data = _import_run_backtest_helpers()
    sym_cfg = data_cfg.for_symbol(sym)
    aux = load_aux_data(sym_cfg)  # 新币种通常没有 aux，全 None
    out: dict[str, pl.DataFrame] = {}
    for tf in sorted(used_tfs):
        raw = _load_ohlcv(sym_cfg, tf)
        if raw is None or raw.height == 0:
            raise FileNotFoundError(f"缺 {sym} {tf} 周期 OHLCV 数据")
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
    return out, aux.get("funding")


def _collect_used_tfs(strategies: list[dict[str, Any]], primary_tf: str) -> set[str]:
    used: set[str] = {primary_tf}
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


class PortfolioBacktester:
    """组合回测器：每个 sleeve 独立 Backtester.run，按时间戳对齐汇总。"""

    def __init__(
        self,
        sleeves: list[SleeveConfig],
        data_cfg: DataConfig,
        bt: Backtester,
        total_balance: float,
    ):
        if not sleeves:
            raise ValueError("sleeves 不能为空")
        total_alloc = sum(s.allocation for s in sleeves)
        if abs(total_alloc - 1.0) > 1e-6:
            logger.warning("分配总和 = %.4f ≠ 1.0；按比例归一化", total_alloc)
            sleeves = [
                SleeveConfig(s.strategy_path, s.symbol, s.allocation / total_alloc)
                for s in sleeves
            ]
        self.sleeves = sleeves
        self.data_cfg = data_cfg
        self.bt_template = bt
        self.total_balance = total_balance

    def _run_sleeve(self, sl: SleeveConfig) -> SleeveRun:
        capital = self.total_balance * sl.allocation
        with sl.strategy_path.open("r", encoding="utf-8") as f:
            strat_yaml = yaml.safe_load(f) or {}
        strategies = strat_yaml.get("strategies", [])
        used_tfs = _collect_used_tfs(strategies, self.bt_template.primary_tf)

        _collect_indicators, _, _ = _import_run_backtest_helpers()
        ind_cfg = _collect_indicators(strategies)
        data_dict, fr_df = _build_data_dict_for_symbol(
            self.data_cfg, sl.symbol, used_tfs, ind_cfg,
        )
        # 用同一回测参数（杠杆/手续费/滑点）但本金按分配缩放
        bt = Backtester(
            initial_balance=capital,
            leverage=self.bt_template.leverage,
            fee_rate=self.bt_template.fee_rate,
            slippage=self.bt_template.slippage,
            maintenance_margin_rate=self.bt_template.maintenance_margin_rate,
            max_drawdown_pct=self.bt_template.max_drawdown_pct,
            funding_rate_epochs_utc=list(self.bt_template.funding_epochs),
            primary_timeframe=self.bt_template.primary_tf,
            daily_max_loss_pct=self.bt_template.daily_max_loss_pct,
        )
        logger.info(
            "运行 sleeve: symbol=%s alloc=%.0f%% capital=%.2f",
            sl.symbol, sl.allocation * 100, capital,
        )
        result = bt.run(data_dict, sl.strategy_path, funding_rate_df=fr_df)
        return SleeveRun(cfg=sl, initial_capital=capital, result=result)

    def run(self) -> PortfolioResult:
        runs = [self._run_sleeve(sl) for sl in self.sleeves]
        return self._aggregate(runs)

    def _aggregate(self, runs: list[SleeveRun]) -> PortfolioResult:
        # 按时间戳对齐：取最晚的 start 和 最早的 end，做一次内连接
        all_dfs: list[pl.DataFrame] = []
        for i, r in enumerate(runs):
            df = pl.DataFrame({
                "timestamp": r.result.timestamps,
                f"eq_{i}": r.result.equity_curve,
            })
            all_dfs.append(df)
        merged = all_dfs[0]
        for df in all_dfs[1:]:
            merged = merged.join(df, on="timestamp", how="inner")
        merged = merged.sort("timestamp")
        ts = merged["timestamp"].to_list()
        eq_cols = [c for c in merged.columns if c.startswith("eq_")]
        # 组合净值 = 各 sleeve 同时点 equity 求和
        total = [0.0] * merged.height
        for col in eq_cols:
            vals = merged[col].to_list()
            for i, v in enumerate(vals):
                total[i] += float(v)

        metrics = self._metrics(total, ts)
        # 贡献度（基于各 sleeve 自身回测期末，不依赖对齐窗口）
        contribution = {
            r.cfg.symbol: r.result.metrics.get("final_equity", 0) - r.initial_capital
            for r in runs
        }

        return PortfolioResult(
            sleeves=runs,
            timestamps=ts,
            equity_curve=total,
            metrics=metrics,
            contribution=contribution,
        )

    def _metrics(self, equity: list[float], ts: list[datetime]) -> dict[str, float]:
        if not equity:
            return {"initial_balance": self.total_balance, "final_equity": self.total_balance}
        final = equity[-1]
        initial = equity[0]
        total_ret = (final - initial) / initial if initial > 0 else 0.0
        days = max((ts[-1] - ts[0]).total_seconds() / 86400.0, 1.0)
        annual = (1 + total_ret) ** (365.0 / days) - 1.0 if (1 + total_ret) > 0 else -1.0
        rets = []
        for i in range(1, len(equity)):
            if equity[i - 1] > 0:
                rets.append((equity[i] - equity[i - 1]) / equity[i - 1])
        sharpe = 0.0
        if rets:
            mean_r = sum(rets) / len(rets)
            var = sum((r - mean_r) ** 2 for r in rets) / max(len(rets) - 1, 1)
            std = var ** 0.5
            tf_per_year = {
                "1m": 525600, "5m": 105120, "15m": 35040,
                "1h": 8760, "4h": 2190, "1d": 365,
            }.get(self.bt_template.primary_tf, 8760)
            sharpe = (mean_r / std * (tf_per_year ** 0.5)) if std > 0 else 0.0
        peak = -float("inf")
        max_dd = 0.0
        for v in equity:
            peak = max(peak, v)
            if peak > 0:
                max_dd = max(max_dd, (peak - v) / peak)
        return {
            "initial_balance": initial,
            "final_equity": final,
            "total_return_pct": total_ret * 100,
            "annualized_return_pct": annual * 100,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd * 100,
            "aligned_bars": len(equity),
        }


def load_portfolio_yaml(path: str | Path) -> tuple[list[SleeveConfig], float]:
    """加载 portfolio.yaml → (sleeves, total_balance)。"""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = raw.get("portfolio", [])
    if not items:
        raise ValueError(f"{path} 缺 portfolio: 配置")
    sleeves = [
        SleeveConfig(
            strategy_path=Path(it["strategy"]).resolve()
            if Path(it["strategy"]).is_absolute()
            else (path.parent.parent / it["strategy"]).resolve(),
            symbol=str(it["symbol"]),
            allocation=float(it["allocation"]),
        )
        for it in items
    ]
    total_balance = float(raw.get("total_balance", 100.0))
    return sleeves, total_balance
