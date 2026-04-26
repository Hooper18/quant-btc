"""蒙特卡洛风险模拟。

思想：策略的样本期表现是众多可能历史中的一条。把已实现的逐笔 PnL 视为可重排的样本，
有放回 bootstrap 采样后重新累加成净值曲线，跑 N 次得到结果分布——回答：
- 收益率 95% 置信区间是多少？
- 最大回撤 95% 置信区间是多少？
- 多大概率破产（NAV 跌破初始资金的 20%）？

注意：此方法假设各笔交易 PnL 独立同分布，忽略时间序列相关性 / 顺序效应。是
"风险下限"估计而非完整模拟。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .backtester import BacktestResult, Trade

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    n_simulations: int
    initial_balance: float
    ruin_threshold: float           # NAV 跌破此值视为破产
    ruin_probability: float          # 破产概率（占总模拟数比例）
    final_returns_pct: np.ndarray    # shape=(n_simulations,) 最终收益率%
    max_drawdowns_pct: np.ndarray    # shape=(n_simulations,) 最大回撤%
    equity_curves: np.ndarray        # shape=(n_simulations, n_trades+1)

    def percentiles(self, arr: np.ndarray, qs: Iterable[float] = (5, 25, 50, 75, 95)) -> dict[int, float]:
        return {int(q): float(np.percentile(arr, q)) for q in qs}


class MonteCarloSimulator:
    """对历史已平仓交易做 bootstrap 重排，构造可能的净值曲线分布。"""

    RUIN_FRACTION_DEFAULT = 0.20  # NAV < 20% × 初始资金 视为破产

    def __init__(self, result: BacktestResult, initial_balance: float | None = None):
        # 提取所有已平仓交易的 PnL（含 0）
        closed = [
            t for t in result.trades
            if t.side.endswith("_close") or t.side == "liquidate"
        ]
        self.trade_pnls = np.array([t.pnl for t in closed], dtype=float)
        self.n_trades = len(self.trade_pnls)
        self.initial_balance = float(
            initial_balance if initial_balance is not None
            else result.metrics.get("initial_balance", 100.0)
        )
        if self.n_trades == 0:
            raise ValueError("BacktestResult 没有已平仓交易，无法 Monte Carlo")

    def run(
        self,
        n_simulations: int = 1000,
        ruin_fraction: float = RUIN_FRACTION_DEFAULT,
        seed: int = 42,
    ) -> MonteCarloResult:
        rng = np.random.default_rng(seed)
        ruin_threshold = self.initial_balance * ruin_fraction

        # 一次性生成所有采样：shape=(n_sim, n_trades)
        # rng.choice 有放回抽样
        sampled = rng.choice(self.trade_pnls, size=(n_simulations, self.n_trades), replace=True)
        # 累加得到净值（前置插入起始资金）
        cumsum = np.cumsum(sampled, axis=1)
        equity = self.initial_balance + cumsum
        # 在每条曲线前置 initial_balance（让首点为起始）
        equity = np.concatenate(
            [np.full((n_simulations, 1), self.initial_balance), equity],
            axis=1,
        )
        # 最终收益率 %
        final_pct = (equity[:, -1] - self.initial_balance) / self.initial_balance * 100.0
        # 每条曲线的运行最大值 + 回撤
        running_peak = np.maximum.accumulate(equity, axis=1)
        # 防止 peak <= 0 时除零（理论上不会发生，因 initial>0）
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = np.where(running_peak > 0, (running_peak - equity) / running_peak, 0.0)
        max_dd_pct = dd.max(axis=1) * 100.0
        # 破产：曲线最低点 < 阈值
        min_eq = equity.min(axis=1)
        ruined = int((min_eq < ruin_threshold).sum())

        return MonteCarloResult(
            n_simulations=n_simulations,
            initial_balance=self.initial_balance,
            ruin_threshold=ruin_threshold,
            ruin_probability=ruined / n_simulations,
            final_returns_pct=final_pct,
            max_drawdowns_pct=max_dd_pct,
            equity_curves=equity,
        )
