"""向量化逐 K 线回测引擎。

支持：杠杆永续合约、多/空、加仓、止损/止盈、强平、资金费率结算、
最大回撤熔断、日最大亏损、滑点 + 双边手续费。

主循环按 primary_timeframe 推进；其它 TF 数据由 RuleEngine 自行对齐。
所有金额以 USDT 计价，仓位 size 以"BTC 张数"表示（正=多，负=空）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from engine import RuleEngine, Signal

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    timestamp: datetime
    side: str          # "long_open" / "long_close" / "short_open" / "short_close" / "liquidate"
    price: float
    size: float        # BTC 张数（绝对值）
    fee: float
    pnl: float = 0.0   # 平仓时的已实现盈亏
    strategy: str = ""

    def as_row(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "side": self.side,
            "price": self.price,
            "size": self.size,
            "fee": self.fee,
            "pnl": self.pnl,
            "strategy": self.strategy,
        }


@dataclass
class BacktestResult:
    equity_curve: list[float]
    timestamps: list[datetime]
    trades: list[Trade]
    metrics: dict[str, float]

    def print_summary(self) -> None:
        print("\n========== 回测结果摘要 ==========")
        print(f"初始资金:      {self.metrics.get('initial_balance', 0):.2f} USDT")
        print(f"最终资产:      {self.metrics.get('final_equity', 0):.2f} USDT")
        print(f"总收益率:      {self.metrics.get('total_return_pct', 0):.2f}%")
        print(f"年化收益率:    {self.metrics.get('annualized_return_pct', 0):.2f}%")
        print(f"夏普比率:      {self.metrics.get('sharpe_ratio', 0):.3f}")
        print(f"最大回撤:      {self.metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"总交易笔数:    {self.metrics.get('total_trades', 0):.0f}")
        print(f"胜率:          {self.metrics.get('win_rate_pct', 0):.2f}%")
        print(f"盈亏比:        {self.metrics.get('profit_loss_ratio', 0):.3f}")
        print(f"平均持仓:      {self.metrics.get('avg_holding_hours', 0):.2f} 小时")
        print(f"是否熔断:      {self.metrics.get('circuit_breaker', False)}")
        print("==================================")

    def to_csv(self, path: str | Path) -> None:
        if not self.trades:
            logger.warning("无交易记录可导出")
            return
        df = pl.DataFrame([t.as_row() for t in self.trades])
        df.write_csv(str(path))
        logger.info("交易记录写入 %s（%d 条）", path, df.height)


@dataclass
class _Position:
    """单一持仓（永续合约）。size 为正多/负空；entry_price 为均价。"""
    size: float = 0.0           # BTC 张数（带方向）
    entry_price: float = 0.0    # 加仓后的均价
    open_ts: datetime | None = None
    open_strategy: str = ""
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None

    @property
    def side(self) -> str:
        if self.size > 0:
            return "long"
        if self.size < 0:
            return "short"
        return "flat"

    @property
    def notional(self) -> float:
        """名义价值（绝对值），用最新价计算请用 unrealized_pnl 配套方法。"""
        return abs(self.size) * self.entry_price

    def margin(self, leverage: float) -> float:
        return self.notional / leverage

    def unrealized_pnl(self, mark_price: float) -> float:
        return self.size * (mark_price - self.entry_price)


class _PositionManager:
    """单一标的仓位管理器。同向加仓更新均价；反向先全平再开新仓。"""

    def __init__(self, leverage: float, fee_rate: float, slippage: float):
        self.leverage = leverage
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.position = _Position()
        self.realized_pnl_total = 0.0
        self.fee_paid_total = 0.0

    # ---------- 内部成交价 ----------
    def _fill_buy(self, price: float) -> float:
        return price * (1.0 + self.slippage)

    def _fill_sell(self, price: float) -> float:
        return price * (1.0 - self.slippage)

    # ---------- 操作 ----------
    def open(
        self,
        side: str,
        size_btc: float,
        price: float,
        ts: datetime,
        strategy: str,
        sl_pct: float | None,
        tp_pct: float | None,
    ) -> Trade:
        """开新仓或同向加仓；反向开仓需调用方先 close_all。"""
        assert size_btc > 0
        if side == "long":
            fill = self._fill_buy(price)
            delta = +size_btc
        else:
            fill = self._fill_sell(price)
            delta = -size_btc

        fee = abs(delta) * fill * self.fee_rate
        self.fee_paid_total += fee

        if self.position.size == 0:
            self.position = _Position(
                size=delta,
                entry_price=fill,
                open_ts=ts,
                open_strategy=strategy,
                stop_loss_pct=sl_pct,
                take_profit_pct=tp_pct,
            )
        else:
            # 同向加仓：均价 = (旧名义 + 新名义) / 总张数
            assert (self.position.size > 0) == (delta > 0), "反向开仓需先 close_all"
            new_total = self.position.size + delta
            new_avg = (self.position.size * self.position.entry_price + delta * fill) / new_total
            self.position.size = new_total
            self.position.entry_price = new_avg
            # 加仓不重置止损止盈、ts、strategy

        return Trade(
            timestamp=ts,
            side=f"{side}_open",
            price=fill,
            size=size_btc,
            fee=fee,
            strategy=strategy,
        )

    def close_all(self, price: float, ts: datetime, reason: str = "close") -> Trade | None:
        if self.position.size == 0:
            return None
        side_label = self.position.side
        if side_label == "long":
            fill = self._fill_sell(price)
        else:
            fill = self._fill_buy(price)
        size_abs = abs(self.position.size)
        fee = size_abs * fill * self.fee_rate
        self.fee_paid_total += fee
        pnl = self.position.size * (fill - self.position.entry_price) - fee
        self.realized_pnl_total += pnl

        trade = Trade(
            timestamp=ts,
            side=(f"{side_label}_close" if reason == "close" else reason),
            price=fill,
            size=size_abs,
            fee=fee,
            pnl=pnl,
            strategy=self.position.open_strategy,
        )
        self.position = _Position()
        return trade

    # ---------- 估值 ----------
    def equity(self, balance: float, mark_price: float) -> float:
        return balance + self.position.unrealized_pnl(mark_price)


class Backtester:
    """杠杆永续合约回测器。

    构造参数从 YAML 读取（推荐用 from_yaml）。run 接收已带指标的 data_dict。
    """

    def __init__(
        self,
        *,
        initial_balance: float,
        leverage: float,
        fee_rate: float,
        slippage: float,
        maintenance_margin_rate: float,
        max_drawdown_pct: float,
        funding_rate_epochs_utc: list[int],
        primary_timeframe: str,
        daily_max_loss_pct: float | None = None,
    ):
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.maintenance_margin_rate = maintenance_margin_rate
        self.max_drawdown_pct = max_drawdown_pct
        self.funding_epochs = set(funding_rate_epochs_utc)
        self.primary_tf = primary_timeframe
        self.daily_max_loss_pct = daily_max_loss_pct

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Backtester":
        with Path(path).open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cls(
            initial_balance=float(cfg["initial_balance"]),
            leverage=float(cfg["leverage"]),
            fee_rate=float(cfg["fee_rate"]),
            slippage=float(cfg["slippage"]),
            maintenance_margin_rate=float(cfg["maintenance_margin_rate"]),
            max_drawdown_pct=float(cfg["max_drawdown_pct"]),
            funding_rate_epochs_utc=list(cfg.get("funding_rate_epochs_utc", [0, 8, 16])),
            primary_timeframe=str(cfg["primary_timeframe"]),
            daily_max_loss_pct=cfg.get("daily_max_loss_pct"),
        )

    # ---------- 主循环 ----------
    def run(
        self,
        data_dict: dict[str, pl.DataFrame],
        rules_path: str | Path,
        funding_rate_df: pl.DataFrame | None = None,
    ) -> BacktestResult:
        if self.primary_tf not in data_dict:
            raise ValueError(f"primary_timeframe={self.primary_tf} 不在 data_dict 中")

        engine = RuleEngine(data_dict, primary_timeframe=self.primary_tf)
        engine.load_rules(rules_path)

        primary = data_dict[self.primary_tf].sort("timestamp")
        n = primary.height
        ts_col = primary["timestamp"].to_list()
        open_col = primary["open"].to_list()
        high_col = primary["high"].to_list()
        low_col = primary["low"].to_list()
        close_col = primary["close"].to_list()

        # 资金费率索引：按 timestamp 排序后，按 search_sorted 取前向值
        fr_ts: list[datetime] = []
        fr_val: list[float] = []
        if funding_rate_df is not None and funding_rate_df.height > 0:
            fr_sorted = funding_rate_df.sort("timestamp")
            fr_ts = fr_sorted["timestamp"].to_list()
            fr_val = fr_sorted["funding_rate"].to_list()

        pm = _PositionManager(self.leverage, self.fee_rate, self.slippage)
        balance = self.initial_balance
        peak_equity = balance
        equity_curve: list[float] = []
        trades: list[Trade] = []
        circuit_breaker = False
        last_funding_check_ts: datetime | None = None
        day_start_equity: float = balance
        current_day: date | None = None

        for i in range(n):
            ts = ts_col[i]
            o, h, l, c = open_col[i], high_col[i], low_col[i], close_col[i]

            # 资金费率结算（每根 K 线检查上次检查点到当前 ts 之间跨过的 epoch）
            # last_funding_check_ts 必须每 bar 更新，不能只在持仓时更新——否则开仓后第
            # 一次结算会丢失中间的 epoch（开仓 ts → 当前 ts 之间被错过）
            if self.funding_epochs:
                if pm.position.size != 0 and last_funding_check_ts is not None:
                    fr = self._apply_funding(
                        pm, last_funding_check_ts, ts, fr_ts, fr_val,
                    )
                    if fr != 0:
                        balance += fr
                last_funding_check_ts = ts

            # 日度风控基线
            day = ts.date() if isinstance(ts, datetime) else date.fromisoformat(str(ts)[:10])
            if current_day != day:
                current_day = day
                day_start_equity = pm.equity(balance, c)

            # 已熔断 → 只更新净值不交易
            if circuit_breaker:
                equity_curve.append(pm.equity(balance, c))
                continue

            # 1) 检查止损/止盈（用 high/low，方向相关）
            sl_tp_trade = self._check_stop_take(pm, h, l, ts)
            if sl_tp_trade:
                trades.append(sl_tp_trade)
                balance += sl_tp_trade.pnl

            # 2) 检查强平
            liq_trade = self._check_liquidation(pm, h, l, ts, balance)
            if liq_trade:
                trades.append(liq_trade)
                balance += liq_trade.pnl

            # 3) 评估规则信号
            try:
                signals = engine.evaluate(primary, i)
            except Exception:
                logger.exception("规则评估异常 @bar=%d ts=%s", i, ts)
                signals = []

            # 4) 执行信号
            # 已持有同向仓位 → 跳过（避免持续状态条件每 bar 重复加仓导致仓位线性放大）
            # 反向信号仍允许，由 _execute_signal 内部先平后开
            for sig in signals:
                if pm.position.size != 0 and pm.position.side == sig.side:
                    continue
                balance += self._execute_signal(pm, sig, c, ts, balance, trades)

            # 5) 净值推进 + 风控熔断
            cur_equity = pm.equity(balance, c)
            equity_curve.append(cur_equity)
            peak_equity = max(peak_equity, cur_equity)
            dd = 0.0 if peak_equity <= 0 else (peak_equity - cur_equity) / peak_equity
            if dd > self.max_drawdown_pct:
                logger.warning("回撤 %.2f%% > 阈值 → 熔断平仓 @%s", dd * 100, ts)
                t = pm.close_all(c, ts, reason="liquidate")
                if t:
                    trades.append(t)
                    balance += t.pnl
                circuit_breaker = True

            # 日内最大亏损
            if self.daily_max_loss_pct is not None and day_start_equity > 0:
                day_loss_pct = (day_start_equity - cur_equity) / day_start_equity
                if day_loss_pct > self.daily_max_loss_pct:
                    logger.warning("日内亏损 %.2f%% > 阈值 → 当日平仓", day_loss_pct * 100)
                    t = pm.close_all(c, ts, reason="liquidate")
                    if t:
                        trades.append(t)
                        balance += t.pnl

        # 结尾若仍有持仓，按最后一根 close 平掉以便核算
        if pm.position.size != 0:
            t = pm.close_all(close_col[-1], ts_col[-1], reason="close")
            if t:
                trades.append(t)
                balance += t.pnl
                equity_curve[-1] = balance

        metrics = self._compute_metrics(equity_curve, ts_col, trades, circuit_breaker)
        return BacktestResult(
            equity_curve=equity_curve,
            timestamps=ts_col,
            trades=trades,
            metrics=metrics,
        )

    # ---------- 风控/资金费率 ----------
    def _apply_funding(
        self,
        pm: _PositionManager,
        prev_ts: datetime,
        cur_ts: datetime,
        fr_ts: list[datetime],
        fr_val: list[float],
    ) -> float:
        """检查 (prev_ts, cur_ts] 内是否跨过资金费率 epoch；若跨过则结算。

        永续合约规则：多头付资金费率 × 名义、空头收（费率为正时）；正负相反则反过来。
        现金流 = -size_signed * notional_at_mark * funding_rate。
        近似：用最近一次资金费率值；若无 funding_rate_df 则按 0 处理。
        """
        if pm.position.size == 0 or not fr_ts:
            return 0.0
        from datetime import timedelta as _td
        # 找出 (prev_ts, cur_ts] 内是否包含 epoch 时刻；按 1h 步进扫描
        crossed: list[datetime] = []
        cursor = prev_ts.replace(minute=0, second=0, microsecond=0) + _td(hours=1)
        while cursor <= cur_ts:
            if cursor > prev_ts and cursor.hour in self.funding_epochs:
                crossed.append(cursor)
            cursor = cursor + _td(hours=1)
        if not crossed:
            return 0.0
        total_cash = 0.0
        for epoch_ts in crossed:
            # 查找 fr_ts 中 <= epoch_ts 的最大者
            idx = self._search_sorted(fr_ts, epoch_ts)
            if idx < 0:
                continue
            rate = fr_val[idx]
            notional = abs(pm.position.size) * pm.position.entry_price
            cash = -1.0 * (1 if pm.position.size > 0 else -1) * notional * rate
            total_cash += cash
        return total_cash

    @staticmethod
    def _search_sorted(arr: list[datetime], target: datetime) -> int:
        lo, hi = 0, len(arr)
        while lo < hi:
            mid = (lo + hi) // 2
            if arr[mid] <= target:
                lo = mid + 1
            else:
                hi = mid
        return lo - 1

    def _check_stop_take(
        self, pm: _PositionManager, high: float, low: float, ts: datetime,
    ) -> Trade | None:
        if pm.position.size == 0:
            return None
        ep = pm.position.entry_price
        sl = pm.position.stop_loss_pct
        tp = pm.position.take_profit_pct
        if pm.position.side == "long":
            if sl is not None and low <= ep * (1 - sl / 100):
                return pm.close_all(ep * (1 - sl / 100), ts, reason="long_close")
            if tp is not None and high >= ep * (1 + tp / 100):
                return pm.close_all(ep * (1 + tp / 100), ts, reason="long_close")
        else:
            if sl is not None and high >= ep * (1 + sl / 100):
                return pm.close_all(ep * (1 + sl / 100), ts, reason="short_close")
            if tp is not None and low <= ep * (1 - tp / 100):
                return pm.close_all(ep * (1 - tp / 100), ts, reason="short_close")
        return None

    def _check_liquidation(
        self, pm: _PositionManager, high: float, low: float, ts: datetime, balance: float,
    ) -> Trade | None:
        """简化强平：当浮亏 + 维持保证金 > 已用保证金 时强平。"""
        if pm.position.size == 0:
            return None
        ep = pm.position.entry_price
        margin = pm.position.margin(self.leverage)
        # 反向最坏价：多头查 low、空头查 high
        worst = low if pm.position.side == "long" else high
        upnl = pm.position.size * (worst - ep)
        notional_at_worst = abs(pm.position.size) * worst
        maint = notional_at_worst * self.maintenance_margin_rate
        if (margin + upnl) < maint:
            return pm.close_all(worst, ts, reason="liquidate")
        return None

    def _execute_signal(
        self,
        pm: _PositionManager,
        sig: Signal,
        price: float,
        ts: datetime,
        balance: float,
        trades: list[Trade],
    ) -> float:
        """执行信号；返回此次调用产生的已实现 pnl 增量（供外层加到 balance）。"""
        delta_balance = 0.0
        # 反向信号：先平后开
        if pm.position.size != 0 and pm.position.side != sig.side:
            t = pm.close_all(price, ts, reason=f"{pm.position.side}_close")
            if t:
                trades.append(t)
                delta_balance += t.pnl
                balance += t.pnl  # 用于下面 equity 计算

        # 仓位金额：按当前权益 × size_pct × leverage 给出名义价值
        equity_now = pm.equity(balance, price)
        if equity_now <= 0:
            return delta_balance
        notional = equity_now * (sig.size_pct / 100.0) * self.leverage
        size_btc = notional / price
        if size_btc <= 0:
            return delta_balance
        t = pm.open(
            side=sig.side, size_btc=size_btc, price=price, ts=ts,
            strategy=sig.strategy_name, sl_pct=sig.stop_loss_pct, tp_pct=sig.take_profit_pct,
        )
        trades.append(t)
        return delta_balance

    # ---------- 指标计算 ----------
    def _compute_metrics(
        self,
        equity_curve: list[float],
        ts_col: list[datetime],
        trades: list[Trade],
        circuit_breaker: bool,
    ) -> dict[str, float]:
        if not equity_curve:
            return {"initial_balance": self.initial_balance, "final_equity": self.initial_balance}
        final_eq = equity_curve[-1]
        total_ret = (final_eq - self.initial_balance) / self.initial_balance
        # 年化：基于实际跨越的天数
        if len(ts_col) >= 2:
            start, end = ts_col[0], ts_col[-1]
            days = max((end - start).total_seconds() / 86400.0, 1.0)
        else:
            days = 1.0
        annual = (1 + total_ret) ** (365.0 / days) - 1.0 if (1 + total_ret) > 0 else -1.0

        # 夏普：用 equity_curve 的对数收益
        rets: list[float] = []
        for j in range(1, len(equity_curve)):
            prev, cur = equity_curve[j - 1], equity_curve[j]
            if prev > 0 and cur > 0:
                rets.append((cur - prev) / prev)
        if rets:
            mean_r = sum(rets) / len(rets)
            var = sum((r - mean_r) ** 2 for r in rets) / max(len(rets) - 1, 1)
            std = var ** 0.5
            # 主 TF → 年化倍数
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
        for v in equity_curve:
            peak = max(peak, v)
            if peak > 0:
                max_dd = max(max_dd, (peak - v) / peak)

        # 胜率/盈亏比/平均持仓时间
        closed = [t for t in trades if t.side.endswith("_close") or t.side == "liquidate"]
        wins = [t.pnl for t in closed if t.pnl > 0]
        losses = [t.pnl for t in closed if t.pnl < 0]
        win_rate = (len(wins) / len(closed)) if closed else 0.0
        avg_win = (sum(wins) / len(wins)) if wins else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
        plr = (avg_win / avg_loss) if avg_loss > 0 else 0.0

        # 平均持仓：配对 open→close
        open_stack: list[Trade] = []
        durations_h: list[float] = []
        for t in trades:
            if t.side.endswith("_open"):
                open_stack.append(t)
            elif (t.side.endswith("_close") or t.side == "liquidate") and open_stack:
                opener = open_stack.pop(0)
                durations_h.append((t.timestamp - opener.timestamp).total_seconds() / 3600.0)
        avg_hold = sum(durations_h) / len(durations_h) if durations_h else 0.0

        return {
            "initial_balance": self.initial_balance,
            "final_equity": final_eq,
            "total_return_pct": total_ret * 100,
            "annualized_return_pct": annual * 100,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd * 100,
            "total_trades": len(closed),
            "win_rate_pct": win_rate * 100,
            "profit_loss_ratio": plr,
            "avg_holding_hours": avg_hold,
            "circuit_breaker": int(circuit_breaker),
        }
