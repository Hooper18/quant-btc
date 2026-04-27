"""模拟交易执行器（Paper Trading）。

每根**主周期 K 线闭合**触发评估：合并新 bar → 重算指标 → RuleEngine 求值 →
按信号开/平仓 → 检查 SL/TP/强平 → 写入交易记录与状态快照 → Telegram 通知。

模拟规则与 Backtester 对齐：
- 同向加仓不允许（与 Backtester._execute_signal 对齐）
- 反向信号先平后开；SL/TP 用 high/low 触发，但 live 流仅有最近已闭合 bar 的 OHLC
- 资金费率每 8 小时（UTC 0/8/16）按固定 0.01% 结算（fapi 受限，无法拉实时费率）

状态持久化：
- data/paper_trades.json：交易明细（追加；启动不会清空）
- data/paper_state.json：余额 / 持仓 / 累计指标的快照（每 bar 覆写）
- 启动时若 paper_state.json 存在则恢复；否则用 initial_balance 起步
"""
from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from engine import RuleEngine
from indicators import IndicatorEngine

logger = logging.getLogger(__name__)

# 每 8 小时按历史均值结算资金费率（fapi REST 受限，不拉实时）
FUNDING_RATE_FALLBACK = 0.0001
FUNDING_HOURS_UTC = (0, 8, 16)
WARMUP_BARS = 200  # 内存中保留的每周期 bar 数（≥ MACD slow+signal=35 即可）


@dataclass
class PaperTrade:
    timestamp: str           # ISO8601 UTC
    side: str                # long_open / long_close / short_open / short_close / liquidate
    price: float
    size: float
    fee: float
    pnl: float = 0.0
    strategy: str = ""
    reason: str = ""         # signal / sl / tp / liq / funding / shutdown


@dataclass
class PaperPosition:
    side: str = "flat"               # long / short / flat
    size: float = 0.0                # BTC 张数（绝对值）
    entry_price: float = 0.0
    open_ts: str | None = None
    open_strategy: str = ""
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None


@dataclass
class PaperState:
    balance: float
    position: PaperPosition = field(default_factory=PaperPosition)
    realized_pnl_total: float = 0.0
    fee_paid_total: float = 0.0
    funding_paid_total: float = 0.0
    last_funding_ts: str | None = None
    last_bar_ts: str | None = None
    trade_count: int = 0


class PaperTrader:
    """模拟交易执行器：用 WebSocket K线驱动 RuleEngine 与持仓管理。"""

    def __init__(
        self,
        *,
        initial_balance: float,
        leverage: float,
        fee_rate: float,
        slippage: float,
        strategy_path: str | Path,
        primary_tf: str = "1h",
        used_tfs: tuple[str, ...] = ("1h", "4h"),
        ind_cfg: list[tuple[str, dict[str, Any]]] | None = None,
        maintenance_margin_rate: float = 0.004,
        trades_path: str | Path = "data/paper_trades.json",
        state_path: str | Path = "data/paper_state.json",
        notifier: Any = None,
        dry_run: bool = False,
    ):
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.maintenance_margin_rate = maintenance_margin_rate
        self.strategy_path = Path(strategy_path)
        self.primary_tf = primary_tf
        self.used_tfs = used_tfs
        self.ind_cfg = ind_cfg or []
        self.trades_path = Path(trades_path)
        self.state_path = Path(state_path)
        self.notifier = notifier
        self.dry_run = dry_run

        self.state = PaperState(balance=initial_balance)
        # 每周期一份 OHLCV 滚动 DataFrame；warmup 后由 on_bar 追加
        self._dfs: dict[str, pl.DataFrame] = {tf: _empty_ohlcv() for tf in used_tfs}

    # ---------- 启动期：warmup + 状态恢复 ----------
    def warmup_from_parquet(self, parquet_dir: str | Path) -> None:
        """从本地 parquet 目录加载最近 N 根 K 线作为指标 warmup。"""
        root = Path(parquet_dir)
        for tf in self.used_tfs:
            files = sorted(root.glob(f"{tf}_*.parquet"))[-3:]  # 取最近 3 个月
            if not files:
                logger.warning("warmup: 缺 %s parquet → 跳过", tf)
                continue
            parts = [pl.read_parquet(f) for f in files]
            df = (
                pl.concat(parts, how="diagonal_relaxed")
                .unique(subset=["timestamp"])
                .sort("timestamp")
                .tail(WARMUP_BARS)
            )
            self._dfs[tf] = df
            logger.info("warmup %s: 加载 %d 根 K 线（%s → %s）",
                        tf, df.height, df["timestamp"][0], df["timestamp"][-1])

    def restore_state(self) -> None:
        """从 state_path 恢复运行状态；不存在时静默跳过。"""
        if not self.state_path.exists():
            logger.info("无历史状态文件，按 initial_balance=%.2f 起步", self.initial_balance)
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            pos_raw = raw.get("position", {}) or {}
            self.state = PaperState(
                balance=float(raw["balance"]),
                position=PaperPosition(**pos_raw),
                realized_pnl_total=float(raw.get("realized_pnl_total", 0.0)),
                fee_paid_total=float(raw.get("fee_paid_total", 0.0)),
                funding_paid_total=float(raw.get("funding_paid_total", 0.0)),
                last_funding_ts=raw.get("last_funding_ts"),
                last_bar_ts=raw.get("last_bar_ts"),
                trade_count=int(raw.get("trade_count", 0)),
            )
            logger.info(
                "状态已恢复：余额=%.2f 持仓=%s 累计已实现=%.2f",
                self.state.balance, self.state.position.side, self.state.realized_pnl_total,
            )
        except Exception:
            logger.exception("状态恢复失败，按 initial_balance 重新起步")
            self.state = PaperState(balance=self.initial_balance)

    # ---------- 主入口：on bar closed ----------
    async def on_bar_closed(self, tf: str, kline: dict[str, Any]) -> None:
        """K 线闭合回调入口；DataFeed 每根 bar 调用一次，按 tf 分发处理。"""
        # 任何 tf 的 bar 都先追加到内部 DF；只有 primary_tf 触发评估
        self._append_bar(tf, kline)
        if tf != self.primary_tf:
            return
        ts = kline["close_time"]
        price = kline["close"]
        high = kline["high"]
        low = kline["low"]
        logger.info("[%s bar 闭合] %s close=%.2f", tf, ts, price)

        # 1) 资金费率结算
        funding_cash = self._apply_funding(ts, price)
        if funding_cash != 0:
            self.state.balance += funding_cash
            self.state.funding_paid_total += -funding_cash if funding_cash < 0 else 0
            logger.info("资金费率结算：%.4f USDT 余额=%.2f", funding_cash, self.state.balance)

        # 2) 检查 SL/TP（用 high/low）
        if not self.dry_run:
            sl_tp = self._check_stop_take(high, low, ts, price)
            if sl_tp:
                self._record_trade(sl_tp)
                self.state.balance += sl_tp.pnl
                await self._notify_trade(sl_tp)

        # 3) 强平
        if not self.dry_run:
            liq = self._check_liquidation(high, low, ts)
            if liq:
                self._record_trade(liq)
                self.state.balance += liq.pnl
                await self._notify_trade(liq)

        # 4) RuleEngine 求值
        signals = self._evaluate_signals()
        if signals:
            logger.info("产生信号 %d 条：%s", len(signals),
                        [(s.strategy_name, s.side) for s in signals])

        # 5) 执行信号
        if not self.dry_run:
            for sig in signals:
                if (
                    self.state.position.side != "flat"
                    and self.state.position.side == sig.side
                ):
                    continue  # 同向加仓跳过（与 Backtester 一致）
                trades = self._execute_signal(sig, price, ts)
                for t in trades:
                    self._record_trade(t)
                    self.state.balance += t.pnl
                    await self._notify_trade(t)
        else:
            for sig in signals:
                logger.info("[DRY-RUN] 信号：%s %s size=%.1f%% SL=%s TP=%s",
                            sig.strategy_name, sig.side, sig.size_pct,
                            sig.stop_loss_pct, sig.take_profit_pct)

        # 6) 更新最后处理 ts + 持久化状态
        self.state.last_bar_ts = ts.isoformat()
        if not self.dry_run:
            self._save_state()

    # ---------- DF 维护 ----------
    def _append_bar(self, tf: str, kline: dict[str, Any]) -> None:
        if tf not in self._dfs:
            return
        new_row = pl.DataFrame({
            "timestamp": [kline["open_time"]],
            "open": [kline["open"]],
            "high": [kline["high"]],
            "low": [kline["low"]],
            "close": [kline["close"]],
            "volume": [kline["volume"]],
            "taker_buy_volume": [kline["taker_buy_volume"]],
        }).with_columns(
            pl.col("timestamp").cast(pl.Datetime("ms")).dt.replace_time_zone("UTC"),
        )
        df = self._dfs[tf]
        # 同 timestamp 覆盖（避免 warmup 与 WS 首根重复）
        merged = (
            pl.concat([df, new_row], how="diagonal_relaxed")
            .unique(subset=["timestamp"], keep="last")
            .sort("timestamp")
            .tail(WARMUP_BARS)
        )
        self._dfs[tf] = merged

    def _evaluate_signals(self) -> list[Any]:
        """按 ind_cfg 重算指标并跑 RuleEngine 在最后一根 bar 上求值。"""
        data_dict: dict[str, pl.DataFrame] = {}
        for tf, df in self._dfs.items():
            if df.height < 35:  # MACD 26+9 起步
                logger.debug("%s 仅 %d 根 bar 不足 warmup", tf, df.height)
                return []
            data_dict[tf] = (
                IndicatorEngine(df).compute_all(self.ind_cfg) if self.ind_cfg else df
            )
        try:
            engine = RuleEngine(data_dict, primary_timeframe=self.primary_tf)
            engine.load_rules(self.strategy_path)
            primary = data_dict[self.primary_tf]
            return engine.evaluate(primary, primary.height - 1)
        except Exception:
            logger.exception("RuleEngine 求值失败")
            return []

    # ---------- 资金费率 ----------
    def _apply_funding(self, ts: datetime, price: float) -> float:
        """结算 (last_funding_ts, ts] 内跨过的 0/8/16 时刻；无持仓返回 0。"""
        if self.state.position.side == "flat":
            self.state.last_funding_ts = ts.isoformat()
            return 0.0
        last = self.state.last_funding_ts
        prev_ts = (
            datetime.fromisoformat(last) if last
            else ts.replace(minute=0, second=0, microsecond=0)
        )
        if prev_ts.tzinfo is None:
            prev_ts = prev_ts.replace(tzinfo=timezone.utc)
        from datetime import timedelta as _td
        cursor = prev_ts.replace(minute=0, second=0, microsecond=0) + _td(hours=1)
        crossed: list[datetime] = []
        while cursor <= ts:
            if cursor > prev_ts and cursor.hour in FUNDING_HOURS_UTC:
                crossed.append(cursor)
            cursor = cursor + _td(hours=1)
        self.state.last_funding_ts = ts.isoformat()
        if not crossed:
            return 0.0
        rate = FUNDING_RATE_FALLBACK
        notional = self.state.position.size * price
        sign = 1 if self.state.position.side == "long" else -1
        return -sign * notional * rate * len(crossed)

    # ---------- SL/TP/强平 ----------
    def _check_stop_take(
        self, high: float, low: float, ts: datetime, fallback_price: float,
    ) -> PaperTrade | None:
        pos = self.state.position
        if pos.side == "flat":
            return None
        ep = pos.entry_price
        if pos.side == "long":
            if pos.stop_loss_pct is not None and low <= ep * (1 - pos.stop_loss_pct / 100):
                return self._close_position(ep * (1 - pos.stop_loss_pct / 100), ts, "sl")
            if pos.take_profit_pct is not None and high >= ep * (1 + pos.take_profit_pct / 100):
                return self._close_position(ep * (1 + pos.take_profit_pct / 100), ts, "tp")
        else:
            if pos.stop_loss_pct is not None and high >= ep * (1 + pos.stop_loss_pct / 100):
                return self._close_position(ep * (1 + pos.stop_loss_pct / 100), ts, "sl")
            if pos.take_profit_pct is not None and low <= ep * (1 - pos.take_profit_pct / 100):
                return self._close_position(ep * (1 - pos.take_profit_pct / 100), ts, "tp")
        return None

    def _check_liquidation(self, high: float, low: float, ts: datetime) -> PaperTrade | None:
        pos = self.state.position
        if pos.side == "flat":
            return None
        ep = pos.entry_price
        margin = (pos.size * ep) / self.leverage
        worst = low if pos.side == "long" else high
        sign = 1 if pos.side == "long" else -1
        upnl = sign * pos.size * (worst - ep)
        notional_at_worst = pos.size * worst
        maint = notional_at_worst * self.maintenance_margin_rate
        if (margin + upnl) < maint:
            return self._close_position(worst, ts, "liq", side_label="liquidate")
        return None

    # ---------- 信号执行 ----------
    def _execute_signal(self, sig: Any, price: float, ts: datetime) -> list[PaperTrade]:
        out: list[PaperTrade] = []
        # 反向先平
        if self.state.position.side != "flat" and self.state.position.side != sig.side:
            t = self._close_position(price, ts, "signal_flip")
            if t:
                out.append(t)
                self.state.balance += t.pnl
        # 开新仓
        equity_now = self._equity(price)
        if equity_now <= 0:
            return out
        notional = equity_now * (sig.size_pct / 100.0) * self.leverage
        size_btc = notional / price
        if size_btc <= 0:
            return out
        fill = price * (1 + self.slippage) if sig.side == "long" else price * (1 - self.slippage)
        fee = size_btc * fill * self.fee_rate
        self.state.fee_paid_total += fee
        self.state.position = PaperPosition(
            side=sig.side,
            size=size_btc,
            entry_price=fill,
            open_ts=ts.isoformat(),
            open_strategy=sig.strategy_name,
            stop_loss_pct=sig.stop_loss_pct,
            take_profit_pct=sig.take_profit_pct,
        )
        out.append(PaperTrade(
            timestamp=ts.isoformat(),
            side=f"{sig.side}_open",
            price=fill,
            size=size_btc,
            fee=fee,
            pnl=-fee,  # 开仓只扣手续费
            strategy=sig.strategy_name,
            reason="signal",
        ))
        return out

    def _close_position(
        self, price: float, ts: datetime, reason: str, side_label: str | None = None,
    ) -> PaperTrade | None:
        pos = self.state.position
        if pos.side == "flat":
            return None
        fill = price * (1 - self.slippage) if pos.side == "long" else price * (1 + self.slippage)
        fee = pos.size * fill * self.fee_rate
        self.state.fee_paid_total += fee
        sign = 1 if pos.side == "long" else -1
        pnl = sign * pos.size * (fill - pos.entry_price) - fee
        self.state.realized_pnl_total += pnl
        side_str = side_label if side_label else f"{pos.side}_close"
        trade = PaperTrade(
            timestamp=ts.isoformat(),
            side=side_str,
            price=fill,
            size=pos.size,
            fee=fee,
            pnl=pnl,
            strategy=pos.open_strategy,
            reason=reason,
        )
        self.state.position = PaperPosition()
        return trade

    def _equity(self, mark_price: float) -> float:
        pos = self.state.position
        if pos.side == "flat":
            return self.state.balance
        sign = 1 if pos.side == "long" else -1
        upnl = sign * pos.size * (mark_price - pos.entry_price)
        return self.state.balance + upnl

    # ---------- 持久化与通知 ----------
    def _record_trade(self, trade: PaperTrade) -> None:
        self.state.trade_count += 1
        self.trades_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trades_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(trade), ensure_ascii=False) + "\n")

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "balance": self.state.balance,
            "position": asdict(self.state.position),
            "realized_pnl_total": self.state.realized_pnl_total,
            "fee_paid_total": self.state.fee_paid_total,
            "funding_paid_total": self.state.funding_paid_total,
            "last_funding_ts": self.state.last_funding_ts,
            "last_bar_ts": self.state.last_bar_ts,
            "trade_count": self.state.trade_count,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    async def _notify_trade(self, trade: PaperTrade) -> None:
        if not self.notifier:
            return
        emoji = {
            "long_open": "📈", "short_open": "📉",
            "long_close": "✅", "short_close": "✅",
            "liquidate": "⚠️",
        }.get(trade.side, "•")
        msg = (
            f"{emoji} *{trade.side}* `{trade.strategy or '-'}`\n"
            f"价格 `{trade.price:.2f}` 数量 `{trade.size:.4f}`\n"
            f"PnL `{trade.pnl:+.2f}` 手续费 `{trade.fee:.2f}`\n"
            f"原因 `{trade.reason}` 余额 `{self.state.balance:.2f}`"
        )
        await self.notifier.send(msg)

    async def shutdown(self, last_price: float | None = None) -> None:
        """优雅退出：若有持仓按 last_price 平仓（可选），保存状态。"""
        if self.state.position.side != "flat" and last_price is not None:
            ts = datetime.now(timezone.utc)
            t = self._close_position(last_price, ts, "shutdown")
            if t:
                self._record_trade(t)
                self.state.balance += t.pnl
        self._save_state()
        logger.info("Paper 退出：余额=%.2f trade_count=%d", self.state.balance, self.state.trade_count)


def _empty_ohlcv() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "timestamp": pl.Datetime("ms", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "taker_buy_volume": pl.Float64,
        }
    )


def parse_strategy_indicators(strategy_path: str | Path) -> tuple[list[tuple[str, dict]], set[str], str]:
    """从策略 YAML 解析所需指标 + 涉及周期。"""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from run_backtest import _collect_required_indicators  # type: ignore

    with open(strategy_path, "r", encoding="utf-8") as f:
        strat_yaml = yaml.safe_load(f) or {}
    strategies = strat_yaml.get("strategies", [])
    ind_cfg = _collect_required_indicators(strategies)
    used_tfs: set[str] = set()
    primary = "1h"
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
    used_tfs.add(primary)
    return ind_cfg, used_tfs, primary
