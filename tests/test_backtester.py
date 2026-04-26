"""Backtester / _PositionManager 单元测试。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from backtest import Backtester, BacktestResult
from backtest.backtester import _PositionManager


# ---------- _PositionManager: 开/平/盈亏 ----------
def test_long_open_close_profit_calculation() -> None:
    """多头：30000 入场 → 31000 平仓，10x 杠杆，0.05% 手续费 + 0.05% 滑点。"""
    pm = _PositionManager(leverage=10, fee_rate=0.0005, slippage=0.0005)
    ts1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ts2 = ts1 + timedelta(hours=1)

    t_open = pm.open("long", size_btc=1.0, price=30000.0, ts=ts1,
                     strategy="t", sl_pct=None, tp_pct=None)
    expected_buy_fill = 30000.0 * 1.0005   # 30015
    assert t_open.price == pytest.approx(expected_buy_fill)
    expected_open_fee = 1.0 * expected_buy_fill * 0.0005
    assert t_open.fee == pytest.approx(expected_open_fee)
    assert pm.position.size == pytest.approx(1.0)
    assert pm.position.entry_price == pytest.approx(expected_buy_fill)

    t_close = pm.close_all(31000.0, ts2)
    expected_sell_fill = 31000.0 * 0.9995   # 30984.5
    assert t_close.price == pytest.approx(expected_sell_fill)
    expected_close_fee = 1.0 * expected_sell_fill * 0.0005
    assert t_close.fee == pytest.approx(expected_close_fee)
    expected_pnl = 1.0 * (expected_sell_fill - expected_buy_fill) - expected_close_fee
    assert t_close.pnl == pytest.approx(expected_pnl)
    # 平仓后无持仓
    assert pm.position.size == 0


def test_short_open_close_profit_when_price_drops() -> None:
    """空头：30000 入场 → 29000 平仓，价格下跌应盈利。"""
    pm = _PositionManager(leverage=10, fee_rate=0.0005, slippage=0.0005)
    ts1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ts2 = ts1 + timedelta(hours=1)

    t_open = pm.open("short", size_btc=1.0, price=30000.0, ts=ts1,
                     strategy="t", sl_pct=None, tp_pct=None)
    expected_sell_fill = 30000.0 * 0.9995   # 29985
    assert t_open.price == pytest.approx(expected_sell_fill)
    assert pm.position.size == pytest.approx(-1.0)

    t_close = pm.close_all(29000.0, ts2)
    expected_buy_fill = 29000.0 * 1.0005    # 29014.5
    expected_close_fee = 1.0 * expected_buy_fill * 0.0005
    expected_pnl = -1.0 * (expected_buy_fill - expected_sell_fill) - expected_close_fee
    assert t_close.pnl == pytest.approx(expected_pnl)
    assert t_close.pnl > 0  # 价格下跌空头应盈利


def test_avg_entry_price_on_same_side_add() -> None:
    """同向加仓应更新均价。"""
    pm = _PositionManager(leverage=10, fee_rate=0.0, slippage=0.0)
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    pm.open("long", 1.0, 30000.0, ts, "t", None, None)
    pm.open("long", 1.0, 32000.0, ts, "t", None, None)
    # 均价 = (30000 + 32000) / 2 = 31000
    assert pm.position.entry_price == pytest.approx(31000.0)
    assert pm.position.size == pytest.approx(2.0)


# ---------- 完整回测：止损 / 止盈 / 强平 / 资金费率 ----------
def _build_data(prices: list[tuple[float, float, float, float]],
                start: datetime | None = None) -> pl.DataFrame:
    """prices: list of (open, high, low, close)；间隔 1h。"""
    start = start or datetime(2024, 1, 1, tzinfo=timezone.utc)
    n = len(prices)
    return pl.DataFrame({
        "timestamp": [start + timedelta(hours=i) for i in range(n)],
        "open": [p[0] for p in prices],
        "high": [p[1] for p in prices],
        "low": [p[2] for p in prices],
        "close": [p[3] for p in prices],
        "volume": [1.0] * n,
        # 触发信号用的列（必有一根 = trigger，之后置 0 防止重复）
        "trigger": [1.0] + [0.0] * (n - 1),
    })


def _write_strategy(tmp_path: Path, side: str, sl: float, tp: float) -> Path:
    p = tmp_path / "strat.yaml"
    p.write_text(f"""
strategies:
  - name: t
    conditions:
      - {{indicator: trigger, operator: ">", value: 0.5, timeframe: "1h"}}
    logic: AND
    action: {{side: {side}, type: market, size_pct: 10}}
    stop_loss_pct: {sl}
    take_profit_pct: {tp}
""", encoding="utf-8")
    return p


def _bt() -> Backtester:
    return Backtester(
        initial_balance=1000.0,
        leverage=10.0,
        fee_rate=0.0,
        slippage=0.0,
        maintenance_margin_rate=0.004,
        max_drawdown_pct=0.99,    # 不熔断
        funding_rate_epochs_utc=[],
        primary_timeframe="1h",
        daily_max_loss_pct=None,
    )


def test_stop_loss_long_triggers(tmp_path: Path) -> None:
    """多头入场后，第二根 K 线最低价跌破 sl 阈值 → 触发止损。"""
    # bar 0: open 100, close 100 — 触发开仓（开仓价 close=100）
    # bar 1: low 96 < 100*(1-3%)=97 → 触发 sl 平仓 @97
    data = _build_data([(100, 100, 100, 100), (100, 100, 96, 100)])
    bt = _bt()
    res = bt.run({"1h": data}, _write_strategy(tmp_path, "long", sl=3, tp=10))
    closes = [t for t in res.trades if t.side.endswith("_close")]
    assert len(closes) == 1
    # 止损价 = entry * (1 - 3%) = 97
    assert closes[0].price == pytest.approx(97.0)
    assert closes[0].pnl < 0


def test_take_profit_long_triggers(tmp_path: Path) -> None:
    """多头入场后，第二根 K 线最高价突破 tp 阈值 → 触发止盈。"""
    data = _build_data([(100, 100, 100, 100), (100, 110, 100, 100)])
    bt = _bt()
    res = bt.run({"1h": data}, _write_strategy(tmp_path, "long", sl=99, tp=5))
    closes = [t for t in res.trades if t.side.endswith("_close")]
    assert len(closes) == 1
    assert closes[0].price == pytest.approx(105.0)  # entry * 1.05
    assert closes[0].pnl > 0


def test_stop_loss_short_triggers(tmp_path: Path) -> None:
    """空头入场后，第二根 K 线最高价突破 sl 阈值 → 触发止损。"""
    data = _build_data([(100, 100, 100, 100), (100, 104, 100, 100)])
    bt = _bt()
    res = bt.run({"1h": data}, _write_strategy(tmp_path, "short", sl=3, tp=10))
    closes = [t for t in res.trades if t.side.endswith("_close")]
    assert len(closes) == 1
    assert closes[0].price == pytest.approx(103.0)


def test_liquidation_triggers_on_extreme_loss(tmp_path: Path) -> None:
    """杠杆 10x 多头，价格跌 ~10% 触发强平。"""
    # bar 0: 入场价 100；保证金 = notional/10 = 10
    # bar 1: low 89 → upnl = 1*(89-100) = -11；margin+upnl = -1 < maint = 89*0.004=0.356 → 强平
    data = _build_data([(100, 100, 100, 100), (100, 100, 89, 100)])
    bt = _bt()
    res = bt.run({"1h": data}, _write_strategy(tmp_path, "long", sl=99, tp=99))
    liqs = [t for t in res.trades if t.side == "liquidate"]
    assert len(liqs) == 1
    assert liqs[0].price == pytest.approx(89.0)  # 强平按 worst 价


def test_fee_and_slippage_deducted(tmp_path: Path) -> None:
    """开有手续费 + 滑点：开仓价应 > 真实价；平仓价 < 真实价（多头）。"""
    bt = Backtester(
        initial_balance=1000.0, leverage=10.0,
        fee_rate=0.001,  # 0.1%
        slippage=0.002,  # 0.2%
        maintenance_margin_rate=0.004, max_drawdown_pct=0.99,
        funding_rate_epochs_utc=[], primary_timeframe="1h",
    )
    data = _build_data([(100, 100, 100, 100), (100, 105, 99, 100)])
    res = bt.run({"1h": data}, _write_strategy(tmp_path, "long", sl=99, tp=4))
    opens = [t for t in res.trades if t.side.endswith("_open")]
    closes = [t for t in res.trades if t.side.endswith("_close")]
    assert len(opens) == 1 and len(closes) == 1
    # 开仓价 > 100（买入加滑点）
    assert opens[0].price > 100.0
    # 手续费 > 0
    assert opens[0].fee > 0
    assert closes[0].fee > 0


def test_funding_rate_paid_by_long(tmp_path: Path) -> None:
    """正费率下，多头应付资金费 → 平仓 pnl 比无费率版小。"""
    # bar 0: 00:00 入场；bar 1: 09:00（跨过 08:00 epoch 一次）→ 有 funding 结算
    start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    data = pl.DataFrame({
        "timestamp": [start, start + timedelta(hours=9)],
        "open": [100.0, 100.0], "high": [100.0, 100.0],
        "low": [100.0, 100.0], "close": [100.0, 100.0],
        "volume": [1.0, 1.0], "trigger": [1.0, 0.0],
    })
    fr_df = pl.DataFrame({
        "timestamp": [start + timedelta(hours=8)],   # 08:00 UTC epoch
        "funding_rate": [0.001],                      # 0.1%
    })
    bt = Backtester(
        initial_balance=1000.0, leverage=10.0,
        fee_rate=0.0, slippage=0.0,
        maintenance_margin_rate=0.004, max_drawdown_pct=0.99,
        funding_rate_epochs_utc=[0, 8, 16],
        primary_timeframe="1h",
    )
    res = bt.run({"1h": data}, _write_strategy(tmp_path, "long", sl=99, tp=99), funding_rate_df=fr_df)
    # 价格未变 → 无 trade pnl；最终 balance 应 < initial_balance（多头付资金费）
    assert res.equity_curve[-1] < 1000.0
    # 资金费 = position_size * entry * rate
    # size = equity * size_pct/100 * leverage / price = 1000 * 0.1 * 10 / 100 = 10
    # cash flow = -size_signed * notional * rate = -1 * (10*100) * 0.001 = -1
    # 但收尾平仓也会触发：用 close=100，entry=100 → pnl=0，无影响
    # 所以最终 equity ≈ 1000 - 1 = 999
    assert res.equity_curve[-1] == pytest.approx(999.0, abs=0.5)


def test_no_trades_when_signal_never_fires(tmp_path: Path) -> None:
    """trigger 全 0 → 无信号 → 无交易，equity 保持 initial。"""
    n = 5
    data = pl.DataFrame({
        "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(n)],
        "open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n, "close": [100.0] * n,
        "volume": [1.0] * n, "trigger": [0.0] * n,
    })
    bt = _bt()
    res = bt.run({"1h": data}, _write_strategy(tmp_path, "long", sl=3, tp=5))
    assert len(res.trades) == 0
    assert all(eq == pytest.approx(1000.0) for eq in res.equity_curve)


def test_dedup_same_direction_signal_not_added(tmp_path: Path) -> None:
    """已修复的 bug：trigger 持续为 1，仍只开一次仓（不会逐 bar 加仓）。"""
    n = 5
    data = pl.DataFrame({
        "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(n)],
        "open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n, "close": [100.0] * n,
        "volume": [1.0] * n, "trigger": [1.0] * n,
    })
    bt = _bt()
    res = bt.run({"1h": data}, _write_strategy(tmp_path, "long", sl=99, tp=99))
    opens = [t for t in res.trades if t.side.endswith("_open")]
    assert len(opens) == 1, f"应只开仓一次，实际 {len(opens)} 次"
