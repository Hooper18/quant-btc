"""IndicatorEngine 单元测试。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from indicators import IndicatorEngine, crossover, crossunder


# ---------- fixtures ----------
@pytest.fixture
def df_random() -> pl.DataFrame:
    """200 根随机游走 1h K 线。"""
    n = 200
    np.random.seed(42)
    ts = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(n)]
    close = np.cumsum(np.random.randn(n)) + 30000
    return pl.DataFrame({
        "timestamp": ts,
        "open": close + np.random.randn(n) * 5,
        "high": close + np.abs(np.random.randn(n)) * 10,
        "low": close - np.abs(np.random.randn(n)) * 10,
        "close": close,
        "volume": np.random.rand(n) * 100,
    })


# ---------- 基础校验 ----------
def test_init_rejects_missing_columns() -> None:
    df = pl.DataFrame({"timestamp": [datetime.now(timezone.utc)], "close": [1.0]})
    with pytest.raises(ValueError, match="缺列"):
        IndicatorEngine(df)


def test_each_indicator_returns_valid_series(df_random: pl.DataFrame) -> None:
    """每个指标至少在末尾有非 NaN 值，且行数不变。"""
    eng = IndicatorEngine(df_random)
    cases = [
        ("sma", {"period": 20}, ["sma_20"]),
        ("ema", {"period": 50}, ["ema_50"]),
        ("rsi", {"period": 14}, ["rsi_14"]),
        ("macd", {"fast": 12, "slow": 26, "signal": 9},
            ["macd_line_12_26_9", "macd_signal_12_26_9", "macd_histogram_12_26_9"]),
        ("adx", {"period": 14}, ["adx_14", "dmp_14", "dmn_14"]),
        ("stoch", {"k_period": 14, "d_period": 3, "smooth_k": 3},
            ["stoch_k_14_3_3", "stoch_d_14_3_3"]),
        ("cci", {"period": 20}, ["cci_20"]),
        ("williams_r", {"period": 14}, ["williams_r_14"]),
        ("mfi", {"period": 14}, ["mfi_14"]),
        ("bollinger", {"period": 20, "std_dev": 2.0},
            ["bb_lower_20_2.0", "bb_middle_20_2.0", "bb_upper_20_2.0"]),
        ("atr", {"period": 14}, ["atr_14"]),
        ("keltner", {"period": 20, "atr_period": 10, "multiplier": 2.0},
            ["kc_lower_20_10_2", "kc_middle_20_10_2", "kc_upper_20_10_2"]),
        ("obv", {}, ["obv"]),
        ("vwap", {}, ["vwap"]),
        ("cmf", {"period": 20}, ["cmf_20"]),
    ]
    for method, params, expected_cols in cases:
        out = getattr(eng, method)(**params)
        assert out.height == df_random.height, f"{method} 行数变化"
        for col in expected_cols:
            assert col in out.columns, f"{method} 缺列 {col}"
            tail = out[col].tail(5)
            non_null = tail.drop_nulls()
            assert non_null.len() > 0, f"{method} 列 {col} 末尾全为 null"


def test_compute_all_dict_form(df_random: pl.DataFrame) -> None:
    eng = IndicatorEngine(df_random)
    out = eng.compute_all({"rsi": {"period": 14}, "ema": {"period": 50}})
    assert "rsi_14" in out.columns and "ema_50" in out.columns


def test_compute_all_list_form_supports_multiple_periods(df_random: pl.DataFrame) -> None:
    eng = IndicatorEngine(df_random)
    out = eng.compute_all([
        ("ema", {"period": 12}),
        ("ema", {"period": 26}),
    ])
    assert "ema_12" in out.columns
    assert "ema_26" in out.columns


def test_compute_all_unknown_indicator_raises(df_random: pl.DataFrame) -> None:
    eng = IndicatorEngine(df_random)
    with pytest.raises(KeyError, match="未知指标"):
        eng.compute_all({"unknown_xx": {}})


# ---------- 边界值 ----------
def test_sma_period_one_equals_close(df_random: pl.DataFrame) -> None:
    """period=1 时 SMA 应与 close 完全相等。"""
    out = IndicatorEngine(df_random).sma(1)
    diff = (out["sma_1"] - out["close"]).abs().max()
    assert diff is not None and diff < 1e-9


def test_sma_period_too_large_returns_all_null(df_random: pl.DataFrame) -> None:
    """period 超过数据长度，所有值都应为 null。"""
    out = IndicatorEngine(df_random).sma(500)
    assert out["sma_500"].null_count() == out.height


def test_rsi_period_one_runs_without_error(df_random: pl.DataFrame) -> None:
    out = IndicatorEngine(df_random).rsi(1)
    assert "rsi_1" in out.columns


# ---------- 交叉检测 ----------
def test_crossover_basic() -> None:
    a = pl.Series("a", [1.0, 2.0, 3.0, 5.0, 4.0])
    b = pl.Series("b", [4.0, 4.0, 4.0, 4.0, 4.0])
    # a 在 idx=3 上穿 b（前一根 3 ≤ 4，当前 5 > 4）
    res = crossover(a, b)
    assert res.to_list() == [False, False, False, True, False]


def test_crossunder_basic() -> None:
    a = pl.Series("a", [5.0, 5.0, 5.0, 3.0, 4.0])
    b = pl.Series("b", [4.0, 4.0, 4.0, 4.0, 4.0])
    # a 在 idx=3 下穿 b（前一根 5 ≥ 4，当前 3 < 4）
    res = crossunder(a, b)
    assert res.to_list() == [False, False, False, True, False]


def test_crossover_no_event_when_a_stays_below() -> None:
    a = pl.Series("a", [1.0, 1.5, 2.0, 2.5])
    b = pl.Series("b", [3.0, 3.0, 3.0, 3.0])
    res = crossover(a, b)
    assert all(not v for v in res.to_list())


def test_crossover_first_bar_is_false() -> None:
    """首行没有 prev，必然返回 False。"""
    a = pl.Series("a", [10.0, 1.0])
    b = pl.Series("b", [5.0, 5.0])
    res = crossover(a, b)
    assert res[0] is False or res[0] is None or not bool(res[0])


def test_crossover_reverse_with_crossunder() -> None:
    """同一份序列，crossover 与 crossunder 在不同 bar 触发。"""
    a = pl.Series("a", [1.0, 2.0, 5.0, 3.0, 1.0])
    b = pl.Series("b", [3.0, 3.0, 3.0, 3.0, 3.0])
    co = crossover(a, b).to_list()
    cu = crossunder(a, b).to_list()
    # idx=2: a 上穿（前 2≤3，当前 5>3）
    # idx=4: a 下穿（前 3≥3，当前 1<3）  —— idx=3 时 a=3 不严格 < 3
    assert co == [False, False, True, False, False]
    assert cu == [False, False, False, False, True]
