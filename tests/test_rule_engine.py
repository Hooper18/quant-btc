"""RuleEngine 单元测试。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from engine import RuleEngine, Signal


# ---------- helpers ----------
def _make_df(rsi_values: list[float], extra: dict[str, list[float]] | None = None) -> pl.DataFrame:
    n = len(rsi_values)
    ts = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(n)]
    cols: dict[str, list] = {
        "timestamp": ts,
        "open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n,
        "close": [100.0] * n, "volume": [1.0] * n,
        "rsi_14": rsi_values,
    }
    if extra:
        cols.update(extra)
    return pl.DataFrame(cols)


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "strat.yaml"
    p.write_text(body, encoding="utf-8")
    return p


# ---------- 阈值条件 ----------
def test_threshold_greater_fires_when_above(tmp_path: Path) -> None:
    df = _make_df([50, 80, 95, 60])
    eng = RuleEngine({"1h": df}, primary_timeframe="1h")
    yaml_path = _write_yaml(tmp_path, """
strategies:
  - name: high_rsi_short
    conditions:
      - indicator: rsi_14
        operator: ">"
        value: 90
        timeframe: "1h"
    logic: AND
    action: {side: short, type: market, size_pct: 10}
""")
    eng.load_rules(yaml_path)
    # row 0/1/3：rsi 不 > 90，不触发
    assert eng.evaluate(df, 0) == []
    assert eng.evaluate(df, 1) == []
    sigs = eng.evaluate(df, 2)
    assert len(sigs) == 1 and sigs[0].side == "short"
    assert eng.evaluate(df, 3) == []


def test_threshold_less_than(tmp_path: Path) -> None:
    df = _make_df([50, 25, 15, 40])
    eng = RuleEngine({"1h": df}, primary_timeframe="1h")
    yaml_path = _write_yaml(tmp_path, """
strategies:
  - name: low_rsi_long
    conditions:
      - indicator: rsi_14
        operator: "<"
        value: 20
        timeframe: "1h"
    logic: AND
    action: {side: long, type: market, size_pct: 10}
""")
    eng.load_rules(yaml_path)
    assert eng.evaluate(df, 1) == []  # 25 not < 20
    sigs = eng.evaluate(df, 2)         # 15 < 20
    assert len(sigs) == 1


def test_threshold_value_can_reference_another_indicator(tmp_path: Path) -> None:
    """value 字段为字符串时按指标名取当前 TF 当前值。"""
    df = _make_df(
        rsi_values=[10, 50, 70, 50],
        extra={"close_x": [101.0, 99.0, 105.0, 95.0], "ref_x": [100.0, 100.0, 100.0, 100.0]},
    )
    eng = RuleEngine({"1h": df}, primary_timeframe="1h")
    yaml_path = _write_yaml(tmp_path, """
strategies:
  - name: ref_indicator_compare
    conditions:
      - indicator: close_x
        operator: ">"
        value: ref_x
        timeframe: "1h"
    logic: AND
    action: {side: long, type: market, size_pct: 10}
""")
    eng.load_rules(yaml_path)
    assert eng.evaluate(df, 0) != []  # 101 > 100
    assert eng.evaluate(df, 1) == []  # 99 > 100 false
    assert eng.evaluate(df, 2) != []  # 105 > 100
    assert eng.evaluate(df, 3) == []


# ---------- 交叉条件 ----------
def test_cross_above_fires_once_per_source_bar(tmp_path: Path) -> None:
    """已修复：cross 触发后，同一 source bar 不再二次触发。"""
    df = _make_df(
        rsi_values=[50] * 6,
        extra={"a": [1.0, 2.0, 3.0, 5.0, 6.0, 7.0], "b": [4.0] * 6},
    )
    eng = RuleEngine({"1h": df}, primary_timeframe="1h")
    yaml_path = _write_yaml(tmp_path, """
strategies:
  - name: cross_long
    conditions:
      - indicator: a
        cross: above
        reference: b
        timeframe: "1h"
    logic: AND
    action: {side: long, type: market, size_pct: 10}
""")
    eng.load_rules(yaml_path)
    fired = [bool(eng.evaluate(df, i)) for i in range(6)]
    # idx=3 时 a=5 上穿 b=4（prev a=3 ≤ 4）
    assert fired == [False, False, False, True, False, False]


def test_cross_below(tmp_path: Path) -> None:
    df = _make_df(
        rsi_values=[50] * 5,
        extra={"a": [10.0, 8.0, 5.0, 2.0, 1.0], "b": [4.0] * 5},
    )
    eng = RuleEngine({"1h": df}, primary_timeframe="1h")
    yaml_path = _write_yaml(tmp_path, """
strategies:
  - name: cross_short
    conditions:
      - indicator: a
        cross: below
        reference: b
        timeframe: "1h"
    logic: AND
    action: {side: short, type: market, size_pct: 10}
""")
    eng.load_rules(yaml_path)
    fired = [bool(eng.evaluate(df, i)) for i in range(5)]
    # idx=3 时 a=2 下穿 b=4（prev a=5 ≥ 4）
    assert fired == [False, False, False, True, False]


# ---------- 状态记忆 ----------
def test_from_above_to_below(tmp_path: Path) -> None:
    """指标先上穿 80，再回落跌破 70 时触发；只触发一次。"""
    df = _make_df([50, 85, 75, 65, 70, 88, 60])
    eng = RuleEngine({"1h": df}, primary_timeframe="1h")
    yaml_path = _write_yaml(tmp_path, """
strategies:
  - name: rsi_pullback_short
    conditions:
      - indicator: rsi_14
        from_above: 80
        to_below: 70
        timeframe: "1h"
    logic: AND
    action: {side: short, type: market, size_pct: 10}
""")
    eng.load_rules(yaml_path)
    fired = [bool(eng.evaluate(df, i)) for i in range(df.height)]
    # 触发点：idx=3 (65 ≤ 70 且曾 ≥ 80) 和 idx=6 (60 ≤ 70 且 idx=5 时 88 ≥ 80)
    assert fired == [False, False, False, True, False, False, True]


def test_from_below_to_above(tmp_path: Path) -> None:
    df = _make_df([50, 15, 25, 35, 30, 12, 40])
    eng = RuleEngine({"1h": df}, primary_timeframe="1h")
    yaml_path = _write_yaml(tmp_path, """
strategies:
  - name: rsi_oversold_bounce_long
    conditions:
      - indicator: rsi_14
        from_below: 20
        to_above: 30
        timeframe: "1h"
    logic: AND
    action: {side: long, type: market, size_pct: 10}
""")
    eng.load_rules(yaml_path)
    fired = [bool(eng.evaluate(df, i)) for i in range(df.height)]
    # 触发点：idx=3 (35 ≥ 30 且曾 ≤ 20)；idx=6 (40 ≥ 30 且 idx=5 时 12 ≤ 20)
    assert fired == [False, False, False, True, False, False, True]


# ---------- AND / OR ----------
def test_and_logic_requires_all(tmp_path: Path) -> None:
    df = _make_df([95, 95, 50], extra={"x": [1, 0, 0]})
    eng = RuleEngine({"1h": df}, primary_timeframe="1h")
    yaml_path = _write_yaml(tmp_path, """
strategies:
  - name: and_test
    conditions:
      - {indicator: rsi_14, operator: ">", value: 90, timeframe: "1h"}
      - {indicator: x,      operator: ">", value: 0,  timeframe: "1h"}
    logic: AND
    action: {side: short, type: market, size_pct: 10}
""")
    eng.load_rules(yaml_path)
    assert eng.evaluate(df, 0) != []  # both true
    assert eng.evaluate(df, 1) == []  # rsi true, x false
    assert eng.evaluate(df, 2) == []  # rsi false


def test_or_logic_requires_any(tmp_path: Path) -> None:
    df = _make_df([95, 50, 50], extra={"x": [0, 1, 0]})
    eng = RuleEngine({"1h": df}, primary_timeframe="1h")
    yaml_path = _write_yaml(tmp_path, """
strategies:
  - name: or_test
    conditions:
      - {indicator: rsi_14, operator: ">", value: 90, timeframe: "1h"}
      - {indicator: x,      operator: ">", value: 0,  timeframe: "1h"}
    logic: OR
    action: {side: short, type: market, size_pct: 10}
""")
    eng.load_rules(yaml_path)
    assert eng.evaluate(df, 0) != []  # rsi true
    assert eng.evaluate(df, 1) != []  # x true
    assert eng.evaluate(df, 2) == []  # neither


def test_nested_conditions(tmp_path: Path) -> None:
    df = _make_df([95, 95, 50, 50], extra={"x": [1, 0, 1, 0]})
    eng = RuleEngine({"1h": df}, primary_timeframe="1h")
    yaml_path = _write_yaml(tmp_path, """
strategies:
  - name: nested
    conditions:
      - conditions:
          - {indicator: rsi_14, operator: ">", value: 90, timeframe: "1h"}
          - {indicator: x,      operator: ">", value: 0,  timeframe: "1h"}
        logic: OR
    logic: AND
    action: {side: long, type: market, size_pct: 10}
""")
    eng.load_rules(yaml_path)
    fired = [bool(eng.evaluate(df, i)) for i in range(df.height)]
    # OR 中至少一个为真：idx 0/1/2 是；idx 3 都假
    assert fired == [True, True, True, False]


# ---------- 冲突仲裁 ----------
def test_conflict_resolution_prefers_higher_priority(tmp_path: Path) -> None:
    """同时触发的反向信号：YAML 顺序靠前的优先（priority 数字小）。"""
    df = _make_df([95])
    eng = RuleEngine({"1h": df}, primary_timeframe="1h")
    yaml_path = _write_yaml(tmp_path, """
strategies:
  - name: short_strategy
    conditions: [{indicator: rsi_14, operator: ">", value: 90, timeframe: "1h"}]
    logic: AND
    action: {side: short, type: market, size_pct: 10}
  - name: long_strategy
    conditions: [{indicator: rsi_14, operator: ">", value: 90, timeframe: "1h"}]
    logic: AND
    action: {side: long, type: market, size_pct: 10}
""")
    eng.load_rules(yaml_path)
    sigs = eng.evaluate(df, 0)
    assert len(sigs) == 1
    assert sigs[0].side == "short"
    assert sigs[0].strategy_name == "short_strategy"


def test_same_direction_signals_kept(tmp_path: Path) -> None:
    df = _make_df([95])
    eng = RuleEngine({"1h": df}, primary_timeframe="1h")
    yaml_path = _write_yaml(tmp_path, """
strategies:
  - name: short_a
    conditions: [{indicator: rsi_14, operator: ">", value: 90, timeframe: "1h"}]
    logic: AND
    action: {side: short, type: market, size_pct: 10}
  - name: short_b
    conditions: [{indicator: rsi_14, operator: ">", value: 90, timeframe: "1h"}]
    logic: AND
    action: {side: short, type: market, size_pct: 5}
""")
    eng.load_rules(yaml_path)
    sigs = eng.evaluate(df, 0)
    assert len(sigs) == 2
    assert all(s.side == "short" for s in sigs)


# ---------- 跨时间周期 ----------
def test_cross_timeframe_alignment(tmp_path: Path) -> None:
    """1h 主周期、4h 条件：1h ts=2024-01-01 04:00 应对齐到 4h 第二根（00:00→ 4h bar，04:00→第二个 4h bar 的开始）。"""
    n_1h = 12
    ts_1h = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(n_1h)]
    df_1h = pl.DataFrame({
        "timestamp": ts_1h,
        "open": [100.0] * n_1h, "high": [101.0] * n_1h, "low": [99.0] * n_1h,
        "close": [100.0] * n_1h, "volume": [1.0] * n_1h,
        "rsi_14": [50.0] * n_1h,
    })
    # 4h: 00:00 / 04:00 / 08:00 三根
    ts_4h = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i * 4) for i in range(3)]
    df_4h = pl.DataFrame({
        "timestamp": ts_4h,
        "open": [100.0] * 3, "high": [101.0] * 3, "low": [99.0] * 3,
        "close": [100.0] * 3, "volume": [1.0] * 3,
        "trend_4h": [1.0, 0.0, 1.0],   # 第二根 4h（04:00）trend=0
    })
    eng = RuleEngine({"1h": df_1h, "4h": df_4h}, primary_timeframe="1h")
    yaml_path = _write_yaml(tmp_path, """
strategies:
  - name: tf_test
    conditions:
      - indicator: trend_4h
        operator: ">"
        value: 0.5
        timeframe: "4h"
    logic: AND
    action: {side: long, type: market, size_pct: 10}
""")
    eng.load_rules(yaml_path)
    fired = [bool(eng.evaluate(df_1h, i)) for i in range(n_1h)]
    # 1h idx 0..3 (00:00..03:00) 落在第一根 4h 内（trend=1）→ 触发
    # 1h idx 4..7 (04:00..07:00) 落在第二根 4h 内（trend=0）→ 不触发
    # 1h idx 8..11 (08:00..11:00) 落在第三根 4h 内（trend=1）→ 触发
    assert fired[0:4] == [True, True, True, True]
    assert fired[4:8] == [False, False, False, False]
    assert fired[8:12] == [True, True, True, True]
