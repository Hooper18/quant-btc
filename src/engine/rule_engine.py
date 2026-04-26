"""符号主义规则引擎：YAML 策略定义 → 信号生成。

支持的条件类型：
- 阈值比较：indicator + operator(>/</>=/<=/==/!=) + value
- 交叉检测：indicator + cross(above/below) + reference（值或另一个 indicator）
- 状态记忆：indicator + from_above/from_below(触发阈) + to_below/to_above(确认阈)
- 嵌套：conditions: [...] + logic: AND/OR

跨周期：每个条件可声明 timeframe；引擎从 data_dict[tf] 中按当前主时间戳向前对齐。
冲突仲裁：同时触发的相反方向信号，按 YAML 顺序保留最高优先级的方向；同向并存。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Signal:
    strategy_name: str
    side: str  # "long" | "short"
    type: str  # "market" | "limit"
    size_pct: float
    stop_loss_pct: float | None
    take_profit_pct: float | None
    timestamp: datetime
    priority: int = 0  # YAML 中越靠前优先级越高（数字越小）


@dataclass
class _Strategy:
    name: str
    conditions: list[dict[str, Any]]
    logic: str
    side: str
    type: str
    size_pct: float
    stop_loss_pct: float | None
    take_profit_pct: float | None
    priority: int


_OPS = {
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


class RuleEngine:
    """符号主义规则引擎。

    用法：
        engine = RuleEngine(data_dict={"1h": df_1h, "4h": df_4h}, primary_timeframe="1h")
        engine.load_rules("config/strategies.yaml")
        signals = engine.evaluate(df_1h, row_index=100)

    `evaluate` 内部会按主 TF 当前 row 的 timestamp，从 `data_dict` 各 TF 找对齐的最新 bar。
    """

    def __init__(self, data_dict: dict[str, pl.DataFrame], primary_timeframe: str):
        if primary_timeframe not in data_dict:
            raise ValueError(f"primary_timeframe={primary_timeframe} 不在 data_dict 中")
        self.data_dict = {tf: df.sort("timestamp") for tf, df in data_dict.items()}
        self.primary_tf = primary_timeframe
        self.strategies: list[_Strategy] = []
        # 状态：strategy_name -> condition_path -> bool（是否到达过触发阈）
        self._memory: dict[str, dict[str, bool]] = {}
        # 已触发的 cross：strategy_name -> condition_path -> 最近触发的 source_bar_idx
        self._cross_fired: dict[str, dict[str, int]] = {}

    # ---------- YAML 加载 ----------
    def load_rules(self, yaml_path: str | Path) -> None:
        path = Path(yaml_path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        raw = data.get("strategies", [])
        self.strategies = []
        for i, item in enumerate(raw):
            action = item.get("action") or {}
            self.strategies.append(_Strategy(
                name=item["name"],
                conditions=item.get("conditions", []),
                logic=str(item.get("logic", "AND")).upper(),
                side=action.get("side", "long"),
                type=action.get("type", "market"),
                size_pct=float(action.get("size_pct", 10)),
                stop_loss_pct=(float(item["stop_loss_pct"]) if "stop_loss_pct" in item else None),
                take_profit_pct=(float(item["take_profit_pct"]) if "take_profit_pct" in item else None),
                priority=i,
            ))
            self._memory.setdefault(item["name"], {})
            self._cross_fired.setdefault(item["name"], {})
        logger.info("加载策略 %d 条", len(self.strategies))

    # ---------- 跨周期取值 ----------
    def _row_idx_at(self, tf: str, ts: datetime) -> int | None:
        """tf 中 timestamp <= ts 的最大行索引；找不到返回 None。"""
        df = self.data_dict[tf]
        ts_col = df["timestamp"]
        # search_sorted side='right' 给出 > ts 的第一个；-1 即 <= ts 的最后一个
        idx = ts_col.search_sorted(ts, side="right") - 1
        if idx < 0:
            return None
        return int(idx)

    def _value(self, indicator: str, tf: str, ts: datetime, lag: int = 0) -> float | None:
        """tf 中 ts 对齐的指标值；lag>0 返回更早的 bar。"""
        idx = self._row_idx_at(tf, ts)
        if idx is None or idx - lag < 0:
            return None
        df = self.data_dict[tf]
        if indicator not in df.columns:
            raise KeyError(f"指标 {indicator} 不在 {tf} DataFrame 中（已有列：{df.columns[:6]}...）")
        v = df[indicator][idx - lag]
        return None if v is None else float(v)

    # ---------- 条件求值 ----------
    def _resolve_reference(self, ref: Any, tf: str, ts: datetime, lag: int = 0) -> float | None:
        """reference 可以是数字也可以是另一个指标列名。"""
        if isinstance(ref, (int, float)):
            return float(ref)
        if isinstance(ref, str):
            return self._value(ref, tf, ts, lag=lag)
        return None

    def _eval_condition(
        self,
        cond: dict[str, Any],
        strategy_name: str,
        ts: datetime,
        path: str,
    ) -> bool:
        # 嵌套条件组
        if "conditions" in cond:
            sub_logic = str(cond.get("logic", "AND")).upper()
            sub_results = [
                self._eval_condition(c, strategy_name, ts, f"{path}.{i}")
                for i, c in enumerate(cond["conditions"])
            ]
            if sub_logic == "OR":
                return any(sub_results)
            return all(sub_results)

        tf = cond.get("timeframe", self.primary_tf)
        indicator = cond.get("indicator")
        if indicator is None:
            raise ValueError(f"条件缺 indicator 字段：{cond}")

        cur = self._value(indicator, tf, ts)
        if cur is None:
            return False

        # 1) 阈值比较
        if "operator" in cond and "value" in cond:
            op = cond["operator"]
            if op not in _OPS:
                raise ValueError(f"未知 operator {op}")
            return bool(_OPS[op](cur, float(cond["value"])))

        # 2) 交叉检测
        if "cross" in cond and "reference" in cond:
            direction = cond["cross"]
            prev = self._value(indicator, tf, ts, lag=1)
            ref_cur = self._resolve_reference(cond["reference"], tf, ts)
            ref_prev = self._resolve_reference(cond["reference"], tf, ts, lag=1)
            if prev is None or ref_cur is None or ref_prev is None:
                return False
            if direction == "above":
                fired = (cur > ref_cur) and (prev <= ref_prev)
            elif direction == "below":
                fired = (cur < ref_cur) and (prev >= ref_prev)
            else:
                raise ValueError(f"未知 cross 方向 {direction}")
            if not fired:
                return False
            # 去重：同一 source bar 只触发一次
            src_idx = self._row_idx_at(tf, ts)
            last_fired = self._cross_fired.setdefault(strategy_name, {}).get(path)
            if last_fired == src_idx:
                return False
            self._cross_fired[strategy_name][path] = src_idx
            return True

        # 3) 状态记忆：from_above + to_below（指标曾上穿阈值后回落）
        if "from_above" in cond and "to_below" in cond:
            trigger_at = float(cond["from_above"])
            confirm_at = float(cond["to_below"])
            mem = self._memory.setdefault(strategy_name, {})
            if cur >= trigger_at:
                mem[path] = True
            if mem.get(path) and cur <= confirm_at:
                mem[path] = False
                return True
            return False

        # 4) 状态记忆：from_below + to_above
        if "from_below" in cond and "to_above" in cond:
            trigger_at = float(cond["from_below"])
            confirm_at = float(cond["to_above"])
            mem = self._memory.setdefault(strategy_name, {})
            if cur <= trigger_at:
                mem[path] = True
            if mem.get(path) and cur >= confirm_at:
                mem[path] = False
                return True
            return False

        raise ValueError(f"无法识别的条件结构：{cond}")

    # ---------- 主入口 ----------
    def evaluate(self, df: pl.DataFrame, row_index: int) -> list[Signal]:
        """对主 TF 当前 row_index 的 K 线评估所有策略，返回触发信号（已仲裁冲突）。"""
        if row_index < 0 or row_index >= df.height:
            raise IndexError(f"row_index={row_index} 越界（height={df.height}）")
        ts = df["timestamp"][row_index]
        # 主 TF df 必须等于 data_dict[primary_tf]，否则跨 TF 对齐会错位
        # 这里只做 ts 对齐，不强校验对象是否同一引用

        fired: list[Signal] = []
        for strat in self.strategies:
            try:
                sub_results = [
                    self._eval_condition(c, strat.name, ts, str(i))
                    for i, c in enumerate(strat.conditions)
                ]
            except Exception:
                logger.exception("策略 %s 条件评估异常", strat.name)
                continue
            triggered = (
                any(sub_results) if strat.logic == "OR" else (all(sub_results) and bool(sub_results))
            )
            if triggered:
                fired.append(Signal(
                    strategy_name=strat.name,
                    side=strat.side,
                    type=strat.type,
                    size_pct=strat.size_pct,
                    stop_loss_pct=strat.stop_loss_pct,
                    take_profit_pct=strat.take_profit_pct,
                    timestamp=ts,
                    priority=strat.priority,
                ))
        return self._resolve_conflicts(fired)

    def _resolve_conflicts(self, signals: list[Signal]) -> list[Signal]:
        """同一时刻方向冲突：保留最高优先级（priority 最小）那条的方向，丢弃反向。"""
        if len(signals) <= 1:
            return signals
        sides = {s.side for s in signals}
        if len(sides) == 1:
            return signals
        winner = min(signals, key=lambda s: s.priority)
        kept = [s for s in signals if s.side == winner.side]
        dropped = [s.strategy_name for s in signals if s.side != winner.side]
        logger.warning(
            "信号方向冲突 @%s：保留 %s 方向，丢弃 %s",
            signals[0].timestamp, winner.side, dropped,
        )
        return kept
