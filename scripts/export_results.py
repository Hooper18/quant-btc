"""把回测产物导出为前端可用的 JSON。

主要变更（Phase 6）：
- 遍历 config/strategies*.yaml 全部策略，逐个跑 BTC/ETH/SOL 回测
- 每个策略生成 web/public/data/strategies/{id}.json（含规则描述、3 币种结果）
- 生成 web/public/data/strategies_index.json 索引（前端排行榜/选择器用）
- 兼容保留：BTCUSDT_*.json / ETHUSDT_*.json / SOLUSDT_*.json（v2_optimized 结果，
  Risk 页/Dashboard 旧链路仍可工作）+ walk_forward / leverage / monte_carlo / sensitivity

输出（路径相对项目根 web/public/data/）：
    strategies/{id}.json            单策略详情（3 币种 metrics + equity + monthly + trades）
    strategies_index.json           策略索引 + 排行用摘要
    {symbol}_*.json                 兼容旧前端（v2_optimized 结果）
    walk_forward_summary.json / leverage_scan.json / monte_carlo.json / sensitivity.json
    portfolio.json / fear_greed_latest.json
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import re
import sys
import tempfile
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from backtest import Backtester, MonteCarloSimulator  # noqa: E402
from backtest.optimizer import set_param  # noqa: E402
from utils.config import DataConfig  # noqa: E402

from run_backtest import (  # noqa: E402
    _collect_required_indicators,
    build_data_dict,
    load_aux_data,
)

DATA_OUT = PROJECT_ROOT / "web" / "public" / "data"
STRATEGY_DIR = DATA_OUT / "strategies"
DEFAULT_BTC_STRATEGY = "strategies_v2_optimized.yaml"  # 用于兼容旧产物
BACKTEST_CFG = PROJECT_ROOT / "config" / "backtest_config.yaml"
DATA_CFG = PROJECT_ROOT / "config" / "data_config.yaml"
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _used_timeframes(strategies: list[dict], primary: str) -> set[str]:
    used: set[str] = {primary}
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


def _has_data(symbol: str, primary_tf: str) -> bool:
    sym_dir = PROJECT_ROOT / "data" / "parquet" / symbol
    return sym_dir.exists() and any(sym_dir.glob(f"{primary_tf}_*.parquet"))


def _write_json(path: Path, payload: Any) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=_json_default)
    path.write_text(text, encoding="utf-8")
    return path.stat().st_size


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Unserializable: {type(obj)}")


# ---------- 策略规则 → 自然语言 ----------

_OP_ZH = {">": "＞", "<": "＜", ">=": "≥", "<=": "≤", "==": "＝", "!=": "≠"}
_SIDE_ZH = {"long": "做多", "short": "做空"}
_CROSS_ZH = {"above": "上穿", "below": "下穿"}


def _describe_condition(c: dict[str, Any]) -> str:
    if "conditions" in c:
        # 嵌套
        sub = " 且 " if c.get("logic", "AND") == "AND" else " 或 "
        return "(" + sub.join(_describe_condition(s) for s in c["conditions"]) + ")"
    ind = c.get("indicator", "?")
    tf = c.get("timeframe")
    tf_str = f"·{tf}" if tf else ""
    if "cross" in c:
        cross_zh = _CROSS_ZH.get(c["cross"], c["cross"])
        return f"{ind}{tf_str} {cross_zh} {c.get('reference', '?')}"
    if "from_above" in c and "to_below" in c:
        return f"{ind}{tf_str} 从 ≥{c['from_above']} 跌破 {c['to_below']}"
    if "from_below" in c and "to_above" in c:
        return f"{ind}{tf_str} 从 ≤{c['from_below']} 突破 {c['to_above']}"
    op = _OP_ZH.get(c.get("operator", ""), c.get("operator", "?"))
    val = c.get("value", "?")
    return f"{ind}{tf_str} {op} {val}"


def _describe_strategies(strat_yaml: dict) -> tuple[str, list[dict]]:
    """生成策略文字摘要 + 结构化条目列表。"""
    items: list[dict] = []
    lines: list[str] = []
    for s in strat_yaml.get("strategies", []):
        name = s.get("name", "?")
        side = s.get("action", {}).get("side", "?")
        size_pct = s.get("action", {}).get("size_pct", "?")
        sl = s.get("stop_loss_pct", "?")
        tp = s.get("take_profit_pct", "?")
        conds = [_describe_condition(c) for c in s.get("conditions", [])]
        joiner = " 且 " if s.get("logic", "AND") == "AND" else " 或 "
        cond_str = joiner.join(conds) if conds else "—"
        side_zh = _SIDE_ZH.get(side, side)
        line = f"{name}：当 {cond_str} → {side_zh}（仓位 {size_pct}% · 止损 {sl}% · 止盈 {tp}%）"
        lines.append(line)
        items.append({
            "name": name,
            "side": side,
            "side_zh": side_zh,
            "conditions_text": cond_str,
            "size_pct": size_pct,
            "stop_loss_pct": sl,
            "take_profit_pct": tp,
        })
    return "\n".join(lines), items


# 文件名 → 策略类型默认标签 + 友好显示名
_STRATEGY_META: dict[str, dict[str, Any]] = {
    "strategies_v2_optimized": {
        "display_name": "V2 最优参数（RSI+MACD）",
        "tags": ["RSI超买", "MACD趋势", "最优参数", "BTC专用"],
        "category": "复合趋势",
    },
    "strategies": {
        "display_name": "经典 RSI+MACD 基础版",
        "tags": ["RSI超买", "MACD趋势", "基础版"],
        "category": "复合趋势",
    },
    "strategies_trend_follow": {
        "display_name": "EMA 交叉趋势跟踪",
        "tags": ["趋势跟踪", "EMA交叉", "ADX 过滤"],
        "category": "趋势跟踪",
    },
    "strategies_mean_reversion": {
        "display_name": "极端均值回归",
        "tags": ["均值回归", "RSI(6)", "布林带"],
        "category": "均值回归",
    },
    "strategies_bollinger_squeeze": {
        "display_name": "布林通道突破",
        "tags": ["布林带突破", "趋势跟踪", "ADX 过滤"],
        "category": "波动率突破",
    },
    "strategies_enhanced_rsi": {
        "display_name": "增强 RSI（情绪+主动盘）",
        "tags": ["RSI超买", "市场情绪", "主动盘比例"],
        "category": "情绪反向",
    },
    "strategies_multi_signal": {
        "display_name": "多周期复合信号",
        "tags": ["EMA交叉", "RSI", "MACD", "多周期"],
        "category": "动量策略",
    },
    "strategies_oi_divergence": {
        "display_name": "OI 持仓量背离",
        "tags": ["持仓量背离", "新高新低"],
        "category": "持仓量背离",
    },
    "strategies_rsi_divergence": {
        "display_name": "RSI 极值回归",
        "tags": ["RSI背离", "动量耗尽"],
        "category": "RSI 背离",
    },
    "strategies_sentiment": {
        "display_name": "市场情绪反向",
        "tags": ["市场情绪", "FNG", "多空比"],
        "category": "情绪反向",
    },
}


def _strategy_meta(stem: str) -> dict[str, Any]:
    return _STRATEGY_META.get(stem, {
        "display_name": stem,
        "tags": [],
        "category": "未分类",
    })


# ---------- 单次 (策略, 币种) 回测 ----------

def _run_one(
    symbol: str, strat_yaml: dict, backtester: Backtester
) -> tuple[Any, pl.DataFrame] | None:
    data_cfg = DataConfig.from_yaml(DATA_CFG).for_symbol(symbol)
    if not _has_data(symbol, backtester.primary_tf):
        return None
    used_tfs = _used_timeframes(strat_yaml["strategies"], backtester.primary_tf)
    ind_cfg = _collect_required_indicators(strat_yaml["strategies"])
    aux = load_aux_data(data_cfg)
    try:
        data_dict = build_data_dict(data_cfg, used_tfs, ind_cfg, aux=aux)
    except (FileNotFoundError, KeyError) as e:
        logging.warning("跳过 %s × %s：%s", symbol, strat_yaml.get("_filename", "?"), e)
        return None

    tmp = Path(tempfile.gettempdir()) / f"strat_{uuid.uuid4().hex}.yaml"
    payload = {k: v for k, v in strat_yaml.items() if not k.startswith("_")}
    tmp.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    try:
        result = backtester.run(data_dict, str(tmp), funding_rate_df=aux.get("funding"))
    except Exception as e:
        logging.warning("回测失败 %s × %s: %s", symbol, strat_yaml.get("_filename", "?"), e)
        return None
    finally:
        tmp.unlink(missing_ok=True)
    return result, data_dict[backtester.primary_tf]


def _equity_daily(result: Any, primary_df: pl.DataFrame) -> list[dict]:
    if not result.timestamps:
        return []
    price_lookup: dict[datetime, float] = {}
    for ts, close in zip(primary_df["timestamp"].to_list(), primary_df["close"].to_list()):
        price_lookup[ts] = float(close)
    daily: dict[date, tuple[datetime, float, float]] = {}
    for ts, eq in zip(result.timestamps, result.equity_curve):
        d = ts.date()
        price = price_lookup.get(ts, float("nan"))
        daily[d] = (ts, float(eq), price)
    return [
        {
            "t": ts.strftime("%Y-%m-%d"),
            "nav": round(eq, 2),
            "price": None if np.isnan(price) else round(price, 2),
        }
        for d in sorted(daily)
        for ts, eq, price in [daily[d]]
    ]


def _equity_sparkline(equity_pts: list[dict], n: int = 60) -> list[float]:
    """采样到约 n 个点用于排行榜的迷你图。"""
    if not equity_pts:
        return []
    if len(equity_pts) <= n:
        return [p["nav"] for p in equity_pts]
    step = len(equity_pts) / n
    return [round(equity_pts[int(i * step)]["nav"], 2) for i in range(n)]


def _metrics_payload(result: Any) -> dict[str, Any]:
    m = result.metrics
    return {
        "initial_balance": round(m.get("initial_balance", 0), 2),
        "final_equity": round(m.get("final_equity", 0), 2),
        "total_return_pct": round(m.get("total_return_pct", 0), 2),
        "annualized_return_pct": round(m.get("annualized_return_pct", 0), 2),
        "sharpe_ratio": round(m.get("sharpe_ratio", 0), 3),
        "max_drawdown_pct": round(m.get("max_drawdown_pct", 0), 2),
        "total_trades": int(m.get("total_trades", 0)),
        "win_rate_pct": round(m.get("win_rate_pct", 0), 2),
        "profit_loss_ratio": round(m.get("profit_loss_ratio", 0), 3),
        "avg_holding_hours": round(m.get("avg_holding_hours", 0), 2),
        "circuit_breaker": int(m.get("circuit_breaker", 0)),
    }


def _trades_payload(result: Any, limit: int = 100) -> dict[str, Any]:
    closed = [t for t in result.trades if t.side.endswith("_close") or t.side == "liquidate"]
    last = closed[-limit:]
    rows = [
        {
            "t": t.timestamp.isoformat(),
            "side": t.side,
            "price": round(t.price, 2),
            "size": round(t.size, 6),
            "pnl": round(t.pnl, 4),
            "fee": round(t.fee, 4),
            "strategy": t.strategy,
        }
        for t in last
    ]
    return {"total_closed": len(closed), "shown": len(rows), "rows": rows}


def _monthly_matrix(result: Any) -> list[dict]:
    if not result.timestamps:
        return []
    month_last: dict[tuple[int, int], float] = {}
    for ts, eq in zip(result.timestamps, result.equity_curve):
        month_last[(ts.year, ts.month)] = float(eq)
    if not month_last:
        return []
    keys = sorted(month_last)
    initial = result.metrics.get("initial_balance", 100.0)
    prev = float(initial)
    rows: dict[int, dict[int, float]] = {}
    for (y, m) in keys:
        cur = month_last[(y, m)]
        ret = (cur - prev) / prev * 100.0 if prev > 0 else 0.0
        rows.setdefault(y, {})[m] = round(ret, 2)
        prev = cur
    return [
        {"year": y, "months": [rows.get(y, {}).get(mm) for mm in range(1, 13)]}
        for y in sorted(rows)
    ]


# ---------- 单策略导出 ----------

def _export_one_strategy(
    yaml_path: Path, backtester: Backtester, symbols: tuple[str, ...] = DEFAULT_SYMBOLS
) -> dict[str, Any] | None:
    """跑 (yaml × 全币种) 回测，写 strategies/{id}.json，返回索引摘要。"""
    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if "strategies" not in raw or not raw["strategies"]:
        logging.warning("跳过 %s：无 strategies 字段", yaml_path.name)
        return None
    raw["_filename"] = yaml_path.name
    stem = yaml_path.stem
    meta = _strategy_meta(stem)
    rules_text, rule_items = _describe_strategies(raw)

    log = logging.getLogger("export_strategy")
    log.info("===== %s (%s) =====", stem, meta["display_name"])

    by_symbol: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        out = _run_one(symbol, raw, backtester)
        if out is None:
            continue
        result, primary_df = out
        equity_pts = _equity_daily(result, primary_df)
        metrics = _metrics_payload(result)
        log.info(
            "  %s: 收益=%.2f%% 夏普=%.2f 回撤=%.2f%% 胜率=%.2f%% 笔数=%d",
            symbol, metrics["total_return_pct"], metrics["sharpe_ratio"],
            metrics["max_drawdown_pct"], metrics["win_rate_pct"], metrics["total_trades"],
        )
        by_symbol[symbol] = {
            "metrics": metrics,
            "equity_points": equity_pts,
            "sparkline": _equity_sparkline(equity_pts, n=60),
            "monthly_matrix": _monthly_matrix(result),
            "trades": _trades_payload(result, limit=100),
        }

    if not by_symbol:
        logging.warning("策略 %s 无任何币种成功 → 跳过导出", stem)
        return None

    default_symbol = "BTCUSDT" if "BTCUSDT" in by_symbol else next(iter(by_symbol))
    payload = {
        "id": stem,
        "file": yaml_path.name,
        "display_name": meta["display_name"],
        "category": meta["category"],
        "tags": meta["tags"],
        "rules_text": rules_text,
        "rule_items": rule_items,
        "applicable_symbols": list(by_symbol.keys()),
        "default_symbol": default_symbol,
        "by_symbol": by_symbol,
    }
    size = _write_json(STRATEGY_DIR / f"{stem}.json", payload)
    log.info("  → 写出 strategies/%s.json (%.1f KB)", stem, size / 1024)

    # 给索引提供基于"默认币种"的关键指标
    primary = by_symbol[default_symbol]["metrics"]
    return {
        "id": stem,
        "file": yaml_path.name,
        "display_name": meta["display_name"],
        "category": meta["category"],
        "tags": meta["tags"],
        "rules_text": rules_text,
        "default_symbol": default_symbol,
        "applicable_symbols": list(by_symbol.keys()),
        "metrics": {
            sym: by_symbol[sym]["metrics"] for sym in by_symbol
        },
        "primary": primary,
        "sparkline": by_symbol[default_symbol]["sparkline"],
    }


def _normalize(values: list[float], inverse: bool = False) -> list[float]:
    """min-max 归一化到 [0,1]；inverse=True 时 1 表示越小越好。"""
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span == 0:
        return [0.5] * len(values)
    out = [(v - lo) / span for v in values]
    if inverse:
        out = [1 - x for x in out]
    return out


def _build_index(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """生成 strategies_index.json，含综合评分。"""
    sharpes = [r["primary"]["sharpe_ratio"] for r in rows]
    annualized = [r["primary"]["annualized_return_pct"] for r in rows]
    drawdowns = [r["primary"]["max_drawdown_pct"] for r in rows]
    winrates = [r["primary"]["win_rate_pct"] for r in rows]

    n_sharpe = _normalize(sharpes)
    n_ann = _normalize(annualized)
    n_dd = _normalize(drawdowns, inverse=True)  # 回撤越小越好
    n_win = _normalize(winrates)

    for i, r in enumerate(rows):
        composite = (
            0.3 * n_sharpe[i] + 0.3 * n_ann[i] + 0.2 * n_dd[i] + 0.2 * n_win[i]
        )
        r["composite_score"] = round(composite * 100, 2)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "weights": {"sharpe": 0.3, "annualized": 0.3, "max_drawdown_inv": 0.2, "win_rate": 0.2},
        "count": len(rows),
        "strategies": rows,
    }


# ---------- 兼容旧前端：导出 v2_optimized 的 BTC 多币种文件 ----------

def _legacy_per_symbol_exports(
    rows: list[dict[str, Any]], backtester: Backtester
) -> dict[str, int]:
    """复用 v2_optimized 的 by_symbol 数据写出 {symbol}_*.json 兼容文件。"""
    sizes: dict[str, int] = {}
    target = next((r for r in rows if r["id"] == "strategies_v2_optimized"), None)
    if target is None:
        target = max(rows, key=lambda r: r["primary"]["sharpe_ratio"])
    detail_path = STRATEGY_DIR / f"{target['id']}.json"
    if not detail_path.exists():
        return sizes
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    for symbol, blob in detail["by_symbol"].items():
        m = blob["metrics"]
        sizes[f"{symbol}_metrics.json"] = _write_json(
            DATA_OUT / f"{symbol}_metrics.json", {"symbol": symbol, **m}
        )
        sizes[f"{symbol}_equity_curve.json"] = _write_json(
            DATA_OUT / f"{symbol}_equity_curve.json",
            {"symbol": symbol, "points": blob["equity_points"]},
        )
        sizes[f"{symbol}_trades.json"] = _write_json(
            DATA_OUT / f"{symbol}_trades.json",
            {"symbol": symbol, **blob["trades"]},
        )
        sizes[f"{symbol}_monthly_returns.json"] = _write_json(
            DATA_OUT / f"{symbol}_monthly_returns.json",
            {"symbol": symbol, "matrix": blob["monthly_matrix"]},
        )
    return sizes


# ---------- 衍生：MC / WF / 杠杆 / 敏感度 / portfolio / FNG（保留 BTC v2 链路）----------

def _export_monte_carlo(symbol: str, result: Any) -> int:
    sim = MonteCarloSimulator(result)
    mc = sim.run(n_simulations=1000, seed=42)
    qs = (5, 25, 50, 75, 95)
    payload = {
        "symbol": symbol,
        "n_simulations": mc.n_simulations,
        "initial_balance": mc.initial_balance,
        "ruin_threshold": mc.ruin_threshold,
        "ruin_probability": round(mc.ruin_probability, 4),
        "return_pct_percentiles": {str(q): round(mc.percentiles(mc.final_returns_pct, [q])[q], 2) for q in qs},
        "max_dd_pct_percentiles": {str(q): round(mc.percentiles(mc.max_drawdowns_pct, [q])[q], 2) for q in qs},
        "return_pct_mean": round(float(np.mean(mc.final_returns_pct)), 2),
        "max_dd_pct_mean": round(float(np.mean(mc.max_drawdowns_pct)), 2),
        "return_histogram": _histogram(mc.final_returns_pct, bins=30),
        "max_dd_histogram": _histogram(mc.max_drawdowns_pct, bins=30),
    }
    return _write_json(DATA_OUT / "monte_carlo.json", payload)


def _histogram(arr: np.ndarray, bins: int = 30) -> dict[str, list[float]]:
    counts, edges = np.histogram(arr, bins=bins)
    return {
        "edges": [round(float(x), 2) for x in edges.tolist()],
        "counts": [int(x) for x in counts.tolist()],
    }


def _export_walk_forward() -> int:
    candidates = sorted((PROJECT_ROOT / "output").glob("walk_forward_*_opt"))
    if not candidates:
        candidates = sorted((PROJECT_ROOT / "output").glob("walk_forward_*"))
    if not candidates:
        logging.warning("跳过 walk_forward：找不到 output/walk_forward_*")
        return 0
    src = candidates[-1]
    csv_path = src / "windows.csv"
    if not csv_path.exists():
        return 0
    windows: list[dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            windows.append({
                "window": int(r["window"]),
                "train_start": r["train_start"],
                "train_end": r["train_end"],
                "test_start": r["test_start"],
                "test_end": r["test_end"],
                "train_return_pct": round(float(r["train_total_return_pct"]), 2),
                "train_sharpe": round(float(r["train_sharpe"]), 2),
                "test_return_pct": round(float(r["test_total_return_pct"]), 2),
                "test_sharpe": round(float(r["test_sharpe"]), 2),
                "test_max_dd_pct": round(float(r["test_max_dd_pct"]), 2),
                "test_win_rate_pct": round(float(r["test_win_rate_pct"]), 2),
                "test_total_trades": int(float(r["test_total_trades"])),
            })
    summary: dict[str, Any] = {}
    rpt = (src / "report.txt").read_text(encoding="utf-8") if (src / "report.txt").exists() else ""
    for key, pat in [
        ("total_return_pct", r"总收益率:\s*([0-9\-.]+)%"),
        ("annualized_return_pct", r"年化收益率:\s*([0-9\-.]+)%"),
        ("sharpe_ratio", r"夏普比率:\s*([0-9\-.]+)"),
        ("max_drawdown_pct", r"最大回撤:\s*([0-9\-.]+)%"),
        ("win_rate_pct", r"胜率（加权）:\s*([0-9\-.]+)%"),
        ("total_trades", r"交易笔数:\s*(\d+)"),
        ("n_windows", r"窗口数:\s*(\d+)"),
    ]:
        m = re.search(pat, rpt)
        if m:
            v = m.group(1)
            summary[key] = int(v) if key in ("total_trades", "n_windows") else float(v)
    return _write_json(
        DATA_OUT / "walk_forward_summary.json",
        {"source": src.name, "summary": summary, "windows": windows},
    )


def _export_leverage_scan() -> int:
    src = PROJECT_ROOT / "output" / "leverage_scan"
    csv_path = src / "table.csv"
    if not csv_path.exists():
        return 0
    rows: list[dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "leverage": int(r["leverage"]),
                "annualized_return_pct": round(float(r["annualized_return_pct"]), 2),
                "sharpe_ratio": round(float(r["sharpe_ratio"]), 3),
                "max_drawdown_pct": round(float(r["max_drawdown_pct"]), 2),
                "win_rate_pct": round(float(r["win_rate_pct"]), 2),
                "total_trades": int(r["total_trades"]),
                "ruin_probability": round(float(r["ruin_probability"]), 4),
            })
    rec = ""
    rpt = (src / "report.txt").read_text(encoding="utf-8") if (src / "report.txt").exists() else ""
    m = re.search(r"建议生产用\s*(\d+)×", rpt)
    if m:
        rec = m.group(1)
    return _write_json(
        DATA_OUT / "leverage_scan.json",
        {"recommended_leverage": int(rec) if rec else None, "rows": rows},
    )


def _export_sensitivity(strat_yaml: dict, backtester: Backtester) -> int:
    data_cfg = DataConfig.from_yaml(DATA_CFG).for_symbol("BTCUSDT")
    if not _has_data("BTCUSDT", backtester.primary_tf):
        return 0
    used_tfs = _used_timeframes(strat_yaml["strategies"], backtester.primary_tf)
    ind_cfg = _collect_required_indicators(strat_yaml["strategies"])
    aux = load_aux_data(data_cfg)
    data_dict = build_data_dict(data_cfg, used_tfs, ind_cfg, aux=aux)
    fr_df = aux.get("funding")

    BEST = {
        "strategies[0].conditions[0].value": 55,
        "strategies[0].action.size_pct": 3,
        "strategies[0].stop_loss_pct": 5,
        "strategies[0].take_profit_pct": 4,
    }
    GRIDS = [
        ("rsi_x_sl",   ("strategies[0].conditions[0].value", "RSI阈值",   [50, 55, 60, 65, 70]),
                       ("strategies[0].stop_loss_pct",       "止损%",     [3, 4, 5, 6, 7])),
        ("rsi_x_tp",   ("strategies[0].conditions[0].value", "RSI阈值",   [50, 55, 60, 65, 70]),
                       ("strategies[0].take_profit_pct",     "止盈%",     [3, 4, 5, 6, 8])),
        ("sl_x_tp",    ("strategies[0].stop_loss_pct",       "止损%",     [3, 4, 5, 6, 7]),
                       ("strategies[0].take_profit_pct",     "止盈%",     [3, 4, 5, 6, 8])),
        ("rsi_x_size", ("strategies[0].conditions[0].value", "RSI阈值",   [50, 55, 60, 65, 70]),
                       ("strategies[0].action.size_pct",     "仓位%",     [2, 3, 5, 7, 10])),
    ]

    def _run_one_inner(yaml_cfg: dict) -> float:
        tmp = Path(tempfile.gettempdir()) / f"sens_{uuid.uuid4().hex}.yaml"
        tmp.write_text(yaml.safe_dump(yaml_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        try:
            res = backtester.run(data_dict, str(tmp), funding_rate_df=fr_df)
            return float(res.metrics.get("sharpe_ratio", 0.0))
        except Exception as e:
            logging.warning("敏感度子任务失败：%s", e)
            return float("nan")
        finally:
            tmp.unlink(missing_ok=True)

    payload_grids = []
    total_runs = sum(len(g[1][2]) * len(g[2][2]) for g in GRIDS)
    cnt = 0
    for name, (p1_path, p1_label, p1_vals), (p2_path, p2_label, p2_vals) in GRIDS:
        matrix = [[None] * len(p2_vals) for _ in p1_vals]
        for i, v1 in enumerate(p1_vals):
            for j, v2 in enumerate(p2_vals):
                cnt += 1
                mod = copy.deepcopy(strat_yaml)
                for path, val in BEST.items():
                    set_param(mod, path, val)
                set_param(mod, p1_path, v1)
                set_param(mod, p2_path, v2)
                sharpe = _run_one_inner(mod)
                matrix[i][j] = round(sharpe, 3) if not np.isnan(sharpe) else None
                logging.info("[sens %s %d/%d] %s=%s × %s=%s → sharpe=%.3f",
                             name, cnt, total_runs, p1_label, v1, p2_label, v2,
                             sharpe if not np.isnan(sharpe) else 0.0)
        payload_grids.append({
            "name": name,
            "x_label": p2_label,
            "x_values": p2_vals,
            "y_label": p1_label,
            "y_values": p1_vals,
            "matrix": matrix,
            "best_x": BEST[p2_path],
            "best_y": BEST[p1_path],
        })

    return _write_json(
        DATA_OUT / "sensitivity.json",
        {"metric": "sharpe_ratio", "best_params": BEST, "grids": payload_grids},
    )


def _export_portfolio() -> int:
    candidates = sorted((PROJECT_ROOT / "output").glob("portfolio_*"))
    if not candidates:
        return 0
    src = candidates[-1]
    summary_txt = (src / "summary.txt").read_text(encoding="utf-8") if (src / "summary.txt").exists() else ""
    payload: dict[str, Any] = {"source": src.name, "summary": {}, "sleeves": [], "equity": []}
    for key, pat in [
        ("initial_balance", r"总本金:\s*([0-9.]+)"),
        ("final_equity", r"期末权益:\s*([0-9.]+)"),
        ("total_return_pct", r"总收益率:\s*([0-9\-.]+)%"),
        ("annualized_return_pct", r"年化:\s*([0-9\-.]+)%"),
        ("sharpe_ratio", r"夏普:\s*([0-9\-.]+)"),
        ("max_drawdown_pct", r"最大回撤:\s*([0-9\-.]+)%"),
    ]:
        m = re.search(pat, summary_txt)
        if m:
            payload["summary"][key] = float(m.group(1))
    for line in summary_txt.splitlines():
        m = re.match(
            r"\s*(\w+)\s+alloc=([\d.]+)%\s+start=([\d.]+)\s+→\s+end=([\d.]+)\s+\(\+([\d.]+)%\)\s+sharpe=([\d.\-]+)\s+MDD=([\d.]+)%\s+trades=(\d+)",
            line,
        )
        if m:
            payload["sleeves"].append({
                "symbol": m.group(1),
                "allocation_pct": float(m.group(2)),
                "start": float(m.group(3)),
                "end": float(m.group(4)),
                "return_pct": float(m.group(5)),
                "sharpe_ratio": float(m.group(6)),
                "max_drawdown_pct": float(m.group(7)),
                "total_trades": int(m.group(8)),
            })
    eq_csv = src / "equity.csv"
    if eq_csv.exists():
        last_per_day: dict[str, float] = {}
        with open(eq_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                d = r["timestamp"][:10]
                last_per_day[d] = float(r["portfolio_equity"])
        payload["equity"] = [
            {"t": d, "v": round(v, 2)} for d, v in sorted(last_per_day.items())
        ]
    return _write_json(DATA_OUT / "portfolio.json", payload)


def _export_fear_greed() -> int:
    fng_path = PROJECT_ROOT / "data" / "parquet" / "BTCUSDT" / "fear_greed_index.parquet"
    if not fng_path.exists():
        return 0
    df = pl.read_parquet(fng_path).sort("timestamp").tail(30)
    rows = [
        {
            "t": ts.strftime("%Y-%m-%d"),
            "value": int(v),
            "classification": c,
        }
        for ts, v, c in zip(df["timestamp"].to_list(), df["value"].to_list(), df["classification"].to_list())
    ]
    return _write_json(DATA_OUT / "fear_greed_latest.json", {"rows": rows})


# ---------- 主入口 ----------

def main() -> int:
    parser = argparse.ArgumentParser(description="导出回测结果为 JSON（多策略）")
    parser.add_argument("--skip-sensitivity", action="store_true",
                        help="跳过敏感度热力图（贵，约 100 次回测）")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--only", nargs="+", default=None,
                        help="只跑指定 stem（不带 .yaml 扩展）；调试用")
    args = parser.parse_args()

    _setup_logging()
    log = logging.getLogger("export_results")

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    STRATEGY_DIR.mkdir(parents=True, exist_ok=True)

    bt = Backtester.from_yaml(str(BACKTEST_CFG))

    # 发现所有策略 yaml（排除非策略 yaml）
    yaml_files: list[Path] = []
    for p in sorted((PROJECT_ROOT / "config").glob("strategies*.yaml")):
        if args.only and p.stem not in args.only:
            continue
        yaml_files.append(p)
    log.info("发现策略文件 %d 个：%s", len(yaml_files), [f.name for f in yaml_files])

    # 跑每个策略
    rows: list[dict[str, Any]] = []
    for p in yaml_files:
        summary = _export_one_strategy(p, bt, tuple(args.symbols))
        if summary:
            rows.append(summary)

    if not rows:
        log.error("无成功导出的策略 → 退出")
        return 1

    # 排行榜索引（含综合评分）
    index = _build_index(rows)
    idx_size = _write_json(DATA_OUT / "strategies_index.json", index)
    log.info("strategies_index.json 已写入 (%.1f KB)", idx_size / 1024)

    # 兼容旧前端：以 v2_optimized 结果生成 {symbol}_*.json
    legacy_sizes = _legacy_per_symbol_exports(rows, bt)

    # MC：用 BTC v2_optimized 结果。需要重跑一次拿到 result 对象（json 不能反序列化 Backtester result）
    btc_result = None
    v2_path = PROJECT_ROOT / "config" / DEFAULT_BTC_STRATEGY
    if v2_path.exists():
        with open(v2_path, encoding="utf-8") as f:
            v2_yaml = yaml.safe_load(f) or {}
            v2_yaml["_filename"] = v2_path.name
        out = _run_one("BTCUSDT", v2_yaml, bt)
        if out is not None:
            btc_result = out[0]

    other_sizes: dict[str, int] = {}
    if btc_result is not None:
        other_sizes["monte_carlo.json"] = _export_monte_carlo("BTCUSDT", btc_result)
    other_sizes["walk_forward_summary.json"] = _export_walk_forward()
    other_sizes["leverage_scan.json"] = _export_leverage_scan()
    other_sizes["portfolio.json"] = _export_portfolio()
    other_sizes["fear_greed_latest.json"] = _export_fear_greed()

    if not args.skip_sensitivity and v2_path.exists():
        log.info("===== 敏感度热力图（5×5×4 ≈ 100 次回测，慢）=====")
        with open(v2_path, encoding="utf-8") as f:
            v2_yaml = yaml.safe_load(f) or {}
        other_sizes["sensitivity.json"] = _export_sensitivity(v2_yaml, bt)

    # 打印排行榜 top3
    print("\n========== 综合评分 Top 5 ==========")
    for r in sorted(rows, key=lambda x: x["composite_score"], reverse=True)[:5]:
        print(f"  {r['composite_score']:6.2f}  {r['display_name']:32s}  "
              f"sharpe={r['primary']['sharpe_ratio']:.2f}  "
              f"年化={r['primary']['annualized_return_pct']:.1f}%  "
              f"回撤={r['primary']['max_drawdown_pct']:.1f}%  "
              f"胜率={r['primary']['win_rate_pct']:.1f}%")
    print()

    print("========== JSON 导出汇总 ==========")
    print(f"输出目录: {DATA_OUT}")
    all_sizes = {**legacy_sizes, **other_sizes, "strategies_index.json": idx_size}
    for name, size in sorted(all_sizes.items()):
        kb = size / 1024
        flag = " ⚠ >500KB" if size > 500 * 1024 else ""
        print(f"  {name:40s} {kb:8.1f} KB{flag}")
    print(f"  strategies/ × {len(rows)} files")
    print("=" * 38)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
