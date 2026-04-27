"""把回测产物导出为前端可用的 JSON。

所有 JSON 写入 web/public/data/，每个文件 < 500KB（曲线按天降采样，蒙特卡洛只存
百分位）。本脚本可独立跑一次，也是 update_and_export.py 的核心步骤。

输出（路径相对项目根 web/public/data/）：
    {symbol}_equity_curve.json     净值曲线（日采样，含价格双 Y 轴）
    {symbol}_metrics.json          关键指标
    {symbol}_trades.json           最近 100 笔交易
    {symbol}_monthly_returns.json  月度收益率矩阵
    walk_forward_summary.json      WF 20 窗口
    leverage_scan.json             杠杆 × 破产概率
    monte_carlo.json               MC 百分位
    sensitivity.json               敏感度热力图（RSI×SL / RSI×TP / SL×TP，小网格）
    portfolio.json                 组合回测（如已存在 portfolio 输出）
    fear_greed_latest.json         最近 30 天恐慌指数
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
STRATEGY_PATH = PROJECT_ROOT / "config" / "strategies_v2_optimized.yaml"
BACKTEST_CFG = PROJECT_ROOT / "config" / "backtest_config.yaml"
DATA_CFG = PROJECT_ROOT / "config" / "data_config.yaml"


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
    size = path.stat().st_size
    return size


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


# ---------- 单币种回测 + 导出 ----------

def _run_symbol_backtest(
    symbol: str, strat_yaml: dict, backtester: Backtester
) -> tuple[Any, pl.DataFrame] | None:
    data_cfg = DataConfig.from_yaml(DATA_CFG).for_symbol(symbol)
    if not _has_data(symbol, backtester.primary_tf):
        logging.warning("跳过 %s：缺 %s parquet", symbol, backtester.primary_tf)
        return None

    used_tfs = _used_timeframes(strat_yaml["strategies"], backtester.primary_tf)
    ind_cfg = _collect_required_indicators(strat_yaml["strategies"])

    # ETH/SOL 没有 funding/OI/FNG 辅助数据 — 静默缺失
    aux = load_aux_data(data_cfg)

    try:
        data_dict = build_data_dict(data_cfg, used_tfs, ind_cfg, aux=aux)
    except FileNotFoundError as e:
        logging.warning("跳过 %s：%s", symbol, e)
        return None

    tmp = Path(tempfile.gettempdir()) / f"strat_{uuid.uuid4().hex}.yaml"
    tmp.write_text(yaml.safe_dump(strat_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    try:
        result = backtester.run(data_dict, str(tmp), funding_rate_df=aux.get("funding"))
    finally:
        tmp.unlink(missing_ok=True)

    return result, data_dict[backtester.primary_tf]


def _export_equity_curve(symbol: str, result: Any, primary_df: pl.DataFrame) -> int:
    """导出按日采样的净值 + BTC 价格序列。"""
    timestamps = result.timestamps
    equity = result.equity_curve
    if not timestamps:
        return 0

    # 价格序列：从主 TF dataframe 抽 close
    price_lookup: dict[datetime, float] = {}
    for ts, close in zip(primary_df["timestamp"].to_list(), primary_df["close"].to_list()):
        price_lookup[ts] = float(close)

    # 按 UTC 日聚合（保留每日最后一个点）
    daily: dict[date, tuple[datetime, float, float]] = {}
    for ts, eq in zip(timestamps, equity):
        d = ts.date()
        price = price_lookup.get(ts, float("nan"))
        daily[d] = (ts, float(eq), price)

    pts = [
        {
            "t": ts.strftime("%Y-%m-%d"),
            "nav": round(eq, 2),
            "price": None if np.isnan(price) else round(price, 2),
        }
        for d in sorted(daily)
        for ts, eq, price in [daily[d]]
    ]
    return _write_json(DATA_OUT / f"{symbol}_equity_curve.json", {"symbol": symbol, "points": pts})


def _export_metrics(symbol: str, result: Any) -> int:
    m = result.metrics
    payload = {
        "symbol": symbol,
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
    return _write_json(DATA_OUT / f"{symbol}_metrics.json", payload)


def _export_trades(symbol: str, result: Any, limit: int = 100) -> int:
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
    return _write_json(
        DATA_OUT / f"{symbol}_trades.json",
        {"symbol": symbol, "total_closed": len(closed), "shown": len(rows), "rows": rows},
    )


def _export_monthly_returns(symbol: str, result: Any) -> int:
    """从 equity 曲线抽月初/月末点，算月度收益率，导出年×月矩阵。"""
    if not result.timestamps:
        return 0

    # 按 (year, month) 取每月最后一点
    month_last: dict[tuple[int, int], float] = {}
    for ts, eq in zip(result.timestamps, result.equity_curve):
        month_last[(ts.year, ts.month)] = float(eq)
    if not month_last:
        return 0

    # 按时间排序
    keys = sorted(month_last)
    initial = result.metrics.get("initial_balance", 100.0)
    prev = float(initial)
    rows: dict[int, dict[int, float]] = {}
    for (y, m) in keys:
        cur = month_last[(y, m)]
        ret = (cur - prev) / prev * 100.0 if prev > 0 else 0.0
        rows.setdefault(y, {})[m] = round(ret, 2)
        prev = cur

    matrix = [
        {"year": y, "months": [rows.get(y, {}).get(mm) for mm in range(1, 13)]}
        for y in sorted(rows)
    ]
    return _write_json(DATA_OUT / f"{symbol}_monthly_returns.json", {"symbol": symbol, "matrix": matrix})


def _export_monte_carlo(symbol: str, result: Any) -> int:
    """对 BTC 主回测做 1000 次 bootstrap MC，导出百分位。"""
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
        # 直方图分桶（避免存全部 1000 条）
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


# ---------- 外部产物聚合 ----------

def _export_walk_forward() -> int:
    """从 output/walk_forward_*_opt/ 读 windows.csv 与 report.txt 汇总。"""
    candidates = sorted((PROJECT_ROOT / "output").glob("walk_forward_*_opt"))
    if not candidates:
        candidates = sorted((PROJECT_ROOT / "output").glob("walk_forward_*"))
    if not candidates:
        logging.warning("跳过 walk_forward：找不到 output/walk_forward_*")
        return 0

    src = candidates[-1]
    csv_path = src / "windows.csv"
    if not csv_path.exists():
        logging.warning("跳过 walk_forward：%s 不存在", csv_path)
        return 0

    windows: list[dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
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

    # 从 report.txt 抽汇总
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
        logging.warning("跳过 leverage_scan：%s 不存在", csv_path)
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

    # 从 report.txt 抽建议
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
    """对 BTC 跑小尺寸敏感度热力图（4-5×4-5 格）。"""
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

    def _run_one(yaml_cfg: dict) -> float:
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
                sharpe = _run_one(mod)
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
        logging.warning("跳过 portfolio：找不到 output/portfolio_*")
        return 0
    src = candidates[-1]
    summary_txt = (src / "summary.txt").read_text(encoding="utf-8") if (src / "summary.txt").exists() else ""

    payload: dict[str, Any] = {"source": src.name, "summary": {}, "sleeves": [], "equity": []}

    # 抽组合指标
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

    # 抽各 sleeve
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

    # equity 曲线（按天采样）
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
        logging.warning("跳过 fear_greed：%s 不存在", fng_path)
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
    parser = argparse.ArgumentParser(description="导出回测结果为 JSON")
    parser.add_argument("--skip-sensitivity", action="store_true",
                        help="跳过敏感度热力图（贵，约 100 次回测）")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    args = parser.parse_args()

    _setup_logging()
    log = logging.getLogger("export_results")

    DATA_OUT.mkdir(parents=True, exist_ok=True)

    with open(STRATEGY_PATH, encoding="utf-8") as f:
        strat_yaml = yaml.safe_load(f) or {}

    bt = Backtester.from_yaml(str(BACKTEST_CFG))

    sizes: dict[str, int] = {}
    btc_result = None

    for symbol in args.symbols:
        log.info("===== 跑 %s 回测 =====", symbol)
        out = _run_symbol_backtest(symbol, strat_yaml, bt)
        if out is None:
            continue
        result, primary_df = out
        log.info("%s: 总收益=%.2f%% 夏普=%.2f 交易=%d",
                 symbol, result.metrics.get("total_return_pct", 0),
                 result.metrics.get("sharpe_ratio", 0),
                 int(result.metrics.get("total_trades", 0)))

        sizes[f"{symbol}_equity_curve.json"] = _export_equity_curve(symbol, result, primary_df)
        sizes[f"{symbol}_metrics.json"] = _export_metrics(symbol, result)
        sizes[f"{symbol}_trades.json"] = _export_trades(symbol, result)
        sizes[f"{symbol}_monthly_returns.json"] = _export_monthly_returns(symbol, result)

        if symbol == "BTCUSDT":
            btc_result = result

    # BTC 衍生：Monte Carlo
    if btc_result is not None:
        sizes["monte_carlo.json"] = _export_monte_carlo("BTCUSDT", btc_result)

    # 外部产物聚合
    sizes["walk_forward_summary.json"] = _export_walk_forward()
    sizes["leverage_scan.json"] = _export_leverage_scan()
    sizes["portfolio.json"] = _export_portfolio()
    sizes["fear_greed_latest.json"] = _export_fear_greed()

    if not args.skip_sensitivity:
        log.info("===== 敏感度热力图（5×5×4 ≈ 100 次回测，慢）=====")
        sizes["sensitivity.json"] = _export_sensitivity(strat_yaml, bt)

    print("\n========== JSON 导出汇总 ==========")
    print(f"输出目录: {DATA_OUT}")
    for name, size in sorted(sizes.items()):
        kb = size / 1024
        flag = " ⚠ >500KB" if size > 500 * 1024 else ""
        print(f"  {name:40s} {kb:8.1f} KB{flag}")
    print("=" * 38)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
