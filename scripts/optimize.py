"""策略参数网格优化入口。

用法：
    uv run python scripts/optimize.py --strategy config/strategies.yaml
    uv run python scripts/optimize.py --strategy config/strategies.yaml \
        --metric sharpe --train-ratio 0.7 --top 10

默认网格（180 组合）：
    strategies[0].conditions[0].value: [70, 75, 80, 85, 90]   # RSI 阈值
    strategies[0].action.size_pct:    [5, 10, 15]
    strategies[0].stop_loss_pct:      [2, 3, 5]
    strategies[0].take_profit_pct:    [4, 6, 8, 10]

可通过 --grid-yaml 指向自定义网格 YAML 覆盖。
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from backtest import Backtester, StrategyOptimizer  # noqa: E402
from backtest.optimizer import _LARGER_IS_BETTER, _METRIC_KEYS  # noqa: E402
from indicators import IndicatorEngine  # noqa: E402
from utils.config import DataConfig  # noqa: E402

# 复用 run_backtest 的指标解析 / OHLCV 加载
from run_backtest import _collect_required_indicators, _load_ohlcv  # noqa: E402


DEFAULT_GRID = {
    "strategies[0].conditions[0].value": [70, 75, 80, 85, 90],
    "strategies[0].action.size_pct": [5, 10, 15],
    "strategies[0].stop_loss_pct": [2, 3, 5],
    "strategies[0].take_profit_pct": [4, 6, 8, 10],
}


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,   # 仅打印 WARNING 及以上，避免污染进度
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("optimize").setLevel(logging.INFO)
    logging.getLogger("backtest.optimizer").setLevel(logging.INFO)


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


def _short_key(path: str) -> str:
    """长 dotted-path → 表头短名。"""
    aliases = {
        "strategies[0].conditions[0].value": "RSI阈值",
        "strategies[0].action.size_pct": "size%",
        "strategies[0].stop_loss_pct": "SL%",
        "strategies[0].take_profit_pct": "TP%",
    }
    return aliases.get(path, path.split(".")[-1])


def _print_top(rows: list[dict], param_keys: list[str], metric: str, top_n: int) -> None:
    metric_key = _METRIC_KEYS[metric]
    larger = _LARGER_IS_BETTER[metric]
    train_metric_col = {
        "sharpe": "train_sharpe",
        "return": "train_total_return_pct",
        "annual_return": "train_total_return_pct",  # 没存年化训练值，用总收益代
        "max_drawdown": "train_max_dd_pct",
        "profit_loss_ratio": "train_pl_ratio",
        "win_rate": "train_win_rate_pct",
    }[metric]
    sorted_rows = sorted(rows, key=lambda r: r.get(train_metric_col, 0), reverse=larger)[:top_n]

    short_keys = [_short_key(k) for k in param_keys]
    headers = (
        ["排名"] + short_keys
        + ["训练收益%", "训练夏普", "训练回撤%", "训练胜率%", "训练交易", "训练熔断"]
    )
    widths = [4] + [max(7, len(h)) for h in short_keys] + [10, 9, 10, 10, 8, 6]

    print()
    print("=" * sum(w + 2 for w in widths))
    print(f"Top {top_n} 参数组合（按训练集 {metric} 排序）")
    print("=" * sum(w + 2 for w in widths))
    print("  ".join(f"{h:>{w}s}" for h, w in zip(headers, widths)))
    print("-" * sum(w + 2 for w in widths))
    for rank, r in enumerate(sorted_rows, 1):
        cells = [f"{rank:>4d}"]
        for k, w in zip(param_keys, widths[1:1 + len(param_keys)]):
            v = r[k]
            cells.append(f"{v:>{w}}" if isinstance(v, str) else f"{v:>{w}}")
        cells += [
            f"{r['train_total_return_pct']:>10.2f}",
            f"{r['train_sharpe']:>9.3f}",
            f"{r['train_max_dd_pct']:>10.2f}",
            f"{r['train_win_rate_pct']:>10.2f}",
            f"{r['train_total_trades']:>8d}",
            f"{('是' if r['train_circuit_breaker'] else '否'):>6s}",
        ]
        print("  ".join(cells))
    print("=" * sum(w + 2 for w in widths))


def main() -> int:
    parser = argparse.ArgumentParser(description="策略参数网格优化")
    parser.add_argument("--strategy", required=True, help="基础策略 YAML 路径")
    parser.add_argument("--backtest", default=str(PROJECT_ROOT / "config" / "backtest_config.yaml"))
    parser.add_argument("--data-config", default=str(PROJECT_ROOT / "config" / "data_config.yaml"))
    parser.add_argument(
        "--metric", default="sharpe",
        choices=list(_METRIC_KEYS.keys()),
        help="排序指标（默认 sharpe）",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument(
        "--grid-yaml", default=None,
        help="自定义参数网格 YAML；缺省使用内置默认（RSI×SL×TP×size = 180 组合）",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    _setup_logging()
    log = logging.getLogger("optimize")

    # 读 grid
    if args.grid_yaml:
        with open(args.grid_yaml, encoding="utf-8") as f:
            grid = yaml.safe_load(f)
    else:
        grid = DEFAULT_GRID
    log.info("参数网格：%s", grid)

    # 读策略 → 推算所需 TF / 指标
    with open(args.strategy, encoding="utf-8") as f:
        strat_yaml = yaml.safe_load(f) or {}
    strategies = strat_yaml.get("strategies", [])
    if not strategies:
        log.error("策略为空：%s", args.strategy)
        return 1

    bt_for_tf = Backtester.from_yaml(args.backtest)
    used_tfs = _used_timeframes(strategies, bt_for_tf.primary_tf)
    log.info("涉及 TF：%s", sorted(used_tfs))
    ind_cfg = _collect_required_indicators(strategies)
    log.info("指标：%s", ind_cfg)

    # 加载数据 + 指标
    data_cfg = DataConfig.from_yaml(args.data_config)
    data_dict: dict[str, pl.DataFrame] = {}
    for tf in used_tfs:
        raw = _load_ohlcv(data_cfg, tf)
        if raw is None or raw.height == 0:
            log.error("缺 %s 周期数据", tf)
            return 2
        data_dict[tf] = (
            IndicatorEngine(raw).compute_all(ind_cfg) if ind_cfg else raw
        )
        log.info("%s: %d 行", tf, data_dict[tf].height)

    # 资金费率
    fr_path = data_cfg.symbol_dir / "funding_rate.parquet"
    fr_df: pl.DataFrame | None = None
    if fr_path.exists():
        tmp = pl.read_parquet(fr_path)
        if tmp.height > 0:
            fr_df = tmp

    # 输出目录
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = PROJECT_ROOT / "output" / f"optimize_{ts_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 跑优化
    optimizer = StrategyOptimizer(args.strategy, data_dict, args.backtest)
    result = optimizer.optimize(
        grid,
        metric=args.metric,
        train_ratio=args.train_ratio,
        funding_rate_df=fr_df,
    )

    # 保存完整结果
    if result.rows:
        pl.DataFrame(result.rows).write_csv(out_dir / "results.csv")
    # 保存最优策略
    best_yaml = optimizer.make_strategy_yaml(result.best_params)
    with open(out_dir / "best_strategy.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(best_yaml, f, allow_unicode=True, sort_keys=False)

    # 打印 top10
    _print_top(result.rows, list(grid.keys()), args.metric, args.top)

    # 训练 vs 测试对比 + 过拟合检测
    print("\n========== 最优参数训练集 vs 测试集 ==========")
    print(f"参数: {result.best_params}")
    print(f"{'指标':<18s}{'训练集':>15s}{'测试集':>15s}")
    print("-" * 48)
    cmp = [
        ("总收益率%", "total_return_pct"),
        ("年化收益率%", "annualized_return_pct"),
        ("夏普比率", "sharpe_ratio"),
        ("最大回撤%", "max_drawdown_pct"),
        ("胜率%", "win_rate_pct"),
        ("盈亏比", "profit_loss_ratio"),
        ("交易笔数", "total_trades"),
        ("是否熔断", "circuit_breaker"),
    ]
    for label, k in cmp:
        tv = result.best_train.metrics.get(k, 0)
        ev = result.best_test.metrics.get(k, 0)
        if k in ("total_trades", "circuit_breaker"):
            print(f"{label:<18s}{int(tv):>15d}{int(ev):>15d}")
        else:
            print(f"{label:<18s}{tv:>15.3f}{ev:>15.3f}")
    print("=" * 48)

    if result.overfit:
        print("\n⚠️  过拟合警告：测试集夏普 < 训练集夏普 × 0.5")
    else:
        print("\n✓  泛化检测通过：测试集夏普未跌破训练集夏普 × 0.5")

    print(f"\n结果已保存到 {out_dir}")
    print(f"  - 全部组合: {out_dir / 'results.csv'}")
    print(f"  - 最优策略: {out_dir / 'best_strategy.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
