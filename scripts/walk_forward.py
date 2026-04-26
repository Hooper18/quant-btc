"""Walk-Forward 验证入口。

用法：
    # 固定参数模式：直接评估策略在滚动样本外的表现
    uv run python scripts/walk_forward.py --strategy config/strategies.yaml

    # 窗口内优化模式：每窗口先 grid search 再用最优参数评估
    uv run python scripts/walk_forward.py --strategy config/strategies.yaml --optimize

可选：
    --train-months 12 --test-months 3 --step-months 3
    --metric sharpe
    --grid-yaml my_grid.yaml   （仅 --optimize 模式生效；缺省用 Phase9 默认网格）

每次运行在 output/walk_forward_{ts}/ 下生成：
    - report.txt        全文报告（同控制台输出）
    - windows.csv       每窗口指标表
    - report.png        拼接净值 + 各窗口测试夏普柱状图
    - best_params.yaml  各窗口最优参数（仅 --optimize）
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path

import polars as pl
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from backtest import Backtester, BacktestVisualizer, WalkForwardValidator  # noqa: E402
from backtest.visualizer import _DARK_RC  # noqa: E402
from indicators import IndicatorEngine  # noqa: E402
from utils.config import DataConfig  # noqa: E402

# 复用 run_backtest 的指标解析 / OHLCV 加载
from run_backtest import _collect_required_indicators, _load_ohlcv  # noqa: E402
# 复用 optimize 的默认网格
from optimize import DEFAULT_GRID  # noqa: E402


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("walk_forward").setLevel(logging.INFO)
    logging.getLogger("backtest.walk_forward").setLevel(logging.INFO)


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


def _save_plot(wf_result, out_path: Path) -> None:
    plt.rcParams.update(_DARK_RC)
    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 1, hspace=0.35, height_ratios=[2.4, 1])
    ax_eq = fig.add_subplot(gs[0, 0])
    ax_sharpe = fig.add_subplot(gs[1, 0])
    # 借用 BacktestVisualizer 的 plot_walk_forward；它是实例方法所以
    # 我们传一个最小占位 result，只用其方法即可（不读 self.result）。
    # 改为直接调用 staticmethod 风格不太自然；这里用 unbound 调用：
    BacktestVisualizer.plot_walk_forward(
        BacktestVisualizer.__new__(BacktestVisualizer),
        ax_eq, ax_sharpe, wf_result,
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-Forward 滚动前向验证")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--backtest", default=str(PROJECT_ROOT / "config" / "backtest_config.yaml"))
    parser.add_argument("--data-config", default=str(PROJECT_ROOT / "config" / "data_config.yaml"))
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--step-months", type=int, default=None)
    parser.add_argument("--optimize", action="store_true",
                        help="启用窗口内参数优化（默认网格 = Phase9 的 DEFAULT_GRID）")
    parser.add_argument("--grid-yaml", default=None,
                        help="自定义参数网格（仅 --optimize 时生效）")
    parser.add_argument("--metric", default="sharpe",
                        choices=["sharpe", "return", "annual_return", "max_drawdown",
                                 "profit_loss_ratio", "win_rate"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    _setup_logging()
    log = logging.getLogger("walk_forward")

    # 1) 读策略 → 推算 TF / 指标
    with open(args.strategy, encoding="utf-8") as f:
        strat_yaml = yaml.safe_load(f) or {}
    strategies = strat_yaml.get("strategies", [])
    if not strategies:
        log.error("策略为空：%s", args.strategy)
        return 1

    bt_for_tf = Backtester.from_yaml(args.backtest)
    used_tfs = _used_timeframes(strategies, bt_for_tf.primary_tf)
    ind_cfg = _collect_required_indicators(strategies)
    log.info("涉及 TF：%s", sorted(used_tfs))
    log.info("指标：%s", ind_cfg)

    # 2) 加载数据 + 计算指标（一次性，全数据）
    data_cfg = DataConfig.from_yaml(args.data_config)
    data_dict: dict[str, pl.DataFrame] = {}
    for tf in used_tfs:
        raw = _load_ohlcv(data_cfg, tf)
        if raw is None or raw.height == 0:
            log.error("缺 %s 周期数据", tf); return 2
        data_dict[tf] = (
            IndicatorEngine(raw).compute_all(ind_cfg) if ind_cfg else raw
        )
        log.info("%s: %d 行 范围 %s → %s",
                 tf, data_dict[tf].height,
                 data_dict[tf]["timestamp"].min(),
                 data_dict[tf]["timestamp"].max())

    # 3) 资金费率
    fr_path = data_cfg.symbol_dir / "funding_rate.parquet"
    fr_df: pl.DataFrame | None = None
    if fr_path.exists():
        tmp = pl.read_parquet(fr_path)
        if tmp.height > 0:
            fr_df = tmp

    # 4) 网格（仅 --optimize 用）
    grid = None
    if args.optimize:
        if args.grid_yaml:
            with open(args.grid_yaml, encoding="utf-8") as f:
                grid = yaml.safe_load(f)
        else:
            grid = DEFAULT_GRID
        log.info("窗口内网格：%s", grid)

    # 5) 输出目录
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "opt" if args.optimize else "fixed"
        out_dir = PROJECT_ROOT / "output" / f"walk_forward_{ts_tag}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 6) 跑 walk-forward
    validator = WalkForwardValidator(args.strategy, data_dict, args.backtest)
    result = validator.run(
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
        param_grid=grid,
        metric=args.metric,
        funding_rate_df=fr_df,
    )

    # 7) 报告
    # 用 StringIO 双写：终端 + report.txt
    buf = StringIO()
    import contextlib
    with contextlib.redirect_stdout(buf):
        result.print_report()
    report_text = buf.getvalue()
    print(report_text)
    (out_dir / "report.txt").write_text(report_text, encoding="utf-8")

    # 8) windows.csv
    rows: list[dict] = []
    for w in result.windows:
        row = {
            "window": w.index,
            "train_start": w.train_start.isoformat(),
            "train_end": w.train_end.isoformat(),
            "test_start": w.test_start.isoformat(),
            "test_end": w.test_end.isoformat(),
            "train_total_return_pct": w.train_metrics.get("total_return_pct", 0),
            "train_sharpe": w.train_metrics.get("sharpe_ratio", 0),
            "train_max_dd_pct": w.train_metrics.get("max_drawdown_pct", 0),
            "test_total_return_pct": w.test_metrics.get("total_return_pct", 0),
            "test_sharpe": w.test_metrics.get("sharpe_ratio", 0),
            "test_max_dd_pct": w.test_metrics.get("max_drawdown_pct", 0),
            "test_win_rate_pct": w.test_metrics.get("win_rate_pct", 0),
            "test_total_trades": int(w.test_metrics.get("total_trades", 0)),
            "test_circuit_breaker": int(w.test_metrics.get("circuit_breaker", 0)),
        }
        if w.best_params:
            for k, v in w.best_params.items():
                row[f"param[{k}]"] = v
        rows.append(row)
    if rows:
        pl.DataFrame(rows).write_csv(out_dir / "windows.csv")

    # 9) best_params.yaml（仅 --optimize）
    if args.optimize and any(w.best_params for w in result.windows):
        bp_dump = [
            {
                "window": w.index,
                "train": f"{w.train_start.date()}→{w.train_end.date()}",
                "test": f"{w.test_start.date()}→{w.test_end.date()}",
                "best_params": w.best_params,
                "test_sharpe": float(w.test_metrics.get("sharpe_ratio", 0)),
                "test_return_pct": float(w.test_metrics.get("total_return_pct", 0)),
            }
            for w in result.windows
        ]
        with open(out_dir / "best_params.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(bp_dump, f, allow_unicode=True, sort_keys=False)

    # 10) 可视化
    try:
        _save_plot(result, out_dir / "report.png")
        print(f"\n图表已保存：{out_dir / 'report.png'}")
    except Exception:
        log.exception("生成图表失败")

    print(f"\n输出目录：{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
