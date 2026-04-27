"""策略组合回测入口。

用法：
    uv run python scripts/run_portfolio.py --config config/portfolio.yaml

输出：output/portfolio_{YYYYmmdd_HHMMSS}/
- summary.txt   组合 + 各 sleeve 摘要
- equity.csv    组合净值曲线（含各 sleeve 列）
- trades_{symbol}.csv  各 sleeve 的交易明细
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from backtest import Backtester, PortfolioBacktester, load_portfolio_yaml  # noqa: E402
from utils.config import DataConfig  # noqa: E402


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="策略组合回测")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "portfolio.yaml"))
    parser.add_argument("--backtest", default=str(PROJECT_ROOT / "config" / "backtest_config.yaml"))
    parser.add_argument("--data-config", default=str(PROJECT_ROOT / "config" / "data_config.yaml"))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    _setup_logging()
    log = logging.getLogger("run_portfolio")

    sleeves, total_balance = load_portfolio_yaml(args.config)
    log.info("组合：%d 个 sleeve，总本金=%.2f", len(sleeves), total_balance)
    for s in sleeves:
        log.info("  - %s @ %.0f%% → %s", s.symbol, s.allocation * 100, s.strategy_path.name)

    data_cfg = DataConfig.from_yaml(args.data_config)
    bt = Backtester.from_yaml(args.backtest)

    pb = PortfolioBacktester(sleeves, data_cfg, bt, total_balance)
    result = pb.run()
    result.print_summary()

    # 输出
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = PROJECT_ROOT / "output" / f"portfolio_{ts_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 组合净值 CSV
    eq_data = {"timestamp": result.timestamps, "portfolio_equity": result.equity_curve}
    pl.DataFrame(eq_data).write_csv(out_dir / "equity.csv")

    # 各 sleeve 交易
    for sl in result.sleeves:
        sl.result.to_csv(out_dir / f"trades_{sl.cfg.symbol}.csv")

    # 摘要 txt
    lines = [
        f"组合回测 - {datetime.now().isoformat()}",
        f"配置: {args.config}",
        f"总本金: {total_balance:.2f}",
        "",
        "===== 组合指标 =====",
        f"期末权益:  {result.metrics['final_equity']:.2f}",
        f"总收益率:  {result.metrics['total_return_pct']:.2f}%",
        f"年化:      {result.metrics['annualized_return_pct']:.2f}%",
        f"夏普:      {result.metrics['sharpe_ratio']:.3f}",
        f"最大回撤:  {result.metrics['max_drawdown_pct']:.2f}%",
        f"对齐 bar:   {result.metrics['aligned_bars']:.0f}",
        "",
        "===== 各 sleeve =====",
    ]
    for sl in result.sleeves:
        m = sl.result.metrics
        lines.append(
            f"{sl.cfg.symbol} alloc={sl.cfg.allocation:.0%} "
            f"start={sl.initial_capital:.2f} → end={m.get('final_equity', 0):.2f} "
            f"({m.get('total_return_pct', 0):+.1f}%) sharpe={m.get('sharpe_ratio', 0):.2f} "
            f"MDD={m.get('max_drawdown_pct', 0):.1f}% trades={int(m.get('total_trades', 0))}"
        )
    lines.append("")
    lines.append("===== 贡献度 =====")
    for sym, pnl in result.contribution.items():
        lines.append(f"{sym}: {pnl:+.2f} USDT")
    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n报告已保存到 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
