"""实时模拟交易入口：WebSocket 数据流 + PaperTrader + Telegram 通知。

用法：
    uv run python scripts/run_paper.py --strategy config/strategies_v2_optimized.yaml
    uv run python scripts/run_paper.py --strategy config/strategies_v2_optimized.yaml --dry-run
    uv run python scripts/run_paper.py --duration 30   # 仅运行 30 秒（冒烟）

启动流程：
1. 读 backtest_config.yaml 取手续费/杠杆/滑点
2. 解析策略 YAML 取所需指标 + 周期
3. 用本地 parquet warmup 指标历史
4. 从 paper_state.json 恢复（若存在）
5. 连接 Binance Spot WS（btcusdt@kline_1m / 1h / 4h）
6. 主循环：bar 闭合 → PaperTrader.on_bar_closed → 信号→交易→通知

退出：Ctrl+C / SIGTERM 优雅关闭，保存状态、发送 "系统已停止" 通知。
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 尝试加载 .env（无 python-dotenv 时退化为手动 parse）
def _load_env() -> None:
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    import os
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_env()

from live import DataFeed, PaperTrader, TelegramNotifier  # noqa: E402
from live.paper_trader import parse_strategy_indicators  # noqa: E402
from utils.config import DataConfig  # noqa: E402


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


async def _run(args: argparse.Namespace) -> int:
    log = logging.getLogger("run_paper")

    # 回测/数据配置共用
    bt_cfg = yaml.safe_load(Path(args.backtest).read_text(encoding="utf-8")) or {}
    data_cfg = DataConfig.from_yaml(args.data_config)
    ind_cfg, used_tfs, primary_tf = parse_strategy_indicators(args.strategy)
    log.info("策略 → 指标=%s 周期=%s 主=%s", ind_cfg, sorted(used_tfs), primary_tf)

    notifier = TelegramNotifier()

    trader = PaperTrader(
        initial_balance=float(bt_cfg["initial_balance"]),
        leverage=float(bt_cfg["leverage"]),
        fee_rate=float(bt_cfg["fee_rate"]),
        slippage=float(bt_cfg["slippage"]),
        strategy_path=args.strategy,
        primary_tf=primary_tf,
        used_tfs=tuple(sorted(used_tfs)),
        ind_cfg=ind_cfg,
        maintenance_margin_rate=float(bt_cfg["maintenance_margin_rate"]),
        trades_path=PROJECT_ROOT / "data" / "paper_trades.json",
        state_path=PROJECT_ROOT / "data" / "paper_state.json",
        notifier=notifier,
        dry_run=args.dry_run,
    )

    trader.warmup_from_parquet(data_cfg.symbol_dir)
    trader.restore_state()

    feed = DataFeed(symbol="btcusdt", timeframes=tuple(sorted(used_tfs | {"1m"})))
    feed.on_kline(trader.on_bar_closed)

    if not args.dry_run:
        await notifier.send(
            f"🟢 *Paper Trader 启动*\n"
            f"策略 `{Path(args.strategy).name}`\n"
            f"余额 `{trader.state.balance:.2f}` 杠杆 `{trader.leverage}x`\n"
            f"持仓 `{trader.state.position.side}`"
        )

    # 优雅退出：捕获 SIGINT/SIGTERM
    stop_event = asyncio.Event()

    def _stop(*_: object) -> None:
        log.info("收到停止信号")
        stop_event.set()
        feed.stop()

    if sys.platform != "win32":
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _stop)

    feed_task = asyncio.create_task(feed.connect())
    try:
        if args.duration > 0:
            await asyncio.sleep(args.duration)
            log.info("达到 --duration=%ds，准备退出", args.duration)
            feed.stop()
        await feed_task
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("被用户中断")
        feed.stop()
        try:
            await feed_task
        except (asyncio.CancelledError, Exception):
            pass

    # 收尾
    last_price = None
    cache = feed.cache(primary_tf)
    if cache:
        last_price = cache[-1]["close"]
    await trader.shutdown(last_price=last_price if not args.dry_run else None)
    if not args.dry_run:
        await notifier.send(
            f"🔴 *Paper Trader 已停止*\n"
            f"最终余额 `{trader.state.balance:.2f}`\n"
            f"累计已实现 `{trader.state.realized_pnl_total:+.2f}`\n"
            f"成交笔数 `{trader.state.trade_count}`"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="实时模拟交易（Paper Trading）")
    parser.add_argument("--strategy", default=str(PROJECT_ROOT / "config" / "strategies_v2_optimized.yaml"))
    parser.add_argument("--backtest", default=str(PROJECT_ROOT / "config" / "backtest_config.yaml"))
    parser.add_argument("--data-config", default=str(PROJECT_ROOT / "config" / "data_config.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="只接收数据/计算信号，不下单/不发通知")
    parser.add_argument("--duration", type=int, default=0, help="运行 N 秒后自动退出（0=不限）")
    args = parser.parse_args()

    _setup_logging()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
