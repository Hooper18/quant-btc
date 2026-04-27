"""一键下载入口：按 config/data_config.yaml 拉取全量 OHLCV + 衍生数据。

用法：
- 全量(BTC + 衍生)：`uv run python scripts/download_all.py`
- 冒烟：`uv run python scripts/download_all.py --test`（只下 1d 2024-01..03 + 恐慌指数）
- 多币种：`uv run python scripts/download_all.py --symbols BTCUSDT,ETHUSDT,SOLUSDT`
- 仅一个币种：`uv run python scripts/download_all.py --symbol ETHUSDT`
- 跳过衍生数据(funding/OI/FNG, 多币种新币常用)：加 `--no-aux`

每个数据源独立 try/except 包裹：单源失败不影响其他源；OHLCV 内部按 timeframe 串行，
单 timeframe 失败也只影响自己。结尾打印各源条数与时间范围摘要。
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data import (  # noqa: E402
    download_fear_greed_index,
    download_funding_rate_vision,
    download_history,
    download_open_interest,
    fetch_recent,
)
from utils.config import DataConfig  # noqa: E402


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_summary(summary: list[str], title: str = "下载摘要") -> None:
    print(f"\n========== {title} ==========")
    for line in summary:
        print(line)
    print("==============================")


def _run_test(cfg: DataConfig, log: logging.Logger) -> int:
    """冒烟测试：只下 1d 2024-01..03 + 恐慌指数。"""
    test_start = date(2024, 1, 1)
    # end_date 在 4 月任意一天 → _last_complete_month 返回 (2024, 3)
    test_end = date(2024, 4, 15)
    summary: list[str] = []

    log.info("=== 冒烟测试模式 ===")
    log.info("=== OHLCV 1d (2024-01..03) ===")
    try:
        hist = download_history(cfg, "1d", start_date=test_start, end_date=test_end)
        log.info("历史：下载=%d 跳过=%d 失败=%d",
                 hist["downloaded"], hist["skipped"], hist["failed"])
        summary.append(
            f"OHLCV 1d (2024-01..03): 下载={hist['downloaded']} "
            f"跳过={hist['skipped']} 失败={hist['failed']}"
        )
    except Exception:
        log.exception("OHLCV 1d 失败")
        summary.append("OHLCV 1d: 失败")

    log.info("=== 贪婪恐慌指数 ===")
    try:
        fng = download_fear_greed_index(cfg)
        summary.append(
            f"贪婪恐慌: rows={fng['rows']} 范围={fng.get('first', '-')} → {fng.get('last', '-')}"
        )
    except Exception:
        log.exception("贪婪恐慌指数失败")
        summary.append("贪婪恐慌: 失败")

    _print_summary(summary, title="冒烟测试摘要")
    return 0


def _run_ohlcv_for_symbol(cfg: DataConfig, log: logging.Logger, summary: list[str]) -> None:
    """对单个 symbol 拉取所有 timeframe 的历史 + 当月增量。"""
    for tf in cfg.timeframes:
        log.info("=== %s OHLCV %s ===", cfg.symbol, tf)
        try:
            hist = download_history(cfg, tf)
            log.info("历史：下载=%d 跳过=%d 失败=%d",
                     hist["downloaded"], hist["skipped"], hist["failed"])
        except Exception:
            log.exception("OHLCV 历史 %s/%s 整体失败", cfg.symbol, tf)
            hist = {"downloaded": 0, "skipped": 0, "failed": -1}
        try:
            recent = fetch_recent(cfg, tf)
            log.info("REST 增量：appended=%d", recent.get("appended", 0))
        except Exception:
            log.exception("OHLCV 增量 %s/%s 失败", cfg.symbol, tf)
            recent = {"appended": 0}
        summary.append(
            f"{cfg.symbol} {tf}: 历史下载={hist['downloaded']} 跳过={hist['skipped']} "
            f"失败={hist['failed']} 增量={recent.get('appended', 0)}"
        )


def _run_full(cfg: DataConfig, log: logging.Logger, *, symbols: list[str], with_aux: bool) -> int:
    summary: list[str] = []

    # 1. OHLCV：对每个 symbol 走自定义配置
    for sym in symbols:
        sym_cfg = cfg.for_symbol(sym)
        log.info(
            "######### %s timeframes=%s start=%s #########",
            sym, sym_cfg.timeframes, sym_cfg.history_start_date,
        )
        _run_ohlcv_for_symbol(sym_cfg, log, summary)

    if not with_aux:
        _print_summary(summary, title="下载摘要(--no-aux)")
        return 0

    # 衍生数据只针对默认 symbol（BTCUSDT，多年覆盖最完整）
    cfg = cfg.for_symbol(cfg.symbol)

    # 2. 资金费率（走 vision CDN，国内可访问）
    log.info("=== 资金费率(vision) ===")
    try:
        fr = download_funding_rate_vision(cfg)
        summary.append(
            f"资金费率: rows={fr['rows']} 范围={fr.get('first', '-')} → {fr.get('last', '-')} "
            f"失败={len(fr.get('failed', []))}月"
        )
    except Exception:
        log.exception("资金费率失败")
        summary.append("资金费率: 失败")

    # 3. 持仓量
    log.info("=== 持仓量 ===")
    try:
        oi = download_open_interest(cfg)
        summary.append(
            f"持仓量: 下载={oi['downloaded_days']}天 跳过={oi['skipped_months']}月 "
            f"写入={oi['months_written']}月 失败={oi['failed_days']}天 缺失={oi['missing_days']}天"
        )
    except Exception:
        log.exception("持仓量失败")
        summary.append("持仓量: 失败")

    # 4. 贪婪恐慌指数
    log.info("=== 贪婪恐慌指数 ===")
    try:
        fng = download_fear_greed_index(cfg)
        summary.append(
            f"贪婪恐慌: rows={fng['rows']} 范围={fng.get('first', '-')} → {fng.get('last', '-')}"
        )
    except Exception:
        log.exception("贪婪恐慌指数失败")
        summary.append("贪婪恐慌: 失败")

    _print_summary(summary, title="下载摘要")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="BTC 量化数据管道下载入口")
    parser.add_argument(
        "--test",
        action="store_true",
        help="冒烟测试：只下 1d 2024-01..03 + 恐慌指数",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="只下单个 symbol（覆盖 config 的 symbols 列表）",
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="覆盖 config.symbols；逗号分隔，如 ETHUSDT,SOLUSDT",
    )
    parser.add_argument(
        "--no-aux",
        action="store_true",
        help="跳过资金费率/OI/FNG（多币种新加币种常用）",
    )
    args = parser.parse_args()

    _setup_logging()
    log = logging.getLogger("download_all")

    cfg_path = PROJECT_ROOT / "config" / "data_config.yaml"
    cfg = DataConfig.from_yaml(cfg_path)

    # 决定本次要下载的 symbols
    if args.symbol:
        symbols = [args.symbol]
    elif args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = cfg.symbols or [cfg.symbol]

    log.info(
        "配置：symbols=%s timeframes(默认)=%s start=%s data_dir=%s",
        symbols, cfg.timeframes, cfg.history_start_date, cfg.data_dir,
    )

    if args.test:
        return _run_test(cfg, log)
    return _run_full(cfg, log, symbols=symbols, with_aux=not args.no_aux)


if __name__ == "__main__":
    raise SystemExit(main())
