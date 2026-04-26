"""一键下载入口：按 config/data_config.yaml 拉取全量 OHLCV + 衍生数据。

用法：
- 全量：`uv run python scripts/download_all.py`
- 冒烟：`uv run python scripts/download_all.py --test`（只下 1d 2024-01..03 + 恐慌指数）

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
    download_funding_rate,
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


def _run_full(cfg: DataConfig, log: logging.Logger) -> int:
    summary: list[str] = []

    # 1. OHLCV 历史 + 当月增量
    for tf in cfg.timeframes:
        log.info("=== OHLCV %s ===", tf)
        try:
            hist = download_history(cfg, tf)
            log.info("历史：下载=%d 跳过=%d 失败=%d",
                     hist["downloaded"], hist["skipped"], hist["failed"])
        except Exception:
            log.exception("OHLCV 历史 %s 整体失败", tf)
            hist = {"downloaded": 0, "skipped": 0, "failed": -1}
        try:
            recent = fetch_recent(cfg, tf)
            log.info("REST 增量：appended=%d", recent.get("appended", 0))
        except Exception:
            log.exception("OHLCV 增量 %s 失败", tf)
            recent = {"appended": 0}
        summary.append(
            f"OHLCV {tf}: 历史下载={hist['downloaded']} 跳过={hist['skipped']} "
            f"失败={hist['failed']} 增量={recent.get('appended', 0)}"
        )

    # 2. 资金费率
    log.info("=== 资金费率 ===")
    try:
        fr = download_funding_rate(cfg)
        summary.append(
            f"资金费率: rows={fr['rows']} 范围={fr.get('first', '-')} → {fr.get('last', '-')}"
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
    args = parser.parse_args()

    _setup_logging()
    log = logging.getLogger("download_all")

    cfg_path = PROJECT_ROOT / "config" / "data_config.yaml"
    cfg = DataConfig.from_yaml(cfg_path)
    log.info(
        "配置：symbol=%s timeframes=%s start=%s data_dir=%s",
        cfg.symbol, cfg.timeframes, cfg.history_start_date, cfg.data_dir,
    )

    if args.test:
        return _run_test(cfg, log)
    return _run_full(cfg, log)


if __name__ == "__main__":
    raise SystemExit(main())
