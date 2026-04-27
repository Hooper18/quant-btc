"""数据 + 回测 + 前端 JSON 一键更新脚本。

用法：
    uv run python scripts/update_and_export.py            # 全量
    uv run python scripts/update_and_export.py --no-push  # 不 git push
    uv run python scripts/update_and_export.py --skip-download  # 跳过下载，只重跑导出

设计为月初 cron 跑一次：
    1. 下载所有币种最新月度 OHLCV + BTC 衍生数据（funding/OI/FNG）
    2. 重跑回测（BTC/ETH/SOL）
    3. 重新导出 JSON 到 web/public/data/
    4. git add web/public/data/ → commit → push（Vercel webhook 自动重新部署）
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _run(cmd: list[str], step: str) -> int:
    log = logging.getLogger("update")
    log.info("▶ %s :: %s", step, " ".join(cmd))
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if proc.returncode != 0:
        log.warning("步骤 %s 退出码=%d", step, proc.returncode)
    return proc.returncode


def _git(args: list[str]) -> int:
    return _run(["git", *args], step=f"git {args[0]}")


def _git_capture(args: list[str]) -> str:
    return subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="数据 + 回测 + 导出 + 推送")
    parser.add_argument("--skip-download", action="store_true", help="跳过 download_all")
    parser.add_argument("--skip-export", action="store_true", help="跳过 export_results")
    parser.add_argument("--skip-sensitivity", action="store_true",
                        help="导出阶段跳过敏感度热力图（更快）")
    parser.add_argument("--no-commit", action="store_true", help="不 git commit")
    parser.add_argument("--no-push", action="store_true", help="不 git push")
    args = parser.parse_args()

    _setup_logging()
    log = logging.getLogger("update")
    started = datetime.now()
    log.info("===== 月度更新启动 %s =====", started.strftime("%Y-%m-%d %H:%M:%S"))

    py = sys.executable

    # 1. 数据下载
    if not args.skip_download:
        rc = _run([py, str(PROJECT_ROOT / "scripts" / "download_all.py")], "download_all")
        if rc != 0:
            log.warning("download_all 失败退出码=%d，仍继续后续步骤（可能是 REST 不可用）", rc)
    else:
        log.info("跳过 download_all（--skip-download）")

    # 2. 重新导出 JSON（内部含 BTC/ETH/SOL 回测）
    if not args.skip_export:
        export_args = [py, str(PROJECT_ROOT / "scripts" / "export_results.py")]
        if args.skip_sensitivity:
            export_args.append("--skip-sensitivity")
        rc = _run(export_args, "export_results")
        if rc != 0:
            log.error("export_results 失败 rc=%d，终止", rc)
            return rc
    else:
        log.info("跳过 export_results（--skip-export）")

    # 3. git add + commit + push
    if args.no_commit:
        log.info("跳过 git commit（--no-commit）")
    else:
        _git(["add", "web/public/data/"])
        status = _git_capture(["status", "--short", "web/public/data/"])
        if not status:
            log.info("web/public/data/ 无变化，无需 commit")
        else:
            msg = f"chore: 月度数据 + 回测结果更新 ({started.strftime('%Y-%m-%d')})"
            rc = _git(["commit", "-m", msg])
            if rc == 0 and not args.no_push:
                _git(["push", "origin", "main"])
            elif args.no_push:
                log.info("跳过 git push（--no-push）")

    elapsed = (datetime.now() - started).total_seconds()
    log.info("===== 完成，用时 %.1fs =====", elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
