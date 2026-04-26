"""数据完整性检查。

扫描 data/parquet/{symbol}/ 下所有 parquet 文件，输出：
- 每个文件：行数 / 文件大小 / 时间范围
- OHLCV 各 timeframe：缺失月份列表（vs 预期 cfg.history_start_date → 上月）
- OI 月度分区：同上
- OHLCV / 资金费率 / FNG 时间戳 gap 检测（>2 个采样间隔的中位数视为 gap）

用法：uv run python scripts/inspect_data.py
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config import DataConfig  # noqa: E402


_HUMAN_TF_TO_HOURS = {
    "1m": 1 / 60, "5m": 5 / 60, "15m": 0.25,
    "1h": 1, "4h": 4, "1d": 24,
}


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:7.1f}{unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f}TB"


def _expected_months(start: date, end: date) -> list[tuple[int, int]]:
    out = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def _last_complete_month(today: date) -> tuple[int, int]:
    if today.month == 1:
        return today.year - 1, 12
    return today.year, today.month - 1


def _scan_files(symbol_dir: Path) -> list[Path]:
    return sorted(symbol_dir.glob("*.parquet"))


def _classify(name: str) -> tuple[str, tuple[int, int] | None]:
    """文件名 → (type, (year, month) or None)。"""
    m = re.match(r"^(1m|5m|15m|1h|4h|1d)_(\d{4})_(\d{2})\.parquet$", name)
    if m:
        return f"ohlcv_{m[1]}", (int(m[2]), int(m[3]))
    m = re.match(r"^(1m|5m|15m|1h|4h|1d)_current\.parquet$", name)
    if m:
        return f"ohlcv_{m[1]}_current", None
    m = re.match(r"^open_interest_(\d{4})_(\d{2})\.parquet$", name)
    if m:
        return "oi", (int(m[1]), int(m[2]))
    if name == "funding_rate.parquet":
        return "funding", None
    if name == "fear_greed_index.parquet":
        return "fng", None
    if name == "long_short_ratio.parquet":
        return "long_short", None
    if name == "top_trader_ratio.parquet":
        return "top_trader", None
    return "other", None


def main() -> int:
    cfg = DataConfig.from_yaml(PROJECT_ROOT / "config" / "data_config.yaml")
    if not cfg.symbol_dir.exists():
        print(f"目录不存在：{cfg.symbol_dir}")
        return 2

    files = _scan_files(cfg.symbol_dir)
    if not files:
        print(f"目录为空：{cfg.symbol_dir}")
        return 1

    print(f"扫描目录：{cfg.symbol_dir}\n")

    # 1) 每个文件的基础信息
    print("=" * 110)
    print(f"{'文件':46s} {'行数':>10s} {'大小':>10s}  {'时间范围':<48s}")
    print("-" * 110)

    by_type: dict[str, list[Path]] = defaultdict(list)
    by_type_months: dict[str, set[tuple[int, int]]] = defaultdict(set)

    for f in files:
        size = f.stat().st_size
        try:
            df = pl.read_parquet(f, columns=["timestamp"])
        except Exception:
            df = pl.read_parquet(f)
        rows = df.height
        if "timestamp" in df.columns and rows > 0:
            tmin = df["timestamp"].min()
            tmax = df["timestamp"].max()
            tr = f"{tmin} → {tmax}"
        else:
            tr = "(无 timestamp)"
        print(f"{f.name:46s} {rows:>10d} {_human_size(size):>10s}  {tr}")
        kind, ym = _classify(f.name)
        by_type[kind].append(f)
        if ym is not None:
            by_type_months[kind].add(ym)

    print("=" * 110)

    # 2) 月度分区的缺失检查
    today = datetime.now(timezone.utc).date()
    last_y, last_m = _last_complete_month(today)
    expected = _expected_months(cfg.history_start_date, date(last_y, last_m, 1))
    expected_set = set(expected)

    print(f"\n预期月份范围：{cfg.history_start_date} → {last_y}-{last_m:02d}（{len(expected)} 个月）")
    print("-" * 60)
    for kind in sorted(by_type_months):
        actual = by_type_months[kind]
        missing = sorted(expected_set - actual)
        extra = sorted(actual - expected_set)
        msg = f"  缺失={len(missing)}"
        if missing:
            sample = ", ".join(f"{y}-{m:02d}" for y, m in missing[:5])
            more = f" …(+{len(missing)-5})" if len(missing) > 5 else ""
            msg += f" [示例: {sample}{more}]"
        if extra:
            msg += f"  额外={len(extra)}"
        print(f"{kind:18s} 实际={len(actual)}{msg}")

    # 3) OHLCV / 资金费率 / FNG 时间戳连续性
    print("\n时间戳连续性（gap = 间隔 > 中位数×2 的相邻点对）：")
    print("-" * 60)
    targets: list[tuple[str, list[Path], float | None]] = []
    for tf in ["1m", "5m", "15m", "1h", "4h", "1d"]:
        ohlcv_files = by_type.get(f"ohlcv_{tf}", []) + by_type.get(f"ohlcv_{tf}_current", [])
        if ohlcv_files:
            targets.append((f"OHLCV {tf}", ohlcv_files, _HUMAN_TF_TO_HOURS[tf] * 3600))
    if "funding" in by_type:
        targets.append(("资金费率", by_type["funding"], 8 * 3600))
    if "fng" in by_type:
        targets.append(("贪婪恐慌", by_type["fng"], 24 * 3600))
    if "long_short" in by_type:
        targets.append(("多空账户比", by_type["long_short"], 5 * 60))
    if "top_trader" in by_type:
        targets.append(("大户多空比", by_type["top_trader"], 5 * 60))

    for label, paths, expected_step_s in targets:
        try:
            df = pl.concat([pl.read_parquet(p, columns=["timestamp"]) for p in paths])
            df = df.unique(subset=["timestamp"]).sort("timestamp")
        except Exception as e:
            print(f"{label:14s} 读取失败：{e}")
            continue
        if df.height < 2:
            print(f"{label:14s} 行数<2，跳过")
            continue
        ts = df["timestamp"].to_list()
        diffs_s = [(ts[i] - ts[i - 1]).total_seconds() for i in range(1, len(ts))]
        median_s = sorted(diffs_s)[len(diffs_s) // 2]
        threshold = max(median_s * 2, expected_step_s * 1.5 if expected_step_s else median_s * 2)
        gaps = [
            (ts[i - 1], ts[i], diffs_s[i - 1])
            for i in range(1, len(ts))
            if diffs_s[i - 1] > threshold
        ]
        head = ", ".join(
            f"{a.isoformat()[:16]} → {b.isoformat()[:16]} ({d/3600:.1f}h)"
            for a, b, d in gaps[:3]
        )
        more = f" …(+{len(gaps)-3})" if len(gaps) > 3 else ""
        print(
            f"{label:14s} rows={df.height:<8d} 中位间隔={median_s/3600:.3f}h "
            f"gap={len(gaps)}{(' [' + head + more + ']') if gaps else ''}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
