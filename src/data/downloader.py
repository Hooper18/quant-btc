"""OHLCV 历史 + 当月增量下载。

数据源：
- 历史完整月：data.binance.vision 月度 ZIP（公开镜像，无频率限制）
- 当月增量：Binance Futures REST `/fapi/v1/klines`

文件命名（写入 cfg.symbol_dir 下）：
- `{timeframe}_{YYYY}_{MM}.parquet`：历史完整月，已存在则幂等跳过
- `{timeframe}_current.parquet`：REST 拉取的尾部增量，每次合并覆盖

下游分析时 glob `{timeframe}_*.parquet` 全部读入 + 按 timestamp 去重即可。

统一 schema：timestamp(Datetime ms, UTC), open/high/low/close/volume(Float64)。
"""
from __future__ import annotations

import io
import logging
import re
import time
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import polars as pl
from tqdm import tqdm

from utils.config import DataConfig

logger = logging.getLogger(__name__)

BINANCE_FAPI_BASE = "https://fapi.binance.com"

_VISION_KLINES_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
# 第 10 列 taker_buy_volume = 主动买入成交量（市价单成交时买方为 taker），
# 用于推导主动买入占比指标（taker_buy_volume / volume）
_OHLCV_PROJECT = ["open_time", "open", "high", "low", "close", "volume", "taker_buy_volume"]


def _month_iter(start: date, end_inclusive: date):
    y, m = start.year, start.month
    while (y, m) <= (end_inclusive.year, end_inclusive.month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def _last_complete_month(today: date) -> tuple[int, int]:
    if today.month == 1:
        return today.year - 1, 12
    return today.year, today.month - 1


def _vision_url(cfg: DataConfig, timeframe: str, year: int, month: int) -> str:
    return (
        f"{cfg.binance_vision_base_url}/data/futures/um/monthly/klines/"
        f"{cfg.symbol}/{timeframe}/{cfg.symbol}-{timeframe}-{year:04d}-{month:02d}.zip"
    )


def _monthly_path(cfg: DataConfig, timeframe: str, year: int, month: int) -> Path:
    return cfg.symbol_dir / f"{timeframe}_{year:04d}_{month:02d}.parquet"


def _current_path(cfg: DataConfig, timeframe: str) -> Path:
    return cfg.symbol_dir / f"{timeframe}_current.parquet"


def _download_with_retry(client: httpx.Client, url: str, cfg: DataConfig) -> bytes | None:
    """下载带指数退避；404 视为"该月不存在"返回 None；其他错误重试到上限。"""
    for attempt in range(cfg.retry_max):
        try:
            resp = client.get(url, timeout=60.0)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.content
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            if attempt < cfg.retry_max - 1:
                wait = cfg.retry_backoff_base ** attempt
                logger.warning("下载失败 %s 重试 %d/%d，%ds 后再试", e, attempt + 1, cfg.retry_max, wait)
                time.sleep(wait)
            else:
                logger.error("下载彻底失败 %s — %s", url, e)
    return None


def _cast_ohlcv(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(
        pl.col("open_time")
            .cast(pl.Int64)
            .cast(pl.Datetime("ms"))
            .dt.replace_time_zone("UTC")
            .alias("timestamp"),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
        pl.col("taker_buy_volume").cast(pl.Float64),
    )


def _parse_vision_zip(zip_bytes: bytes) -> pl.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        if not names:
            raise ValueError("ZIP 空")
        with zf.open(names[0]) as f:
            csv_bytes = f.read()
    first = csv_bytes.split(b"\n", 1)[0]
    has_header = first.startswith(b"open_time")
    if has_header:
        df = pl.read_csv(io.BytesIO(csv_bytes), has_header=True)
    else:
        df = pl.read_csv(io.BytesIO(csv_bytes), has_header=False, new_columns=_VISION_KLINES_COLS)
    return _cast_ohlcv(df.select(_OHLCV_PROJECT))


def download_history(
    cfg: DataConfig,
    timeframe: str,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[str, Any]:
    """下载 [start_date 或 cfg.history_start_date, 上月底] 区间所有月度 OHLCV。

    幂等：已存在的 monthly parquet 跳过。失败月份记录到 failed_list 但不中断后续。
    `start_date` / `end_date` 为可选关键字参数，默认走 cfg 与"今天上月"。
    """
    today = end_date or datetime.now(timezone.utc).date()
    last_y, last_m = _last_complete_month(today)
    start = start_date or cfg.history_start_date
    months = list(_month_iter(start, date(last_y, last_m, 1)))

    cfg.symbol_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {"downloaded": 0, "skipped": 0, "failed": 0, "failed_list": []}

    with httpx.Client(follow_redirects=True) as client:
        for y, m in tqdm(months, desc=f"OHLCV {timeframe}", unit="月"):
            out_path = _monthly_path(cfg, timeframe, y, m)
            if out_path.exists():
                stats["skipped"] += 1
                continue
            zip_bytes = _download_with_retry(client, _vision_url(cfg, timeframe, y, m), cfg)
            if zip_bytes is None:
                logger.warning("跳过 %s %04d-%02d（不存在或下载失败）", timeframe, y, m)
                stats["failed"] += 1
                stats["failed_list"].append((timeframe, y, m))
                continue
            try:
                df = _parse_vision_zip(zip_bytes)
                df.write_parquet(out_path)
                stats["downloaded"] += 1
            except Exception as e:
                logger.error("解析/写入失败 %s %04d-%02d：%s", timeframe, y, m, e)
                stats["failed"] += 1
                stats["failed_list"].append((timeframe, y, m))
    return stats


def _max_local_ts_ms(cfg: DataConfig, timeframe: str) -> int | None:
    """扫描本地 monthly + current 文件，取最大 timestamp（ms）。"""
    if not cfg.symbol_dir.exists():
        return None
    monthly_re = re.compile(rf"^{re.escape(timeframe)}_\d{{4}}_\d{{2}}\.parquet$")
    current_name = f"{timeframe}_current.parquet"
    monthlies = sorted(
        f for f in cfg.symbol_dir.glob(f"{timeframe}_*.parquet")
        if monthly_re.match(f.name)
    )
    candidates: list[Path] = []
    if monthlies:
        candidates.append(monthlies[-1])
    current_path = cfg.symbol_dir / current_name
    if current_path.exists():
        candidates.append(current_path)
    if not candidates:
        return None
    max_ms = 0
    for f in candidates:
        ts = pl.read_parquet(f, columns=["timestamp"]).select(pl.col("timestamp").max()).item()
        if ts is not None:
            max_ms = max(max_ms, int(ts.timestamp() * 1000))
    return max_ms or None


def fetch_recent(cfg: DataConfig, timeframe: str, batch_limit: int = 1500) -> dict[str, Any]:
    """从本地最新 ts 之后用 REST 补齐到当前，合并写入 `{timeframe}_current.parquet`。"""
    cfg.symbol_dir.mkdir(parents=True, exist_ok=True)

    last_ms = _max_local_ts_ms(cfg, timeframe)
    if last_ms is None:
        last_ms = int(
            datetime.combine(cfg.history_start_date, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000
        ) - 1
    cursor = last_ms + 1
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    if cursor >= now_ms:
        logger.info("%s 已是最新（last_ms=%d）", timeframe, last_ms)
        return {"appended": 0, "rows_pulled": 0}

    rows: list[list[Any]] = []
    with httpx.Client(follow_redirects=True) as client:
        with tqdm(desc=f"REST {timeframe}", unit="批") as bar:
            while cursor < now_ms:
                params = {
                    "symbol": cfg.symbol,
                    "interval": timeframe,
                    "startTime": cursor,
                    "limit": batch_limit,
                }
                got: list[Any] | None = None
                for attempt in range(cfg.retry_max):
                    try:
                        resp = client.get(f"{BINANCE_FAPI_BASE}/fapi/v1/klines", params=params, timeout=30.0)
                        resp.raise_for_status()
                        got = resp.json()
                        break
                    except (httpx.HTTPError, httpx.TimeoutException) as e:
                        if attempt < cfg.retry_max - 1:
                            wait = cfg.retry_backoff_base ** attempt
                            logger.warning("REST 失败 %s 重试 %d/%d", e, attempt + 1, cfg.retry_max)
                            time.sleep(wait)
                        else:
                            logger.error("REST 拉取彻底失败 %s @ %d", timeframe, cursor)
                if not got:
                    break
                rows.extend(got)
                cursor = int(got[-1][0]) + 1
                bar.update(1)
                if len(got) < batch_limit:
                    break

    if not rows:
        return {"appended": 0, "rows_pulled": 0}

    raw = pl.DataFrame({
        "open_time": [int(r[0]) for r in rows],
        "open": [r[1] for r in rows],
        "high": [r[2] for r in rows],
        "low": [r[3] for r in rows],
        "close": [r[4] for r in rows],
        "volume": [r[5] for r in rows],
        # Binance fapi REST klines 第 10 列（index 9）= taker_buy_base_asset_volume
        "taker_buy_volume": [r[9] for r in rows],
    })
    new_df = _cast_ohlcv(raw).unique(subset=["timestamp"]).sort("timestamp")

    out_path = _current_path(cfg, timeframe)
    if out_path.exists():
        existing = pl.read_parquet(out_path)
        merged = (
            pl.concat([existing, new_df])
            .unique(subset=["timestamp"], keep="last")
            .sort("timestamp")
        )
        appended = merged.height - existing.height
        merged.write_parquet(out_path)
    else:
        appended = new_df.height
        new_df.write_parquet(out_path)

    return {"appended": appended, "rows_pulled": len(rows)}
