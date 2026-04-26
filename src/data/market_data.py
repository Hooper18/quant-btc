"""市场情绪/衍生数据下载：资金费率、持仓量(OI)、贪婪恐慌指数(FNG)。"""
from __future__ import annotations

import io
import logging
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
import polars as pl
from tqdm import tqdm

from utils.config import DataConfig

logger = logging.getLogger(__name__)

BINANCE_FAPI_BASE = "https://fapi.binance.com"
ALTERNATIVE_FNG_URL = "https://api.alternative.me/fng/"


def _retry_get(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, Any] | None,
    cfg: DataConfig,
    timeout: float,
) -> httpx.Response | None:
    """通用 GET + 指数退避；404 直接返回响应不重试；其他错误重试到上限。"""
    for attempt in range(cfg.retry_max):
        try:
            resp = client.get(url, params=params, timeout=timeout)
            if resp.status_code == 404:
                return resp
            resp.raise_for_status()
            return resp
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            if attempt < cfg.retry_max - 1:
                wait = cfg.retry_backoff_base ** attempt
                logger.warning("请求失败 %s 重试 %d/%d", e, attempt + 1, cfg.retry_max)
                time.sleep(wait)
            else:
                logger.error("请求彻底失败 %s — %s", url, e)
    return None


def download_funding_rate(cfg: DataConfig) -> dict[str, Any]:
    """全量历史资金费率（REST：fapi.binance.com/fapi/v1/fundingRate，覆盖写入 funding_rate.parquet）。

    注意：fapi.binance.com 在中国大陆 DNS 级被墙；如本机网络无法直连，请改用
    `download_funding_rate_vision`（走 data.binance.vision CDN，国内可访问）。
    """
    out_path = cfg.symbol_dir / "funding_rate.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_ms = int(
        datetime.combine(cfg.history_start_date, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000
    )
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    rows: list[dict[str, Any]] = []
    with httpx.Client(follow_redirects=True) as client:
        cursor = start_ms
        with tqdm(desc="资金费率", unit="批") as bar:
            while cursor < now_ms:
                params: dict[str, Any] = {
                    "symbol": cfg.symbol,
                    "startTime": cursor,
                    "limit": 1000,
                }
                resp = _retry_get(
                    client,
                    f"{BINANCE_FAPI_BASE}/fapi/v1/fundingRate",
                    params=params,
                    cfg=cfg,
                    timeout=30.0,
                )
                if resp is None:
                    logger.error("资金费率 @ cursor=%d 彻底失败，停止", cursor)
                    break
                got = resp.json()
                if not got:
                    break
                rows.extend(got)
                cursor = int(got[-1]["fundingTime"]) + 1
                bar.update(1)
                if len(got) < 1000:
                    break

    if not rows:
        return {"rows": 0, "path": str(out_path)}

    df = (
        pl.DataFrame({
            "timestamp": [int(r["fundingTime"]) for r in rows],
            "funding_rate": [float(r["fundingRate"]) for r in rows],
        })
        .with_columns(
            pl.col("timestamp").cast(pl.Datetime("ms")).dt.replace_time_zone("UTC"),
        )
        .unique(subset=["timestamp"])
        .sort("timestamp")
    )
    df.write_parquet(out_path)
    return {
        "rows": int(df.height),
        "path": str(out_path),
        "first": str(df["timestamp"].min()),
        "last": str(df["timestamp"].max()),
    }


def _parse_funding_zip(zip_bytes: bytes) -> pl.DataFrame:
    """解析 vision 月度 fundingRate ZIP；CSV 列：calc_time(ms), funding_interval_hours, last_funding_rate。"""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        if not names:
            raise ValueError("ZIP 空")
        with zf.open(names[0]) as f:
            csv_bytes = f.read()
    # 历史上 vision 偶尔出过无表头的旧文件，统一兜底
    first = csv_bytes.split(b"\n", 1)[0]
    has_header = first.startswith(b"calc_time")
    if has_header:
        df = pl.read_csv(io.BytesIO(csv_bytes), has_header=True)
    else:
        df = pl.read_csv(
            io.BytesIO(csv_bytes), has_header=False,
            new_columns=["calc_time", "funding_interval_hours", "last_funding_rate"],
        )
    return df.select(
        pl.col("calc_time")
            .cast(pl.Int64)
            .cast(pl.Datetime("ms"))
            .dt.replace_time_zone("UTC")
            .alias("timestamp"),
        pl.col("last_funding_rate").cast(pl.Float64).alias("funding_rate"),
        pl.col("funding_interval_hours").cast(pl.Int32),
    )


def download_funding_rate_vision(cfg: DataConfig) -> dict[str, Any]:
    """从 data.binance.vision 月度 fundingRate ZIP 拉取全量历史，合并写入 funding_rate.parquet。

    与 REST 版输出 schema 兼容（timestamp/funding_rate）；额外保留 funding_interval_hours 列。
    单月失败（404 或解析错）记录到 failed_list 但不中断；其余月份照写。
    """
    out_path = cfg.symbol_dir / "funding_rate.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    today = datetime.now(timezone.utc).date()
    last_y, last_m = (today.year - 1, 12) if today.month == 1 else (today.year, today.month - 1)

    months: list[tuple[int, int]] = []
    y, m = cfg.history_start_date.year, cfg.history_start_date.month
    while (y, m) <= (last_y, last_m):
        months.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1

    parts: list[pl.DataFrame] = []
    failed: list[tuple[int, int]] = []
    with httpx.Client(follow_redirects=True) as client:
        for y, m in tqdm(months, desc="资金费率(vision)", unit="月"):
            url = (
                f"{cfg.binance_vision_base_url}/data/futures/um/monthly/fundingRate/"
                f"{cfg.symbol}/{cfg.symbol}-fundingRate-{y:04d}-{m:02d}.zip"
            )
            resp = _retry_get(client, url, params=None, cfg=cfg, timeout=60.0)
            if resp is None or resp.status_code == 404:
                logger.warning("资金费率 %04d-%02d 不存在或下载失败", y, m)
                failed.append((y, m))
                continue
            try:
                parts.append(_parse_funding_zip(resp.content))
            except Exception as e:
                logger.error("资金费率 %04d-%02d 解析失败：%s", y, m, e)
                failed.append((y, m))

    if not parts:
        return {"rows": 0, "path": str(out_path), "failed": failed}

    merged = pl.concat(parts).unique(subset=["timestamp"]).sort("timestamp")
    merged.write_parquet(out_path)
    return {
        "rows": int(merged.height),
        "path": str(out_path),
        "first": str(merged["timestamp"].min()),
        "last": str(merged["timestamp"].max()),
        "failed": failed,
    }


def _parse_metrics_zip(zip_bytes: bytes) -> pl.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        if not names:
            raise ValueError("ZIP 空")
        with zf.open(names[0]) as f:
            csv_bytes = f.read()
    df = pl.read_csv(
        io.BytesIO(csv_bytes),
        has_header=True,
        schema_overrides={"create_time": pl.Utf8},
    )
    base = df.select(
        pl.col("create_time")
            .str.to_datetime(format="%Y-%m-%d %H:%M:%S", time_unit="ms")
            .dt.replace_time_zone("UTC")
            .alias("timestamp"),
        pl.col("sum_open_interest").cast(pl.Float64).alias("open_interest"),
        pl.col("sum_open_interest_value").cast(pl.Float64).alias("open_interest_value"),
        pl.col("count_long_short_ratio").cast(pl.Float64).alias("_ls_ratio"),
        pl.col("count_toptrader_long_short_ratio").cast(pl.Float64).alias("_tt_count_ratio"),
        pl.col("sum_toptrader_long_short_ratio").cast(pl.Float64).alias("_tt_pos_ratio"),
    )
    # 仅保留 OI 三列（保持向后兼容已有月度 parquet schema）
    return base.select(["timestamp", "open_interest", "open_interest_value"])


def _parse_metrics_zip_full(zip_bytes: bytes) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """解析 vision daily metrics ZIP，一次产出三组对齐 DataFrame：
    - oi: timestamp, open_interest, open_interest_value
    - long_short_ratio: timestamp, long_account, short_account, long_short_ratio
    - top_trader_ratio: timestamp, long_account, short_account, long_short_ratio

    多空比换算：vision 仅给出 ratio，由 ratio 反推占比
        long_short_ratio = N_long / N_short
        long_account     = ratio / (1 + ratio)
        short_account    = 1     / (1 + ratio)
    （long_account + short_account = 1，可直接当作市场情绪比例使用）
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        if not names:
            raise ValueError("ZIP 空")
        with zf.open(names[0]) as f:
            csv_bytes = f.read()
    # 历史上某些天的多空比列为空字符串（如 2022-12 部分日期），需要 strict=False 兜底
    df = pl.read_csv(
        io.BytesIO(csv_bytes),
        has_header=True,
        schema_overrides={
            "create_time": pl.Utf8,
            "count_long_short_ratio": pl.Utf8,
            "count_toptrader_long_short_ratio": pl.Utf8,
            "sum_toptrader_long_short_ratio": pl.Utf8,
        },
    )
    base = df.select(
        pl.col("create_time")
            .str.to_datetime(format="%Y-%m-%d %H:%M:%S", time_unit="ms")
            .dt.replace_time_zone("UTC")
            .alias("timestamp"),
        pl.col("sum_open_interest").cast(pl.Float64).alias("open_interest"),
        pl.col("sum_open_interest_value").cast(pl.Float64).alias("open_interest_value"),
        pl.col("count_long_short_ratio").cast(pl.Float64, strict=False).alias("ls_ratio"),
        pl.col("sum_toptrader_long_short_ratio").cast(pl.Float64, strict=False).alias("tt_pos_ratio"),
    )

    oi = base.select(["timestamp", "open_interest", "open_interest_value"])
    ls = base.select(
        pl.col("timestamp"),
        (pl.col("ls_ratio") / (1 + pl.col("ls_ratio"))).alias("long_account"),
        (1 / (1 + pl.col("ls_ratio"))).alias("short_account"),
        pl.col("ls_ratio").alias("long_short_ratio"),
    ).filter(pl.col("long_short_ratio").is_not_null())
    # 大户多空：用 position（仓位金额）口径，比 count（账户数）更代表"实际下注"
    tt = base.select(
        pl.col("timestamp"),
        (pl.col("tt_pos_ratio") / (1 + pl.col("tt_pos_ratio"))).alias("long_account"),
        (1 / (1 + pl.col("tt_pos_ratio"))).alias("short_account"),
        pl.col("tt_pos_ratio").alias("long_short_ratio"),
    ).filter(pl.col("long_short_ratio").is_not_null())
    return oi, ls, tt


def download_open_interest(cfg: DataConfig) -> dict[str, Any]:
    """从 vision daily metrics 一次性下载并写入三组 parquet：
    - open_interest_{YYYY}_{MM}.parquet（按月分区，沿用原约定）
    - long_short_ratio.parquet（全量单文件）
    - top_trader_ratio.parquet（全量单文件）

    跳过策略：当某月 OI 文件 + 两个全量文件都存在时跳过该月；否则下载并累积。
    """
    cfg.symbol_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).date()
    cur = cfg.history_start_date

    stats: dict[str, Any] = {
        "downloaded_days": 0,
        "skipped_months": 0,
        "failed_days": 0,
        "missing_days": 0,
        "months_written": 0,
        "ls_rows": 0,
        "tt_rows": 0,
    }

    oi_buckets: dict[tuple[int, int], list[pl.DataFrame]] = {}
    ls_parts: list[pl.DataFrame] = []
    tt_parts: list[pl.DataFrame] = []

    ls_path = cfg.symbol_dir / "long_short_ratio.parquet"
    tt_path = cfg.symbol_dir / "top_trader_ratio.parquet"
    aggregates_exist = ls_path.exists() and tt_path.exists()

    def _flush_oi(ym: tuple[int, int]) -> None:
        parts = oi_buckets.pop(ym, None)
        if not parts:
            return
        merged = pl.concat(parts).unique(subset=["timestamp"]).sort("timestamp")
        out_path = cfg.symbol_dir / f"open_interest_{ym[0]:04d}_{ym[1]:02d}.parquet"
        merged.write_parquet(out_path)
        stats["months_written"] += 1

    total_days = max((today - cur).days, 1)
    with httpx.Client(follow_redirects=True) as client:
        with tqdm(desc="持仓量+多空比", unit="天", total=total_days) as bar:
            while cur < today:
                ym = (cur.year, cur.month)
                oi_month_path = cfg.symbol_dir / f"open_interest_{cur.year:04d}_{cur.month:02d}.parquet"
                # 仅当 OI 月文件 + 两个汇总文件都已就绪时整月跳过
                if oi_month_path.exists() and aggregates_exist:
                    if cur.month == 12:
                        nxt = date(cur.year + 1, 1, 1)
                    else:
                        nxt = date(cur.year, cur.month + 1, 1)
                    skip_days = min((nxt - cur).days, (today - cur).days)
                    bar.update(skip_days)
                    cur = nxt
                    stats["skipped_months"] += 1
                    continue

                url = (
                    f"{cfg.binance_vision_base_url}/data/futures/um/daily/metrics/"
                    f"{cfg.symbol}/{cfg.symbol}-metrics-"
                    f"{cur.year:04d}-{cur.month:02d}-{cur.day:02d}.zip"
                )
                resp = _retry_get(client, url, params=None, cfg=cfg, timeout=60.0)
                if resp is None:
                    stats["failed_days"] += 1
                elif resp.status_code == 404:
                    stats["missing_days"] += 1
                else:
                    try:
                        oi_df, ls_df, tt_df = _parse_metrics_zip_full(resp.content)
                        # OI 仍按月聚合；ls/tt 累积成单文件
                        if not oi_month_path.exists():
                            oi_buckets.setdefault(ym, []).append(oi_df)
                        ls_parts.append(ls_df)
                        tt_parts.append(tt_df)
                        stats["downloaded_days"] += 1
                    except Exception as e:
                        logger.error("metrics %s 解析失败：%s", cur, e)
                        stats["failed_days"] += 1

                bar.update(1)
                cur += timedelta(days=1)
                if cur.month != ym[1]:
                    _flush_oi(ym)

    for ym in list(oi_buckets.keys()):
        _flush_oi(ym)

    # 写入两个汇总 parquet（合并已存在的旧文件以避免数据丢失）
    if ls_parts:
        merged_ls = pl.concat(ls_parts).unique(subset=["timestamp"]).sort("timestamp")
        if ls_path.exists():
            old = pl.read_parquet(ls_path)
            merged_ls = pl.concat([old, merged_ls]).unique(subset=["timestamp"]).sort("timestamp")
        merged_ls.write_parquet(ls_path)
        stats["ls_rows"] = int(merged_ls.height)
    elif ls_path.exists():
        stats["ls_rows"] = int(pl.read_parquet(ls_path).height)

    if tt_parts:
        merged_tt = pl.concat(tt_parts).unique(subset=["timestamp"]).sort("timestamp")
        if tt_path.exists():
            old = pl.read_parquet(tt_path)
            merged_tt = pl.concat([old, merged_tt]).unique(subset=["timestamp"]).sort("timestamp")
        merged_tt.write_parquet(tt_path)
        stats["tt_rows"] = int(merged_tt.height)
    elif tt_path.exists():
        stats["tt_rows"] = int(pl.read_parquet(tt_path).height)

    return stats


def download_fear_greed_index(cfg: DataConfig) -> dict[str, Any]:
    """alternative.me 全量历史贪婪恐慌指数（覆盖写入）。"""
    out_path = cfg.symbol_dir / "fear_greed_index.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(follow_redirects=True) as client:
        resp = _retry_get(
            client,
            ALTERNATIVE_FNG_URL,
            params={"limit": 0, "format": "json"},
            cfg=cfg,
            timeout=30.0,
        )
    if resp is None or resp.status_code == 404:
        logger.error("贪婪恐慌指数下载失败")
        return {"rows": 0, "path": str(out_path)}

    data = resp.json().get("data", [])
    if not data:
        return {"rows": 0, "path": str(out_path)}

    df = (
        pl.DataFrame({
            "timestamp": [int(r["timestamp"]) * 1000 for r in data],
            "value": [int(r["value"]) for r in data],
            "classification": [str(r["value_classification"]) for r in data],
        })
        .with_columns(
            pl.col("timestamp").cast(pl.Datetime("ms")).dt.replace_time_zone("UTC"),
        )
        .unique(subset=["timestamp"])
        .sort("timestamp")
    )
    df.write_parquet(out_path)
    return {
        "rows": int(df.height),
        "path": str(out_path),
        "first": str(df["timestamp"].min()),
        "last": str(df["timestamp"].max()),
    }
