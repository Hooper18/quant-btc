"""数据合并：把多源时序数据按 timestamp 对齐到 OHLCV bar。

低频数据（日频 FGI、5 分钟 OI / 多空比 / 资金费率）通过 `join_asof(strategy="backward")`
向后填充到每根 K 线 —— 即每个 bar 取"截止该 bar timestamp 时的最新值"。
这样不会偷看未来，且高频 K 线上每根 bar 都拿到一个最新可用的低频字段。

字段重命名约定：
- funding_rate.parquet     → 列 funding_rate            注入为 funding_rate
- open_interest_*.parquet  → 列 open_interest           注入为 open_interest
- fear_greed_index.parquet → 列 value(int) → 重命名为   fear_greed
- long_short_ratio.parquet → 列 long_short_ratio        注入为 ls_ratio
- top_trader_ratio.parquet → 列 long_short_ratio        注入为 tt_ratio
"""
from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)


def _asof_join(left: pl.DataFrame, right: pl.DataFrame, suffix: str = "") -> pl.DataFrame:
    """以 left.timestamp 为基准，从 right 取时间戳 ≤ 当前 bar 的最新一行；
    避免主键冲突时 right 的同名 timestamp 由 join_asof 自动处理（不进入结果）。"""
    return left.join_asof(
        right.sort("timestamp"),
        on="timestamp",
        strategy="backward",
        suffix=suffix,
    )


def merge_market_data(
    ohlcv_df: pl.DataFrame,
    funding_df: pl.DataFrame | None = None,
    oi_df: pl.DataFrame | None = None,
    fgi_df: pl.DataFrame | None = None,
    long_short_df: pl.DataFrame | None = None,
    top_trader_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """把可选的市场数据 forward-fill 对齐到 OHLCV，返回扩展 DataFrame。

    - 入参中 None 的数据源直接跳过（不会注入对应列）
    - 输入 OHLCV 不会被破坏（按 timestamp 排序后链式合并）
    - 输出列固定按入参顺序追加：funding_rate / open_interest /
      fear_greed / ls_ratio / tt_ratio（缺哪个就略哪个）

    用法示例：
        ohlcv = pl.read_parquet("data/parquet/BTCUSDT/1h_2024_06.parquet")
        oi    = pl.read_parquet("data/parquet/BTCUSDT/open_interest_2024_06.parquet")
        fgi   = pl.read_parquet("data/parquet/BTCUSDT/fear_greed_index.parquet")
        merged = merge_market_data(ohlcv, oi_df=oi, fgi_df=fgi)
    """
    out = ohlcv_df.sort("timestamp")

    if funding_df is not None and funding_df.height > 0:
        sub = funding_df.select([
            pl.col("timestamp"),
            pl.col("funding_rate").cast(pl.Float64),
        ])
        out = _asof_join(out, sub)

    if oi_df is not None and oi_df.height > 0:
        sub = oi_df.select([
            pl.col("timestamp"),
            pl.col("open_interest").cast(pl.Float64),
        ])
        out = _asof_join(out, sub)

    if fgi_df is not None and fgi_df.height > 0:
        sub = fgi_df.select([
            pl.col("timestamp"),
            pl.col("value").cast(pl.Float64).alias("fear_greed"),
        ])
        out = _asof_join(out, sub)

    if long_short_df is not None and long_short_df.height > 0:
        sub = long_short_df.select([
            pl.col("timestamp"),
            pl.col("long_short_ratio").cast(pl.Float64).alias("ls_ratio"),
        ])
        out = _asof_join(out, sub)

    if top_trader_df is not None and top_trader_df.height > 0:
        sub = top_trader_df.select([
            pl.col("timestamp"),
            pl.col("long_short_ratio").cast(pl.Float64).alias("tt_ratio"),
        ])
        out = _asof_join(out, sub)

    return out
