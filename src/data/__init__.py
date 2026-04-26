"""数据下载与管道。"""
from .data_merger import merge_market_data
from .downloader import download_history, fetch_recent
from .market_data import (
    download_fear_greed_index,
    download_funding_rate,
    download_funding_rate_vision,
    download_open_interest,
)

__all__ = [
    "download_history",
    "fetch_recent",
    "download_funding_rate",
    "download_funding_rate_vision",
    "download_open_interest",
    "download_fear_greed_index",
    "merge_market_data",
]
