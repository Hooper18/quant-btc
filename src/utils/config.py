"""数据管道配置加载（YAML → 类型化 dataclass）。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    symbol: str
    timeframes: list[str]
    history_start_date: date
    data_dir: Path
    binance_vision_base_url: str
    retry_max: int
    retry_backoff_base: int

    @property
    def symbol_dir(self) -> Path:
        return self.data_dir / self.symbol

    @classmethod
    def from_yaml(cls, path: Path | str) -> "DataConfig":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            symbol=str(raw["symbol"]),
            timeframes=[str(tf) for tf in raw["timeframes"]],
            history_start_date=date.fromisoformat(str(raw["history_start_date"])),
            data_dir=Path(raw["data_dir"]),
            binance_vision_base_url=str(raw["binance_vision_base_url"]).rstrip("/"),
            retry_max=int(raw["retry_max"]),
            retry_backoff_base=int(raw["retry_backoff_base"]),
        )
