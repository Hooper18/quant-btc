"""数据管道配置加载（YAML → 类型化 dataclass）。

支持单币种（symbol）+ 多币种（symbols）共存：旧脚本只读 symbol/timeframes 走默认；
新脚本可走 symbols + symbol_overrides 拿到每个币种的独立配置。
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
    # 多币种扩展（Phase: 多币种支持）；旧 YAML 缺省时退化为 [symbol]
    symbols: list[str] = field(default_factory=list)
    symbol_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def symbol_dir(self) -> Path:
        return self.data_dir / self.symbol

    def for_symbol(self, sym: str) -> "DataConfig":
        """返回切换 symbol 的副本；自动应用 symbol_overrides[sym] 中的 timeframes / start。"""
        ov = self.symbol_overrides.get(sym, {})
        return DataConfig(
            symbol=sym,
            timeframes=[str(tf) for tf in ov.get("timeframes", self.timeframes)],
            history_start_date=(
                date.fromisoformat(str(ov["history_start_date"]))
                if "history_start_date" in ov else self.history_start_date
            ),
            data_dir=self.data_dir,
            binance_vision_base_url=self.binance_vision_base_url,
            retry_max=self.retry_max,
            retry_backoff_base=self.retry_backoff_base,
            symbols=self.symbols,
            symbol_overrides=self.symbol_overrides,
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "DataConfig":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        symbol = str(raw["symbol"])
        symbols = [str(s) for s in raw.get("symbols", [symbol])] or [symbol]
        return cls(
            symbol=symbol,
            timeframes=[str(tf) for tf in raw["timeframes"]],
            history_start_date=date.fromisoformat(str(raw["history_start_date"])),
            data_dir=Path(raw["data_dir"]),
            binance_vision_base_url=str(raw["binance_vision_base_url"]).rstrip("/"),
            retry_max=int(raw["retry_max"]),
            retry_backoff_base=int(raw["retry_backoff_base"]),
            symbols=symbols,
            symbol_overrides=dict(raw.get("symbol_overrides", {}) or {}),
        )
