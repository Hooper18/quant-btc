"""Binance 现货 WebSocket 实时K线数据流。

国内网络下 fapi.binance.com 被墙，但现货 stream.binance.com 可达；现货价与永续合约价
在主流币上价差极小（< 0.05%），用现货数据驱动模拟期货交易够用。

Combined stream 同时订阅 1m / 1h / 4h，按周期分别维护本地缓存（最多 500 根）。

主要约定：
- 仅在 K 线**闭合**时（kline.x == True）触发回调，未闭合的 tick 仅更新缓存
- 心跳：30s 无任何消息触发重连；指数退避 1s → 2s → 4s → ... → 上限 60s
- 不保存订单状态（PaperTrader 负责），DataFeed 是纯数据通路
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)

WS_BASE = "wss://stream.binance.com:9443/stream"
DEFAULT_SYMBOL = "btcusdt"
DEFAULT_TIMEFRAMES = ("1m", "1h", "4h")
CACHE_SIZE = 500
HEARTBEAT_TIMEOUT = 30.0
BACKOFF_MAX = 60.0


KlineCallback = Callable[[str, dict[str, Any]], Awaitable[None] | None]


def _build_combined_url(symbol: str, timeframes: tuple[str, ...]) -> str:
    streams = "/".join(f"{symbol.lower()}@kline_{tf}" for tf in timeframes)
    return f"{WS_BASE}?streams={streams}"


def _parse_kline(k: dict[str, Any]) -> dict[str, Any]:
    """Binance kline payload → 内部统一格式。"""
    return {
        "open_time": datetime.fromtimestamp(k["t"] / 1000, tz=timezone.utc),
        "close_time": datetime.fromtimestamp(k["T"] / 1000, tz=timezone.utc),
        "open": float(k["o"]),
        "high": float(k["h"]),
        "low": float(k["l"]),
        "close": float(k["c"]),
        "volume": float(k["v"]),
        "taker_buy_volume": float(k["V"]),
        "is_closed": bool(k["x"]),
    }


class DataFeed:
    """Binance 现货实时K线数据流，支持多周期 combined stream + 自动重连。"""

    def __init__(
        self,
        symbol: str = DEFAULT_SYMBOL,
        timeframes: tuple[str, ...] = DEFAULT_TIMEFRAMES,
    ):
        self.symbol = symbol.lower()
        self.timeframes = timeframes
        self._url = _build_combined_url(self.symbol, self.timeframes)
        self._caches: dict[str, deque[dict[str, Any]]] = {
            tf: deque(maxlen=CACHE_SIZE) for tf in timeframes
        }
        self._callbacks: list[KlineCallback] = []
        self._stop_event = asyncio.Event()
        self._last_msg_at: float = 0.0

    # ---------- 注册接口 ----------
    def on_kline(self, callback: KlineCallback) -> None:
        """注册 K 线闭合回调；callback(timeframe, kline_dict) 可同步或 async。"""
        self._callbacks.append(callback)

    def cache(self, timeframe: str) -> list[dict[str, Any]]:
        """返回某周期的本地缓存（最近 N 根，含未闭合的当前根）。"""
        return list(self._caches.get(timeframe, deque()))

    def stop(self) -> None:
        self._stop_event.set()

    # ---------- 主循环 ----------
    async def connect(self) -> None:
        """建立 WebSocket 连接；断线后指数退避自动重连，直到 stop()。"""
        attempt = 0
        while not self._stop_event.is_set():
            try:
                logger.info("WS 连接中：%s", self._url)
                async with websockets.connect(
                    self._url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    logger.info("WS 已连接：%s", ", ".join(self.timeframes))
                    attempt = 0
                    self._last_msg_at = asyncio.get_event_loop().time()
                    await self._consume(ws)
            except (ConnectionClosed, OSError, asyncio.TimeoutError) as e:
                logger.warning("WS 断线：%s", e)
            except Exception:
                logger.exception("WS 主循环异常")

            if self._stop_event.is_set():
                break
            wait = min(2 ** attempt, BACKOFF_MAX)
            attempt += 1
            logger.info("WS %.1fs 后重连（第 %d 次）", wait, attempt)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=wait)
                break
            except asyncio.TimeoutError:
                continue
        logger.info("WS 主循环退出")

    async def _consume(self, ws: Any) -> None:
        """读消息 + 心跳监控；30s 无数据抛超时触发外层重连。"""
        while not self._stop_event.is_set():
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=HEARTBEAT_TIMEOUT)
            except asyncio.TimeoutError:
                logger.warning("WS 心跳超时（%ss 无数据）→ 触发重连", HEARTBEAT_TIMEOUT)
                raise
            self._last_msg_at = asyncio.get_event_loop().time()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("WS 收到非 JSON 数据：%r", raw[:200])
                continue
            await self._handle_message(payload)

    async def _handle_message(self, payload: dict[str, Any]) -> None:
        # combined stream 格式：{"stream": "btcusdt@kline_1h", "data": {...}}
        stream = payload.get("stream", "")
        data = payload.get("data") or payload
        if "k" not in data:
            return
        k = data["k"]
        tf = k.get("i")
        if tf not in self._caches:
            return
        kline = _parse_kline(k)
        cache = self._caches[tf]
        # 同一根 K 线（未闭合）持续更新最后一个；闭合后追加
        if cache and cache[-1]["open_time"] == kline["open_time"]:
            cache[-1] = kline
        else:
            cache.append(kline)
        if not kline["is_closed"]:
            return
        logger.debug("K线闭合 %s @ %s close=%.2f", tf, kline["close_time"], kline["close"])
        for cb in self._callbacks:
            try:
                ret = cb(tf, kline)
                if asyncio.iscoroutine(ret):
                    await ret
            except Exception:
                logger.exception("on_kline 回调异常")
