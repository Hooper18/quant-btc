"""Telegram Bot 通知器（直接调 Bot HTTP API，不依赖第三方 SDK）。

凭证从 .env 读取（TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID）；
未配置时静默降级为日志输出，便于 dry-run 与本地调试。
"""
from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"


class TelegramNotifier:
    """轻量 Telegram 通知客户端。"""

    def __init__(self, bot_token: str | None = None, chat_id: str | None = None):
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        self.enabled = bool(self.bot_token and self.chat_id)
        if not self.enabled:
            logger.warning("Telegram 未配置（TELEGRAM_BOT_TOKEN/CHAT_ID 缺失）→ 通知降级为日志")

    async def send(self, message: str) -> bool:
        """发送消息；返回是否成功。未配置时仅日志，不算失败。"""
        if not self.enabled:
            logger.info("[TG-LOG] %s", message)
            return True
        url = f"{API_BASE}/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                logger.warning("Telegram 发送失败 status=%d body=%s", resp.status_code, resp.text[:200])
                return False
            return True
        except Exception:
            logger.exception("Telegram 发送异常")
            return False
