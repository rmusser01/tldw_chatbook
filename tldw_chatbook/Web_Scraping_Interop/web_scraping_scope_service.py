"""Source-aware routing for server-owned web-scraping management."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any


class WebScrapingBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "media.web_scraping.remote_only.local",
        "source": "local",
        "supported": False,
        "reason_code": "remote_only_contract",
        "user_message": "Server web-scraping management is unavailable in local/offline mode.",
        "affected_action_ids": [
            "media.web_scraping.status.server",
            "media.web_scraping.detail.server",
            "media.web_scraping.cancel.server",
            "media.web_scraping.observe.server",
            "media.web_scraping.cookies.detail.server",
            "media.web_scraping.cookies.update.server",
            "media.web_scraping.inspect.server",
        ],
    }
]

_SERVER_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "media.web_scraping.service_controls.server",
        "source": "server",
        "supported": False,
        "reason_code": "deferred_server_process_control",
        "user_message": "Server web-scraping initialize/shutdown controls are intentionally not exposed through Chatbook.",
        "affected_action_ids": [
            "media.web_scraping.service.initialize.server",
            "media.web_scraping.service.shutdown.server",
        ],
    }
]


class WebScrapingScopeService:
    """Route web-scraping management calls to the active server only."""

    def __init__(self, *, server_service: Any, policy_enforcer: Any | None = None) -> None:
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: WebScrapingBackend | str | None) -> WebScrapingBackend:
        if mode is None:
            return WebScrapingBackend.LOCAL
        if isinstance(mode, WebScrapingBackend):
            return mode
        try:
            return WebScrapingBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid web-scraping backend: {mode}") from exc

    def _require_server(self, mode: WebScrapingBackend | str | None) -> Any:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode != WebScrapingBackend.SERVER:
            raise ValueError("Server web-scraping management is server-only; switch to server mode to use it.")
        if self.server_service is None:
            raise ValueError("Server web-scraping backend is unavailable.")
        return self.server_service

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def list_unsupported_capabilities(
        self,
        *,
        mode: WebScrapingBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == WebScrapingBackend.LOCAL:
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        return [dict(item) for item in _SERVER_UNSUPPORTED_CAPABILITIES]

    async def get_status(self, *, mode: WebScrapingBackend | str | None = None) -> dict[str, Any]:
        return await self._maybe_await(self._require_server(mode).get_status())

    async def get_job_status(
        self,
        *,
        mode: WebScrapingBackend | str | None = None,
        job_id: str,
    ) -> dict[str, Any]:
        return await self._maybe_await(self._require_server(mode).get_job_status(job_id))

    async def cancel_job(
        self,
        *,
        mode: WebScrapingBackend | str | None = None,
        job_id: str,
    ) -> dict[str, Any]:
        return await self._maybe_await(self._require_server(mode).cancel_job(job_id))

    async def get_progress(
        self,
        *,
        mode: WebScrapingBackend | str | None = None,
        task_id: str,
    ) -> dict[str, Any]:
        return await self._maybe_await(self._require_server(mode).get_progress(task_id))

    async def get_cookies(
        self,
        *,
        mode: WebScrapingBackend | str | None = None,
        domain: str,
    ) -> dict[str, Any]:
        return await self._maybe_await(self._require_server(mode).get_cookies(domain))

    async def set_cookies(
        self,
        *,
        mode: WebScrapingBackend | str | None = None,
        domain: str,
        cookies: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return await self._maybe_await(self._require_server(mode).set_cookies(domain, cookies))

    async def check_duplicate(
        self,
        *,
        mode: WebScrapingBackend | str | None = None,
        url: str,
    ) -> dict[str, Any]:
        return await self._maybe_await(self._require_server(mode).check_duplicate(url))
