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
            return WebScrapingBackend.SERVER
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

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    async def _call_server(
        self,
        *,
        mode: WebScrapingBackend | str | None,
        action_id: str,
        method_name: str,
        args: tuple[Any, ...] = (),
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server(normalized_mode)
        self._enforce_policy(action_id)
        return await self._maybe_await(getattr(service, method_name)(*args))

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
        return await self._call_server(
            mode=mode,
            action_id="media.web_scraping.status.server",
            method_name="get_status",
        )

    async def get_job_status(
        self,
        *,
        mode: WebScrapingBackend | str | None = None,
        job_id: str,
    ) -> dict[str, Any]:
        return await self._call_server(
            mode=mode,
            action_id="media.web_scraping.detail.server",
            method_name="get_job_status",
            args=(job_id,),
        )

    async def cancel_job(
        self,
        *,
        mode: WebScrapingBackend | str | None = None,
        job_id: str,
    ) -> dict[str, Any]:
        return await self._call_server(
            mode=mode,
            action_id="media.web_scraping.cancel.server",
            method_name="cancel_job",
            args=(job_id,),
        )

    async def get_progress(
        self,
        *,
        mode: WebScrapingBackend | str | None = None,
        task_id: str,
    ) -> dict[str, Any]:
        return await self._call_server(
            mode=mode,
            action_id="media.web_scraping.observe.server",
            method_name="get_progress",
            args=(task_id,),
        )

    async def get_cookies(
        self,
        *,
        mode: WebScrapingBackend | str | None = None,
        domain: str,
    ) -> dict[str, Any]:
        return await self._call_server(
            mode=mode,
            action_id="media.web_scraping.cookies.detail.server",
            method_name="get_cookies",
            args=(domain,),
        )

    async def set_cookies(
        self,
        *,
        mode: WebScrapingBackend | str | None = None,
        domain: str,
        cookies: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return await self._call_server(
            mode=mode,
            action_id="media.web_scraping.cookies.update.server",
            method_name="set_cookies",
            args=(domain, cookies),
        )

    async def check_duplicate(
        self,
        *,
        mode: WebScrapingBackend | str | None = None,
        url: str,
    ) -> dict[str, Any]:
        return await self._call_server(
            mode=mode,
            action_id="media.web_scraping.inspect.server",
            method_name="check_duplicate",
            args=(url,),
        )
