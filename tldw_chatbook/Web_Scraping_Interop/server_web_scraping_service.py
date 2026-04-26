"""Policy-gated active-server web-scraping management service."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import TLDWAPIClient


class ServerWebScrapingService:
    """Expose safe server web-scraping management actions through runtime policy."""

    def __init__(
        self,
        client: Optional[TLDWAPIClient],
        *,
        policy_enforcer: Any | None = None,
    ) -> None:
        self.client = client
        self.policy_enforcer = policy_enforcer

    @classmethod
    def from_config(
        cls,
        app_config: Mapping[str, Any],
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerWebScrapingService":
        return cls(
            client=build_runtime_api_client_from_config(app_config),
            policy_enforcer=policy_enforcer,
        )

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server web-scraping operations.")
        return self.client

    def _enforce(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        require_allowed = getattr(self.policy_enforcer, "require_allowed", None)
        require_ui_action_allowed = getattr(self.policy_enforcer, "require_ui_action_allowed", None)
        if callable(require_allowed):
            require_allowed(action_id=action_id)
            return
        if callable(require_ui_action_allowed):
            decision = require_ui_action_allowed(action_id=action_id)
            if decision is not None and getattr(decision, "allowed", True) is False:
                raise PolicyDeniedError(
                    action_id=action_id,
                    reason_code=getattr(decision, "reason_code", None) or "authority_denied",
                    user_message=getattr(decision, "user_message", None)
                    or "Server web-scraping action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @staticmethod
    def _dump(response: Any) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        return dict(response or {})

    @staticmethod
    def _with_record_id(record_id: str, response: Any) -> dict[str, Any]:
        record = ServerWebScrapingService._dump(response)
        record.setdefault("record_id", record_id)
        return record

    async def get_status(self) -> dict[str, Any]:
        self._enforce("media.web_scraping.status.server")
        return self._with_record_id(
            "server:web_scraping:status",
            await self._require_client().get_web_scraping_status(),
        )

    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        self._enforce("media.web_scraping.detail.server")
        return self._with_record_id(
            f"server:web_scraping_job:{job_id}",
            await self._require_client().get_web_scraping_job_status(job_id),
        )

    async def cancel_job(self, job_id: str) -> dict[str, Any]:
        self._enforce("media.web_scraping.cancel.server")
        return self._with_record_id(
            f"server:web_scraping_job:{job_id}",
            await self._require_client().cancel_web_scraping_job(job_id),
        )

    async def get_progress(self, task_id: str) -> dict[str, Any]:
        self._enforce("media.web_scraping.observe.server")
        return self._with_record_id(
            f"server:web_scraping_progress:{task_id}",
            await self._require_client().get_web_scraping_progress(task_id),
        )

    async def get_cookies(self, domain: str) -> dict[str, Any]:
        self._enforce("media.web_scraping.cookies.detail.server")
        return self._with_record_id(
            f"server:web_scraping_cookies:{domain}",
            await self._require_client().get_web_scraping_cookies(domain),
        )

    async def set_cookies(self, domain: str, cookies: list[dict[str, Any]]) -> dict[str, Any]:
        self._enforce("media.web_scraping.cookies.update.server")
        return self._with_record_id(
            f"server:web_scraping_cookies:{domain}",
            await self._require_client().set_web_scraping_cookies(domain, cookies),
        )

    async def check_duplicate(self, url: str) -> dict[str, Any]:
        self._enforce("media.web_scraping.inspect.server")
        return self._with_record_id(
            f"server:web_scraping_duplicate:{url}",
            await self._require_client().check_web_scraping_duplicate(url),
        )
