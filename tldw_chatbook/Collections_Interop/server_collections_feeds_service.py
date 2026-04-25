"""Server-backed collections feed subscription service."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import (
    CollectionsFeedCreateRequest,
    CollectionsFeedUpdateRequest,
    TLDWAPIClient,
)


class ServerCollectionsFeedsService:
    """Policy-gated access to server collection feed APIs."""

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
    ) -> "ServerCollectionsFeedsService":
        return cls(
            client=build_runtime_api_client_from_config(app_config),
            policy_enforcer=policy_enforcer,
        )

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server collections feed operations.")
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
                    or "Server collections feed action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @staticmethod
    def _dump(response: Any) -> Any:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        if isinstance(response, (dict, list, bool)):
            return response
        return dict(response or {})

    async def create_feed(
        self,
        *,
        url: str,
        name: str | None = None,
        tags: list[str] | None = None,
        schedule_expr: str | None = None,
        timezone: str | None = None,
        active: bool = True,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._enforce("collections.feeds.create.server")
        request = CollectionsFeedCreateRequest(
            url=url,  # type: ignore[arg-type]
            name=name,
            tags=tags or [],
            schedule_expr=schedule_expr,
            timezone=timezone,
            active=active,
            settings=settings,
        )
        return self._dump(await self._require_client().create_collections_feed(request))

    async def list_feeds(
        self,
        *,
        q: str | None = None,
        page: int = 1,
        size: int = 20,
    ) -> dict[str, Any]:
        self._enforce("collections.feeds.list.server")
        return self._dump(await self._require_client().list_collections_feeds(q=q, page=page, size=size))

    async def get_feed(self, feed_id: int) -> dict[str, Any]:
        self._enforce("collections.feeds.detail.server")
        return self._dump(await self._require_client().get_collections_feed(int(feed_id)))

    async def update_feed(
        self,
        feed_id: int,
        *,
        name: str | None = None,
        url: str | None = None,
        tags: list[str] | None = None,
        schedule_expr: str | None = None,
        timezone: str | None = None,
        active: bool | None = None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._enforce("collections.feeds.update.server")
        request = CollectionsFeedUpdateRequest(
            name=name,
            url=url,  # type: ignore[arg-type]
            tags=tags,
            schedule_expr=schedule_expr,
            timezone=timezone,
            active=active,
            settings=settings,
        )
        return self._dump(await self._require_client().update_collections_feed(int(feed_id), request))

    async def delete_feed(self, feed_id: int) -> bool:
        self._enforce("collections.feeds.delete.server")
        return bool(await self._require_client().delete_collections_feed(int(feed_id)))
