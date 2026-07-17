"""Server-backed browser web-clipper service."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_provider_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import (
    WebClipperEnrichmentPayload,
    WebClipperSaveRequest,
)
if TYPE_CHECKING:
    from ..tldw_api import TLDWAPIClient


class ServerWebClipperService:
    """Policy-gated access to server web-clipper APIs."""

    def __init__(
        self,
        client: Optional[TLDWAPIClient],
        *,
        client_provider: Any | None = None,
        policy_enforcer: Any | None = None,
    ) -> None:
        self.client = client
        self.client_provider = client_provider
        self.policy_enforcer = policy_enforcer

    @classmethod
    def from_config(
        cls,
        app_config: Mapping[str, Any],
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerWebClipperService":
        return cls(
            client=None,
            client_provider=build_runtime_api_client_provider_from_config(app_config),
            policy_enforcer=policy_enforcer,
        )

    @classmethod
    def from_server_context_provider(
        cls,
        provider: Any,
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerWebClipperService":
        return cls(
            client=None,
            client_provider=provider,
            policy_enforcer=policy_enforcer,
        )

    def _require_client(self) -> TLDWAPIClient:
        if self.client is not None:
            return self.client
        if self.client_provider is not None:
            return self.client_provider.build_client()
        raise ValueError("TLDW API client is required for server web-clipper operations.")

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
                    or "Server web-clipper action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @staticmethod
    def _dump(response: Any) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        return dict(response or {})

    async def save_clip(
        self,
        *,
        clip_id: str,
        clip_type: str,
        source_url: str,
        source_title: str,
        destination_mode: str = "note",
        note: dict[str, Any] | None = None,
        workspace: dict[str, Any] | None = None,
        content: dict[str, Any] | None = None,
        attachments: list[dict[str, Any]] | None = None,
        enhancements: dict[str, Any] | None = None,
        capture_metadata: dict[str, Any] | None = None,
        source_note_version: int | None = None,
    ) -> dict[str, Any]:
        self._enforce("web_clipper.capture.server")
        request = WebClipperSaveRequest(
            clip_id=clip_id,
            clip_type=clip_type,
            source_url=source_url,
            source_title=source_title,
            destination_mode=destination_mode,  # type: ignore[arg-type]
            note=note or {},
            workspace=workspace,
            content=content or {},
            attachments=attachments or [],
            enhancements=enhancements or {},
            capture_metadata=capture_metadata or {},
            source_note_version=source_note_version,
        )
        return self._dump(await self._require_client().save_web_clip(request))

    async def get_status(self, clip_id: str) -> dict[str, Any]:
        self._enforce("web_clipper.status.server")
        return self._dump(await self._require_client().get_web_clip_status(clip_id))

    async def persist_enrichment(
        self,
        *,
        clip_id: str,
        enrichment_type: str,
        source_note_version: int,
        status: str = "pending",
        inline_summary: str | None = None,
        structured_payload: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        self._enforce("web_clipper.capture.server")
        request = WebClipperEnrichmentPayload(
            clip_id=clip_id,
            enrichment_type=enrichment_type,  # type: ignore[arg-type]
            status=status,  # type: ignore[arg-type]
            inline_summary=inline_summary,
            structured_payload=structured_payload or {},
            source_note_version=source_note_version,
            error=error,
        )
        return self._dump(await self._require_client().persist_web_clip_enrichment(clip_id, request))
