"""Policy-gated active-server meeting session service."""

from __future__ import annotations

from typing import Any, AsyncGenerator, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_provider_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import (
    MeetingArtifactCreate,
    MeetingFinalizeRequest,
    MeetingSessionCreate,
    MeetingSessionStatusUpdate,
    MeetingShareRequest,
    MeetingTemplateCreate,
    TLDWAPIClient,
)


class ServerMeetingsService:
    """Execute stable REST/SSE-backed meeting operations against the active server."""

    supports_websocket_live_ingest = False

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
    ) -> "ServerMeetingsService":
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
    ) -> "ServerMeetingsService":
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
        raise ValueError("TLDW API client is required for server meeting operations.")

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
                    user_message=getattr(decision, "user_message", None) or "Server meeting action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @classmethod
    def _dump(cls, response: Any) -> Any:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="python", by_alias=True)
        if isinstance(response, list):
            return [cls._dump(item) for item in response]
        if isinstance(response, dict):
            return {key: cls._dump(value) for key, value in response.items()}
        return response

    @staticmethod
    def _model(request_data: Any, model_type: type[Any]) -> Any:
        if isinstance(request_data, model_type):
            return request_data
        return model_type(**dict(request_data or {}))

    @staticmethod
    def _with_record_id(kind: str, payload: dict[str, Any], identifier: Any | None = None) -> dict[str, Any]:
        record = dict(payload or {})
        record.setdefault("backend", "server")
        if identifier is None:
            identifier = record.get("id") or record.get("session_id") or record.get("dispatch_id")
        if identifier is not None:
            record.setdefault("record_id", f"server:{kind}:{identifier}")
        return record

    @classmethod
    def _normalize_response(cls, payload: Any, *, kind: str | None = None) -> Any:
        payload = cls._dump(payload)
        if isinstance(payload, list):
            return [cls._normalize_response(item, kind=kind) for item in payload]
        if not isinstance(payload, dict):
            return payload
        if kind == "meeting_session":
            return cls._with_record_id("meeting_session", payload)
        if kind == "meeting_template":
            return cls._with_record_id("meeting_template", payload)
        if kind == "meeting_artifact":
            return cls._with_record_id("meeting_artifact", payload)
        if kind == "meeting_share":
            return cls._with_record_id("meeting_share", payload, payload.get("dispatch_id"))
        if kind == "meeting_health":
            return cls._with_record_id("meetings", payload, "health")

        record = dict(payload)
        record.setdefault("backend", "server")
        if record.get("session_id") is not None:
            record = cls._with_record_id("meeting_session", record, record.get("session_id"))
        if isinstance(record.get("artifacts"), list):
            record["artifacts"] = [
                cls._with_record_id("meeting_artifact", item) if isinstance(item, dict) else item
                for item in record["artifacts"]
            ]
        return record

    async def get_meetings_health(self) -> dict[str, Any]:
        self._enforce("meetings.health.detail.server")
        return self._normalize_response(await self._require_client().get_meetings_health(), kind="meeting_health")

    async def create_meeting_session(self, request_data: MeetingSessionCreate | dict[str, Any]) -> dict[str, Any]:
        self._enforce("meetings.sessions.create.server")
        request = self._model(request_data, MeetingSessionCreate)
        return self._normalize_response(
            await self._require_client().create_meeting_session(request),
            kind="meeting_session",
        )

    async def list_meeting_sessions(self, **kwargs: Any) -> list[dict[str, Any]]:
        self._enforce("meetings.sessions.list.server")
        return self._normalize_response(
            await self._require_client().list_meeting_sessions(**kwargs),
            kind="meeting_session",
        )

    async def get_meeting_session(self, session_id: str) -> dict[str, Any]:
        self._enforce("meetings.sessions.detail.server")
        return self._normalize_response(
            await self._require_client().get_meeting_session(session_id),
            kind="meeting_session",
        )

    async def update_meeting_session_status(
        self,
        session_id: str,
        request_data: MeetingSessionStatusUpdate | dict[str, Any],
    ) -> dict[str, Any]:
        self._enforce("meetings.sessions.update.server")
        request = self._model(request_data, MeetingSessionStatusUpdate)
        return self._normalize_response(
            await self._require_client().update_meeting_session_status(session_id, request),
            kind="meeting_session",
        )

    async def create_meeting_template(self, request_data: MeetingTemplateCreate | dict[str, Any]) -> dict[str, Any]:
        self._enforce("meetings.templates.create.server")
        request = self._model(request_data, MeetingTemplateCreate)
        return self._normalize_response(
            await self._require_client().create_meeting_template(request),
            kind="meeting_template",
        )

    async def list_meeting_templates(self, **kwargs: Any) -> list[dict[str, Any]]:
        self._enforce("meetings.templates.list.server")
        return self._normalize_response(
            await self._require_client().list_meeting_templates(**kwargs),
            kind="meeting_template",
        )

    async def get_meeting_template(self, template_id: str) -> dict[str, Any]:
        self._enforce("meetings.templates.detail.server")
        return self._normalize_response(
            await self._require_client().get_meeting_template(template_id),
            kind="meeting_template",
        )

    async def create_meeting_artifact(
        self,
        session_id: str,
        request_data: MeetingArtifactCreate | dict[str, Any],
    ) -> dict[str, Any]:
        self._enforce("meetings.artifacts.create.server")
        request = self._model(request_data, MeetingArtifactCreate)
        return self._normalize_response(
            await self._require_client().create_meeting_artifact(session_id, request),
            kind="meeting_artifact",
        )

    async def list_meeting_artifacts(self, session_id: str) -> list[dict[str, Any]]:
        self._enforce("meetings.artifacts.list.server")
        return self._normalize_response(
            await self._require_client().list_meeting_artifacts(session_id),
            kind="meeting_artifact",
        )

    async def finalize_meeting_session(
        self,
        session_id: str,
        request_data: MeetingFinalizeRequest | dict[str, Any],
    ) -> dict[str, Any]:
        self._enforce("meetings.sessions.launch.server")
        request = self._model(request_data, MeetingFinalizeRequest)
        return self._normalize_response(await self._require_client().finalize_meeting_session(session_id, request))

    async def share_meeting_session_to_slack(
        self,
        session_id: str,
        request_data: MeetingShareRequest | dict[str, Any],
    ) -> dict[str, Any]:
        self._enforce("meetings.share.launch.server")
        request = self._model(request_data, MeetingShareRequest)
        return self._normalize_response(
            await self._require_client().share_meeting_session_to_slack(session_id, request),
            kind="meeting_share",
        )

    async def share_meeting_session_to_webhook(
        self,
        session_id: str,
        request_data: MeetingShareRequest | dict[str, Any],
    ) -> dict[str, Any]:
        self._enforce("meetings.share.launch.server")
        request = self._model(request_data, MeetingShareRequest)
        return self._normalize_response(
            await self._require_client().share_meeting_session_to_webhook(session_id, request),
            kind="meeting_share",
        )

    async def stream_meeting_session_events(self, session_id: str) -> AsyncGenerator[dict[str, Any], None]:
        self._enforce("meetings.events.observe.server")
        async for event in self._require_client().stream_meeting_session_events(session_id):
            payload = self._dump(event)
            if isinstance(payload, dict):
                payload.setdefault("backend", "server")
            yield payload
