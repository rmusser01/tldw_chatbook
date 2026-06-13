"""Source-aware routing for server-owned meeting sessions and event streams."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any, AsyncGenerator


class MeetingsBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "meetings.remote_only.local",
        "source": "local",
        "supported": False,
        "reason_code": "remote_only_surface",
        "user_message": "Server meeting sessions, templates, artifacts, sharing, finalization, and event streams are unavailable in local/offline mode.",
        "affected_action_ids": [],
    }
]

_SERVER_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "meetings.websocket_live_ingest.server",
        "source": "server",
        "supported": False,
        "reason_code": "client_adapter_missing",
        "user_message": "The server exposes websocket live transcript ingestion, but this Chatbook meetings adapter only exposes REST CRUD/finalization/sharing and SSE event observation.",
        "affected_action_ids": [],
    }
]


class MeetingsScopeService:
    """Route meeting operations through the active server without local meeting authority."""

    def __init__(self, *, server_service: Any = None, policy_enforcer: Any = None):
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: MeetingsBackend | str | None) -> MeetingsBackend:
        if mode is None:
            return MeetingsBackend.SERVER
        if isinstance(mode, MeetingsBackend):
            return mode
        try:
            return MeetingsBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid meetings backend: {mode}") from exc

    def _require_server_service(self, mode: MeetingsBackend) -> Any:
        if mode == MeetingsBackend.LOCAL:
            raise ValueError("Server meeting records are server-only; switch to server mode to manage them.")
        if self.server_service is None:
            raise ValueError("Server meetings backend is unavailable.")
        return self.server_service

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    @staticmethod
    def _dump(payload: Any) -> Any:
        if hasattr(payload, "model_dump"):
            return payload.model_dump(mode="python", by_alias=True)
        if isinstance(payload, dict):
            return {key: MeetingsScopeService._dump(value) for key, value in payload.items()}
        if isinstance(payload, list):
            return [MeetingsScopeService._dump(item) for item in payload]
        return payload

    @staticmethod
    def _with_record_id(mode: MeetingsBackend, kind: str, payload: dict[str, Any], identifier: Any | None = None) -> dict[str, Any]:
        record = dict(payload or {})
        record.setdefault("backend", mode.value)
        if identifier is None:
            identifier = record.get("id") or record.get("session_id") or record.get("dispatch_id")
        if identifier is not None:
            record.setdefault("record_id", f"{mode.value}:{kind}:{identifier}")
        return record

    @classmethod
    def _normalize_response(cls, mode: MeetingsBackend, payload: Any, *, kind: str | None = None) -> Any:
        payload = cls._dump(payload)
        if isinstance(payload, list):
            return [cls._normalize_response(mode, item, kind=kind) for item in payload]
        if not isinstance(payload, dict):
            return payload
        if kind == "meeting_session":
            return cls._with_record_id(mode, "meeting_session", payload)
        if kind == "meeting_template":
            return cls._with_record_id(mode, "meeting_template", payload)
        if kind == "meeting_artifact":
            return cls._with_record_id(mode, "meeting_artifact", payload)
        if kind == "meeting_share":
            return cls._with_record_id(mode, "meeting_share", payload, payload.get("dispatch_id"))
        if kind == "meeting_health":
            return cls._with_record_id(mode, "meetings", payload, "health")

        record = dict(payload)
        record.setdefault("backend", mode.value)
        if record.get("session_id") is not None:
            record = cls._with_record_id(mode, "meeting_session", record, record.get("session_id"))
        if isinstance(record.get("artifacts"), list):
            record["artifacts"] = [
                cls._with_record_id(mode, "meeting_artifact", item) if isinstance(item, dict) else item
                for item in record["artifacts"]
            ]
        return record

    def list_unsupported_capabilities(
        self,
        *,
        mode: MeetingsBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == MeetingsBackend.LOCAL:
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        reports = []
        for item in _SERVER_UNSUPPORTED_CAPABILITIES:
            if item["operation_id"] == "meetings.websocket_live_ingest.server" and self._has_websocket_live_ingest_adapter():
                continue
            reports.append(dict(item))
        return reports

    def _has_websocket_live_ingest_adapter(self) -> bool:
        service = self.server_service
        if service is None:
            return False
        explicit_support = getattr(service, "supports_websocket_live_ingest", None)
        if explicit_support is False:
            return False
        if explicit_support is True:
            return True
        return callable(getattr(service, "stream_meeting_session_websocket", None))

    async def _call(
        self,
        *,
        mode: MeetingsBackend | str | None,
        action_id: str,
        method_name: str,
        normalize_kind: str | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy(action_id)
        result = await self._maybe_await(getattr(service, method_name)(*args, **(kwargs or {})))
        return self._normalize_response(normalized_mode, result, kind=normalize_kind)

    async def get_meetings_health(self, *, mode: MeetingsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="meetings.health.detail.server",
            method_name="get_meetings_health",
            normalize_kind="meeting_health",
        )

    async def create_meeting_session(self, request_data: Any, *, mode: MeetingsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="meetings.sessions.create.server",
            method_name="create_meeting_session",
            normalize_kind="meeting_session",
            args=(request_data,),
        )

    async def list_meeting_sessions(self, *, mode: MeetingsBackend | str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        return await self._call(
            mode=mode,
            action_id="meetings.sessions.list.server",
            method_name="list_meeting_sessions",
            normalize_kind="meeting_session",
            kwargs=kwargs,
        )

    async def get_meeting_session(self, session_id: str, *, mode: MeetingsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="meetings.sessions.detail.server",
            method_name="get_meeting_session",
            normalize_kind="meeting_session",
            args=(session_id,),
        )

    async def update_meeting_session_status(self, session_id: str, request_data: Any, *, mode: MeetingsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="meetings.sessions.update.server",
            method_name="update_meeting_session_status",
            normalize_kind="meeting_session",
            args=(session_id, request_data),
        )

    async def create_meeting_template(self, request_data: Any, *, mode: MeetingsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="meetings.templates.create.server",
            method_name="create_meeting_template",
            normalize_kind="meeting_template",
            args=(request_data,),
        )

    async def list_meeting_templates(self, *, mode: MeetingsBackend | str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        return await self._call(
            mode=mode,
            action_id="meetings.templates.list.server",
            method_name="list_meeting_templates",
            normalize_kind="meeting_template",
            kwargs=kwargs,
        )

    async def get_meeting_template(self, template_id: str, *, mode: MeetingsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="meetings.templates.detail.server",
            method_name="get_meeting_template",
            normalize_kind="meeting_template",
            args=(template_id,),
        )

    async def create_meeting_artifact(self, session_id: str, request_data: Any, *, mode: MeetingsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="meetings.artifacts.create.server",
            method_name="create_meeting_artifact",
            normalize_kind="meeting_artifact",
            args=(session_id, request_data),
        )

    async def list_meeting_artifacts(self, session_id: str, *, mode: MeetingsBackend | str | None = None) -> list[dict[str, Any]]:
        return await self._call(
            mode=mode,
            action_id="meetings.artifacts.list.server",
            method_name="list_meeting_artifacts",
            normalize_kind="meeting_artifact",
            args=(session_id,),
        )

    async def finalize_meeting_session(self, session_id: str, request_data: Any, *, mode: MeetingsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="meetings.sessions.launch.server",
            method_name="finalize_meeting_session",
            args=(session_id, request_data),
        )

    async def share_meeting_session_to_slack(self, session_id: str, request_data: Any, *, mode: MeetingsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="meetings.share.launch.server",
            method_name="share_meeting_session_to_slack",
            normalize_kind="meeting_share",
            args=(session_id, request_data),
        )

    async def share_meeting_session_to_webhook(self, session_id: str, request_data: Any, *, mode: MeetingsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="meetings.share.launch.server",
            method_name="share_meeting_session_to_webhook",
            normalize_kind="meeting_share",
            args=(session_id, request_data),
        )

    async def stream_meeting_session_events(
        self,
        session_id: str,
        *,
        mode: MeetingsBackend | str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy("meetings.events.observe.server")
        async for event in service.stream_meeting_session_events(session_id):
            payload = self._dump(event)
            if isinstance(payload, dict):
                payload.setdefault("backend", normalized_mode.value)
            yield payload
