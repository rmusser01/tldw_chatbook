from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from .enforcement import classify_backend_exception
from .types import RuntimeSourceState


class ActiveServerCapabilityService:
    """Refresh a source-honest capability snapshot for the configured active server."""

    def __init__(
        self,
        *,
        runtime_context: Any,
        server_runtime_scope_service: Any,
    ) -> None:
        self.runtime_context = runtime_context
        self.server_runtime_scope_service = server_runtime_scope_service

    async def refresh(self) -> dict[str, Any]:
        state = self._current_state()
        now = datetime.now(timezone.utc)
        if not state.server_configured or not state.active_server_id:
            updated_state = replace(
                state,
                server_reachability="unknown",
                server_reachability_checked_at=None,
                server_auth_state="unknown",
                server_auth_checked_at=None,
            )
            if updated_state != state:
                self.runtime_context.state = updated_state
                persist = getattr(self.runtime_context, "persist", None)
                if callable(persist):
                    persist()
            return self._snapshot(
                state=updated_state,
                now=now,
                reachability="unknown",
                auth_state="unknown",
                errors=[
                    {
                        "reason_code": "server_not_configured",
                        "message": "No active server is configured.",
                    }
                ],
            )

        health: dict[str, Any] = {}
        readiness: dict[str, Any] = {}
        docs_info: dict[str, Any] = {}
        errors: list[dict[str, Any]] = []
        reachability = "reachable"
        auth_state = "authenticated"

        try:
            health = await self._call_discovery_method("probe_health", "get_health")
            readiness = await self._call_discovery_method("probe_readiness", "get_readiness")
            docs_info = await self._call_discovery_method("probe_docs_info", "get_docs_info")
        except Exception as exc:  # noqa: BLE001 - discovery must convert backend failures into state.
            reason_code = classify_backend_exception(exc) or "capability_discovery_failed"
            errors.append({"reason_code": reason_code, "message": str(exc)})
            if reason_code == "server_unreachable":
                reachability = "unreachable"
                auth_state = "unknown"
            elif reason_code in {"server_auth_required", "server_session_invalid"}:
                reachability = "reachable"
                auth_state = "session_invalid" if reason_code == "server_session_invalid" else "auth_required"
            else:
                reachability = "reachable"
                auth_state = "unknown"

        updated_state = replace(
            state,
            server_reachability=reachability,
            server_reachability_checked_at=now,
            server_auth_state=auth_state,
            server_auth_checked_at=now,
        )
        self.runtime_context.state = updated_state
        persist = getattr(self.runtime_context, "persist", None)
        if callable(persist):
            persist()

        return self._snapshot(
            state=updated_state,
            now=now,
            reachability=reachability,
            auth_state=auth_state,
            health=health,
            readiness=readiness,
            docs_info=docs_info,
            errors=errors,
        )

    @staticmethod
    def _snapshot(
        *,
        state: RuntimeSourceState,
        now: datetime,
        reachability: str,
        auth_state: str,
        health: dict[str, Any] | None = None,
        readiness: dict[str, Any] | None = None,
        docs_info: dict[str, Any] | None = None,
        errors: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        resolved_server_id = state.active_server_id or "unconfigured"
        docs = dict(docs_info or {})
        return {
            "backend": "server",
            "record_id": f"server:capability_snapshot:{resolved_server_id}",
            "active_server_id": state.active_server_id,
            "server_configured": state.server_configured,
            "last_known_server_label": state.last_known_server_label,
            "checked_at": now.isoformat().replace("+00:00", "Z"),
            "reachability": reachability,
            "auth_state": auth_state,
            "health": dict(health or {}),
            "readiness": dict(readiness or {}),
            "docs_info": docs,
            "capabilities": dict(docs.get("capabilities") or {}),
            "supported_features": dict(docs.get("supported_features") or {}),
            "errors": list(errors or []),
        }

    def _current_state(self) -> RuntimeSourceState:
        state = getattr(self.runtime_context, "state", None)
        if isinstance(state, RuntimeSourceState):
            return state
        return RuntimeSourceState()

    async def _call_discovery_method(self, probe_name: str, scope_method_name: str) -> dict[str, Any]:
        server_service = getattr(self.server_runtime_scope_service, "server_service", None)
        probe = getattr(server_service, probe_name, None)
        if callable(probe):
            result = await probe()
            return dict(result or {})
        scope_method = getattr(self.server_runtime_scope_service, scope_method_name)
        result = await scope_method(mode="server")
        return dict(result or {})
