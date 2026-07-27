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
        target_store: Any | None = None,
    ) -> None:
        self.runtime_context = runtime_context
        self.server_runtime_scope_service = server_runtime_scope_service
        self.target_store = target_store

    async def refresh(self) -> dict[str, Any]:
        state, revision = self.runtime_context.snapshot()
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
                if not self.runtime_context.commit_state(
                    updated_state,
                    expected_revision=revision,
                ):
                    fresh_state, _ = self.runtime_context.snapshot()
                    return self._superseded_snapshot(fresh_state, now=now)
            else:
                fresh_state, fresh_revision = self.runtime_context.snapshot()
                if fresh_revision != revision or fresh_state != state:
                    return self._superseded_snapshot(fresh_state, now=now)
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
            readiness = await self._call_discovery_method(
                "probe_readiness", "get_readiness"
            )
            docs_info = await self._call_discovery_method(
                "probe_docs_info", "get_docs_info"
            )
        except Exception as exc:  # noqa: BLE001 - discovery must convert backend failures into state.
            reason_code = (
                classify_backend_exception(exc) or "capability_discovery_failed"
            )
            errors.append(
                {
                    "reason_code": reason_code,
                    "message": self._capability_error_message(reason_code),
                }
            )
            if reason_code == "server_unreachable":
                reachability = "unreachable"
                auth_state = "unknown"
            elif reason_code in {"server_auth_required", "server_session_invalid"}:
                reachability = "reachable"
                auth_state = (
                    "session_invalid"
                    if reason_code == "server_session_invalid"
                    else "auth_required"
                )
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
        if not self.runtime_context.commit_state(
            updated_state,
            expected_revision=revision,
        ):
            fresh_state, _ = self.runtime_context.snapshot()
            return self._superseded_snapshot(fresh_state, now=now)
        self._persist_target_status(
            state=updated_state,
            checked_at=now,
            reachability=reachability,
            auth_state=auth_state,
            errors=errors,
        )

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

    @classmethod
    def _superseded_snapshot(
        cls,
        state: RuntimeSourceState,
        *,
        now: datetime,
    ) -> dict[str, Any]:
        return cls._snapshot(
            state=state,
            now=now,
            reachability=state.server_reachability,
            auth_state=state.server_auth_state,
            errors=[
                {
                    "reason_code": "capability_result_superseded",
                    "message": (
                        "Capability refresh was superseded by a newer "
                        "runtime selection."
                    ),
                }
            ],
        )

    @staticmethod
    def _capability_error_message(reason_code: str) -> str:
        return {
            "server_unreachable": "The active server could not be reached.",
            "server_auth_required": "The active server requires authentication.",
            "server_session_invalid": "The active server session is no longer valid.",
        }.get(
            reason_code,
            "Capability discovery failed.",
        )

    async def _call_discovery_method(
        self, probe_name: str, scope_method_name: str
    ) -> dict[str, Any]:
        server_service = getattr(
            self.server_runtime_scope_service, "server_service", None
        )
        probe = getattr(server_service, probe_name, None)
        if callable(probe):
            result = await probe()
            return dict(result or {})
        scope_method = getattr(self.server_runtime_scope_service, scope_method_name)
        result = await scope_method(mode="server")
        return dict(result or {})

    def _persist_target_status(
        self,
        *,
        state: RuntimeSourceState,
        checked_at: datetime,
        reachability: str,
        auth_state: str,
        errors: list[dict[str, Any]],
    ) -> None:
        if not state.active_server_id or self.target_store is None:
            return

        update_target_status = getattr(self.target_store, "update_target_status", None)
        if not callable(update_target_status):
            return

        try:
            update_target_status(
                state.active_server_id,
                last_known_server_label=state.last_known_server_label,
                last_known_reachability=reachability,
                last_known_auth_state=auth_state,
                updated_at=checked_at,
            )
        except KeyError:
            errors.append(
                {
                    "reason_code": "target_profile_missing",
                    "message": "Active server target profile was not found.",
                }
            )
