"""Server-backed deep research run service."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncGenerator, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_provider_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import (
    ResearchCheckpointPatchApproveRequest,
    ResearchRunCreateRequest,
)
if TYPE_CHECKING:
    from ..tldw_api import TLDWAPIClient
from .research_normalizers import ResearchRecord, ResearchRecordList


class ServerResearchService:
    """Policy-gated access to server deep research runs."""

    supports_run_delete = True

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
    ) -> "ServerResearchService":
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
    ) -> "ServerResearchService":
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
        raise ValueError("TLDW API client is required for server research operations.")

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
                    user_message=getattr(decision, "user_message", None) or "Server research action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @staticmethod
    def _dump(response: Any) -> ResearchRecord:
        if hasattr(response, "model_dump"):
            payload = response.model_dump(mode="json")
        else:
            payload = dict(response or {})
        payload.setdefault("source", "server")
        return ResearchRecord(payload)

    @staticmethod
    def _dump_list(response: Any) -> ResearchRecordList:
        return ResearchRecordList(ServerResearchService._dump(item) for item in list(response or []))

    @staticmethod
    def _dump_event(response: Any) -> ResearchRecord:
        if hasattr(response, "model_dump"):
            payload = response.model_dump(mode="json")
        else:
            payload = dict(response or {})
        event_id = payload.pop("event_id", None)
        if event_id is not None:
            payload["id"] = event_id
        return ResearchRecord(payload)

    async def create_run(self, **kwargs: Any) -> ResearchRecord:
        return await self.launch_run(**kwargs)

    async def launch_run(
        self,
        *,
        query: str,
        source_policy: str = "balanced",
        autonomy_mode: str = "checkpointed",
        limits_json: dict[str, Any] | None = None,
        provider_overrides: dict[str, Any] | None = None,
        chat_handoff: dict[str, Any] | None = None,
        follow_up: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._enforce("research.runs.launch.server")
        request = ResearchRunCreateRequest(
            query=query,
            source_policy=source_policy,
            autonomy_mode=autonomy_mode,
            limits_json=limits_json,
            provider_overrides=provider_overrides,
            chat_handoff=chat_handoff,
            follow_up=follow_up,
        )
        return self._dump(await self._require_client().create_research_run(request))

    async def list_runs(
        self,
        *,
        limit: int = 25,
        offset: int = 0,
        session_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        self._enforce("research.runs.list.server")
        return self._dump_list(
            await self._require_client().list_research_runs(
                limit=limit,
                offset=offset,
                session_id=session_id,
                status=status,
            )
        )

    async def get_run(self, session_id: str) -> dict[str, Any]:
        self._enforce("research.runs.detail.server")
        return self._dump(await self._require_client().get_research_run(session_id))

    async def observe_run_events(
        self,
        session_id: str,
        *,
        after_id: int = 0,
    ) -> AsyncGenerator[dict[str, Any], None]:
        self._enforce("research.runs.observe.server")
        async for event in self._require_client().stream_research_run_events(session_id, after_id=after_id):
            yield self._dump_event(event)

    async def stream_run_events(
        self,
        session_id: str,
        *,
        after_id: int = 0,
    ) -> AsyncGenerator[dict[str, Any], None]:
        async for event in self.observe_run_events(session_id, after_id=after_id):
            yield event

    async def pause_run(self, session_id: str) -> dict[str, Any]:
        self._enforce("research.runs.update.server")
        return self._dump(await self._require_client().pause_research_run(session_id))

    async def resume_run(self, session_id: str) -> dict[str, Any]:
        self._enforce("research.runs.update.server")
        return self._dump(await self._require_client().resume_research_run(session_id))

    async def cancel_run(self, session_id: str) -> dict[str, Any]:
        self._enforce("research.runs.update.server")
        return self._dump(await self._require_client().cancel_research_run(session_id))

    async def delete_run(self, session_id: str, *, expected_version: int | None = None) -> bool:
        self._enforce("research.runs.delete.server")
        response = await self._require_client().delete_research_run(session_id)
        if isinstance(response, Mapping):
            return bool(response.get("deleted"))
        return bool(response)

    async def get_bundle(self, session_id: str) -> dict[str, Any]:
        self._enforce("research.runs.detail.server")
        return await self._require_client().get_research_bundle(session_id)

    async def get_artifact(self, session_id: str, artifact_name: str) -> dict[str, Any]:
        self._enforce("research.runs.detail.server")
        return self._dump(await self._require_client().get_research_artifact(session_id, artifact_name))

    async def patch_and_approve_checkpoint(
        self,
        session_id: str,
        checkpoint_id: str,
        *,
        patch_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._enforce("research.runs.update.server")
        request = ResearchCheckpointPatchApproveRequest(patch_payload=patch_payload)
        return self._dump(
            await self._require_client().patch_and_approve_research_checkpoint(
                session_id,
                checkpoint_id,
                request,
            )
        )
