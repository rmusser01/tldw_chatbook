"""Thin server-backed chat-loop execution service around the shared API client."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_from_config
from ..tldw_api import ChatLoopStartRequest, TLDWAPIClient


class ServerChatLoopService:
    """Adapt server chat-loop endpoints to a source-aware Chatbook service seam."""

    def __init__(self, client: Optional[TLDWAPIClient]):
        self.client = client

    @classmethod
    def from_config(cls, app_config: Mapping[str, Any]) -> "ServerChatLoopService":
        return cls(client=build_runtime_api_client_from_config(app_config))

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server chat loop operations.")
        return self.client

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return dict(value)
        if hasattr(value, "model_dump"):
            return value.model_dump(exclude_none=True, mode="json")
        return dict(value)

    async def start_run(self, *, messages: list[dict[str, Any]], **payload: Any) -> dict[str, Any]:
        request_data = ChatLoopStartRequest(messages=messages, **payload)
        response = await self._require_client().start_chat_loop_run(request_data)
        return self._as_dict(response)

    async def list_events(self, run_id: str, *, after_seq: int = 0) -> dict[str, Any]:
        response = await self._require_client().list_chat_loop_events(str(run_id), after_seq=after_seq)
        return self._as_dict(response)

    async def approve(self, run_id: str, approval_id: str) -> dict[str, Any]:
        response = await self._require_client().approve_chat_loop_call(str(run_id), str(approval_id))
        return self._as_dict(response)

    async def reject(self, run_id: str, approval_id: str) -> dict[str, Any]:
        response = await self._require_client().reject_chat_loop_call(str(run_id), str(approval_id))
        return self._as_dict(response)

    async def cancel(self, run_id: str) -> dict[str, Any]:
        response = await self._require_client().cancel_chat_loop_run(str(run_id))
        return self._as_dict(response)
