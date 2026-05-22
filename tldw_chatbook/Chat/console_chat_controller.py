"""Native Console chat controller for send, stream, stop, and retry flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleProviderSelection,
    ConsoleRunState,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore


class ConsoleProviderGatewayProtocol(Protocol):
    """Provider gateway surface required by the Console controller."""

    async def resolve_for_send(self, selection: ConsoleProviderSelection) -> Any:
        """Resolve provider readiness for a send."""

    async def stream_chat(self, resolution: Any, messages: list[dict[str, str]]) -> Any:
        """Stream response chunks for provider messages."""


@dataclass(frozen=True)
class ConsoleSubmitResult:
    """Result returned to the composer after a Console submit attempt."""

    accepted: bool
    should_clear_draft: bool
    visible_copy: str = ""


class ConsoleChatController:
    """Coordinate native Console chat state between store and provider gateway."""

    def __init__(
        self,
        *,
        store: ConsoleChatStore,
        provider_gateway: ConsoleProviderGatewayProtocol,
        provider: str = "llama_cpp",
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.store = store
        self.provider_gateway = provider_gateway
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.run_state = ConsoleRunState()
        self.run_state_history: list[ConsoleRunStatus] = [self.run_state.status]
        self._active_assistant_message_id: str | None = None
        self._stop_requested = False

    async def submit_draft(self, draft: str) -> ConsoleSubmitResult:
        """Submit a composer draft through native Console validation and provider resolution."""
        normalized_draft = draft.strip()
        session = self.store.ensure_session(
            workspace_id=self.store.workspace_context.active_workspace_id,
        )
        if not normalized_draft:
            return self._block(session.id, "Type a message before sending.")
        if self.store.workspace_context.has_policy_blocks:
            return self._block(session.id, self.store.workspace_context.recovery_copy)

        resolution = await self.provider_gateway.resolve_for_send(self._provider_selection())
        if not getattr(resolution, "ready", False):
            visible_copy = getattr(resolution, "visible_copy", "") or "Provider blocked."
            return self._block(session.id, visible_copy)

        self.store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content=normalized_draft,
            persist=self.store.persistence is not None,
        )
        provider_messages = self._provider_messages_for_session(session.id)
        assistant = self.store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            persist=self.store.persistence is not None,
        )
        return await self._stream_assistant_response(
            resolution=resolution,
            provider_messages=provider_messages,
            assistant_message_id=assistant.id,
        )

    def stop_active_run(self) -> bool:
        """Request the active stream to stop at the next safe boundary."""
        if self.run_state.status is not ConsoleRunStatus.STREAMING:
            return False
        if self._active_assistant_message_id is None:
            return False
        self._stop_requested = True
        return True

    async def retry_message(self, message_id: str) -> ConsoleSubmitResult:
        """Retry a failed assistant message using the original turn context."""
        session_id = self.store.active_session_id
        if session_id is None:
            return ConsoleSubmitResult(False, False, "No active Console session.")
        message = self.store.get_message(message_id)
        if message.status != "failed":
            return self._block(session_id, "Only failed messages can be retried.")

        self._set_run_state(ConsoleRunState.retrying("Retrying failed response."))
        resolution = await self.provider_gateway.resolve_for_send(self._provider_selection())
        if not getattr(resolution, "ready", False):
            visible_copy = getattr(resolution, "visible_copy", "") or "Provider blocked."
            return self._block(session_id, visible_copy)

        provider_messages = self._provider_messages_for_session(
            session_id,
            before_message_id=message_id,
        )
        return await self._stream_assistant_response(
            resolution=resolution,
            provider_messages=provider_messages,
            assistant_message_id=message_id,
            prepare_retry=True,
        )

    def _provider_selection(self) -> ConsoleProviderSelection:
        return ConsoleProviderSelection(
            provider=self.provider,
            base_url=self.base_url,
            explicit_model=self.model,
            workspace_context=self.store.workspace_context,
        )

    def _block(self, session_id: str, visible_copy: str) -> ConsoleSubmitResult:
        self._set_run_state(ConsoleRunState.blocked(visible_copy))
        self.store.append_message(
            session_id,
            role=ConsoleMessageRole.SYSTEM,
            content=visible_copy,
        )
        return ConsoleSubmitResult(
            accepted=False,
            should_clear_draft=False,
            visible_copy=visible_copy,
        )

    async def _stream_assistant_response(
        self,
        *,
        resolution: Any,
        provider_messages: list[dict[str, str]],
        assistant_message_id: str,
        prepare_retry: bool = False,
    ) -> ConsoleSubmitResult:
        self._active_assistant_message_id = assistant_message_id
        self._stop_requested = False
        self._set_run_state(ConsoleRunState(ConsoleRunStatus.STREAMING, "Streaming response."))
        retry_prepared = False
        try:
            async for chunk in self.provider_gateway.stream_chat(resolution, provider_messages):
                if self._stop_requested:
                    stopped = (
                        self.store.mark_message_stopped(assistant_message_id)
                        if not prepare_retry or retry_prepared
                        else self.store.get_message(assistant_message_id)
                    )
                    self._set_run_state(ConsoleRunState(ConsoleRunStatus.STOPPED, "Response stopped."))
                    return ConsoleSubmitResult(True, True, stopped.content)
                if prepare_retry and not retry_prepared:
                    self.store.prepare_message_retry(assistant_message_id)
                    retry_prepared = True
                self.store.append_stream_chunk(assistant_message_id, chunk)
            if self._stop_requested:
                stopped = (
                    self.store.mark_message_stopped(assistant_message_id)
                    if not prepare_retry or retry_prepared
                    else self.store.get_message(assistant_message_id)
                )
                self._set_run_state(ConsoleRunState(ConsoleRunStatus.STOPPED, "Response stopped."))
                return ConsoleSubmitResult(True, True, stopped.content)
            if prepare_retry and not retry_prepared:
                failed = self.store.get_message(assistant_message_id)
                self._set_run_state(
                    ConsoleRunState(
                        ConsoleRunStatus.FAILED,
                        "Provider stream ended without replacement content.",
                    )
                )
                return ConsoleSubmitResult(True, True, failed.content)
            completed = self.store.mark_message_complete(assistant_message_id)
            self._set_run_state(ConsoleRunState(ConsoleRunStatus.COMPLETED, "Response complete."))
            return ConsoleSubmitResult(True, True, completed.content)
        except Exception as exc:
            failed = (
                self.store.mark_message_failed(assistant_message_id)
                if not prepare_retry or retry_prepared
                else self.store.get_message(assistant_message_id)
            )
            visible_copy = f"Provider stream failed: {exc}"
            self._set_run_state(ConsoleRunState(ConsoleRunStatus.FAILED, visible_copy))
            return ConsoleSubmitResult(True, True, failed.content)
        finally:
            self._active_assistant_message_id = None
            self._stop_requested = False

    def _provider_messages_for_session(
        self,
        session_id: str,
        *,
        before_message_id: str | None = None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for message in self.store.messages_for_session(session_id):
            if message.id == before_message_id:
                break
            if not message.content:
                continue
            if message.role not in {ConsoleMessageRole.USER, ConsoleMessageRole.ASSISTANT}:
                continue
            if message.status == "failed":
                continue
            messages.append({"role": message.role.value, "content": message.content})
        return messages

    def _set_run_state(self, run_state: ConsoleRunState) -> None:
        self.run_state = run_state
        self.run_state_history.append(run_state.status)
