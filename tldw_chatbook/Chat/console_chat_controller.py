"""Native Console chat controller for send, stream, stop, and retry flows."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleProviderSelection,
    ConsoleRunState,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession, ConsoleChatStore
from tldw_chatbook.Utils.input_validation import sanitize_string, validate_text_input


MAX_CONSOLE_DRAFT_LENGTH = 100_000


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
        configured_model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.store = store
        self.provider_gateway = provider_gateway
        self.provider = provider
        self.model = model
        self.configured_model = configured_model
        self.base_url = base_url
        self.run_state = ConsoleRunState()
        self.run_state_history: list[ConsoleRunStatus] = [self.run_state.status]
        self._active_assistant_message_id: str | None = None
        self._active_stream_task: asyncio.Task | None = None
        self._stop_requested = False

    async def submit_draft(self, draft: str) -> ConsoleSubmitResult:
        """Submit a composer draft through native Console validation and provider resolution."""
        active_rejection = self._active_run_rejection()
        if active_rejection is not None:
            return active_rejection

        session = self.store.ensure_session(
            workspace_id=self.store.workspace_context.active_workspace_id,
        )
        clean_draft, validation_error = self._validated_draft(draft)
        if validation_error is not None:
            return self._block(session.id, validation_error)
        if self.store.workspace_context.has_policy_blocks:
            return self._block(session.id, self.store.workspace_context.recovery_copy)

        self._set_run_state(ConsoleRunState(ConsoleRunStatus.VALIDATING, "Validating provider."))
        resolution = await self.provider_gateway.resolve_for_send(self._provider_selection())
        if not getattr(resolution, "ready", False):
            visible_copy = self._blocked_visible_copy(getattr(resolution, "visible_copy", ""))
            return self._block(session.id, visible_copy)

        self.store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content=clean_draft,
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

    def new_session(self, *, title: str | None = None) -> ConsoleChatSession:
        """Create and activate a new native Console session."""
        next_number = len(self.store.sessions()) + 1
        return self.store.create_session(title=title or f"Chat {next_number}")

    def switch_session(self, session_id: str) -> ConsoleChatSession:
        """Activate an existing native Console session."""
        return self.store.switch_session(session_id)

    def stop_active_run(self) -> bool:
        """Request the active stream to stop at the next safe boundary."""
        if self.run_state.status is not ConsoleRunStatus.STREAMING:
            return False
        if self._active_assistant_message_id is None:
            return False
        self._stop_requested = True
        if self._active_stream_task is not None and self._active_stream_task is not asyncio.current_task():
            self._active_stream_task.cancel()
        return True

    async def retry_message(self, message_id: str) -> ConsoleSubmitResult:
        """Retry a failed assistant message using the original turn context."""
        active_rejection = self._active_run_rejection()
        if active_rejection is not None:
            return active_rejection

        session_id = self.store.active_session_id
        if session_id is None:
            return ConsoleSubmitResult(False, False, "No active Console session.")
        message = self.store.get_message(message_id)
        message_session_id = self.store.session_id_for_message(message_id)
        if message_session_id != session_id:
            visible_copy = "Open the original session before retrying this message."
            self._set_run_state(ConsoleRunState.blocked(visible_copy))
            return ConsoleSubmitResult(False, False, visible_copy)
        if message.status != "failed":
            return self._block(session_id, "Only failed messages can be retried.")

        self._set_run_state(ConsoleRunState.retrying("Retrying failed response."))
        resolution = await self.provider_gateway.resolve_for_send(self._provider_selection())
        if not getattr(resolution, "ready", False):
            visible_copy = self._blocked_visible_copy(getattr(resolution, "visible_copy", ""))
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

    async def continue_from_message(self, message_id: str) -> ConsoleSubmitResult:
        """Continue from a selected message by streaming a new assistant turn."""
        active_rejection = self._active_run_rejection()
        if active_rejection is not None:
            return active_rejection

        session_id = self.store.active_session_id
        if session_id is None:
            return ConsoleSubmitResult(False, False, "No active Console session.")
        message_session_id = self.store.session_id_for_message(message_id)
        if message_session_id != session_id:
            visible_copy = "Open the original session before continuing from this message."
            self._set_run_state(ConsoleRunState.blocked(visible_copy))
            return ConsoleSubmitResult(False, False, visible_copy)

        self._set_run_state(ConsoleRunState(ConsoleRunStatus.VALIDATING, "Validating provider."))
        resolution = await self.provider_gateway.resolve_for_send(self._provider_selection())
        if not getattr(resolution, "ready", False):
            visible_copy = self._blocked_visible_copy(getattr(resolution, "visible_copy", ""))
            return self._block(session_id, visible_copy)

        provider_messages = self._provider_messages_through_message(session_id, message_id)
        assistant = self.store.append_message(
            session_id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            persist=self.store.persistence is not None,
        )
        return await self._stream_assistant_response(
            resolution=resolution,
            provider_messages=provider_messages,
            assistant_message_id=assistant.id,
        )

    async def regenerate_message(self, message_id: str) -> ConsoleSubmitResult:
        """Regenerate a selected assistant message into a newly selected variant."""
        active_rejection = self._active_run_rejection()
        if active_rejection is not None:
            return active_rejection

        session_id = self.store.active_session_id
        if session_id is None:
            return ConsoleSubmitResult(False, False, "No active Console session.")
        message = self.store.get_message(message_id)
        if message.role is not ConsoleMessageRole.ASSISTANT:
            return self._block(session_id, "Only assistant messages can be regenerated.")
        if self.store.session_id_for_message(message_id) != session_id:
            visible_copy = "Open the original session before regenerating this message."
            self._set_run_state(ConsoleRunState.blocked(visible_copy))
            return ConsoleSubmitResult(False, False, visible_copy)

        self._set_run_state(ConsoleRunState(ConsoleRunStatus.VALIDATING, "Validating provider."))
        resolution = await self.provider_gateway.resolve_for_send(self._provider_selection())
        if not getattr(resolution, "ready", False):
            visible_copy = self._blocked_visible_copy(getattr(resolution, "visible_copy", ""))
            return self._block(session_id, visible_copy)

        provider_messages = self._provider_messages_for_session(
            session_id,
            before_message_id=message_id,
        )
        self._set_run_state(ConsoleRunState(ConsoleRunStatus.STREAMING, "Regenerating response."))
        chunks: list[str] = []
        try:
            async for chunk in self.provider_gateway.stream_chat(resolution, provider_messages):
                if chunk:
                    chunks.append(chunk)
        except Exception as exc:
            visible_copy = f"Provider stream failed: {exc}"
            self._set_run_state(ConsoleRunState(ConsoleRunStatus.FAILED, visible_copy))
            return ConsoleSubmitResult(True, True, visible_copy)

        content = "".join(chunks)
        if not content:
            visible_copy = "Provider stream ended without content."
            self._set_run_state(ConsoleRunState(ConsoleRunStatus.FAILED, visible_copy))
            return ConsoleSubmitResult(True, True, visible_copy)

        updated = self.store.add_variant(message_id, content)
        self._set_run_state(ConsoleRunState(ConsoleRunStatus.COMPLETED, "Response regenerated."))
        return ConsoleSubmitResult(True, True, updated.content)

    def _provider_selection(self) -> ConsoleProviderSelection:
        return ConsoleProviderSelection(
            provider=self.provider,
            base_url=self.base_url,
            explicit_model=self.model,
            configured_model=self.configured_model,
            workspace_context=self.store.workspace_context,
        )

    @staticmethod
    def _validated_draft(draft: str) -> tuple[str, str | None]:
        raw_draft = str(draft or "")
        if not raw_draft.strip():
            return "", "Type a message before sending."
        if not validate_text_input(
            raw_draft,
            max_length=MAX_CONSOLE_DRAFT_LENGTH,
            allow_html=False,
        ):
            return "", "Message blocked: remove unsafe markup or shorten your message."
        clean_draft = sanitize_string(raw_draft, max_length=MAX_CONSOLE_DRAFT_LENGTH)
        if not clean_draft.strip():
            return "", "Type a message before sending."
        return clean_draft, None

    @staticmethod
    def _blocked_visible_copy(copy: str) -> str:
        if "Provider blocked" in copy:
            return copy
        if copy.startswith("WIP:"):
            return f"Provider blocked: {copy}"
        return copy or "Provider blocked."

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
        self._active_stream_task = asyncio.current_task()
        self._stop_requested = False
        self._set_run_state(ConsoleRunState(ConsoleRunStatus.STREAMING, "Streaming response."))
        retry_prepared = False
        emitted_content = False
        try:
            async for chunk in self.provider_gateway.stream_chat(resolution, provider_messages):
                if not chunk:
                    continue
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
                if chunk:
                    emitted_content = True
            if self._stop_requested:
                stopped = (
                    self.store.mark_message_stopped(assistant_message_id)
                    if not prepare_retry or retry_prepared
                    else self.store.get_message(assistant_message_id)
                )
                self._set_run_state(ConsoleRunState(ConsoleRunStatus.STOPPED, "Response stopped."))
                return ConsoleSubmitResult(True, True, stopped.content)
            if not emitted_content:
                failed = self.store.get_message(assistant_message_id)
                self._set_run_state(
                    ConsoleRunState(
                        ConsoleRunStatus.FAILED,
                        "Provider stream ended without content.",
                    )
                )
                if not prepare_retry:
                    failed = self.store.mark_message_failed(assistant_message_id)
                return ConsoleSubmitResult(True, True, failed.content)
            completed = self.store.mark_message_complete(assistant_message_id)
            self._set_run_state(ConsoleRunState(ConsoleRunStatus.COMPLETED, "Response complete."))
            return ConsoleSubmitResult(True, True, completed.content)
        except asyncio.CancelledError:
            if self._stop_requested:
                stopped = (
                    self.store.mark_message_stopped(assistant_message_id)
                    if not prepare_retry or retry_prepared
                    else self.store.get_message(assistant_message_id)
                )
                self._set_run_state(ConsoleRunState(ConsoleRunStatus.STOPPED, "Response stopped."))
                return ConsoleSubmitResult(True, True, stopped.content)
            raise
        except Exception as exc:
            visible_copy = f"Provider stream failed: {exc}"
            if not prepare_retry or retry_prepared:
                try:
                    self.store.append_stream_chunk(assistant_message_id, f"\n{visible_copy}")
                except ValueError:
                    pass
            failed = (
                self.store.mark_message_failed(assistant_message_id)
                if not prepare_retry or retry_prepared
                else self.store.get_message(assistant_message_id)
            )
            self._set_run_state(ConsoleRunState(ConsoleRunStatus.FAILED, visible_copy))
            return ConsoleSubmitResult(True, True, failed.content)
        finally:
            if self._active_stream_task is asyncio.current_task():
                self._active_assistant_message_id = None
                self._active_stream_task = None
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

    def _provider_messages_through_message(
        self,
        session_id: str,
        message_id: str,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for message in self.store.messages_for_session(session_id):
            content = (
                message.variants.current.content
                if message.variants is not None
                else message.content
            )
            if content and message.role in {ConsoleMessageRole.USER, ConsoleMessageRole.ASSISTANT}:
                messages.append({"role": message.role.value, "content": content})
            if message.id == message_id:
                break
        return messages

    def _set_run_state(self, run_state: ConsoleRunState) -> None:
        self.run_state = run_state
        self.run_state_history.append(run_state.status)

    def _active_run_rejection(self) -> ConsoleSubmitResult | None:
        if self.run_state.is_send_allowed:
            return None
        return ConsoleSubmitResult(
            accepted=False,
            should_clear_draft=False,
            visible_copy="A Console run is already running.",
        )
