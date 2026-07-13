"""Native Console chat controller for send, stream, stop, and retry flows."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from tldw_chatbook.Chat.attachment_core import (
    image_content_parts,
    max_history_images,
    vision_block_reason,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleProviderSelection,
    ConsoleRunState,
    ConsoleRunStatus,
    derive_console_session_title,
    is_default_console_session_title,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession, ConsoleChatStore
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Utils.input_validation import sanitize_string, validate_text_input
from tldw_chatbook.model_capabilities import is_vision_capable


MAX_CONSOLE_DRAFT_LENGTH = 100_000
CONSOLE_CONTINUE_INSTRUCTION = "Continue and extend the selected message."


def describe_stream_failure(exc: BaseException) -> str:
    """Return user-facing copy classifying a provider stream failure.

    ``str(exc)`` alone can be empty (observed live as ``"Provider stream
    failed: "`` rendering ``"[failed]"``), so the failure class is always
    included in user terms: timeout vs connection vs HTTP status.

    Args:
        exc: The exception raised by the provider stream.

    Returns:
        A short, user-readable failure description that is never empty.
    """
    detail = str(exc).strip()
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None) or getattr(exc, "status_code", None)
    exc_name = type(exc).__name__
    lowered_name = exc_name.lower()

    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or "timeout" in lowered_name:
        summary = "request timed out waiting for the provider"
    elif isinstance(exc, ConnectionRefusedError) or "connectrefused" in lowered_name.replace("_", ""):
        summary = "connection refused - is the provider server running?"
    elif isinstance(exc, ConnectionError) or "connect" in lowered_name:
        summary = "could not connect to the provider"
    elif status_code is not None:
        summary = f"provider returned HTTP {status_code}"
    else:
        summary = f"{exc_name} error"

    if detail and detail.lower() != summary.lower():
        return f"{summary} ({detail})"
    return summary


class ConsoleProviderGatewayProtocol(Protocol):
    """Provider gateway surface required by the Console controller."""

    async def resolve_for_send(self, selection: ConsoleProviderSelection) -> Any:
        """Resolve provider readiness for a send."""

    async def stream_chat(self, resolution: Any, messages: list[dict[str, Any]]) -> Any:
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
        temperature: float | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        verbosity: str | None = None,
        thinking_effort: str | None = None,
        thinking_budget_tokens: int | None = None,
        streaming: bool = True,
    ) -> None:
        self.store = store
        self.provider_gateway = provider_gateway
        self.provider = provider
        self.model = model
        self.configured_model = configured_model
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p
        self.min_p = min_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.seed = seed
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        self.verbosity = verbosity
        self.thinking_effort = thinking_effort
        self.thinking_budget_tokens = thinking_budget_tokens
        self.streaming = streaming
        self.run_state = ConsoleRunState()
        self.run_state_history: list[ConsoleRunStatus] = [self.run_state.status]
        #: Optional owner hook invoked once a submit is accepted (user message
        #: persisted, run about to start) so the composer can clear immediately
        #: instead of holding the sent text for the whole run.
        self.on_submission_accepted: Callable[[], None] | None = None
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
        pending = self.store.pending_attachment(session.id)
        pending_image = (
            pending
            if pending is not None
            and pending.insert_mode == "attachment"
            and pending.data is not None
            else None
        )
        clean_draft, validation_error = self._validated_draft(
            draft, allow_empty=pending_image is not None
        )
        if validation_error is not None:
            return self._block(session.id, validation_error)
        if pending_image is not None:
            vision_model = self.model or self.configured_model
            model_is_vision_capable = bool(vision_model) and is_vision_capable(
                self.provider, vision_model or ""
            )
            if not model_is_vision_capable:
                block_reason = vision_block_reason(self.provider, vision_model)
                if block_reason is not None:
                    return self._block(session.id, block_reason)
        if self.store.workspace_context.has_policy_blocks:
            return self._block(session.id, self.store.workspace_context.recovery_copy)

        self._set_run_state(ConsoleRunState(ConsoleRunStatus.VALIDATING, "Validating provider."))
        resolution = await self.provider_gateway.resolve_for_send(self._provider_selection())
        if not getattr(resolution, "ready", False):
            visible_copy = self._blocked_visible_copy(getattr(resolution, "visible_copy", ""))
            return self._block(session.id, visible_copy)

        self._maybe_auto_title_session(session, clean_draft)
        self.store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content=clean_draft,
            image_data=pending_image.data if pending_image is not None else None,
            image_mime_type=pending_image.mime_type if pending_image is not None else None,
            attachment_label=pending_image.label if pending_image is not None else None,
            persist=self.store.persistence is not None,
        )
        if pending_image is not None:
            self.store.clear_pending_attachment(session.id)
        self._notify_submission_accepted()
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

    def new_session(
        self,
        *,
        title: str | None = None,
        settings: ConsoleSessionSettings | None = None,
    ) -> ConsoleChatSession:
        """Create and activate a new native Console session."""
        next_number = len(self.store.sessions()) + 1
        session = self.store.create_session(
            title=title or f"Chat {next_number}",
            settings=settings,
        )
        self._clear_terminal_run_state()
        return session

    def _maybe_auto_title_session(self, session: ConsoleChatSession, draft: str) -> None:
        """Title a default-named session from its first accepted message."""
        if session.persisted_conversation_id is not None:
            return
        if not is_default_console_session_title(session.title):
            return
        derived = derive_console_session_title(draft)
        if derived:
            self.store.rename_session(session.id, derived)

    def update_provider_selection(self, selection: ConsoleProviderSelection) -> None:
        """Sync controller provider settings from a Console selection."""
        previous_selection = (
            self.provider,
            self.model,
            self.configured_model,
            self.base_url,
            self.temperature,
            self.top_p,
            self.min_p,
            self.top_k,
            self.max_tokens,
            self.seed,
            self.presence_penalty,
            self.frequency_penalty,
            self.reasoning_effort,
            self.reasoning_summary,
            self.verbosity,
            self.thinking_effort,
            self.thinking_budget_tokens,
            self.streaming,
        )
        self.provider = selection.provider
        self.model = selection.explicit_model
        self.configured_model = selection.configured_model
        self.base_url = selection.base_url
        self.temperature = selection.temperature
        self.top_p = selection.top_p
        self.min_p = selection.min_p
        self.top_k = selection.top_k
        self.max_tokens = selection.max_tokens
        self.seed = selection.seed
        self.presence_penalty = selection.presence_penalty
        self.frequency_penalty = selection.frequency_penalty
        self.reasoning_effort = selection.reasoning_effort
        self.reasoning_summary = selection.reasoning_summary
        self.verbosity = selection.verbosity
        self.thinking_effort = selection.thinking_effort
        self.thinking_budget_tokens = selection.thinking_budget_tokens
        self.streaming = selection.streaming
        current_selection = (
            self.provider,
            self.model,
            self.configured_model,
            self.base_url,
            self.temperature,
            self.top_p,
            self.min_p,
            self.top_k,
            self.max_tokens,
            self.seed,
            self.presence_penalty,
            self.frequency_penalty,
            self.reasoning_effort,
            self.reasoning_summary,
            self.verbosity,
            self.thinking_effort,
            self.thinking_budget_tokens,
            self.streaming,
        )
        if current_selection != previous_selection:
            self._clear_terminal_run_state()

    def switch_session(self, session_id: str) -> ConsoleChatSession:
        """Activate an existing native Console session."""
        session = self.store.switch_session(session_id)
        self._clear_terminal_run_state()
        return session

    def close_session(self, session_id: str) -> ConsoleChatSession | None:
        """Close an existing native Console session.

        Args:
            session_id: Native Console session ID to close.

        Returns:
            The session activated after closing, or ``None`` when no sessions remain.
        """
        if self._active_stream_belongs_to_session(session_id):
            self._stop_requested = True
            if (
                self._active_stream_task is not None
                and self._active_stream_task is not asyncio.current_task()
            ):
                self._active_stream_task.cancel()
            self._set_run_state(
                ConsoleRunState(ConsoleRunStatus.STOPPED, "Session closed.")
            )
        return self.store.close_session(session_id)

    def stop_active_run(self) -> bool:
        """Request the active stream to stop at the next safe boundary."""
        if self.run_state.status is not ConsoleRunStatus.STREAMING:
            assistant_message_id = self._active_streaming_assistant_message_id()
            if assistant_message_id is None:
                return False
        else:
            assistant_message_id = (
                self._active_assistant_message_id
                or self._active_streaming_assistant_message_id()
            )
        if assistant_message_id is None:
            return False
        self._stop_requested = True
        self._mark_stream_stopped(
            assistant_message_id,
            visible_copy="Response stopped.",
        )
        if self._active_stream_task is not None and self._active_stream_task is not asyncio.current_task():
            self._active_stream_task.cancel()
        return True

    async def shutdown(self) -> None:
        """Stop and await the active stream task before owner teardown."""
        task = self._active_stream_task
        if task is None:
            return
        if not self.stop_active_run():
            self._stop_requested = True
            if task is not asyncio.current_task():
                task.cancel()
        if task is asyncio.current_task():
            return
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            # Shutdown is a teardown path; stale task failures should not crash owner cleanup.
            pass
        finally:
            if self._active_stream_task is task:
                self._active_assistant_message_id = None
                self._active_stream_task = None
                self._stop_requested = False

    def _active_streaming_assistant_message_id(self) -> str | None:
        """Return the visible streaming assistant message for the active session."""
        session_id = self.store.active_session_id
        if session_id is None:
            return None
        try:
            messages = self.store.messages_for_session(session_id)
        except KeyError:
            return None
        for message in reversed(messages):
            if message.role is ConsoleMessageRole.ASSISTANT and message.status == "streaming":
                return message.id
        return None

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
        self._ensure_user_continuation_instruction(provider_messages)
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
        self._ensure_user_continuation_instruction(provider_messages)
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
        self._ensure_user_continuation_instruction(provider_messages)
        self._set_run_state(ConsoleRunState(ConsoleRunStatus.STREAMING, "Regenerating response."))
        chunks: list[str] = []
        try:
            async for chunk in self.provider_gateway.stream_chat(resolution, provider_messages):
                if chunk:
                    chunks.append(chunk)
        except Exception as exc:
            visible_copy = f"Provider stream failed: {describe_stream_failure(exc)}"
            self._append_failure_system_row(session_id, visible_copy)
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
            temperature=self.temperature,
            top_p=self.top_p,
            min_p=self.min_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            seed=self.seed,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            reasoning_effort=self.reasoning_effort,
            reasoning_summary=self.reasoning_summary,
            verbosity=self.verbosity,
            thinking_effort=self.thinking_effort,
            thinking_budget_tokens=self.thinking_budget_tokens,
            streaming=self.streaming,
            workspace_context=self.store.workspace_context,
        )

    @staticmethod
    def _ensure_user_continuation_instruction(
        provider_messages: list[dict[str, Any]],
    ) -> None:
        if (
            provider_messages
            and provider_messages[-1].get("role") == ConsoleMessageRole.ASSISTANT.value
        ):
            provider_messages.append(
                {"role": ConsoleMessageRole.USER.value, "content": CONSOLE_CONTINUE_INSTRUCTION}
            )

    @staticmethod
    def _validated_draft(draft: str, *, allow_empty: bool = False) -> tuple[str, str | None]:
        raw_draft = str(draft or "")
        if not raw_draft.strip():
            if allow_empty:
                return "", None
            return "", "Type a message before sending."
        if not validate_text_input(
            raw_draft,
            max_length=MAX_CONSOLE_DRAFT_LENGTH,
            allow_html=False,
        ):
            return "", "Message blocked: remove unsafe markup or shorten your message."
        clean_draft = sanitize_string(raw_draft, max_length=MAX_CONSOLE_DRAFT_LENGTH)
        if not clean_draft.strip():
            if allow_empty:
                return "", None
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

    def _notify_submission_accepted(self) -> None:
        """Invoke the owner accepted-hook without letting UI errors kill the run."""
        callback = self.on_submission_accepted
        if callback is None:
            return
        try:
            callback()
        except Exception:
            # The hook is a UI convenience (composer clearing); a failure there
            # must never abort an already-accepted provider run.
            pass

    def _append_failure_system_row(self, session_id: str, visible_copy: str) -> None:
        """Append a transcript-only system row describing a provider failure."""
        try:
            self.store.append_message(
                session_id,
                role=ConsoleMessageRole.SYSTEM,
                content=visible_copy,
            )
        except KeyError:
            # Session vanished mid-failure (e.g. closed); the run-state copy
            # still carries the failure for the control surfaces.
            pass

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
                    try:
                        stopped = self._mark_stream_stopped(
                            assistant_message_id,
                            visible_copy="Response stopped.",
                            prepare_retry=prepare_retry,
                            retry_prepared=retry_prepared,
                        )
                    except KeyError:
                        return self._session_closed_result()
                    return ConsoleSubmitResult(True, True, stopped.content)
                if prepare_retry and not retry_prepared:
                    self.store.prepare_message_retry(assistant_message_id)
                    retry_prepared = True
                try:
                    self.store.append_stream_chunk(assistant_message_id, chunk)
                except KeyError:
                    return self._session_closed_result()
                if chunk:
                    emitted_content = True
            if self._stop_requested:
                try:
                    stopped = self._mark_stream_stopped(
                        assistant_message_id,
                        visible_copy="Response stopped.",
                        prepare_retry=prepare_retry,
                        retry_prepared=retry_prepared,
                    )
                except KeyError:
                    return self._session_closed_result()
                return ConsoleSubmitResult(True, True, stopped.content)
            if not emitted_content:
                try:
                    failed = self.store.get_message(assistant_message_id)
                except KeyError:
                    return self._session_closed_result()
                self._set_run_state(
                    ConsoleRunState(
                        ConsoleRunStatus.FAILED,
                        "Provider stream ended without content.",
                    )
                )
                if not prepare_retry:
                    try:
                        failed = self.store.mark_message_failed(assistant_message_id)
                    except KeyError:
                        return self._session_closed_result()
                return ConsoleSubmitResult(True, True, failed.content)
            try:
                completed = self.store.mark_message_complete(assistant_message_id)
            except KeyError:
                return self._session_closed_result()
            self._set_run_state(
                ConsoleRunState(ConsoleRunStatus.COMPLETED, "Response complete.")
            )
            return ConsoleSubmitResult(True, True, completed.content)
        except asyncio.CancelledError:
            if self._stop_requested:
                try:
                    stopped = self._mark_stream_stopped(
                        assistant_message_id,
                        visible_copy="Response stopped.",
                        prepare_retry=prepare_retry,
                        retry_prepared=retry_prepared,
                    )
                except KeyError:
                    return self._session_closed_result()
                return ConsoleSubmitResult(True, True, stopped.content)
            raise
        except Exception as exc:
            # Provider failures are surfaced as run status plus a transcript
            # system row; they must never be written into assistant message
            # content, which is persisted and replayed as model context.
            visible_copy = f"Provider stream failed: {describe_stream_failure(exc)}"
            try:
                if not prepare_retry or retry_prepared:
                    self.store.mark_message_failed(assistant_message_id)
                else:
                    self.store.get_message(assistant_message_id)
            except KeyError:
                return self._session_closed_result()
            try:
                session_id = self.store.session_id_for_message(assistant_message_id)
            except KeyError:
                session_id = None
            if session_id is not None:
                self._append_failure_system_row(session_id, visible_copy)
            self._set_run_state(ConsoleRunState(ConsoleRunStatus.FAILED, visible_copy))
            return ConsoleSubmitResult(True, True, visible_copy)
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
    ) -> list[dict[str, Any]]:
        collected: list[ConsoleChatMessage] = []
        for message in self.store.messages_for_session(session_id):
            if message.id == before_message_id:
                break
            collected.append(message)
        return self._provider_message_payloads(collected, skip_failed=True)

    def _provider_messages_through_message(
        self,
        session_id: str,
        message_id: str,
    ) -> list[dict[str, Any]]:
        collected: list[ConsoleChatMessage] = []
        for message in self.store.messages_for_session(session_id):
            collected.append(message)
            if message.id == message_id:
                break
        return self._provider_message_payloads(
            collected, skip_failed=False, use_variant_content=True
        )

    def _provider_message_payloads(
        self,
        session_messages: list[ConsoleChatMessage],
        *,
        skip_failed: bool,
        use_variant_content: bool = False,
    ) -> list[dict[str, Any]]:
        model = self.model or self.configured_model
        vision = bool(model) and is_vision_capable(self.provider, model or "")
        image_budget = max_history_images(self.provider, model)
        image_ids = [
            message.id
            for message in session_messages
            if message.role is ConsoleMessageRole.USER and message.image_data is not None
        ]
        allowed_image_ids = set(image_ids[-image_budget:]) if vision else set()
        payloads: list[dict[str, Any]] = []
        for message in session_messages:
            if message.role not in {ConsoleMessageRole.USER, ConsoleMessageRole.ASSISTANT}:
                continue
            if skip_failed and message.status == "failed":
                continue
            text = (
                message.variants.current.content
                if use_variant_content and message.variants is not None
                else message.content
            )
            if (
                message.id in allowed_image_ids
                and message.image_data is not None
                and message.image_mime_type
            ):
                payloads.append(
                    {
                        "role": message.role.value,
                        "content": image_content_parts(
                            text, message.image_data, message.image_mime_type
                        ),
                    }
                )
                continue
            if not text:
                continue
            payloads.append({"role": message.role.value, "content": text})
        return payloads

    def _mark_stream_stopped(
        self,
        assistant_message_id: str,
        *,
        visible_copy: str,
        prepare_retry: bool = False,
        retry_prepared: bool = True,
    ) -> ConsoleChatMessage:
        """Mark a streaming assistant message stopped, tolerating an earlier stop request."""
        if prepare_retry and not retry_prepared:
            stopped = self.store.get_message(assistant_message_id)
        else:
            try:
                stopped = self.store.mark_message_stopped(assistant_message_id)
            except ValueError:
                stopped = self.store.get_message(assistant_message_id)
                if stopped.status != "stopped":
                    raise
        self._set_run_state(ConsoleRunState(ConsoleRunStatus.STOPPED, visible_copy))
        return stopped

    def _set_run_state(self, run_state: ConsoleRunState) -> None:
        self.run_state = run_state
        self.run_state_history.append(run_state.status)

    def _clear_terminal_run_state(self) -> None:
        """Clear stale terminal status copy when the active session changes."""
        if self.run_state.status in {
            ConsoleRunStatus.BLOCKED,
            ConsoleRunStatus.COMPLETED,
            ConsoleRunStatus.FAILED,
            ConsoleRunStatus.STOPPED,
        }:
            self._set_run_state(ConsoleRunState())

    def _active_stream_belongs_to_session(self, session_id: str) -> bool:
        if self._active_assistant_message_id is None:
            return False
        try:
            return self.store.session_id_for_message(self._active_assistant_message_id) == session_id
        except KeyError:
            return False

    def _session_closed_result(self) -> ConsoleSubmitResult:
        visible_copy = "Session closed."
        self._set_run_state(ConsoleRunState(ConsoleRunStatus.STOPPED, visible_copy))
        return ConsoleSubmitResult(True, True, visible_copy)

    def _active_run_rejection(self) -> ConsoleSubmitResult | None:
        if self.run_state.is_send_allowed:
            return None
        return ConsoleSubmitResult(
            accepted=False,
            should_clear_draft=False,
            visible_copy="A Console run is already running.",
        )
