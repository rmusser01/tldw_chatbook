"""Native Console chat controller for send, stream, stop, and retry flows."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol

from tldw_chatbook.Chat.attachment_core import (
    image_url_part,
    max_history_images,
    vision_block_reason,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleProviderSelection,
    ConsoleRunState,
    ConsoleRunStatus,
    MessageAttachment,
    derive_console_session_title,
    is_default_console_session_title,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession, ConsoleChatStore
from tldw_chatbook.Chat.console_command_grammar import COMMAND_PREFIX
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_skill_resolver import (
    SKILL_UNTRUSTED_REFUSE,
    SkillCommandCandidate,
    cap_skill_args,
    resolve_skill_command,
)
from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError
from tldw_chatbook.Utils.input_validation import sanitize_string, validate_text_input
from tldw_chatbook.model_capabilities import is_vision_capable

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge


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


def _split_skill_command_word(text: str) -> tuple[str, str]:
    """Split a ``/word rest`` string into its leading token and the remainder.

    Mirrors ``console_command_grammar._split_leading_token``'s single-
    whitespace-character split rule. That helper is module-private (by
    design -- callers own their own tokenization per its module docstring),
    so this is a deliberate small duplicate rather than an import, the same
    precedent ``chat_screen.ChatScreen._split_console_skill_name_args``
    already follows. ``text`` is assumed to already start with
    `COMMAND_PREFIX`.
    """
    for index, character in enumerate(text):
        if character.isspace():
            return text[:index], text[index + 1 :]
    return text, ""


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
        system_prompt: str | None = None,
        agent_bridge: "ConsoleAgentBridge | None" = None,
        agent_runtime_enabled: bool = True,
        skills_service: Any | None = None,
        skill_substitution_enabled: bool = True,
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
        self.system_prompt = system_prompt
        self._agent_bridge = agent_bridge
        self._agent_runtime_enabled = agent_runtime_enabled
        self._skills_service = skills_service
        self._skill_substitution_enabled = skill_substitution_enabled
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
        pendings = self.store.pending_attachments(session.id)
        attachment_mode_pendings = [
            pending
            for pending in pendings
            if pending.insert_mode == "attachment" and pending.data is not None
        ]
        has_pending_attachment = bool(attachment_mode_pendings)
        clean_draft, validation_error = self._validated_draft(
            draft, allow_empty=has_pending_attachment
        )
        if validation_error is not None:
            return self._block(session.id, validation_error)
        if has_pending_attachment:
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
        staged_attachments = tuple(
            MessageAttachment(
                data=pending.data,
                mime_type=pending.mime_type or "image/png",
                display_name=pending.display_name,
                position=index,
            )
            for index, pending in enumerate(attachment_mode_pendings)
        )
        self.store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content=clean_draft,
            attachments=staged_attachments,
            persist=self.store.persistence is not None,
        )
        if pendings:
            self.store.clear_pending_attachments(session.id)
        provider_messages = self._provider_messages_for_session(session.id)
        provider_messages, refuse = await self._apply_skill_substitution(provider_messages)
        if refuse is not None:
            return self._block(session.id, refuse)
        # The accepted-hook fires only once the turn is confirmed to
        # actually proceed (Qodo finding 3, PR #636 bot review): it used to
        # fire right after the USER row was appended, BEFORE this skill
        # substitution/trust check ran. In the real ChatScreen, this hook
        # is the sole consume point for a staged resolved-skill "driving
        # this turn" TOOL marker (see `_on_console_submission_accepted`'s
        # own docstring) -- firing it before a substitution refusal meant
        # a refused/untrusted skill submit still consumed and appended
        # that marker, claiming the skill drove the turn right before the
        # refuse row that says it never ran. A substitution refusal is a
        # `_block()` outcome exactly like any other (provider not ready,
        # policy block, validation failure) and those already never reach
        # this hook -- this reorder just extends that same rule to cover
        # it too.
        self._notify_submission_accepted()
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
            self.system_prompt,
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
        self.system_prompt = selection.system_prompt
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
            self.system_prompt,
        )
        if current_selection != previous_selection:
            self._clear_terminal_run_state()

    def update_agent_runtime(
        self, *, enabled: bool, bridge: "ConsoleAgentBridge | None"
    ) -> None:
        """Refresh the agent-runtime gate and bridge from a fresh config read.

        Both were previously read only once, at controller construction
        (Plan-B Task 6 Important 3): the ``[console] agent_runtime``
        kill-switch is meant to take effect on the next send, but a
        controller built before a config change stayed on its original
        path until the owning screen tore it down. The owner must call
        this every time it refreshes provider selection (see
        ``update_provider_selection``) so the gate and bridge presence
        never go stale.
        """
        self._agent_runtime_enabled = enabled
        self._agent_bridge = bridge

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
        provider_messages, refuse = await self._apply_skill_substitution(provider_messages)
        if refuse is not None:
            return self._block(session_id, refuse)
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
        provider_messages, refuse = await self._apply_skill_substitution(provider_messages)
        if refuse is not None:
            return self._block(session_id, refuse)
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
        provider_messages, refuse = await self._apply_skill_substitution(provider_messages)
        if refuse is not None:
            return self._block(session_id, refuse)
        return await self._stream_assistant_response(
            resolution=resolution,
            provider_messages=provider_messages,
            assistant_message_id=message_id,
            variant_mode=True,
        )

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
            system_prompt=self.system_prompt,
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

    async def _apply_skill_substitution(
        self, provider_messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Render-fresh the triggering turn's skill command at payload build time.

        Spec: "Invocation semantics" §5 (the substitution rule) -- one rule
        for fresh sends AND retry/regenerate/continue. Only the FINAL
        ``role == "user"`` message in ``provider_messages`` (the turn
        actually driving this send) is ever a substitution candidate; every
        earlier message -- including an earlier raw skill command sitting
        in history -- is left untouched, so the persisted transcript always
        keeps the literal text the user typed (the raw command is what gets
        submitted and stored; only the ephemeral provider payload for this
        turn is ever rendered). Re-resolves against a FRESH candidate
        snapshot and re-verifies trust through ``execute_skill`` on every
        call (never a cached snapshot), so a retry issued after a skill was
        edited (now untrusted) refuses instead of silently re-running a
        stale render.

        Args:
            provider_messages: The fully-built payload about to be sent to
                the provider (already includes any leading session-system
                message and any synthesized continuation instruction).

        Returns:
            ``(provider_messages, None)`` unchanged when there is no skills
            service configured, substitution is disabled, there is no final
            user message, or that message's content does not resolve to a
            known skill command (not a string, doesn't start with
            `COMMAND_PREFIX`, or `resolve_skill_command` doesn't return
            ``"resolved"``). ``(new_messages, None)`` when a skill resolves
            and renders: ``inline`` replaces just the final message in
            place (history preserved); ``fork`` drops every message before
            it except a leading ``role == "system"`` message (clean context
            = session system prompt + rendered turn only). ``(provider_
            messages, refuse_copy)`` -- the ORIGINAL, unmodified messages,
            paired with `SKILL_UNTRUSTED_REFUSE` copy -- when the resolved
            skill is no longer trusted (`SkillTrustBlockedError` at
            execute-time); the caller must append `refuse_copy` as a system
            row and abort the turn without sending.
        """
        if self._skills_service is None or not self._skill_substitution_enabled:
            return provider_messages, None

        final_index: int | None = None
        for index in range(len(provider_messages) - 1, -1, -1):
            if provider_messages[index].get("role") == ConsoleMessageRole.USER.value:
                final_index = index
                break
        if final_index is None:
            return provider_messages, None

        content = provider_messages[final_index].get("content")
        if not isinstance(content, str) or not content.startswith(COMMAND_PREFIX):
            return provider_messages, None

        word, rest = _split_skill_command_word(content)
        name = word[len(COMMAND_PREFIX) :]
        if not name:
            return provider_messages, None

        context = await self._skills_service.get_context(mode="local")
        candidates = self._skill_candidates_from_context(context)
        resolution = resolve_skill_command(name, rest, candidates)
        if resolution.kind != "resolved":
            return provider_messages, None

        args = cap_skill_args(rest)
        try:
            result = await self._skills_service.execute_skill(
                resolution.name, mode="local", args=args
            )
        except SkillTrustBlockedError as exc:
            refuse = SKILL_UNTRUSTED_REFUSE.format(name=resolution.name, reason=exc.reason_code)
            return provider_messages, refuse

        rendered = result.get("rendered_prompt", "") if isinstance(result, Mapping) else ""
        rendered_message = {"role": ConsoleMessageRole.USER.value, "content": rendered}
        execution_mode = result.get("execution_mode") if isinstance(result, Mapping) else None
        if execution_mode == "fork":
            leading = (
                [provider_messages[0]]
                if provider_messages
                and provider_messages[0].get("role") == ConsoleMessageRole.SYSTEM.value
                else []
            )
            return leading + [rendered_message], None

        new_messages = list(provider_messages)
        new_messages[final_index] = rendered_message
        return new_messages, None

    @staticmethod
    def _skill_candidates_from_context(
        context: Any,
    ) -> tuple[SkillCommandCandidate, ...]:
        """Build the user-invocable, trusted skill candidate population.

        Mirrors ``chat_screen.ChatScreen.
        _console_skill_trusted_candidates_from_context``'s filter -- kept as
        a small duplicate rather than a shared import because `Chat/`
        business logic must not depend on `UI/Screens/` (project layering),
        and `console_skill_resolver` deliberately stays unaware of trust/
        context shape (see its own module docstring).
        """
        available = context.get("available_skills") if isinstance(context, Mapping) else None
        return tuple(
            SkillCommandCandidate(
                name=str(item.get("name")),
                description=str(item.get("description") or ""),
            )
            for item in (available or [])
            if isinstance(item, Mapping)
            and item.get("name")
            and item.get("user_invocable", True)
            and not item.get("trust_blocked", False)
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
        variant_mode: bool = False,
    ) -> ConsoleSubmitResult:
        if self._agent_runtime_enabled and self._agent_bridge is not None:
            return await self._run_agent_reply(
                resolution=resolution,
                provider_messages=provider_messages,
                assistant_message_id=assistant_message_id,
                prepare_retry=prepare_retry,
                variant_mode=variant_mode,
            )
        self._active_assistant_message_id = assistant_message_id
        self._active_stream_task = asyncio.current_task()
        self._stop_requested = False
        if variant_mode:
            self.store.begin_variant_stream(assistant_message_id)
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
                if variant_mode:
                    completed = self.store.finalize_variant_stream(assistant_message_id)
                else:
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

    async def _run_agent_reply(
        self,
        *,
        resolution: Any,
        provider_messages: list[dict[str, Any]],
        assistant_message_id: str,
        prepare_retry: bool,
        variant_mode: bool,
    ) -> ConsoleSubmitResult:
        """Run the agent loop as the reply engine, streaming into the target row."""
        self._active_assistant_message_id = assistant_message_id
        self._active_stream_task = asyncio.current_task()
        self._stop_requested = False
        self._set_run_state(ConsoleRunState(ConsoleRunStatus.STREAMING, "Agent running."))
        try:
            session_id = self.store.session_id_for_message(assistant_message_id)
        except KeyError:
            return self._session_closed_result()
        if variant_mode:
            self.store.begin_variant_stream(assistant_message_id)
        elif prepare_retry:
            self.store.prepare_message_retry(assistant_message_id)

        # Split the leading session system message off the payload; the
        # agent config carries it (composed with the operating prompt).
        session_system_prompt = ""
        agent_messages = list(provider_messages)
        if agent_messages and agent_messages[0].get("role") == ConsoleMessageRole.SYSTEM.value:
            session_system_prompt = str(agent_messages[0].get("content", ""))
            agent_messages = agent_messages[1:]

        conversation_id = self._agent_conversation_id(session_id)
        should_cancel = lambda: self._stop_requested  # noqa: E731 — tiny closure

        # Swap site: the agent loop runs synchronously on a worker thread via
        # asyncio.to_thread, so Stop is cooperative-only -- `should_cancel` is
        # polled between chunks/steps inside the bridge, never preempts the
        # thread itself. A provider that hangs mid-request without emitting a
        # single chunk cannot be interrupted here; RunBudget.max_wall_seconds
        # (agent_models.py) is what bounds a run overall, but only once
        # control returns to a checkpoint the loop actually polls -- it is
        # not a hard timeout on an in-flight, zero-chunk provider call.
        try:
            outcome = await asyncio.to_thread(
                self._agent_bridge.run_reply,
                conversation_id=conversation_id,
                session_id=session_id,
                resolution=resolution,
                assistant_message_id=assistant_message_id,
                model=self.model or self.configured_model or "",
                session_system_prompt=session_system_prompt,
                agent_messages=agent_messages,
                should_cancel=should_cancel,
                supersede_previous=bool(prepare_retry or variant_mode),
            )
        except asyncio.CancelledError:
            if self._stop_requested:
                try:
                    stopped = self._mark_stream_stopped(
                        assistant_message_id, visible_copy="Response stopped.")
                except KeyError:
                    return self._session_closed_result()
                return ConsoleSubmitResult(True, True, stopped.content)
            raise
        except Exception as exc:
            # Bridge failures can originate OUTSIDE AgentService's own
            # narrow loop guard (agent_service.py wraps only
            # `run_agent_loop`; `db.create_run`, `_persist`
            # (append_steps/set_status), and `supersede_run_tree` are not
            # covered). Left uncaught here, run_state would stay STREAMING
            # forever and every future send on every session would be
            # rejected ("A Console run is already running.") until app
            # restart (Plan-B Task 6 Critical 1). Mirror the legacy stream
            # path's catch-all above, including the Task-1 variant-restore
            # semantics: `begin_variant_stream`/`prepare_message_retry`
            # already ran before the bridge call, so `mark_message_failed`
            # resolves the correct terminal content on its own (restores
            # the pre-regenerate base + status for a failed regenerate;
            # preserves whatever partial content already streamed
            # otherwise).
            visible_copy = f"Agent run failed: {describe_stream_failure(exc)}"
            try:
                self.store.mark_message_failed(assistant_message_id)
            except KeyError:
                return self._session_closed_result()
            self._append_failure_system_row(session_id, visible_copy)
            self._set_run_state(ConsoleRunState(ConsoleRunStatus.FAILED, visible_copy))
            return ConsoleSubmitResult(True, True, visible_copy)
        finally:
            if self._active_stream_task is asyncio.current_task():
                self._active_assistant_message_id = None
                self._active_stream_task = None
                self._stop_requested = False

        return self._finalize_agent_reply(
            assistant_message_id, session_id, outcome, variant_mode=variant_mode)

    def _agent_conversation_id(self, session_id: str) -> str:
        """Return the durable id the run store is keyed by (persisted id when set)."""
        for session in self.store.sessions():
            if session.id == session_id:
                return session.persisted_conversation_id or session_id
        return session_id

    def _finalize_agent_reply(
        self, assistant_message_id: str, session_id: str, outcome: Any,
        *, variant_mode: bool,
    ) -> ConsoleSubmitResult:
        from tldw_chatbook.Agents.agent_models import RUN_CANCELLED, RUN_DONE

        if outcome.status == RUN_CANCELLED:
            try:
                stopped = self._mark_stream_stopped(
                    assistant_message_id, visible_copy="Response stopped.")
            except KeyError:
                return self._session_closed_result()
            return ConsoleSubmitResult(True, True, stopped.content)

        if outcome.status != RUN_DONE:
            # RUN_ERROR/RUN_STUCK (and any other non-done outcome) are
            # failures, never a silent "complete" (Plan-B Task 6 Critical
            # 2): a failing regenerate must not clobber a good prior
            # answer with a fake "[agent error]" variant, and a failed
            # message must stay retryable and excluded from model context
            # (skip_failed=True). `mark_message_failed` carries the Task-1
            # variant-restore semantics on its own -- for a regenerate it
            # restores the pre-regenerate base content + status untouched;
            # for a plain send/retry it keeps whatever partial prose had
            # already streamed, matching legacy failure behavior.
            visible_copy = self._agent_failure_visible_copy(outcome)
            try:
                failed = self.store.mark_message_failed(assistant_message_id)
            except KeyError:
                return self._session_closed_result()
            self._append_failure_system_row(session_id, visible_copy)
            self._set_run_state(ConsoleRunState(ConsoleRunStatus.FAILED, visible_copy))
            return ConsoleSubmitResult(True, True, failed.content)

        try:
            if variant_mode:
                completed = self.store.finalize_variant_stream(assistant_message_id)
            else:
                completed = self.store.mark_message_complete(assistant_message_id)
        except KeyError:
            return self._session_closed_result()
        self._set_run_state(ConsoleRunState(ConsoleRunStatus.COMPLETED, "Response complete."))
        return ConsoleSubmitResult(True, True, completed.content)

    @staticmethod
    def _agent_failure_visible_copy(outcome: Any) -> str:
        """Return user-facing copy for a non-done agent outcome, naming the reason.

        ``RUN_STUCK`` in particular must read as visibly distinct from a
        generic failure -- it means the run hit a budget or loop-detection
        limit (agent_runtime.py), not a raw exception -- so the concrete
        reason recorded on the last ``STEP_ERROR`` step (e.g. "step budget
        exhausted", "wall-clock budget exhausted", "loop detected: ...") is
        surfaced when available.
        """
        from tldw_chatbook.Agents.agent_models import RUN_STUCK, STEP_ERROR

        reason = ""
        for step in reversed(getattr(outcome, "steps", None) or []):
            if getattr(step, "kind", None) == STEP_ERROR and getattr(step, "summary", ""):
                reason = step.summary
                break
        if outcome.status == RUN_STUCK:
            return f"Agent run stuck: {reason or 'budget or loop limit reached'}."
        return f"Agent run failed: {reason or outcome.status}."

    def _leading_system_message(self) -> list[dict[str, str]]:
        """Return a single-item system message list when a system prompt is set.

        Applies to every native Console send path (submit, retry, regenerate,
        continue) since they all build their provider payload by prepending
        this to the transcript-derived messages. Blank/whitespace-only prompts
        are treated as "no system prompt" (native Console default stays silent
        unless a user has explicitly set one for this session) -- ``strip()``
        is used ONLY for that emptiness check. The message content itself is
        ``self.system_prompt`` verbatim: leading/trailing whitespace and
        internal formatting (blank lines, indentation) are never altered, so
        a formatting-sensitive prompt reaches the provider unchanged.
        """
        raw_system_prompt = self.system_prompt
        if not isinstance(raw_system_prompt, str) or not raw_system_prompt.strip():
            return []
        return [{"role": ConsoleMessageRole.SYSTEM.value, "content": raw_system_prompt}]

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
        return self._leading_system_message() + self._provider_message_payloads(
            collected, skip_failed=True
        )

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
        return self._leading_system_message() + self._provider_message_payloads(
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

        # Reserve the image budget newest-message-first, counting IMAGES (not
        # messages): a message with several attachments can consume more than
        # one unit of budget, and the walk stops as soon as the budget is
        # exhausted regardless of how many messages remain.
        budget = max_history_images(self.provider, model) if vision else 0
        allowed_counts: dict[str, int] = {}
        for message in reversed(session_messages):
            if budget <= 0:
                break
            if message.role is not ConsoleMessageRole.USER:
                continue
            usable = [attachment for attachment in message.attachments if attachment.data is not None]
            if not usable:
                continue
            take = min(len(usable), budget)
            allowed_counts[message.id] = take
            budget -= take

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
            take = allowed_counts.get(message.id, 0)
            if take > 0:
                # Partially-budgeted messages emit their images in POSITION
                # order up to the reserved count (oldest-attached first),
                # not in reservation order.
                usable = [
                    attachment for attachment in message.attachments if attachment.data is not None
                ]
                parts: list[dict[str, Any]] = []
                if text:
                    parts.append({"type": "text", "text": text})
                for attachment in usable[:take]:
                    # An attachment can reach here with an empty mime_type
                    # (e.g. a resumed message whose persisted
                    # image_mime_type column was NULL --
                    # ``_console_messages_from_conversation_tree`` falls back
                    # to ``""`` for display purposes). Emitting a bare
                    # ``data:;base64,...`` URL produces an invalid data URI
                    # most providers reject outright, so fall back to the
                    # same default mime the send-time staging path already
                    # uses (see ``pending.mime_type or "image/png"`` above
                    # and ``ConsoleChatStore.append_message``).
                    parts.append(
                        image_url_part(attachment.data, attachment.mime_type or "image/png")
                    )
                payloads.append({"role": message.role.value, "content": parts})
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
        """Mark a streaming assistant message stopped, tolerating an earlier stop request.

        ``stop_active_run`` finalizes the message synchronously and then
        cancels the active stream task; that task's own ``CancelledError``
        handler in ``_stream_assistant_response`` calls this a second,
        redundant time. ``store.mark_message_stopped`` raises ``ValueError``
        for that redundant call because the message is no longer pending/
        streaming -- i.e. some earlier call already finalized it -- so any
        such error here is tolerated by simply reading back the
        already-finalized message rather than re-raising. Before Plan-B
        final-review Medium-2, the only reachable terminal status from this
        path was "stopped" itself; a mid-regenerate stop now legitimately
        settles the message at its pre-regenerate status instead (e.g.
        "complete"), so this must tolerate any terminal status, not just
        "stopped".
        """
        if prepare_retry and not retry_prepared:
            stopped = self.store.get_message(assistant_message_id)
        else:
            try:
                stopped = self.store.mark_message_stopped(assistant_message_id)
            except ValueError:
                stopped = self.store.get_message(assistant_message_id)
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
