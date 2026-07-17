"""Native Console chat controller for send, stream, stop, and retry flows."""

from __future__ import annotations

import asyncio
import threading
import time
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
from loguru import logger

from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider
from tldw_chatbook.config import get_cli_setting
from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError
from tldw_chatbook.Utils.input_validation import sanitize_string, validate_text_input
from tldw_chatbook.model_capabilities import is_vision_capable

if TYPE_CHECKING:
    from tldw_chatbook.Agents.agent_models import ToolCall
    from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge


#: Fallback used when no `mcp_approval_timeout_seconds` seam is injected --
#: mirrors `UnifiedMCPControlPlaneService.approval_timeout_seconds`'s own
#: default (task-201/T2), read directly here since the controller has no
#: dependency on that service (T6 wires the service into `MCPToolProvider`,
#: not into this controller).
_DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS = 120.0
#: Poll granularity for `request_mcp_approvals`'s wait loop (binding, from
#: the Phase-5 plan) -- also the worst-case slack added on top of a
#: configured timeout/cancellation before this method observes it.
_MCP_APPROVAL_POLL_SECONDS = 1.0


MAX_CONSOLE_DRAFT_LENGTH = 100_000
CONSOLE_CONTINUE_INSTRUCTION = "Continue and extend the selected message."


def build_mcp_review_hook(
    provider: MCPToolProvider,
    request_mcp_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
) -> Callable[[list["ToolCall"]], dict[str, str]]:
    """Build this run's T4 `review_tool_calls` hook for one composed MCP provider.

    Handed to `ConsoleAgentBridge.run_reply` (P5-T6), which forwards it
    straight through to `AgentService`/`LoopDeps.review_tool_calls` (T4):
    called ONCE per turn with the full batch of tool calls about to be
    dispatched, before any of them is invoked.

    For every call in the batch, `provider.pending_gate_for(name, args)`
    resolves whether it needs human gating (`None` for both "not an MCP
    call this provider owns" and "an MCP call whose current state doesn't
    need asking" -- `invoke()` re-resolves either case for itself, so
    this hook does not need to distinguish them). When at least one call
    needs asking, this makes exactly ONE `request_mcp_approvals` round
    trip for the whole batch (never one per call) and hands the resulting
    decisions to `provider.apply_batch_decisions` -- a per-turn stamp
    `invoke()` consumes on its very next call for that name.

    Design choice (binding, per the Phase-5 plan): this hook never
    returns a refusal string itself. Every MCP call it stamped is left to
    resolve through `invoke()`'s own gate on dispatch -- `invoke()`
    already handles every decision string uniformly (`approve_once`/
    `approve_session`/`always_allow` execute; `deny`/`timeout` refuse with
    the exact model-facing copy AND record the audit decision), so
    routing every decision through that ONE place keeps the refusal copy
    and the audit trail single-sourced instead of duplicating that logic
    here. The verdict map this hook returns therefore only ever contains
    `"proceed"` entries (for calls it gated this turn) -- purely
    documentary, since `run_agent_loop` already treats any name this hook
    doesn't mention as `"proceed"` by default; returning `{}` when nothing
    needed gating is exactly as correct as omitting entries would be.
    Non-MCP calls are untouched either way: `pending_gate_for` returns
    `None` for any name the provider doesn't own, so they never enter
    `pending` and are never mentioned in the returned map.

    Args:
        provider: This run's already-composed `MCPToolProvider` (P5-T6:
            built and `compose_catalog()`-ed by the caller on the main
            loop before the run's worker thread starts).
        request_mcp_approvals: The bound `ConsoleChatController.
            request_mcp_approvals` method for THIS run -- runs on the
            agent bridge's worker thread and blocks until the batch is
            decided, cancelled, or times out (T5).

    Returns:
        A `review_tool_calls`-shaped callable suitable for `LoopDeps`/
        `AgentService(review_tool_calls=...)`.
    """

    def review_tool_calls(calls: list) -> dict[str, str]:
        pending: list = []
        for call in calls:
            gate = provider.pending_gate_for(call.name, call.args)
            if gate is not None:
                pending.append(gate)
        if not pending:
            return {}
        decisions = request_mcp_approvals(pending)
        provider.apply_batch_decisions(decisions)
        return {call.llm_name: "proceed" for call in pending}

    return review_tool_calls


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
        chat_dictionary_applier: "Callable[[str | None, str], str] | None" = None,
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
        self._chat_dictionary_applier = chat_dictionary_applier
        self.run_state = ConsoleRunState()
        self.run_state_history: list[ConsoleRunStatus] = [self.run_state.status]
        #: Optional owner hook invoked once a submit is accepted (user message
        #: persisted, run about to start) so the composer can clear immediately
        #: instead of holding the sent text for the whole run.
        self.on_submission_accepted: Callable[[], None] | None = None
        self._active_assistant_message_id: str | None = None
        self._active_stream_task: asyncio.Task | None = None
        self._stop_requested = False
        #: Per-run cancellation flag for the agent bridge's background
        #: thread (see ``_run_agent_reply``). ``threading.Event`` rather
        #: than a shared bool: ``asyncio.to_thread`` survives Task
        #: cancellation (the coroutine detaches from the still-running OS
        #: thread), so the closure handed to that thread must observe a
        #: signal that, once set, is never reset for THIS run -- unlike
        #: ``_stop_requested``, which the run's own ``finally`` block
        #: resets as soon as the coroutine side is done (task-227).
        self._active_cancel_event: threading.Event | None = None

        # -- MCP batch-approval bridge (task-5) ------------------------------
        #: Textual App-like object exposing ``call_from_thread`` -- assigned
        #: by the owning screen (``ChatScreen._ensure_console_chat_
        #: controller``), mirroring how ``on_submission_accepted`` is wired.
        #: ``None`` (e.g. in most existing controller-only tests) makes
        #: ``request_mcp_approvals`` a safe no-op UI bridge that still
        #: resolves via cancellation/timeout.
        self.app: Any | None = None
        #: UI-thread callback that pushes/clears the pending-approval batch
        #: into the owning screen's task-resume state (``ChatScreen.
        #: _set_console_pending_approval``). Always invoked through
        #: ``self.app.call_from_thread`` from ``request_mcp_approvals``.
        self.set_pending_approval: Callable[[dict[str, Any] | None], None] | None = None
        #: Optional override for how long ``request_mcp_approvals`` waits
        #: for a human decision before failing every undecided call to
        #: ``"timeout"``. Defaults to reading ``[mcp] approval_timeout_
        #: seconds`` (T2's ``approval_timeout_seconds``) when unset.
        self.mcp_approval_timeout_seconds: Callable[[], float] | None = None
        #: The active batch-approval round's release signal + shared
        #: decisions holder, set for the duration of one ``request_mcp_
        #: approvals`` call (worker thread) and read/written from the UI
        #: thread by ``resolve_pending_approval`` /
        #: ``_deny_pending_approval_on_context_change``. ``None`` whenever
        #: no approval round is in flight.
        self._pending_approval_event: threading.Event | None = None
        self._pending_approval_decisions: dict[str, str] | None = None

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
            # ONE capability check decides the gate AND the copy: this
            # module's is_vision_capable (the documented monkeypatch seam) is
            # injected into vision_block_reason instead of being re-checked
            # around it — the two seams could otherwise disagree under test.
            block_reason = vision_block_reason(
                self.provider, vision_model, is_capable=is_vision_capable
            )
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
        provider_messages = await self._apply_chat_dictionaries(provider_messages, session.id)
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
        # Binding threading contract (task-5): a conversation switch denies
        # any pending MCP approval round rather than leaving its worker
        # thread blocked on a card the user just navigated away from. Only
        # one run (and therefore one approval round) can be active
        # controller-wide at a time (`_active_run_rejection` blocks a new
        # send while one is running), so this is unconditional -- a no-op
        # whenever no round is in flight.
        self._deny_pending_approval_on_context_change()
        return session

    def close_session(self, session_id: str) -> ConsoleChatSession | None:
        """Close an existing native Console session.

        Args:
            session_id: Native Console session ID to close.

        Returns:
            The session activated after closing, or ``None`` when no sessions remain.
        """
        if self._active_stream_belongs_to_session(session_id):
            self._signal_stop()
            if (
                self._active_stream_task is not None
                and self._active_stream_task is not asyncio.current_task()
            ):
                self._active_stream_task.cancel()
            self._set_run_state(
                ConsoleRunState(ConsoleRunStatus.STOPPED, "Session closed.")
            )
        return self.store.close_session(session_id)

    def _signal_stop(self) -> None:
        """Set the shared UI-facing stop flag AND the active run's own
        permanent per-run cancel signal.

        ``_stop_requested`` is reset by the run's own ``finally`` block as
        soon as the coroutine side of ``_run_agent_reply`` is done handling
        a cancellation -- but ``asyncio.to_thread`` survives Task
        cancellation, so the agent bridge's background OS thread can still
        be running at that point and poll ``should_cancel()`` afterward
        (task-227). ``_active_cancel_event``, once set here, is never reset
        for that run, so a still-running bridge thread always observes the
        Stop correctly regardless of what the coroutine side has already
        reset.
        """
        self._stop_requested = True
        if self._active_cancel_event is not None:
            self._active_cancel_event.set()

    # -- MCP batch-approval bridge (task-5) ----------------------------------

    def request_mcp_approvals(self, pending: list[MCPPendingCall]) -> dict[str, str]:
        """Bridge one batch of pending MCP tool calls to the Console UI and back.

        WORKER THREAD. Bound as ``MCPToolProvider``'s ``approval_callback``
        (T6's closure hands ``self.request_mcp_approvals`` straight
        through), so this runs on the agent bridge's background OS thread
        (the ``asyncio.to_thread`` call inside ``_run_agent_reply``) --
        it must never touch a widget directly, only through
        ``self.app.call_from_thread``.

        Builds a fresh ``threading.Event`` + shared decisions dict, surfaces
        the batch via ``self.set_pending_approval`` (marshaled onto the UI
        thread), then polls ``event.wait(1.0)`` re-checking this run's
        cancel signals and a deadline every second until one of three things
        happens: the user submits a decision (``resolve_pending_approval``,
        called from the UI thread, sets the Event), the run is
        cancelled/stopped/torn down (``_stop_requested``/
        ``_active_cancel_event`` -- already wired by ``stop_active_run``,
        ``close_session``, and ``shutdown`` via ``_signal_stop``), or the
        configured approval timeout elapses. Whichever unique ``llm_name``
        never received an explicit decision by then fails closed to
        ``"deny"`` (cancellation) or ``"timeout"`` (deadline) -- see
        ``MCPToolProvider._apply_verdict`` for how each decision string is
        consumed. The card is always cleared afterwards (``finally``),
        regardless of outcome.

        Args:
            pending: One turn's pending tool calls awaiting approval,
                possibly containing repeated ``llm_name``s (T3: calls
                sharing a name share one verdict).

        Returns:
            A decision string (``approve_once``/``approve_session``/
            ``always_allow``/``deny``/``timeout``) for every unique
            ``llm_name`` in ``pending``.
        """
        unique_names: list[str] = []
        seen: set[str] = set()
        for call in pending:
            if call.llm_name not in seen:
                seen.add(call.llm_name)
                unique_names.append(call.llm_name)
        if not unique_names:
            return {}

        event = threading.Event()
        decisions: dict[str, str] = {}
        self._pending_approval_event = event
        self._pending_approval_decisions = decisions

        timeout_seconds = self._resolve_mcp_approval_timeout_seconds()
        deadline = time.monotonic() + timeout_seconds
        payload = {
            "calls": [
                {
                    "llm_name": call.llm_name,
                    "server_key": call.server_key,
                    "tool_name": call.tool_name,
                    "server_label": call.server_label,
                    "arguments": dict(call.arguments or {}),
                    "reason": call.reason,
                }
                for call in pending
            ],
            "timeout_seconds": timeout_seconds,
        }

        try:
            self._marshal_pending_approval(payload)
            while not event.wait(_MCP_APPROVAL_POLL_SECONDS):
                if self._stop_requested or (
                    self._active_cancel_event is not None
                    and self._active_cancel_event.is_set()
                ):
                    for name in unique_names:
                        decisions.setdefault(name, "deny")
                    break
                if time.monotonic() >= deadline:
                    for name in unique_names:
                        decisions.setdefault(name, "timeout")
                    break
            # Any name the resolution path above didn't already cover (e.g.
            # a partial/empty decisions dict handed to `resolve_pending_
            # approval`) fails closed to "deny" rather than silently
            # dropping the call from the returned mapping.
            for name in unique_names:
                decisions.setdefault(name, "deny")
            return dict(decisions)
        finally:
            self._pending_approval_event = None
            self._pending_approval_decisions = None
            try:
                self._marshal_pending_approval(None)
            except Exception:  # noqa: BLE001 -- suppress teardown-time errors
                logger.opt(exception=True).debug(
                    "Failed to marshal approval clear during teardown"
                )

    def _marshal_pending_approval(self, payload: dict[str, Any] | None) -> None:
        """Push ``payload`` (or clear it) onto the UI thread, if wired."""
        if self.app is not None and self.set_pending_approval is not None:
            self.app.call_from_thread(self.set_pending_approval, payload)

    def _resolve_mcp_approval_timeout_seconds(self) -> float:
        if self.mcp_approval_timeout_seconds is not None:
            try:
                return float(self.mcp_approval_timeout_seconds())
            except Exception:  # noqa: BLE001 -- fail open to the documented default
                pass
        try:
            return float(get_cli_setting("mcp", "approval_timeout_seconds", _DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS))
        except (TypeError, ValueError):
            return _DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS

    # -- MCP provider registration (task-6) ----------------------------------

    def _publish_mcp_inspector_counts(
        self, tool_count: int | None, not_connected_count: int | None,
    ) -> None:
        """Publish this run's MCP catalog counts for the inspector's "MCP" row.

        ``setattr`` onto ``self.app`` -- the exact same object
        ``ChatScreen._console_mcp_tool_count``/``_console_mcp_not_connected_
        count`` ``getattr`` from (wired onto this controller as ``self.app``
        by ``ChatScreen._ensure_console_chat_controller``). Every
        ``_compose_mcp_provider`` return path calls this: ``(None, None)``
        is the row's documented "absent" contract (see
        ``console_display_state._mcp_inspector_row``) for the no-service /
        kill-switch-on / compose-failed / empty-catalog paths; the eligible
        path publishes the real counts.

        No separate UI refresh is triggered here by design -- piggybacking
        on machinery the screen already runs, not a new mechanism:
        ``_compose_mcp_provider`` always executes on the main loop while
        this run's state is already STREAMING (set moments earlier by
        ``_run_agent_reply``), so the screen's own active-run poll timer
        (``ChatScreen._start_console_transcript_sync_timer``, already
        ticking every 0.2s by the time this runs -- started before
        ``submit_draft`` is even awaited) and the guaranteed post-
        ``submit_draft`` sync (``ChatScreen._submit_console_native_draft``)
        both already re-derive inspector state from these attributes on
        their own next pass.
        """
        if self.app is None:
            return
        self.app.console_mcp_tool_count = tool_count
        self.app.console_mcp_not_connected_count = not_connected_count

    async def _compose_mcp_provider(
        self,
    ) -> tuple[MCPToolProvider | None, Callable[[list["ToolCall"]], dict[str, str]] | None]:
        """Build + compose THIS run's MCPToolProvider on the running main loop.

        MUST be awaited from an async caller with the real Textual main
        loop running (``_run_agent_reply``, BEFORE its own
        ``asyncio.to_thread`` call) -- never from the agent bridge's
        worker thread. See ``MCPToolProvider``'s own module docstring:
        ``compose_catalog()`` performs async I/O
        (``local_external_catalog()``) that is documented to run on the
        main loop at registration time.

        Returns ``(None, None)`` whenever MCP tools should not be offered
        this run: no ``unified_mcp_service`` on the app, the kill switch
        is on, ``get_kill_switch``/``compose_catalog`` raised, or the
        composed catalog is empty (nothing to register, and -- since
        ``not_connected_count`` is only ever non-zero for servers that
        already contributed at least one eligible tool -- nothing an
        empty catalog could usefully report either). Every return path
        also publishes this run's inspector counts via
        ``_publish_mcp_inspector_counts`` -- see that method's docstring;
        this is the only production writer of ``console_mcp_tool_count``/
        ``console_mcp_not_connected_count``.

        Returns:
            ``(provider, review_tool_calls)`` when eligible -- a composed
            ``MCPToolProvider`` ready to hand to ``ConsoleAgentBridge.
            run_reply`` and this run's ``build_mcp_review_hook``-built
            batch-review closure; ``(None, None)`` otherwise.
        """
        service = getattr(self.app, "unified_mcp_service", None)
        if service is None:
            self._publish_mcp_inspector_counts(None, None)
            return None, None
        try:
            kill_switch = service.get_kill_switch()
        except Exception:  # noqa: BLE001 -- fail closed to "no MCP this run"
            logger.opt(exception=True).warning(
                "ConsoleChatController: get_kill_switch failed; skipping MCP this run")
            self._publish_mcp_inspector_counts(None, None)
            return None, None
        if kill_switch:
            self._publish_mcp_inspector_counts(None, None)
            return None, None
        provider = MCPToolProvider(
            service=service,
            main_loop=asyncio.get_running_loop(),
            approval_callback=self.request_mcp_approvals,
        )
        try:
            await provider.compose_catalog()
        except Exception:  # noqa: BLE001 -- a composition failure must not abort the send
            logger.opt(exception=True).warning(
                "ConsoleChatController: MCP compose_catalog failed; skipping MCP this run")
            self._publish_mcp_inspector_counts(None, None)
            return None, None
        catalog = provider.list_catalog()
        if not catalog:
            self._publish_mcp_inspector_counts(None, None)
            return None, None
        self._publish_mcp_inspector_counts(len(catalog), provider.not_connected_count)
        return provider, build_mcp_review_hook(provider, self.request_mcp_approvals)

    def resolve_pending_approval(self, decisions: dict[str, str]) -> None:
        """UI THREAD: apply the user's batch decision, releasing the waiting worker thread.

        Called by ``ChatScreen``'s ``ChatApprovalCard.ApprovalDecided``
        handler. A no-op when there is no active round (e.g. a stale
        message arriving after a timeout/cancellation already resolved and
        cleared it).

        NOTE: Snapshots ``_pending_approval_decisions`` and ``_pending_approval_event``
        into locals to avoid TOCTOU race: the worker thread's ``finally`` block nulls
        both attributes concurrently. Guard and act only on the snapshots.
        """
        # Snapshot both at once to prevent TOCTOU race with worker thread's finally block
        decisions_dict = self._pending_approval_decisions
        approval_event = self._pending_approval_event
        if decisions_dict is None or approval_event is None:
            return
        decisions_dict.update(decisions or {})
        approval_event.set()

    def _deny_pending_approval_on_context_change(self) -> None:
        """Force-resolve a pending approval round as denied for undecided calls.

        Sets the round's Event without pre-filling ``decisions`` --
        ``request_mcp_approvals``'s own post-loop fill-in resolves every
        name that still lacks an explicit entry to ``"deny"``. A no-op when
        no round is pending.

        NOTE: Snapshots ``_pending_approval_event`` into a local to avoid TOCTOU race
        with the worker thread's ``finally`` block that nulls it concurrently.
        """
        # Snapshot to prevent TOCTOU race with worker thread's finally block
        approval_event = self._pending_approval_event
        if approval_event is not None:
            approval_event.set()

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
        self._signal_stop()
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
            self._signal_stop()
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
                self._active_cancel_event = None

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
        provider_messages = await self._apply_chat_dictionaries(provider_messages, session_id)
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
        provider_messages = await self._apply_chat_dictionaries(provider_messages, session_id)
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
        provider_messages = await self._apply_chat_dictionaries(provider_messages, session_id)
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

    async def _apply_chat_dictionaries(
        self, provider_messages: list[dict[str, Any]], session_id: str
    ) -> list[dict[str, Any]]:
        """Apply the active conversation chat dictionaries to the final user
        message of the ephemeral provider payload (never the stored transcript).

        Mirrors `_apply_skill_substitution` (final `role == "user"` message
        only, one rule for fresh sends AND retry/continue/regenerate). The
        synchronous DB read + regex substitution are offloaded via
        `asyncio.to_thread` because native sends run as async workers on the UI
        event loop. Skill commands are left untouched. Any failure returns the
        payload unchanged so a dictionary problem can never break a send;
        `asyncio.CancelledError` is re-raised so a mid-send Stop still cancels.
        """
        applier = self._chat_dictionary_applier
        if applier is None:
            return provider_messages

        session = next((s for s in self.store.sessions() if s.id == session_id), None)
        conversation_id = session.persisted_conversation_id if session is not None else None
        if not conversation_id:
            return provider_messages

        final_index: int | None = None
        for index in range(len(provider_messages) - 1, -1, -1):
            if provider_messages[index].get("role") == ConsoleMessageRole.USER.value:
                final_index = index
                break
        if final_index is None:
            return provider_messages

        message = provider_messages[final_index]
        content = message.get("content")
        if isinstance(content, str) and content.startswith(COMMAND_PREFIX):
            return provider_messages

        try:
            if isinstance(content, str):
                new_content: Any = await asyncio.to_thread(applier, conversation_id, content)
                if new_content == content:
                    return provider_messages
            elif isinstance(content, list):
                new_parts: list[Any] = []
                changed = False
                for part in content:
                    if (
                        isinstance(part, dict)
                        and part.get("type") == "text"
                        and isinstance(part.get("text"), str)
                    ):
                        new_text = await asyncio.to_thread(applier, conversation_id, part["text"])
                        if new_text != part["text"]:
                            changed = True
                            new_parts.append({**part, "text": new_text})
                            continue
                    new_parts.append(part)
                if not changed:
                    return provider_messages
                new_content = new_parts
            else:
                return provider_messages
        except asyncio.CancelledError:
            raise
        except Exception:
            return provider_messages

        new_messages = list(provider_messages)
        new_messages[final_index] = {**message, "content": new_content}
        return new_messages

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
        # A fresh per-run Event, captured by `should_cancel` below by
        # closure (not read off `self` each time) -- see
        # `_active_cancel_event`'s docstring for why this, rather than
        # `_stop_requested` alone, is what makes a still-running bridge
        # thread observe a Stop correctly (task-227).
        cancel_event = threading.Event()
        self._active_cancel_event = cancel_event
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
        # noqa: E731 — tiny closure. Reads BOTH signals: `_stop_requested`
        # for same-tick responsiveness (and test doubles that flip it
        # directly), and `cancel_event` -- captured by value, not via
        # `self._active_cancel_event` -- for correctness once this run's
        # `finally` below has already reset `_stop_requested` while the
        # bridge's background thread is still running (task-227: an
        # `asyncio.to_thread` call survives Task cancellation, so the
        # coroutine can finish handling a Stop and reset its own shared
        # bookkeeping well before the OS thread it detached from actually
        # returns). `stop_active_run`/`close_session`/`shutdown` all set
        # `cancel_event` via `_signal_stop()` the moment Stop is
        # requested, and nothing ever clears it again for this run, so a
        # late poll from the surviving thread still sees the cancellation.
        should_cancel = lambda: self._stop_requested or cancel_event.is_set()  # noqa: E731

        # P5-T6: compose this run's MCP tool provider (if eligible) HERE,
        # on the running main loop, BEFORE the bridge is dispatched onto
        # asyncio.to_thread below -- see `_compose_mcp_provider`'s own
        # docstring for why `compose_catalog()`'s async I/O can never run
        # from the worker thread. `(None, None)` (no service, kill switch
        # on, or nothing composed) leaves the bridge's MCP-free path
        # byte-identical to before this task.
        mcp_provider, mcp_review_hook = await self._compose_mcp_provider()

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
                mcp_provider=mcp_provider,
                review_tool_calls=mcp_review_hook,
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
                # NOT cancel_event.clear(): the closure above captured
                # this exact Event object by value, so any still-running
                # bridge thread keeps observing whatever it was last set
                # to, forever, regardless of this attribute now pointing
                # elsewhere (or nowhere) for the NEXT run (task-227).
                self._active_cancel_event = None

        # Captured here, before `_finalize_agent_reply` runs: this run's own
        # cancel_event is the authority on whether IT was stopped,
        # independent of what status `mark_message_stopped` may have left
        # the message at (task-227 AC3 follow-up -- see the guard below).
        return self._finalize_agent_reply(
            assistant_message_id, session_id, outcome, variant_mode=variant_mode,
            cancel_event=cancel_event)

    def _agent_conversation_id(self, session_id: str) -> str:
        """Return the durable id the run store is keyed by (persisted id when set)."""
        for session in self.store.sessions():
            if session.id == session_id:
                return session.persisted_conversation_id or session_id
        return session_id

    def _finalize_agent_reply(
        self, assistant_message_id: str, session_id: str, outcome: Any,
        *, variant_mode: bool, cancel_event: threading.Event | None = None,
    ) -> ConsoleSubmitResult:
        from tldw_chatbook.Agents.agent_models import RUN_CANCELLED, RUN_DONE

        try:
            current = self.store.get_message(assistant_message_id)
        except KeyError:
            return self._session_closed_result()
        # task-227 LOW-2 (+ AC3 follow-up): a Stop can land in the
        # ultra-narrow window after asyncio.to_thread returns an outcome
        # but before this method runs. `current.status == "stopped"` alone
        # only catches a plain send/retry -- `mark_message_stopped`
        # (console_chat_store.py) RESTORES a mid-regenerate message to its
        # *prior* status (e.g. "complete"), not "stopped", so that check
        # never fires for a stopped regenerate. Trust the run's own
        # per-run `cancel_event` instead: it is set by `_signal_stop` the
        # instant Stop is requested and never cleared for this run, so
        # `.is_set()` is true here if and only if THIS run was stopped --
        # regardless of which status `mark_message_stopped` left the
        # message at. Every branch below would otherwise either raise via
        # _validate_can_mark_terminal (mark_message_complete /
        # mark_message_failed) or silently resurrect the message back to
        # "complete" with a phantom variant (finalize_variant_stream,
        # which has no such guard at all). The `current.status`
        # comparison stays as a belt for any future caller that reaches
        # this method without a `cancel_event` in scope. Stop already won
        # and settled the message (mark_message_stopped's own restore --
        # prior status for a regenerate, "stopped" for a plain send) and
        # the variant base (already popped), so this is a benign no-op
        # read-back, never an error, in either case.
        if current.status == "stopped" or (cancel_event is not None and cancel_event.is_set()):
            self._set_run_state(
                ConsoleRunState(ConsoleRunStatus.STOPPED, "Response stopped.")
            )
            return ConsoleSubmitResult(True, True, current.content)

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
        exhausted", "model-turn budget exhausted", "wall-clock budget
        exhausted", "loop detected: ...") is surfaced when available.
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
                # An image-only user turn whose images all fell outside the
                # budget (over-cap, or a non-vision model) must not vanish —
                # a silently dropped turn distorts the conversation shape the
                # model sees. Emit a text placeholder instead.
                omitted = [
                    attachment
                    for attachment in message.attachments
                    if attachment.data is not None
                ]
                if message.role is ConsoleMessageRole.USER and omitted:
                    placeholder = (
                        "[image omitted]"
                        if len(omitted) == 1
                        else f"[{len(omitted)} images omitted]"
                    )
                    payloads.append(
                        {"role": message.role.value, "content": placeholder}
                    )
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
