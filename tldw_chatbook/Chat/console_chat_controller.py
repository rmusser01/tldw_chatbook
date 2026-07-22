"""Native Console chat controller for send, stream, stop, and retry flows."""

from __future__ import annotations

import asyncio
import copy
import re
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol

from tldw_chatbook.Chat.attachment_core import (
    image_url_part,
    max_history_images,
    vision_block_reason,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleContextSnapshot,
    ConsoleMessageRole,
    ConsoleProviderSelection,
    ConsoleRunState,
    ConsoleRunStatus,
    ConsoleStagedSource,
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
from tldw_chatbook.Chat.provider_failures import (  # noqa: F401  (re-export: tests and callers import describe_stream_failure from here)
    describe_stream_failure,
)
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


def _normalize_world_info_history(
    messages: "list[dict[str, Any]]",
) -> "list[dict[str, Any]]":
    """Flatten messages to ``{"role","content": str}`` for world-info scanning.

    ``WorldInfoProcessor.process_messages`` types content as ``str``; native
    provider messages may carry multimodal list content, so extract the text
    parts (joined) and drop images before scanning. System messages are
    skipped entirely -- world-info should scan only the user/assistant
    conversation, matching the legacy path; keywords in the system prompt
    must not spuriously activate entries.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == ConsoleMessageRole.SYSTEM.value:
            continue
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(
                part["text"]
                for part in content
                if isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            )
        else:
            text = ""
        out.append({"role": message.get("role", ""), "content": text})
    return out


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
    every same-named call `invoke()` makes THIS turn peeks (Finding F1:
    never popped, so two calls to the same tool in one batch both see the
    approval, not just the first).

    Finding F1 also requires this hook to call
    `provider.apply_batch_decisions` on EVERY invocation, even when
    `pending` ends up empty (a turn whose calls are all non-MCP, or all
    already resolved without asking) -- passing `{}` in that case.
    `apply_batch_decisions` REPLACES the stamp set rather than merging, so
    this is what guarantees a stamp from an earlier turn can never survive
    into a later one and be misread as this turn's verdict for a
    repeated tool name.

    I3 (probe-verified): that clear happens at hook ENTRY, before
    `pending_gate_for` is even resolved and before the
    `request_mcp_approvals` round trip -- not only after a successful one.
    `request_mcp_approvals` can raise (e.g. the unguarded
    `_marshal_pending_approval` call mid-shutdown); `run_agent_loop`'s own
    hook-exception handling fails the WHOLE batch open (treats every call
    in it as `"proceed"`) when that happens. If the clear only ran after a
    successful round trip, a raise would leave THIS turn's stamp set
    exactly as the PREVIOUS turn left it -- so the fail-open runtime would
    hand `invoke()` a stale prior-turn stamp (e.g. a real `"approve_once"`)
    for a call the user never decided on this turn. Clearing first means a
    raised round trip always leaves `invoke()` with no stamp to peek,
    falling through to its own fresh gate -- which fails closed for an
    `"ask"` tool with no approval_callback wired.

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

    def review_tool_calls(calls: list["ToolCall"]) -> dict[str, str]:
        # I3: clear THIS turn's stamps FIRST, before pending_gate_for/the
        # approval round trip even run -- subsumes the `if not pending`
        # branch's own clear below (every invocation of this hook clears,
        # unconditionally). See this function's own docstring for why the
        # clear must happen at entry, not only after a successful round
        # trip: a raising `request_mcp_approvals` must never leave a stale
        # prior-turn stamp live for the fail-open runtime to hand straight
        # to `invoke()`.
        provider.apply_batch_decisions({})
        pending: list["MCPPendingCall"] = []
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
        world_info_applier: "Callable[[str | None, str, list], str] | None" = None,
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
        self._world_info_applier = world_info_applier
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
        #: The composed MCP provider for the current agent run, captured
        #: on the main loop in ``_run_agent_reply`` so ``build_context_snapshot``
        #: can read tool metadata later without recomposing.
        self._mcp_provider: Any | None = None

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

        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.VALIDATING, "Validating provider.")
        )
        resolution = await self.provider_gateway.resolve_for_send(
            self._provider_selection()
        )
        if not getattr(resolution, "ready", False):
            visible_copy = self._blocked_visible_copy(
                getattr(resolution, "visible_copy", "")
            )
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
        provider_messages, refuse = await self._apply_skill_substitution(
            provider_messages
        )
        if refuse is not None:
            return self._block(session.id, refuse)
        provider_messages = await self._apply_chat_dictionaries(
            provider_messages, session.id
        )
        provider_messages = await self._apply_world_info(
            provider_messages, session.id
        )
        prefill, prefill_from_one_shot = self._resolve_submit_prefill(session.id)
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
            prefill=prefill,
            prefill_from_one_shot=prefill_from_one_shot,
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

    def _maybe_auto_title_session(
        self, session: ConsoleChatSession, draft: str
    ) -> None:
        """Title a default-named session from its first accepted message."""
        if session.persisted_conversation_id is not None:
            return
        if not is_default_console_session_title(session.title):
            return
        derived = derive_console_session_title(draft)
        if derived:
            self.store.rename_session(session.id, derived)  # (session, persisted) — auto-title best-effort

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
        call_by_name: dict[str, "MCPPendingCall"] = {}
        for call in pending:
            if call.llm_name not in seen:
                seen.add(call.llm_name)
                unique_names.append(call.llm_name)
                call_by_name[call.llm_name] = call
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
                    # Finding I3: a stop/unmount that resolves THIS round
                    # denies every still-undecided call, but
                    # `run_agent_loop`'s own `should_cancel()` check fires
                    # for every call in this turn's batch BEFORE any of
                    # them reaches `invoke()` -- so the "deny" verdict
                    # stamped below is never consumed there and would
                    # otherwise leave no audit record at all (contrast
                    # with the timeout branch, whose calls DO still reach
                    # `invoke()`'s own gate and get logged there, since a
                    # timeout is not itself a cancellation). Log directly
                    # here, best-effort, for exactly the names this branch
                    # is about to fail closed.
                    cancelled_names = [
                        name for name in unique_names if name not in decisions
                    ]
                    for name in unique_names:
                        decisions.setdefault(name, "deny")
                    self._record_cancelled_approval_decisions(
                        cancelled_names,
                        call_by_name,
                    )
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
            # Finding F4: build the snapshot by keyed lookup over the
            # (locally-owned, never-mutated) `unique_names` list rather
            # than `dict(decisions)` -- the latter iterates `decisions`
            # itself, which `resolve_pending_approval` can concurrently
            # `.update()` from the UI thread; a same-size update can't
            # change dict length, so this is unreachable today, but a
            # keyed `.get()` per name can never raise "dictionary changed
            # size during iteration" regardless. The `setdefault` pass
            # above already guarantees every name resolves, so `.get`'s
            # own "deny" fallback here is a belt-and-suspenders no-op, not
            # a second source of truth.
            return {name: decisions.get(name, "deny") for name in unique_names}
        finally:
            self._pending_approval_event = None
            self._pending_approval_decisions = None
            try:
                self._marshal_pending_approval(None)
            except Exception:  # noqa: BLE001 -- suppress teardown-time errors
                logger.opt(exception=True).debug(
                    "Failed to marshal approval clear during teardown"
                )

    def _record_cancelled_approval_decisions(
        self,
        names: list[str],
        call_by_name: dict[str, "MCPPendingCall"],
    ) -> None:
        """Best-effort audit log for calls denied by a stop/unmount mid-approval.

        Finding I3: see the cancellation branch's own comment in
        ``request_mcp_approvals`` for why this direct call is necessary --
        `MCPToolProvider._record_decision_safe` (the normal recording
        path) is never reached for these calls, since `run_agent_loop`
        cancels the whole turn before dispatching any of them. Reached via
        `self.app.unified_mcp_service` (the same object
        `_compose_mcp_provider` built this run's `MCPToolProvider` from --
        see that method), never raises: a missing app/service, or the
        service lacking `record_tool_decision`, is a silent no-op, and any
        exception the real call raises is logged and swallowed, mirroring
        `MCPToolProvider._record_decision_safe`'s own never-raise
        contract.
        """
        service = getattr(self.app, "unified_mcp_service", None)
        if service is None:
            return
        record = getattr(service, "record_tool_decision", None)
        if not callable(record):
            return
        for name in names:
            call = call_by_name.get(name)
            if call is None:
                continue
            try:
                record(
                    call.server_key,
                    call.tool_name,
                    decision="denied",
                    initiator="agent",
                    error="run stopped while approval pending",
                )
            except Exception:  # noqa: BLE001 -- best-effort audit trail only
                logger.opt(exception=True).debug(
                    "Failed to record cancelled MCP approval decision"
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
            return float(
                get_cli_setting(
                    "mcp",
                    "approval_timeout_seconds",
                    _DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS,
                )
            )
        except (TypeError, ValueError):
            return _DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS

    # -- MCP provider registration (task-6) ----------------------------------

    def _publish_mcp_inspector_counts(
        self,
        tool_count: int | None,
        not_connected_count: int | None,
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
    ) -> tuple[
        MCPToolProvider | None, Callable[[list["ToolCall"]], dict[str, str]] | None
    ]:
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
                "ConsoleChatController: get_kill_switch failed; skipping MCP this run"
            )
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
                "ConsoleChatController: MCP compose_catalog failed; skipping MCP this run"
            )
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

    def stop_active_run(self, *, record_user_stop: bool = True) -> bool:
        """Request the active stream to stop at the next safe boundary.

        Args:
            record_user_stop: Append the explicit "stopped by user"
                transcript record (TASK-337 AC3). ``shutdown`` passes
                ``False`` — a teardown stop is not a user action.

        Returns:
            True when an active run was found and stopped.
        """
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
        if record_user_stop:
            # TASK-337 AC3: a durable, explicit record — the run-state chip
            # copy is transient and the review found nothing else marked
            # the interruption.
            try:
                session_id = self.store.session_id_for_message(assistant_message_id)
                self.store.append_message(
                    session_id,
                    role=ConsoleMessageRole.SYSTEM,
                    content="Response stopped by user.",
                )
            except KeyError:
                pass
        if (
            self._active_stream_task is not None
            and self._active_stream_task is not asyncio.current_task()
        ):
            self._active_stream_task.cancel()
        return True

    async def shutdown(self) -> None:
        """Stop and await the active stream task before owner teardown."""
        task = self._active_stream_task
        if task is None:
            return
        if not self.stop_active_run(record_user_stop=False):
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
            if (
                message.role is ConsoleMessageRole.ASSISTANT
                and message.status == "streaming"
            ):
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
        resolution = await self.provider_gateway.resolve_for_send(
            self._provider_selection()
        )
        if not getattr(resolution, "ready", False):
            visible_copy = self._blocked_visible_copy(
                getattr(resolution, "visible_copy", "")
            )
            return self._block(session_id, visible_copy)

        provider_messages = self._provider_messages_for_session(
            session_id,
            before_message_id=message_id,
        )
        self._ensure_user_continuation_instruction(provider_messages)
        provider_messages, refuse = await self._apply_skill_substitution(
            provider_messages
        )
        if refuse is not None:
            return self._block(session_id, refuse)
        provider_messages = await self._apply_chat_dictionaries(
            provider_messages, session_id
        )
        provider_messages = await self._apply_world_info(
            provider_messages, session_id
        )
        prefill = self._pinned_prefill_for_session(session_id)
        return await self._stream_assistant_response(
            resolution=resolution,
            provider_messages=provider_messages,
            assistant_message_id=message_id,
            prepare_retry=True,
            prefill=prefill,
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
            visible_copy = (
                "Open the original session before continuing from this message."
            )
            self._set_run_state(ConsoleRunState.blocked(visible_copy))
            return ConsoleSubmitResult(False, False, visible_copy)

        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.VALIDATING, "Validating provider.")
        )
        resolution = await self.provider_gateway.resolve_for_send(
            self._provider_selection()
        )
        if not getattr(resolution, "ready", False):
            visible_copy = self._blocked_visible_copy(
                getattr(resolution, "visible_copy", "")
            )
            return self._block(session_id, visible_copy)

        provider_messages = self._provider_messages_through_message(
            session_id, message_id
        )
        self._ensure_user_continuation_instruction(provider_messages)
        if not self._has_user_turn(provider_messages):
            return self._block(
                session_id,
                "Nothing to continue before the character's opening line.",
            )
        provider_messages, refuse = await self._apply_skill_substitution(
            provider_messages
        )
        if refuse is not None:
            return self._block(session_id, refuse)
        provider_messages = await self._apply_chat_dictionaries(
            provider_messages, session_id
        )
        provider_messages = await self._apply_world_info(
            provider_messages, session_id
        )
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
            return self._block(
                session_id, "Only assistant messages can be regenerated."
            )
        if self.store.session_id_for_message(message_id) != session_id:
            visible_copy = "Open the original session before regenerating this message."
            self._set_run_state(ConsoleRunState.blocked(visible_copy))
            return ConsoleSubmitResult(False, False, visible_copy)

        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.VALIDATING, "Validating provider.")
        )
        resolution = await self.provider_gateway.resolve_for_send(
            self._provider_selection()
        )
        if not getattr(resolution, "ready", False):
            visible_copy = self._blocked_visible_copy(
                getattr(resolution, "visible_copy", "")
            )
            return self._block(session_id, visible_copy)

        provider_messages = self._provider_messages_for_session(
            session_id,
            before_message_id=message_id,
        )
        self._ensure_user_continuation_instruction(provider_messages)
        if not self._has_user_turn(provider_messages):
            return self._block(
                session_id,
                "Nothing to regenerate before the character's opening line.",
            )
        provider_messages, refuse = await self._apply_skill_substitution(
            provider_messages
        )
        if refuse is not None:
            return self._block(session_id, refuse)
        provider_messages = await self._apply_chat_dictionaries(
            provider_messages, session_id
        )
        provider_messages = await self._apply_world_info(
            provider_messages, session_id
        )
        prefill = self._pinned_prefill_for_session(session_id)
        return await self._stream_assistant_response(
            resolution=resolution,
            provider_messages=provider_messages,
            assistant_message_id=message_id,
            variant_mode=True,
            prefill=prefill,
        )

    async def build_context_snapshot(
        self,
        draft: str,
        attachments: Iterable[MessageAttachment] | None = None,
        staged_sources: Iterable[ConsoleStagedSource] | None = None,
    ) -> ConsoleContextSnapshot:
        """Return a read-only snapshot of the current transcript and the assembled next-send payload.

        Skills with side effects are NOT executed; only chat dictionaries are applied.

        Args:
            draft: The current composer draft text to include as a synthetic user turn.
            attachments: Pending attachments to include with the synthetic user turn.
            staged_sources: Staged workspace sources to include in the payload.

        Returns:
            A ``ConsoleContextSnapshot`` containing a deep-copied transcript and the
            redacted next-send provider payload. If assembly fails, the payload may
            contain an ``"error"`` key with a human-readable message.
        """
        session_id = self.store.active_session_id
        if not session_id:
            return ConsoleContextSnapshot(current_messages=[], next_send_payload={})

        current_messages = list(self.store.messages_for_session(session_id))
        staged_sources_list = [
            {"source_id": s.source_id, "label": s.label, "type": s.source_type}
            for s in (staged_sources or ())
        ]

        provider_messages: list[dict[str, Any]] = []

        try:
            # Build the next-send payload as submit_draft would, but do not persist.
            provider_messages = self._provider_messages_for_session(session_id)

            # Append a synthetic user turn for the draft so the preview matches what would be sent.
            attachment_tuple = tuple(attachments or ())
            synthetic_turn_added = bool(draft.strip() or attachment_tuple)
            if synthetic_turn_added:
                synthetic_user = self._provider_message_payloads(
                    [
                        ConsoleChatMessage(
                            role=ConsoleMessageRole.USER,
                            content=draft,
                            attachments=attachment_tuple,
                        )
                    ],
                    skip_failed=True,
                )
                provider_messages.extend(synthetic_user)

            # Do NOT call _apply_skill_substitution because it may execute skills with side effects.
            # Instead, annotate the final user message if a synthetic turn was appended and it
            # starts with a skill command. Historical turns have already been resolved at send time
            # and must not be annotated.
            provider_messages = self._annotate_skill_commands(
                provider_messages, synthetic_turn_added=synthetic_turn_added
            )

            # Chat dictionaries are safe to apply (string replacements only).
            provider_messages = await self._apply_chat_dictionaries(provider_messages, session_id)

            # task-401: mirror the send path's response prefill exactly --
            # same resolution (one-shot wins over pinned) and same trailing
            # assistant turn -- WITHOUT consuming the one-shot (this is a
            # read-only preview). Placed after dictionaries to match
            # `_stream_assistant_response`'s ordering (dictionaries never
            # rewrite prefill text).
            prefill, prefill_from_one_shot = self._resolve_submit_prefill(session_id)
            if prefill:
                provider_messages = [
                    *provider_messages,
                    {
                        "role": ConsoleMessageRole.ASSISTANT.value,
                        "content": prefill,
                    },
                ]

            # Replace image data with placeholders for the preview, including historical images.
            provider_messages = self._replace_image_data_with_placeholders(provider_messages)

            # Gather native tool schemas and MCP note.
            tools_info = self._build_tools_info_for_snapshot()

            # Redact secrets before returning.
            redacted_messages = self._redact_secrets(provider_messages)
            redacted_system = self._redact_secrets(self._leading_system_message())

            # Deep-copy messages so the snapshot is independent of the store.
            copied_messages = copy.deepcopy(current_messages)

            next_send_payload: dict[str, Any] = {
                "model": self.model or self.configured_model,
                "messages": redacted_messages,
                # `system` is intentionally duplicated from the leading system
                # message in `messages` so the preview viewer can show the
                # effective system prompt at a glance without scanning the
                # message list.  It is the same redacted value.
                "system": redacted_system,
                "staged_sources": staged_sources_list,
                "tools": tools_info,
            }
            if prefill:
                # Text mirrors the redacted trailing assistant turn so the
                # indicator can never leak what the messages list redacted.
                next_send_payload["response_prefill"] = {
                    "source": "one-shot" if prefill_from_one_shot else "pinned",
                    "text": redacted_messages[-1]["content"]
                    if redacted_messages
                    else prefill,
                    "agent_loop_bypassed": True,
                }
            return ConsoleContextSnapshot(
                current_messages=copied_messages,
                next_send_payload=next_send_payload,
            )
        except Exception as exc:
            logger.exception(
                "Failed to build context snapshot: session_id={session_id} "
                "draft_length={draft_length} attachments={attachments_count} "
                "staged_sources={staged_sources_count}",
                session_id=session_id,
                draft_length=len(draft),
                attachments_count=len(tuple(attachments or ())),
                staged_sources_count=len(tuple(staged_sources or ())),
            )
            # Preserve whatever was assembled before the failure so the viewer
            # still sees the transcript-derived payload and effective system
            # prompt rather than an empty placeholder.
            degraded_messages = self._replace_image_data_with_placeholders(
                self._redact_secrets(provider_messages)
            )
            degraded_system = self._redact_secrets(self._leading_system_message())
            return ConsoleContextSnapshot(
                current_messages=copy.deepcopy(current_messages),
                next_send_payload={
                    "model": self.model or self.configured_model,
                    "messages": degraded_messages,
                    "system": degraded_system,
                    "staged_sources": staged_sources_list,
                    "tools": {
                        "native_schemas": [],
                        "mcp_note": None,
                        "preview_note": "Preview unavailable due to an internal error.",
                    },
                    "error": f"Failed to build context snapshot: {exc}",
                },
            )

    @staticmethod
    def _replace_image_data_with_placeholders(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = copy.deepcopy(messages)

        def _is_data_url(value: Any) -> bool:
            return isinstance(value, str) and value.startswith("data:")

        def _redact_image_url_value(value: Any) -> Any:
            """Redact an image URL value while preserving its original shape."""
            if isinstance(value, dict) and _is_data_url(value.get("url")):
                return {**value, "url": "[image: data redacted for preview]"}
            if isinstance(value, str) and _is_data_url(value):
                return "[image: data redacted for preview]"
            return value

        def _redact_image_source(source: dict[str, Any]) -> dict[str, Any]:
            """Redact base64 or data-URL content inside an image source dict."""
            if not isinstance(source, dict):
                return source
            redacted = {**source}
            if _is_data_url(redacted.get("data")) or redacted.get("type") == "base64":
                redacted["data"] = "[image: data redacted for preview]"
            if _is_data_url(redacted.get("url")):
                redacted["url"] = "[image: data redacted for preview]"
            return redacted

        for message in result:
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "image_url":
                        part["image_url"] = _redact_image_url_value(part.get("image_url"))
                    if part.get("type") == "image":
                        # Anthropic-style image parts use a ``source`` dict with
                        # base64 data; preserve the surrounding structure.
                        if isinstance(part.get("source"), dict):
                            part["source"] = _redact_image_source(part["source"])
                        if "image" in part:
                            part["image"] = _redact_image_url_value(part["image"])
            elif isinstance(content, str):
                # Some providers may inline image data URLs directly in a string
                # content body; redact them so they never leak into the preview.
                message["content"] = re.sub(
                    r"data:[^\s\"'<>]+",
                    "[image: data redacted for preview]",
                    content,
                )
        return result

    @staticmethod
    def _annotate_skill_commands(
        messages: list[dict[str, Any]],
        *,
        synthetic_turn_added: bool = True,
    ) -> list[dict[str, Any]]:
        result = copy.deepcopy(messages)
        if not synthetic_turn_added or not result or result[-1].get("role") != "user":
            return result

        content = result[-1].get("content", "")
        annotation = (
            "[Skill command not resolved in preview; "
            "actual substitution happens at send time.]"
        )

        if isinstance(content, str) and content.lstrip().startswith("/"):
            result[-1]["content"] = f"{content}\n\n{annotation}"
            return result

        if isinstance(content, list):
            new_parts: list[Any] = []
            annotated = False
            for part in content:
                text = part.get("text") if isinstance(part, dict) else None
                if (
                    not annotated
                    and isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(text, str)
                    and text.lstrip().startswith("/")
                ):
                    new_parts.append({**part, "text": f"{text}\n\n{annotation}"})
                    annotated = True
                else:
                    new_parts.append(part)
            if annotated:
                result[-1]["content"] = new_parts
        return result

    def _build_tools_info_for_snapshot(self) -> dict[str, Any]:
        """Return native tool schemas and preview notes for the snapshot."""
        tools: list[dict[str, Any]] = []
        if self._agent_bridge is not None:
            # Native tools only; live MCP catalog composition is out of scope.
            tools = self._agent_bridge.native_tool_schemas()
        mcp_note: str | None = None
        if self._mcp_provider:
            mcp_note = (
                "MCP tools are configured but live catalog composition is not shown in this preview."
            )
        if tools:
            preview_note = (
                "This preview shows only builtin native tools. "
                "The live run may add skills/MCP tools."
            )
        else:
            preview_note = "No native tools are configured for preview."
        return {
            "native_schemas": tools,
            "mcp_note": mcp_note,
            "preview_note": preview_note,
        }

    _SECRET_REDACTION_KEYS = {"api_key", "apikey", "token", "password", "secret", "bearer"}
    _SECRET_REDACTION_KEYS_NORMALIZED = {
        k.replace("-", "").replace("_", "") for k in _SECRET_REDACTION_KEYS
    }
    _SECRET_REDACTION_PATTERN = re.compile(
        r"(?P<open_quote>[\"']?)"
        r"(?P<key>"
        + "|".join(re.escape(k) for k in _SECRET_REDACTION_KEYS)
        + r")"
        r"(?P=open_quote)"
        r"(?P<sep>\s*[:=]\s*)"
        r"(?P<value>"
        + r'"(?:\\.|[^"\\])*"'
        + r"|'(?:\\.|[^'\\])*'"
        + r"|[^\s,;}\]\)\"']+"
        + r")",
        re.IGNORECASE,
    )

    @staticmethod
    def _redact_secrets(payload: Any) -> Any:
        """Return a deep-copied payload with likely secret values replaced.

        Redaction is best-effort and intended for preview/export convenience
        only. Do not rely on it for security-sensitive export or disclosure
        scenarios.
        """
        redacted = copy.deepcopy(payload)

        def _redact_string(value: str) -> str:
            def _replace_value(match: re.Match[str]) -> str:
                matched_value = match.group("value")
                if matched_value.startswith('"'):
                    redacted_value = '"[redacted]"'
                elif matched_value.startswith("'"):
                    redacted_value = "'[redacted]'"
                else:
                    redacted_value = "[redacted]"
                open_quote = match.group("open_quote")
                key = match.group("key")
                sep = match.group("sep")
                return f"{open_quote}{key}{open_quote}{sep}{redacted_value}"

            return ConsoleChatController._SECRET_REDACTION_PATTERN.sub(_replace_value, value)

        def _matches_secret_key(key: str) -> bool:
            """Return True when ``key`` matches or ends with a secret word.

            Matches exact keys such as ``api_key``, suffixed keys such as
            ``my_api_key``, and hyphenated/camelCase variants such as
            ``x-api-key`` or ``apiKey``.
            """
            lowered = key.lower()
            normalized = lowered.replace("-", "").replace("_", "")
            if normalized in ConsoleChatController._SECRET_REDACTION_KEYS_NORMALIZED:
                return True
            for secret in ConsoleChatController._SECRET_REDACTION_KEYS:
                if lowered.endswith(f"_{secret}"):
                    return True
                normalized_secret = secret.replace("-", "").replace("_", "")
                if normalized.endswith(normalized_secret):
                    return True
            return False

        def _redact_obj(obj: Any, under_secret: bool = False) -> Any:
            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    key_is_secret = _matches_secret_key(key)
                    if key_is_secret and isinstance(value, str):
                        result[key] = "[redacted]"
                    elif key_is_secret:
                        # Structured values under a secret key are recursively
                        # redacted so nested strings do not leak.
                        result[key] = _redact_obj(value, under_secret=True)
                    elif under_secret and isinstance(value, str):
                        result[key] = "[redacted]"
                    else:
                        result[key] = _redact_obj(value, under_secret=under_secret)
                return result
            if isinstance(obj, list):
                return [_redact_obj(item, under_secret=under_secret) for item in obj]
            if isinstance(obj, str):
                if under_secret:
                    return "[redacted]"
                return _redact_string(obj)
            return obj

        return _redact_obj(redacted)

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
                {
                    "role": ConsoleMessageRole.USER.value,
                    "content": CONSOLE_CONTINUE_INSTRUCTION,
                }
            )

    @staticmethod
    def _has_user_turn(provider_messages: list[dict[str, Any]]) -> bool:
        return any(
            m.get("role") == ConsoleMessageRole.USER.value for m in provider_messages
        )

    def _pinned_prefill_for_session(self, session_id: str) -> str | None:
        """Return the session's pinned response prefill, if any."""
        settings = self.store.session_settings(session_id)
        pinned = getattr(settings, "pinned_prefill", None) if settings else None
        return pinned or None

    def _resolve_submit_prefill(self, session_id: str) -> tuple[str | None, bool]:
        """Return ``(prefill, from_one_shot)`` for a normal send.

        One-shot wins over pinned for the send it is armed for; pinned
        resumes afterward (the one-shot is only cleared on a complete or
        stopped outcome — see ``_consume_one_shot_prefill``).
        """
        one_shot = self.store.session_one_shot_prefill(session_id)
        if one_shot:
            return one_shot, True
        return self._pinned_prefill_for_session(session_id), False

    def _consume_one_shot_prefill(
        self, assistant_message_id: str, used_prefill: str | None
    ) -> None:
        """Clear the armed one-shot after a send that used it terminated
        ``complete`` or ``stopped``. Blocked and failed sends never call
        this, so retry reproduces the original intent (spec §2).

        Compare-and-clear: ``used_prefill`` is the exact one-shot text this
        send consumed (or ``None`` if this send did not use a one-shot at
        all, in which case this is a no-op). The session's armed one-shot
        slot is only cleared when it still holds that same text. If a
        ``/prefill`` re-armed a *different* one-shot while this send was
        streaming, that newer one-shot must survive the in-flight send's
        completion untouched.
        """
        if used_prefill is None:
            return
        try:
            session_id = self.store.session_id_for_message(assistant_message_id)
        except KeyError:
            return
        if self.store.session_one_shot_prefill(session_id) == used_prefill:
            self.store.set_session_one_shot_prefill(session_id, None)

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
            refuse = SKILL_UNTRUSTED_REFUSE.format(
                name=resolution.name, reason=exc.reason_code
            )
            return provider_messages, refuse

        rendered = (
            result.get("rendered_prompt", "") if isinstance(result, Mapping) else ""
        )
        rendered_message = {"role": ConsoleMessageRole.USER.value, "content": rendered}
        execution_mode = (
            result.get("execution_mode") if isinstance(result, Mapping) else None
        )
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

    async def _apply_world_info(
        self, provider_messages: list[dict[str, Any]], session_id: str
    ) -> list[dict[str, Any]]:
        """Inject conversation world-info into the final user message of the
        ephemeral provider payload (never the stored transcript).

        Runs AFTER `_apply_chat_dictionaries` so world-info matches the
        dict-substituted text the model will see. Conversation-only (the bound
        applier passes `char_data=None`). Offloaded via `asyncio.to_thread`;
        any failure returns the payload unchanged; `CancelledError` re-raised.
        """
        applier = self._world_info_applier
        if applier is None:
            return provider_messages

        session = next((s for s in self.store.sessions() if s.id == session_id), None)
        conversation_id = (
            session.persisted_conversation_id if session is not None else None
        )
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

        history = _normalize_world_info_history(provider_messages[:final_index])

        try:
            if isinstance(content, str):
                injected: Any = await asyncio.to_thread(
                    applier, conversation_id, content, history
                )
                if injected == content:
                    return provider_messages
                new_content = injected
            elif isinstance(content, list):
                combined = "\n".join(
                    part["text"]
                    for part in content
                    if isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                )
                if not combined:
                    return provider_messages
                injected = await asyncio.to_thread(
                    applier, conversation_id, combined, history
                )
                if injected == combined:
                    return provider_messages
                prefix, _, suffix = injected.partition(combined)
                text_indices = [
                    i
                    for i, part in enumerate(content)
                    if isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                ]
                first_idx = text_indices[0]
                last_idx = text_indices[-1]
                new_parts: list[Any] = []
                for i, part in enumerate(content):
                    if i == first_idx or i == last_idx:
                        new_text = part["text"]
                        if i == first_idx:
                            new_text = prefix + new_text
                        if i == last_idx:
                            new_text = new_text + suffix
                        new_parts.append({**part, "text": new_text})
                    else:
                        new_parts.append(part)
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
        conversation_id = (
            session.persisted_conversation_id if session is not None else None
        )
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
                new_content: Any = await asyncio.to_thread(
                    applier, conversation_id, content
                )
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
                        new_text = await asyncio.to_thread(
                            applier, conversation_id, part["text"]
                        )
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
        available = (
            context.get("available_skills") if isinstance(context, Mapping) else None
        )
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
    def _validated_draft(
        draft: str, *, allow_empty: bool = False
    ) -> tuple[str, str | None]:
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

    _IMAGE_REJECTION_RECOVERY_HINT = (
        " This conversation includes an image attachment; if the model can't "
        "accept images, remove that message (select it and use Delete) or "
        "switch to a vision-capable model."
    )

    def _session_history_carries_images(self, session_id: str) -> bool:
        """Return whether any message in the session carries an image.

        TASK-335: history re-sends attachments on every turn, so a provider
        that rejects images fails ALL later sends in the conversation with
        the same opaque status — the failure copy names the likely cause.
        """
        try:
            messages = self.store.messages_for_session(session_id)
        except KeyError:
            return False
        for message in messages:
            if getattr(message, "attachments", None):
                return True
            if getattr(message, "image_data", None) is not None:
                return True
        return False

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
        prefill: str | None = None,
        prefill_from_one_shot: bool = False,
    ) -> ConsoleSubmitResult:
        owner_id = self.store.session_id_for_message(assistant_message_id)
        owner = next(
            (s for s in self.store.sessions() if s.id == owner_id), None
        )
        # task-427: a character session always takes the plain-provider
        # path, even with the global agent runtime enabled and a bridge
        # present. Keyed on the message's OWNING session (looked up here,
        # not the controller's active session) so a session switch racing
        # this send can't flip which branch a still-in-flight message uses.
        force_plain = owner is not None and owner.character_id is not None
        if (
            self._agent_runtime_enabled
            and self._agent_bridge is not None
            and not prefill
            and not force_plain
        ):
            return await self._run_agent_reply(
                resolution=resolution,
                provider_messages=provider_messages,
                assistant_message_id=assistant_message_id,
                prepare_retry=prepare_retry,
                variant_mode=variant_mode,
            )
        one_shot_used = prefill if prefill_from_one_shot else None
        if prefill:
            provider_messages = [
                *provider_messages,
                {
                    "role": ConsoleMessageRole.ASSISTANT.value,
                    "content": prefill,
                },
            ]
        self._active_assistant_message_id = assistant_message_id
        self._active_stream_task = asyncio.current_task()
        self._stop_requested = False
        if variant_mode:
            self.store.begin_variant_stream(assistant_message_id)
        if prefill and not prepare_retry:
            try:
                self.store.append_stream_chunk(assistant_message_id, prefill)
            except KeyError:
                return self._session_closed_result()
        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "Streaming response.")
        )
        retry_prepared = False
        emitted_content = False
        try:
            async for chunk in self.provider_gateway.stream_chat(
                resolution, provider_messages
            ):
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
                    self._consume_one_shot_prefill(
                        assistant_message_id, one_shot_used
                    )
                    return ConsoleSubmitResult(True, True, stopped.content)
                if prepare_retry and not retry_prepared:
                    self.store.prepare_message_retry(assistant_message_id)
                    retry_prepared = True
                    if prefill:
                        try:
                            self.store.append_stream_chunk(
                                assistant_message_id, prefill
                            )
                        except KeyError:
                            return self._session_closed_result()
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
                self._consume_one_shot_prefill(
                    assistant_message_id, one_shot_used
                )
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
            self._consume_one_shot_prefill(assistant_message_id, one_shot_used)
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
                self._consume_one_shot_prefill(
                    assistant_message_id, one_shot_used
                )
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
        logger.info(
            "console agent reply start",
            assistant_message_id=assistant_message_id,
            variant_mode=variant_mode,
            prepare_retry=prepare_retry,
        )
        self._active_assistant_message_id = assistant_message_id
        self._active_stream_task = asyncio.current_task()
        self._stop_requested = False
        self._mcp_provider = None
        # A fresh per-run Event, captured by `should_cancel` below by
        # closure (not read off `self` each time) -- see
        # `_active_cancel_event`'s docstring for why this, rather than
        # `_stop_requested` alone, is what makes a still-running bridge
        # thread observe a Stop correctly (task-227).
        cancel_event = threading.Event()
        self._active_cancel_event = cancel_event
        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "Agent running.")
        )
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
        if (
            agent_messages
            and agent_messages[0].get("role") == ConsoleMessageRole.SYSTEM.value
        ):
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
        self._mcp_provider = mcp_provider

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
                        assistant_message_id, visible_copy="Response stopped."
                    )
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
            if (
                getattr(getattr(exc, "response", None), "status_code", None)
                is not None
                and self._session_history_carries_images(session_id)
            ):
                visible_copy += self._IMAGE_REJECTION_RECOVERY_HINT
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
            logger.info(
                "console agent reply end",
                assistant_message_id=assistant_message_id,
                run_status=self.run_state.status.value,
                run_copy=self.run_state.visible_copy,
            )

        # Captured here, before `_finalize_agent_reply` runs: this run's own
        # cancel_event is the authority on whether IT was stopped,
        # independent of what status `mark_message_stopped` may have left
        # the message at (task-227 AC3 follow-up -- see the guard below).
        return self._finalize_agent_reply(
            assistant_message_id,
            session_id,
            outcome,
            variant_mode=variant_mode,
            cancel_event=cancel_event,
        )

    def _agent_conversation_id(self, session_id: str) -> str:
        """Return the durable id the run store is keyed by (persisted id when set)."""
        for session in self.store.sessions():
            if session.id == session_id:
                return session.persisted_conversation_id or session_id
        return session_id

    def _finalize_agent_reply(
        self,
        assistant_message_id: str,
        session_id: str,
        outcome: Any,
        *,
        variant_mode: bool,
        cancel_event: threading.Event | None = None,
    ) -> ConsoleSubmitResult:
        from tldw_chatbook.Agents.agent_models import RUN_CANCELLED, RUN_DONE

        current = self._ensure_assistant_placeholder(assistant_message_id, session_id)
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
        stopped_now = (
            (current is not None and current.status == "stopped")
            or (cancel_event is not None and cancel_event.is_set())
        )
        if stopped_now:
            self._set_run_state(
                ConsoleRunState(ConsoleRunStatus.STOPPED, "Response stopped.")
            )
            return ConsoleSubmitResult(True, True, current.content if current is not None else "")

        if outcome.status == RUN_CANCELLED:
            return self._finalize_agent_cancelled(
                assistant_message_id, session_id, variant_mode=variant_mode)

        if outcome.status != RUN_DONE:
            return self._finalize_agent_failure(
                assistant_message_id, session_id, outcome, variant_mode=variant_mode)

        return self._finalize_agent_success(
            assistant_message_id, session_id, outcome, variant_mode=variant_mode)

    def _ensure_assistant_placeholder(
        self, assistant_message_id: str, session_id: str,
    ) -> ConsoleChatMessage | None:
        """Return the assistant placeholder message if it still exists.

        ``KeyError`` means the session/placeholder was closed/removed mid-run;
        ``None`` is returned so callers can recover by appending a fresh
        assistant message instead of aborting the whole turn.
        """
        try:
            return self.store.get_message(assistant_message_id)
        except KeyError:
            return None

    def _find_runtime_written_assistant(
        self, session_id: str,
    ) -> ConsoleChatMessage | None:
        """Return the most recent assistant message in ``session_id``, if any."""
        try:
            messages = self.store.messages_for_session(session_id)
        except KeyError:
            return None
        for message in reversed(messages):
            if message.role is ConsoleMessageRole.ASSISTANT:
                return message
        return None

    def _complete_agent_message(
        self, assistant_message_id: str, variant_mode: bool, outcome: Any,
    ) -> ConsoleChatMessage:
        """Finalize a placeholder, applying the empty-final-text fallback.

        The fallback text is streamed into the placeholder so the store's
        existing persistence/validation paths stay unchanged.
        """
        if not getattr(outcome, "final_text", ""):
            self.store.append_stream_chunk(assistant_message_id, "No response was generated.")
        if variant_mode:
            return self.store.finalize_variant_stream(assistant_message_id)
        return self.store.mark_message_complete(assistant_message_id)

    def _finalize_agent_cancelled(
        self, assistant_message_id: str, session_id: str, *, variant_mode: bool,
    ) -> ConsoleSubmitResult:
        """Handle a ``RUN_CANCELLED`` outcome: the placeholder becomes ``failed``.

        Per the agent turn-control spec, a runtime-reported cancellation is a
        terminal failure, not a user-initiated stop. If the placeholder has
        vanished, append a failed assistant message carrying the visible copy.
        """
        visible_copy = "Response stopped/cancelled."
        placeholder = self._ensure_assistant_placeholder(assistant_message_id, session_id)
        if placeholder is not None:
            failed = self.store.mark_message_failed(assistant_message_id)
        else:
            failed = self._append_failed_assistant(session_id, visible_copy)
        self._set_run_state(ConsoleRunState(ConsoleRunStatus.FAILED, visible_copy))
        return ConsoleSubmitResult(True, True, failed.content)

    def _finalize_agent_failure(
        self, assistant_message_id: str, session_id: str, outcome: Any,
        *, variant_mode: bool,
    ) -> ConsoleSubmitResult:
        """Handle ``RUN_ERROR``, ``RUN_STUCK``, or any unknown non-done outcome.

        A present placeholder is marked ``failed`` and a system row explains
        the failure (preserving the existing failure UX). If the placeholder
        is missing, the runtime may have already written an assistant message
        (e.g. streamed partial content before the error); use it when
        possible, otherwise append a new failed assistant message.
        """
        visible_copy = self._agent_failure_visible_copy(outcome)
        if "provider returned HTTP" in visible_copy and (
            self._session_history_carries_images(session_id)
        ):
            visible_copy += self._IMAGE_REJECTION_RECOVERY_HINT
        placeholder = self._ensure_assistant_placeholder(assistant_message_id, session_id)
        if placeholder is not None:
            failed = self.store.mark_message_failed(assistant_message_id)
            self._append_failure_system_row(session_id, visible_copy)
            self._set_run_state(ConsoleRunState(ConsoleRunStatus.FAILED, visible_copy))
            return ConsoleSubmitResult(True, True, failed.content)

        runtime_written = self._find_runtime_written_assistant(session_id)
        if runtime_written is not None and runtime_written.status in {"pending", "streaming"}:
            self.store.append_stream_chunk(runtime_written.id, f"\n\n{visible_copy}")
            failed = self.store.mark_message_failed(runtime_written.id)
        else:
            failed = self._append_failed_assistant(session_id, visible_copy)
        self._set_run_state(ConsoleRunState(ConsoleRunStatus.FAILED, visible_copy))
        return ConsoleSubmitResult(True, True, failed.content)

    def _finalize_agent_success(
        self, assistant_message_id: str, session_id: str, outcome: Any,
        *, variant_mode: bool,
    ) -> ConsoleSubmitResult:
        """Handle ``RUN_DONE``: complete the placeholder (or a runtime-written one).

        An empty ``final_text`` is replaced with the fallback copy ``No
        response was generated.``. If the placeholder is missing, the runtime
        may have streamed content into an assistant row already; complete it
        when possible, otherwise append a new assistant message.
        """
        placeholder = self._ensure_assistant_placeholder(assistant_message_id, session_id)
        if placeholder is not None:
            completed = self._complete_agent_message(assistant_message_id, variant_mode, outcome)
            self._set_run_state(ConsoleRunState(ConsoleRunStatus.COMPLETED, "Response complete."))
            return ConsoleSubmitResult(True, True, completed.content)

        runtime_written = self._find_runtime_written_assistant(session_id)
        if runtime_written is not None and runtime_written.status in {"pending", "streaming"}:
            completed = self._complete_agent_message(runtime_written.id, variant_mode=False, outcome=outcome)
            self._set_run_state(ConsoleRunState(ConsoleRunStatus.COMPLETED, "Response complete."))
            return ConsoleSubmitResult(True, True, completed.content)

        final_text = getattr(outcome, "final_text", "") or "No response was generated."
        completed = self.store.append_message(
            session_id, role=ConsoleMessageRole.ASSISTANT, content=final_text)
        self._set_run_state(ConsoleRunState(ConsoleRunStatus.COMPLETED, "Response complete."))
        return ConsoleSubmitResult(True, True, completed.content)

    def _append_failed_assistant(
        self, session_id: str, visible_copy: str,
    ) -> ConsoleChatMessage:
        """Append a failed assistant message carrying ``visible_copy``.

        The store's terminal-status validation only accepts pending/streaming
        assistant messages, so the message is created empty, the copy is
        streamed in, and then it is marked failed.
        """
        message = self.store.append_message(
            session_id, role=ConsoleMessageRole.ASSISTANT, content="")
        self.store.append_stream_chunk(message.id, visible_copy)
        return self.store.mark_message_failed(message.id)

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
            if getattr(step, "kind", None) == STEP_ERROR and getattr(
                step, "summary", ""
            ):
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
            usable = [
                attachment
                for attachment in message.attachments
                if attachment.data is not None
            ]
            if not usable:
                continue
            take = min(len(usable), budget)
            allowed_counts[message.id] = take
            budget -= take

        payloads: list[dict[str, Any]] = []
        seen_user = False
        for message in session_messages:
            if message.role not in {
                ConsoleMessageRole.USER,
                ConsoleMessageRole.ASSISTANT,
            }:
                continue
            if skip_failed and message.status == "failed":
                continue
            # A seeded character greeting is a display-only assistant turn:
            # keep it out of the provider payload so strict providers (Anthropic,
            # Gemini) never see an assistant-first message array (task-427).
            if not seen_user and message.role is ConsoleMessageRole.ASSISTANT:
                continue
            if message.role is ConsoleMessageRole.USER:
                seen_user = True
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
                    attachment
                    for attachment in message.attachments
                    if attachment.data is not None
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
                        image_url_part(
                            attachment.data, attachment.mime_type or "image/png"
                        )
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
            return (
                self.store.session_id_for_message(self._active_assistant_message_id)
                == session_id
            )
        except KeyError:
            return False

    def streaming_session_id(self) -> str | None:
        """Return the session with an in-flight stream, for tab status glyphs."""
        if self._active_assistant_message_id is None:
            return None
        try:
            return self.store.session_id_for_message(self._active_assistant_message_id)
        except KeyError:
            return None

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
