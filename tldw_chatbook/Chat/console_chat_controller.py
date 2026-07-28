"""Native Console chat controller for send, stream, stop, and retry flows."""

from __future__ import annotations

import asyncio
import copy
import functools
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal, Protocol
from uuid import uuid4

from tldw_chatbook.Chat.attachment_core import (
    image_url_part,
    max_history_images,
    vision_block_reason,
)
from tldw_chatbook.Chat.console_chat_models import (
    CONSOLE_CAP_REFUSAL_TITLE_LIMIT,
    CONSOLE_DEFAULT_MAX_PARALLEL_RUNS,
    ConsoleChatMessage,
    ConsoleCitationNoticeCode,
    ConsoleCitationPhase,
    ConsoleCitationPresentation,
    ConsoleContextSnapshot,
    ConsoleMessageRole,
    ConsoleProviderSelection,
    ConsoleRunMarker,
    ConsoleRunState,
    ConsoleRunStatus,
    ConsoleStagedSource,
    MessageAttachment,
    derive_console_session_title,
    is_default_console_session_title,
)
from tldw_chatbook.Chat.citation_repair import (
    REPAIR_ANSWER_BODY_UTF8_BYTES_MAX,
    CitationRepairContract,
    CitationRepairDecision,
    build_citation_repair_messages,
    decide_citation_repair,
    repair_request_fits_model_window,
    select_repaired_body,
)
from tldw_chatbook.Chat.citation_trace_builder import (
    CitationTraceBuilder,
    CitationTraceBuildUnavailable,
)
from tldw_chatbook.Chat.citation_trace_models import SealedCitationWrite
from tldw_chatbook.Chat.console_chat_store import (
    ConsoleChatSession,
    ConsoleChatStore,
    TerminalCitationFinalizer,
)
from tldw_chatbook.Chat.console_command_grammar import COMMAND_PREFIX
from tldw_chatbook.Chat.console_history_budget import (
    DEFAULT_RESPONSE_RESERVATION,
    bound_messages_to_window,
    count_console_messages_tokens,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_skill_resolver import (
    MENTION_SIGIL,
    SKILL_MENTION_SKIPPED_NOTE,
    SKILL_UNTRUSTED_REFUSE,
    SkillCommandCandidate,
    cap_skill_args,
    find_embedded_mentions,
    resolve_skill_command,
)
from loguru import logger

from tldw_chatbook.Agents.builtin_tool_gate import build_builtin_gate
from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall, MCPToolProvider
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider
from tldw_chatbook.config import get_cli_setting
from tldw_chatbook.Internal_Prompts import get_internal_prompt
from tldw_chatbook.MCP.permission_store import BUILTIN_TOOL_SERVER_KEY
from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError
from tldw_chatbook.Tools.file_operation_tools import path_precheck_failed
from tldw_chatbook.Utils.input_validation import sanitize_string, validate_text_input
from tldw_chatbook.Chat.provider_failures import (  # noqa: F401  (re-export: tests and callers import describe_stream_failure from here)
    describe_stream_failure,
)
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderResolution,
    ConsoleProviderStreamSignals,
)
from tldw_chatbook.model_capabilities import is_vision_capable

if TYPE_CHECKING:
    from tldw_chatbook.Agents.agent_models import ToolCall
    from tldw_chatbook.Agents.builtin_tool_gate import BuiltinToolGate
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge


#: Fallback used when no `mcp_approval_timeout_seconds` seam is injected --
#: mirrors `UnifiedMCPControlPlaneService.approval_timeout_seconds`'s own
#: default (task-201/T2), read directly here since the controller has no
#: dependency on that service (T6 wires the service into `MCPToolProvider`,
#: not into this controller).
#:
#: task-545/T6: built-in tool approvals reuse this SAME timeout (routed
#: through `request_mcp_approvals`/`build_tool_review_hook`), so this value
#: must stay strictly BELOW `RunBudget.max_tool_call_seconds` (300s at
#: defaults, `Agents/agent_models.py`) -- never equal, never above. The
#: approval wait happens INSIDE `agent_service._call_with_timeout`'s own
#: per-call wrapper (task-327): if the approval timeout were >= the
#: tool-call ceiling, `_call_with_timeout` would fire first, tell the agent
#: the call failed/timed out, and the underlying `invoke()` call would
#: still be running on the (by then abandoned) worker thread -- so a late
#: approval from the user would execute the tool for real after the
#: runtime already moved on and reported failure. Any future change to
#: either constant must preserve `approval_timeout < max_tool_call_seconds`.
_DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS = 120.0
#: Poll granularity for `request_mcp_approvals`'s wait loop (binding, from
#: the Phase-5 plan) -- also the worst-case slack added on top of a
#: configured timeout/cancellation before this method observes it.
_MCP_APPROVAL_POLL_SECONDS = 1.0
#: Fallback used when no `skill_install_confirm_timeout_seconds` seam is
#: injected -- mirrors `_DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS`'s role for
#: `request_skill_install_confirm`'s own wait loop.
_DEFAULT_SKILL_INSTALL_CONFIRM_TIMEOUT_SECONDS = 120.0
#: Fallback used when no `skill_script_confirm_timeout_seconds` seam is
#: injected -- mirrors `_DEFAULT_SKILL_INSTALL_CONFIRM_TIMEOUT_SECONDS`'s
#: role for `request_skill_script_confirm`'s own wait loop.
_DEFAULT_SKILL_SCRIPT_CONFIRM_TIMEOUT_SECONDS = 120.0
#: TASK-1050: synthetic round id `set_run_pending_approval`'s deprecated
#: boolean shim registers under, internally, so its add/discard composes
#: safely with the round-keyed `_pending_approvals` accounting (see that
#: method's docstring) without ever colliding with a real bridge round's
#: `uuid4()` id -- every genuine round id is a UUID string; this is not.
_LEGACY_PENDING_APPROVAL_ROUND_ID = "__legacy_pending_approval__"


MAX_CONSOLE_DRAFT_LENGTH = 100_000
CONSOLE_CONTINUE_INSTRUCTION = "Continue and extend the selected message."

# Private payload-row key threading a transcript message's native id from the
# payload builder to the dispatch choke point, where `/rewind`
# "summarize up to here" compaction anchors the boundary by IDENTITY rather
# than by content (see `_apply_context_summary_compaction`). It is opt-in
# (send paths only, `annotate_ids=True`) and ALWAYS stripped from every row
# before the payload leaves the controller for a provider/agent, so no
# provider ever sees it.
NATIVE_MESSAGE_ID_KEY = "_native_message_id"


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


def _collect_mcp_pending(
    provider: MCPToolProvider, calls: list["ToolCall"]
) -> list["MCPPendingCall"]:
    """Resolve each call's MCP gate; return the subset that needs asking.

    Extracted so `build_mcp_review_hook` (MCP-only, still used directly by
    its own long-standing tests) and `build_tool_review_hook` (T6: the
    run-level hook that folds built-ins in too) share this ONE walk over
    `provider.pending_gate_for` rather than one copying the other's body.
    `None` per call means either "not an MCP call this provider owns" or
    "an MCP call whose current state doesn't need asking" -- see
    `pending_gate_for`'s own docstring for why callers do not need to
    distinguish those two cases.
    """
    pending: list["MCPPendingCall"] = []
    for call in calls:
        gate = provider.pending_gate_for(call.name, call.args)
        if gate is not None:
            pending.append(gate)
    return pending


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
        pending = _collect_mcp_pending(provider, calls)
        if not pending:
            return {}
        decisions = request_mcp_approvals(pending)
        provider.apply_batch_decisions(decisions)
        return {call.llm_name: "proceed" for call in pending}

    return review_tool_calls


def build_tool_review_hook(
    builtin_gate: "BuiltinToolGate",
    builtin_provider: "BuiltinToolProvider",
    mcp_provider: MCPToolProvider | None,
    request_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
    *,
    workspace_id: str | None = None,
) -> Callable[[list["ToolCall"]], dict[str, str]]:
    """Build THIS run's run-level `review_tool_calls` hook (P5-T6/task-545).

    Unlike `build_mcp_review_hook`, this is wired UNCONDITIONALLY -- every
    run gets one, even a user with no MCP servers configured at all --
    because built-in tools (calculator/datetime today, more later) must be
    gated regardless of whether MCP happens to be composed this turn.
    `BuiltinToolProvider.invoke` already enforces the gate as defense in
    depth, but without this hook the ONLY review a built-in call would ever
    get is that per-call fallback -- never the batched, one-card-per-turn
    review MCP calls already get, and never a chance to ask before
    dispatch for calls this hook doesn't stamp.

    Routing per call, MCP first: `mcp_provider.pending_gate_for` (when a
    provider was composed this run) is asked before the built-in provider,
    so a name that provider actually owns is never mistakenly re-resolved
    against the built-in side too. Note this hook's own precedent is the
    OPPOSITE of `console_agent_bridge._non_colliding_mcp_names`, which
    resolves a name collision the other way -- it drops the colliding MCP
    name from the run's registry so the built-in wins composition. That
    inconsistency is moot in practice: `MCP/tool_naming.py:106` always
    mints MCP tool names as `mcp__<server>__<tool>`, which can never equal
    a bare built-in name like `calculator`/`get_current_datetime`, so no
    call is ever ambiguous between the two orders. A name neither provider
    claims (a skill, `spawn_subagent`, `find_tools`, ...) passes through
    unreviewed, exactly as it does for `build_mcp_review_hook` today.

    Built-in rows use `server_key=BUILTIN_TOOL_SERVER_KEY`
    (`"agent:builtin"`), `server_label="Built-in"`, and `reason=
    "risk_floored"` when `EffectiveToolState.risk_floored` else `"ask"`
    (built-ins never set `config_changed` -- see `resolve_builtin_state`'s
    own docstring for why). Every built-in row's `path_precheck_failed`
    (TASK-1231/F3 AC2) is set via `Tools.file_operation_tools.
    path_precheck_failed`: for `read_file`/`list_directory`/`write_file`
    this pre-flights the SAME `allowed_file_roots`/`validate_path_multi`
    check `invoke()` runs at dispatch, so the approval card can warn the
    user this exact call will fail even if approved -- it never gates or
    auto-denies; `False` for every other builtin tool and every MCP row.
    Only a resolved `"ask"` state ever produces a row: `"allow"` never
    prompts, and `"deny"` is refused outright by
    `invoke()`'s own gate WITHOUT ever reaching the user -- a tool the
    operator switched Off must not appear on the approval card at all.
    Nor does an `"ask"` tool that already has a live session approval
    (`builtin_gate.is_session_approved(name)`) -- review finding 1
    (T6 review): `resolve()`/`resolve_builtin_state` read the permission
    store ONLY, never session approvals, so without this check a user who
    picked "Approve for session" on turn 1 would be re-prompted on turn 2
    even though `invoke()`'s own `check()` already honors that same
    session approval and would execute it anyway. Mirrors MCP's own
    `pending_gate_for`, which applies the identical
    `_is_session_approved_safe` skip for exactly this reason.

    `options=("approve_once", "approve_session", "deny")` -- deliberately
    excluding ONLY `"always_allow"` (verified at
    `Agents/mcp_tool_provider.py:556-564`: `always_allow` is the sole
    PERSISTENT write via `set_tool_state`; `approve_session` is an
    in-memory session cache and `deny`/`timeout` are turn-scoped refusals
    that persist nothing). `"deny"` MUST stay offered -- an earlier draft
    of this design mistakenly dropped it too, which would have made a
    built-in row impossible to refuse from the card at all (the bulk "Deny
    all" button would silently leave it on whatever the row's default
    was).

    Mirrors `build_mcp_review_hook`'s I3 clear-at-entry discipline, extended
    to the built-in side: `builtin_gate.begin_turn()` runs FIRST,
    unconditionally -- before the MCP stamp clear, before any
    `pending_gate_for`/`resolve` call, before the `request_approvals` round
    trip -- so a raising round trip can never leave a stale built-in stamp
    (or a stale cached permission payload) live for the next turn to
    consume. `mcp_provider.apply_batch_decisions({})` follows the same
    reasoning for the MCP side, only when a provider was actually composed
    this run.

    Exactly ONE `request_approvals` round trip is made per turn, carrying
    BOTH the MCP and built-in pending rows together -- never one call per
    owner. Decisions are then applied back to each owner separately:
    `mcp_provider.apply_batch_decisions(...)` for MCP rows,
    `builtin_gate.stamp(name, decision)` for built-in rows. The returned
    verdict map is `{name: "proceed"}` for every call this hook gated this
    turn (MCP or built-in), purely documentary like
    `build_mcp_review_hook`'s own -- the actual allow/deny outcome is left
    to `invoke()`'s gate on dispatch, which is the single place that
    produces refusal copy and records the audit decision.

    Args:
        builtin_gate: THIS run's `BuiltinToolGate` -- the SAME instance
            the run's `BuiltinToolProvider.invoke` checks, so a stamp
            written here is visible there. Two separate instances would
            mean a decision made here is invisible to `invoke()`, silently
            re-prompting (a stamp `invoke()` never sees) or failing closed
            (an approval that never reaches the gate that checks it).
        builtin_provider: THIS run's `BuiltinToolProvider` (only
            `.tool_for(name)` is used here, to resolve a `ToolCall.name`
            to the `Tool` object `builtin_gate.resolve` needs).
        mcp_provider: THIS run's already-composed `MCPToolProvider`, or
            `None` when no MCP tools should be offered this run (no
            service, kill switch on, or composition yielded nothing) --
            the entire point of this hook existing separately from
            `build_mcp_review_hook` is that built-in gating must not
            depend on this being non-`None`.
        request_approvals: The bound `ConsoleChatController.
            request_mcp_approvals` method for THIS run (the name predates
            built-in gating; the method itself is owner-agnostic -- it
            only reads `MCPPendingCall` fields, never assumes MCP
            ownership).
        workspace_id: THIS run's OWN workspace id (round 1 review CRITICAL
            1) -- e.g. `self.store.session_workspace_id(session_id)` --
            threaded into every builtin file-tool row's `path_precheck_
            failed` computation via `Tools.file_operation_tools.
            path_precheck_failed`'s own `workspace_id=` parameter. Must be
            the SAME workspace id `ConsoleAgentBridge.run_reply` resolves
            for this run's real dispatch (`BuiltinToolProvider(workspace_
            id=...)`) -- otherwise the pre-flight can resolve a DIFFERENT
            workspace than the one the call will actually run against
            (e.g. whatever happens to be active in the UI for a parked
            background session), making the warning wrong in either
            direction. `None` (the default) reproduces the pre-existing
            active-workspace fallback for a caller with no session
            context at all; every caller that has a real session id MUST
            resolve and pass its workspace id.

    Returns:
        A `review_tool_calls`-shaped callable suitable for `LoopDeps`/
        `AgentService(review_tool_calls=...)`.
    """

    def review_tool_calls(calls: list["ToolCall"]) -> dict[str, str]:
        builtin_gate.begin_turn()
        if mcp_provider is not None:
            mcp_provider.apply_batch_decisions({})

        mcp_pending = (
            _collect_mcp_pending(mcp_provider, calls)
            if mcp_provider is not None
            else []
        )
        mcp_claimed_names = {row.llm_name for row in mcp_pending}

        # Minor (round 1 review): memoize `allowed_file_roots` across every
        # builtin file-tool row THIS batch checks -- `workspace_id` is fixed
        # for the whole call, so a turn with several read_file/write_file
        # rows would otherwise re-hit the workspace registry (a fresh
        # sqlite3 connection per `WorkspaceDB` operation) once per row.
        # Fresh dict per `review_tool_calls` call -- never reused across
        # turns, so a folder binding added/removed between turns is still
        # picked up on the very next call.
        path_roots_cache: dict[bool, tuple] = {}

        builtin_pending: list["MCPPendingCall"] = []
        for call in calls:
            if call.name in mcp_claimed_names:
                continue
            tool = builtin_provider.tool_for(call.name)
            if tool is None:
                continue  # not ours either -- a skill/native tool, unreviewed
            state = builtin_gate.resolve(tool)
            if state.state != "ask":
                # "allow" never prompts; "deny" is refused outright by
                # invoke()'s own gate -- neither is offered a card.
                continue
            if builtin_gate.is_session_approved(call.name):
                # Review finding 1 (T6 review): already approved for this
                # session -- `invoke()`'s own `check()` will honor it via
                # the identical `is_session_approved` read, so re-asking
                # here would just re-prompt for a decision the user
                # already made. Not added to `builtin_pending` and so
                # never mentioned in the returned verdict map either --
                # exactly as undecided-but-not-needed-this-turn MCP calls
                # already work (see this function's own docstring).
                continue
            builtin_pending.append(
                MCPPendingCall(
                    llm_name=call.name,
                    server_key=BUILTIN_TOOL_SERVER_KEY,
                    tool_name=call.name,
                    server_label="Built-in",
                    arguments=dict(call.args or {}),
                    reason="risk_floored" if state.risk_floored else "ask",
                    options=("approve_once", "approve_session", "deny"),
                    # TASK-1231/F3 AC2: pre-flight the roots check for the
                    # three file tools -- never gates or auto-denies, just
                    # tells the card this specific path is doomed even if
                    # approved (see path_precheck_failed's own docstring).
                    # `workspace_id=workspace_id` (round 1 review CRITICAL
                    # 1): the pre-flight MUST resolve THIS run's own
                    # workspace, never whatever happens to be active in the
                    # UI -- see this function's own docstring.
                    path_precheck_failed=path_precheck_failed(
                        call.name,
                        call.args,
                        workspace_id=workspace_id,
                        roots_cache=path_roots_cache,
                    ),
                )
            )

        all_pending = mcp_pending + builtin_pending
        if not all_pending:
            return {}
        decisions = request_approvals(all_pending)
        if mcp_provider is not None:
            mcp_decisions = {
                name: decisions[name] for name in mcp_claimed_names if name in decisions
            }
            mcp_provider.apply_batch_decisions(mcp_decisions)
        for row in builtin_pending:
            decision = decisions.get(row.llm_name)
            if decision is not None:
                builtin_gate.stamp(row.llm_name, decision)
        return {row.llm_name: "proceed" for row in all_pending}

    return review_tool_calls


def _split_skill_command_word(text: str) -> tuple[str, str]:
    """Split a ``$word rest`` string into its leading token and the remainder.

    Mirrors ``console_command_grammar._split_leading_token``'s single-
    whitespace-character split rule. That helper is module-private (by
    design -- callers own their own tokenization per its module docstring),
    so this is a deliberate small duplicate rather than an import, the same
    precedent ``chat_screen.ChatScreen._split_console_skill_name_args``
    already follows. ``text`` is assumed to already start with
    `MENTION_SIGIL` (the `$`-mention leading form, not `COMMAND_PREFIX`'s
    `/` -- its sole caller is `_apply_skill_substitution`'s leading-form
    branch).
    """
    for index, character in enumerate(text):
        if character.isspace():
            return text[:index], text[index + 1 :]
    return text, ""


def _render_skill_bundle_block(results: Iterable[Mapping[str, Any]]) -> str:
    """Render one combined "Bundled files" block for a turn's bound skills.

    Task 5 (skills-fork-reachability): `_apply_skill_substitution` builds
    this as pure string work from `execute_skill` results it already holds
    -- no re-execution, no extra service calls -- for every skill actually
    bound this turn (leading-resolved, or embedded mentions that spliced).
    `run_reply` (never here) is the only place that ever appends the
    returned string to a message, so plain sends and the stored transcript
    never see it.

    Row format matches `_BridgeSkillRunner.run`'s own bundle-pointer block
    byte-for-byte (Task 4) -- ``{path} ({size} bytes)`` / ``{path} ({size}
    bytes, binary)``, comma-joined under one combined header -- so a bound
    skill's `skill_file` reads look identical whether granted turn-side
    (this function) or fork-side (a spawned skill reading its own bundle).

    Args:
        results: `execute_skill` result mappings for the bound skills, in
            any order; a result missing `reference_files` (absent when a
            skill has no bundle beyond SKILL.md) or with it empty
            contributes no rows.

    Returns:
        The combined block, or ``""`` when no result carries any rows.
    """
    rows: list[str] = []
    for result in results:
        refs = result.get("reference_files") if isinstance(result, Mapping) else None
        if not refs:
            continue
        rows.extend(
            f"{ref['path']} ({ref['size']} bytes"
            f"{'' if ref.get('is_text', True) else ', binary'})"
            for ref in refs
        )
    if not rows:
        return ""
    return "Bundled files (readable via skill_file): " + ", ".join(rows)


class ConsoleProviderGatewayProtocol(Protocol):
    """Provider gateway surface required by the Console controller."""

    async def resolve_for_send(self, selection: ConsoleProviderSelection) -> Any:
        """Resolve provider readiness for a send."""

    async def stream_chat(
        self,
        resolution: Any,
        messages: list[dict[str, Any]],
        signals: ConsoleProviderStreamSignals | None = None,
    ) -> Any:
        """Stream response chunks for provider messages."""


@dataclass(slots=True)
class ConsoleCitationRepairSession:
    contract: CitationRepairContract | None
    resolution: ConsoleProviderResolution | None
    attempt_started: bool = False
    selection_committed: bool = False
    phase: str = "initial_streaming"
    cancel_reason: Literal["user", "session_close", "shutdown"] | None = None

    def clear_governed_state(self) -> None:
        """Release request content and provider configuration after cleanup."""
        self.contract = None
        self.resolution = None


@dataclass(frozen=True, slots=True)
class ConsoleCitationSelectionOutcome:
    selected_body: str
    state: Literal["bypassed", "valid", "repaired", "unavailable", "canceled"]


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
        rag_capture_provider: "Callable[[str], Awaitable[Any]] | None" = None,
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
        self._rag_capture_provider = rag_capture_provider
        # Parallel-agents spec §2: run state is a PER-SESSION map, not a
        # single global slot -- two sessions can each have their own
        # in-flight/terminal run without stamping each other. `run_state`/
        # `run_state_history` below become read-only facades over these maps
        # (see the property block right after `__init__`); every WRITE goes
        # through `_set_run_state`/`_clear_terminal_run_state`, which take an
        # explicit `session_id` so a background completion can target its
        # OWNING session instead of whatever the user currently has open.
        self._run_states: dict[str, ConsoleRunState] = {}
        self._run_state_histories: dict[str, list[ConsoleRunStatus]] = {}
        # Parallel-agents spec §6: run-marker state (Task 7). Both maps are
        # keyed by session id like the run-state maps above, but track
        # marker-only bookkeeping that ``_run_states`` doesn't capture on
        # its own:
        #   - `_pending_approvals`: TASK-1050 (round-keyed accounting):
        #     session id -> the set of outstanding approval-like round ids
        #     (real bridge round/request ids, or the deprecated shim's
        #     `_LEGACY_PENDING_APPROVAL_ROUND_ID` sentinel) currently
        #     blocking that session. A session is "pending" iff it is a KEY
        #     here with a non-empty value set -- `add_pending_round`/
        #     `discard_pending_round` are the ONLY writers and keep that
        #     invariant (an emptied set is popped, never left as `{}`), so
        #     every reader (`run_marker_for`, `fleet_summary_counts`,
        #     plain `in`/`not in` membership tests throughout the test
        #     suite) can keep treating this exactly like the plain
        #     `set[str]` of session ids it used to be. Was a single global
        #     boolean per session (`set_run_pending_approval`) shared by
        #     THREE independent bridges (MCP tool approvals, skill-install
        #     confirms, skill-script confirms) -- whichever bridge's round
        #     finished first cleared the badge even if a SIBLING round from
        #     a different bridge (or a second round from the SAME bridge)
        #     was still outstanding for that session. `set_run_pending_
        #     approval` is now a deprecated boolean shim kept ONLY for
        #     callers that genuinely have no round id of their own (see its
        #     docstring) -- Task 9 originally wired the approval paths that
        #     called it; TASK-1050 migrated all three bridges themselves to
        #     `add_pending_round`/`discard_pending_round` with their real
        #     round/request ids. Named to avoid colliding with the
        #     PRE-EXISTING `self.set_pending_approval` INSTANCE ATTRIBUTE
        #     below (the MCP batch-approval UI callback slot, task-5) -- a
        #     same-named method here would be silently clobbered by that
        #     assignment.
        #   - `_unvisited_outcomes`: sessions whose run reached a terminal
        #     COMPLETED/FAILED status while NOT the active (viewed)
        #     session, stamped by `_set_run_state` and cleared by
        #     `mark_session_visited` (called from `switch_session`). The
        #     viewed session's own terminal transition is seen live and is
        #     deliberately never stamped here.
        self._pending_approvals: dict[str, set[str]] = {}
        self._unvisited_outcomes: dict[str, ConsoleRunMarker] = {}
        #: F2b fix (Qodo wave): guards every mutation of `_pending_
        #: approvals`, `_parked_approval_payloads`, and `_pending_
        #: approval_rounds` -- the three approval-marker collections a
        #: worker thread (`request_mcp_approvals`'s own body/`finally`) can
        #: mutate WHILE the UI thread iterates them every ~0.2s sync tick
        #: (`fleet_summary_counts`). An unguarded set/dict mutation racing
        #: an unguarded iteration over the SAME object can raise
        #: `RuntimeError: Set/dictionary changed size during iteration`.
        #: `_unvisited_outcomes`/`_run_states` are NOT covered here: both
        #: are written only from the main thread (`_set_run_state`,
        #: `mark_session_visited`), never from a worker thread, so they
        #: carry no cross-thread hazard this lock needs to close.
        self._approval_state_lock = threading.Lock()
        #: Optional owner hook invoked once a submit is accepted (user message
        #: persisted, run about to start) so the composer can clear immediately
        #: instead of holding the sent text for the whole run.
        self.on_submission_accepted: Callable[[], None] | None = None
        # Task 3b: PER-SESSION maps, mirroring `_run_states`' own keying --
        # two sessions can each have their own in-flight stream/cancel state
        # without clobbering each other. Written/cleared at the SAME
        # lifecycle points the old singulars were (`_stream_assistant_
        # response`/`_run_agent_reply`'s start and `finally`), keyed by the
        # run's OWNING session id (the same `owner_id`/`session_id` locals
        # Task 1 threaded), never by whatever session the user currently has
        # open. `stop_active_run` is the one place that DELIBERATELY reads
        # by the ACTIVE (viewed) session -- see its own docstring.
        self._active_assistant_message_ids: dict[str, str] = {}
        self._active_stream_tasks: dict[str, asyncio.Task] = {}
        self._stop_requested = False
        #: F5 fix (Qodo wave): set ONLY by ``shutdown()`` and NEVER reset
        #: (unlike ``_stop_requested``, which every run's own lifecycle
        #: resets to ``False`` -- see ``shutdown``/``_run_agent_reply``/
        #: ``_stream_assistant_response``'s own resets -- making it
        #: race-dependent whether a still-polling bridge thread observes a
        #: Stop that raced a reset). The three worker-thread approval/
        #: confirm bridges (``request_mcp_approvals``, ``request_skill_
        #: install_confirm``, ``request_skill_script_confirm``) OR this
        #: with ``_is_active_session_cancelled()`` at their poll sites
        #: instead of the old, session-agnostic ``_stop_requested`` --
        #: a single session's Stop must never deny another session's
        #: unrelated approval round; only real process teardown (the one
        #: case where every session's run legitimately ends at once) does.
        self._shutdown_requested = threading.Event()
        # Rebase note (dev citation-repair vs. Task 3b): dev added this as a
        # singular slot (no per-session awareness); rescoped here the same
        # way as the two maps above -- keyed by the run's OWNING session id,
        # so a background session's in-flight repair can never be read/
        # cleared by another session's close/stop/teardown path.
        self._active_citation_repair_sessions: dict[
            str, ConsoleCitationRepairSession
        ] = {}
        self._original_attempts: OrderedDict[str, str] = OrderedDict()
        #: Per-run cancellation flag for the agent bridge's background
        #: thread (see ``_run_agent_reply``), keyed by owning session id
        #: like the two maps above. ``threading.Event`` rather than a
        #: shared bool: ``asyncio.to_thread`` survives Task cancellation
        #: (the coroutine detaches from the still-running OS thread), so
        #: the closure handed to that thread must observe a signal that,
        #: once set, is never reset for THIS run -- unlike
        #: ``_stop_requested``, which the run's own ``finally`` block
        #: resets as soon as the coroutine side is done (task-227).
        #: ``_stop_requested`` itself stays a single shared flag (Task 3b
        #: did not rescope it) -- see ``_is_active_session_cancelled``'s
        #: docstring for the resulting, deliberately-scoped-down, limit on
        #: the three worker-thread approval/confirm bridges below.
        self._active_cancel_events: dict[str, threading.Event] = {}
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
        #: Task 9 (parked background approvals): UI-thread callback invoked
        #: (via ``self.app.call_from_thread``) when ``request_mcp_approvals``
        #: raises a round for a NON-active session -- sets the fleet
        #: pending-approval badge and fires the one-per-card toast, WITHOUT
        #: touching ``set_pending_approval``'s mounted-card slot (that stays
        #: reserved for whichever session is actually being viewed). Wired
        #: to ``ChatScreen._park_console_approval`` by ``_ensure_console_
        #: chat_controller``, mirroring ``set_pending_approval``'s own
        #: wiring. ``None`` in most controller-only tests, matching every
        #: other UI bridge slot here.
        self.park_pending_approval: Callable[[str], None] | None = None
        #: Task 10 (background completion toasts): UI-thread callback
        #: invoked DIRECTLY (never via ``self.app.call_from_thread`` --
        #: unlike the two bridges above, every terminal ``_set_run_state``
        #: call already runs on the main event-loop thread: worker-thread
        #: agent runs resume here only after ``await asyncio.to_thread(...)``
        #: returns in ``_run_agent_reply``) from ``_set_run_state``'s
        #: non-active COMPLETED/FAILED branch, once per transition INTO a
        #: terminal state. Wired to ``ChatScreen._notify_console_run_
        #: outcome`` by ``_ensure_console_chat_controller``, mirroring
        #: ``park_pending_approval``'s wiring and reusing its exact
        #: session-title/workspace-name resolution
        #: (``ChatScreen._console_workspace_display_name``). ``None`` in
        #: most controller-only tests, matching every other UI bridge slot
        #: here.
        self.notify_run_outcome: Callable[[str, ConsoleRunStatus], None] | None = None
        #: Optional override for how long ``request_mcp_approvals`` waits
        #: for a human decision before failing every undecided call to
        #: ``"timeout"``. Defaults to reading ``[mcp] approval_timeout_
        #: seconds`` (T2's ``approval_timeout_seconds``) when unset.
        self.mcp_approval_timeout_seconds: Callable[[], float] | None = None
        #: Task 9 (Fix round 1): each batch-approval round's release signal
        #: + shared decisions holder + owning session id, keyed by a
        #: freshly minted ROUND id (``uuid4()``, stamped into the payload
        #: as ``"round_id"`` and round-tripped through ``ChatApprovalCard``
        #: -> ``ApprovalDecided`` -> ``resolve_pending_approval``) --
        #: mirrors ``_pending_skill_script_rounds``'s identical
        #: ``request_id``-keyed design. Superseded TWO earlier, both-wrong
        #: shapes: the pre-Task-9 single ``_pending_approval_event``/
        #: ``_pending_approval_decisions`` pair (only ever tracked ONE
        #: round controller-wide -- fatal once two sessions can each have
        #: their own concurrent pending approval), and this task's own
        #: first draft keyed by session id alone (still wrong: `Approval
        #: Decided` travels as an async Textual message, so a
        #: `switch_session` landing in the gap between the user's click and
        #: the handler running could resolve session A's decision against
        #: session B's completely different batch -- review CRITICAL
        #: finding, fix round 1). Read/written from the UI thread by
        #: ``resolve_pending_approval``, which resolves ONLY the round
        #: whose id was stamped onto the card the user actually decided --
        #: never "whichever session happens to be active right now".
        self._pending_approval_rounds: dict[str, dict[str, Any]] = {}
        #: Task 9: retained payload for a PARKED round (session_id !=
        #: active_session_id at round-start), keyed by owning session id --
        #: the exact dict ``request_mcp_approvals`` would otherwise have
        #: pushed straight to ``set_pending_approval``. ``switch_session``
        #: re-derives the mounted card from this map every time the user
        #: visits (or re-visits) the session, per the spec's "card state
        #: derives from the run's pending review state, not mounted-widget
        #: lifetime" contract -- never mutated by mount/unmount itself, only
        #: by the round's own start (park) and end (any resolution path).
        self._parked_approval_payloads: dict[str, dict[str, Any]] = {}
        #: UI-thread callback that pushes/clears the pending skill-install
        #: confirm payload into the owning screen's task-resume state
        #: (ChatScreen._set_console_pending_skill_install). Invoked through
        #: self.app.call_from_thread from request_skill_install_confirm.
        self.set_pending_skill_install: Callable[[dict | None], None] | None = None
        #: Optional test override for the confirm timeout.
        self.skill_install_confirm_timeout_seconds: Callable[[], float] | None = None
        #: TASK-910: per-round release Event + shared decision box + owning
        #: session id, keyed by a freshly minted request id -- mirrors
        #: `_pending_skill_script_rounds`' identical shape (itself task-581's
        #: fix for the same "single shared slot clobbers a second concurrent
        #: round" hazard `request_mcp_approvals` solved with `round_id`).
        #: Pre-TASK-910 this was a single `_pending_skill_install_event`/
        #: `_pending_skill_install_decision` pair -- fine while only one
        #: session could ever have a live install confirm, but parking makes
        #: two DIFFERENT background sessions' install confirms genuinely
        #: concurrent.
        self._pending_skill_install_rounds: dict[str, dict[str, Any]] = {}
        self._pending_skill_install_lock = threading.Lock()
        #: TASK-910 (parked background skill confirms): retained payload for
        #: a session-attributed `request_skill_install_confirm` round --
        #: mounted or parked, exactly like `_parked_approval_payloads`.
        #: `switch_session`/`new_session`/`close_session` re-derive the
        #: mounted card from this map on every activation, never from
        #: whatever the card happened to already be showing.
        self._parked_skill_install_payloads: dict[str, dict[str, Any]] = {}
        #: UI-thread callback that pushes/clears the pending skill-SCRIPT
        #: confirm payload into the owning screen's task-resume state.
        #: Invoked through self.app.call_from_thread from
        #: request_skill_script_confirm. Mirrors set_pending_skill_install,
        #: but the round-trip decision carries a "remember" flag too.
        self.set_pending_skill_script: Callable[[dict | None], None] | None = None
        #: Optional test override for the confirm timeout, mirroring
        #: `skill_install_confirm_timeout_seconds`.
        self.skill_script_confirm_timeout_seconds: Callable[[], float] | None = None
        #: The active script-confirm round's release Event + shared
        #: decision box ({"allow": bool, "remember": bool}), now also
        #: carrying the round's owning session id (TASK-910) so teardown can
        #: tell whether ANOTHER still-armed round belongs to the SAME
        #: session (must not clear the mounted card out from under it --
        #: see `request_skill_script_confirm`) independently of whether some
        #: OTHER session also has a round outstanding.
        #: task-581: rounds keyed by request_id, not a single slot. Two rounds
        #: armed at once previously clobbered each other's event/decision and
        #: both worker threads then blocked to their full deadline.
        self._pending_skill_script_rounds: dict[str, dict[str, Any]] = {}
        self._pending_skill_script_lock = threading.Lock()
        #: TASK-910 (parked background skill confirms): retained payload for
        #: a session-attributed `request_skill_script_confirm` round --
        #: mirrors `_parked_skill_install_payloads` above.
        self._parked_skill_script_payloads: dict[str, dict[str, Any]] = {}
        #: The currently-armed round's unique id (see `request_skill_script_
        #: confirm` / `resolve_pending_skill_script`). A resolve carrying any
        #: other id (including None) is dropped -- this is what stops a
        #: late button press from a torn-down round 1 from authorizing
        #: round 2's script. `None` whenever no round is armed.

    @property
    def run_state(self) -> ConsoleRunState:
        """The ACTIVE session's run state (parallel-agents spec §2).

        Read-only facade: the ~16 pre-existing read sites in chat_screen
        keep their semantics ("the viewed session's run"), while writes go
        through ``_set_run_state``/``_clear_terminal_run_state`` with an
        explicit owning session id. There is deliberately no setter --
        assigning ``controller.run_state = ...`` now raises ``AttributeError``
        so a stray direct-assignment writer (bypassing the per-session map)
        fails loudly instead of silently reintroducing the single-slot bug.

        Returns:
            The active session's recorded ``ConsoleRunState`` (a fresh idle
            state when the active session has no recorded run).
        """
        return self.run_state_for(self.store.active_session_id or "")

    def run_state_for(self, session_id: str) -> ConsoleRunState:
        """Return ``session_id``'s own run state (a fresh idle one when unset).

        Args:
            session_id: The session id to look up.

        Returns:
            The session's recorded ``ConsoleRunState``, or a fresh idle
            ``ConsoleRunState`` when the session has no recorded run.
        """
        return self._run_states.get(session_id) or ConsoleRunState()

    def run_states(self) -> dict[str, ConsoleRunState]:
        """Raw map snapshot incl. entries for closed sessions.

        This is the UNFILTERED ``self._run_states`` copy -- it can contain
        orphaned entries for sessions ``ConsoleChatStore.close_session`` has
        already removed (closing never touches the controller's map). Use
        ``in_flight_run_count`` (or ``_live_busy_session_ids``) for cap/fleet
        math; those exclude orphans. This raw snapshot is for callers that
        want the full recorded history regardless of session lifetime.

        Returns:
            A shallow copy of the internal session-id -> ``ConsoleRunState``
            map, including entries for sessions the store has since closed.
        """
        return dict(self._run_states)

    def _live_busy_session_ids(self) -> list[str]:
        """Busy session ids that still exist in the store, insertion-ordered.

        Intersects ``self._run_states`` with ``store.sessions()``: a session
        closed mid-VALIDATING leaves its entry in the map behind (Task 1
        review finding), and neither cap/fleet math nor the refusal copy's
        session list may count or name a session that no longer exists.
        Shared by ``in_flight_run_count`` and ``send_refusal_copy`` so both
        apply the same live-session filter.
        """
        live_ids = {session.id for session in self.store.sessions()}
        return [
            sid
            for sid, state in self._run_states.items()
            if sid in live_ids and not state.is_send_allowed
        ]

    def in_flight_run_count(self) -> int:
        """Count of LIVE sessions whose recorded run currently disallows a new send.

        Excludes orphaned entries for sessions the store no longer has (see
        ``_live_busy_session_ids``) -- consumers (cap math, fleet UX) must
        never see a closed session's stale run inflate this count.

        Returns:
            The number of live sessions whose recorded run currently
            disallows a new send.
        """
        return len(self._live_busy_session_ids())

    def add_pending_round(self, session_id: str, round_id: str) -> None:
        """Register ``round_id`` as an outstanding approval-like round for ``session_id``.

        TASK-1050 (Defect A): the fleet-visible pending-approval badge used
        to be a single boolean per session (``_pending_approvals`` as a
        plain ``set[str]``, flipped by the now-deprecated ``set_run_
        pending_approval``) shared by THREE independent bridges -- MCP tool
        approvals, skill-install confirms, and skill-script confirms. Any
        one bridge's teardown cleared the badge for its own session_id
        regardless of whether a SIBLING round (same bridge or a different
        one) was still outstanding for that same session, so the badge
        could go dark while a live confirm was still waiting on the user.

        ``_pending_approvals`` is now keyed by session id to the SET of
        round ids currently outstanding for it -- a session reads as
        "pending" (``run_marker_for``/``fleet_summary_counts``) iff that
        set is non-empty. Idempotent: adding an already-registered
        ``round_id`` again is a no-op (set semantics), so a caller never
        needs to check first.

        Every genuine bridge round already mints a fresh ``uuid4()`` round/
        request id before arming (``request_mcp_approvals``'s ``round_id``,
        ``request_skill_install_confirm``'s/``request_skill_script_
        confirm``'s ``request_id``) -- this is the id each bridge now
        passes here instead of the old boolean.

        Args:
            session_id: The session the round belongs to.
            round_id: The round's own unique id (a real bridge round id, or
                the reserved ``_LEGACY_PENDING_APPROVAL_ROUND_ID`` sentinel
                -- see ``set_run_pending_approval``).
        """
        # F2b fix (Qodo wave), preserved: reachable from a worker thread
        # while the UI thread concurrently iterates `_pending_approvals`
        # via `fleet_summary_counts` -- guard the mutation with the shared
        # lock so iteration never observes a torn add/discard.
        with self._approval_state_lock:
            self._pending_approvals.setdefault(session_id, set()).add(round_id)

    def discard_pending_round(self, session_id: str, round_id: str) -> None:
        """Clear ``round_id`` from ``session_id``'s outstanding approval-like rounds.

        TASK-1050 (Defect A) counterpart to ``add_pending_round``: discards
        only THIS round's id from the session's round-id set. The fleet
        badge (``run_marker_for``) clears only once that set is empty --
        i.e. once every bridge round for the session has resolved, not just
        this one. Idempotent: discarding an id that was never added (or was
        already discarded) is a safe no-op, and discarding the SAME id
        twice never double-decrements anything (set semantics -- there is
        nothing to corrupt).

        Args:
            session_id: The session the round belongs to.
            round_id: The round's own unique id, as passed to the matching
                ``add_pending_round`` call.
        """
        with self._approval_state_lock:
            rounds = self._pending_approvals.get(session_id)
            if rounds is None:
                return
            rounds.discard(round_id)
            if not rounds:
                self._pending_approvals.pop(session_id, None)

    def has_pending_approval_round(self, session_id: str) -> bool:
        """Return whether ``session_id`` currently has ANY outstanding approval-like round.

        TASK-1050: exposed so a caller that lacks a round id of its own
        (see ``set_run_pending_approval``'s docstring) can check whether a
        REAL round is already registered before redundantly stamping the
        deprecated boolean shim -- ``ChatScreen._park_console_approval`` is
        the one production caller that needs this (its owning bridge always
        registers the real round id via ``add_pending_round`` moments
        before invoking the park callback, so by the time this runs, the
        real round is normally already present).

        Args:
            session_id: The session to check.

        Returns:
            ``True`` iff at least one round id is currently registered for
            ``session_id``.
        """
        with self._approval_state_lock:
            return session_id in self._pending_approvals

    def set_run_pending_approval(self, session_id: str, pending: bool) -> None:
        """DEPRECATED boolean shim -- prefer ``add_pending_round``/``discard_pending_round``.

        Parallel-agents spec §6 (Task 7 stores/exposes the flag; Task 9
        wired the approval paths -- MCP batch approvals, skill-install/
        script confirms -- that originally called this). TASK-1050 (Defect
        A) migrated all three bridges to the round-keyed ``add_pending_
        round``/``discard_pending_round`` instead, since a plain boolean
        cannot represent "N independent rounds outstanding for one
        session" without one clobbering another's clear.

        This shim survives for the ONE remaining caller genuinely without a
        round id of its own: ``ChatScreen._park_console_approval`` (wired
        as ``park_pending_approval``), whose own public contract is a
        single-arg ``Callable[[str], None]`` with no room for a round id --
        changing that would ripple into every test that wires ``park_
        pending_approval = some_list.append`` -- and it is ALSO used
        directly, standalone, by tests exercising the marker/badge
        lifecycle without a live round (mirrors how those tests already
        drive other controller seams directly).

        Internally represented as the reserved
        ``_LEGACY_PENDING_APPROVAL_ROUND_ID`` sentinel round id, so it
        composes safely alongside real round ids in the same per-session
        set -- ``pending=True`` adds the sentinel, ``pending=False``
        discards ONLY the sentinel (a real round registered separately via
        ``add_pending_round`` is untouched either way). Because of this, a
        caller that calls this with ``pending=True`` while a REAL round is
        ALREADY registered for the session adds a harmless, redundant
        no-op-visible entry -- but that same caller must not rely on this
        call's own ``pending=False`` (or a real round's ``discard_pending_
        round``) to fully clear the badge on its own; whichever one runs
        last is the one that actually clears it. ``ChatScreen._park_
        console_approval`` avoids this ambiguity by checking ``has_
        pending_approval_round`` first and only falling back to this shim
        when no real round is registered yet.

        Args:
            session_id: The session whose pending-approval flag to update.
            pending: ``True`` to mark the session as awaiting a decision,
                ``False`` to clear it.
        """
        if pending:
            self.add_pending_round(session_id, _LEGACY_PENDING_APPROVAL_ROUND_ID)
        else:
            self.discard_pending_round(session_id, _LEGACY_PENDING_APPROVAL_ROUND_ID)

    def mark_session_visited(self, session_id: str) -> None:
        """Clear ``session_id``'s unvisited terminal outcome.

        Parallel-agents spec §6. Called from ``switch_session`` once the
        store has swapped to ``session_id`` -- visiting a session is what
        "sees" its terminal outcome, so that marker resets to steady state
        (``run_marker_for`` then falls through to ``ConsoleRunMarker.NONE``
        unless a fresh run starts).

        Task 9 correction: this used to ALSO discard ``session_id``'s
        pending-approval flag, which directly contradicted the parked-
        approval design once background sessions could carry a live
        approval round -- a plain visit (e.g. just checking on a background
        session, or the auto-mount ``switch_session`` now performs to show
        its parked card) would silently deny-in-spirit the outstanding
        round's badge before the human ever made a decision. The flag now
        clears ONLY on the round's own resolution (``request_mcp_
        approvals``' ``finally``) or a terminal run-state transition
        (``_set_run_state``) -- never merely from being looked at. See
        ``switch_session`` for the (separate) mount-the-parked-card step
        that visiting now performs instead.

        Args:
            session_id: The session just switched to (or otherwise
                visited).
        """
        self._unvisited_outcomes.pop(session_id, None)

    def run_marker_for(self, session_id: str) -> ConsoleRunMarker:
        """Fleet-visible marker for ``session_id`` (parallel-agents spec §6).

        Precedence, checked in order:

        1. ``NEEDS_APPROVAL`` -- outranks ``RUNNING`` even though a parked
           run is technically still in-flight: the marker must announce
           the thing that needs a human, not just "something is
           happening".
        2. ``RUNNING`` -- derived from the same live/busy definition as
           ``in_flight_run_count`` (``_live_busy_session_ids``), so this
           never invents a second notion of "in-flight".
        3. ``FINISHED_OK``/``FINISHED_FAILED`` -- from
           ``_unvisited_outcomes``, stamped only for non-active sessions
           by ``_set_run_state``'s terminal transitions and cleared by
           ``mark_session_visited``.
        4. ``NONE`` otherwise.

        Args:
            session_id: The session to compute the marker for.

        Returns:
            The single ``ConsoleRunMarker`` that best describes
            ``session_id``'s current fleet-visible state, per the
            precedence above.
        """
        # TASK-1050: `_pending_approvals` is keyed by session id to a SET of
        # outstanding round ids (see `add_pending_round`) -- a plain `in`
        # dict-key check is exactly "does this session have ANY pending
        # round", which is what NEEDS_APPROVAL means; `add_pending_round`/
        # `discard_pending_round` guarantee an emptied round set is popped
        # rather than left behind as a stale `{}`, so this can never read
        # "pending" for a session with zero live rounds.
        if session_id in self._pending_approvals:
            return ConsoleRunMarker.NEEDS_APPROVAL
        if session_id in self._live_busy_session_ids():
            return ConsoleRunMarker.RUNNING
        return self._unvisited_outcomes.get(session_id, ConsoleRunMarker.NONE)

    def fleet_summary_counts(self) -> tuple[int, int]:
        """Counts of OTHER live sessions running / needing approval.

        Parallel-agents spec §6. Returns ``(other running, other pending-
        approval)`` relative to the active (viewed) session -- its own
        status is visible directly in the transcript, not through the
        fleet summary, so it is excluded from both counts. Sessions the
        store no longer has (orphaned ``_pending_approvals``/`
        `_run_states`` entries) are excluded via the same live-session
        filter ``_live_busy_session_ids`` applies. A session that is both
        busy and pending-approval is counted only as pending, mirroring
        ``run_marker_for``'s NEEDS_APPROVAL-outranks-RUNNING precedence --
        neither count double-books it.

        Returns:
            A ``(other_running, other_pending_approval)`` tuple of counts,
            both excluding the active (viewed) session.
        """
        active = self.store.active_session_id or ""
        live_ids = {session.id for session in self.store.sessions()}
        # F2b fix (Qodo wave): snapshot under the lock rather than
        # iterating `_pending_approvals` live -- this runs on the UI
        # thread's ~0.2s sync tick while a worker thread can concurrently
        # add/discard entries (`request_mcp_approvals`'s own body/
        # `finally`), so an unguarded comprehension here risked
        # `RuntimeError: Set changed size during iteration`. The
        # comprehension itself runs OUTSIDE the lock, over the snapshot.
        with self._approval_state_lock:
            pending_snapshot = set(self._pending_approvals)
        other_pending = {
            sid for sid in pending_snapshot if sid in live_ids and sid != active
        }
        other_busy = {sid for sid in self._live_busy_session_ids() if sid != active}
        return len(other_busy - other_pending), len(other_pending)

    def busy_fleet_session_count(self) -> int:
        """Count of LIVE sessions ``shutdown()`` would tear down right now.

        TASK-1143 (F5): union of ``_live_busy_session_ids()`` (a session
        with an active stream/citation-repair task -- the same set
        ``in_flight_run_count`` reports) and every LIVE session with at
        least one outstanding approval-like round, mounted or parked
        (``_pending_approvals``, the same registry ``run_marker_for``'s
        NEEDS_APPROVAL branch and ``has_pending_approval_round`` read --
        MCP tool approvals, skill-install, and skill-script confirms all
        register through the same ``add_pending_round``). A session that
        is both busy and mid-approval is counted once: this answers "how
        many agent runs" for fleet-teardown UX (the Console
        confirm-on-navigate guard and its post-navigate record), not "how
        many independent events" -- no new definition of "busy" beyond
        the union of the two predicates those existing callers already
        use.

        Returns:
            The number of live sessions with in-flight work and/or an
            outstanding approval-like round -- 0 when the fleet is idle.
        """
        live_ids = {session.id for session in self.store.sessions()}
        with self._approval_state_lock:
            pending_ids = set(self._pending_approvals)
        busy_ids = set(self._live_busy_session_ids())
        return len(busy_ids | (pending_ids & live_ids))

    @property
    def max_parallel_runs(self) -> int:
        """User-adjustable global cap on simultaneous runs (parallel-agents spec §4).

        Reads ``[console] max_parallel_runs`` through the same
        ``get_cli_setting`` seam used elsewhere in this module (see
        ``_resolve_mcp_approval_timeout_seconds``). Floored at 1 and
        defaulted to ``CONSOLE_DEFAULT_MAX_PARALLEL_RUNS`` so a bad/blank
        config value can never lock every session out of sending.

        Returns:
            The configured cap on simultaneous runs, floored at 1.
        """
        raw = get_cli_setting(
            "console", "max_parallel_runs", CONSOLE_DEFAULT_MAX_PARALLEL_RUNS
        )
        if raw is None:
            return CONSOLE_DEFAULT_MAX_PARALLEL_RUNS
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = CONSOLE_DEFAULT_MAX_PARALLEL_RUNS
        return max(1, value)

    def send_refusal_copy(self, session_id: str) -> str | None:
        """Why a send to ``session_id`` must be refused right now, or ``None``.

        Parallel-agents spec §4. Two gates, checked in order:

        1. Per-session -- ``session_id``'s own run is still in flight.
        2. Global cap -- ``max_parallel_runs`` busy sessions already exist,
           so a NEW send (from any session, including an idle one) must
           wait.

        The cap's busy list comes from ``_live_busy_session_ids`` (shared
        with ``in_flight_run_count``): a session closed mid-VALIDATING
        leaves its entry in ``self._run_states`` behind
        (``ConsoleChatStore.close_session`` never touches the controller's
        map -- Task 1 review finding), and a session that no longer exists
        must not consume a cap slot or be named in the refusal copy.

        Args:
            session_id: The session id attempting to send.

        Returns:
            A human-readable refusal message if the send must be blocked
            right now, otherwise ``None`` when the send is allowed.
        """
        if not self.run_state_for(session_id).is_send_allowed:
            return "A run is already running in this tab."
        busy_ids = self._live_busy_session_ids()
        if len(busy_ids) < self.max_parallel_runs:
            return None
        live_sessions = {session.id: session for session in self.store.sessions()}
        limit = CONSOLE_CAP_REFUSAL_TITLE_LIMIT
        titles = [live_sessions[sid].title for sid in busy_ids[:limit]]
        suffix = f" and {len(busy_ids) - limit} more" if len(busy_ids) > limit else ""
        busy_count = len(busy_ids)
        # Fleet-UX expert review F7 (task-1234): number agreement -- "1
        # agents already running" read as a grammar bug on the very first
        # cap refusal a solo user could ever see (max_parallel_runs=1).
        agent_noun = "agent" if busy_count == 1 else "agents"
        return (
            f"{busy_count} {agent_noun} already running "
            f"({', '.join(titles)}{suffix}). "
            "Wait for one to finish or interrupt it."
        )

    @property
    def run_state_history(self) -> list[ConsoleRunStatus]:
        """The ACTIVE session's run-status history (read-only facade, mirrors ``run_state``).

        Returns:
            The active session's list of recorded ``ConsoleRunStatus`` values.
        """
        return self.run_state_history_for(self.store.active_session_id or "")

    def run_state_history_for(self, session_id: str) -> list[ConsoleRunStatus]:
        """Return (creating if absent) ``session_id``'s run-status history.

        Args:
            session_id: The session id to look up.

        Returns:
            The session's list of recorded ``ConsoleRunStatus`` values,
            initialized to ``[ConsoleRunStatus.IDLE]`` when absent.
        """
        return self._run_state_histories.setdefault(session_id, [ConsoleRunStatus.IDLE])

    async def submit_draft(
        self, draft: str, *, session_id: str | None = None
    ) -> ConsoleSubmitResult:
        """Submit a composer draft through native Console validation and provider resolution.

        F4 fix (Qodo wave, parallel-agents spec §2): sends are dispatched
        per-session -- ``chat_screen._dispatch_console_draft_send`` captures
        the target session at DISPATCH time and threads it through
        ``run_worker``'s coroutine args (see ``_submit_console_native_
        draft``). Before this fix, this method always re-resolved "the
        session to submit into" via ``store.ensure_session()``/
        ``store.active_session_id`` at EXECUTION time instead -- a session
        switch during the scheduling gap between ``run_worker(...)`` and
        this coroutine's body actually running could silently submit the
        draft into whichever session the user switched TO, not the one
        that was showing when Send was pressed.

        Args:
            draft: The raw composer text to submit.
            session_id: The session this draft was dispatched for, captured
                by the caller at dispatch time. ``None`` (the default)
                preserves the pre-fix behavior -- resolve/create the
                CURRENTLY active session -- for direct-call test idioms and
                other callers that have no per-session dispatch to capture.
                An empty string is treated the same as ``None`` (the
                dispatch-time sentinel for "no session existed yet").

        Returns:
            The submission outcome: ``accepted`` False (with an explanatory
            ``visible_copy``) when blocked before any provider call, or
            when ``session_id`` names a session that no longer exists by
            the time this runs (see ``_session_closed_result``); ``True``
            once the turn actually proceeds.
        """
        active_rejection = self._active_run_rejection(session_id=session_id)
        if active_rejection is not None:
            return active_rejection

        if session_id:
            session = next(
                (s for s in self.store.sessions() if s.id == session_id), None
            )
            if session is None:
                # The dispatching session was closed during the gap between
                # dispatch and this coroutine actually running -- there is
                # nothing left to submit into. Stamp the (now-orphaned)
                # session id, never whatever is active now (see
                # `_session_closed_result`'s own docstring).
                return self._session_closed_result(session_id=session_id)
        else:
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

        # TASK-457(a): echo the USER message BEFORE resolving the provider, so a
        # slow/cold readiness probe no longer leaves the transcript blank while
        # the composer clears — the message reads as "sent", not lost. On a
        # not-ready provider the row persists next to the honest block-row below
        # (the message is no longer silently dropped) and the draft is kept (the
        # composer clears only on the accepted path via
        # `_notify_submission_accepted`), so the user can re-attempt. Staged
        # attachments are embedded on the row here but only CLEARED on the
        # success path below, so a blocked attempt leaves them staged for retry.
        #
        # Auto-title BEFORE the append: a persisting append creates the durable
        # conversation from `session.title` (persist_session_if_needed) and sets
        # `persisted_conversation_id`, after which `_maybe_auto_title_session`
        # early-returns. Titling first means the conversation is created as the
        # derived title (e.g. "hello") instead of the default "Chat 1", so the
        # workspace rail shows it immediately after persistence.
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
        # TASK-485: the optimistic echo is appended WITHOUT persistence. A send
        # that is blocked/fails before it reaches the provider must leave no
        # durable record — otherwise the resume path (which reconstructs every
        # row as "complete") would silently drop the row's failed state and let a
        # never-sent message re-enter the next send's context, and the orphan
        # would render as a lonely user prompt. The row is flushed to storage
        # only once the turn is confirmed to proceed (below).
        echoed_user = self.store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content=clean_draft,
            attachments=staged_attachments,
            persist=False,
        )

        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.VALIDATING, "Validating provider."),
            session_id=session.id,
        )
        try:
            resolution = await self.provider_gateway.resolve_for_send(
                self._provider_selection()
            )
        except BaseException:
            # A readiness probe that raises or is cancelled AFTER the optimistic
            # USER echo must still fail that row — otherwise a never-sent USER
            # message leaks into the NEXT send's provider context (`skip_failed`
            # only drops "failed" rows). Fail it, then re-raise so the caller
            # still sees the probe failure.
            self.store.mark_message_send_blocked(echoed_user.id)
            raise
        if not getattr(resolution, "ready", False):
            visible_copy = self._blocked_visible_copy(
                getattr(resolution, "visible_copy", "")
            )
            # The echoed row stays visible but never reached a provider — fail it
            # so it is excluded from the NEXT send's provider context
            # (`skip_failed`) and reads honestly as unsent rather than polluting
            # the history.
            self.store.mark_message_send_blocked(echoed_user.id)
            return self._block(session.id, visible_copy)

        if pendings:
            self.store.clear_pending_attachments(session.id)
        citation_context: str | None = None
        citation_trace_builder: CitationTraceBuilder | None = None
        prompt_evidence_set_id: str | None = None
        citation_repair_contract: CitationRepairContract | None = None
        terminal_citation_finalizer: TerminalCitationFinalizer | None = None
        try:
            provider_messages = self._provider_messages_for_session(
                session.id, annotate_ids=True
            )
            (
                provider_messages,
                refuse,
                skill_notes,
                skill_bindings,
                skill_bundle_block,
            ) = await self._apply_skill_substitution(provider_messages)
            if refuse is not None:
                # A substitution refusal is a block outcome like any other
                # (provider not ready, probe raise): fail the echoed row so the
                # refused command never enters the next send's provider context.
                self.store.mark_message_send_blocked(echoed_user.id)
                return self._block(session.id, refuse)
            for note in skill_notes:
                # An embedded skipped-skill note is never an abort: append the
                # same system-row copy `_block` would, then let the turn proceed.
                self.store.append_message(
                    session.id, role=ConsoleMessageRole.SYSTEM, content=note
                )
            (
                citation_context,
                citation_trace_builder,
                prompt_evidence_set_id,
                citation_repair_contract,
            ) = await self._capture_rag_context(clean_draft)
            has_exact_citation_context = (
                citation_trace_builder is not None
                or citation_repair_contract is not None
            )
            if citation_context and not has_exact_citation_context:
                provider_messages = self._prepend_evidence_context(
                    provider_messages,
                    citation_context,
                )
            provider_messages = await self._apply_chat_dictionaries(
                provider_messages, session.id
            )
            provider_messages = await self._apply_world_info(
                provider_messages, session.id
            )
            if citation_context and has_exact_citation_context:
                provider_messages = self._prepend_evidence_context(
                    provider_messages,
                    citation_context,
                )
            prefill, prefill_from_one_shot = self._resolve_submit_prefill(session.id)
            terminal_citation_finalizer = self._build_terminal_citation_finalizer(
                context=citation_context,
                builder=citation_trace_builder,
                prompt_evidence_set_id=prompt_evidence_set_id,
            )
        except BaseException:
            # Any failure between the optimistic echo and the confirmed turn
            # (dictionary/world-info application, prefill resolution) must also
            # fail the echoed row, or a never-sent message leaks into the next
            # send's provider context (`skip_failed` only drops "failed" rows).
            self.store.mark_message_send_blocked(echoed_user.id)
            raise
        # The accepted-hook fires only once the turn is confirmed to
        # actually proceed (Qodo finding 3, PR #636 bot review): it used to
        # fire right after the USER row was appended, BEFORE this skill
        # substitution/trust check ran. In the real ChatScreen this hook
        # clears the composer, so firing it before a substitution refusal
        # ate the refused draft the user needs to correct. A substitution
        # refusal is a `_block()` outcome exactly like any other (provider
        # not ready, policy block, validation failure) and those already
        # never reach this hook -- this ordering just extends that same
        # rule to cover it too.
        self._notify_submission_accepted()
        # TASK-485: the turn is confirmed to proceed — flush the deferred USER
        # echo to durable storage now (creating the conversation), BEFORE the
        # assistant row, so a reload shows the user's prompt ahead of its reply.
        self.store.persist_message_if_needed(echoed_user.id)
        assistant: ConsoleChatMessage | None = None
        citation_repair_session = (
            ConsoleCitationRepairSession(
                contract=citation_repair_contract,
                resolution=resolution,
            )
            if citation_repair_contract is not None
            else None
        )
        try:
            assistant = self.store.append_message(
                session.id,
                role=ConsoleMessageRole.ASSISTANT,
                content="",
                persist=self.store.persistence is not None,
                terminal_citation_finalizer=terminal_citation_finalizer,
                defer_terminal_persistence=citation_repair_session is not None,
            )
            return await self._stream_assistant_response(
                resolution=resolution,
                provider_messages=provider_messages,
                assistant_message_id=assistant.id,
                prefill=prefill,
                prefill_from_one_shot=prefill_from_one_shot,
                skill_bindings=skill_bindings,
                skill_bundle_block=skill_bundle_block,
                citation_repair_session=citation_repair_session,
            )
        finally:
            if assistant is not None:
                self.store.clear_terminal_citation_state(assistant.id)
            del terminal_citation_finalizer
            del citation_trace_builder

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
        # `create_session` above already activated the new session, so the
        # default (no explicit session_id -> active session) targets the
        # session JUST created here -- which is fresh/never-recorded and
        # therefore already idle, making this call a no-op today. Left
        # unchanged (rather than reaching for the session being replaced)
        # for the same reason `switch_session` below is: per-session run
        # state is meant to persist on the session you're leaving, not be
        # wiped just because a sibling session appeared.
        self._clear_terminal_run_state()
        # Fix wave (IMPORTANT 2, final review): re-derive the mounted
        # approval card for the brand-new (now active) session, exactly
        # like `switch_session`/`close_session`'s neighbor-activation
        # branch already do -- without this, a round mounted on the
        # session being left behind stayed rendered over the new tab
        # (`create_session` above activates `session`, but nothing else
        # ever told the card to re-derive for it). A fresh session can
        # never itself have a parked payload, so this always resolves to
        # `None` here -- i.e. it always clears -- but going through the
        # same `_parked_approval_payloads` lookup (rather than a bespoke
        # unconditional clear) keeps this call site honest with the same
        # "card state derives from the run's pending review state" rule
        # every other activation path follows.
        if self.set_pending_approval is not None:
            # F2b fix (Qodo wave): guard the read for consistency with
            # every other `_parked_approval_payloads` access, even though
            # a single `.get()` is not itself an iteration hazard.
            with self._approval_state_lock:
                parked_payload = self._parked_approval_payloads.get(session.id)
            self.set_pending_approval(parked_payload)
        # TASK-910: same re-derive for the skill-install/script cards -- a
        # brand-new session can never itself have a parked confirm, so this
        # always resolves to clearing whatever the session being left behind
        # had shown (mirrors the approval re-derive immediately above).
        self._remount_parked_skill_install(session.id)
        self._remount_parked_skill_script(session.id)
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
            self.store.rename_session(
                session.id, derived
            )  # (session, persisted) — auto-title best-effort

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
            # No session in scope here -- this is a global provider/model
            # settings change, not tied to any particular session's run.
            # Active-session UI path: clears whatever the user is currently
            # looking at (parallel-agents spec §2).
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
        # Resolve the OUTGOING session BEFORE `store.switch_session` below
        # moves `active_session_id` -- the no-arg default on
        # `_clear_terminal_run_state` would otherwise target the session
        # being ARRIVED AT (active_session_id already points there by the
        # time it runs). Per the spec's "clear the session you are leaving
        # if terminal" semantic, every session-scoped write in this refactor
        # is explicit, so this one is too: pass the outgoing session's id
        # directly. A session you're ARRIVING AT keeps whatever terminal/
        # in-flight state it already had (parallel-agents spec §2).
        previous_session_id = self.store.active_session_id
        session = self.store.switch_session(session_id)
        # Parallel-agents spec §6: visiting the session you just switched TO
        # clears its unvisited outcome marker -- must run AFTER the store
        # swap above so `session_id` really is the new active session by
        # the time downstream reads (e.g. `run_marker_for`) observe it.
        self.mark_session_visited(session_id)
        if previous_session_id is not None:
            self._clear_terminal_run_state(session_id=previous_session_id)
        # Task 9 (parked background approvals): mount `session_id`'s
        # parked round, if any, through the SAME UI bridge
        # `request_mcp_approvals` uses for an active session's round --
        # `self.set_pending_approval` is always safe to call with `None`
        # too (clears whatever the session being LEFT had shown), so this
        # single call both mounts a newly-visited parked card AND hides a
        # departing session's card in one step. No `call_from_thread`
        # marshal needed: `switch_session` always runs on the UI/main
        # thread already (same convention as `mark_session_visited`/
        # `_clear_terminal_run_state` above). Card state is entirely
        # derived from `_parked_approval_payloads` (the round's own
        # retained pending-review payload) every time this runs, never
        # from whatever the card happened to be showing before -- so
        # switching away and back re-mounts it unchanged (spec).
        #
        # Supersedes the pre-Task-9 `_deny_pending_approval_on_context_
        # change()` call that used to run here: that assumed only one
        # approval round could ever be in flight controller-wide (true
        # before Task 3's concurrent runs), so ANY switch force-denied it.
        # Once a background session can carry its own live round, denying
        # it just for being switched away from directly contradicts
        # parking -- the round now stays alive until its own resolution
        # (decision, cancel, or timeout).
        if self.set_pending_approval is not None:
            # F2b fix (Qodo wave): guard the read for consistency with
            # every other `_parked_approval_payloads` access.
            with self._approval_state_lock:
                parked_payload = self._parked_approval_payloads.get(session_id)
            self.set_pending_approval(parked_payload)
        # TASK-910: skill-install/script confirms now get the SAME park/
        # re-derive treatment as MCP batch approvals above -- a context
        # change (switch away) no longer force-denies either bridge's
        # pending confirm; the round stays alive (parked, badge + one
        # toast via `park_pending_approval`) until its own resolution,
        # cancellation, or shutdown. Superseded the pre-TASK-910
        # `_deny_pending_skill_install_on_context_change()`/`_deny_pending_
        # skill_script_on_context_change()` calls that used to run here
        # unconditionally on every switch.
        self._remount_parked_skill_install(session_id)
        self._remount_parked_skill_script(session_id)
        return session

    def close_session(self, session_id: str) -> ConsoleChatSession | None:
        """Close an existing native Console session.

        Args:
            session_id: Native Console session ID to close.

        Returns:
            The session activated after closing, or ``None`` when no sessions remain.
        """
        repair_session = self._active_citation_repair_sessions.get(session_id)
        self.clear_original_attempts_for_session(session_id)
        owns_active_stream = self._active_stream_belongs_to_session(session_id)
        if repair_session is not None and owns_active_stream:
            repair_session.cancel_reason = "session_close"
        if owns_active_stream:
            self._signal_stop(session_id=session_id)
            task = self._active_stream_tasks.get(session_id)
            if task is not None and task is not asyncio.current_task():
                task.cancel()
            self._set_run_state(
                ConsoleRunState(ConsoleRunStatus.STOPPED, "Session closed."),
                # `session_id` here is the session being CLOSED, which the
                # `_active_stream_belongs_to_session` guard above confirms
                # owns the active stream -- not necessarily the currently
                # ACTIVE session (you can close a background tab while
                # viewing another one), so this must be explicit rather
                # than falling back to the active-session default.
                session_id=session_id,
            )
        previous_active_id = self.store.active_session_id
        closed = self.store.close_session(session_id)
        new_active_id = self.store.active_session_id
        if (
            owns_active_stream
            and repair_session is not None
            and self._active_citation_repair_sessions.get(session_id) is repair_session
        ):
            self._active_citation_repair_sessions.pop(session_id, None)
        # Parallel-agents spec §6: closing the ACTIVE session auto-activates
        # a neighbor (`ConsoleChatStore.close_session`, console_chat_store.py
        # ~594-604) -- that neighbor is now the VIEWED session exactly as if
        # `switch_session` had navigated to it, so its unvisited outcome
        # must clear the same way, AND (Task 9) its parked approval card
        # (if any) must mount the same way too -- closing a background tab
        # must never leave the newly-viewed session's own pending approval
        # invisible just because it arrived here via auto-activation rather
        # than an explicit switch. Closing a BACKGROUND (non-active) session
        # leaves `active_session_id` unchanged, so this is a no-op in that
        # case.
        if new_active_id is not None and new_active_id != previous_active_id:
            self.mark_session_visited(new_active_id)
            if self.set_pending_approval is not None:
                # F2b fix (Qodo wave): guard the read for consistency with
                # every other `_parked_approval_payloads` access.
                with self._approval_state_lock:
                    parked_payload = self._parked_approval_payloads.get(new_active_id)
                self.set_pending_approval(parked_payload)
            # TASK-910: same re-derive for the skill-install/script cards --
            # closing the ACTIVE session auto-activates a neighbor, which is
            # now the VIEWED session exactly as if `switch_session` had
            # navigated to it.
            self._remount_parked_skill_install(new_active_id)
            self._remount_parked_skill_script(new_active_id)
        return closed

    def original_attempt_for_message(self, message_id: str) -> str | None:
        """Return and refresh one current-session original attempt."""
        body = self._original_attempts.get(message_id)
        if body is None:
            return None
        try:
            message = self.store.get_message(message_id)
        except KeyError:
            self._original_attempts.pop(message_id, None)
            return None
        presentation = message.citation_presentation
        if presentation is None or not presentation.original_attempt_available:
            self._original_attempts.pop(message_id, None)
            return None
        self._original_attempts.move_to_end(message_id)
        return body

    def clear_original_attempt(self, message_id: str) -> None:
        """Forget one preview and clear its content-free availability flag."""
        self._original_attempts.pop(message_id, None)
        self._set_original_attempt_availability(message_id, False)

    def clear_original_attempts_for_session(self, session_id: str) -> None:
        """Forget every original attempt owned by one Console session."""
        for message_id in tuple(self._original_attempts):
            try:
                owner_session_id = self.store.session_id_for_message(message_id)
            except KeyError:
                self._original_attempts.pop(message_id, None)
                continue
            if owner_session_id == session_id:
                self.clear_original_attempt(message_id)

    def _remember_original_attempt(
        self,
        message_id: str,
        body: str,
        *,
        update_presentation: bool = True,
    ) -> None:
        """Insert one successful repair preview into the eight-entry LRU."""
        self._original_attempts.pop(message_id, None)
        self._original_attempts[message_id] = body
        if update_presentation:
            self._set_original_attempt_availability(message_id, True)
        while len(self._original_attempts) > 8:
            evicted_id, _evicted_body = self._original_attempts.popitem(last=False)
            self._set_original_attempt_availability(evicted_id, False)

    def _set_original_attempt_availability(
        self,
        message_id: str,
        available: bool,
    ) -> None:
        """Update only the bounded presentation flag for a live message."""
        try:
            message = self.store.get_message(message_id)
        except KeyError:
            return
        presentation = message.citation_presentation
        if presentation is None:
            return
        self.store.set_citation_presentation(
            message_id,
            ConsoleCitationPresentation(
                phase=presentation.phase,
                notice_code=presentation.notice_code,
                original_attempt_available=available,
            ),
        )

    def _signal_stop(self, *, session_id: str) -> None:
        """Set the shared UI-facing stop flag AND ``session_id``'s own
        permanent per-run cancel signal.

        ``_stop_requested`` stays a single shared flag, but as of Fix
        round 1 (Critical 1) NO run's own cancellation-check loop reads it
        any more -- ``should_cancel`` (``_run_agent_reply``) and the
        direct/legacy stream path's own checks (``_stream_assistant_
        response``) read ONLY their run's own ``_active_cancel_events[
        owner_id]``, captured by closure. Reading the shared flag from
        inside a specific run's loop let ANY session's Stop/Close silently
        truncate an unrelated, untouched session's still-streaming reply
        (Fix round 1 finding).

        F5 fix (Qodo wave): the three worker-thread approval/confirm
        bridges no longer read ``_stop_requested`` either (see
        ``_is_active_session_cancelled``) -- a single session's Stop/Close
        must not deny an unrelated session's in-flight approval round any
        more than it may truncate an unrelated session's stream. Real
        process teardown (``shutdown()``) is the one case where denying
        every session's round at once is correct; that now goes through
        the dedicated, never-reset ``_shutdown_requested`` instead.
        ``_stop_requested`` itself is left set here for any other/legacy
        reader (kept for back-compat; this method's own contract has
        always been "set it," not "this is its only reader").

        ``_active_cancel_events[session_id]``, once set here, is never
        reset for that run, so a still-running bridge thread always
        observes the Stop correctly regardless of what the coroutine side
        has already reset (task-227). Every caller (``close_session``,
        ``stop_active_run``, ``shutdown``) already knows the exact session
        it is signalling -- there is deliberately no active-session
        fallback here, unlike ``_set_run_state``.
        """
        self._stop_requested = True
        cancel_event = self._active_cancel_events.get(session_id)
        if cancel_event is not None:
            cancel_event.set()

    def _is_active_session_cancelled(self) -> bool:
        """Best-effort cancel-signal check that falls back to the VIEWED
        session -- the pre-Task-9 behavior of the three worker-thread
        approval/confirm bridges below (``request_mcp_approvals``,
        ``request_skill_install_confirm``, ``request_skill_script_
        confirm``), preserved here as the fallback ``_is_session_
        cancelled`` uses when a caller has no ``session_id`` of its own to
        pass (e.g. a legacy direct call in an older test). See
        ``_is_session_cancelled``'s own docstring for the Task 9 fix this
        was carved out of, and for the F5 fix (Qodo wave) that replaced
        the shared, lifecycle-reset ``_stop_requested`` flag with the
        never-reset ``_shutdown_requested`` in that same fallback branch.
        """
        cancel_event = self._active_cancel_events.get(
            self.store.active_session_id or ""
        )
        return cancel_event is not None and cancel_event.is_set()

    def _is_session_cancelled(self, session_id: str | None) -> bool:
        """Cancellation check for the three worker-thread approval/confirm
        bridges below, scoped to ``session_id``'s OWN cancel event when
        known (PA-T9 finding #1 fix).

        Pre-Task-9, all three bridges checked ``self._stop_requested or
        self._is_active_session_cancelled()`` -- the shared global flag
        OR'd with the VIEWED session's cancel event, regardless of which
        session's round was actually waiting. Once background sessions can
        each carry their own in-flight approval round (parked or not),
        that was a real cross-session bug: Session A's Stop
        (``stop_active_run``/``close_session``, via ``_signal_stop``)
        always sets the shared ``_stop_requested`` flag alongside A's own
        cancel event, so the OR-check let A's Stop spuriously deny B's
        completely unrelated, still-waiting approval batch.

        Fix: when ``session_id`` is known, check ONLY that session's own
        ``_active_cancel_events`` entry -- never the shared flag. This
        still correctly resolves every INTENTIONAL global-reach case:
        ``shutdown()`` (the one caller that must stop every session at
        once) signals every live session's cancel event individually
        (``_signal_stop(session_id=...)`` in its own per-session loop), so
        a round scoped to ANY session still observes shutdown via its own
        event -- ``_stop_requested`` was never the mechanism that made
        shutdown reach a specific round, just a side effect of
        ``_signal_stop`` also setting it.

        ``session_id=None`` (a caller with no session context of its own --
        e.g. an existing test calling ``request_mcp_approvals`` directly
        with no ``session_id=`` kwarg) falls back to the exact pre-Task-9
        behavior via ``_is_active_session_cancelled``, so those callers'
        existing global-flag expectations are unchanged.

        F5 fix (Qodo wave, folded in during the PR2 restack): that
        ``session_id=None`` fallback used to OR in the shared
        ``_stop_requested`` flag, which (a) any session's Stop set
        regardless of which round was waiting, and (b) various run
        lifecycles reset to ``False`` mid-flight, making whether a
        still-polling bridge observed an earlier Stop a race. It now ORs
        in ``_shutdown_requested`` instead -- set exactly once, only by
        ``shutdown()``, and never reset -- so a legacy no-session caller's
        "global stop denies" expectation is preserved for the one case
        this controller INSTANCE is ever torn down for (see ``shutdown()``
        's own docstring for exactly what that covers -- NOT only real
        process exit) where that is actually correct, without
        reintroducing cross-session poisoning for everyday per-session
        Stop/Close.

        TASK-1052 fix: the ``session_id is not None`` branch now ALSO ORs
        in ``_shutdown_requested``. Previously it checked ONLY that
        session's own ``_active_cancel_events`` entry, relying entirely on
        ``shutdown()``'s per-session ``_signal_stop`` fanout (see the
        docstring paragraph above) to ever reach a real-session round --
        but that fanout walks a SNAPSHOT of ``_active_stream_tasks`` taken
        when ``shutdown()``/``close_session`` runs. A round armed for a
        session BEFORE that session is registered there is invisible to
        the snapshot and was previously left to fail closed only via its
        own (up to ~120s) confirm/approval timeout -- promptness, not
        correctness, but still a real gap for a signal this controller
        instance's teardown is supposed to reach every live round with,
        unconditionally.

        Correction (review, TASK-1052): an earlier revision of this
        docstring justified ORing in ``_shutdown_requested`` here by
        calling it "real process teardown" and treating that as
        inherently global/safe. That premise was FALSE: ``shutdown()`` is
        also called from ordinary Console-screen unmount
        (``ChatScreen.on_unmount``), which fires on every navigation AWAY
        from the Console tab, not only on app exit -- so
        ``_shutdown_requested`` can be set on a controller instance the
        user is still actively using the app around. The actual safety
        argument does not rest on "global by definition"; it rests on
        this controller's OWN lifecycle: ``ChatScreen`` only ever
        constructs a fresh ``ConsoleChatController`` lazily
        (``_ensure_console_chat_controller``) after ``on_unmount`` has
        both run this instance's ``shutdown()`` and dropped the screen's
        reference to it, so a torn-down instance -- flag permanently set
        or not -- is never reused for a later Console visit, and no round
        still parked on it could ever be resolved through a UI that no
        longer exists anyway. ``_shutdown_requested`` is set exactly once,
        only by ``shutdown()``, and never reset for THIS instance's
        lifetime (see ``shutdown()``'s own docstring and its ``self.
        _shutdown_requested.set()`` call), so ORing it in here for a real
        ``session_id`` can never wrongly deny a live round while this
        controller instance is still the one actually in use -- it can
        only ever fire once this instance itself is being (or has been)
        torn down. This does NOT widen scoping for everyday per-session
        Stop/Close: ``_signal_stop`` still only touches the ONE session's
        own cancel event; an unrelated session's Stop still leaves both
        this branch's checks unset.
        """
        if session_id is not None:
            if self._shutdown_requested.is_set():
                return True
            cancel_event = self._active_cancel_events.get(session_id)
            return cancel_event is not None and cancel_event.is_set()
        return self._shutdown_requested.is_set() or self._is_active_session_cancelled()

    # -- MCP batch-approval bridge (task-5) ----------------------------------

    def request_mcp_approvals(
        self, pending: list[MCPPendingCall], *, session_id: str | None = None
    ) -> dict[str, str]:
        """Bridge one batch of pending MCP tool calls to the Console UI and back.

        WORKER THREAD. Bound (via a ``functools.partial`` binding this
        run's ``session_id``, Task 9) as ``MCPToolProvider``'s
        ``approval_callback`` and ``build_tool_review_hook``'s
        ``request_approvals``, so this runs on the agent bridge's
        background OS thread (the ``asyncio.to_thread`` call inside
        ``_run_agent_reply``) -- it must never touch a widget directly,
        only through ``self.app.call_from_thread``.

        Builds a fresh ``threading.Event`` + shared decisions dict (stored
        under this round's own entry in ``_pending_approval_rounds``, keyed
        by a freshly minted ``round_id`` -- see that map's own docstring
        for why a single shared slot, or a slot keyed by session id alone,
        could not survive concurrent sessions or same-session round
        replacement). Either MOUNTS the card immediately (``session_id`` is
        the currently ACTIVE/viewed session, or unknown -- legacy
        no-session callers keep the pre-Task-9 always-mount behavior) or
        PARKS it (``session_id`` is a DIFFERENT, background session --
        Task 9: the retained ``payload`` goes into
        ``_parked_approval_payloads`` for ``switch_session`` to mount
        later, and ``park_pending_approval`` fires the fleet badge +
        one-shot toast instead of touching the mounted-card slot). Either
        way it then polls ``event.wait(1.0)`` re-checking this run's OWN
        cancel signal (``_is_session_cancelled``) and a deadline every
        second until one of three things happens: the user submits a
        decision (``resolve_pending_approval``, called from the UI thread
        once the card's own stamped ``round_id`` is delivered back, sets
        the Event -- Fix round 1: NOT "whichever round belongs to the
        active session", see ``resolve_pending_approval``'s own docstring
        for why that was a real cross-session misattribution hazard), the
        run is cancelled/torn down (``_is_session_cancelled`` -- F5 fix,
        Qodo wave: this round's OWN cancel event, or real process teardown
        via ``_shutdown_requested``, never any OTHER session's bare Stop --
        see that method's own docstring), or the configured approval
        timeout elapses. Whichever unique ``llm_name`` never received an
        explicit decision by then fails closed to ``"deny"``
        (cancellation) or ``"timeout"`` (deadline) -- see
        ``MCPToolProvider._apply_verdict`` for how each decision string is
        consumed. The mounted card (if any) is always cleared afterwards
        (``finally``), regardless of outcome -- but ONLY if this round's
        session is STILL the active one at that moment, so a background
        round resolving (timeout/cancel) while some OTHER session's card is
        showing never clobbers it.

        Args:
            pending: One turn's pending tool calls awaiting approval,
                possibly containing repeated ``llm_name``s (T3: calls
                sharing a name share one verdict).
            session_id: The run's OWNING session (Task 3 threads it through
                ``_run_agent_reply``). ``None`` preserves every pre-Task-9
                call site's behavior (always mounts against whatever
                session is active at ROUND-key time; no parking).

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
        # Fix round 1 (review CRITICAL finding): keyed by a freshly minted
        # ROUND id, not by session id (or the active session) -- a session-
        # keyed slot is ambiguous the moment either (a) the ACTIVE session
        # changes between this round starting and the user's decision
        # arriving (`ApprovalDecided` travels as an async Textual message,
        # so a `switch_session` can land in that gap), or (b) a second
        # round starts for the SAME session before a first round's stale
        # decision message is delivered -- either way a session-keyed slot
        # would let a stale/misdirected decision resolve the WRONG round.
        # `round_id` is stamped into `payload` below, round-trips through
        # `ChatApprovalCard.set_batch` -> `ApprovalDecided` ->
        # `resolve_pending_approval`, and is the ONLY thing that round is
        # ever resolved by -- mirrors `_pending_skill_script_rounds`'
        # identical `request_id`-keyed defense.
        round_id = str(uuid4())
        owning_session_id = (
            session_id
            if session_id is not None
            else (self.store.active_session_id or "")
        )
        # F2b fix (Qodo wave): guard the round registration -- the UI
        # thread's `resolve_pending_approval` (TASK-913: fails closed by
        # round_id now, no more active-session scan) and the
        # `fleet_summary_counts` sync tick can read/iterate this map
        # concurrently with this worker thread's own writes.
        with self._approval_state_lock:
            self._pending_approval_rounds[round_id] = {
                "event": event,
                "decisions": decisions,
                "session_id": owning_session_id,
            }

        timeout_seconds = self._resolve_mcp_approval_timeout_seconds()
        deadline = time.monotonic() + timeout_seconds
        payload = {
            "round_id": round_id,
            "session_id": owning_session_id,
            "calls": [
                {
                    "llm_name": call.llm_name,
                    "server_key": call.server_key,
                    "tool_name": call.tool_name,
                    "server_label": call.server_label,
                    "arguments": dict(call.arguments or {}),
                    "reason": call.reason,
                    "options": list(call.options),
                    "path_precheck_failed": call.path_precheck_failed,
                }
                for call in pending
            ],
            "timeout_seconds": timeout_seconds,
        }
        # Task 9: park rather than mount when this round's session is a
        # DIFFERENT, background session -- `session_id is None` (a legacy
        # caller with no session context) always mounts, matching every
        # pre-Task-9 call site.
        is_parked = session_id is not None and session_id != (
            self.store.active_session_id or ""
        )
        if session_id is not None:
            # Register THIS round's own id directly here (worker thread,
            # plain-dict/set mutation -- same no-marshal convention as
            # `_active_cancel_events` elsewhere in this class) so it is
            # authoritative regardless of whether a UI bridge happens to be
            # wired. TASK-1050 (Defect A): round-keyed, not a plain
            # boolean -- a sibling round from this bridge or either of the
            # other two (skill-install/skill-script confirms) for the SAME
            # session stays independently tracked, so THIS round's own
            # teardown can never clear a badge a sibling still needs.
            # `park_pending_approval`/`ChatScreen._park_console_approval`
            # only falls back to the deprecated boolean shim when NO round
            # is registered yet (`has_pending_approval_round`), so it never
            # double-counts against this call.
            self.add_pending_round(session_id, round_id)
            # Fix wave (CRITICAL 1, final review): retain THIS round's
            # payload for EVERY session-attributed round -- mounted or
            # parked -- not just a parked one. `switch_session` re-derives
            # the card EXCLUSIVELY from `_parked_approval_payloads` (never
            # from whatever the card happened to already be showing), so a
            # round that mounted immediately (session_id was the active
            # session at round-start) was previously unrecoverable the
            # moment the user switched away and back: the lookup found
            # nothing, mounted `None`, and the round silently hung with a
            # stale NEEDS_APPROVAL badge and no card until its 120s
            # timeout. The `finally` below already pops this key
            # unconditionally (whenever `session_id is not None`,
            # regardless of `is_parked`) -- storing it unconditionally
            # here too makes retention symmetric with that cleanup, per
            # spec §5 ("card state survives tab switches") for every round,
            # not only parked ones.
            # F2b fix (Qodo wave): guard the store -- `switch_session`'s
            # own re-derive read (`.get()`) runs on the UI thread and can
            # race this worker-thread write.
            with self._approval_state_lock:
                self._parked_approval_payloads[session_id] = payload

        try:
            if is_parked:
                if self.app is not None and self.park_pending_approval is not None:
                    self.app.call_from_thread(self.park_pending_approval, session_id)
            else:
                self._marshal_pending_approval(payload)
            while not event.wait(_MCP_APPROVAL_POLL_SECONDS):
                if self._is_session_cancelled(session_id):
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
            # F2b fix (Qodo wave): guard both pops -- `resolve_pending_
            # approval`'s round_id lookup and `switch_session`'s re-derive
            # read can each observe these maps from the UI thread while
            # this worker thread tears the round down.
            with self._approval_state_lock:
                self._pending_approval_rounds.pop(round_id, None)
                # TASK-1050 (Defect B) fix round 1 (review): `_parked_
                # approval_payloads` is a SINGLE per-session slot that
                # always holds whichever round's payload was LAST WRITTEN
                # (arming always overwrites it) -- mirrors `request_skill_
                # install_confirm`'s/`request_skill_script_confirm`'s
                # identical guard. The original fix also popped whenever
                # the STORED payload was still this round's own id, on the
                # theory that meant "nothing has overwritten it since" --
                # but that condition is true exactly when THIS round is
                # the newest-armed one, which is also the common case where
                # an OLDER sibling round is still outstanding (arming a
                # round re-mounts/re-parks its card, which typically gets
                # decided before an already-waiting sibling does). Popping
                # there discarded the still-armed OLDER round's only
                # remaining payload, so a switch-away/back re-derive found
                # nothing and mounted `None` -- the reviewer reproduced
                # this live. Only the order-independent "no armed round
                # left for this session" test is safe: pop ONLY when this
                # is the LAST armed MCP round for the session. (Accepted
                # scope limitation: because the slot is single-payload,
                # last-armed-wins regardless of resolution order -- a
                # remount after the newest round resolves first shows the
                # newest round's now-stale payload, not the still-live
                # older round's; per-round payload storage is a larger
                # change out of scope here.)
                still_armed_same_session = session_id is not None and any(
                    state.get("session_id") == session_id
                    for state in self._pending_approval_rounds.values()
                )
                if session_id is not None and not still_armed_same_session:
                    self._parked_approval_payloads.pop(session_id, None)
            if session_id is not None:
                # TASK-1050 (Defect A): discard ONLY this round's own id --
                # the badge clears only once every bridge round for this
                # session (this one included) has resolved.
                self.discard_pending_round(session_id, round_id)
            # TASK-1050 fix round 2 (review): clearing the mounted card
            # here used to be guarded by `still_active`/`still_armed_same_
            # session` booleans computed BEFORE enqueueing the clear via
            # `call_from_thread` -- a race window between that snapshot and
            # the UI thread actually running the clear let a NEWER
            # same-session round arm, mount its own card, and then get
            # wiped by this round's now-stale clear. `_clear_pending_
            # approval_if_round_is_current` closes this by deferring the
            # ENTIRE decision (round-identity check included) to the UI
            # thread's own execution of the enqueued callable -- see its
            # docstring for the full race analysis.
            try:
                self._clear_pending_approval_if_round_is_current(
                    round_id, session_id
                )
            except Exception:  # noqa: BLE001 -- suppress teardown-time errors
                logger.opt(exception=True).debug(
                    "Failed to marshal approval clear during teardown"
                )

    def _clear_pending_approval_if_round_is_current(
        self, round_id: str | None, session_id: str | None
    ) -> None:
        """WORKER THREAD: enqueue a round-identity-guarded clear of the mounted MCP approval card.

        TASK-1050 fix round 2 (review): `request_mcp_approvals`'s
        `finally` used to decide whether to clear the mounted card via a
        plain boolean snapshot (``still_active and not still_armed_same_
        session``) computed BEFORE enqueueing ``self._marshal_pending_
        approval(None)`` through ``call_from_thread``. A NEWER same-
        session round could arm -- and, if this session is the one being
        viewed, fully mount ITS OWN card via its own ``call_from_thread``
        call -- in the window between that snapshot and the UI thread
        actually running THIS round's clear, which would then wipe the
        newer round's just-mounted card, stranding it until a manual
        remount (switch away/back) or its own timeout. Recomputing the
        same boolean any earlier -- e.g. right before enqueueing --
        narrows that window but cannot close it: checking and enqueueing
        are still two separate steps a concurrent round's own check-and-
        enqueue can interleave with. The only race-proof fix is to defer
        the ENTIRE decision to the single-threaded UI event loop's own
        execution of the enqueued callable, which re-reads the CURRENT
        authoritative state (never a snapshot) at the last possible
        moment: once that callable starts running, Textual's
        ``call_from_thread`` callables run to completion, one at a time,
        on the UI thread, so no further worker-thread interleaving can
        change the outcome mid-decision.

        The check is TWO-PART, and BOTH parts must pass before clearing:

        1. Round-IDENTITY based, not boolean: ``_parked_approval_payloads
           [session_id]`` always holds whichever round's payload was LAST
           WRITTEN (arming overwrites it -- mirrors the payload-pop
           guard's own "last-armed-wins" contract in the ``finally``
           block above). If it no longer names THIS round's own
           ``round_id``, a newer round has already claimed the slot (and,
           if ``session_id`` is the currently active session, already
           marshaled its own mount), so this round's clear must no-op.
           This closes the ORIGINAL Qodo TOCTOU (a newer round's own
           mount getting wiped by an older round's stale clear).

        2. Fix round 3 (re-review) regression fix: the identity check
           ALONE only detects "payload overwritten by a newer arm" -- it
           says nothing about whether a DIFFERENT, OLDER sibling round is
           still armed. When the newest-armed round resolves FIRST (the
           natural live ordering, per this file's own fix-round-1
           docstrings: arming a round typically gets it decided before an
           already-waiting older sibling does), the identity check
           trivially PASSES (nothing has overwritten the slot since this
           round armed) even though an older round is still pending --
           the old snapshot-based ``still_armed_same_session`` guard this
           closure replaced used to catch exactly this case; dropping it
           entirely (rather than also re-checking it live) reintroduced a
           stranded-card regression: the card cleared while the badge
           stayed lit, leaving the older round undecidable through the
           UI until its own timeout. Closed by ALSO re-reading (live,
           under the same lock, at the same last-possible-moment as the
           identity check -- never a pre-enqueue snapshot)
           ``_pending_approval_rounds`` filtered to this session: if ANY
           round remains registered there (this round's own entry is
           already popped earlier in ``finally``, before this closure
           even runs, so any hit here is necessarily a DIFFERENT,
           still-armed sibling), the clear must no-op just as surely as a
           failed identity check does.

        Only once both checks pass does it fall through to the
        ``still_active`` check -- also re-read live here, not from a
        snapshot -- before actually clearing.

        Args:
            round_id: This round's own id. Only consulted when
                ``session_id`` is not ``None`` (every session-attributed
                round is 1:1 with a real round id).
            session_id: This round's owning session. ``None`` preserves
                the pre-existing unconditional-clear behavior for legacy
                no-session callers -- there is no "newer round for this
                session" concept without a session to key by.
        """
        if self.app is None or self.set_pending_approval is None:
            return

        def _clear_if_still_current() -> None:
            if session_id is not None:
                # F2b-style guard: this runs on the UI thread, but a
                # worker thread can concurrently write `_parked_approval_
                # payloads`/`_pending_approval_rounds` under this same
                # lock (MCP's round registry shares `_approval_state_
                # lock` with the payload map, so both reads happen in one
                # atomic critical section).
                with self._approval_state_lock:
                    current = self._parked_approval_payloads.get(session_id)
                    still_armed_same_session = any(
                        state.get("session_id") == session_id
                        for state in self._pending_approval_rounds.values()
                    )
                if current is not None and current.get("round_id") != round_id:
                    # A newer round already claimed this session's
                    # retained-payload slot -- whatever the mounted card
                    # is currently showing (if this session is even the
                    # one being viewed) belongs to THAT round, not this
                    # one. Leave it alone.
                    return
                if still_armed_same_session:
                    # A DIFFERENT round (necessarily -- this round's own
                    # entry was already popped before this closure runs)
                    # is still armed for this session. Clearing now would
                    # strand it: card gone, badge still lit, undecidable
                    # through the UI until its own timeout.
                    return
                if session_id != (self.store.active_session_id or ""):
                    # Not (or no longer) the session being viewed --
                    # nothing of THIS round's own was ever mounted here
                    # (a parked round never marshals), or the user has
                    # since switched away (`switch_session`'s own
                    # explicit clear already handled the departing
                    # card). Clearing here would blank whatever the
                    # CURRENTLY active session's own card is showing.
                    return
            self.set_pending_approval(None)

        self.app.call_from_thread(_clear_if_still_current)

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
        session_id: str | None = None,
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

        Args:
            session_id: The run's OWNING session (Task 3/9) -- threaded
                into the composed provider's ``approval_callback`` (via a
                ``functools.partial`` binding, since ``MCPToolProvider``
                calls it with a fixed ``[pending]`` single-list arg) so a
                single-call fallback approval raised through
                ``invoke()``'s own gate parks/mounts and scopes its cancel
                check exactly like the batch review-hook path does.
                ``None`` (the default -- every pre-Task-9 call site) keeps
                ``request_mcp_approvals``' legacy no-session behavior.

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
        bound_request_approvals = functools.partial(
            self.request_mcp_approvals, session_id=session_id
        )
        provider = MCPToolProvider(
            service=service,
            main_loop=asyncio.get_running_loop(),
            approval_callback=bound_request_approvals,
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
        return provider, build_mcp_review_hook(provider, bound_request_approvals)

    def resolve_pending_approval(
        self, decisions: dict[str, str], *, round_id: str | None = None
    ) -> None:
        """UI THREAD: apply the user's batch decision, releasing the waiting worker thread.

        Called by ``ChatScreen``'s ``ChatApprovalCard.ApprovalDecided``
        handler, which forwards ``event.round_id`` -- the SAME id
        ``request_mcp_approvals`` stamped into the payload the card was
        built from (``ChatApprovalCard.set_batch`` stashes it;
        ``_submit_batch_decisions`` echoes it back on submit, mirroring
        ``resolve_pending_skill_script``'s identical ``request_id``
        round-trip).

        Fix round 1 (review CRITICAL finding): resolves ONLY the round
        whose id matches ``round_id`` -- never "whichever round belongs to
        the currently active session". ``ApprovalDecided`` travels as an
        async Textual message: a ``switch_session`` landing in the gap
        between the user's click and this handler running would otherwise
        let session A's decision resolve session B's completely different,
        unreviewed batch (or, for the same session, let a STALE decision
        from an already-ended round 1 resolve a newer round 2 that
        happened to arm before the stale message was delivered). A
        mismatched or stale ``round_id`` -- including one belonging to a
        round that already resolved and was popped -- is a safe no-op: the
        real round (if any) stays pending and its card re-derives
        unchanged on the next visit; nothing is ever auto-approved or
        denied-by-accident here.

        TASK-913 (AC#2): ``round_id=None`` no longer falls back to
        "whichever round belongs to the currently active session" -- it
        fails closed immediately, mirroring
        ``resolve_pending_skill_script``'s/``resolve_pending_skill_install``'s
        identical ``if request_id is None: return`` contract. Production
        (``ChatApprovalCard``/``ChatScreen``) has only ever had a single
        emitter (``ChatApprovalCard._submit_batch_decisions``) and it
        always threads the real ``round_id`` through; the active-session
        fallback existed only for legacy direct-call tests, which have
        been migrated to pass the real round id captured from the
        mounted/parked payload instead.

        A no-op both when ``round_id`` is ``None`` and when it doesn't
        match any currently-armed round (e.g. a stale message arriving
        after a timeout/cancellation already resolved and cleared it) --
        the real round (if any) stays pending and undecided; nothing is
        ever auto-approved or denied-by-accident here.

        NOTE: Snapshots the round's ``decisions``/``event`` into locals to
        avoid TOCTOU race: the worker thread's ``finally`` block pops the
        round entry out of ``_pending_approval_rounds`` concurrently. Guard
        and act only on the snapshots.

        Args:
            decisions: The user's per-``llm_name`` decision strings
                (``approve_once``/``approve_session``/``always_allow``/
                ``deny``) to merge into the round's shared decisions dict.
            round_id: The specific round to resolve (the id stamped onto
                the card the user actually decided). ``None`` (the
                default) never matches an armed round, so an un-migrated
                or malformed caller fails closed by omission.
        """
        # TASK-913 (AC#2): fail closed on a missing round_id rather than
        # scanning `_pending_approval_rounds.values()` for "whichever round
        # belongs to the active session" -- that active-session fallback
        # was production-unreachable (see docstring) and is now removed
        # entirely, taking its AC#1 lock-guarded-snapshot protection with
        # it (moot once the scan itself is gone). The remaining branch's
        # `.get()` read stays guarded: the worker thread's own registration
        # (`request_mcp_approvals`) and teardown (its `finally`) can mutate
        # this dict concurrently.
        if round_id is None:
            return
        with self._approval_state_lock:
            round_state = self._pending_approval_rounds.get(round_id)
        if round_state is None:
            return
        # Snapshot both at once to prevent TOCTOU race with worker thread's finally block
        decisions_dict = round_state["decisions"]
        approval_event = round_state["event"]
        decisions_dict.update(decisions or {})
        approval_event.set()

    # -- Skill-install confirm bridge (task-5, parked TASK-910) --------------

    def request_skill_install_confirm(
        self, url: str, *, session_id: str | None = None
    ) -> bool:
        """WORKER THREAD: ask the user to confirm a skill install before any fetch.

        TASK-910: mirrors ``request_mcp_approvals``' park/mount/retain
        contract. Registers a fresh round (event + decision box + owning
        session id) under a freshly minted request id in
        ``_pending_skill_install_rounds`` (mirrors ``_pending_skill_script_
        rounds``' identical per-round design -- the pre-TASK-910 single
        ``_pending_skill_install_event``/``_pending_skill_install_decision``
        pair could not survive two DIFFERENT sessions each raising their own
        install confirm concurrently, exactly the hazard task-581 already
        fixed for skill-script). Either MOUNTS the card immediately
        (``session_id`` is the active/viewed session, or unknown -- legacy
        no-session callers keep the pre-TASK-910 always-mount behavior) or
        PARKS it (a different, background session -- the retained payload
        goes into ``_parked_skill_install_payloads`` for ``switch_session``/
        ``new_session``/``close_session`` to remount later, and
        ``park_pending_approval`` fires the SAME fleet badge + one-shot
        toast machinery ``request_mcp_approvals`` uses, per the train's
        toast-copy convention).

        Then polls re-checking this round's OWN cancel signal
        (``_is_session_cancelled``, scoped to ``session_id`` when known) and
        a deadline. Cancel/stop (of the OWNING session, or real process
        teardown via ``_shutdown_requested``), timeout, or no wired UI all
        resolve to DENY (fail-closed). A plain switch away no longer denies
        -- the round parks and stays alive until its own resolution,
        cancellation, or shutdown. Returns True only on an explicit Allow.

        Args:
            url: The skill source URL the model wants to install, surfaced
                verbatim on the confirm card for the user to inspect.
            session_id: The run's OWNING session (Task 3/9/TASK-910).
                ``None`` preserves the pre-Task-9 VIEWED-session/global-flag
                fallback (see ``_is_session_cancelled``) and never parks.

        Returns:
            True only on an explicit Allow; every other path (deny, cancel,
            stop, timeout, or no wired UI) returns False.
        """
        # No UI bridge wired means the marshal below is a no-op and nothing
        # can ever set the Event -- fail closed immediately instead of
        # blocking for the full timeout with no way to be resolved.
        if self.app is None or self.set_pending_skill_install is None:
            return False

        event = threading.Event()
        decision: dict[str, bool] = {}
        request_id = str(uuid4())
        owning_session_id = session_id if session_id is not None else (
            self.store.active_session_id or ""
        )
        with self._pending_skill_install_lock:
            self._pending_skill_install_rounds[request_id] = {
                "event": event,
                "decision": decision,
                "session_id": owning_session_id,
            }

        timeout_seconds = (
            self.skill_install_confirm_timeout_seconds()
            if self.skill_install_confirm_timeout_seconds is not None
            else _DEFAULT_SKILL_INSTALL_CONFIRM_TIMEOUT_SECONDS
        )
        deadline = time.monotonic() + timeout_seconds
        payload = {
            "url": url,
            "timeout_seconds": timeout_seconds,
            "request_id": request_id,
            "session_id": owning_session_id,
        }
        # TASK-910: park rather than mount when this round's session is a
        # DIFFERENT, background session -- mirrors `request_mcp_approvals`'
        # identical `is_parked` gate. `session_id is None` (a legacy caller
        # with no session context) always mounts.
        is_parked = (
            session_id is not None
            and session_id != (self.store.active_session_id or "")
        )
        if session_id is not None:
            # TASK-1050 (Defect A): round-keyed, not a plain boolean -- see
            # `request_mcp_approvals`' identical `add_pending_round` call
            # for the full rationale (a sibling round from this bridge or
            # either of the other two must not have its badge stolen by
            # THIS round's own teardown).
            self.add_pending_round(session_id, request_id)
            # Retain THIS round's payload for EVERY session-attributed
            # round -- mounted or parked -- not just a parked one, mirroring
            # `request_mcp_approvals`' identical retention (Fix wave,
            # CRITICAL 1): a round that mounted immediately must still be
            # recoverable after a switch-away-and-back.
            with self._approval_state_lock:
                self._parked_skill_install_payloads[session_id] = payload
        try:
            if is_parked:
                if self.app is not None and self.park_pending_approval is not None:
                    self.app.call_from_thread(self.park_pending_approval, session_id)
            else:
                self._marshal_pending_skill_install(payload)
            while not event.wait(_MCP_APPROVAL_POLL_SECONDS):
                if self._is_session_cancelled(session_id):
                    break
                if time.monotonic() >= deadline:
                    break
            return bool(decision.get("allow", False))
        finally:
            with self._pending_skill_install_lock:
                self._pending_skill_install_rounds.pop(request_id, None)
                still_armed_same_session = any(
                    state.get("session_id") == owning_session_id
                    for state in self._pending_skill_install_rounds.values()
                )
            if session_id is not None:
                # TASK-1050 (Defect B) fix round 1 (review): `_parked_
                # skill_install_payloads` is a SINGLE per-session slot
                # holding whichever round's payload was LAST WRITTEN
                # (arming overwrites it) -- an unconditional pop here would
                # let the EARLIER round's teardown discard the NEWER
                # round's still-armed retained payload. The original fix
                # also popped whenever the stored payload was still this
                # round's own id ("nothing has overwritten it since"), but
                # that is true exactly when THIS round is the newest-armed
                # one -- which is also the common case where an OLDER
                # sibling is still outstanding (arming re-mounts/re-parks a
                # card, which typically gets decided before an
                # already-waiting sibling does). Popping there discarded
                # the still-armed OLDER round's only remaining payload
                # (reviewer reproduced live). Only the order-independent
                # "no armed round left for this session" test is safe: pop
                # ONLY when this is the LAST armed round for the session.
                # (Accepted scope limitation: last-armed-wins regardless of
                # resolution order -- see `request_mcp_approvals`' mirror
                # of this comment for the full rationale.)
                with self._approval_state_lock:
                    if not still_armed_same_session:
                        self._parked_skill_install_payloads.pop(session_id, None)
                # TASK-1050 (Defect A): discard ONLY this round's own id --
                # the badge clears only once every bridge round for this
                # session (this one included) has resolved.
                self.discard_pending_round(session_id, request_id)
            # TASK-1050 fix round 2 (review): mirrors `request_mcp_
            # approvals`' identical fix -- a plain boolean snapshot
            # (`still_active`/`still_armed_same_session`) computed before
            # enqueueing the clear via `call_from_thread` leaves a race
            # window where a NEWER same-session round can arm, mount its
            # own card, and then get wiped by this round's now-stale
            # clear. `_clear_pending_skill_install_if_round_is_current`
            # defers the whole decision (round-identity check included)
            # to the UI thread's own execution of the enqueued callable --
            # see `_clear_pending_approval_if_round_is_current`'s
            # docstring for the full race analysis.
            try:
                self._clear_pending_skill_install_if_round_is_current(
                    request_id, session_id
                )
            except Exception:  # noqa: BLE001
                logger.opt(exception=True).debug(
                    "Failed to clear skill-install confirm during teardown"
                )

    def _clear_pending_skill_install_if_round_is_current(
        self, request_id: str | None, session_id: str | None
    ) -> None:
        """WORKER THREAD: enqueue a round-identity-guarded clear of the mounted skill-install card.

        TASK-1050 fix round 2 (review): mirrors ``_clear_pending_
        approval_if_round_is_current``'s identical race-proofing -- see
        that method's docstring for the full analysis of why a boolean
        snapshot computed before enqueueing (however late) cannot close
        this race, only deferring the whole identity check to the UI
        thread's own execution of the enqueued callable can.

        Fix round 3 (re-review) regression fix: mirrors ``_clear_pending_
        approval_if_round_is_current``'s identical two-part guard -- the
        round-identity check alone only catches "payload overwritten by a
        newer arm", not "a DIFFERENT, OLDER sibling round is still armed"
        (true exactly when THIS round is the newest-armed one and
        resolves FIRST, the natural live ordering). Also re-reads
        ``_pending_skill_install_rounds`` live (under ``_pending_skill_
        install_lock``, sequentially after -- never nested with --
        ``_approval_state_lock``, matching this file's existing lock-
        ordering discipline) filtered to this session; any hit there is
        necessarily a still-armed sibling (this round's own entry is
        already popped earlier in ``finally``), so the clear must no-op
        exactly as it does on a failed identity check.

        Args:
            request_id: This round's own id. Only consulted when
                ``session_id`` is not ``None``.
            session_id: This round's owning session. ``None`` preserves
                the pre-existing unconditional-clear behavior for legacy
                no-session callers.
        """
        if self.app is None or self.set_pending_skill_install is None:
            return

        def _clear_if_still_current() -> None:
            if session_id is not None:
                with self._approval_state_lock:
                    current = self._parked_skill_install_payloads.get(session_id)
                if current is not None and current.get("request_id") != request_id:
                    return
                with self._pending_skill_install_lock:
                    still_armed_same_session = any(
                        state.get("session_id") == session_id
                        for state in self._pending_skill_install_rounds.values()
                    )
                if still_armed_same_session:
                    return
                if session_id != (self.store.active_session_id or ""):
                    return
            self.set_pending_skill_install(None)

        self.app.call_from_thread(_clear_if_still_current)

    def _remount_parked_skill_install(self, session_id: str) -> None:
        """Re-derive the mounted skill-install confirm card for ``session_id``.

        TASK-910: called from `switch_session`/`new_session`/`close_session`
        exactly like the MCP approval card's own re-derive -- mounts
        ``session_id``'s retained payload (if any) and clears whatever the
        departing session had shown, all in one call. A no-op when no UI
        bridge is wired.

        Args:
            session_id: The session now being activated/viewed.
        """
        if self.set_pending_skill_install is None:
            return
        with self._approval_state_lock:
            parked_payload = self._parked_skill_install_payloads.get(session_id)
        self.set_pending_skill_install(parked_payload)

    def _marshal_pending_skill_install(self, payload: dict[str, Any] | None) -> None:
        """WORKER THREAD: hand a skill-install confirm payload to the UI thread.

        No-op when no UI bridge is wired (``self.app`` or
        ``set_pending_skill_install`` is None).

        Args:
            payload: The pending confirm's ``{"url", "timeout_seconds"}``
                dict to show, or None to clear/hide the card.
        """
        if self.app is not None and self.set_pending_skill_install is not None:
            self.app.call_from_thread(self.set_pending_skill_install, payload)

    def resolve_pending_skill_install(
        self, allow: bool, *, request_id: str | None = None
    ) -> None:
        """UI THREAD: apply the user's Allow/Deny, releasing the worker thread.

        TASK-910: strict match against ``request_id``, mirroring
        ``resolve_pending_skill_script``'s identical contract -- a resolve
        carrying no id, or an id belonging to any round other than the one
        it names, is silently dropped rather than resolved. This closes the
        same stale-late-click hazard ``resolve_pending_skill_script``'s own
        docstring documents: once two sessions can each have their own
        concurrent install-confirm round (TASK-910 parking), "whichever
        round happens to be active" is no longer a safe fallback the way it
        was pre-TASK-910 (a single global slot could only ever have one
        candidate).

        Args:
            allow: True to allow the pending install, False to deny it.
            request_id: The armed round's id, as echoed back by the UI
                (``SkillInstallConfirmCard.InstallDecided.request_id``).
                ``None`` (the default) never matches an armed round, so an
                un-migrated or malformed caller fails closed by omission.
        """
        if request_id is None:
            return
        with self._pending_skill_install_lock:
            round_state = self._pending_skill_install_rounds.get(request_id)
        if round_state is None:
            return
        round_state["decision"]["allow"] = bool(allow)
        round_state["event"].set()

    def pending_skill_install_ids(self) -> list[str]:
        """Return the request ids of every currently-armed install-confirm round.

        Mirrors ``pending_skill_script_ids`` -- exposed for tests and for
        any surface that needs to know whether a decision is outstanding.

        Returns:
            The armed round ids, in insertion order. Empty when none is
            pending.
        """
        with self._pending_skill_install_lock:
            return list(self._pending_skill_install_rounds)

    # -- Skill-script confirm bridge -----------------------------------------

    def request_skill_script_confirm(
        self, payload: dict[str, Any], *, session_id: str | None = None
    ) -> dict[str, bool]:
        """WORKER THREAD: ask the user to confirm running a skill's script.

        Mirrors request_skill_install_confirm, but carries a two-part decision:
        allow this run, and whether to remember the choice for this skill.

        Each call arms a fresh round under a newly-generated request id
        (embedded in the payload handed to the UI as ``"request_id"``) so
        that ``resolve_pending_skill_script`` can reject a decision left
        over from a prior, already-torn-down round -- see that method's
        docstring for why this matters.

        TASK-910: also carries the SAME park/mount/retain contract as
        ``request_mcp_approvals``/``request_skill_install_confirm`` -- see
        ``request_skill_install_confirm``'s docstring for the full
        mount-vs-park/retain rationale, identical here. The per-round
        registry (keyed by ``request_id``, task-581) now also stores this
        round's owning session id, so teardown can distinguish "another
        round for a DIFFERENT session is still armed" (must not suppress
        clearing THIS session's card) from "another round for the SAME
        session is still armed" (must not clear it out from under that
        sibling round, preserving task-581's original guarantee).

        Args:
            payload: Confirm details to render ({"skill_name", "script_path",
                "mechanism", "args", ...}); "timeout_seconds" and
                "request_id" keys are added before marshaling to the UI.
            session_id: The run's OWNING session (Task 3/9/TASK-910), scoping
                the cancel check (``_is_session_cancelled`` -- PA-T9 finding
                #1) and the park/mount decision. ``None`` preserves the
                pre-Task-9 VIEWED-session/global-flag fallback and never
                parks.

        Returns:
            ``{"allow": bool, "remember": bool}``. Every non-Allow path (deny,
            cancel, stop, timeout, no wired UI) returns ``allow=False``.
        """
        if self.app is None or self.set_pending_skill_script is None:
            return {"allow": False, "remember": False}

        event = threading.Event()
        decision: dict[str, bool] = {}
        request_id = str(uuid4())
        owning_session_id = session_id if session_id is not None else (
            self.store.active_session_id or ""
        )
        with self._pending_skill_script_lock:
            self._pending_skill_script_rounds[request_id] = {
                "event": event,
                "decision": decision,
                "session_id": owning_session_id,
            }

        timeout_seconds = (
            self.skill_script_confirm_timeout_seconds()
            if self.skill_script_confirm_timeout_seconds is not None
            else _DEFAULT_SKILL_SCRIPT_CONFIRM_TIMEOUT_SECONDS
        )
        deadline = time.monotonic() + timeout_seconds
        card_payload = dict(payload)
        card_payload["timeout_seconds"] = timeout_seconds
        card_payload["request_id"] = request_id
        card_payload["session_id"] = owning_session_id
        is_parked = (
            session_id is not None
            and session_id != (self.store.active_session_id or "")
        )
        if session_id is not None:
            # TASK-1050 (Defect A): round-keyed, not a plain boolean -- see
            # `request_mcp_approvals`' identical `add_pending_round` call
            # for the full rationale.
            self.add_pending_round(session_id, request_id)
            with self._approval_state_lock:
                self._parked_skill_script_payloads[session_id] = card_payload
        try:
            if is_parked:
                if self.app is not None and self.park_pending_approval is not None:
                    self.app.call_from_thread(self.park_pending_approval, session_id)
            else:
                self._marshal_pending_skill_script(card_payload)
            while not event.wait(_MCP_APPROVAL_POLL_SECONDS):
                if self._is_session_cancelled(session_id):
                    break
                if time.monotonic() >= deadline:
                    break
            return {
                "allow": bool(decision.get("allow", False)),
                "remember": bool(decision.get("remember", False)),
            }
        finally:
            with self._pending_skill_script_lock:
                self._pending_skill_script_rounds.pop(request_id, None)
                still_armed_same_session = any(
                    state.get("session_id") == owning_session_id
                    for state in self._pending_skill_script_rounds.values()
                )
            if session_id is not None:
                # TASK-1050 (Defect B) fix round 1 (review): mirrors
                # `request_skill_install_confirm`'s identical guard --
                # `_parked_skill_script_payloads` is a SINGLE per-session
                # slot holding whichever round's payload was LAST WRITTEN.
                # The original fix also popped whenever the stored payload
                # was still this round's own id, but that is true exactly
                # when THIS round is the newest-armed one -- which is also
                # the common case where an OLDER sibling is still
                # outstanding. Popping there discarded the still-armed
                # OLDER round's only remaining payload (reviewer reproduced
                # live). Only pop when this is the LAST armed round for the
                # session -- see `request_mcp_approvals`' mirror of this
                # comment for the full rationale and the accepted
                # single-slot/last-armed-wins scope limitation.
                with self._approval_state_lock:
                    if not still_armed_same_session:
                        self._parked_skill_script_payloads.pop(session_id, None)
                # TASK-1050 (Defect A): discard ONLY this round's own id --
                # the badge clears only once every bridge round for this
                # session (this one included) has resolved.
                self.discard_pending_round(session_id, request_id)
            # TASK-1050 fix round 2 (review): mirrors `request_mcp_
            # approvals`'/`request_skill_install_confirm`'s identical
            # fix -- see `_clear_pending_approval_if_round_is_current`'s
            # docstring for the full race analysis a plain boolean
            # snapshot (however late it is recomputed) cannot close.
            try:
                self._clear_pending_skill_script_if_round_is_current(
                    request_id, session_id
                )
            except Exception:  # noqa: BLE001
                logger.opt(exception=True).debug(
                    "Failed to clear skill-script confirm during teardown"
                )

    def _clear_pending_skill_script_if_round_is_current(
        self, request_id: str | None, session_id: str | None
    ) -> None:
        """WORKER THREAD: enqueue a round-identity-guarded clear of the mounted skill-script card.

        TASK-1050 fix round 2 (review): mirrors ``_clear_pending_
        approval_if_round_is_current``'s identical race-proofing -- see
        that method's docstring for the full analysis.

        Fix round 3 (re-review) regression fix: mirrors ``_clear_pending_
        approval_if_round_is_current``'s/``_clear_pending_skill_install_
        if_round_is_current``'s identical two-part guard -- also
        re-reads ``_pending_skill_script_rounds`` live (under
        ``_pending_skill_script_lock``, sequentially after -- never
        nested with -- ``_approval_state_lock``) filtered to this
        session, so a still-armed OLDER sibling round (true exactly when
        THIS round is the newest-armed one and resolves FIRST) blocks the
        clear just as surely as a failed identity check does.

        Args:
            request_id: This round's own id. Only consulted when
                ``session_id`` is not ``None``.
            session_id: This round's owning session. ``None`` preserves
                the pre-existing unconditional-clear behavior for legacy
                no-session callers.
        """
        if self.app is None or self.set_pending_skill_script is None:
            return

        def _clear_if_still_current() -> None:
            if session_id is not None:
                with self._approval_state_lock:
                    current = self._parked_skill_script_payloads.get(session_id)
                if current is not None and current.get("request_id") != request_id:
                    return
                with self._pending_skill_script_lock:
                    still_armed_same_session = any(
                        state.get("session_id") == session_id
                        for state in self._pending_skill_script_rounds.values()
                    )
                if still_armed_same_session:
                    return
                if session_id != (self.store.active_session_id or ""):
                    return
            self.set_pending_skill_script(None)

        self.app.call_from_thread(_clear_if_still_current)

    def _remount_parked_skill_script(self, session_id: str) -> None:
        """Re-derive the mounted skill-script confirm card for ``session_id``.

        TASK-910: called from `switch_session`/`new_session`/`close_session`
        exactly like the MCP approval card's own re-derive -- mounts
        ``session_id``'s retained payload (if any) and clears whatever the
        departing session had shown, all in one call. A no-op when no UI
        bridge is wired.

        Args:
            session_id: The session now being activated/viewed.
        """
        if self.set_pending_skill_script is None:
            return
        with self._approval_state_lock:
            parked_payload = self._parked_skill_script_payloads.get(session_id)
        self.set_pending_skill_script(parked_payload)

    def _marshal_pending_skill_script(self, payload: dict[str, Any] | None) -> None:
        """WORKER THREAD: hand a skill-script confirm payload to the UI thread.

        Args:
            payload: The pending confirm dict to show, or None to hide it.
        """
        if self.app is not None and self.set_pending_skill_script is not None:
            self.app.call_from_thread(self.set_pending_skill_script, payload)

    def resolve_pending_skill_script(
        self, allow: bool, remember: bool, request_id: str | None = None
    ) -> None:
        """UI THREAD: apply the user's decision, releasing the worker thread.

        ``request_id`` must be the exact ``"request_id"`` value the pending
        confirm's payload carried (``request_skill_script_confirm`` embeds
        a fresh one per round, and the confirm card built in a later task
        MUST echo it back here unchanged). This is a strict match: a
        resolve carrying no id, or an id from any round other than the one
        currently armed, is silently dropped rather than resolved.

        This guards against a real arbitrary-code-execution hazard: if
        round 1 ends (deadline, cancel, stop, conversation switch) and the
        agent immediately issues a second ``run_skill_script`` call
        arming round 2, a ``Button.Pressed`` queued for round 1 just
        before its teardown could otherwise be handled after round 2 is
        armed -- resolving round 2 (a script the user never saw) with
        round 1's stale click. Widget messages and ``call_from_thread``
        calls are separate queues, so ordering across a round boundary is
        not guaranteed.

        Args:
            allow: True to run the script this once.
            remember: True to also grant this skill standing permission.
            request_id: The armed round's id, as echoed back by the UI.
                ``None`` (the default) never matches an armed round, so an
                un-migrated or malformed caller fails closed by omission.
        """
        if request_id is None:
            return
        with self._pending_skill_script_lock:
            round_state = self._pending_skill_script_rounds.get(request_id)
        if round_state is None:
            return
        round_state["decision"]["allow"] = bool(allow)
        round_state["decision"]["remember"] = bool(remember)
        round_state["event"].set()

    def pending_skill_script_ids(self) -> list[str]:
        """Return the request ids of every currently-armed confirm round.

        Returns:
            The armed round ids, in insertion order. Empty when none is
            pending. Exposed for tests and for any surface that needs to
            know whether a decision is outstanding.
        """
        with self._pending_skill_script_lock:
            return list(self._pending_skill_script_rounds)

    def stop_active_run(self, *, record_user_stop: bool = True) -> bool:
        """Request the ACTIVE (viewed) session's stream to stop at the next
        safe boundary.

        Task 3b requirement 2: name and public semantics are unchanged --
        this is the Stop button's contract, and it only ever targets
        whatever session ``self.store.active_session_id`` currently is,
        never a background run in another tab. A background run is
        completely unaffected by this call (its own entries in the
        per-session maps below are untouched); see ``shutdown`` for the
        teardown path that stops every session at once.

        Args:
            record_user_stop: Append the explicit "stopped by user"
                transcript record (TASK-337 AC3). ``shutdown`` passes
                ``False`` — a teardown stop is not a user action.

        Returns:
            True when the viewed session had an active run and it was
            stopped; False (a no-op) when it did not.
        """
        session_id = self.store.active_session_id or ""
        repair_session = self._active_citation_repair_sessions.get(session_id)
        if repair_session is not None and repair_session.selection_committed:
            return False
        if repair_session is not None and repair_session.phase in {
            "checking",
            "repair_streaming",
        }:
            if self._active_assistant_message_ids.get(session_id) is None:
                return False
            repair_session.cancel_reason = "user" if record_user_stop else "shutdown"
            self._signal_stop(session_id=session_id)
            task = self._active_stream_tasks.get(session_id)
            if task is not None and task is not asyncio.current_task():
                task.cancel()
            return True

        if self.run_state.status is not ConsoleRunStatus.STREAMING:
            assistant_message_id = self._active_streaming_assistant_message_id()
            if assistant_message_id is None:
                return False
        else:
            assistant_message_id = (
                self._active_assistant_message_ids.get(session_id)
                or self._active_streaming_assistant_message_id()
            )
        if assistant_message_id is None:
            return False
        self._signal_stop(session_id=session_id)
        self._mark_stream_stopped(
            assistant_message_id,
            visible_copy="Response stopped.",
        )
        if record_user_stop:
            # TASK-337 AC3: a durable, explicit record — the run-state chip
            # copy is transient and the review found nothing else marked
            # the interruption.
            try:
                owner_id = self.store.session_id_for_message(assistant_message_id)
                self.store.append_message(
                    owner_id,
                    role=ConsoleMessageRole.SYSTEM,
                    content="Response stopped by user.",
                )
            except KeyError:
                pass
        task = self._active_stream_tasks.get(session_id)
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        return True

    async def shutdown(self) -> None:
        """Stop and await EVERY session's active stream task before owner
        teardown.

        Task 3b requirement 3: unlike ``stop_active_run`` (deliberately
        scoped to the VIEWED session only), teardown is global across THIS
        controller instance's OWN sessions -- a background run must never
        survive this instance's shutdown just because the user was looking
        at a different tab. Mirrors ``stop_active_run``'s manual
        signal-then-cancel fallback for every session with a live entry,
        rather than reusing ``stop_active_run`` itself, which by contract
        only ever resolves the active session.

        Callers: real process exit (owner app teardown) is one caller, but
        NOT the only one -- ``ChatScreen.on_unmount`` also awaits this on
        every ordinary navigation AWAY from the Console screen (switching
        tabs unmounts the outgoing screen), which is far more frequent
        than process exit. Any docstring here or in ``_is_session_
        cancelled`` that called ``_shutdown_requested`` "real process
        teardown" was describing only one of its two callers.

        F5 fix (Qodo wave): sets ``_shutdown_requested`` unconditionally
        and FIRST -- before the no-tasks early return below -- so a
        worker-thread approval/confirm bridge polling on behalf of a run
        this method doesn't (yet) see in ``_active_stream_tasks`` still
        observes this instance's teardown. TASK-1052: this was true
        immediately only for a legacy ``session_id=None`` caller (whose
        fallback branch in ``_is_session_cancelled`` already OR'd in this
        flag); a round armed with a REAL ``session_id`` before its session
        reached ``_active_stream_tasks`` -- exactly the case this
        docstring describes -- previously still had to fall through to
        its own confirm/approval timeout, since the per-session
        ``_signal_stop`` fanout below only reaches sessions present in
        this method's ``tasks`` snapshot. ``_is_session_cancelled``'s
        real-``session_id`` branch now also ORs in ``_shutdown_requested``
        directly, closing that gap so this paragraph is accurate for
        every caller.

        Correction (review, TASK-1052): setting ``_shutdown_requested``
        unconditionally here is safe NOT because this method only ever
        runs at real process exit (it doesn't -- see "Callers" above), but
        because it is scoped to THIS controller instance, and
        ``ChatScreen`` never reuses an instance after unmounting it:
        ``_ensure_console_chat_controller`` only ever (re)builds a fresh
        ``ConsoleChatController`` lazily, and ``on_unmount`` both awaits
        this method AND drops the screen's reference to the instance
        before that lazy rebuild can ever fire again. A round still armed
        on an instance whose ``shutdown()`` already ran cannot be resolved
        through a UI that no longer exists regardless of this flag, and a
        LATER Console visit's rounds run against a brand-new instance with
        its own, unset ``_shutdown_requested`` -- so the permanently-set
        flag on the old instance can never poison it.
        """
        self._shutdown_requested.set()
        for message_id in tuple(self._original_attempts):
            self.clear_original_attempt(message_id)
        tasks = dict(self._active_stream_tasks)
        if not tasks:
            return
        current = asyncio.current_task()
        for session_id in tasks:
            # Dev's citation-repair feature threads a `cancel_reason`
            # ("user" vs "shutdown") through `ConsoleCitationRepairSession`
            # so `commit_canceled()` knows whether to append a "canceled by
            # user" system row (`stop_active_run` sets this for the VIEWED
            # session it targets) -- global teardown must set the same
            # field for EVERY session's own in-flight repair, or a
            # still-checking/repair-streaming session falls back to
            # whatever `cancel_reason` (if any) was already there.
            repair_session = self._active_citation_repair_sessions.get(session_id)
            if (
                repair_session is not None
                and not repair_session.selection_committed
                and repair_session.phase in {"checking", "repair_streaming"}
            ):
                repair_session.cancel_reason = "shutdown"
            self._signal_stop(session_id=session_id)
        for session_id, task in tasks.items():
            if task is not current:
                task.cancel()
        for session_id, task in tasks.items():
            if task is current:
                # Shutdown was invoked from within its own run's task --
                # cannot cancel/await itself; that run's own finally will
                # still fire once this coroutine naturally unwinds.
                continue
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                # Shutdown is a teardown path; stale task failures should not crash owner cleanup.
                pass
        self._stop_requested = False
        # Safety net: each task's own `finally` (in `_stream_assistant_
        # response`/`_run_agent_reply`) already pops ITS OWN session's
        # entries on the happy path -- this only catches a task that
        # somehow never reached that finally (e.g. a test double, or a
        # task that failed before it), so teardown never leaves a stale
        # entry behind for any session.
        for session_id, task in tasks.items():
            if self._active_stream_tasks.get(session_id) is task:
                self._active_stream_tasks.pop(session_id, None)
                self._active_assistant_message_ids.pop(session_id, None)
                self._active_cancel_events.pop(session_id, None)

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
            self._set_run_state(
                ConsoleRunState.blocked(visible_copy), session_id=session_id
            )
            return ConsoleSubmitResult(False, False, visible_copy)
        if message.status != "failed":
            return self._block(session_id, "Only failed messages can be retried.")

        self._set_run_state(
            ConsoleRunState.retrying("Retrying failed response."),
            session_id=session_id,
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
            annotate_ids=True,
        )
        self._ensure_user_continuation_instruction(provider_messages)
        (
            provider_messages,
            refuse,
            skill_notes,
            skill_bindings,
            skill_bundle_block,
        ) = await self._apply_skill_substitution(provider_messages)
        if refuse is not None:
            return self._block(session_id, refuse)
        for note in skill_notes:
            # An embedded skipped-skill note is never an abort: append the
            # same system-row copy `_block` would, then let the turn proceed.
            self.store.append_message(
                session_id, role=ConsoleMessageRole.SYSTEM, content=note
            )
        provider_messages = await self._apply_chat_dictionaries(
            provider_messages, session_id
        )
        provider_messages = await self._apply_world_info(provider_messages, session_id)
        prefill = self._pinned_prefill_for_session(session_id)
        return await self._stream_assistant_response(
            resolution=resolution,
            provider_messages=provider_messages,
            assistant_message_id=message_id,
            prepare_retry=True,
            prefill=prefill,
            skill_bindings=skill_bindings,
            skill_bundle_block=skill_bundle_block,
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
            self._set_run_state(
                ConsoleRunState.blocked(visible_copy), session_id=session_id
            )
            return ConsoleSubmitResult(False, False, visible_copy)

        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.VALIDATING, "Validating provider."),
            session_id=session_id,
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
            session_id, message_id, annotate_ids=True
        )
        self._ensure_user_continuation_instruction(provider_messages)
        if not self._has_user_turn(provider_messages):
            return self._block(
                session_id,
                "Nothing to continue before the first message.",
            )
        (
            provider_messages,
            refuse,
            skill_notes,
            skill_bindings,
            skill_bundle_block,
        ) = await self._apply_skill_substitution(provider_messages)
        if refuse is not None:
            return self._block(session_id, refuse)
        for note in skill_notes:
            # An embedded skipped-skill note is never an abort: append the
            # same system-row copy `_block` would, then let the turn proceed.
            self.store.append_message(
                session_id, role=ConsoleMessageRole.SYSTEM, content=note
            )
        provider_messages = await self._apply_chat_dictionaries(
            provider_messages, session_id
        )
        provider_messages = await self._apply_world_info(provider_messages, session_id)
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
            skill_bindings=skill_bindings,
            skill_bundle_block=skill_bundle_block,
        )

    async def regenerate_message(self, message_id: str) -> ConsoleSubmitResult:
        """Regenerate a selected assistant message by forking a sibling branch.

        Unlike the pre-Task-6 behavior (streaming a replacement *variant*
        into the SAME message via ``variant_mode=True`` /
        ``begin_variant_stream``/``finalize_variant_stream``), this forks a
        new assistant node alongside ``message_id`` under its own parent
        (``store.create_sibling``) and streams into that NEW node normally
        (``variant_mode=False``). The anchor (and any old tail beneath it,
        for a mid-conversation regenerate) is left untouched and simply
        drops off the active path -- still reachable via
        ``store.set_active_leaf``, never deleted.

        All validation/blocking checks (provider readiness, "nothing to
        regenerate before the first message", a refusing skill) run BEFORE
        the sibling is created, mirroring the old mutate-only-once-committed
        discipline: a blocked regenerate must not leave a stray empty node
        forked into the tree. Because the fork shares the anchor's own
        parent, ``provider_messages`` computed with
        ``before_message_id=message_id`` (while ``message_id`` is still on
        the active path) is identical to computing it against the new
        sibling's id afterward -- both yield the anchor's ancestor chain --
        so it is safe to build once, up front.

        On stream FAILURE, the new sibling node itself becomes a ``failed``
        node on the active path (retryable via ``retry_message``), rather
        than restoring the anchor's prior reply in place -- this is the
        intended node-model behavior, not a regression: the anchor is a
        completely separate node and was never touched.
        """
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
            self._set_run_state(
                ConsoleRunState.blocked(visible_copy), session_id=session_id
            )
            return ConsoleSubmitResult(False, False, visible_copy)

        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.VALIDATING, "Validating provider."),
            session_id=session_id,
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
            annotate_ids=True,
        )
        self._ensure_user_continuation_instruction(provider_messages)
        if not self._has_user_turn(provider_messages):
            return self._block(
                session_id,
                "Nothing to regenerate before the first message.",
            )
        (
            provider_messages,
            refuse,
            skill_notes,
            skill_bindings,
            skill_bundle_block,
        ) = await self._apply_skill_substitution(provider_messages)
        if refuse is not None:
            return self._block(session_id, refuse)
        for note in skill_notes:
            # An embedded skipped-skill note is never an abort: append the
            # same system-row copy `_block` would, then let the turn proceed.
            self.store.append_message(
                session_id, role=ConsoleMessageRole.SYSTEM, content=note
            )
        provider_messages = await self._apply_chat_dictionaries(
            provider_messages, session_id
        )
        provider_messages = await self._apply_world_info(provider_messages, session_id)
        prefill = self._pinned_prefill_for_session(session_id)
        self.clear_original_attempt(message_id)
        new_message = self.store.create_sibling(
            message_id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            persist=self.store.persistence is not None,
        )
        return await self._stream_assistant_response(
            resolution=resolution,
            provider_messages=provider_messages,
            assistant_message_id=new_message.id,
            variant_mode=False,
            prefill=prefill,
            skill_bindings=skill_bindings,
            skill_bundle_block=skill_bundle_block,
        )

    #: Guidance cap for the transcript span fed to the summarizer (Task 3).
    #: Well above any realistic single-summary span so it never trims in tests
    #: or normal use; a runaway history drops its OLDEST turns before the call.
    _SUMMARY_SPAN_TOKEN_BUDGET = 12000

    async def summarize_up_to(self, message_id: str) -> ConsoleSubmitResult:
        """Summarize the active path up to (excluding) a USER message.

        Console `/rewind` "Summarize up to here" (SP2, Task 3). Runs the
        session's resolved provider (non-streaming) over the active-path turns
        before ``message_id`` and stores the result as the session's boundary
        summary (``store.set_session_context_summary``). The visible transcript
        is never mutated -- only the provider CONTEXT is later compacted at the
        dispatch choke point (see ``_apply_context_summary_compaction``).

        Gates run FIRST and NONE of them mutates transcript state (the Phase B
        discipline): an active run, a missing session, an off-path or non-USER
        target, a target with nothing before it, and provider-not-ready each
        return a blocked ``ConsoleSubmitResult`` via ``_summarize_block`` --
        which only sets the run state, never appends a system row. Rolling
        re-summarize (a prior boundary already on the path before ``message_id``)
        prepends the prior summary and only re-sends the turns SINCE that
        boundary. On an empty reply or a provider error the stored summary is
        left untouched.

        Args:
            message_id: Native id of the USER turn to summarize UP TO.

        Returns:
            ``ConsoleSubmitResult`` -- ``accepted`` True only when a non-empty
            summary was generated and stored.
        """
        active_rejection = self._active_run_rejection()
        if active_rejection is not None:
            return active_rejection

        session_id = self.store.active_session_id
        if session_id is None:
            return ConsoleSubmitResult(False, False, "No active Console session.")

        if message_id not in self.store.active_path_message_ids(session_id):
            return self._summarize_block(
                session_id, "Switch to that branch before summarizing."
            )
        try:
            target = self.store.get_message(message_id)
        except KeyError:
            return self._summarize_block(
                session_id, "Switch to that branch before summarizing."
            )
        if target.role is not ConsoleMessageRole.USER:
            return self._summarize_block(
                session_id, "Only your own messages can be summarized up to here."
            )

        messages = self.store.messages_for_session(session_id)
        target_index = next(
            (i for i, m in enumerate(messages) if m.id == message_id), None
        )
        if target_index is None:
            return self._summarize_block(
                session_id, "Switch to that branch before summarizing."
            )
        before = [
            m
            for m in messages[:target_index]
            if m.role in {ConsoleMessageRole.USER, ConsoleMessageRole.ASSISTANT}
        ]
        if not before:
            return self._summarize_block(
                session_id, "Nothing to summarize before that message."
            )

        # Rolling compaction: when a prior boundary sits on this path BEFORE the
        # target, the prior summary already covers everything strictly before
        # it, so re-summarize only from that boundary (inclusive) forward and
        # fold the prior summary in.
        prev_summary, prev_boundary_id = self.store.session_context_summary(session_id)
        start_index = 0
        rolling_summary: str | None = None
        if prev_boundary_id is not None and prev_summary:
            prev_index = next(
                (i for i, m in enumerate(messages) if m.id == prev_boundary_id), None
            )
            if prev_index is not None and prev_index < target_index:
                start_index = prev_index
                rolling_summary = prev_summary
        span = [
            m
            for m in messages[start_index:target_index]
            if m.role in {ConsoleMessageRole.USER, ConsoleMessageRole.ASSISTANT}
        ]

        # "Summarizing..." run state, set the way regenerate sets VALIDATING.
        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.VALIDATING, "Summarizing conversation…"),
            session_id=session_id,
        )
        resolution = await self.provider_gateway.resolve_for_send(
            self._provider_selection()
        )
        if not getattr(resolution, "ready", False):
            return self._summarize_block(
                session_id,
                self._blocked_visible_copy(getattr(resolution, "visible_copy", "")),
            )

        span_text = self._build_summary_span_text(
            span, rolling_summary, model=getattr(resolution, "model", None) or ""
        )
        summarize_messages = [
            {
                "role": ConsoleMessageRole.SYSTEM.value,
                "content": get_internal_prompt("console.rewind_summarize"),
            },
            {"role": ConsoleMessageRole.USER.value, "content": span_text},
        ]
        try:
            summary_text = await self._collect_summary_completion(
                resolution, summarize_messages
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 -- failure = no-op + honest copy
            logger.opt(exception=True).warning(
                "Console summarize-up-to failed", error=str(error)
            )
            visible_copy = "Couldn't summarize the conversation. Try again."
            self._set_run_state(
                ConsoleRunState(ConsoleRunStatus.FAILED, visible_copy),
                session_id=session_id,
            )
            return ConsoleSubmitResult(False, False, visible_copy)

        if not summary_text.strip():
            return self._summarize_block(
                session_id, "The model returned an empty summary."
            )

        self.store.set_session_context_summary(session_id, summary_text, message_id)
        turns = sum(1 for m in before if m.role is ConsoleMessageRole.USER)
        visible_copy = f"Summarized {turns} earlier turn{'s' if turns != 1 else ''}."
        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.COMPLETED, visible_copy),
            session_id=session_id,
        )
        return ConsoleSubmitResult(True, False, visible_copy)

    def _summarize_block(
        self, session_id: str, visible_copy: str
    ) -> ConsoleSubmitResult:
        """Blocked-summarize result that mutates NO transcript state.

        Unlike ``_block`` (which appends a SYSTEM row), a blocked summarize
        must leave the transcript untouched -- the run-state copy alone carries
        the reason to the control surfaces (Phase B discipline).
        """
        self._set_run_state(
            ConsoleRunState.blocked(visible_copy), session_id=session_id
        )
        return ConsoleSubmitResult(False, False, visible_copy)

    def _build_summary_span_text(
        self,
        span: list[ConsoleChatMessage],
        prior_summary: str | None,
        *,
        model: str,
    ) -> str:
        """Build the plain-text transcript span fed to the summarizer.

        Emits ``User: ...`` / ``Assistant: ...`` lines, prepending a
        ``[Previous summary]`` block when rolling. If the assembled span blows
        past ``_SUMMARY_SPAN_TOKEN_BUDGET`` (counted with
        ``count_console_messages_tokens``), the OLDEST turns are dropped until
        it fits -- the newest detail and the prior summary are always kept.
        """

        def assemble(rows: list[ConsoleChatMessage]) -> str:
            lines = [
                f"{'User' if m.role is ConsoleMessageRole.USER else 'Assistant'}: {m.content}"
                for m in rows
            ]
            transcript_text = "\n".join(lines)
            if prior_summary:
                return (
                    f"[Previous summary]\n{prior_summary}\n\n{transcript_text}".rstrip()
                )
            return transcript_text

        rows = list(span)
        body = assemble(rows)
        while (
            len(rows) > 1
            and count_console_messages_tokens(
                [{"role": "user", "content": body}], model
            )
            > self._SUMMARY_SPAN_TOKEN_BUDGET
        ):
            rows = rows[1:]
            body = assemble(rows)
        return body

    async def _collect_summary_completion(
        self, resolution: Any, messages: list[dict[str, Any]]
    ) -> str:
        """Collect a NON-streaming completion via the gateway's streaming seam.

        The provider gateway protocol exposes only ``stream_chat``; there is no
        separate non-streaming completion method on the Console surface, so the
        summary is accumulated from its chunks WITHOUT appending to any
        transcript message (summarize never mutates the tree). Non-``str``
        yields (e.g. tool-call payloads, never requested here) are ignored.
        """
        chunks: list[str] = []
        async for chunk in self.provider_gateway.stream_chat(resolution, messages):
            if isinstance(chunk, str) and chunk:
                chunks.append(chunk)
        return "".join(chunks)

    async def edit_and_resend_message(
        self, message_id: str, new_content: str
    ) -> ConsoleSubmitResult:
        """Edit a USER message and resend it, forking a NEW sibling branch.

        Sibling counterpart to ``regenerate_message``, but the anchor is a
        USER message rather than an assistant one, and this creates TWO new
        nodes instead of one: a USER sibling of ``message_id`` (``store.
        create_sibling``, parented at the anchor's own parent, carrying the
        edited text) followed by an empty ASSISTANT node appended under it
        (``store.append_message``, which always parents at the current
        active leaf -- the freshly created sibling). The anchor
        (``message_id``) and any old tail beneath it (its prior assistant
        reply, and anything after it for a mid-conversation edit) are left
        untouched and simply drop off the active path -- still reachable via
        ``store.set_active_leaf``, never deleted.

        All validation/blocking checks (active run, message role/session
        ownership, non-blank content, provider readiness) AND every payload
        transform (skill substitution, chat dictionaries, world info) run
        BEFORE either new node is created, mirroring ``regenerate_message``'s
        "mutate last" discipline: a blocked or refused edit-and-resend must
        not leave a stray orphan sibling -- or an un-streamed, un-retryable
        ``"pending"`` assistant node -- forked into the tree. Unlike
        ``regenerate_message`` (whose anchor is still on the active path, so
        its payload can be read straight off the store), the edited text
        does not exist as a stored node yet, so ``provider_messages`` is
        built from the anchor's ancestors (``_provider_messages_for_session``
        with ``before_message_id=message_id``, which excludes the anchor and
        its subtree) plus a synthesized ``{"role": "user", "content":
        clean_content}`` dict standing in for the not-yet-created sibling.
        The transform pipeline operates purely on that ``list[dict]``
        payload and never needs the real nodes to exist, so a
        skill-substitution refusal aborts the turn via ``_block`` with
        nothing to clean up. Only once every transform has succeeded are
        ``new_user`` (``store.create_sibling``) and the empty ``assistant``
        node (``store.append_message``) actually created, and the stream is
        started against them.

        On stream FAILURE, the new assistant node becomes a ``failed`` node
        on the active path (retryable via ``retry_message``), rather than
        restoring the anchor's prior reply in place -- this is the intended
        node-model behavior, not a regression: the anchor is a completely
        separate node and was never touched.

        Args:
            message_id: Native id of the USER message being edited (the
                anchor whose ancestor chain -- read with
                ``before_message_id=message_id``, which excludes the anchor
                and its own subtree -- becomes the base for the new branch).
            new_content: The edited text to resend as the new sibling USER
                message.

        Returns:
            A ``ConsoleSubmitResult``. ``accepted`` is ``True`` once the new
            USER/ASSISTANT sibling pair has been created and streaming has
            started (whether the stream itself later completes or fails);
            ``False`` if any pre-mutation block gate (active run, message
            role, session ownership, off-active-path anchor, blank content,
            provider readiness, skill refusal) rejected the resend before
            either new node was created. ``visible_copy`` carries the
            block/refusal copy shown to the user when ``accepted`` is
            ``False`` (and the streamed/failure copy otherwise).
        """
        active_rejection = self._active_run_rejection()
        if active_rejection is not None:
            return active_rejection

        session_id = self.store.active_session_id
        if session_id is None:
            return ConsoleSubmitResult(False, False, "No active Console session.")
        message = self.store.get_message(message_id)
        if message.role is not ConsoleMessageRole.USER:
            return self._block(
                session_id, "Only your messages can be edited and re-sent."
            )
        if self.store.session_id_for_message(message_id) != session_id:
            visible_copy = "Open the original session before editing this message."
            self._set_run_state(
                ConsoleRunState.blocked(visible_copy), session_id=session_id
            )
            return ConsoleSubmitResult(False, False, visible_copy)
        if message_id not in self.store.active_path_message_ids(session_id):
            # Task 2 review fix (Qodo finding 2): `_provider_messages_for_session`
            # builds the resend payload by scanning the ACTIVE-PATH transcript
            # until `message_id` is seen. If the anchor is not on the active
            # path, that scan never breaks and the payload would be built from
            # the wrong branch entirely. Edit is only exposed on active-path
            # rows today, so this is currently unreachable from the UI -- but
            # guard it here too so the method is safe to call directly.
            return self._block(
                session_id,
                "Switch to that branch before editing and re-sending this message.",
            )

        clean_content, validation_error = self._validated_draft(new_content)
        if validation_error is not None:
            return self._block(session_id, validation_error)

        # task-573: the resend carries the anchor's attachments, so the same
        # vision gate a fresh send applies (see ``submit_draft``) must fire
        # here too -- BEFORE any node is created (mutate-last discipline).
        anchor_attachments = tuple(message.attachments)
        if any(a.data is not None for a in anchor_attachments):
            vision_model = self.model or self.configured_model
            block_reason = vision_block_reason(
                self.provider, vision_model, is_capable=is_vision_capable
            )
            if block_reason is not None:
                return self._block(session_id, block_reason)

        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.VALIDATING, "Validating provider."),
            session_id=session_id,
        )
        resolution = await self.provider_gateway.resolve_for_send(
            self._provider_selection()
        )
        if not getattr(resolution, "ready", False):
            visible_copy = self._blocked_visible_copy(
                getattr(resolution, "visible_copy", "")
            )
            return self._block(session_id, visible_copy)

        # Build + transform the payload BEFORE creating either new node
        # (task-2 review fix): the edited turn is synthesized as a
        # not-yet-stored ``ConsoleChatMessage`` standing in for the sibling,
        # so a skill-substitution refusal (or any other transform failure)
        # has nothing to clean up -- no orphan sibling, no stuck "pending"
        # assistant node. task-573: running ancestors + the synthesized turn
        # through ONE ``_provider_message_payloads`` pass gives the carried
        # attachments the same image-budget/vision/mime treatment as a fresh
        # send (newest-first reservation included), instead of a hand-rolled
        # text-only dict.
        ancestors: list[ConsoleChatMessage] = []
        for candidate in self.store.messages_for_session(session_id):
            if candidate.id == message_id:
                break
            ancestors.append(candidate)
        ancestors.append(
            ConsoleChatMessage(
                role=ConsoleMessageRole.USER,
                content=clean_content,
                attachments=anchor_attachments,
            )
        )
        provider_messages = self._leading_system_message() + (
            self._provider_message_payloads(
                ancestors, skip_failed=True, annotate_ids=True
            )
        )
        self._ensure_user_continuation_instruction(provider_messages)
        (
            provider_messages,
            refuse,
            skill_notes,
            skill_bindings,
            skill_bundle_block,
        ) = await self._apply_skill_substitution(provider_messages)
        if refuse is not None:
            return self._block(session_id, refuse)
        for note in skill_notes:
            # An embedded skipped-skill note is never an abort: append the
            # same system-row copy `_block` would, then let the turn proceed.
            self.store.append_message(
                session_id, role=ConsoleMessageRole.SYSTEM, content=note
            )
        provider_messages = await self._apply_chat_dictionaries(
            provider_messages, session_id
        )
        provider_messages = await self._apply_world_info(provider_messages, session_id)
        prefill = self._pinned_prefill_for_session(session_id)

        # Every transform succeeded: now (and only now) fork the edited USER
        # sibling and append the empty ASSISTANT node to stream into.
        active_path = self.store.active_path_message_ids(session_id)
        anchor_index = active_path.index(message_id)
        for replaced_message_id in active_path[anchor_index:]:
            self.clear_original_attempt(replaced_message_id)
        self.store.create_sibling(
            message_id,
            role=ConsoleMessageRole.USER,
            content=clean_content,
            persist=self.store.persistence is not None,
            attachments=anchor_attachments,
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
            variant_mode=False,
            prefill=prefill,
            skill_bindings=skill_bindings,
            skill_bundle_block=skill_bundle_block,
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
            # task-548: annotate native ids so the boundary-summary compaction
            # below can anchor by identity, exactly like the real dispatch path
            # (the keys are stripped again before the snapshot is returned).
            provider_messages = self._provider_messages_for_session(
                session_id, annotate_ids=True
            )

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
            provider_messages = await self._apply_chat_dictionaries(
                provider_messages, session_id
            )

            # task-548: mirror the dispatch choke point's boundary-summary
            # compaction so the preview matches what is actually sent when a
            # `/rewind` summary is active (pre-boundary turns replaced by the
            # summary folded into the leading system row). Applied after the
            # transforms, exactly like the send path; a payload without the
            # boundary row (or no stored summary) is untouched. The private
            # id-threading key is stripped immediately after, so it can never
            # appear in the preview rows.
            provider_messages = self._apply_context_summary_compaction(
                session_id, provider_messages
            )
            provider_messages = [
                {k: v for k, v in row.items() if k != NATIVE_MESSAGE_ID_KEY}
                for row in provider_messages
            ]

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
            provider_messages = self._replace_image_data_with_placeholders(
                provider_messages
            )

            # Gather native tool schemas and MCP note.
            tools_info = self._build_tools_info_for_snapshot()

            # Redact secrets before returning.
            redacted_messages = self._redact_secrets(provider_messages)
            # task-548: derive the duplicated `system` field from the payload's
            # own leading system row when present, so a folded boundary summary
            # shows there too (falling back to the bare session prompt when the
            # payload carries no system row).
            leading_system: list[dict[str, Any]] = (
                [provider_messages[0]]
                if provider_messages
                and provider_messages[0].get("role") == ConsoleMessageRole.SYSTEM.value
                else self._leading_system_message()
            )
            redacted_system = self._redact_secrets(leading_system)

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
            # prompt rather than an empty placeholder. A failure inside the
            # annotate->strip window leaves the private id-threading key on the
            # assembled rows, so strip it here too (Qodo, PR #860).
            degraded_messages = self._replace_image_data_with_placeholders(
                self._redact_secrets(
                    [
                        {k: v for k, v in row.items() if k != NATIVE_MESSAGE_ID_KEY}
                        for row in provider_messages
                    ]
                )
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
    def _replace_image_data_with_placeholders(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
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
                        part["image_url"] = _redact_image_url_value(
                            part.get("image_url")
                        )
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
        """Flag a draft that LOOKS like an unresolved leading `$name` skill mention.

        Cheap textual heuristic only (a leading `MENTION_SIGIL`) -- this
        preview path deliberately never calls `_apply_skill_substitution`
        (see the caller's comment), so it has no candidate snapshot to
        actually resolve the word against. Re-sigiled for the `$`-mention
        migration (Task 5): a leading ``/`` is now a registered slash
        command (``/skills``, ``/prompt``, ...), not a skill invocation, so
        it must NOT be annotated here. Embedded ``$name`` mentions
        elsewhere in the draft are intentionally not flagged -- this only
        covers the leading form, mirroring `_apply_skill_substitution`'s
        own "leading form tried first" precedence.

        Only STRING content is ever annotated. A multimodal (list-content)
        draft -- e.g. a text part plus an image attachment -- is left
        completely unchanged, even when its text part starts with a
        `$name` mention: `_apply_skill_substitution` early-returns on
        non-str content at send time (replacing list content outright would
        drop the attachments), so this preview never actually substitutes a
        multimodal draft's skill mention. Annotating it here would promise
        a substitution the send never performs -- a dishonest preview
        (Qodo fix 4, PR #801 review).
        """
        result = copy.deepcopy(messages)
        if not synthetic_turn_added or not result or result[-1].get("role") != "user":
            return result

        content = result[-1].get("content", "")
        annotation = (
            "[Skill command not resolved in preview; "
            "actual substitution happens at send time.]"
        )

        if isinstance(content, str) and content.lstrip().startswith(MENTION_SIGIL):
            result[-1]["content"] = f"{content}\n\n{annotation}"

        return result

    def _build_tools_info_for_snapshot(self) -> dict[str, Any]:
        """Return native tool schemas and preview notes for the snapshot."""
        tools: list[dict[str, Any]] = []
        if self._agent_bridge is not None:
            # Native tools only; live MCP catalog composition is out of scope.
            tools = self._agent_bridge.native_tool_schemas()
        mcp_note: str | None = None
        if self._mcp_provider:
            mcp_note = "MCP tools are configured but live catalog composition is not shown in this preview."
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

    _SECRET_REDACTION_KEYS = {
        "api_key",
        "apikey",
        "token",
        "password",
        "secret",
        "bearer",
    }
    _SECRET_REDACTION_KEYS_NORMALIZED = {
        k.replace("-", "").replace("_", "") for k in _SECRET_REDACTION_KEYS
    }
    _SECRET_REDACTION_PATTERN = re.compile(
        r"(?P<open_quote>[\"']?)"
        r"(?P<key>" + "|".join(re.escape(k) for k in _SECRET_REDACTION_KEYS) + r")"
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

            return ConsoleChatController._SECRET_REDACTION_PATTERN.sub(
                _replace_value, value
            )

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
    ) -> tuple[list[dict[str, Any]], str | None, tuple[str, ...], tuple[str, ...], str]:
        """Render-fresh the triggering turn's skill mention(s) at payload build time.

        Spec: "Invocation semantics" §5 (the substitution rule) -- one rule
        for fresh sends AND retry/regenerate/continue. Only the FINAL
        ``role == "user"`` message in ``provider_messages`` (the turn
        actually driving this send) is ever a substitution candidate; every
        earlier message -- including an earlier raw skill mention sitting
        in history -- is left untouched, so the persisted transcript always
        keeps the literal text the user typed (the raw mention is what gets
        submitted and stored; only the ephemeral provider payload for this
        turn is ever rendered). Re-resolves against a FRESH candidate
        snapshot and re-verifies trust through ``execute_skill`` on every
        call (never a cached snapshot), so a retry issued after a skill was
        edited (now untrusted) refuses/skips instead of silently re-running
        a stale render.

        Both forms are DETECTED against trusted candidates UNION
        user-invocable blocked (needs-review) skills -- a blocked skill must
        still be found (leading refuses, embedded degrades to literal +
        note) rather than silently staying plain, sigil-prefixed text with
        no signal at all. `execute_skill` remains the sole trust authority;
        detection here never grants execution.

        Two independent forms, tried in order:

        Leading form -- the message, with leading whitespace stripped
        (mirroring `_annotate_skill_commands`'s own preview `lstrip()`
        assumption -- a resolved leading mention replaces the whole message
        either way, so the leading whitespace simply disappears), starts
        with `MENTION_SIGIL` and the leading word resolves to a known
        skill: the REST of the (stripped) message is passed as that skill's
        args (`cap_skill_args`). A resolved leading mention is never also
        scanned for embedded mentions -- its args are opaque payload, not
        further mentions to expand.

        Embedded form -- tried whenever the leading form doesn't apply (no
        leading `MENTION_SIGIL`, or the leading word doesn't resolve):
        scans the ORIGINAL (unstripped) message. Every `$skill-name`
        mention anywhere in the message (case-sensitive, code-span-masked,
        document order -- `find_embedded_mentions`) is looked up ARGLESS
        (`execute_skill(name, mode="local", args="")`, once per unique
        name, right-to-left splice so earlier spans stay valid) and spliced
        in place at the mention's exact span, preserving all surrounding
        prose. Only an ``execution_mode == "inline"`` result splices;
        anything else (e.g. ``fork``, which has no "in place" meaning for
        an embedded mention) silently leaves that mention's literal `$name`
        text untouched. A trust-blocked mention (`SkillTrustBlockedError`)
        also leaves the literal text untouched but records a
        `SKILL_MENTION_SKIPPED_NOTE` for the caller to surface as a
        non-aborting system row.

        Args:
            provider_messages: The fully-built payload about to be sent to
                the provider (already includes any leading session-system
                message and any synthesized continuation instruction).

        Returns:
            A 5-tuple ``(provider_messages, refuse, notes, skill_bindings,
            skill_bundle_block)`` (Task 5, skills-fork-reachability).
            ``skill_bindings`` is the leading-RESOLVED skill's name (both
            ``inline`` and ``fork`` outcomes -- never on refuse) plus every
            embedded mention name that actually SPLICED (never a
            trust-blocked-literal or fork-literal mention).
            ``skill_bundle_block`` is the fully-rendered "Bundled files"
            block (`_render_skill_bundle_block`) for every bound skill
            whose `execute_skill` result carried non-empty
            `reference_files`, built as pure string work from the results
            already in hand this call (no re-execution, no extra service
            calls), or ``""`` when nothing bound has any. It is NEVER
            inserted into ``provider_messages`` here -- only ``run_reply``
            (bridge-side) ever appends it, so plain sends and the stored
            transcript never see it.

            ``(provider_messages, None, (), (), "")`` unchanged when there
            is no skills service configured, substitution is disabled,
            there is no final user message, that message's content isn't a
            string, or neither form applies. ``(new_messages, None, notes,
            skill_bindings, skill_bundle_block)`` when the leading form
            resolves and renders (``notes`` always empty for the leading
            form) or when the embedded pass splices one or more mentions
            (``notes`` carries one `SKILL_MENTION_SKIPPED_NOTE` per unique
            trust-blocked mention name, in document order); ``inline``
            replaces just the final message in place (history preserved);
            leading-form ``fork`` drops every message before it except a
            leading ``role == "system"`` message (clean context = session
            system prompt + rendered turn only).
            ``(provider_messages, refuse_copy, (), (), "")`` -- the
            ORIGINAL, unmodified messages, paired with
            `SKILL_UNTRUSTED_REFUSE` copy -- when a LEADING resolved skill
            is no longer trusted (`SkillTrustBlockedError` at
            execute-time); the caller must append `refuse_copy` as a
            system row and abort the turn without sending. An embedded
            mention never refuses/aborts -- it degrades to a
            literal-plus-note instead, and the send proceeds.
        """
        if self._skills_service is None or not self._skill_substitution_enabled:
            return provider_messages, None, (), (), ""

        final_index: int | None = None
        for index in range(len(provider_messages) - 1, -1, -1):
            if provider_messages[index].get("role") == ConsoleMessageRole.USER.value:
                final_index = index
                break
        if final_index is None:
            return provider_messages, None, (), (), ""

        content = provider_messages[final_index].get("content")
        if not isinstance(content, str):
            return provider_messages, None, (), (), ""
        if MENTION_SIGIL not in content:
            # Fast path: no sigil anywhere means neither form can possibly
            # apply -- plain-text sends never touch the skills service.
            return provider_messages, None, (), (), ""

        context = await self._skills_service.get_context(mode="local")
        candidates = self._skill_candidates_from_context(context)
        # DETECTION population = trusted candidates UNION user-invocable
        # blocked (needs-review) skills. A blocked skill must still be
        # DETECTED -- leading refuses, embedded degrades to literal + note
        # -- rather than silently staying plain, sigil-prefixed text with no
        # signal at all. `execute_skill` (not this resolution step) remains
        # the sole authority on whether a resolved name may actually run:
        # a name that resolves here to a blocked skill hits
        # `SkillTrustBlockedError` at the `execute_skill` call below/in the
        # embedded loop, which already drives the refuse/skip-with-note
        # paths.
        detection_candidates = candidates + self._skill_blocked_candidates_from_context(
            context
        )

        # --- Leading form: message starts with a resolvable $skill-name.
        # Leading whitespace is tolerated (stripped before the sigil check
        # and the word/rest split) to match `_annotate_skill_commands`'s own
        # `lstrip()` assumption in the preview -- a resolved leading mention
        # replaces the ENTIRE message on both the inline-replace and fork
        # paths, so the leading whitespace simply disappears either way.
        stripped_content = content.lstrip()
        if stripped_content.startswith(MENTION_SIGIL):
            word, rest = _split_skill_command_word(stripped_content)
            name = word[len(MENTION_SIGIL) :]
            if name:
                resolution = resolve_skill_command(name, rest, detection_candidates)
                if resolution.kind == "resolved":
                    args = cap_skill_args(rest)
                    try:
                        result = await self._skills_service.execute_skill(
                            resolution.name, mode="local", args=args
                        )
                    except SkillTrustBlockedError as exc:
                        refuse = SKILL_UNTRUSTED_REFUSE.format(
                            name=resolution.name, reason=exc.reason_code
                        )
                        return provider_messages, refuse, (), (), ""

                    rendered = (
                        result.get("rendered_prompt", "")
                        if isinstance(result, Mapping)
                        else ""
                    )
                    rendered_message = {
                        "role": ConsoleMessageRole.USER.value,
                        "content": rendered,
                    }
                    execution_mode = (
                        result.get("execution_mode")
                        if isinstance(result, Mapping)
                        else None
                    )
                    # Task 5: a resolved leading mention always binds its
                    # name (fork AND inline outcomes -- never on refuse,
                    # which already returned above) and its block is
                    # rendered from this single execute_skill result.
                    bindings = (resolution.name,)
                    block = (
                        _render_skill_bundle_block([result])
                        if isinstance(result, Mapping)
                        else ""
                    )
                    if execution_mode == "fork":
                        leading = (
                            [provider_messages[0]]
                            if provider_messages
                            and provider_messages[0].get("role")
                            == ConsoleMessageRole.SYSTEM.value
                            else []
                        )
                        return leading + [rendered_message], None, (), bindings, block

                    new_messages = list(provider_messages)
                    new_messages[final_index] = {
                        **provider_messages[final_index],
                        "content": rendered,
                    }
                    return new_messages, None, (), bindings, block

        # --- Embedded pass: no leading mention, or the leading word didn't
        # resolve to a known skill. Scans the ORIGINAL (unstripped) content
        # -- the leading-whitespace tolerance above only applies to the
        # leading form. `names` is the same detection population (trusted
        # UNION user-invocable blocked) so a blocked mention is found and
        # routed through the trust-blocked-note path below instead of
        # staying invisible.
        names = frozenset(candidate.name for candidate in detection_candidates)
        mentions = find_embedded_mentions(content, names)
        if not mentions:
            return provider_messages, None, (), (), ""

        rendered_by_name: dict[str, str | None] = {}
        # Task 5: results_by_name only keeps a name's execute_skill result
        # when that mention actually SPLICED (execution_mode == "inline")
        # -- a blocked-literal or fork-literal mention's result is
        # discarded here, so it can never leak into skill_bindings or the
        # rendered bundle block below.
        results_by_name: dict[str, Mapping[str, Any]] = {}
        notes: list[str] = []
        for mention in mentions:
            if mention.name in rendered_by_name:
                continue
            try:
                result = await self._skills_service.execute_skill(
                    mention.name, mode="local", args=""
                )
            except SkillTrustBlockedError:
                rendered_by_name[mention.name] = None
                notes.append(SKILL_MENTION_SKIPPED_NOTE.format(name=mention.name))
                continue
            execution_mode = (
                result.get("execution_mode") if isinstance(result, Mapping) else None
            )
            rendered = (
                result.get("rendered_prompt", "") if isinstance(result, Mapping) else ""
            )
            # Fork (or anything non-inline) cannot splice in place: leave
            # the mention literal, no note (this is not a trust failure).
            rendered_by_name[mention.name] = (
                rendered if execution_mode == "inline" else None
            )
            if execution_mode == "inline" and isinstance(result, Mapping):
                results_by_name[mention.name] = result

        new_content = content
        for mention in reversed(mentions):
            body = rendered_by_name.get(mention.name)
            if body is None:
                continue
            new_content = (
                new_content[: mention.start] + body + new_content[mention.end :]
            )
        if new_content == content:
            return provider_messages, None, tuple(notes), (), ""

        # Task 5: bound names are every unique mention that actually
        # spliced, in first-occurrence document order (`dict.fromkeys` on
        # `mentions` dedups while preserving order) -- never a
        # blocked-literal or fork-literal mention, which never reached
        # `results_by_name`.
        spliced_names = tuple(
            name
            for name in dict.fromkeys(mention.name for mention in mentions)
            if rendered_by_name.get(name) is not None
        )
        block = _render_skill_bundle_block(
            results_by_name[name] for name in spliced_names if name in results_by_name
        )
        new_messages = list(provider_messages)
        new_messages[final_index] = {
            **provider_messages[final_index],
            "content": new_content,
        }
        return new_messages, None, tuple(notes), spliced_names, block

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
    def _skill_blocked_candidates_from_context(
        context: Any,
    ) -> tuple[SkillCommandCandidate, ...]:
        """Build the user-invocable, trust-BLOCKED (needs-review) skill
        candidate population.

        Companion to `_skill_candidates_from_context`: unioned with it in
        `_apply_skill_substitution` to widen the DETECTION population (never
        the executable one) so a `$blocked-name` mention resolves a name
        instead of silently staying literal, sigil-prefixed text with no
        refusal or note at all. `execute_skill` remains the sole authority
        on whether a resolved name may actually run -- candidates built here
        are never executed directly by this method's caller. A blocked
        skill flagged ``user_invocable: False`` is excluded, mirroring
        `_skill_candidates_from_context`'s own filter discipline.
        """
        blocked = (
            context.get("blocked_skills") if isinstance(context, Mapping) else None
        )
        return tuple(
            SkillCommandCandidate(
                name=str(item.get("name")),
                description=str(item.get("description") or ""),
            )
            for item in (blocked or [])
            if isinstance(item, Mapping)
            and item.get("name")
            and item.get("user_invocable", True)
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
        self._set_run_state(
            ConsoleRunState.blocked(visible_copy), session_id=session_id
        )
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

    async def _capture_rag_context(
        self,
        draft: str,
    ) -> tuple[
        str | None,
        CitationTraceBuilder | None,
        str | None,
        CitationRepairContract | None,
    ]:
        """Resolve optional staged RAG context without exposing request state."""

        provider = self._rag_capture_provider
        if provider is None:
            return None, None, None, None
        try:
            captured = await provider(draft)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error(
                "Console RAG capture unavailable; "
                f"reason=capture_provider_failure; draft_length={len(draft)}"
            )
            return None, None, None, None
        captured_context = getattr(captured, "context", None)
        context = (
            captured_context
            if isinstance(captured_context, str) and captured_context.strip()
            else None
        )
        captured_builder = getattr(captured, "citation_builder", None)
        builder = (
            captured_builder
            if isinstance(captured_builder, CitationTraceBuilder)
            else None
        )
        captured_prompt_id = getattr(captured, "prompt_evidence_set_id", None)
        prompt_evidence_set_id = (
            captured_prompt_id
            if isinstance(captured_prompt_id, str) and captured_prompt_id.strip()
            else None
        )
        captured_repair_contract = getattr(
            captured,
            "citation_repair_contract",
            None,
        )
        repair_contract = (
            captured_repair_contract
            if isinstance(captured_repair_contract, CitationRepairContract)
            and context is not None
            and captured_repair_contract.evidence_context == context
            else None
        )
        return context, builder, prompt_evidence_set_id, repair_contract

    @staticmethod
    def _build_terminal_citation_finalizer(
        *,
        context: str | None,
        builder: CitationTraceBuilder | None,
        prompt_evidence_set_id: str | None,
    ) -> TerminalCitationFinalizer | None:
        """Build exact-body citation finalization for one eligible initial send."""

        if (
            not isinstance(context, str)
            or not context.strip()
            or not isinstance(builder, CitationTraceBuilder)
            or not isinstance(prompt_evidence_set_id, str)
            or not prompt_evidence_set_id.strip()
        ):
            return None

        def finalize(answer_body: str) -> SealedCitationWrite | None:
            terminal_at = datetime.now(UTC)
            try:
                attempt_id = builder.record_initial_answer_attempt(
                    prompt_evidence_set_id=prompt_evidence_set_id,
                    answer_body=answer_body,
                    completed_at=terminal_at,
                )
                return builder.seal(
                    selected_attempt_id=attempt_id,
                    sealed_at=terminal_at,
                )
            except CitationTraceBuildUnavailable:
                logger.warning(
                    "Console citation finalization unavailable; "
                    "reason=occurrence_mapping_unavailable"
                )
            except Exception:
                logger.warning(
                    "Console citation finalization unavailable; "
                    "reason=attempt_or_seal_failure"
                )
            return None

        return finalize

    @staticmethod
    def _prepend_evidence_context(
        provider_messages: list[dict[str, Any]],
        context: str,
    ) -> list[dict[str, Any]]:
        """Prefix exact evidence to the final provider-only user message."""

        final_index = next(
            (
                index
                for index in range(len(provider_messages) - 1, -1, -1)
                if provider_messages[index].get("role") == ConsoleMessageRole.USER.value
            ),
            None,
        )
        if final_index is None:
            return provider_messages
        prefix = f"Evidence: {context}\n\n---\n\n"
        message = provider_messages[final_index]
        content = message.get("content")
        if isinstance(content, str):
            new_content: Any = prefix + content
        elif isinstance(content, list):
            new_content = list(content)
            text_index = next(
                (
                    index
                    for index, part in enumerate(new_content)
                    if isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                ),
                None,
            )
            if text_index is None:
                new_content.insert(0, {"type": "text", "text": prefix})
            else:
                text_part = new_content[text_index]
                new_content[text_index] = {
                    **text_part,
                    "text": prefix + text_part["text"],
                }
        else:
            return provider_messages
        updated = list(provider_messages)
        updated[final_index] = {**message, "content": new_content}
        return updated

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

    def _append_history_trimmed_note(self, session_id: str, dropped: int) -> None:
        """Append a transcript-only system row noting history was trimmed."""
        try:
            self.store.append_message(
                session_id,
                role=ConsoleMessageRole.SYSTEM,
                content=(
                    "Earlier messages were trimmed to fit the model's context "
                    f"window ({dropped} dropped)."
                ),
            )
        except KeyError:
            # Session vanished mid-send; the dispatched payload was still bounded.
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
        skill_bindings: tuple[str, ...] = (),
        skill_bundle_block: str = "",
        citation_repair_session: ConsoleCitationRepairSession | None = None,
    ) -> ConsoleSubmitResult:
        try:
            return await self._stream_assistant_response_inner(
                resolution=resolution,
                provider_messages=provider_messages,
                assistant_message_id=assistant_message_id,
                prepare_retry=prepare_retry,
                variant_mode=variant_mode,
                prefill=prefill,
                prefill_from_one_shot=prefill_from_one_shot,
                skill_bindings=skill_bindings,
                skill_bundle_block=skill_bundle_block,
                citation_repair_session=citation_repair_session,
            )
        finally:
            if citation_repair_session is not None:
                citation_repair_session.clear_governed_state()

    async def _stream_assistant_response_inner(
        self,
        *,
        resolution: Any,
        provider_messages: list[dict[str, str]],
        assistant_message_id: str,
        prepare_retry: bool = False,
        variant_mode: bool = False,
        prefill: str | None = None,
        prefill_from_one_shot: bool = False,
        skill_bindings: tuple[str, ...] = (),
        skill_bundle_block: str = "",
        citation_repair_session: ConsoleCitationRepairSession | None = None,
    ) -> ConsoleSubmitResult:
        try:
            owner_id = self.store.session_id_for_message(assistant_message_id)
        except KeyError:
            # The message itself is already gone -- no owning session to
            # attribute this to; default (active session) is a harmless
            # no-op since nothing will ever read a closed session's state.
            return self._session_closed_result()
        owner = next((s for s in self.store.sessions() if s.id == owner_id), None)
        # task-427: a character session always takes the plain-provider
        # path, even with the global agent runtime enabled and a bridge
        # present. Keyed on the message's OWNING session (looked up here,
        # not the controller's active session) so a session switch racing
        # this send can't flip which branch a still-in-flight message uses.
        force_plain = owner is not None and owner.character_id is not None
        # SP2 /rewind "summarize up to here": at the SINGLE dispatch choke point
        # (agent + direct both flow through here), fold the session's boundary
        # summary into the payload -- but ONLY when the boundary message is
        # actually present in it (the leak rule; see
        # _apply_context_summary_compaction). Runs BEFORE bound_messages_to_
        # window so the summary lands in the leading system prefix the trimmer
        # preserves.
        provider_messages = self._apply_context_summary_compaction(
            owner_id, provider_messages
        )
        # task-322: bound the dispatched history by real tokens before the
        # agent-vs-direct branch below, so both paths send a windowed payload.
        # Budget against the captured `resolution` -- the same model/provider/
        # max_tokens the dispatch below actually sends -- not the controller's
        # mutable self.* fields, which a provider/model switch racing the awaits
        # between resolve_for_send and here could have changed underneath us.
        bound = bound_messages_to_window(
            provider_messages,
            model=getattr(resolution, "model", None) or "",
            provider=getattr(resolution, "provider", "") or "",
            response_reservation=(
                getattr(resolution, "max_tokens", None) or DEFAULT_RESPONSE_RESERVATION
            ),
        )
        provider_messages = bound.messages
        # Strip the private id-threading key from every row before dispatch:
        # it existed solely so the compaction above could anchor the boundary
        # by identity (see NATIVE_MESSAGE_ID_KEY). This is the single latest
        # point covering BOTH the direct stream path (`stream_chat` below) and
        # the agent path (`agent_messages = list(provider_messages)` in
        # `_run_agent_reply`), so no provider/gateway/agent ever sees the key.
        # Rebuild fresh row dicts rather than mutating in place, since transforms
        # can leave earlier rows aliased to freshly-built builder dicts.
        provider_messages = [
            {k: v for k, v in row.items() if k != NATIVE_MESSAGE_ID_KEY}
            for row in provider_messages
        ]
        if bound.dropped_count:
            # Reuse the guarded owner_id resolved above; the note helper
            # swallows a store-close race that happens during the append.
            self._append_history_trimmed_note(owner_id, bound.dropped_count)
        active_task = asyncio.current_task()
        self._active_assistant_message_ids[owner_id] = assistant_message_id
        self._active_stream_tasks[owner_id] = active_task
        self._stop_requested = False
        self._active_citation_repair_sessions[owner_id] = citation_repair_session
        stream_signals = (
            ConsoleProviderStreamSignals()
            if citation_repair_session is not None
            else None
        )
        try:
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
                    skill_bindings=skill_bindings,
                    skill_bundle_block=skill_bundle_block,
                    citation_repair_session=citation_repair_session,
                    stream_signals=stream_signals,
                )
            return await self._run_direct_provider_reply(
                resolution=resolution,
                provider_messages=provider_messages,
                assistant_message_id=assistant_message_id,
                prepare_retry=prepare_retry,
                variant_mode=variant_mode,
                prefill=prefill,
                prefill_from_one_shot=prefill_from_one_shot,
                citation_repair_session=citation_repair_session,
                stream_signals=stream_signals,
            )
        finally:
            if (
                self._active_stream_tasks.get(owner_id) is active_task
                and self._active_assistant_message_ids.get(owner_id)
                == assistant_message_id
            ):
                self._active_stream_tasks.pop(owner_id, None)
                self._active_assistant_message_ids.pop(owner_id, None)
                self._stop_requested = False
                if (
                    self._active_citation_repair_sessions.get(owner_id)
                    is citation_repair_session
                ):
                    self._active_citation_repair_sessions.pop(owner_id, None)
                # Task 3b (agent path): `_run_agent_reply`'s own finally
                # deliberately leaves its cancel_event live past its own
                # return (see that finally's docstring) so the citation-
                # repair post-generation check -- which runs afterward, on
                # this same task, via `_finalize_agent_reply` -- still
                # observes it. This is the one place left to retire it, now
                # that the whole run (agent OR direct) has fully finished.
                # A no-op for the direct path, whose own finally already
                # popped its own cancel_event before returning.
                self._active_cancel_events.pop(owner_id, None)

    async def _run_direct_provider_reply(
        self,
        *,
        resolution: Any,
        provider_messages: list[dict[str, Any]],
        assistant_message_id: str,
        prepare_retry: bool,
        variant_mode: bool,
        prefill: str | None,
        prefill_from_one_shot: bool,
        citation_repair_session: ConsoleCitationRepairSession | None,
        stream_signals: ConsoleProviderStreamSignals | None,
    ) -> ConsoleSubmitResult:
        # Dev's citation-repair refactor extracted this streaming body out of
        # the wrapper (`_stream_assistant_response_inner`) into its own
        # method, which left it without the `owner_id` the wrapper already
        # resolved for ITSELF -- re-resolve independently here, same as
        # `_run_agent_reply` does for its own `session_id`, rather than
        # threading it through as a parameter (`None` on KeyError mirrors
        # every other guarded call site below: no owning session to
        # attribute a closed-session result to).
        try:
            owner_id = self.store.session_id_for_message(assistant_message_id)
        except KeyError:
            return self._session_closed_result()
        one_shot_used = prefill if prefill_from_one_shot else None
        if prefill:
            provider_messages = [
                *provider_messages,
                {
                    "role": ConsoleMessageRole.ASSISTANT.value,
                    "content": prefill,
                },
            ]
        # Fix round 1 (Critical 1): a per-session cancel signal for this
        # direct/legacy stream path too, mirroring `_run_agent_reply`'s own
        # `cancel_event` -- the shared `_stop_requested` flag below is
        # GLOBAL (set by ANY session's Stop/Close via `_signal_stop`), so
        # reading it inside a specific run's own loop let stopping session
        # B silently truncate an untouched session A's still-streaming
        # reply. Captured by closure/local (not re-read off `self.
        # _active_cancel_events` each poll) for the same reason
        # `should_cancel` isn't: a concurrent NEXT run for this same
        # session (after this one's own finally already popped its entry)
        # must never be torn down by a stale reference to THIS run's event.
        cancel_event = threading.Event()
        self._active_cancel_events[owner_id] = cancel_event
        if variant_mode:
            self.store.begin_variant_stream(assistant_message_id)
        if prefill and not prepare_retry:
            try:
                self.store.append_stream_chunk(assistant_message_id, prefill)
            except KeyError:
                return self._session_closed_result(session_id=owner_id)
        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "Streaming response."),
            session_id=owner_id,
        )
        retry_prepared = False
        emitted_content = False
        try:
            if stream_signals is None:
                provider_stream = self.provider_gateway.stream_chat(
                    resolution,
                    provider_messages,
                )
            else:
                provider_stream = self.provider_gateway.stream_chat(
                    resolution,
                    provider_messages,
                    signals=stream_signals,
                )
            async for chunk in provider_stream:
                if not chunk:
                    continue
                if cancel_event.is_set():
                    try:
                        stopped = self._mark_stream_stopped(
                            assistant_message_id,
                            visible_copy="Response stopped.",
                            prepare_retry=prepare_retry,
                            retry_prepared=retry_prepared,
                        )
                    except KeyError:
                        return self._session_closed_result(session_id=owner_id)
                    self._consume_one_shot_prefill(assistant_message_id, one_shot_used)
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
                            return self._session_closed_result(session_id=owner_id)
                try:
                    self.store.append_stream_chunk(assistant_message_id, chunk)
                except KeyError:
                    return self._session_closed_result(session_id=owner_id)
                if chunk:
                    emitted_content = True
            if cancel_event.is_set():
                try:
                    stopped = self._mark_stream_stopped(
                        assistant_message_id,
                        visible_copy="Response stopped.",
                        prepare_retry=prepare_retry,
                        retry_prepared=retry_prepared,
                    )
                except KeyError:
                    return self._session_closed_result(session_id=owner_id)
                self._consume_one_shot_prefill(assistant_message_id, one_shot_used)
                return ConsoleSubmitResult(True, True, stopped.content)
            if not emitted_content:
                try:
                    failed = self.store.get_message(assistant_message_id)
                except KeyError:
                    return self._session_closed_result(session_id=owner_id)
                self._set_run_state(
                    ConsoleRunState(
                        ConsoleRunStatus.FAILED,
                        "Provider stream ended without content.",
                    ),
                    session_id=owner_id,
                )
                if not prepare_retry:
                    try:
                        failed = self.store.mark_message_failed(assistant_message_id)
                    except KeyError:
                        return self._session_closed_result(session_id=owner_id)
                return ConsoleSubmitResult(True, True, failed.content)
            if citation_repair_session is not None and stream_signals is not None:
                try:
                    selection = await self._select_post_generation_body(
                        assistant_message_id=assistant_message_id,
                        repair_session=citation_repair_session,
                        stream_signals=stream_signals,
                    )
                except KeyError:
                    # F4 fix (Qodo wave): `owner_id` was already resolved
                    # above (line ~4290) and is in scope here, same as
                    # every other guarded call site in this method -- the
                    # bare no-arg call defaulted to whatever session is
                    # ACTIVE right now, wrongly stamping a STOPPED run
                    # state on an unrelated live session instead of this
                    # run's own (now-orphaned) one.
                    return self._session_closed_result(session_id=owner_id)
                if selection.state == "canceled":
                    self._consume_one_shot_prefill(
                        assistant_message_id,
                        one_shot_used,
                    )
                    return ConsoleSubmitResult(True, True, selection.selected_body)
            try:
                if variant_mode:
                    completed = self.store.finalize_variant_stream(assistant_message_id)
                else:
                    completed = self.store.mark_message_complete(assistant_message_id)
            except KeyError:
                return self._session_closed_result(session_id=owner_id)
            self._set_run_state(
                ConsoleRunState(ConsoleRunStatus.COMPLETED, "Response complete."),
                session_id=owner_id,
            )
            self._consume_one_shot_prefill(assistant_message_id, one_shot_used)
            return ConsoleSubmitResult(True, True, completed.content)
        except asyncio.CancelledError:
            if cancel_event.is_set():
                try:
                    stopped = self._mark_stream_stopped(
                        assistant_message_id,
                        visible_copy="Response stopped.",
                        prepare_retry=prepare_retry,
                        retry_prepared=retry_prepared,
                    )
                except KeyError:
                    return self._session_closed_result(session_id=owner_id)
                self._consume_one_shot_prefill(assistant_message_id, one_shot_used)
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
                return self._session_closed_result(session_id=owner_id)
            # Reuse the guarded owner_id resolved at the top of this method
            # (rather than re-deriving it) -- this is the run's OWNING
            # session regardless of whatever the user currently has open.
            self._append_failure_system_row(owner_id, visible_copy)
            self._set_run_state(
                ConsoleRunState(ConsoleRunStatus.FAILED, visible_copy),
                session_id=owner_id,
            )
            return ConsoleSubmitResult(True, True, visible_copy)
        finally:
            # Fix round 1 (Critical 1): this run's own per-session cancel
            # signal (created above, mirroring `_run_agent_reply`'s own)
            # must not survive the run -- a stale entry would let a LATER,
            # unrelated run on this same session inherit an already-set
            # Event. Not `cancel_event.clear()`: `_select_post_generation_
            # body` (already returned by the time this fires) captured this
            # by session-id lookup rather than closure, so identity-gated
            # pop (not reset-in-place) is what matches `_run_agent_reply`'s
            # own matching pop.
            if self._active_cancel_events.get(owner_id) is cancel_event:
                self._active_cancel_events.pop(owner_id, None)

    async def _select_post_generation_body(
        self,
        *,
        assistant_message_id: str,
        repair_session: ConsoleCitationRepairSession,
        stream_signals: ConsoleProviderStreamSignals,
    ) -> ConsoleCitationSelectionOutcome:
        """Select one bounded reply before terminal persistence."""

        try:
            initial_message = self.store.get_message(assistant_message_id)
            owner_session_id = self.store.session_id_for_message(assistant_message_id)
        except KeyError:
            return ConsoleCitationSelectionOutcome("", "unavailable")
        initial_body = initial_message.content

        def owns_request() -> bool:
            # Task 3b: check the OWNING session's own map entries, not a
            # global singular slot -- a concurrent session's own in-flight
            # stream/repair must never be mistaken for this one.
            if (
                self._active_assistant_message_ids.get(owner_session_id)
                != assistant_message_id
                or self._active_stream_tasks.get(owner_session_id)
                is not asyncio.current_task()
                or self._active_citation_repair_sessions.get(owner_session_id)
                is not repair_session
            ):
                return False
            try:
                return (
                    self.store.session_id_for_message(assistant_message_id)
                    == owner_session_id
                )
            except KeyError:
                return False

        def cancellation_requested() -> bool:
            # Fix round 1 (Critical 1): this run's own per-session cancel
            # signal, not the shared `_stop_requested` flag -- reading the
            # global flag here let an UNRELATED session's Stop/Close
            # silently cancel this session's still-running citation repair,
            # the exact hazard this fix closes for the sibling stream
            # loops. `_run_direct_provider_reply` registers this session's
            # `cancel_event` in `_active_cancel_events[owner_session_id]`
            # before ever calling this method.
            cancel_event = self._active_cancel_events.get(owner_session_id)
            return (
                repair_session.cancel_reason is not None
                and cancel_event is not None
                and cancel_event.is_set()
                and not repair_session.selection_committed
                and repair_session.phase in {"checking", "repair_streaming"}
            )

        def commit_canceled() -> ConsoleCitationSelectionOutcome:
            if not owns_request():
                visible_copy = (
                    "Session closed."
                    if repair_session.cancel_reason == "session_close"
                    else initial_body
                )
                return ConsoleCitationSelectionOutcome(
                    visible_copy,
                    "canceled",
                )
            try:
                current = self.store.get_message(assistant_message_id)
            except KeyError:
                return ConsoleCitationSelectionOutcome(
                    "Session closed.",
                    "canceled",
                )
            if current.content != initial_body:
                return ConsoleCitationSelectionOutcome(
                    current.content,
                    "canceled",
                )

            repair_session.phase = "selected"
            repair_session.selection_committed = True
            self.store.set_citation_presentation(
                assistant_message_id,
                ConsoleCitationPresentation(
                    phase=ConsoleCitationPhase.SELECTED,
                    notice_code=ConsoleCitationNoticeCode.CANCELED,
                    original_attempt_available=False,
                ),
            )
            completed = self.store.mark_message_complete(assistant_message_id)
            self._set_run_state(
                ConsoleRunState(
                    ConsoleRunStatus.STOPPED,
                    "Citation repair canceled.",
                ),
                session_id=owner_session_id,
            )
            if repair_session.cancel_reason == "user":
                try:
                    self.store.append_message(
                        owner_session_id,
                        role=ConsoleMessageRole.SYSTEM,
                        content="Citation repair canceled by user.",
                        persist=self.store.persistence is not None,
                    )
                except Exception:
                    logger.warning(
                        "Console citation repair cancellation record unavailable; "
                        "reason=citation_repair_cancel_record_persistence_failed"
                    )
            return ConsoleCitationSelectionOutcome(
                completed.content,
                "canceled",
            )

        def commit(
            state: Literal["valid", "repaired", "unavailable"],
            *,
            notice_code: ConsoleCitationNoticeCode | None = None,
            selected_body: str | None = None,
        ) -> ConsoleCitationSelectionOutcome:
            if cancellation_requested():
                return commit_canceled()
            if not owns_request():
                return ConsoleCitationSelectionOutcome(
                    initial_body,
                    "unavailable",
                )
            if selected_body is not None:
                try:
                    self.store.replace_deferred_terminal_body(
                        assistant_message_id,
                        selected_body,
                    )
                except ValueError:
                    state = "unavailable"
                    notice_code = ConsoleCitationNoticeCode.UNAVAILABLE
            repair_session.phase = "selected"
            repair_session.selection_committed = True
            self.store.set_citation_presentation(
                assistant_message_id,
                ConsoleCitationPresentation(
                    phase=ConsoleCitationPhase.SELECTED,
                    notice_code=notice_code,
                    original_attempt_available=state == "repaired",
                ),
            )
            if state == "repaired":
                self._remember_original_attempt(
                    assistant_message_id,
                    initial_body,
                    update_presentation=False,
                )
            selected = self.store.get_message(assistant_message_id)
            return ConsoleCitationSelectionOutcome(selected.content, state)

        if (
            not initial_body
            or stream_signals.synthetic_fallback_emitted
            or repair_session.selection_committed
        ):
            repair_session.phase = "selected"
            repair_session.selection_committed = True
            return ConsoleCitationSelectionOutcome(initial_body, "bypassed")

        contract = repair_session.contract
        resolution = repair_session.resolution
        if contract is None or resolution is None:
            return ConsoleCitationSelectionOutcome(initial_body, "unavailable")

        decision = decide_citation_repair(initial_body, contract)
        if decision is CitationRepairDecision.VALID:
            return commit("valid")
        if decision is CitationRepairDecision.UNAVAILABLE:
            return commit(
                "unavailable",
                notice_code=ConsoleCitationNoticeCode.UNAVAILABLE,
            )

        repair_session.phase = "checking"
        self.store.set_citation_presentation(
            assistant_message_id,
            ConsoleCitationPresentation(phase=ConsoleCitationPhase.CHECKING),
        )
        self._set_run_state(
            ConsoleRunState(
                ConsoleRunStatus.CHECKING_CITATIONS,
                "Checking citations…",
            ),
            session_id=owner_session_id,
        )
        repaired_chunks: list[str] = []
        repair_output_available = False
        try:
            await asyncio.sleep(0)
            if cancellation_requested():
                return commit_canceled()
            try:
                current_message = self.store.get_message(assistant_message_id)
            except KeyError:
                return ConsoleCitationSelectionOutcome(
                    "Session closed.",
                    "canceled",
                )
            if (
                not owns_request()
                or current_message.content != initial_body
                or repair_session.attempt_started
                or stream_signals.synthetic_fallback_emitted
            ):
                return commit(
                    "unavailable",
                    notice_code=ConsoleCitationNoticeCode.UNAVAILABLE,
                )

            repair_messages = build_citation_repair_messages(
                contract,
                initial_body,
            )
            if repair_messages is None or not repair_request_fits_model_window(
                repair_messages,
                initial_answer=initial_body,
                model=resolution.model or "",
                provider=resolution.provider,
                max_tokens=resolution.max_tokens,
            ):
                return commit(
                    "unavailable",
                    notice_code=ConsoleCitationNoticeCode.UNAVAILABLE,
                )
            if cancellation_requested():
                return commit_canceled()

            repair_session.attempt_started = True
            repair_session.phase = "repair_streaming"
            self.store.set_citation_presentation(
                assistant_message_id,
                ConsoleCitationPresentation(phase=ConsoleCitationPhase.REPAIRING),
            )

            repaired_size = 0
            repair_output_available = True
            async for chunk in self.provider_gateway.stream_chat(
                resolution,
                repair_messages,
                signals=stream_signals,
            ):
                if cancellation_requested():
                    repaired_chunks.clear()
                    return commit_canceled()
                if type(chunk) is not str:
                    repair_output_available = False
                    break
                if not chunk:
                    continue
                try:
                    repaired_size += len(chunk.encode("utf-8"))
                except UnicodeEncodeError:
                    repair_output_available = False
                    break
                if repaired_size > REPAIR_ANSWER_BODY_UTF8_BYTES_MAX:
                    repair_output_available = False
                    break
                repaired_chunks.append(chunk)
        except asyncio.CancelledError:
            if cancellation_requested():
                return commit_canceled()
            raise
        except Exception:
            repair_output_available = False

        if cancellation_requested():
            repaired_chunks.clear()
            return commit_canceled()
        if not repair_output_available or stream_signals.synthetic_fallback_emitted:
            return commit(
                "unavailable",
                notice_code=ConsoleCitationNoticeCode.UNAVAILABLE,
            )

        repaired_body = "".join(repaired_chunks)
        selected = select_repaired_body(
            initial_body,
            repaired_body,
            contract,
        )
        if not selected.repaired:
            return commit(
                "unavailable",
                notice_code=ConsoleCitationNoticeCode.UNAVAILABLE,
            )
        return commit(
            "repaired",
            notice_code=ConsoleCitationNoticeCode.REPAIRED,
            selected_body=selected.selected_body,
        )

    async def _run_agent_reply(
        self,
        *,
        resolution: Any,
        provider_messages: list[dict[str, Any]],
        assistant_message_id: str,
        prepare_retry: bool,
        variant_mode: bool,
        skill_bindings: tuple[str, ...] = (),
        skill_bundle_block: str = "",
        citation_repair_session: ConsoleCitationRepairSession | None = None,
        stream_signals: ConsoleProviderStreamSignals | None = None,
    ) -> ConsoleSubmitResult:
        """Run the agent loop as the reply engine, streaming into the target row."""
        logger.info(
            "console agent reply start",
            assistant_message_id=assistant_message_id,
            variant_mode=variant_mode,
            prepare_retry=prepare_retry,
        )
        # Resolve the run's OWNING session FIRST (Task 3b): every write
        # below -- the per-session stream/cancel maps AND run state -- must
        # target it explicitly rather than whatever the user currently has
        # open (parallel-agents spec §2). Moved ahead of those writes
        # (previously ran after them, back when they were single shared
        # slots with no session to key by).
        try:
            session_id = self.store.session_id_for_message(assistant_message_id)
        except KeyError:
            return self._session_closed_result()
        self._active_assistant_message_ids[session_id] = assistant_message_id
        self._active_stream_tasks[session_id] = asyncio.current_task()
        self._stop_requested = False
        self._mcp_provider = None
        # A fresh per-run Event, captured by `should_cancel` below by
        # closure (not read off `self` each time) -- see
        # `_active_cancel_events`'s docstring for why this, rather than
        # `_stop_requested` alone, is what makes a still-running bridge
        # thread observe a Stop correctly (task-227).
        cancel_event = threading.Event()
        self._active_cancel_events[session_id] = cancel_event
        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "Agent running."),
            session_id=session_id,
        )
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
        # noqa: E731 — tiny closure. Fix round 1 (Critical 1): reads ONLY
        # `cancel_event` -- captured by value, not via `self.
        # _active_cancel_events[session_id]` -- never the shared
        # `_stop_requested` flag. `_stop_requested` is GLOBAL (set by ANY
        # session's Stop/Close via `_signal_stop`), so OR'ing it in here
        # let stopping an unrelated session silently cancel THIS run too.
        # `cancel_event` alone is still correct once this run's own
        # `finally` below has already reset `_stop_requested` while the
        # bridge's background thread is still running (task-227: an
        # `asyncio.to_thread` call survives Task cancellation, so the
        # coroutine can finish handling a Stop and reset its own shared
        # bookkeeping well before the OS thread it detached from actually
        # returns) -- `stop_active_run`/`close_session`/`shutdown` all set
        # THIS session's `cancel_event` via `_signal_stop(session_id=...)`
        # the moment Stop is requested, and nothing ever clears it again
        # for this run, so a late poll from the surviving thread still
        # sees the cancellation regardless of `_stop_requested`'s state.
        should_cancel = lambda: cancel_event.is_set()  # noqa: E731

        # P5-T6: compose this run's MCP tool provider (if eligible) HERE,
        # on the running main loop, BEFORE the bridge is dispatched onto
        # asyncio.to_thread below -- see `_compose_mcp_provider`'s own
        # docstring for why `compose_catalog()`'s async I/O can never run
        # from the worker thread. `(None, None)` (no service, kill switch
        # on, or nothing composed) leaves the bridge's MCP-free path
        # byte-identical to before this task.
        #
        # task-545/T6: `_compose_mcp_provider`'s own `mcp_review_hook`
        # (built from `build_mcp_review_hook`) is deliberately discarded
        # here rather than wired -- it is `None` whenever MCP is not
        # eligible for this run, and built-in tools (calculator/datetime
        # today) must be gated regardless of MCP eligibility. Changing
        # `_compose_mcp_provider`'s own return contract to drop that
        # second element was considered and rejected: several existing
        # test suites (`Tests/Chat/test_console_agent_swap.py`,
        # `Tests/UI/test_console_internals_decomposition.py`) assert its
        # exact `(provider, hook)` / `(None, None)` shape directly and sit
        # outside this task's file scope, so keeping that function
        # byte-identical and building the run-level hook separately here
        # is the lower-blast-radius choice.
        mcp_provider, _unused_mcp_only_review_hook = await self._compose_mcp_provider(
            session_id
        )
        self._mcp_provider = mcp_provider

        # task-545/T6: build THIS run's built-in permission gate and hand
        # the SAME instance to both the review hook (below) and
        # `ConsoleAgentBridge.run_reply` (which threads it into the
        # `BuiltinToolProvider` that actually invokes tools) -- a second,
        # independently-built gate would silently desynchronize stamps:
        # a decision made here would never be visible to `invoke()`'s own
        # gate, and vice versa. `build_builtin_gate(None)` (no
        # `unified_mcp_service` on the app) is fail-closed-correct, not
        # "ungated" -- see that function's own docstring.
        builtin_gate = build_builtin_gate(
            getattr(self.app, "unified_mcp_service", None)
        )
        # Only `.tool_for(name)` is used by the review hook below, to
        # resolve a `ToolCall.name` to the `Tool` object `builtin_gate.
        # resolve` needs -- this instance is never used to invoke a tool,
        # so it does not need to be the SAME `BuiltinToolProvider` object
        # the bridge's registry actually dispatches through (its `_tools`
        # dict is stateless data rebuilt identically by any instance).
        builtin_review_provider = BuiltinToolProvider(gate=builtin_gate)
        # Round 1 review CRITICAL 1: resolve THIS run's OWN workspace id --
        # the SAME lookup `ConsoleAgentBridge.run_reply` makes
        # (`self._store.session_workspace_id(session_id)`) for the real
        # `BuiltinToolProvider(workspace_id=...)` dispatch below -- and
        # thread it into the review hook so its `path_precheck_failed`
        # pre-flight resolves the IDENTICAL workspace dispatch will, never
        # whatever happens to be active in the UI for a parked/background
        # session. `KeyError` (an already-closed session) degrades to
        # `None`, matching `allowed_file_roots`'s own fail-safe posture.
        try:
            review_workspace_id = self.store.session_workspace_id(session_id)
        except KeyError:
            review_workspace_id = None
        # Task 9: bind THIS run's owning session id into the approval
        # bridge so `request_mcp_approvals` can (a) scope its cancellation
        # check to this run's own cancel event rather than falling back to
        # whichever session is currently VIEWED (finding #1), and (b) park
        # rather than mount when `session_id` is not the active session.
        review_hook = build_tool_review_hook(
            builtin_gate,
            builtin_review_provider,
            mcp_provider,
            functools.partial(self.request_mcp_approvals, session_id=session_id),
            workspace_id=review_workspace_id,
        )

        # Swap site: the agent loop runs synchronously on a worker thread via
        # asyncio.to_thread, so Stop is cooperative-only -- `should_cancel` is
        # polled between chunks/steps inside the bridge, never preempts the
        # thread itself. A provider that hangs mid-request without emitting a
        # single chunk cannot be interrupted here; RunBudget.max_wall_seconds
        # (agent_models.py) is what bounds a run overall, but only once
        # control returns to a checkpoint the loop actually polls -- it is
        # not a hard timeout on an in-flight, zero-chunk provider call.
        try:
            # run_reply returns (run_id, outcome): run_id lets us write the
            # produced reply's PERSISTED id back onto the run after
            # completion (the load-bearing write for resume marker anchoring).
            run_id, outcome = await asyncio.to_thread(
                self._agent_bridge.run_reply,
                conversation_id=conversation_id,
                session_id=session_id,
                resolution=resolution,
                assistant_message_id=assistant_message_id,
                model=self.model or self.configured_model or "",
                session_system_prompt=session_system_prompt,
                agent_messages=agent_messages,
                should_cancel=should_cancel,
                provider_stream_signals=stream_signals,
                supersede_previous=bool(prepare_retry or variant_mode),
                mcp_provider=mcp_provider,
                builtin_gate=builtin_gate,
                review_tool_calls=review_hook,
                turn_skill_bindings=skill_bindings,
                turn_bundle_block=skill_bundle_block,
                request_skill_install_confirm=functools.partial(
                    self.request_skill_install_confirm, session_id=session_id
                ),
                # Advertised must equal usable (the #847 lesson, restated in
                # the run_skill_script docstring below): only pass the
                # confirm callback -- and therefore only let the bridge
                # build/advertise the run_skill_script tool at all -- once a
                # UI sink is actually wired. Until then
                # `request_skill_script_confirm`'s own no-UI guard would
                # auto-deny every call, offering the model a tool it can
                # never successfully use.
                request_skill_script_confirm=(
                    functools.partial(
                        self.request_skill_script_confirm, session_id=session_id
                    )
                    if self.set_pending_skill_script is not None
                    else None
                ),
            )
        except asyncio.CancelledError:
            if cancel_event.is_set():
                try:
                    stopped = self._mark_stream_stopped(
                        assistant_message_id, visible_copy="Response stopped."
                    )
                except KeyError:
                    return self._session_closed_result(session_id=session_id)
                # task-543: this is the dominant user-Stop path --
                # ``task.cancel()`` raised before ``(run_id, outcome)`` ever
                # bound, so recover the active run's id via the bridge's
                # latest-unanchored-primary lookup and record the stopped
                # reply's persisted id, same as every finalizer terminal
                # path. A never-persisted stop (or an anchored/missing row)
                # no-ops and leaves the row NULL -> ordinal fallback.
                self._record_run_assistant_message(
                    self._latest_unanchored_primary_run_id(conversation_id),
                    stopped,
                )
                return ConsoleSubmitResult(True, True, stopped.content)
            raise
        except Exception as exc:
            # Bridge failures can originate OUTSIDE AgentService's own
            # narrow loop guard (agent_service.py wraps only
            # `run_agent_loop`; `db.create_run`, `_persist`
            # (append_steps/set_status), and `supersede_run_tree` are not
            # covered). Left uncaught here, run_state would stay STREAMING
            # forever and every future send on every session would be
            # rejected ("A run is already running in this tab.") until app
            # restart (Plan-B Task 6 Critical 1). Mirror the legacy stream
            # path's catch-all above, including the Task-1 variant-restore
            # semantics: `begin_variant_stream`/`prepare_message_retry`
            # already ran before the bridge call, so `mark_message_failed`
            # resolves the correct terminal content on its own (restores
            # the pre-regenerate base + status for a failed regenerate;
            # preserves whatever partial content already streamed
            # otherwise).
            visible_copy = f"Agent run failed: {describe_stream_failure(exc)}"
            if getattr(
                getattr(exc, "response", None), "status_code", None
            ) is not None and self._session_history_carries_images(session_id):
                visible_copy += self._IMAGE_REJECTION_RECOVERY_HINT
            try:
                self.store.mark_message_failed(assistant_message_id)
            except KeyError:
                return self._session_closed_result(session_id=session_id)
            self._append_failure_system_row(session_id, visible_copy)
            self._set_run_state(
                ConsoleRunState(ConsoleRunStatus.FAILED, visible_copy),
                session_id=session_id,
            )
            return ConsoleSubmitResult(True, True, visible_copy)
        finally:
            # Task 3b: this finally intentionally pops NONE of the three
            # per-session entries (stream task, assistant message id,
            # cancel event) -- `_finalize_agent_reply` below (and, through
            # it, `_finalize_agent_success`'s citation-repair post-
            # generation check) still runs AFTER this try/finally, on this
            # SAME task, and both `owns_request()` (stream task/assistant
            # message id) and `cancellation_requested()` (cancel_event,
            # NOT clear()'d for the same reason noted where it was created
            # above -- task-227) need to see this run as still live and
            # still cancellable. The wrapper (`_stream_assistant_response_
            # inner`), which awaits this entire call including
            # `_finalize_agent_reply`, is what clears every per-session
            # entry once everything has actually finished.
            run_state = self.run_state_for(session_id)
            logger.info(
                "console agent reply end",
                assistant_message_id=assistant_message_id,
                run_status=run_state.status.value,
                run_copy=run_state.visible_copy,
            )

        # Captured here, before `_finalize_agent_reply` runs: this run's own
        # cancel_event is the authority on whether IT was stopped,
        # independent of what status `mark_message_stopped` may have left
        # the message at (task-227 AC3 follow-up -- see the guard below).
        return await self._finalize_agent_reply(
            assistant_message_id,
            session_id,
            outcome,
            variant_mode=variant_mode,
            cancel_event=cancel_event,
            run_id=run_id,
            citation_repair_session=citation_repair_session,
            stream_signals=stream_signals,
        )

    def _agent_conversation_id(self, session_id: str) -> str:
        """Return the durable id the run store is keyed by (persisted id when set)."""
        for session in self.store.sessions():
            if session.id == session_id:
                return session.persisted_conversation_id or session_id
        return session_id

    async def _finalize_agent_reply(
        self,
        assistant_message_id: str,
        session_id: str,
        outcome: Any,
        *,
        variant_mode: bool,
        cancel_event: threading.Event | None = None,
        run_id: str | None = None,
        citation_repair_session: ConsoleCitationRepairSession | None = None,
        stream_signals: ConsoleProviderStreamSignals | None = None,
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
        stopped_now = (current is not None and current.status == "stopped") or (
            cancel_event is not None and cancel_event.is_set()
        )
        if stopped_now:
            # The stopped message was already persisted by
            # `mark_message_stopped` (`_persist_existing_message`), so its
            # durable persisted id is available NOW -- record it onto the run
            # so resume can anchor markers by it. Without this the run keeps
            # whatever `create_run` stored (a stale native id pre-fix, NULL
            # post-fix); a never-persisted stop leaves `current` without a
            # persisted id and the helper no-ops (row stays NULL -> ordinal
            # fallback -- correct).
            self._record_run_assistant_message(run_id, current)
            self._set_run_state(
                ConsoleRunState(ConsoleRunStatus.STOPPED, "Response stopped."),
                session_id=session_id,
            )
            return ConsoleSubmitResult(
                True, True, current.content if current is not None else ""
            )

        if outcome.status == RUN_CANCELLED:
            return self._finalize_agent_cancelled(
                assistant_message_id,
                session_id,
                variant_mode=variant_mode,
                run_id=run_id,
            )

        if outcome.status != RUN_DONE:
            return self._finalize_agent_failure(
                assistant_message_id,
                session_id,
                outcome,
                variant_mode=variant_mode,
                run_id=run_id,
            )

        return await self._finalize_agent_success(
            assistant_message_id,
            session_id,
            outcome,
            variant_mode=variant_mode,
            run_id=run_id,
            citation_repair_session=citation_repair_session,
            stream_signals=stream_signals,
        )

    def _ensure_assistant_placeholder(
        self,
        assistant_message_id: str,
        session_id: str,
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
        self,
        session_id: str,
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
        self,
        assistant_message_id: str,
        variant_mode: bool,
        outcome: Any,
    ) -> ConsoleChatMessage:
        """Finalize a placeholder, applying the empty-final-text fallback.

        The fallback text is streamed into the placeholder so the store's
        existing persistence/validation paths stay unchanged.
        """
        if not getattr(outcome, "final_text", ""):
            self.store.clear_terminal_citation_state(assistant_message_id)
            self.store.append_stream_chunk(
                assistant_message_id,
                "No response was generated.",
            )
        if variant_mode:
            return self.store.finalize_variant_stream(assistant_message_id)
        return self.store.mark_message_complete(assistant_message_id)

    def _finalize_agent_cancelled(
        self,
        assistant_message_id: str,
        session_id: str,
        *,
        variant_mode: bool,
        run_id: str | None = None,
    ) -> ConsoleSubmitResult:
        """Handle a ``RUN_CANCELLED`` outcome: the placeholder becomes ``failed``.

        Per the agent turn-control spec, a runtime-reported cancellation is a
        terminal failure, not a user-initiated stop. If the placeholder has
        vanished, append a failed assistant message carrying the visible copy.
        The terminal message (``mark_message_failed``/``_append_failed_assistant``,
        both persisted) has its durable id recorded onto the run so resume can
        anchor markers by it; a never-persisted reply no-ops (row stays NULL ->
        ordinal fallback -- see ``_record_run_assistant_message``).
        """
        visible_copy = "Response stopped/cancelled."
        placeholder = self._ensure_assistant_placeholder(
            assistant_message_id, session_id
        )
        if placeholder is not None:
            failed = self.store.mark_message_failed(assistant_message_id)
        else:
            failed = self._append_failed_assistant(session_id, visible_copy)
        self._record_run_assistant_message(run_id, failed)
        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.FAILED, visible_copy),
            session_id=session_id,
        )
        return ConsoleSubmitResult(True, True, failed.content)

    def _finalize_agent_failure(
        self,
        assistant_message_id: str,
        session_id: str,
        outcome: Any,
        *,
        variant_mode: bool,
        run_id: str | None = None,
    ) -> ConsoleSubmitResult:
        """Handle ``RUN_ERROR``, ``RUN_STUCK``, or any unknown non-done outcome.

        A present placeholder is marked ``failed`` and a system row explains
        the failure (preserving the existing failure UX). If the placeholder
        is missing, the runtime may have already written an assistant message
        (e.g. streamed partial content before the error); use it when
        possible, otherwise append a new failed assistant message.

        Whichever terminal message resolves (all persisted via
        ``mark_message_failed``/``_append_failed_assistant``) has its durable id
        recorded onto the run so resume can anchor markers by it; a
        never-persisted reply no-ops (row stays NULL -> ordinal fallback -- see
        ``_record_run_assistant_message``).
        """
        visible_copy = self._agent_failure_visible_copy(outcome)
        if "provider returned HTTP" in visible_copy and (
            self._session_history_carries_images(session_id)
        ):
            visible_copy += self._IMAGE_REJECTION_RECOVERY_HINT
        placeholder = self._ensure_assistant_placeholder(
            assistant_message_id, session_id
        )
        if placeholder is not None:
            failed = self.store.mark_message_failed(assistant_message_id)
            self._record_run_assistant_message(run_id, failed)
            self._append_failure_system_row(session_id, visible_copy)
            self._set_run_state(
                ConsoleRunState(ConsoleRunStatus.FAILED, visible_copy),
                session_id=session_id,
            )
            return ConsoleSubmitResult(True, True, failed.content)

        runtime_written = self._find_runtime_written_assistant(session_id)
        if runtime_written is not None and runtime_written.status in {
            "pending",
            "streaming",
        }:
            self.store.append_stream_chunk(runtime_written.id, f"\n\n{visible_copy}")
            failed = self.store.mark_message_failed(runtime_written.id)
        else:
            failed = self._append_failed_assistant(session_id, visible_copy)
        self._record_run_assistant_message(run_id, failed)
        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.FAILED, visible_copy),
            session_id=session_id,
        )
        return ConsoleSubmitResult(True, True, failed.content)

    async def _finalize_agent_success(
        self,
        assistant_message_id: str,
        session_id: str,
        outcome: Any,
        *,
        variant_mode: bool,
        run_id: str | None = None,
        citation_repair_session: ConsoleCitationRepairSession | None = None,
        stream_signals: ConsoleProviderStreamSignals | None = None,
    ) -> ConsoleSubmitResult:
        """Handle ``RUN_DONE``: complete the placeholder (or a runtime-written one).

        An empty ``final_text`` is replaced with the fallback copy ``No
        response was generated.``. If the placeholder is missing, the runtime
        may have streamed content into an assistant row already; complete it
        when possible, otherwise append a new assistant message.

        Once the reply is completed (and its durable ``persisted_message_id``
        assigned), that persisted id is written back onto the agent run via
        ``_record_run_assistant_message`` -- the load-bearing correction of
        the native id ``create_run`` recorded, which resume anchors markers by.
        """
        placeholder = self._ensure_assistant_placeholder(
            assistant_message_id, session_id
        )
        if placeholder is not None:
            if (
                citation_repair_session is not None
                and stream_signals is not None
                and placeholder.content
                and bool(getattr(outcome, "final_text", ""))
                and not stream_signals.synthetic_fallback_emitted
            ):
                try:
                    selection = await self._select_post_generation_body(
                        assistant_message_id=assistant_message_id,
                        repair_session=citation_repair_session,
                        stream_signals=stream_signals,
                    )
                except KeyError:
                    # F4 fix (Qodo wave): `session_id` is a REQUIRED
                    # parameter of this method (always known, never
                    # re-derived) -- the bare no-arg call defaulted to
                    # whatever session is ACTIVE right now, wrongly
                    # stamping a STOPPED run state on an unrelated live
                    # session instead of this run's own owning session.
                    return self._session_closed_result(session_id=session_id)
                if selection.state == "canceled":
                    completed = self._ensure_assistant_placeholder(
                        assistant_message_id,
                        session_id,
                    )
                    if completed is None:
                        return self._session_closed_result(session_id=session_id)
                    self._record_run_assistant_message(run_id, completed)
                    return ConsoleSubmitResult(
                        True,
                        True,
                        selection.selected_body,
                    )
            completed = self._complete_agent_message(
                assistant_message_id, variant_mode, outcome
            )
            self._record_run_assistant_message(run_id, completed)
            self._set_run_state(
                ConsoleRunState(ConsoleRunStatus.COMPLETED, "Response complete."),
                session_id=session_id,
            )
            return ConsoleSubmitResult(True, True, completed.content)

        runtime_written = self._find_runtime_written_assistant(session_id)
        if runtime_written is not None and runtime_written.status in {
            "pending",
            "streaming",
        }:
            completed = self._complete_agent_message(
                runtime_written.id, variant_mode=False, outcome=outcome
            )
            self._record_run_assistant_message(run_id, completed)
            self._set_run_state(
                ConsoleRunState(ConsoleRunStatus.COMPLETED, "Response complete."),
                session_id=session_id,
            )
            return ConsoleSubmitResult(True, True, completed.content)

        final_text = getattr(outcome, "final_text", "") or "No response was generated."
        completed = self.store.append_message(
            session_id, role=ConsoleMessageRole.ASSISTANT, content=final_text
        )
        self._record_run_assistant_message(run_id, completed)
        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.COMPLETED, "Response complete."),
            session_id=session_id,
        )
        return ConsoleSubmitResult(True, True, completed.content)

    def _record_run_assistant_message(
        self,
        run_id: str | None,
        completed: ConsoleChatMessage,
    ) -> None:
        """Write the completed reply's PERSISTED id onto the agent run.

        On resume, markers anchor by matching a transcript message's durable
        ``persisted_message_id``; the id recorded at ``create_run`` time is
        the native in-memory id (the reply was not persisted yet), so it must
        be corrected here, once the reply has its persisted id. A no-op when
        there is no run id, no bridge, or no persistence (the native id would
        be useless to resume). Never fails the turn -- a marker-anchoring
        bookkeeping write, wrapped defensively like the file's other seams.
        """
        persisted = getattr(completed, "persisted_message_id", None)
        if not run_id or persisted is None or self._agent_bridge is None:
            return
        try:
            self._agent_bridge.record_run_assistant_message(run_id, persisted)
        except Exception:  # noqa: BLE001 -- bookkeeping must never fail the turn
            logger.opt(exception=True).warning(
                "failed to record persisted assistant id on agent run",
                run_id=run_id,
                persisted_message_id=persisted,
            )

    def _latest_unanchored_primary_run_id(self, conversation_id: str) -> str | None:
        """Return the active run's id for the stopped-via-cancel path.

        task-543: thin defensive wrapper over the bridge's
        ``latest_unanchored_primary_run_id`` (see its docstring for the
        NULL-anchor guard) -- a bookkeeping lookup on the Stop path must
        never fail the stop itself.

        Args:
            conversation_id: Durable conversation id whose runs to inspect.

        Returns:
            The recoverable run id, or ``None`` when there is no bridge, no
            matching unanchored primary run, or the lookup fails.
        """
        if self._agent_bridge is None:
            return None
        try:
            return self._agent_bridge.latest_unanchored_primary_run_id(conversation_id)
        except Exception:  # noqa: BLE001 -- bookkeeping must never fail the stop
            logger.opt(exception=True).warning(
                "failed to look up unanchored primary run for stop recording",
                conversation_id=conversation_id,
            )
            return None

    def _append_failed_assistant(
        self,
        session_id: str,
        visible_copy: str,
    ) -> ConsoleChatMessage:
        """Append a failed assistant message carrying ``visible_copy``.

        The store's terminal-status validation only accepts pending/streaming
        assistant messages, so the message is created empty, the copy is
        streamed in, and then it is marked failed.
        """
        message = self.store.append_message(
            session_id, role=ConsoleMessageRole.ASSISTANT, content=""
        )
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
        exhausted", or the loop-guard's own user-facing "Agent stopped:
        ..." copy -- TASK-1231/F3 AC4) is surfaced when available.
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
            if reason.startswith("Agent stopped:"):
                # Round 1 review (Minor): the loop-guard's own copy
                # (agent_runtime.py) already reads as a complete,
                # user-facing sentence -- prefixing "Agent run stuck: "
                # here would double the lead-in ("Agent run stuck: Agent
                # stopped: ...").
                return reason
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

    def _apply_context_summary_compaction(
        self, session_id: str, provider_messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Fold the session's boundary summary into ``provider_messages``.

        THE LEAK RULE (spec-review fix): compaction applies ONLY when the
        boundary USER message is actually PRESENT in this payload. When present,
        the payload rows BEFORE it are dropped and the summary is appended to
        the leading system prefix (which ``bound_messages_to_window`` preserves).
        When ABSENT -- e.g. regenerating a message that sits BEFORE the boundary,
        whose ancestors-only payload ends pre-boundary -- the payload is returned
        untouched: a summary covering LATER turns must never be substituted into
        an earlier point's context.

        Payload-row -> boundary matching mechanism: match by native message
        IDENTITY, not by content. Send-path payload builds thread each row's
        source transcript id onto it (``annotate_ids=True`` ->
        ``NATIVE_MESSAGE_ID_KEY``); the boundary is the row whose id equals the
        stored ``boundary_native_id``. The transform pipeline between build and
        this choke point only ever rewrites/drops the FINAL user turn (skill
        fork drops leading rows; chat-dictionary/world-info AND skill-
        substitution's own inline rewrites -- leading-mention replace and
        embedded-mention splice -- rewrite the last user row via ``{**row}``
        spreads that PRESERVE the key) and appends a synthesized continuation
        turn (no key) -- so every earlier row, and thus any strictly-earlier
        boundary, keeps its id intact.

        This is the genuine fail-safe: if the boundary id is not present on any
        row -- because the boundary sits after the payload's end
        (pre-boundary regenerate/retry/continue/edit-resend), or a branch
        switch/deletion made it dangling, or the payload was built WITHOUT id
        annotation -- NOTHING matches and the FULL history is sent unchanged.
        A byte-identical earlier duplicate of the boundary's text (e.g. a repeat
        "continue"/"yes") can no longer false-fire the way first-occurrence
        content matching did, so the summary of LATER turns is never injected
        into an EARLIER point's context.

        Args:
            session_id: Session owning the payload being dispatched.
            provider_messages: The fully-built, post-transform payload
                (id-annotated on the send path).

        Returns:
            The compacted payload, or ``provider_messages`` unchanged.
        """
        summary, boundary_native_id = self.store.session_context_summary(session_id)
        if not summary or boundary_native_id is None:
            return provider_messages

        boundary_index: int | None = None
        for index, row in enumerate(provider_messages):
            if row.get(NATIVE_MESSAGE_ID_KEY) == boundary_native_id:
                boundary_index = index
                break
        if boundary_index is None:
            return provider_messages

        sys_end = 0
        while (
            sys_end < len(provider_messages)
            and provider_messages[sys_end].get("role")
            == ConsoleMessageRole.SYSTEM.value
        ):
            sys_end += 1
        system_prefix = provider_messages[:sys_end]
        tail = provider_messages[boundary_index:]

        summary_suffix = "\n\n[Summary of earlier conversation]\n" + summary
        if system_prefix:
            first = system_prefix[0]
            merged_first = {
                **first,
                "content": (first.get("content") or "") + summary_suffix,
            }
            new_system = [merged_first, *system_prefix[1:]]
        else:
            new_system = [
                {
                    "role": ConsoleMessageRole.SYSTEM.value,
                    "content": summary_suffix.lstrip(),
                }
            ]
        return new_system + tail

    def _provider_messages_for_session(
        self,
        session_id: str,
        *,
        before_message_id: str | None = None,
        annotate_ids: bool = False,
    ) -> list[dict[str, Any]]:
        collected: list[ConsoleChatMessage] = []
        for message in self.store.messages_for_session(session_id):
            if message.id == before_message_id:
                break
            collected.append(message)
        return self._leading_system_message() + self._provider_message_payloads(
            collected, skip_failed=True, annotate_ids=annotate_ids
        )

    def _provider_messages_through_message(
        self,
        session_id: str,
        message_id: str,
        *,
        annotate_ids: bool = False,
    ) -> list[dict[str, Any]]:
        collected: list[ConsoleChatMessage] = []
        for message in self.store.messages_for_session(session_id):
            collected.append(message)
            if message.id == message_id:
                break
        return self._leading_system_message() + self._provider_message_payloads(
            collected,
            skip_failed=False,
            use_variant_content=True,
            annotate_ids=annotate_ids,
        )

    def _provider_message_payloads(
        self,
        session_messages: list[ConsoleChatMessage],
        *,
        skip_failed: bool,
        use_variant_content: bool = False,
        annotate_ids: bool = False,
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
            if skip_failed and message.status == "failed":
                # A send-blocked echo keeps its attachment data but is dropped
                # from the emitted payload below (skip_failed); it must not
                # reserve image budget a real message would then lose (TASK-457
                # code-review finding 2).
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

        def _emit(content: Any, source: ConsoleChatMessage) -> None:
            # Optionally thread the source transcript message's native id onto
            # the row so the dispatch choke point can anchor `/rewind` summary
            # compaction by identity (stripped before any provider sees it).
            row: dict[str, Any] = {"role": source.role.value, "content": content}
            if annotate_ids:
                row[NATIVE_MESSAGE_ID_KEY] = source.id
            payloads.append(row)

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
                _emit(parts, message)
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
                    _emit(placeholder, message)
                continue
            _emit(text, message)
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
        # Derive the owning session the same way `_active_stream_belongs_to_
        # session`/`streaming_session_id` do, rather than requiring every
        # caller to thread it through -- `assistant_message_id` is stable
        # even once the run finishes, so this is always resolvable unless
        # the session was closed out from under the run (in which case
        # there is nothing left to attribute the STOPPED stamp to).
        try:
            owner_id = self.store.session_id_for_message(assistant_message_id)
        except KeyError:
            owner_id = None
        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STOPPED, visible_copy),
            session_id=owner_id,
        )
        return stopped

    def _set_run_state(
        self, run_state: ConsoleRunState, *, session_id: str | None = None
    ) -> None:
        """Write ``run_state`` for ``session_id`` (default: the active session).

        Parallel-agents spec §2: this is the ONLY path that mutates the
        per-session run-state map -- ``run_state``/``run_state_history``
        stay read-only facades (see their property definitions near
        ``__init__``). ``session_id=None`` preserves every pre-existing
        call site's behavior (targets whatever session is currently active);
        callers that know the run's OWNING session (which may not be the
        active one once a background run outlives a session switch) pass it
        explicitly.
        """
        target = (
            session_id
            if session_id is not None
            else (self.store.active_session_id or "")
        )
        # Task 10 (background completion toasts): captured BEFORE the
        # overwrite below so the once-guard downstream can tell a genuine
        # transition INTO a terminal outcome (toast) apart from a
        # defensive re-stamp of the SAME terminal status onto an already-
        # terminal session (no toast -- the brief's own re-set test pins
        # this).
        previous_status = self.run_state_for(target).status
        self._run_states[target] = run_state
        self.run_state_history_for(target).append(run_state.status)
        # Task 9 finding #2 (deferred from Task 7 review): a terminal run
        # has no live approval left to decide, so the pending-approval flag
        # must be discarded for ANY terminal transition -- including the
        # currently ACTIVE session's own. Pre-Task-9 this discard lived
        # ONLY inside the non-active branch below (alongside the unvisited-
        # outcome stamp), so a pending flag on the session you were actually
        # LOOKING AT survived its own run ending, leaving a misleading
        # NEEDS_APPROVAL badge with no round left behind it. Kept as its own
        # unconditional block, separate from the unvisited-outcome stamp,
        # which deliberately STAYS non-active-only (the viewed session's own
        # COMPLETED/FAILED transition is visible live in its transcript and
        # must never grow a stale "unvisited" fleet marker on itself).
        # TASK-1050: a terminal run state means NO approval-like round can
        # legitimately remain live for this session from ANY bridge -- pop
        # the session's ENTIRE round-id set (not just the deprecated shim's
        # sentinel), unlike a single bridge's own teardown which only ever
        # discards ITS OWN round id.
        if run_state.status in {
            ConsoleRunStatus.BLOCKED,
            ConsoleRunStatus.COMPLETED,
            ConsoleRunStatus.FAILED,
            ConsoleRunStatus.STOPPED,
        }:
            # F2b fix (Qodo wave): this call always runs on the main
            # thread today, but guard it with the same lock as every other
            # `_pending_approvals` mutation for consistency (and so it
            # stays correct if a future caller ever moves this off-thread).
            with self._approval_state_lock:
                self._pending_approvals.pop(target, None)
        # Parallel-agents spec §6: stamp an unvisited terminal outcome, but
        # ONLY for a session other than the currently active (viewed) one --
        # the viewed session's own COMPLETED/FAILED transition is visible
        # live in its transcript and must never grow a stale "unvisited"
        # fleet marker on itself. `mark_session_visited` is the sole path
        # that clears an entry stamped here.
        if target != (self.store.active_session_id or ""):
            if run_state.status is ConsoleRunStatus.COMPLETED:
                self._unvisited_outcomes[target] = ConsoleRunMarker.FINISHED_OK
            elif run_state.status is ConsoleRunStatus.FAILED:
                self._unvisited_outcomes[target] = ConsoleRunMarker.FINISHED_FAILED
            # Task 10 (background completion toasts, parallel-agents spec):
            # ONE toast on a non-active session's run finishing/failing --
            # the viewed session's own terminal transition is visible live
            # in its transcript and gets none (same "user is watching" rule
            # as the unvisited-outcome stamp just above). Once-guarded on
            # the transition INTO a terminal state: `previous_status` was
            # NOT already one of the four terminal statuses, so re-setting
            # the same COMPLETED/FAILED status again (e.g. a defensive
            # re-stamp) does not re-toast.
            if (
                run_state.status
                in (ConsoleRunStatus.COMPLETED, ConsoleRunStatus.FAILED)
                and previous_status
                not in {
                    ConsoleRunStatus.BLOCKED,
                    ConsoleRunStatus.COMPLETED,
                    ConsoleRunStatus.FAILED,
                    ConsoleRunStatus.STOPPED,
                }
                and self.notify_run_outcome is not None
            ):
                self.notify_run_outcome(target, run_state.status)

    def _clear_terminal_run_state(self, session_id: str | None = None) -> None:
        """Clear stale terminal status copy for ``session_id`` (default: active).

        Parallel-agents spec §2: terminal-only guard preserved verbatim --
        a NON-terminal (e.g. STREAMING) run is never reset by this, so a
        background run in progress on another session is untouched when the
        viewed session changes.
        """
        target = (
            session_id
            if session_id is not None
            else (self.store.active_session_id or "")
        )
        if self.run_state_for(target).status in {
            ConsoleRunStatus.BLOCKED,
            ConsoleRunStatus.COMPLETED,
            ConsoleRunStatus.FAILED,
            ConsoleRunStatus.STOPPED,
        }:
            self._set_run_state(ConsoleRunState(), session_id=target)

    def _active_stream_belongs_to_session(self, session_id: str) -> bool:
        """Whether ``session_id`` has its own registered in-flight stream.

        Task 3b: a direct membership check now that the underlying map is
        keyed by session id -- no lookup (or ``KeyError`` guard) needed.
        """
        return session_id in self._active_assistant_message_ids

    def streaming_session_id(self) -> str | None:
        """Return A session with an in-flight stream, for tab status glyphs.

        Task 3b: this single-value contract predates true concurrency
        (Task 3) -- under concurrent runs there can be MULTIPLE streaming
        sessions at once, and this still returns only one. Prefers the
        ACTIVE (viewed) session when it has a live entry (keeps today's
        "the tab you're looking at shows the spinner" behavior), else an
        arbitrary (insertion-order) live entry. Full multi-session tab/
        fleet markers are PA-T8's job; this just keeps the existing
        single-glyph caller (``console_session_surface``'s tab strip) from
        going stale now that the underlying map is per-session.
        """
        active = self.store.active_session_id
        if active is not None and active in self._active_assistant_message_ids:
            return active
        for session_id in self._active_assistant_message_ids:
            return session_id
        return None

    def _session_closed_result(
        self, *, session_id: str | None = None
    ) -> ConsoleSubmitResult:
        """Result for a KeyError caused by the message's session vanishing mid-run.

        ``session_id`` is the run's owning session when the caller still has
        it in scope (most call sites do -- ``owner_id``/``session_id``
        resolved earlier in the same method); ``None`` only where the very
        first lookup of that owning session is what failed, i.e. there is
        genuinely nothing to attribute the STOPPED stamp to. Either way the
        owning session no longer exists in the store (``close_session``
        purges it), so this write is at worst an orphaned map entry -- never
        a stamp on a live, currently-viewed session.
        """
        visible_copy = "Session closed."
        self._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STOPPED, visible_copy),
            session_id=session_id,
        )
        return ConsoleSubmitResult(True, True, visible_copy)

    def _active_run_rejection(
        self, *, session_id: str | None = None
    ) -> ConsoleSubmitResult | None:
        """Defense-in-depth double-send guard for ``submit_draft``.

        F4 fix (Qodo wave): accepts an optional ``session_id`` so
        ``submit_draft`` can check the DISPATCHED session's own run state
        rather than whichever session happens to be active right now (the
        two can differ once a session switch races a background
        dispatch -- see ``submit_draft``'s own docstring). Every
        pre-existing caller (``retry_message``/``continue_from_message``/
        etc., which operate only on the active session by construction --
        each already blocks with "Open the original session..." if a
        target message belongs elsewhere) omits ``session_id`` and keeps
        checking the active session exactly as before.

        Args:
            session_id: The session to check, or ``None``/empty to check
                the currently active session (the pre-fix behavior).

        Returns:
            ``None`` when a new send may proceed; otherwise a blocked
            ``ConsoleSubmitResult`` carrying the refusal copy.
        """
        target_id = session_id if session_id else (self.store.active_session_id or "")
        if self.run_state_for(target_id).is_send_allowed:
            return None
        return ConsoleSubmitResult(
            accepted=False,
            should_clear_draft=False,
            # Must match the screen gate's `send_refusal_copy` own-session
            # copy (parallel-agents spec §4) -- a rapid double-send can hit
            # this internal defense-in-depth check instead of the screen's
            # gate (the loser of the exclusive-worker creation race), and a
            # mismatched copy there would read as two different bugs instead
            # of one lost race.
            visible_copy="A run is already running in this tab.",
        )
