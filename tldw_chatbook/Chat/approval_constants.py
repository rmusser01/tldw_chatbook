"""Shared capture cap for tool descriptions (widget, controller, permission summary, local tool provider)."""

#: ADR-332 (Qodo review #7): single named cap for tool-description capture
#: at the three pending-row producers (MCP, local, builtin) and the
#: summarizer prompt -- one constant so the egress bound cannot drift
#: between tool owners.
TOOL_DESCRIPTION_CAPTURE_CAP = 300


#: Fallback used when no `mcp_approval_timeout_seconds` seam is injected --
#: mirrors `UnifiedMCPControlPlaneService.approval_timeout_seconds`'s own
#: default (ADR-314), read directly here since the controller has no
#: dependency on that service (T6 wires the service into `MCPToolProvider`,
#: not into this controller).
#:
#: The PRE-ADR default was 120.0, kept strictly below
#: ``RunBudget.max_tool_call_seconds`` (300s at defaults,
#: ``Agents/agent_models.py``) because the invoke-path approval wait runs
#: INSIDE ``agent_service._call_with_timeout``'s per-call wrapper: an
#: approval timeout at/above the wrapper ceiling would let the wrapper fire
#: first, report the call failed, and a late approval would then execute
#: the tool for real on the abandoned thread. That invariant is SUPERSEDED:
#: the wrapper's deadline now PAUSES while a human decision is pending for
#: the run (``Agents/human_input_wait`` marks the wait;
#: ``_call_with_timeout``'s ``pauses_deadline`` re-arms the ceiling each
#: poll slice), so wall-clock counts tool execution, not human deliberation
#: -- an indefinite wait can no longer lose that race. Cancellation was
#: already closed by ``revoke_approval_rounds_for_run`` and is unaffected.
_DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS = 0.0
#: Poll granularity for `request_mcp_approvals`'s wait loop (binding, from
#: the Phase-5 plan) -- also the worst-case slack added on top of a
#: configured timeout/cancellation before this method observes it.
_MCP_APPROVAL_POLL_SECONDS = 1.0
#: Same ADR-314 contract as `_DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS`, for
#: `request_skill_install_confirm`'s own wait loop (fallback used when no
#: `skill_install_confirm_timeout_seconds` seam is injected).
_DEFAULT_SKILL_INSTALL_CONFIRM_TIMEOUT_SECONDS = 0.0
#: Same ADR-314 contract as `_DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS`, for
#: `request_skill_script_confirm`'s own wait loop (fallback used when no
#: `skill_script_confirm_timeout_seconds` seam is injected).
_DEFAULT_SKILL_SCRIPT_CONFIRM_TIMEOUT_SECONDS = 0.0
#: Same ADR-314 contract as `_DEFAULT_SKILL_SCRIPT_CONFIRM_TIMEOUT_SECONDS`,
#: for `request_chat_create_confirm`'s own wait loop (fallback used when no
#: `chat_create_confirm_timeout_seconds` seam is injected).
_DEFAULT_CHAT_CREATE_CONFIRM_TIMEOUT_SECONDS = 0.0
_CHAT_CREATE_SESSION_GONE = "session_gone"
#: Same ADR-314 contract, for `request_worktree_merge_confirm`'s own wait
#: loop (fallback used when no `worktree_merge_confirm_timeout_seconds`
#: seam is injected).
_DEFAULT_WORKTREE_MERGE_CONFIRM_TIMEOUT_SECONDS = 0.0
#: PRD A7: `[console] ask_user_timeout_seconds`. 0 = no deadline (ADR-314);
#: a positive value auto-continues the run with `answered: false`.
_DEFAULT_ASK_USER_TIMEOUT_SECONDS = 0.0
#: Env override for `[console] ask_user_timeout_seconds` (env -> config -> default).
ASK_USER_TIMEOUT_ENV_VAR = "TLDW_CONSOLE_ASK_USER_TIMEOUT_SECONDS"


#: A9 bounce counters are keyed by run id; a run that only ever bounced has no
#: end-of-run hook here, so the map is bounded by evicting the oldest run.
_MAX_TRACKED_QUESTION_BOUNCE_RUNS = 64


#: TASK-1050: synthetic round id `set_run_pending_approval`'s deprecated
#: boolean shim registers under, internally, so its add/discard composes
#: safely with the round-keyed `_pending_approvals` accounting (see that
#: method's docstring) without ever colliding with a real bridge round's
#: `uuid4()` id -- every genuine round id is a UUID string; this is not.
_LEGACY_PENDING_APPROVAL_ROUND_ID = "__legacy_pending_approval__"


#: TASK-1861: how broad each approval scope is. A session/always grant is
#: recorded against a tool NAME, so when per-call rows of one tool are
#: approved at different scopes only one can be stamped -- and it must be the
#: broadest the user chose, or the grant they asked for is silently dropped
#: and re-prompted. Unknown values rank 0 (narrower than anything known).
_APPROVAL_SCOPE_RANK: dict[str, int] = {
    "approve_once": 1,
    "approve_session": 2,
    "always_allow": 3,
}

#: The card answers that approve a call for every tool owner. MCP rows also
#: accept ``"allow_matching"``; built-in and local rows do not.
_APPROVING_DECISIONS: tuple[str, ...] = (
    "approve_once",
    "approve_session",
    "always_allow",
)


APPROVAL_OPTIONS = (
    ("Once", "approve_once"),
    ("Until Chatbook exits", "approve_session"),
    ("Always · these args", "allow_matching"),
    ("Always", "always_allow"),
    ("Deny", "deny"),
)


def approval_options(requested: object) -> list[tuple[str, str]]:
    """Apply the native default and supported-option filter for ordinary tools."""
    if not isinstance(requested, (list, tuple)) or not requested:
        return list(APPROVAL_OPTIONS)
    wanted = set(requested)
    narrowed = [pair for pair in APPROVAL_OPTIONS if pair[1] in wanted]
    return narrowed or list(APPROVAL_OPTIONS)
