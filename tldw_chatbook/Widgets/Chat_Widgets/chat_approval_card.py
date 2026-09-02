"""Inline chat approval card: the live Phase-5 MCP batch-approval flow
(task-5).

``set_batch``/``ApprovalDecided`` is the UI half of the Console MCP
tool-call approval round-trip: ``ConsoleChatController.
request_mcp_approvals`` (worker thread) pushes a pending batch into
``ChatScreen.chat_state.task_resume_state.pending_approval`` via
``app.call_from_thread``, which flows down through ``sync_task_resume_state``
-> ``ChatTaskCards.sync_state`` -> this card's ``set_batch``. The user's
decisions travel back up as an ``ApprovalDecided`` message that
``ChatScreen`` forwards to ``ConsoleChatController.resolve_pending_approval``
(UI thread), which releases the worker thread's wait.

Every method here stays synchronous end-to-end: ``ChatScreen.
set_task_resume_state``/``sync_task_resume_state`` and ``ChatTaskCards.
sync_state`` are plain sync calls, so ``set_batch`` cannot ``await``
anything either -- see its own docstring for how row remounts stay
collision-safe without awaiting ``remove_children()``.

(task-914: this card used to also carry a legacy single-approval API --
``set_approval``/``#approval-single-body`` with "Allow once"/"Deny"
buttons -- for the pre-task-649 ``Chat_Window_Enhanced`` composition.
That composition was fully retired in task-649, which deleted its last
caller and its dedicated pinning suite (``Tests/UI/test_chat_approvals_
and_resume.py``) without also removing the now-orphaned widget code;
``set_batch`` is the sole production entry point today.)
"""

from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any, Mapping, Sequence

from rich.markup import escape
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import Button, Select, Static, TextArea

from tldw_chatbook.MCP.redaction import redact_mapping
from tldw_chatbook.Tools.raw_cli_executor import MAX_RAW_COMMAND_BYTES

#: Per-row decision options, in display order. Values are the exact
#: decision strings `MCPToolProvider._apply_verdict` consumes.
_DECISION_OPTIONS: list[tuple[str, str]] = [
    ("Approve once", "approve_once"),
    ("Approve for session", "approve_session"),
    # TASK-26012: persists an allow scoped to EXACTLY the arguments shown
    # on this card (AC#3: the rule is created from what the user read);
    # the same tool with different arguments still asks.
    ("Always allow this exact input", "allow_matching"),
    ("Always allow", "always_allow"),
    ("Deny", "deny"),
]
_DEFAULT_DECISION = "approve_once"
_RAW_SHELL_SERVER_KEY = "local:__local__"
_RAW_SHELL_TOOL_NAME = "shell_exec"
_RAW_SHELL_DECISION_OPTIONS: list[tuple[str, str]] = [
    ("Run once", "approve_once"),
    ("Allow all raw shell commands for this Console session", "approve_session"),
    ("Deny", "deny"),
]
_RAW_SHELL_COPY_LIMIT = 2048


def _is_raw_shell_row(call: Mapping[str, Any]) -> bool:
    """Return whether a row is the reserved model raw-shell capability."""
    return (
        call.get("server_key") == _RAW_SHELL_SERVER_KEY
        and call.get("tool_name") == _RAW_SHELL_TOOL_NAME
    )

_EFFECT_LABELS: dict[str, str] = {
    "private_read": "may read private local data",
    "mutates_local": "may modify local data",
    "network": "may access the network",
    "llm_spend": "may incur LLM usage costs",
}


def format_approval_effects(entry: Mapping[str, Any]) -> str:
    """Render code-owned effects for an approval row without inspecting args."""
    effects = entry.get("effects")
    if not isinstance(effects, (list, tuple)):
        return ""
    labels = [
        _EFFECT_LABELS[str(effect)]
        for effect in effects
        if str(effect) in _EFFECT_LABELS
    ]
    return f"Effects: {'; '.join(labels)}" if labels else ""


def _options_for_row(call: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Decision options for one row, honoring an optional ``options`` key.

    Rows that omit ``options`` (every MCP row) get the full set, so MCP
    behavior is unchanged. A row may narrow it -- built-in tools offer
    only the session-scoped choices in P1, because persistent decisions
    for them cannot yet be undone in the UI. Unknown values are dropped,
    and an empty result falls back to the full set rather than rendering
    an unusable empty ``Select``.
    """
    if _is_raw_shell_row(call):
        return _RAW_SHELL_DECISION_OPTIONS
    requested = call.get("options") if isinstance(call, Mapping) else None
    if not isinstance(requested, (list, tuple)) or not requested:
        return _DECISION_OPTIONS
    wanted = set(requested)
    narrowed = [pair for pair in _DECISION_OPTIONS if pair[1] in wanted]
    return narrowed or _DECISION_OPTIONS


def _default_decision_for_row(
    call: Mapping[str, Any], row_values: Sequence[str]
) -> str:
    """Choose Deny for raw shell and preserve Approve once everywhere else."""
    preferred = "deny" if _is_raw_shell_row(call) else _DEFAULT_DECISION
    return preferred if preferred in row_values else row_values[0]


def _bounded_text(value: Any, byte_limit: int) -> str:
    """Return a UTF-8-safe bounded string for one optional approval field."""
    text = value if isinstance(value, str) else str(value or "")
    encoded = text.encode("utf-8")
    if len(encoded) <= byte_limit:
        return text
    return encoded[:byte_limit].decode("utf-8", errors="ignore")


def _raw_shell_metadata(entry: Mapping[str, Any]) -> str:
    """Render the validated shell selector, directory, and timeout literally."""
    arguments = entry.get("arguments")
    args = arguments if isinstance(arguments, Mapping) else {}
    return (
        f"Shell: {args.get('shell', 'auto')}\n"
        f"Directory: {args.get('initial_directory', '')}\n"
        f"Timeout: {args.get('timeout_seconds', '')} seconds"
    )


#: Reason-badge suffixes appended to a row's header line.
_REASON_SUFFIXES: dict[str, str] = {
    "config_changed": " (definition changed)",
    "risk_floored": " (high risk)",
}

#: Fleet-UX expert review F5/F7 (task-1234, item g): "(high risk)" on a
#: plain read (e.g. `read_file`) reads as alarmist with no explanation --
#: this is the row header's tooltip, a why-affordance for the badge alone.
#: `config_changed` isn't included: its badge already names the concrete
#: fact ("definition changed") and needs no further explanation.
_REASON_TOOLTIPS: dict[str, str] = {
    "risk_floored": (
        "Reads can exfiltrate file contents; built-in file tools always "
        "ask before running."
    ),
}


def _row_header_tooltip(entry: Mapping[str, Any]) -> str:
    """Return the row header's why-affordance tooltip, or ``""`` for none.

    Args:
        entry: One collapsed pending-call entry (see
            ``_collapse_pending_calls``).

    Returns:
        The tooltip text for ``entry``'s reason code, or ``""`` when that
        code carries no explanation (e.g. no reason at all, or
        ``config_changed``, whose badge is already self-explanatory).
    """
    return _REASON_TOOLTIPS.get(str(entry.get("reason", "") or ""), "")


#: TASK-1231/F3 AC2: appended (in addition to any `_REASON_SUFFIXES` badge)
#: when the row's `path_precheck_failed` flag is set -- a file tool
#: (read_file/list_directory/write_file) whose path argument will be
#: rejected by the roots check regardless of the user's decision. This is
#: a WARNING, never a gate: the row still offers every normal decision and
#: the user can still approve it (it will then fail with the same
#: recovery-route error `validate_path_multi` raises at dispatch).
_PATH_PRECHECK_SUFFIX = " -- path outside allowed folders; will fail even if approved"

#: TASK-1845: needs-decision was a border + 10% tint with no text change, so
#: the state vanished in monochrome. PRODUCT.md: "colour must never be the
#: only carrier of meaning."
NEEDS_DECISION_PREFIX = "needs decision · "


def format_approval_deadline(timeout_seconds: float | None) -> str:
    """Return the countdown copy for an armed approval deadline.

    TASK-1844: `set_batch` accepted `timeout_seconds` and never read it,
    while its own docstring claimed the value was "surfaced on the card".
    The controller arms a 120s auto-deny, so a clock the user could not see
    was making the decision for them.

    Args:
        timeout_seconds: The round's approval timeout, or None/0 when no
            deadline is armed.

    Returns:
        "Auto-denies in M:SS", or "" when nothing is armed -- say nothing
        rather than invent a number.
    """
    try:
        total = int(timeout_seconds or 0)
    except (TypeError, ValueError):
        return ""
    if total <= 0:
        return ""
    return f"Auto-denies in {total // 60}:{total % 60:02d}"


def _collapse_pending_calls(calls: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Group ``calls`` into one row per addressable verdict, first-seen order.

    Rows are keyed by ``call_id`` when the call has one, so two reads of two
    different files are two decisions -- the user can allow ``spec.md`` and
    refuse ``secrets.md``. Tools are how an agent reaches the outside world,
    so per-target granularity is the point of the gate.

    Calls with NO ``call_id`` still collapse by ``llm_name``, and that is
    deliberate rather than a leftover: the fence path builds ToolCalls
    without ids (``agent_runtime.parse_tool_call``), so the runtime can only
    apply a NAME-keyed verdict to them. Splitting those into separate rows
    would offer the user a decision the runtime cannot honour -- the row
    would say "deny this one" and every same-name call would stop.

    Each entry carries ``count`` (for the "×N" suffix) and ``all_arguments``
    (every grouped call's arguments, so a count never conceals a target).
    ADR-090 (task 5): each entry also carries the group's first non-empty
    ``rationale`` (the row's advisory context line) and ``description``
    (the tool definition's own text, for the external summarizer).
    """
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for call in calls:
        # Key by the per-call id when the runtime can address it; otherwise
        # by name, which is the only verdict key that reaches such a call.
        call_id = str(call.get("call_id", "") or "")
        name = call_id or str(call.get("llm_name", ""))
        if name not in grouped:
            entry = dict(call)
            entry["count"] = 1
            # TASK-1845: keep EVERY call's arguments, not just the first.
            # Grouping by name is the verdict contract and stays; hiding the
            # other calls' targets behind the count was the hazard -- three
            # reads of three different files rendered as one, so the user
            # approved three things having seen one.
            entry["all_arguments"] = [call.get("arguments")]
            # ADR-090 (task 5): seed the group's advisory-context slots so
            # every entry carries them; the fill-in below lets the first
            # NON-EMPTY value win even when it arrives on a later call.
            entry.setdefault("rationale", "")
            entry.setdefault("description", "")
            grouped[name] = entry
            order.append(name)
        else:
            grouped[name]["count"] += 1
            grouped[name].setdefault("all_arguments", []).append(call.get("arguments"))
        # ADR-090 (task 5): first non-empty wins -- a blank rationale on an
        # earlier call must not mask a real reason stated on a later one.
        if call.get("rationale") and not grouped[name].get("rationale"):
            grouped[name]["rationale"] = str(call.get("rationale"))
        if call.get("description") and not grouped[name].get("description"):
            grouped[name]["description"] = str(call.get("description"))
    return [grouped[name] for name in order]


def _format_row_header(entry: Mapping[str, Any]) -> str:
    """Return one row's header line: ``"server · tool"`` (+ ×N, + badges).

    Badge order: the reason badge (``config_changed``/``risk_floored``)
    first, then the roots pre-flight warning (TASK-1231/F3 AC2) last --
    a row can carry both (e.g. a high-risk `write_file` call whose path is
    ALSO outside every allowed root), and the pre-flight warning is the
    more actionable of the two, so it reads last/closest to the reader's
    eye rather than being buried before another badge.
    """
    server_label = str(entry.get("server_label", "") or "")
    tool_name = str(entry.get("tool_name", "") or entry.get("llm_name", "") or "")
    header = f"{server_label} · {tool_name}" if server_label else tool_name
    # TASK-1845: carry the needs-decision state in TEXT, not colour alone.
    if entry.get("needs_decision"):
        header = f"{NEEDS_DECISION_PREFIX}{header}"
    count = int(entry.get("count", 1) or 1)
    if count > 1:
        header += f" ×{count}"
    header += _REASON_SUFFIXES.get(str(entry.get("reason", "") or ""), "")
    if entry.get("path_precheck_failed"):
        header += _PATH_PRECHECK_SUFFIX
    return header


# ---------------------------------------------------------------------------
# Shared display helpers (ADR-090). These lived in a dedicated module
# (``tldw_chatbook.Chat.approval_display``) earlier in this PR; that module
# pushed the ``_ui_ready`` module census one over its never-rises ratchet
# (ADR-097 / TASK-23029), so they are folded back into this -- already
# mount-leg-resident -- module, their original pre-PR home. The public
# names stay importable from here for the summarizer service and the
# pending-row producers.
# ---------------------------------------------------------------------------

#: ADR-090: display cap for one advisory line (tail-biased).
RATIONALE_DISPLAY_CAP = 240
CONTEXT_LABEL = "Model context:"
SUMMARY_LABEL = "Summary:"

#: ADR-090 (Qodo review #7): single named cap for tool-description capture
#: at the three pending-row producers (MCP, local, builtin) and the
#: summarizer prompt -- one constant so the egress bound cannot drift
#: between tool owners.
TOOL_DESCRIPTION_CAPTURE_CAP = 300

_ARGS_SUMMARY_LIMIT = 80

#: TASK-695: per-VALUE budget inside the summary above. Without it a single
#: bulk argument (a `write_file` body, a pasted document) consumes the whole
#: line and every other argument -- including the destination the decision
#: turns on -- is clipped away. Sized so a typical path survives intact
#: while a payload is obviously abbreviated.
_ARGS_VALUE_LIMIT = 34

#: Floor for a shared value budget: below this a value is all ellipsis and
#: tells the reader nothing, so it is better to overflow the line cap (which
#: clips the tail) than to render every argument as noise.
_ARGS_MIN_VALUE_LIMIT = 10

#: TASK-695: argument names that say WHERE a call acts. Matched as whole
#: tokens (see `_is_destination_key`), so `profile` is not a file and
#: `urinal` is not a URL.
_DESTINATION_TOKENS: frozenset[str] = frozenset(
    {
        "path", "paths", "filepath", "file", "files", "filename",
        "dir", "dirs", "directory", "folder",
        "dest", "destination", "target", "output", "out",
        "src", "source", "input",
        "url", "uri", "endpoint", "host", "hostname",
        "cmd", "command", "script",
    }
)


def _snake_case(key: Any) -> str:
    """Return ``key`` lowercased with camelCase split into ``_`` tokens.

    Takes ``Any``, not ``str``: these keys come straight from model output,
    where a malformed payload can carry a non-string key. `re.sub` raises
    TypeError on one, which used to take down the whole approval row -- an
    approval the user cannot answer, blocking the run until the auto-deny
    fires. Coerced here, at the boundary.

    Args:
        key: One argument name from a tool call, of any type.

    Returns:
        The key as a lowercase, ``_``-separated string.
    """
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key)).lower()


def _is_destination_key(key: Any) -> bool:
    """Whether ``key`` names WHERE a call acts, rather than what it carries.

    TASK-695: these are the arguments an approval decision actually turns
    on -- the file being written, the URL being fetched, the command being
    run. They are hoisted ahead of bulk payloads so a large ``content`` can
    never push the destination off the end of the summary.

    Args:
        key: One argument name from the model's tool call. Any type -- see
            `_snake_case` for why this is not narrowed to ``str``.

    Returns:
        True when the name looks like a destination/target.
    """
    # Matched on TOKENS, not substrings: `profile` contains "file" and
    # `urinal` contains "uri", and a false positive reorders the line so the
    # real destination lands later in a budget-limited summary -- the exact
    # failure this hoisting exists to prevent.
    tokens = {
        token
        for token in re.split(r"[^a-z0-9]+|(?<=[a-z])(?=[0-9])", _snake_case(key))
        if token
    }
    return bool(tokens & _DESTINATION_TOKENS)


def summarize_arguments(arguments: Mapping[str, Any] | None) -> str:
    """Return ONE payload as a compact, ``markup=False``-safe summary.

    TASK-695: the summary used to be one ``json.dumps`` blob clipped at
    ``_ARGS_SUMMARY_LIMIT``, and ``json.dumps`` preserves the model's key
    order -- so a ``write_file`` emitting ``content`` before ``file_path``
    showed 67 characters of file body and truncated the destination out of
    view. The card asked "may I write this?" without showing where.

    Two changes fix that without raising the global cap (which would only
    move the cliff): destination-like keys are rendered FIRST, and every
    value gets its own budget so one bulk payload cannot consume the line.

    Secret-looking values (``api_key``, ``token``, ``password``, ...) are
    redacted before rendering -- redaction parity with every other MCP
    display/log boundary (see ``tldw_chatbook.MCP.redaction``'s module
    docstring); the approval card is the one place a raw secret argument
    was still reaching the screen unredacted. Redaction runs BEFORE the
    reordering and clipping below, so neither can expose a secret.
    """
    try:
        redacted = redact_mapping(dict(arguments or {}))
    except Exception:  # noqa: BLE001 -- a bad arg must never crash rendering
        return str(arguments or {})[:_ARGS_SUMMARY_LIMIT]
    if not redacted:
        return "{}"

    # Destinations first, each group keeping the model's own order so a call
    # with several paths still reads in the order it was made.
    ordered = sorted(redacted.items(), key=lambda kv: not _is_destination_key(kv[0]))

    def _render(value: Any, budget: int) -> str:
        try:
            text = json.dumps(value, default=str, separators=(",", ":"))
        except Exception:  # noqa: BLE001
            text = json.dumps(str(value))
        if len(text) > budget:
            return text[: max(1, budget - 1)] + "…"
        return text

    # Destinations are rendered first and at the full per-value budget; what
    # they leave is split evenly among the remaining arguments. Without the
    # split, the second bulk argument still starves everything after it --
    # a fixed per-value cap only moves the cliff along by one key.
    destinations = [kv for kv in ordered if _is_destination_key(kv[0])]
    payloads = [kv for kv in ordered if not _is_destination_key(kv[0])]
    overhead = sum(len(json.dumps(str(key))) + 2 for key, _ in ordered) + 2
    spent = sum(len(_render(v, _ARGS_VALUE_LIMIT)) for _, v in destinations)
    share = _ARGS_VALUE_LIMIT
    if payloads:
        remaining = _ARGS_SUMMARY_LIMIT - overhead - spent
        share = max(
            _ARGS_MIN_VALUE_LIMIT, min(_ARGS_VALUE_LIMIT, remaining // len(payloads))
        )

    parts = [
        f"{json.dumps(str(key))}:"
        f"{_render(value, _ARGS_VALUE_LIMIT if _is_destination_key(key) else share)}"
        for key, value in ordered
    ]
    text = "{" + ",".join(parts) + "}"
    if len(text) > _ARGS_SUMMARY_LIMIT:
        return text[: _ARGS_SUMMARY_LIMIT - 1] + "…"
    return text


def summarize_row_arguments(entry: Mapping[str, Any]) -> str:
    """Return the summary for one COLLAPSED row -- every call's arguments.

    TASK-1845: a row that says "x3" must show all three targets or the count
    is concealing the decision, so each grouped call's payload is rendered on
    its own line and capped independently (one long payload cannot push the
    others off screen). Redaction applies to every line, not just the first.

    Takes the collapsed ENTRY, not an arguments mapping. The two shapes were
    once distinguished by sniffing for an ``all_arguments`` key inside the
    arguments themselves, which both mis-fires on a tool that genuinely has
    an argument by that name and -- as shipped -- silently did nothing,
    because the render site passed the first call's arguments and the branch
    never ran.
    """
    sets = entry.get("all_arguments")
    if not sets:
        # Not a collapsed entry (or a row with no arguments at all): fall
        # back to the single payload so a caller can never render nothing.
        return summarize_arguments(entry.get("arguments"))
    rendered = [summarize_arguments(payload) for payload in sets]
    # De-duplicate identical payloads while preserving order: N identical
    # calls are one decision with one target, and repeating it N times
    # would bury a genuinely different target further down.
    seen: set[str] = set()
    unique = [r for r in rendered if not (r in seen or seen.add(r))]
    return "\n".join(unique)


# Historical underscore names this module's own code and the existing
# suites import.
_summarize_arguments = summarize_arguments
_summarize_row_arguments = summarize_row_arguments


def format_context_line(text: object, cap: int = RATIONALE_DISPLAY_CAP) -> str:
    """Tail-biased display clip for one advisory context/summary line.

    Args:
        text: Raw advisory text (model rationale or summarizer output).
        cap: Maximum rendered length including the ellipsis.

    Returns:
        The clipped line, or "" for blank/absent input.
    """
    from tldw_chatbook.Agents.agent_models import normalize_rationale

    return normalize_rationale(text, cap=cap)


class ChatApprovalCard(Container):
    """Inline approval card for privileged agent actions."""

    #: TASK-1846: `.ds-approval-card` is the design system's approval
    #: treatment (thick border in the approval-required colour, 12% tint) and
    #: it was applied by NOTHING -- `#chat-approval-card` had no rules at all,
    #: so the card asking permission for an agent to reach the outside world
    #: rendered as ordinary body text. This is the surface's whole visual
    #: identity; it belongs on the class, not on each mount site.
    DEFAULT_CLASSES = "ds-approval-card"

    def first_focus_widget_id(self) -> str:
        """Return the id the keyboard should land on when this card is reached.

        TASK-1845. Every row is pre-armed to ``_DEFAULT_DECISION``
        ("approve_once") because a blank Select breaks ``allow_blank=False``
        and the bulk-assign path. That default is fine; landing the keyboard
        on the COMMIT control was not. Both review entry points focused
        ``#approval-submit``, so the documented route -- jump to the card,
        press Enter -- granted a tool access to a call the user had not read.

        Tools are how an agent reaches the outside world, so this card is the
        egress boundary; arriving on it should present a choice, not a
        pre-signed one.

        Returns:
            The row's decision Select when there is one to decide, else the
            card's own container id -- never the Submit button.
        """
        if self._batch_phase == "finishing":
            return "chat-approval-card"
        try:
            selects = self.query(".approval-row-decision")
        except Exception:
            return "chat-approval-card"
        for select in selects:
            if select.id:
                return select.id
        return "chat-approval-card"

    def focus_first_decision(self) -> None:
        """Move focus to this card's first undecided control.

        The single seam both review entry points use, so a third caller
        cannot reintroduce a focus target that commits on Enter.
        """
        if self._batch_phase == "finishing":
            self.focus()
            return
        try:
            selects = list(self.query(".approval-row-decision"))
        except Exception:
            selects = []
        for select in selects:
            if not select.disabled and select.can_focus:
                select.focus()
                return
        # No rows to decide (card shown for a batch that has since resolved):
        # focus the card itself rather than an action button.
        try:
            self.focus()
        except Exception:
            pass

    class ApprovalDecided(Message):
        """Posted when the user submits per-row decisions for a pending batch.

        ``round_id`` (Task 9 fix round 1) is the id ``ConsoleChatController.
        request_mcp_approvals`` stamped onto the batch payload this card
        was built from (see ``set_batch``) -- ``ConsoleChatController.
        resolve_pending_approval`` uses it to resolve THIS exact round,
        never "whichever session happens to be active" (an async-message
        misattribution hazard fixed alongside this: `ApprovalDecided` can
        be delivered after a `switch_session` already moved the active
        session elsewhere). ``None`` when no round id was supplied (e.g.
        a test constructing this message directly).
        """

        def __init__(
            self, decisions: dict[str, str], *, round_id: str | None = None
        ) -> None:
            self.decisions = decisions
            self.round_id = round_id
            super().__init__()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize batch state before mount to avoid AttributeError in pre-mount calls."""
        super().__init__(*args, **kwargs)
        # Batch approval state (initialized here, not in on_mount, so pre-mount
        # calls like set_batch() or on_button_pressed() don't AttributeError).
        self._batch_generation = 0
        self._batch_names: list[str] = []
        self._batch_selects: list[Select] = []
        self._batch_legal_values: list[list[str]] = []
        self._batch_rows: list[Vertical] = []
        #: The current batch's fast-approval buttons (task-1234 review
        #: round 1), if any -- membership-guards `on_button_pressed`
        #: against a stale press the same way `_on_batch_row_select_
        #: changed` guards `self._batch_selects`. See `_submit_fast_
        #: decision`'s docstring for why this guard exists.
        self._batch_fast_buttons: list[Button] = []
        #: The currently-rendered batch's round id (Task 9 fix round 1),
        #: echoed back unchanged in `ApprovalDecided` on submit. `None`
        #: whenever no batch (or a caller that predates round ids) is
        #: showing.
        self._batch_round_id: str | None = None
        self._batch_phase = "approval"
        self._batch_calls_snapshot: list[dict[str, Any]] = []
        #: ADR-090 (task 5): the current batch's advisory summary (payload
        #: carriage on ``set_batch``, patchable in place via ``set_summary``
        #: for a matching round), re-rendered by ``_render_summary_line``.
        self._batch_summary: str | None = None
        # task-17500: the initial hide is CONSTRUCTION state, never deferred
        # mount work. This used to live in `on_mount` (`self.display =
        # False` plus a `call_after_refresh(_hide_batch_body)` for the batch
        # body) -- and on a real terminal the fresh Console screen's first
        # paint takes longer than the screen's 0.05s mount sync, so the
        # deferred hide landed AFTER `set_batch` had rendered a headless
        # round's card and unrendered it: the "Approval required"-and-
        # nothing-else pane the close-out live pass reproduced on both
        # headless paths. A hide applied here cannot race anything; the
        # batch body gets the same treatment in `compose`.
        self.display = False

    def compose(self) -> ComposeResult:
        yield Static("Approval required", id="approval-title")
        # TASK-1844: the countdown lives beside the title, not buried in a
        # row -- it applies to the whole batch and it decides for the user.
        deadline = Static("", id="approval-deadline", markup=False)
        deadline.display = False
        yield deadline
        # ADR-090 (task 5): the batch-level advisory summary line, built
        # hidden like the deadline above (task-17500 pattern -- see
        # __init__). markup=True because the dim/italic styling requires
        # markup; the payload text goes through rich's ``escape()`` before
        # it lands here, so brackets in model output cannot inject tags.
        summary = Static("", id="approval-summary", markup=True)
        summary.display = False
        yield summary
        # task-17500: built hidden, like the deadline above -- see __init__.
        batch_body = Container(id="approval-batch-body")
        batch_body.display = False
        with batch_body:
            yield Vertical(id="approval-batch-rows")
            with Horizontal(id="approval-batch-actions"):
                yield Button(
                    "Approve all",
                    id="approval-approve-all",
                    tooltip="Set every pending tool call's decision to Approve once.",
                )
                yield Button(
                    "Submit",
                    id="approval-submit",
                    variant="primary",
                    tooltip="Apply each row's selected decision and resume the run.",
                )
                yield Button(
                    "Deny all",
                    id="approval-deny-all",
                    variant="error",
                    tooltip="Set every pending tool call's decision to Deny.",
                )

    # -- batch-approval API (task-5) -----------------------------------------

    def set_batch(
        self,
        calls: list[dict[str, Any]],
        *,
        timeout_seconds: float,
        round_id: str | None = None,
        phase: str = "approval",
        summary: str | None = None,
    ) -> None:
        """Render one row per unique ``llm_name`` in ``calls``.

        Synchronous throughout -- see the module docstring for why this
        cannot ``await``. Repeated resume-state syncs for one unchanged,
        identified round preserve its mounted controls. Changed rounds,
        calls, or phases prune old rows via a fire-and-forget
        ``remove_children()`` (Textual 8.2.7 defers the actual detachment
        to the next event-loop tick -- see ``Widget.remove_children``'s
        ``AwaitRemove``/``App._prune``), while every new row gets an id
        tagged with a fresh, monotonically increasing generation number.
        That makes a still-pruning previous batch's ids structurally
        unable to collide with the incoming batch's ids, without this
        method ever needing to await the removal.

        Args:
            calls: This turn's pending tool calls.
            timeout_seconds: The round's configured approval timeout,
                surfaced on the card.
            round_id: The round's unique id (Task 9 fix round 1), stashed
                and echoed back verbatim in ``ApprovalDecided`` on submit
                so ``ConsoleChatController.resolve_pending_approval`` can
                resolve THIS exact round rather than guessing from
                whichever session happens to be active when the decision
                is delivered.
            summary: ADR-090 advisory batch summary carried by the
                payload, re-rendered on every remount.

        Raises:
            NoMatches: When the card's composed containers are not attached
                yet. Raised BEFORE any state is touched (task-17500): the
                production caller (``sync_task_resume_state``) swallows
                ``QueryError`` with no retry, so a ``set_batch`` that
                mutated ``display`` first would strand a visible,
                title-only card -- the same user-visible state as the
                mount-ordering bug, through a different writer.
        """
        normalized_phase = "finishing" if phase == "finishing" else "approval"
        if (
            round_id is not None
            and round_id == self._batch_round_id
            and normalized_phase == self._batch_phase
            and calls == self._batch_calls_snapshot
        ):
            return

        # task-17500: all-or-nothing -- resolve every container this method
        # writes to before mutating anything, including the round-id stash.
        title = self.query_one("#approval-title", Static)
        batch_body = self.query_one("#approval-batch-body")
        rows_container = self.query_one("#approval-batch-rows", Vertical)
        self._batch_round_id = round_id
        self._batch_phase = normalized_phase
        self._batch_calls_snapshot = deepcopy(calls)
        finishing = self._batch_phase == "finishing"
        # A finishing card is status, not a decision form. Keep the existing
        # card container as its keyboard inspection target while every
        # decision control is disabled.
        self.can_focus = finishing
        title.update(
            "Finishing — Stop will not cancel" if finishing else "Approval required"
        )
        # ADR-090 (task 5): stash the payload-carried summary so any remount
        # re-renders it (a live `set_summary` patch is for THIS mount only).
        self._batch_summary = format_context_line(summary) if summary else None
        # TASK-1844: actually surface the deadline the docstring promised.
        try:
            deadline = self.query_one("#approval-deadline", Static)
            text = format_approval_deadline(timeout_seconds)
            deadline.update(text)
            deadline.display = bool(text)
        except NoMatches:
            pass
        self._render_summary_line()
        if not calls:
            self.display = False
            self.can_focus = False
            batch_body.display = False
            self._batch_names = []
            self._batch_selects = []
            self._batch_legal_values = []
            self._batch_rows = []
            self._batch_fast_buttons = []
            return

        self.display = True
        batch_body.display = True
        # task-1234 review round 1: a submit-shaped control (Submit, or
        # either fast button) disables itself right after a press to close
        # the double-submit window -- see `_disable_batch_submit_controls`.
        # A NEW batch must start every submitting control re-enabled,
        # otherwise a round whose PREDECESSOR was resolved via Submit would
        # render with a permanently-disabled Submit button.
        for button_id in (
            "#approval-approve-all",
            "#approval-submit",
            "#approval-deny-all",
        ):
            try:
                self.query_one(button_id, Button).disabled = finishing
            except NoMatches:
                pass

        grouped = _collapse_pending_calls(calls)
        self._batch_generation += 1
        generation = self._batch_generation
        # Fleet-UX expert review F5 (task-1234): a single-decision card
        # still forced a two-step Select-then-Submit commit. Both fast
        # decisions ("approve_once"/"deny") are legal for EVERY row this
        # card ever renders -- MCP rows get the full `_DECISION_OPTIONS`
        # set unconditionally, and the one narrowed case in production
        # (built-in tools, `options=("approve_once", "approve_session",
        # "deny")` -- see `ConsoleChatController`'s review-hook docstring)
        # deliberately keeps both -- so no legality check is needed here,
        # unlike the bulk Approve-all/Deny-all buttons' `legal_values` dance.
        single_row = len(grouped) == 1
        names: list[str] = []
        selects: list[Select] = []
        legal_values: list[list[str]] = []
        rows: list[Vertical] = []
        fast_buttons: list[Button] = []
        for index, entry in enumerate(grouped):
            # The verdict key must match what the RUNTIME looks up, and it
            # looks up `call_id` first, then name. Emitting the name here
            # while grouping rows per call would give the user a per-call
            # decision the runtime then applies to every same-name call.
            names.append(str(entry.get("call_id", "") or entry.get("llm_name", "")))
            row_options = _options_for_row(entry)
            row_values = [value for _label, value in row_options]
            default_value = _default_decision_for_row(entry, row_values)
            select = Select(
                row_options,
                value=default_value,
                allow_blank=False,
                id=f"approval-row-decision-{generation}-{index}",
                classes="approval-row-decision",
            )
            select.disabled = finishing
            selects.append(select)
            legal_values.append(row_values)
            header_static = Static(
                _format_row_header(entry),
                markup=False,
                classes="approval-row-header",
            )
            header_tooltip = _row_header_tooltip(entry)
            if header_tooltip:
                header_static.tooltip = header_tooltip
            # TASK-1846 AC#2: the controls are FIXED width (26 + 14 + 14 =
            # 54 cells), so sharing one line with the text left the arguments
            # 10 cells on an 80-column terminal -- `{"path":"~/` of
            # `{"path":"~/notes/secrets.md"}`. Since TASK-1861 the card offers
            # one decision per TARGET, so telling those apart IS the row's
            # job. Header, arguments and controls each get their own full
            # width line.
            #
            # Keeping the header BESIDE the controls was tried and is wrong:
            # in the Console's ~52-cell chat pane those 54 fixed cells starve
            # the header to ONE cell, which wraps to nine lines and pushes the
            # arguments out of the card entirely -- worse than the layout it
            # replaced. Only a real terminal showed that; every mounted-widget
            # measurement at 80/120/212 looked fine.
            control_children: list[Any] = [select]
            detail_children: list[Any]
            if _is_raw_shell_row(entry):
                full_command = entry.get("full_command")
                if not isinstance(full_command, str):
                    arguments = entry.get("arguments")
                    args = arguments if isinstance(arguments, Mapping) else {}
                    full_command = str(args.get("command", "") or "")
                detail_children = [
                    Static(
                        "Complete command:",
                        markup=False,
                        classes="approval-row-raw-label",
                    ),
                    TextArea(
                        _bounded_text(full_command, MAX_RAW_COMMAND_BYTES),
                        read_only=True,
                        show_line_numbers=False,
                        classes="approval-row-full-command",
                    ),
                    Static(
                        _raw_shell_metadata(entry),
                        markup=False,
                        classes="approval-row-raw-metadata",
                    ),
                    Static(
                        "DANGER: "
                        + _bounded_text(entry.get("warning"), _RAW_SHELL_COPY_LIMIT),
                        markup=False,
                        classes="approval-row-raw-warning",
                    ),
                    Static(
                        "Session scope: "
                        + _bounded_text(
                            entry.get("scope_notice"), _RAW_SHELL_COPY_LIMIT
                        ),
                        markup=False,
                        classes="approval-row-raw-scope",
                    ),
                ]
            else:
                detail_children = [
                    Static(
                        _summarize_row_arguments(entry),
                        markup=False,
                        classes="approval-row-args",
                    )
                ]
            effect_copy = format_approval_effects(entry)
            if effect_copy:
                detail_children.append(
                    Static(
                        effect_copy,
                        markup=False,
                        classes="approval-row-effects",
                    )
                )
            # ADR-090 (task 5): the row's advisory context line, directly
            # below the details it annotates. markup=True because the
            # dim/italic styling requires markup; ``escape()`` neutralizes
            # bracket injection in the model-authored text. Rendered only
            # when the collapsed entry carries a non-empty rationale.
            context = format_context_line(entry.get("rationale"))
            if context:
                detail_children.append(
                    Static(
                        f"[dim italic]{CONTEXT_LABEL} {escape(context)}[/dim italic]",
                        markup=True,
                        id=f"approval-context-{generation}-{index}",
                    )
                )
            if single_row:
                fast_approve = Button(
                    "Run once" if _is_raw_shell_row(entry) else "Approve once",
                    id=f"approval-fast-approve-{generation}-{index}",
                    variant="success",
                    compact=True,
                    classes="approval-row-fast-approve",
                    tooltip=(
                        "Approve once and resume immediately (skips Select + Submit)."
                    ),
                )
                fast_deny = Button(
                    "Deny",
                    id=f"approval-fast-deny-{generation}-{index}",
                    variant="error",
                    compact=True,
                    classes="approval-row-fast-deny",
                    tooltip=("Deny and resume immediately (skips Select + Submit)."),
                )
                fast_buttons.extend((fast_approve, fast_deny))
                fast_approve.disabled = finishing
                fast_deny.disabled = finishing
                control_children.append(fast_approve)
                control_children.append(fast_deny)
            rows.append(
                Vertical(
                    header_static,
                    *detail_children,
                    Horizontal(
                        *control_children,
                        classes="approval-row-controls",
                    ),
                    id=f"approval-row-{generation}-{index}",
                    classes="approval-row",
                )
            )
        self._batch_names = names
        self._batch_selects = selects
        self._batch_legal_values = legal_values
        self._batch_rows = rows
        self._batch_fast_buttons = fast_buttons

        rows_container.remove_children()
        if rows:
            rows_container.mount(*rows)

    def _render_summary_line(self) -> None:
        """Render the batch-level advisory summary line (ADR-090).

        Plain, dim/italic, visually subordinate to every machine-owned
        field; hidden entirely when there is nothing to show.
        """
        try:
            summary = self.query_one("#approval-summary", Static)
        except NoMatches:
            return
        text = self._batch_summary or ""
        if text:
            summary.update(
                f"[dim italic]{SUMMARY_LABEL} {escape(text)}[/dim italic]"
            )
            summary.display = True
        else:
            summary.update("")
            summary.display = False

    def set_summary(self, round_id: str | None, text: str) -> None:
        """Patch ONLY the batch summary line for a matching round (ADR-090).

        Guarded by the card's current round id -- a late result from a
        prior round must never land on the current card -- and never
        re-runs ``set_batch``, so per-row Selects and in-progress decisions
        are untouched.

        Args:
            round_id: The approval round the summary belongs to; dropped
                unless it equals this card's current round id.
            text: The advisory summary text; clipped and control-stripped
                by ``format_context_line`` before rendering.
        """
        if round_id is None or self._batch_round_id != round_id:
            return
        self._batch_summary = format_context_line(text)
        self._render_summary_line()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "approval-approve-all":
            event.stop()
            self._set_all_batch_decisions(("approve_once", "approve_session"))
        elif button_id == "approval-deny-all":
            event.stop()
            self._set_all_batch_decisions(("deny",))
        elif button_id == "approval-submit":
            event.stop()
            self._submit_batch_decisions()
        elif button_id.startswith("approval-fast-approve-"):
            event.stop()
            # task-1234 review round 1: membership-guard against a STALE
            # button -- see `_submit_fast_decision`'s docstring for the
            # race this closes. `event.button` (not the id string) so this
            # matches by widget identity, the same way `_on_batch_row_
            # select_changed` guards `self._batch_selects`.
            if event.button in self._batch_fast_buttons:
                self._submit_fast_decision("approve_once")
        elif button_id.startswith("approval-fast-deny-"):
            event.stop()
            if event.button in self._batch_fast_buttons:
                self._submit_fast_decision("deny")

    def _set_all_batch_decisions(self, candidates: tuple[str, ...]) -> None:
        """Bulk-set every row to the first of ``candidates`` that row legally offers.

        A narrowed row (task-5's per-row ``options``) may not offer every
        bulk target -- assigning a value outside a ``Select``'s own option
        list raises Textual's ``InvalidSelectValueError``. ``candidates`` is
        a preference order (e.g. "Approve all" prefers ``approve_once``,
        falling back to ``approve_session`` -- both are approvals, so
        falling back between them is honest); a row that legally offers
        none of ``candidates`` is left on its current value rather than
        crashing or silently doing nothing useful.

        A row left untouched this way is otherwise visually identical to a
        row nobody has looked at yet, so it also gets a ``needs-decision``
        class on its row container -- a visible "this one still needs an
        explicit choice" signal. A row that DOES receive a bulk value has
        the class cleared, so a stale flag from an earlier bulk press
        (e.g. "Approve all" skipped it, then "Deny all" successfully set
        it) never lingers.
        """
        for select, legal_values, row in zip(
            self._batch_selects, self._batch_legal_values, self._batch_rows
        ):
            applied = False
            for candidate in candidates:
                if candidate in legal_values:
                    select.value = candidate
                    applied = True
                    break
            if applied:
                row.remove_class("needs-decision")
            else:
                row.add_class("needs-decision")

    @on(Select.Changed)
    def _on_batch_row_select_changed(self, event: Select.Changed) -> None:
        """Clear a row's ``needs-decision`` flag once it has an explicit choice.

        The only ``Select`` widgets under this card are the per-row batch
        decision selects, so no id/class scoping is needed on the
        decorator -- membership in ``self._batch_selects`` is enough to
        identify "one of our rows."
        This does not stop the event: nothing else in this widget handles
        ``Select.Changed`` today, and it must keep bubbling exactly as it
        did before this handler existed (``TldwCli.on_select_changed`` in
        ``app.py`` filters by ``select.id``, which these rows never set, so
        it already ignores them).
        """
        select = event.select
        if select not in self._batch_selects:
            return
        index = self._batch_selects.index(select)
        self._batch_rows[index].remove_class("needs-decision")

    def _disable_batch_submit_controls(self) -> None:
        """Disable this round's submitting controls right after a press.

        task-1234 review round 1: Submit and both fast buttons each
        resolve the ENTIRE round with one press. Before this, a second
        click landing in the brief window before the round's teardown
        (the next ``set_batch``/hide) would post a SECOND ``ApprovalDecided``
        for a round that may already be resolved -- safe only incidentally,
        by whatever the controller does with a duplicate resolution, not by
        anything this widget guaranteed. Disabling immediately, rather than
        waiting on a re-render, closes that window directly. ``set_batch``
        re-enables ``#approval-submit`` (and repopulates
        ``self._batch_fast_buttons`` with fresh, enabled buttons) at the
        start of every new round, so this is never a permanent lockout.
        """
        for button in self._batch_fast_buttons:
            button.disabled = True
        try:
            self.query_one("#approval-submit", Button).disabled = True
        except NoMatches:
            pass

    def _submit_batch_decisions(self) -> None:
        decisions = {
            name: select.value
            for name, select in zip(self._batch_names, self._batch_selects)
        }
        self._disable_batch_submit_controls()
        self.post_message(
            self.ApprovalDecided(decisions, round_id=self._batch_round_id)
        )

    def _submit_fast_decision(self, decision: str) -> None:
        """Single-row fast path (task-1234/F5): resolve without touching Selects.

        Only ever reachable when ``set_batch`` rendered exactly one row
        (the fast buttons are gated on ``single_row`` there) AND the
        pressed button is still a member of ``self._batch_fast_buttons``
        (``on_button_pressed``'s guard, task-1234 review round 1) -- so
        ``self._batch_names[0]`` is unambiguously THIS round's row. Without
        that membership guard, a fire-and-forget ``remove_children()`` (see
        ``set_batch``'s docstring) leaves a stale-generation button mounted
        and clickable for one event-loop tick after a NEW batch supersedes
        it; pressing it would otherwise resolve the NEW round using
        whatever ``self._batch_names``/``self._batch_round_id`` the newer
        ``set_batch`` call just overwrote them with -- silently deciding a
        tool call the user never reviewed. ``round_id`` alone does not
        catch this: ``set_batch`` overwrites ``_batch_round_id`` wholesale
        on every call, so an old and a new round's messages are not
        distinguishable by id at this layer.

        Posts the SAME ``ApprovalDecided`` message, through the SAME
        ``round_id``, as ``_submit_batch_decisions`` -- no new resolution
        seam; ``ConsoleChatController.resolve_pending_approval`` cannot
        tell this apart from a normal Select+Submit round trip.

        Args:
            decision: The verdict to submit. Only ever ``"approve_once"``
                or ``"deny"`` (the two call sites in ``on_button_pressed``)
                -- a fast click can never grant ``"approve_session"``/
                ``"always_allow"``; those stay reachable only through the
                row's own Select + Submit.
        """
        if not self._batch_names:
            return
        self._disable_batch_submit_controls()
        self.post_message(
            self.ApprovalDecided(
                {self._batch_names[0]: decision}, round_id=self._batch_round_id
            )
        )
