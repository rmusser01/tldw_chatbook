"""Inline chat approval card: legacy single-approval API + the live
Phase-5 MCP batch-approval flow (task-5).

The batch path (`set_batch`/`ApprovalDecided`) is the UI half of the
Console MCP tool-call approval round-trip: ``ConsoleChatController.
request_mcp_approvals`` (worker thread) pushes a pending batch into
``ChatScreen.chat_state.task_resume_state.pending_approval`` via
``app.call_from_thread``, which flows down through ``sync_task_resume_state``
-> ``ChatTaskCards.sync_state`` -> this card's ``set_batch``. The user's
decisions travel back up as an ``ApprovalDecided`` message that
``ChatScreen`` forwards to ``ConsoleChatController.resolve_pending_approval``
(UI thread), which releases the worker thread's wait.

Every method here stays synchronous end-to-end (mirrors ``set_approval``):
``ChatScreen.set_task_resume_state``/``sync_task_resume_state`` and
``ChatTaskCards.sync_state`` are plain sync calls frozen by
``Tests/UI/test_chat_approvals_and_resume.py``, so ``set_batch`` cannot
``await`` anything either -- see its own docstring for how row remounts
stay collision-safe without awaiting ``remove_children()``.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Select, Static

from tldw_chatbook.MCP.redaction import redact_mapping

#: Per-row decision options, in display order. Values are the exact
#: decision strings `MCPToolProvider._apply_verdict` consumes.
_DECISION_OPTIONS: list[tuple[str, str]] = [
    ("Approve once", "approve_once"),
    ("Approve for session", "approve_session"),
    ("Always allow", "always_allow"),
    ("Deny", "deny"),
]
_DEFAULT_DECISION = "approve_once"


def _options_for_row(call: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Decision options for one row, honoring an optional ``options`` key.

    Rows that omit ``options`` (every MCP row) get the full set, so MCP
    behavior is unchanged. A row may narrow it -- built-in tools offer
    only the session-scoped choices in P1, because persistent decisions
    for them cannot yet be undone in the UI. Unknown values are dropped,
    and an empty result falls back to the full set rather than rendering
    an unusable empty ``Select``.
    """
    requested = call.get("options") if isinstance(call, Mapping) else None
    if not isinstance(requested, (list, tuple)) or not requested:
        return _DECISION_OPTIONS
    wanted = set(requested)
    narrowed = [pair for pair in _DECISION_OPTIONS if pair[1] in wanted]
    return narrowed or _DECISION_OPTIONS


#: Reason-badge suffixes appended to a row's header line.
_REASON_SUFFIXES: dict[str, str] = {
    "config_changed": " (definition changed)",
    "risk_floored": " (high risk)",
}

_ARGS_SUMMARY_LIMIT = 80


def _collapse_pending_calls(calls: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Collapse ``calls`` to one entry per unique ``llm_name``, first-seen order.

    Matches T3's decisions-keyed-by-llm_name contract: same-name calls in
    one turn share a single verdict, so the batch card only ever needs one
    row per unique name. Each returned entry carries a ``count`` of how
    many original calls shared that name (for the row's "×N" suffix).
    """
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for call in calls:
        name = str(call.get("llm_name", ""))
        if name not in grouped:
            entry = dict(call)
            entry["count"] = 1
            grouped[name] = entry
            order.append(name)
        else:
            grouped[name]["count"] += 1
    return [grouped[name] for name in order]


def _format_row_header(entry: Mapping[str, Any]) -> str:
    """Return one row's header line: ``"server · tool"`` (+ ×N, + reason badge)."""
    server_label = str(entry.get("server_label", "") or "")
    tool_name = str(entry.get("tool_name", "") or "")
    header = f"{server_label} · {tool_name}"
    count = int(entry.get("count", 1) or 1)
    if count > 1:
        header += f" ×{count}"
    header += _REASON_SUFFIXES.get(str(entry.get("reason", "") or ""), "")
    return header


def _summarize_arguments(arguments: Mapping[str, Any] | None) -> str:
    """Return a compact, ``markup=False``-safe argument summary, capped at 80 chars.

    Secret-looking values (``api_key``, ``token``, ``password``, ...) are
    redacted before rendering -- redaction parity with every other MCP
    display/log boundary (see ``tldw_chatbook.MCP.redaction``'s module
    docstring); the approval card is the one place a raw secret argument
    was still reaching the screen unredacted.
    """
    try:
        text = json.dumps(
            redact_mapping(dict(arguments or {})),
            default=str,
            separators=(",", ":"),
        )
    except Exception:  # noqa: BLE001 -- a non-serializable arg must never crash rendering
        text = str(arguments or {})
    if len(text) > _ARGS_SUMMARY_LIMIT:
        return text[: _ARGS_SUMMARY_LIMIT - 1] + "…"
    return text


class ChatApprovalCard(Container):
    """Inline approval card for privileged agent actions."""

    class ApprovalDecided(Message):
        """Posted when the user submits per-row decisions for a pending batch."""

        def __init__(self, decisions: dict[str, str]) -> None:
            self.decisions = decisions
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

    def compose(self) -> ComposeResult:
        yield Static("Approval required", id="approval-title")
        with Container(id="approval-single-body"):
            yield Static("", id="approval-summary")
            yield Static("", id="approval-copy")
            with Horizontal(id="approval-actions"):
                yield Button("Allow once", id="approval-allow-once", variant="primary")
                yield Button("Deny", id="approval-deny", variant="error")
                yield Button("Review details", id="approval-details")
        with Container(id="approval-batch-body"):
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

    def on_mount(self) -> None:
        self.display = False
        self.query_one("#approval-batch-body").display = False

    # -- legacy single-approval API (unchanged; kept for existing callers) --

    def set_approval(self, approval: dict[str, Any] | None) -> None:
        """Update the card with the latest single approval request."""
        has_approval = bool(approval)
        self.display = has_approval
        if not has_approval:
            return

        self.query_one("#approval-batch-body").display = False
        self.query_one("#approval-single-body").display = True

        summary = approval.get("summary", "Approval required")
        details = approval.get("details", "")
        allow_label = approval.get("allow_label", "Allow once")

        self.query_one("#approval-summary", Static).update(summary)
        self.query_one("#approval-copy", Static).update(details)
        self.query_one("#approval-details", Button).label = approval.get(
            "details_label", "Review details"
        )
        self.query_one("#approval-allow-once", Button).label = allow_label

    # -- batch-approval API (task-5) -----------------------------------------

    def set_batch(self, calls: list[dict[str, Any]], *, timeout_seconds: float) -> None:
        """Render one row per unique ``llm_name`` in ``calls``.

        Synchronous throughout -- see the module docstring for why this
        cannot ``await``. Old rows are pruned via a fire-and-forget
        ``remove_children()`` (Textual 8.2.7 defers the actual detachment
        to the next event-loop tick -- see ``Widget.remove_children``'s
        ``AwaitRemove``/``App._prune``), while every new row gets an id
        tagged with a fresh, monotonically increasing generation number.
        That makes a still-pruning previous batch's ids structurally
        unable to collide with the incoming batch's ids, without this
        method ever needing to await the removal.
        """
        if not calls:
            self.display = False
            self.query_one("#approval-batch-body").display = False
            self._batch_names = []
            self._batch_selects = []
            self._batch_legal_values = []
            return

        self.display = True
        self.query_one("#approval-single-body").display = False
        self.query_one("#approval-batch-body").display = True

        grouped = _collapse_pending_calls(calls)
        self._batch_generation += 1
        generation = self._batch_generation
        names: list[str] = []
        selects: list[Select] = []
        legal_values: list[list[str]] = []
        rows: list[Horizontal] = []
        for index, entry in enumerate(grouped):
            names.append(str(entry.get("llm_name", "")))
            row_options = _options_for_row(entry)
            row_values = [value for _label, value in row_options]
            default_value = (
                _DEFAULT_DECISION
                if _DEFAULT_DECISION in row_values
                else row_options[0][1]
            )
            select = Select(
                row_options,
                value=default_value,
                allow_blank=False,
                classes="approval-row-decision",
            )
            selects.append(select)
            legal_values.append(row_values)
            rows.append(
                Horizontal(
                    Static(
                        _format_row_header(entry),
                        markup=False,
                        classes="approval-row-header",
                    ),
                    Static(
                        _summarize_arguments(entry.get("arguments")),
                        markup=False,
                        classes="approval-row-args",
                    ),
                    select,
                    id=f"approval-row-{generation}-{index}",
                    classes="approval-row",
                )
            )
        self._batch_names = names
        self._batch_selects = selects
        self._batch_legal_values = legal_values

        rows_container = self.query_one("#approval-batch-rows", Vertical)
        rows_container.remove_children()
        if rows:
            rows_container.mount(*rows)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "approval-approve-all":
            event.stop()
            self._set_all_batch_decisions(("approve_once", "approve_session"))
        elif button_id == "approval-deny-all":
            event.stop()
            self._set_all_batch_decisions(("deny",))
        elif button_id == "approval-submit":
            event.stop()
            self._submit_batch_decisions()

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
        """
        for select, legal_values in zip(self._batch_selects, self._batch_legal_values):
            for candidate in candidates:
                if candidate in legal_values:
                    select.value = candidate
                    break

    def _submit_batch_decisions(self) -> None:
        decisions = {
            name: select.value
            for name, select in zip(self._batch_names, self._batch_selects)
        }
        self.post_message(self.ApprovalDecided(decisions))
