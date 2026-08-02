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

import json
from typing import Any, Mapping, Sequence

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.css.query import NoMatches
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

_ARGS_SUMMARY_LIMIT = 80

#: TASK-1845: needs-decision was a border + 10% tint with no text change, so
#: the state vanished in monochrome. PRODUCT.md: "colour must never be the
#: only carrier of meaning."
NEEDS_DECISION_PREFIX = "needs decision · "


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
            # TASK-1845: keep EVERY call's arguments, not just the first.
            # Grouping by name is the verdict contract and stays; hiding the
            # other calls' targets behind the count was the hazard -- three
            # reads of three different files rendered as one, so the user
            # approved three things having seen one.
            entry["all_arguments"] = [call.get("arguments")]
            grouped[name] = entry
            order.append(name)
        else:
            grouped[name]["count"] += 1
            grouped[name].setdefault("all_arguments", []).append(
                call.get("arguments")
            )
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


def _summarize_arguments(arguments: Mapping[str, Any] | None) -> str:
    """Return a compact, ``markup=False``-safe argument summary.

    TASK-1845: when handed a COLLAPSED entry (one carrying ``all_arguments``),
    every call's arguments are rendered, one per line -- a row that says "x3"
    must show all three targets or the count is concealing the decision. Each
    line is capped independently so one long payload cannot push the others
    off screen. Redaction is applied to every line, not just the first.

    Secret-looking values (``api_key``, ``token``, ``password``, ...) are
    redacted before rendering -- redaction parity with every other MCP
    display/log boundary (see ``tldw_chatbook.MCP.redaction``'s module
    docstring); the approval card is the one place a raw secret argument
    was still reaching the screen unredacted.
    """
    def _one(payload: Any) -> str:
        try:
            text = json.dumps(
                redact_mapping(dict(payload or {})),
                default=str,
                separators=(",", ":"),
            )
        except Exception:  # noqa: BLE001 -- a bad arg must never crash rendering
            text = str(payload or {})
        if len(text) > _ARGS_SUMMARY_LIMIT:
            return text[: _ARGS_SUMMARY_LIMIT - 1] + "…"
        return text

    if isinstance(arguments, Mapping) and "all_arguments" in arguments:
        sets = arguments.get("all_arguments") or []
        rendered = [_one(payload) for payload in sets]
        # De-duplicate identical payloads while preserving order: N identical
        # calls are one decision with one target, and repeating it N times
        # would bury a genuinely different target further down.
        seen: set[str] = set()
        unique = [r for r in rendered if not (r in seen or seen.add(r))]
        return "\n".join(unique)
    return _one(arguments)


class ChatApprovalCard(Container):
    """Inline approval card for privileged agent actions."""

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
        try:
            selects = list(self.query(".approval-row-decision"))
        except Exception:
            selects = []
        if selects:
            selects[0].focus()
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
        self._batch_rows: list[Horizontal] = []
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

    def compose(self) -> ComposeResult:
        yield Static("Approval required", id="approval-title")
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
        # Mount can fire before this widget's composed children are attached
        # to the DOM (observed as a NoMatches crash on '#approval-batch-body'
        # during Console screen mount), so defer the initial batch-body hide
        # until after the children have settled.
        self.call_after_refresh(self._hide_batch_body)

    def _hide_batch_body(self) -> None:
        try:
            self.query_one("#approval-batch-body").display = False
        except NoMatches:
            # The card is hidden anyway (display = False above); a missing
            # batch body at this point is harmless.
            pass

    # -- batch-approval API (task-5) -----------------------------------------

    def set_batch(
        self,
        calls: list[dict[str, Any]],
        *,
        timeout_seconds: float,
        round_id: str | None = None,
    ) -> None:
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
        """
        self._batch_round_id = round_id
        if not calls:
            self.display = False
            self.query_one("#approval-batch-body").display = False
            self._batch_names = []
            self._batch_selects = []
            self._batch_legal_values = []
            self._batch_rows = []
            self._batch_fast_buttons = []
            return

        self.display = True
        self.query_one("#approval-batch-body").display = True
        # task-1234 review round 1: a submit-shaped control (Submit, or
        # either fast button) disables itself right after a press to close
        # the double-submit window -- see `_disable_batch_submit_controls`.
        # A NEW batch must start every submitting control re-enabled,
        # otherwise a round whose PREDECESSOR was resolved via Submit would
        # render with a permanently-disabled Submit button.
        try:
            self.query_one("#approval-submit", Button).disabled = False
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
        rows: list[Horizontal] = []
        fast_buttons: list[Button] = []
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
            header_static = Static(
                _format_row_header(entry),
                markup=False,
                classes="approval-row-header",
            )
            header_tooltip = _row_header_tooltip(entry)
            if header_tooltip:
                header_static.tooltip = header_tooltip
            row_children: list[Any] = [
                header_static,
                Static(
                    _summarize_arguments(entry.get("arguments")),
                    markup=False,
                    classes="approval-row-args",
                ),
                select,
            ]
            if single_row:
                fast_approve = Button(
                    "Approve once",
                    id=f"approval-fast-approve-{generation}-{index}",
                    variant="success",
                    compact=True,
                    classes="approval-row-fast-approve",
                    tooltip=(
                        "Approve once and resume immediately "
                        "(skips Select + Submit)."
                    ),
                )
                fast_deny = Button(
                    "Deny",
                    id=f"approval-fast-deny-{generation}-{index}",
                    variant="error",
                    compact=True,
                    classes="approval-row-fast-deny",
                    tooltip=(
                        "Deny and resume immediately "
                        "(skips Select + Submit)."
                    ),
                )
                fast_buttons.extend((fast_approve, fast_deny))
                row_children.append(fast_approve)
                row_children.append(fast_deny)
            rows.append(
                Horizontal(
                    *row_children,
                    id=f"approval-row-{generation}-{index}",
                    classes="approval-row",
                )
            )
        self._batch_names = names
        self._batch_selects = selects
        self._batch_legal_values = legal_values
        self._batch_rows = rows
        self._batch_fast_buttons = fast_buttons

        rows_container = self.query_one("#approval-batch-rows", Vertical)
        rows_container.remove_children()
        if rows:
            rows_container.mount(*rows)

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
