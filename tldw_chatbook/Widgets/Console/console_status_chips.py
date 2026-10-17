"""Console status strip: one clickable text line (provider/model/assistant/
source/tool/approval plus the retrieval-scope, run, rate and cost segments).
The Library policy readout lives in the composer menu (TASK-35160 follow-up):
its segment left the line and the menu's "Library access..." entry opens the
same two-axis policy editor the segment click used to open.

TASK-35160: the strip used to compose ~11 bordered chip widgets -- each with
background, margin, padding and a 34-cell ellipsis cap, roughly 30-40 columns
of overhead that pushed labels off the row. The strip is now ONE focusable
``Static`` rendering a single markup line whose segments are ``[@click]``
spans: the same information as text, hover-underline on actionable segments,
and clicks that post the SAME typed messages the chips posted
(``OpenRequested`` / ``SaveRequested`` / ``ReviewRequested`` /
``ConsoleCostChipPressed``), so ``ChatScreen``'s ``@on`` handlers are
unchanged. The chip widget classes remain as message-carrier namespaces --
their names are the strip's public contract (imports in extensions, tests and
the screen all reference ``ConsoleRunChip.OpenRequested`` and friends).

Keyboard parity: the line is focusable; ``left``/``right`` move a visible
segment cursor, ``home``/``end`` jump, ``enter``/``space`` activate the
cursored segment (the cost breakdown modal's keyboard entry lives here, as
it did on the cost chip). Per-segment tooltips no longer exist (one widget,
one tooltip); segment detail lives in the destinations the segments open.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

from rich import markup as rich_markup
from rich.cells import cell_len
from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, HorizontalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_cost_tracker import ConsoleCostState
from tldw_chatbook.Chat.console_display_state import (
    CONSOLE_INSPECTOR_NO_APPROVAL_REASON,
    ConsoleControlState,
    ConsoleRetrievalScopeState,
)
from tldw_chatbook.Chat.console_ephemeral import TEMPORARY_LABEL, TEMPORARY_TOOLTIP
from tldw_chatbook.Chat.rag_scope import SCOPE_REASON_EMPTY, scope_empty_notice
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard
from tldw_chatbook.Widgets.Console.console_retrieval_scope_row import (
    UNSCOPED_LABEL as SCOPE_ROW_UNSCOPED_LABEL,
)


def _cost_label_forms(state: ConsoleCostState) -> list[str]:
    """Every form of the cost label, longest first: the full and compact
    labels, then each with whole " · " segments dropped from its end.

    Ported from the chip era (dev 3212e3984d's room-aware fitting): the
    cost segment picks the longest form that fits the room left on the
    line, and "…" only when none does.
    """
    forms: list[str] = []
    for label in (state.label, state.compact_label):
        parts = label.split(" · ")
        forms += (" · ".join(parts[:end]) for end in range(len(parts), 0, -1))
    return sorted(dict.fromkeys(forms), key=cell_len, reverse=True)


def _items_word(count: int) -> str:
    """Return the singular/plural scope-unit word for a count."""
    return "item" if count == 1 else "items"


# ---------------------------------------------------------------------------
# Segment message namespaces. These were focusable chip widgets until
# TASK-35160 collapsed the strip into one text line; the classes remain as
# the carriers of the strip's typed activation messages, because every
# ``ChatScreen`` handler, test and extension import references them by name.


class ConsoleChip:
    """Namespace root for the strip's activation messages (see module doc)."""

    class OpenRequested(Message):
        """Generic segment activation (subclassed per namespace below)."""


class ConsoleApprovalsChip:
    """Approvals segment: activation asks the strip to focus the approval card."""

    class ReviewRequested(Message):
        """Posted when the approvals segment is activated."""


class ConsoleModelChip:
    """Provider/model segment: activation opens Switch model (Alt+M's popover)."""

    class OpenRequested(Message):
        """Posted when a provider/model segment is activated."""


class ConsoleAssistantChip:
    """Assistant segment: activation opens the character picker."""

    class OpenRequested(Message):
        """Posted when the assistant/character segment is activated."""


class ConsoleSystemPromptChip:
    """System-prompt segment: activation opens the system-prompt editor."""

    class OpenRequested(Message):
        """Posted when the system-prompt segment is activated."""


class ConsoleScopeChip:
    """Retrieval-scope segment: activation opens the scope picker."""

    class OpenRequested(Message):
        """Posted when the scope segment is activated."""


class ConsoleSourcesChip:
    """Staged-sources segment: activation opens the Inspector rail."""

    class OpenRequested(Message):
        """Posted when the sources segment is activated from keyboard or mouse."""


class ConsoleToolsChip:
    """Tools-readiness segment: activation opens the Inspector rail."""

    class OpenRequested(Message):
        """Posted when the tools segment is activated from keyboard or mouse."""


class ConsoleRunChip:
    """Active-run segment: activation opens the Inspector rail."""

    class OpenRequested(Message):
        """Posted when the run segment is activated from keyboard or mouse."""


class ConsoleRateChip:
    """Settled tokens-per-second segment (TASK-35100). Display-only -- activation
    belongs to the run segment (Inspector) and cost segment (breakdown modal)."""


class ConsoleTemporaryChip:
    """Temporary-chat segment: activation saves the chat."""

    class SaveRequested(Message):
        """Posted when the temporary segment is activated."""


class ConsoleCostChip:
    """Cost segment: activation opens the cost breakdown modal (task-4)."""

    class ConsoleCostChipPressed(Message):
        """Posted when the cost segment is activated from keyboard or mouse."""


@dataclass(frozen=True)
class StatusSegment:
    """One rendered segment of the single-line status strip.

    Attributes:
        name: Stable identifier (drives the ``[@click]`` action argument and
            the keyboard cursor).
        label: Plain text rendered for the segment (markup-escaped on render).
        visible: ``False`` drops the segment from the line entirely (the old
            chips' display toggles).
        actionable: Whether activation/click posts the segment's message
            (``False`` for read-only segments like the rate badge).
        dim: Low-emphasis styling (inactive counters, normal cost).
        alert: High-emphasis styling (action-required counters, empty scope).
    """

    name: str
    label: str = ""
    visible: bool = True
    actionable: bool = True
    dim: bool = False
    alert: bool = False


class ConsoleStatusText(Static):
    """The whole status strip as one focusable, clickable text line.

    Segments arrive via :meth:`set_segments`; the line re-renders only when
    the composed markup changed. Actionable segments carry a
    ``[@click="open_segment('<name>')"]`` span -- bare content actions
    resolve on this widget (probe-verified on Textual 8.2.8) -- and post the
    legacy chip message classes, so the host screen's handlers are unchanged.
    Keyboard: ``left``/``right`` move a reverse-video cursor across the
    visible segments, ``home``/``end`` jump, ``enter``/``space`` activate.
    """

    can_focus = True

    BINDINGS = [
        Binding("left", "cursor_left", "Previous segment", show=False),
        Binding("right", "cursor_right", "Next segment", show=False),
        Binding("home", "cursor_home", "First segment", show=False),
        Binding("end", "cursor_end", "Last segment", show=False),
        Binding("enter", "activate_segment", "Open segment", show=False),
        Binding("space", "activate_segment", "Open segment", show=False),
        # Wheel parity for keyboard-only users: page the viewport without
        # moving the segment cursor (the cursor walks one segment per key
        # press and scroll-visible -- these move the VIEW itself).
        Binding("shift+right", "viewport_right", "Scroll strip right", show=False),
        Binding("shift+left", "viewport_left", "Scroll strip left", show=False),
    ]

    #: Segment name -> the message class activation posts. The same classes
    #: the chip widgets used to post, so screen handlers are untouched.
    SEGMENT_MESSAGES: dict[str, Callable[[], Message]] = {
        "temporary": ConsoleTemporaryChip.SaveRequested,
        "run": ConsoleRunChip.OpenRequested,
        "provider": ConsoleModelChip.OpenRequested,
        "model": ConsoleModelChip.OpenRequested,
        "system-prompt": ConsoleSystemPromptChip.OpenRequested,
        "assistant": ConsoleAssistantChip.OpenRequested,
        "sources": ConsoleSourcesChip.OpenRequested,
        "tools": ConsoleToolsChip.OpenRequested,
        "approvals": ConsoleApprovalsChip.ReviewRequested,
        "scope": ConsoleScopeChip.OpenRequested,
        "cost": ConsoleCostChip.ConsoleCostChipPressed,
    }

    def __init__(self, segments: list[StatusSegment], **kwargs: Any) -> None:
        super().__init__(markup=True, **kwargs)
        self._segments = list(segments)
        #: The cursored segment's NAME, not its index: segments appear and
        #: disappear between syncs (a run ends, tools hide at zero), and an
        #: index cursor would silently jump onto a different segment -- or
        #: index past the visible list and crash the render pass.
        self._cursor_name: str | None = None
        self._rendered_markup: str | None = None
        # Column spans of the VISIBLE segments in the plain line
        # ``(name, start, end)`` -- click/activation target metadata, and
        # the seam tests use to click individual segments.
        self.segment_spans: list[tuple[str, int, int]] = []

    # -- state ---------------------------------------------------------

    def set_segments(self, segments: list[StatusSegment]) -> None:
        """Replace segment state; re-render only if the composed line changed.

        A cursor on a segment that just vanished resets (there is nothing
        sensible to highlight); a cursor on a segment that only moved
        keeps its position by name.
        """
        self._segments = list(segments)
        if self._cursor_name is not None and not any(
            s.name == self._cursor_name for s in self.visible_segments
        ):
            self._cursor_name = None
        self._rerender()

    @property
    def visible_segments(self) -> list[StatusSegment]:
        """The segments currently on the line, in render order."""
        return [s for s in self._segments if s.visible and s.label]

    def _plain_label(self, segment: StatusSegment) -> str:
        """The segment's on-screen text (before styling), for span tracking."""
        return segment.label

    # -- rendering -----------------------------------------------------

    def _render_line(self) -> str:
        """Compose the markup line and refresh ``segment_spans``."""
        parts: list[str] = []
        spans: list[tuple[str, int, int]] = []
        column = 0
        visible = self.visible_segments
        for segment in visible:
            if column:
                parts.append(" · ")
                column += 3
            plain = self._plain_label(segment)
            start = column
            column += len(plain)
            spans.append((segment.name, start, start + len(plain)))
            body = rich_markup.escape(plain)
            if self._cursor_name == segment.name:
                body = f"[reverse]{body}[/reverse]"
            elif segment.dim:
                body = f"[dim]{body}[/dim]"
            elif segment.alert:
                body = f"[bold]{body}[/bold]"
            if segment.actionable and segment.name in self.SEGMENT_MESSAGES:
                # The [@click] meta exists for the HOVER LINK STYLING only;
                # its action is a no-op because dispatch goes through
                # ``_on_click``'s span table (keeping the real action here
                # double-posted wherever the meta broker does resolve).
                body = f"[@click=\"segment_styled('{segment.name}')\"]{body}[/]"
            parts.append(body)
        self.segment_spans = spans
        return "".join(parts)

    def _rerender(self) -> None:
        """Recompose and update the widget when the line actually changed."""
        line = self._render_line()
        if line == self._rendered_markup:
            return
        self._rendered_markup = line
        self.update(line)

    # -- keyboard ------------------------------------------------------

    def _cursor_names(self) -> list[str]:
        return [segment.name for segment in self.visible_segments]

    @property
    def _cursor(self) -> int:
        """Index of the cursored segment in the visible list (-1 = none)."""
        names = self._cursor_names()
        return names.index(self._cursor_name) if self._cursor_name in names else -1

    def _move_cursor(self, target: int) -> None:
        names = self._cursor_names()
        if not names:
            self._cursor_name = None
            self._rerender()
            return
        clamped = max(0, min(target, len(names) - 1))
        name = names[clamped]
        if name == self._cursor_name:
            return
        self._cursor_name = name
        self._rerender()
        self.scroll_visible(animate=False)
        self._notify_arrows()

    def on_focus(self) -> None:
        """Entering the strip parks the cursor on the first visible segment."""
        if self._cursor_name is None:
            self._move_cursor(0)

    def action_cursor_left(self) -> None:
        """Move the segment cursor one segment left."""
        self._move_cursor(max(0, self._cursor - 1))

    def action_cursor_right(self) -> None:
        """Move the segment cursor one segment right."""
        self._move_cursor(self._cursor + 1)

    def action_cursor_home(self) -> None:
        """Move the segment cursor to the first visible segment."""
        self._move_cursor(0)

    def action_cursor_end(self) -> None:
        """Move the segment cursor to the last visible segment."""
        self._move_cursor(len(self._cursor_names()) - 1)

    def action_viewport_right(self) -> None:
        """Page the strip viewport right without moving the cursor."""
        scroller = self._scroll_container()
        if scroller is not None and scroller.max_scroll_x > 0:
            scroller.scroll_relative(
                x=self.PAGE_COLUMNS, animate=False
            )
            self._notify_arrows()

    def action_viewport_left(self) -> None:
        """Page the strip viewport left without moving the cursor."""
        scroller = self._scroll_container()
        if scroller is not None and scroller.max_scroll_x > 0:
            scroller.scroll_relative(
                x=-self.PAGE_COLUMNS, animate=False
            )
            self._notify_arrows()

    def action_activate_segment(self) -> None:
        """Activate the cursored segment (keyboard parity with the old chips)."""
        if self._cursor_name is not None:
            self._post_for(self._cursor_name)

    # -- activation ----------------------------------------------------

    #: Columns a single wheel tick scrolls the strip's viewport.
    WHEEL_COLUMNS = 8
    #: Columns a click of the ◀ ▶ arrows (or a Shift+arrow press) pages the
    #: strip's viewport.
    PAGE_COLUMNS = 32

    def _scroll_container(self) -> HorizontalScroll | None:
        """The nearest scrolling ancestor (the strip's viewport shell)."""
        ancestor = self.parent
        while ancestor is not None and not isinstance(ancestor, HorizontalScroll):
            ancestor = ancestor.parent
        return ancestor

    def _notify_arrows(self) -> None:
        """Let the strip re-dim its ◀ ▶ arrows after this widget scrolled."""
        ancestor = self.parent
        while ancestor is not None and not isinstance(ancestor, ConsoleStatusChips):
            ancestor = ancestor.parent
        if ancestor is not None:
            ancestor._update_arrow_state()

    def _on_resize(self, event: events.Resize) -> None:
        """Arrows re-evaluate after layout: scroll ranges only exist once
        both the line and its viewport have their real sizes, so the check
        runs one refresh cycle after this resize lands."""
        self.call_after_refresh(self._notify_arrows)

    def _wheel_scroll(self, event: events.MouseScrollDown | events.MouseScrollUp) -> None:
        """Translate a plain vertical wheel tick into horizontal scrolling.

        The strip's viewport only ever scrolls horizontally, so the
        up/down wheel (which a ``HorizontalScroll`` ignores) is remapped:
        down scrolls right, up scrolls left. A tick is a few columns --
        the line is text, not a grid, and the segment cursor (or a
        shift+wheel) remains the precise instruments. When there is
        nothing to scroll the event is left to bubble untouched.
        """
        scroller = self._scroll_container()
        if scroller is None or scroller.max_scroll_x <= 0:
            return
        event.stop()
        columns = self.WHEEL_COLUMNS * max(1, int(round(abs(event.delta_y))))
        if isinstance(event, events.MouseScrollDown):
            scroller.scroll_relative(x=columns, animate=False)
        else:
            scroller.scroll_relative(x=-columns, animate=False)
        self._notify_arrows()

    def _on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        self._wheel_scroll(event)

    def _on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        self._wheel_scroll(event)

    async def _on_click(self, event: events.Click) -> None:
        """Route a click to the segment under the pointer.

        Dispatch goes through OUR span table, not Textual's ``@click`` meta
        broker: the markup stays in the line for the hover link styling, but
        meta dispatch proved unreliable here (the focus-driven cursor
        re-render between mousedown and click can swap the content the
        click's style snapshot pointed at). The span table is authoritative
        -- this widget composed the line.

        ``event.x`` is widget-relative; the spans are line columns and the
        line starts at the widget's left edge, so they align directly.
        """
        event.stop()
        for name, start, end in self.segment_spans:
            if start <= event.x < end:
                self._post_for(name)
                return

    def action_open_segment(self, name: str) -> None:
        """The historical content-action entry point: post the segment's message."""
        self._post_for(name)

    def action_segment_styled(self, name: str) -> None:
        """No-op target of the line's ``[@click]`` meta (hover styling only).

        Dispatch is ``_on_click``'s span table; a real action here double-
        posted wherever Textual's meta broker also resolved the click.
        """

    def _post_for(self, name: str) -> None:
        message_factory = self.SEGMENT_MESSAGES.get(name)
        if message_factory is None:
            return
        self.post_message(message_factory())


class ConsoleStatusChips(Horizontal):
    """Full-width status strip: one clickable text line of Console readiness
    segments (provider/model/assistant/RAG/source/tool/approval plus the
    retrieval-scope, run, rate and cost segments).

    TASK-2154.5 (LY-03): the expanded presentation's inner strip scrolls
    horizontally when the line outgrows the viewport instead of silently
    clipping it -- keyboard reachability comes from the text widget's own
    segment cursor plus focus auto-scroll, mouse from Shift+wheel /
    trackpad swipe; the scrollbar itself is hidden (single-row strip,
    tab-strip precedent).

    Exposes ``sync_state`` so ``ChatScreen`` can refresh the segment labels
    and emphasis after provider/model/source/tool/approval state changes.
    The ``sync_*`` method set and the chip message namespaces are the
    strip's public contract and are preserved from the chip-widget era.
    """

    #: Generic per-segment detail no longer exists on one text widget (T1):
    #: the tooltip teaches interaction, and detail lives in the destinations.
    _LINE_TOOLTIP = (
        "Click a segment to open it · arrows move · Enter opens ·"
        " Shift+arrows scroll."
    )

    def __init__(
        self,
        state: ConsoleControlState,
        *,
        scope_state: ConsoleRetrievalScopeState | None = None,
        ephemeral: bool = False,
        cost_state: ConsoleCostState | None = None,
        run_copy: str = "",
        **kwargs: Any,
    ) -> None:
        """Initialize the strip.

        Args:
            state: Display-state snapshot for the readiness segments.
            scope_state: Display-state snapshot for the "Scope" segment
                (task-10). ``None`` renders as unscoped (hidden).
            ephemeral: Whether the active session is temporary at
                construction time (final-review F1) -- the Temporary segment
                renders on the first frame rather than waiting for a
                post-mount ``sync_temporary_chip`` call.
            cost_state: Display-state snapshot for the cost segment (task-4,
                PR3 cost ticker) -- same F1 precedent. ``None`` renders
                hidden (non-Console-native contexts have no cost to show).
            run_copy: Active-run copy for the run segment (TASK-2154.18,
                FB-08) -- same F1 precedent. ``""`` renders hidden.
            **kwargs: Additional Textual widget arguments (id/classes).
        """
        classes = kwargs.pop("classes", "")
        # Reuse the existing chip-row class so its CSS continues to apply.
        super().__init__(
            classes=f"console-control-chip-row console-status-chips {classes}".strip(),
            **kwargs,
        )
        self.state = state
        self.scope_state = scope_state
        self.ephemeral = ephemeral
        self._cost_state = cost_state
        # The strip's max_scroll_x and width, and the cost chip's width, at
        # the last layout check (see _keep_cost_chip_in_view).
        self._cost_strip_end = 0.0
        self._cost_strip_width = 0
        self._cost_chip_width = 0
        # What places the cost chip on the strip at the last layout of this
        # screen (see _refit_after_layout).
        self._cost_chip_place: tuple[object, ...] | None = None
        self._run_chip_state: tuple[bool, str] = (bool(run_copy), run_copy)
        # TASK-35100: settled-only badge, hidden at mount (nothing has
        # settled yet); the live rate rides the run segment instead.
        self._rate_chip_state: tuple[bool, str, str] = (False, "", "")
        self.remove_class(*(name for name in self.classes if name.startswith("h-")))
        self.set_styles(height=None)
        self.add_class("h-1")
        self.styles.min_height = 1
        self.styles.max_height = 1

    # -- segment state -> ConsoleStatusText -----------------------------

    def _status_text(self) -> ConsoleStatusText:
        """The one-line text widget (only child of the scroll shell)."""
        return self.query_one("#console-status-text", ConsoleStatusText)

    def _run_segment(self) -> StatusSegment:
        label, _tooltip, hidden = self._run_chip_render(*self._run_chip_state)
        return StatusSegment(
            name="run", label=label, visible=not hidden, actionable=True
        )

    def _rate_segment(self) -> StatusSegment:
        label, _tooltip, hidden = self._rate_chip_render(*self._rate_chip_state)
        # Display-only by design: activation belongs to the run segment
        # (Inspector) and the cost segment (breakdown modal).
        return StatusSegment(
            name="rate",
            label=label,
            visible=bool(label) and not hidden,
            actionable=False,
        )

    def _cost_segment(self) -> StatusSegment:
        label, _tooltip, hidden, alert, cold = self._cost_chip_render(
            self._cost_state
        )
        if self._cost_state is not None:
            label = (
                self._cost_state.compact_label
                if self.screen.size.width < 120
                else self._cost_state.label
            )
        return StatusSegment(
            name="cost",
            label=label,
            visible=not hidden,
            actionable=True,
            # Line emphasis is two-valued (alert vs dim): the chip era's
            # third "cold" state carried an info COLOR, which a markup span
            # cannot express token-legally -- cold and normal both read dim,
            # and the state itself stays on the cost state (tooltip/modal).
            dim=bool(self._cost_state) and not alert,
            alert=alert,
        )

    def _temporary_segment(self) -> StatusSegment:
        label, _tooltip, hidden = self._temporary_chip_render(self.ephemeral)
        return StatusSegment(
            name="temporary", label=label, visible=not hidden, actionable=True
        )

    def _scope_segment(self) -> StatusSegment:
        label, _tooltip, hidden, alert = self._scope_chip_render(self.scope_state)
        return StatusSegment(
            name="scope",
            label=label,
            visible=not hidden,
            actionable=True,
            alert=alert,
        )

    def _build_segments(self) -> list[StatusSegment]:
        """The full segment list, in the strip's order (see old compose)."""
        state = self.state
        return [
            self._temporary_segment(),
            # TASK-2154.18 (FB-08): the run segment sits left-most so it
            # stays visible when the strip scrolls horizontally.
            self._run_segment(),
            # TASK-35100: the settled rate badge rides with the transient
            # cluster, in the slot the run segment vacates at terminal.
            self._rate_segment(),
            StatusSegment(name="provider", label=state.provider_label),
            StatusSegment(name="model", label=state.model_label),
            StatusSegment(name="system-prompt", label=state.system_prompt_label),
            StatusSegment(name="assistant", label=state.assistant_label),
            StatusSegment(
                name="sources",
                label=state.sources_label,
                dim=not state.sources_active,
                alert=state.sources_active,
            ),
            # TASK-2154.12 (TX-04): hidden entirely at a zero tool count --
            # the old "Tools: not loaded" placeholder exposed a lazy-loading
            # implementation detail (Console UX review 2026-08).
            StatusSegment(
                name="tools",
                label=state.tools_label,
                visible=state.tools_active,
                dim=not state.tools_active,
                alert=state.tools_active,
            ),
            StatusSegment(
                name="approvals",
                label=state.approvals_label,
                dim=not state.approvals_active,
                alert=state.approvals_active,
            ),
            self._scope_segment(),
            # task-4 (PR3 cost ticker): last on the line, hidden entirely
            # when there is no cost state (see ``_cost_chip_render``).
            self._cost_segment(),
        ]

    def _refresh_text(self) -> None:
        """Rebuild all segments from state and push them at the line."""
        try:
            text_widget = self._status_text()
        except NoMatches:
            return
        text_widget.set_segments(self._build_segments())
        self._update_arrow_state()

    # -- shell ----------------------------------------------------------

    #: Visible only while the line overflows its viewport; each click pages
    #: the line by ``ConsoleStatusText.PAGE_COLUMNS``.
    def _scroll_scroller(self) -> HorizontalScroll | None:
        try:
            return self.query_one("#console-status-chip-scroll", HorizontalScroll)
        except NoMatches:
            return None

    def _page_viewport(self, direction: int) -> None:
        """Page the line's viewport one step left (-1) or right (+1)."""
        scroller = self._scroll_scroller()
        if scroller is None or scroller.max_scroll_x <= 0:
            return
        scroller.scroll_relative(
            x=direction * ConsoleStatusText.PAGE_COLUMNS, animate=False
        )
        self._update_arrow_state()

    def _update_arrow_state(self) -> None:
        """Request an arrow-state refresh on the next refresh cycle.

        Deferred deliberately, and scheduled on the SCROLLER's own message
        pump when one exists: every scroll path (wheel, cursor walk,
        paging, shift+arrows) applies its scroll through that pump, so a
        callback queued there runs strictly after the new position lands.
        Scheduling on the strip raced it and read zeros.
        """
        scroller = self._scroll_scroller()
        target = scroller if scroller is not None else self
        target.call_after_refresh(self._apply_arrow_state)

    def _apply_arrow_state(self) -> None:
        """Dim/disable the arrows at the line's ends; hide them when it fits."""
        try:
            left = self.query_one("#console-status-scroll-left", Button)
            right = self.query_one("#console-status-scroll-right", Button)
        except NoMatches:
            return
        scroller = self._scroll_scroller()
        if scroller is None:
            return
        # Mount ordering race: the line can report scroll range 0 before its
        # first real layout even when it overflows. The line's own width is
        # the ground truth -- if it exceeds the viewport, one more deferred
        # pass will see the settled range (self-limiting: once laid out the
        # condition goes false and the loop stops).
        line = self._status_text()
        if (
            scroller.max_scroll_x <= 0
            and line.region.width > scroller.content_region.width
        ):
            self.call_after_refresh(self._apply_arrow_state)
            return
        position = max(scroller.scroll_x, scroller.scroll_target_x)
        fits = scroller.max_scroll_x <= 0
        left.display = not fits
        right.display = not fits
        left.disabled = fits or position <= 0
        right.disabled = fits or position >= scroller.max_scroll_x

    @on(Button.Pressed, "#console-status-scroll-left")
    def _scroll_line_left(self, event: Button.Pressed) -> None:
        """Page the status line left (mouse affordance, TASK-35160)."""
        event.stop()
        self._page_viewport(-1)

    @on(Button.Pressed, "#console-status-scroll-right")
    def _scroll_line_right(self, event: Button.Pressed) -> None:
        """Page the status line right (mouse affordance, TASK-35160)."""
        event.stop()
        self._page_viewport(1)

    def compose(self) -> ComposeResult:
        # TASK-35160 follow-up: the Status collapsible is gone. The strip is
        # one compact text line; hiding it hid information, and the
        # collapse/expand button pair cost a tab stop, a second presentation,
        # and persisted config state for no remaining value. The scroll
        # shell composes directly, flanked by the ◀ ▶ click-to-page arrows.
        left_arrow = Button("◀", id="console-status-scroll-left",
                            classes="console-status-arrow", compact=True)
        right_arrow = Button("▶", id="console-status-scroll-right",
                             classes="console-status-arrow", compact=True)
        for arrow in (left_arrow, right_arrow):
            # Click-only affordances: keyboard users page with Shift+arrows
            # on the line, so the arrows add no tab stops.
            arrow.can_focus = False
            arrow.tooltip = "Scroll the status line."
            arrow.remove_class(
                *(name for name in arrow.classes if name.startswith("w-"))
            )
            arrow.set_styles(width=None)
            # Button's default line_pad would eat both content cells at
            # width 2 and crash the render (chop_cells at width 0) -- the
            # same fix the old collapse button carried.
            arrow.styles.line_pad = 0
        left_arrow.disabled = True  # the line starts at its left end
        right_arrow.display = False  # arrow state settles on the first sync

        chip_scroll = HorizontalScroll(id="console-status-chip-scroll")
        # The viewport shell is NOT a tab stop -- the line owns keyboard
        # traversal (segment cursor + shift+arrows viewport paging).
        chip_scroll.can_focus = False
        chip_scroll.remove_class(
            *(name for name in chip_scroll.classes if name.startswith("h-"))
        )
        chip_scroll.set_styles(height=None)
        chip_scroll.add_class("h-1")
        chip_scroll.styles.min_height = 1
        chip_scroll.styles.max_height = 1
        chip_scroll.styles.scrollbar_size_horizontal = 0
        with chip_scroll:
            status_text = ConsoleStatusText(
                self._build_segments(),
                id="console-status-text",
                classes="console-status-text",
            )
            status_text.tooltip = self._LINE_TOOLTIP
            yield status_text
        yield left_arrow
        yield right_arrow
        self._update_arrow_state()

    # -- renders (pure, kept from the chip era; labels + hidden flags) ---

    @staticmethod
    def _run_chip_render(visible: bool, copy: str) -> tuple[str, str, bool]:
        """Pure ``(label, tooltip, hidden)`` render for the run segment.

        Args:
            visible: Whether the viewed session's run status is active
                (the screen owns the ``CONSOLE_ACTIVE_RUN_STATUSES``
                membership call; the widget stays display-only).
            copy: The run's ``visible_copy`` (e.g. "Streaming response.").
        """
        if not visible:
            return "", "", True
        text = copy.strip() or "Working."
        return (
            f"Run: {text}",
            f"Active run: {text} Open the Inspector for run details.",
            False,
        )

    def sync_run_chip(self, visible: bool, copy: str) -> None:
        """Refresh the run segment's visibility and copy (TASK-2154.18, FB-08).

        Deliberately NOT folded into ``sync_state``: run state changes on
        its own cadence (send/stop transitions and the 0.2s transcript
        poll while a run is active), independently of
        ``ConsoleControlState``. Equality-guarded so unchanged ticks are
        free.

        Args:
            visible: Whether the viewed session's run is in an active
                status. Terminal states hide the segment (their ambient
                signals live elsewhere -- see ``ConsoleRunChip``).
            copy: The run's ``visible_copy``; ignored when not visible.
        """
        next_state = (visible, copy if visible else "")
        if next_state == self._run_chip_state:
            return
        self._run_chip_state = next_state
        self._refresh_text()

    @staticmethod
    def _rate_chip_render(
        visible: bool, copy: str, tooltip: str
    ) -> tuple[str, str, bool]:
        """Pure ``(label, tooltip, hidden)`` render for the rate segment.

        TASK-35100 redesign: this segment only ever shows the SETTLED
        figure, in the slot the run segment vacates at terminal -- the
        live rate rides the run segment's copy, so the strip never gains a
        slot while streaming.

        Args:
            visible: Whether the tracker has a settled reading to show.
            copy: The compact label ("80.0 tok/s avg"), from
                ``Chat/console_token_rate.format_rate_chip_settled``.
            tooltip: The detail line (token total, measured window);
                detail lives in the destinations since TASK-35160.
        """
        if not visible:
            return "", "", True
        text = copy.strip()
        if not text:
            return "", "", True
        return (text, tooltip or f"Last response: {text}.", False)

    def sync_rate_chip(self, visible: bool, copy: str, tooltip: str = "") -> None:
        """Refresh the rate segment's visibility and label (TASK-35100).

        Pushed by the screen on its own cadence (the 0.2s transcript
        poll); the whole-line equality guard makes unchanged ticks free.

        Args:
            visible: Whether the tracker has a settled reading to show.
            copy: The compact label; ignored when not visible.
            tooltip: The detail line; carried for API compatibility, not
                rendered (per-segment tooltips ended with TASK-35160).
        """
        next_state = (
            visible,
            copy if visible else "",
            tooltip if visible else "",
        )
        if next_state == self._rate_chip_state:
            return
        self._rate_chip_state = next_state
        self._refresh_text()

    @staticmethod
    def _temporary_chip_render(ephemeral: bool) -> tuple[str, str, bool]:
        """Pure ``(label, tooltip, hidden)`` render for the temporary segment.

        Args:
            ephemeral: Whether the active session is temporary.

        Returns:
            ``label``: segment text. ``tooltip``: hover text, which is
            where the save affordance is spelled out. ``hidden``: ``True``
            for a normal chat -- a "Saved" segment on every ordinary
            conversation would be noise, and the strip is width-bounded.
        """
        if not ephemeral:
            return TEMPORARY_LABEL, TEMPORARY_TOOLTIP, True
        return TEMPORARY_LABEL, TEMPORARY_TOOLTIP, False

    def sync_temporary_chip(self, ephemeral: bool) -> None:
        """Refresh the temporary segment from the active session's flag.

        Separate from ``sync_state`` for the same reason ``sync_scope_chip``
        is: this is pushed from the screen when the active session changes,
        not on every control-bar sync tick.

        Args:
            ephemeral: Whether the active session is temporary.
        """
        if ephemeral == self.ephemeral:
            return
        self.ephemeral = ephemeral
        self._refresh_text()

    @staticmethod
    def _scope_chip_render(
        state: ConsoleRetrievalScopeState | None,
    ) -> tuple[str, str, bool, bool]:
        """Pure ``(label, tooltip, hidden, alert)`` render for the scope segment.

        ``item_count`` is always the EFFECTIVE (post-intersection) count
        (task-13). The tooltip spells the active scope levels out in words
        (TASK-2154.12/TX-03): a single active level reads as "Only
        searching: conversation scope (N items)"; both active levels read
        as the full breakdown with the shared count named in words.

        Args:
            state: Display-state snapshot, or ``None`` (renders unscoped).

        Returns:
            ``label``: segment text. ``tooltip``: detail text (the EMPTY
            branch folds the plain-language cause in; carried for API
            compatibility, not rendered since TASK-35160). ``hidden``:
            ``True`` when unscoped (hidden rather than shown as
            "everything", matching the brief). ``alert``: ``True`` only
            for EMPTY.
        """
        if state is None or (not state.is_scoped and not state.is_empty):
            return SCOPE_ROW_UNSCOPED_LABEL, "", True, False
        if state.is_empty:
            return (
                "Scope: no sources",
                scope_empty_notice(state.cause or SCOPE_REASON_EMPTY),
                False,
                True,
            )
        label = f"Scope: {state.item_count}"
        if state.conv_item_count is not None and state.ws_item_count is not None:
            tooltip = (
                f"Only searching: conversation scope "
                f"({state.conv_item_count} {_items_word(state.conv_item_count)}) "
                f"and workspace scope "
                f"({state.ws_item_count} {_items_word(state.ws_item_count)}) — "
                f"{state.item_count} in both."
            )
        elif state.ws_item_count is not None:
            tooltip = (
                f"Only searching: workspace scope "
                f"({state.ws_item_count} {_items_word(state.ws_item_count)})."
            )
        else:
            tooltip = (
                f"Only searching: conversation scope "
                f"({state.item_count} {_items_word(state.item_count)})."
            )
        return label, tooltip, False, False

    def sync_scope_chip(self, scope_state: ConsoleRetrievalScopeState | None) -> None:
        """Refresh the "Scope" segment from a new snapshot (task-10).

        Deliberately NOT folded into ``sync_state``: pushed directly from
        ``ChatScreen._sync_console_retrieval_scope_row`` with the exact
        same ``ConsoleRetrievalScopeState`` instance passed to the
        Inspector row's own ``sync_state`` in the same call -- one state,
        two renderers, computed once.

        Args:
            scope_state: Updated display-state snapshot to render.
        """
        if scope_state == self.scope_state:
            return
        self.scope_state = scope_state
        self._refresh_text()

    @staticmethod
    def _cost_chip_render(
        state: ConsoleCostState | None,
    ) -> tuple[str, str, bool, bool, bool]:
        """Pure ``(label, tooltip, hidden, alert, cold)`` render for the cost segment.

        Args:
            state: Display-state snapshot, or ``None`` (hides the segment --
                non-Console-native contexts have no cost to show).

        Returns:
            ``label``: segment text (the full ``state.label``; the
            room-aware form selection applies after layout via
            ``_apply_cost_segment_forms``). ``tooltip``: hover text
            (carried for API compatibility, not rendered since TASK-35160
            -- the breakdown modal carries the provenance). ``hidden``:
            ``True`` when ``state`` is ``None``. ``alert``/``cold``:
            ``state.alert``/``state.cold``.
        """
        if state is None:
            return "", "", True, False, False
        return state.label, state.tooltip, False, state.alert, state.cold

    def sync_cost_state(self, state: ConsoleCostState | None) -> None:
        """Refresh the cost segment from a new cost-state snapshot (task-4).

        Deliberately NOT folded into ``sync_state`` below, for the same
        reason ``sync_scope_chip`` isn't: cost state is computed and
        pushed on its own cadence (the general control-bar sync tick, but
        independently of whether ``ConsoleControlState`` itself changed),
        so it owns its own equality guard.

        Args:
            state: Updated cost snapshot, or ``None`` to hide the segment.
        """
        if state == self._cost_state:
            return
        self._cost_state = state
        self._refresh_text()

    def _apply_cost_segment_forms(self) -> None:
        """Fit the cost segment: pick the longest label form that fits.

        Port of the chip era's room-aware fitting (dev 3212e3984d, capture
        fix 3 finding 4): segments before the cost segment leave less room
        than the label needs, and the line cut it mid-word ("Current $0."
        at 211 columns). The cost segment is LAST on the line, so its room
        is the viewport width minus its start column; the longest form
        whose whole " · " segments fit is shown, and "…" when none does.
        Runs deferred so the spans reflect the just-rendered line.
        """
        try:
            line = self._status_text()
            scroller = self._scroll_scroller()
        except NoMatches:
            return
        state = self._cost_state
        if state is None:
            return
        cost_start = next(
            (
                start
                for name, start, _end in line.segment_spans
                if name == "cost"
            ),
            None,
        )
        if cost_start is None:
            return
        room = scroller.scrollable_content_region.width - cost_start
        if room <= 0:
            return
        forms = _cost_label_forms(state)
        label = next((f for f in forms if cell_len(f) <= room), "…")
        for segment in line._segments:
            if segment.name == "cost" and segment.label != label:
                line.set_segments(
                    [label if s.name == "cost" else s.label for s in line._segments]
                    and [
                        replace(s, label=label) if s.name == "cost" else s
                        for s in line._segments
                    ]
                )
                break

    def _refit_cost_segment(self) -> None:
        """Deferred cost-segment refit (layout watcher entry point)."""
        self.call_after_refresh(self._apply_cost_segment_forms)

    def on_layout_refresh(self, _event: object) -> None:
        """Refit the cost segment after a layout of this screen moved it.

        The one refit for a move: a segment before it that changes width,
        shows or hides, and a viewport that resizes. Capture findings 4
        (02c): a refit the change scheduled itself runs after the ACTIVE
        screen's refresh, which can come before this screen lays out the
        new widths (under a dialog; or Switch model's result callback
        beating the segments' Layout messages), so it fitted from the old
        places. Still deferred: relabelled straight from here the segment
        kept its old width (a layout was pending on it).
        """
        self._refit_cost_segment()

    def on_mount(self) -> None:
        """Check the cost segment after any layout that changes the line's
        total width (Textual posts no Resize for that, only for a new strip
        size)."""
        strip = self.query_one("#console-status-chip-scroll", HorizontalScroll)
        self.watch(
            strip,
            "virtual_size",
            self._refit_cost_segment,
            init=False,
        )
        self.screen.screen_layout_refresh_signal.subscribe(
            self, self.on_layout_refresh
        )

    def on_resize(self, _event: events.Resize) -> None:
        """Refresh segments (and the cost segment's room-aware form).

        Args:
            _event: Textual resize notification; sizes are read from the
                widgets.
        """
        self._refresh_text()
        self._refit_cost_segment()

    def sync_state(self, state: ConsoleControlState) -> None:
        """Refresh segment labels and emphasis from a new snapshot."""
        if state == self.state:
            return
        self.state = state
        self._refresh_text()

    @on(ConsoleApprovalsChip.ReviewRequested)
    def on_approval_review_requested(
        self, event: ConsoleApprovalsChip.ReviewRequested
    ) -> None:
        """Focus the pending approval card in the transcript.

        Falls back to a notification when no approval is pending so the
        keyboard-only path never dead-ends silently.
        """
        event.stop()
        self._focus_pending_approval_card()

    def _focus_pending_approval_card(self) -> None:
        """Scroll the displayed approval card into view and focus its action."""
        try:
            cards = list(self.screen.query("#chat-approval-card"))
        except Exception:
            cards = []
        card = next(
            (
                candidate
                for candidate in cards
                if isinstance(candidate, ChatApprovalCard) and candidate.display
            ),
            None,
        )
        if card is None:
            self.app.notify(CONSOLE_INSPECTOR_NO_APPROVAL_REASON, severity="warning")
            return
        try:
            card.scroll_visible(animate=False)
        except Exception:
            pass
        # `set_batch` (the card's sole production entry point, task-914) is
        # the only body it ever renders, so a displayed card's action is
        # always its "Submit" button.
        try:
            card.focus_first_decision()
        except NoMatches:
            pass
