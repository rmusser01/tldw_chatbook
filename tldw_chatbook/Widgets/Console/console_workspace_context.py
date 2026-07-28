"""Console-native workspace context tray."""

from __future__ import annotations

from typing import Any

from rich.cells import cell_len
from rich.markup import escape as _escape_markup
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Input, Static

from tldw_chatbook.Chat.console_chat_models import (
    CONSOLE_RUN_MARKER_MEANINGS_BY_GLYPH,
)
from tldw_chatbook.Chat.console_glyphs import (
    GLYPH_COLLAPSED,
    GLYPH_EXPANDED,
)
from tldw_chatbook.Workspaces.conversation_browser_state import (
    console_conversation_status_detail,
    CONSOLE_DEFAULT_CONVERSATION_DETAIL,
    ConsoleConversationBrowserGroup,
    ConsoleConversationBrowserRow,
    ConsoleConversationBrowserSection,
    ConsoleConversationBrowserState,
)
from tldw_chatbook.Workspaces.display_state import (
    ConsoleWorkspaceContextState,
)
from tldw_chatbook.Widgets.recompose_capture_guard import RecomposeCaptureGuard


# One vocabulary for persisted-but-not-archived chats across surfaces:
# "saved chat". Rows reach this map with either a workspace-membership role
# ("workspace-thread"/"workspace") or a persisted conversation state
# ("in-progress" is the default state normalize_conversation_row assigns) --
# all of them mean the same thing to the user: a chat that is saved locally
# and not currently open in a tab. Library Browse ▸ Conversations lists the
# same records, so these labels must not contradict its copy.
_STATUS_LABELS = {
    "workspace-thread": "saved",
    "workspace": "saved",
    "in-progress": "saved",
    "active": "active",
    "open": "open",
}
# TASK-356: the "saved chat"/"active session"/"open session" detail vocabulary
# now lives once in conversation_browser_state.console_conversation_status_detail
# (which `_conversation_detail_status` below delegates to); the former local
# `_STATUS_DETAIL_LABELS` copy was removed to keep a single source of truth.
_CONVERSATION_BROWSER_HEADER_HEIGHT = 1
_CONVERSATION_BROWSER_EMPTY_COPY_HEIGHT = 1

_TITLE_WRAP_MAX_LINES = 2
_MIN_TITLE_WRAP_BUDGET = 10
_ROW_ELLIPSIS = "…"
# Shared fallback for a conversation with no usable title. Used by both the
# wrap helper and ``ConsoleWorkspaceContextTray._conversation_title`` so the
# two normalizations cannot drift.
_UNTITLED_CONVERSATION = "Untitled conversation"


def _cut_prefix_cells(text: str, budget: int) -> str:
    """Return the longest prefix of ``text`` that fits within ``budget`` cells."""
    used = 0
    for index, char in enumerate(text):
        width = cell_len(char)
        if used + width > budget:
            return text[:index]
        used += width
    return text


def truncate_console_row_cells(text: str, budget: int) -> str:
    """Truncate raw row text to at most ``budget`` terminal cells.

    Cell-aware (CJK/emoji safe). Appends an ellipsis only when truncation
    actually occurred. Operates on raw text -- markup escaping happens later
    in ``format_console_conversation_row_label``.

    Args:
        text: Raw (unescaped) row text to fit.
        budget: Maximum width in terminal cells; clamped to a floor of 1.

    Returns:
        The text unchanged when it already fits, otherwise the longest
        cell-measured prefix that leaves room for a trailing ellipsis, with
        the ellipsis appended.
    """
    budget = max(1, int(budget))
    text = str(text)
    if cell_len(text) <= budget:
        return text
    keep = _cut_prefix_cells(text, budget - cell_len(_ROW_ELLIPSIS))
    return f"{keep.rstrip()}{_ROW_ELLIPSIS}"


def wrap_console_conversation_title(title: str, budget: int) -> tuple[str, ...]:
    """Word-wrap a raw conversation title into at most two budget-width lines.

    Widths are measured in terminal cells, not characters. Spaceless tokens
    longer than one line hard-break at the budget. When two lines are still
    insufficient the second line is ellipsized. The budget is clamped to
    ``_MIN_TITLE_WRAP_BUDGET`` to avoid degenerate wraps on absurdly narrow
    rails. Blank titles normalize to ``_UNTITLED_CONVERSATION`` (the shared
    fallback ``ConsoleWorkspaceContextTray._conversation_title`` also uses).

    Args:
        title: Raw (unescaped) conversation title to wrap.
        budget: Target line width in terminal cells; clamped up to a floor of
            ``_MIN_TITLE_WRAP_BUDGET``.

    Returns:
        A tuple of one or two raw text lines, each at most ``budget`` cells
        wide; the second line carries a trailing ellipsis when the title
        still overflows two lines.
    """
    budget = max(_MIN_TITLE_WRAP_BUDGET, int(budget))
    remaining = str(title).strip() or _UNTITLED_CONVERSATION
    lines: list[str] = []
    while remaining:
        if len(lines) == _TITLE_WRAP_MAX_LINES - 1:
            lines.append(truncate_console_row_cells(remaining, budget))
            break
        if cell_len(remaining) <= budget:
            lines.append(remaining)
            break
        head = _cut_prefix_cells(remaining, budget)
        on_boundary = head.endswith(" ") or (
            len(head) < len(remaining) and remaining[len(head)] == " "
        )
        if on_boundary:
            lines.append(head.rstrip())
            remaining = remaining[len(head) :].lstrip()
            continue
        break_at = head.rfind(" ")
        if break_at > 0:
            lines.append(remaining[:break_at].rstrip())
            remaining = remaining[break_at + 1 :].lstrip()
        else:
            lines.append(head)
            remaining = remaining[len(head) :].lstrip()
    return tuple(lines)


def wrap_console_plain_text_uncapped(text: str, budget: int) -> tuple[str, ...]:
    """Word-wrap ``text`` into budget-width lines with NO cap on line count
    and no ellipsis.

    TASK-1142 round-2 review (Qodo, PR #1050): ``_empty_copy_line_count``
    used to call ``wrap_console_conversation_title``, which intentionally
    hard-caps at ``_TITLE_WRAP_MAX_LINES`` (2) and ellipsizes the remainder
    -- correct for a row TITLE (which really is rendered that way), but an
    empty-copy ``Static`` has no such cap: Rich wraps it to as many lines
    as it needs. Any empty-copy string that legitimately needs 3+ lines
    (a narrower rail, longer copy, a future localization) would silently
    undercount again through the capped helper, re-opening the exact
    clipping bug TASK-1142 round 1 fixed (a later section's header pushed
    out of the tray's own visible bounds). This is the SAME greedy
    word-wrap rule ``wrap_console_conversation_title`` uses per line
    (word-boundary wrap; a single token longer than one line hard-breaks
    at the budget) with the two-line cap removed, so it cannot disagree
    with that helper for text that already fits within the cap -- this is
    strictly what it would compute with the cap lifted.

    Args:
        text: Raw text to wrap (already-rendered plain copy, not a title --
            no blank-text fallback is applied).
        budget: Target line width in terminal cells; clamped up to a floor
            of ``_MIN_TITLE_WRAP_BUDGET``.

    Returns:
        A tuple of one or more raw text lines, each at most ``budget``
        cells wide (an over-length spaceless token is the sole exception,
        hard-broken at the budget instead). Empty for blank/whitespace-only
        input.
    """
    budget = max(_MIN_TITLE_WRAP_BUDGET, int(budget))
    remaining = str(text).strip()
    if not remaining:
        return ()
    lines: list[str] = []
    while remaining:
        if cell_len(remaining) <= budget:
            lines.append(remaining)
            break
        head = _cut_prefix_cells(remaining, budget)
        on_boundary = head.endswith(" ") or (
            len(head) < len(remaining) and remaining[len(head)] == " "
        )
        if on_boundary:
            lines.append(head.rstrip())
            remaining = remaining[len(head) :].lstrip()
            continue
        break_at = head.rfind(" ")
        if break_at > 0:
            lines.append(remaining[:break_at].rstrip())
            remaining = remaining[break_at + 1 :].lstrip()
        else:
            lines.append(head)
            remaining = remaining[len(head) :].lstrip()
    return tuple(lines)


def _marker_prefixed_name_lines(
    title: str, run_marker: str, budget: int
) -> tuple[str, ...]:
    """Wrap ``title`` and prefix the first line with its fleet run-marker glyph.

    Parallel-agents spec PA-T8: ``run_marker`` is already the resolved glyph
    string (``ConsoleConversationBrowserRow.run_marker``, empty for the
    steady state), so an unmarked row wraps exactly as it did before this
    helper existed -- no stray leading space. When a marker is present, the
    wrap budget is reduced by the glyph's cell width *before* wrapping (not
    prefixed after) so the first rendered line still fits the row's actual
    width, and the blank-title fallback in ``wrap_console_conversation_title``
    still triggers off the raw (unprefixed) title.

    This is the SAME helper both ``_compose_conversation_browser_row``
    (render) and ``_conversation_browser_rows_height`` (height precompute)
    call, so the two can never disagree about rendered line count -- the
    same invariant ``_conversation_browser_rows_height``'s docstring already
    documents for the un-marked case.
    """
    marker = str(run_marker or "").strip()
    if not marker:
        return wrap_console_conversation_title(title, budget)
    prefix = f"{marker} "
    adjusted_budget = max(_MIN_TITLE_WRAP_BUDGET, budget - cell_len(prefix))
    lines = wrap_console_conversation_title(title, adjusted_budget)
    if not lines:
        return lines
    return (f"{prefix}{lines[0]}", *lines[1:])


def _marker_meaning_tooltip_suffix(marker_glyph: str) -> str:
    """Return `" — <meaning>"` for a non-empty fleet run-marker glyph, else "".

    Fleet-UX expert review F4 (task-1233): sidebar row and header tooltips
    decode whichever marker glyph they carry in context, same as Console
    session tab tooltips (``ConsoleSessionSurface._session_tab_tooltip``).
    An unrecognized or empty glyph (the steady state) adds no suffix, so a
    caller can always append this unconditionally.
    """
    meaning = CONSOLE_RUN_MARKER_MEANINGS_BY_GLYPH.get(str(marker_glyph or "").strip(), "")
    return f" — {meaning}" if meaning else ""


# Pre-measurement fallback for the tray's usable row width. Only the first
# frame before `_fit_height_to_content` measures the real width renders with
# it; the guarded relabel pass corrects it immediately (see
# `_maybe_relabel_for_width`).
_FALLBACK_ROW_CONTENT_WIDTH = 20
# Grouped-browser rows share their line with the star control (width 3 +
# 1 margin) and carry 1 cell of button padding per side.
_BROWSER_ROW_CHROME_WIDTH = 6
# Every row button carries a 1-line bottom margin (see the row CSS).
_ROW_BOTTOM_MARGIN = 1
# Minimum measured-width change (in cells) that triggers a relabel recompose
# after the first measurement. The rail body scrollbar is one cell wide
# (`scrollbar-size: 1 1`), so adding or removing rows -- e.g. collapsing a
# browser section -- toggles the scrollbar and shifts `content_region.width`
# by exactly one cell. `scrollbar-gutter: stable` reserves that cell in the
# real app, but a one-cell change never alters two-line wrapping and must not
# provoke a recompose regardless: recomposing on it would race an in-progress
# state-change recompose (observed as a collapse failing to render) and, in
# any environment where the gutter CSS is absent, oscillate the relabel.
_RELABEL_MIN_WIDTH_DELTA = 2


def _conversation_row_render_height(
    name_line_count: int, subagent_count: int
) -> int:
    """Return the button height for a row: name lines + metadata line,
    plus a dedicated badge line when this conversation has historical
    sub-agent runs (see `format_console_conversation_row_label`)."""
    height = max(1, int(name_line_count)) + 1
    if subagent_count > 0:
        height += 1
    return height


def format_console_conversation_row_label(
    title: str, *, subagent_count: int = 0
) -> str:
    """Return a markup-safe conversation-row label with an optional badge.

    The badge renders on its own trailing line rather than being appended to
    whatever line ``title`` already ends on. Conversation rows already pack a
    marker, a (possibly ellipsized) title, and a secondary detail line
    (workspace / status / age) whose combined length is unbounded -- if the
    badge shared that last line, a long secondary line could push the badge
    past the rail's rendered width and clip it (observed as a bare ``[1`` in
    the agent-runtime live gate; see task-226). Giving the badge its own
    short, fixed-length line decouples its visibility from how long the
    other lines happen to be.

    Args:
        title: Raw row label text (escaped before any markup is appended).
            May already contain newlines (e.g. a title line plus a secondary
            detail line); the badge is appended as one further line.
        subagent_count: Historical sub-agent run count for this conversation.
            When greater than zero, a dim ``[N Sub-Agents]`` badge is
            appended on its own line.

    Returns:
        Rich-markup text safe to render via ``Text.from_markup``.
    """
    base = _escape_markup(str(title))
    if subagent_count > 0:
        return f"{base}\n[dim]\\[{subagent_count} Sub-Agents][/dim]"
    return base


class ConsoleWorkspaceStatusPair(Horizontal):
    """Render workspace authority metadata as a structured status row.

    Attributes:
        label: User-facing row label.
        value: User-facing row value.
        label_id: Textual widget id for the label cell.
        value_id: Textual widget id for the value cell.
    """

    def __init__(
        self,
        label: str,
        value: str,
        *,
        label_id: str,
        value_id: str,
        **kwargs: Any,
    ) -> None:
        """Initialize the label/value status row.

        Args:
            label: User-facing row label.
            value: User-facing row value.
            label_id: Textual widget id for the label cell.
            value_id: Textual widget id for the value cell.
            **kwargs: Additional keyword arguments passed to ``Horizontal``.
        """
        super().__init__(classes="console-workspace-status-pair", **kwargs)
        self.label = label
        self.value = value
        self.label_id = label_id
        self.value_id = value_id
        self.styles.height = "auto"
        self.styles.min_height = 1

    def compose(self) -> ComposeResult:
        """Render the row as queryable Textual widgets.

        Returns:
            ComposeResult containing the label and value widgets.
        """
        label_widget = Static(
            self.label,
            id=self.label_id,
            classes="console-workspace-status-label",
            markup=False,
        )
        label_widget.styles.width = 12
        label_widget.styles.min_width = 12
        yield label_widget

        value_widget = Static(
            self.value,
            id=self.value_id,
            classes="console-workspace-status-value",
            markup=False,
        )
        value_widget.styles.width = "1fr"
        value_widget.styles.min_width = 0
        # TASK-384: at narrow rail widths the value column shrinks to a few cells
        # and a value like "Default" word-wrapped into a "Def / aul / t" letter
        # stack. Truncate the whole token with an ellipsis on one line instead,
        # and keep the full value reachable on hover.
        value_widget.styles.text_wrap = "nowrap"
        value_widget.styles.text_overflow = "ellipsis"
        if self.value:
            # Tooltips render Rich markup (unlike the markup=False value above),
            # so escape the raw value or bracket tokens in a workspace name would
            # be interpreted/styled instead of shown literally (Qodo #821).
            value_widget.tooltip = _escape_markup(self.value)
        yield value_widget


class ConsoleWorkspaceContextTray(RecomposeCaptureGuard, Vertical):
    """Render workspace selection, conversation scope, and recovery copy."""

    class Relabeled(Message):
        """Posted after a width-driven relabel recompose.

        A relabel rebuilds the tray's children from compose(), which
        discards any controls the screen mounted out-of-band (the
        transitional legacy "New conversation" alias). The screen listens
        for this message and re-applies them.
        """

    def __init__(
        self,
        state: ConsoleWorkspaceContextState,
        *,
        show_heading: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.state = state
        self.show_heading = show_heading
        self._row_content_width = _FALLBACK_ROW_CONTENT_WIDTH
        # False until the first real content-width measurement is adopted.
        # The first measurement always relabels (to replace the pre-measure
        # fallback budget); later ones apply the hysteresis threshold.
        self._row_width_measured = False
        self.styles.height = "auto"
        self.styles.min_height = 0

    def on_mount(self) -> None:
        """Fit the tray to content after Textual has measured child widgets.

        Returns:
            None.
        """

        self.call_after_refresh(self._fit_height_to_content)

    def on_resize(self, event: Any) -> None:
        """Refit wrapped status rows when the rail width changes.

        Args:
            event: Textual resize event emitted for this tray.

        Returns:
            None.
        """

        self.call_after_refresh(self._fit_height_to_content)

    def sync_state(self, state: ConsoleWorkspaceContextState) -> None:
        """Refresh the mounted workspace context tray from new display state.

        Args:
            state: Latest workspace context display state.

        Returns:
            None.
        """
        # TASK-251 -- DEVIATION FROM THE BRIEF, documented in the task-251
        # report: the brief's Change 2 asked for the same
        # `if state == self.state: return` guard `ConsoleRunInspector` uses.
        # Measured against the real test suite, that guard broke click
        # targeting on grouped browser rows (workspace-conversation search
        # + resume flows) -- skipping `refresh(recompose=True)` also skips
        # rebuilding the row children, and this widget's own scroll/fit-pass
        # (`_schedule_recomposed_content_fit`) alone does not settle correct
        # on-screen regions for the (unrebuilt) existing children. Tried
        # narrowing the guard to skip only the recompose while still always
        # scheduling the fit-pass -- still broke the same two tests -- so
        # this keeps the original unconditional recompose. The screen-side
        # skip in `_sync_console_workspace_context` (the `call_after_refresh`
        # legacy-alias kick) still applies and is safe/tested.
        self.state = state
        self.styles.min_height = 0
        scroll_parent = self._nearest_scroll_parent()
        parent_scroll_y = getattr(scroll_parent, "scroll_y", None)
        restore_scroll_y = int(parent_scroll_y) if parent_scroll_y is not None else None
        self.refresh(recompose=True)
        if self.is_mounted:
            self._schedule_recomposed_content_fit(restore_scroll_y=restore_scroll_y)

    def _nearest_scroll_parent(self) -> Any | None:
        """Return the nearest ancestor that owns vertical scrolling.

        Returns:
            Scrollable ancestor widget, or ``None`` when unavailable.
        """

        for ancestor in self.ancestors:
            if getattr(ancestor, "id", None) == "console-left-rail-body":
                return ancestor
        for ancestor in self.ancestors:
            if getattr(ancestor, "max_scroll_y", 0) > 0 and callable(
                getattr(ancestor, "scroll_to", None)
            ):
                return ancestor
        return None

    def _schedule_recomposed_content_fit(
        self,
        *,
        restore_scroll_y: int | None = None,
    ) -> None:
        """Schedule a single fit pass after recomposing tray content.

        TASK-1191: this used to fan out into two `call_later` hops plus a
        0.01s scroll-restore timer on every sync, theorized (commit
        1115fa624) as compensating for recompose settling child layout over
        "more than one message turn" in scrolled rails. TASK-1142 investigated
        that theory twice (looking for a click-eating race) and could not
        reproduce it under 15 rapid sync cycles. What TASK-1142 round 1/2
        *did* find, twice, was an actual bug in this area: the grouped
        browser's own auto-height ESTIMATE
        (`_conversation_browser_list_height`) undercounting wrapped
        empty-copy/row lines, silently clipping later siblings out of the
        tray's box. That is a height-computation bug, not a layout-timing
        one, and it is now fixed with wrap-aware, budget-based estimators
        (`_empty_copy_line_count`, `_marker_prefixed_name_lines`) so the
        conversation list's height is set explicitly at compose time
        (`_compose_conversation_browser`) rather than discovered from
        settled geometry.
        `call_after_refresh` -- the same primitive `on_mount`/`on_resize`
        already use for this exact job -- defers the callback until Textual
        has processed the pending refresh (recompose + layout), so
        `virtual_region` is current when `_fit_height_to_content` reads it.
        See `_fit_height_to_content`'s own docstring for why that single
        read is trustworthy (not just deferred-until-current). One deferred
        pass is sufficient; a second pass is still scheduled by
        `_maybe_relabel_for_width` itself, but only on the rare occasions a
        relabel is actually needed (first real width measurement, or a
        multi-cell width change) -- not unconditionally on every sync.

        Args:
            restore_scroll_y: Parent rail scroll offset to restore after fitting.

        Returns:
            None.
        """

        def fit_and_restore_scroll() -> None:
            self._fit_height_to_content()
            self._restore_parent_scroll(restore_scroll_y)

        self.call_after_refresh(fit_and_restore_scroll)

    def _restore_parent_scroll(self, scroll_y: int | None) -> None:
        """Restore the parent rail scroll position after a deferred fit pass.

        Args:
            scroll_y: Parent rail scroll offset to restore.

        Returns:
            None.
        """

        if scroll_y is None:
            return
        scroll_parent = self._nearest_scroll_parent()
        scroll_to = getattr(scroll_parent, "scroll_to", None)
        if callable(scroll_to):
            scroll_to(y=max(0, scroll_y), animate=False)

    def _should_relabel_at_width(self, measured: int) -> bool:
        """Return whether a measured content width warrants a rewrap recompose.

        The first measurement always relabels, replacing the pre-measurement
        fallback budget. Afterwards a change is honored only when it moves at
        least ``_RELABEL_MIN_WIDTH_DELTA`` cells, so a one-cell scrollbar
        toggle (from rows being added or removed) neither races a concurrent
        state-change recompose nor oscillates the relabel.

        Args:
            measured: Freshly measured content-region width in cells.

        Returns:
            True when the tray should recompose to rewrap at ``measured``.
        """
        if not self._row_width_measured:
            return measured != self._row_content_width
        return abs(measured - self._row_content_width) >= _RELABEL_MIN_WIDTH_DELTA

    def _maybe_relabel_for_width(self) -> bool:
        """Rewrap row labels when the measured content width has changed.

        The check lives in the fit pass rather than ``on_resize`` because the
        tray's frame variant (solid <-> none) changes the *content* width
        without changing the outer size, so no resize event fires for it.
        ``_should_relabel_at_width`` is what prevents recompose feedback loops:
        steady-state passes and one-cell scrollbar-toggle flaps are free, so a
        section collapse (which toggles the scrollbar) never spawns a recompose
        that would race its own state-change recompose. Returns True when a
        relabel recompose was scheduled (the caller should skip fitting; the
        scheduled passes re-fit after the recompose).
        """
        region = getattr(self, "content_region", None)
        if region is None or region.width <= 0:
            return False
        measured = int(region.width)
        should_relabel = self._should_relabel_at_width(measured)
        # Latch on the first *real* measurement regardless of whether it
        # relabels: when the measured width coincides with the fallback the
        # decision is a no-op, but the tray has still been measured, so a
        # later one-cell change must be treated as a scrollbar flap (hysteresis
        # branch), not as another first measurement.
        self._row_width_measured = True
        if not should_relabel:
            return False
        self._row_content_width = measured
        scroll_parent = self._nearest_scroll_parent()
        parent_scroll_y = getattr(scroll_parent, "scroll_y", None)
        restore_scroll_y = (
            int(parent_scroll_y) if parent_scroll_y is not None else None
        )
        self.refresh(recompose=True)
        if self.is_mounted:
            self._schedule_recomposed_content_fit(
                restore_scroll_y=restore_scroll_y
            )
            self.post_message(self.Relabeled())
        return True

    def _fit_height_to_content(self) -> None:
        """Expose the full tray content height to the parent scroll container.

        TASK-1191: called from a single deferred `call_after_refresh` pass
        (see `_schedule_recomposed_content_fit`), not several. One pass is
        trustworthy here, not merely convenient, for two independent
        reasons:

        1. Every child's height is either resolved synchronously within
           Textual's own layout pass or set explicitly before it. The
           grouped conversation list's height is computed up front from
           state at compose time (`_conversation_browser_list_height`,
           wrap-aware since TASK-1142) rather than discovered from settled
           geometry. The remaining top-level children -- free-text
           `Static`s like `console-workspace-recovery` / `recovery_copy`
           included -- are still plain WRAPPING Statics with no explicit
           height, but Textual resolves a `Static`'s wrapped auto-height
           synchronously as part of arranging its parent, in the same
           layout pass `call_after_refresh` waits for; it is not a
           multi-turn process for any of them. Nothing here is
           fixed-height; the point is that "wrapped" and "settles over
           several message turns" are not the same claim, and only the
           first is true.
        2. `#console-left-rail-body`'s CSS pins `scrollbar-gutter: stable`
           (see that rule's own comment) specifically so this tray's
           content width -- and therefore every wrapped child's line
           count -- cannot shift out from under an already-scheduled fit
           pass just because a row was added/removed and the scrollbar
           toggled. Do not remove that CSS without re-auditing this
           single-pass assumption.

        Returns:
            None.
        """

        region = getattr(self, "region", None)
        if region is None or region.height <= 0:
            return

        if self._maybe_relabel_for_width():
            return

        content_top = 0
        content_bottom = 1
        for child in self.children:
            if not getattr(child, "display", True):
                continue
            child_virtual = getattr(child, "virtual_region", None)
            if child_virtual is None or child_virtual.height <= 0:
                continue
            content_bottom = max(
                content_bottom,
                child_virtual.y + child_virtual.height,
            )

        target_height = max(1, content_bottom - content_top)
        # When the tray is nested inside a collapsible left-rail section body,
        # size to natural content only. Stretching to the shared scroll parent
        # would push sibling sections (Context/Model/Details) out of the rail
        # viewport.
        if not self._is_inside_rail_section_body():
            scroll_parent = self._nearest_scroll_parent()
            parent_region = getattr(scroll_parent, "region", None)
            if parent_region is not None and parent_region.height > 0:
                target_height = max(target_height, int(parent_region.height))

        if int(region.height) != target_height:
            self.styles.height = target_height

    def _is_inside_rail_section_body(self) -> bool:
        """Return whether this tray is nested in a collapsible rail section body.

        Returns:
            True when an ancestor carries the ``console-rail-section-body``
            class, meaning the tray shares the left-rail scroll container with
            sibling sections and must not stretch to fill it.
        """

        for ancestor in self.ancestors:
            has_class = getattr(ancestor, "has_class", None)
            if callable(has_class) and has_class("console-rail-section-body"):
                return True
        return False

    @staticmethod
    def _static(text: str, *, id: str, classes: str = "") -> Static:
        """Create a plain Static widget with markup disabled.

        Args:
            text: Text to display.
            id: Textual widget id.
            classes: Optional CSS classes.

        Returns:
            Static widget configured for plain text.
        """
        return Static(str(text), id=id, classes=classes, markup=False)

    @staticmethod
    def _conversation_button(
        text: str,
        *,
        id: str,
        conversation_id: str,
        tooltip_label: str | None = None,
        selected: bool = False,
        subagent_count: int = 0,
        name_line_count: int = 1,
    ) -> Button:
        # Escaped-then-markup rendering round-trips plain text unchanged while
        # letting `format_console_conversation_row_label` safely append a dim
        # "[N Sub-Agents]" badge when this conversation has historical runs.
        label = format_console_conversation_row_label(
            text, subagent_count=subagent_count
        )
        button = Button(
            Text.from_markup(label),
            id=id,
            classes="console-workspace-conversation-row",
            compact=True,
        )
        button.conversation_id = conversation_id
        fallback_tooltip = text.splitlines()[0].strip() if text else text
        button.tooltip = f"Switch to {tooltip_label or fallback_tooltip}"
        button.set_class(selected, "console-workspace-conversation-row-selected")
        row_height = _conversation_row_render_height(name_line_count, subagent_count)
        button.styles.height = row_height
        button.styles.min_height = row_height
        return button

    @staticmethod
    def _conversation_browser_rows_height(
        rows: tuple[ConsoleConversationBrowserRow, ...],
        budget: int,
    ) -> int:
        """Total height for a row sequence: per-row button height (from the
        same wrap the labels use, so the two cannot disagree) plus margin."""
        return sum(
            _conversation_row_render_height(
                len(_marker_prefixed_name_lines(row.title, row.run_marker, budget)),
                row.subagent_count,
            )
            + _ROW_BOTTOM_MARGIN
            for row in rows
        )

    @staticmethod
    def _empty_copy_line_count(text: str, budget: int) -> int:
        """Return the rendered line count for an empty-copy Static at
        ``budget`` cells wide.

        TASK-1142 round 1 review: ``_conversation_browser_list_height`` used
        to assume every empty-copy line ("No starred conversations." etc.)
        always renders as exactly one row. Unlike a row title, this Static
        is NOT reduced by ``_BROWSER_ROW_CHROME_WIDTH`` (there is no star
        column beside it), so at the tray's real content width it can (and
        does) wrap to two lines while the flat-constant heuristic still
        counted one. That single missing row silently undercounted the
        `#console-workspace-conversations` container's explicit height,
        clipping whatever composed after it -- including a LATER section's
        entire header, caret and all -- out of the tray's own visible
        bounds. Confirmed via a coordinate-honest repro: after a real click
        collapsed "Chats", "Chats"'s own re-expand caret vanished from the
        rendered pane entirely (not just off in a click-miss sense -- not
        painted at all), because "No starred conversations." and "No
        workspace conversations." both silently wrapped to two lines above
        it.

        TASK-1142 round-2 review (Qodo, PR #1050): the first fix here called
        ``wrap_console_conversation_title``, which hard-caps at two lines
        and ellipsizes -- correct for a row title, but an empty-copy
        ``Static`` wraps with no such cap. Any empty-copy string that
        legitimately needs 3+ lines would silently undercount again through
        that capped helper. Now uses ``wrap_console_plain_text_uncapped``
        instead, which is the same per-line word-wrap rule with the cap
        removed -- see its docstring for why a dedicated helper (rather
        than reusing the row-title one) is required.
        """
        return max(
            _CONVERSATION_BROWSER_EMPTY_COPY_HEIGHT,
            len(wrap_console_plain_text_uncapped(text, budget)),
        )

    @staticmethod
    def _conversation_browser_list_height(
        browser: ConsoleConversationBrowserState,
        row_title_budget: int,
        empty_copy_budget: int,
    ) -> int:
        """Return the full content height for the grouped browser rows.

        Args:
            browser: The grouped browser state being measured.
            row_title_budget: Wrap budget for row titles (reduced for the
                star column, matches ``_browser_title_budget()``).
            empty_copy_budget: Wrap budget for empty-copy lines, which span
                the tray's full row width with no star-column reduction
                (matches ``_row_content_width``) -- see
                ``_empty_copy_line_count`` for why this must differ from
                ``row_title_budget``.
        """
        height = 0
        for section in browser.sections:
            height += _CONVERSATION_BROWSER_HEADER_HEIGHT
            if section.collapsed:
                continue
            if section.groups:
                for group in section.groups:
                    height += _CONVERSATION_BROWSER_HEADER_HEIGHT
                    if group.collapsed:
                        continue
                    if group.rows:
                        height += ConsoleWorkspaceContextTray._conversation_browser_rows_height(
                            group.rows, row_title_budget
                        )
                    elif group.empty_copy:
                        height += ConsoleWorkspaceContextTray._empty_copy_line_count(
                            group.empty_copy, empty_copy_budget
                        )
                continue
            if section.rows:
                height += ConsoleWorkspaceContextTray._conversation_browser_rows_height(
                    section.rows, row_title_budget
                )
            elif section.empty_copy:
                height += ConsoleWorkspaceContextTray._empty_copy_line_count(
                    section.empty_copy, empty_copy_budget
                )
        return max(_CONVERSATION_BROWSER_EMPTY_COPY_HEIGHT, height)

    def compose(self) -> ComposeResult:
        if self.show_heading:
            yield self._static(
                self.state.heading,
                id="console-workspace-context-title",
                classes="destination-section",
            )

        workspace_value = self.state.workspace_name or self._workspace_selector_label()
        yield ConsoleWorkspaceStatusPair(
            "Workspace",
            workspace_value,
            label_id="console-active-workspace-label",
            value_id="console-active-workspace-value",
            id="console-active-workspace",
        )

        with Horizontal(
            id="console-workspace-action-row",
            classes="console-workspace-action-row",
        ):
            yield Button(
                "Switch",
                id="console-change-workspace",
                classes="console-workspace-action",
                compact=True,
                disabled=not self.state.change_workspace_enabled,
            )
            yield Button(
                "New",
                id="console-new-workspace",
                classes="console-workspace-action",
                compact=True,
                disabled=not self.state.new_workspace_enabled,
            )

        # task-13: workspace-level RAG retrieval scope entry point.
        # Named "RAG Scope" (not "Scope") to avoid colliding with the
        # unrelated "Scope" status pair below, which shows the active
        # conversation's identity, not a RAG retrieval scope. Enabled
        # only for a real registry workspace (`rag_scope_enabled`) --
        # never for the "Local Default"/error/no-registry sentinel
        # states, which have no real workspace_id to scope against.
        #
        # This lives on its OWN row (task-14) rather than sharing the
        # Switch/New row: the narrow Console left rail body is only wide
        # enough for ~2 compact buttons (Textual's default Button
        # min-width is 16 columns each). A third button packed into the
        # same Horizontal overflowed the rail's clipped width, so the
        # button's clickable region extended past the rail body -- real
        # clicks (and `pilot.click`) landed on the rail backdrop instead
        # of the button.
        with Horizontal(
            id="console-workspace-rag-scope-row",
            classes="console-workspace-action-row",
        ):
            scope_button = Button(
                "RAG Scope",
                id="console-workspace-rag-scope-open",
                classes="console-workspace-action",
                compact=True,
                disabled=not self.state.rag_scope_enabled,
            )
            scope_button.tooltip = "Narrow RAG retrieval to items in this workspace"
            yield scope_button

        if (
            not self.state.change_workspace_enabled
            and self.state.change_workspace_recovery
        ):
            yield self._static(
                self.state.change_workspace_recovery,
                id="console-change-workspace-recovery",
                classes="console-workspace-recovery",
            )

        scope_value = self.state.scope_label or ""
        scope_pair = ConsoleWorkspaceStatusPair(
            "Scope",
            scope_value,
            label_id="console-active-scope-label",
            value_id="console-active-scope-value",
            id="console-active-scope",
        )
        # TASK-373 AC#2: keep the raw conversation identifier available on hover
        # rather than in the primary row.
        if self.state.scope_detail:
            scope_pair.tooltip = f"Conversation id: {self.state.scope_detail}"
        yield scope_pair

        if self.state.recovery_copy:
            yield self._static(
                self.state.recovery_copy,
                id="console-workspace-recovery",
                classes="console-workspace-recovery",
            )

        # TASK-1190: the transitional legacy conversation-list compose path
        # (taken only when ``conversation_browser is None``) was retired.
        # Reachability sweep (grep every constructor/caller that could
        # produce a state reaching this widget): the SOLE production
        # builder, ``ChatScreen._build_console_workspace_context_state``,
        # always finishes by calling ``_with_console_conversation_browser_
        # state``, which unconditionally attaches a real
        # ``ConsoleConversationBrowserState`` via ``build_console_
        # conversation_browser_state`` (that builder has exactly one
        # ``return`` and it is never ``None``). No other production call
        # site constructs a ``ConsoleWorkspaceContextState`` or calls this
        # tray's ``sync_state`` with ``conversation_browser=None`` -- only
        # test fixtures did, to pin the now-removed legacy render path. The
        # ``browser is not None`` guard below stays as defense in depth
        # (render nothing rather than crash) rather than an assertion.
        browser = self.state.conversation_browser
        if browser is not None:
            yield from self._compose_conversation_browser(browser)

    def _compose_conversation_browser(
        self,
        browser: ConsoleConversationBrowserState,
    ) -> ComposeResult:
        """Render the grouped all-workspaces conversation browser."""

        with Horizontal(
            id="console-workspace-conversations-header",
            classes="console-rail-header console-workspace-conversations-header",
        ):
            title = self._static(
                "Conversations",
                id="console-workspace-conversations-title",
                classes="console-rail-section-title",
            )
            title.styles.width = "1fr"
            yield title
        yield self._static(
            browser.selected_summary or "No active conversation.",
            id="console-workspace-selected-conversation",
            classes="console-workspace-selected-conversation",
        )
        with Horizontal(
            id="console-workspace-conversation-search-row",
            classes="console-workspace-conversation-search-row",
        ):
            search_input = Input(
                value=browser.query,
                placeholder="Search conversations",
                id="console-workspace-conversation-search",
                classes="console-workspace-conversation-search",
            )
            search_input.styles.width = "1fr"
            yield search_input
            clear_button = Button(
                "Clear",
                id="console-workspace-conversation-search-clear",
                classes=(
                    "console-workspace-action "
                    "console-workspace-conversation-search-clear"
                ),
                compact=True,
                disabled=not bool(str(browser.query or "").strip()),
            )
            clear_button.tooltip = "Clear conversation search"
            yield clear_button
        if browser.status_copy:
            yield self._static(
                browser.status_copy,
                id="console-workspace-conversation-search-status",
                classes="console-workspace-empty-copy",
            )
        if browser.error_copy:
            yield self._static(
                browser.error_copy,
                id="console-workspace-conversation-search-error",
                classes="console-workspace-recovery",
            )
        if not browser.marks_available:
            yield self._static(
                "Local stars unavailable",
                id="console-conversation-browser-marks-unavailable",
                classes="console-workspace-recovery",
            )

        row_index = 0
        conversation_list = Vertical(id="console-workspace-conversations")
        conversation_list.styles.height = self._conversation_browser_list_height(
            browser,
            self._browser_title_budget(),
            self._row_content_width,
        )
        conversation_list.styles.min_height = 0
        with conversation_list:
            for section in browser.sections:
                yield from self._compose_conversation_browser_section_header(section)
                if section.collapsed:
                    continue
                if section.groups:
                    for group_index, group in enumerate(section.groups):
                        yield from self._compose_conversation_browser_group_header(
                            group,
                            group_index,
                        )
                        if group.collapsed:
                            continue
                        if group.rows:
                            for row in group.rows:
                                yield from self._compose_conversation_browser_row(
                                    row,
                                    row_index,
                                    marks_available=browser.marks_available,
                                )
                                row_index += 1
                        elif group.empty_copy:
                            yield self._static(
                                group.empty_copy,
                                id=(
                                    "console-conversation-browser-group-empty-"
                                    f"{group_index}"
                                ),
                                classes="console-workspace-empty-copy",
                            )
                    if not section.groups and section.empty_copy:
                        yield self._static(
                            section.empty_copy,
                            id=(
                                "console-conversation-browser-"
                                f"{section.section_id}-empty"
                            ),
                            classes="console-workspace-empty-copy",
                        )
                    continue
                if section.rows:
                    # Flat sections have no workspace group header. The Starred
                    # section pins conversations from multiple workspaces, so the
                    # workspace is the differentiator that keeps same-titled rows
                    # distinguishable; the all-global Chats section does not need
                    # it (Qodo #812).
                    show_workspace = section.section_id == "starred"
                    for row in section.rows:
                        yield from self._compose_conversation_browser_row(
                            row,
                            row_index,
                            marks_available=browser.marks_available,
                            show_workspace=show_workspace,
                        )
                        row_index += 1
                elif section.empty_copy:
                    yield self._static(
                        section.empty_copy,
                        id=f"console-conversation-browser-{section.section_id}-empty",
                        classes="console-workspace-empty-copy",
                    )

    def _compose_conversation_browser_section_header(
        self,
        section: ConsoleConversationBrowserSection,
    ) -> ComposeResult:
        """Render one grouped browser section header.

        TASK-912 AC#1: collapsing a top-level section (Starred/Workspaces/
        Chats) hides every row -- and every workspace group -- beneath it,
        so any fleet run-marker glyph among them is otherwise invisible.
        When collapsed AND `section.run_marker` is non-empty (the
        most-urgent glyph across the section's full pre-cap contents,
        computed in `conversation_browser_state`), it is appended to the
        header label -- same plain-string threading and `markup=False`
        Static the group-header marker already uses.

        TASK-912 review fix round 1: an expanded flat section (Starred/
        Chats -- `"workspaces"` never carries its own `rows`, so this is
        always empty there) is subject to the identical per-section row cap
        as a workspace group, and shows every row's own marker just like an
        expanded group does -- UNLESS the section's row count exceeds the
        cap, in which case `section.capped_run_marker` (the most-urgent
        glyph among only the hidden overflow rows) is borrowed onto the
        header instead, mirroring `_compose_conversation_browser_group_
        header` exactly. A visible marked row within the cap needs no
        header echo, which is exactly what `capped_run_marker` excludes.
        """

        label = section.label
        if section.collapsed and section.run_marker:
            label = f"{label} {section.run_marker}"
        elif not section.collapsed and section.capped_run_marker:
            label = f"{label} {section.capped_run_marker}"
        with Horizontal(classes="console-conversation-browser-section-header"):
            title = self._static(
                label,
                id=f"console-conversation-browser-{section.section_id}-title",
                classes="console-conversation-browser-section-title",
            )
            yield title
            toggle = Button(
                GLYPH_COLLAPSED if section.collapsed else GLYPH_EXPANDED,
                id=f"console-conversation-browser-section-toggle-{section.section_id}",
                classes=(
                    "console-workspace-action console-workspace-conversations-toggle"
                ),
                compact=True,
            )
            toggle.group_id = f"section:{section.section_id}"
            # TASK-1233 AC#1: decode whichever aggregate marker the header
            # itself is currently showing (same collapsed/capped split as
            # the label above) rather than inventing a new tooltip surface.
            header_marker = (
                section.run_marker if section.collapsed else section.capped_run_marker
            )
            escaped_section_label = _escape_markup(section.label)
            toggle.tooltip = (
                f"Expand {escaped_section_label}"
                f"{_marker_meaning_tooltip_suffix(header_marker)}"
                if section.collapsed
                else f"Collapse {escaped_section_label}"
                f"{_marker_meaning_tooltip_suffix(header_marker)}"
            )
            yield toggle

    def _compose_conversation_browser_group_header(
        self,
        group: ConsoleConversationBrowserGroup,
        index: int,
    ) -> ComposeResult:
        """Render one workspace group header.

        PA-T8 review fix round 1 (IMPORTANT 2): a collapsed group hides its
        rows entirely, so any fleet run-marker glyph on those rows is
        otherwise invisible -- defeating "which workspace's agents are busy
        is visible at a glance" for exactly the user who collapsed the
        group. When collapsed AND at least one hidden row carries a marker,
        the single most-urgent glyph (``group.run_marker``, precomputed in
        ``conversation_browser_state._build_workspace_groups``) is appended
        to the header label, same plain-string threading the row label
        prefix already uses (this Static renders with ``markup=False``, so
        no separate escaping step applies here either). An expanded group
        already shows every row's own marker, so its header stays
        unchanged -- UNLESS the group's row count exceeds
        `CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT`: a marked row pushed
        past that cap is hidden despite the group being expanded, so
        `group.capped_run_marker` (the most-urgent glyph among only the
        hidden overflow rows, TASK-912 AC#2) is borrowed onto the header in
        that case instead. A visible marked row within the cap needs no
        header echo, which is exactly what `capped_run_marker` excludes.
        """

        with Horizontal(classes="console-conversation-browser-group-header"):
            label = group.label
            if group.collapsed and group.run_marker:
                label = f"{label} {group.run_marker}"
            elif not group.collapsed and group.capped_run_marker:
                label = f"{label} {group.capped_run_marker}"
            title = self._static(
                label,
                id=f"console-conversation-browser-group-title-{index}",
                classes="console-conversation-browser-group-title",
            )
            yield title
            toggle = Button(
                GLYPH_COLLAPSED if group.collapsed else GLYPH_EXPANDED,
                id=f"console-conversation-browser-group-toggle-{index}",
                classes=(
                    "console-workspace-action console-workspace-conversations-toggle"
                ),
                compact=True,
            )
            toggle.group_id = group.group_id
            # TASK-1233 AC#1: same collapsed/capped aggregate-marker split
            # as the label above, decoded into the already-existing toggle
            # tooltip rather than a new tooltip surface.
            header_marker = (
                group.run_marker if group.collapsed else group.capped_run_marker
            )
            escaped_group_label = _escape_markup(group.label)
            toggle.tooltip = (
                f"Expand {escaped_group_label}"
                f"{_marker_meaning_tooltip_suffix(header_marker)}"
                if group.collapsed
                else f"Collapse {escaped_group_label}"
                f"{_marker_meaning_tooltip_suffix(header_marker)}"
            )
            yield toggle

    def _compose_conversation_browser_row(
        self,
        row: ConsoleConversationBrowserRow,
        index: int,
        *,
        marks_available: bool,
        show_workspace: bool = False,
    ) -> ComposeResult:
        """Render one grouped browser row plus its local star control.

        ``show_workspace`` carries the owning workspace into the subtitle for rows
        rendered without a workspace group header (the cross-workspace Starred
        section), where it is the disambiguator (Qodo #812).
        """

        with Horizontal(classes="console-conversation-browser-row-line"):
            budget = self._browser_title_budget()
            title = self._conversation_title(row.title)
            name_lines = _marker_prefixed_name_lines(row.title, row.run_marker, budget)
            status = self._conversation_status(row.status)
            detail = self._conversation_detail_status(row.status)
            secondary = truncate_console_row_cells(
                self._conversation_row_secondary(
                    detail,
                    row.updated_label,
                    workspace_label=row.workspace_label if show_workspace else "",
                )
                or "conversation",
                budget,
            )
            status_suffix = f" [{status}]" if status else ""
            marker_suffix = _marker_meaning_tooltip_suffix(row.run_marker)
            row_button = self._conversation_button(
                "\n".join((*name_lines, secondary)),
                id=f"console-workspace-conversation-{index}",
                conversation_id=row.conversation_id or row.row_key,
                # TASK-1233 AC#1: the tooltip decodes the row's fleet
                # run-marker glyph in context. Escaped because `Button.
                # tooltip` renders Rich markup and `title` is user data
                # (a conversation title), unlike `status_suffix`/
                # `marker_suffix`, which are both fixed app vocabulary.
                tooltip_label=f"{_escape_markup(title)}{status_suffix}{marker_suffix}",
                selected=row.selected,
                subagent_count=row.subagent_count,
                name_line_count=len(name_lines),
            )
            row_button.row_key = row.row_key
            row_button.native_session_id = row.native_session_id
            row_button.scope_type = row.scope_type
            row_button.workspace_id = row.workspace_id
            row_button.styles.width = "1fr"
            row_button.styles.min_width = 0
            if not row.openable:
                # TASK-717: a prior open proved this record is missing; the
                # row must stop presenting as an openable conversation.
                row_button.disabled = True
                row_button.add_class("console-workspace-conversation-row-broken")
                row_button.tooltip = (
                    "This conversation's record is missing and cannot be opened."
                )
            yield row_button

            star_disabled = not marks_available or not row.star_enabled
            star_button = Button(
                # TASK-357: a recognizable filled/hollow star pair — the old
                # one-cell '*'/'.' distinction was nearly invisible and led to
                # accidental silent toggles.
                "★" if row.starred else "☆",
                id=f"console-conversation-star-{index}",
                classes="console-workspace-action console-conversation-star",
                compact=True,
                disabled=star_disabled,
            )
            # Match the row button's height so the star control still spans
            # the full row whatever the name-line and badge count.
            star_row_height = _conversation_row_render_height(
                len(name_lines), row.subagent_count
            )
            star_button.styles.height = star_row_height
            star_button.styles.min_height = star_row_height
            if not marks_available:
                star_button.tooltip = "Local stars unavailable"
            elif not row.star_enabled:
                star_button.tooltip = "Send or save this conversation before starring."
            else:
                star_button.tooltip = (
                    "Unstar conversation" if row.starred else "Star conversation"
                )
            star_button.row_key = row.row_key
            star_button.conversation_id = row.conversation_id
            star_button.starred = row.starred
            # TASK-357: carry the title so the press handler can confirm the
            # toggle ("Starred <title>") instead of changing state silently.
            star_button.conversation_title = row.title
            yield star_button

    def _workspace_selector_label(self) -> str:
        """Return the visible active-workspace selector affordance."""
        workspace_label = self.state.workspace_label
        if workspace_label.startswith("Workspace: "):
            workspace_label = workspace_label.removeprefix("Workspace: ").strip()
        return workspace_label

    @staticmethod
    def _conversation_title(title: str) -> str:
        """Return a readable conversation label."""
        return str(title).strip() or _UNTITLED_CONVERSATION

    def _browser_title_budget(self) -> int:
        """Cells available to grouped-browser row text."""
        return max(
            _MIN_TITLE_WRAP_BUDGET,
            self._row_content_width - _BROWSER_ROW_CHROME_WIDTH,
        )

    @staticmethod
    def _conversation_status(status: str) -> str:
        """Return a short user-facing conversation status badge."""
        normalized = str(status or "").strip().lower()
        if not normalized:
            return ""
        return _STATUS_LABELS.get(normalized, normalized.replace("-", " "))

    @staticmethod
    def _conversation_detail_status(status: str) -> str:
        """Return second-line row metadata for row disambiguation.

        TASK-356: delegates to the shared vocabulary so the rail and the
        Ctrl+K switcher never disagree on the same conversation's state.
        """
        return console_conversation_status_detail(status)

    @staticmethod
    def _conversation_row_secondary(
        detail: str,
        updated_label: str,
        *,
        workspace_label: str = "",
    ) -> str:
        """Compress the row's second line to just its differentiator.

        TASK-374: the subtitle used to read ``<workspace> - saved chat - <age>``
        on every row, so only the age differed and half the section's vertical
        space carried no information. For a row under a workspace group header the
        workspace is redundant and ``saved chat`` is the common default -- so keep
        the age always and the state only when it is a non-default differentiator.

        ``workspace_label`` is supplied only for rows rendered WITHOUT a workspace
        group header -- the cross-workspace ``Starred`` section -- where the
        workspace is itself the differentiator that keeps same-titled
        conversations from different workspaces distinguishable (Qodo #812).

        Args:
            detail: The friendly state label from ``_conversation_detail_status``.
            updated_label: The compact relative age (e.g. ``2d``).
            workspace_label: The owning workspace, included as the leading
                differentiator only for header-less sections; omitted otherwise.

        Returns:
            The age alone for a default saved grouped row, ``<workspace> - <age>``
            for a starred cross-workspace row, ``<state> - <age>`` when the state
            differentiates, or ``""`` when nothing is present.
        """
        parts = [
            part
            for part in (
                workspace_label,
                detail if detail and detail != CONSOLE_DEFAULT_CONVERSATION_DETAIL else "",
                updated_label,
            )
            if str(part or "").strip()
        ]
        return " - ".join(parts)
