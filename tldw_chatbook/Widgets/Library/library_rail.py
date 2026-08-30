"""Library shell rail: search box, source sections, and Details disclosure."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from rich.markup import escape as escape_markup
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.scalar import Scalar
from textual.css.query import NoMatches
from textual.events import Focus, Key, MouseDown, Resize
from textual.widget import Widget
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_rail_state import (
    LibraryLifecycle,
    LibraryRailPreferences,
)
from tldw_chatbook.Utils.library_rail_width import (
    LIBRARY_DEFAULT_MAX_WIDTH,
    LIBRARY_MIN_WIDTH,
    OrdinaryRailStyleContract,
)
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_CREATE_NOTE,
    LIBRARY_ROW_INGEST_MEDIA,
    LibraryRailRow,
    LibraryRailSectionState,
    LibraryShellState,
)
from tldw_chatbook.Widgets.Library.library_canvas_sync import PostRecomposeCallback
from tldw_chatbook.Widgets.destination_rail import (
    DestinationRailHandle,
    DestinationRailSectionHeader,
)
from tldw_chatbook.Widgets.recompose_capture_guard import RecomposeCaptureGuard

LIBRARY_RAIL_ROW_PREFIX = "library-row-"

_MAX_LIBRARY_ROW_TITLE = 20


def library_dim_label_text(label: str, value: str) -> Text:
    """Build a "label · value" line with a dimmed label and a literal value.

    Used for the Details rail's scannable rows (e.g. "Source · Local",
    "Active · Local Default") so the label reads as secondary context while
    the value stays at normal emphasis.

    Args:
        label: Short constant label rendered with a dim style.
        value: Value text appended literally. Rich markup embedded in
            ``value`` (e.g. a user-supplied workspace name) is never
            interpreted, only displayed, so untrusted content cannot inject
            styling.

    Returns:
        A Rich ``Text`` combining the dimmed label separator and the plain
        value, with a single "dim" style span over the label only.
    """
    text = Text()
    text.append(f"{label} · ", style="dim")
    text.append(str(value))
    return text


def _truncate_row_title(title: str, budget: int = _MAX_LIBRARY_ROW_TITLE) -> str:
    """Return the raw row title capped to ``budget`` cells with "...".

    The result is NOT markup-escaped: callers truncate first (escaping
    before truncating could slice through an escape sequence) and escape
    at label-build time, since ``Button`` labels parse Rich markup -- an
    unescaped user title like ``[draft] Q3 plan [wip]`` would render with
    its bracketed segments consumed as (or crashing on) markup tags.
    """
    readable = str(title).strip()
    if len(readable) > budget:
        readable = f"{readable[: max(1, budget - 3)].rstrip()}..."
    return readable


def _fit_title_no_mid_word_cut(title: str, budget: int) -> str:
    """Fit ``title`` within ``budget`` cells without cutting inside a word.

    LIB-18: unlike ``_truncate_row_title`` (used elsewhere for arbitrary
    user content, where a hard character cut at the budget is the
    established, expected behavior), the rail's own fixed navigational row
    titles must never ellipsize mid-word at the rail's real widths
    (120/100/80 columns all pin the rail to the same row-content budget via
    its own ``min_width``). Returns ``title`` unchanged when it already
    fits; otherwise ellipsizes at the last whitespace boundary within
    budget. A single unbroken word that still does not fit (no internal
    space to retreat to, and no ``short_title`` override defined for it by
    the caller) falls back to the same hard character cut as a last
    resort -- callers should prefer supplying a ``short_title`` for any
    row whose title risks landing here.
    """
    readable = str(title).strip()
    if len(readable) <= budget:
        return readable
    ellipsis_budget = max(1, budget - 3)
    head = readable[:ellipsis_budget]
    boundary = head.rfind(" ")
    if boundary > 0:
        head = head[:boundary].rstrip()
    else:
        head = head.rstrip()
    if not head:
        head = readable[:ellipsis_budget].rstrip()
    return f"{head}..."


#: task-4023 AC#7: the CANVAS row-title budget. The content canvases
#: (media/conversations) used to inherit the RAIL's 20-cell cap, so a
#: 170-column terminal rendered "Podcast episode 1..." with ~115 blank
#: columns beside it. The canvas rows are full-width with CSS
#: ``text-overflow: ellipsis``, so this cap only bounds pathological
#: titles; real widths ellipsize at the rendered edge.
_MAX_LIBRARY_CANVAS_ROW_TITLE = 120


def _visible_row_title(title: str, budget: int = _MAX_LIBRARY_CANVAS_ROW_TITLE) -> str:
    """Return the canvas-safe row title: capped at ``budget`` cells, escaped.

    Used by the Library content canvases (conversations/media rows),
    which interpolate the result into markup-parsed ``Button`` labels.
    The rail itself builds labels via ``LibraryRail._row_label`` (F-015
    width-fitting), which truncates raw and escapes at build time --
    the rail's own 20-cell ``_MAX_LIBRARY_ROW_TITLE`` no longer leaks
    into the (much wider) canvases.
    """
    return escape_markup(_truncate_row_title(title, budget))


class SelectAllOnFocusingClickInput(Input):
    """An ``Input`` whose FIRST click (the one that also focuses it) selects
    all text instead of just positioning the cursor there (LIB-17).

    Textual's ``Input`` already defaults ``select_on_focus=True`` (its own
    ``_on_focus`` sets ``Selection(0, len(value))``), but ``Input.
    _on_mouse_down`` ALWAYS repositions the cursor to the click offset --
    and ``Screen._forward_event`` calls ``set_focus()`` *synchronously*,
    before the click is even forwarded to this widget, so ``self.has_focus``
    already reads ``True`` by the time ``_on_mouse_down`` runs regardless of
    whether THIS click is the one that focused the box. Checking
    ``has_focus`` there cannot distinguish the two cases; the ``Focus``
    event (posted by ``set_focus``) and this ``MouseDown`` (posted right
    after, by the same click) are instead queued back-to-back on THIS
    widget's own message pump and processed in that order, so ``_on_focus``
    marks a one-shot "a focusing click may still be inbound" flag that
    ``_on_mouse_down`` consumes if it is the very next thing processed --
    the same-gesture window this whole mechanism depends on.

    A second Textual quirk this override must also account for: message
    dispatch (``MessagePump._get_dispatch_methods``) walks the class's
    entire MRO and invokes EVERY class's own ``_on_mouse_down`` in turn
    (most-derived first) -- calling ``super()`` is not what wires this up,
    and NOT calling it does not skip it either. Without
    ``event.prevent_default()`` in the select-all branch below, ``Input.
    _on_mouse_down`` (the base class, further up the MRO) still runs
    immediately afterward and silently overwrites the select-all with its
    own ``Selection.cursor(click_offset)`` -- ``prevent_default()`` is the
    documented mechanism (checked at the top of that MRO walk) that stops
    it, the exact same seam this class's own ``_on_key`` already leans on
    for the "/" re-arm.

    Without this fix, a plain mouse click on a not-yet-focused Input
    silently wins the race and undoes ``select_on_focus``'s "replace me"
    framing entirely, regardless of where in the box the click lands. For a
    prefilled query box this means the box's stale text survives the very
    interaction (click, then type) a user relies on to replace it:
    live-reproduced by a click landing near the start of "quokka" and
    typing a character, which PREPENDED instead of replacing ("Zquokka").

    Scoped to the focusing click only: once the box already has focus (no
    new ``Focus`` event, so the flag is never armed), a click positions the
    cursor precisely as normal -- expected mid-text editing is unaffected.
    The flag also self-clears shortly after arming (``call_after_refresh``)
    so a LATER, unrelated click on an already-focused box (e.g. after a
    Tab-focus with no immediately-following click) never inherits a stale
    "select all" from an earlier, unconsumed focus event.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._select_all_pending_click = False

    def _on_focus(self, event: Focus) -> None:
        self._select_all_pending_click = True
        self.call_after_refresh(self._clear_select_all_pending_click)

    def _clear_select_all_pending_click(self) -> None:
        self._select_all_pending_click = False

    async def _on_mouse_down(self, event: MouseDown) -> None:
        if self._select_all_pending_click:
            self._select_all_pending_click = False
            self._pause_blink(visible=True)
            self.select_all()
            self._selecting = True
            self.capture_mouse()
            # See the class docstring: stops Textual's own MRO walk from
            # ALSO invoking Input._on_mouse_down for this event, which
            # would otherwise overwrite the select-all above.
            event.prevent_default()
            return
        await super()._on_mouse_down(event)


class LibraryRailSearchInput(SelectAllOnFocusingClickInput):
    """Rail search box where a second "/" re-arms the query instead of typing.

    "/" is the Library screen's focus-the-search key (F-012); once the box
    itself has focus the screen's on_key never sees printable keys, so a
    second "/" would insert a literal slash into the query -- the settings
    screen's task-1584 live trap, solved the same way here: intercept it
    and select-all so the next keystroke replaces the stale text. LIB-17:
    a stale query surviving a screen re-entry (the rail rebuilds a fresh
    box seeded from the screen's persisted query, unfocused) is covered by
    the SAME "select-all, don't clear" promise this seam already makes for
    "/" -- ``SelectAllOnFocusingClickInput`` extends it to the box's first
    click too, so whichever way the user re-enters the box (click or "/"),
    typing replaces rather than appends.
    """

    async def _on_key(self, event: Key) -> None:
        # Same slash representations the screen-level handler accepts --
        # some platforms/layouts emit key="slash" without character="/".
        if event.key in {"/", "slash"} or event.character == "/":
            self.select_all()
            event.stop()
            event.prevent_default()
            return
        await super()._on_key(event)


class LibraryRailRowButton(Button):
    """Rail row button that refits its own label when its width changes.

    F-015: Textual clips an over-long label at the right edge, which ate
    the count exactly when it mattered ("Conversations …" at 100
    columns). The row owns the truncation instead -- the F-013 subtitle
    drops first, then the title ellipsizes, and the count is the last
    thing standing. Per-button ``on_resize`` (not one rail-level pass) so
    the fit also follows the vertical scrollbar's gutter: the rail's own
    size does not change when its scrollbar appears, but each row's
    content width does.
    """

    #: The row record the label is rebuilt from on every width change.
    library_row: "LibraryRailRow | None" = None

    def on_resize(self, event: Resize) -> None:
        row = self.library_row
        width = self.content_region.width
        if row is None or width <= 0:
            return
        self.label = LibraryRail._row_label(
            row, self.has_class("library-rail-row-selected"), width
        )


class LibraryNavigationRailHandle(DestinationRailHandle):
    """Compact, focusable handle for reopening Library navigation."""

    WIDTH = 3

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            label="Nav",
            button_id="library-rail-open",
            badge_id="library-rail-handle-badge",
            side="left",
            open_tooltip="Expand Library navigation",
            **kwargs,
        )
        self.add_class("console-rail-handle-vertical")
        self.styles.width = self.WIDTH
        self.styles.min_width = self.WIDTH
        self.styles.max_width = self.WIDTH

    def compose(self) -> ComposeResult:
        """Render the narrow vertical button used to expand Library navigation.

        Returns:
            ComposeResult yielding the configured navigation button.
        """
        for child in super().compose():
            if isinstance(child, Button):
                child.add_class("console-rail-handle-button-vertical")
                child.styles.width = 1
                child.styles.max_width = 1
                child.styles.height = "1fr"
                child.styles.clear_rule("min_height")
                child.styles.clear_rule("max_height")
                child.styles.line_pad = 0
            yield child

    def _display_label(self) -> str:
        return "N\na\nv"


class LibraryRail(PostRecomposeCallback, RecomposeCaptureGuard, Vertical):
    """Render the Library shell rail: search, source sections, and Details.

    Attributes:
        shell: Current Library shell display state.
        preferences: Section open/collapsed preferences.
        query: Active search box text, re-seeded into the ``Input`` so a
            recompose does not silently clear a submitted query.
        workspaces_body_factory: Callable building the Workspaces body widgets
            (depth panel + actions) so the screen keeps its service/callback
            wiring; rendered inside the collapsed Details section.
    """

    def __init__(
        self,
        shell: LibraryShellState,
        preferences: LibraryRailPreferences,
        *,
        query: str = "",
        search_placeholder: str = "Search conversations…",
        workspaces_body_factory: Callable[[], Iterable[Widget]] | None = None,
        top_action_factory: Callable[[], Iterable[Widget]] | None = None,
        lifecycle: LibraryLifecycle = LibraryLifecycle.EXPANDED,
        onboarding_all_empty: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.shell = shell
        self.preferences = preferences
        self.query = query
        self.search_placeholder = search_placeholder
        self.workspaces_body_factory = workspaces_body_factory
        self.top_action_factory = top_action_factory
        self.lifecycle = lifecycle
        self.onboarding_all_empty = onboarding_all_empty
        self._last_ordinary_width_contract: OrdinaryRailStyleContract | None = None
        self.apply_ordinary_width_contract(
            OrdinaryRailStyleContract(
                True,
                "3fr",
                LIBRARY_MIN_WIDTH,
                LIBRARY_DEFAULT_MAX_WIDTH,
            )
        )

    @staticmethod
    def _ordinary_width_inline_declarations(
        styles: Any,
    ) -> tuple[
        bool | None,
        tuple[float, object, object] | None,
        tuple[float, object, object] | None,
        tuple[float, object, object] | None,
    ]:
        """Return the four inline declarations in a Scalar-safe comparison shape."""
        inline_styles = styles.inline

        def scalar_rule(rule_name: str) -> tuple[float, object, object] | None:
            if not inline_styles.has_rule(rule_name):
                return None
            value = inline_styles.get_rule(rule_name)
            if not isinstance(value, Scalar):
                return None
            return (value.value, value.unit, value.percent_unit)

        display = (
            inline_styles.get_rule("display") == "block"
            if inline_styles.has_rule("display")
            else None
        )
        return (
            display,
            scalar_rule("width"),
            scalar_rule("min_width"),
            scalar_rule("max_width"),
        )

    @staticmethod
    def _ordinary_width_contract_declarations(
        contract: OrdinaryRailStyleContract,
    ) -> tuple[
        bool,
        tuple[float, object, object] | None,
        tuple[float, object, object] | None,
        tuple[float, object, object] | None,
    ]:
        """Normalize contract scalars to Textual's inline declaration form."""

        def scalar_value(
            value: str | int | None,
        ) -> tuple[float, object, object] | None:
            if value is None:
                return None
            scalar = (
                Scalar.parse(value)
                if isinstance(value, str)
                else Scalar.from_number(value)
            )
            return (scalar.value, scalar.unit, scalar.percent_unit)

        return (
            contract.display,
            scalar_value(contract.width),
            scalar_value(contract.min_width),
            scalar_value(contract.max_width),
        )

    def apply_ordinary_width_contract(
        self, contract: OrdinaryRailStyleContract
    ) -> None:
        """Apply ordinary-shell geometry as reversible inline declarations.

        ``None`` width values deliberately remove the matching inline rule,
        allowing the next ordinary presentation to re-establish its exact
        bounded declaration.
        """
        expected = self._ordinary_width_contract_declarations(contract)
        if (
            self._last_ordinary_width_contract == contract
            and self._ordinary_width_inline_declarations(self.styles) == expected
        ):
            return

        self.styles.display = "block" if contract.display else "none"
        for rule_name, value in (
            ("width", contract.width),
            ("min_width", contract.min_width),
            ("max_width", contract.max_width),
        ):
            if value is None:
                self.styles.clear_rule(rule_name)
            else:
                setattr(self.styles, rule_name, value)
        self._last_ordinary_width_contract = contract

    def invalidate_width_contract_owner(self) -> None:
        """Require the next ordinary application to reclaim rail geometry."""
        self._last_ordinary_width_contract = None

    def sync_state(
        self,
        shell: LibraryShellState,
        preferences: LibraryRailPreferences,
        *,
        query: str = "",
        lifecycle: LibraryLifecycle = LibraryLifecycle.EXPANDED,
        onboarding_all_empty: bool = False,
    ) -> None:
        """Refresh the rail from new state.

        The search placeholder is not synced here: the Library screen
        refreshes the rail by full ``recompose`` (rebuilding a fresh
        ``LibraryRail`` whose ``__init__`` recomputes the context-aware
        placeholder), so a ``sync_state`` placeholder argument would be
        dead. It is set once at construction instead.

        Args:
            shell: Latest Library shell display state.
            preferences: Latest section preferences.
            query: Latest search box text.
            lifecycle: Current progressive-disclosure lifecycle.
            onboarding_all_empty: Whether one fresh evidence generation was empty.

        Returns:
            None.
        """
        section_shape = tuple(
            (
                section.section_id,
                section.title,
                tuple(row.row_id for row in section.rows),
            )
            for section in shell.sections
        )
        previous_shape = tuple(
            (
                section.section_id,
                section.title,
                tuple(row.row_id for row in section.rows),
            )
            for section in self.shell.sections
        )
        details_shape_matches = bool(
            len(self.shell.details_lines) > 2 and self.shell.details_lines[2]
        ) == bool(len(shell.details_lines) > 2 and shell.details_lines[2])
        can_patch_in_place = (
            self.is_mounted
            and self.lifecycle is LibraryLifecycle.EXPANDED
            and lifecycle is self.lifecycle
            and preferences == self.preferences
            and query == self.query
            and onboarding_all_empty == self.onboarding_all_empty
            and section_shape == previous_shape
            and details_shape_matches
        )

        self.shell = shell
        self.preferences = preferences
        self.query = query
        self.lifecycle = lifecycle
        self.onboarding_all_empty = onboarding_all_empty
        if not can_patch_in_place:
            self.refresh(recompose=True)
            return

        for section in shell.sections:
            for row in section.rows:
                try:
                    button = self.query_one(
                        f"#{LIBRARY_RAIL_ROW_PREFIX}{row.row_id}",
                        LibraryRailRowButton,
                    )
                except NoMatches:
                    continue
                previous_row = button.library_row
                if previous_row is not None and previous_row.count_emphasis:
                    button.remove_class(
                        f"library-rail-row-due-{previous_row.count_emphasis}"
                    )
                selected = row.row_id == shell.selected_row_id
                button.library_row = row
                button.row_id = row.row_id
                button.target_kind = row.target_kind
                button.target_id = row.target_id
                button.disabled = row.disabled
                button.tooltip = (
                    row.disabled_tooltip
                    if row.disabled and row.disabled_tooltip
                    else row.title
                )
                button.label = self._row_label(
                    row,
                    selected,
                    width=button.content_region.width,
                )
                button.set_class(selected, "library-rail-row-selected")
                if row.count_emphasis:
                    button.add_class(f"library-rail-row-due-{row.count_emphasis}")
                is_handoff = row.target_kind == "handoff"
                button.styles.height = 2 if is_handoff else 1
                button.styles.min_height = 2 if is_handoff else 1

        details_lines = shell.details_lines
        try:
            self.query_one("#library-details-runtime", Static).update(
                library_dim_label_text(
                    "Source", details_lines[0] if details_lines else ""
                )
            )
            self.query_one("#library-details-body", Static).update(
                details_lines[1] if len(details_lines) > 1 else ""
            )
            if len(details_lines) > 2 and details_lines[2]:
                self.query_one("#library-details-db-sizes", Static).update(
                    library_dim_label_text("DB sizes", details_lines[2])
                )
        except NoMatches:
            self.refresh(recompose=True)

    def apply_selection(
        self,
        shell: LibraryShellState,
        *,
        lifecycle: LibraryLifecycle | None = None,
        onboarding_all_empty: bool | None = None,
    ) -> None:
        """Update route selection without recomposing the whole rail.

        Route selection changes only the leading marker and selected class;
        counts, sections, search state, and Details remain unchanged. Updating
        the two affected rows in place avoids remounting the full Library rail
        during a canvas-to-canvas route switch.

        Args:
            shell: Latest shell state carrying the new selected row.
            lifecycle: Current progressive-disclosure lifecycle, when supplied.
            onboarding_all_empty: Latest fresh all-empty evidence, when supplied.

        Returns:
            None.
        """
        previous_row_id = self.shell.selected_row_id
        self.shell = shell
        if lifecycle is not None:
            self.lifecycle = lifecycle
        if onboarding_all_empty is not None:
            self.onboarding_all_empty = onboarding_all_empty
        changed_row_ids = {previous_row_id, shell.selected_row_id} - {""}
        rows = {
            row.row_id: row
            for section in shell.sections
            for row in section.rows
            if row.row_id in changed_row_ids
        }
        for row_id, row in rows.items():
            try:
                button = self.query_one(
                    f"#{LIBRARY_RAIL_ROW_PREFIX}{row_id}", LibraryRailRowButton
                )
            except NoMatches:
                continue
            selected = row_id == shell.selected_row_id
            button.library_row = row
            button.label = self._row_label(
                row,
                selected,
                width=button.content_region.width,
            )
            if button.has_class("library-rail-row-selected") != selected:
                button.set_class(selected, "library-rail-row-selected")

    def _section_open(self, section_id: str) -> bool:
        return bool(getattr(self.preferences, f"{section_id}_open", True))

    @staticmethod
    def _count_suffix(count: int | None, count_known: bool) -> str:
        if count is None:
            return ""
        if count_known:
            return f" ({count})"
        return f" ({count}+)"

    #: LIB-15: matches the F-014 loading placeholder's own width (" (…)").
    #: A ``count_pending`` row's gloss-fit check reserves at least this much
    #: room for the count that has not arrived yet, so the gloss's fate does
    #: not flip the instant a real (but still short) count lands -- see
    #: ``LibraryRailRow.count_pending``'s docstring for the full rationale.
    _GLOSS_FIT_MIN_COUNT_WIDTH = len(" (…)")

    @staticmethod
    def _row_label(row: LibraryRailRow, selected: bool, width: int = 0) -> str:
        """Build a rail row's label, fitted to ``width`` cells when known.

        Fitting order (F-015): the F-013 subtitle drops first, then the
        title ellipsize-truncates (never mid-word, LIB-18), then the
        handoff meta line drops; the count is never truncated. ``width`` 0
        (compose time, before layout) renders the full label -- each
        ``LibraryRailRowButton`` refits itself via ``on_resize`` once it
        learns its rendered width.
        """
        prefix = f"{'▸' if selected else ' '} "
        # F-014: one count policy -- a dim "(…)" placeholder while the
        # count is in flight, the count (or "+" estimate) when known, and
        # no suffix at all when the source is off.
        if row.count_loading:
            count_markup = " [dim](…)[/dim]"
            count_plain = " (…)"
        else:
            count_markup = row.count_display or LibraryRail._count_suffix(
                row.count, row.count_known
            )
            count_plain = count_markup
        raw_title = _truncate_row_title(row.title)
        subtitle_markup = ""
        if row.subtitle:
            subtitle_markup = f" [dim]— {escape_markup(row.subtitle)}[/dim]"
        if width > 0:
            fixed_plain = f"{prefix}{raw_title}{count_plain}"
            if len(fixed_plain) > width:
                # The title absorbs the squeeze; the count never clips.
                subtitle_markup = ""
                title_budget = width - len(prefix) - len(count_plain)
                candidate_title = row.short_title or raw_title
                raw_title = (
                    _fit_title_no_mid_word_cut(candidate_title, title_budget)
                    if title_budget >= 4
                    else ""
                )
            elif row.subtitle:
                # task-2236 (R2): the gloss renders whole or not at all --
                # a partial gloss is noise at real rail widths (the
                # review's "imported…"/"saved…" complaint), so when the
                # full gloss doesn't fit after title + count, it drops
                # and the row falls back to its tooltip.
                #
                # LIB-15: a ``count_pending`` row (count not fetched yet,
                # but will be) reserves at least the F-014 placeholder's
                # width for the count column here, even while the count is
                # still ``None`` -- otherwise the gloss decision uses 0
                # cells for "count not here yet" and a real (if short)
                # count's width once it lands, silently flipping the
                # gloss's visibility at an UNCHANGED terminal width.
                reserved_count_len = len(count_plain)
                if row.count_pending and row.count is None:
                    reserved_count_len = max(
                        reserved_count_len, LibraryRail._GLOSS_FIT_MIN_COUNT_WIDTH
                    )
                gloss_fit_len = len(prefix) + len(raw_title) + reserved_count_len
                if gloss_fit_len + len(f" — {row.subtitle}") > width:
                    subtitle_markup = ""
        label = f"{prefix}{escape_markup(raw_title)}{count_markup}{subtitle_markup}"
        if row.target_kind == "handoff":
            # F-011: a meta line survives ONLY where it discriminates --
            # handoff rows are a two-step trip out of Library, unlike the
            # plain canvas rows around them. task-2854: "opens Study" was
            # false for THIS click -- it opens a Library-local staging
            # canvas ("Continue in Study" lives inside that canvas, one
            # click later, and is the one that actually leaves Library for
            # the Study screen family).
            # task-4023 AC#7: "opens staging canvas" traded the old lie
            # ("opens Study") for internal jargon, printed three times in
            # the primary nav. User language: the click shows what will
            # carry over to Study before anything leaves the Library.
            meta_text = "see what carries over"
            meta_line = f"\n    {meta_text}"
            # LIB-18: the meta line is the LOWEST-priority element (below
            # even the gloss) -- at width 0 (compose time) it always shows,
            # matching every other element's unfitted-until-resize
            # behavior; once a real width is known, it shows only if it
            # fits whole. Textual's own Static/Button overflow otherwise
            # silently ellipsizes it mid-word ("opens stagin…"), exactly
            # the F-015 pathology this file already exists to prevent for
            # the count.
            if width <= 0 or len(meta_line) - 1 <= width:
                label += meta_line
        return label

    def compose(self) -> ComposeResult:
        """Render the search input, source sections, and Details disclosure.

        Returns:
            ComposeResult with the search box, one header + body per section,
            and the Details header + body.
        """
        heading_row = Horizontal(id="library-rail-heading")
        heading_row.styles.height = 1
        heading_row.styles.min_height = 1
        with heading_row:
            heading = Static("Navigation", id="library-rail-heading-label")
            heading.styles.height = 1
            heading.styles.width = "1fr"
            yield heading
            collapse = Button(
                "Collapse",
                id="library-rail-collapse",
                compact=True,
            )
            collapse.tooltip = "Collapse Library navigation"
            collapse.styles.width = "auto"
            collapse.styles.height = 1
            collapse.styles.min_height = 1
            collapse.styles.padding = (0, 1)
            collapse.styles.border = ("none", "transparent")
            yield collapse
        if self.lifecycle in (LibraryLifecycle.UNKNOWN, LibraryLifecycle.STARTER):
            for row_id in (LIBRARY_ROW_INGEST_MEDIA, LIBRARY_ROW_CREATE_NOTE):
                yield self._compose_row_button(self._row(row_id))
            yield Button("Explore all tools", id="library-rail-explore-all")
            return
        if self.top_action_factory is not None:
            yield from self.top_action_factory()
        yield LibraryRailSearchInput(
            value=self.query,
            placeholder=self.search_placeholder,
            id="library-search-input",
        )
        for section in self.shell.sections:
            yield from self._compose_section(section)

        details_open = self._section_open("details")
        yield DestinationRailSectionHeader(
            "Details",
            section_id="library-details",
            open=details_open,
            id="library-rail-section-header-details",
        )
        details_body = Vertical(
            id="library-rail-section-body-details",
            classes="library-rail-section-body",
        )
        details_body.styles.height = "auto"
        details_body.display = details_open
        # TASK-23025 considered growing this body on demand (it is ~13
        # widgets mounted display=False on the default route), but the
        # closed body's children are a QUERIED contract: the counts line and
        # the workspace Console-handoff state are read while the disclosure
        # is closed by seven tests across four files (e.g.
        # ``#library-use-in-console`` disabled-state in
        # test_destination_shells). Deferring it is a contract change, left
        # to its own task.
        with details_body:
            yield from self._compose_details_body_children()
        if self.lifecycle is LibraryLifecycle.EXPANDED and self.onboarding_all_empty:
            yield Button(
                "Back to Get started",
                id="library-rail-back-to-starter",
                compact=True,
            )

    def _compose_details_body_children(self) -> ComposeResult:
        """Build the Details disclosure's children from current shell state."""
        yield Static(
            "Status",
            id="library-details-group-status",
            classes="library-details-group library-details-group-first",
        )
        details_lines = self.shell.details_lines
        runtime_value = details_lines[0] if details_lines else ""
        yield Static(
            # F-013: "Source", not "Runtime" -- the line says where the
            # Library's content lives (this device or a server), and
            # "Runtime" taught nothing.
            library_dim_label_text("Source", runtime_value),
            id="library-details-runtime",
            classes="library-details-row",
        )
        counts_or_error = details_lines[1] if len(details_lines) > 1 else ""
        yield Static(
            counts_or_error,
            id="library-details-body",
            classes="library-details-row",
            markup=False,
        )
        if len(details_lines) > 2 and details_lines[2]:
            # F-014: the DB-size telemetry relocated out of the app
            # footer lives here -- third Status row, only when the
            # shell actually carries it (never an "N/A" triplet).
            yield Static(
                library_dim_label_text("DB sizes", details_lines[2]),
                id="library-details-db-sizes",
                classes="library-details-row",
            )
        if self.workspaces_body_factory is not None:
            yield from self.workspaces_body_factory()

    def _row(self, row_id: str) -> LibraryRailRow:
        """Return one canonical row from the full shell state."""
        return next(
            row
            for section in self.shell.sections
            for row in section.rows
            if row.row_id == row_id
        )

    def _compose_row_button(self, row: LibraryRailRow) -> LibraryRailRowButton:
        """Build one production Library rail row button."""
        selected = row.row_id == self.shell.selected_row_id
        is_handoff = row.target_kind == "handoff"
        button = LibraryRailRowButton(
            self._row_label(row, selected),
            id=f"{LIBRARY_RAIL_ROW_PREFIX}{row.row_id}",
            classes="library-rail-row",
            compact=True,
            disabled=row.disabled,
        )
        button.row_id = row.row_id
        button.target_kind = row.target_kind
        button.target_id = row.target_id
        button.library_row = row
        button.tooltip = (
            row.disabled_tooltip if row.disabled and row.disabled_tooltip else row.title
        )
        button.set_class(selected, "library-rail-row-selected")
        if row.count_emphasis:
            button.add_class(f"library-rail-row-due-{row.count_emphasis}")
        button.styles.height = 2 if is_handoff else 1
        button.styles.min_height = 2 if is_handoff else 1
        return button

    def _compose_section(self, section: LibraryRailSectionState) -> ComposeResult:
        open_state = self._section_open(section.section_id)
        yield DestinationRailSectionHeader(
            section.title,
            section_id=f"library-{section.section_id}",
            open=open_state,
            id=f"library-rail-section-header-{section.section_id}",
        )
        body = Vertical(
            id=f"library-rail-section-body-{section.section_id}",
            classes="library-rail-section-body",
        )
        body.styles.height = "auto"
        body.display = open_state
        with body:
            for row in section.rows:
                yield self._compose_row_button(row)
