"""Library shell rail: search box, source sections, and Details disclosure."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from rich.markup import escape as escape_markup
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.events import Key, Resize
from textual.widget import Widget
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_rail_state import LibraryRailPreferences
from tldw_chatbook.Library.library_shell_state import (
    LibraryRailRow,
    LibraryRailSectionState,
    LibraryShellState,
)
from tldw_chatbook.Widgets.destination_rail import DestinationRailSectionHeader
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


def _visible_row_title(title: str) -> str:
    """Return the rail-safe row title: capped at the rail budget, escaped.

    Used by the Library content canvases (conversations/media rows),
    which interpolate the result into markup-parsed ``Button`` labels.
    The rail itself builds labels via ``LibraryRail._row_label`` (F-015
    width-fitting), which truncates raw and escapes at build time.
    """
    return escape_markup(_truncate_row_title(title))


class LibraryRailSearchInput(Input):
    """Rail search box where a second "/" re-arms the query instead of typing.

    "/" is the Library screen's focus-the-search key (F-012); once the box
    itself has focus the screen's on_key never sees printable keys, so a
    second "/" would insert a literal slash into the query -- the settings
    screen's task-1584 live trap, solved the same way here: intercept it
    and select-all so the next keystroke replaces the stale text.
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


class LibraryRail(RecomposeCaptureGuard, Vertical):
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
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.shell = shell
        self.preferences = preferences
        self.query = query
        self.search_placeholder = search_placeholder
        self.workspaces_body_factory = workspaces_body_factory
        self.top_action_factory = top_action_factory
        self.styles.width = "3fr"
        self.styles.min_width = 24

    def sync_state(
        self,
        shell: LibraryShellState,
        preferences: LibraryRailPreferences,
        *,
        query: str = "",
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

        Returns:
            None.
        """
        self.shell = shell
        self.preferences = preferences
        self.query = query
        self.refresh(recompose=True)

    def _section_open(self, section_id: str) -> bool:
        return bool(getattr(self.preferences, f"{section_id}_open", True))

    @staticmethod
    def _count_suffix(count: int | None, count_known: bool) -> str:
        if count is None:
            return ""
        if count_known:
            return f" ({count})"
        return f" ({count}+)"

    @staticmethod
    def _row_label(row: LibraryRailRow, selected: bool, width: int = 0) -> str:
        """Build a rail row's label, fitted to ``width`` cells when known.

        Fitting order (F-015): the F-013 subtitle drops first, then the
        title ellipsize-truncates; the count is never truncated. ``width``
        0 (compose time, before layout) renders the full label -- each
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
                raw_title = (
                    _truncate_row_title(raw_title, title_budget)
                    if title_budget >= 4
                    else ""
                )
            elif row.subtitle:
                # task-2236 (R2): the gloss renders whole or not at all --
                # a partial gloss is noise at real rail widths (the
                # review's "imported…"/"saved…" complaint), so when the
                # full gloss doesn't fit after title + count, it drops
                # and the row falls back to its tooltip.
                if len(fixed_plain) + len(f" — {row.subtitle}") > width:
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
            label += "\n    opens staging canvas"
        return label

    def compose(self) -> ComposeResult:
        """Render the search input, source sections, and Details disclosure.

        Returns:
            ComposeResult with the search box, one header + body per section,
            and the Details header + body.
        """
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
        with details_body:
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
                selected = row.row_id == self.shell.selected_row_id
                # F-011: one-line rows by default -- the old blanket second
                # line ("in Library" on all ~11 rows) was pure stutter and
                # the reason the Create section was unreachable at 100x30
                # and Details clipped even at 170x50 (2-line rows + 1-line
                # bottom margin = 3 terminal lines per row). A meta line
                # survives ONLY where it discriminates: handoff rows leave
                # the Library entirely for the Study screen family.
                # Label construction lives in `_row_label` (count policy
                # F-014, width-fitting F-015); compose renders the unfitted
                # label and `_refit_row_labels` fits it post-layout.
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
                # Read by the button's own on_resize to rebuild the label
                # whenever its width changes (F-015).
                button.library_row = row
                button.tooltip = (
                    row.disabled_tooltip
                    if row.disabled and row.disabled_tooltip
                    else row.title
                )
                button.set_class(selected, "library-rail-row-selected")
                if row.count_emphasis:
                    button.add_class(f"library-rail-row-due-{row.count_emphasis}")
                button.styles.height = 2 if is_handoff else 1
                button.styles.min_height = 2 if is_handoff else 1
                yield button
