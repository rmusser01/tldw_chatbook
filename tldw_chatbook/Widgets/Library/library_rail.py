"""Library shell rail: search box, source sections, and Details disclosure."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from rich.markup import escape as escape_markup
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_rail_state import LibraryRailPreferences
from tldw_chatbook.Library.library_shell_state import (
    LibraryRailSectionState,
    LibraryShellState,
)
from tldw_chatbook.Widgets.Console.console_rail_section import ConsoleRailSectionHeader

LIBRARY_RAIL_ROW_PREFIX = "library-row-"

_MAX_LIBRARY_ROW_TITLE = 20


def library_dim_label_text(label: str, value: str) -> Text:
    """Build a "label · value" line with a dimmed label and a literal value.

    Used for the Details rail's scannable rows (e.g. "Runtime · Local",
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


def _visible_row_title(title: str) -> str:
    """Return a rail-safe visible title that does not clip in narrow panes.

    The result is markup-escaped (after truncating) because every caller
    interpolates it into a ``Button`` label, which Textual parses as Rich
    markup: an unescaped user title like ``[draft] Q3 plan [wip]`` would
    render with its bracketed segments consumed as (or crashing on) markup
    tags -- the same bug class as the search-history Button-label lesson.

    Args:
        title: Full row title (may contain user-supplied text).

    Returns:
        The escaped title, truncated with an ellipsis when longer than the
        rail budget.
    """
    readable = str(title).strip()
    if len(readable) > _MAX_LIBRARY_ROW_TITLE:
        readable = f"{readable[: _MAX_LIBRARY_ROW_TITLE - 3].rstrip()}..."
    return escape_markup(readable)


class LibraryRail(Vertical):
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
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.shell = shell
        self.preferences = preferences
        self.query = query
        self.search_placeholder = search_placeholder
        self.workspaces_body_factory = workspaces_body_factory
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

    def compose(self) -> ComposeResult:
        """Render the search input, source sections, and Details disclosure.

        Returns:
            ComposeResult with the search box, one header + body per section,
            and the Details header + body.
        """
        yield Input(
            value=self.query,
            placeholder=self.search_placeholder,
            id="library-search-input",
        )
        for section in self.shell.sections:
            yield from self._compose_section(section)

        details_open = self._section_open("details")
        yield ConsoleRailSectionHeader(
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
                library_dim_label_text("Runtime", runtime_value),
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
            if self.workspaces_body_factory is not None:
                yield from self.workspaces_body_factory()

    def _compose_section(self, section: LibraryRailSectionState) -> ComposeResult:
        open_state = self._section_open(section.section_id)
        yield ConsoleRailSectionHeader(
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
                marker = "▸" if selected else " "
                count_suffix = row.count_display or self._count_suffix(
                    row.count, row.count_known
                )
                section_hint = "opens screen" if row.target_kind == "screen" else "in Library"
                button = Button(
                    f"{marker} {_visible_row_title(row.title)}{count_suffix}"
                    f"\n    {section_hint}",
                    id=f"{LIBRARY_RAIL_ROW_PREFIX}{row.row_id}",
                    classes="library-rail-row",
                    compact=True,
                    disabled=row.disabled,
                )
                button.row_id = row.row_id
                button.target_kind = row.target_kind
                button.target_id = row.target_id
                button.tooltip = row.disabled_tooltip if row.disabled and row.disabled_tooltip else row.title
                button.set_class(selected, "library-rail-row-selected")
                if row.count_emphasis:
                    button.add_class(f"library-rail-row-due-{row.count_emphasis}")
                button.styles.height = 2
                button.styles.min_height = 2
                yield button
