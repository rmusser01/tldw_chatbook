"""Library shell rail: search box, source sections, and Details disclosure."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_rail_state import LibraryRailPreferences
from tldw_chatbook.Library.library_shell_state import (
    LibraryRailSectionState,
    LibraryShellState,
)
from tldw_chatbook.Widgets.Console.console_rail_section import ConsoleRailSectionHeader

LIBRARY_RAIL_ROW_PREFIX = "library-row-"

_MAX_LIBRARY_ROW_TITLE = 20


def _visible_row_title(title: str) -> str:
    """Return a rail-safe visible title that does not clip in narrow panes.

    Args:
        title: Full row title.

    Returns:
        The title, truncated with an ellipsis when longer than the rail budget.
    """
    readable = str(title).strip()
    if len(readable) <= _MAX_LIBRARY_ROW_TITLE:
        return readable
    return f"{readable[: _MAX_LIBRARY_ROW_TITLE - 3].rstrip()}..."


class LibraryRail(Vertical):
    """Render the Library shell rail: search, source sections, and Details.

    Attributes:
        shell: Current Library shell display state.
        preferences: Section open/collapsed preferences.
    """

    def __init__(
        self,
        shell: LibraryShellState,
        preferences: LibraryRailPreferences,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.shell = shell
        self.preferences = preferences
        self.styles.width = "3fr"
        self.styles.min_width = 24

    def sync_state(
        self,
        shell: LibraryShellState,
        preferences: LibraryRailPreferences,
    ) -> None:
        """Refresh the rail from new state.

        Args:
            shell: Latest Library shell display state.
            preferences: Latest section preferences.

        Returns:
            None.
        """
        self.shell = shell
        self.preferences = preferences
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
            placeholder="Search conversations…",
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
        if not details_open:
            details_body.styles.display = "none"
        with details_body:
            yield Static(
                "\n".join(self.shell.details_lines),
                id="library-details-body",
                classes="library-rail-empty-copy",
                markup=False,
            )

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
        if not open_state:
            body.styles.display = "none"
        with body:
            for row in section.rows:
                selected = row.row_id == self.shell.selected_row_id
                marker = "▸" if selected else " "
                count_suffix = self._count_suffix(row.count, row.count_known)
                section_hint = row.target_kind == "screen" and "opens screen" or "in Library"
                button = Button(
                    f"{marker} {_visible_row_title(row.title)}{count_suffix}"
                    f"\n    {section_hint}",
                    id=f"{LIBRARY_RAIL_ROW_PREFIX}{row.row_id}",
                    classes="library-rail-row",
                    compact=True,
                )
                button.row_id = row.row_id
                button.target_kind = row.target_kind
                button.target_id = row.target_id
                button.tooltip = row.title
                button.set_class(selected, "library-rail-row-selected")
                button.styles.height = 2
                button.styles.min_height = 2
                yield button
