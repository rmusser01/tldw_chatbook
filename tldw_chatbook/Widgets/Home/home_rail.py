"""Home triage rail: sections of selectable work rows."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

from tldw_chatbook.Home.dashboard_state import HomeRailSectionState, HomeTriageState
from tldw_chatbook.Home.home_rail_state import HomeRailPreferences
from tldw_chatbook.Widgets.Console.console_rail_section import ConsoleRailSectionHeader

HOME_RAIL_ROW_PREFIX = "home-row-"

_MAX_HOME_ROW_TITLE = 20


def _visible_row_title(title: str) -> str:
    """Return a rail-safe visible title that does not clip in narrow panes."""
    readable = str(title).strip()
    if len(readable) <= _MAX_HOME_ROW_TITLE:
        return readable
    return f"{readable[: _MAX_HOME_ROW_TITLE - 3].rstrip()}..."


class HomeRail(Vertical):
    """Render the Home triage sections as selectable rows.

    Attributes:
        triage: Current triage display state.
        preferences: Section open/collapsed preferences.
    """

    def __init__(
        self,
        triage: HomeTriageState,
        preferences: HomeRailPreferences,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.triage = triage
        self.preferences = preferences
        self.styles.width = "3fr"
        self.styles.min_width = 24

    def sync_state(
        self,
        triage: HomeTriageState,
        preferences: HomeRailPreferences,
    ) -> None:
        """Refresh the rail from new state.

        Args:
            triage: Latest triage display state.
            preferences: Latest section preferences.

        Returns:
            None.
        """
        self.triage = triage
        self.preferences = preferences
        self.refresh(recompose=True)

    def _section_open(self, section_id: str) -> bool:
        return bool(getattr(self.preferences, f"{section_id}_open", True))

    @staticmethod
    def _section_title(section: HomeRailSectionState) -> str:
        return f"{section.title} ({section.count})" if section.count else section.title

    def compose(self) -> ComposeResult:
        """Render section headers, rows, and the details disclosure.

        Returns:
            ComposeResult with one header + body block per section.
        """
        row_index = 0
        for section in self.triage.sections:
            open_state = self._section_open(section.section_id)
            yield ConsoleRailSectionHeader(
                self._section_title(section),
                section_id=f"home-{section.section_id}",
                open=open_state,
                id=f"home-rail-section-header-{section.section_id}",
            )
            body = Vertical(
                id=f"home-rail-section-body-{section.section_id}",
                classes="home-rail-section-body",
            )
            body.styles.height = "auto"
            if not open_state:
                body.styles.display = "none"
            with body:
                if section.rows:
                    for row in section.rows:
                        selected = row.row_id == self.triage.selected_row_id
                        marker = "▸" if selected else " "
                        source_line = (
                            f"{row.source} - {row.age_label}"
                            if row.age_label
                            else row.source
                        )
                        button = Button(
                            f"{marker} {row.glyph} {_visible_row_title(row.title)}"
                            f"\n    {source_line}",
                            id=f"{HOME_RAIL_ROW_PREFIX}{row_index}",
                            classes="home-rail-row",
                            compact=True,
                        )
                        button.row_id = row.row_id
                        button.tooltip = row.title
                        button.set_class(selected, "home-rail-row-selected")
                        button.styles.height = 2
                        button.styles.min_height = 2
                        yield button
                        row_index += 1
                else:
                    yield Static(
                        section.empty_copy,
                        id=f"home-rail-empty-{section.section_id}",
                        classes="home-rail-empty-copy",
                        markup=False,
                    )
        details_open = self._section_open("details")
        yield ConsoleRailSectionHeader(
            "Details",
            section_id="home-details",
            open=details_open,
            id="home-rail-section-header-details",
        )
        details_body = Vertical(
            id="home-rail-section-body-details",
            classes="home-rail-section-body",
        )
        details_body.styles.height = "auto"
        if not details_open:
            details_body.styles.display = "none"
        with details_body:
            yield Static(
                "\n".join(self.triage.details_lines),
                id="home-details-body",
                classes="home-rail-empty-copy",
                markup=False,
            )
