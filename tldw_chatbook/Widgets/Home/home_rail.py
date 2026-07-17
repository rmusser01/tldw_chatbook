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

        B3 (task-282): when the section/row structure and open/collapsed
        preferences are byte-for-byte unchanged -- the common case for a
        row-selection click or a background count refresh that didn't
        actually add/remove rows -- patch the affected widgets directly
        instead of tearing down and remounting every row. Any change to
        the rows/sections themselves (added, removed, reordered, or
        re-labelled) still falls back to a full recompose.

        Args:
            triage: Latest triage display state.
            preferences: Latest section preferences.

        Returns:
            None.
        """
        previous_triage = self.triage
        previous_preferences = self.preferences
        self.triage = triage
        self.preferences = preferences
        if triage.sections == previous_triage.sections and preferences == previous_preferences:
            try:
                self._patch_selection(triage, previous_triage)
                return
            except Exception:
                # Defensive: fall through to a full recompose rather than
                # leave the rail showing a half-patched state.
                pass
        self.refresh(recompose=True)

    def _patch_selection(
        self, triage: HomeTriageState, previous_triage: HomeTriageState
    ) -> None:
        """Patch selection markers and details text in place (no recompose)."""
        if triage.selected_row_id != previous_triage.selected_row_id:
            self._patch_row_selection(triage.selected_row_id, previous_triage.selected_row_id)
        if triage.details_lines != previous_triage.details_lines:
            self.query_one("#home-details-body", Static).update(
                "\n".join(triage.details_lines)
            )

    def _patch_row_selection(
        self, new_selected_row_id: str, previous_selected_row_id: str
    ) -> None:
        """Toggle the marker/class on just the previously/newly selected rows."""
        changed_ids = {
            row_id for row_id in (new_selected_row_id, previous_selected_row_id) if row_id
        }
        if not changed_ids:
            return
        row_lookup = {
            row.row_id: row for section in self.triage.sections for row in section.rows
        }
        for button in self.query("Button.home-rail-row"):
            row_id = getattr(button, "row_id", "")
            if row_id not in changed_ids:
                continue
            row = row_lookup.get(row_id)
            if row is None:
                continue
            selected = row_id == new_selected_row_id
            marker = "▸" if selected else " "
            source_line = (
                f"{row.source} - {row.age_label}" if row.age_label else row.source
            )
            button.label = (
                f"{marker} {row.glyph} {_visible_row_title(row.title)}\n    {source_line}"
            )
            button.set_class(selected, "home-rail-row-selected")

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
