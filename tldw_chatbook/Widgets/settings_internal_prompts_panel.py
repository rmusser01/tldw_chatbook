"""Settings "Internal Prompts" panel: browse the registry prompts grouped by
subsystem with customized / default-changed badges, filter by search, and
(Task 4) open the editor modal to save/reset overrides.

Self-contained editor pattern (mirrors SettingsThemeEditor): owns its state,
posts a Modified message the screen watches for the sidebar dirty-marker."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches, QueryError
from textual.message import Message
from textual.widgets import Button, Input, Static

from tldw_chatbook.Internal_Prompts import authoring
from tldw_chatbook.Internal_Prompts.catalog import CATALOG


def _row_id(prompt_id: str) -> str:
    return "prompt-row-" + prompt_id.replace(".", "__")


class InternalPromptsPanel(Vertical):
    """Browse + edit internal prompts. Title is rendered by the screen."""

    class Modified(Message):
        def __init__(self, customized_count: int) -> None:
            self.customized_count = customized_count
            super().__init__()

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Search prompts…", id="internal-prompts-search")
        with VerticalScroll(id="internal-prompts-list"):
            for subsystem, specs in authoring.iter_specs_by_subsystem():
                yield Static(
                    f"{subsystem}  ({len(specs)})",
                    classes="internal-prompts-group-header",
                    id="group-header-" + subsystem,
                )
                for spec in specs:
                    yield self._make_row(spec)

    def _make_row(self, spec) -> Button:
        st = authoring.override_state(spec.id)
        label = spec.title
        badges = []
        if st.customized:
            badges.append("● customized")
        if st.default_changed:
            badges.append("⟳ default changed")
        if badges:
            label = f"{spec.title}   [{'  '.join(badges)}]"
        row = Button(label, id=_row_id(spec.id), classes="internal-prompt-row")
        row.tooltip = spec.title + " — " + spec.description
        if st.customized:
            row.add_class("row-customized")
        if st.default_changed:
            row.add_class("row-default-changed")
        # carry the prompt id for Task 4's activation handler
        row.prompt_id = spec.id  # type: ignore[attr-defined]
        return row

    @on(Input.Changed, "#internal-prompts-search")
    def _on_search(self, event: Input.Changed) -> None:
        needle = event.value.strip().lower()
        for subsystem, specs in authoring.iter_specs_by_subsystem():
            any_visible = False
            for spec in specs:
                match = (not needle) or needle in spec.title.lower() \
                    or needle in spec.description.lower() or needle in spec.id.lower()
                try:
                    self.query_one("#" + _row_id(spec.id), Button).display = match
                except (NoMatches, QueryError):
                    continue
                any_visible = any_visible or match
            try:
                self.query_one("#group-header-" + subsystem, Static).display = any_visible
            except (NoMatches, QueryError):
                pass

    def _refresh_row(self, prompt_id: str) -> None:
        """Targeted in-place badge refresh for one row (no recompose)."""
        try:
            row = self.query_one("#" + _row_id(prompt_id), Button)
        except (NoMatches, QueryError):
            return
        st = authoring.override_state(prompt_id)
        spec = CATALOG[prompt_id]
        badges = []
        if st.customized:
            badges.append("● customized")
        if st.default_changed:
            badges.append("⟳ default changed")
        row.label = spec.title + (f"   [{'  '.join(badges)}]" if badges else "")
        row.set_class(st.customized, "row-customized")
        row.set_class(st.default_changed, "row-default-changed")
