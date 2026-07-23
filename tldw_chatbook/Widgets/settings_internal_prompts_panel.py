"""Settings "Internal Prompts" panel: browse the registry prompts grouped by
subsystem with customized / default-changed badges, filter by search, and
open the editor modal to save/reset overrides.

Self-contained editor pattern (mirrors SettingsThemeEditor): owns its state,
posts a Modified message the screen watches for the sidebar dirty-marker."""

from __future__ import annotations

import asyncio

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches, QueryError
from textual.message import Message
from textual.widgets import Button, Input, Static

from tldw_chatbook.Internal_Prompts import authoring
from tldw_chatbook.Internal_Prompts.catalog import CATALOG
from tldw_chatbook.Widgets.settings_internal_prompts_editor_modal import (
    InternalPromptEditorModal,
)


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
        # carry the prompt id for the row-activation handler below
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

    @on(Button.Pressed, ".internal-prompt-row")
    def _open_editor(self, event: Button.Pressed) -> None:
        event.stop()
        prompt_id = getattr(event.button, "prompt_id", None)
        if prompt_id is None:
            return
        spec = CATALOG[prompt_id]
        st = authoring.override_state(prompt_id)
        self.app.push_screen(
            InternalPromptEditorModal(spec=spec, active_text=st.active_text),
            lambda result, pid=prompt_id: self._on_editor_closed(pid, result),
        )

    def _on_editor_closed(self, prompt_id: str, result) -> None:
        if result is None:
            return
        # schedule the async apply (worker + refresh)
        self.run_worker(self._apply_editor_result(prompt_id, result), exclusive=False)

    async def _apply_editor_result(self, prompt_id: str, result: dict) -> None:
        action = result.get("action")
        if action == "save":
            ok = await self._persist(prompt_id, result.get("text", ""), reset=False)
        elif action == "reset":
            ok = await self._persist(prompt_id, "", reset=True)
        else:
            return
        if ok:
            self._refresh_row(prompt_id)
            self.post_message(self.Modified(authoring.customized_count()))
        else:
            self.app.notify("Could not save the prompt override.", severity="error")

    async def _persist(self, prompt_id: str, text: str, reset: bool) -> bool:
        def _io() -> bool:
            try:
                return (
                    authoring.reset_override(prompt_id)
                    if reset
                    else authoring.save_override(prompt_id, text)
                )
            except Exception:  # never let the worker crash the app
                return False
        return await asyncio.to_thread(_io)

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
