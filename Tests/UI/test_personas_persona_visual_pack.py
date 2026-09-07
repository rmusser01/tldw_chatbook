"""Mounted contract tests for Persona Visual authoring controls."""

from __future__ import annotations

import pytest
from textual.app import ComposeResult
from textual.widgets import Button, Input, OptionList, Select, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Persona_Visual.authoring import (
    PersonaVisualDraftInventory,
    PersonaVisualDraftRow,
)
from tldw_chatbook.Persona_Visual.contracts import RESERVED_STATES
from tldw_chatbook.Widgets.Persona_Widgets.persona_profile_editor_widget import (
    PersonaProfileEditorWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_persona_visual_pack_widget import (
    PersonasPersonaVisualPackWidget,
    PersonaVisualAddCustomRequested,
    PersonaVisualCancelRequested,
    PersonaVisualClearRequested,
    PersonaVisualCustomStateDialog,
    PersonaVisualImportRequested,
    PersonaVisualPreviewRequested,
    PersonaVisualReplaceRequested,
    PersonaVisualSaveRequested,
)

pytestmark = pytest.mark.asyncio


def _inventory(*, custom: bool = False, activatable: bool = True):
    rows = tuple(
        PersonaVisualDraftRow(
            state=state,
            label=state.replace("_", " ").title(),
            custom=False,
            configured=state == "idle",
            animation_id="idle" if state == "idle" else None,
            asset_key="idle-frame" if state == "idle" else None,
        )
        for state in RESERVED_STATES
    )
    if custom:
        rows += (
            PersonaVisualDraftRow(
                state="operator-ready",
                label="[bold]Operator ready[/bold]",
                custom=True,
                configured=True,
                animation_id="operator-ready",
                asset_key="operator-ready-frame",
            ),
        )
    return PersonaVisualDraftInventory(
        rows=rows,
        asset_count=2 if custom else 1,
        activatable=activatable,
        validation_reason=None if activatable else "persona_visual_draft_incomplete",
    )


class PackApp(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield PersonasPersonaVisualPackWidget()


async def test_saved_local_pack_shows_nine_baseline_rows_and_posts_typed_actions():
    received: list[object] = []

    class CaptureApp(PackApp):
        def on_persona_visual_preview_requested(self, message):
            received.append(message)

        def on_persona_visual_replace_requested(self, message):
            received.append(message)

        def on_persona_visual_clear_requested(self, message):
            received.append(message)

        def on_persona_visual_add_custom_requested(self, message):
            received.append(message)

        def on_persona_visual_import_requested(self, message):
            received.append(message)

    async with CaptureApp().run_test(size=(100, 32)) as pilot:
        widget = pilot.app.query_one(PersonasPersonaVisualPackWidget)
        widget.show_inventory(_inventory(), dirty=False)
        await pilot.pause()

        options = pilot.app.query_one("#personas-persona-visual-results", OptionList)
        assert options.option_count == 9
        await pilot.click("#personas-persona-visual-replace")
        await pilot.click("#personas-persona-visual-clear")
        await pilot.click("#personas-persona-visual-add-custom")
        await pilot.click("#personas-persona-visual-import")
        await pilot.pause()

    assert isinstance(received[0], PersonaVisualPreviewRequested)
    assert isinstance(received[1], PersonaVisualReplaceRequested)
    assert isinstance(received[2], PersonaVisualClearRequested)
    assert isinstance(received[3], PersonaVisualAddCustomRequested)
    assert isinstance(received[4], PersonaVisualImportRequested)
    assert received[0].state == received[1].state == received[2].state == "idle"


async def test_custom_label_is_plain_text_and_selected_preview_is_lazy():
    async with PackApp().run_test(size=(100, 32)) as pilot:
        widget = pilot.app.query_one(PersonasPersonaVisualPackWidget)
        widget.show_inventory(_inventory(custom=True), dirty=False)
        await pilot.pause()
        options = pilot.app.query_one("#personas-persona-visual-results", OptionList)
        assert options.option_count == 10
        prompt = str(options.get_option_at_index(9).prompt)
        assert "[bold]Operator ready[/bold]" in prompt
        assert (
            pilot.app.query_one("#personas-persona-visual-preview", Static).renderable
            == "Loading selected preview…"
        )


@pytest.mark.parametrize(
    ("availability", "copy"),
    (("unsaved", "Save Persona first"), ("server", "Save Local Copy first")),
)
async def test_ineligible_persona_states_explain_recovery_and_disable_actions(
    availability: str,
    copy: str,
):
    async with PackApp().run_test() as pilot:
        widget = pilot.app.query_one(PersonasPersonaVisualPackWidget)
        widget.set_availability(availability)
        await pilot.pause()
        assert copy in str(
            pilot.app.query_one("#personas-persona-visual-notice", Static).renderable
        )
        for button in widget.query(Button):
            assert button.disabled is True


async def test_staged_and_busy_states_keep_save_cancel_copy_honest():
    received: list[type] = []

    class CaptureApp(PackApp):
        def on_persona_visual_save_requested(self, message):
            received.append(type(message))

        def on_persona_visual_cancel_requested(self, message):
            received.append(type(message))

    async with CaptureApp().run_test() as pilot:
        widget = pilot.app.query_one(PersonasPersonaVisualPackWidget)
        widget.show_inventory(_inventory(), dirty=True)
        await pilot.pause()
        save = pilot.app.query_one("#personas-persona-visual-save", Button)
        cancel = pilot.app.query_one("#personas-persona-visual-cancel", Button)
        assert save.disabled is False and cancel.disabled is False
        await pilot.click(save)
        await pilot.click(cancel)
        widget.set_busy("importing")
        assert "Importing" in str(
            pilot.app.query_one("#personas-persona-visual-status", Static).renderable
        )
        assert cancel.disabled is False
        widget.set_busy("saving")
        assert "Saving" in str(
            pilot.app.query_one("#personas-persona-visual-status", Static).renderable
        )
        assert cancel.disabled is True

    assert received == [PersonaVisualSaveRequested, PersonaVisualCancelRequested]


async def test_profile_editor_embeds_pack_section_and_compact_layout_keeps_actions():
    class EditorApp(ConsolidatedCSSApp):
        def compose(self) -> ComposeResult:
            yield PersonaProfileEditorWidget()

    async with EditorApp().run_test(size=(80, 24)) as pilot:
        editor = pilot.app.query_one(PersonaProfileEditorWidget)
        editor.load_persona({"id": "p-1", "version": 2, "name": "Archivist"})
        await pilot.pause()
        pack = pilot.app.query_one(PersonasPersonaVisualPackWidget)
        assert pack is not None
        assert pack.has_class("-narrow")
        assert pilot.app.query_one(
            "#personas-persona-visual-import", Button
        ).label.plain == ("Import Pack…")
        assert not getattr(type(pack), "BINDINGS", ())


async def test_profile_editor_visual_session_token_changes_on_load_and_save():
    class EditorApp(ConsolidatedCSSApp):
        def compose(self) -> ComposeResult:
            yield PersonaProfileEditorWidget()

    async with EditorApp().run_test() as pilot:
        editor = pilot.app.query_one(PersonaProfileEditorWidget)
        initial = editor.persona_visual_session_token
        editor.load_persona({"id": "p-1", "version": 2, "name": "Archivist"})
        loaded = editor.persona_visual_session_token
        editor.mark_saved({"id": "p-1", "version": 3, "name": "Archivist"})

        assert loaded == initial + 1
        assert editor.persona_visual_session_token == loaded + 1


async def test_custom_state_dialog_returns_plain_typed_values():
    class DialogApp(ConsolidatedCSSApp):
        result: tuple[str, str, str] | None = None

        def on_mount(self) -> None:
            self.push_screen(
                PersonaVisualCustomStateDialog(),
                callback=lambda value: setattr(self, "result", value),
            )

    async with DialogApp().run_test() as pilot:
        await pilot.pause()
        pilot.app.screen.query_one(
            "#persona-visual-custom-key", Input
        ).value = "deep_focus"
        pilot.app.screen.query_one(
            "#persona-visual-custom-label", Input
        ).value = "[bold]Deep focus[/bold]"
        pilot.app.screen.query_one("#persona-visual-custom-kind", Select).value = "mood"
        await pilot.click("#persona-visual-custom-confirm")
        await pilot.pause()

        assert pilot.app.result == (
            "deep_focus",
            "[bold]Deep focus[/bold]",
            "mood",
        )
