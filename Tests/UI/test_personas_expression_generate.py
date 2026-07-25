"""Image-gen P3 Task 2: generate buttons + messages on the avatar row and
per-state expression slots in the character editor.

Widget-level coverage only, mirroring ``test_personas_character_editor_avatar.py``'s
bare-``PersonasCharacterEditorWidget`` host harness (a small ``App`` subclass
with ``on_<message>`` handlers that record posted messages) and
``test_personas_expression_slots.py``'s ``_UnsavedEditorHost``/
``test_import_export_buttons_disabled_for_unsaved_character`` gating pattern.
No DB/screen wiring is needed: these buttons only post messages and
participate in ``_sync_expression_slots_enabled``, both of which are
observable straight off the bare widget.
"""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button

from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
    CharacterAvatarGenerateRequested,
    CharacterExpressionGenerateAllRequested,
    CharacterExpressionGenerateRequested,
)

pytestmark = pytest.mark.asyncio

EXPRESSION_STATES = ("thinking", "speaking", "error")


class _CaptureApp(App):
    """Bare editor host that records the three new generate messages."""

    def __init__(self) -> None:
        super().__init__()
        self.avatar_generate: list[CharacterAvatarGenerateRequested] = []
        self.expr_generate: list[CharacterExpressionGenerateRequested] = []
        self.expr_generate_all: list[CharacterExpressionGenerateAllRequested] = []

    def compose(self) -> ComposeResult:
        yield PersonasCharacterEditorWidget()

    def on_character_avatar_generate_requested(
        self, message: CharacterAvatarGenerateRequested
    ) -> None:
        self.avatar_generate.append(message)

    def on_character_expression_generate_requested(
        self, message: CharacterExpressionGenerateRequested
    ) -> None:
        self.expr_generate.append(message)

    def on_character_expression_generate_all_requested(
        self, message: CharacterExpressionGenerateAllRequested
    ) -> None:
        self.expr_generate_all.append(message)


# ===== Buttons exist =====


async def test_generate_buttons_present_in_compose():
    app = _CaptureApp()
    async with app.run_test():
        editor = app.query_one(PersonasCharacterEditorWidget)
        assert editor.query_one("#personas-char-editor-avatar-generate", Button) is not None
        assert (
            editor.query_one("#personas-char-editor-expr-generate-all", Button) is not None
        )
        for state in EXPRESSION_STATES:
            assert (
                editor.query_one(f"#personas-char-editor-expr-{state}-generate", Button)
                is not None
            )


# ===== Mandatory assertion (a): pressing each generate button posts the
# right message type (+ state for the per-state one). =====


async def test_avatar_generate_button_posts_avatar_generate_requested():
    app = _CaptureApp()
    async with app.run_test() as pilot:
        app.query_one("#personas-char-editor-avatar-generate", Button).press()
        await pilot.pause()
        assert len(app.avatar_generate) == 1
        assert isinstance(app.avatar_generate[0], CharacterAvatarGenerateRequested)


async def test_expression_generate_button_posts_message_with_correct_state():
    app = _CaptureApp()
    async with app.run_test() as pilot:
        editor = app.query_one(PersonasCharacterEditorWidget)
        # A saved character (has "id") is required for the per-state slot
        # buttons to be enabled/pressable - see the gating tests below.
        editor.load_character({"id": 1, "name": "A"})
        await pilot.pause()
        for state in EXPRESSION_STATES:
            app.expr_generate.clear()
            app.query_one(f"#personas-char-editor-expr-{state}-generate", Button).press()
            await pilot.pause()
            assert len(app.expr_generate) == 1
            assert app.expr_generate[0].state == state


async def test_generate_all_button_posts_generate_all_requested():
    app = _CaptureApp()
    async with app.run_test() as pilot:
        editor = app.query_one(PersonasCharacterEditorWidget)
        editor.load_character({"id": 1, "name": "A"})
        await pilot.pause()
        app.query_one("#personas-char-editor-expr-generate-all", Button).press()
        await pilot.pause()
        assert len(app.expr_generate_all) == 1
        assert isinstance(app.expr_generate_all[0], CharacterExpressionGenerateAllRequested)


# ===== Mandatory assertion (b): per-state generate + generate-all are
# disabled before the character is saved, enabled after - driving
# _sync_expression_slots_enabled the same way test_personas_expression_slots.py's
# import/export gating tests do. =====


async def test_expression_generate_buttons_disabled_for_unsaved_character():
    app = _CaptureApp()
    async with app.run_test() as pilot:
        editor = app.query_one(PersonasCharacterEditorWidget)
        editor.load_character({"name": "A"})  # no "id" key -> unsaved
        await pilot.pause()
        assert editor.expression_character_id() is None
        for state in EXPRESSION_STATES:
            assert (
                editor.query_one(
                    f"#personas-char-editor-expr-{state}-generate", Button
                ).disabled
                is True
            )
        assert (
            editor.query_one(
                "#personas-char-editor-expr-generate-all", Button
            ).disabled
            is True
        )


async def test_expression_generate_buttons_enabled_after_save():
    app = _CaptureApp()
    async with app.run_test() as pilot:
        editor = app.query_one(PersonasCharacterEditorWidget)
        editor.load_character({"name": "A"})
        await pilot.pause()
        # mark_saved is the save-in-place path (see its docstring): it
        # re-baselines the record with a freshly-assigned id and re-runs
        # _sync_expression_slots_enabled, the exact moment the slots flip
        # from disabled to enabled for a create session's first Save.
        editor.mark_saved({"id": 42, "name": "A", "version": 2})
        await pilot.pause()
        assert editor.expression_character_id() == 42
        for state in EXPRESSION_STATES:
            assert (
                editor.query_one(
                    f"#personas-char-editor-expr-{state}-generate", Button
                ).disabled
                is False
            )
        assert (
            editor.query_one(
                "#personas-char-editor-expr-generate-all", Button
            ).disabled
            is False
        )


# ===== Mandatory assertion (c): avatar-generate stays enabled pre-save
# (staged path - same as avatar Upload/Remove, which never gate on
# expression_character_id()). =====


async def test_avatar_generate_button_enabled_for_unsaved_character():
    app = _CaptureApp()
    async with app.run_test() as pilot:
        editor = app.query_one(PersonasCharacterEditorWidget)
        editor.load_character({"name": "A"})  # no "id" key -> unsaved
        await pilot.pause()
        assert editor.expression_character_id() is None
        assert (
            editor.query_one("#personas-char-editor-avatar-generate", Button).disabled
            is False
        )
