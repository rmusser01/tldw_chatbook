# ruff: noqa: F811
"""Image-gen P3 Tasks 2-3: generate buttons + messages on the avatar row and
per-state expression slots in the character editor (Task 2), plus the
screen-level generate handler + worker that turns those messages into an
actual image-gen call (Task 3).

Widget-level coverage (Task 2) mirrors ``test_personas_character_editor_avatar.py``'s
bare-``PersonasCharacterEditorWidget`` host harness (a small ``App`` subclass
with ``on_<message>`` handlers that record posted messages) and
``test_personas_expression_slots.py``'s ``_UnsavedEditorHost``/
``test_import_export_buttons_disabled_for_unsaved_character`` gating pattern.
No DB/screen wiring is needed there: those buttons only post messages and
participate in ``_sync_expression_slots_enabled``, both of which are
observable straight off the bare widget.

Screen-level coverage (Task 3) reuses ``test_personas_expression_slots.py``'s
``personas_editor_with_saved_character`` fixture (imported directly rather
than duplicated - the ``# ruff: noqa: F811`` above is the repo's established
idiom for this, e.g. ``Tests/Character_Chat/test_character_persona_scope_
service.py``, since pyflakes otherwise flags every test that takes the
imported fixture name as a parameter as "redefining" the import).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button

from tldw_chatbook.UI.Screens import personas_screen as personas_screen_module
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
    CharacterAvatarGenerateRequested,
    CharacterExpressionGenerateAllRequested,
    CharacterExpressionGenerateRequested,
)

from Tests.UI.test_personas_expression_slots import (
    expr_db,  # noqa: F401 -- fixture dependency, not referenced by name
    personas_editor_with_saved_character,  # noqa: F401 -- used as a fixture
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


# ===== Image-gen P3 Task 3: screen-level handler + worker =====
#
# Screen-level coverage, reusing test_personas_expression_slots.py's
# ``personas_editor_with_saved_character`` fixture (real ``CharactersRAGDB``,
# a saved character, the editor open on a real ``PersonasScreen``) - the
# same fixture that flow's upload tests use. The worker is exercised
# directly (mirrors Tests/Chat/test_console_generation_actions.py's
# ``_regenerate_console_generation_variant`` pattern: call the async method,
# don't round-trip through run_worker/wait_for_complete); the handler's
# gating is exercised by monkeypatching ``screen.run_worker`` to record
# whether a worker was dispatched (mirrors this same file's
# ``test_export_handler_blocked_while_io_dialog_active`` pattern).


def _configure_backend(monkeypatch, name: str = "test_backend") -> None:
    """Patch the screen module's image-gen config/listing to a configured backend."""
    monkeypatch.setattr(
        personas_screen_module,
        "get_image_generation_config",
        lambda: SimpleNamespace(default_backend=name),
    )
    monkeypatch.setattr(
        personas_screen_module,
        "list_image_models_for_catalog",
        lambda: [{"name": name, "is_configured": True}],
    )


def _set_description(screen, text: str) -> PersonasCharacterEditorWidget:
    editor = screen.query_one(PersonasCharacterEditorWidget)
    editor._area("description").text = text
    return editor


def _capture_notifications(app) -> list:
    notifications: list = []
    app.notify = lambda message, severity="information", **kwargs: notifications.append(
        (str(message), severity)
    )
    return notifications


# ----- Worker: mandatory assertions (a), (b), (e) -----------------------


async def test_generate_worker_happy_path_applies_result(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character
    _configure_backend(monkeypatch)
    _set_description(screen, "A cheerful adventurer.")

    fake_result = SimpleNamespace(content=b"png-bytes", content_type="image/png")
    monkeypatch.setattr(
        personas_screen_module, "run_generation", lambda request: fake_result
    )
    apply_mock = AsyncMock()
    monkeypatch.setattr(screen, "_apply_expression_upload", apply_mock)
    screen._expression_generate_inflight.add((char_id, "thinking"))

    await screen._generate_expression_image_worker(char_id, "thinking")

    apply_mock.assert_awaited_once_with(
        char_id, "thinking", b"png-bytes", "image/png"
    )
    assert (char_id, "thinking") not in screen._expression_generate_inflight


async def test_generate_worker_stale_token_skips_write(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character
    _configure_backend(monkeypatch)
    _set_description(screen, "A cheerful adventurer.")

    def _mutating_run_generation(request):
        # Simulate the character editor session changing (e.g. Cancel, a
        # new create/edit, a save-in-place) while generation was in flight.
        screen._character_editor_generation += 1
        return SimpleNamespace(content=b"png-bytes", content_type="image/png")

    monkeypatch.setattr(
        personas_screen_module, "run_generation", _mutating_run_generation
    )
    apply_mock = AsyncMock()
    monkeypatch.setattr(screen, "_apply_expression_upload", apply_mock)

    await screen._generate_expression_image_worker(char_id, "speaking")

    apply_mock.assert_not_awaited()


async def test_generate_worker_failure_notifies_and_clears_inflight(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character
    _configure_backend(monkeypatch)
    _set_description(screen, "A cheerful adventurer.")

    def _boom(request):
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(personas_screen_module, "run_generation", _boom)
    apply_mock = AsyncMock()
    monkeypatch.setattr(screen, "_apply_expression_upload", apply_mock)
    notifications = _capture_notifications(app)
    screen._expression_generate_inflight.add((char_id, "error"))

    await screen._generate_expression_image_worker(char_id, "error")

    apply_mock.assert_not_awaited()
    assert (char_id, "error") not in screen._expression_generate_inflight
    assert notifications
    assert notifications[-1][1] == "error"
    assert "backend exploded" in notifications[-1][0]


# ----- Handler: mandatory assertions (c), (d), (f) -----------------------


async def test_generate_requested_empty_description_notifies_no_worker(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character
    # Deliberately NOT configuring a backend here: the empty-description
    # gate must fire before the backend-resolution gate, so this notify
    # text (not the backend-refusal text) proves the ordering.
    _set_description(screen, "   ")

    calls: list = []
    monkeypatch.setattr(screen, "run_worker", lambda *a, **k: calls.append(1))
    notifications = _capture_notifications(app)

    screen._handle_character_expression_generate_requested(
        CharacterExpressionGenerateRequested("thinking")
    )

    assert calls == []
    assert notifications
    assert notifications[-1] == ("Add a description first.", "warning")


async def test_generate_requested_inflight_second_click_single_generation(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character
    _configure_backend(monkeypatch)
    _set_description(screen, "A cheerful adventurer.")

    calls: list = []

    def _fake_run_worker(coro, *a, **k):
        calls.append(1)
        coro.close()  # avoid a "coroutine was never awaited" warning

    monkeypatch.setattr(screen, "run_worker", _fake_run_worker)
    notifications = _capture_notifications(app)

    screen._handle_character_expression_generate_requested(
        CharacterExpressionGenerateRequested("thinking")
    )
    screen._handle_character_expression_generate_requested(
        CharacterExpressionGenerateRequested("thinking")
    )

    assert calls == [1]  # only the first click dispatched a worker
    assert (char_id, "thinking") in screen._expression_generate_inflight
    assert notifications
    assert notifications[-1][1] == "warning"
    assert "already generating" in notifications[-1][0].lower()


async def test_generate_requested_unsaved_character_notifies_save_first(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character
    editor = screen.query_one(PersonasCharacterEditorWidget)
    editor.new_character()  # clears to an unsaved (id-less) session

    calls: list = []
    monkeypatch.setattr(screen, "run_worker", lambda *a, **k: calls.append(1))
    notifications = _capture_notifications(app)

    screen._handle_character_expression_generate_requested(
        CharacterExpressionGenerateRequested("thinking")
    )

    assert calls == []
    assert notifications
    # Same copy as the upload flow's save-first refusal.
    assert notifications[-1] == ("Save the character to add expressions.", "warning")


# ----- Handler: backend refusal (Console's exact copy) -------------------


async def test_generate_requested_backend_not_configured_notifies(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character
    monkeypatch.setattr(
        personas_screen_module,
        "get_image_generation_config",
        lambda: SimpleNamespace(default_backend="stable_diffusion_cpp"),
    )
    monkeypatch.setattr(
        personas_screen_module,
        "list_image_models_for_catalog",
        lambda: [{"name": "stable_diffusion_cpp", "is_configured": False}],
    )
    _set_description(screen, "A cheerful adventurer.")

    calls: list = []
    monkeypatch.setattr(screen, "run_worker", lambda *a, **k: calls.append(1))
    notifications = _capture_notifications(app)

    screen._handle_character_expression_generate_requested(
        CharacterExpressionGenerateRequested("thinking")
    )

    assert calls == []
    assert notifications
    assert notifications[-1][1] == "warning"
    assert (
        notifications[-1][0]
        == "Image backend 'stable_diffusion_cpp' is not enabled/configured. "
        "Check [image_generation] settings."
    )
