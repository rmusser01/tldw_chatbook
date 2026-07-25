# ruff: noqa: F811
"""Image-gen P3 Tasks 2-4: generate buttons + messages on the avatar row and
per-state expression slots in the character editor (Task 2), the
screen-level generate handler + worker that turns those messages into an
actual image-gen call (Task 3), and avatar generation / Generate-all /
the style picker (Task 4).

Widget-level coverage (Task 2) mirrors ``test_personas_character_editor_avatar.py``'s
bare-``PersonasCharacterEditorWidget`` host harness (a small ``App`` subclass
with ``on_<message>`` handlers that record posted messages) and
``test_personas_expression_slots.py``'s ``_UnsavedEditorHost``/
``test_import_export_buttons_disabled_for_unsaved_character`` gating pattern.
No DB/screen wiring is needed there: those buttons only post messages and
participate in ``_sync_expression_slots_enabled``, both of which are
observable straight off the bare widget.

Screen-level coverage (Tasks 3-4) reuses ``test_personas_expression_slots.py``'s
``personas_editor_with_saved_character`` fixture (imported directly rather
than duplicated - the ``# ruff: noqa: F811`` above is the repo's established
idiom for this, e.g. ``Tests/Character_Chat/test_character_persona_scope_
service.py``, since pyflakes otherwise flags every test that takes the
imported fixture name as a parameter as "redefining" the import).

Task 4's worker-level tests exercise ``_generate_expression_image_worker``
(single-slot, including the avatar) and ``_generate_all_expression_images_
worker`` directly - the same "call the async method, don't round-trip
through run_worker" pattern Task 3's own tests use - to prove the Task 4
refactor (``_generate_one_slot`` factored out of the original Task 3 worker
body) is behavior-preserving: this file's own 14 Task 3 tests are left
completely unchanged and must stay green.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from tldw_chatbook.Media_Creation.generation_templates import get_template
from tldw_chatbook.UI.Screens import personas_screen as personas_screen_module
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
    CharacterAvatarGenerateRequested,
    CharacterExpressionGenerateAllRequested,
    CharacterExpressionGenerateRequested,
    CharacterExpressionSetExportRequested,
    CharacterExpressionSetImportRequested,
    CharacterExpressionStylePickRequested,
)

from Tests.UI.test_personas_expression_slots import (
    expr_db,  # noqa: F401 -- fixture dependency, not referenced by name
    personas_editor_with_saved_character,  # noqa: F401 -- used as a fixture
)

pytestmark = pytest.mark.asyncio

EXPRESSION_STATES = ("thinking", "speaking", "error")


class _CaptureApp(App):
    """Bare editor host that records the three new generate messages, plus
    (Task 4) the style-pick message."""

    def __init__(self) -> None:
        super().__init__()
        self.avatar_generate: list[CharacterAvatarGenerateRequested] = []
        self.expr_generate: list[CharacterExpressionGenerateRequested] = []
        self.expr_generate_all: list[CharacterExpressionGenerateAllRequested] = []
        self.style_pick: list[CharacterExpressionStylePickRequested] = []
        self.expr_import: list[CharacterExpressionSetImportRequested] = []
        self.expr_export: list[CharacterExpressionSetExportRequested] = []

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

    def on_character_expression_style_pick_requested(
        self, message: CharacterExpressionStylePickRequested
    ) -> None:
        self.style_pick.append(message)

    def on_character_expression_set_import_requested(
        self, message: CharacterExpressionSetImportRequested
    ) -> None:
        self.expr_import.append(message)

    def on_character_expression_set_export_requested(
        self, message: CharacterExpressionSetExportRequested
    ) -> None:
        self.expr_export.append(message)


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


@pytest.mark.parametrize("width", [80, 120, 200])
async def test_expr_set_row_buttons_reachable_and_functional_at_width(width):
    """task-563 AC1: supersedes the original 200-col-only regression test
    below this docstring's history - a bare ``Static`` used to default to
    Textual's ``1fr`` width, letting the "Expressions" section header claim
    the whole row and push every sibling control off it (fixed by the
    ``width: auto`` on the header, still in effect). That fix only proved
    reachability at a 200-column viewport; live tmux verification separately
    found the row's four action buttons (Style pick, Generate all, Import
    set, Export set) still overflow and become unreachable at realistic
    narrower widths (120 cols: Import/Export; 80 cols: Generate all too),
    since a plain ``Horizontal`` has no wrap and (until this fix) no
    horizontal scrollbar either.

    The fix makes the row horizontally scrollable (``overflow-x: auto`` on
    ``.personas-char-editor-expr-set-row``, mirroring ``MainNavigationBar``'s
    own ``.main-nav`` idiom) instead of hard-clipping. This proves the fix
    at all three widths in two parts:

    - Reachable: after ``scroll_visible()`` (what a real user's mouse
      wheel/shift-wheel, or Tab - which auto-scrolls the focused widget into
      view - would do), the button's rendered region must fall entirely
      within the actual terminal viewport (``0 <= region.x`` and
      ``region.x + region.width <= width``). This is the assertion that
      actually discriminates the fix: ``Pilot.click`` itself was found to
      dispatch successfully even at an off-screen region (headless-mode
      leniency a real terminal's mouse-reporting protocol cannot replicate,
      since it cannot report coordinates beyond its own dimensions), so a
      raw click-succeeds check alone does not prove real-terminal
      reachability - the in-bounds region check does.
    - Functional: a still-independent proof the button itself works, via
      the same ``Button.press()`` this file's other tests already use.
    """
    app = _CaptureApp()
    async with app.run_test(size=(width, 50)) as pilot:
        editor = app.query_one(PersonasCharacterEditorWidget)
        editor.load_character({"id": 1, "name": "A"})  # saved -> row's buttons enabled
        await pilot.pause()

        for widget_id, captured, message_type in (
            (
                "personas-char-editor-style-pick",
                app.style_pick,
                CharacterExpressionStylePickRequested,
            ),
            (
                "personas-char-editor-expr-generate-all",
                app.expr_generate_all,
                CharacterExpressionGenerateAllRequested,
            ),
            (
                "personas-char-editor-expr-import",
                app.expr_import,
                CharacterExpressionSetImportRequested,
            ),
            (
                "personas-char-editor-expr-export",
                app.expr_export,
                CharacterExpressionSetExportRequested,
            ),
        ):
            node = editor.query_one(f"#{widget_id}", Button)
            node.scroll_visible(animate=False)
            await pilot.pause()
            await pilot.pause()  # let the (possibly animated-off) scroll settle
            region = node.region
            assert 0 <= region.x and region.x + region.width <= width, (
                f"{widget_id} rendered at x={region.x}, width={region.width} "
                f"- outside the {width}-column viewport even after "
                "scroll_visible(), so a real terminal's mouse could never "
                "reach it."
            )

            before = len(captured)
            node.press()
            await pilot.pause()
            assert len(captured) == before + 1, (
                f"{widget_id} did not post {message_type.__name__} at "
                f"width={width}."
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


async def test_generate_worker_no_open_editor_session_skips_write(
    personas_editor_with_saved_character, monkeypatch
):
    """Final review F2 (confirmed): the editor is cancelled after the
    handler dispatched this worker but before the worker's own session
    token is captured (the token is read fresh inside ``_generate_one_
    slot``, not passed in by the caller) - e.g. the user hits Cancel in the
    moment between a generate-button click and the worker actually
    starting. Without the fix, the freshly-captured token is ``None``
    (no editor open) and the post-generation re-check compares
    ``None != None``, which is ``False`` - so the stale-write guard
    silently passes and the result gets written with no editor open at
    all. The fix refuses outright when the freshly-captured token is
    ``None``."""
    app, screen, db, char_id = personas_editor_with_saved_character
    _configure_backend(monkeypatch)
    _set_description(screen, "A cheerful adventurer.")

    monkeypatch.setattr(
        personas_screen_module,
        "run_generation",
        lambda request: SimpleNamespace(
            content=b"png-bytes", content_type="image/png"
        ),
    )
    apply_mock = AsyncMock()
    monkeypatch.setattr(screen, "_apply_expression_upload", apply_mock)

    # The user cancels the editor after the handler dispatched the worker
    # but before the worker itself captures a session token.
    screen._finish_cancel_edit()
    assert screen._character_editor_session_token() is None

    screen._expression_generate_inflight.add((char_id, "thinking"))
    await screen._generate_expression_image_worker(char_id, "thinking")

    apply_mock.assert_not_awaited()
    assert (char_id, "thinking") not in screen._expression_generate_inflight


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


# ===== task-563 AC2: in-slot "Generating…" affordance =====
#
# Spec §1 promised a per-slot busy indicator while a generation runs; today
# the only feedback is the completion/failure notify or the "already
# generating" refusal on a second click. These tests pin the widget-level
# setters (PersonasCharacterEditorWidget.set_expression_generating /
# set_avatar_generating) and the screen-level wiring: set at dispatch,
# cleared on success AND failure (both flow through _generate_one_slot's
# existing ``finally``), and never leaked onto a DIFFERENT character's
# same-named slot after a mid-generation editor switch.


async def test_set_expression_generating_shows_and_clears_hint():
    app = _CaptureApp()
    async with app.run_test():
        editor = app.query_one(PersonasCharacterEditorWidget)
        editor.load_character({"id": 1, "name": "A"})  # saved -> enabled ("" hint)
        hint = editor.query_one("#personas-char-editor-expr-thinking-hint", Static)
        assert str(hint.renderable) == ""

        editor.set_expression_generating("thinking", True)
        assert str(hint.renderable) == "Generating…"

        editor.set_expression_generating("thinking", False)
        assert str(hint.renderable) == ""


async def test_set_expression_generating_clear_restores_unsaved_hint():
    app = _CaptureApp()
    async with app.run_test():
        editor = app.query_one(PersonasCharacterEditorWidget)
        editor.load_character({"name": "A"})  # no id -> unsaved
        editor.set_expression_generating("thinking", True)
        hint = editor.query_one("#personas-char-editor-expr-thinking-hint", Static)
        assert str(hint.renderable) == "Generating…"

        editor.set_expression_generating("thinking", False)
        assert str(hint.renderable) == "Save the character to add expressions."


async def test_set_avatar_generating_shows_and_clears_status():
    app = _CaptureApp()
    async with app.run_test():
        editor = app.query_one(PersonasCharacterEditorWidget)
        editor.load_character({"id": 1, "name": "A"})
        status = editor.query_one("#personas-char-editor-avatar-status", Static)
        assert str(status.renderable) == "Avatar: none"

        editor.set_avatar_generating(True)
        assert str(status.renderable) == "Avatar: generating…"

        editor.set_avatar_generating(False)
        assert str(status.renderable) == "Avatar: none"


async def test_generate_worker_shows_and_clears_in_slot_generating_hint(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character
    _configure_backend(monkeypatch)
    editor = _set_description(screen, "A cheerful adventurer.")
    hint = editor.query_one("#personas-char-editor-expr-thinking-hint", Static)

    seen: dict = {}

    def _run_generation(request):
        seen["during"] = str(hint.renderable)
        return SimpleNamespace(content=b"png-bytes", content_type="image/png")

    monkeypatch.setattr(personas_screen_module, "run_generation", _run_generation)
    screen._expression_generate_inflight.add((char_id, "thinking"))
    editor.set_expression_generating("thinking", True)  # what the handler does at dispatch

    await screen._generate_expression_image_worker(char_id, "thinking")

    assert seen["during"] == "Generating…"
    assert str(hint.renderable) == ""


async def test_generate_worker_failure_clears_in_slot_generating_hint(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character
    _configure_backend(monkeypatch)
    editor = _set_description(screen, "A cheerful adventurer.")
    hint = editor.query_one("#personas-char-editor-expr-error-hint", Static)

    def _boom(request):
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(personas_screen_module, "run_generation", _boom)
    screen._expression_generate_inflight.add((char_id, "error"))
    editor.set_expression_generating("error", True)

    await screen._generate_expression_image_worker(char_id, "error")

    assert str(hint.renderable) == ""


async def test_avatar_generate_worker_shows_and_clears_generating_status(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character
    _configure_backend(monkeypatch)
    editor = _set_description(screen, "A cheerful adventurer.")
    status = editor.query_one("#personas-char-editor-avatar-status", Static)

    seen: dict = {}

    def _run_generation(request):
        seen["during"] = str(status.renderable)
        return SimpleNamespace(content=b"png-bytes", content_type="image/png")

    monkeypatch.setattr(personas_screen_module, "run_generation", _run_generation)
    _capture_avatar_render_worker(screen)
    screen._expression_generate_inflight.add((char_id, "avatar"))
    editor.set_avatar_generating(True)

    await screen._generate_expression_image_worker(char_id, "avatar")

    assert seen["during"] == "Avatar: generating…"
    assert str(status.renderable) == "Avatar: embedded"


async def test_generate_worker_does_not_clear_generating_hint_of_switched_to_character(
    personas_editor_with_saved_character, monkeypatch
):
    """The leak guard: while A's "thinking" generation is in flight, the
    user switches the SAME editor widget to a DIFFERENT character B, which
    happens to have its own INDEPENDENT "thinking" generation already
    showing "Generating…" (started via a separate click after the switch).
    A's stale completion must not clear B's legitimately-in-flight
    indicator - the clear must only ever touch the (character_id, state)
    pair it was set for."""
    app, screen, db, char_id = personas_editor_with_saved_character
    _configure_backend(monkeypatch)
    editor = _set_description(screen, "CHARACTER A: a cheerful adventurer.")
    other_id = db.add_character_card({"name": "Grim"})
    hint = editor.query_one("#personas-char-editor-expr-thinking-hint", Static)

    def _run_generation(request):
        # Mid-flight for A's request: the user cancels and opens a
        # DIFFERENT character (B) in the same editor widget.
        screen._finish_cancel_edit()
        screen._character_editor_generation += 1
        screen._edit_mode = "edit"
        screen.state.select_entity(
            entity_kind="character", entity_id=str(other_id), entity_name="Grim"
        )
        screen._show_center("#ccp-character-editor-view")
        editor.load_character(
            {"id": other_id, "name": "Grim", "description": "B", "version": 1}
        )
        # B has its own, unrelated "thinking" generation already in flight.
        editor.set_expression_generating("thinking", True)
        return SimpleNamespace(content=b"png-bytes", content_type="image/png")

    monkeypatch.setattr(personas_screen_module, "run_generation", _run_generation)
    apply_mock = AsyncMock()
    monkeypatch.setattr(screen, "_apply_expression_upload", apply_mock)
    screen._expression_generate_inflight.add((char_id, "thinking"))
    editor.set_expression_generating("thinking", True)

    await screen._generate_expression_image_worker(char_id, "thinking")

    apply_mock.assert_not_awaited()  # A's stale write is dropped, as before
    # B's own (unrelated) "Generating…" must survive A's stale completion.
    assert str(hint.renderable) == "Generating…"


# ===== Image-gen P3 Task 4: avatar generate, Generate-all, style picker =====
#
# The refactor's own proof is above: this file's 14 original Task 3 tests
# (unchanged) stay green against ``_generate_one_slot`` factored out of the
# original ``_generate_expression_image_worker`` body. Everything below is
# new Task 4 coverage.


def _capture_avatar_render_worker(screen) -> list:
    """Patch ``screen.run_worker`` to record + swallow the avatar-render-kick
    coroutine ``_generate_one_slot``'s avatar branch dispatches, mirroring
    this file's existing ``coro.close()`` idiom for an un-awaited worker
    coroutine in a monkeypatched ``run_worker``."""
    calls: list = []

    def _fake_run_worker(coro, *args, **kwargs):
        calls.append(kwargs.get("group"))
        coro.close()

    screen.run_worker = _fake_run_worker
    return calls


# ----- Mandatory (a): avatar happy path stages via set_avatar_image --------


async def test_avatar_generate_worker_happy_path_stages_via_set_avatar_image(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character
    _configure_backend(monkeypatch)
    editor = _set_description(screen, "A cheerful adventurer.")

    fake_result = SimpleNamespace(content=b"avatar-bytes", content_type="image/png")
    monkeypatch.setattr(
        personas_screen_module, "run_generation", lambda request: fake_result
    )
    apply_mock = AsyncMock()
    monkeypatch.setattr(screen, "_apply_expression_upload", apply_mock)
    render_calls = _capture_avatar_render_worker(screen)
    notifications = _capture_notifications(app)
    screen._expression_generate_inflight.add((char_id, "avatar"))

    await screen._generate_expression_image_worker(char_id, "avatar")

    assert editor.current_avatar_bytes() == b"avatar-bytes"
    apply_mock.assert_not_awaited()  # NOT the expression-slot DB seam
    assert render_calls == ["personas-avatar-render"]
    assert (char_id, "avatar") not in screen._expression_generate_inflight
    assert notifications
    assert notifications[-1] == (
        "Avatar image generated — Save to keep it.",
        "information",
    )


async def test_avatar_generate_worker_stages_for_unsaved_character(
    personas_editor_with_saved_character, monkeypatch
):
    """Avatar generation is allowed pre-save - the in-flight key's
    character_id may be ``None`` (Task 4's gate widening)."""
    app, screen, db, char_id = personas_editor_with_saved_character
    editor = screen.query_one(PersonasCharacterEditorWidget)
    editor.new_character()  # clears to an unsaved (id-less) session
    _configure_backend(monkeypatch)
    editor._area("description").text = "A cheerful adventurer."

    fake_result = SimpleNamespace(content=b"avatar-bytes", content_type="image/png")
    monkeypatch.setattr(
        personas_screen_module, "run_generation", lambda request: fake_result
    )
    _capture_avatar_render_worker(screen)
    screen._expression_generate_inflight.add((None, "avatar"))

    await screen._generate_expression_image_worker(None, "avatar")

    assert editor.current_avatar_bytes() == b"avatar-bytes"
    assert (None, "avatar") not in screen._expression_generate_inflight


# ----- Mandatory (b): oversized avatar result -------------------------------


async def test_avatar_generate_worker_oversized_result_notifies_no_staging(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character
    _configure_backend(monkeypatch)
    editor = _set_description(screen, "A cheerful adventurer.")

    oversized = b"x" * (personas_screen_module.PERSONAS_AVATAR_MAX_BYTES + 1)
    fake_result = SimpleNamespace(content=oversized, content_type="image/png")
    monkeypatch.setattr(
        personas_screen_module, "run_generation", lambda request: fake_result
    )
    render_calls = _capture_avatar_render_worker(screen)
    notifications = _capture_notifications(app)
    screen._expression_generate_inflight.add((char_id, "avatar"))

    await screen._generate_expression_image_worker(char_id, "avatar")

    assert editor.current_avatar_bytes() is None  # not staged
    assert render_calls == []  # no render kick for a rejected result
    assert (char_id, "avatar") not in screen._expression_generate_inflight
    assert notifications
    assert notifications[-1][1] == "error"
    assert "5 MB" in notifications[-1][0]


# ----- Handler: avatar generate gates (no saved-character gate) ------------


async def test_avatar_generate_requested_unsaved_character_dispatches_worker(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character
    editor = screen.query_one(PersonasCharacterEditorWidget)
    editor.new_character()
    editor._area("description").text = "A cheerful adventurer."
    _configure_backend(monkeypatch)

    calls: list = []

    def _fake_run_worker(coro, *a, **k):
        calls.append(1)
        coro.close()

    monkeypatch.setattr(screen, "run_worker", _fake_run_worker)
    notifications = _capture_notifications(app)

    screen._handle_character_avatar_generate_requested(CharacterAvatarGenerateRequested())

    # No "save the character first" refusal - unlike the per-state handler.
    assert calls == [1]
    assert (None, "avatar") in screen._expression_generate_inflight
    assert not notifications


async def test_avatar_generate_requested_empty_description_notifies_no_worker(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character
    _set_description(screen, "   ")

    calls: list = []
    monkeypatch.setattr(screen, "run_worker", lambda *a, **k: calls.append(1))
    notifications = _capture_notifications(app)

    screen._handle_character_avatar_generate_requested(CharacterAvatarGenerateRequested())

    assert calls == []
    assert notifications
    assert notifications[-1] == ("Add a description first.", "warning")


async def test_avatar_generate_requested_inflight_second_click_single_generation(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character
    _configure_backend(monkeypatch)
    _set_description(screen, "A cheerful adventurer.")

    calls: list = []

    def _fake_run_worker(coro, *a, **k):
        calls.append(1)
        coro.close()

    monkeypatch.setattr(screen, "run_worker", _fake_run_worker)
    notifications = _capture_notifications(app)

    screen._handle_character_avatar_generate_requested(CharacterAvatarGenerateRequested())
    screen._handle_character_avatar_generate_requested(CharacterAvatarGenerateRequested())

    assert calls == [1]
    assert (char_id, "avatar") in screen._expression_generate_inflight
    assert notifications
    assert notifications[-1][1] == "warning"
    assert "already generating" in notifications[-1][0].lower()


async def test_generate_requested_refused_while_generate_all_in_flight(
    personas_editor_with_saved_character, monkeypatch
):
    """task-563 AC5: a per-slot key is freed the instant that slot's own
    generation finishes inside the Generate-all sweep (``_generate_one_
    slot``'s ``finally``), while the sweep itself keeps running the
    remaining states. Without an explicit guard, a single-slot click for
    that just-finished (or not-yet-started) state during the same sweep
    would dispatch an independent, redundant regeneration. Closing this at
    the "all" key's full lifetime (held for the whole sweep) is simpler and
    stronger than trying to reason about per-slot timing windows."""
    app, screen, db, char_id = personas_editor_with_saved_character
    _configure_backend(monkeypatch)
    _set_description(screen, "A cheerful adventurer.")
    screen._expression_generate_inflight.add((char_id, "all"))

    calls: list = []
    monkeypatch.setattr(screen, "run_worker", lambda *a, **k: calls.append(1))
    notifications = _capture_notifications(app)

    screen._handle_character_expression_generate_requested(
        CharacterExpressionGenerateRequested("thinking")
    )

    assert calls == []
    assert notifications
    assert notifications[-1][1] == "warning"
    assert "generate all" in notifications[-1][0].lower()


async def test_avatar_generate_requested_refused_while_generate_all_in_flight(
    personas_editor_with_saved_character, monkeypatch
):
    """task-563 AC5: same guard as the expression-slot case, for the avatar
    generate button - the avatar is itself one of the sweep's four states."""
    app, screen, db, char_id = personas_editor_with_saved_character
    _configure_backend(monkeypatch)
    _set_description(screen, "A cheerful adventurer.")
    screen._expression_generate_inflight.add((char_id, "all"))

    calls: list = []
    monkeypatch.setattr(screen, "run_worker", lambda *a, **k: calls.append(1))
    notifications = _capture_notifications(app)

    screen._handle_character_avatar_generate_requested(CharacterAvatarGenerateRequested())

    assert calls == []
    assert notifications
    assert notifications[-1][1] == "warning"
    assert "generate all" in notifications[-1][0].lower()


# ----- task-563 AC3: Generate-all overwrite confirmation --------------------
#
# The sweep's blast radius (avatar + 3 expression states) exceeds the
# per-slot regenerate-by-click contract, so it must confirm first whenever
# it would actually overwrite something. Uses the same ConfirmationDialog /
# push_screen_wait idiom as _confirm_delete / _confirm_dictionary_revert
# (monkeypatch app.push_screen_wait, mirroring this file's own style-pick
# tests above).


async def test_generate_all_worker_no_confirmation_when_nothing_would_be_overwritten(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character
    _configure_backend(monkeypatch)
    _set_description(screen, "A cheerful adventurer.")

    push_calls: list = []

    async def _fake_push_screen_wait(dialog):
        push_calls.append(dialog)
        return True

    monkeypatch.setattr(app, "push_screen_wait", _fake_push_screen_wait)
    monkeypatch.setattr(
        personas_screen_module,
        "run_generation",
        lambda request: SimpleNamespace(content=b"png-bytes", content_type="image/png"),
    )
    apply_mock = AsyncMock()
    monkeypatch.setattr(screen, "_apply_expression_upload", apply_mock)
    _capture_avatar_render_worker(screen)
    screen._expression_generate_inflight.add((char_id, "all"))

    await screen._generate_all_expression_images_worker(char_id)

    assert push_calls == []  # nothing to overwrite -> no dialog
    assert apply_mock.await_count == 3  # the sweep still ran normally


async def test_generate_all_worker_confirms_before_overwriting_staged_avatar(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character
    _configure_backend(monkeypatch)
    editor = _set_description(screen, "A cheerful adventurer.")
    editor.set_avatar_image(b"already-staged-avatar")

    push_calls: list = []

    async def _fake_push_screen_wait(dialog):
        push_calls.append(dialog)
        return True  # user confirms

    monkeypatch.setattr(app, "push_screen_wait", _fake_push_screen_wait)
    monkeypatch.setattr(
        personas_screen_module,
        "run_generation",
        lambda request: SimpleNamespace(content=b"png-bytes", content_type="image/png"),
    )
    apply_mock = AsyncMock()
    monkeypatch.setattr(screen, "_apply_expression_upload", apply_mock)
    _capture_avatar_render_worker(screen)
    screen._expression_generate_inflight.add((char_id, "all"))

    await screen._generate_all_expression_images_worker(char_id)

    assert len(push_calls) == 1
    assert editor.current_avatar_bytes() == b"png-bytes"  # sweep proceeded
    assert apply_mock.await_count == 3


async def test_generate_all_worker_confirms_before_overwriting_existing_expression_image(
    personas_editor_with_saved_character, monkeypatch
):
    """An existing expression-state image (not the avatar) also counts as
    "would overwrite"."""
    app, screen, db, char_id = personas_editor_with_saved_character
    _configure_backend(monkeypatch)
    _set_description(screen, "A cheerful adventurer.")
    db.set_character_expression_image(char_id, "thinking", b"already-there")

    push_calls: list = []

    async def _fake_push_screen_wait(dialog):
        push_calls.append(dialog)
        return True

    monkeypatch.setattr(app, "push_screen_wait", _fake_push_screen_wait)
    monkeypatch.setattr(
        personas_screen_module,
        "run_generation",
        lambda request: SimpleNamespace(content=b"png-bytes", content_type="image/png"),
    )
    _capture_avatar_render_worker(screen)
    screen._expression_generate_inflight.add((char_id, "all"))

    await screen._generate_all_expression_images_worker(char_id)

    assert len(push_calls) == 1
    assert db.get_character_expression_image(char_id, "thinking") == b"png-bytes"


async def test_generate_all_worker_declining_confirmation_aborts_with_no_writes(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character
    _configure_backend(monkeypatch)
    editor = _set_description(screen, "A cheerful adventurer.")
    editor.set_avatar_image(b"already-staged-avatar")

    async def _fake_push_screen_wait(dialog):
        return False  # Cancel

    monkeypatch.setattr(app, "push_screen_wait", _fake_push_screen_wait)
    run_calls: list = []

    def _run_generation(request):
        run_calls.append(request)
        return SimpleNamespace(content=b"png-bytes", content_type="image/png")

    monkeypatch.setattr(personas_screen_module, "run_generation", _run_generation)
    notifications = _capture_notifications(app)
    screen._expression_generate_inflight.add((char_id, "all"))

    await screen._generate_all_expression_images_worker(char_id)

    assert run_calls == []  # nothing generated
    assert editor.current_avatar_bytes() == b"already-staged-avatar"  # untouched
    assert (char_id, "all") not in screen._expression_generate_inflight
    assert not any("generated" in msg.lower() for msg, _sev in notifications)


# ----- Mandatory (c): Generate-all, one failing state -----------------------


async def test_generate_all_worker_one_failing_state_reports_partial_summary(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character
    _configure_backend(monkeypatch)
    editor = _set_description(screen, "A cheerful adventurer.")

    def _run_generation(request):
        if "mid-speech" in request.prompt:  # the "speaking" state modifier
            raise RuntimeError("backend exploded")
        return SimpleNamespace(content=b"png-bytes", content_type="image/png")

    monkeypatch.setattr(personas_screen_module, "run_generation", _run_generation)
    apply_mock = AsyncMock()
    monkeypatch.setattr(screen, "_apply_expression_upload", apply_mock)
    _capture_avatar_render_worker(screen)
    notifications = _capture_notifications(app)
    screen._expression_generate_inflight.add((char_id, "all"))

    await screen._generate_all_expression_images_worker(char_id)

    # avatar + thinking + error succeed (3 writes); speaking fails.
    assert editor.current_avatar_bytes() == b"png-bytes"
    assert apply_mock.await_count == 2
    applied_states = {call.args[1] for call in apply_mock.await_args_list}
    assert applied_states == {"thinking", "error"}
    assert (char_id, "all") not in screen._expression_generate_inflight
    for state in ("avatar", "thinking", "speaking", "error"):
        assert (char_id, state) not in screen._expression_generate_inflight
    assert notifications
    assert notifications[-1] == ("3/4 generated.", "information")


async def test_generate_all_worker_counts_only_genuinely_persisted_slots(
    personas_editor_with_saved_character, monkeypatch
):
    """task-563 AC4: exercises the REAL ``_apply_expression_upload`` (not a
    mock) so a DB-write failure for one state - generation itself
    succeeded, only the persist step failed - must not be counted as a
    success in the final summary, even though it already got its own
    per-slot error notify."""
    app, screen, db, char_id = personas_editor_with_saved_character
    _configure_backend(monkeypatch)
    _set_description(screen, "A cheerful adventurer.")

    monkeypatch.setattr(
        personas_screen_module,
        "run_generation",
        lambda request: SimpleNamespace(content=b"png-bytes", content_type="image/png"),
    )
    original_write = db.set_character_expression_image

    def _flaky_write(character_id, state, image, mime=None):
        if state == "speaking":
            raise RuntimeError("disk full")
        return original_write(character_id, state, image, mime)

    monkeypatch.setattr(db, "set_character_expression_image", _flaky_write)
    _capture_avatar_render_worker(screen)
    notifications = _capture_notifications(app)
    screen._expression_generate_inflight.add((char_id, "all"))

    await screen._generate_all_expression_images_worker(char_id)

    # avatar (staged) + thinking + error persisted; speaking's DB write
    # failed, so it must not count despite generation itself succeeding.
    assert db.get_character_expression_image(char_id, "speaking") is None
    assert db.get_character_expression_image(char_id, "thinking") is not None
    assert db.get_character_expression_image(char_id, "error") is not None
    severities = [severity for _msg, severity in notifications]
    assert "error" in severities  # the per-slot failure notify still fires
    assert notifications[-1] == ("3/4 generated.", "information")


async def test_generate_all_worker_skips_a_slot_already_in_flight(
    personas_editor_with_saved_character, monkeypatch
):
    """A slot claimed by an independent single-slot generate click (started
    just before Generate-all) is skipped, not raced."""
    app, screen, db, char_id = personas_editor_with_saved_character
    _configure_backend(monkeypatch)
    _set_description(screen, "A cheerful adventurer.")

    fake_result = SimpleNamespace(content=b"png-bytes", content_type="image/png")
    monkeypatch.setattr(
        personas_screen_module, "run_generation", lambda request: fake_result
    )
    apply_mock = AsyncMock()
    monkeypatch.setattr(screen, "_apply_expression_upload", apply_mock)
    _capture_avatar_render_worker(screen)
    notifications = _capture_notifications(app)
    screen._expression_generate_inflight.add((char_id, "thinking"))  # already busy

    await screen._generate_all_expression_images_worker(char_id)

    applied_states = {call.args[1] for call in apply_mock.await_args_list}
    assert applied_states == {"speaking", "error"}  # thinking was skipped
    assert notifications[-1] == ("3/4 generated.", "information")


async def test_generate_all_worker_stops_sweep_when_character_switches_mid_generation(
    personas_editor_with_saved_character, monkeypatch
):
    """Final review F1 (confirmed, High): the user cancels the editor mid-
    sweep and opens a DIFFERENT character (B) in the same editor widget
    while slot 1 (avatar) is still generating. Slot 1 itself is caught by
    its own pre-existing per-call stale-token guard in ``_generate_one_
    slot`` (the switch happens mid-call, before ``run_generation``
    returns). But WITHOUT this fix, slots 2-4 each capture a brand-new
    session token at the START of their own ``_generate_one_slot`` call -
    by then already matching character B's now-current session, since no
    further switch happens during THEIR calls - so they'd sail past their
    own stale-write guard and write real bytes into character A's DB rows,
    composed from B's live (unsaved) text. The fix re-verifies session
    identity against the character id THIS sweep was launched for at the
    top of every loop iteration and stops the sweep the moment they
    diverge, so the sweep must not even attempt slots 2-4."""
    app, screen, db, char_id = personas_editor_with_saved_character
    _configure_backend(monkeypatch)
    editor = _set_description(screen, "CHARACTER A: a cheerful adventurer.")
    other_id = db.add_character_card({"name": "Grim"})

    prompts = []

    def _run_generation(request):
        prompts.append(request.prompt)
        if len(prompts) == 1:
            # The user cancels the editor after slot 1 (avatar) dispatched
            # its generation call, but before that call returns, then
            # opens a DIFFERENT character (B) in the same editor widget -
            # mirroring what _handle_edit_requested does for a fresh
            # EditCharacterRequested.
            screen._finish_cancel_edit()
            screen._character_editor_generation += 1
            screen._edit_mode = "edit"
            screen.state.select_entity(
                entity_kind="character",
                entity_id=str(other_id),
                entity_name="Grim",
            )
            screen._show_center("#ccp-character-editor-view")
            editor.load_character(
                {
                    "id": other_id,
                    "name": "Grim",
                    "description": "CHARACTER B: a grim necromancer.",
                    "version": 1,
                }
            )
        return SimpleNamespace(content=b"png-bytes", content_type="image/png")

    monkeypatch.setattr(personas_screen_module, "run_generation", _run_generation)
    apply_mock = AsyncMock()
    monkeypatch.setattr(screen, "_apply_expression_upload", apply_mock)
    _capture_avatar_render_worker(screen)
    notifications = _capture_notifications(app)
    screen._expression_generate_inflight.add((char_id, "all"))

    await screen._generate_all_expression_images_worker(char_id)

    apply_mock.assert_not_awaited()  # zero writes, into A or B, after the switch
    assert len(prompts) == 1  # only the avatar attempt; nothing dispatched after
    assert (char_id, "all") not in screen._expression_generate_inflight
    for state in ("avatar", "thinking", "speaking", "error"):
        assert (char_id, state) not in screen._expression_generate_inflight
    # The summary reflects only genuinely-completed slots (zero here - the
    # switch lands before slot 1 even finishes).
    assert notifications[-1] == ("0/4 generated.", "information")


# ----- Handler: Generate-all gates (saved character required) --------------


async def test_generate_all_requested_unsaved_character_notifies_save_first(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character
    editor = screen.query_one(PersonasCharacterEditorWidget)
    editor.new_character()

    calls: list = []
    monkeypatch.setattr(screen, "run_worker", lambda *a, **k: calls.append(1))
    notifications = _capture_notifications(app)

    screen._handle_character_expression_generate_all_requested(
        CharacterExpressionGenerateAllRequested()
    )

    assert calls == []
    assert notifications
    assert notifications[-1] == ("Save the character to add expressions.", "warning")


async def test_generate_all_requested_saved_character_dispatches_worker(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character
    _configure_backend(monkeypatch)
    _set_description(screen, "A cheerful adventurer.")

    calls: list = []

    def _fake_run_worker(coro, *a, **k):
        calls.append(1)
        coro.close()

    monkeypatch.setattr(screen, "run_worker", _fake_run_worker)

    screen._handle_character_expression_generate_all_requested(
        CharacterExpressionGenerateAllRequested()
    )

    assert calls == [1]
    assert (char_id, "all") in screen._expression_generate_inflight


async def test_generate_all_requested_inflight_second_click_blocked(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character
    _configure_backend(monkeypatch)
    _set_description(screen, "A cheerful adventurer.")

    calls: list = []

    def _fake_run_worker(coro, *a, **k):
        calls.append(1)
        coro.close()

    monkeypatch.setattr(screen, "run_worker", _fake_run_worker)
    notifications = _capture_notifications(app)

    screen._handle_character_expression_generate_all_requested(
        CharacterExpressionGenerateAllRequested()
    )
    screen._handle_character_expression_generate_all_requested(
        CharacterExpressionGenerateAllRequested()
    )

    assert calls == [1]
    assert notifications
    assert notifications[-1][1] == "warning"
    assert "already generating" in notifications[-1][0].lower()


# ----- Mandatory (d) + (e): style picker ------------------------------------


async def test_style_pick_worker_stores_template_and_updates_readout(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character
    editor = screen.query_one(PersonasCharacterEditorWidget)

    async def _fake_push_screen_wait(modal):
        return {"id": "portrait_realistic", "name": "Realistic Portrait"}

    monkeypatch.setattr(app, "push_screen_wait", _fake_push_screen_wait)

    await screen._expression_style_pick_dialog_worker()

    assert screen._expression_generate_style is not None
    assert screen._expression_generate_style.id == "portrait_realistic"
    readout = editor.query_one("#personas-char-editor-style-readout", Static)
    assert str(readout.renderable) == "Style: Realistic Portrait"
    assert screen._io_dialog_active is False


async def test_style_pick_worker_cancel_keeps_previous_style_and_readout(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character
    editor = screen.query_one(PersonasCharacterEditorWidget)
    previous = get_template("portrait_realistic")
    screen._expression_generate_style = previous
    screen._update_expression_style_readout()

    async def _fake_push_screen_wait(modal):
        return None  # Escape/Cancel

    monkeypatch.setattr(app, "push_screen_wait", _fake_push_screen_wait)

    await screen._expression_style_pick_dialog_worker()

    assert screen._expression_generate_style is previous  # unchanged
    readout = editor.query_one("#personas-char-editor-style-readout", Static)
    assert str(readout.renderable) == "Style: Realistic Portrait"


async def test_style_pick_requested_dispatches_dialog_worker(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character
    calls: list = []

    def _fake_run_worker(coro, *a, **k):
        calls.append(1)
        coro.close()

    monkeypatch.setattr(screen, "run_worker", _fake_run_worker)
    assert screen._io_dialog_active is False

    screen._handle_expression_style_pick_requested(
        CharacterExpressionStylePickRequested()
    )

    assert calls == [1]
    assert screen._io_dialog_active is True


async def test_style_pick_requested_blocked_while_io_dialog_active(
    personas_editor_with_saved_character, monkeypatch
):
    app, screen, db, char_id = personas_editor_with_saved_character
    calls: list = []
    monkeypatch.setattr(screen, "run_worker", lambda *a, **k: calls.append(1))
    screen._io_dialog_active = True

    screen._handle_expression_style_pick_requested(
        CharacterExpressionStylePickRequested()
    )

    assert calls == []


async def test_style_pick_affects_subsequent_generation_build_request_kwargs(
    personas_editor_with_saved_character, monkeypatch
):
    """Mandatory (d): the picked style's params flow into the next
    generation's ``build_request`` call."""
    app, screen, db, char_id = personas_editor_with_saved_character
    _configure_backend(monkeypatch)
    _set_description(screen, "A cheerful adventurer.")
    template = get_template("portrait_realistic")
    screen._expression_generate_style = template

    captured_kwargs: dict = {}

    def _fake_build_request(**kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(backend=kwargs.get("backend"))

    monkeypatch.setattr(personas_screen_module, "build_request", _fake_build_request)
    fake_result = SimpleNamespace(content=b"png-bytes", content_type="image/png")
    monkeypatch.setattr(
        personas_screen_module, "run_generation", lambda request: fake_result
    )
    apply_mock = AsyncMock()
    monkeypatch.setattr(screen, "_apply_expression_upload", apply_mock)
    screen._expression_generate_inflight.add((char_id, "thinking"))

    await screen._generate_expression_image_worker(char_id, "thinking")

    assert captured_kwargs["width"] == template.default_params.get("width")
    assert captured_kwargs["height"] == template.default_params.get("height")
    assert captured_kwargs["steps"] == template.default_params.get("steps")
    assert captured_kwargs["cfg_scale"] == template.default_params.get("cfg_scale")
    apply_mock.assert_awaited_once_with(char_id, "thinking", b"png-bytes", "image/png")


# ----- Widget: style-pick button + readout present --------------------------


async def test_style_pick_button_and_readout_present_in_compose():
    app = _CaptureApp()
    async with app.run_test():
        editor = app.query_one(PersonasCharacterEditorWidget)
        assert editor.query_one("#personas-char-editor-style-pick", Button) is not None
        readout = editor.query_one("#personas-char-editor-style-readout", Static)
        assert str(readout.renderable) == "Style: Custom"


async def test_style_pick_button_posts_style_pick_requested():
    app = _CaptureApp()
    async with app.run_test() as pilot:
        app.query_one("#personas-char-editor-style-pick", Button).press()
        await pilot.pause()
        assert len(app.style_pick) == 1
        assert isinstance(app.style_pick[0], CharacterExpressionStylePickRequested)


# ===== Fix round 1: the picked style must not bleed across editor sessions =
#
# Verified finding: _expression_generate_style was set once (style pick) and
# never reset, so it lived for the whole PersonasScreen instance's lifetime
# rather than "this editor session" (the spec's own words). Opening a
# different character (or starting a new create session, or cancelling)
# after picking a style on character A would silently carry that style over
# to character B. The fix resets it (+ the readout) at the same 3 sites
# where ``_character_editor_generation`` bumps to mark a genuinely NEW/ended
# session - NOT the other bump sites in this file (avatar Remove, expression
# -set apply, expression upload/clear, save-in-place), which bump the same
# counter merely to invalidate a stale in-flight render within the SAME
# session and must leave a picked style untouched.


async def test_style_pick_reset_on_new_create_session(
    personas_editor_with_saved_character, monkeypatch
):
    """Mandatory fix-round-1 pin: pick a style, cross a real session
    boundary (_begin_create_character - the cheapest boundary to drive
    directly in this harness), then assert the style is cleared, the
    readout reverts, and the NEXT generation's build_request carries no
    template params."""
    app, screen, db, char_id = personas_editor_with_saved_character
    template = get_template("portrait_realistic")
    screen._expression_generate_style = template
    screen._update_expression_style_readout()
    editor = screen.query_one(PersonasCharacterEditorWidget)
    readout = editor.query_one("#personas-char-editor-style-readout", Static)
    assert str(readout.renderable) == "Style: Realistic Portrait"  # sanity

    await screen._begin_create_character()

    assert screen._expression_generate_style is None
    assert str(readout.renderable) == "Style: Custom"

    # The next generation (now an unsaved/id-less avatar, post-create) must
    # carry no template params - proves the reset actually affects
    # composition, not just the bookkeeping fields.
    _configure_backend(monkeypatch)
    editor._area("description").text = "A cheerful adventurer."
    captured_kwargs: dict = {}

    def _fake_build_request(**kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(backend=kwargs.get("backend"))

    monkeypatch.setattr(personas_screen_module, "build_request", _fake_build_request)
    fake_result = SimpleNamespace(content=b"avatar-bytes", content_type="image/png")
    monkeypatch.setattr(
        personas_screen_module, "run_generation", lambda request: fake_result
    )
    _capture_avatar_render_worker(screen)
    screen._expression_generate_inflight.add((None, "avatar"))

    await screen._generate_expression_image_worker(None, "avatar")

    assert captured_kwargs["negative_prompt"] is None
    assert captured_kwargs["width"] is None
    assert captured_kwargs["height"] is None
    assert captured_kwargs["steps"] is None
    assert captured_kwargs["cfg_scale"] is None


async def test_style_pick_reset_on_edit_requested_reopen(
    personas_editor_with_saved_character
):
    """Opening a character for edit (EditCharacterRequested - the
    "open-different-character" boundary the review flagged by name) also
    clears a previously-picked style."""
    from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
        EditCharacterRequested,
    )

    app, screen, db, char_id = personas_editor_with_saved_character
    template = get_template("portrait_realistic")
    screen._expression_generate_style = template
    screen._update_expression_style_readout()

    screen._handle_edit_requested(EditCharacterRequested(str(char_id)))

    assert screen._expression_generate_style is None
    editor = screen.query_one(PersonasCharacterEditorWidget)
    readout = editor.query_one("#personas-char-editor-style-readout", Static)
    assert str(readout.renderable) == "Style: Custom"


async def test_style_pick_reset_on_cancel_edit(
    personas_editor_with_saved_character
):
    """Cancelling the editor (_finish_cancel_edit) also clears a
    previously-picked style - the session it belonged to just ended."""
    app, screen, db, char_id = personas_editor_with_saved_character
    template = get_template("portrait_realistic")
    screen._expression_generate_style = template
    screen._update_expression_style_readout()

    screen._finish_cancel_edit()

    assert screen._expression_generate_style is None
    editor = screen.query_one(PersonasCharacterEditorWidget)
    readout = editor.query_one("#personas-char-editor-style-readout", Static)
    assert str(readout.renderable) == "Style: Custom"
