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

import asyncio
import time
from dataclasses import replace
from io import BytesIO
from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image
from textual.app import ComposeResult
from textual.widgets import Button, Static

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_personas_expression_slots import (
    expr_db,  # noqa: F401 -- fixture dependency, not referenced by name
    personas_editor_with_bound_pack,  # noqa: F401 -- used as a fixture
    personas_editor_with_saved_character,  # noqa: F401 -- used as a fixture
)
from tldw_chatbook.Character_Chat.visual_identity import (
    SAMIRA_EXPRESSION_KEYS,
    SAMIRA_REACTION_LABELS,
    VisualIdentityPublicationError,
    VisualIdentityPublicationResult,
)
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
    CharacterSaveRequested,
)

pytestmark = pytest.mark.asyncio

EXPRESSION_STATES = ("thinking", "speaking", "error")


def _valid_png(color=(20, 40, 60)) -> bytes:
    stream = BytesIO()
    Image.new("RGB", (8, 8), color).save(stream, format="PNG")
    return stream.getvalue()


def _fail_on_cancelled_task_reshield(monkeypatch) -> None:
    """Bound drain regressions so a cancelled child cannot spin the test loop."""

    shield = asyncio.shield

    def bounded(awaitable):
        if isinstance(awaitable, asyncio.Task) and awaitable.cancelled():
            raise RuntimeError("cancelled task was re-shielded")
        return shield(awaitable)

    monkeypatch.setattr(asyncio, "shield", bounded)


class _CaptureApp(ConsolidatedCSSApp):
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


async def test_generate_all_worker_unreadable_expression_state_fails_closed_into_confirmation(
    personas_editor_with_saved_character, monkeypatch
):
    """A DB read failure while checking for existing images must fail CLOSED:
    when overwrite status is unknown, the confirmation dialog is forced
    rather than silently skipped (Qodo PR #865 — consent gate must not fail
    open on the exception path)."""
    app, screen, db, char_id = personas_editor_with_saved_character
    _configure_backend(monkeypatch)
    _set_description(screen, "A cheerful adventurer.")

    def _raise(*args, **kwargs):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(db, "get_character_expression_image", _raise)

    push_calls: list = []

    async def _fake_push_screen_wait(dialog):
        push_calls.append(dialog)
        return False  # Cancel — sweep must abort without writes

    monkeypatch.setattr(app, "push_screen_wait", _fake_push_screen_wait)
    run_calls: list = []
    monkeypatch.setattr(
        personas_screen_module,
        "run_generation",
        lambda request: run_calls.append(request),
    )
    screen._expression_generate_inflight.add((char_id, "all"))

    await screen._generate_all_expression_images_worker(char_id)

    assert len(push_calls) == 1  # unknown overwrite state ⇒ dialog forced
    assert run_calls == []


async def test_generate_all_worker_confirm_dialog_failure_notifies_user(
    personas_editor_with_saved_character, monkeypatch
):
    """When the overwrite-confirmation dialog itself fails to show, the
    aborted sweep must tell the user instead of silently doing nothing
    (Qodo PR #865 — the feature otherwise appears broken with only a log
    line as evidence)."""
    app, screen, db, char_id = personas_editor_with_saved_character
    _configure_backend(monkeypatch)
    editor = _set_description(screen, "A cheerful adventurer.")
    editor.set_avatar_image(b"already-staged-avatar")

    async def _broken_push_screen_wait(dialog):
        raise RuntimeError("screen stack unavailable")

    monkeypatch.setattr(app, "push_screen_wait", _broken_push_screen_wait)
    run_calls: list = []
    monkeypatch.setattr(
        personas_screen_module,
        "run_generation",
        lambda request: run_calls.append(request),
    )
    notifications = _capture_notifications(app)
    screen._expression_generate_inflight.add((char_id, "all"))

    await screen._generate_all_expression_images_worker(char_id)

    assert run_calls == []
    assert any(
        "confirmation" in msg.lower() and "cancel" in msg.lower()
        for msg, _sev in notifications
    ), f"expected a user-facing abort notification, got: {notifications}"


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


# ===== TASK-16319.3 Task 14: immutable Visual Identity authoring =====


async def test_visual_identity_clear_stages_without_changing_active_version(
    personas_editor_with_bound_pack,
):
    _app, screen, db, char_id, _preview_calls = personas_editor_with_bound_pack
    browser = screen.query_one(personas_screen_module.PersonasVisualIdentityPackWidget)
    before = personas_screen_module.VisualIdentityRepository(db).get_active_actor_pack(
        "character", char_id
    )
    asset = browser.selected_asset
    assert asset is not None

    assert await screen._stage_visual_identity_clear(asset)

    after = personas_screen_module.VisualIdentityRepository(db).get_active_actor_pack(
        "character", char_id
    )
    assert after["version"]["id"] == before["version"]["id"]
    candidate = screen._visual_identity_authoring.candidate
    assert asset.expression_key in candidate.cleared_expression_keys
    assert "1 staged change" in str(
        browser.query_one("#personas-visual-identity-dirty", Static).renderable
    )


async def test_visual_identity_generate_all_decline_makes_zero_provider_calls(
    personas_editor_with_bound_pack, monkeypatch
):
    app, screen, _db, _char_id, _preview_calls = personas_editor_with_bound_pack
    dialogs = []

    async def decline(dialog):
        dialogs.append(dialog)
        return False

    monkeypatch.setattr(app, "push_screen_wait", decline)
    calls = []
    monkeypatch.setattr(
        personas_screen_module, "run_generation", lambda request: calls.append(request)
    )

    await screen._generate_visual_identity_pack_all()

    assert len(dialogs) == 1
    assert "31 provider calls" in dialogs[0].message
    assert calls == []


async def test_visual_identity_generate_all_uses_three_threads_one_reference_and_event(
    personas_editor_with_bound_pack, monkeypatch
):
    app, screen, _db, char_id, _preview_calls = personas_editor_with_bound_pack
    _set_description(screen, "silver hair, amber eyes")
    monkeypatch.setattr(app, "push_screen_wait", AsyncMock(return_value=True))
    monkeypatch.setattr(
        personas_screen_module,
        "get_image_generation_config",
        lambda: SimpleNamespace(default_backend="fal"),
    )
    reference_bytes = _valid_png((1, 2, 3))
    monkeypatch.setattr(
        personas_screen_module,
        "resolve_visual_identity",
        lambda *_args, **_kwargs: SimpleNamespace(
            actor_kind="character",
            actor_id=str(char_id),
            pack_id=1,
            pack_version_id=1,
            asset_id=21,
            content_type="image/png",
            image_bytes=reference_bytes,
        ),
    )
    lock = Lock()
    active = 0
    peak = 0
    requests = []

    def generate(request):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            requests.append(request)
        time.sleep(0.01)
        with lock:
            active -= 1
        return SimpleNamespace(content=_valid_png(), content_type="image/png")

    monkeypatch.setattr(personas_screen_module, "run_generation", generate)

    assert await screen._generate_visual_identity_pack_all()

    assert len(requests) == len(SAMIRA_REACTION_LABELS) == 31
    assert peak == 3
    assert len({id(request.reference_image) for request in requests}) == 1
    assert requests[0].reference_image.content == reference_bytes
    assert len({id(request.cancel_event) for request in requests}) == 1
    assert isinstance(requests[0].cancel_event, Event)
    assert len(screen._visual_identity_authoring.candidate.replaced_expression_keys) == 31


async def test_visual_identity_generate_all_cancellation_discards_candidate_and_drains(
    personas_editor_with_bound_pack, monkeypatch
):
    app, screen, _db, char_id, _preview_calls = personas_editor_with_bound_pack
    _set_description(screen, "silver hair, amber eyes")
    monkeypatch.setattr(app, "push_screen_wait", AsyncMock(return_value=True))
    monkeypatch.setattr(
        personas_screen_module,
        "get_image_generation_config",
        lambda: SimpleNamespace(default_backend="fal"),
    )
    monkeypatch.setattr(
        personas_screen_module,
        "resolve_visual_identity",
        lambda *_args, **_kwargs: SimpleNamespace(
            actor_kind="character",
            actor_id=str(char_id),
            pack_id=1,
            pack_version_id=1,
            asset_id=21,
            content_type="image/png",
            image_bytes=_valid_png(),
        ),
    )
    started = Event()
    exited = Event()
    seen_events = []

    def generate(request):
        seen_events.append(request.cancel_event)
        if len(seen_events) >= 3:
            started.set()
        request.cancel_event.wait(2)
        exited.set()
        return SimpleNamespace(content=_valid_png(), content_type="image/png")

    monkeypatch.setattr(personas_screen_module, "run_generation", generate)
    task = asyncio.create_task(screen._generate_visual_identity_pack_all())
    assert await asyncio.to_thread(started.wait, 2)

    screen._request_visual_identity_generation_cancel()
    assert await task is False

    assert exited.is_set()
    assert seen_events and len({id(event) for event in seen_events}) == 1
    assert seen_events[0].is_set()
    assert screen._visual_identity_authoring is None


async def test_visual_identity_generate_all_one_failure_discards_without_publication(
    personas_editor_with_bound_pack, monkeypatch
):
    app, screen, db, char_id, _preview_calls = personas_editor_with_bound_pack
    _set_description(screen, "silver hair, amber eyes")
    monkeypatch.setattr(app, "push_screen_wait", AsyncMock(return_value=True))
    monkeypatch.setattr(
        personas_screen_module,
        "get_image_generation_config",
        lambda: SimpleNamespace(default_backend="fal"),
    )
    monkeypatch.setattr(
        personas_screen_module,
        "resolve_visual_identity",
        lambda *_args, **_kwargs: SimpleNamespace(
            actor_kind="character",
            actor_id=str(char_id),
            pack_id=1,
            pack_version_id=1,
            asset_id=21,
            content_type="image/png",
            image_bytes=_valid_png(),
        ),
    )
    before = personas_screen_module.VisualIdentityRepository(db).get_active_actor_pack(
        "character", char_id
    )

    def generate(request):
        if "Admiration expression" in request.prompt:
            raise RuntimeError("provider failed")
        return SimpleNamespace(content=_valid_png(), content_type="image/png")

    monkeypatch.setattr(personas_screen_module, "run_generation", generate)

    assert await screen._generate_visual_identity_pack_all() is False

    after = personas_screen_module.VisualIdentityRepository(db).get_active_actor_pack(
        "character", char_id
    )
    assert after["version"]["id"] == before["version"]["id"]
    assert screen._visual_identity_authoring is None


async def test_visual_identity_unsupported_reference_fails_before_candidate_or_call(
    personas_editor_with_bound_pack, monkeypatch
):
    app, screen, _db, _char_id, _preview_calls = personas_editor_with_bound_pack
    monkeypatch.setattr(app, "push_screen_wait", AsyncMock(return_value=True))
    monkeypatch.setattr(
        personas_screen_module,
        "get_image_generation_config",
        lambda: SimpleNamespace(default_backend="stable_diffusion_cpp"),
    )
    calls = []
    monkeypatch.setattr(
        personas_screen_module, "run_generation", lambda request: calls.append(request)
    )

    assert await screen._generate_visual_identity_pack_all() is False

    assert calls == []
    assert screen._visual_identity_authoring is None


async def test_visual_identity_save_publishes_once_then_invalidates_before_refresh(
    personas_editor_with_bound_pack, monkeypatch, tmp_path
):
    _app, screen, _db, char_id, _preview_calls = personas_editor_with_bound_pack
    browser = screen.query_one(personas_screen_module.PersonasVisualIdentityPackWidget)
    asset = browser.selected_asset
    assert asset is not None
    assert await screen._stage_visual_identity_replacement(
        asset, _valid_png(), source="upload"
    )
    order = []
    result = VisualIdentityPublicationResult(
        actor_kind="character",
        actor_id=str(char_id),
        old_pack_id=1,
        old_version_id=1,
        new_pack_id=2,
        new_version_id=2,
        version_directory=tmp_path,
    )

    def publish(*args, **kwargs):
        order.append("publish")
        return result

    async def invalidate(_result):
        order.append("invalidate")

    async def refresh(_snapshot):
        order.append("refresh")
        current = browser.pack
        assert current is not None
        browser.pack = replace(
            current,
            pack_id=2,
            pack_version_id=2,
            assets=tuple(
                replace(asset, asset_id=asset.asset_id + 100)
                for asset in current.assets
            ),
        )

    monkeypatch.setattr(personas_screen_module, "publish_visual_identity_candidate", publish)
    monkeypatch.setattr(screen, "_invalidate_visual_identity_publication", invalidate)
    monkeypatch.setattr(screen, "_configure_character_visual_identity", refresh)

    assert await screen._save_visual_identity_pack(browser.pack)
    assert not await screen._save_visual_identity_pack(browser.pack)

    assert order == ["publish", "invalidate", "refresh"]
    assert browser.pack is not None
    assert browser.pack.pack_version_id == 2
    assert len(browser.pack.assets) == 31
    assert all(asset.asset_id > 0 for asset in browser.pack.assets)


async def test_visual_identity_concurrent_save_attempt_publishes_exactly_once(
    personas_editor_with_bound_pack, monkeypatch, tmp_path
):
    _app, screen, _db, char_id, _preview_calls = personas_editor_with_bound_pack
    browser = screen.query_one(personas_screen_module.PersonasVisualIdentityPackWidget)
    asset = browser.selected_asset
    assert asset is not None
    assert await screen._stage_visual_identity_replacement(
        asset, _valid_png(), source="upload"
    )
    entered = Event()
    release = Event()
    calls = []

    def publish(*args, **kwargs):
        calls.append(1)
        entered.set()
        release.wait(2)
        return VisualIdentityPublicationResult(
            actor_kind="character",
            actor_id=str(char_id),
            old_pack_id=1,
            old_version_id=1,
            new_pack_id=2,
            new_version_id=2,
            version_directory=tmp_path,
        )

    monkeypatch.setattr(
        personas_screen_module, "publish_visual_identity_candidate", publish
    )
    monkeypatch.setattr(
        screen, "_invalidate_visual_identity_publication", AsyncMock()
    )
    monkeypatch.setattr(screen, "_configure_character_visual_identity", AsyncMock())

    first = asyncio.create_task(screen._save_visual_identity_pack(browser.pack))
    assert await asyncio.to_thread(entered.wait, 2)
    assert not await screen._save_visual_identity_pack(browser.pack)
    release.set()
    assert await first
    assert calls == [1]


async def test_drain_async_returns_cancelled_child_without_reshielding(monkeypatch):
    _fail_on_cancelled_task_reshield(monkeypatch)

    async def cancel_child():
        raise asyncio.CancelledError("child-cancelled")

    outcome = await asyncio.wait_for(
        personas_screen_module._drain_async(
            cancel_child(), task_name="personas-cancelled-child-probe"
        ),
        1,
    )

    assert outcome.error is None
    assert isinstance(outcome.cancellation, asyncio.CancelledError)


async def test_cancelled_reaction_save_drains_commit_before_releasing_admission(
    personas_editor_with_bound_pack, monkeypatch, tmp_path
):
    _app, screen, _db, char_id, _preview_calls = personas_editor_with_bound_pack
    browser = screen.query_one(personas_screen_module.PersonasVisualIdentityPackWidget)
    asset = browser.selected_asset
    assert asset is not None
    assert await screen._stage_visual_identity_replacement(
        asset, _valid_png(), source="upload"
    )
    entered = Event()
    release = Event()
    order = []

    def publish(*_args, **_kwargs):
        entered.set()
        assert release.wait(2)
        order.append("publish")
        return VisualIdentityPublicationResult(
            actor_kind="character",
            actor_id=str(char_id),
            old_pack_id=1,
            old_version_id=1,
            new_pack_id=2,
            new_version_id=2,
            version_directory=tmp_path,
        )

    async def invalidate(_result):
        order.append("invalidate")

    async def refresh(_snapshot):
        order.append("refresh")

    monkeypatch.setattr(
        personas_screen_module, "publish_visual_identity_candidate", publish
    )
    monkeypatch.setattr(screen, "_invalidate_visual_identity_publication", invalidate)
    monkeypatch.setattr(screen, "_configure_character_visual_identity", refresh)
    save = asyncio.create_task(
        screen._save_visual_identity_pack(browser.pack), name="cancelled-reaction-save"
    )
    assert await asyncio.to_thread(entered.wait, 2)

    save.cancel()
    await asyncio.sleep(0)
    assert screen._visual_identity_publication_inflight
    assert screen._visual_identity_operation_task is save
    assert not await screen._stage_visual_identity_clear(asset)
    save.cancel()
    await asyncio.sleep(0)
    assert screen._visual_identity_publication_inflight
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await save
    assert order == ["publish", "invalidate", "refresh"]
    assert not screen._visual_identity_publication_inflight
    assert screen._visual_identity_operation_task is None
    assert screen._visual_identity_authoring is None
    assert browser._staged == {}


async def test_cancelled_reaction_save_drains_failure_and_cleans_exact_orphan(
    personas_editor_with_bound_pack, monkeypatch, tmp_path
):
    app, screen, db, _char_id, _preview_calls = personas_editor_with_bound_pack
    browser = screen.query_one(personas_screen_module.PersonasVisualIdentityPackWidget)
    asset = browser.selected_asset
    assert asset is not None
    assert await screen._stage_visual_identity_replacement(
        asset, _valid_png(), source="upload"
    )
    before = personas_screen_module.VisualIdentityRepository(db).get_active_actor_pack(
        "character", screen._visual_identity_authoring.snapshot.character_id
    )
    token = "packs/profile-0123456789abcdef0123456789abcdef/versions/" + "c" * 32
    user_root = tmp_path / "user-root"
    entered = Event()
    release = Event()
    cleanup_calls = []
    notifications = _capture_notifications(app)

    def publish(*_args, **_kwargs):
        entered.set()
        assert release.wait(2)
        raise VisualIdentityPublicationError(
            "visual_identity_database_failed", cleanup_candidate_relpath=token
        )

    def cleanup(cleanup_db, cleanup_token, *, user_data_dir):
        cleanup_calls.append((cleanup_db, cleanup_token, user_data_dir))
        return True

    monkeypatch.setattr(personas_screen_module, "get_user_data_dir", lambda: user_root)
    monkeypatch.setattr(
        personas_screen_module, "publish_visual_identity_candidate", publish
    )
    monkeypatch.setattr(
        personas_screen_module,
        "cleanup_visual_identity_publication_candidate",
        cleanup,
    )
    save = asyncio.create_task(
        screen._save_visual_identity_pack(browser.pack), name="cancelled-reaction-error"
    )
    assert await asyncio.to_thread(entered.wait, 2)

    save.cancel()
    await asyncio.sleep(0)
    assert screen._visual_identity_publication_inflight
    assert not await screen._stage_visual_identity_clear(asset)
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await save
    assert cleanup_calls == [(db, token, user_root)]
    assert token not in " ".join(message for message, _severity in notifications)
    after = personas_screen_module.VisualIdentityRepository(db).get_active_actor_pack(
        "character", screen._visual_identity_authoring.snapshot.character_id
    )
    assert after["version"]["id"] == before["version"]["id"]
    assert not screen._visual_identity_publication_inflight
    assert screen._visual_identity_operation_task is None


async def test_cancelled_reaction_save_drains_post_commit_reconciliation(
    personas_editor_with_bound_pack, monkeypatch, tmp_path
):
    _app, screen, _db, char_id, _preview_calls = personas_editor_with_bound_pack
    browser = screen.query_one(personas_screen_module.PersonasVisualIdentityPackWidget)
    asset = browser.selected_asset
    assert asset is not None
    assert await screen._stage_visual_identity_replacement(
        asset, _valid_png(), source="upload"
    )
    invalidation_entered = asyncio.Event()
    release_invalidation = asyncio.Event()
    order = []

    def publish(*_args, **_kwargs):
        order.append("publish")
        return VisualIdentityPublicationResult(
            actor_kind="character",
            actor_id=str(char_id),
            old_pack_id=1,
            old_version_id=1,
            new_pack_id=2,
            new_version_id=2,
            version_directory=tmp_path,
        )

    async def invalidate(_result):
        order.append("invalidate-start")
        invalidation_entered.set()
        await release_invalidation.wait()
        order.append("invalidate-done")

    async def refresh(_snapshot):
        order.append("refresh")

    monkeypatch.setattr(
        personas_screen_module, "publish_visual_identity_candidate", publish
    )
    monkeypatch.setattr(screen, "_invalidate_visual_identity_publication", invalidate)
    monkeypatch.setattr(screen, "_configure_character_visual_identity", refresh)
    save = asyncio.create_task(
        screen._save_visual_identity_pack(browser.pack),
        name="cancelled-reaction-reconcile",
    )
    await asyncio.wait_for(invalidation_entered.wait(), 2)

    save.cancel()
    await asyncio.sleep(0)
    assert screen._visual_identity_publication_inflight
    assert screen._visual_identity_operation_task is save
    assert not await screen._stage_visual_identity_clear(asset)
    save.cancel()
    await asyncio.sleep(0)
    assert screen._visual_identity_publication_inflight
    release_invalidation.set()

    with pytest.raises(asyncio.CancelledError):
        await save
    assert order == ["publish", "invalidate-start", "invalidate-done", "refresh"]
    assert not screen._visual_identity_publication_inflight
    assert screen._visual_identity_operation_task is None
    assert screen._visual_identity_authoring is None
    assert browser._staged == {}


async def test_cancelled_reaction_reconciliation_releases_operation_guards(
    personas_editor_with_bound_pack, monkeypatch, tmp_path
):
    app, screen, _db, char_id, _preview_calls = personas_editor_with_bound_pack
    browser = screen.query_one(personas_screen_module.PersonasVisualIdentityPackWidget)
    asset = browser.selected_asset
    assert asset is not None
    assert await screen._stage_visual_identity_replacement(
        asset, _valid_png(), source="upload"
    )
    notifications = _capture_notifications(app)
    _fail_on_cancelled_task_reshield(monkeypatch)

    monkeypatch.setattr(
        personas_screen_module,
        "publish_visual_identity_candidate",
        lambda *_args, **_kwargs: VisualIdentityPublicationResult(
            actor_kind="character",
            actor_id=str(char_id),
            old_pack_id=1,
            old_version_id=1,
            new_pack_id=2,
            new_version_id=2,
            version_directory=tmp_path,
        ),
    )

    async def cancel_invalidation(_result):
        raise asyncio.CancelledError("invalidation-cancelled")

    refresh = AsyncMock()
    monkeypatch.setattr(
        screen, "_invalidate_visual_identity_publication", cancel_invalidation
    )
    monkeypatch.setattr(screen, "_configure_character_visual_identity", refresh)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(screen._save_visual_identity_pack(browser.pack), 2)

    refresh.assert_not_awaited()
    assert not screen._visual_identity_publication_inflight
    assert screen._visual_identity_operation_task is None
    assert screen._visual_identity_authoring is None
    assert browser._staged == {}
    assert notifications == []


async def test_visual_identity_first_clear_admits_one_candidate_and_cancel_reaches_it(
    personas_editor_with_bound_pack, monkeypatch
):
    app, screen, _db, char_id, _preview_calls = personas_editor_with_bound_pack
    browser = screen.query_one(personas_screen_module.PersonasVisualIdentityPackWidget)
    asset = browser.selected_asset
    assert asset is not None
    entered = Event()
    release = Event()
    create_calls = []
    original_create = personas_screen_module.create_visual_identity_candidate

    def blocked_create(*args, **kwargs):
        create_calls.append(1)
        entered.set()
        assert release.wait(2)
        return original_create(*args, **kwargs)

    monkeypatch.setattr(
        personas_screen_module, "create_visual_identity_candidate", blocked_create
    )
    monkeypatch.setattr(app, "push_screen_wait", AsyncMock(return_value=True))
    monkeypatch.setattr(
        personas_screen_module,
        "get_image_generation_config",
        lambda: SimpleNamespace(default_backend="fal"),
    )
    monkeypatch.setattr(
        personas_screen_module,
        "resolve_visual_identity",
        lambda *_args, **_kwargs: SimpleNamespace(
            actor_kind="character",
            actor_id=str(char_id),
            pack_id=1,
            pack_version_id=1,
            asset_id=21,
            content_type="image/png",
            image_bytes=_valid_png(),
        ),
    )
    provider_calls = []
    monkeypatch.setattr(
        personas_screen_module,
        "run_generation",
        lambda request: provider_calls.append(request)
        or SimpleNamespace(content=_valid_png(), content_type="image/png"),
    )

    clear = asyncio.create_task(screen._stage_visual_identity_clear(asset))
    assert await asyncio.to_thread(entered.wait, 2)
    assert browser.query_one(
        "#personas-visual-identity-cancel", Button
    ).display
    assert (
        str(browser.query_one("#personas-visual-identity-dirty", Static).renderable)
        == "Preparing reactions…"
    )
    generate = asyncio.create_task(screen._generate_visual_identity_assets((asset,)))
    generate_all = asyncio.create_task(screen._generate_visual_identity_pack_all())
    await asyncio.sleep(0)
    browser.query_one("#personas-visual-identity-cancel", Button).press()
    await asyncio.sleep(0)
    release.set()

    assert await asyncio.gather(clear, generate, generate_all) == [False, False, False]
    assert create_calls == [1]
    assert provider_calls == []
    assert screen._visual_identity_authoring is None


async def test_visual_identity_duplicate_generation_shares_global_three_call_ceiling(
    personas_editor_with_bound_pack, monkeypatch
):
    app, screen, _db, char_id, _preview_calls = personas_editor_with_bound_pack
    _set_description(screen, "silver hair, amber eyes")
    browser = screen.query_one(personas_screen_module.PersonasVisualIdentityPackWidget)
    asset = browser.selected_asset
    assert asset is not None
    monkeypatch.setattr(app, "push_screen_wait", AsyncMock(return_value=True))
    monkeypatch.setattr(
        personas_screen_module,
        "get_image_generation_config",
        lambda: SimpleNamespace(default_backend="fal"),
    )
    monkeypatch.setattr(
        personas_screen_module,
        "resolve_visual_identity",
        lambda *_args, **_kwargs: SimpleNamespace(
            actor_kind="character",
            actor_id=str(char_id),
            pack_id=1,
            pack_version_id=1,
            asset_id=21,
            content_type="image/png",
            image_bytes=_valid_png(),
        ),
    )
    lock = Lock()
    active = 0
    peak = 0
    three_started = Event()

    def generate(request):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active >= 3:
                three_started.set()
        request.cancel_event.wait(2)
        with lock:
            active -= 1
        return SimpleNamespace(content=_valid_png(), content_type="image/png")

    monkeypatch.setattr(personas_screen_module, "run_generation", generate)
    generate_all = asyncio.create_task(screen._generate_visual_identity_pack_all())
    assert await asyncio.to_thread(three_started.wait, 2)
    duplicate = asyncio.create_task(
        screen._generate_visual_identity_assets((asset,))
    )
    await asyncio.sleep(0.1)

    screen._request_visual_identity_generation_cancel()
    assert await generate_all is False
    assert await duplicate is False
    assert peak == 3


async def test_visual_identity_mode_guard_decline_preserves_staged_candidate(
    personas_editor_with_bound_pack, monkeypatch
):
    app, screen, _db, _char_id, _preview_calls = personas_editor_with_bound_pack
    browser = screen.query_one(personas_screen_module.PersonasVisualIdentityPackWidget)
    asset = browser.selected_asset
    assert asset is not None
    assert await screen._stage_visual_identity_clear(asset)
    state = screen._visual_identity_authoring
    assert state is not None
    monkeypatch.setattr(app, "push_screen_wait", AsyncMock(return_value=False))

    await screen._run_guarded(lambda: screen._apply_mode("personas"))
    await app.workers.wait_for_complete()

    assert screen.state.active_mode == "characters"
    assert screen._visual_identity_authoring is state
    assert not state.cancel_event.is_set()
    assert asset.expression_key in state.candidate.cleared_expression_keys


async def test_visual_identity_actor_guard_discards_only_after_approval(
    personas_editor_with_bound_pack, monkeypatch
):
    app, screen, _db, _char_id, _preview_calls = personas_editor_with_bound_pack
    browser = screen.query_one(personas_screen_module.PersonasVisualIdentityPackWidget)
    asset = browser.selected_asset
    assert asset is not None
    assert await screen._stage_visual_identity_clear(asset)
    state = screen._visual_identity_authoring
    transition = AsyncMock()
    monkeypatch.setattr(screen, "_select_character", transition)
    monkeypatch.setattr(app, "push_screen_wait", AsyncMock(side_effect=[False, True]))

    await screen._run_guarded(lambda: screen._select_character("999", "Other"))
    await app.workers.wait_for_complete()
    assert transition.await_count == 0
    assert screen._visual_identity_authoring is state
    assert state is not None and not state.cancel_event.is_set()

    await screen._run_guarded(lambda: screen._select_character("999", "Other"))
    await app.workers.wait_for_complete()
    transition.assert_awaited_once_with("999", "Other")
    assert state.cancel_event.is_set()
    assert screen._visual_identity_authoring is None


async def test_visual_identity_mode_transition_signals_then_drains_adapter(
    personas_editor_with_bound_pack, monkeypatch
):
    app, screen, _db, char_id, _preview_calls = personas_editor_with_bound_pack
    _set_description(screen, "silver hair, amber eyes")
    browser = screen.query_one(personas_screen_module.PersonasVisualIdentityPackWidget)
    asset = browser.selected_asset
    assert asset is not None
    monkeypatch.setattr(app, "push_screen_wait", AsyncMock(return_value=True))
    monkeypatch.setattr(
        personas_screen_module,
        "get_image_generation_config",
        lambda: SimpleNamespace(default_backend="fal"),
    )
    monkeypatch.setattr(
        personas_screen_module,
        "resolve_visual_identity",
        lambda *_args, **_kwargs: SimpleNamespace(
            actor_kind="character",
            actor_id=str(char_id),
            pack_id=1,
            pack_version_id=1,
            asset_id=21,
            content_type="image/png",
            image_bytes=_valid_png(),
        ),
    )
    entered = Event()
    saw_cancel = Event()
    allow_exit = Event()

    def generate(request):
        entered.set()
        if request.cancel_event.wait(2):
            saw_cancel.set()
        assert allow_exit.wait(2)
        return SimpleNamespace(content=_valid_png(), content_type="image/png")

    monkeypatch.setattr(personas_screen_module, "run_generation", generate)
    generation = asyncio.create_task(
        screen._generate_visual_identity_assets((asset,))
    )
    assert await asyncio.to_thread(entered.wait, 2)
    try:
        await screen._run_guarded(lambda: screen._apply_mode("personas"))
        assert await asyncio.to_thread(saw_cancel.wait, 1)
        assert screen.state.active_mode == "characters"
        allow_exit.set()
        assert await generation is False
        await app.workers.wait_for_complete()
        assert screen.state.active_mode == "personas"
        assert screen._visual_identity_authoring is None
    finally:
        allow_exit.set()


@pytest.mark.parametrize("cleanup_result", [True, False])
async def test_visual_identity_save_consumes_orphan_token_without_exposing_path(
    personas_editor_with_bound_pack, monkeypatch, tmp_path, cleanup_result
):
    app, screen, db, _char_id, _preview_calls = personas_editor_with_bound_pack
    browser = screen.query_one(personas_screen_module.PersonasVisualIdentityPackWidget)
    asset = browser.selected_asset
    assert asset is not None
    assert await screen._stage_visual_identity_replacement(
        asset, _valid_png(), source="upload"
    )
    before = personas_screen_module.VisualIdentityRepository(db).get_active_actor_pack(
        "character", screen._visual_identity_authoring.snapshot.character_id
    )
    token = "packs/profile-0123456789abcdef0123456789abcdef/versions/" + "a" * 32
    user_root = tmp_path / "user-root"
    calls = []
    notifications = _capture_notifications(app)

    def publish(*_args, **_kwargs):
        raise VisualIdentityPublicationError(
            "visual_identity_database_failed", cleanup_candidate_relpath=token
        )

    def cleanup(cleanup_db, cleanup_token, *, user_data_dir):
        calls.append((cleanup_db, cleanup_token, user_data_dir))
        if not cleanup_result:
            raise VisualIdentityPublicationError("visual_identity_cleanup_referenced")
        return True

    monkeypatch.setattr(personas_screen_module, "get_user_data_dir", lambda: user_root)
    monkeypatch.setattr(personas_screen_module, "publish_visual_identity_candidate", publish)
    monkeypatch.setattr(
        personas_screen_module,
        "cleanup_visual_identity_publication_candidate",
        cleanup,
        raising=False,
    )

    assert not await screen._save_visual_identity_pack(browser.pack)

    assert calls == [(db, token, user_root)]
    assert token not in " ".join(message for message, _severity in notifications)
    assert any(
        "visual_identity_database_failed" in message
        for message, _severity in notifications
    )
    after = personas_screen_module.VisualIdentityRepository(db).get_active_actor_pack(
        "character", screen._visual_identity_authoring.snapshot.character_id
    )
    assert after["version"]["id"] == before["version"]["id"]


async def test_visual_identity_orphan_cleanup_never_notifies_reloaded_editor(
    personas_editor_with_bound_pack, monkeypatch, tmp_path
):
    app, screen, _db, char_id, _preview_calls = personas_editor_with_bound_pack
    browser = screen.query_one(personas_screen_module.PersonasVisualIdentityPackWidget)
    asset = browser.selected_asset
    assert asset is not None
    assert await screen._stage_visual_identity_replacement(
        asset, _valid_png(), source="upload"
    )
    token = "packs/profile-0123456789abcdef0123456789abcdef/versions/" + "b" * 32
    cleanup_started = Event()
    cleanup_release = Event()
    notifications = _capture_notifications(app)

    def publish(*_args, **_kwargs):
        raise VisualIdentityPublicationError(
            "visual_identity_database_failed", cleanup_candidate_relpath=token
        )

    def cleanup(_db, cleanup_token, *, user_data_dir):
        assert cleanup_token == token
        assert user_data_dir == tmp_path / "user-root"
        cleanup_started.set()
        assert cleanup_release.wait(2)
        return True

    monkeypatch.setattr(
        personas_screen_module, "get_user_data_dir", lambda: tmp_path / "user-root"
    )
    monkeypatch.setattr(personas_screen_module, "publish_visual_identity_candidate", publish)
    monkeypatch.setattr(
        personas_screen_module,
        "cleanup_visual_identity_publication_candidate",
        cleanup,
    )
    save = asyncio.create_task(screen._save_visual_identity_pack(browser.pack))
    assert await asyncio.to_thread(cleanup_started.wait, 2)
    editor = screen.query_one(PersonasCharacterEditorWidget)
    editor.load_character(
        {"id": char_id, "name": "Reloaded", "description": "new session"},
        visual_identity_pending=True,
    )
    cleanup_release.set()

    assert await save is False
    assert notifications == []


async def test_character_save_refuses_staged_reaction_changes_without_mutation(
    personas_editor_with_bound_pack, monkeypatch
):
    app, screen, _db, _char_id, _preview_calls = personas_editor_with_bound_pack
    editor = _set_description(screen, "unsaved character edit")
    browser = screen.query_one(personas_screen_module.PersonasVisualIdentityPackWidget)
    asset = browser.selected_asset
    assert asset is not None
    assert await screen._stage_visual_identity_clear(asset)
    authoring = screen._visual_identity_authoring
    assert authoring is not None
    generation = screen._character_editor_generation
    save_calls = []
    monkeypatch.setattr(
        screen,
        "_save_character_worker",
        lambda *args: save_calls.append(args),
    )
    notifications = _capture_notifications(app)

    screen._handle_save_requested(CharacterSaveRequested(editor.get_character_data()))

    assert save_calls == []
    assert notifications == [
        ("Save or Cancel reaction changes before saving the character.", "warning")
    ]
    assert editor._area("description").text == "unsaved character edit"
    assert screen._character_editor_generation == generation
    assert screen._visual_identity_authoring is authoring
    assert not authoring.cancel_event.is_set()
    assert asset.expression_key in authoring.candidate.cleared_expression_keys
    assert not screen._character_save_inflight


async def test_character_save_refuses_inflight_reaction_generation_without_cancelling(
    personas_editor_with_bound_pack, monkeypatch
):
    app, screen, _db, char_id, _preview_calls = personas_editor_with_bound_pack
    editor = _set_description(screen, "unsaved character edit")
    browser = screen.query_one(personas_screen_module.PersonasVisualIdentityPackWidget)
    asset = browser.selected_asset
    assert asset is not None
    monkeypatch.setattr(
        personas_screen_module,
        "get_image_generation_config",
        lambda: SimpleNamespace(default_backend="fal"),
    )
    monkeypatch.setattr(
        personas_screen_module,
        "resolve_visual_identity",
        lambda *_args, **_kwargs: SimpleNamespace(
            actor_kind="character",
            actor_id=str(char_id),
            pack_id=1,
            pack_version_id=1,
            asset_id=21,
            content_type="image/png",
            image_bytes=_valid_png(),
        ),
    )
    entered = Event()
    release = Event()

    def generate(request):
        entered.set()
        assert release.wait(2)
        return SimpleNamespace(content=_valid_png(), content_type="image/png")

    monkeypatch.setattr(personas_screen_module, "run_generation", generate)
    generation_task = asyncio.create_task(
        screen._generate_visual_identity_assets((asset,))
    )
    assert await asyncio.to_thread(entered.wait, 2)
    authoring = screen._visual_identity_authoring
    assert authoring is not None
    editor_generation = screen._character_editor_generation
    save_calls = []
    monkeypatch.setattr(
        screen,
        "_save_character_worker",
        lambda *args: save_calls.append(args),
    )
    notifications = _capture_notifications(app)
    try:
        screen._handle_save_requested(
            CharacterSaveRequested(editor.get_character_data())
        )

        assert save_calls == []
        assert notifications == [
            ("Save or Cancel reaction changes before saving the character.", "warning")
        ]
        assert editor._area("description").text == "unsaved character edit"
        assert screen._character_editor_generation == editor_generation
        assert screen._visual_identity_authoring is authoring
        assert screen._visual_identity_operation_task is generation_task
        assert not authoring.cancel_event.is_set()
        assert not screen._character_save_inflight
    finally:
        screen._request_visual_identity_generation_cancel()
        release.set()
        assert await generation_task is False


async def test_character_save_dispatches_normally_without_reaction_authoring(
    personas_editor_with_bound_pack, monkeypatch
):
    _app, screen, _db, _char_id, _preview_calls = personas_editor_with_bound_pack
    editor = _set_description(screen, "ordinary character edit")
    generation = screen._character_editor_generation
    save_calls = []
    monkeypatch.setattr(
        screen,
        "_save_character_worker",
        lambda *args: save_calls.append(args),
    )

    screen._handle_save_requested(CharacterSaveRequested(editor.get_character_data()))

    assert len(save_calls) == 1
    assert save_calls[0][0]["description"] == "ordinary character edit"
    assert save_calls[0][1:] == (screen.state.selected_entity_id, screen._edit_mode)
    assert screen._character_editor_generation == generation
    assert screen._character_save_inflight


async def test_visual_identity_clear_refuses_character_save_inflight_without_state(
    personas_editor_with_bound_pack, monkeypatch
):
    app, screen, _db, _char_id, _preview_calls = personas_editor_with_bound_pack
    browser = screen.query_one(personas_screen_module.PersonasVisualIdentityPackWidget)
    asset = browser.selected_asset
    assert asset is not None
    candidate_calls = []
    original_create = personas_screen_module.create_visual_identity_candidate

    def create_candidate(*args, **kwargs):
        candidate_calls.append((args, kwargs))
        return original_create(*args, **kwargs)

    monkeypatch.setattr(
        personas_screen_module,
        "create_visual_identity_candidate",
        create_candidate,
    )
    notifications = _capture_notifications(app)
    screen._character_save_inflight = True

    assert not await screen._stage_visual_identity_clear(asset)

    assert notifications == [
        ("Wait for Character Save to finish before editing reactions.", "warning")
    ]
    assert candidate_calls == []
    assert screen._visual_identity_authoring is None
    assert screen._visual_identity_operation_task is None
    assert screen._visual_identity_operation_event is None
    assert browser._staged == {}


async def test_visual_identity_generation_refuses_character_save_before_work(
    personas_editor_with_bound_pack, monkeypatch
):
    app, screen, _db, _char_id, _preview_calls = personas_editor_with_bound_pack
    browser = screen.query_one(personas_screen_module.PersonasVisualIdentityPackWidget)
    asset = browser.selected_asset
    assert asset is not None
    admitted = AsyncMock(return_value=True)
    provider_calls = []
    dialogs = AsyncMock(return_value=True)
    monkeypatch.setattr(screen, "_generate_visual_identity_assets_admitted", admitted)
    monkeypatch.setattr(
        personas_screen_module,
        "run_generation",
        lambda request: provider_calls.append(request),
    )
    monkeypatch.setattr(app, "push_screen_wait", dialogs)
    notifications = _capture_notifications(app)
    screen._character_save_inflight = True

    assert not await screen._generate_visual_identity_assets((asset,))
    assert not await screen._generate_visual_identity_pack_all()

    assert notifications == [
        ("Wait for Character Save to finish before editing reactions.", "warning"),
        ("Wait for Character Save to finish before editing reactions.", "warning"),
    ]
    assert admitted.await_count == 0
    assert dialogs.await_count == 0
    assert provider_calls == []
    assert screen._visual_identity_authoring is None
    assert screen._visual_identity_operation_task is None
    assert screen._visual_identity_operation_event is None
    assert browser._staged == {}


async def test_visual_identity_operation_admits_when_character_save_idle(
    personas_editor_with_bound_pack,
):
    _app, screen, _db, _char_id, _preview_calls = personas_editor_with_bound_pack
    snapshot = screen._visual_identity_author_snapshot()
    assert snapshot is not None

    admission = screen._begin_visual_identity_operation(snapshot)

    assert admission is not None
    task, event = admission
    assert task is asyncio.current_task()
    assert not event.is_set()
    screen._finish_visual_identity_operation(task)


async def test_visual_identity_operation_admits_after_character_save_completion(
    personas_editor_with_bound_pack,
):
    app, screen, _db, char_id, _preview_calls = personas_editor_with_bound_pack
    screen._character_save_inflight = True

    await screen._after_character_save(str(char_id), "Packed")
    await app.workers.wait_for_complete()
    snapshot = screen._visual_identity_author_snapshot()

    assert not screen._character_save_inflight
    assert snapshot is not None
    admission = screen._begin_visual_identity_operation(snapshot)
    assert admission is not None
    task, _event = admission
    screen._finish_visual_identity_operation(task)


async def test_idle_pack_cancel_discards_candidate_and_widget_staging(
    personas_editor_with_bound_pack,
):
    app, screen, _db, _char_id, _preview_calls = personas_editor_with_bound_pack
    browser = screen.query_one(personas_screen_module.PersonasVisualIdentityPackWidget)
    original_pack = browser.pack
    asset = browser.selected_asset
    assert asset is not None
    assert await screen._stage_visual_identity_clear(asset)
    candidate = screen._visual_identity_authoring.candidate

    browser.query_one("#personas-visual-identity-cancel", Button).press()
    await asyncio.sleep(0.05)
    await app.workers.wait_for_complete()

    assert candidate._cancelled
    assert screen._visual_identity_authoring is None
    assert browser.pack == original_pack
    assert browser._staged == {}
    assert (
        str(browser.query_one("#personas-visual-identity-dirty", Static).renderable)
        == "No staged changes"
    )


@pytest.mark.parametrize("newer_authority", ("binding", "session"))
async def test_cancel_never_restores_pack_over_newer_editor_authority(
    personas_editor_with_bound_pack, newer_authority
):
    _app, screen, _db, _char_id, _preview_calls = personas_editor_with_bound_pack
    browser = screen.query_one(personas_screen_module.PersonasVisualIdentityPackWidget)
    asset = browser.selected_asset
    assert asset is not None
    assert await screen._stage_visual_identity_clear(asset)
    candidate = screen._visual_identity_authoring.candidate
    current = browser.pack
    assert current is not None
    newer_pack = replace(current, title="Externally refreshed")
    if newer_authority == "binding":
        newer_pack = replace(
            newer_pack,
            binding_id=current.binding_id + 1,
            pack_id=current.pack_id + 1,
            pack_version_id=current.pack_version_id + 1,
        )
    else:
        editor = screen.query_one(PersonasCharacterEditorWidget)
        editor._visual_identity_session_token += 1
    browser.pack = newer_pack

    screen._discard_visual_identity_authoring()

    assert candidate._cancelled
    assert screen._visual_identity_authoring is None
    assert browser.pack == newer_pack


async def test_candidate_authority_mismatch_refuses_before_staging_or_provider(
    personas_editor_with_bound_pack, monkeypatch
):
    _app, screen, _db, char_id, _preview_calls = personas_editor_with_bound_pack
    _set_description(screen, "silver hair, amber eyes")
    browser = screen.query_one(personas_screen_module.PersonasVisualIdentityPackWidget)
    asset = browser.selected_asset
    assert asset is not None
    original_create = personas_screen_module.create_visual_identity_candidate
    candidates = []

    def mismatched_candidate(*args, **kwargs):
        candidate = original_create(*args, **kwargs)
        candidate.old_binding_id += 1
        candidates.append(candidate)
        return candidate

    monkeypatch.setattr(
        personas_screen_module, "create_visual_identity_candidate", mismatched_candidate
    )
    monkeypatch.setattr(
        personas_screen_module,
        "get_image_generation_config",
        lambda: SimpleNamespace(default_backend="fal"),
    )
    monkeypatch.setattr(
        personas_screen_module,
        "resolve_visual_identity",
        lambda *_args, **_kwargs: SimpleNamespace(
            actor_kind="character",
            actor_id=str(char_id),
            pack_id=1,
            pack_version_id=1,
            asset_id=21,
            content_type="image/png",
            image_bytes=_valid_png(),
        ),
    )
    provider_calls = []
    monkeypatch.setattr(
        personas_screen_module,
        "run_generation",
        lambda request: provider_calls.append(request),
    )

    assert not await screen._generate_visual_identity_assets((asset,))

    assert len(candidates) == 1
    assert candidates[0]._cancelled
    assert provider_calls == []
    assert screen._visual_identity_authoring is None
    assert browser._staged == {}


async def test_publication_invalidation_isolated_and_busy_state_always_restored(
    personas_editor_with_bound_pack, monkeypatch, tmp_path
):
    app, screen, _db, char_id, _preview_calls = personas_editor_with_bound_pack
    browser = screen.query_one(personas_screen_module.PersonasVisualIdentityPackWidget)
    asset = browser.selected_asset
    assert asset is not None
    assert await screen._stage_visual_identity_replacement(
        asset, _valid_png(), source="upload"
    )
    result = VisualIdentityPublicationResult(
        actor_kind="character",
        actor_id=str(char_id),
        old_pack_id=1,
        old_version_id=1,
        new_pack_id=2,
        new_version_id=2,
        version_directory=tmp_path,
    )
    invalidated = []

    async def broken_invalidator(*_args):
        invalidated.append("broken")
        raise RuntimeError("/private/secret/cache")

    async def healthy_invalidator(*_args):
        invalidated.append("healthy")

    fake_screens = [
        SimpleNamespace(
            _session=SimpleNamespace(
                invalidate_visual_identity_actor=broken_invalidator
            )
        ),
        SimpleNamespace(
            _session=SimpleNamespace(
                invalidate_visual_identity_actor=healthy_invalidator
            )
        ),
    ]
    monkeypatch.setattr(
        personas_screen_module,
        "publish_visual_identity_candidate",
        lambda *_args, **_kwargs: result,
    )
    reload_metadata = AsyncMock()
    monkeypatch.setattr(
        screen, "_configure_character_visual_identity", reload_metadata
    )

    async def invalidate(result):
        host = SimpleNamespace(app=SimpleNamespace(screen_stack=fake_screens))
        await personas_screen_module.PersonasScreen._invalidate_visual_identity_publication(
            host, result
        )

    monkeypatch.setattr(screen, "_invalidate_visual_identity_publication", invalidate)
    saved = await screen._save_visual_identity_pack(browser.pack)

    assert saved
    assert invalidated == ["broken", "healthy"]
    reload_metadata.assert_awaited_once()
    assert screen._visual_identity_authoring is None
    assert browser._staged == {}
    assert (
        str(browser.query_one("#personas-visual-identity-dirty", Static).renderable)
        == "No staged changes"
    )


async def test_character_save_presentation_failure_releases_all_admission_guards(
    personas_editor_with_bound_pack, monkeypatch
):
    app, screen, _db, char_id, _preview_calls = personas_editor_with_bound_pack
    notifications = _capture_notifications(app)
    monkeypatch.setattr(
        personas_screen_module.ccp_character_handler,
        "update_character",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        screen,
        "_after_character_save",
        AsyncMock(side_effect=RuntimeError("/private/secret/refresh")),
    )
    screen._character_save_inflight = True

    await personas_screen_module.PersonasScreen._save_character_worker.__wrapped__(
        screen, {"name": "Packed"}, str(char_id), "edit"
    )

    assert not screen._character_save_inflight
    assert notifications == [
        ("Character saved, but the editor could not refresh.", "warning")
    ]
    snapshot = screen._visual_identity_author_snapshot()
    assert snapshot is not None
    admission = screen._begin_visual_identity_operation(snapshot)
    assert admission is not None
    task, _event = admission
    screen._finish_visual_identity_operation(task)


@pytest.mark.parametrize("stale_session", (False, True))
async def test_cancelled_character_save_drains_commit_before_reconcile_and_release(
    personas_editor_with_bound_pack, monkeypatch, stale_session
):
    _app, screen, _db, char_id, _preview_calls = personas_editor_with_bound_pack
    entered = Event()
    release = Event()

    def persist(*_args, **_kwargs):
        entered.set()
        assert release.wait(2)
        return True

    reconcile = AsyncMock()
    monkeypatch.setattr(
        personas_screen_module.ccp_character_handler, "update_character", persist
    )
    monkeypatch.setattr(screen, "_after_character_save", reconcile)
    screen._character_save_inflight = True
    save = asyncio.create_task(
        personas_screen_module.PersonasScreen._save_character_worker.__wrapped__(
            screen, {"name": "Packed"}, str(char_id), "edit"
        ),
        name="cancelled-character-save",
    )
    assert await asyncio.to_thread(entered.wait, 2)
    if stale_session:
        editor = screen.query_one(PersonasCharacterEditorWidget)
        editor._visual_identity_session_token += 1

    save.cancel()
    await asyncio.sleep(0)
    assert screen._character_save_inflight
    snapshot = screen._visual_identity_author_snapshot()
    assert snapshot is not None
    assert screen._begin_visual_identity_operation(snapshot) is None
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await save
    assert not screen._character_save_inflight
    if stale_session:
        reconcile.assert_not_awaited()
    else:
        reconcile.assert_awaited_once_with(str(char_id), "Packed")
    admission = screen._begin_visual_identity_operation(snapshot)
    assert admission is not None
    task, _event = admission
    screen._finish_visual_identity_operation(task)


async def test_cancelled_character_save_drains_late_error_and_releases_guard(
    personas_editor_with_bound_pack, monkeypatch
):
    app, screen, _db, char_id, _preview_calls = personas_editor_with_bound_pack
    entered = Event()
    release = Event()

    def persist(*_args, **_kwargs):
        entered.set()
        assert release.wait(2)
        raise RuntimeError("/private/secret/character.db")

    monkeypatch.setattr(
        personas_screen_module.ccp_character_handler, "update_character", persist
    )
    notifications = _capture_notifications(app)
    screen._character_save_inflight = True
    save = asyncio.create_task(
        personas_screen_module.PersonasScreen._save_character_worker.__wrapped__(
            screen, {"name": "Packed"}, str(char_id), "edit"
        ),
        name="cancelled-character-error",
    )
    assert await asyncio.to_thread(entered.wait, 2)

    save.cancel()
    await asyncio.sleep(0)
    assert screen._character_save_inflight
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await save
    assert not screen._character_save_inflight
    assert notifications == [("Character save failed.", "error")]


async def test_cancelled_character_save_drains_post_commit_reconciliation(
    personas_editor_with_bound_pack, monkeypatch
):
    _app, screen, _db, char_id, _preview_calls = personas_editor_with_bound_pack
    reconciliation_entered = asyncio.Event()
    release_reconciliation = asyncio.Event()
    order = []

    def persist(*_args, **_kwargs):
        order.append("persist")
        return True

    async def reconcile(saved_id, submitted_name):
        order.append(("reconcile-start", saved_id, submitted_name))
        reconciliation_entered.set()
        await release_reconciliation.wait()
        order.append("reconcile-done")

    monkeypatch.setattr(
        personas_screen_module.ccp_character_handler, "update_character", persist
    )
    monkeypatch.setattr(screen, "_after_character_save", reconcile)
    screen._character_save_inflight = True
    save = asyncio.create_task(
        personas_screen_module.PersonasScreen._save_character_worker.__wrapped__(
            screen, {"name": "Packed"}, str(char_id), "edit"
        ),
        name="cancelled-character-reconcile",
    )
    await asyncio.wait_for(reconciliation_entered.wait(), 2)

    save.cancel()
    await asyncio.sleep(0)
    assert screen._character_save_inflight
    snapshot = screen._visual_identity_author_snapshot()
    assert snapshot is not None
    assert screen._begin_visual_identity_operation(snapshot) is None
    save.cancel()
    await asyncio.sleep(0)
    assert screen._character_save_inflight
    release_reconciliation.set()

    with pytest.raises(asyncio.CancelledError):
        await save
    assert order == [
        "persist",
        ("reconcile-start", str(char_id), "Packed"),
        "reconcile-done",
    ]
    assert not screen._character_save_inflight
    admission = screen._begin_visual_identity_operation(snapshot)
    assert admission is not None
    task, _event = admission
    screen._finish_visual_identity_operation(task)


async def test_cancelled_character_reconciliation_releases_save_guard(
    personas_editor_with_bound_pack, monkeypatch
):
    app, screen, _db, char_id, _preview_calls = personas_editor_with_bound_pack
    notifications = _capture_notifications(app)
    _fail_on_cancelled_task_reshield(monkeypatch)
    monkeypatch.setattr(
        personas_screen_module.ccp_character_handler,
        "update_character",
        lambda *_args, **_kwargs: True,
    )

    async def cancel_reconciliation(_saved_id, _submitted_name):
        raise asyncio.CancelledError("character-reconcile-cancelled")

    monkeypatch.setattr(screen, "_after_character_save", cancel_reconciliation)
    screen._character_save_inflight = True

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(
            personas_screen_module.PersonasScreen._save_character_worker.__wrapped__(
                screen, {"name": "Packed"}, str(char_id), "edit"
            ),
            2,
        )

    assert not screen._character_save_inflight
    assert notifications == []


async def test_generate_all_restores_missing_canonical_asset_and_direction(
    personas_editor_with_bound_pack, monkeypatch
):
    app, screen, db, char_id, _preview_calls = personas_editor_with_bound_pack
    _set_description(screen, "silver hair, amber eyes")
    browser = screen.query_one(personas_screen_module.PersonasVisualIdentityPackWidget)
    pack = browser.pack
    assert pack is not None
    omitted_label = "remorse"
    omitted_key = SAMIRA_EXPRESSION_KEYS[omitted_label]
    omitted = next(
        asset for asset in pack.assets if asset.expression_key == omitted_key
    )
    with db.transaction() as cursor:
        cursor.execute(
            "DELETE FROM visual_identity_assets WHERE pack_version_id = ? AND expression_key = ?",
            (pack.pack_version_id, omitted_key),
        )
    browser.pack = replace(
        pack,
        assets=tuple(asset for asset in pack.assets if asset is not omitted),
    )
    authoritative_pack = browser.pack
    authoritative_graph = personas_screen_module.VisualIdentityRepository(
        db
    ).get_active_actor_pack("character", char_id)
    assert authoritative_graph is not None
    monkeypatch.setattr(app, "push_screen_wait", AsyncMock(return_value=True))
    monkeypatch.setattr(
        personas_screen_module,
        "get_image_generation_config",
        lambda: SimpleNamespace(default_backend="fal"),
    )
    monkeypatch.setattr(
        personas_screen_module,
        "resolve_visual_identity",
        lambda *_args, **_kwargs: SimpleNamespace(
            actor_kind="character",
            actor_id=str(char_id),
            pack_id=pack.pack_id,
            pack_version_id=pack.pack_version_id,
            asset_id=21,
            content_type="image/png",
            image_bytes=_valid_png(),
        ),
    )
    requests = []
    monkeypatch.setattr(
        personas_screen_module,
        "run_generation",
        lambda request: requests.append(request)
        or SimpleNamespace(content=_valid_png(), content_type="image/png"),
    )

    assert await screen._generate_visual_identity_pack_all()

    assert len(requests) == 31
    assert any(
        "lowered gaze and accountable regret" in request.prompt
        for request in requests
    )
    candidate = screen._visual_identity_authoring.candidate
    assert set(candidate.replaced_expression_keys) == set(
        SAMIRA_EXPRESSION_KEYS.values()
    )
    assert browser.pack is not None
    assert len(browser.pack.assets) == 31
    assert any(asset.asset_id < 0 for asset in browser.pack.assets)

    browser.query_one("#personas-visual-identity-cancel", Button).press()
    await asyncio.sleep(0.05)
    await app.workers.wait_for_complete()

    assert candidate._cancelled
    assert screen._visual_identity_authoring is None
    assert browser.pack == authoritative_pack
    assert len(browser.pack.assets) == 30
    assert all(asset.asset_id > 0 for asset in browser.pack.assets)
    live_graph = personas_screen_module.VisualIdentityRepository(
        db
    ).get_active_actor_pack("character", char_id)
    assert live_graph is not None
    assert live_graph["version"]["id"] == authoritative_graph["version"]["id"]
    assert tuple(asset["id"] for asset in live_graph["assets"]) == tuple(
        asset["id"] for asset in authoritative_graph["assets"]
    )


async def test_provider_failure_notifies_once_without_private_detail(
    personas_editor_with_bound_pack, monkeypatch
):
    app, screen, _db, char_id, _preview_calls = personas_editor_with_bound_pack
    _set_description(screen, "silver hair, amber eyes")
    browser = screen.query_one(personas_screen_module.PersonasVisualIdentityPackWidget)
    asset = browser.selected_asset
    assert asset is not None
    monkeypatch.setattr(
        personas_screen_module,
        "get_image_generation_config",
        lambda: SimpleNamespace(default_backend="fal"),
    )
    monkeypatch.setattr(
        personas_screen_module,
        "resolve_visual_identity",
        lambda *_args, **_kwargs: SimpleNamespace(
            actor_kind="character",
            actor_id=str(char_id),
            pack_id=1,
            pack_version_id=1,
            asset_id=21,
            content_type="image/png",
            image_bytes=_valid_png(),
        ),
    )
    monkeypatch.setattr(
        personas_screen_module,
        "run_generation",
        lambda _request: (_ for _ in ()).throw(
            RuntimeError("/private/secret/provider-key")
        ),
    )
    notifications = _capture_notifications(app)

    assert not await screen._generate_visual_identity_assets((asset,))

    assert notifications == [("Reaction generation failed. Try again.", "error")]
    assert screen._visual_identity_authoring is None
