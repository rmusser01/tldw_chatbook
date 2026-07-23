"""Mounted regressions for the collapsible Console composer."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from textual.app import App, ComposeResult
from textual.events import Paste
from textual.widgets import Button, Static

from Tests.UI.test_console_native_chat_flow import (
    _configure_native_ready_console,
)
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Widgets.Console import ConsoleComposerBar, ConsoleTranscript


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_STYLESHEET = _REPO_ROOT / "tldw_chatbook/css/components/_agentic_terminal.tcss"
_BUNDLED_STYLESHEET = _REPO_ROOT / "tldw_chatbook/css/tldw_cli_modular.tcss"


class _ComposerGeometryApp(App[None]):
    """Mount a composer with the production stylesheet for geometry assertions."""

    CSS_PATH = str(_BUNDLED_STYLESHEET)

    def __init__(self, *, collapsed: bool = False) -> None:
        super().__init__()
        self._initially_collapsed = collapsed

    def compose(self) -> ComposeResult:
        yield ConsoleComposerBar(
            id="console-native-composer",
            collapsed=self._initially_collapsed,
        )


def _ready_console_host() -> ConsoleHarness:
    app = _build_test_app()
    _configure_native_ready_console(app)
    return ConsoleHarness(app)


async def _mounted_console(host: ConsoleHarness, pilot):
    console = host.screen_stack[-1]
    await _wait_for_selector(console, pilot, "#console-native-composer")
    return console


@pytest.mark.asyncio
async def test_collapse_button_moves_focus_to_transcript_without_sending():
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep me")
        store = console._ensure_console_chat_store()
        message_count = len(store.messages_for_session(store.active_session_id))
        collapse = composer.query_one("#console-composer-collapse", Button)
        collapse.focus()
        await pilot.press("enter")
        await pilot.pause()

        assert composer.collapsed is True
        assert composer.draft_text() == "keep me"
        assert isinstance(host.focused, ConsoleTranscript)
        assert len(store.messages_for_session(store.active_session_id)) == message_count


@pytest.mark.asyncio
async def test_expand_button_and_one_escape_expand_and_focus_draft():
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        await pilot.click("#console-composer-collapse")
        await pilot.pause()
        assert composer.collapsed is True

        await pilot.press("escape")
        await pilot.pause()
        assert composer.collapsed is False
        assert host.focused is composer

        await pilot.click("#console-composer-collapse")
        await pilot.pause()
        assert composer.collapsed is True
        expand = composer.query_one("#console-composer-expand", Button)
        expand.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert composer.collapsed is False
        assert host.focused is composer


@pytest.mark.asyncio
async def test_collapsed_composer_hidden_input_and_paste_do_not_mutate_or_send():
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep me")
        store = console._ensure_console_chat_store()
        message_count = len(store.messages_for_session(store.active_session_id))
        console._set_console_composer_collapsed(True)
        await pilot.pause()

        for key in ("x", "backspace", "delete", "enter"):
            await pilot.press(key)
            await pilot.pause()
            assert composer.draft_text() == "keep me"
            assert (
                len(store.messages_for_session(store.active_session_id))
                == message_count
            )

        console.on_paste(Paste("pasted"))
        await pilot.pause()
        assert composer.draft_text() == "keep me"
        assert len(store.messages_for_session(store.active_session_id)) == message_count

        run_worker = Mock()
        console.run_worker = run_worker
        console.on_paste(Paste("/tmp/hidden-image.png"))
        await pilot.pause()
        run_worker.assert_not_called()
        assert composer.draft_text() == "keep me"
        assert len(store.messages_for_session(store.active_session_id)) == message_count


@pytest.mark.asyncio
async def test_collapsed_escape_action_is_dynamic_and_setup_controls_are_inert(
    monkeypatch,
):
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)

        assert console.check_action("expand_collapsed_console_composer", ()) is False
        with monkeypatch.context() as context:
            context.setattr(console, "_console_setup_modal_blocking", lambda: True)
            console._set_console_composer_collapsed(True)
            assert composer.collapsed is False

        console._set_console_composer_collapsed(True)
        await pilot.pause()
        assert console.check_action("expand_collapsed_console_composer", ()) is True

        with monkeypatch.context() as context:
            context.setattr(console, "_console_setup_modal_blocking", lambda: True)
            assert (
                console.check_action("expand_collapsed_console_composer", ()) is False
            )
            console._set_console_composer_collapsed(False)
            assert composer.collapsed is True


@pytest.mark.asyncio
async def test_expanded_escape_clears_transcript_selection_before_focus_fallback():
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        store = console._ensure_console_chat_store()
        message = store.append_message(
            store.active_session_id,
            role=ConsoleMessageRole.ASSISTANT,
            content="selected answer",
        )
        await console._sync_native_console_chat_ui()
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.select_message(message.id)
        transcript.focus()
        await pilot.pause()

        await pilot.press("escape")
        await pilot.pause()

        assert transcript.selected_message_id is None
        assert host.focused is transcript


@pytest.mark.asyncio
async def test_screen_collapse_disarms_unfurl_without_changing_pasted_content():
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        pasted_text = "pending unfurl paste " * 20
        composer.insert_pasted_text(pasted_text)
        await pilot.click("#console-command-visible-text")
        await pilot.pause()
        assert composer.has_pending_paste_confirmation() is True

        console._set_console_composer_collapsed(True)
        await pilot.pause()

        assert composer.has_pending_paste_confirmation() is False
        assert composer.has_paste_segments() is True
        assert composer.draft_text() == pasted_text


@pytest.mark.asyncio
async def test_collapsed_stop_routes_native_run_control_without_expanding():
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.sync_action_state(
            has_draft=False,
            run_active=True,
            can_save_chatbook=False,
        )
        console._set_console_composer_collapsed(True)
        await pilot.pause()
        stop_run = AsyncMock()
        console._stop_console_generation_from_visible_action = stop_run

        composer.query_one("#console-collapsed-stop-generation", Button).press()
        await pilot.pause()

        stop_run.assert_awaited_once_with()
        assert composer.collapsed is True


@pytest.mark.asyncio
async def test_collapsed_stop_is_inert_while_setup_blocks(monkeypatch):
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.sync_action_state(
            has_draft=False,
            run_active=True,
            can_save_chatbook=False,
        )
        console._set_console_composer_collapsed(True)
        await pilot.pause()
        stop_run = AsyncMock()
        console._stop_console_generation_from_visible_action = stop_run
        monkeypatch.setattr(console, "_console_setup_modal_blocking", lambda: True)

        stop = composer.query_one("#console-collapsed-stop-generation", Button)
        await console.handle_console_stop_generation(Button.Pressed(stop))

        stop_run.assert_not_awaited()
        assert composer.collapsed is True


@pytest.mark.asyncio
async def test_stale_composer_layout_revision_does_not_override_current_focus():
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        console._set_console_composer_collapsed(True)
        await pilot.pause()
        expand = console.query_one("#console-composer-expand", Button)
        expand.focus()
        await pilot.pause()

        console._finish_console_composer_layout_change(
            console._console_composer_layout_revision - 1,
            True,
            console._capture_console_transcript_reading_state(),
        )
        await pilot.pause()

        assert host.focused is expand


@pytest.mark.asyncio
async def test_screen_collapse_round_trip_restores_transcript_reading_state():
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        store = console._ensure_console_chat_store()
        selected_message_id = ""
        for index in range(24):
            message = store.append_message(
                store.active_session_id,
                role=(
                    ConsoleMessageRole.USER
                    if index % 2 == 0
                    else ConsoleMessageRole.ASSISTANT
                ),
                content="\n".join(f"message {index} line {line}" for line in range(3)),
            )
            selected_message_id = message.id
        await console._sync_native_console_chat_ui()
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        assert transcript.max_scroll_y > 0
        transcript.select_message(selected_message_id)
        await pilot.pause()
        transcript.anchor(False)
        transcript.scroll_to(y=min(4, transcript.max_scroll_y), animate=False)
        await pilot.pause()
        assert transcript.is_anchored is False
        expected_scroll_y = transcript.scroll_y

        console._set_console_composer_collapsed(True)
        await pilot.pause()
        assert transcript.is_anchored is False
        assert transcript.scroll_y == min(expected_scroll_y, transcript.max_scroll_y)
        assert transcript.selected_message_id == selected_message_id

        console._set_console_composer_collapsed(False)
        await pilot.pause()
        assert transcript.is_anchored is False
        assert transcript.scroll_y == min(expected_scroll_y, transcript.max_scroll_y)
        assert transcript.selected_message_id == selected_message_id


@pytest.mark.asyncio
async def test_replacement_composer_inherits_screen_state_and_active_session_draft():
    host = _ready_console_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        store = console._ensure_console_chat_store()
        store.set_session_draft(store.active_session_id, "persisted draft")
        console._set_console_composer_collapsed(True)
        await pilot.pause()

        console.refresh(recompose=True)
        for _ in range(40):
            await pilot.pause(0.05)
            replacement = console.query_one(
                "#console-native-composer", ConsoleComposerBar
            )
            if replacement is not composer:
                break
        else:
            raise AssertionError("Expected screen recompose to replace the composer")

        assert replacement.collapsed is True
        assert replacement.draft_text() == "persisted draft"


@pytest.mark.asyncio
async def test_console_composer_defaults_expanded_and_toggles_idempotently():
    host = _ready_console_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        expanded = composer.query_one("#console-composer-expanded")
        collapsed = composer.query_one("#console-composer-collapsed")

        assert composer.collapsed is False
        assert expanded.display is True
        assert collapsed.display is False
        assert composer.can_focus is True

        composer.set_collapsed(True)
        composer.set_collapsed(True)
        await pilot.pause()

        assert composer.collapsed is True
        assert expanded.display is False
        assert collapsed.display is True
        assert composer.can_focus is False

        composer.set_collapsed(False)
        composer.set_collapsed(False)
        await pilot.pause()

        assert composer.collapsed is False
        assert expanded.display is True
        assert collapsed.display is False
        assert composer.can_focus is True


@pytest.mark.asyncio
async def test_console_composer_geometry_is_bounded_then_exactly_one_row():
    app = _ComposerGeometryApp()

    async with app.run_test(size=(140, 42)) as pilot:
        composer = app.query_one("#console-native-composer", ConsoleComposerBar)

        assert 5 <= composer.region.height <= 8

        composer.set_collapsed(True)
        await pilot.pause()

        assert composer.region.height == 1

        composer.set_collapsed(False)
        await pilot.pause()

        assert 5 <= composer.region.height <= 8


@pytest.mark.asyncio
async def test_console_composer_compact_geometry_keeps_status_and_expand_visible():
    app = _ComposerGeometryApp(collapsed=True)

    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.pause()
        composer = app.query_one("#console-native-composer", ConsoleComposerBar)

        assert composer.region.height == 1
        assert composer.query_one("#console-composer-expand", Button).region.width > 0
        assert (
            composer.query_one(
                "#console-composer-collapsed-status", Static
            ).region.width
            > 0
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("draft", "attachment", "run_active", "expected"),
    [
        ("", None, False, "Composer hidden"),
        (" ", None, False, "Composer hidden · Draft retained"),
        (
            "draft",
            "photo.png · 12 B",
            False,
            "Composer hidden · Draft retained · Attachment retained",
        ),
        ("", None, True, "Composer hidden · Generating"),
        (
            "draft",
            "photo.png · 12 B",
            True,
            "Composer hidden · Generating · Draft retained · Attachment retained",
        ),
    ],
)
async def test_console_collapsed_status_uses_presence_only(
    draft: str,
    attachment: str | None,
    run_active: bool,
    expected: str,
):
    host = _ready_console_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft(draft)
        composer.set_pending_attachment_label(attachment)
        composer.sync_action_state(
            has_draft=bool(draft.strip()),
            run_active=run_active,
            can_save_chatbook=False,
        )
        composer.set_collapsed(True)
        await pilot.pause()

        status = composer.query_one("#console-composer-collapsed-status", Static)
        assert str(status.renderable) == expected
        stop = composer.query_one("#console-collapsed-stop-generation", Button)
        assert stop.display is run_active
        assert "photo.png" not in str(status.renderable)


@pytest.mark.asyncio
async def test_console_composer_round_trip_preserves_editor_and_attachment_state():
    host = _ready_console_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        pasted_text = "preserved large paste " * 20

        composer.insert_pasted_text(pasted_text)
        composer.move_cursor_left()
        composer.set_pending_attachment_label("photo.png · 12 B")
        before_caret_round_trip = (
            composer.draft_text(),
            composer.cursor_index,
            composer.has_paste_segments(),
            composer.has_full_draft_selection(),
        )

        composer.set_collapsed(True)
        composer.set_collapsed(False)
        await pilot.pause()

        assert (
            composer.draft_text(),
            composer.cursor_index,
            composer.has_paste_segments(),
            composer.has_full_draft_selection(),
        ) == before_caret_round_trip

        assert composer.select_all_draft() is True
        before_selection_round_trip = (
            composer.cursor_index,
            composer.has_full_draft_selection(),
        )

        composer.set_collapsed(True)
        composer.set_collapsed(False)
        await pilot.pause()

        assert (
            composer.cursor_index,
            composer.has_full_draft_selection(),
        ) == before_selection_round_trip
        attachment = composer.query_one("#console-attachment-indicator", Static)
        assert "photo.png · 12 B" in str(attachment.renderable)


@pytest.mark.asyncio
async def test_console_composer_collapse_preserves_pending_unfurl_segment():
    host = _ready_console_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        pasted_text = "pending unfurl paste " * 20

        composer.insert_pasted_text(pasted_text)
        await pilot.click("#console-command-visible-text")
        await pilot.pause()
        assert composer.has_pending_paste_confirmation() is True

        composer.set_collapsed(True)
        await pilot.pause()

        assert composer.draft_text() == pasted_text
        assert composer.has_paste_segments() is True
        assert composer.has_pending_paste_confirmation() is True


@pytest.mark.asyncio
async def test_console_composer_collapse_pauses_cursor_timer_despite_lingering_focus():
    host = _ready_console_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        timer = composer._cursor_blink_timer
        assert timer is not None
        assert timer._active.is_set() is True

        composer.set_collapsed(True)
        composer._sync_cursor_blink_state()

        assert timer._active.is_set() is False


@pytest.mark.parametrize(
    "stylesheet",
    (_SOURCE_STYLESHEET, _BUNDLED_STYLESHEET),
    ids=("source", "bundle"),
)
def test_console_composer_collapsed_styles_are_pinned(stylesheet: Path):
    css = stylesheet.read_text(encoding="utf-8")
    required = (
        "#console-native-composer.console-composer-collapsed",
        "#console-composer-collapsed-status",
        "#console-composer-expand",
        "text-overflow: ellipsis",
    )

    for token in required:
        assert token in css
