"""Caret blinking must not move the surrounding draft (TASK-32012.1)."""

from __future__ import annotations

import asyncio
from contextlib import closing

import pytest
from rich.cells import cell_len
from textual.widgets import Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_native_chat_flow import _persist_console_provider_config
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.app import TldwCli
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Widgets.Console import console_composer_bar as module


def _assert_only_caret_changed(shown: str, hidden: str, glyph: str = "▌") -> None:
    assert len(shown) == len(hidden), (shown, hidden)
    differences = [(a, b) for a, b in zip(shown, hidden) if a != b]
    assert differences == [(glyph, " ")], (shown, hidden, differences)


@pytest.mark.parametrize("ascii_caret", [False, True])
@pytest.mark.parametrize(
    "draft,width,cursor,ghost",
    [
        ("hello world", 11, 11, ""),
        ("hello world again", 8, 7, ""),
        ("one two three four " * 12, 20, 170, ""),
        ("測試 hello world", 14, 14, ""),
        ("🧪 hello world", 13, 13, ""),
        ("literal ▌ and | hello world", 12, 17, ""),
        ("hello", 11, 5, " world again"),
        ("a\tb", 20, 3, ""),
        ("a\tb\nhello world", 20, 15, ""),
        ("", 11, 0, ""),
    ],
)
def test_blink_preserves_wrapped_text_and_styles(
    monkeypatch, ascii_caret, draft, width, cursor, ghost
):
    glyph = "|" if ascii_caret else "▌"
    monkeypatch.setattr(module, "resolve_glyph", lambda value: glyph)
    args = {
        "width": width,
        "focused": True,
        "cursor_index": cursor,
        "ghost_suffix": ghost,
        "style_ranges": [(0, len(draft), "bold")],
    }
    shown = module.ConsoleComposerBar._draft_renderable(
        draft, cursor_visible=True, **args
    )
    hidden = module.ConsoleComposerBar._draft_renderable(
        draft, cursor_visible=False, **args
    )
    _assert_only_caret_changed(shown.plain, hidden.plain, glyph)
    assert shown.spans == hidden.spans


def test_blink_preserves_style_after_expanding_pasted_tabs():
    for phase in (True, False):
        painted = module.ConsoleComposerBar._draft_renderable(
            "a\tb", focused=True, cursor_visible=phase, style_ranges=[(2, 3, "bold")]
        )
        assert [(span.start, span.end) for span in painted.spans] == [(8, 9)]
        assert painted.plain[8:9] == "b"


@pytest.mark.parametrize(
    "prefix", ["", "\t", "界\t", "🧪\t", "e\u0301\t", "👩\u200d💻\t", "e\u0301❤️\t"]
)
@pytest.mark.parametrize("size", [(80, 24), (97, 30)])
async def test_failed_enter_restores_draft_without_blink_text_movement(
    monkeypatch, tmp_path, size, prefix
):
    app = _build_test_app()
    with closing(CharactersRAGDB(tmp_path / "chat.sqlite", "blink-wrap")) as database:
        console = None
        try:
            app.chachanotes_db = database
            _persist_console_provider_config(
                app,
                provider="openai",
                model="gpt-4.1",
                provider_settings={"api_key": "synthetic-test-key"},
            )
            host = ConsoleHarness(app)
            host.CSS_PATH = TldwCli.CSS_PATH
            async with host.run_test(size=size) as pilot:
                console = host.screen
                await _wait_for_selector(console, pilot, "#console-native-composer")
                controller = console._ensure_console_chat_controller()
                runtime = console._console_runtime()
                accepted_tasks = []
                accept_turn = runtime.accept_turn

                def capture_turn(request, **kwargs):
                    turn_id = accept_turn(request, **kwargs)
                    accepted_tasks.append(runtime._turn_custody[turn_id].task)
                    return turn_id

                monkeypatch.setattr(runtime, "accept_turn", capture_turn)

                async def fail_resolution(selection):
                    raise ValueError("Synthetic validation interruption")

                monkeypatch.setattr(
                    controller.provider_gateway, "resolve_for_send", fail_resolution
                )
                composer = console._console_composer_or_none()
                width = composer._draft_render_width()
                prefix_width = cell_len(prefix.expandtabs(8))
                draft = prefix + "x" * (width - prefix_width - 6) + " world"
                composer.load_draft(draft)
                composer.focus()
                await pilot.pause()
                await pilot.press("enter")
                await asyncio.wait_for(host.workers.wait_for_complete(), timeout=10)
                assert len(accepted_tasks) == 1
                with pytest.raises(
                    ValueError, match="Synthetic validation interruption"
                ):
                    await asyncio.wait_for(accepted_tasks[0], timeout=10)
                # Runtime custody owns completion; the disposable poller then
                # reconciles the retained recovery without overwriting the draft.
                async with asyncio.timeout(5):
                    while console._console_transcript_sync_timer is not None:
                        await pilot.pause(0.05)
                assert composer.draft_text() == ""
                session_id = controller.store.active_session_id
                (recovery,) = runtime.recoveries_for_session(session_id)
                assert recovery.session_id == session_id
                assert recovery.draft == draft
                await console._prompt_queue.handle_primary_intent(
                    session_id,
                    action=f"turn-recovery:restore:{recovery.turn_id}",
                    expected_revision=controller.lifecycle_impact(
                        session_id=session_id
                    ).revision,
                )
                assert runtime.recoveries_for_session(session_id) == ()
                composer._cursor_blink_timer.pause()
                assert composer.draft_text() == draft
                assert controller.run_state.is_send_allowed
                assert console._console_transcript_sync_timer is None

                visible = composer.query_one("#console-command-visible-text", Static)
                phases = []
                for phase in (True, False):
                    composer._cursor_visible = not phase
                    composer._toggle_cursor_blink()
                    await pilot.pause()
                    phases.append(
                        "\n".join(
                            strip.text for strip in console._compositor.render_strips()
                        )
                    )
                    assert (
                        len(visible.renderable.plain.splitlines())
                        <= visible.size.height
                    )
                    assert "world" in visible.renderable.plain
                    if "❤️" in prefix:
                        assert composer._display_index_at(1, 0) == 2
                        assert composer._display_index_at(2, 0) == 2
                    if prefix:
                        assert (
                            composer._display_index_at(prefix_width - 1, 0)
                            == len(prefix) - 1
                        )
                        assert composer._display_index_at(prefix_width, 0) == len(
                            prefix
                        )
                    for row, text in enumerate(visible.renderable.plain.splitlines()):
                        if "world" in text:
                            assert (
                                composer._display_index_at(text.index("world") + 1, row)
                                == len(draft) - 4
                            )
                _assert_only_caret_changed(*phases)
                assert composer.draft_text() == draft
                if "❤️" in prefix:
                    await pilot.click("#console-command-visible-text", offset=(2, 0))
                    await pilot.press("!")
                    assert composer.draft_text() == draft[:2] + "!" + draft[2:]
        finally:
            if console is not None:
                await console._console_runtime().dispose()
            try:
                with database.quiesce_connections(timeout_seconds=5):
                    assert database.registered_connection_count() == 0
            finally:
                await asyncio.to_thread(app.ui_responsiveness_monitor.close)
