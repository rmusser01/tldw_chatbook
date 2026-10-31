"""A custom-capture press must preserve an identically reauthored draft."""

from __future__ import annotations

import pytest

from Tests.UI.test_console_approval_compact_layout import (
    _wait_for_reconciled_console,
)
from Tests.UI.test_console_send_acknowledgement import (
    DRAFT,
    ConsoleComposerBar,
    _wait_for_selector,
    build,
    eager_tasks,
    paint_state,
    press,
    until,
)
from Tests.UI.test_console_send_admission_off_pump import (
    B_DRAFT,
    ENTRY_SECONDS,
    HeldMcpRead,
    _second_tab,
    _sent_or_queued,
)
from Tests.UI.test_console_send_resend_guard import _idle
from Tests.UI.test_console_turn_resend_ui import _select_ready_llamacpp_console

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize("hidden_owner", [False, True], ids=["visible", "hidden"])
async def test_custom_capture_keeps_identically_reauthored_pressed_draft(hidden_owner):
    from tldw_chatbook.MCP.console_snapshot import standard_console_sources

    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console = host.screen_stack[-1]
            runtime = console._console_runtime()
            hold = None
            try:
                await _wait_for_selector(console, pilot, "#console-native-composer")
                await _wait_for_reconciled_console(console, pilot)
                await _select_ready_llamacpp_console(console, pilot)
                composer = console.query_one(
                    "#console-native-composer", ConsoleComposerBar
                )
                store = console._ensure_console_chat_store()
                session_a = store.active_session_id
                session_b = None
                if hidden_owner:
                    session_a, session_b = await _second_tab(console, pilot, ready=True)
                    await _wait_for_reconciled_console(console, pilot)
                    await _select_ready_llamacpp_console(console, pilot)
                composer.load_draft(DRAFT)
                composer.focus()
                await until(
                    lambda: composer.draft_text() == DRAFT
                    and store.session_draft(session_a) == DRAFT
                    and console._console_visible_draft_session_id == session_a
                    and store.active_session_id == session_a
                    and host.focused is composer
                    and paint_state(host).draft_in_composer
                )
                original_inputs = store.session_input_snapshot(session_a)
                hold = HeldMcpRead(host.app_instance.unified_mcp_service)
                assert not standard_console_sources(
                    host.app_instance.unified_mcp_service
                )
                gateway.validation_release.set()
                try:
                    press(host, "enter", "\r")
                    await until(hold.entered.is_set, timeout=ENTRY_SECONDS)
                    assert not hold.timed_out
                    assert gateway.stream_calls == 0
                    press(host, "ctrl+u")
                    await until(lambda: composer.draft_text() == "", timeout=10)
                    for character in DRAFT:
                        press(
                            host,
                            "space" if character == " " else character,
                            character,
                        )
                    await until(
                        lambda: composer.draft_text() == DRAFT
                        and store.session_draft(session_a) == DRAFT
                        and store.session_input_snapshot(session_a).draft_revision
                        > original_inputs.draft_revision,
                        timeout=10,
                    )
                    reauthored_revision = store.session_input_snapshot(
                        session_a
                    ).draft_revision
                    assert not hold.timed_out
                    if hidden_owner:
                        press(host, "alt+2")
                        await until(
                            lambda: console._console_visible_draft_session_id
                            == session_b
                            and composer.draft_text() == B_DRAFT,
                            timeout=10,
                        )
                finally:
                    hold.release.set()
                await until(lambda: gateway.stream_calls == 1)
                await _idle(host, console, pilot, session_a)
                assert not hold.timed_out
                assert _sent_or_queued(console, session_a) == [DRAFT]
                assert gateway.stream_calls == 1
                assert store.session_draft(session_a) == DRAFT, {
                    "stored_a": store.session_draft(session_a),
                    "composer": composer.draft_text(),
                    "reauthored_revision": reauthored_revision,
                    "final_revision": store.session_input_snapshot(
                        session_a
                    ).draft_revision,
                }
                if hidden_owner:
                    assert store.session_draft(session_b) == B_DRAFT
                    assert store.active_session_id == session_b
                    assert console._console_visible_draft_session_id == session_b
                    assert composer.draft_text() == B_DRAFT
                else:
                    assert composer.draft_text() == DRAFT
            finally:
                if hold is not None:
                    hold.release.set()
                gateway.validation_release.set()
                await runtime.dispose()
