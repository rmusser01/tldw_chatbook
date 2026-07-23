"""TASK-339: keystrokes typed during the session-switch settle window.

`_sync_console_session_draft` used to save the LIVE composer text to the old
session and reload the new session's stored draft whenever the deferred swap
finally ran — keystrokes typed after the switch keypress were misattributed
to the old session and wiped/reordered in the composer (UX review finding
j2-post-switch-typing-reordered; the mangled text was sent and persisted).

The settle window is simulated deterministically: the coalescing guard is
held so the activation's inline sync defers, typing happens in the gap, and
the swap then runs exactly as a later sync pass would.
"""

import pytest

from Tests.UI.test_console_native_chat_flow import _select_llamacpp_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Widgets.Console import ConsoleComposerBar


async def _settle_window_swap(console, pilot, *, type_during_window: str):
    """Switch to a fresh session while the sync is held, typing in the gap."""
    store = console._ensure_console_chat_store()
    session_a = store.ensure_session(title="Chat A")
    composer = console.query_one("#console-native-composer", ConsoleComposerBar)
    composer.focus()
    await pilot.pause()

    composer.load_draft("old draft")
    console._sync_console_session_draft()  # settle tracker onto A

    # Hold the coalescing guard: the new-tab path's inline sync defers, so
    # the draft swap runs only on a LATER pass — the real settle window.
    console._console_sync_in_progress = True
    try:
        await console._create_native_console_session_from_active_context()
        if type_during_window:
            composer.insert_text(type_during_window)
    finally:
        console._console_sync_in_progress = False

    session_b_id = store.active_session_id
    assert session_b_id != session_a.id
    console._sync_console_session_draft()
    return store, session_a, session_b_id, composer


@pytest.mark.asyncio
async def test_console_switch_carries_settle_window_typing_to_new_session():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)

        store, session_a, session_b_id, composer = await _settle_window_swap(
            console, pilot, type_during_window="typed in window"
        )

        # The keystrokes belong to the NEW session, in order…
        assert composer.draft_text() == "typed in window"
        # …and the OLD session keeps only what it actually had at the
        # switch keypress (no misattributed keystrokes).
        assert store.session_draft(session_a.id) == "old draft"


@pytest.mark.asyncio
async def test_console_switch_without_typing_swaps_drafts_exactly_as_before():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)

        store, session_a, session_b_id, composer = await _settle_window_swap(
            console, pilot, type_during_window=""
        )

        assert composer.draft_text() == ""
        assert store.session_draft(session_a.id) == "old draft"
