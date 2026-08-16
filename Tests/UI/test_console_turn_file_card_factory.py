"""The transcript renders a card for change-summary rows -- switch-gated.

OFF must be byte-identical to today's marker row: the kill switch is a
pure presentation toggle (spec §2 re-scoped)."""

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Widgets.Console.console_turn_file_card import (
    ConsoleTurnFileCard,
)

MARKER = "✎ Edited 2 files  +8 −3 — review with `v`"


def _append_summary_message(store):
    """Append a change-summary TOOL marker with a real run id to append to.

    ``ConsoleChatStore.append_message`` takes a session id plus scalar
    fields, not a pre-built ``ConsoleChatMessage`` -- matches the
    established pattern other Console harness tests use.
    """
    session = store.ensure_session()
    return store.append_message(
        session.id,
        role=ConsoleMessageRole.TOOL,
        content=MARKER,
        change_review_run_id="run-1",
    )


@pytest.mark.asyncio
async def test_summary_row_renders_card_when_enabled(monkeypatch):
    from Tests.UI.test_console_native_chat_flow import (
        ConsoleHarness,
        _build_test_app,
        _wait_for_selector,
    )

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        _append_summary_message(store)
        await console._sync_native_console_chat_ui()
        for _ in range(40):
            if console.query(ConsoleTurnFileCard):
                break
            await pilot.pause(0.02)
        assert console.query(ConsoleTurnFileCard)


@pytest.mark.asyncio
async def test_summary_row_stays_plain_marker_when_disabled(monkeypatch):
    from Tests.UI.test_console_native_chat_flow import (
        ConsoleHarness,
        _build_test_app,
        _wait_for_selector,
        _wait_for_text,
    )
    import tldw_chatbook.Widgets.Console.console_transcript as transcript_mod

    monkeypatch.setattr(
        transcript_mod,
        "get_cli_setting",
        lambda section, key, default=None: (
            False
            if (section, key) == ("console", "turn_file_cards")
            else default
        ),
    )
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        _append_summary_message(store)
        await console._sync_native_console_chat_ui()
        # The plain marker text still renders, byte-identical, and no
        # card mounts -- the switch is a pure presentation toggle.
        await _wait_for_text(console, pilot, MARKER)
        assert not console.query(ConsoleTurnFileCard)
