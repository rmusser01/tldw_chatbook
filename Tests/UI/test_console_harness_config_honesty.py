"""The test app factory must hand Console a config a test can actually set.

task-15270. `_build_test_app` used to patch `load_settings` to a three-key
synthetic dict carrying no `[console]`/`[chat_defaults]` section and none of
the sections `load_settings()` always emits. Production correctly refuses to
refresh a snapshot that never came from disk
(`ChatScreen._console_config_snapshot_is_disk_loaded`), so every mounted
Console test read a `ConsoleTurnExecutionContext` frozen at defaults no
matter what the test had persisted -- which is how
`test_send_proceeds_when_auto_retrieve_fails` stayed green while never once
calling the exploding backend it existed to exercise (task-15210).

These tests pin the harness contract itself: what a test persists is what the
turn-context snapshot reads.
"""

from __future__ import annotations

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.config import save_setting_to_cli_config
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


@pytest.fixture(autouse=True)
def _restore_toggles():
    """Leave the isolated config as this suite found it."""
    yield
    save_setting_to_cli_config("chat_defaults", "rag_auto_retrieve_on_send", False)


def test_factory_config_carries_the_disk_load_markers():
    """The snapshot must look disk-loaded, or Console never refreshes it."""
    app = _build_test_app()

    assert ChatScreen._console_config_snapshot_is_disk_loaded(app.app_config) is True


def test_factory_config_reflects_a_persisted_chat_default():
    """A `[chat_defaults]` value a test persists reaches `app_config`."""
    save_setting_to_cli_config("chat_defaults", "rag_auto_retrieve_on_send", True)

    app = _build_test_app()

    assert app.app_config["chat_defaults"]["rag_auto_retrieve_on_send"] is True


@pytest.mark.asyncio
async def test_persisted_chat_default_reaches_the_turn_context_snapshot():
    """End to end: persisted toggle -> mounted Console -> frozen turn context.

    This is the read that `_maybe_auto_retrieve_for_send` gates on since
    task-14803, and the one that silently defaulted for every mounted test.
    """
    save_setting_to_cli_config("chat_defaults", "rag_auto_retrieve_on_send", True)
    app = _build_test_app()

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        controller = screen._ensure_console_chat_controller()
        session_id = controller.store.active_session_id

        context = controller.resolve_turn_configuration_snapshot(session_id)

        assert context.rag_defaults["auto_retrieve_on_send"] is True
