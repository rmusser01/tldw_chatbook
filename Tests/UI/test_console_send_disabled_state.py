"""TASK-2154.6 (FR-04, DS-07): Send's real disabled state + persistent reason.

Before this task Send was never Textual-``disabled`` -- the blocked state was
conveyed only via CSS classes and a hover tooltip (and an empty draft got no
tooltip at all), while the ``#console-send-disabled-reason`` Static stayed
permanently ``display:none``. These tests pin the new contract:

- Send is genuinely disabled whenever setup/pre-acceptance/full-queue state
  blocks dispatch or the draft is empty, and becomes Queue once an accepted
  turn can own the next prompt.
- The blocked/empty reason is perceivable WITHOUT hover: the reason strip
  renders inline (the 1fr draft yields the cells; the strip never adds
  height to the one-row composer) and clears immediately.
- The Enter hotkey keeps both of its behaviors: it sends when enabled, and
  it still drives the blocked-attempt feedback (toast + transcript system
  row) while disabled -- ``Button.press()`` no-ops on a disabled control,
  so the key handler dispatches the Pressed handler directly.
- DS-07: the idle Stop button no longer carries an unreachable tooltip.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_console_command_composer import _spy_submit_draft
from Tests.UI.test_console_dictation import _mounted_console, _ready_host
from Tests.UI.test_console_native_chat_flow import (
    CapturingGateway,
    _select_llamacpp_console,
    _wait_for_text,
)
from Tests.UI.test_console_regenerate_feedback import GatedGateway
from Tests.UI.test_destination_shells import _build_test_app
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_display_state import build_console_disabled_reason
from tldw_chatbook.Widgets.Console import ConsoleComposerBar


async def _wait_for_condition(pilot, predicate, timeout: float = 4.0) -> None:
    """Poll ``predicate`` until it holds, advancing the pilot meanwhile."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.05)
    assert predicate()


# --- pure copy helper -------------------------------------------------------


def test_disabled_reason_helper_covers_run_blocked_state():
    """A block with no setup copy is an active run; say so with or without a draft."""
    for has_draft in (False, True):
        assert (
            build_console_disabled_reason(
                action_id="send",
                has_draft=has_draft,
                send_blocked=True,
                setup_blocked_reason="",
            )
            == "Send blocked — wait for the active run to finish"
        )
    # Not blocked at all: the empty-draft guidance still applies.
    assert (
        build_console_disabled_reason(
            action_id="send", has_draft=False, send_blocked=False
        )
        == "Send disabled: type a message"
    )


# --- widget-level contract ---------------------------------------------------


@pytest.mark.asyncio
async def test_empty_draft_disables_send_with_visible_idle_reason():
    app, host = _ready_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        send_button = composer.query_one("#console-send-message", Button)
        reason = composer.query_one("#console-send-disabled-reason", Static)

        await _wait_for_condition(
            pilot, lambda: reason.styles.display == "block"
        )

        assert send_button.disabled is True
        assert reason.renderable.plain == "Send disabled: type a message"
        assert reason.has_class("console-send-disabled-reason-idle")
        assert send_button.tooltip == "Type a message to send."


@pytest.mark.asyncio
async def test_typing_enables_send_and_clears_reason_immediately():
    app, host = _ready_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        send_button = composer.query_one("#console-send-message", Button)
        reason = composer.query_one("#console-send-disabled-reason", Static)

        await _wait_for_condition(pilot, lambda: send_button.disabled is True)

        # AC#3: the very first character clears the disabled state and the
        # reason synchronously -- no waiting for the 0.2s UI-sync tick.
        composer.insert_text("h")
        assert send_button.disabled is False
        assert reason.styles.display == "none"
        assert send_button.tooltip == "Send the active Console session draft."

        # Deleting back to empty restores both, again synchronously.
        composer.delete_left()
        assert send_button.disabled is True
        assert reason.styles.display == "block"
        assert reason.renderable.plain == "Send disabled: type a message"


@pytest.mark.asyncio
async def test_setup_block_shows_reason_and_clears_when_unblocked():
    app, host = _ready_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        send_button = composer.query_one("#console-send-message", Button)
        reason = composer.query_one("#console-send-disabled-reason", Static)

        composer.load_draft("draft ready")
        composer.sync_action_state(
            has_draft=True,
            run_active=False,
            can_save_chatbook=False,
            send_blocked=True,
            setup_blocked_reason="Choose a model in Console Settings before sending.",
        )
        await pilot.pause(0.1)

        assert send_button.disabled is True
        assert reason.styles.display == "block"
        # TASK-21145 (UAT H-3): a setup blocker carries its own way out —
        # the reason strip appends an "Open setup" action link wired to
        # app.run_setup_wizard.
        assert reason.renderable.plain == (
            "Send blocked — choose a model to continue ›"
        )
        assert not reason.has_class("console-send-disabled-reason-idle")
        assert (
            send_button.tooltip
            == "Choose a model in Console Settings before sending."
        )

        # Configuring the model unblocks: same sync, blockers cleared.
        composer.sync_action_state(
            has_draft=True,
            run_active=False,
            can_save_chatbook=False,
            send_blocked=False,
            setup_blocked_reason="",
        )
        await pilot.pause(0.1)

        assert send_button.disabled is False
        assert reason.styles.display == "none"
        assert send_button.tooltip == "Send the active Console session draft."


@pytest.mark.asyncio
async def test_active_run_block_shows_wait_reason():
    app, host = _ready_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        send_button = composer.query_one("#console-send-message", Button)
        reason = composer.query_one("#console-send-disabled-reason", Static)

        composer.load_draft("queued behind the run")
        composer.sync_action_state(
            has_draft=True,
            run_active=True,
            can_save_chatbook=False,
            send_blocked=True,
            setup_blocked_reason="",
        )
        await pilot.pause(0.1)

        assert send_button.disabled is True
        assert reason.styles.display == "block"
        assert (
            reason.renderable.plain
            == "Send blocked — wait for the active run to finish"
        )
        assert (
            send_button.tooltip
            == "Wait for the active Console run to finish before sending."
        )

        # Run finishing clears the block even though the draft never changed.
        composer.sync_action_state(
            has_draft=True,
            run_active=False,
            can_save_chatbook=False,
            send_blocked=False,
            setup_blocked_reason="",
        )
        await pilot.pause(0.1)

        assert send_button.disabled is False
        assert reason.styles.display == "none"


@pytest.mark.asyncio
async def test_reason_strip_never_adds_height_to_the_composer_row():
    app, host = _ready_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        reason = composer.query_one("#console-send-disabled-reason", Static)
        send_button = composer.query_one("#console-send-message", Button)
        expanded = composer.query_one("#console-composer-expanded")

        await _wait_for_condition(pilot, lambda: reason.styles.display == "block")
        await pilot.pause(0.2)

        # One-row composer, reason visible: the strip shares the Send row.
        assert expanded.region.height == 1
        assert reason.region.height <= 1
        assert reason.region.y == send_button.region.y


@pytest.mark.asyncio
async def test_idle_stop_button_has_no_unreachable_tooltip():
    """DS-07: the hidden idle Stop must not carry copy nobody can hover."""
    app, host = _ready_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        stop_button = composer.query_one("#console-stop-generation", Button)

        assert stop_button.styles.display == "none"
        assert stop_button.tooltip is None

        composer.sync_action_state(
            has_draft=False,
            run_active=True,
            can_save_chatbook=False,
            send_blocked=True,
        )
        await pilot.pause(0.1)

        assert stop_button.styles.display == "block"
        assert stop_button.tooltip == "Stop this tab's run."


# --- Enter hotkey (AC#4) -----------------------------------------------------


@pytest.mark.asyncio
async def test_enter_hotkey_still_sends_when_send_is_enabled():
    gateway = CapturingGateway()
    app, host = _ready_host()
    app.console_provider_gateway_factory = lambda: gateway

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("enter sends this")
        composer.focus()
        submit_spy = await _spy_submit_draft(console)

        await pilot.press("enter")
        await _wait_for_text(console, pilot, "accepted")

        submit_spy.assert_awaited_once_with(
            "enter sends this",
            session_id=console._ensure_console_chat_store().active_session_id,
        )
        assert gateway.sent_messages[-1][-1]["content"] == "enter sends this"


@pytest.mark.asyncio
async def test_enter_hotkey_queues_draft_behind_accepted_run():
    """An accepted live turn changes Send to Queue and preserves exact text."""
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    gateway = GatedGateway()
    app.console_provider_gateway_factory = lambda: gateway
    notify_mock = Mock()
    app.notify = notify_mock
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        send_button = composer.query_one("#console-send-message", Button)
        reason = composer.query_one("#console-send-disabled-reason", Static)

        # Start a held run: draft present, no run yet -- Send is enabled.
        composer.load_draft("stream and hold")
        await pilot.pause(0.1)
        assert send_button.disabled is False
        console.query_one("#console-send-message").press()
        store = console._ensure_console_chat_store()
        await _wait_for_condition(
            pilot,
            lambda: any(
                m.content.startswith("first-chunk")
                for m in store.messages_for_session(store.active_session_id)
                if m.role is ConsoleMessageRole.ASSISTANT
            ),
        )

        try:
            # An empty draft stays disabled, but the action truthfully names
            # the now-available queue path.
            await _wait_for_condition(pilot, lambda: send_button.disabled is True)
            assert send_button.label.plain == "Queue"
            assert reason.styles.display == "block"
            assert reason.renderable.plain == "Send disabled: type a message"

            # A real draft enables Queue; Enter admits the exact text behind
            # the accepted turn through the same dispatcher as the button.
            composer.load_draft("queued behind run")
            composer.focus()
            await pilot.pause(0.1)
            assert send_button.disabled is False
            await pilot.press("enter")

            await _wait_for_condition(
                pilot,
                lambda: console._ensure_console_chat_controller()
                .prompt_queue_registry.snapshot(store.active_session_id)
                .total_count
                == 1,
            )
            snapshot = (
                console._ensure_console_chat_controller()
                .prompt_queue_registry.snapshot(store.active_session_id)
            )
            queued = snapshot.entries[0]
            text = (
                console._ensure_console_chat_controller()
                .prompt_queue_registry.read_waiting_text(
                    store.active_session_id,
                    entry_id=queued.entry_id,
                    expected_revision=snapshot.revision,
                )
            )
            assert text.text == "queued behind run"
            assert composer.draft_text() == ""
            assert console._console_pending_send_stash is None
        finally:
            gateway.release.set()
