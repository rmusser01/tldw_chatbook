"""A send whose durable commit fails never jams its conversation.

TASK-33621.53 and TASK-33620.3 (Console UX review 2026-09-29). A commit that
failed before the store reserved it -- the store's own preflight ("Durable
identity owner changed." and its siblings) or the controller's "no committed
owner" check -- left the preparation COMMITTING. COMMITTING is not a pause,
so neither the card nor the shelf could end it: Discard dropped the shelf
entry and every later send was refused ("Console session already has an
admission owner."). It now pauses as PERSISTENCE, exactly like a commit that
failed inside its transaction: the card says "Send could not be saved", and
Retry send, Cancel send, Restore and Discard each end it.

These drive the real ChatScreen, controller, store, runtime and an in-memory
ChaChaNotes database. Only the provider gateway is a double, and the forced
failure is injected inside the real store's commit, before its reservation.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.test_console_library_auto_outcome import (
    FIRST,
    REPLY,
    SECOND,
    _Gateway,
    _shelf_offers_unsent_turn,
)
from Tests.UI.test_console_native_chat_flow import (
    _build_console_send_test_app,
    _select_llamacpp_console,
    _wait_for_selector,
)
from Tests.UI.test_console_send_acknowledgement import (
    _painted_lines,
    eager_tasks,
    press,
    until,
)
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsoleTurnPreparationState,
)
from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_setup_modal import ConsoleSetupModal

pytestmark = pytest.mark.bootstrap_profile

_NOT_SAVED = "Send could not be saved; your message was not sent"


def _build(gateway):
    from tldw_chatbook.Chat.chat_conversation_scope_service import (
        ChatConversationScopeService,
    )
    from tldw_chatbook.Chat.chat_conversation_service import (
        ChatConversationService,
    )

    app = _build_console_send_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: gateway
    # The real conversation services over the app's own ChaChaNotes DB: a
    # second turn reads its chat's archive state, and resume reads the tree.
    service = ChatConversationService(app.chachanotes_db)
    app.local_chat_conversation_service = service
    app.chat_conversation_scope_service = ChatConversationScopeService(
        local_service=service, server_service=None
    )
    return ConsoleHarness(app), service


def _screen_text(host) -> str:
    return "\n".join(_painted_lines(host))


def _session(store, session_id: str):
    return next(item for item in store.sessions() if item.id == session_id)


def _user_rows(db, conversation_id: str, content: str) -> int:
    row = db.execute_query(
        "SELECT COUNT(*) FROM messages WHERE conversation_id = ? AND sender = ? "
        "AND content = ? AND deleted = 0",
        (conversation_id, "user", content),
    ).fetchone()
    return int(row[0])


async def _ready(host, pilot):
    console = host.screen_stack[-1]
    await _wait_for_selector(console, pilot, "#console-native-composer")
    await _select_llamacpp_console(console)
    await pilot.pause(0.3)
    composer = console.query_one("#console-native-composer", ConsoleComposerBar)
    composer.focus()
    return console, console._ensure_console_chat_controller(), composer


def _setup_modal_blocking(console) -> bool:
    return any(modal.is_blocking for modal in console.query(ConsoleSetupModal))


def _composer_focused(host, composer) -> bool:
    focused = host.focused
    return focused is not None and (
        focused is composer or composer in focused.ancestors
    )


async def _enter(host, pilot, composer) -> None:
    """Press Enter in the composer, as a user would.

    Resuming a chat re-reads its settings before the test selects llama.cpp,
    so the first-run setup modal can block the Console for a moment and take
    focus; an Enter pressed then lands on the modal's button and sends
    nothing. Wait for it to clear and put focus back in the composer first.
    """
    await until(lambda: not _setup_modal_blocking(composer.screen))
    composer.focus()
    await pilot.pause()
    await until(lambda: _composer_focused(host, composer))
    press(host, "enter", "\r")


async def _send(host, pilot, composer, text: str) -> None:
    composer.load_draft(text)
    await _enter(host, pilot, composer)


def _idle(controller, session_id: str) -> bool:
    activity = controller.activity_for(session_id)
    return not (activity.occupies_slot or activity.accepted_live_turn)


def _fail_next_commit_before_reservation(monkeypatch) -> list[int]:
    """Make the store's next durable commit fail in its preflight, once.

    ``durable_acceptance`` runs inside ``ConsoleChatStore.commit_durable_turn``
    before the commit reservation exists, where "Durable identity owner
    changed." and its siblings are raised. Its error leaves the store's own
    PERSISTENCE pause unreachable, which is the TASK-33621.53 jam.
    """
    from tldw_chatbook.Chat import console_legacy_flat_roots as flat_roots

    original = flat_roots.durable_acceptance
    calls: list[int] = []

    def fail_once(acceptance, nodes):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("Durable identity owner changed.")
        return original(acceptance, nodes)

    monkeypatch.setattr(flat_roots, "durable_acceptance", fail_once)
    return calls


def _preparation(controller, session_id: str):
    return controller.store.preparation_for_session(session_id)


async def _failed_send(host, pilot, monkeypatch):
    """Send FIRST and let its durable commit fail before the reservation."""
    calls = _fail_next_commit_before_reservation(monkeypatch)
    console, controller, composer = await _ready(host, pilot)
    session_id = controller.store.active_session_id
    await _send(host, pilot, composer, FIRST)
    await until(lambda: bool(calls))
    # The runtime shelves the refused turn once the controller has returned.
    await until(lambda: _shelf_offers_unsent_turn(console))
    return console, controller, composer, session_id


@pytest.mark.asyncio
@pytest.mark.parametrize("shelf_action", ["Discard", "Restore"])
async def test_a_send_whose_commit_fails_before_its_reservation_never_jams(
    shelf_action, monkeypatch
):
    """TASK-33621.53 AC#1/#3/#4: Discard or Restore lets the next send through.

    Before the fix the preparation stayed COMMITTING: the shelf's Discard
    dropped its entry and the next send was refused; Restore put the text
    back, and sending it was refused the same way. Measured on dev 3efe8476ce:
    both notify "Console session already has an admission owner." and never
    reach the provider.
    """
    gateway = _Gateway()
    host, _service = _build(gateway)
    db = host.app_instance.chachanotes_db
    async with host.run_test(size=(120, 40)) as pilot:
        with eager_tasks():
            console, controller, composer, session_id = await _failed_send(
                host, pilot, monkeypatch
            )
            store = controller.store
            assert gateway.stream_calls == 0
            button_id = (
                "#console-prompt-queue-pause"
                if shelf_action == "Discard"
                else "#console-prompt-queue-manage"
            )
            assert str(console.query_one(button_id).label) == shelf_action
            console.query_one(button_id, Button).press()
            await until(lambda: not _shelf_offers_unsent_turn(console))
            notices: list[str] = []
            if shelf_action == "Restore":
                # The message is back in the composer; sending it sends it once.
                await until(lambda: composer.draft_text() == FIRST)
                host.app_instance.notify = lambda message, **_k: notices.append(
                    str(message)
                )
                await _enter(host, pilot, composer)
                expected = FIRST
            else:
                host.app_instance.notify = lambda message, **_k: notices.append(
                    str(message)
                )
                await _send(host, pilot, composer, SECOND)
                expected = SECOND
            await until(lambda: gateway.stream_calls == 1 or bool(notices))
            # Before the fix the failed send still held the chat, so this
            # send was refused ("Console session already has an admission
            # owner.") and never reached the provider.
            assert not notices, notices
            await until(lambda: REPLY in _screen_text(host))
            assert gateway.stream_calls == 1
            assert _NOT_SAVED not in _screen_text(host)
            conversation_id = _session(store, session_id).persisted_conversation_id
            assert conversation_id is not None
            assert _user_rows(db, conversation_id, expected) == 1
            if shelf_action == "Discard":
                assert _user_rows(db, conversation_id, FIRST) == 0


@pytest.mark.asyncio
async def test_a_failed_commit_says_not_saved_and_leaves_the_running_state(
    monkeypatch,
):
    """TASK-33620.3 AC#1/#2: honest copy, no live run, and the way out.

    Before the fix the send stayed COMMITTING: no card, no "not saved" copy,
    and nothing but a shelf entry whose Discard released nothing.
    """
    gateway = _Gateway()
    host, _service = _build(gateway)
    async with host.run_test(size=(120, 40)) as pilot:
        with eager_tasks():
            console, controller, composer, session_id = await _failed_send(
                host, pilot, monkeypatch
            )
            # The failed send is a pause the card and the shelf can both end.
            preparation = _preparation(controller, session_id)
            assert preparation is not None
            assert preparation.state is ConsoleTurnPreparationState.PAUSED, (
                preparation.state
            )
            assert preparation.pause_kind is ConsolePreparationPauseKind.PERSISTENCE
            await until(lambda: _NOT_SAVED in _screen_text(host))
            screen = _screen_text(host)
            # Not saved and not sent; never provider setup.
            assert "Impact: The provider was not contacted." in screen
            assert "Run: Blocked — send not saved" in screen
            assert "finish provider setup" not in screen
            assert "Preparing..." not in screen
            assert "Running" not in screen
            assert "Sending" not in screen
            assert not composer.has_class("console-composer-setup-blocked")
            # Not a live run: the quit dialog has nothing to count.
            assert _idle(controller, session_id)
            assert controller.lifecycle_impact().live_run_count == 0
            # The card offers the retry and the way to take the message back.
            retry = console.query_one("#console-trace-retry", Button)
            assert retry.display and str(retry.label) == "Retry send"
            assert console.query_one("#console-trace-cancel", Button).display
            assert gateway.stream_calls == 0

            # Cancel send puts the message back and the conversation sends again.
            console.query_one("#console-trace-cancel", Button).press()
            await until(lambda: _preparation(controller, session_id) is None)
            await until(lambda: composer.draft_text() == FIRST)
            await until(lambda: _NOT_SAVED not in _screen_text(host))
            await _enter(host, pilot, composer)
            await until(lambda: REPLY in _screen_text(host))
            assert gateway.stream_calls == 1


@pytest.mark.asyncio
async def test_card_retry_after_a_commit_failure_sends_the_turn_once(monkeypatch):
    """TASK-33620.3 AC#3: Retry send from the failure state answers the turn."""
    gateway = _Gateway()
    host, _service = _build(gateway)
    db = host.app_instance.chachanotes_db
    async with host.run_test(size=(120, 40)) as pilot:
        with eager_tasks():
            console, controller, _composer, session_id = await _failed_send(
                host, pilot, monkeypatch
            )
            store = controller.store
            assert _preparation(controller, session_id).state is (
                ConsoleTurnPreparationState.PAUSED
            )
            await until(lambda: _NOT_SAVED in _screen_text(host))
            retry = console.query_one("#console-trace-retry", Button)
            await until(lambda: retry.display and not retry.disabled)
            retry.press()
            await until(lambda: gateway.stream_calls == 1)
            await until(lambda: REPLY in _screen_text(host))
            await until(lambda: not _shelf_offers_unsent_turn(console))
            assert _NOT_SAVED not in _screen_text(host)
            conversation_id = _session(store, session_id).persisted_conversation_id
            assert _user_rows(db, conversation_id, FIRST) == 1
