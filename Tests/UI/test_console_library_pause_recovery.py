"""A Library-paused send never jams, and is offered on one surface at a time.

TASK-33621.20 pre-merge review. Three ways the paused send's two surfaces --
the transcript card (Retry / Send once without Library / Cancel) and the
unsent-turn shelf (Restore / Discard) -- disagreed:

1. Send once (or Retry) with a provider that is not ready re-paused the send
   as DESTINATION_CHANGED: no card showed it, and the shelf's Discard did not
   release it, so every later send was refused ("Last send is blocked").
2. While a card action was sending the turn, the shelf still offered Restore
   and Discard: Restore put the message back for a second send, and Discard
   did nothing while the message went out anyway.
3. The card's Cancel put the text back but dropped the staged attachments.

These drive the real ChatScreen, controller, store and runtime. Only the
provider gateway and the Library search service are doubles.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from textual.widgets import Button

from Tests.Chat.test_console_automatic_library_preparation import _pending_image
from Tests.UI.test_console_library_auto_outcome import (
    FIRST,
    REPLY,
    SECOND,
    _CARD_ACTIONS,
    _Gateway,
    _StalledLibrary,
    _automatic_console,
    _paused_retrieval,
    _shelf_offers_unsent_turn,
)
from Tests.UI.test_console_native_chat_flow import _build_console_send_test_app
from Tests.UI.test_console_send_acknowledgement import (
    _painted_lines,
    eager_tasks,
    press,
    until,
)
from Tests.UI.test_console_send_admission_off_pump import _allow_second_turns
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsoleTurnPreparationState,
)
from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
    TraceCallRecoveryCallout,
)

pytestmark = pytest.mark.bootstrap_profile


class _HeldGateway(_Gateway):
    """Ready provider whose stream stays open until ``release`` is set."""

    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()

    async def stream_chat(self, resolution, messages, **kwargs):
        self.stream_calls += 1
        await self.release.wait()
        yield REPLY


def _build_with(gateway):
    app = _build_console_send_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    host = ConsoleHarness(app)
    app.console_provider_gateway_factory = lambda: gateway
    _allow_second_turns(host)
    return host


def _screen_text(host) -> str:
    return "\n".join(_painted_lines(host))


def _paused(controller, kind: ConsolePreparationPauseKind) -> bool:
    preparation = controller.store.preparation_for_session(
        controller.store.active_session_id
    )
    return (
        preparation is not None
        and preparation.state is ConsoleTurnPreparationState.PAUSED
        and preparation.pause_kind is kind
    )


async def _timed_out_send(host, pilot, library):
    """Send FIRST with Library on Automatic and let its search time out."""
    console, controller, composer = await _automatic_console(host, pilot)
    controller._library_preparation_timeout = 0.2
    composer.load_draft(FIRST)
    await pilot.pause()
    press(host, "enter", "\r")
    await until(lambda: _paused_retrieval(controller))
    await until(lambda: _shelf_offers_unsent_turn(console))
    await until(lambda: console.query_one(_CARD_ACTIONS["Cancel send"], Button).display)
    return console, controller, composer


@pytest.mark.asyncio
async def test_send_once_with_a_provider_that_is_not_ready_never_jams_the_chat():
    """CRITICAL 1: the re-paused send is shown, cancellable, and not a jam."""
    gateway = _Gateway()
    host = _build_with(gateway)
    library = _StalledLibrary()
    host.app_instance.library_rag_search_service = library
    async with host.run_test(size=(120, 40)) as pilot:
        with eager_tasks():
            console, controller, composer = await _timed_out_send(host, pilot, library)
            ready_resolver = controller._resolve_for_send_bounded

            async def not_ready(selection):
                return SimpleNamespace(
                    ready=False, visible_copy="Provider is not ready."
                )

            controller._resolve_for_send_bounded = not_ready
            console.query_one(
                _CARD_ACTIONS["Send once without Library"], Button
            ).press()
            await until(
                lambda: _paused(
                    controller, ConsolePreparationPauseKind.DESTINATION_CHANGED
                )
            )
            # The card now shows the re-paused send, with what it can do.
            await until(lambda: "Provider not ready" in _screen_text(host))
            await until(
                lambda: console.query_one("#console-trace-retry", Button).display
            )
            # The card covers it, so the runtime keeps it off the shelf at once.
            # The painted shelf follows on the next Console sync, which a sync
            # already running defers (it trailed under load).
            session_id = controller.store.active_session_id
            assert console._console_runtime().recoveries_for_session(session_id) == ()
            await until(lambda: not _shelf_offers_unsent_turn(console))
            assert gateway.stream_calls == 0

            controller._resolve_for_send_bounded = ready_resolver
            library.release.set()
            console.query_one(_CARD_ACTIONS["Cancel send"], Button).press()
            await until(lambda: composer.draft_text() == FIRST)
            assert controller.store.preparation_for_session(
                controller.store.active_session_id
            ) is None or not _paused(
                controller, ConsolePreparationPauseKind.DESTINATION_CHANGED
            )

            composer.load_draft(SECOND)
            await pilot.pause()
            press(host, "enter", "\r")
            await until(lambda: gateway.stream_calls == 1)
            await until(lambda: REPLY in _screen_text(host))
            assert "Last send is blocked" not in _screen_text(host)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["search", "stream"])
async def test_the_shelf_never_offers_a_send_the_card_is_sending(phase):
    """IMPORTANT 2: Retry's send is not also on the shelf; restore refuses."""
    gateway = _HeldGateway()
    host = _build_with(gateway)
    library = _StalledLibrary()
    host.app_instance.library_rag_search_service = library
    async with host.run_test(size=(120, 40)) as pilot:
        with eager_tasks():
            console, controller, composer = await _timed_out_send(host, pilot, library)
            runtime = console._console_runtime()
            session_id = controller.store.active_session_id
            (entry,) = runtime.recoveries_for_session(session_id)
            # The retried search may take its time; only Stop or a hit ends it.
            controller._library_preparation_timeout = 60.0
            if phase == "stream":
                library.release.set()
            console.query_one(_CARD_ACTIONS["Retry Library search"], Button).press()
            if phase == "search":
                await until(lambda: library.calls == 2)
            else:
                await until(lambda: gateway.stream_calls == 1)
            await pilot.pause()

            # Never offered: the runtime refuses it at once. The painted shelf
            # follows on the next Console sync (a running sync defers it).
            assert runtime.recoveries_for_session(session_id) == ()
            assert runtime.discard_turn_recovery(entry.turn_id) is False
            with pytest.raises(RuntimeError):
                runtime.restore_turn_recovery(entry.turn_id)
            await until(lambda: not _shelf_offers_unsent_turn(console))
            assert composer.draft_text() == ""

            library.release.set()
            gateway.release.set()
            await until(lambda: REPLY in _screen_text(host))
            await until(lambda: runtime.recoveries_for_session(session_id) == ())
            await until(lambda: not _shelf_offers_unsent_turn(console))
            assert gateway.stream_calls == 1
            assert composer.draft_text() == ""


@pytest.mark.asyncio
async def test_card_cancel_returns_the_message_with_its_staged_attachment(
    monkeypatch,
):
    """IMPORTANT 3: Cancel puts back the text and the staged image."""
    from tldw_chatbook.Chat import attachment_core
    from tldw_chatbook.Chat import console_chat_controller as controller_module

    monkeypatch.setattr(attachment_core, "vision_block_reason", lambda *a, **k: None)
    monkeypatch.setattr(controller_module, "vision_block_reason", lambda *a, **k: None)
    gateway = _Gateway()
    host = _build_with(gateway)
    library = _StalledLibrary()
    host.app_instance.library_rag_search_service = library
    async with host.run_test(size=(120, 40)) as pilot:
        with eager_tasks():
            console, controller, composer = await _automatic_console(host, pilot)
            store = controller.store
            session_id = store.active_session_id
            image = _pending_image("diagram.png", b"\x89PNG\r\n\x1a\n-diagram")
            assert store.add_pending_attachment(session_id, image)
            controller._library_preparation_timeout = 0.2
            composer.load_draft(FIRST)
            await pilot.pause()
            press(host, "enter", "\r")
            await until(lambda: _paused_retrieval(controller))
            await until(lambda: _shelf_offers_unsent_turn(console))
            # The send took the image with it.
            assert store.pending_attachments(session_id) == []

            await until(
                lambda: console.query_one(_CARD_ACTIONS["Cancel send"], Button).display
            )
            console.query_one(_CARD_ACTIONS["Cancel send"], Button).press()
            await until(lambda: composer.draft_text() == FIRST)
            await pilot.pause()
            restored = store.pending_attachments(session_id)
            assert [item.display_name for item in restored] == ["diagram.png"]
            assert console._console_runtime().recoveries_for_session(session_id) == ()
            await until(lambda: not _shelf_offers_unsent_turn(console))
            assert gateway.stream_calls == 0


@pytest.mark.asyncio
async def test_a_card_retry_cancelled_by_a_center_swap_leaves_a_recoverable_pause():
    """Re-review IMPORTANT 1: a cancelled card Retry must not recreate the jam.

    The card's actions run as a worker on the card. Opening the Console
    terminal swaps the transcript region out, which cancels that worker; mid
    search, the send was left PREPARING with no owner: no card, a shelf
    Discard that released nothing, and every later send refused.
    """
    gateway = _Gateway()
    host = _build_with(gateway)
    library = _StalledLibrary()
    host.app_instance.library_rag_search_service = library
    async with host.run_test(size=(120, 40)) as pilot:
        with eager_tasks():
            console, controller, composer = await _timed_out_send(host, pilot, library)
            controller._library_preparation_timeout = 60.0
            console.query_one(_CARD_ACTIONS["Retry Library search"], Button).press()
            await until(lambda: library.calls == 2)

            await console._swap_console_center()  # what opening the terminal does

            await until(lambda: _paused_retrieval(controller))
            await until(lambda: _shelf_offers_unsent_turn(console))
            # The fresh card shows the paused send again, with its reason.
            await until(
                lambda: console.query_one(_CARD_ACTIONS["Cancel send"], Button).display
            )
            await until(lambda: "Library search stopped" in _screen_text(host))
            console.query_one("#console-prompt-queue-pause", Button).press()
            await until(lambda: not _shelf_offers_unsent_turn(console))

            assert not _paused_retrieval(controller)

            library.release.set()
            composer.load_draft(SECOND)
            composer.focus()  # the swap moved focus to the transcript
            await pilot.pause()
            press(host, "enter", "\r")
            await until(lambda: gateway.stream_calls == 1)
            await until(lambda: REPLY in _screen_text(host))
            screen = _screen_text(host)
            assert "Another send is still preparing" not in screen
            assert "Last send is blocked" not in screen


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action", ["Retry Library search", "Send once without Library"]
)
async def test_a_card_action_interrupted_after_its_send_went_out_fails_that_reply(
    action,
):
    """Pre-merge review (blocking): what the User Guide says about this window.

    Closing the card pauses the send only while Retry is still searching.
    Once the message has gone out and its reply is streaming, the swap
    cancels the card's worker and with it the stream: that reply ends as
    failed, the message stays in the conversation, nothing is listed again,
    and the next send goes through. The guide claimed a pause here.
    """
    gateway = _HeldGateway()
    host = _build_with(gateway)
    library = _StalledLibrary()
    host.app_instance.library_rag_search_service = library
    async with host.run_test(size=(120, 40)) as pilot:
        with eager_tasks():
            console, controller, composer = await _timed_out_send(host, pilot, library)
            store = controller.store
            session_id = store.active_session_id
            runtime = console._console_runtime()
            controller._library_preparation_timeout = 60.0
            library.release.set()
            console.query_one(_CARD_ACTIONS[action], Button).press()
            await until(lambda: gateway.stream_calls == 1)

            await console._swap_console_center()  # what opening the terminal does

            await until(
                lambda: (
                    controller.run_state_for(session_id).status
                    is ConsoleRunStatus.FAILED
                )
            )
            assert store.preparation_for_session(session_id) is None or (
                store.preparation_for_session(session_id).state
                is not ConsoleTurnPreparationState.PAUSED
            )
            await pilot.pause()
            assert runtime.recoveries_for_session(session_id) == ()
            assert not _shelf_offers_unsent_turn(console)
            rows = store.messages_for_session(session_id)
            assert [
                row.content for row in rows if row.role is ConsoleMessageRole.USER
            ] == [FIRST]
            assert rows[-1].role is ConsoleMessageRole.ASSISTANT
            assert REPLY not in rows[-1].content

            gateway.release.set()
            composer.load_draft(SECOND)
            composer.focus()  # the swap moved focus to the transcript
            await pilot.pause()
            press(host, "enter", "\r")
            await until(lambda: gateway.stream_calls == 2)
            await until(lambda: REPLY in _screen_text(host))
            screen = _screen_text(host)
            assert "Another send is still preparing" not in screen
            assert "Last send is blocked" not in screen


@pytest.mark.asyncio
async def test_a_recovery_sync_during_a_center_swap_never_fails_the_sync():
    """A sync that lands while the center swap removes the card's region.

    Textual prunes children before their parent, so for a moment the card is
    still in the DOM while its title is already gone. A Console sync then
    (the interrupted card action's own re-sync lands exactly then, and the
    0.2 s tick can) raised NoMatches('#console-trace-title') in the
    console-sync worker, and a failed worker exits the app. Found under load
    on the head rebased onto dev 0fef489ece.
    """
    gateway = _Gateway()
    host = _build_with(gateway)
    library = _StalledLibrary()
    host.app_instance.library_rag_search_service = library
    async with host.run_test(size=(120, 40)) as pilot:
        with eager_tasks():
            console, controller, composer = await _timed_out_send(host, pilot, library)
            region = console.query_one("#console-main-column")
            card = region.query_one(TraceCallRecoveryCallout)
            title = card.query_one("#console-trace-title")

            swap = asyncio.ensure_future(console._swap_console_center())
            for _ in range(10_000):
                if title.parent is None:
                    break
                await asyncio.sleep(0)
            # The window: the card's title has left the DOM, the card has not.
            assert title.parent is None
            assert card.parent is not None

            region.sync_recovery()  # raised NoMatches before the fix

            await swap
            await until(lambda: _paused_retrieval(controller))
            await until(
                lambda: console.query_one(_CARD_ACTIONS["Cancel send"], Button).display
            )
            library.release.set()


@pytest.mark.asyncio
async def test_a_card_action_whose_first_repaint_fails_gives_the_turn_back():
    """Re-review MINOR 6: the shelf hold is released however the action ends.

    The card takes the turn off the shelf and repaints before its action.
    That hold and repaint ran outside the action's error handling, so a
    repaint that failed left the turn held for good: off the shelf, with
    Restore and Discard refused, and the action never run.
    """
    gateway = _Gateway()
    host = _build_with(gateway)
    library = _StalledLibrary()
    host.app_instance.library_rag_search_service = library
    async with host.run_test(size=(120, 40)) as pilot:
        with eager_tasks():
            console, controller, composer = await _timed_out_send(host, pilot, library)
            runtime = console._console_runtime()
            session_id = controller.store.active_session_id
            (entry,) = runtime.recoveries_for_session(session_id)
            real_sync = console._sync_native_console_chat_ui
            failed: list[str] = []

            async def first_held_repaint_fails():
                if not failed and runtime.recoveries_for_session(session_id) == ():
                    failed.append(entry.preparation_id)
                    raise RuntimeError("repaint failed")
                await real_sync()

            console._sync_native_console_chat_ui = first_held_repaint_fails
            console.query_one(_CARD_ACTIONS["Retry Library search"], Button).press()
            await until(lambda: failed == [entry.preparation_id])
            # The runtime, not the painted shelf (still painted from before).
            await until(lambda: runtime.recoveries_for_session(session_id) == (entry,))
            await until(lambda: "Recovery did not complete" in _screen_text(host))
            assert _shelf_offers_unsent_turn(console)
            assert _paused_retrieval(controller)
            assert library.calls == 1  # the retry itself never ran

            console.query_one("#console-prompt-queue-pause", Button).press()
            await until(lambda: not _shelf_offers_unsent_turn(console))
            assert not _paused_retrieval(controller)


@pytest.mark.asyncio
async def test_card_cancel_after_a_tab_switch_returns_the_text_to_its_own_chat():
    """Re-review MINOR 7: Cancel never fills another conversation's composer.

    The user switches tab while the card's Cancel runs. The cancelled
    message goes back to its own conversation's draft; the composer, which
    now shows the other conversation, is left as it was.
    """
    gateway = _Gateway()
    host = _build_with(gateway)
    library = _StalledLibrary()
    host.app_instance.library_rag_search_service = library
    async with host.run_test(size=(120, 40)) as pilot:
        with eager_tasks():
            console, controller, composer = await _timed_out_send(host, pilot, library)
            store = controller.store
            first_id = store.active_session_id
            runtime = console._console_runtime()
            real_cancel = controller.cancel_library_preparation
            other_ids: list[str] = []

            def cancel_after_a_tab_switch(preparation_id):
                other = store.create_session(title="Other")
                other_ids.append(other.id)
                # The real switch (#3065): saves the composer to its own chat's
                # draft and shows the other chat's (empty) draft at once.
                console._session.switch_and_bind(other.id)
                return real_cancel(preparation_id)

            controller.cancel_library_preparation = cancel_after_a_tab_switch
            console.query_one(_CARD_ACTIONS["Cancel send"], Button).press()
            await until(lambda: other_ids != [])
            await until(lambda: runtime.recoveries_for_session(first_id) == ())
            await pilot.pause()

            assert composer.draft_text() == ""
            assert store.session_draft(other_ids[0]) == ""
            assert store.session_draft(first_id) == FIRST
            assert gateway.stream_calls == 0


@pytest.mark.asyncio
async def test_card_cancel_after_a_tab_switch_keeps_a_follow_up_typed_meanwhile():
    """A Cancel that finishes after a tab switch never overwrites a newer draft.

    The user typed a follow-up while the card was up, pressed Cancel, and the
    tab switched before the action finished. The follow-up stays as that
    conversation's draft (as it does when its composer is still shown and not
    empty), and the cancelled message stays on the shelf to Restore.
    """
    gateway = _Gateway()
    host = _build_with(gateway)
    library = _StalledLibrary()
    host.app_instance.library_rag_search_service = library
    async with host.run_test(size=(120, 40)) as pilot:
        with eager_tasks():
            console, controller, composer = await _timed_out_send(host, pilot, library)
            store = controller.store
            first_id = store.active_session_id
            runtime = console._console_runtime()
            (entry,) = runtime.recoveries_for_session(first_id)
            composer.load_draft(SECOND)  # typed while the card is up
            await pilot.pause()
            await until(lambda: store.session_draft(first_id) == SECOND)
            real_cancel = controller.cancel_library_preparation
            other_ids: list[str] = []

            def cancel_after_a_tab_switch(preparation_id):
                other = store.create_session(title="Other")
                other_ids.append(other.id)
                # The real switch (#3065): saves the composer to its own chat's
                # draft and shows the other chat's (empty) draft at once.
                console._session.switch_and_bind(other.id)
                return real_cancel(preparation_id)

            controller.cancel_library_preparation = cancel_after_a_tab_switch
            console.query_one(_CARD_ACTIONS["Cancel send"], Button).press()
            await until(lambda: other_ids != [])
            await until(
                lambda: (
                    store.preparation_by_id(entry.preparation_id) is None
                    or store.preparation_by_id(entry.preparation_id).state
                    is ConsoleTurnPreparationState.CANCELLED
                )
            )
            # The card's action settles, one way or the other.
            await until(
                lambda: (
                    store.session_draft(first_id) != SECOND
                    or runtime.recoveries_for_session(first_id) == (entry,)
                )
            )
            await pilot.pause()

            assert store.session_draft(first_id) == SECOND  # not overwritten
            # Offered again to Restore, rather than dropped.
            assert runtime.recoveries_for_session(first_id) == (entry,)
            assert store.session_draft(other_ids[0]) == ""
            assert composer.draft_text() == ""
            assert gateway.stream_calls == 0


@pytest.mark.asyncio
async def test_a_card_retry_whose_durable_commit_fails_keeps_the_turn_on_the_shelf():
    """Re-review MINOR 8: the shelf trusts the store, which says "not sent".

    A Retry whose durable commit then fails returns ``accepted=False`` and
    leaves its preparation COMMITTING, so the turn stays restorable on the
    shelf rather than being forgotten as sent.
    """
    gateway = _Gateway()
    host = _build_with(gateway)
    library = _StalledLibrary()
    host.app_instance.library_rag_search_service = library
    async with host.run_test(size=(120, 40)) as pilot:
        with eager_tasks():
            console, controller, composer = await _timed_out_send(host, pilot, library)
            store = controller.store
            session_id = store.active_session_id
            runtime = console._console_runtime()

            def refuse_commit(*_args, **_kwargs):
                raise RuntimeError("durable commit refused for this test")

            store.commit_durable_turn = refuse_commit
            library.release.set()
            console.query_one(_CARD_ACTIONS["Retry Library search"], Button).press()
            await until(
                lambda: (
                    getattr(store.preparation_for_session(session_id), "state", None)
                    is ConsoleTurnPreparationState.COMMITTING
                )
            )
            # The card's action has settled (no longer held off the shelf).
            await until(lambda: len(runtime.recoveries_for_session(session_id)) == 1)
            await until(lambda: _shelf_offers_unsent_turn(console))
            assert gateway.stream_calls == 0
