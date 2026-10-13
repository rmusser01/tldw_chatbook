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
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsoleTurnPreparationState,
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
            assert not _shelf_offers_unsent_turn(console)
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

            assert not _shelf_offers_unsent_turn(console)
            assert runtime.recoveries_for_session(session_id) == ()
            assert runtime.discard_turn_recovery(entry.turn_id) is False
            with pytest.raises(RuntimeError):
                runtime.restore_turn_recovery(entry.turn_id)
            assert composer.draft_text() == ""

            library.release.set()
            gateway.release.set()
            await until(lambda: REPLY in _screen_text(host))
            await until(lambda: runtime.recoveries_for_session(session_id) == ())
            assert not _shelf_offers_unsent_turn(console)
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
            assert not _shelf_offers_unsent_turn(console)
            assert console._console_runtime().recoveries_for_session(session_id) == ()
            assert gateway.stream_calls == 0
