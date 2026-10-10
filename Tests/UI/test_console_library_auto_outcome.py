"""Automatic Library preparation on the real Console send path (TASK-33621.20).

Live on dev 64579cce2c (2026-09-29): the first Automatic send after launch
hit the 5 s ``library_preparation_timeout`` while the RAG service was still
cold-initialising, and paused the turn. The shelf offered only Restore and
Discard, and neither released the paused preparation: every later send in
that conversation was refused ("Last send is blocked; resolve it first.").

These tests drive the real ChatScreen, controller, store and runtime. Only
the provider and the Library search service are doubles.
"""

from __future__ import annotations

import asyncio

import pytest

from Tests.UI.test_console_native_chat_flow import (
    _ReadyResolutionGateway,
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
from Tests.UI.test_console_send_admission_off_pump import _allow_second_turns
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicyCandidate,
)
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsoleTurnPreparationState,
)
from textual.widgets import Button

from tldw_chatbook.Widgets.Console import ConsoleComposerBar

pytestmark = pytest.mark.bootstrap_profile

FIRST = "first automatic send"
SECOND = "second automatic send"
REPLY = "library reply"


class _Gateway(_ReadyResolutionGateway):
    """Ready provider that records each streamed request."""

    def __init__(self) -> None:
        self.stream_calls = 0

    async def resolve_context_window(self, settings):
        return self.cached_context_window(settings)

    async def stream_chat(self, resolution, messages, **kwargs):
        self.stream_calls += 1
        yield REPLY


class _StalledLibrary:
    """A Library search that never answers until it is released."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.calls = 0

    async def search(self, query, source_types, mode, **kwargs):
        self.calls += 1
        await self.release.wait()
        return {"results": []}


def _build():
    app = _build_console_send_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    host = ConsoleHarness(app)
    gateway = _Gateway()
    app.console_provider_gateway_factory = lambda: gateway
    _allow_second_turns(host)
    return host, gateway


async def _automatic_console(host, pilot):
    console = host.screen_stack[-1]
    await _wait_for_selector(console, pilot, "#console-native-composer")
    _select_llamacpp_console(console)
    await pilot.pause(0.3)
    controller = console._ensure_console_chat_controller()
    store = controller.store
    store.stage_session_library_policy(
        store.active_session_id,
        ConsoleLibraryPolicyCandidate(
            auto_retrieve=ConsoleAutoRetrieve.AUTOMATIC,
            assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
        ),
    )
    composer = console.query_one("#console-native-composer", ConsoleComposerBar)
    composer.focus()
    return console, controller, composer


def _paused_retrieval(controller) -> bool:
    preparation = controller.store.preparation_for_session(
        controller.store.active_session_id
    )
    return (
        preparation is not None
        and preparation.state is ConsoleTurnPreparationState.PAUSED
        and preparation.pause_kind is ConsolePreparationPauseKind.RETRIEVAL
    )


def _shelf_offers_unsent_turn(console) -> bool:
    shelf = console.query_one("#console-prompt-queue-pause")
    return shelf.display and str(shelf.label) == "Discard"


@pytest.mark.asyncio
@pytest.mark.parametrize("shelf_action", ["Discard", "Restore"])
async def test_shelf_action_on_a_timed_out_preparation_lets_the_next_send_through(
    shelf_action,
):
    """AC#3/#5: Discard or Restore releases the paused preparation."""
    host, gateway = _build()
    library = _StalledLibrary()
    host.app_instance.library_rag_search_service = library
    async with host.run_test(size=(120, 40)) as pilot:
        with eager_tasks():
            console, controller, composer = await _automatic_console(host, pilot)
            controller._library_preparation_timeout = 0.2
            composer.load_draft(FIRST)
            await pilot.pause()
            press(host, "enter", "\r")
            await until(lambda: _paused_retrieval(controller))
            await until(lambda: _shelf_offers_unsent_turn(console))
            assert gateway.stream_calls == 0

            button_id = (
                "#console-prompt-queue-pause"
                if shelf_action == "Discard"
                else "#console-prompt-queue-manage"
            )
            assert str(console.query_one(button_id).label) == shelf_action
            console.query_one(button_id, Button).press()
            await until(lambda: not _shelf_offers_unsent_turn(console))

            library.release.set()
            composer.load_draft(SECOND)
            await pilot.pause()
            press(host, "enter", "\r")
            await until(lambda: gateway.stream_calls == 1)
            await until(lambda: REPLY in "\n".join(_painted_lines(host)))
            screen_text = "\n".join(_painted_lines(host))
            assert SECOND in screen_text
            assert "Last send is blocked" not in screen_text
            assert controller.store.preparation_for_session(
                controller.store.active_session_id
            ) is None or not _paused_retrieval(controller)


@pytest.mark.asyncio
async def test_a_slow_library_search_is_shown_and_stop_pauses_the_send():
    """AC#4: a long Library wait says so, offers Stop, and Stop pauses it."""
    host, gateway = _build()
    library = _StalledLibrary()
    host.app_instance.library_rag_search_service = library
    async with host.run_test(size=(120, 40)) as pilot:
        with eager_tasks():
            console, controller, composer = await _automatic_console(host, pilot)
            # Long enough that only Stop, never the budget, can end the wait.
            controller._library_preparation_timeout = 120.0
            composer.load_draft(FIRST)
            await pilot.pause()
            press(host, "enter", "\r")
            await until(lambda: library.calls == 1)
            await until(
                lambda: "Run: Searching Library…" in "\n".join(_painted_lines(host))
            )
            stop = console.query_one("#console-stop-generation", Button)
            await until(lambda: stop.styles.display == "block" and not stop.disabled)
            stop.press()
            await until(lambda: _paused_retrieval(controller))
            await until(lambda: _shelf_offers_unsent_turn(console))
            assert gateway.stream_calls == 0
            assert controller._preparation_outcomes[
                controller.store.preparation_for_session(
                    controller.store.active_session_id
                ).preparation_id
            ].error_code == "library_retrieval_stopped"
            library.release.set()
