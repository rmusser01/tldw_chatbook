"""A wrapped collaborator must not change the Console received Send route (TASK-41643).

ADR-356 rule 4: authority comes from runtime-owned registration and current
state, not from function or code identity. The received route used to compare
the screen's selectors, the session's turn-context builder, the scratch
provider and the RAG readers with their definition-time code, so a decorator
or a test patch around any of them sent every Send down the ordinary dispatch.
Each wrapped case wraps one collaborator with ``functools.wraps`` and drives
the real route; ``stock`` is the control. Each foreign case puts a callback on
one object (or replaces a registered slot), which still declines.
"""

import functools
from functools import partial
from types import MethodType

import pytest

from Tests.UI.test_console_received_intent_feedback import _received_console_case

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


def _wrap(monkeypatch, owner, name, calls):
    original = getattr(owner, name)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        calls.append(name)
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, wrapped)


def _receive(case):
    from tldw_chatbook.UI.Console_Modules.wiring import receive_console_visible_intent

    stash = case.composer.capture_draft_for_send()
    captured = case.store.session_input_snapshot(case.session.id)
    return receive_console_visible_intent(
        case.console, case.draft, case.session.id, stash, _captured_inputs=captured
    )


@pytest.mark.parametrize(
    "wrapped",
    ["stock", "screen_selector", "turn_context_builder", "scratch_snapshot", "rag_reader"],
)
async def test_wrapped_received_collaborator_keeps_the_received_route(
    monkeypatch, wrapped
):
    from tldw_chatbook.UI.Console_Modules import session as session_source
    from tldw_chatbook.UI.Console_Modules import wiring
    from tldw_chatbook.UI.Screens import chat_screen

    calls = []
    if wrapped == "scratch_snapshot":
        # The screen's wiring binds this into the session's scratch slot when
        # it builds the controllers, so the decorator must already be there.
        _wrap(monkeypatch, wiring, "_stock_console_scratch_snapshot", calls)
    async with _received_console_case(monkeypatch, f"wrapped-{wrapped}") as case:
        if wrapped == "screen_selector":
            _wrap(monkeypatch, chat_screen.ChatScreen, "_console_send_gate", calls)
        elif wrapped == "turn_context_builder":
            _wrap(
                monkeypatch,
                session_source.ConsoleSessionController,
                "_build_console_turn_execution_context",
                calls,
            )
        elif wrapped == "rag_reader":
            _wrap(monkeypatch, chat_screen, "_console_library_rag_source_scope", calls)
        turn_id = _receive(case)
        assert turn_id, f"{wrapped}: the received route declined"
        record = case.runtime._turn_custody[turn_id]
        assert record.received_intent.inputs.draft == case.draft
        assert record.session_id == case.session.id
        if wrapped == "rag_reader":
            assert calls, "the received route never read the wrapped RAG scope"
        assert case.provider_calls == []


@pytest.mark.parametrize(
    "foreign",
    [
        "instance_selector",
        "instance_turn_context_builder",
        "foreign_scratch_slot",
        "foreign_rag_slot",
    ],
)
async def test_foreign_received_collaborator_is_refused(monkeypatch, foreign):
    async with _received_console_case(monkeypatch, f"foreign-{foreign}") as case:
        screen, owner = case.console, case.console._session
        foreign_calls = []

        def foreign_callback(*args, **kwargs):
            foreign_calls.append(foreign)
            raise AssertionError(f"{foreign}: the foreign callback ran")

        if foreign == "instance_selector":
            monkeypatch.setattr(
                screen, "_console_send_gate", MethodType(foreign_callback, screen)
            )
        elif foreign == "instance_turn_context_builder":
            monkeypatch.setattr(
                owner,
                "_build_console_turn_execution_context",
                MethodType(foreign_callback, owner),
            )
        elif foreign == "foreign_scratch_slot":
            # A stock-shaped partial that is not the one the wiring registered.
            monkeypatch.setattr(
                owner,
                "_scratch_snapshot_provider",
                partial(owner._scratch_snapshot_provider.func, screen),
            )
        else:
            monkeypatch.setattr(owner, "_rag_top_k_accessor", foreign_callback)
        assert _receive(case) is None, f"{foreign}: the received route accepted"
        assert not foreign_calls
        assert not case.runtime._turn_custody
        assert case.provider_calls == []


def test_own_methods_accepts_wrapped_class_methods_but_not_instance_overrides():
    from tldw_chatbook.Chat.console_received_dispatch import own_methods

    class Controller:
        def queue_prompt(self):
            return "stock"

    controller = Controller()
    assert own_methods(controller, ("queue_prompt",))
    original = Controller.queue_prompt
    Controller.queue_prompt = functools.wraps(original)(
        lambda self, *a, **k: original(self, *a, **k)
    )
    assert own_methods(controller, ("queue_prompt",))
    vars(controller)["queue_prompt"] = MethodType(original, controller)
    assert not own_methods(controller, ("queue_prompt",))
    del vars(controller)["queue_prompt"]
    Controller.queue_prompt = staticmethod(lambda: None)
    assert not own_methods(controller, ("queue_prompt",))
    assert not own_methods(controller, ("missing",))
