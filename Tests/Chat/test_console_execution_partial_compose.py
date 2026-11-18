"""Incomplete selected content keeps its error panel and real detach path."""

from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import pytest


def test_partial_selected_compose_keeps_original_panel_until_complete(monkeypatch):
    from textual.widgets import Static

    from tldw_chatbook.UI.Console_Modules import execution_state
    from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    partial = Static("partial")
    failure = RuntimeError("original compose failure")
    state = {"fail": True, "prefix_calls": 0}

    def prefix():
        state["prefix_calls"] += 1
        return None

    def content(*_):
        yield partial
        if state["fail"]:
            raise failure

    screen = SimpleNamespace(
        app_instance=SimpleNamespace(console_execution_client=object()),
        screen_name="Console",
        _console_process_visit=3,
        _console_derivation_scope=nullcontext,
        _consume_pending_console_launch=prefix,
        _build_console_control_state=lambda _: object(),
        _compose_console_content=content,
    )
    screen.compose_content = MethodType(ChatScreen.compose_content, screen)
    monkeypatch.setattr(
        execution_state, "selected_client",
        lambda actual: SimpleNamespace(
            require_view=lambda owner, visit: None,
        ) if actual is screen else pytest.fail("wrong screen"),
    )

    # The original generator still raises the original object, rather than
    # silently making a failed partial tree into a successful compose.
    generator = screen.compose_content()
    assert next(generator) is partial
    with pytest.raises(RuntimeError) as raised:
        next(generator)
    assert raised.value is failure
    assert screen._console_process_content_failed is True

    # The actual BaseAppScreen boundary keeps already-yielded content and
    # its stock error panel. No selected retry duplicates the consumed prefix.
    state["prefix_calls"] = 0
    nodes = list(BaseAppScreen._compose_content_or_failure(screen))
    assert nodes[0] is partial
    assert nodes[1].id == "screen-content-error"
    assert state["prefix_calls"] == 1
    ChatScreen.on_mount(screen)
    ChatScreen.on_screen_resume(screen)
    ChatScreen._start_console_attach_reconciliation(screen)
    execution_state.refresh_selected_when_due(screen)
    assert state["prefix_calls"] == 1

    state["fail"] = False
    generator = screen.compose_content()
    assert next(generator) is partial
    assert screen._console_process_content_failed is True
    with pytest.raises(StopIteration):
        next(generator)
    assert screen._console_process_content_failed is False


@pytest.mark.asyncio
async def test_partial_selected_compose_skips_late_display_but_still_detaches(monkeypatch):
    from tldw_chatbook.UI.Console_Modules import execution_state
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    screen = SimpleNamespace(
        app_instance=SimpleNamespace(console_execution_client=object()),
        _console_process_content_failed=True,
        _console_process_prepare_failed=False,
    )
    monkeypatch.setattr(
        execution_state, "selected_client",
        lambda _: pytest.fail("partial content must not refresh"),
    )
    assert await execution_state.refresh_selected_view(screen) is False
    await execution_state.reconcile_selected_view(screen)
    await ChatScreen._reconcile_console_after_attach(screen)

    detached = []
    stop = RuntimeError("stop after original detach entry")

    def detach(actual):
        detached.append(actual)
        raise stop

    monkeypatch.setattr(execution_state, "detach_selected_view", detach)
    with pytest.raises(RuntimeError) as raised:
        await ChatScreen.on_unmount(screen)
    assert raised.value is stop
    assert detached == [screen]
