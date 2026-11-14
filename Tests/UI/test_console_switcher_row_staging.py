"""Yielding row construction must not expose partial or obsolete choices."""

from __future__ import annotations

import asyncio

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Button, Input

from Tests.UI.test_console_character_switcher import (
    _character_row,
    _CharacterSwitcherApp,
)
from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    CharacterConversationPage,
)
from tldw_chatbook.Chat.console_conversation_activation import (
    ConsoleActivationResultKind,
    ConsoleConversationActivationResult,
)
from tldw_chatbook.Chat.console_switcher_state import SwitcherMode

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["finish", "query", "cancel", "authority"])
async def test_row_construction_yields_before_publishing_current_page(
    monkeypatch, transition
):
    rows = tuple(
        _character_row(f"chat-{index}", f"Chat {index}", "2026-09-03T12:00:00Z")
        for index in range(50)
    )
    expanded = False

    async def loader(*, query, **_kwargs):
        selected = rows[-1:] if query else rows if expanded else rows[:1]
        return CharacterConversationPage(selected, len(selected), None, 7)

    app = _CharacterSwitcherApp(
        character_loader=loader, initial_mode=SwitcherMode.CHARACTER_CHATS
    )
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        screen = app.screen
        old_row = screen.query_one(".console-switcher-result", Button)
        old_row.focus()
        await pilot.pause()
        assert app.focused is old_row
        original_generation = screen._committed_result_generation
        built = []
        observed = []
        cancellation = []
        original_init = Button.__init__

        def observe_and_interrupt():
            observed.append((len(built), screen._query_pending))
            if transition == "query":
                screen.query_one("#console-switcher-query", Input).value = "latest"
            elif transition == "cancel":
                cancellation.append(
                    asyncio.create_task(screen.request_safe_cancel(source="escape"))
                )
            elif transition == "authority":
                screen._authority_snapshot = lambda: ("other-profile", "runtime-a", 0)

        def construct(button, *args, **kwargs):
            original_init(button, *args, **kwargs)
            if str(kwargs.get("id", "")).startswith("console-switcher-result-"):
                built.append(button)
                if len(built) == 1:
                    asyncio.get_running_loop().call_soon(observe_and_interrupt)

        monkeypatch.setattr(Button, "__init__", construct)
        expanded = True
        committed = await screen._refresh_results("")
        if cancellation:
            await asyncio.gather(*cancellation)
        if transition == "query":
            async with asyncio.timeout(2):
                while screen._query_pending:
                    await asyncio.sleep(0.01)
        await pilot.pause()

        assert observed and 0 < observed[0][0] < 50
        assert observed[0][1] is True
        if transition == "finish":
            assert committed is True
            assert len(screen._result_buttons()) == 50
            assert not any(button.disabled for button in screen._result_buttons())
            assert screen._query_pending is False
            assert (
                screen._payload_by_widget_id[app.focused.id].target.conversation_id
                == "chat-0"
            )
        elif transition == "query":
            assert committed is False
            assert screen._rendered_query == "latest"
            assert [entry.title for entry in screen._entries] == ["Chat 49"]
            assert screen._committed_result_generation == original_generation + 1
        elif transition == "cancel":
            assert committed is False
            assert app.result is None
            assert screen._closed is True
            assert screen._committed_result_generation == original_generation
        else:
            assert committed is False
            assert screen._query_pending is True
            assert screen._committed_result_generation == original_generation


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transition", ["finish", "query", "reconcile", "cancel", "authority"]
)
async def test_mounting_page_stays_non_actionable_until_complete(
    monkeypatch, transition
):
    attempts = []
    rows = tuple(
        _character_row(f"chat-{index}", f"Chat {index}", "2026-09-03T12:00:00Z")
        for index in range(50)
    )

    async def loader(*, query, **_kwargs):
        selected = rows[-1:] if query else rows
        return CharacterConversationPage(selected, len(selected), None, 7)

    async def activate(request, _cancellation):
        attempts.append(request)
        return ConsoleConversationActivationResult(
            ConsoleActivationResultKind.FAILED, request.target, False
        )

    app = _CharacterSwitcherApp(
        character_loader=loader,
        character_activate=activate,
        initial_mode=SwitcherMode.CHARACTER_CHATS,
    )
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        screen = app.screen
        original_generation = screen._committed_result_generation
        original_mount = VerticalScroll.mount_all
        mounted = asyncio.Event()
        release = asyncio.Event()
        following = None

        async def mount_and_hold(container, widgets, **kwargs):
            await original_mount(container, widgets, **kwargs)
            if not mounted.is_set():
                mounted.set()
                await release.wait()

        monkeypatch.setattr(VerticalScroll, "mount_all", mount_and_hold)
        refresh = asyncio.create_task(screen._refresh_results(""))
        try:
            await asyncio.wait_for(mounted.wait(), 5)
            assert screen._query_pending is True
            assert screen._committed_result_generation == original_generation
            assert all(button.disabled for button in screen._result_buttons())
            assert (
                screen.query_one(
                    "#console-switcher-results", VerticalScroll
                ).styles.visibility
                == "hidden"
            )
            row = screen._result_buttons()[0]
            row.post_message(Button.Pressed(row))
            await pilot.pause()
            assert attempts == []
            if transition == "query":
                screen.query_one("#console-switcher-query", Input).value = "latest"
            elif transition == "reconcile":
                following = asyncio.create_task(screen._refresh_results(""))
                await asyncio.sleep(0)
            elif transition == "cancel":
                await screen.request_safe_cancel(source="escape")
            elif transition == "authority":
                screen._authority_snapshot = lambda: ("other-profile", "runtime-a", 0)
        finally:
            release.set()
            committed = await refresh
            if following is not None:
                assert await following is True
        if transition == "query":
            async with asyncio.timeout(2):
                while screen._query_pending:
                    await asyncio.sleep(0.01)
        await pilot.pause()
        assert committed is (transition == "finish")
        if transition == "cancel":
            assert app.result is None
            assert screen._committed_result_generation == original_generation
        elif transition == "authority":
            assert screen._query_pending is True
            assert screen._committed_result_generation == original_generation
        else:
            assert screen._query_pending is False
            assert screen._committed_result_generation == original_generation + 1
            assert len(screen._committed_payload_by_widget_id) == (
                1 if transition == "query" else 50
            )
