"""Character-chat mode contracts for the Console Ctrl+K switcher."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import ClassVar

import pytest
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Button, Input, Static

from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    CharacterConversationPage,
    CharacterConversationRow,
    CharacterKeywordIndexStatus,
    CharacterKeywordSnapshot,
    LocalCharacterConversationTarget,
    ResolvedLocalCharacterKey,
    UnavailableCharacterReason,
    UnresolvedConversationKey,
)
from tldw_chatbook.Chat.console_conversation_activation import (
    CharacterConversationActivationRequest,
    ConsoleActivationResultKind,
    ConsoleConversationActivationResult,
)
from tldw_chatbook.Chat.console_switcher_state import (
    ActivityGroup,
    ConsoleSwitcherCharacterResult,
    ConsoleSwitcherEntry,
    ConsoleSwitcherHistoryPage,
    ConsoleSwitcherTarget,
    SwitcherMode,
    SwitcherTargetKind,
    build_console_character_results,
)
from tldw_chatbook.UI.Workbench.help import WorkbenchHelpPanel
from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (
    SEARCH_DEBOUNCE_SECONDS,
    ConsoleSessionSwitcherModal,
)


def _target(name: str, character_id: int = 7) -> LocalCharacterConversationTarget:
    return LocalCharacterConversationTarget(
        ResolvedLocalCharacterKey("authority-a", character_id), name
    )


def _character_row(
    name: str,
    title: str,
    modified: str,
    *,
    excerpt: str = "",
    is_current: bool = False,
) -> CharacterConversationRow:
    return CharacterConversationRow.resolved(
        _target(name),
        character_label="Ada",
        title=title,
        last_modified=modified,
        created_at="2026-09-01T00:00:00Z",
        is_current=is_current,
        selected_excerpt=excerpt,
    )


def _unavailable_row(
    name: str,
    title: str,
    reason: UnavailableCharacterReason,
    modified: str,
) -> CharacterConversationRow:
    return CharacterConversationRow.unavailable(
        UnresolvedConversationKey("authority-a", name),
        reason=reason,
        character_label="Historical Ada",
        title=title,
        last_modified=modified,
        created_at="2026-09-01T00:00:00Z",
    )


def _active_entry(name: str, title: str) -> ConsoleSwitcherEntry:
    return ConsoleSwitcherEntry(
        row_key=name,
        title=title,
        subtitle="OPEN · now",
        native_session_id=name,
        conversation_id=None,
        scope_type="workspace",
        workspace_id="workspace-a",
        is_active=False,
        openable=True,
        group=ActivityGroup.CURRENT,
    )


def _active_character_entry(
    conversation_id: str,
    title: str,
    *,
    profile_authority: str = "profile-a",
    authority_token: str = "runtime-a",
    is_active: bool = False,
) -> ConsoleSwitcherEntry:
    session_id = f"session-{conversation_id}"
    target = ConsoleSwitcherTarget(
        kind=SwitcherTargetKind.NATIVE_SESSION,
        profile_authority=profile_authority,
        authority_token=authority_token,
        session_id=session_id,
        conversation_id=conversation_id,
        scope_type="workspace",
        workspace_id="workspace-a",
    )
    group = ActivityGroup.CURRENT if is_active else ActivityGroup.OTHER_OPEN
    state = "CURRENT" if is_active else "OPEN AGENT"
    return ConsoleSwitcherEntry(
        row_key=f"session:{profile_authority}:{session_id}",
        title=title,
        subtitle=f"{state} · CONSOLE TAB · Workspace A · now",
        native_session_id=session_id,
        conversation_id=conversation_id,
        scope_type="workspace",
        workspace_id="workspace-a",
        is_active=is_active,
        section=group.value,
        state_label=state,
        target=target,
        group=group,
        activity_state="current" if is_active else "other-open",
    )


class _CharacterSwitcherApp(ConsolidatedCSSApp):
    CSS_PATH: ClassVar[list[str]] = [str(path) for path in APP_STYLESHEETS]

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("f1", "show_workbench_help", "Help", priority=True)
    ]

    def __init__(
        self,
        *,
        active: tuple[ConsoleSwitcherEntry, ...] = (),
        history_loader=None,
        character_loader=None,
        character_activate=None,
        character_commit_waiter=None,
        character_open_library=None,
        initial_mode: SwitcherMode = SwitcherMode.ACTIVE,
        initial_character_query: str = "",
    ) -> None:
        super().__init__()
        self.active = active
        self.history_loader = history_loader
        self.character_loader = character_loader
        self.character_activate = character_activate
        self.character_commit_waiter = character_commit_waiter
        self.character_open_library = character_open_library
        self.initial_mode = initial_mode
        self.initial_character_query = initial_character_query
        self.result = "unset"

    async def on_mount(self) -> None:
        await self.push_screen(
            ConsoleSessionSwitcherModal(
                active_results=self.active,
                history_loader=self.history_loader,
                character_loader=self.character_loader,
                character_activate=self.character_activate,
                character_commit_waiter=self.character_commit_waiter,
                character_open_library=self.character_open_library,
                profile_authority="profile-a",
                authority_token="runtime-a",
                initial_mode=self.initial_mode,
                initial_character_query=self.initial_character_query,
            ),
            callback=lambda result: setattr(self, "result", result),
        )

    async def action_show_workbench_help(self) -> None:
        handler = getattr(self.screen, "action_show_workbench_help", None)
        if handler is not None:
            await handler()


def test_character_results_sort_by_activity_then_stable_identity() -> None:
    rows = (
        _character_row("conversation-b", "B", "2026-09-01T12:00:00+00:00"),
        _character_row("conversation-c", "C", "2026-09-02T12:00:00+00:00"),
        _character_row("conversation-a", "A", "2026-09-01T12:00:00+00:00"),
    )

    results = build_console_character_results(
        rows, now=datetime(2026, 9, 3, 12, tzinfo=UTC)
    )

    assert [result.target.conversation_id for result in results if result.target] == [
        "conversation-c",
        "conversation-a",
        "conversation-b",
    ]
    assert all(isinstance(result, ConsoleSwitcherCharacterResult) for result in results)
    assert results[0].relative_time == "1d"
    assert results[0].absolute_time.startswith("Updated 2026-09-02 ")


@pytest.mark.asyncio
async def test_f3_cycles_active_history_character_and_back() -> None:
    async def history_loader(**_kwargs):
        return ConsoleSwitcherHistoryPage((), 0, 50, 0)

    async def character_loader(**_kwargs):
        return CharacterConversationPage((), 0, None, 3)

    app = _CharacterSwitcherApp(
        history_loader=history_loader, character_loader=character_loader
    )
    async with app.run_test(size=(52, 20)) as pilot:
        modal = app.screen
        assert modal._mode is SwitcherMode.ACTIVE
        await pilot.press("f3")
        await pilot.pause()
        assert modal._mode is SwitcherMode.HISTORY
        await pilot.press("f3")
        await pilot.pause()
        assert modal._mode is SwitcherMode.CHARACTER_CHATS
        await pilot.press("f3")
        await pilot.pause()
        assert modal._mode is SwitcherMode.ACTIVE


@pytest.mark.asyncio
async def test_active_and_history_share_query_but_character_query_is_independent() -> (
    None
):
    async def history_loader(**_kwargs):
        return ConsoleSwitcherHistoryPage((), 0, 50, 0)

    async def character_loader(**_kwargs):
        return CharacterConversationPage((), 0, None, 3)

    app = _CharacterSwitcherApp(
        history_loader=history_loader, character_loader=character_loader
    )
    async with app.run_test(size=(52, 20)) as pilot:
        query = app.screen.query_one("#console-switcher-query", Input)
        query.value = "operations"
        await pilot.press("f3")
        await pilot.pause()
        assert query.value == "operations"

        await pilot.press("f3")
        await pilot.pause()
        assert query.value == ""
        query.value = "Ada"

        await pilot.press("f3")
        await pilot.pause()
        assert query.value == "operations"
        await pilot.press("f3")
        await pilot.pause()
        await pilot.press("f3")
        await pilot.pause()
        assert query.value == "Ada"


@pytest.mark.asyncio
async def test_active_zero_match_widens_but_character_zero_match_never_widens() -> None:
    history_queries: list[str] = []

    async def history_loader(**kwargs):
        history_queries.append(kwargs["query"])
        return ConsoleSwitcherHistoryPage(
            (_active_entry("history-row", "Needle in History"),), 0, 50, 1
        )

    async def character_loader(**_kwargs):
        return CharacterConversationPage((), 0, None, 3)

    app = _CharacterSwitcherApp(
        active=(_active_entry("active-row", "Live work"),),
        history_loader=history_loader,
        character_loader=character_loader,
    )
    async with app.run_test(size=(52, 20)) as pilot:
        query = app.screen.query_one("#console-switcher-query", Input)
        query.value = "Needle"
        await pilot.pause(SEARCH_DEBOUNCE_SECONDS + 0.1)
        assert "Active · showing History matches" in str(
            app.screen.query_one("#console-switcher-scope", Static).renderable
        )
        assert history_queries == ["Needle"]

        app.screen._set_mode(SwitcherMode.CHARACTER_CHATS)
        await pilot.pause()
        query.value = "No character result"
        await pilot.pause(SEARCH_DEBOUNCE_SECONDS + 0.1)
        assert history_queries == ["Needle"]
        assert "No Keyword matches" in str(
            app.screen.query_one("#console-switcher-empty", Static).renderable
        )


@pytest.mark.asyncio
async def test_character_f2_is_a_noop_with_truthful_hint() -> None:
    async def character_loader(**_kwargs):
        return CharacterConversationPage(
            (_character_row("conversation-a", "Ada's plan", "2026-09-02T12:00:00Z"),),
            1,
            None,
            3,
        )

    app = _CharacterSwitcherApp(
        character_loader=character_loader,
        initial_mode=SwitcherMode.CHARACTER_CHATS,
    )
    async with app.run_test(size=(52, 20)) as pilot:
        await pilot.pause()
        row = app.screen.query_one(".console-switcher-result", Button)
        row.focus()
        await pilot.press("f2")
        await pilot.pause()
        assert app.result == "unset"
        assert "F2" not in str(
            app.screen.query_one("#console-switcher-hints", Static).renderable
        )
        assert (
            "cannot be renamed"
            in str(
                app.screen.query_one("#console-switcher-status", Static).renderable
            ).lower()
        )


@pytest.mark.asyncio
async def test_character_loader_receives_validated_query_and_fifty_row_page() -> None:
    calls: list[dict[str, object]] = []

    async def character_loader(**kwargs):
        calls.append(kwargs)
        return CharacterConversationPage((), 0, None, 3)

    app = _CharacterSwitcherApp(
        character_loader=character_loader,
        initial_mode=SwitcherMode.CHARACTER_CHATS,
        initial_character_query="needle",
    )
    async with app.run_test(size=(52, 20)) as pilot:
        await pilot.pause()
        assert calls == [{"query": "needle", "offset": 0, "limit": 50}]
        assert await app.screen._refresh_results("x" * 513) is False
        assert calls == [{"query": "needle", "offset": 0, "limit": 50}]


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["x" * 201, " " * 201, "bad\x00query"])
async def test_character_invalid_query_is_visible_and_recovers_without_loader(invalid):
    queries = []

    async def loader(**kwargs):
        queries.append(kwargs["query"])
        return CharacterConversationPage((), 0, None, 7)

    app = _CharacterSwitcherApp(
        character_loader=loader, initial_mode=SwitcherMode.CHARACTER_CHATS
    )
    async with app.run_test(size=(52, 20)) as pilot:
        await pilot.pause()
        query = app.screen.query_one("#console-switcher-query", Input)
        query.value = invalid
        await pilot.pause(SEARCH_DEBOUNCE_SECONDS + 0.05)
        assert queries == [""]
        detail = app.screen.query_one("#console-switcher-selected-detail", Static)
        assert "200" in str(detail.renderable)
        assert "Keyword search unavailable" not in str(detail.renderable)
        query.value = "界" * 200
        await pilot.pause(SEARCH_DEBOUNCE_SECONDS + 0.05)
        assert queries == ["", "界" * 200]


@pytest.mark.asyncio
async def test_character_keyword_failure_is_not_reported_as_zero_matches() -> None:
    async def character_loader(**_kwargs):
        return CharacterConversationPage(
            (), 0, None, 3, CharacterKeywordIndexStatus.FAILED
        )

    app = _CharacterSwitcherApp(
        character_loader=character_loader,
        initial_mode=SwitcherMode.CHARACTER_CHATS,
        initial_character_query="needle",
    )
    async with app.run_test(size=(52, 20)) as pilot:
        await pilot.pause()
        assert "Keyword search unavailable" in str(
            app.screen.query_one("#console-switcher-selected-detail", Static).renderable
        )
        recovery = app.screen.query_one("#console-switcher-recovery", Button)
        assert recovery.display is True
        assert str(recovery.label) == "Refresh results"


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(52, 20), (120, 50)])
async def test_character_activation_freezes_exact_target_and_closes_only_on_opened(
    size: tuple[int, int],
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    attempts: list[CharacterConversationActivationRequest] = []

    async def character_loader(**_kwargs):
        return CharacterConversationPage(
            (
                _character_row(
                    "conversation-a",
                    "Ada's exact plan",
                    "2026-09-02T12:00:00Z",
                    excerpt="selected excerpt only",
                ),
            ),
            1,
            None,
            9,
            CharacterKeywordIndexStatus.READY,
            CharacterKeywordSnapshot("prior-corpus", 1, 3, "2026-09-01T00:00:00Z"),
        )

    async def activate(request, _cancellation):
        attempts.append(request)
        entered.set()
        await release.wait()
        return ConsoleConversationActivationResult(
            ConsoleActivationResultKind.OPENED, request.target, True
        )

    async def wait_for_commit(_request):
        await entered.wait()

    app = _CharacterSwitcherApp(
        character_loader=character_loader,
        character_activate=activate,
        character_commit_waiter=wait_for_commit,
        initial_mode=SwitcherMode.CHARACTER_CHATS,
    )
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        app.screen.query_one(".console-switcher-result", Button).focus()
        await pilot.press("enter", "enter")
        await entered.wait()
        assert app.screen._committed_character_result is not None
        assert app.screen._committed_character_result.target == _target(
            "conversation-a"
        )
        assert "Finishing" in str(
            app.screen.query_one("#console-switcher-status", Static).renderable
        )
        await pilot.press("escape")
        assert app.screen is not app
        release.set()
        await pilot.pause()

    assert len(attempts) == 1
    assert attempts[0].data_revision == 9
    assert app.result is None


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(52, 20), (120, 50)])
async def test_character_pointer_press_keeps_immutable_target_across_repaint(
    size: tuple[int, int],
) -> None:
    first = _character_row("conversation-a", "First", "2026-09-03T12:00:00Z")
    second = _character_row("conversation-b", "Second", "2026-09-02T12:00:00Z")
    page_rows = [first, second]
    attempts: list[CharacterConversationActivationRequest] = []

    async def character_loader(**_kwargs):
        return CharacterConversationPage(tuple(page_rows), 2, None, 4)

    async def activate(request, _cancellation):
        attempts.append(request)
        return ConsoleConversationActivationResult(
            ConsoleActivationResultKind.FAILED, request.target, False
        )

    app = _CharacterSwitcherApp(
        character_loader=character_loader,
        character_activate=activate,
        initial_mode=SwitcherMode.CHARACTER_CHATS,
    )
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await pilot.mouse_down(".console-switcher-result")
        page_rows[:] = [
            _character_row("conversation-b", "Second", "2026-09-04T12:00:00Z"),
            _character_row("conversation-a", "First", "2026-09-01T12:00:00Z"),
        ]
        await app.screen._refresh_results("")
        await pilot.mouse_up(".console-switcher-result")
        await pilot.pause()

        assert attempts[0].target == _target("conversation-a")
        assert app.screen._committed_character_result is not None
        assert app.screen._committed_character_result.title == "First"


@pytest.mark.asyncio
async def test_character_pointer_release_cancels_when_pressed_target_disappears() -> (
    None
):
    page_rows = [
        _character_row("conversation-a", "First", "2026-09-03T12:00:00Z"),
        _character_row("conversation-b", "Second", "2026-09-02T12:00:00Z"),
    ]
    attempts: list[CharacterConversationActivationRequest] = []

    async def character_loader(**_kwargs):
        return CharacterConversationPage(tuple(page_rows), len(page_rows), None, 5)

    async def activate(request, _cancellation):
        attempts.append(request)
        return ConsoleConversationActivationResult(
            ConsoleActivationResultKind.FAILED, request.target, False
        )

    app = _CharacterSwitcherApp(
        character_loader=character_loader,
        character_activate=activate,
        initial_mode=SwitcherMode.CHARACTER_CHATS,
    )
    async with app.run_test(size=(52, 20)) as pilot:
        await pilot.pause()
        await pilot.mouse_down(".console-switcher-result")
        page_rows.pop(0)
        await app.screen._refresh_results("")
        await pilot.mouse_up(".console-switcher-result")
        await pilot.pause()

        assert attempts == []
        assert app.screen._committed_character_result is None


@pytest.mark.asyncio
async def test_escape_cancels_character_open_only_before_commit() -> None:
    entered = asyncio.Event()

    async def character_loader(**_kwargs):
        return CharacterConversationPage(
            (_character_row("conversation-a", "First", "2026-09-03T12:00:00Z"),),
            1,
            None,
            4,
        )

    async def activate(request, cancellation):
        entered.set()
        await cancellation.wait()
        return ConsoleConversationActivationResult(
            ConsoleActivationResultKind.CANCELLED_PRECOMMIT,
            request.target,
            False,
        )

    app = _CharacterSwitcherApp(
        character_loader=character_loader,
        character_activate=activate,
        initial_mode=SwitcherMode.CHARACTER_CHATS,
    )
    async with app.run_test(size=(52, 20)) as pilot:
        await pilot.pause()
        await pilot.press("down", "enter")
        await entered.wait()
        await pilot.press("escape")
        await pilot.pause()

        assert isinstance(app.screen, ConsoleSessionSwitcherModal)
        assert app.screen._activation_phase.value == "idle"
        assert app.result == "unset"


@pytest.mark.parametrize(
    ("kind", "copy", "action"),
    [
        (
            ConsoleActivationResultKind.NOT_FOUND,
            "Conversation no longer exists",
            "Refresh results",
        ),
        (
            ConsoleActivationResultKind.DATA_PROFILE_CHANGED,
            "Profile changed",
            "Refresh results",
        ),
        (
            ConsoleActivationResultKind.CHARACTER_UNAVAILABLE,
            "Character unavailable",
            "Open Library",
        ),
    ],
)
@pytest.mark.asyncio
async def test_character_activation_typed_failure_keeps_modal_and_prior_state(
    kind: ConsoleActivationResultKind, copy: str, action: str
) -> None:
    async def character_loader(**_kwargs):
        return CharacterConversationPage(
            (_character_row("conversation-a", "First", "2026-09-03T12:00:00Z"),),
            1,
            None,
            6,
        )

    async def activate(request, _cancellation):
        return ConsoleConversationActivationResult(kind, request.target, False)

    app = _CharacterSwitcherApp(
        character_loader=character_loader,
        character_activate=activate,
        initial_mode=SwitcherMode.CHARACTER_CHATS,
    )
    async with app.run_test(size=(52, 20)) as pilot:
        await pilot.pause()
        await pilot.press("down", "enter")
        await pilot.pause()

        assert isinstance(app.screen, ConsoleSessionSwitcherModal)
        assert app.result == "unset"
        assert copy in str(
            app.screen.query_one("#console-switcher-status", Static).renderable
        )
        recovery = app.screen.query_one("#console-switcher-recovery", Button)
        assert recovery.display
        assert str(recovery.label) == action
        divider = app.screen.query_one("#console-switcher-divider", Static)
        assert divider.display
        assert copy in str(divider.renderable)
        detail = app.screen.query_one("#console-switcher-selected-detail", Static)
        assert copy in str(detail.renderable)


@pytest.mark.asyncio
async def test_opened_for_a_different_target_does_not_close_character_switcher() -> (
    None
):
    async def character_loader(**_kwargs):
        return CharacterConversationPage(
            (_character_row("conversation-a", "First", "2026-09-03T12:00:00Z"),),
            1,
            None,
            6,
        )

    async def activate(_request, _cancellation):
        other = _target("conversation-b")
        return ConsoleConversationActivationResult(
            ConsoleActivationResultKind.OPENED, other, True
        )

    app = _CharacterSwitcherApp(
        character_loader=character_loader,
        character_activate=activate,
        initial_mode=SwitcherMode.CHARACTER_CHATS,
    )
    async with app.run_test(size=(52, 20)) as pilot:
        await pilot.pause()
        await pilot.press("down", "enter")
        await pilot.pause()

        assert isinstance(app.screen, ConsoleSessionSwitcherModal)
        assert app.result == "unset"
        assert "Could not open chat" in str(
            app.screen.query_one("#console-switcher-status", Static).renderable
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("gesture", ["click", "focused-enter"])
async def test_pending_character_query_cannot_activate_a_stale_row(
    gesture: str,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    attempts: list[CharacterConversationActivationRequest] = []

    async def character_loader(**kwargs):
        if kwargs["query"]:
            entered.set()
            await release.wait()
            return CharacterConversationPage((), 0, None, 7)
        return CharacterConversationPage(
            (_character_row("conversation-a", "Old result", "2026-09-03T12:00:00Z"),),
            1,
            None,
            6,
        )

    async def activate(request, _cancellation):
        attempts.append(request)
        return ConsoleConversationActivationResult(
            ConsoleActivationResultKind.FAILED, request.target, False
        )

    app = _CharacterSwitcherApp(
        character_loader=character_loader,
        character_activate=activate,
        initial_mode=SwitcherMode.CHARACTER_CHATS,
    )
    async with app.run_test(size=(52, 20)) as pilot:
        await pilot.pause()
        row = app.screen.query_one(".console-switcher-result", Button)
        if gesture == "focused-enter":
            row.focus()
        app.screen.query_one("#console-switcher-query", Input).value = "new query"
        await entered.wait()

        assert row.disabled is True
        if gesture == "click":
            # Exercise the generation fence independently of the disabled
            # presentation guard (e.g. a queued Pressed message).
            row.disabled = False
            row.press()
        else:
            await pilot.press("enter")
        await pilot.pause()
        assert attempts == []

        release.set()
        await pilot.pause()


@pytest.mark.asyncio
async def test_pending_history_query_uses_the_same_exact_generation_fence() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def history_loader(**kwargs):
        if kwargs["query"]:
            entered.set()
            await release.wait()
            return ConsoleSwitcherHistoryPage((), 0, 50, 0)
        return ConsoleSwitcherHistoryPage(
            (_active_entry("saved-a", "Old saved result"),), 0, 50, 1
        )

    app = _CharacterSwitcherApp(
        history_loader=history_loader,
        initial_mode=SwitcherMode.HISTORY,
    )
    async with app.run_test(size=(52, 20)) as pilot:
        await pilot.pause()
        row = app.screen.query_one(".console-switcher-result", Button)
        app.screen.query_one("#console-switcher-query", Input).value = "new query"
        await entered.wait()

        assert row.disabled is True
        row.disabled = False
        row.press()
        await pilot.pause()
        assert app.result == "unset"

        release.set()
        await pilot.pause()


@pytest.mark.asyncio
async def test_raising_character_loader_is_typed_unavailable_not_zero_match() -> None:
    async def character_loader(**_kwargs):
        raise RuntimeError("database offline")

    app = _CharacterSwitcherApp(
        character_loader=character_loader,
        initial_mode=SwitcherMode.CHARACTER_CHATS,
        initial_character_query="needle",
    )
    async with app.run_test(size=(52, 20)) as pilot:
        await pilot.pause()
        query = app.screen.query_one("#console-switcher-query", Input)
        assert query.value == "needle"
        assert "Keyword search unavailable" in str(
            app.screen.query_one("#console-switcher-selected-detail", Static).renderable
        )
        assert (
            not app.screen.query("#console-switcher-empty")
            .first()
            .renderable.startswith("No Keyword matches")
        )
        recovery = app.screen.query_one("#console-switcher-recovery", Button)
        assert recovery.display is True
        assert str(recovery.label) == "Refresh results"


@pytest.mark.asyncio
async def test_unresolved_recovery_retains_exact_result_until_route_is_accepted() -> (
    None
):
    unresolved = _unavailable_row(
        "lost-conversation",
        "Lost conversation",
        UnavailableCharacterReason.DELETED_CARD,
        "2026-09-03T12:00:00Z",
    )
    accepted = False
    received: list[ConsoleSwitcherCharacterResult] = []

    async def character_loader(**_kwargs):
        return CharacterConversationPage((unresolved,), 1, None, 8)

    async def open_library(result, **_kwargs):
        received.append(result)
        return accepted

    app = _CharacterSwitcherApp(
        character_loader=character_loader,
        character_open_library=open_library,
        initial_mode=SwitcherMode.CHARACTER_CHATS,
    )
    async with app.run_test(size=(52, 20)) as pilot:
        await pilot.pause()
        await pilot.press("down", "enter")
        await pilot.pause()
        assert app.screen._committed_character_result is not None
        app.screen.query_one("#console-switcher-recovery", Button).press()
        await pilot.pause()

        assert isinstance(app.screen, ConsoleSessionSwitcherModal)
        assert app.result == "unset"
        assert received[0].unresolved == unresolved.unresolved
        assert app.screen._committed_character_result == received[0]


@pytest.mark.asyncio
async def test_becomes_unavailable_recovery_dismisses_only_after_exact_acceptance() -> (
    None
):
    received: list[ConsoleSwitcherCharacterResult] = []

    async def character_loader(**_kwargs):
        return CharacterConversationPage(
            (
                _character_row(
                    "conversation-a", "Changed source", "2026-09-03T12:00:00Z"
                ),
            ),
            1,
            None,
            8,
        )

    async def activate(request, _cancellation):
        return ConsoleConversationActivationResult(
            ConsoleActivationResultKind.CHARACTER_UNAVAILABLE,
            request.target,
            False,
        )

    async def open_library(result, **_kwargs):
        received.append(result)
        return True

    app = _CharacterSwitcherApp(
        character_loader=character_loader,
        character_activate=activate,
        character_open_library=open_library,
        initial_mode=SwitcherMode.CHARACTER_CHATS,
    )
    async with app.run_test(size=(52, 20)) as pilot:
        await pilot.pause()
        await pilot.press("down", "enter")
        await pilot.pause()
        app.screen.query_one("#console-switcher-recovery", Button).press()
        await pilot.pause()

    assert received[0].target == _target("conversation-a")
    assert app.result is None


@pytest.mark.asyncio
async def test_mounted_library_recovery_keeps_visit_until_app_navigation_completes():
    from types import SimpleNamespace

    from Tests.UI.test_console_character_context import _controller
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    messages = []
    posted = asyncio.Event()
    database = SimpleNamespace(
        get_local_authority_id=lambda: "authority-a",
        get_character_conversation_search_revision=lambda: 7,
    )

    def post_message(message):
        messages.append(message)
        posted.set()
        return True

    owner = SimpleNamespace(
        _character_context=_controller(database_accessor=lambda: database),
        post_message=post_message,
    )

    async def loader(**_kwargs):
        return CharacterConversationPage(
            (
                _unavailable_row(
                    "lost",
                    "Exact lost chat",
                    UnavailableCharacterReason.DELETED_CARD,
                    "2026-09-03T12:00:00Z",
                ),
            ),
            1,
            None,
            7,
        )

    async def recover(result, **kwargs):
        return await ChatScreen._open_console_character_library(owner, result, **kwargs)

    app = _CharacterSwitcherApp(
        character_loader=loader,
        character_open_library=recover,
        initial_mode=SwitcherMode.CHARACTER_CHATS,
        initial_character_query="needle",
    )
    async with app.run_test(size=(52, 20)) as pilot:
        await pilot.pause()
        modal = app.screen
        await pilot.press("enter")
        await pilot.pause()
        committed = modal._committed_character_result
        modal.query_one("#console-switcher-recovery", Button).press()
        await asyncio.wait_for(posted.wait(), 1)
        assert app.screen is modal
        assert app.result == "unset"
        assert modal.query_one("#console-switcher-query", Input).value == "needle"
        assert modal._committed_character_result is committed
        messages[0].report_completion(False)
        await pilot.pause()
        assert app.screen is modal
        assert modal._committed_character_result is committed
        posted.clear()
        modal.query_one("#console-switcher-recovery", Button).press()
        await asyncio.wait_for(posted.wait(), 1)
        assert app.screen is modal
        messages[1].report_completion(True)
        await pilot.pause()
        assert app.result is None


@pytest.mark.asyncio
async def test_f3_restores_history_page_two_and_stable_selection() -> None:
    calls: list[int] = []

    async def history_loader(**kwargs):
        offset = kwargs["offset"]
        calls.append(offset)
        entries = tuple(
            _active_entry(f"history-{index}", f"History {index}")
            for index in range(offset, min(offset + 50, 75))
        )
        return ConsoleSwitcherHistoryPage(entries, offset, 50, 75)

    async def character_loader(**_kwargs):
        return CharacterConversationPage((), 0, None, 3)

    app = _CharacterSwitcherApp(
        history_loader=history_loader,
        character_loader=character_loader,
        initial_mode=SwitcherMode.HISTORY,
    )
    async with app.run_test(size=(52, 20)) as pilot:
        await pilot.pause()
        app.screen.query_one("#console-switcher-next-page", Button).press()
        await pilot.pause()
        await pilot.press("down", "down", "down")
        selected = app.screen._candidate_key()
        assert app.screen._page_offset == 50

        await pilot.press("f3", "f3", "f3")
        await pilot.pause()

        assert app.screen._mode is SwitcherMode.HISTORY
        assert app.screen._page_offset == 50
        assert app.screen._candidate_key() == selected
        assert calls[-1] == 50


@pytest.mark.asyncio
async def test_f3_restores_scrolled_character_selection() -> None:
    async def history_loader(**_kwargs):
        return ConsoleSwitcherHistoryPage((), 0, 50, 0)

    async def character_loader(**_kwargs):
        rows = tuple(
            _character_row(
                f"conversation-{index}",
                f"Conversation {index}",
                f"2026-09-{28 - index:02d}T12:00:00Z",
            )
            for index in range(18)
        )
        return CharacterConversationPage(rows, 18, None, 9)

    app = _CharacterSwitcherApp(
        history_loader=history_loader,
        character_loader=character_loader,
        initial_mode=SwitcherMode.CHARACTER_CHATS,
    )
    async with app.run_test(size=(52, 20)) as pilot:
        await pilot.pause()
        for _ in range(9):
            await pilot.press("down")
        results = app.screen.query_one("#console-switcher-results", VerticalScroll)
        selected = app.screen._candidate_key()
        scroll_y = float(results.scroll_y)
        assert scroll_y > 0

        await pilot.press("f3", "f3", "f3")
        await pilot.pause()

        assert app.screen._mode is SwitcherMode.CHARACTER_CHATS
        assert app.screen._candidate_key() == selected
        assert float(results.scroll_y) == scroll_y


@pytest.mark.asyncio
async def test_character_rows_expose_truthful_state_and_action_vocabulary() -> None:
    rows = (
        _character_row("current", "Current", "2026-09-05T12:00:00Z", is_current=True),
        _character_row("other-open", "Other open", "2026-09-04T12:00:00Z"),
        _character_row("saved", "Saved", "2026-09-03T12:00:00Z"),
        _unavailable_row(
            "deleted",
            "Deleted",
            UnavailableCharacterReason.DELETED_CARD,
            "2026-09-02T12:00:00Z",
        ),
        _unavailable_row(
            "source-changed",
            "Source changed",
            UnavailableCharacterReason.MISSING_CHARACTER_AUTHORITY_LINK,
            "2026-09-01T12:00:00Z",
        ),
    )

    async def character_loader(**_kwargs):
        return CharacterConversationPage(rows, len(rows), None, 10)

    current_open = _active_character_entry("current", "Current", is_active=True)
    other_open = _active_character_entry("other-open", "Other open")
    app = _CharacterSwitcherApp(
        active=(current_open, other_open),
        character_loader=character_loader,
        initial_mode=SwitcherMode.CHARACTER_CHATS,
    )
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        presentations = {
            app.screen._payload_by_widget_id[button.id].title: (
                str(button.label),
                str(button.tooltip),
            )
            for button in app.screen.query(".console-switcher-result")
        }

        assert "CURRENT TAB" in presentations["Current"][0]
        assert presentations["Current"][1].startswith("OPEN TAB:")
        assert "OPEN TAB" in presentations["Other open"][0]
        assert presentations["Other open"][1].startswith("OPEN TAB:")
        assert "RESUME CHAT" in presentations["Saved"][0]
        assert presentations["Saved"][1].startswith("RESUME CHAT:")
        assert "DELETED CARD" in presentations["Deleted"][0]
        assert presentations["Deleted"][1].startswith("VIEW DETAILS:")
        assert "CHARACTER SOURCE CHANGED" in presentations["Source changed"][0]
        assert presentations["Source changed"][1].startswith("VIEW DETAILS:")
        assert "CURRENT TAB" in str(
            app.screen.query_one("#console-switcher-selected-detail", Static).renderable
        )


@pytest.mark.asyncio
async def test_current_character_grouping_never_claims_a_closed_chat_is_open() -> None:
    rows = (
        _character_row(
            "open", "Actually current tab", "2026-09-05T12:00:00Z", is_current=True
        ),
        _character_row(
            "closed",
            "Closed current-character chat",
            "2026-09-04T12:00:00Z",
            is_current=True,
        ),
    )

    async def character_loader(**_kwargs):
        return CharacterConversationPage(rows, len(rows), None, 11)

    active = (
        _active_character_entry("open", "Actually current tab", is_active=True),
        _active_character_entry(
            "closed",
            "Stale profile impostor",
            profile_authority="profile-b",
        ),
        _active_character_entry(
            "closed",
            "Stale runtime impostor",
            authority_token="runtime-b",
        ),
    )
    app = _CharacterSwitcherApp(
        active=active,
        character_loader=character_loader,
        initial_mode=SwitcherMode.CHARACTER_CHATS,
    )
    async with app.run_test(size=(52, 20)) as pilot:
        await pilot.pause()
        buttons = list(app.screen.query(".console-switcher-result"))
        open_button, closed_button = buttons
        open_result = app.screen._payload_by_widget_id[open_button.id]
        closed_result = app.screen._payload_by_widget_id[closed_button.id]

        assert "CURRENT TAB" in str(open_button.label)
        assert app.screen._entry_action(open_result) == "OPEN TAB"
        assert str(open_button.tooltip).startswith("OPEN TAB:")
        assert "CURRENT TAB" in str(
            app.screen.query_one("#console-switcher-selected-detail", Static).renderable
        )

        closed_button.focus()
        await pilot.pause()
        assert "RESUME CHAT" in str(closed_button.label)
        assert app.screen._entry_action(closed_result) == "RESUME CHAT"
        assert str(closed_button.tooltip).startswith("RESUME CHAT:")
        assert "RESUME CHAT" in str(
            app.screen.query_one("#console-switcher-selected-detail", Static).renderable
        )

        await pilot.press("f1")
        await pilot.pause()
        assert isinstance(app.screen, WorkbenchHelpPanel)
        rendered = app.screen.state.render_text()
        assert "Closed current-character chat" in rendered
        assert "RESUME CHAT" in rendered


@pytest.mark.asyncio
async def test_character_opening_repaints_row_and_accessible_action() -> None:
    entered = asyncio.Event()

    async def character_loader(**_kwargs):
        return CharacterConversationPage(
            (
                _character_row(
                    "conversation-a", "Opening target", "2026-09-03T12:00:00Z"
                ),
            ),
            1,
            None,
            4,
        )

    async def activate(request, cancellation):
        entered.set()
        await cancellation.wait()
        return ConsoleConversationActivationResult(
            ConsoleActivationResultKind.CANCELLED_PRECOMMIT, request.target, False
        )

    app = _CharacterSwitcherApp(
        character_loader=character_loader,
        character_activate=activate,
        initial_mode=SwitcherMode.CHARACTER_CHATS,
    )
    async with app.run_test(size=(52, 20)) as pilot:
        await pilot.pause()
        await pilot.press("down", "enter")
        await entered.wait()
        button = app.screen.query_one(".console-switcher-result", Button)
        assert "OPENING" in str(button.label)
        assert str(button.tooltip).startswith("OPENING:")
        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.asyncio
async def test_global_f1_shows_full_selected_unicode_title_and_action() -> None:
    title = "研究🙂" * 30

    async def character_loader(**_kwargs):
        return CharacterConversationPage(
            (_character_row("conversation-a", title, "2026-09-03T12:00:00Z"),),
            1,
            None,
            4,
        )

    app = _CharacterSwitcherApp(
        character_loader=character_loader,
        initial_mode=SwitcherMode.CHARACTER_CHATS,
    )
    async with app.run_test(size=(52, 20)) as pilot:
        await pilot.pause()
        await pilot.press("down", "f1")
        await pilot.pause()

        assert isinstance(app.screen, WorkbenchHelpPanel)
        rendered = app.screen.state.render_text()
        assert title in rendered
        assert "RESUME CHAT" in rendered


@pytest.mark.asyncio
async def test_pending_character_query_paints_busy_without_stale_detail_or_enter():
    entered = asyncio.Event()
    release = asyncio.Event()
    row = _character_row(
        "a", "First chat", "2026-09-03T12:00:00Z", excerpt="FIRST EXCERPT"
    )

    async def loader(*, query, **_kwargs):
        if query:
            entered.set()
            await release.wait()
        return CharacterConversationPage((row,), 1, None, 4)

    app = _CharacterSwitcherApp(
        character_loader=loader, initial_mode=SwitcherMode.CHARACTER_CHATS
    )
    async with app.run_test(size=(52, 20)) as pilot:
        try:
            await pilot.pause()
            app.screen.query_one("#console-switcher-query", Input).value = "new"
            await asyncio.wait_for(entered.wait(), 2)
            await pilot.pause()
            frame = "\n".join(
                strip.text for strip in app.screen._compositor.render_strips()
            )
            assert "Searching local chats" in frame
            assert "FIRST EXCERPT" not in frame
            assert "Enter:" not in frame
            assert "Cancel" in frame
        finally:
            release.set()
            await pilot.pause()


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["selection", "query", "mode"])
@pytest.mark.parametrize(
    "failure",
    [
        ConsoleActivationResultKind.FAILED,
        ConsoleActivationResultKind.CHARACTER_UNAVAILABLE,
    ],
)
async def test_failed_character_owner_is_released_when_selection_context_changes(
    transition, failure
):
    rows = (
        _character_row(
            "a", "First chat", "2026-09-03T12:00:00Z", excerpt="FIRST EXCERPT"
        ),
        _character_row(
            "b", "Second chat", "2026-09-02T12:00:00Z", excerpt="SECOND EXCERPT"
        ),
    )

    async def loader(*, query, **_kwargs):
        selected = (rows[1],) if query else rows
        return CharacterConversationPage(selected, len(selected), None, 4)

    async def activate(request, _cancellation):
        return ConsoleConversationActivationResult(failure, request.target, False)

    app = _CharacterSwitcherApp(
        active=(_active_entry("live", "Active chat"),),
        character_loader=loader,
        character_activate=activate,
        initial_mode=SwitcherMode.CHARACTER_CHATS,
    )
    async with app.run_test(size=(52, 20)) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        screen = app.screen
        recovery = screen.query_one("#console-switcher-recovery", Button)
        assert recovery.display
        assert "FIRST EXCERPT" in str(
            screen.query_one("#console-switcher-selected-detail", Static).renderable
        )
        if transition == "selection":
            screen.query(".console-switcher-result")[1].focus()
        elif transition == "query":
            screen.query_one("#console-switcher-query", Input).value = "Second"
        else:
            await pilot.press("f3")
        await pilot.pause(SEARCH_DEBOUNCE_SECONDS + 0.05)
        detail = str(
            screen.query_one("#console-switcher-selected-detail", Static).renderable
        )
        assert "FIRST EXCERPT" not in detail
        if transition != "mode":
            assert "SECOND EXCERPT" in detail
        assert not recovery.display
        await pilot.press("f1")
        await pilot.pause()
        rendered = app.screen.state.render_text()
        assert "First chat" not in rendered
        assert ("Active chat" if transition == "mode" else "Second chat") in rendered
