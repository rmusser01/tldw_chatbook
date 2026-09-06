"""Task 4 contract tests for Console Character navigation."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace
from typing import ClassVar

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    CharacterConversationGroup,
    CharacterConversationPage,
    CharacterConversationRow,
    CharacterKeywordIndexStatus,
    CharacterRepairCandidate,
    CharacterRepairPage,
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
from tldw_chatbook.Chat.console_rail_state import (
    CONSOLE_CHARACTER_DISCLOSURE_EXPLICIT_KEY,
    ConsoleRailPreferenceKey,
    build_console_rail_state,
    serialize_console_rail_stored_preferences,
)
from tldw_chatbook.UI.Console_Modules.character_context import (
    CONSOLE_CHARACTER_GROUP_LIMIT,
    CONSOLE_CHARACTER_ROW_LIMIT,
    CONSOLE_CHARACTER_SEARCH_LIMIT,
    ConsoleCharacterContextController,
    ConsoleCharacterContextState,
    ConsoleCharacterFocusIdentity,
    ConsoleCharacterOperationPhase,
    ConsoleCharacterQueryHandoff,
    ConsoleCharacterQueryHandoffCapability,
    ConsoleCharacterScopeFingerprint,
    ConsoleCharacterUnavailableDetail,
)
from tldw_chatbook.UI.Navigation.character_conversation_navigation import (
    LibraryCharacterRepairContext,
    LibraryUnavailableConversationInspection,
    LibraryUnavailableConversationsBrowse,
    RoleplayCharacterConversationLink,
    RoleplayReturnTarget,
)
from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module
from tldw_chatbook.Widgets.Console.console_character_context import (
    ConsoleCharacterContext,
)


def _resolved(character_id: int, conversation_id: str) -> CharacterConversationRow:
    key = ResolvedLocalCharacterKey("authority", character_id)
    return CharacterConversationRow.resolved(
        LocalCharacterConversationTarget(key, conversation_id),
        character_label=f"Character {character_id}",
        title=f"Chat {conversation_id}",
        last_modified=f"2026-09-0{character_id}T12:00:00Z",
        created_at="2026-09-01T00:00:00Z",
    )


def _groups() -> tuple[CharacterConversationGroup, ...]:
    groups = []
    for character_id in range(1, 5):
        key = ResolvedLocalCharacterKey("authority", character_id)
        rows = tuple(_resolved(character_id, f"{character_id}-{n}") for n in range(5))
        groups.append(
            CharacterConversationGroup(
                key,
                f"Character {character_id}",
                rows,
                9,
                character_id == 1,
            )
        )
    return tuple(groups)


class _Service:
    recent_calls: ClassVar[list[tuple[int, int]]] = []
    search_calls: ClassVar[list[tuple[str, int]]] = []

    def __init__(self, _db, *, current_character=None):
        self.current_character = current_character

    def recent_groups(self, *, group_limit: int, row_limit: int):
        self.recent_calls.append((group_limit, row_limit))
        return _groups()[:group_limit]

    def ensure_keyword_index(self):
        return CharacterKeywordIndexStatus.READY

    def keyword_search(self, query: str, *, limit: int):
        self.search_calls.append((query, limit))
        rows = tuple(_resolved(1, f"search-{n}") for n in range(12))
        return CharacterConversationPage(
            rows=rows[:limit],
            total=12,
            next_cursor=None,
            data_revision=7,
            keyword_status=CharacterKeywordIndexStatus.READY,
        )

    def refresh_unresolved_evidence(self, _key):
        return 3, "Historical Ada"

    def repair_candidates(self, _key, *, limit=20):
        return CharacterRepairPage((), 0, None)


def _controller(**overrides):
    async def activate(request, _cancel):
        return ConsoleConversationActivationResult(
            ConsoleActivationResultKind.OPENED, request.target, True
        )

    database = SimpleNamespace(
        get_local_authority_id=lambda: "authority",
        get_character_conversation_search_revision=lambda: 7,
    )
    params = {
        "database_accessor": lambda: database,
        "current_character_accessor": lambda: (1, "Character 1"),
        "open_conversation_accessor": lambda: None,
        "activate_target": activate,
        "navigate_roleplay": lambda _link: None,
        "navigate_repair": lambda _context: None,
        "navigate_inspection": lambda _link: None,
        "navigate_unavailable_browse": lambda _link: None,
        "navigate_roleplay_home": lambda: None,
        "navigate_library_home": lambda: None,
        "start_console": lambda _id, _name: None,
        "service_factory": _Service,
    }
    params.update(overrides)
    return ConsoleCharacterContextController(**params)


@pytest.mark.asyncio
async def test_controller_enforces_four_by_five_and_eight_search_bounds() -> None:
    _Service.recent_calls.clear()
    _Service.search_calls.clear()
    controller = _controller()

    await controller.refresh()
    focus = ConsoleCharacterFocusIdentity("group", group_key=_groups()[0].key)
    controller.capture_browse(focus=focus, scroll_offset=6)
    await controller.search("needle")

    assert _Service.recent_calls == [
        (CONSOLE_CHARACTER_GROUP_LIMIT, CONSOLE_CHARACTER_ROW_LIMIT)
    ]
    assert _Service.search_calls == [("needle", CONSOLE_CHARACTER_SEARCH_LIMIT)]
    assert len(controller.state.groups) == 4
    assert all(len(group.rows) <= 5 for group in controller.state.groups)
    assert len(controller.state.search_rows) == 8
    assert all(row.selected_excerpt == "" for row in controller.state.search_rows)

    await controller.search("")
    assert controller.state.expanded_key == _groups()[0].key
    assert controller.state.restore_focus == focus
    assert controller.state.restore_scroll_offset == 6


def test_accordion_keeps_at_most_one_stable_typed_key_expanded() -> None:
    controller = _controller()
    controller._publish(ConsoleCharacterContextState(groups=_groups()))

    controller.toggle_group(_groups()[1].key)
    assert controller.state.expanded_key == ResolvedLocalCharacterKey("authority", 2)
    controller.toggle_group(_groups()[2].key)
    assert controller.state.expanded_key == ResolvedLocalCharacterKey("authority", 3)
    controller.toggle_group(_groups()[2].key)
    assert controller.state.expanded_key is None


def test_character_disclosure_new_explicit_and_legacy_rules() -> None:
    key = ConsoleRailPreferenceKey("global", "layout", "key")

    assert (
        build_console_rail_state(
            preference_key=key,
            stored_preferences={},
            character_context_exists=True,
        ).character_open
        is True
    )


@pytest.mark.asyncio
async def test_canonical_writer_persists_first_use_close_reopen_and_legacy(
    monkeypatch,
) -> None:
    from copy import deepcopy

    from Tests.UI.test_console_inspector_compact_access import (
        _stored_rail_preferences,
    )
    from Tests.UI.test_console_left_rail import make_console_pilot

    key = ConsoleRailPreferenceKey("global", "layout", "key")

    async with make_console_pilot(size=(120, 40)) as pilot:
        screen = pilot.app.screen
        app = screen.app_instance
        screen._character_context._publish(
            ConsoleCharacterContextState(groups=_groups(), data_revision=7)
        )
        await pilot.pause()
        assert screen._current_console_rail_state().character_open is True

        screen._toggle_console_rail_section("character", next_open=False)
        stored = _stored_rail_preferences(app)
        assert stored["character_open"] is False
        assert stored[CONSOLE_CHARACTER_DISCLOSURE_EXPLICIT_KEY] is True
        assert screen._current_console_rail_state().character_open is False

        screen._toggle_console_rail_section("character", next_open=True)
        stored = _stored_rail_preferences(app)
        assert stored["character_open"] is True
        assert stored[CONSOLE_CHARACTER_DISCLOSURE_EXPLICIT_KEY] is True

        rail_key = next(iter(app.app_config["console"]["rail_state"]))
        legacy = deepcopy(stored)
        legacy["character_open"] = False
        legacy.pop(CONSOLE_CHARACTER_DISCLOSURE_EXPLICIT_KEY)
        app.app_config["console"]["rail_state"][rail_key] = legacy
        screen._toggle_console_rail_section("details", next_open=True)
        stored = _stored_rail_preferences(app)
        assert stored["character_open"] is False
        assert CONSOLE_CHARACTER_DISCLOSURE_EXPLICIT_KEY not in stored

        writes: list[dict[str, bool]] = []
        monkeypatch.setattr(
            type(screen),
            "_save_console_rail_preferences",
            lambda _screen, _key, payload, **_kwargs: writes.append(dict(payload)),
        )
        await pilot.resize_terminal(52, 20)
        await pilot.pause(0.2)
        assert writes == []
    assert (
        build_console_rail_state(
            preference_key=key,
            stored_preferences={
                "character_open": False,
                CONSOLE_CHARACTER_DISCLOSURE_EXPLICIT_KEY: True,
            },
            character_context_exists=True,
        ).character_open
        is False
    )
    assert (
        build_console_rail_state(
            preference_key=key,
            stored_preferences={"character_open": True},
            character_context_exists=False,
            available_columns=52,
        ).character_open
        is True
    )
    assert "character_open" not in serialize_console_rail_stored_preferences(None)
    assert (
        serialize_console_rail_stored_preferences(
            {CONSOLE_CHARACTER_DISCLOSURE_EXPLICIT_KEY: True}
        )["character_open"]
        is False
    )


class _CharacterApp(App):
    def __init__(self, controller, state):
        super().__init__()
        self.controller = controller
        self.state = state
        controller._publish(state)

    def compose(self) -> ComposeResult:
        widget = ConsoleCharacterContext(self.controller)
        widget._state = self.state
        yield widget


class _CharacterIdentityApp(App):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.identity = Static()

    def compose(self) -> ComposeResult:
        yield self.identity
        widget = ConsoleCharacterContext(
            self.controller,
            identity_state=self.identity,
        )
        self.controller._state_changed = widget.sync_state
        yield widget


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("current", "open_conversation_id", "groups", "expected"),
    [
        (
            None,
            "ordinary-open",
            _groups(),
            "No current character · Local · No open chat",
        ),
        (
            (9, "Zero"),
            None,
            (
                CharacterConversationGroup(
                    ResolvedLocalCharacterKey("authority", 9),
                    "Zero",
                    (),
                    0,
                    True,
                ),
            ),
            "Zero · Local · No open chat",
        ),
        ((1, "Saved"), None, _groups(), "Saved · Local · No open chat"),
        ((1, "Open"), "1-1", _groups(), "Open · Local · Open"),
    ],
)
async def test_identity_line_uses_only_owner_current_and_exact_open_state(
    current,
    open_conversation_id,
    groups,
    expected,
) -> None:
    class IdentityService(_Service):
        def recent_groups(self, *, group_limit: int, row_limit: int):
            return groups[:group_limit]

    controller = _controller(
        current_character_accessor=lambda: current,
        open_conversation_accessor=lambda: open_conversation_id,
        service_factory=IdentityService,
    )
    app = _CharacterIdentityApp(controller)
    async with app.run_test(size=(80, 35)) as pilot:
        await controller.refresh()
        await pilot.pause()
        assert str(app.identity.render()) == expected


@pytest.mark.asyncio
async def test_production_avatar_repaint_never_overwrites_identity_variants() -> None:
    from Tests.UI.test_console_left_rail import make_console_pilot

    variants = (
        (None, "", "No current character · Local · No open chat"),
        ((9, "Zero"), "", "Zero · Local · No open chat"),
        ((1, "Saved"), "", "Saved · Local · No open chat"),
        ((1, "Open"), "1-1", "Open · Local · Open"),
    )
    async with make_console_pilot(size=(120, 40)) as pilot:
        screen = pilot.app.screen
        for current, open_id, expected in variants:
            character_id = None if current is None else current[0]
            character_label = "" if current is None else current[1]
            state = replace(
                screen._character_context.state,
                scope_fingerprint=ConsoleCharacterScopeFingerprint(
                    database_identity=1,
                    data_authority_id="authority",
                    data_revision=7,
                    current_character_id=character_id,
                    current_character_label=character_label,
                    open_conversation_id=open_id,
                ),
            )
            screen._character_context._publish(state)
            await pilot.pause()

            await screen._render_character_avatar_into_section(
                spec=None,
                name="Avatar painter must not own identity",
                manual_label="Happy",
                is_current=lambda: True,
            )
            await pilot.pause()

            identity = screen.query_one("#console-character-identity", Static)
            assert str(identity.renderable) == expected


def test_character_picker_invalidates_context_before_current_mutation() -> None:
    calls: list[str] = []

    async def apply(_choice) -> None:
        calls.append("mutated")

    def run_worker(coroutine, **_kwargs) -> None:
        calls.append("scheduled")
        coroutine.close()

    screen = SimpleNamespace(
        _character_context=SimpleNamespace(
            invalidate_scope=lambda: calls.append("invalidated")
        ),
        _character=SimpleNamespace(_apply_console_character_choice_async=apply),
        run_worker=run_worker,
    )

    chat_screen_module.ChatScreen._apply_console_character_choice(
        screen,
        SimpleNamespace(),
    )

    assert calls == ["invalidated", "scheduled"]


@pytest.mark.asyncio
async def test_start_current_invalidates_context_before_current_mutation() -> None:
    calls: list[tuple[str, bool]] = []
    controller = None

    async def start(_character_id: int, _name: str) -> None:
        assert controller is not None
        calls.append(("mutated", controller.state.scope_fingerprint is None))

    controller = _controller(start_console=start)
    await controller.refresh()
    group = _groups()[0]

    await controller.start_current(group)

    assert calls == [("mutated", True)]


@pytest.mark.asyncio
async def test_mounted_widget_has_four_headers_five_rows_and_no_future_handoff_copy() -> (
    None
):
    state = ConsoleCharacterContextState(
        groups=_groups(), expanded_key=_groups()[0].key, data_revision=7
    )
    app = _CharacterApp(_controller(), state)
    async with app.run_test(size=(72, 35)) as pilot:
        await pilot.pause()
        assert len(app.screen.query(".console-character-group")) == 4
        assert len(app.screen.query(".console-character-row")) == 5
        assert "Continue search in Character chats" not in str(app.screen.render())
        assert "View all 9 in Roleplay" in str(
            app.screen.query_one(
                f"#{ConsoleCharacterContext.action_dom_id('view-all', _groups()[0].key)}"
            ).label
        )


@pytest.mark.asyncio
async def test_unavailable_row_is_repair_only_and_zero_chat_current_can_start() -> None:
    unresolved = UnresolvedConversationKey("authority", "lost")
    unavailable = CharacterConversationGroup(
        unresolved,
        "Chats with unavailable characters",
        (
            CharacterConversationRow.unavailable(
                unresolved,
                reason=UnavailableCharacterReason.MISSING_CARD,
                character_label="Historical Ada",
                title="Lost chat",
                last_modified="2026-09-03T12:00:00Z",
                created_at="2026-09-01T00:00:00Z",
            ),
        ),
        1,
        False,
    )
    current = CharacterConversationGroup(
        ResolvedLocalCharacterKey("authority", 1), "Ada", (), 0, True
    )
    state = ConsoleCharacterContextState(
        groups=(current, unavailable), expanded_key=current.key
    )
    app = _CharacterApp(_controller(), state)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.controller._publish(state)
        app.screen.query_one(ConsoleCharacterContext).sync_state(state)
        await pilot.pause()
        assert app.screen.query_one(
            f"#{ConsoleCharacterContext.action_dom_id('start', current.key)}"
        )
        assert "No chats with Ada yet" == str(
            app.screen.query_one(".console-character-empty").renderable
        )
        assert not app.screen.query(
            f"#{ConsoleCharacterContext.action_dom_id('view-all', current.key)}"
        )
        await pilot.click(f"#{ConsoleCharacterContext.group_dom_id(unavailable.key)}")
        await pilot.pause()
        assert "Card missing" in str(
            app.screen.query_one(
                f"#{ConsoleCharacterContext.row_dom_id(unavailable.rows[0].row_key)}"
            ).label
        )
        assert "View all 1 in Library" in str(
            app.screen.query_one(
                f"#{ConsoleCharacterContext.action_dom_id('view-all', unavailable.key)}"
            ).label
        )


@pytest.mark.asyncio
async def test_controller_routes_only_typed_activation_roleplay_and_repair() -> None:
    activations: list[CharacterConversationActivationRequest] = []
    roleplay: list[RoleplayCharacterConversationLink] = []
    repairs: list[LibraryCharacterRepairContext] = []
    inspections: list[LibraryUnavailableConversationInspection] = []
    unavailable_browses: list[LibraryUnavailableConversationsBrowse] = []

    async def activate(request, _cancel):
        activations.append(request)
        return ConsoleConversationActivationResult(
            ConsoleActivationResultKind.OPENED,
            request.target,
            True,
        )

    controller = _controller(
        activate_target=activate,
        navigate_roleplay=roleplay.append,
        navigate_repair=repairs.append,
        navigate_inspection=inspections.append,
        navigate_unavailable_browse=unavailable_browses.append,
        service_factory=type(
            "CandidateService",
            (_Service,),
            {
                "repair_candidates": lambda self, _key, *, limit=20: (
                    CharacterRepairPage(
                        (
                            CharacterRepairCandidate(
                                ResolvedLocalCharacterKey("authority", 7),
                                "Replacement",
                                1,
                            ),
                        ),
                        1,
                        None,
                    )
                )
            },
        ),
    )
    controller._publish(
        ConsoleCharacterContextState(
            groups=_groups(),
            expanded_key=_groups()[0].key,
            data_revision=7,
        )
    )
    target = _groups()[0].rows[0].target
    assert target is not None

    await controller.activate(target)
    await controller.view_group(_groups()[0])
    unresolved = UnresolvedConversationKey("authority", "lost")
    await controller.open_unavailable(unresolved, row_key="lost-row")
    await controller.repair_unavailable(unresolved, row_key="lost-row")
    unavailable_group = CharacterConversationGroup(
        unresolved,
        "Chats with unavailable characters",
        (
            CharacterConversationRow.unavailable(
                unresolved,
                reason=UnavailableCharacterReason.MISSING_CARD,
                character_label="Historical Ada",
                title="Lost chat",
                last_modified="2026-09-03T12:00:00Z",
                created_at="2026-09-01T00:00:00Z",
            ),
        ),
        9,
        False,
    )
    controller.select_unavailable(unavailable_group.rows[0].row_key)
    await controller.view_group(unavailable_group)

    assert activations == [
        CharacterConversationActivationRequest(target, "authority", 7)
    ]
    assert roleplay == [
        RoleplayCharacterConversationLink(
            character=_groups()[0].key,
            return_target=RoleplayReturnTarget.console_context_character(),
        )
    ]
    assert inspections == [
        LibraryUnavailableConversationInspection(
            unresolved=unresolved,
            return_target=RoleplayReturnTarget.console_context_character(),
        )
    ]
    assert repairs == [
        LibraryCharacterRepairContext(
            unresolved=unresolved,
            expected_conversation_version=3,
            historical_display_snapshot="Historical Ada",
            return_target=RoleplayReturnTarget.console_context_character(),
        )
    ]
    assert unavailable_browses == [
        LibraryUnavailableConversationsBrowse(
            selected=unresolved,
            return_target=RoleplayReturnTarget.console_context_character(),
        )
    ]


@pytest.mark.asyncio
async def test_scope_fingerprint_refreshes_current_profile_revision_and_activation() -> (
    None
):
    calls: list[ResolvedLocalCharacterKey | None] = []
    current = {"value": (1, "Ada")}
    database = SimpleNamespace(
        authority="authority",
        revision=7,
        get_local_authority_id=lambda: database.authority,
        get_character_conversation_search_revision=lambda: database.revision,
    )

    class ScopeService(_Service):
        def recent_groups(self, *, group_limit: int, row_limit: int):
            calls.append(self.current_character)
            return _groups()[:group_limit]

    controller = _controller(
        database_accessor=lambda: database,
        current_character_accessor=lambda: current["value"],
        service_factory=ScopeService,
    )
    await controller.refresh_if_scope_changed()
    await controller.refresh_if_scope_changed()
    assert len(calls) == 1

    current["value"] = (2, "Bea")
    await controller.refresh_if_scope_changed()
    database.revision = 8
    await controller.refresh_if_scope_changed()
    database.authority = "authority-2"
    await controller.refresh_if_scope_changed()
    assert [key.character_id if key else None for key in calls] == [1, 2, 2, 2]
    assert controller.state.scope_fingerprint is not None
    assert controller.state.scope_fingerprint.data_authority_id == "authority-2"

    target = LocalCharacterConversationTarget(
        ResolvedLocalCharacterKey("authority-2", 2), "after-open"
    )
    controller._publish(
        replace(controller.state, data_revision=8, scope_fingerprint=None)
    )
    await controller.activate(target, row_key="activation-row")
    assert len(calls) == 5


@pytest.mark.asyncio
async def test_scope_capture_retries_when_database_switches_during_revision_read() -> (
    None
):
    """Catch a pre-await database identity being committed after profile switch."""

    entered = threading.Event()
    release = threading.Event()

    class Database:
        def __init__(self, authority: str, *, block: bool = False) -> None:
            self.authority = authority
            self.block = block

        def get_local_authority_id(self) -> str:
            if self.block:
                entered.set()
                assert release.wait(2)
            return self.authority

        def get_character_conversation_search_revision(self) -> int:
            return 7

    old = Database("old-authority")
    new = Database("new-authority")
    active = {"database": old}
    calls: list[str] = []

    class ScopeService(_Service):
        def __init__(self, database, *, current_character=None):
            super().__init__(database, current_character=current_character)
            calls.append(database.authority)

    controller = _controller(
        database_accessor=lambda: active["database"],
        service_factory=ScopeService,
    )
    await controller.refresh()
    old.block = True
    pending = asyncio.create_task(controller.refresh_if_scope_changed())
    assert await asyncio.to_thread(entered.wait, 2)
    active["database"] = new
    release.set()

    assert await pending is True
    assert controller.state.scope_fingerprint is not None
    assert controller.state.scope_fingerprint.data_authority_id == "new-authority"
    assert calls[-1] == "new-authority"


@pytest.mark.asyncio
async def test_scope_capture_retries_when_current_character_switches_during_read() -> (
    None
):
    """Catch a pre-await character identity being retained after session switch."""

    entered = threading.Event()
    release = threading.Event()

    class Database:
        block = False

        def get_local_authority_id(self) -> str:
            if self.block:
                entered.set()
                assert release.wait(2)
            return "authority"

        def get_character_conversation_search_revision(self) -> int:
            return 7

    database = Database()
    current = {"value": (1, "Ada")}
    calls: list[int | None] = []

    class ScopeService(_Service):
        def recent_groups(self, *, group_limit: int, row_limit: int):
            calls.append(
                None
                if self.current_character is None
                else self.current_character.character_id
            )
            return _groups()[:group_limit]

    controller = _controller(
        database_accessor=lambda: database,
        current_character_accessor=lambda: current["value"],
        service_factory=ScopeService,
    )
    await controller.refresh()
    database.block = True
    pending = asyncio.create_task(controller.refresh_if_scope_changed())
    assert await asyncio.to_thread(entered.wait, 2)
    current["value"] = (2, "Bea")
    release.set()

    assert await pending is True
    assert controller.state.scope_fingerprint is not None
    assert controller.state.scope_fingerprint.current_character_id == 2
    assert calls[-1] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("mutated_member", ["authority", "revision"])
async def test_scope_capture_retries_same_handle_metadata_mutation_across_await(
    mutated_member: str,
) -> None:
    """Every database-owned fingerprint member is recaptured after the await."""

    entered = threading.Event()
    release = threading.Event()

    class Database:
        authority = "old-authority"
        revision = 7
        block_member = ""
        blocked_once = False

        def _read(self, member: str):
            value = getattr(self, member)
            if self.block_member == member and not self.blocked_once:
                self.blocked_once = True
                entered.set()
                assert release.wait(2)
            return value

        def get_local_authority_id(self) -> str:
            return str(self._read("authority"))

        def get_character_conversation_search_revision(self) -> int:
            return int(self._read("revision"))

    database = Database()
    controller = _controller(database_accessor=lambda: database)
    await controller.refresh()
    database.block_member = mutated_member
    database.blocked_once = False
    pending = asyncio.create_task(controller.refresh_if_scope_changed())
    assert await asyncio.to_thread(entered.wait, 2)
    if mutated_member == "authority":
        database.authority = "new-authority"
    else:
        database.revision = 8
    release.set()

    assert await pending is True
    assert controller.state.scope_fingerprint is not None
    assert controller.state.scope_fingerprint.data_authority_id == database.authority
    assert controller.state.scope_fingerprint.data_revision == database.revision


@pytest.mark.asyncio
async def test_stable_scope_metadata_exception_settles_idle_retryable_error() -> None:
    class Database:
        fail = False

        def get_local_authority_id(self) -> str:
            if self.fail:
                raise RuntimeError("metadata unavailable")
            return "authority"

        def get_character_conversation_search_revision(self) -> int:
            return 7

    database = Database()
    controller = _controller(database_accessor=lambda: database)
    await controller.refresh()
    database.fail = True

    await controller.refresh()

    assert controller.state.phase is ConsoleCharacterOperationPhase.IDLE
    assert controller.state.loading is False
    assert controller.state.error == "Could not load local character chats · Retry"


@pytest.mark.asyncio
async def test_search_scope_metadata_exception_settles_idle_retryable_error() -> None:
    """A stable metadata failure cannot leave Keyword search latched busy."""

    class Database:
        fail = False

        def get_local_authority_id(self) -> str:
            if self.fail:
                raise RuntimeError("metadata unavailable")
            return "authority"

        def get_character_conversation_search_revision(self) -> int:
            return 7

    database = Database()
    controller = _controller(database_accessor=lambda: database)
    await controller.refresh()
    groups = controller.state.groups
    database.fail = True

    await controller.search("needle")

    assert controller.state.phase is ConsoleCharacterOperationPhase.IDLE
    assert controller.state.loading is False
    assert controller.state.operation_row_key == ""
    assert controller.state.error == "Could not search local character chats · Retry"
    assert controller.state.query == "needle"
    assert controller.state.search_rows == ()
    assert controller.state.groups == groups


@pytest.mark.asyncio
async def test_repair_scope_metadata_exception_settles_idle_retryable_error() -> None:
    """A stable metadata failure cannot leave Library repair latched busy."""

    class Database:
        fail = False

        def get_local_authority_id(self) -> str:
            if self.fail:
                raise RuntimeError("metadata unavailable")
            return "authority"

        def get_character_conversation_search_revision(self) -> int:
            return 7

    database = Database()
    controller = _controller(database_accessor=lambda: database)
    await controller.refresh()
    previous_details = controller.state.unavailable_details
    database.fail = True

    repaired = await controller.repair_unavailable(
        UnresolvedConversationKey("authority", "lost"),
        row_key="lost-row",
    )

    assert repaired is False
    assert controller.state.phase is ConsoleCharacterOperationPhase.IDLE
    assert controller.state.loading is False
    assert controller.state.operation_row_key == ""
    assert controller.state.error == "Could not refresh Library details · Retry"
    assert controller.state.unavailable_details == previous_details


@pytest.mark.asyncio
async def test_replaced_closed_database_error_never_paints_new_profile() -> None:
    """Catch an exception from a replaced handle becoming the new profile's error."""

    entered = threading.Event()
    release = threading.Event()

    class OldDatabase:
        closed = False

        def get_local_authority_id(self) -> str:
            entered.set()
            assert release.wait(2)
            if self.closed:
                raise RuntimeError("closed old database")
            return "old-authority"

        def get_character_conversation_search_revision(self) -> int:
            return 7

    new = SimpleNamespace(
        authority="new-authority",
        get_local_authority_id=lambda: "new-authority",
        get_character_conversation_search_revision=lambda: 8,
    )
    old = OldDatabase()
    active = {"database": old}
    controller = _controller(database_accessor=lambda: active["database"])
    pending = asyncio.create_task(controller.refresh())
    assert await asyncio.to_thread(entered.wait, 2)
    active["database"] = new
    old.closed = True
    release.set()

    await pending
    assert controller.state.error == ""
    assert controller.state.scope_fingerprint is not None
    assert controller.state.scope_fingerprint.data_authority_id == "new-authority"


@pytest.mark.asyncio
async def test_semantic_focus_survives_reorder_and_falls_back_to_group() -> None:
    groups = _groups()[:2]
    state = ConsoleCharacterContextState(
        groups=groups,
        expanded_key=groups[0].key,
        data_revision=7,
    )
    app = _CharacterApp(_controller(), state)
    async with app.run_test(size=(80, 35)) as pilot:
        await pilot.pause()
        row_key = groups[0].rows[1].row_key
        row = app.screen.query_one(f"#{ConsoleCharacterContext.row_dom_id(row_key)}")
        row.focus()
        await pilot.pause()
        reordered_group = replace(
            groups[0], rows=(groups[0].rows[1], groups[0].rows[0], *groups[0].rows[2:])
        )
        reordered = ConsoleCharacterContextState(
            groups=(groups[1], reordered_group),
            expanded_key=reordered_group.key,
            data_revision=8,
        )
        app.screen.query_one(ConsoleCharacterContext).sync_state(reordered)
        await pilot.pause()
        assert getattr(
            app.screen.focused, "id", None
        ) == ConsoleCharacterContext.row_dom_id(row_key)

        disappeared = replace(
            reordered, groups=(groups[1], replace(reordered_group, rows=()))
        )
        app.screen.query_one(ConsoleCharacterContext).sync_state(disappeared)
        await pilot.pause()
        assert getattr(
            app.screen.focused, "id", None
        ) == ConsoleCharacterContext.group_dom_id(reordered_group.key)


@pytest.mark.asyncio
async def test_unavailable_detail_reason_exact_open_and_candidate_gated_repair() -> (
    None
):
    unresolved = UnresolvedConversationKey("authority", "lost")
    row = CharacterConversationRow.unavailable(
        unresolved,
        reason=UnavailableCharacterReason.DELETED_CARD,
        character_label="Historical Ada",
        title="Lost chat",
        last_modified="2026-09-03T12:00:00Z",
        created_at="2026-09-01T00:00:00Z",
    )
    group = CharacterConversationGroup(
        unresolved, "Chats with unavailable characters", (row,), 1, False
    )
    context = LibraryCharacterRepairContext(
        unresolved=unresolved,
        expected_conversation_version=3,
        historical_display_snapshot="Historical Ada",
        return_target=RoleplayReturnTarget.console_context_character(),
    )
    inspections: list[LibraryUnavailableConversationInspection] = []
    state = ConsoleCharacterContextState(
        groups=(group,),
        expanded_key=unresolved,
        unavailable_details=(
            ConsoleCharacterUnavailableDetail(
                row_key=row.row_key,
                reason_copy="Card deleted",
                context=context,
                candidate_count=0,
            ),
        ),
        selected_unavailable_row_key=row.row_key,
    )
    controller = _controller(navigate_inspection=inspections.append)
    app = _CharacterApp(controller, state)
    async with app.run_test(size=(80, 35)) as pilot:
        await pilot.pause()
        controller._publish(state)
        app.screen.query_one(ConsoleCharacterContext).sync_state(state)
        await pilot.pause()
        assert (
            str(
                app.screen.query_one("#console-character-unavailable-reason").renderable
            )
            == "Card deleted"
        )
        assert app.screen.query_one("#console-character-open-library")
        assert not app.screen.query("#console-character-repair-library")
        app.screen.query_one("#console-character-open-library", Button).press()
        await pilot.pause()
        assert inspections == [
            LibraryUnavailableConversationInspection(
                unresolved=unresolved,
                return_target=RoleplayReturnTarget.console_context_character(),
            )
        ]

    candidate = CharacterRepairCandidate(
        ResolvedLocalCharacterKey("authority", 7), "Replacement", 1
    )

    class CandidateService(_Service):
        def repair_candidates(self, _key, *, limit=20):
            assert limit == 20
            return CharacterRepairPage((candidate,), 27, 1)

    controller = _controller(service_factory=CandidateService)
    controller._publish(replace(state, unavailable_details=()))
    await controller.refresh_unavailable_details((group,))
    assert controller.state.unavailable_details[0].candidate_count == 27
    repairs = []
    controller._navigate_repair = repairs.append
    assert await controller.repair_unavailable(unresolved, row_key=row.row_key)
    assert controller.state.unavailable_details[0].candidate_count == 27
    assert repairs[0].unresolved == unresolved


@pytest.mark.asyncio
async def test_opening_phase_preserves_row_blocks_duplicates_and_escape_cancels() -> (
    None
):
    entered = asyncio.Event()
    release = asyncio.Event()
    cancellations: list[asyncio.Event] = []

    async def activate(request, cancellation):
        cancellations.append(cancellation)
        entered.set()
        await release.wait()
        return ConsoleConversationActivationResult(
            ConsoleActivationResultKind.CANCELLED_PRECOMMIT
            if cancellation.is_set()
            else ConsoleActivationResultKind.OPENED,
            request.target,
            False,
        )

    state = ConsoleCharacterContextState(
        groups=_groups(), expanded_key=_groups()[0].key, data_revision=7
    )
    controller = _controller(activate_target=activate)
    app = _CharacterApp(controller, state)
    async with app.run_test(size=(80, 35)) as pilot:
        await pilot.pause()
        controller._publish(state)
        app.screen.query_one(ConsoleCharacterContext).sync_state(state)
        await pilot.pause()
        row = app.screen.query_one(
            f"#{ConsoleCharacterContext.row_dom_id(_groups()[0].rows[0].row_key)}"
        )
        row.focus()
        await pilot.pause()
        target = _groups()[0].rows[0].target
        assert target is not None
        pending = asyncio.create_task(
            controller.activate(target, row_key=_groups()[0].rows[0].row_key)
        )
        await asyncio.wait_for(entered.wait(), timeout=1)
        app.screen.query_one(ConsoleCharacterContext).sync_state(controller.state)
        await pilot.pause()
        opening = app.screen.query_one(
            f"#{ConsoleCharacterContext.row_dom_id(_groups()[0].rows[0].row_key)}"
        )
        assert controller.state.phase is ConsoleCharacterOperationPhase.OPENING
        assert "Opening…" in str(opening.label)
        assert opening.has_class("-opening")
        assert app.screen.focused is opening
        duplicate = await controller.activate(
            target, row_key=_groups()[0].rows[0].row_key
        )
        assert duplicate.kind is ConsoleActivationResultKind.FAILED
        assert len(cancellations) == 1
        await pilot.press("escape")
        assert cancellations[0].is_set()
        release.set()
        await pending
        await pilot.pause()
        assert controller.state.phase is ConsoleCharacterOperationPhase.IDLE


@pytest.mark.asyncio
async def test_opening_search_result_escape_preserves_query_rows_and_focus() -> None:
    """Catch cancellation falling through into idle search clearing."""

    entered = asyncio.Event()
    release = asyncio.Event()
    cancellations: list[asyncio.Event] = []

    async def activate(request, cancellation):
        cancellations.append(cancellation)
        entered.set()
        await release.wait()
        return ConsoleConversationActivationResult(
            ConsoleActivationResultKind.CANCELLED_PRECOMMIT,
            request.target,
            False,
        )

    rows = tuple(_resolved(1, f"search-{index}") for index in range(2))
    state = ConsoleCharacterContextState(
        groups=_groups(),
        query="needle",
        search_rows=rows,
        data_revision=7,
    )
    controller = _controller(activate_target=activate)
    state = replace(state, scope_fingerprint=await controller._fingerprint())
    app = _CharacterApp(controller, state)
    async with app.run_test(size=(80, 35)) as pilot:
        widget = app.screen.query_one(ConsoleCharacterContext)
        controller._state_changed = widget.sync_state
        controller._publish(state)
        await pilot.pause()
        row_id = ConsoleCharacterContext.row_dom_id(rows[0].row_key)
        row = app.screen.query_one(f"#{row_id}", Button)
        row.focus()
        row.press()
        await asyncio.wait_for(entered.wait(), timeout=1)
        await pilot.pause()

        await pilot.press("escape")
        await pilot.pause()

        assert cancellations[0].is_set()
        assert controller.state.phase is ConsoleCharacterOperationPhase.OPENING
        assert controller.state.query == "needle"
        assert controller.state.search_rows == rows
        assert getattr(app.screen.focused, "id", None) == row_id

        release.set()
        for _ in range(40):
            if controller.state.phase is ConsoleCharacterOperationPhase.IDLE:
                break
            await pilot.pause(0.05)
        assert controller.state.query == "needle"
        assert controller.state.search_rows == rows
        assert getattr(app.screen.focused, "id", None) == row_id


def test_dormant_typed_query_handoff_is_default_false() -> None:
    handed_off: list[ConsoleCharacterQueryHandoff] = []
    controller = _controller(query_handoff=handed_off.append)
    assert controller.handoff_query("needle") is False
    assert handed_off == []

    controller = _controller(
        query_handoff_capability=ConsoleCharacterQueryHandoffCapability(True),
        query_handoff=handed_off.append,
    )
    assert controller.handoff_query("needle") is True
    assert handed_off == [ConsoleCharacterQueryHandoff("needle")]


@pytest.mark.asyncio
async def test_newer_search_generation_fences_late_activation_presentation() -> None:
    release = asyncio.Event()

    async def activate(request, _cancel):
        await release.wait()
        return ConsoleConversationActivationResult(
            ConsoleActivationResultKind.OPENED,
            request.target,
            True,
        )

    controller = _controller(activate_target=activate)
    controller._publish(ConsoleCharacterContextState(groups=_groups(), data_revision=7))
    target = _groups()[0].rows[0].target
    assert target is not None
    pending = asyncio.create_task(controller.activate(target))
    await asyncio.sleep(0)
    await controller.search("newer")
    release.set()
    await pending

    assert controller.state.query == "newer"
    assert controller.state.error == ""
    assert len(controller.state.search_rows) == 8


@pytest.mark.asyncio
async def test_search_reports_index_build_instead_of_false_empty_success() -> None:
    class _BuildingService(_Service):
        def ensure_keyword_index(self):
            return CharacterKeywordIndexStatus.BUILDING

        def keyword_search(self, query: str, *, limit: int):
            return CharacterConversationPage(
                rows=(),
                total=0,
                next_cursor=None,
                data_revision=7,
                keyword_status=CharacterKeywordIndexStatus.BUILDING,
            )

    controller = _controller(service_factory=_BuildingService)
    await controller.search("needle")

    assert controller.state.search_rows == ()
    assert controller.state.error == "Character chat search is rebuilding · Retry"
    assert controller.state.keyword_status is CharacterKeywordIndexStatus.BUILDING


@pytest.mark.asyncio
async def test_browse_reports_missing_local_database_instead_of_false_empty() -> None:
    controller = _controller(database_accessor=lambda: None)

    await controller.refresh()

    assert controller.state.groups == ()
    assert controller.state.error == "Local character data is unavailable · Retry"


@pytest.mark.asyncio
async def test_pointer_selects_then_double_click_and_enter_activate_exact_row() -> None:
    opened: list[LocalCharacterConversationTarget] = []

    async def activate(request, _cancel):
        opened.append(request.target)
        return ConsoleConversationActivationResult(
            ConsoleActivationResultKind.OPENED,
            request.target,
            True,
        )

    state = ConsoleCharacterContextState(
        groups=_groups(), expanded_key=_groups()[0].key, data_revision=7
    )
    app = _CharacterApp(_controller(activate_target=activate), state)
    async with app.run_test(size=(72, 35)) as pilot:
        await pilot.pause()
        app.controller._publish(state)
        app.screen.query_one(ConsoleCharacterContext).sync_state(state)
        await pilot.pause()
        row = app.screen.query_one(
            f"#{ConsoleCharacterContext.row_dom_id(_groups()[0].rows[0].row_key)}"
        )
        assert await pilot.click(row)
        await pilot.pause()
        assert opened == []
        assert await pilot.click(row, times=2)
        await pilot.pause()
        assert opened == [row.character_row.target]

        second = app.screen.query_one(
            f"#{ConsoleCharacterContext.row_dom_id(_groups()[0].rows[1].row_key)}"
        )
        second.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert opened[-1] == second.character_row.target

        header = app.screen.query_one(
            f"#{ConsoleCharacterContext.group_dom_id(_groups()[0].key)}"
        )
        header.focus()
        await pilot.press("left")
        await pilot.pause()
        assert app.controller.state.expanded_key is None
        await pilot.press("right")
        await pilot.pause()
        assert app.controller.state.expanded_key == _groups()[0].key

        second = app.screen.query_one(
            f"#{ConsoleCharacterContext.row_dom_id(_groups()[0].rows[1].row_key)}"
        )
        second.focus()
        await pilot.press("space")
        await pilot.pause()
        assert opened[-1] == second.character_row.target
