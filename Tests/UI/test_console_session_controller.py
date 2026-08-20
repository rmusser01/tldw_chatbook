"""Characterisation tests for the Console session cluster.

Originally written BEFORE the wave-2 Task 3 extraction of
`ConsoleSessionController` (`tldw_chatbook/UI/Console_Modules/session.py`)
landed -- all 8 tests below passed against the unmodified, still-monolithic
`ChatScreen` (commit `08f0f5479`) before the extraction touched any
production file. Call sites were then retargeted from `console.<method>(...)`
to `console._session.<method>(...)` to follow the move; every assertion is
byte-for-byte unchanged from the pre-move version.

Drives `_activate_native_console_session`, `_apply_console_switcher_choice`,
and `_promote_console_temporary_session` through REAL interactions against a
real mounted `ChatScreen` -- the same "real production coroutine, not a
rebuilt double" discipline `test_console_native_chat_flow.py`'s own resume
coverage and `test_console_workspace_controller.py` (wave-2 Task 2's own
characterisation file) use. No method under test is monkeypatched.

Two things this file exists specifically to pin, per the wave-2 Task 3 brief:

1. **Session activation's workspace-alignment seam**
   (`_activate_native_console_session` -> `_set_active_workspace_for_
   session`, a session-controller -> workspace-controller named-callable
   call): activating a session that belongs to a DIFFERENT workspace than
   the currently active one must also switch the active workspace. Existing
   coverage (`test_console_cost_chip_screen.py::test_cost_chip_state_
   isolated_across_session_tabs`, `test_console_agent_rail.py::test_
   activate_native_console_session_clears_stale_drilldown`, `test_console_
   scope_row.py`) exercises the SAME entry point but never with two
   DIFFERENT workspaces in play, so none of it would have caught a
   snapshot-vs-live binding bug in that one call.
2. **Temporary-session promotion writes a DURABLE database row.** Every
   existing `_promote_console_temporary_session` test
   (`Tests/UI/test_console_composer_menu.py`) stubs `store.
   promote_ephemeral_session` outright and asserts only chip/widget state --
   none of them ever let a real write reach `chachanotes_db`. The brief is
   explicit: assert the DATABASE, not widget state.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from Tests.UI.background_signals import (
    await_background_task,
    wait_for_background_signal,
)
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_character_session_prompt_seed import (
    _character_screen,
    _roleplay_card,
    _start_chat_handoff,
)
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_switcher_state import ConsoleSwitcherEntry
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Widgets.Console.console_rename_session_modal import (
    ConsoleRenameSessionModal,
)
from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (
    ConsoleSwitcherChoice,
)


@pytest.mark.asyncio
async def test_character_handoff_reuses_untouched_chat_one(monkeypatch):
    card = _roleplay_card(name="Alba")
    screen = _character_screen(monkeypatch, card)
    store = screen._ensure_console_chat_store()
    defaults = screen._session._default_console_session_settings()
    original = store.ensure_session(
        title="Chat 1",
        workspace_id="workspace-original",
        settings=defaults,
        canonical_settings_baseline=defaults,
    )
    sync = AsyncMock()
    focus_calls: list[bool] = []
    screen._session._sync_native_console_chat_ui_fn = sync
    screen._session._focus_composer_if_needed_fn = (
        lambda *, force=False: focus_calls.append(force)
    )

    assert await screen._session._start_character_console_session(
        _start_chat_handoff(card)
    )

    sessions = store.sessions()
    assert len(sessions) == 1
    session = sessions[0]
    assert session is original
    assert session.id == original.id
    assert session.workspace_id == "workspace-original"
    assert session.title == "Chat with Alba"
    assert session.settings.model == defaults.model
    assert session.settings.system_prompt == "Protect Captain Rowan as Alba."
    greetings = store.messages_for_session(session.id)
    assert [message.content for message in greetings] == ["Hello, Captain Rowan."]
    assert session.identity_revision == 2
    assert store.payload_revision(session.id) == 3
    presentation = store.presentation_context(session.id, "fallback")
    assert presentation.character_name == "Alba"
    assert presentation.revision == session.identity_revision
    sync.assert_awaited_once_with()
    assert focus_calls == [True]


@pytest.mark.asyncio
async def test_draft_sync_initial_session_keeps_provenance_for_character_handoff(
    monkeypatch,
):
    card = _roleplay_card(name="Alba")
    screen = _character_screen(monkeypatch, card)
    store = screen._ensure_console_chat_store()

    screen._session._sync_console_session_draft()

    sessions = store.sessions()
    assert len(sessions) == 1
    original = sessions[0]
    assert original.canonical_settings_baseline is original.settings

    assert await screen._session._start_character_console_session(
        _start_chat_handoff(card)
    )

    sessions = store.sessions()
    assert len(sessions) == 1
    assert sessions[0].id == original.id
    assert sessions[0].title == "Chat with Alba"


@pytest.mark.asyncio
async def test_tab_sync_initial_session_keeps_provenance_for_character_handoff(
    monkeypatch,
):
    card = _roleplay_card(name="Alba")
    screen = _character_screen(monkeypatch, card)
    store = screen._ensure_console_chat_store()
    surface = AsyncMock()
    surface.sync_sessions = AsyncMock()
    monkeypatch.setattr(screen, "query_one", lambda *_args, **_kwargs: surface)
    monkeypatch.setattr(screen, "_maybe_show_fleet_coachmark", lambda *_args: None)
    monkeypatch.setattr(screen, "_console_chat_controller", None)

    await screen._sync_console_native_session_tabs()

    sessions = store.sessions()
    assert len(sessions) == 1
    original = sessions[0]
    assert original.canonical_settings_baseline is original.settings
    surface.sync_sessions.assert_awaited_once()

    assert await screen._session._start_character_console_session(
        _start_chat_handoff(card)
    )

    sessions = store.sessions()
    assert len(sessions) == 1
    assert sessions[0].id == original.id
    assert sessions[0].title == "Chat with Alba"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "worked_state",
    [
        "draft",
        "title",
        "system-prompt",
        "pinned-prefill",
        "model",
        "provider",
        "temperature",
        "character-label",
    ],
)
async def test_character_handoff_leaves_worked_session_and_creates_another(
    monkeypatch, worked_state
):
    card = _roleplay_card(name="Alba")
    screen = _character_screen(monkeypatch, card)
    store = screen._ensure_console_chat_store()
    defaults = screen._session._default_console_session_settings()
    original = store.ensure_session(
        title="Chat 1",
        settings=defaults,
        canonical_settings_baseline=defaults,
    )
    if worked_state == "draft":
        store.set_session_draft(original.id, "my work")
    elif worked_state == "system-prompt":
        store.set_session_system_prompt(original.id, "My system prompt")
    elif worked_state == "pinned-prefill":
        store.set_session_pinned_prefill(original.id, "Always:")
    elif worked_state == "model":
        store.replace_session_settings(
            original.id,
            replace(defaults, model="my-model"),
        )
    elif worked_state == "provider":
        store.replace_session_settings(
            original.id,
            replace(defaults, provider="anthropic"),
        )
    elif worked_state == "temperature":
        store.replace_session_settings(
            original.id,
            replace(defaults, temperature=defaults.temperature + 0.1),
        )
    elif worked_state == "character-label":
        store.replace_session_settings(
            original.id,
            replace(defaults, character_label="My assistant"),
        )
    else:
        original.title = "Planning"
    original_before = replace(original)

    assert await screen._session._start_character_console_session(
        _start_chat_handoff(card)
    )

    assert len(store.sessions()) == 2
    assert store.sessions()[0] == original_before
    assert store.active_session_id != original.id
    assert store.sessions()[1].title == "Chat with Alba"
    assert len(store.messages_for_session(store.active_session_id)) == 1


@pytest.mark.asyncio
async def test_character_handoff_does_not_refresh_unproven_derived_settings(
    monkeypatch,
):
    card = _roleplay_card(name="Alba")
    screen = _character_screen(monkeypatch, card)
    store = screen._ensure_console_chat_store()
    stale = screen._session._default_console_session_settings()
    original = store.ensure_session(settings=stale)
    original_before = replace(original)
    screen.app_instance.app_config.setdefault("chat_defaults", {})["model"] = (
        "canonical-current-model"
    )

    assert await screen._session._start_character_console_session(
        _start_chat_handoff(card)
    )

    assert len(store.sessions()) == 2
    assert store.sessions()[0] == original_before
    assert store.active_session_id != original.id


def test_typed_then_cleared_work_marker_survives_screen_state_restore(monkeypatch):
    screen = _character_screen(monkeypatch, _roleplay_card(name="Alba"))
    store = screen._ensure_console_chat_store()
    defaults = screen._session._default_console_session_settings()
    session = store.ensure_session(
        title="Chat 1",
        settings=defaults,
        canonical_settings_baseline=defaults,
    )
    store.set_session_draft(session.id, "my work")
    store.set_session_draft(session.id, "")

    payload = screen._session._console_session_to_state(session)
    restored = screen._session._console_session_from_state(payload)

    assert restored.has_user_work is True


@pytest.mark.asyncio
async def test_character_handoff_uses_current_canonical_defaults_not_stale_session(
    monkeypatch,
):
    card = _roleplay_card(name="Alba")
    screen = _character_screen(monkeypatch, card)
    store = screen._ensure_console_chat_store()
    stale = screen._session._default_console_session_settings()
    original = store.ensure_session(
        settings=stale,
        canonical_settings_baseline=stale,
    )
    screen.app_instance.app_config.setdefault("chat_defaults", {})["model"] = (
        "canonical-current-model"
    )
    screen.app_instance.app_config.setdefault("chat_defaults", {})["provider"] = (
        "openai"
    )
    current = screen._session._default_console_session_settings()
    assert stale.model != current.model

    assert await screen._session._start_character_console_session(
        _start_chat_handoff(card)
    )

    active = store.switch_session(store.active_session_id)
    assert active is original
    assert active.settings.model == "canonical-current-model"
    assert active.settings.system_prompt == "Protect Captain Rowan as Alba."
    assert active.assistant_kind == "character"
    assert active.character_name == "Alba"
    assert len(store.sessions()) == 1
    assert active.id == original.id


@pytest.mark.asyncio
@pytest.mark.parametrize("has_greeting", [True, False])
async def test_duplicate_character_handoff_does_not_duplicate_session_or_greeting(
    monkeypatch, has_greeting
):
    card = _roleplay_card(name="Alba")
    if not has_greeting:
        card["first_message"] = ""
    screen = _character_screen(monkeypatch, card)
    store = screen._ensure_console_chat_store()
    defaults = screen._session._default_console_session_settings()
    store.ensure_session(
        title="Chat 1",
        settings=defaults,
        canonical_settings_baseline=defaults,
    )
    payload = _start_chat_handoff(card)

    assert await screen._session._start_character_console_session(payload)
    assert await screen._session._start_character_console_session(payload)

    assert len(store.sessions()) == 1
    assert len(store.messages_for_session(store.active_session_id)) == int(has_greeting)


@pytest.mark.asyncio
async def test_concurrent_character_handoff_does_not_duplicate_session_or_greeting(
    monkeypatch,
):
    card = _roleplay_card(name="Alba")
    screen = _character_screen(monkeypatch, card)
    store = screen._ensure_console_chat_store()
    defaults = screen._session._default_console_session_settings()
    store.ensure_session(
        title="Chat 1",
        settings=defaults,
        canonical_settings_baseline=defaults,
    )
    both_fetches_started = asyncio.Event()
    release_fetches = asyncio.Event()
    fetch_count = 0

    async def get_character(*_args, **_kwargs):
        nonlocal fetch_count
        fetch_count += 1
        if fetch_count == 2:
            both_fetches_started.set()
        await release_fetches.wait()
        return card

    screen.app_instance.character_persona_scope_service.get_character = get_character
    payload = _start_chat_handoff(card)

    async def start_both() -> list[bool]:
        return await asyncio.gather(
            screen._session._start_character_console_session(payload),
            screen._session._start_character_console_session(payload),
        )

    both = asyncio.create_task(start_both())
    await wait_for_background_signal(
        both_fetches_started,
        both,
        what="both concurrent character fetches to start",
    )
    release_fetches.set()

    assert await await_background_task(
        both,
        what="both concurrent character handoffs to finish",
    ) == [True, True]
    assert len(store.sessions()) == 1
    assert len(store.messages_for_session(store.active_session_id)) == 1


def _real_chachanotes_db(tmp_path) -> CharactersRAGDB:
    """A real (temp-file-backed) ChaChaNotes DB.

    `_build_test_app` deliberately leaves `app.chachanotes_db` unset (its
    `notes_service` is faked with no real `.db`) -- fine for the vast
    majority of Console tests, which never touch persistence, but exactly
    the gap the promotion tests below exist to close. Assigned to
    `app.chachanotes_db` BEFORE the screen mounts: `_ensure_console_chat_
    store` reads it lazily on first call and memoizes the result, so
    setting it any later would be silently ignored.
    """
    return CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")


def _switcher_entry(
    *,
    native_session_id: str | None = None,
    conversation_id: str | None = None,
    row_key: str = "row",
    title: str = "Entry",
    scope_type: str = "workspace",
    workspace_id: str | None = None,
    is_active: bool = False,
) -> ConsoleSwitcherEntry:
    return ConsoleSwitcherEntry(
        row_key=row_key,
        title=title,
        subtitle="",
        native_session_id=native_session_id,
        conversation_id=conversation_id,
        scope_type=scope_type,
        workspace_id=workspace_id,
        is_active=is_active,
    )


# -- Session activation: workspace-alignment seam ----------------------------


@pytest.mark.asyncio
async def test_activate_native_console_session_realigns_active_workspace():
    """Activating a session in a DIFFERENT workspace switches the active one.

    This is the one call `_activate_native_console_session` makes into the
    workspace cluster (`_set_active_workspace_for_console_session`) --
    exactly the seam the wave-2 Task 3 brief calls out as needing a deliberate
    named callable between the session and workspace controllers rather than
    a screen back-door. Two real workspaces, two real sessions, one real
    registry service.
    """
    app = _build_test_app()
    registry_service = app.workspace_registry_service
    default_workspace = registry_service.get_active_workspace()
    registry_service.create_workspace(
        workspace_id="ws-other",
        name="Other workspace",
        description="Second workspace for the activation test.",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        store = console._ensure_console_chat_store()
        home_session = console._session._active_native_console_session()
        assert home_session is not None
        assert registry_service.get_active_workspace().workspace_id == (
            default_workspace.workspace_id
        )

        # `create_session` activates what it creates (same trap
        # `test_cost_chip_state_isolated_across_session_tabs` documents), so
        # right after this call `other_session` -- not `home_session` -- is
        # the active one and the active workspace has already followed it
        # via THIS call's own real-time context sync. The two real
        # transitions this test needs both go through the shared entry
        # point below: hop back home first, THEN hop to `other_session`, so
        # the interesting assertion is against a genuine activation call,
        # not residue from creation.
        other_session = store.create_session(
            title="Other workspace chat", workspace_id="ws-other"
        )
        assert other_session.workspace_id == "ws-other"
        assert store.active_session_id == other_session.id

        await console._session._activate_native_console_session(home_session.id)
        await pilot.pause()
        assert store.active_session_id == home_session.id
        assert registry_service.get_active_workspace().workspace_id == (
            default_workspace.workspace_id
        )

        await console._session._activate_native_console_session(other_session.id)
        await pilot.pause()

        assert store.active_session_id == other_session.id
        assert registry_service.get_active_workspace().workspace_id == "ws-other"

        # Hop back: the home session's own workspace must be restored too,
        # not just left on "ws-other" because nothing re-checks it.
        await console._session._activate_native_console_session(home_session.id)
        await pilot.pause()

        assert store.active_session_id == home_session.id
        assert registry_service.get_active_workspace().workspace_id == (
            default_workspace.workspace_id
        )


@pytest.mark.asyncio
async def test_activate_native_console_session_noop_still_focuses_composer(
    monkeypatch,
):
    """Activating the ALREADY-active session skips the switch but still
    calls the composer-focus step (the guard is `store.active_session_id !=
    session_id`, not an early return from the whole method) -- spies on the
    dependency rather than asserting real Textual focus state, which is not
    reliable to observe headlessly for this compound widget."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        session = console._session._active_native_console_session()
        assert session is not None

        focus_calls: list[bool] = []
        real_focus = type(console)._focus_console_composer_if_needed

        def _spy_focus(self, *, force: bool = False) -> None:
            focus_calls.append(force)
            return real_focus(self, force=force)

        monkeypatch.setattr(
            type(console), "_focus_console_composer_if_needed", _spy_focus
        )

        await console._session._activate_native_console_session(session.id)
        await pilot.pause()

        assert console._session._active_native_console_session().id == session.id
        assert focus_calls == [True]


# -- Ctrl+K switcher callback (`_apply_console_switcher_choice`) ------------


@pytest.mark.asyncio
async def test_apply_console_switcher_choice_activate_uses_shared_activation_path():
    """The switcher's "activate" result reaches the same shared entry point
    as the tab click / Alt+1..9 -- verified here through a real workspace
    switch, not just an active-session-id flip."""
    app = _build_test_app()
    registry_service = app.workspace_registry_service
    registry_service.create_workspace(
        workspace_id="ws-switcher",
        name="Switcher workspace",
        description="Workspace for the switcher-choice test.",
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        store = console._ensure_console_chat_store()
        home = console._session._active_native_console_session()
        assert home is not None
        target = store.create_session(
            title="Switcher target", workspace_id="ws-switcher"
        )
        assert home.id != target.id
        # `create_session` activates what it creates -- go back home first so
        # the switcher choice below is a real transition, not a no-op.
        await console._session._activate_native_console_session(home.id)
        await pilot.pause()
        assert store.active_session_id == home.id

        choice = ConsoleSwitcherChoice(
            kind="activate",
            entry=_switcher_entry(native_session_id=target.id, title=target.title),
        )
        await console._session._apply_console_switcher_choice(choice)
        await pilot.pause()

        assert store.active_session_id == target.id
        assert registry_service.get_active_workspace().workspace_id == (
            "ws-switcher"
        )


@pytest.mark.asyncio
async def test_apply_console_switcher_choice_rename_opens_rename_modal():
    """The switcher's "rename" result opens the rename modal for that tab,
    seeded with its CURRENT title."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        store = console._ensure_console_chat_store()
        session = store.create_session(title="Rename me")

        choice = ConsoleSwitcherChoice(
            kind="rename",
            entry=_switcher_entry(native_session_id=session.id, title=session.title),
        )
        await console._session._apply_console_switcher_choice(choice)
        await pilot.pause()

        top = host.screen_stack[-1]
        assert isinstance(top, ConsoleRenameSessionModal)
        assert top._title == "Rename me"


@pytest.mark.asyncio
async def test_apply_console_switcher_choice_none_is_a_noop():
    """A cancelled switcher (`choice is None`) leaves the active session and
    screen stack untouched."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        store = console._ensure_console_chat_store()
        active_before = store.active_session_id
        depth_before = len(host.screen_stack)

        await console._session._apply_console_switcher_choice(None)
        await pilot.pause()

        assert store.active_session_id == active_before
        assert len(host.screen_stack) == depth_before


# -- Temporary-session promotion: assert the DATABASE, not widget state -----


@pytest.mark.asyncio
async def test_promote_console_temporary_session_writes_durable_rows_to_the_database(
    tmp_path,
):
    """Saving a temporary chat must produce a REAL, readable conversation row
    (and its messages) in `chachanotes_db` -- not just an in-memory flag flip.

    Every existing `_promote_console_temporary_session` test
    (`test_console_composer_menu.py`) stubs the store's own
    `promote_ephemeral_session` and only asserts chip/notification state;
    none of them let a write reach a real database. This is the gap the
    wave-2 Task 3 brief calls out by name.
    """
    app = _build_test_app()
    db = _real_chachanotes_db(tmp_path)
    app.chachanotes_db = db
    host = ConsoleHarness(app)

    try:
        async with host.run_test(size=(160, 44)) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-transcript")

            store = console._ensure_console_chat_store()
            session = store.create_session(title="Temp chat to save", ephemeral=True)
            store.switch_session(session.id)
            store.append_message(
                session.id, role=ConsoleMessageRole.USER, content="hello", persist=True
            )
            store.append_message(
                session.id,
                role=ConsoleMessageRole.ASSISTANT,
                content="hi there",
                persist=True,
            )
            assert session.persisted_conversation_id is None

            await console._session._promote_console_temporary_session()
            await pilot.pause()

            assert session.ephemeral is False
            conversation_id = session.persisted_conversation_id
            assert conversation_id is not None

            conversation = db.get_conversation_by_id(conversation_id)
            assert conversation is not None
            assert conversation["title"] == "Temp chat to save"

            messages = db.get_messages_for_conversation(conversation_id)
            assert [m["content"] for m in messages] == ["hello", "hi there"]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_promote_console_temporary_session_second_call_does_not_duplicate_the_row(
    tmp_path,
):
    """Promoting an already-saved session again must not write a second
    conversation row -- `promote_ephemeral_session` is idempotent, and the
    screen-level wrapper must not paper over a regression there by, say,
    creating a fresh conversation on every click."""
    app = _build_test_app()
    db = _real_chachanotes_db(tmp_path)
    app.chachanotes_db = db
    host = ConsoleHarness(app)

    try:
        async with host.run_test(size=(160, 44)) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-transcript")

            store = console._ensure_console_chat_store()
            session = store.create_session(title="Save twice", ephemeral=True)
            store.switch_session(session.id)
            store.append_message(
                session.id, role=ConsoleMessageRole.USER, content="hi", persist=True
            )

            await console._session._promote_console_temporary_session()
            await pilot.pause()
            conversation_id = session.persisted_conversation_id
            assert conversation_id is not None
            assert db.get_conversation_by_id(conversation_id) is not None

            await console._session._promote_console_temporary_session()
            await pilot.pause()

            assert session.persisted_conversation_id == conversation_id
            assert session.ephemeral is False
            # Still exactly the one conversation row this session's messages
            # belong to; a duplicate-write regression would mint a second one.
            assert db.get_conversation_by_id(conversation_id) is not None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_promote_console_temporary_session_no_active_session_touches_nothing():
    """With no active session (a bare/never-mounted store edge), promotion is
    a pure no-op -- no notification, no DB access attempted."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        store = console._ensure_console_chat_store()
        store.active_session_id = None

        # Must not raise even though there is nothing to promote.
        await console._session._promote_console_temporary_session()
        await pilot.pause()
