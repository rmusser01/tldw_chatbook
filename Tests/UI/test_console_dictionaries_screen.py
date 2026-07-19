"""P1g Task 4: `ChatScreen` caches the "what's in play" chat-dictionary
summary on native Console session change and feeds it into the Console
inspector build with ZERO DB I/O on recompose.

`refresh_active_dictionaries_summary()` is the ONLY place that is allowed to
call the scope service's `summarize_active_dictionaries`; a bare
`_build_console_inspector_state()` call (the same path every Console
recompose/refresh goes through) must read only the cached
`self._active_dictionaries_summary` set by the last refresh.

Fix-wave (Critical, Task 4 review): the original wiring sourced the current
conversation/character from the APP-LEVEL `current_chat_conversation_id`/
`current_chat_active_character_data` reactives and recomputed via app.py
watchers on those reactives. Those reactives are written ONLY by the
*legacy* sidebar chat flow (`Event_Handlers/Chat_Events/chat_events.py`) --
the native Console (`ChatScreen` + `ConsoleChatStore`) never touches them, so
in the real app the watcher never fired and the summary stayed permanently
`None` ("No active chat"), despite every test in the original version of
this file passing (they poked the app reactive directly instead of driving a
real native-session change).

The fix rewires both the SOURCE (the active native Console session's
`persisted_conversation_id`, via `_active_console_dictionary_scope_ids()` /
`_current_console_rail_conversation_id()`) and the TRIGGER
(`_sync_native_console_chat_ui()`, the central Console UI-sync entrypoint
that runs on every native session switch/resume). The tests below drive a
REAL native session switch (`_activate_native_console_session` via a session
tab click, exactly as a user would) and assert the built inspector state
tracks whichever session is actually active -- this is the scenario the
original tests missed entirely.
"""

import pytest

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import ConsoleHarness
from tldw_chatbook.Chat.console_display_state import ConsoleDisplayRow
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


class _FakeDictionaryScopeService:
    """Records every `summarize_active_dictionaries` call; returns a single
    canned summary regardless of arguments. Used to prove the DB-shaped
    summarize call happens ONLY on `refresh_active_dictionaries_summary()`,
    never on an inspector build."""

    def __init__(self, summary):
        self.summary = summary
        self.calls: list[tuple] = []

    async def summarize_active_dictionaries(self, conversation_id, character_id, mode="local"):
        self.calls.append((conversation_id, character_id, mode))
        return self.summary


class _PerConversationFakeDictionaryScopeService:
    """Returns a distinct canned summary per conversation id (default: no
    dictionaries for unknown ids); records every call. Lets a test prove the
    summary genuinely tracks whichever native Console session is ACTIVE,
    rather than a single global value."""

    def __init__(self, summaries_by_conversation_id: dict):
        self.summaries_by_conversation_id = summaries_by_conversation_id
        self.calls: list[tuple] = []

    async def summarize_active_dictionaries(self, conversation_id, character_id, mode="local"):
        self.calls.append((conversation_id, character_id, mode))
        return self.summaries_by_conversation_id.get(conversation_id, {"dictionaries": []})


async def _wait_for_active_session_id(store, pilot, expected_session_id: str, *, attempts: int = 40) -> None:
    """Wait for the Console store to report the expected active session."""
    for _ in range(attempts):
        if store.active_session_id == expected_session_id:
            return
        await pilot.pause(0.05)
    raise AssertionError(
        "Console active session did not match expected session. "
        f"expected={expected_session_id!r}; active={store.active_session_id!r}"
    )


def _active_native_session(console: ChatScreen):
    store = console._ensure_console_chat_store()
    return next(s for s in store.sessions() if s.id == store.active_session_id)


# --- Hard-rule tests (brief scenarios a + b) --------------------------------
# `refresh_active_dictionaries_summary()` calls `_sync_console_control_bar()`
# to push the rebuilt inspector state to the mounted widget, which needs a
# real (running) app context -- so these run inside `ConsoleHarness`, the
# same Console `ChatScreen` mounting harness `test_console_persistent_rails`/
# `test_product_maturity_gate1_core_loop_screen_adaptation` already use.

@pytest.mark.asyncio
async def test_refresh_caches_summary_and_build_projects_dictionary_rows():
    app = _build_test_app()
    summary = {
        "dictionaries": [
            {
                "name": "Slang",
                "source": "conversation",
                "enabled": True,
                "entry_count": 3,
                "shadowed": False,
            },
        ],
        "source": "local",
    }
    service = _FakeDictionaryScopeService(summary)
    app.chat_dictionary_scope_service = service

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        _active_native_session(screen).persisted_conversation_id = "conv-1"

        await screen.refresh_active_dictionaries_summary()

        assert screen._active_dictionaries_summary == summary
        assert service.calls == [("conv-1", None, "local")]

        inspector_state = screen._build_console_inspector_state(None)
        assert inspector_state.dictionary_rows == (
            ConsoleDisplayRow("Slang", "from conversation"),
        )


@pytest.mark.asyncio
async def test_build_console_inspector_state_never_re_queries_the_summarize_service():
    app = _build_test_app()
    summary = {
        "dictionaries": [
            {
                "name": "Slang",
                "source": "conversation",
                "enabled": True,
                "entry_count": 3,
                "shadowed": False,
            },
        ],
        "source": "local",
    }
    service = _FakeDictionaryScopeService(summary)
    app.chat_dictionary_scope_service = service

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        _active_native_session(screen).persisted_conversation_id = "conv-1"

        await screen.refresh_active_dictionaries_summary()
        assert len(service.calls) == 1

        # A bare, recompose-equivalent build must read only the cache -- no
        # re-invocation of the summarize service.
        screen._build_console_inspector_state(None)
        screen._build_console_inspector_state(None)
        screen._build_console_inspector_state(None)

        assert len(service.calls) == 1


# --- Guard / edge-case coverage ---------------------------------------------

@pytest.mark.asyncio
async def test_refresh_with_no_service_and_no_chat_caches_empty():
    app = _build_test_app()
    app.chat_dictionary_scope_service = None

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        # Fresh default native session has no persisted conversation yet.
        assert _active_native_session(screen).persisted_conversation_id is None

        await screen.refresh_active_dictionaries_summary()

        assert screen._active_dictionaries_summary == {"dictionaries": []}
        rows = screen._console_dictionary_inspector_rows()
        assert rows == (ConsoleDisplayRow("No active chat", ""),)


@pytest.mark.asyncio
async def test_refresh_with_service_but_no_active_chat_caches_empty_without_calling_service():
    app = _build_test_app()
    service = _FakeDictionaryScopeService({"dictionaries": []})
    app.chat_dictionary_scope_service = service

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        assert _active_native_session(screen).persisted_conversation_id is None

        await screen.refresh_active_dictionaries_summary()

        assert service.calls == []
        assert screen._active_dictionaries_summary == {"dictionaries": []}


@pytest.mark.asyncio
async def test_empty_dictionaries_summary_renders_empty_row():
    app = _build_test_app()
    service = _FakeDictionaryScopeService({"dictionaries": [], "source": "local"})
    app.chat_dictionary_scope_service = service

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        _active_native_session(screen).persisted_conversation_id = "conv-1"

        await screen.refresh_active_dictionaries_summary()

        rows = screen._console_dictionary_inspector_rows()
        assert rows == (ConsoleDisplayRow("No dictionaries in play", ""),)


@pytest.mark.asyncio
async def test_shadowed_and_disabled_suffixes_render_on_the_row_value():
    app = _build_test_app()
    summary = {
        "dictionaries": [
            {"name": "Slang", "source": "conversation", "enabled": True, "entry_count": 2, "shadowed": False},
            {"name": "Slang", "source": "character", "enabled": True, "entry_count": 1, "shadowed": True},
            {"name": "Lore", "source": "character", "enabled": False, "entry_count": 5, "shadowed": False},
        ],
        "source": "local",
    }
    service = _FakeDictionaryScopeService(summary)
    app.chat_dictionary_scope_service = service

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        _active_native_session(screen).persisted_conversation_id = "conv-1"

        await screen.refresh_active_dictionaries_summary()

        # Native Console sessions do not yet track a numeric character id
        # (Roleplay P1e Attachments is net-new work) -- character_id is
        # always None on this call, even though the canned summary still
        # contains character-sourced entries (the fake service returns them
        # unconditionally; this proves row *rendering* of shadowed/disabled
        # character entries is unaffected by that).
        assert service.calls == [("conv-1", None, "local")]
        rows = screen._console_dictionary_inspector_rows()
        assert rows == (
            ConsoleDisplayRow("Slang", "from conversation"),
            ConsoleDisplayRow("Slang", "from character (shadowed)"),
            ConsoleDisplayRow("Lore", "from character (disabled)"),
        )


@pytest.mark.asyncio
async def test_actions_reflect_conversation_and_attach_state():
    app = _build_test_app()
    service = _FakeDictionaryScopeService({"dictionaries": [], "source": "local"})
    app.chat_dictionary_scope_service = service

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")

        # No conversation on the active native session: attach must be
        # disabled (with the recovery copy) and detach has no
        # conversation-source dict to detach either.
        assert _active_native_session(screen).persisted_conversation_id is None
        await screen.refresh_active_dictionaries_summary()

        actions = screen._console_dictionary_inspector_actions()
        attach = next(a for a in actions if a.widget_id == "console-inspector-dictionaries-attach")
        detach = next(a for a in actions if a.widget_id == "console-inspector-dictionaries-detach")
        assert attach.enabled is False
        assert attach.disabled_reason == "Start or load a conversation first"
        assert detach.enabled is False

        # Once the active session's conversation exists and owns a
        # dictionary, both actions light up.
        _active_native_session(screen).persisted_conversation_id = "conv-2"
        service.summary = {
            "dictionaries": [
                {"name": "Slang", "source": "conversation", "enabled": True, "entry_count": 1, "shadowed": False},
            ],
            "source": "local",
        }
        await screen.refresh_active_dictionaries_summary()

        actions2 = screen._console_dictionary_inspector_actions()
        attach2 = next(a for a in actions2 if a.widget_id == "console-inspector-dictionaries-attach")
        detach2 = next(a for a in actions2 if a.widget_id == "console-inspector-dictionaries-detach")
        assert attach2.enabled is True
        assert detach2.enabled is True


# --- Real native Console session-switch wiring (the fix under test) --------

@pytest.mark.asyncio
async def test_real_native_console_session_switch_drives_dictionary_summary_per_session():
    """Drives an ACTUAL native Console session switch -- the scenario the
    original (app-reactive) wiring never handled, because clicking a session
    tab never touches `app.current_chat_conversation_id`.

    Creates two native sessions, each bound to a distinct
    `persisted_conversation_id` (one with an attached dictionary, one
    without), and activates each through the real production path: a click
    on `#console-session-tab-{id}`, which `ChatScreen`'s button handler
    routes to `_activate_native_console_session()` ->
    `_sync_native_console_chat_ui()`. Asserts the built inspector state's
    `dictionary_rows` track whichever session is ACTUALLY active, and that
    the DB-backed summarize call fires exactly once per genuine scope
    change (not once per sync pass -- `_sync_native_console_chat_ui` also
    runs on the 0.2s transcript poll timer).
    """
    app = _build_test_app()
    service = _PerConversationFakeDictionaryScopeService(
        {
            "conv-with-dict": {
                "dictionaries": [
                    {
                        "name": "Slang",
                        "source": "conversation",
                        "enabled": True,
                        "entry_count": 3,
                        "shadowed": False,
                    },
                ],
                "source": "local",
            },
            "conv-without-dict": {"dictionaries": [], "source": "local"},
        }
    )
    app.chat_dictionary_scope_service = service

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        console = pilot.app.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        store = console._ensure_console_chat_store()
        session_a_id = store.active_session_id
        assert session_a_id is not None
        _active_native_session(console).persisted_conversation_id = "conv-with-dict"

        # Create and activate a second native session through the real
        # "new chat" action (mirrors how a user opens a second tab).
        await pilot.click("#console-new-chat-tab")
        session_b_id = await _wait_for_active_session_change(store, pilot, session_a_id)
        assert session_b_id != session_a_id
        session_b = next(s for s in store.sessions() if s.id == session_b_id)
        session_b.persisted_conversation_id = "conv-without-dict"

        await _wait_for_selector(console, pilot, f"#console-session-tab-{session_a_id}")
        await _wait_for_selector(console, pilot, f"#console-session-tab-{session_b_id}")

        # --- Switch to session A (has an attached conversation dictionary) ---
        calls_before = len(service.calls)
        await pilot.click(f"#console-session-tab-{session_a_id}")
        await _wait_for_active_session_id(store, pilot, session_a_id)

        assert len(service.calls) == calls_before + 1
        assert service.calls[-1] == ("conv-with-dict", None, "local")

        inspector_state = console._build_console_inspector_state(None)
        assert inspector_state.dictionary_rows == (
            ConsoleDisplayRow("Slang", "from conversation"),
        )

        # No-DB-on-recompose still holds after the switch: repeated bare
        # builds must not re-invoke the summarize service.
        console._build_console_inspector_state(None)
        console._build_console_inspector_state(None)
        assert len(service.calls) == calls_before + 1

        # --- Switch to session B (no dictionaries attached) ---
        calls_before_b = len(service.calls)
        await pilot.click(f"#console-session-tab-{session_b_id}")
        await _wait_for_active_session_id(store, pilot, session_b_id)

        assert len(service.calls) == calls_before_b + 1
        assert service.calls[-1] == ("conv-without-dict", None, "local")

        inspector_state_b = console._build_console_inspector_state(None)
        assert inspector_state_b.dictionary_rows == (
            ConsoleDisplayRow("No dictionaries in play", ""),
        )


async def _wait_for_active_session_change(store, pilot, previous_session_id, *, attempts: int = 40) -> str:
    """Wait for the Console store to activate a session other than
    `previous_session_id` and return its id."""
    for _ in range(attempts):
        active_session_id = store.active_session_id
        if active_session_id is not None and active_session_id != previous_session_id:
            return active_session_id
        await pilot.pause(0.05)
    raise AssertionError(
        "Console active session did not change. "
        f"previous={previous_session_id!r}; active={store.active_session_id!r}"
    )


@pytest.mark.asyncio
async def test_sync_native_console_chat_ui_does_not_resummarize_when_scope_is_unchanged():
    """`_sync_native_console_chat_ui()` is also invoked by the 0.2s
    transcript-poll timer while a run is streaming
    (`_start_console_transcript_sync_timer`). Without a change-guard, every
    poll would re-run the DB-backed summarize call. Repeated syncs against
    the SAME active session's scope must not grow the call count past the
    first one."""
    app = _build_test_app()
    service = _FakeDictionaryScopeService({"dictionaries": [], "source": "local"})
    app.chat_dictionary_scope_service = service

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        console = pilot.app.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _active_native_session(console).persisted_conversation_id = "conv-steady"

        await console._sync_native_console_chat_ui()
        calls_after_first = len(service.calls)
        assert calls_after_first >= 1

        for _ in range(5):
            await console._sync_native_console_chat_ui()

        assert len(service.calls) == calls_after_first
