"""P1g Task 4: `ChatScreen` caches the "what's in play" chat-dictionary
summary on conversation/character change and feeds it into the Console
inspector build with ZERO DB I/O on recompose.

`refresh_active_dictionaries_summary()` is the ONLY place that is allowed to
call the scope service's `summarize_active_dictionaries`; a bare
`_build_console_inspector_state()` call (the same path every Console
recompose/refresh goes through) must read only the cached
`self._active_dictionaries_summary` set by the last refresh.
"""

import asyncio
from types import SimpleNamespace

import pytest

from Tests.UI.test_destination_shells import _build_test_app
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import ConsoleHarness
from tldw_chatbook.Chat.console_display_state import ConsoleDisplayRow
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


class _FakeDictionaryScopeService:
    """Records every `summarize_active_dictionaries` call; returns a canned
    summary. Used to prove the DB-shaped summarize call happens ONLY on
    `refresh_active_dictionaries_summary()`, never on an inspector build."""

    def __init__(self, summary):
        self.summary = summary
        self.calls: list[tuple] = []

    async def summarize_active_dictionaries(self, conversation_id, character_id, mode="local"):
        self.calls.append((conversation_id, character_id, mode))
        return self.summary


# --- Hard-rule tests (brief scenarios a + b) --------------------------------
# `refresh_active_dictionaries_summary()` calls `_sync_console_control_bar()`
# to push the rebuilt inspector state to the mounted widget, which needs a
# real (running) app context -- so these run inside `ConsoleHarness`, the
# same Console `ChatScreen` mounting harness `test_console_persistent_rails`/
# `test_product_maturity_gate1_core_loop_screen_adaptation` already use.

@pytest.mark.asyncio
async def test_refresh_caches_summary_and_build_projects_dictionary_rows():
    app = _build_test_app()
    app.current_chat_conversation_id = "conv-1"
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
    app.current_chat_conversation_id = "conv-1"
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
    assert app.current_chat_conversation_id is None
    assert app.current_chat_active_character_data is None

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]

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

        await screen.refresh_active_dictionaries_summary()

        assert service.calls == []
        assert screen._active_dictionaries_summary == {"dictionaries": []}


@pytest.mark.asyncio
async def test_empty_dictionaries_summary_renders_empty_row():
    app = _build_test_app()
    app.current_chat_conversation_id = "conv-1"
    service = _FakeDictionaryScopeService({"dictionaries": [], "source": "local"})
    app.chat_dictionary_scope_service = service

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]

        await screen.refresh_active_dictionaries_summary()

        rows = screen._console_dictionary_inspector_rows()
        assert rows == (ConsoleDisplayRow("No dictionaries in play", ""),)


@pytest.mark.asyncio
async def test_shadowed_and_disabled_suffixes_render_on_the_row_value():
    app = _build_test_app()
    app.current_chat_conversation_id = "conv-1"
    app.current_chat_active_character_data = {"id": 7, "name": "Nyx"}
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

        await screen.refresh_active_dictionaries_summary()

        assert service.calls == [("conv-1", 7, "local")]
        rows = screen._console_dictionary_inspector_rows()
        assert rows == (
            ConsoleDisplayRow("Slang", "from conversation"),
            ConsoleDisplayRow("Slang", "from character (shadowed)"),
            ConsoleDisplayRow("Lore", "from character (disabled)"),
        )


@pytest.mark.asyncio
async def test_actions_reflect_conversation_and_attach_state():
    app = _build_test_app()
    # No conversation, but a character is active: attach must be disabled
    # (with the recovery copy) and detach has no conversation-source dict to
    # detach either.
    app.current_chat_active_character_data = {"id": 9}
    summary = {
        "dictionaries": [
            {"name": "Lore", "source": "character", "enabled": True, "entry_count": 1, "shadowed": False},
        ],
        "source": "local",
    }
    service = _FakeDictionaryScopeService(summary)
    app.chat_dictionary_scope_service = service

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]

        await screen.refresh_active_dictionaries_summary()

        actions = screen._console_dictionary_inspector_actions()
        attach = next(a for a in actions if a.widget_id == "console-inspector-dictionaries-attach")
        detach = next(a for a in actions if a.widget_id == "console-inspector-dictionaries-detach")
        assert attach.enabled is False
        assert attach.disabled_reason == "Start or load a conversation first"
        assert detach.enabled is False

        # Once a conversation exists and owns a dictionary, both actions
        # light up.
        app.current_chat_conversation_id = "conv-2"
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


# --- Reactive-watcher wiring (app.py owns the reactives) --------------------

def test_conversation_and_character_change_schedule_a_refresh_worker(monkeypatch):
    """`current_chat_conversation_id`/`current_chat_active_character_data`
    live on the app (see `app.py:2296/2299`), so `ChatScreen` cannot define
    its own `watch_*` for them. The app's watchers must dispatch to the
    active Console screen's `refresh_active_dictionaries_summary()` --
    mirroring the existing `isinstance(self.screen, ChatScreen)` pattern used
    by `set_current_chat_is_streaming`.

    Uses a `type(app).screen` monkeypatch rather than mounting: `TldwCli` is
    the real running app in production (its own screen stack), but no test
    harness in this suite ever actually runs `TldwCli` itself -- only a
    lightweight wrapper `App` (`ConsoleHarness`) that pushes `ChatScreen`
    onto ITS OWN stack. There is no existing precedent for driving this
    app-level watcher through a live screen stack.
    """
    app = _build_test_app()
    screen = ChatScreen(app)
    monkeypatch.setattr(type(app), "screen", property(lambda self: screen))

    scheduled = []

    def fake_run_worker(work, **kwargs):
        scheduled.append(kwargs)
        if asyncio.iscoroutine(work):
            work.close()  # never actually awaited by this fake
        return SimpleNamespace(cancel=lambda: None)

    app.run_worker = fake_run_worker

    app.current_chat_conversation_id = "conv-live"
    assert len(scheduled) == 1
    assert scheduled[0].get("group") == "console-dictionary-summary"

    app.current_chat_active_character_data = {"id": 3, "name": "Nyx"}
    assert len(scheduled) == 2


def test_conversation_id_watcher_is_a_no_op_when_value_is_unchanged(monkeypatch):
    """Textual's `reactive(...)` factory defaults `init=True`, so the very
    first touch of `current_chat_conversation_id` on a fresh app instance
    fires the watcher once with old==new==default (None) before any real
    change. The watcher must not schedule a refresh for that no-op call."""
    app = _build_test_app()
    screen = ChatScreen(app)
    monkeypatch.setattr(type(app), "screen", property(lambda self: screen))

    scheduled = []
    app.run_worker = lambda work, **kwargs: scheduled.append(kwargs) or (
        work.close() if asyncio.iscoroutine(work) else None
    )

    # First-ever touch: internally triggers the init-quirk call AND (since
    # None != None is False here) no real-change call either.
    assert app.current_chat_conversation_id is None
    assert scheduled == []
