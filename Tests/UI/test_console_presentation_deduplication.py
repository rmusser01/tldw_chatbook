"""Presentation shares repeated reads without extending storage authority."""

import sys
from types import SimpleNamespace

import pytest

from Tests.UI.test_console_dictation import _ready_host
from Tests.UI.test_console_run_tick_workspace_reads import APP_SIZE, _settled_console
from Tests.UI.test_console_workspace_controller import _workspace_controller
from tldw_chatbook.UI.Console_Modules import workspace
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_run_inspector import ConsoleRunInspector
from tldw_chatbook.Widgets.Console.console_send_authority_summary import (
    ConsoleSendAuthoritySummary,
)


def _appearances(monkeypatch):
    calls = []
    values = {"known": SimpleNamespace(icon="icon", color="color")}

    def read(ids):
        calls.append(tuple(ids))
        return {key: values[key] for key in ids if key in values}

    service = SimpleNamespace(get_conversation_appearances=read)
    app = SimpleNamespace(local_chat_conversation_service=service)
    controller = _workspace_controller(app_instance=app)
    clock = [100.0]
    monkeypatch.setattr(workspace.time, "monotonic", lambda: clock[0])
    return controller, app, calls, values, clock


@pytest.mark.parametrize("absent_id", ["missing", "deleted"])
def test_absent_appearance_ids_share_the_existing_display_ttl(monkeypatch, absent_id):
    controller, _app, calls, _values, _clock = _appearances(monkeypatch)
    for _ in range(3):
        assert controller._console_conversation_appearance_map(
            ["known", absent_id]
        ) == {"known": ("icon", "color")}
    assert calls == [tuple(sorted((absent_id, "known")))]
    # A genuinely new ID still causes a fresh batch; missing IDs remain absent
    # from the display map so a row's existing fallback is never overwritten.
    controller._console_conversation_appearance_map([absent_id, "new"])
    controller._console_conversation_appearance_map([absent_id, "new"])
    assert len(calls) == 2


def test_appearance_expiry_and_explicit_invalidation_read_again(monkeypatch):
    controller, _app, calls, values, clock = _appearances(monkeypatch)
    assert controller._console_conversation_appearance_map(["missing"]) == {}
    values["missing"] = SimpleNamespace(icon="new", color="")
    clock[0] += workspace.CONSOLE_APPEARANCE_MAP_CACHE_TTL_SECONDS + 1
    assert controller._console_conversation_appearance_map(["missing"]) == {
        "missing": ("new", "")
    }
    controller._console_appearance_map_cache = None
    controller._console_appearance_map_cached_at = 0.0
    controller._console_conversation_appearance_map(["missing"])
    assert len(calls) == 3


def test_deleted_appearance_disappears_after_display_ttl(monkeypatch):
    controller, _app, calls, values, clock = _appearances(monkeypatch)
    assert controller._console_conversation_appearance_map(["known"])
    values.clear()
    clock[0] += workspace.CONSOLE_APPEARANCE_MAP_CACHE_TTL_SECONDS + 1
    assert controller._console_conversation_appearance_map(["known"]) == {}
    assert controller._console_conversation_appearance_map(["known"]) == {}
    assert len(calls) == 2


def test_new_appearance_queries_do_not_renew_an_older_missing_id(monkeypatch):
    controller, _app, calls, values, clock = _appearances(monkeypatch)
    assert controller._console_conversation_appearance_map(["missing"]) == {}
    clock[0] += workspace.CONSOLE_APPEARANCE_MAP_CACHE_TTL_SECONDS / 2
    controller._console_conversation_appearance_map(["known"])
    values["missing"] = SimpleNamespace(icon="arrived", color="")
    assert controller._console_conversation_appearance_map(["missing"])["missing"] == (
        "arrived",
        "",
    )
    assert len(calls) == 3


@pytest.mark.parametrize("stale_display_reads", [False, True])
def test_replacing_appearance_service_drops_the_previous_display_map(
    monkeypatch, stale_display_reads
):
    controller, app, calls, _values, _clock = _appearances(monkeypatch)
    assert controller._console_conversation_appearance_map(["known"])
    controller._stale_display_reads_on = stale_display_reads
    replacement_calls = []

    def replacement(ids):
        replacement_calls.append(tuple(ids))
        return {}

    app.local_chat_conversation_service = SimpleNamespace(
        get_conversation_appearances=replacement
    )
    assert controller._console_conversation_appearance_map(["known"]) == {}
    assert len(calls) == len(replacement_calls) == 1


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
async def test_original_rail_controls_build_only_the_consumed_inspectors():
    _app, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        screen = await _settled_console(host, pilot)
        code = ChatScreen._build_console_inspector_state.__code__
        builds = []

        def observe(frame, event, _argument):
            if event == "call" and frame.f_code is code:
                builds.append(frame.f_back.f_code.co_name)

        previous = sys.getprofile()
        sys.setprofile(observe)
        try:
            with screen._workspace.tick_workspace_build_scope():
                assert screen._sync_console_rail_and_controls() is True
        finally:
            sys.setprofile(previous)
        # Agent section needs its rail state; the controls consume one inspector
        # directly and supply it to their rail projection. No leading throwaway.
        assert len(builds) == 2, builds
        inspector = screen.query_one(
            "#console-run-inspector-state", ConsoleRunInspector
        )
        summary = screen.query_one(
            "#console-send-authority-summary", ConsoleSendAuthoritySummary
        )
        assert inspector.state == summary.last_state
        assert screen._last_console_rail_state == screen._current_console_rail_state(
            inspector_state=inspector.state
        )
