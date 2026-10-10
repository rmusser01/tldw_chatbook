"""ADR-220: actual region callbacks resolve the current screen at action time."""

from types import SimpleNamespace
import pytest

pytestmark = pytest.mark.bootstrap_profile
from tldw_chatbook.UI.Console_Modules.wiring import build_console_controllers
from tldw_chatbook.UI.Screens import chat_screen


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action,draft,cleared",
    [
        ("cancel", "", True),
        ("cancel", "new draft", True),
        ("cancel", "", False),
        ("send_without_compacting", "", True),
    ],
)
async def test_region_recovery_callbacks_read_live_screen_and_preserve_order(
    monkeypatch, action, draft, cleared
):
    screen = chat_screen.ChatScreen.__new__(chat_screen.ChatScreen)
    screen.app_instance = SimpleNamespace()
    build_console_controllers(
        screen,
        resume_screen_is_torn_down=lambda: chat_screen._console_screen_is_torn_down(
            screen
        ),
        read_resume_asyncio=lambda: chat_screen.asyncio,
        resume_isawaitable=lambda result: chat_screen.inspect.isawaitable(result),
        read_resume_logger=lambda: chat_screen.logger,
        display_pricing_catalog=lambda: chat_screen.spend.pricing_catalog_for_display(
            screen, chat_screen.get_pricing_catalog
        ),
        build_cost_snapshot=lambda *args, **kwargs: chat_screen.build_cost_snapshot(
            *args, **kwargs
        ),
        rag_source_types_accessor=lambda: (),
        rag_top_k_accessor=lambda: 8,
        read_trace_recovery_dispatch=lambda: (
            chat_screen.dispatch_trace_call_recovery_action
        ),
        read_trace_recovery_state=lambda: chat_screen.trace_call_recovery_state,
    )
    events = []
    hold = [object()]
    held = SimpleNamespace(preparation_id="p", executed_draft="held draft")

    def preparation():
        events.append("held")
        return held

    def context_hold(_):
        events.append("hold")
        return hold[0]

    controller = SimpleNamespace(
        trace_call_recovery_preparation=preparation,
        context_compaction_hold=context_hold,
    )
    screen._ensure_console_chat_controller = lambda: controller
    screen._console_terminal_open = False
    region = screen._build_console_center()
    callback = region._on_trace_recovery_action
    state_callback = region._trace_recovery_state_builder
    assert callback.__self__ is screen._session
    assert callback.__func__ is type(screen._session)._dispatch_console_trace_recovery
    assert state_callback.__self__ is screen._session
    assert (
        state_callback.__func__ is type(screen._session)._console_trace_recovery_state
    )
    started = lambda: None
    finished = lambda: None
    screen._start_console_transcript_sync_timer = started
    screen._sync_native_console_chat_ui = finished
    loaded = []
    composer = SimpleNamespace(draft_text=lambda: draft, load_draft=loaded.append)
    screen._console_composer_or_none = lambda: (events.append("composer"), composer)[1]
    result = object()

    async def dispatch(actual_controller, actual_action, preparation_id, **kwargs):
        events.append("dispatch")
        assert actual_controller is controller
        assert (actual_action, preparation_id) == (action, "p")
        assert kwargs == {"on_started": started, "on_finished": finished}
        if cleared:
            hold[0] = None
        return result

    monkeypatch.setattr(chat_screen, "dispatch_trace_call_recovery_action", dispatch)
    assert await callback(action, "p") is result
    assert events[:4] == ["held", "hold", "dispatch", "composer"]
    assert loaded == (
        ["held draft"] if action == "cancel" and cleared and not draft else []
    )
    state = object()

    def project(actual_preparation, *, context_hold):
        assert actual_preparation is held
        assert context_hold is hold[0]
        return state

    monkeypatch.setattr(chat_screen, "trace_call_recovery_state", project)
    assert state_callback() is state
