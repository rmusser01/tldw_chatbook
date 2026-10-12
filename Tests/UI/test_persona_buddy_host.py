"""Production host navigation and explicit Workbench Buddy selection."""

from types import SimpleNamespace

import pytest
from textual.screen import ModalScreen
from textual.widgets import Static

from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
from tldw_chatbook.UI.Navigation.persona_buddy_host import (
    buddy_screen_allowed,
    trusted_console_states,
)


def test_buddy_only_uses_primary_screens_and_hides_under_modal():
    app = SimpleNamespace(splash_screen_active=False)
    primary = BaseAppScreen(app, "personas")
    assert buddy_screen_allowed(app, primary)
    assert not buddy_screen_allowed(app, ModalScreen())
    app.splash_screen_active = True
    assert not buddy_screen_allowed(app, primary)
    app.splash_screen_active = False
    assert not buddy_screen_allowed(app, BaseAppScreen(app, "recovery"))


def test_console_signals_are_source_scoped_across_parallel_runs():
    states = {"a": "streaming", "b": "validating", "c": "failed"}
    controller = SimpleNamespace(
        run_state_for=lambda sid: SimpleNamespace(status=states[sid]),
        has_pending_approval_round=lambda sid: sid == "b",
    )
    runtime = SimpleNamespace(
        chat_controller=controller,
        chat_store=SimpleNamespace(
            sessions=lambda: [SimpleNamespace(id=sid) for sid in states]
        ),
        view=None,
    )
    assert trusted_console_states(runtime) == {
        "console:a:run": "thinking",
        "console:b:run": "thinking",
        "console:b:approval": "approval_needed",
        "console:c:run": "error",
    }
    states["a"] = "completed"
    result = trusted_console_states(runtime)
    assert "console:a:run" not in result
    assert result["console:b:run"] == "thinking"


@pytest.mark.asyncio
async def test_workbench_buddy_action_requires_saved_clean_local_pack():
    from textual.widgets import Button

    from Tests.UI.test_personas_persona_visual_pack import PackApp, _inventory
    from tldw_chatbook.Widgets.Persona_Widgets.personas_persona_visual_pack_widget import (
        PersonasPersonaVisualPackWidget,
    )

    async with PackApp().run_test(size=(100, 40)) as pilot:
        widget = pilot.app.query_one(PersonasPersonaVisualPackWidget)
        widget.show_inventory(_inventory(), dirty=False)
        await pilot.pause()
        button = widget.query_one("#personas-persona-visual-buddy", Button)
        assert not button.disabled
        widget.show_inventory(_inventory(), dirty=True)
        assert button.disabled
        widget.set_availability("server")
        assert button.disabled


from Tests.Persona_Visual.test_persona_buddy import PERSONA
from Tests.Persona_Visual.test_persona_buddy import (
    environment as environment,  # noqa: PLC0414 -- pytest fixture re-export
)
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Persona_Visual.buddy import BuddyController
from tldw_chatbook.UI.Navigation.persona_buddy_host import PersonaBuddyHost


class BuddyPrimary(BaseAppScreen):
    def compose(self):
        from textual.widgets import Input

        yield Input(id="composer")
        yield Static("Primary work area", id="work-area")


class BuddyApp(ConsolidatedCSSApp):
    def __init__(self, controller):
        super().__init__()
        self.host = PersonaBuddyHost(self, controller)
        self.splash_screen_active = False
        self.console_runtime = None

    async def on_mount(self):
        await self.push_screen(BuddyPrimary(self, "personas"))
        self.host.start()

    async def on_buddy_preferences_requested(self, message):
        if message.control is self.host.current_view:
            await self.host.update_preferences(**message.changes)


async def settle_host(app, pilot):
    for _ in range(3):
        app.host.request_refresh()
        if app.host._task:
            await app.host._task
        await pilot.pause()


@pytest.mark.asyncio
async def test_real_local_buddy_mounts_navigates_hides_for_modal_and_restores(
    environment,
):
    from textual.widgets import Input

    from tldw_chatbook.Widgets.Persona_Widgets.persona_buddy import PersonaBuddyView

    controller = BuddyController(*environment[:3])
    app = BuddyApp(controller)
    async with app.run_test(size=(100, 36)) as pilot:
        await settle_host(app, pilot)
        assert not app.screen.query(PersonaBuddyView)
        assert await app.host.select(PERSONA)
        await settle_host(app, pilot)
        first = app.screen.query_one(PersonaBuddyView)
        assert first.display
        assert first.region.width > 0
        assert first in app.screen._compositor.visible_widgets
        composer = app.screen.query_one(Input)
        composer.focus()
        controller.signal("test:trusted-tool", "tool_running")
        await settle_host(app, pilot)
        assert app.screen.focused is composer
        assert controller.preferences.local_persona_id == PERSONA
        await app.push_screen(ModalScreen())
        await settle_host(app, pilot)
        assert not first.display
        assert app.host.current_view is None
        await app.pop_screen()
        await settle_host(app, pilot)
        assert app.host.current_view is first
        assert first.display
        await app.switch_screen(BuddyPrimary(app, "notes"))
        await settle_host(app, pilot)
        second = app.screen.query_one(PersonaBuddyView)
        assert second is not first
        assert not first.is_attached
        assert controller.preferences.local_persona_id == PERSONA
        await app.host.update_preferences(open=False)
        await settle_host(app, pilot)
        assert not second.display
        assert controller.preferences.enabled
        await app.host.shutdown()


def test_terminal_failure_lease_expires_while_other_session_keeps_running(environment):
    clock = [0.0]
    buddy = BuddyController(*environment[:3], clock=lambda: clock[0])
    app = type("AppOwner", (), {})()
    host = PersonaBuddyHost(app, buddy)
    signals = {"console:old:run": "error", "console:new:run": "thinking"}
    host.sync_trusted_signals(signals)
    assert buddy.requested_state == "error"
    clock[0] = 10.0
    host.sync_trusted_signals(signals)
    assert buddy.requested_state == "thinking"
    host.sync_trusted_signals({"console:old:run": "thinking"})
    host.sync_trusted_signals({"console:old:run": "error"})
    assert buddy.requested_state == "error"


@pytest.mark.asyncio
async def test_host_honors_canonical_appearance_reduce_motion(environment, monkeypatch):
    from tldw_chatbook import config

    monkeypatch.setattr(
        config,
        "get_cli_setting",
        lambda section, key, default=None: (
            section == "appearance" and key == "reduce_motion"
        ),
    )
    controller = BuddyController(*environment[:3])
    app = BuddyApp(controller)
    async with app.run_test(size=(100, 36)) as pilot:
        assert await app.host.select(PERSONA)
        await settle_host(app, pilot)
        assert app.host.current_view._snapshot.resolution.cache_identity.reduced_motion
        await app.host.shutdown()


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("idle", None),
        ("connecting", None),
        ("live", "listening"),
        ("thinking", "thinking"),
        ("speaking", "speaking"),
        ("reconnecting", None),
    ],
)
def test_realtime_voice_signals_follow_actual_fsm_with_idle_dictation(state, expected):
    from tldw_chatbook.Chat.console_realtime_loop import RealtimeLoopController

    loop = RealtimeLoopController(
        lambda event: None, acoustic_barge_in=False, idle_timeout_seconds=60
    )
    if state != "idle":
        loop.enter()
    if state not in {"idle", "connecting"}:
        loop.on_session_ready()
    if state in {"thinking", "speaking"}:
        loop.on_turn_committed(now=1)
    if state == "speaking":
        loop.on_first_audio()
    if state == "reconnecting":
        loop.on_transport_closed(error=True)
    assert loop.state == state
    runtime = SimpleNamespace(
        view=SimpleNamespace(
            _console_dictation_state="idle",
            _console_speaking_message_id=None,
            _console_realtime=SimpleNamespace(controller=loop),
        )
    )
    assert trusted_console_states(runtime) == (
        {"console:realtime": expected} if expected else {}
    )


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("idle", None),
        ("listening", "listening"),
        ("countdown", "listening"),
        ("awaiting_reply", "thinking"),
        ("speaking", "speaking"),
    ],
)
def test_hands_free_voice_signals_follow_actual_fsm(state, expected):
    from tldw_chatbook.Chat.console_hands_free import HandsFreeController

    loop = HandsFreeController(lambda event: None)
    if state != "idle":
        loop.enter(capture_live=True)
    if state in {"countdown", "awaiting_reply", "speaking"}:
        loop.on_voice_final()
    if state in {"awaiting_reply", "speaking"}:
        loop.tick(0)
        loop.tick(2)
    if state == "speaking":
        loop.on_first_utterance()
    assert loop.state == state
    runtime = SimpleNamespace(
        view=SimpleNamespace(
            _console_dictation_state="idle",
            _console_speaking_message_id=None,
            _console_hands_free=SimpleNamespace(controller=loop),
        )
    )
    assert trusted_console_states(runtime) == (
        {"console:hands-free": expected} if expected else {}
    )


@pytest.fixture
def production_agent_bridge():
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge

    # Snapshot publication is an in-memory production seam. These dependencies
    # are unused; any accidental DB or provider read fails immediately.
    return ConsoleAgentBridge(
        agent_runs_db=None,
        store=None,
        provider_gateway=None,
        registry=ToolCatalogRegistry(),
    )


@pytest.mark.parametrize("persisted_id", [None, "durable-conversation"])
@pytest.mark.parametrize(
    ("status", "kind", "expected"),
    [
        ("running", "tool_call", True),
        ("done", "tool_call", False),
        ("running", "tool_result", False),
        ("running", "llm", False),
    ],
)
def test_tool_signal_reads_production_primary_execution_metadata(
    production_agent_bridge,
    persisted_id,
    status,
    kind,
    expected,
):
    from tldw_chatbook.Chat.console_agent_bridge import AgentLiveSnapshot, AgentLiveStep

    bridge = production_agent_bridge
    bridge._publish_live(
        persisted_id or "session-a",
        "primary-run",
        AgentLiveSnapshot(
            status=status,
            steps=(AgentLiveStep(kind, "tool_call error offline", "primary"),),
        ),
        primary=True,
    )
    runtime = SimpleNamespace(
        agent_bridge=bridge,
        chat_store=SimpleNamespace(
            sessions=lambda: [
                SimpleNamespace(
                    id="session-a",
                    persisted_conversation_id=persisted_id,
                )
            ]
        ),
        chat_controller=SimpleNamespace(
            run_state_for=lambda sid: SimpleNamespace(status="streaming"),
            has_pending_approval_round=lambda sid: False,
        ),
        view=None,
    )
    result = trusted_console_states(runtime)
    assert result.get("console:session-a:tool") == (
        "tool_running" if expected else None
    )
    assert result["console:session-a:run"] == "thinking"
    # A child from a previous turn cannot retarget the primary summary.
    bridge._publish_live(
        persisted_id or "session-a",
        "previous-child",
        AgentLiveSnapshot(
            status="running", steps=(AgentLiveStep("tool_call", "ignored", "subagent"),)
        ),
        primary=False,
    )
    assert trusted_console_states(runtime) == result
    bridge._publish_live(
        persisted_id or "session-a",
        "primary-run",
        AgentLiveSnapshot(
            status="running",
            steps=(AgentLiveStep("tool_result", "tool_running", "primary"),),
        ),
        primary=False,
    )
    assert "console:session-a:tool" not in trusted_console_states(runtime)
