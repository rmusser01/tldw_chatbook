"""Master shell destination wrapper tests."""

import inspect
from pathlib import Path
from unittest.mock import Mock

import pytest
from textual.app import App
from textual.widgets import Button, Select, Static, TextArea

from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_unified_mcp_panel import FakeUnifiedMCPService
from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore
from tldw_chatbook.MCP.unified_control_models import ConfiguredServerTarget
from tldw_chatbook.runtime_policy.types import PolicyDeniedError
from tldw_chatbook.UI.MCP_Modules.unified_mcp_panel import UnifiedMCPPanel
from tldw_chatbook.UI.Screens.artifacts_screen import ArtifactsScreen
from tldw_chatbook.UI.Screens.acp_screen import ACPScreen
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.UI.Screens.mcp_screen import MCPScreen
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.UI.Screens.schedules_screen import SchedulesScreen
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.UI.Screens.skills_screen import SkillsScreen
from tldw_chatbook.UI.Screens.watchlists_collections_screen import WatchlistsCollectionsScreen
from tldw_chatbook.UI.Screens.workflows_screen import WorkflowsScreen
from tldw_chatbook.UI.Screens import skills_screen as skills_screen_module


SCREEN_BY_ROUTE = {
    "library": LibraryScreen,
    "artifacts": ArtifactsScreen,
    "personas": PersonasScreen,
    "watchlists_collections": WatchlistsCollectionsScreen,
    "schedules": SchedulesScreen,
    "workflows": WorkflowsScreen,
    "mcp": MCPScreen,
    "tools_settings": MCPScreen,
    "acp": ACPScreen,
    "skills": SkillsScreen,
    "settings": SettingsScreen,
}

PHASE4_MCP_ADOPTION_EVIDENCE = Path(
    "Docs/superpowers/qa/unified-shell/phase-4/2026-05-04-mcp-destination-service-adoption.md"
)
PHASE4_SKILLS_ADOPTION_EVIDENCE = Path(
    "Docs/superpowers/qa/unified-shell/phase-4/2026-05-04-skills-destination-service-adoption.md"
)


class StaticSkillsScopeService:
    def __init__(self, skills):
        self.skills = tuple(skills)
        self.calls = []

    async def list_skills(self, **kwargs):
        self.calls.append(kwargs)
        return {"skills": list(self.skills), "total": len(self.skills)}


class RaisingSkillsScopeService:
    async def list_skills(self, **kwargs):
        raise RuntimeError("skills registry unavailable")


class PolicyDeniedSkillsScopeService:
    async def list_skills(self, **kwargs):
        raise PolicyDeniedError(
            action_id="skills.list.local",
            reason_code="authority_denied",
            user_message="Local Skills are disabled by workspace policy.",
            effective_source="local",
            authority_owner="local",
        )


class DestinationHarness(App):
    def __init__(self, app_instance, route, seen_routes=None, restored_state=None):
        super().__init__()
        self.app_instance = app_instance
        self.route = route
        self.seen_routes = seen_routes if seen_routes is not None else []
        self.restored_state = restored_state

    async def on_mount(self) -> None:
        screen = SCREEN_BY_ROUTE[self.route](self.app_instance)
        if self.restored_state is not None:
            screen.restore_state(self.restored_state)
        await self.push_screen(screen)

    def on_navigate_to_screen(self, message) -> None:
        self.seen_routes.append(message.screen_name)


def _active_destination_screen(host: DestinationHarness):
    return host.screen_stack[-1]


def _static_text(widget: Static) -> str:
    renderable = widget.renderable
    return getattr(renderable, "plain", str(renderable))


def _visible_text(screen) -> str:
    return " ".join(
        _static_text(widget)
        for widget in screen.query(Static)
        if widget.display and hasattr(widget, "renderable")
    )


@pytest.mark.parametrize(
    ("route", "title_id", "purpose_text"),
    [
        ("library", "#library-title", "source material"),
        ("artifacts", "#artifacts-title", "generated"),
        ("personas", "#personas-title", "behavior"),
    ],
)
@pytest.mark.asyncio
async def test_primary_destination_wrappers_mount(route, title_id, purpose_text):
    app = _build_test_app()
    host = DestinationHarness(app, route)

    async with host.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_destination_screen(host)

        title = screen.query_one(title_id)
        assert title
        assert title.has_class("ds-destination-header")
        assert purpose_text in _static_text(screen.query_one(".destination-purpose", Static)).lower()
        assert screen.query_one(".ds-panel")


@pytest.mark.asyncio
async def test_watchlists_collections_uses_compact_title_and_clear_sections():
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")

    async with host.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_destination_screen(host)

        assert _static_text(screen.query_one("#watchlists-collections-title", Static)) == "W+C"
        visible_text = _visible_text(screen)
        assert "Watchlists" in visible_text
        assert "Collections" in visible_text


@pytest.mark.asyncio
async def test_library_exposes_source_sections_and_import_export_boundary():
    app = _build_test_app()
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_destination_screen(host)

        for selector in [
            "#library-open-notes",
            "#library-open-media",
            "#library-open-conversations",
            "#library-open-import-export",
            "#library-open-search",
        ]:
            assert screen.query_one(selector)


@pytest.mark.parametrize(
    ("route", "selector", "target_route"),
    [
        ("library", "#library-open-notes", "notes"),
        ("library", "#library-open-media", "media"),
        ("library", "#library-open-conversations", "conversation"),
        ("library", "#library-open-import-export", "ingest"),
        ("library", "#library-open-search", "search"),
        ("artifacts", "#artifacts-open-chatbooks", "chatbooks"),
        ("personas", "#personas-open-profiles", "ccp"),
        ("watchlists_collections", "#wc-open-watchlists", "subscriptions"),
    ],
)
@pytest.mark.asyncio
async def test_destination_action_buttons_emit_compatibility_routes(route, selector, target_route):
    app = _build_test_app()
    seen_routes = []
    host = DestinationHarness(app, route, seen_routes)

    async with host.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        await pilot.click(selector)
        await pilot.pause(0.1)

    assert seen_routes[-1] == target_route


@pytest.mark.parametrize("route", SCREEN_BY_ROUTE)
@pytest.mark.asyncio
async def test_destination_action_buttons_explain_their_outcome(route):
    app = _build_test_app()
    host = DestinationHarness(app, route)

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_destination_screen(host)

        for button in screen.query(Button):
            tooltip = getattr(button, "tooltip", None)
            assert tooltip is not None, button.id
            assert str(tooltip).strip().lower() not in {"", "none"}, button.id


@pytest.mark.parametrize(
    ("route", "expected_sections"),
    [
        ("watchlists_collections", ["Watchlists", "Collections"]),
        ("schedules", ["Next Run", "Paused", "Failed"]),
        ("workflows", ["Recipes", "Dry Run", "Console launch unavailable"]),
    ],
)
@pytest.mark.asyncio
async def test_automation_destination_wrappers_explain_ownership(route, expected_sections):
    app = _build_test_app()
    host = DestinationHarness(app, route)

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_destination_screen(host)

        visible_text = _visible_text(screen)
        for section in expected_sections:
            assert section in visible_text


@pytest.mark.parametrize(
    ("route", "expected_text"),
    [
        ("mcp", "tools and servers"),
        ("acp", "Agent Client Protocol"),
        ("skills", "SKILL.md"),
        ("settings", "global preferences"),
    ],
)
@pytest.mark.asyncio
async def test_protocol_and_settings_wrappers_have_distinct_boundaries(route, expected_text):
    app = _build_test_app()
    host = DestinationHarness(app, route)

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_destination_screen(host)

        visible_text = _visible_text(screen)
        assert expected_text in visible_text


@pytest.mark.parametrize(
    ("route", "selector", "copy"),
    [
        ("skills", "#skills-import-skill", "Skill import is not wired in this shell yet."),
    ],
)
@pytest.mark.asyncio
async def test_unwired_destination_actions_are_disabled_with_honest_copy(route, selector, copy):
    app = _build_test_app()
    host = DestinationHarness(app, route)

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_destination_screen(host)

        button = screen.query_one(selector, Button)
        assert button.disabled is True
        assert copy in _visible_text(screen)


@pytest.mark.asyncio
async def test_mcp_destination_embeds_unified_mcp_management_panel():
    app = _build_test_app()
    host = DestinationHarness(app, "mcp")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = _active_destination_screen(host)

        assert screen.query_one(UnifiedMCPPanel)
        assert screen.query_one("#unified-mcp-source", Select)
        assert screen.query_one("#unified-mcp-server-target", Select)
        assert screen.query_one("#unified-mcp-scope", Select)
        assert screen.query_one("#unified-mcp-section", Select)
        assert screen.query_one("#unified-mcp-action", Select)
        assert screen.query_one("#unified-mcp-action-payload", TextArea)
        assert screen.query_one("#unified-mcp-action-run", Button)
        assert not screen.query("#mcp-open-management")
        assert "Unified MCP management is not embedded in this shell yet." not in _visible_text(screen)


@pytest.mark.asyncio
async def test_mcp_destination_restores_unified_mcp_view_state_after_mount(tmp_path):
    target_store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    target_store.save_targets(
        [ConfiguredServerTarget(server_id="server-a", label="Server A", base_url="https://a.example/api", is_default=True)]
    )
    app = _build_test_app()
    app.unified_mcp_service = FakeUnifiedMCPService(target_store)
    host = DestinationHarness(
        app,
        "mcp",
        restored_state={
            "unified_mcp_view_state": {
                "selected_source": "server",
                "selected_active_server_id": "server-a",
                "selected_scope": "team",
                "selected_scope_ref": "21",
                "selected_section": "inventory",
            }
        },
    )

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        panel = _active_destination_screen(host).query_one(UnifiedMCPPanel)

        assert panel.context.selected_source == "server"
        assert panel.context.selected_active_server_id == "server-a"
        assert panel.context.selected_scope == "team"
        assert panel.context.selected_scope_ref == "21"
        assert panel.context.selected_section == "inventory"


@pytest.mark.asyncio
async def test_mcp_destination_runtime_refresh_uses_exclusive_worker(monkeypatch):
    app = _build_test_app()
    screen = MCPScreen(app)
    scheduled = {}

    class FakePanel:
        async def load_context(self):
            return None

    def capture_worker(coro, **kwargs):
        scheduled["kwargs"] = kwargs
        coro.close()

    screen.mcp_panel = FakePanel()
    monkeypatch.setattr(screen, "run_worker", capture_worker)

    await screen.handle_runtime_backend_changed("server")

    assert scheduled["kwargs"]["name"] == "mcp-screen-runtime-refresh"
    assert scheduled["kwargs"]["group"] == "mcp-screen-runtime-refresh"
    assert scheduled["kwargs"]["exclusive"] is True


def test_mcp_destination_service_adoption_tracking_evidence_exists():
    evidence = PHASE4_MCP_ADOPTION_EVIDENCE.read_text(encoding="utf-8")
    roadmap = Path("Docs/superpowers/trackers/unified-shell-maturity-roadmap.md").read_text(encoding="utf-8")
    task = Path(
        "backlog/tasks/task-5.1 - Phase-4.1-Adopt-Unified-MCP-panel-in-MCP-destination.md"
    ).read_text(encoding="utf-8")

    assert "Phase 4.1 MCP Destination Service Adoption" in evidence
    assert "TASK-5.1" in evidence
    assert "Phase 4.1: Adopt Unified MCP panel in MCP destination - `TASK-5.1`" in roadmap
    assert "UnifiedMCPPanel" in task


def test_skills_screen_public_initializer_is_typed():
    signature = inspect.signature(SkillsScreen.__init__)

    assert signature.parameters["app_instance"].annotation is not inspect.Parameter.empty
    assert signature.parameters["kwargs"].annotation is not inspect.Parameter.empty
    assert signature.return_annotation in {None, "None"}


@pytest.mark.asyncio
async def test_skills_destination_lists_local_skills_from_scope_service():
    app = _build_test_app()
    app.skills_scope_service = StaticSkillsScopeService(
        [
            {
                "name": "summarize-notes",
                "description": "Summarize note collections",
                "argument_hint": "note id",
                "record_id": "local:skill:summarize-notes",
            },
            {
                "name": "code-review",
                "description": "Review code changes",
                "record_id": "local:skill:code-review",
            },
        ]
    )
    host = DestinationHarness(app, "skills")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = _active_destination_screen(host)
        text = _visible_text(screen)
        button = screen.query_one("#skills-attach-to-console", Button)

        assert "Installed local skills: 2" in text
        assert "summarize-notes" in text
        assert "Summarize note collections" in text
        assert "code-review" in text
        assert button.disabled is False

    assert app.skills_scope_service.calls[0]["mode"] == "local"
    assert app.skills_scope_service.calls[0]["limit"] == getattr(
        skills_screen_module,
        "SKILLS_LOCAL_PAGE_SIZE",
        None,
    )


@pytest.mark.asyncio
async def test_skills_destination_empty_state_disables_console_attach():
    app = _build_test_app()
    app.skills_scope_service = StaticSkillsScopeService([])
    host = DestinationHarness(app, "skills")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = _active_destination_screen(host)
        button = screen.query_one("#skills-attach-to-console", Button)

        assert "No local Agent Skills are installed yet." in _visible_text(screen)
        assert button.disabled is True
        assert "Stage local skill context" in str(button.tooltip)


@pytest.mark.asyncio
async def test_skills_destination_service_failure_uses_recovery_copy():
    app = _build_test_app()
    app.skills_scope_service = RaisingSkillsScopeService()
    host = DestinationHarness(app, "skills")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = _active_destination_screen(host)
        button = screen.query_one("#skills-attach-to-console", Button)

        assert "Skills service unavailable; retry Skills later." in _visible_text(screen)
        assert button.disabled is True
        assert "Skills service is unavailable" in str(button.tooltip)


@pytest.mark.asyncio
async def test_skills_destination_missing_service_uses_unavailable_state():
    app = _build_test_app()
    app.skills_scope_service = object()
    host = DestinationHarness(app, "skills")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = _active_destination_screen(host)
        button = screen.query_one("#skills-attach-to-console", Button)

        assert "Skills service is unavailable in this runtime." in _visible_text(screen)
        assert button.disabled is True
        assert "Skills service is unavailable" in str(button.tooltip)


@pytest.mark.asyncio
async def test_skills_destination_policy_denied_surfaces_policy_message():
    app = _build_test_app()
    app.skills_scope_service = PolicyDeniedSkillsScopeService()
    host = DestinationHarness(app, "skills")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = _active_destination_screen(host)
        button = screen.query_one("#skills-attach-to-console", Button)

        assert "Local Skills are disabled by workspace policy." in _visible_text(screen)
        assert button.disabled is True
        assert "Skills service unavailable; retry Skills later." not in _visible_text(screen)


@pytest.mark.asyncio
async def test_skills_attach_to_console_uses_listed_skill_context():
    app = _build_test_app()
    app.skills_scope_service = StaticSkillsScopeService(
        [
            {
                "name": "summarize-notes",
                "description": "Summarize note collections",
                "argument_hint": "note id",
                "record_id": "local:skill:summarize-notes",
            }
        ]
    )
    app.open_chat_with_handoff = Mock()
    host = DestinationHarness(app, "skills")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        await pilot.click("#skills-attach-to-console")
        await pilot.pause(0.1)

    app.open_chat_with_handoff.assert_called_once()
    payload = app.open_chat_with_handoff.call_args.args[0]
    assert isinstance(payload, ChatHandoffPayload)
    assert payload.source == "skills"
    assert payload.item_type == "skills-context"
    assert payload.title == "Local Agent Skills (1)"
    assert "summarize-notes" in payload.body
    assert "Summarize note collections" in payload.body
    assert "argument hint: note id" in payload.body
    assert "Stage installed skills, SKILL.md instructions" not in payload.body
    assert payload.metadata["skill_count"] == 1
    assert payload.metadata["skill_names"] == ["summarize-notes"]


@pytest.mark.asyncio
async def test_skills_attach_to_console_sanitizes_listed_skill_text():
    app = _build_test_app()
    app.skills_scope_service = StaticSkillsScopeService(
        [
            {
                "name": "unsafe-skill",
                "description": "Summarize <script>alert(1)</script> notes",
                "argument_hint": "note id onclick=steal",
                "record_id": "local:skill:unsafe-skill\x00",
            }
        ]
    )
    app.open_chat_with_handoff = Mock()
    host = DestinationHarness(app, "skills")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = _active_destination_screen(host)
        visible_text = _visible_text(screen).lower()
        await pilot.click("#skills-attach-to-console")
        await pilot.pause(0.1)

    app.open_chat_with_handoff.assert_called_once()
    payload = app.open_chat_with_handoff.call_args.args[0]

    assert "unsafe-skill" in payload.body
    assert "<script" not in payload.body.lower()
    assert "onclick=" not in payload.body.lower()
    assert "\x00" not in payload.body
    assert "<script" not in visible_text
    assert "onclick=" not in visible_text
    assert payload.metadata["skill_names"] == ["unsafe-skill"]


def test_skills_destination_service_adoption_tracking_evidence_exists():
    evidence = PHASE4_SKILLS_ADOPTION_EVIDENCE.read_text(encoding="utf-8")
    roadmap = Path("Docs/superpowers/trackers/unified-shell-maturity-roadmap.md").read_text(encoding="utf-8")
    task = Path(
        "backlog/tasks/task-5.2 - Phase-4.2-Adopt-Skills-services-in-Skills-destination.md"
    ).read_text(encoding="utf-8")

    assert "Phase 4.2 Skills Destination Service Adoption" in evidence
    assert "TASK-5.2" in evidence
    assert "Phase 4.2: Adopt Skills services in Skills destination - `TASK-5.2`" in roadmap
    assert "skills_scope_service" in task


@pytest.mark.asyncio
async def test_settings_appearance_action_routes_to_customize_surface():
    app = _build_test_app()
    seen_routes = []
    host = DestinationHarness(app, "settings", seen_routes)

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        await pilot.click("#settings-open-appearance")
        await pilot.pause(0.1)

    assert seen_routes[-1] == "customize"


@pytest.mark.asyncio
async def test_legacy_tools_settings_route_opens_mcp_not_global_settings():
    app = _build_test_app()
    screen_name, current_tab, screen_class = app._resolve_screen_navigation_target("tools_settings")

    assert screen_name == "tools_settings"
    assert current_tab == "mcp"
    assert screen_class is MCPScreen

    host = DestinationHarness(app, "tools_settings")

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_destination_screen(host)

        visible_text = " ".join(
            _static_text(widget)
            for widget in screen.query(Static)
            if hasattr(widget, "renderable")
        )
        assert "tools and servers" in visible_text
        assert "global preferences" not in visible_text
