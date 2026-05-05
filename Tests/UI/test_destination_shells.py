"""Master shell destination wrapper tests."""

import asyncio
import inspect
import time
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
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from tldw_chatbook.UI.Screens import personas_screen as personas_screen_module
from tldw_chatbook.UI.Screens import skills_screen as skills_screen_module
from tldw_chatbook.UI.Screens import watchlists_collections_screen as wc_screen_module


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
PHASE4_LIBRARY_ADOPTION_EVIDENCE = Path(
    "Docs/superpowers/qa/unified-shell/phase-4/2026-05-04-library-source-service-adoption.md"
)
PHASE4_PERSONAS_ADOPTION_EVIDENCE = Path(
    "Docs/superpowers/qa/unified-shell/phase-4/2026-05-04-personas-service-adoption.md"
)
PHASE4_WC_ADOPTION_EVIDENCE = Path(
    "Docs/superpowers/qa/unified-shell/phase-4/2026-05-04-wc-service-adoption.md"
)


class StaticPersonasScopeService:
    def __init__(self, *, characters=(), profiles=()):
        self.characters = tuple(characters)
        self.profiles = tuple(profiles)
        self.character_calls = []
        self.profile_calls = []

    async def list_characters(self, **kwargs):
        self.character_calls.append(kwargs)
        return list(self.characters)

    async def list_persona_profiles(self, **kwargs):
        self.profile_calls.append(kwargs)
        return list(self.profiles)


class RaisingPersonasScopeService:
    async def list_characters(self, **kwargs):
        raise RuntimeError("characters unavailable")

    async def list_persona_profiles(self, **kwargs):
        return []


class SlowPersonasScopeService(StaticPersonasScopeService):
    async def list_characters(self, **kwargs):
        await asyncio.sleep(0.35)
        return await super().list_characters(**kwargs)


class StaticLibraryNotesScopeService:
    def __init__(self, notes):
        self.notes = tuple(notes)
        self.calls = []

    async def list_notes(self, **kwargs):
        self.calls.append(kwargs)
        return {"items": list(self.notes), "pagination": {"total": len(self.notes)}}


class StaticLibraryNotesListScopeService:
    def __init__(self, notes):
        self.notes = tuple(notes)
        self.calls = []

    async def list_notes(self, **kwargs):
        self.calls.append(kwargs)
        limit = kwargs.get("limit")
        if isinstance(limit, int):
            return list(self.notes[:limit])
        return list(self.notes)


class StaticLibraryMediaScopeService:
    def __init__(self, media_items):
        self.media_items = tuple(media_items)
        self.calls = []

    async def list_media_items(self, **kwargs):
        self.calls.append(kwargs)
        return {"items": list(self.media_items), "pagination": {"total_items": len(self.media_items)}}


class StaticLibraryConversationScopeService:
    def __init__(self, conversations):
        self.conversations = tuple(conversations)
        self.calls = []

    async def list_conversations(self, **kwargs):
        self.calls.append(kwargs)
        return {"items": list(self.conversations), "pagination": {"total": len(self.conversations)}}


class RaisingLibraryNotesScopeService:
    async def list_notes(self, **kwargs):
        raise RuntimeError("notes unavailable")


class PolicyDeniedLibraryNotesScopeService:
    def __init__(
        self,
        *,
        reason_code="wrong_source",
        user_message="Server Library sources require server mode.",
        effective_source="local",
        authority_owner="active server",
    ):
        self.reason_code = reason_code
        self.user_message = user_message
        self.effective_source = effective_source
        self.authority_owner = authority_owner

    async def list_notes(self, **kwargs):
        raise PolicyDeniedError(
            action_id="library.sources.list",
            reason_code=self.reason_code,
            user_message=self.user_message,
            effective_source=self.effective_source,
            authority_owner=self.authority_owner,
        )


class StaticWatchlistsScopeService:
    def __init__(self, watch_items):
        self.watch_items = tuple(watch_items)
        self.calls = []

    async def list_watch_items(self, **kwargs):
        self.calls.append(kwargs)
        return list(self.watch_items)


class RaisingWatchlistsScopeService:
    async def list_watch_items(self, **kwargs):
        raise RuntimeError("watchlists unavailable")


class PolicyDeniedWatchlistsScopeService:
    def __init__(
        self,
        *,
        reason_code="server_session_invalid",
        user_message="The W+C server session has expired.",
        effective_source="server",
        authority_owner="active server",
    ):
        self.reason_code = reason_code
        self.user_message = user_message
        self.effective_source = effective_source
        self.authority_owner = authority_owner

    async def list_watch_items(self, **kwargs):
        raise PolicyDeniedError(
            action_id="wc.watchlists.list",
            reason_code=self.reason_code,
            user_message=self.user_message,
            effective_source=self.effective_source,
            authority_owner=self.authority_owner,
        )


class StaticReadItLaterScopeService:
    def __init__(self, items):
        self.items = tuple(items)
        self.calls = []

    async def list_read_it_later(self, **kwargs):
        self.calls.append(kwargs)
        return {"items": list(self.items), "total": len(self.items)}


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
    def __init__(
        self,
        *,
        reason_code="authority_denied",
        user_message="Local Skills are disabled by workspace policy.",
        effective_source="local",
        authority_owner="local",
    ):
        self.reason_code = reason_code
        self.user_message = user_message
        self.effective_source = effective_source
        self.authority_owner = authority_owner

    async def list_skills(self, **kwargs):
        raise PolicyDeniedError(
            action_id="skills.list.local",
            reason_code=self.reason_code,
            user_message=self.user_message,
            effective_source=self.effective_source,
            authority_owner=self.authority_owner,
        )


class PolicyDeniedPersonasScopeService:
    def __init__(
        self,
        *,
        reason_code="server_auth_required",
        user_message="Server Personas require sign-in.",
        effective_source="server",
        authority_owner="active server",
    ):
        self.reason_code = reason_code
        self.user_message = user_message
        self.effective_source = effective_source
        self.authority_owner = authority_owner

    async def list_characters(self, **kwargs):
        raise PolicyDeniedError(
            action_id="personas.characters.list",
            reason_code=self.reason_code,
            user_message=self.user_message,
            effective_source=self.effective_source,
            authority_owner=self.authority_owner,
        )

    async def list_persona_profiles(self, **kwargs):
        return []


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


async def _wait_for_personas_snapshot(screen, pilot, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    terminal_selectors = (
        "#personas-service-error",
        "#personas-empty-state",
        "#personas-characters-summary",
        "#personas-profiles-summary",
    )
    while time.monotonic() < deadline:
        terminal_state_visible = any(screen.query(selector) for selector in terminal_selectors)
        buttons = list(screen.query("#personas-attach-to-console"))
        if buttons:
            button = buttons[0]
            if button.disabled is False:
                await pilot.pause()
                return
            if terminal_state_visible:
                await pilot.pause()
                return
        await pilot.pause(0.01)
    raise AssertionError(f"Timed out waiting for Personas snapshot. Visible text: {_visible_text(screen)}")


async def _wait_for_wc_snapshot(screen, pilot, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    terminal_selectors = (
        "#wc-service-error",
        "#wc-empty-state",
        "#wc-watchlists-summary",
        "#wc-collections-summary",
    )
    while time.monotonic() < deadline:
        if any(screen.query(selector) for selector in terminal_selectors):
            await pilot.pause()
            return
        await pilot.pause(0.01)
    raise AssertionError(f"Timed out waiting for W+C snapshot. Visible text: {_visible_text(screen)}")


async def _wait_for_library_snapshot(screen, pilot, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    terminal_selectors = (
        "#library-source-error",
        "#library-source-empty",
        "#library-notes-summary",
        "#library-media-summary",
        "#library-conversations-summary",
    )
    while time.monotonic() < deadline:
        if any(screen.query(selector) for selector in terminal_selectors):
            await pilot.pause()
            return
        await pilot.pause(0.01)
    raise AssertionError(f"Timed out waiting for Library snapshot. Visible text: {_visible_text(screen)}")


async def _wait_for_skills_snapshot(screen, pilot, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    terminal_selectors = (
        "#skills-service-error",
        "#skills-empty-state",
        "#skills-local-summary",
    )
    while time.monotonic() < deadline:
        if any(screen.query(selector) for selector in terminal_selectors):
            await pilot.pause()
            return
        await pilot.pause(0.01)
    raise AssertionError(f"Timed out waiting for Skills snapshot. Visible text: {_visible_text(screen)}")


async def _wait_for_mock_call(mock: Mock, pilot, *, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if mock.call_count:
            return
        await pilot.pause()
    raise AssertionError("Timed out waiting for mock call")


def _assert_policy_recovery_copy(
    *,
    visible_text: str,
    button: Button,
    status_label: str,
    unavailable_what: str,
    why: str,
    next_action: str,
    recovery_action: str,
    authority_owner: str,
) -> None:
    assert status_label in visible_text
    assert f"Unavailable: {unavailable_what}." in visible_text
    assert f"Why: {why}." in visible_text
    assert f"Next: {next_action}" in visible_text
    assert f"Recovery: {recovery_action}." in visible_text
    assert f"Owner: {authority_owner}." in visible_text
    assert button.disabled is True
    assert why in str(button.tooltip)
    assert next_action in str(button.tooltip)


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
async def test_watchlists_collections_lists_local_snapshot_from_services():
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService(
        [
            {"title": "Research feeds", "id": 1, "record_id": "local:subscription:1"},
            {"name": "Vendor changelogs", "id": 2, "record_id": "local:subscription:2"},
        ]
    )
    app.media_reading_scope_service = StaticReadItLaterScopeService(
        [
            {"title": "Saved article", "id": 10, "record_id": "local:media:10"},
        ]
    )
    host = DestinationHarness(app, "watchlists_collections")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_wc_snapshot(screen, pilot)
        text = _visible_text(screen)
        button = screen.query_one("#wc-attach-to-console", Button)

        assert "Local W+C snapshot" in text
        assert "Watchlists (showing up to 5): 2" in text
        assert "Collections: 1" in text
        assert "Research feeds" in text
        assert "Vendor changelogs" in text
        assert "Saved article" in text
        assert button.disabled is False

    assert app.watchlist_scope_service.calls[0] == {
        "runtime_backend": "local",
        "limit": getattr(wc_screen_module, "WC_LOCAL_PAGE_SIZE", None),
        "offset": 0,
    }
    assert app.media_reading_scope_service.calls[0] == {
        "mode": "local",
        "limit": getattr(wc_screen_module, "WC_LOCAL_PAGE_SIZE", None),
        "offset": 0,
    }


@pytest.mark.asyncio
async def test_watchlists_collections_empty_state_disables_console_attach():
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    app.media_reading_scope_service = StaticReadItLaterScopeService([])
    host = DestinationHarness(app, "watchlists_collections")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_wc_snapshot(screen, pilot)
        button = screen.query_one("#wc-attach-to-console", Button)

        assert "No local Watchlists or Collections are available yet." in _visible_text(screen)
        assert button.disabled is True
        assert "Stage local W+C context" in str(button.tooltip)


@pytest.mark.asyncio
async def test_watchlists_collections_service_failure_uses_recovery_copy():
    app = _build_test_app()
    app.watchlist_scope_service = RaisingWatchlistsScopeService()
    app.media_reading_scope_service = StaticReadItLaterScopeService([])
    host = DestinationHarness(app, "watchlists_collections")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_wc_snapshot(screen, pilot)
        button = screen.query_one("#wc-attach-to-console", Button)

        assert "W+C services unavailable; retry W+C later." in _visible_text(screen)
        assert button.disabled is True
        assert "W+C services are unavailable" in str(button.tooltip)


@pytest.mark.asyncio
async def test_watchlists_collections_policy_denial_uses_runtime_recovery_taxonomy():
    app = _build_test_app()
    app.watchlist_scope_service = PolicyDeniedWatchlistsScopeService()
    app.media_reading_scope_service = StaticReadItLaterScopeService([])
    host = DestinationHarness(app, "watchlists_collections")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_wc_snapshot(screen, pilot)
        error = screen.query_one("#wc-service-error", Static)
        button = screen.query_one("#wc-attach-to-console", Button)

        _assert_policy_recovery_copy(
            visible_text=_static_text(error),
            button=button,
            status_label="Server session expired",
            unavailable_what="Stage W+C context in Console",
            why="The W+C server session has expired",
            next_action="Re-authenticate the active server profile before retrying.",
            recovery_action="Settings",
            authority_owner="active server",
        )


@pytest.mark.asyncio
async def test_watchlists_collections_attach_to_console_uses_listed_context():
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService(
        [{"title": "Research feeds", "id": 1, "record_id": "local:subscription:1"}]
    )
    app.media_reading_scope_service = StaticReadItLaterScopeService(
        [{"title": "Saved article", "id": 10, "record_id": "local:media:10"}]
    )
    app.open_chat_with_handoff = Mock()
    host = DestinationHarness(app, "watchlists_collections")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_wc_snapshot(screen, pilot)
        await pilot.click("#wc-attach-to-console")
        await _wait_for_mock_call(app.open_chat_with_handoff, pilot)

    payload = app.open_chat_with_handoff.call_args.args[0]
    assert isinstance(payload, ChatHandoffPayload)
    assert payload.source == "watchlists_collections"
    assert payload.item_type == "wc-context"
    assert payload.title == "Local W+C snapshot"
    assert "Research feeds" in payload.body
    assert "Saved article" in payload.body
    assert payload.metadata["watchlist_count"] == 1
    assert payload.metadata["collection_count"] == 1


@pytest.mark.asyncio
async def test_watchlists_collections_preserves_safe_comparison_titles_and_rejects_dangerous_text():
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService(
        [{"title": "Model A < Model B > Baseline", "id": 1}]
    )
    app.media_reading_scope_service = StaticReadItLaterScopeService(
        [{"title": "javascript:alert(1)", "id": 10}]
    )
    app.open_chat_with_handoff = Mock()
    host = DestinationHarness(app, "watchlists_collections")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_wc_snapshot(screen, pilot)
        text = _visible_text(screen)

        assert "Model A < Model B > Baseline" in text
        assert "javascript:alert(1)" not in text
        assert "alert(1)" not in text

        await pilot.click("#wc-attach-to-console")
        await _wait_for_mock_call(app.open_chat_with_handoff, pilot)

    payload = app.open_chat_with_handoff.call_args.args[0]
    assert "Model A < Model B > Baseline" in payload.body
    assert "javascript:alert(1)" not in payload.body
    assert "alert(1)" not in payload.body


@pytest.mark.asyncio
async def test_personas_destination_lists_local_behavior_snapshot_from_service():
    app = _build_test_app()
    app.character_persona_scope_service = StaticPersonasScopeService(
        characters=[
            {"name": "Research Mentor", "id": 1},
            {"name": "Code Reviewer", "id": 2},
        ],
        profiles=[
            {"name": "Socratic Tutor", "id": "persona-1", "description": "Guides by asking questions."},
        ],
    )
    host = DestinationHarness(app, "personas")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_personas_snapshot(screen, pilot)
        text = _visible_text(screen)
        button = screen.query_one("#personas-attach-to-console", Button)

        assert "Local Personas snapshot" in text
        assert "Characters: 2" in text
        assert "Persona profiles: 1" in text
        assert "Research Mentor" in text
        assert "Code Reviewer" in text
        assert "Socratic Tutor" in text
        assert button.disabled is False

    assert app.character_persona_scope_service.character_calls[0] == {
        "mode": "local",
        "limit": getattr(personas_screen_module, "PERSONAS_LOCAL_PAGE_SIZE", None),
        "offset": 0,
    }
    assert app.character_persona_scope_service.profile_calls[0] == {
        "mode": "local",
        "active_only": True,
        "include_deleted": False,
        "limit": getattr(personas_screen_module, "PERSONAS_LOCAL_PAGE_SIZE", None),
        "offset": 0,
    }


@pytest.mark.asyncio
async def test_personas_destination_waits_for_threaded_snapshot_without_fixed_sleep():
    app = _build_test_app()
    app.character_persona_scope_service = SlowPersonasScopeService(
        characters=[{"name": "Delayed Mentor", "id": 1}],
        profiles=[],
    )
    host = DestinationHarness(app, "personas")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_personas_snapshot(screen, pilot)

        assert "Delayed Mentor" in _visible_text(screen)


@pytest.mark.asyncio
async def test_personas_destination_empty_state_disables_console_attach():
    app = _build_test_app()
    app.character_persona_scope_service = StaticPersonasScopeService()
    host = DestinationHarness(app, "personas")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_personas_snapshot(screen, pilot)
        button = screen.query_one("#personas-attach-to-console", Button)

        assert "No local characters or persona profiles are available yet." in _visible_text(screen)
        assert button.disabled is True
        assert "Stage local persona context" in str(button.tooltip)


@pytest.mark.asyncio
async def test_personas_destination_service_failure_uses_recovery_copy():
    app = _build_test_app()
    app.character_persona_scope_service = RaisingPersonasScopeService()
    host = DestinationHarness(app, "personas")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_personas_snapshot(screen, pilot)
        button = screen.query_one("#personas-attach-to-console", Button)

        assert "Personas service unavailable; retry Personas later." in _visible_text(screen)
        assert button.disabled is True
        assert "Personas service is unavailable" in str(button.tooltip)


@pytest.mark.asyncio
async def test_personas_policy_denial_uses_runtime_recovery_taxonomy():
    app = _build_test_app()
    app.character_persona_scope_service = PolicyDeniedPersonasScopeService()
    host = DestinationHarness(app, "personas")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_personas_snapshot(screen, pilot)
        error = screen.query_one("#personas-service-error", Static)
        button = screen.query_one("#personas-attach-to-console", Button)

        _assert_policy_recovery_copy(
            visible_text=_static_text(error),
            button=button,
            status_label="Server sign-in required",
            unavailable_what="Attach Personas context to Console",
            why="Server Personas require sign-in",
            next_action="Reconnect or configure server credentials in Settings before retrying.",
            recovery_action="Settings",
            authority_owner="active server",
        )


@pytest.mark.asyncio
async def test_personas_attach_to_console_uses_listed_behavior_context():
    app = _build_test_app()
    app.character_persona_scope_service = StaticPersonasScopeService(
        characters=[
            {"name": "Research Mentor", "id": 1, "description": "Helps reason over sources."},
        ],
        profiles=[
            {"name": "Socratic Tutor", "id": "persona-1", "description": "Guides by asking questions."},
        ],
    )
    app.open_chat_with_handoff = Mock()
    host = DestinationHarness(app, "personas")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_personas_snapshot(screen, pilot)
        await pilot.click("#personas-attach-to-console")
        await _wait_for_mock_call(app.open_chat_with_handoff, pilot)

    app.open_chat_with_handoff.assert_called_once()
    payload = app.open_chat_with_handoff.call_args.args[0]
    assert isinstance(payload, ChatHandoffPayload)
    assert payload.source == "personas"
    assert payload.item_type == "personas-context"
    assert payload.title == "Local Personas Context"
    assert "Characters: 1" in payload.body
    assert "Persona profiles: 1" in payload.body
    assert "Research Mentor" in payload.body
    assert "Socratic Tutor" in payload.body
    assert "Stage characters, prompts, dictionaries" not in payload.body
    assert payload.metadata["character_count"] == 1
    assert payload.metadata["persona_profile_count"] == 1
    assert payload.metadata["character_names"] == ["Research Mentor"]
    assert payload.metadata["persona_profile_names"] == ["Socratic Tutor"]


def test_personas_destination_service_adoption_tracking_evidence_exists():
    evidence = PHASE4_PERSONAS_ADOPTION_EVIDENCE.read_text(encoding="utf-8")
    roadmap = Path("Docs/superpowers/trackers/unified-shell-maturity-roadmap.md").read_text(encoding="utf-8")
    task = Path(
        "backlog/tasks/task-5.4 - Phase-4.4-Adopt-Personas-service-in-Personas-destination.md"
    ).read_text(encoding="utf-8")

    assert "Phase 4.4 Personas Service Adoption" in evidence
    assert "TASK-5.4" in evidence
    assert "Phase 4.4: Adopt Personas service in Personas destination - `TASK-5.4`" in roadmap
    assert "character_persona_scope_service" in task


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


@pytest.mark.asyncio
async def test_library_destination_lists_local_source_snapshot_from_services():
    app = _build_test_app()
    app.notes_user_id = "unit-user"
    app.notes_scope_service = StaticLibraryNotesScopeService(
        [
            {"title": "Research Note", "id": "note-1"},
            {"title": "Meeting Note", "id": "note-2"},
        ]
    )
    app.media_reading_scope_service = StaticLibraryMediaScopeService(
        [{"title": "Transcript A", "id": "media-1"}]
    )
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(
        [{"title": "Planning Chat", "id": "chat-1"}]
    )
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = _active_destination_screen(host)
        text = _visible_text(screen)
        button = screen.query_one("#library-use-in-console", Button)

        assert "Local Library snapshot" in text
        assert "Notes: 2" in text
        assert "Media: 1" in text
        assert "Conversations: 1" in text
        assert "Research Note" in text
        assert "Transcript A" in text
        assert "Planning Chat" in text
        assert button.disabled is False

    assert app.notes_scope_service.calls[0] == {
        "scope": "local_note",
        "limit": getattr(library_screen_module, "LIBRARY_SOURCE_PAGE_SIZE", None),
        "offset": 0,
        "user_id": "unit-user",
    }
    assert app.media_reading_scope_service.calls[0] == {
        "mode": "local",
        "page": 1,
        "results_per_page": getattr(library_screen_module, "LIBRARY_SOURCE_PAGE_SIZE", None),
        "include_keywords": False,
    }
    assert app.chat_conversation_scope_service.calls[0] == {
        "mode": "local",
        "limit": getattr(library_screen_module, "LIBRARY_SOURCE_PAGE_SIZE", None),
        "offset": 0,
    }


@pytest.mark.asyncio
async def test_library_destination_empty_state_disables_console_handoff():
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = _active_destination_screen(host)
        button = screen.query_one("#library-use-in-console", Button)

        assert "No local Library sources are available yet." in _visible_text(screen)
        assert button.disabled is True
        assert "Stage Library source context" in str(button.tooltip)


@pytest.mark.asyncio
async def test_library_destination_labels_plain_list_notes_as_sample_snapshot():
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesListScopeService(
        [{"title": f"Research Note {index}", "id": f"note-{index}"} for index in range(1, 7)]
    )
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    app.open_chat_with_handoff = Mock()
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert "Notes (showing up to 5): 5" in text
        assert "Notes: 5" not in text
        assert "Research Note 1" in text
        assert "Research Note 5" in text
        assert "Research Note 6" not in text

        await pilot.click("#library-use-in-console")
        await pilot.pause(0.1)

    app.open_chat_with_handoff.assert_called_once()
    payload = app.open_chat_with_handoff.call_args.args[0]
    assert "Notes (showing up to 5): 5" in payload.body
    assert "Notes: 5" not in payload.body
    assert payload.metadata["notes_sample_count"] == 5
    assert payload.metadata["notes_total_count"] is None
    assert "notes_count" not in payload.metadata
    assert payload.metadata["note_titles"] == [
        "Research Note 1",
        "Research Note 2",
        "Research Note 3",
        "Research Note 4",
        "Research Note 5",
    ]


@pytest.mark.asyncio
async def test_library_destination_service_failure_uses_recovery_copy():
    app = _build_test_app()
    app.notes_scope_service = RaisingLibraryNotesScopeService()
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = _active_destination_screen(host)
        button = screen.query_one("#library-use-in-console", Button)

        assert "Library source services unavailable; retry Library later." in _visible_text(screen)
        assert button.disabled is True
        assert "Library source services are unavailable" in str(button.tooltip)


@pytest.mark.asyncio
async def test_library_policy_denial_uses_runtime_recovery_taxonomy():
    app = _build_test_app()
    app.notes_scope_service = PolicyDeniedLibraryNotesScopeService()
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        error = screen.query_one("#library-source-error", Static)
        button = screen.query_one("#library-use-in-console", Button)

        _assert_policy_recovery_copy(
            visible_text=_static_text(error),
            button=button,
            status_label="Wrong source",
            unavailable_what="Use Library sources in Console",
            why="Server Library sources require server mode",
            next_action="Switch to the required source, then retry this workflow.",
            recovery_action="Source switch or Settings",
            authority_owner="active server",
        )


@pytest.mark.asyncio
async def test_library_use_in_console_uses_source_snapshot_context():
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService(
        [{"title": "Research Note", "id": "note-1"}]
    )
    app.media_reading_scope_service = StaticLibraryMediaScopeService(
        [{"title": "Transcript A", "id": "media-1"}]
    )
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(
        [{"title": "Planning Chat", "id": "chat-1"}]
    )
    app.open_chat_with_handoff = Mock()
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        await pilot.click("#library-use-in-console")
        await pilot.pause(0.1)

    app.open_chat_with_handoff.assert_called_once()
    payload = app.open_chat_with_handoff.call_args.args[0]
    assert isinstance(payload, ChatHandoffPayload)
    assert payload.source == "library"
    assert payload.item_type == "library-source-snapshot"
    assert payload.title == "Local Library Sources"
    assert "Notes: 1" in payload.body
    assert "Media: 1" in payload.body
    assert "Conversations: 1" in payload.body
    assert "Research Note" in payload.body
    assert "Transcript A" in payload.body
    assert "Planning Chat" in payload.body
    assert "Stage Library source material" not in payload.body
    assert payload.metadata["notes_count"] == 1
    assert payload.metadata["media_count"] == 1
    assert payload.metadata["conversations_count"] == 1


def test_library_destination_service_adoption_tracking_evidence_exists():
    evidence = PHASE4_LIBRARY_ADOPTION_EVIDENCE.read_text(encoding="utf-8")
    roadmap = Path("Docs/superpowers/trackers/unified-shell-maturity-roadmap.md").read_text(encoding="utf-8")
    task = Path(
        "backlog/tasks/task-5.3 - Phase-4.3-Adopt-Library-source-services-in-Library-destination.md"
    ).read_text(encoding="utf-8")

    assert "Phase 4.3 Library Source Service Adoption" in evidence
    assert "TASK-5.3" in evidence
    assert "Phase 4.3: Adopt Library source services in Library destination - `TASK-5.3`" in roadmap
    assert "notes_scope_service" in task
    assert "media_reading_scope_service" in task


def test_wc_service_adoption_tracking_evidence_exists():
    evidence = PHASE4_WC_ADOPTION_EVIDENCE.read_text(encoding="utf-8")
    roadmap = Path("Docs/superpowers/trackers/unified-shell-maturity-roadmap.md").read_text(encoding="utf-8")
    task = Path(
        "backlog/tasks/task-5.5 - Phase-4.5-Adopt-WC-services-in-WC-destination.md"
    ).read_text(encoding="utf-8")

    assert "Phase 4.5 W+C Service Adoption" in evidence
    assert "TASK-5.5" in evidence
    assert "Phase 4.5: Adopt W+C services in W+C destination - `TASK-5.5`" in roadmap
    assert "watchlist_scope_service" in task
    assert "media_reading_scope_service" in task


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
        screen = _active_destination_screen(host)
        await _wait_for_skills_snapshot(screen, pilot)
        button = screen.query_one("#skills-attach-to-console", Button)

        _assert_policy_recovery_copy(
            visible_text=_visible_text(screen),
            button=button,
            status_label="Policy denied",
            unavailable_what="Attach local Skills to Console",
            why="Local Skills are disabled by workspace policy",
            next_action="Review workspace policy or ask the authority owner to allow this action.",
            recovery_action="Workspace policy",
            authority_owner="local",
        )
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
