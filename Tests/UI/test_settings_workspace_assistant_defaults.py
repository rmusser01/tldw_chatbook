"""Settings ▸ Workspaces "Default assistant" section (Task 10).

Covers the pure posture-preview helper plus the pane section: effective
status, persona/profile pickers, the two-press read_write confirm, clear,
degraded copy, locked default workspace, and the tool-catalog degrade.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import threading
from types import SimpleNamespace
from typing import Any

import pytest
from textual.widgets import Button, OptionList, Static

from Tests.UI.test_settings_configuration_hub import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _open_settings_category,
    _visible_text,
)
from tldw_chatbook.Workspaces.assistant_defaults import compose_posture_preview
from tldw_chatbook.Workspaces.models import WorkspaceAssistantDefaults
from tldw_chatbook.Tool_Packs.binding import (
    ToolProfileBindingReview,
    ToolProfileBindingSummary,
    ToolProfileConfirmationRequired,
)
from tldw_chatbook.Widgets.Settings_Widgets.tool_pack_import_review import (
    ToolProfileFirstBindReviewModal,
)


class FakePersonaService:
    """Minimal stub of LocalCharacterPersonaService's list/get surface."""

    def __init__(self, personas: list[dict[str, Any]]) -> None:
        self._personas = personas

    def list_persona_profiles(self, **_: Any) -> list[dict[str, Any]]:
        return [dict(persona) for persona in self._personas]

    def get_persona_profile(self, persona_id: str) -> dict[str, Any] | None:
        for persona in self._personas:
            if persona["id"] == persona_id:
                if persona.get("deleted"):
                    return {"id": persona_id, "deleted": True, "name": persona.get("name", "")}
                return dict(persona)
        return None


class FakePermissionStore:
    def __init__(self, payload: dict[str, Any], profiles: list[str]) -> None:
        self._payload = payload
        self._profiles = profiles

    def load(self) -> dict[str, Any]:
        return self._payload

    def list_profiles(self) -> list[str]:
        return list(self._profiles)


def _fake_unified_service(
    tool_names: list[str] | None = None,
    store: Any = None,
) -> SimpleNamespace:
    service = SimpleNamespace()
    if tool_names is not None:
        service.local_service = SimpleNamespace(
            get_inventory=lambda: {
                "tools": [{"name": name} for name in tool_names]
            }
        )
    if store is not None:
        service.permission_store = store
    return service


def _stub_assistant_services(
    app: Any,
    personas: list[dict[str, Any]] | None = None,
    tool_names: list[str] | None = None,
    permission_payload: dict[str, Any] | None = None,
    profiles: list[str] | None = None,
) -> None:
    app.local_character_persona_service = FakePersonaService(personas or [])
    app.unified_mcp_service = _fake_unified_service(
        tool_names=tool_names,
        store=FakePermissionStore(
            permission_payload or {}, profiles or ["default"]
        ),
    )


class _FakeOptionSelected:
    """Duck-typed stand-in for OptionList.OptionSelected events."""

    def __init__(self, option: Any) -> None:
        self.option = option

    def stop(self) -> None:  # noqa: D102 - event protocol
        return None


def _persona(persona_id: str, name: str, **extra: Any) -> dict[str, Any]:
    return {"id": persona_id, "name": name, "system_prompt": "", **extra}


def _first_bind_review(
    workspace_id: str,
    defaults: WorkspaceAssistantDefaults,
    action: str,
) -> ToolProfileBindingReview:
    return ToolProfileBindingReview(
        workspace_id=workspace_id,
        action=action,  # type: ignore[arg-type]
        intended_defaults_digest="6" * 64,
        profile_id=str(defaults.tool_policy_profile_id),
        policy_digest="a" * 64,
        revision=3,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
        summary=ToolProfileBindingSummary(
            global_fallback="ask",
            builtin_fallback="deny",
            allow_server_fallbacks=(),
            stored_exact_allows=(),
            effective_allows=(),
            unavailable_allows=(),
            downgraded_allows=(),
            high_risk_allows=(),
            allow_count=0,
            ask_count=1,
            deny_count=0,
            inventory_digest="4" * 64,
        ),
    )


class _FirstBindGuard:
    """Registry guard that accepts only the token issued by the fake service."""

    def __init__(self) -> None:
        self.accepted: list[tuple[str, bool]] = []

    @contextmanager
    def mutation_scope(self, **context: Any):
        defaults = context["intended_defaults"]
        if (
            defaults is not None
            and defaults.tool_policy_profile_id == "research"
        ):
            token = context["confirmation_token"]
            if token is None:
                raise ToolProfileConfirmationRequired()
            if token != "exact-first-bind-token":
                raise AssertionError("unexpected first-bind token")
            self.accepted.append((context["workspace_id"], token is not None))
        yield


class _FirstBindService:
    def __init__(self) -> None:
        self.reviews: list[ToolProfileBindingReview] = []
        self.confirmed: list[ToolProfileBindingReview] = []

    def review_first_bind(
        self,
        workspace_id: str,
        defaults: WorkspaceAssistantDefaults,
        *,
        action: str,
    ) -> ToolProfileBindingReview:
        review = _first_bind_review(workspace_id, defaults, action)
        self.reviews.append(review)
        return review

    def confirm_first_bind(self, review: ToolProfileBindingReview) -> str:
        assert self.reviews[-1] is review
        self.confirmed.append(review)
        return "exact-first-bind-token"


# ---------------------------------------------------------------------------
# Pure helper: compose_posture_preview
# ---------------------------------------------------------------------------


def test_posture_preview_kill_switch_denies_everything() -> None:
    lines = compose_posture_preview(
        [],
        {"kill_switch": True},
        "default",
        ["search_notes", "write_note"],
    )
    assert lines == [
        "search_notes: denied — kill switch",
        "write_note: denied — kill switch",
    ]


def test_posture_preview_persona_layers() -> None:
    rules = [
        {"rule_kind": "mcp_tool", "rule_name": "search_notes", "allowed": False},
        {
            "rule_kind": "mcp_tool",
            "rule_name": "write_note",
            "require_confirmation": True,
        },
        {
            "rule_kind": "mcp_tool",
            "rule_name": "fetch_url",
            "max_calls_per_turn": 2,
        },
    ]
    lines = compose_posture_preview(
        rules, {}, "default", ["search_notes", "write_note", "fetch_url"]
    )
    assert lines == [
        "search_notes: denied — persona policy",
        "write_note: ask — persona policy",
        "fetch_url: capped (2) — persona policy",
    ]


def test_posture_preview_permission_resolution() -> None:
    payload = {
        "profiles": {
            "default": {
                "servers": {
                    "builtin:tldw_chatbook": {
                        "tools": {
                            "search_notes": {"state": "allow"},
                            "write_note": {"state": "deny"},
                        }
                    }
                }
            },
            "profile-b": {
                "servers": {
                    "builtin:tldw_chatbook": {
                        "tools": {"search_notes": {"state": "ask"}}
                    }
                }
            },
        }
    }
    # Named profile shadows the default profile, level by level.
    lines = compose_posture_preview([], payload, "profile-b", ["search_notes", "write_note"])
    assert lines == [
        "search_notes: ask — permissions",
        "write_note: denied — permissions",
    ]
    # The default profile resolves search_notes at full fidelity.
    lines = compose_posture_preview([], payload, "default", ["search_notes", "write_note"])
    assert lines == [
        "search_notes: available — permissions",
        "write_note: denied — permissions",
    ]


def test_posture_preview_empty_tool_names_degrades() -> None:
    assert compose_posture_preview([], {}, "default", []) == [
        "Tool catalog unavailable"
    ]


# ---------------------------------------------------------------------------
# UI section
# ---------------------------------------------------------------------------


async def _open_workspace_card(pilot, app, workspace_id: str):
    screen = _active_destination_screen(app)
    await _open_settings_category(pilot, "#settings-category-workspaces")
    screen.query_one(f"#settings-workspace-row-{workspace_id}", Button).press()
    await pilot.pause(0.2)
    return screen


@pytest.mark.asyncio
async def test_section_renders_effective_status_and_pickers() -> None:
    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-a", name="Alpha WS")
    registry.set_assistant_defaults(
        "ws-a",
        WorkspaceAssistantDefaults(assistant_id="persona-1"),
    )
    _stub_assistant_services(
        app,
        personas=[
            _persona("persona-1", "Helper"),
            _persona("persona-2", "Other"),
        ],
        tool_names=["search_notes"],
        profiles=["default", "profile-b"],
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _open_workspace_card(pilot, host, "ws-a")

        assert "Default assistant" in _visible_text(screen)
        status = screen.query_one(
            "#settings-workspace-assistant-status", Static
        )
        assert "Helper" in str(status.renderable)
        assert "read_only" in str(status.renderable)
        assert screen.query_one("#settings-workspace-persona-picker", OptionList)
        assert screen.query_one("#settings-workspace-profile-picker", OptionList)
        assert screen.query_one("#settings-workspace-memory-toggle", Button)
        assert screen.query_one("#settings-workspace-assistant-clear", Button)
        preview = screen.query_one(
            "#settings-workspace-posture-preview", Static
        )
        # No persona policy rules and an empty store: global default is ask.
        assert "search_notes: ask — permissions" in str(preview.renderable)


@pytest.mark.asyncio
async def test_selecting_persona_then_press_applies_read_only() -> None:
    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-b", name="Bravo WS")
    _stub_assistant_services(
        app,
        personas=[_persona("persona-1", "Helper")],
        tool_names=["search_notes"],
        profiles=["default"],
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _open_workspace_card(pilot, host, "ws-b")

        picker = screen.query_one("#settings-workspace-persona-picker", OptionList)
        screen.handle_workspace_persona_selected(
            _FakeOptionSelected(picker.get_option_at_index(0))
        )
        await pilot.pause(0.2)

        screen.query_one("#settings-workspace-memory-toggle", Button).press()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        defaults = registry.get_workspace("ws-b").assistant_defaults
        assert defaults is not None
        assert defaults.assistant_id == "persona-1"
        assert defaults.persona_memory_mode == "read_only"


@pytest.mark.asyncio
async def test_read_write_requires_two_presses() -> None:
    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-c", name="Charlie WS")
    registry.set_assistant_defaults(
        "ws-c",
        WorkspaceAssistantDefaults(assistant_id="persona-1"),
    )
    _stub_assistant_services(
        app,
        personas=[_persona("persona-1", "Helper")],
        tool_names=[],
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _open_workspace_card(pilot, host, "ws-c")
        toggle = screen.query_one("#settings-workspace-memory-toggle", Button)

        # First press arms: label flips, nothing persists yet.
        toggle.press()
        await pilot.pause(0.2)
        assert "Confirm read_write?" in _visible_text(screen)
        assert (
            registry.get_workspace("ws-c").assistant_defaults.persona_memory_mode
            == "read_only"
        )

        # Second press applies with the explicit confirmation.
        screen.query_one("#settings-workspace-memory-toggle", Button).press()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        defaults = registry.get_workspace("ws-c").assistant_defaults
        assert defaults.persona_memory_mode == "read_write"
        assert "read_write" in _visible_text(screen)


@pytest.mark.asyncio
async def test_clear_removes_default() -> None:
    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-d", name="Delta WS")
    registry.set_assistant_defaults(
        "ws-d",
        WorkspaceAssistantDefaults(assistant_id="persona-1"),
    )
    _stub_assistant_services(
        app,
        personas=[_persona("persona-1", "Helper")],
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _open_workspace_card(pilot, host, "ws-d")
        screen.query_one("#settings-workspace-assistant-clear", Button).press()
        await pilot.pause(0.3)
        assert registry.get_workspace("ws-d").assistant_defaults is None


@pytest.mark.asyncio
async def test_degraded_persona_shows_reason_copy() -> None:
    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-e", name="Echo WS")
    registry.set_assistant_defaults(
        "ws-e",
        WorkspaceAssistantDefaults(assistant_id="persona-gone"),
    )
    _stub_assistant_services(
        app,
        personas=[_persona("persona-1", "Helper")],
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _open_workspace_card(pilot, host, "ws-e")
        status = screen.query_one("#settings-workspace-assistant-status", Static)
        assert "persona_deleted" in str(status.renderable)


@pytest.mark.asyncio
async def test_default_workspace_renders_locked_note_not_picker() -> None:
    app = _build_test_app()
    _stub_assistant_services(app, personas=[_persona("persona-1", "Helper")])
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-workspaces")
        screen.query_one(
            "#settings-workspace-row-workspace-default", Button
        ).press()
        await pilot.pause(0.2)
        assert not screen.query("#settings-workspace-persona-picker")
        assert not screen.query("#settings-workspace-memory-toggle")
        # Dev drifted the Default-workspace protection copy; the feature's
        # contract is only that the Default card shows a locked note and no
        # assistant-defaults controls.
        assert "built-in Default workspace" in _visible_text(screen)


@pytest.mark.asyncio
async def test_posture_preview_degrades_without_tool_catalog() -> None:
    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-f", name="Foxtrot WS")
    _stub_assistant_services(
        app,
        personas=[_persona("persona-1", "Helper")],
        tool_names=None,
    )
    # No local_service on the unified stub -> catalog unavailable.
    app.unified_mcp_service = SimpleNamespace(
        permission_store=FakePermissionStore({}, ["default"])
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _open_workspace_card(pilot, host, "ws-f")
        preview = screen.query_one("#settings-workspace-posture-preview", Static)
        assert "Tool catalog unavailable" in str(preview.renderable)


@pytest.mark.asyncio
async def test_profile_selection_is_applied_with_default() -> None:
    app = _build_test_app()
    registry = app.workspace_registry_service
    bootstrap = app._tool_pack_guard_bootstrap

    class AllowingGuard:
        @contextmanager
        def mutation_scope(self, **_context: Any):
            yield

    assert bootstrap.activate(AllowingGuard()) is True
    registry.create_workspace(workspace_id="ws-g", name="Golf WS")
    _stub_assistant_services(
        app,
        personas=[_persona("persona-1", "Helper")],
        tool_names=["search_notes"],
        profiles=["default", "profile-b"],
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _open_workspace_card(pilot, host, "ws-g")

        persona_picker = screen.query_one(
            "#settings-workspace-persona-picker", OptionList
        )
        screen.handle_workspace_persona_selected(
            _FakeOptionSelected(persona_picker.get_option_at_index(0))
        )
        await pilot.pause(0.2)
        profile_picker = screen.query_one(
            "#settings-workspace-profile-picker", OptionList
        )
        options = [
            profile_picker.get_option_at_index(index)
            for index in range(profile_picker.option_count)
        ]
        profile_option = next(
            option for option in options if getattr(option, "profile_id", None) == "profile-b"
        )
        screen.handle_workspace_profile_selected(
            _FakeOptionSelected(profile_option)
        )
        await pilot.pause(0.2)

        screen.query_one("#settings-workspace-memory-toggle", Button).press()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        defaults = registry.get_workspace("ws-g").assistant_defaults
        assert defaults is not None
        assert defaults.assistant_id == "persona-1"
        assert defaults.tool_policy_profile_id == "profile-b"


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement_workspace_id", ("ws-slow", "ws-new"))
async def test_slow_apply_does_not_clear_newer_staging(
    monkeypatch: pytest.MonkeyPatch,
    replacement_workspace_id: str,
) -> None:
    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-slow", name="Slow WS")
    registry.create_workspace(workspace_id="ws-new", name="New WS")
    _stub_assistant_services(
        app,
        personas=[_persona("persona-1", "Helper")],
        tool_names=[],
    )
    entered = threading.Event()
    release = threading.Event()
    original = registry.set_assistant_defaults

    def slow_set(*args: Any, **kwargs: Any):
        entered.set()
        if not release.wait(5):
            raise RuntimeError("test apply was not released")
        return original(*args, **kwargs)

    monkeypatch.setattr(registry, "set_assistant_defaults", slow_set)
    host = DestinationHarness(app, "settings")

    try:
        async with host.run_test(size=(180, 50)) as pilot:
            screen = await _open_workspace_card(pilot, host, "ws-slow")
            picker = screen.query_one(
                "#settings-workspace-persona-picker", OptionList
            )
            screen.handle_workspace_persona_selected(
                _FakeOptionSelected(picker.get_option_at_index(0))
            )
            await pilot.pause(0.2)
            screen.query_one("#settings-workspace-memory-toggle", Button).press()
            await pilot.pause()
            started = await asyncio.to_thread(entered.wait, 2)
            if not started:
                release.set()
            assert started

            replacement = {
                "workspace_id": replacement_workspace_id,
                "profile_id": None,
                "persona_id": "persona-1",
                "memory_mode": "read_write",
            }
            screen._settings_selected_workspace_id = replacement_workspace_id
            screen._settings_workspace_assistant_pending = replacement
            screen._settings_workspace_memory_armed = replacement_workspace_id
            release.set()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()

            assert screen._settings_workspace_assistant_pending is replacement
            assert (
                screen._settings_workspace_memory_armed == replacement_workspace_id
            )
            assert registry.get_workspace("ws-slow").assistant_defaults is not None
    finally:
        release.set()


@pytest.mark.asyncio
async def test_imported_profile_first_bind_requires_current_review_and_exact_token() -> None:
    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-bind", name="Bind WS")
    guard = _FirstBindGuard()
    registry.attach_tool_profile_guard(guard)
    service = _FirstBindService()
    app.tool_pack_service = service
    _stub_assistant_services(
        app,
        personas=[_persona("persona-1", "Helper")],
        tool_names=["search_notes"],
        profiles=["default", "research"],
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _open_workspace_card(pilot, host, "ws-bind")
        persona_picker = screen.query_one(
            "#settings-workspace-persona-picker", OptionList
        )
        screen.handle_workspace_persona_selected(
            _FakeOptionSelected(persona_picker.get_option_at_index(0))
        )
        profile_picker = screen.query_one(
            "#settings-workspace-profile-picker", OptionList
        )
        profile = next(
            profile_picker.get_option_at_index(index)
            for index in range(profile_picker.option_count)
            if getattr(
                profile_picker.get_option_at_index(index), "profile_id", None
            )
            == "research"
        )
        screen.handle_workspace_profile_selected(_FakeOptionSelected(profile))
        await pilot.pause()

        screen.query_one("#settings-workspace-memory-toggle", Button).press()
        await pilot.pause(0.3)

        assert isinstance(host.screen, ToolProfileFirstBindReviewModal)
        assert registry.get_workspace("ws-bind").assistant_defaults is None
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        defaults = registry.get_workspace("ws-bind").assistant_defaults
        assert defaults is not None
        assert defaults.assistant_id == "persona-1"
        assert defaults.persona_memory_mode == "read_only"
        assert defaults.tool_policy_profile_id == "research"
        assert service.confirmed == service.reviews
        assert guard.accepted == [("ws-bind", True)]


@pytest.mark.asyncio
async def test_read_write_acknowledgement_remains_separate_from_first_bind_review() -> None:
    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-bind-rw", name="Bind RW")
    registry.set_assistant_defaults(
        "ws-bind-rw",
        WorkspaceAssistantDefaults(assistant_id="persona-1"),
    )
    registry.attach_tool_profile_guard(_FirstBindGuard())
    app.tool_pack_service = _FirstBindService()
    _stub_assistant_services(
        app,
        personas=[_persona("persona-1", "Helper")],
        tool_names=["search_notes"],
        profiles=["default", "research"],
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _open_workspace_card(pilot, host, "ws-bind-rw")
        profile_picker = screen.query_one(
            "#settings-workspace-profile-picker", OptionList
        )
        profile = next(
            profile_picker.get_option_at_index(index)
            for index in range(profile_picker.option_count)
            if getattr(
                profile_picker.get_option_at_index(index), "profile_id", None
            )
            == "research"
        )
        screen.handle_workspace_profile_selected(_FakeOptionSelected(profile))
        await pilot.pause()

        toggle = screen.query_one("#settings-workspace-memory-toggle", Button)
        toggle.press()
        await pilot.pause()
        assert "Confirm read_write?" in _visible_text(screen)
        assert not isinstance(host.screen, ToolProfileFirstBindReviewModal)
        assert (
            registry.get_workspace("ws-bind-rw").assistant_defaults.persona_memory_mode
            == "read_only"
        )

        screen.query_one("#settings-workspace-memory-toggle", Button).press()
        await pilot.pause(0.3)
        assert isinstance(host.screen, ToolProfileFirstBindReviewModal)
        assert (
            registry.get_workspace("ws-bind-rw").assistant_defaults.persona_memory_mode
            == "read_only"
        )

        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        defaults = registry.get_workspace("ws-bind-rw").assistant_defaults
        assert defaults.persona_memory_mode == "read_write"
        assert defaults.tool_policy_profile_id == "research"
