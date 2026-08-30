"""Settings ▸ Workspaces "Default assistant" section (Task 10).

Covers the pure posture-preview helper plus the pane section: effective
status, persona/profile pickers, the two-press read_write confirm, clear,
degraded copy, locked default workspace, and the tool-catalog degrade.
"""

from __future__ import annotations

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
        await pilot.pause(0.3)

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
        await pilot.pause(0.3)
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
        assert "stays tool-less" in _visible_text(screen)


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
        await pilot.pause(0.3)

        defaults = registry.get_workspace("ws-g").assistant_defaults
        assert defaults is not None
        assert defaults.assistant_id == "persona-1"
        assert defaults.tool_policy_profile_id == "profile-b"
