"""A memory acknowledgement belongs to the reviewed workspace/defaults."""

import asyncio
from contextlib import contextmanager
from dataclasses import replace

import pytest
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_rag_result_focus import _assert_painted
from Tests.UI.test_settings_overview_search_journeys import _category, _painted
from Tests.UI.test_settings_provider_keyboard_journeys import _settle
from Tests.UI.test_settings_speech_tts_panel import _StyledDestinationHarness
from Tests.UI.test_settings_workspace_assistant_defaults import (
    _persona,
    _stub_assistant_services,
)
from tldw_chatbook.Workspaces.models import WorkspaceAssistantDefaults


def _fixture():
    app = _build_test_app()
    registry = app.workspace_registry_service

    class LocalProfileGuard:
        @contextmanager
        def mutation_scope(self, **_context):
            yield

    assert app._tool_pack_guard_bootstrap.activate(LocalProfileGuard())
    registry.create_workspace(workspace_id="ws-review", name="Review workspace")
    registry.create_workspace(workspace_id="ws-other", name="Other workspace")
    defaults = WorkspaceAssistantDefaults(
        assistant_id="helper", tool_policy_profile_id="research"
    )
    registry.set_assistant_defaults("ws-review", defaults)
    _stub_assistant_services(
        app,
        personas=[_persona("helper", "Research helper"), _persona("other", "Other")],
        profiles=["default", "research", "other"],
    )
    return app, registry, defaults


async def _press(host, pilot, selector):
    button = host.screen.query_one(selector, Button)
    button.focus()
    await _settle(host, pilot)
    async with asyncio.timeout(2):
        while button.has_class("-active"):
            await pilot.pause(0.02)
    _assert_painted(host.screen, button)
    await pilot.press("enter")
    await _settle(host, pilot)


async def _arm(host, pilot, registry, defaults):
    await _press(host, pilot, "#settings-workspace-memory-toggle")
    assert registry.get_workspace("ws-review").assistant_defaults == defaults
    assert str(host.screen.query_one("#settings-workspace-memory-toggle").label) == (
        "Confirm read_write?"
    )
    _receipt(host, "press again to confirm")


def _receipt(host, needle):
    receipt = host.screen.query_one("#settings-workspace-assistant-result", Static)
    assert needle in str(receipt.renderable)
    _assert_painted(host.screen, host.focused)
    assert " ".join(str(receipt.renderable).split()) in " ".join(
        _painted(host, receipt).split()
    )


@pytest.mark.parametrize("navigation", ["workspace", "category", "modal"])
@private_profile_test
async def test_navigation_requires_fresh_memory_acknowledgement(request, navigation):
    app, registry, defaults = _fixture()
    host = _StyledDestinationHarness(app, "settings")
    async with host.run_test(size=(170, 48)) as pilot:
        await _category(host, pilot, "Workspaces")
        await _press(host, pilot, "#settings-workspace-row-ws-review")
        await _arm(host, pilot, registry, defaults)
        if navigation == "workspace":
            await _press(host, pilot, "#settings-workspace-row-ws-other")
            await _press(host, pilot, "#settings-workspace-row-ws-review")
        elif navigation == "category":
            await _category(host, pilot, "Overview")
            await _category(host, pilot, "Workspaces")
            await _press(host, pilot, "#settings-workspace-row-ws-review")
        else:
            await host.push_screen(ModalScreen())
            await pilot.pause()
            await host.pop_screen()
            await _settle(host, pilot)
        assert str(
            host.screen.query_one("#settings-workspace-memory-toggle").label
        ) == ("Set memory: read_write")
        assert not host.screen.query_one(
            "#settings-workspace-assistant-result"
        ).renderable
        await _arm(host, pilot, registry, defaults)
        await _press(host, pilot, "#settings-workspace-memory-toggle")
        assert registry.get_workspace("ws-review").assistant_defaults == replace(
            defaults, persona_memory_mode="read_write"
        )
        _receipt(host, "Default assistant applied")


@pytest.mark.parametrize("changed_field", ["persona", "profile", "memory", "clear"])
@private_profile_test
async def test_changed_saved_defaults_reject_stale_memory_acknowledgement(
    request, changed_field
):
    app, registry, defaults = _fixture()
    host = _StyledDestinationHarness(app, "settings")
    async with host.run_test(size=(170, 48)) as pilot:
        await _category(host, pilot, "Workspaces")
        await _press(host, pilot, "#settings-workspace-row-ws-review")
        await _arm(host, pilot, registry, defaults)
        changed = {
            "persona": replace(defaults, assistant_id="other"),
            "profile": replace(defaults, tool_policy_profile_id="other"),
            "memory": replace(defaults, persona_memory_mode="read_write"),
            "clear": None,
        }[changed_field]
        if changed is None:
            registry.clear_assistant_defaults("ws-review")
        else:
            registry.set_assistant_defaults(
                "ws-review", changed, confirm_read_write=True
            )
        await _press(host, pilot, "#settings-workspace-memory-toggle")
        assert registry.get_workspace("ws-review").assistant_defaults == changed
        assert host.screen._settings_workspace_memory_armed is None
        _receipt(host, "changed")
        if changed_field in {"persona", "profile"}:
            await _arm(host, pilot, registry, changed)
            await _press(host, pilot, "#settings-workspace-memory-toggle")
            assert registry.get_workspace("ws-review").assistant_defaults == replace(
                changed, persona_memory_mode="read_write"
            )
            _receipt(host, "Default assistant applied")
