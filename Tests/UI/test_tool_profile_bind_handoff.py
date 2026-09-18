"""Bind hands off to a visible assistant draft without silently applying it."""

from contextlib import contextmanager

import pytest
from textual.widgets import Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_settings_configuration_hub import StyledSettingsDestinationHarness
from Tests.UI.test_settings_tool_profiles import (
    ToolProfileListing,
    _build_test_app,
    _open_settings_category,
    _profile,
    _WorkflowService,
)
from Tests.UI.test_settings_workspace_assistant_defaults import (
    _persona,
    _stub_assistant_services,
)
from Tests.UI.test_tool_profile_removal_outcomes import _visible
from Tests.UI.test_tool_profile_review_lifetime import _activate, _wait
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Workspaces.models import WorkspaceAssistantDefaults


def _host(*, active=True):
    app = _build_test_app()
    registry = app.workspace_registry_service
    for index in range(8):
        registry.create_workspace(workspace_id=f"ws-{index}", name=f"Workspace {index}")
    if active:
        registry.set_active_workspace("ws-7")
    _stub_assistant_services(
        app,
        personas=[_persona("helper", "Local helper")],
        profiles=["default", "research"],
    )
    app.tool_pack_service = _WorkflowService(
        ToolProfileListing(
            profiles=(_profile("research", origin="imported", binding_state="unbound"),)
        )
    )
    return StyledSettingsDestinationHarness(app, "settings"), registry


async def _settle(pilot):
    await pilot.app.workers.wait_for_complete()
    await pilot.wait_for_scheduled_animations()
    await pilot.pause()


async def _bind(pilot):
    await _open_settings_category(pilot, "#settings-category-tool-profiles")
    await _settle(pilot)
    await _activate(pilot, "#tool-profile-bind-0")
    await _wait(pilot, lambda: pilot.app.screen.active_category == "workspaces")
    await _settle(pilot)


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(80, 24), (170, 48)])
@private_profile_test
async def test_bind_preserves_persona_draft_and_reveals_apply(request, theme, size):
    host, registry = _host()
    host.theme = theme
    async with host.run_test(size=size) as pilot:
        await _open_settings_category(pilot, "#settings-category-workspaces")
        await _settle(pilot)
        await _activate(pilot, "#settings-workspace-row-ws-7")
        await _settle(pilot)
        picker = host.screen.query_one("#settings-workspace-persona-picker")
        picker.focus()
        picker.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        await pilot.press("home", "enter")
        await _settle(pilot)
        before = dict(host.screen._settings_workspace_assistant_pending)
        assert before["persona_id"] == "helper"
        await _bind(pilot)
        assert host.screen._settings_workspace_assistant_pending == {
            **before,
            "profile_id": "research",
        }
        assert registry.get_workspace("ws-7").assistant_defaults is None
        assert host.focused.id == "settings-workspace-memory-toggle"
        _visible(host, host.focused)
        receipt = host.screen.query_one("#settings-workspace-assistant-result", Static)
        assert "staged" in str(receipt.renderable)
        _visible(host, receipt)


@private_profile_test
async def test_bind_without_persona_reveals_picker_and_guidance(request):
    host, registry = _host()
    async with host.run_test(size=(80, 24)) as pilot:
        await _bind(pilot)
        assert registry.get_workspace("ws-7").assistant_defaults is None
        assert host.focused.id == "settings-workspace-persona-picker"
        _visible(host, host.focused)
        receipt = host.screen.query_one("#settings-workspace-assistant-result", Static)
        assert "choose a persona" in str(receipt.renderable)
        _visible(host, receipt)


@private_profile_test
async def test_bind_without_explicit_workspace_reveals_recovery(request):
    host, registry = _host(active=False)
    async with host.run_test(size=(80, 24)) as pilot:
        await _bind(pilot)
        assert host.screen._settings_workspace_assistant_pending is None
        assert registry.get_workspace("ws-7").assistant_defaults is None
        assert host.focused.id == "settings-workspace-create"
        _visible(host, host.focused)
        receipt = host.screen.query_one("#settings-workspaces-result", Static)
        assert "return to Tool Profiles" in str(receipt.renderable)
        _visible(host, receipt)


@pytest.mark.parametrize("draft", ["same", "other", "none"])
@private_profile_test
async def test_bind_preserves_only_the_target_workspace_draft(request, draft):
    host, registry = _host()

    class LocalProfileGuard:
        @contextmanager
        def mutation_scope(self, **_context):
            yield

    assert host.app_instance._tool_pack_guard_bootstrap.activate(LocalProfileGuard())
    saved = WorkspaceAssistantDefaults(
        assistant_id="helper", tool_policy_profile_id="default"
    )
    registry.set_assistant_defaults("ws-7", saved)
    async with host.run_test(size=(80, 24)) as pilot:
        await _open_settings_category(pilot, "#settings-category-tool-profiles")
        await _settle(pilot)
        if draft != "none":
            host.screen._settings_workspace_assistant_pending = {
                "workspace_id": "ws-7" if draft == "same" else "ws-6",
                "persona_id": "helper",
                "memory_mode": "read_write",
                "profile_id": "default",
            }
        await _activate(pilot, "#tool-profile-bind-0")
        await _settle(pilot)
        assert host.screen._settings_workspace_assistant_pending == {
            "workspace_id": "ws-7",
            "persona_id": "helper",
            "memory_mode": "read_write" if draft == "same" else "read_only",
            "profile_id": "research",
        }
        assert registry.get_workspace("ws-7").assistant_defaults == saved


@pytest.mark.parametrize("newer", ["focus", "category", "workspace", "dialog"])
@private_profile_test
async def test_late_bind_continuation_respects_newer_intent(
    request, monkeypatch, newer
):
    host, registry = _host()
    async with host.run_test(size=(80, 24)) as pilot:
        settings = host.screen
        finish = settings._finish_tool_profile_bind_handoff
        calls = []
        monkeypatch.setattr(
            settings,
            "_finish_tool_profile_bind_handoff",
            lambda *args: calls.append(args),
        )
        await _bind(pilot)
        assert len(calls) == 1
        if newer == "category":
            await _open_settings_category(pilot, "#settings-category-theme")
        elif newer == "workspace":
            await _activate(pilot, "#settings-workspace-row-ws-6")
        elif newer == "dialog":
            host.push_screen(ConfirmationDialog("Newer review", "Keep this open?"))
        else:
            settings.query_one("#settings-workspaces-show-archived").focus()
        await _settle(pilot)
        focused = host.focused
        pane = settings.query_one("#settings-detail-pane")
        scroll = pane.scroll_offset
        finish(*calls[0])
        await _settle(pilot)
        assert host.focused is focused
        assert pane.scroll_offset == scroll
        assert registry.get_workspace("ws-7").assistant_defaults is None


@private_profile_test
async def test_delayed_bind_continuation_reveals_after_layout_settled(
    request, monkeypatch
):
    host, registry = _host()
    async with host.run_test(size=(80, 24)) as pilot:
        settings = host.screen
        finish = settings._finish_tool_profile_bind_handoff
        calls = []
        monkeypatch.setattr(
            settings,
            "_finish_tool_profile_bind_handoff",
            lambda *args: calls.append(args),
        )
        await _bind(pilot)
        assert len(calls) == 1
        finish(*calls[0])
        await _settle(pilot)
        assert host.focused.id == "settings-workspace-persona-picker"
        _visible(host, host.focused)
        receipt = settings.query_one("#settings-workspace-assistant-result", Static)
        _visible(host, receipt)
        assert registry.get_workspace("ws-7").assistant_defaults is None
