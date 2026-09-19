"""Mounted first-bind review ownership and keyboard evidence."""

import asyncio
import threading

import pytest
from textual.screen import ModalScreen
from textual.widgets import Button, OptionList

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_rag_result_focus import _assert_painted
from Tests.UI.test_settings_overview_search_journeys import _category
from Tests.UI.test_settings_speech_tts_panel import _StyledDestinationHarness
from Tests.UI.test_settings_workspace_assistant_defaults import (
    _FirstBindGuard,
    _FirstBindService,
    _persona,
    _stub_assistant_services,
)
from tldw_chatbook.Widgets.Settings_Widgets.tool_pack_import_review import (
    ToolProfileFirstBindReviewModal,
)


async def _wait(pilot, predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await pilot.pause(0.02)
    await pilot.pause()


async def _press(host, pilot, selector):
    control = host.screen.query_one(selector)
    control.focus()
    await pilot.pause()
    await pilot.wait_for_scheduled_animations()
    if isinstance(control, Button):
        await _wait(pilot, lambda: not control.has_class("-active"))
    _assert_painted(host.screen, control)
    await pilot.press("enter")
    await pilot.pause()


async def _choose(host, pilot, kind, value):
    picker = host.screen.query_one(f"#settings-workspace-{kind}-picker", OptionList)
    index = next(
        index
        for index in range(picker.option_count)
        if getattr(picker.get_option_at_index(index), f"{kind}_id") == value
    )
    picker.focus()
    await pilot.pause()
    await pilot.press("home", *(["down"] * index), "enter")
    await pilot.pause()
    await _wait(pilot, lambda: not host.screen._category_pane_swap_pending)


@pytest.mark.parametrize(
    "navigation", ["category_return", "workspace_return", "modal_return", "new_draft"]
)
@pytest.mark.parametrize("phase", ["review", "confirmation"])
@private_profile_test
async def test_delayed_first_bind_review_does_not_reopen_an_old_intent(
    request, navigation, phase
):
    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-review", name="Review workspace")
    registry.create_workspace(workspace_id="ws-other", name="Other workspace")
    guard = _FirstBindGuard()
    assert app._tool_pack_guard_bootstrap.activate(guard)
    started, release, returned = threading.Event(), threading.Event(), threading.Event()

    class DelayedReview(_FirstBindService):
        def delayed(self, value):
            started.set()
            assert release.wait(20)
            returned.set()
            return value

        def review_first_bind(self, *args, **kwargs):
            candidate = super().review_first_bind(*args, **kwargs)
            return self.delayed(candidate) if phase == "review" else candidate

        def confirm_first_bind(self, candidate):
            token = super().confirm_first_bind(candidate)
            return self.delayed(token) if phase == "confirmation" else token

    service = app.tool_pack_service = DelayedReview()
    _stub_assistant_services(
        app,
        personas=[_persona("helper", "Helper"), _persona("other", "Other helper")],
        profiles=["default", "research"],
    )
    host = _StyledDestinationHarness(app, "settings")
    try:
        async with host.run_test(size=(80, 24)) as pilot:
            await _category(host, pilot, "Workspaces")
            await _press(host, pilot, "#settings-workspace-row-ws-review")
            await _choose(host, pilot, "persona", "helper")
            await _choose(host, pilot, "profile", "research")
            screen = host.screen
            await _press(host, pilot, "#settings-workspace-memory-toggle")
            if phase == "confirmation":
                await _wait(
                    pilot,
                    lambda: isinstance(host.screen, ToolProfileFirstBindReviewModal),
                )
                await _press(host, pilot, "#tool-profile-bind-confirm")
                await _wait(pilot, lambda: host.screen is screen)
            await _wait(pilot, started.is_set)
            if navigation == "category_return":
                for name in ("Overview", "Workspaces"):
                    await pilot.press("escape", "/", *name, "enter")
                    await pilot.pause()
                    await _wait(pilot, lambda: not screen._category_pane_swap_pending)
                await _press(host, pilot, "#settings-workspace-row-ws-review")
            elif navigation == "workspace_return":
                await _press(host, pilot, "#settings-workspace-row-ws-other")
                await _press(host, pilot, "#settings-workspace-row-ws-review")
            elif navigation == "modal_return":
                await host.push_screen(ModalScreen())
                await pilot.pause()
                await host.pop_screen()
                await pilot.pause()
            else:
                await _choose(host, pilot, "persona", "other")
            release.set()
            await _wait(pilot, returned.is_set)
            await pilot.pause(0.15)
            assert not isinstance(host.screen, ToolProfileFirstBindReviewModal)
            assert registry.get_workspace("ws-review").assistant_defaults is None
            assert len(service.confirmed) == (1 if phase == "confirmation" else 0)
            assert not guard.accepted
            if navigation == "new_draft":
                assert (
                    screen._settings_workspace_assistant_pending["persona_id"]
                    == "other"
                )
    finally:
        release.set()


@private_profile_test
async def test_failed_clear_preserves_memory_confirmation_and_label(
    request, monkeypatch
):
    from Tests.UI.test_settings_workspace_memory_confirmation import (
        _arm,
        _fixture,
    )
    from Tests.UI.test_settings_workspace_memory_confirmation import (
        _press as press_and_settle,
    )
    from tldw_chatbook.Workspaces.registry_service import WorkspaceRegistryServiceError

    app, registry, defaults = _fixture()
    host = _StyledDestinationHarness(app, "settings")
    async with host.run_test(size=(80, 24)) as pilot:
        await _category(host, pilot, "Workspaces")
        await press_and_settle(host, pilot, "#settings-workspace-row-ws-review")
        await _arm(host, pilot, registry, defaults)
        confirmation = host.screen._settings_workspace_memory_armed

        def refuse(_workspace_id):
            raise WorkspaceRegistryServiceError("Clear unavailable; retry.")

        monkeypatch.setattr(registry, "clear_assistant_defaults", refuse)
        await press_and_settle(host, pilot, "#settings-workspace-assistant-clear")
        assert registry.get_workspace("ws-review").assistant_defaults == defaults
        assert host.screen._settings_workspace_memory_armed is confirmation
        assert str(
            host.screen.query_one("#settings-workspace-memory-toggle").label
        ) == ("Confirm read_write?")


@private_profile_test
async def test_workspace_assistant_name_is_painted_literally(request):
    from Tests.UI.test_settings_overview_search_journeys import _painted
    from Tests.UI.test_settings_workspace_memory_confirmation import _fixture

    app, _registry, _defaults = _fixture()
    name = "Review [helper]"
    app.local_character_persona_service._personas[0]["name"] = name
    host = _StyledDestinationHarness(app, "settings")
    async with host.run_test(size=(80, 24)) as pilot:
        await _category(host, pilot, "Workspaces")
        await _press(host, pilot, "#settings-workspace-row-ws-review")
        status = host.screen.query_one("#settings-workspace-assistant-status")
        status.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        assert name in " ".join(_painted(host, status).split())
        picker = host.screen.query_one("#settings-workspace-persona-picker")
        picker.focus()
        await pilot.pause()
        assert name in " ".join(_painted(host, picker).split())
