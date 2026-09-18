"""Keyboard/registry evidence for single-field workspace assistant edits."""

import asyncio
from contextlib import contextmanager

import pytest
from textual.widgets import Button, OptionList, Static

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
from tldw_chatbook.Workspaces.registry_service import WorkspaceRegistryServiceError


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(80, 24), (170, 48)])
@private_profile_test
async def test_single_field_edits_preserve_saved_defaults(
    request, monkeypatch, theme, size
):
    app = _build_test_app()
    registry = app.workspace_registry_service

    class LocalProfileGuard:
        @contextmanager
        def mutation_scope(self, **_context):
            yield

    # This journey uses ordinary local profiles; imported-profile review/token
    # behavior is covered separately by the original first-bind tests.
    assert app._tool_pack_guard_bootstrap.activate(LocalProfileGuard())
    registry.create_workspace(workspace_id="ws-review", name="Review workspace")
    registry.create_workspace(workspace_id="ws-other", name="Other workspace")
    registry.set_assistant_defaults(
        "ws-review",
        WorkspaceAssistantDefaults(
            assistant_id="helper", tool_policy_profile_id="research"
        ),
    )
    _stub_assistant_services(
        app,
        personas=[
            _persona("helper", "Research helper"),
            _persona("alternate", "Other helper"),
            *[_persona(f"extra-{i}", f"Extra helper {i}") for i in range(20)],
        ],
        profiles=["default", "research", "other", *[f"extra-{i}" for i in range(20)]],
    )
    host = _StyledDestinationHarness(app, "settings")
    host.theme = theme
    async with host.run_test(size=size) as pilot:
        await _category(host, pilot, "Workspaces")

        def saved():
            return registry.get_workspace("ws-review").assistant_defaults

        async def focus(selector):
            control = host.screen.query_one(selector)
            control.focus()
            await pilot.pause()
            await _settle(host, pilot)
            if isinstance(control, Button):
                async with asyncio.timeout(2):
                    while control.has_class("-active"):
                        await pilot.pause(0.02)
            _assert_painted(host.screen, control)
            assert host.focused is control and control.is_attached
            return control

        async def press(selector):
            await focus(selector)
            await pilot.press("enter")
            await _settle(host, pilot)
            assert host.focused.is_attached

        async def choose(kind, index):
            before = saved()
            await focus(f"#settings-workspace-{kind}-picker")
            await pilot.press("home", *(["down"] * index), "enter")
            await _settle(host, pilot)
            assert saved() == before  # Selection never commits on its own.
            assert host.focused.is_attached
            _assert_painted(host.screen, host.focused)
            option = host.focused.get_option_at_index(index)
            assert str(option.prompt) in _painted(host, host.focused)
            visible_receipt()

        def visible_receipt():
            receipt = host.screen.query_one(
                "#settings-workspace-assistant-result", Static
            )
            _assert_painted(host.screen, receipt)
            assert " ".join(str(receipt.renderable).split()) in " ".join(
                _painted(host, receipt).split()
            )
            _assert_painted(host.screen, host.focused)

        async def apply(memory="read_only"):
            button = await focus("#settings-workspace-memory-toggle")
            assert str(button.label) == f"Apply (memory: {memory})"
            assert str(button.label) in _painted(host, button)
            before = saved()
            await pilot.press("enter")
            await _settle(host, pilot)
            if memory == "read_write":
                assert saved() == before
                assert str(button.label) == "Confirm read_write?"
                visible_receipt()
                await press("#settings-workspace-memory-toggle")
            assert host.focused.is_attached
            visible_receipt()

        def assert_default(persona, profile, memory="read_only"):
            assert saved() == WorkspaceAssistantDefaults(
                assistant_id=persona,
                tool_policy_profile_id=profile,
                persona_memory_mode=memory,
            )

        await press("#settings-workspace-row-ws-review")
        assert not host.screen.query("#settings-save-category")
        await choose("persona", 1)
        profile_picker = host.screen.query_one(
            "#settings-workspace-profile-picker", OptionList
        )
        assert (
            profile_picker.get_option_at_index(profile_picker.highlighted).profile_id
            == "research"
        )
        await apply()
        assert_default("alternate", "research")

        await choose("profile", 2)
        result = host.screen.query_one("#settings-workspace-assistant-result", Static)
        assert "select a persona" not in str(result.renderable)
        await apply()
        assert_default("alternate", "other")

        await press("#settings-workspace-memory-toggle")
        assert_default("alternate", "other")
        assert (
            str(
                host.screen.query_one("#settings-workspace-memory-toggle", Button).label
            )
            == "Confirm read_write?"
        )
        await press("#settings-workspace-memory-toggle")
        assert_default("alternate", "other", "read_write")

        await choose("profile", 1)
        await apply("read_write")
        assert_default("alternate", "research", "read_write")

        # Re-selecting the saved persona retains its read-write mode; changing
        # persona resets to read-only while keeping the selected profile.
        await choose("persona", 1)
        await apply("read_write")
        assert_default("alternate", "research", "read_write")
        await choose("persona", 0)
        real_set = registry.set_assistant_defaults
        with monkeypatch.context() as patch:

            def refused(*args, **kwargs):
                raise WorkspaceRegistryServiceError("Assistant save refused; retry.")

            patch.setattr(registry, "set_assistant_defaults", refused)
            await apply()
        assert_default("alternate", "research", "read_write")
        assert registry.set_assistant_defaults == real_set
        assert "retry" in str(
            host.screen.query_one(
                "#settings-workspace-assistant-result", Static
            ).renderable
        )
        await apply()
        assert_default("helper", "research")

        # Switching cards drops this unapplied profile selection.
        await choose("profile", 0)
        await press("#settings-workspace-row-ws-other")
        await press("#settings-workspace-row-ws-review")
        assert host.screen._settings_workspace_assistant_pending is None
        assert host.screen._settings_workspace_memory_armed is None
        assert_default("helper", "research")
        await choose("profile", 0)
        await apply()
        assert_default("helper", "default")

        await press("#settings-workspace-assistant-clear")
        assert saved() is None
        visible_receipt()
        await choose("profile", 2)
        await press("#settings-workspace-memory-toggle")
        assert saved() is None
        await choose("persona", 1)
        await apply()
        assert_default("alternate", "other")
        # A guarded reveal must do the actual scroll now. Older receipts and
        # a move to navigation must not schedule a later viewport change.
        screen = host.screen
        receipt = screen.query_one("#settings-workspace-assistant-result", Static)
        current = screen._settings_workspace_assistant_result
        calls = []
        real_scroll = receipt.scroll_visible
        with monkeypatch.context() as patch:

            def record_scroll(**kwargs):
                calls.append(kwargs)
                real_scroll(**kwargs)

            patch.setattr(receipt, "scroll_visible", record_scroll)
            screen._reveal_workspace_assistant_result(current)
            assert calls == [{"animate": False, "immediate": True}]
            screen._reveal_workspace_assistant_result((current[0], current[1], "stale"))
            assert len(calls) == 1
            await focus("#settings-category-search")
            screen._reveal_workspace_assistant_result(current)
            await _settle(host, pilot)
            assert len(calls) == 1
            _assert_painted(screen, host.focused)
        await focus("#settings-workspace-memory-toggle")
        await pilot.press("tab")
        await _settle(host, pilot)
        assert host.focused.is_attached
        _assert_painted(host.screen, host.focused)
