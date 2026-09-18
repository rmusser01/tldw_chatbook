"""Workspace folder actions expose their outcome at the keyboard target."""

import pytest
from textual.widgets import Button, Input, Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_rag_result_focus import _assert_painted
from Tests.UI.test_settings_overview_search_journeys import _category, _painted
from Tests.UI.test_settings_provider_keyboard_journeys import _settle
from Tests.UI.test_settings_speech_tts_panel import _StyledDestinationHarness
from tldw_chatbook.Workspaces.registry_service import WorkspaceRegistryServiceError


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(80, 24), (170, 48)])
@pytest.mark.timeout(180)
@private_profile_test
async def test_folder_actions_keep_feedback_and_recovery_visible(
    request, tmp_path, monkeypatch, theme, size
):
    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-folders", name="Research project")
    registry.create_workspace(workspace_id="ws-other", name="Other project")
    folder = tmp_path / "project[notes]"
    folder.mkdir()
    host = _StyledDestinationHarness(app, "settings")
    host.theme = theme

    async def focus(selector):
        widget = host.screen.query_one(selector)
        widget.focus()
        # focus() is queued on the app; let it admit scrolling before waiting.
        await pilot.pause()
        await _settle(host, pilot)
        _assert_painted(host.screen, widget)
        return widget

    async def press(selector):
        control = await focus(selector)
        if isinstance(control, Button):
            assert str(control.label) in _painted(host, control)
        await pilot.press("enter")
        await _settle(host, pilot)

    async def path(value):
        field = await focus("#settings-workspace-folder-path")
        await pilot.press("home", "shift+end", "backspace", *value)
        await _settle(host, pilot)
        assert field.value == value
        await pilot.press("tab")
        await _settle(host, pilot)
        assert host.focused.id == "settings-workspace-folder-add"
        await pilot.press("enter")
        await _settle(host, pilot)

    async def feedback(needle):
        for _ in range(100):
            await pilot.pause(0.03)
            if not host.screen._category_pane_swap_pending and any(
                needle in str(item.renderable) for item in host.screen.query(Static)
            ):
                break
        await _settle(host, pilot)
        results = [
            item for item in host.screen.query(Static) if needle in str(item.renderable)
        ]
        assert len(results) == 1
        result = results[0]
        _assert_painted(host.screen, result)
        assert " ".join(str(result.renderable).split()) in " ".join(
            _painted(host, result).split()
        )
        _assert_painted(host.screen, host.focused)

    async with host.run_test(size=size) as pilot:
        await _category(host, pilot, "Workspaces")
        await press("#settings-workspace-row-ws-folders")
        assert not host.screen.query("#settings-save-category")
        await path("/missing-workspace-folder")
        await feedback("not a directory")
        assert (
            host.screen.query_one("#settings-workspace-folder-path", Input).value
            == "/missing-workspace-folder"
        )
        assert registry.list_folder_bindings("ws-folders") == ()

        original_add = registry.add_folder_binding
        with monkeypatch.context() as patch:

            def refuse(*args, **kwargs):
                raise WorkspaceRegistryServiceError(
                    "Folder add refused. Retry Add folder."
                )

            patch.setattr(registry, "add_folder_binding", refuse)
            await path(str(folder))
            await feedback("Folder add refused")
            assert host.screen.query_one(
                "#settings-workspace-folder-path", Input
            ).value == str(folder)
            assert registry.list_folder_bindings("ws-folders") == ()
        assert registry.add_folder_binding == original_add
        await press("#settings-workspace-folder-add")
        assert host.focused.id == "settings-workspace-folder-add"
        await feedback("Folder added (read-only).")
        (binding,) = registry.list_folder_bindings("ws-folders")
        assert binding.locator == str(folder.resolve())
        assert binding.metadata["access"] == "ro"
        assert host.focused.id == "settings-workspace-folder-add"
        toggle = f"#settings-workspace-folder-toggle-{binding.binding_id}"

        async def binding_state(access):
            await focus(toggle)
            row = host.screen.query_one(
                f"#settings-workspace-folder-{binding.binding_id}"
            )
            # A deeply nested fixture can scroll above the viewport while its
            # basename and access state still paint beside the focused toggle.
            text = "".join(_painted(host, row).split())
            assert f"[{access}]" in text
            assert "project[notes]" in text

        await binding_state("ro")

        with monkeypatch.context() as patch:

            def refuse(*args, **kwargs):
                raise WorkspaceRegistryServiceError("Access change refused. Retry.")

            patch.setattr(registry, "set_folder_binding_access", refuse)
            await press(toggle)
            await feedback("Access change refused")
            assert (
                registry.list_folder_bindings("ws-folders")[0].metadata["access"]
                == "ro"
            )
        await press(toggle)
        await feedback("Folder access: read-write.")
        assert registry.list_folder_bindings("ws-folders")[0].metadata["access"] == "rw"
        assert host.focused.id == toggle[1:]
        await binding_state("rw")
        await press(toggle)
        await feedback("Folder access: read-only.")
        assert registry.list_folder_bindings("ws-folders")[0].metadata["access"] == "ro"

        await press("#settings-workspace-row-ws-other")
        assert "Folder access:" not in " ".join(
            str(item.renderable) for item in host.screen.query(Static)
        )
        await press("#settings-workspace-row-ws-folders")
        await binding_state("ro")
        remove = f"#settings-workspace-folder-remove-{binding.binding_id}"
        with monkeypatch.context() as patch:

            def refuse(*args, **kwargs):
                raise WorkspaceRegistryServiceError("Folder removal refused. Retry.")

            patch.setattr(registry, "remove_runtime_binding", refuse)
            await press(remove)
            await feedback("Folder removal refused")
            assert len(registry.list_folder_bindings("ws-folders")) == 1
        # Native removal exposed focus queued for the soon-to-be-removed Add.
        # Release that old control's focus request only after pane replacement.
        old_add = host.screen.query_one("#settings-workspace-folder-add", Button)
        delayed_focus = []
        with monkeypatch.context() as patch:

            def defer_old_focus(*args, **kwargs):
                delayed_focus.append(lambda: Button.focus(old_add, *args, **kwargs))
                return old_add

            patch.setattr(old_add, "focus", defer_old_focus)
            await press(remove)
            for callback in delayed_focus:
                callback()
            await pilot.pause()
            await _settle(host, pilot)
        await feedback("Folder removed.")
        assert registry.list_folder_bindings("ws-folders") == ()
        assert host.focused is host.screen.query_one("#settings-workspace-folder-add")
        assert host.focused.is_attached
        await pilot.press("tab")
        await _settle(host, pilot)
        assert host.focused.is_attached
        assert host.focused.id != "settings-workspace-folder-add"
        _assert_painted(host.screen, host.focused)
        await pilot.press("shift+tab")
        await _settle(host, pilot)
        assert host.focused is host.screen.query_one("#settings-workspace-folder-add")
        assert folder.is_dir()
        assert registry.get_active_workspace().workspace_id == "workspace-default"
        await pilot.resize_terminal(*((80, 24) if size == (170, 48) else (170, 48)))
        await _settle(host, pilot)
        assert host.focused.id == "settings-workspace-folder-add"
        _assert_painted(host.screen, host.focused)
