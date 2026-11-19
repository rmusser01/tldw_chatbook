"""Console exposes folder keyword status and a supported Settings route."""

import pytest

from Tests.private_profile import private_profile_test


@private_profile_test
async def test_index_settings_action_exists(request):
    from tldw_chatbook.Widgets.Console.console_workspace_files_modal import (
        ConsoleWorkspaceFilesModal,
    )

    assert hasattr(
        ConsoleWorkspaceFilesModal, "_request_index_status"
    ), "Console needs captured binding keyword status"


@pytest.mark.parametrize(
    "theme,size", [("textual-dark", (80, 24)), ("textual-light", (170, 48))]
)
@private_profile_test
async def test_console_status_binding_fence_and_safe_settings_route(
    request, theme, size
):
    import asyncio
    from dataclasses import replace
    from threading import Event
    from textual.widgets import Button, Static
    from Tests.UI.test_console_workspace_files_modal import _Host, _Inspector, _scope
    from tldw_chatbook.Widgets.Console.console_workspace_files_modal import (
        ConsoleWorkspaceFilesModal,
        WorkspaceFilesBinding,
    )

    started, release = Event(), Event()
    calls, routed = [], []

    def read(binding):
        calls.append(binding.binding_id)
        if binding.binding_id == "binding-a":
            started.set()
            assert release.wait(10)
            return "Keyword index: Ready A"
        return "Keyword index: Paused B"

    first = WorkspaceFilesBinding("binding-a", "A", _scope())
    second = WorkspaceFilesBinding(
        "binding-b", "B", replace(_scope(), binding_id="binding-b")
    )
    host = _Host()
    host.theme = theme
    modal = ConsoleWorkspaceFilesModal(
        inspector=_Inspector([]),
        inspected_workspace_id="ws-a",
        inspected_workspace_name="A",
        active_workspace_id="ws-a",
        active_workspace_name="A",
        bindings=(first, second),
        index_status_reader=read,
        on_index_settings=lambda: routed.append(modal not in host.screen_stack),
    )
    try:
        async with host.run_test(size=size) as pilot:
            await host.push_screen(modal)
            assert await asyncio.to_thread(started.wait, 5)
            modal.query_one("#console-workspace-files-binding-1", Button).press()
            await pilot.pause()
            release.set()
            for _ in range(100):
                await pilot.pause(0.02)
                if "Paused B" in str(
                    modal.query_one(
                        "#console-workspace-files-index-status", Static
                    ).render()
                ):
                    break
            assert "Paused B" in str(
                modal.query_one(
                    "#console-workspace-files-index-status", Static
                ).render()
            )
            button = modal.query_one("#console-workspace-files-index-settings", Button)
            button.focus()
            await pilot.pause()
            assert host.focused is button
            assert button.region.intersection(modal.region) == button.region
            from pathlib import Path

            qa = (
                Path(__file__).resolve().parents[2]
                / ".superpowers/sdd/2026-10-20-workspace-project-keyword-index"
            )
            host.save_screenshot(
                filename=f"k5-console-{theme}-{size[0]}.svg", path=str(qa)
            )
            await pilot.press("enter")
            for _ in range(50):
                await pilot.pause(0.02)
                if routed:
                    break
            assert routed == [True]
            assert calls == ["binding-a", "binding-b"]
    finally:
        release.set()
