"""Disposable production-styled Workspace evidence capture."""

from pathlib import Path

import pytest
from textual.widgets import Button

from Tests.UI.test_post_release_workspaces_library_depth import (
    LibraryWorkspaceHarness,
    _active_destination_screen,
    _build_test_app,
    _open_library_details,
    _seed_cross_workspace_library,
    _wait_for_library_shell_ready,
)

OUT = Path("Docs/superpowers/qa/2026-09-14-workspace-handoff")


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_capture(theme):
    app = _build_test_app()
    _seed_cross_workspace_library(app)
    host = LibraryWorkspaceHarness(app, "library")
    async with host.run_test(size=(120, 45)) as pilot:
        host.theme = theme
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)
        await _open_library_details(screen, pilot)
        for section in ("browse", "create", "study", "ingest"):
            screen._set_library_rail_section(section, False)
        handoff = screen.query_one("#library-use-in-console", Button)
        handoff.focus()
        await pilot.pause()
        await pilot.pause()
        painted = (
            "\n".join(
                s.text.rstrip() for s in host.screen._compositor.render_strips()
            ).rstrip()
            + "\n"
        )
        assert "2 items" in painted
        assert "Use in Console" in painted
        assert handoff.has_focus
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"workspace-blocked-{theme}.txt").write_text(painted)
        (OUT / f"workspace-blocked-{theme}.svg").write_text(host.export_screenshot())
