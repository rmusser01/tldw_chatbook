"""Disposable differential checks; no production changes."""

import json
import sys
from collections import Counter

import pytest
from Tests.UI._library_audit_capture import OUT, capture
from textual.widgets import Button

from Tests.UI import test_library_resize_focus_gates_t23025 as costs
from Tests.UI import test_post_release_workspaces_library_depth as workspaces
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _build_test_app,
    _seed_conversations,
    _two_conversations,
    _wait_for_library_shell,
)
from tldw_chatbook.app import TldwCli


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name",
    [
        "test_library_workspaces_mode_preserves_global_visibility_and_blocks_cross_workspace_handoff",
        "test_library_workspaces_create_local_workspace_mouse_clicks",
        "test_library_details_section_renders_grouped_headers_and_drops_policy_prose",
        "test_create_workspace_preserves_rail_scroll",
    ],
)
async def test_workspace_with_production_styles(monkeypatch, name):
    monkeypatch.setattr(workspaces.DestinationHarness, "CSS_PATH", TldwCli.CSS_PATH)
    await getattr(workspaces, name)()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name",
    [
        "test_resize_gate_skips_library_query_work_on_non_crossing_frames",
        "test_tab_focus_path_library_query_volume_is_bounded",
    ],
)
async def test_query_cost_sites(monkeypatch, name):
    calls = Counter()

    def nearest():
        frame = sys._getframe(2)
        for _ in range(40):
            if frame is None:
                return ""
            filename = frame.f_code.co_filename
            if "tldw_chatbook/" in filename and "/site-packages/" not in filename:
                if "UI/Screens/library_screen.py" in filename:
                    calls[f"{frame.f_code.co_name}:{frame.f_lineno}"] += 1
                return filename
            frame = frame.f_back
        return ""

    monkeypatch.setattr(costs, "_nearest_repo_frame", nearest)
    try:
        await getattr(costs, name)()
    finally:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"query-cost-{name}.json").write_text(json.dumps(calls, indent=2))


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_rail_fold_focus_paint(monkeypatch, theme):
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(80, 24)) as pilot:
        host.theme = theme
        screen = host.screen
        await _wait_for_library_shell(screen, pilot)
        first = screen.query_one("#library-row-browse-search", Button)
        first.focus()
        await pilot.pause()
        for _ in range(25):
            await pilot.press("tab")
            await pilot.pause()
            if (
                host.focused
                and host.focused.id == "console-rail-section-toggle-library-create"
            ):
                break
        assert host.focused.id == "console-rail-section-toggle-library-create"
        await pilot.pause(0.3)
        capture(host, f"rail-fold-focus-{theme}")
        widget = host.focused
        region = widget.region
        strips = screen._compositor.render_strips()
        painted = "\n".join(
            strips[y].crop(region.x, region.right).text
            for y in range(region.y, region.bottom)
        )
        (OUT / f"rail-fold-focus-{theme}-detail.json").write_text(
            json.dumps(
                {
                    "focus_id": widget.id,
                    "region": tuple(region),
                    "painted": painted,
                    "fold_cue": tuple(
                        screen.query_one("#library-rail-fold-cue").region
                    ),
                    "rail_scroll_y": screen.query_one("#library-rail").scroll_y,
                },
                indent=2,
            )
        )
        assert str(widget.label) in painted, repr(painted)


@pytest.mark.asyncio
async def test_workspace_projection_and_painted_receipt_agree():
    app = _build_test_app()
    workspaces._seed_cross_workspace_library(app)
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(120, 45)) as pilot:
        screen = host.screen
        await _wait_for_library_shell(screen, pilot)
        await pilot.pause(0.2)
        screen.query_one("#console-rail-section-toggle-library-details", Button).press()
        await pilot.pause()
        await pilot.pause()
        cached = screen._library_workspace_depth_state()
        fresh = screen._library_workspace_depth_state(refresh=True)
        receipt = screen.query_one("#library-workspaces-handoff")
        receipt.scroll_visible(animate=False)
        await pilot.pause()
        capture(host, "workspace-projection-mismatch")
        diagnostic = {
            "counts": screen._local_source_counts,
            "records": screen._workspace_source_records(),
            "cached_handoff": cached.handoff_label,
            "fresh_handoff": fresh.handoff_label,
            "painted_receipt": str(receipt.renderable),
            "source_row_count": len(fresh.source_rows),
        }
        (OUT / "workspace-projection-mismatch-detail.json").write_text(
            json.dumps(diagnostic, indent=2, default=str)
        )
        assert "2 eligible" in str(receipt.renderable), diagnostic
