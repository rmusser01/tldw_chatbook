"""Reveal the Test Tool within the existing Settings scroll owners."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.containers import VerticalScroll
from textual.css.query import NoMatches
from textual.errors import NoWidget
from textual.geometry import Region, clamp
from textual.widget import Widget
from textual.widgets import Button, Static

if TYPE_CHECKING:
    from .mcp_inspector import MCPInspector

_TEST_TOOL = "mcp-inspector-test-tool"
_RUN = "mcp-inspector-test-run"


def _top(inspector: MCPInspector) -> bool:
    return (
        inspector.is_attached
        and inspector.display
        and inspector.app.screen is inspector.screen
    )


def _tools_body(inspector: MCPInspector) -> VerticalScroll | None:
    if not _top(inspector):
        return None
    screen = inspector.screen
    parents = inspector.ancestors[:4]
    if (
        getattr(screen, "screen_name", None) != "settings"
        or getattr(screen, "active_category", None) != "tools"
        or tuple(p.id for p in parents) != (
            "mcp-hub-grid",
            "mcp-hub-workbench",
            "settings-tools-hub",
            "settings-detail-pane-body",
        )
        or getattr(parents[1], "active_mode", None) != "tools"
        or not isinstance(body := parents[3], VerticalScroll)
    ):
        return None
    return body


def _compact_body(inspector: MCPInspector, target: Widget) -> VerticalScroll | None:
    if target.id not in {_TEST_TOOL, _RUN, "mcp-inspector-test-close"} and not any(
        p.id == "mcp-inspector-test-form" for p in target.ancestors
    ):
        return None
    if not any(
        p.id == "settings-workbench" and p.has_class("settings-workbench-compact")
        for p in inspector.ancestors
    ):
        return None
    return _tools_body(inspector)


def _displayed(target: Widget) -> bool:
    return target.is_attached and all(
        node.display and node.visible for node in target.ancestors_with_self
    )


def queue_test_focus(target: Widget) -> bool:
    """Defer only this control's compact reveal, tied to its panel lifetime."""
    from .mcp_inspector import MCPInspector

    inspector = next((p for p in target.ancestors if isinstance(p, MCPInspector)), None)
    if inspector is None or _compact_body(inspector, target) is None:
        return False
    token = inspector.test_panel_token

    def reveal() -> None:
        body = _compact_body(inspector, target)
        if (
            body is None
            or inspector.app.focused is not target
            or inspector not in target.ancestors
            or inspector.test_panel_token is not token
            or (token is None and target.id != _TEST_TOOL)
            or not target.focusable
            or not _displayed(target)
        ):
            return
        try:
            if not _offsets_match_layout(inspector, body):
                return
            placement = body.screen.find_widget(target)
            if not _reveal_extent(inspector, body, target, placement.region, placement.clip):
                target.scroll_visible(animate=False, immediate=True)
        except (NoMatches, NoWidget):
            return

    return inspector.call_after_refresh(reveal)


def reveal_other_focus(inspector: MCPInspector) -> None:
    """Keep the original Inspector resize reveal outside compact Test Tool."""
    if not _top(inspector):
        return
    focused = inspector.app.focused
    if (
        focused is not None
        and inspector in focused.ancestors
        and _compact_body(inspector, focused) is None
    ):
        focused.scroll_visible(animate=False, immediate=True)


def _offsets_match_layout(inspector: MCPInspector, body: VerticalScroll) -> bool:
    # Textual can update an owner's offset before repainting its in-flow child.
    for owner, child_id in (
        (inspector, "mcp-inspector-tool"),
        (body, "settings-tools-hub"),
    ):
        child = owner.get_child_by_id(child_id)
        placement = owner.screen.find_widget(child)
        virtual_top = owner.content_region.y + placement.virtual_region.y + child.offset.y
        if placement.region.y != virtual_top - owner.scroll_offset.y:
            return False
    return True


def _scroll_limits(owner: VerticalScroll) -> tuple[int, int]:
    if not owner.allow_vertical_scroll:
        return 0, 0
    y = owner.scroll_offset.y
    return -y, owner.max_scroll_y - y


def _reveal_extent(
    inspector: MCPInspector, body: VerticalScroll, target: Widget, extent: Region, clip: Region
) -> bool:
    inner = inspector.scrollable_content_region
    outer = body.scrollable_content_region.intersection(body.screen.find_widget(body).clip)
    below_inspector = True
    for parent in target.ancestors:
        if parent is body:
            break
        if parent is inspector:
            below_inspector = False
        elif parent.is_scrollable:
            viewport = parent.scrollable_content_region
            if below_inspector:
                if extent not in viewport:
                    return False
            else:
                inner = inner.intersection(viewport)

    effective = inner.intersection(outer)
    if (
        not extent
        or extent.x < effective.x
        or extent.right > effective.right
        or extent.intersection(effective) != extent.intersection(clip)
    ):
        return False

    inner_min, inner_max = _scroll_limits(inspector)
    outer_min, outer_max = _scroll_limits(body)
    total_min, total_max = extent.bottom - outer.bottom, extent.y - outer.y
    if total_min > total_max:
        return False
    inner_min = max(inner_min, extent.bottom - inner.bottom, total_min - outer_max)
    inner_max = min(inner_max, extent.y - inner.y, total_max - outer_min)
    if inner_min > inner_max:
        return False

    # Prefer outer movement when the target fits inside the Inspector.
    inner_delta = clamp(0, inner_min, inner_max)
    outer_delta = clamp(
        0, max(outer_min, total_min - inner_delta), min(outer_max, total_max - inner_delta)
    )
    inner_y = inspector.scroll_offset.y + inner_delta
    outer_y = body.scroll_offset.y + outer_delta
    if inner_delta:
        inspector.scroll_to(y=inner_y, animate=False, immediate=True)
    if outer_delta:
        body.scroll_to(y=outer_y, animate=False, immediate=True)
    return True


def reveal_test_approval(inspector: MCPInspector) -> bool:
    """Reveal only the current ask preview above the exact enabled focused Run."""
    body = _tools_body(inspector)
    if body is None:
        return False
    run = inspector.app.focused
    tool = inspector.current_tool
    profile = inspector._current_tool_profile_context
    preview = inspector._test_preview
    if (
        not isinstance(run, Button)
        or run.id != _RUN
        or inspector not in run.ancestors
        or not run.focusable
        or not _displayed(run)
        or inspector.test_panel_token is None
        or tool is None
        or profile is None
        or preview is None
        or preview.rendered_gate != "ask"
        or (preview.server_key, preview.tool_name) != (tool.server_key, tool.name)
        or preview.profile_id != profile.profile_id
        or preview.profile_policy_digest != profile.policy_digest
        or preview.profile_revision != profile.revision
    ):
        return False

    panel = run.parent
    if not isinstance(panel, Widget) or panel.id != "mcp-inspector-test-panel":
        return False
    try:
        status = panel.get_child_by_id("mcp-inspector-test-preview")
        if (
            not isinstance(status, Static)
            or not _displayed(status)
            or not _offsets_match_layout(inspector, body)
        ):
            return False
        status_geometry = body.screen.find_widget(status)
        run_geometry = body.screen.find_widget(run)
        clip = run_geometry.clip
        if (
            not status_geometry.region
            or not run_geometry.region
            or status_geometry.clip != clip
            or run_geometry.region not in clip
            or status_geometry.region.y >= clip.y
        ):
            return False
        extent = status_geometry.region.union(run_geometry.region)
        return _reveal_extent(inspector, body, status, extent, clip)
    except (NoMatches, NoWidget):
        return False
