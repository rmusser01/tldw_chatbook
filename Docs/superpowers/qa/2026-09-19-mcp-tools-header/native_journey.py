"""Native Tools header journey loaded after private-profile validation and isolation."""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import traceback
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

from loguru import logger
from textual.css.query import NoMatches, QueryError
from textual.widgets import Button, DataTable

from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
from tldw_chatbook.UI.MCP_Modules.mcp_servers_mode import MCPServersMode
from tldw_chatbook.UI.MCP_Modules.mcp_tools_mode import MCPToolsMode
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.Utils.app_shutdown import claim_process_exit
from tldw_chatbook.Utils.terminal_utils import warm_up_image_protocol


def run(args: Namespace, attempts: list[str]) -> int:
    """Drive Tools refresh, resize and selection in the prepared native terminal.

    Args:
        args: Validated private root, tmux socket/session and executable path.
        attempts: Network attempts collected by the launcher's socket guard.

    Returns:
        Zero for success, or one for a journey or application failure.
    """
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    root, tmux_socket, session = args.root, args.tmux_socket, args.session
    args_tmux_path = args.tmux_path
    logger.remove()
    logger.add(root / "native.log", level="INFO")
    evidence = root / "evidence"
    evidence.mkdir(exist_ok=False)
    (root / "launch.json").write_text(json.dumps({"pid": os.getpid()}))
    claim_process_exit()
    warm_up_image_protocol()
    app = TldwCli()
    paths = [
        "tldw_chatbook/UI/MCP_Modules/" + p + ".py"
        for p in ("mcp_tools_mode", "mcp_workbench", "mcp_inspector")
    ]
    paths += [
        "tldw_chatbook/css/" + p
        for p in (
            "tldw_cli_modular.tcss",
            "widget_defaults_scoped.tcss",
            "widget_defaults_self.tcss",
            "core/_variables.tcss",
            "components/_agentic_terminal.tcss",
        )
    ]
    paths += [
        "tldw_chatbook/MCP/execution_log.py",
        "tldw_chatbook/MCP/local_control_service.py",
        "tldw_chatbook/MCP/unified_control_plane_service.py",
        "tldw_chatbook/UI/MCP_Modules/mcp_inspector.py",
    ]
    result = {
        "pid": os.getpid(),
        "cells": [],
        "source_hashes": {
            p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in paths
        },
        "journey_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(
            (here / "native_check.py").read_bytes()
        ).hexdigest(),
        "fixture_scope": "Real private catalog restricted to three existing tools with synthetic display metadata. The Tools filter renders once before queued measurement to exercise the observed ordering. No tool execution, external server connection or permission change.",
    }

    def record():
        (evidence / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    async def tmux(*args):
        return await asyncio.to_thread(
            subprocess.run,
            [args_tmux_path, "-L", tmux_socket, *args],
            check=True,
            text=True,
            capture_output=True,
        )

    async def journey(pilot):
        async def settle():
            await pilot.pause()
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()

        async def wait_for(predicate, label):
            result["waiting_for"] = label
            record()
            async with asyncio.timeout(35):
                while True:
                    try:
                        if predicate():
                            break
                    except (NoMatches, QueryError):
                        pass
                    await pilot.pause(0.03)
            await settle()

        async def focus(selector):
            control = app.screen.query_one(selector)
            control.focus()
            await settle()
            assert control.has_focus
            region, clip = app.screen._compositor.visible_widgets[control]
            assert region.width > 0 and region.height > 0, (selector, region)
            assert region.intersection(clip) == region, (selector, region, clip)
            if isinstance(control, Button):
                hit, _ = app.screen.get_widget_at(
                    region.x + region.width // 2, region.y + region.height // 2
                )
                assert hit is control, (selector, hit)
                painted = "\n".join(
                    strip.crop(region.x, region.right).text
                    for strip in app.screen._compositor.render_strips()[
                        region.y : region.bottom
                    ]
                )
                assert control.label.plain in painted, (selector, painted)
                await wait_for(lambda: not control.has_class("-active"), "button ready")
            return control

        async def capture(stem):
            await wait_for(lambda: not app._notifications, "notices clear")
            app.save_screenshot(stem + ".svg", path=str(evidence))
            pane = await tmux("capture-pane", "-p", "-t", session)
            (evidence / (stem + ".txt")).write_text(pane.stdout)

        try:
            assert app._instance_lock_status.acquired
            assert type(app._driver).__name__ == "LinuxDriver"
            assert (
                sys.stdout.isatty()
                and sys.stderr.isatty()
                and app.console.file.isatty()
            )
            result.update(driver="LinuxDriver", tty_streams=True, lock_acquired=True)
            await wait_for(lambda: getattr(app, "_ui_ready", False), "startup")
            await app.handle_screen_navigation(NavigateToScreen("mcp"))
            await wait_for(
                lambda: getattr(app.screen, "screen_name", None) == "mcp",
                "MCP destination",
            )
            workbench = app.screen.workbench
            await wait_for(
                lambda: (
                    not workbench.is_loading
                    and not workbench._reloading
                    and bool(workbench.query(MCPServersMode))
                ),
                "MCP loaded",
            )
            service = app.unified_mcp_service
            log = service.execution_log
            assert log is not None
            target = next(
                tool
                for tool in workbench._last_hub_tools
                if tool.name == "chat_with_llm"
            )
            profile_before = service.permission_store.read_snapshot_strict().payload[
                "profiles"
            ]
            records_before = log.read_recent(200)
            collect = workbench._collect_hub_tools
            others = [
                tool.tool_id for tool in collect() if tool.tool_id != target.tool_id
            ][:2]
            chosen = {target.tool_id, *others}
            phase = 0

            def collect_review_catalog():
                return [
                    replace(
                        tool,
                        server_label="Docs 文档" if phase else "Docs",
                        tags=("read-only",) if phase else (),
                    )
                    for tool in collect()
                    if tool.tool_id in chosen
                ]

            workbench._collect_hub_tools = collect_review_catalog
            await focus("#mcp-mode-tools")
            await pilot.press("enter")
            await settle()
            canvas = workbench.query_one(MCPToolsMode)
            table = canvas.query_one("#mcp-tools-table", DataTable)
            await workbench._sync_children()
            await settle()
            assert await canvas.select_tool_row(target.tool_id)
            table.focus()
            await settle()
            apply_filter = canvas._apply_filter
            paints = 0

            def paint_before_measurement():
                nonlocal paints
                apply_filter()
                table.render_line(0)
                paints += 1

            canvas._apply_filter = paint_before_measurement

            def starts(strip):
                columns, offset = {}, 0
                for segment in strip:
                    meta = segment.style.meta if segment.style else {}
                    column = meta.get("column")
                    if column is not None and not meta.get("out_of_bounds"):
                        columns.setdefault(column, offset)
                    offset += segment.cell_length
                return columns

            for theme in ("textual-dark", "textual-light"):
                for width, height in ((120, 40), (170, 48)):
                    app.theme = theme
                    await tmux(
                        "resize-window",
                        "-t",
                        session,
                        "-x",
                        str(width),
                        "-y",
                        str(height),
                    )
                    await wait_for(
                        lambda width=width, height=height: app.size == (width, height),
                        "resize",
                    )
                    phase = 0
                    await workbench._sync_children()
                    await settle()
                    phase = 1
                    await workbench._sync_children()
                    await settle()
                    key, _ = table.coordinate_to_cell_key((table.cursor_row, 0))
                    assert key.value == target.tool_id
                    assert table.has_focus
                    assert table.row_count == 3
                    # Read the composed screen, not a new table render that
                    # could differ from the header still visible in the terminal.
                    region = table.content_region
                    strips = app.screen._compositor.render_strips()
                    header = strips[region.y].crop(region.x, region.right)
                    row = strips[region.y + table.header_height].crop(
                        region.x, region.right
                    )
                    header_starts, row_starts = starts(header), starts(row)
                    assert row_starts
                    stem = f"{theme}-{width}x{height}-tools"
                    await capture(stem)
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": [width, height],
                            "header": header.text,
                            "row": row.text,
                            "header_columns": header_starts,
                            "body_columns": row_starts,
                            "aligned": header_starts == row_starts,
                            "selected_tool": key.value,
                            "focus_retained": True,
                        }
                    )
                    record()
            await pilot.press("enter")
            await settle()
            inspector = workbench.query_one(MCPInspector)
            assert inspector.current_tool.tool_id == target.tool_id
            assert inspector.current_tool.server_label == "Docs 文档"
            result["next_enter_selects_current_tool"] = True
            result["paint_before_measurement_count"] = paints
            assert (
                service.permission_store.read_snapshot_strict().payload["profiles"]
                == profile_before
            )
            result["permission_profiles_unchanged"] = True
            assert log.read_recent(200) == records_before
            result["execution_log_unchanged"] = True
            result["network_attempts"] = list(attempts)
            assert not attempts, attempts
            result["passed"] = all(cell["aligned"] for cell in result["cells"])
        except Exception:  # noqa: BLE001 - preserve failure and shut down the native app
            result.update(passed=False, error=traceback.format_exc())
            app.save_screenshot("failed-state.svg", path=str(evidence))
        finally:
            record()
            await tmux("send-keys", "-t", session, "C-q")

    app.run(auto_pilot=journey, size=(170, 48))
    result.update(
        app_run_returned=True,
        app_return_code=app.return_code,
        app_exception=type(app._exception).__name__
        if app._exception is not None
        else None,
    )
    if app.return_code != 0 or app._exception is not None:
        result["passed"] = False
    record()
    return 0 if result.get("passed") else 1
