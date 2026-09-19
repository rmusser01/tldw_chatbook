"""Native Audit journey loaded after private-profile validation and isolation."""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import traceback
from argparse import Namespace
from pathlib import Path

from loguru import logger
from textual.css.query import NoMatches, QueryError
from textual.widgets import Button, DataTable, Input

from tldw_chatbook.app import TldwCli
from tldw_chatbook.MCP.execution_log import build_record
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
from tldw_chatbook.UI.MCP_Modules.mcp_servers_mode import MCPServersMode
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.Utils.app_shutdown import claim_process_exit
from tldw_chatbook.Utils.terminal_utils import warm_up_image_protocol


def run(args: Namespace, attempts: list[str]) -> int:
    """Drive both Audit navigation actions in the prepared native terminal.

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
        for p in ("mcp_audit_mode", "mcp_workbench", "mcp_inspector")
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
        "fixture_scope": "Real private catalog and two synthetic Audit metadata records. No tool executes, no external server connects, no permission changes.",
        "runner_sha256": hashlib.sha256(
            (here / "native_check.py").read_bytes()
        ).hexdigest(),
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
            for key, name in [
                (target.server_key, target.name),
                ("local:removed", "missing_review_tool"),
            ]:
                log.append(
                    build_record(
                        server_key=key,
                        tool_name=name,
                        initiator="test",
                        ok=True,
                        duration_ms=42,
                        decision="allowed",
                    )
                )
            await workbench._sync_audit_log_entries()
            inspector = workbench.query_one(MCPInspector)
            profile_before = service.permission_store.read_snapshot_strict().payload[
                "profiles"
            ]
            notices = []
            notify = app.notify

            def observe(message, **kwargs):
                notices.append((str(message), kwargs.get("severity")))
                return notify(message, **kwargs)

            app.notify = observe

            async def mode(name):
                await focus("#mcp-mode-" + name)
                await pilot.press("enter")
                await settle()
                assert workbench.active_mode == name

            async def audit(missing=False):
                await mode("audit")
                table = await focus("#mcp-audit-table")
                assert table.row_count == 2
                await pilot.press("ctrl+home" if missing else "ctrl+end", "enter")
                await settle()
                assert inspector._current_audit_entry["tool_name"] == (
                    "missing_review_tool" if missing else target.name
                )

            for theme in ("textual-dark", "textual-light"):
                for width, height in ((120, 40), (170, 48)):
                    stem = f"{theme}-{width}x{height}"
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
                    for destination, button_id, field_id, table_id in [
                        (
                            "tools",
                            "mcp-audit-open-tool",
                            "mcp-tools-filter-text",
                            "mcp-tools-table",
                        ),
                        (
                            "permissions",
                            "mcp-audit-adjust-permission",
                            "mcp-perm-filter-text",
                            "mcp-perm-table",
                        ),
                    ]:
                        # Seed a previously entered filter in the destination.
                        field = workbench.query_one("#" + field_id, Input)
                        field.value = "no_matching_review_tool"
                        await settle()
                        await audit()
                        await focus("#" + button_id)
                        await capture(stem + "-source-" + destination)
                        await pilot.press("enter")
                        await wait_for(
                            lambda destination=destination: (
                                workbench.active_mode == destination
                                and (
                                    inspector.current_tool
                                    if destination == "tools"
                                    else inspector.current_permission_tool
                                )
                                is not None
                            ),
                            "drill destination",
                        )
                        selected = (
                            inspector.current_tool
                            if destination == "tools"
                            else inspector.current_permission_tool
                        )
                        assert (selected.server_key, selected.name) == (
                            target.server_key,
                            target.name,
                        )
                        table = workbench.query_one("#" + table_id, DataTable)
                        key, _ = table.coordinate_to_cell_key((table.cursor_row, 0))
                        assert key.value == target.tool_id
                        assert field.value == ""
                        assert not workbench.query("#mcp-audit-open-tool")
                        await capture(stem + "-" + destination)
                        await audit(missing=True)
                        await focus("#" + button_id)
                        await pilot.press("enter")
                        await settle()
                        assert workbench.active_mode == "audit"
                        assert notices[-1] == (
                            "local:removed::missing_review_tool: tool no longer available.",
                            "warning",
                        )
                        # Preserve the visible warning before notification expiry.
                        app.save_screenshot(
                            stem + "-missing-" + destination + ".svg",
                            path=str(evidence),
                        )
                        pane = await tmux("capture-pane", "-p", "-t", session)
                        (
                            evidence / (stem + "-missing-" + destination + ".txt")
                        ).write_text(pane.stdout)
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": [width, height],
                            "tool_identity": [target.server_key, target.name],
                            "both_filtered_targets_selected": True,
                            "both_missing_targets_warn_without_mode_change": True,
                        }
                    )
                    record()
            assert (
                service.permission_store.read_snapshot_strict().payload["profiles"]
                == profile_before
            )
            result["permission_profiles_unchanged"] = True
            result["record_count"] = len(log.read_recent(200))
            assert result["record_count"] == 2
            assert not attempts
            result["passed"] = True
        except Exception:  # noqa: BLE001 - preserve failure and shut down the native app
            result.update(passed=False, error=traceback.format_exc())
            app.save_screenshot("failed-state.svg", path=str(evidence))
        finally:
            result["network_attempts"] = attempts
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
