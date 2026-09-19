"""Native permission navigation from Tools and Test Tool with the real workbench."""

import asyncio
import hashlib
import json
import os
import runpy
import socket
import subprocess
import sys
import traceback
from pathlib import Path


def main() -> None:
    """Run ROOT TMUX_SOCKET SESSION in an existing native tmux session.

    ROOT is an unused, prepared private profile under /tmp. Arguments, profile
    paths and tmux availability are checked before app startup or output writes.
    The run writes native.log, launch.json and evidence/ under ROOT. It exits 0
    after a successful journey, 1 for a journey failure and 2 for invalid input.
    """
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    sys.path.insert(0, str(repo))
    args = runpy.run_path(str(here.parent / "native_runner_args.py"))[
        "parse_native_args"
    ]()
    root, tmux_socket, session = args.root, args.tmux_socket, args.session
    tmux_path = args.tmux_path
    os.environ.update(
        HOME=str(root / "home"),
        USERPROFILE=str(root / "home"),
        TLDW_CONFIG_PATH=str(root / "config.toml"),
        XDG_DATA_HOME=str(root / "data"),
        XDG_CONFIG_HOME=str(root / "config"),
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
    )
    os.environ.pop("NO_COLOR", None)
    attempts = []
    original_connect = socket.socket.connect

    def guard_connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            attempts.append("network connect")
            raise RuntimeError("Network is disabled in this disposable UI journey")
        return original_connect(sock, address)

    socket.socket.connect = guard_connect
    from loguru import logger
    from textual.widgets import Button, DataTable

    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.MCP.local_store import LocalExternalMCPProfile
    from tldw_chatbook.MCP.permission_store import definition_hash
    from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
    from tldw_chatbook.UI.MCP_Modules.mcp_permissions_mode import MCPPermissionsMode
    from tldw_chatbook.UI.MCP_Modules.mcp_tools_mode import MCPToolsMode
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.Utils.app_shutdown import claim_process_exit
    from tldw_chatbook.Utils.terminal_utils import warm_up_image_protocol

    logger.remove()
    logger.add(root / "native.log", level="INFO")
    evidence = root / "evidence"
    evidence.mkdir(exist_ok=False)
    (root / "launch.json").write_text(json.dumps({"pid": os.getpid()}))
    claim_process_exit()
    warm_up_image_protocol()
    app = TldwCli()
    sources = [
        "tldw_chatbook/UI/MCP_Modules/mcp_inspector.py",
        "tldw_chatbook/UI/MCP_Modules/mcp_workbench.py",
        "tldw_chatbook/UI/MCP_Modules/mcp_permissions_mode.py",
        "tldw_chatbook/MCP/unified_control_plane_service.py",
        "tldw_chatbook/Agents/builtin_tool_gate.py",
        "tldw_chatbook/css/tldw_cli_modular.tcss",
        "tldw_chatbook/css/widget_defaults_scoped.tcss",
        "tldw_chatbook/css/widget_defaults_self.tcss",
        "tldw_chatbook/css/core/_variables.tcss",
    ]
    result = {
        "pid": os.getpid(),
        "cells": [],
        "source_hashes": {
            p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in sources
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture_scope": "Disposable disconnected discovery fixture, real local store, permission store, service, inspector and workbench. Both navigation actions use the normal keyboard handlers. No server connects or tool executes.",
    }

    def record():
        (evidence / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    async def tmux(*args):
        return await asyncio.to_thread(
            subprocess.run,
            [tmux_path, "-L", tmux_socket, *args],
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
                while not predicate():
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
                lambda: not workbench.is_loading and not workbench._reloading,
                "MCP loaded",
            )
            service = app.unified_mcp_service
            local_store = service.local_service.store
            local_store.save_profile(
                LocalExternalMCPProfile("review-docs", "/usr/bin/false")
            )
            local_store.save_discovery_snapshot(
                "review-docs",
                {
                    "tools": [
                        {
                            "name": "search",
                            "description": "Search the disposable review catalog.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {},
                            },
                        }
                    ]
                },
            )
            await workbench.reload()
            tool = workbench._tool_for("local:review-docs", "search")
            assert tool is not None
            store = service.permission_store
            store.set_tool_state(
                tool.server_key,
                tool.name,
                "deny",
                definition_hash=definition_hash(tool.description, tool.input_schema),
            )
            permission_before = store.get_tool_entry(tool.server_key, tool.name)
            await workbench.reload()
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
                    stem = f"{theme}-{width}x{height}"
                    for entry in ("details", "test"):
                        await focus("#mcp-mode-tools")
                        await pilot.press("enter")
                        await settle()
                        canvas = workbench.query_one(MCPToolsMode)
                        assert await canvas.select_tool_row(tool.tool_id)
                        table = canvas.query_one("#mcp-tools-table", DataTable)
                        table.focus()
                        await settle()
                        assert table.has_focus
                        await pilot.press("enter")
                        inspector = workbench.query_one(MCPInspector)
                        await wait_for(
                            lambda inspector=inspector: (
                                inspector.current_permission_tool is not None
                                and inspector.current_permission_tool.tool_id
                                == tool.tool_id
                            ),
                            "tool detail",
                        )
                        if entry == "test":
                            await focus("#mcp-inspector-test-tool")
                            await pilot.press("enter")
                            await wait_for(
                                lambda: (
                                    bool(
                                        workbench.query(
                                            "#mcp-inspector-goto-permission-test"
                                        )
                                    )
                                    and workbench.query_one(
                                        "#mcp-inspector-goto-permission-test"
                                    ).display
                                ),
                                "permission preview",
                            )
                            selector = "#mcp-inspector-goto-permission-test"
                        else:
                            selector = "#mcp-inspector-goto-permission"
                        await focus(selector)
                        await capture(stem + "-" + entry)
                        await pilot.press("enter")
                        await wait_for(
                            lambda inspector=inspector: (
                                workbench.active_mode == "permissions"
                                and inspector.current_permission_tool is not None
                                and inspector.current_permission_tool.tool_id
                                == tool.tool_id
                            ),
                            "permission destination",
                        )
                        permission_canvas = workbench.query_one(MCPPermissionsMode)
                        permission_table = permission_canvas.query_one(
                            "#mcp-perm-table", DataTable
                        )
                        row_key = permission_table.coordinate_to_cell_key(
                            permission_table.cursor_coordinate
                        ).row_key.value
                        assert row_key == tool.tool_id, (row_key, tool.tool_id)
                        assert (
                            store.get_tool_entry(tool.server_key, tool.name)
                            == permission_before
                        )
                        if entry == "test":
                            await focus("#mcp-perm-table")
                            await capture(stem + "-destination")
                        result["cells"].append(
                            {
                                "theme": theme,
                                "size": [width, height],
                                "entry": entry,
                                "destination_tool": tool.tool_id,
                                "permission_unchanged": True,
                            }
                        )
                        record()
                    assert not attempts
            result["passed"] = True
        except Exception:  # noqa: BLE001 - preserve evidence and shut down the native app
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
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
