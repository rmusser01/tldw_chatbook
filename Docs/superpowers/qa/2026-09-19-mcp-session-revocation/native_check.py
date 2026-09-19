"""Native session-grant review and keyboard revocation with the real service."""

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
    args_tmux_path = args.tmux_path
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

    from tldw_chatbook.Agents.builtin_tool_gate import BuiltinToolGate
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Tools.tool_executor import CalculatorTool
    from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
    from tldw_chatbook.UI.MCP_Modules.mcp_permissions_mode import MCPPermissionsMode
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
        "fixture_scope": "Disposable in-memory grants use the real permission store, service, inspector, workbench and builtin gate. The gate is checked before and after revocation; no tool executes.",
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
            service.permission_store.ensure_profile("review-other")
            service.set_tool_state("agent:builtin", "calculator", "ask")
            service.approve_for_session(
                "agent:builtin", "calculator", profile_id="review-other"
            )
            gate = BuiltinToolGate(service)
            tool = CalculatorTool()
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
                    service.approve_for_session("agent:builtin", "calculator")
                    service.approve_for_session("agent:builtin", "get_current_datetime")
                    await focus("#mcp-mode-permissions")
                    await pilot.press("enter")
                    await settle()
                    async with workbench._sync_children_lock:
                        await workbench._sync_permissions_mode()
                    canvas = workbench.query_one(MCPPermissionsMode)
                    table = canvas.query_one("#mcp-perm-table", DataTable)
                    assert canvas.select_tool_row("agent:builtin", "calculator")
                    table.focus()
                    await pilot.press("enter")
                    await wait_for(
                        lambda: bool(
                            workbench.query("#mcp-inspector-session-approval-revoke-0")
                        ),
                        "grant list",
                    )
                    assert gate.check(tool, "review-before") is None
                    inspector = workbench.query_one(MCPInspector)
                    assert inspector._current_permission_session_approvals == [
                        ("agent:builtin", "calculator"),
                        ("agent:builtin", "get_current_datetime"),
                    ]
                    await focus("#mcp-inspector-session-approval-revoke-0")
                    stem = f"{theme}-{width}x{height}"
                    await capture(stem + "-before-revoke")
                    await pilot.press("enter")
                    await wait_for(
                        lambda inspector=inspector: (
                            inspector._current_permission_session_approvals
                            == [("agent:builtin", "get_current_datetime")]
                        ),
                        "grant revoked",
                    )
                    assert not service.is_session_approved(
                        "agent:builtin", "calculator"
                    )
                    assert service.is_session_approved(
                        "agent:builtin", "get_current_datetime"
                    )
                    assert service.is_session_approved(
                        "agent:builtin", "calculator", profile_id="review-other"
                    )
                    assert gate.check(tool, "review-after") is not None
                    assert (
                        service.permission_store.get_tool_entry(
                            "agent:builtin", "calculator"
                        )["state"]
                        == "ask"
                    )
                    rows = {
                        row.tool_name: row.state_label
                        for row in canvas._rows_by_key.values()
                        if row.server_key == "agent:builtin" and row.tool_name
                    }
                    assert "(session)" not in rows["calculator"]
                    assert "(session)" in rows["get_current_datetime"]
                    await focus("#mcp-inspector-session-approval-revoke-0")
                    await capture(stem + "-after-revoke")
                    assert not attempts
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": [width, height],
                            "real_service_revoke": True,
                            "runtime_gate_asks_again": True,
                            "other_tool_and_profile_preserved": True,
                            "tool_dispatches": 0,
                        }
                    )
                    record()
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
