"""Native MCP inspector catalog refresh with real preview admission, no execution."""

import asyncio
import hashlib
import json
import os
import runpy
import socket
import subprocess
import sys
import traceback
from dataclasses import replace
from pathlib import Path


def main() -> None:
    """Run ROOT TMUX_SOCKET SESSION in an existing native tmux session.

    ROOT is an unused, prepared private profile under /tmp. Shared validation
    checks arguments, profile paths and tmux before app imports or output writes.
    The journey records retained argument drafts, refreshed inspector detail,
    painted controls and normal shutdown in native.log and evidence/.

    Raises:
        SystemExit: Status 0 after success, 1 after a journey or application
            failure, or 2 for invalid command-line input or profile paths.
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
    from textual.widgets import Button, DataTable, Input, Static

    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
    from tldw_chatbook.UI.MCP_Modules.mcp_servers_mode import MCPServersMode
    from tldw_chatbook.UI.MCP_Modules.mcp_tools_mode import MCPToolsMode
    from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCPWorkbench
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.Utils.app_shutdown import claim_process_exit
    from tldw_chatbook.Utils.terminal_utils import warm_up_image_protocol

    module_origins = {
        name: str(Path(sys.modules[name].__file__).resolve())
        for name in (
            "tldw_chatbook",
            "tldw_chatbook.app",
            "tldw_chatbook.UI.MCP_Modules.mcp_inspector",
            "tldw_chatbook.UI.MCP_Modules.mcp_workbench",
            "tldw_chatbook.Utils.input_validation",
            "tldw_chatbook.Utils.path_validation",
        )
    }
    assert all(Path(path).is_relative_to(repo) for path in module_origins.values())
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
        "tldw_chatbook/UI/Screens/mcp_screen.py",
        "tldw_chatbook/css/tldw_cli_modular.tcss",
        "tldw_chatbook/css/widget_defaults_scoped.tcss",
        "tldw_chatbook/css/widget_defaults_self.tcss",
        "tldw_chatbook/css/core/_variables.tcss",
        "Docs/superpowers/qa/native_runner_args.py",
    ]
    result = {
        "pid": os.getpid(),
        "cells": [],
        "module_origins": module_origins,
        "source_hashes": {
            p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in sources
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture_scope": "Controlled collector changes replace one real built-in tool definition in the UI. The real service prepares/revokes permission previews; no tool runs. This does not qualify external MCP discovery or connected runtime behavior.",
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
                lambda: getattr(app.screen, "screen_name", None) == "mcp", "MCP"
            )
            workbench = app.screen.workbench
            assert isinstance(workbench, MCPWorkbench)
            await wait_for(
                lambda: (
                    not workbench.is_loading
                    and not workbench._reloading
                    and bool(workbench.query(MCPServersMode))
                ),
                "MCP loaded",
            )
            service = app.unified_mcp_service
            profiles_before = service.permission_store.read_snapshot_strict().payload[
                "profiles"
            ]
            records_before = service.execution_log.read_recent(200)
            executions = []

            async def forbid_execution(*args, **kwargs):
                executions.append("execution attempted")
                raise AssertionError("This journey never executes tools")

            service.execute_prepared_hub_test = forbid_execution
            collect = workbench._collect_hub_tools
            target = next(t for t in collect() if t.name == "search_notes")
            phase = "original"

            def review_catalog():
                tools = collect()
                if phase == "removed":
                    return [t for t in tools if t.tool_id != target.tool_id]
                return [
                    replace(
                        t,
                        description="Search notes using the current catalog.",
                    )
                    if phase == "changed" and t.tool_id == target.tool_id
                    else replace(t)
                    for t in tools
                ]

            workbench._collect_hub_tools = review_catalog
            await focus("#mcp-mode-tools")
            await pilot.press("enter")
            await settle()
            canvas = workbench.query_one(MCPToolsMode)
            table = canvas.query_one(DataTable)
            inspector = workbench.query_one(MCPInspector)
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
                    await inspector.show_tool(None)
                    phase = "original"
                    await workbench._sync_children()
                    assert await canvas.select_tool_row(target.tool_id)
                    table.focus()
                    await settle()
                    await pilot.press("enter")
                    await wait_for(
                        lambda: (
                            inspector.current_tool is not None
                            and inspector.current_tool.tool_id == target.tool_id
                        ),
                        "selected search_notes",
                    )
                    await focus("#mcp-inspector-test-tool")
                    await pilot.press("enter")
                    await wait_for(
                        lambda: inspector._test_preview is not None,
                        "permission preview",
                    )
                    field = await focus("#mcp-schema-field-0")
                    assert isinstance(field, Input)
                    await pilot.press(*"review note")
                    await settle()
                    nonce = inspector._test_preview.nonce
                    cursor = field.cursor_position
                    await workbench._sync_children()
                    await settle()
                    assert inspector.query_one("#mcp-schema-field-0") is field
                    assert field.value == "review note" and field.has_focus
                    assert field.cursor_position == cursor
                    assert inspector._test_preview.nonce == nonce
                    stem = f"{theme}-{width}x{height}"
                    await capture(stem + "-draft-preserved")
                    phase = "changed"
                    await workbench._sync_children()
                    await wait_for(
                        lambda: not inspector.query("#mcp-inspector-test-panel"),
                        "obsolete form retired",
                    )
                    assert inspector._test_preview is None
                    assert (
                        inspector.current_tool.description
                        == "Search notes using the current catalog."
                    )
                    assert inspector.query_one(
                        "#mcp-inspector-test-tool", Button
                    ).has_focus
                    note = inspector.query_one(
                        "#mcp-inspector-tool-refresh-note", Static
                    )
                    assert "Reopen Test Tool" in str(note.renderable)
                    await capture(stem + "-details-refreshed")
                    phase = "removed"
                    await workbench._sync_children()
                    await settle()
                    assert inspector.current_tool is None
                    assert not inspector.query_one("#mcp-inspector-tool").display
                    assert not inspector.query("#mcp-inspector-test-panel")
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": [width, height],
                            "draft_preserved": True,
                            "cursor_and_focus_preserved": True,
                            "preview_preserved_on_unchanged_refresh": True,
                            "changed_definition_retires_preview": True,
                            "guidance_visible": True,
                            "removed_tool_clears_detail": True,
                        }
                    )
                    record()
            assert (
                service.permission_store.read_snapshot_strict().payload["profiles"]
                == profiles_before
            )
            assert service.execution_log.read_recent(200) == records_before
            assert not executions and not attempts
            result.update(
                permission_profiles_unchanged=True,
                execution_log_unchanged=True,
                tool_dispatches=0,
            )
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
