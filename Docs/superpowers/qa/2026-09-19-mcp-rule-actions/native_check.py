"""Native exact-input removal and Re-allow with the real service and saved catalog."""

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


def main():
    root = Path(sys.argv[1]).resolve()
    tmux_socket, session = sys.argv[2:4]
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    runpy.run_path(str(here.parent / "2026-09-16-ingest-lifecycle/native_check.py"))[
        "validate_profile"
    ](root)
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
    sys.path.insert(0, str(repo))
    from loguru import logger
    from textual.widgets import Button, DataTable

    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.MCP.local_store import LocalExternalMCPProfile
    from tldw_chatbook.MCP.permission_store import definition_hash
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
        "fixture_scope": "Disposable disconnected discovery fixture, real local store, permission store, service, inspector and workbench. Both actions persist through the normal keyboard handlers. No server connects or tool executes.",
    }

    def record():
        (evidence / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    async def tmux(*args):
        return await asyncio.to_thread(
            subprocess.run,
            ["/opt/homebrew/bin/tmux", "-L", tmux_socket, *args],
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
                                "properties": {"query": {"type": "string"}},
                            },
                        }
                    ]
                },
            )
            await workbench.reload()
            tool = workbench._tool_for("local:review-docs", "search")
            assert tool is not None
            current_hash = definition_hash(tool.description, tool.input_schema)
            store = service.permission_store
            store.ensure_profile("review-other")
            store.add_tool_arg_rule(
                tool.server_key,
                tool.name,
                args={"query": "other profile"},
                definition_hash=current_hash,
                profile_id="review-other",
            )
            other_rules = store.list_tool_arg_rules(
                tool.server_key, tool.name, profile_id="review-other"
            )
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
                    store.set_tool_state(
                        tool.server_key, tool.name, "allow", definition_hash="a" * 64
                    )
                    store.add_tool_arg_rule(
                        tool.server_key,
                        tool.name,
                        args={"query": "remove this"},
                        definition_hash=current_hash,
                    )
                    await focus("#mcp-mode-permissions")
                    await pilot.press("enter")
                    await settle()
                    async with workbench._sync_children_lock:
                        await workbench._sync_permissions_mode()
                    canvas = workbench.query_one(MCPPermissionsMode)
                    table = canvas.query_one("#mcp-perm-table", DataTable)
                    assert canvas.select_tool_row(tool.server_key, tool.name)
                    table.focus()
                    await pilot.press("enter")
                    await wait_for(
                        lambda: bool(workbench.query("#mcp-inspector-reallow")),
                        "review controls",
                    )
                    assert workbench._effective_for_display(tool).config_changed
                    await focus("#mcp-inspector-arg-rule-remove-0")
                    stem = f"{theme}-{width}x{height}"
                    await capture(stem + "-before-remove")
                    await pilot.press("enter")
                    await wait_for(
                        lambda: not workbench.query("#mcp-inspector-arg-rule-remove-0"),
                        "rule removed",
                    )
                    assert store.list_tool_arg_rules(tool.server_key, tool.name) == []
                    assert (
                        store.list_tool_arg_rules(
                            tool.server_key, tool.name, profile_id="review-other"
                        )
                        == other_rules
                    )
                    assert workbench._effective_for_display(tool).config_changed
                    await focus("#mcp-inspector-reallow")
                    await capture(stem + "-before-reallow")
                    await pilot.press("enter")
                    await wait_for(
                        lambda: not workbench.query("#mcp-inspector-reallow"),
                        "definition reallowed",
                    )
                    entry = store.get_tool_entry(tool.server_key, tool.name)
                    assert (
                        entry["state"] == "allow"
                        and entry["definition_hash"] == current_hash
                    )
                    assert workbench._effective_for_display(tool).state == "allow"
                    assert not workbench._effective_for_display(tool).config_changed
                    assert (
                        store.list_tool_arg_rules(
                            tool.server_key, tool.name, profile_id="review-other"
                        )
                        == other_rules
                    )
                    await focus("#mcp-perm-table")
                    await capture(stem + "-after-reallow")
                    assert not attempts
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": [width, height],
                            "real_service_remove": True,
                            "real_service_reallow": True,
                            "other_profile_preserved": True,
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
