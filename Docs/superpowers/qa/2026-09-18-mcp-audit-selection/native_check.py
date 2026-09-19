"""Native Audit selection with real JSONL storage and two live same-name tool catalogs.

Audit records are synthetic fixtures, not claims of executed tool calls.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
"""

import asyncio
import hashlib
import json
import os
import runpy
import subprocess
import sys
import traceback
from pathlib import Path


def main():
    root = Path(sys.argv[1]).resolve()
    socket, session = sys.argv[2:4]
    here = Path(__file__).resolve()
    repo = here.parents[4]
    runpy.run_path(
        str(repo / "Docs/superpowers/qa/2026-09-16-ingest-lifecycle/native_check.py")
    )["validate_profile"](root)
    os.environ.update(
        HOME=str(root / "home"),
        USERPROFILE=str(root / "home"),
        TLDW_CONFIG_PATH=str(root / "config.toml"),
        XDG_CONFIG_HOME=str(root / "config"),
        XDG_DATA_HOME=str(root / "data"),
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
    )
    for key in (
        "NO_COLOR",
        "OPENAI_API_KEY",
        "LLAMA_CPP_API_KEY",
        "SEARX_URL",
        "SERPER_API_KEY",
    ):
        os.environ.pop(key, None)
    sys.path.insert(0, str(repo))
    from textual.css.query import NoMatches, QueryError
    from textual.widgets import Button, DataTable, Input, Select
    from textual_image._terminal import probe_terminal

    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.MCP.execution_log import build_record
    from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
    from tldw_chatbook.UI.MCP_Modules.mcp_servers_mode import MCPServersMode
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.Utils.app_shutdown import claim_process_exit

    evidence = root / "evidence"
    evidence.mkdir(exist_ok=False)
    (root / "launch.json").write_text(json.dumps({"pid": os.getpid()}))
    claim_process_exit()
    probe_terminal()
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
        "tldw_chatbook/MCP/client.py",
        "tldw_chatbook/MCP/local_control_service.py",
        "tldw_chatbook/MCP/unified_control_plane_service.py",
        "tldw_chatbook/UI/MCP_Modules/mcp_inspector.py",
        "Docs/superpowers/qa/2026-09-18-mcp-audit-selection/stdio_server.py",
    ]
    result = {
        "pid": os.getpid(),
        "cells": [],
        "source_hashes": {
            p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in paths
        },
        "runner_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
    }

    def record():
        (evidence / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    async def tmux(*args):
        return await asyncio.to_thread(
            subprocess.run,
            ["/opt/homebrew/bin/tmux", "-L", socket, *args],
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

        def visible(control):
            region, clip = app.screen._compositor.visible_widgets[control]
            assert region.intersection(clip) == region, (control.id, region, clip)

        async def focus(selector):
            control = app.screen.query_one(selector)
            control.focus()
            await settle()
            assert control.has_focus
            visible(control)
            if isinstance(control, Button):
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
            trace = root / "fixture-trace.jsonl"
            state = root / "fixture-result.json"
            state.write_text(json.dumps({"content": []}))
            processes = []
            for name in ("audit-alpha", "audit-beta"):
                await service.save_local_profile(
                    {
                        "profile_id": name,
                        "command": sys.executable,
                        "args": [
                            str(here.with_name("stdio_server.py")),
                            str(state),
                            str(trace),
                        ],
                        "env_placeholders": {},
                        "env_literals": {},
                    }
                )
                await service.connect_local_profile(name)
                processes.append(
                    service.local_service._get_client().sessions[name].process
                )
            result["fixture_pids"] = [process.pid for process in processes]
            workbench._snapshots = await workbench._collect_snapshots()
            await workbench._sync_children()
            log = service.execution_log

            def append(name, decision="allowed", initiator="test"):
                log.append(
                    build_record(
                        server_key="local:" + name,
                        tool_name="review_echo",
                        initiator=initiator,
                        ok=True,
                        duration_ms=42,
                        decision=decision,
                    )
                )

            append("audit-beta")
            append("audit-alpha", "denied", "agent")
            inspector = workbench.query_one(MCPInspector)
            for theme in ("textual-dark", "textual-light"):
                for width, height in ((80, 24), (170, 48)):
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
                    workbench.set_mode("audit")
                    await workbench._sync_audit_log_entries()
                    await settle()
                    field = workbench.query_one("#mcp-audit-filter-text", Input)
                    decision = workbench.query_one("#mcp-audit-filter-decision", Select)
                    initiator = workbench.query_one(
                        "#mcp-audit-filter-initiator", Select
                    )
                    field.value = ""
                    decision.value = Select.NULL
                    initiator.value = Select.NULL
                    await settle()
                    table = await focus("#mcp-audit-table")
                    beta_row = next(
                        i
                        for i in range(table.row_count)
                        if "audit-beta" in table.get_row_at(i)[1].plain
                    )
                    table.move_cursor(row=beta_row)
                    await settle()
                    await pilot.press("enter")
                    await wait_for(
                        lambda: inspector._current_audit_entry is not None,
                        "execution selected",
                    )
                    selected = dict(inspector._current_audit_entry)
                    assert selected["server_key"] == "local:audit-beta"
                    await capture(stem + "-selected")
                    append("audit-alpha", "denied", "agent")
                    await workbench._sync_audit_log_entries()
                    await settle()
                    assert "audit-beta" in table.get_row_at(table.cursor_row)[1].plain
                    assert inspector._current_audit_entry == selected
                    await pilot.press("enter")
                    await settle()
                    assert inspector._current_audit_entry == selected
                    await capture(stem + "-refreshed")
                    field = await focus("#mcp-audit-filter-text")
                    field.value = "audit-alpha"
                    await settle()
                    assert not inspector.query_one("#mcp-inspector-audit").display
                    assert not inspector.query("#mcp-audit-open-tool")
                    assert field.has_focus and table.row_count > 0
                    await capture(stem + "-filtered")
                    field.value = "missing"
                    await settle()
                    assert table.row_count == 0
                    field.value = "review_echo"
                    decision.value = "allowed"
                    initiator.value = "test"
                    await settle()
                    assert table.row_count == 1
                    assert not inspector.query_one("#mcp-inspector-audit").display
                    await focus("#mcp-audit-table")
                    await pilot.press("enter")
                    await wait_for(
                        lambda: inspector._current_audit_entry is not None,
                        "filtered event selected",
                    )
                    assert inspector._current_audit_entry == selected
                    await capture(stem + "-combined-filters")
                    cell = {
                        "theme": theme,
                        "size": [width, height],
                        "refresh_retained_event": True,
                        "filter_cleared_detail": True,
                        "zero_and_combined_filters": True,
                    }
                    if width == 170:
                        await focus("#mcp-audit-open-tool")
                        await pilot.press("enter")
                        await wait_for(
                            lambda: (
                                workbench.active_mode == "tools"
                                and inspector.current_tool is not None
                            ),
                            "exact tool drilldown",
                        )
                        assert inspector.current_tool.server_key == "local:audit-beta"
                        assert inspector.current_tool.name == "review_echo"
                        tools_table = workbench.query_one("#mcp-tools-table", DataTable)
                        assert (
                            tools_table.coordinate_to_cell_key(
                                (tools_table.cursor_row, 0)
                            )[0].value
                            == "local:audit-beta::review_echo"
                        )
                        assert not inspector.query_one("#mcp-inspector-audit").display
                        await capture(stem + "-exact-tool")
                        cell["exact_same_name_tool_drilldown"] = True
                    result["cells"].append(cell)
                    record()
            result["records"] = log.read_recent(200)
            requests = [json.loads(line) for line in trace.read_text().splitlines()]
            assert not any(row["method"] == "tools/call" for row in requests)
            result["wire_requests"] = requests
            for name in ("audit-alpha", "audit-beta"):
                await service.disconnect_local_profile(name)
                await service.delete_local_profile(name)
            result["fixture_returncodes"] = [
                process.returncode for process in processes
            ]
            result["passed"] = True
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
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
