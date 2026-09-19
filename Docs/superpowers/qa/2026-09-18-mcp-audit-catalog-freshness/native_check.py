"""Native Audit navigation with controlled same-ID catalog replacement.

Real app/catalog; one synthetic metadata record, no tool execution.
"""

import asyncio
import hashlib
import json
import os
import runpy
import subprocess
import sys
import traceback
from dataclasses import replace
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
    from textual.widgets import Button, DataTable, Static
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
            log = service.execution_log
            assert log is not None
            target = next(
                tool
                for tool in workbench._last_hub_tools
                if tool.name == "chat_with_llm"
            )
            log.append(
                build_record(
                    server_key=target.server_key,
                    tool_name=target.name,
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
            collect = workbench._collect_hub_tools
            clear_audit = inspector.show_audit_entry
            replacement_enabled = False
            replace_on_clear = False
            refresh_count = 0

            def collect_review_catalog():
                tools = collect()
                if not replacement_enabled:
                    return tools
                return [
                    replace(
                        tool,
                        server_label="Refreshed catalog",
                        description="Catalog refreshed during Audit navigation.",
                        input_schema={
                            "type": "object",
                            "properties": {"review_query": {"type": "string"}},
                        },
                    )
                    if tool.tool_id == target.tool_id
                    else tool
                    for tool in tools
                ]

            async def refresh_during_clear(entry, **kwargs):
                nonlocal replacement_enabled, replace_on_clear, refresh_count
                await clear_audit(entry, **kwargs)
                if entry is None and replace_on_clear:
                    replace_on_clear = False
                    replacement_enabled = True
                    await workbench._sync_children()
                    refresh_count += 1

            workbench._collect_hub_tools = collect_review_catalog
            inspector.show_audit_entry = refresh_during_clear
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
                    for destination, button_id, table_id in [
                        ("tools", "mcp-audit-open-tool", "mcp-tools-table"),
                        (
                            "permissions",
                            "mcp-audit-adjust-permission",
                            "mcp-perm-table",
                        ),
                    ]:
                        replacement_enabled = False
                        await workbench._sync_children()
                        assert (
                            workbench._tool_for(
                                target.server_key, target.name
                            ).server_label
                            != "Refreshed catalog"
                        )
                        await focus("#mcp-mode-audit")
                        await pilot.press("enter")
                        await settle()
                        table = await focus("#mcp-audit-table")
                        assert table.row_count == 1
                        await pilot.press("enter")
                        await settle()
                        assert (
                            inspector._current_audit_entry["tool_name"] == target.name
                        )
                        replace_on_clear = True
                        await focus("#" + button_id)
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
                            "refreshed destination",
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
                        assert selected.server_label == "Refreshed catalog"
                        assert (
                            selected.description
                            == "Catalog refreshed during Audit navigation."
                        )
                        assert selected.input_schema["properties"] == {
                            "review_query": {"type": "string"}
                        }
                        if destination == "tools":
                            assert (
                                str(
                                    inspector.query_one(
                                        "#mcp-inspector-tool-description", Static
                                    ).renderable
                                )
                                == selected.description
                            )
                        table = workbench.query_one("#" + table_id, DataTable)
                        key, _ = table.coordinate_to_cell_key((table.cursor_row, 0))
                        assert key.value == target.tool_id
                        await capture(stem + "-" + destination)
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": [width, height],
                            "identity": [target.server_key, target.name],
                            "both_destinations_show_refreshed_definition": True,
                        }
                    )
                    record()
            result["controlled_catalog_replacements"] = refresh_count
            assert refresh_count == 8
            assert (
                service.permission_store.read_snapshot_strict().payload["profiles"]
                == profile_before
            )
            result["permission_profiles_unchanged"] = True
            result["record_count"] = len(log.read_recent(200))
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
