"""TASK-32790 native MCP Tool/State readability and selection continuity.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
All workflow and fixture writes stay in the private profile.
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
    (root / "launch.json").write_text(json.dumps({"pid": os.getpid()}))
    qa = Path(__file__).resolve().parents[1]
    runpy.run_path(str(qa / "2026-09-16-ingest-lifecycle/native_check.py"))[
        "validate_profile"
    ](root)
    evidence = root / "evidence"
    evidence.mkdir(exist_ok=False)
    os.environ.update(
        HOME=str(root / "home"),
        USERPROFILE=str(root / "home"),
        TLDW_CONFIG_PATH=str(root / "config.toml"),
        XDG_DATA_HOME=str(root / "data"),
        XDG_CONFIG_HOME=str(root / "config"),
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

    from textual.css.query import NoMatches, QueryError
    from textual_image._terminal import probe_terminal

    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Utils.app_shutdown import claim_process_exit

    claim_process_exit()
    probe_terminal()
    app = TldwCli()
    source = Path(__file__).resolve().parents[4]
    result = {
        "pid": os.getpid(),
        "cells": [],
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_hashes": {
            p: hashlib.sha256((source / p).read_bytes()).hexdigest()
            for p in (
                "tldw_chatbook/app.py",
                "tldw_chatbook/UI/Screens/mcp_screen.py",
                "tldw_chatbook/UI/MCP_Modules/mcp_rail.py",
                "tldw_chatbook/UI/MCP_Modules/mcp_tools_mode.py",
                "tldw_chatbook/UI/MCP_Modules/mcp_inspector.py",
                "tldw_chatbook/css/components/_agentic_terminal.tcss",
                "tldw_chatbook/UI/MCP_Modules/mcp_permissions_mode.py",
                "tldw_chatbook/UI/MCP_Modules/mcp_workbench.py",
                "tldw_chatbook/Tool_Packs/service.py",
                "tldw_chatbook/Tool_Packs/export.py",
                "tldw_chatbook/Tool_Packs/publication.py",
                "tldw_chatbook/Tool_Packs/activation.py",
                "tldw_chatbook/Tool_Packs/importer.py",
                "tldw_chatbook/Tool_Packs/removal.py",
                "tldw_chatbook/Widgets/enhanced_file_picker.py",
                "tldw_chatbook/Widgets/Settings_Widgets/tool_pack_import_review.py",
                "tldw_chatbook/Widgets/Settings_Widgets/tool_profiles_panel.py",
                "tldw_chatbook/Workspaces/agent_provisioning.py",
                "tldw_chatbook/UI/Screens/settings_screen.py",
                "tldw_chatbook/Widgets/workspace_persona_default.py",
                "tldw_chatbook/Widgets/workspace_create_modal.py",
                "tldw_chatbook/Character_Chat/local_character_persona_service.py",
                "tldw_chatbook/css/tldw_cli_modular.tcss",
                "tldw_chatbook/css/features/_settings.tcss",
                "tldw_chatbook/css/core/_variables.tcss",
                "tldw_chatbook/css/components/_dialogs.tcss",
                "tldw_chatbook/css/widget_defaults_scoped.tcss",
                "tldw_chatbook/css/widget_defaults_self.tcss",
                "tldw_chatbook/css/screen_agentic_settings.tcss",
                "tldw_chatbook/Workspaces/registry_service.py",
                "tldw_chatbook/config.py",
            )
        },
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
        from textual.coordinate import Coordinate
        from textual.geometry import Offset
        from textual.widgets import Button, DataTable, Input, Select

        from tldw_chatbook.config import get_cli_setting
        from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
        from tldw_chatbook.UI.MCP_Modules.mcp_tools_mode import MCPToolsMode
        from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

        async def wait_for(predicate, label):
            result["waiting_for"] = label
            record()
            deadline = asyncio.get_running_loop().time() + 35
            while asyncio.get_running_loop().time() < deadline:
                try:
                    if predicate():
                        await settle()
                        return
                except (NoMatches, QueryError):
                    pass
                await pilot.pause(0.03)
            raise AssertionError(label)

        async def settle():
            await pilot.pause()
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()

        def visible(control):
            region, clip = app.screen._compositor.visible_widgets[control]
            assert region.intersection(clip) == region

        async def focus(selector):
            control = app.screen.query_one(selector)
            assert not control.disabled
            control.focus()
            await settle()
            assert app.focused is control
            visible(control)
            if isinstance(control, Button):
                await wait_for(lambda: not control.has_class("-active"), "Button ready")
            return control

        def paint(region):
            return "\n".join(
                strip.crop(region.x, region.right).text
                for strip in app.screen._compositor.render_strips()[
                    region.y : region.bottom
                ]
            )

        def compact(text):
            return "".join(text.split())

        def cell_paint(table, column):
            region = table._get_cell_region(Coordinate(table.cursor_row, column))
            region = region.translate(
                Offset(
                    table.content_region.x - int(table.scroll_x),
                    table.content_region.y - int(table.scroll_y),
                )
            )
            clip = app.screen._compositor.visible_widgets[table][1]
            assert region.intersection(clip) == region
            assert region.intersection(table.scrollable_content_region) == region
            return paint(region)

        def readable(table, tool):
            assert table.scroll_x == 0
            assert compact(tool.name) in compact(cell_paint(table, 0))
            assert table.get_row_at(table.cursor_row)[1].plain in cell_paint(table, 1)

        async def capture(stem):
            await wait_for(lambda: not app.screen.query("Toast"), "Notices clear")
            app.save_screenshot(stem + ".svg", path=str(evidence))
            pane = await tmux("capture-pane", "-p", "-t", session)
            (evidence / (stem + ".txt")).write_text(pane.stdout)

        try:
            assert app._instance_lock_status.acquired
            assert type(app._driver).__name__ == "LinuxDriver"
            assert sys.stdout.isatty() and sys.stderr.isatty()
            assert app.console.file.isatty()
            result.update(driver="LinuxDriver", tty_streams=True, lock_acquired=True)
            await wait_for(lambda: getattr(app, "_ui_ready", False), "Startup")
            result["live_database_attributes"] = sorted(
                key
                for key, value in vars(app).items()
                if key.endswith("_db") and value is not None
            )
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
                    and bool(workbench.query("#mcp-tools-table"))
                ),
                "MCP loaded",
            )
            workbench.set_mode("tools")
            await settle()
            canvas = workbench.query_one(MCPToolsMode)
            table = canvas.query_one("#mcp-tools-table", DataTable)
            store = app.unified_mcp_service.permission_store
            policy_before = store.read_snapshot_strict().payload["profiles"]
            assert get_cli_setting("console", "local_tools_enabled", True) is True
            result["fixture_setup"] = (
                "Fresh private profile, real MCP inventory and configuration service. "
                "Route through the production navigation handler; subsequent controls "
                "use focus and keyboard. Read-only catalog operations only. "
                "No tool execution or provider requests."
            )
            for theme in ("textual-dark", "textual-light"):
                for size in ((80, 24), (170, 48)):
                    stem = f"{theme}-{size[0]}x{size[1]}"
                    app.theme = theme
                    await tmux(
                        "resize-window",
                        "-t",
                        session,
                        "-x",
                        str(size[0]),
                        "-y",
                        str(size[1]),
                    )
                    await wait_for(
                        lambda size=size: (app.size.width, app.size.height) == size,
                        "Terminal resized",
                    )
                    await wait_for(lambda: table.row_count > 0, "Catalog rows")
                    await focus("#mcp-tools-table")
                    tool = max(canvas._tools, key=lambda item: len(item.name))
                    await pilot.press("ctrl+home")
                    await pilot.press(*(["down"] * table.get_row_index(tool.tool_id)))
                    await settle()
                    target_key, _ = table.coordinate_to_cell_key((table.cursor_row, 0))
                    assert target_key.value == tool.tool_id
                    readable(table, tool)
                    other_size = (170, 48) if size == (80, 24) else (80, 24)
                    for resized in (other_size, size):
                        await tmux(
                            "resize-window",
                            "-t",
                            session,
                            "-x",
                            str(resized[0]),
                            "-y",
                            str(resized[1]),
                        )
                        await wait_for(
                            lambda resized=resized: (
                                (app.size.width, app.size.height) == resized
                            ),
                            "Resize with selected row",
                        )
                        assert app.focused is table
                        assert (
                            table.coordinate_to_cell_key((table.cursor_row, 0))[0]
                            == target_key
                        )
                        visible(table)
                        readable(table, tool)
                    await capture(stem + "-tool-state")
                    await pilot.press("end")
                    await settle()
                    last_column = len(table.columns) - 1
                    assert table.get_row_at(table.cursor_row)[
                        last_column
                    ].plain in cell_paint(table, last_column)
                    await capture(stem + "-metadata")
                    await pilot.press("home")
                    await settle()
                    readable(table, tool)
                    await pilot.press("enter")
                    await wait_for(
                        lambda tool=tool: (
                            workbench.query_one(MCPInspector).current_tool == tool
                        ),
                        "Exact row inspected",
                    )
                    field = await focus("#mcp-tools-filter-text")
                    assert isinstance(field, Input)
                    await pilot.press("home", "shift+end", "backspace", *tool.name)
                    await wait_for(
                        lambda field=field, tool=tool: field.value == tool.name,
                        "Tool name filter",
                    )
                    await focus("#mcp-tools-table")
                    assert (
                        table.coordinate_to_cell_key((table.cursor_row, 0))[0]
                        == target_key
                    )
                    readable(table, tool)
                    await capture(stem + "-filtered")
                    await focus("#mcp-tools-filter-text")
                    await pilot.press("home", "shift+end", "backspace")
                    await wait_for(
                        lambda field=field: field.value == "", "Filter cleared"
                    )
                    select = await focus("#mcp-tools-filter-server")
                    assert isinstance(select, Select)
                    assert "All servers" in paint(select.content_region)
                    server_index = next(
                        index
                        for index, (_, key) in enumerate(canvas._server_options(), 1)
                        if key == tool.server_key
                    )
                    await pilot.press(
                        "enter", "home", *(["down"] * server_index), "enter"
                    )
                    await wait_for(
                        lambda: canvas._filter_server_key is not None,
                        "Server filter chosen",
                    )
                    assert table.row_count > 0
                    assert canvas._filter_server_key == tool.server_key
                    await pilot.press("enter", "home", "enter")
                    await wait_for(
                        lambda: canvas._filter_server_key is None,
                        "All servers restored",
                    )
                    assert (
                        store.read_snapshot_strict().payload["profiles"]
                        == policy_before
                    )
                    assert get_cli_setting("console", "local_tools_enabled") is True
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": size,
                            "longest_tool_and_full_state_painted": True,
                            "last_metadata_column_accessible": True,
                            "selected_row_visible_through_resize": True,
                            "exact_row_inspected": tool.tool_id,
                            "text_and_server_filters_usable": True,
                            "selected_tool_retained_while_filtering": True,
                            "permission_profiles_unchanged": True,
                        }
                    )
                    record()
            result["passed"] = True
        except Exception:  # noqa: BLE001 - retain failed-run diagnostics
            result.update(
                passed=False,
                error=traceback.format_exc(),
                focused_id=getattr(app.focused, "id", None),
                focused_region=str(getattr(app.focused, "region", None)),
            )
            app.save_screenshot("failed-state.svg", path=str(evidence))
        finally:
            record()
            await tmux("send-keys", "-t", session, "C-q")

    app.run(auto_pilot=journey, size=(170, 48))
    result.update(app_run_returned=True)
    record()
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
