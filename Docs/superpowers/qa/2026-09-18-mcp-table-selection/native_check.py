"""TASK-32796 native MCP table refresh and inspector selection.

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
                "tldw_chatbook/UI/Widgets/table_click_select.py",
                "tldw_chatbook/UI/MCP_Modules/mcp_audit_mode.py",
                "tldw_chatbook/UI/Voice_Cloning_Window.py",
                "tldw_chatbook/UI/Screens/mcp_screen.py",
                "tldw_chatbook/UI/MCP_Modules/mcp_rail.py",
                "tldw_chatbook/UI/MCP_Modules/mcp_servers_mode.py",
                "tldw_chatbook/MCP/local_config_saves.py",
                "tldw_chatbook/UI/MCP_Modules/mcp_local_master_button.py",
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
        from textual.widgets import Button, DataTable

        from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
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
                    and bool(workbench.query("#mcp-perm-table"))
                ),
                "MCP loaded",
            )
            store = app.unified_mcp_service.permission_store
            policy_before = store.read_snapshot_strict().payload["profiles"]
            result["fixture_setup"] = (
                "Fresh private profile; real app, catalog and keyboard. Select a tool, "
                "refresh with its table focused, navigate via the All servers rail button, "
                "then refresh the focused table after clearing. No external server or tool execution."
            )
            inspector = workbench.query_one(MCPInspector)
            reloads = []
            original_reload = workbench.reload

            async def observed_reload():
                await original_reload()
                reloads.append(True)

            workbench.reload = observed_reload

            async def refresh():
                count = len(reloads)
                await pilot.press("r")
                await wait_for(
                    lambda: len(reloads) > count, "Keyboard refresh completed"
                )
                await settle()

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
                    await focus("#mcp-mode-tools")
                    await pilot.press("enter")
                    await settle()
                    table = workbench.query_one("#mcp-tools-table", DataTable)
                    assert table.row_count > 2
                    table.focus()
                    await settle()
                    await pilot.press("home", "down")
                    await wait_for(
                        lambda: inspector.current_tool is not None,
                        "Tool selected by keyboard",
                    )
                    tool_id = inspector.current_tool.tool_id
                    detail = inspector.query_one("#mcp-inspector-tool-name")
                    selected_row = table.cursor_row
                    await refresh()
                    assert table.has_focus and table.cursor_row == selected_row
                    assert inspector.current_tool.tool_id == tool_id
                    assert inspector.query_one("#mcp-inspector-tool-name") is detail
                    await capture(stem + "-selected-after-refresh")
                    # The existing compact rail label is horizontally clipped.
                    # Qualify keyboard activation and record its paint boundary;
                    # this run does not approve that separate rail layout.
                    rail_button = workbench.query_one("#mcp-rail-row-0", Button)
                    rail_button.focus()
                    await settle()
                    assert app.focused is rail_button and not rail_button.disabled
                    rail_region, rail_clip = app.screen._compositor.visible_widgets[
                        rail_button
                    ]
                    assert rail_region.intersection(rail_clip).area > 0
                    rail_fully_visible = (
                        rail_region.intersection(rail_clip) == rail_region
                    )
                    await pilot.press("enter")
                    await wait_for(
                        lambda: inspector.current_tool is None,
                        "Rail navigation cleared tool",
                    )
                    table.focus()
                    await settle()
                    await refresh()
                    assert table.has_focus and inspector.current_tool is None
                    assert not inspector.query("#mcp-inspector-tool-name")
                    assert not inspector.query_one("#mcp-inspector-tool").display
                    await capture(stem + "-cleared-after-refresh")
                    await pilot.press("down")
                    await wait_for(
                        lambda: inspector.current_tool is not None, "Next arrow selects"
                    )
                    assert (
                        store.read_snapshot_strict().payload["profiles"]
                        == policy_before
                    )
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": size,
                            "rail_button_fully_visible": rail_fully_visible,
                            "rail_region": str(rail_region),
                            "rail_clip": str(rail_clip),
                            "selected_tool": tool_id,
                            "focused_refresh_retains_detail_node": True,
                            "rail_clear_survives_focused_refresh": True,
                            "next_arrow_selects": True,
                            "permission_profiles_unchanged": True,
                        }
                    )
                    record()
            workbench.reload = original_reload
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
