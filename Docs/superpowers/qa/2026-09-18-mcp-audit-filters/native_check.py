"""Native Audit filter readability and keyboard access with real JSONL fixtures.

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
            for i in range(48):
                blocked = bool(i % 2)
                log.append(
                    build_record(
                        server_key="local:audit-review",
                        tool_name=f"review_{i:02}",
                        initiator="test",
                        ok=not blocked,
                        duration_ms=42,
                        decision="denied-killswitch" if blocked else "allowed",
                    )
                )
            workbench.set_mode("audit")
            await workbench._sync_audit_log_entries()
            await settle()
            table = workbench.query_one("#mcp-audit-table", DataTable)
            field = workbench.query_one("#mcp-audit-filter-text", Input)
            decision = workbench.query_one("#mcp-audit-filter-decision", Select)
            initiator = workbench.query_one("#mcp-audit-filter-initiator", Select)

            def painted(control):
                region = control.content_region
                strips = app.screen._compositor.render_strips()
                return "".join(
                    "".join(strips[y].crop(region.x, region.right).text.split())
                    for y in range(region.y, region.bottom)
                )

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
                    field.value = ""
                    decision.value = Select.NULL
                    initiator.value = Select.NULL
                    await settle()
                    await focus("#mcp-audit-filter-text")
                    assert "Filtertoolorserver…" in painted(field)
                    await pilot.press(*"review")
                    await settle()
                    assert field.value == "review" and "review" in painted(field)
                    assert table.row_count == 48
                    await capture(stem + "-text")
                    await pilot.press("tab")
                    await settle()
                    assert decision.has_focus
                    visible(decision)
                    assert "Alldecisions" in painted(decision)
                    await pilot.press("enter", "home", *(["down"] * 6))
                    await settle()
                    if width == 80:
                        await capture(stem + "-decision-menu")
                    await pilot.press("enter")
                    await settle()
                    assert decision.value == "denied-killswitch"
                    assert "Blocked(killswitch)" in painted(decision)
                    assert table.row_count == 24
                    await capture(stem + "-decision")
                    await pilot.press("tab")
                    await settle()
                    assert initiator.has_focus
                    visible(initiator)
                    assert "Allinitiators" in painted(initiator)
                    await pilot.press("enter", "home", "down", "enter")
                    await settle()
                    assert initiator.value == "test"
                    assert "Test" in painted(initiator)
                    assert table.row_count == 24
                    await capture(stem + "-initiator")
                    await pilot.press("tab")
                    await settle()
                    assert table.has_focus
                    visible(table)
                    await pilot.press("ctrl+end")
                    await settle()
                    assert table.cursor_row == 23
                    row = table._get_row_region(table.cursor_row)
                    y = table.content_region.y + row.y - int(table.scroll_y)
                    clip = app.screen._compositor.visible_widgets[table][1]
                    assert clip.y <= y and y + row.height <= clip.bottom
                    assert (
                        field.value == "review"
                        and decision.value == "denied-killswitch"
                        and initiator.value == "test"
                    )
                    await capture(stem + "-last-row")
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": [width, height],
                            "text_typed_and_visible": True,
                            "decision_keyboard_menu": True,
                            "initiator_keyboard_menu": True,
                            "tab_to_visible_last_row": True,
                            "filter_values_retained": True,
                            "filtered_count": table.row_count,
                        }
                    )
                    record()
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
