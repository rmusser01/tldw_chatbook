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
from datetime import UTC, datetime
from pathlib import Path

AUDIT_FIXTURE_COUNT = 48


def main() -> None:
    """Qualify Audit filters in an unused private native terminal profile.

    ROOT TMUX_SOCKET SESSION are admitted before app imports or output writes.
    The journey uses synthetic private JSONL records and executes no tools.

    Raises:
        SystemExit: Zero for success, one for journey failure, two for invalid CLI.
    """
    here = Path(__file__).resolve()
    repo = here.parents[4]
    sys.path.insert(0, str(repo))
    args = runpy.run_path(str(repo / "Docs/superpowers/qa/native_runner_args.py"))[
        "parse_native_args"
    ]()
    root, socket, session = args.root, args.tmux_socket, args.session
    tmux_path = args.tmux_path
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
    from loguru import logger
    from textual.css.query import NoMatches, QueryError
    from textual.widgets import Button, DataTable, Input, Select

    from Tests.network_guard import blocked_attempts, install

    install()
    logger.remove()
    logger.add(root / "native.log", level="INFO")

    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.MCP.execution_log import build_record
    from tldw_chatbook.UI.MCP_Modules.mcp_servers_mode import MCPServersMode
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.Utils.app_shutdown import claim_process_exit
    from tldw_chatbook.Utils.terminal_utils import warm_up_image_protocol

    evidence = root / "evidence"
    evidence.mkdir(exist_ok=False)
    (root / "launch.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "started_at": datetime.now(UTC).isoformat(),
                "repo": str(repo),
                "head": subprocess.check_output(
                    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
                ).strip(),
                "dirty_status": subprocess.check_output(
                    ["git", "-C", str(repo), "status", "--short"], text=True
                ),
            },
            indent=2,
        )
        + "\n"
    )
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
        "tldw_chatbook/app.py",
        "tldw_chatbook/config.py",
        "Docs/superpowers/qa/native_runner_args.py",
    ]
    result = {
        "pid": os.getpid(),
        "cells": [],
        "module_origins": {},
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
            [tmux_path, "-L", socket, *args],
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
            assert region.width > 0 and region.height > 0
            assert region.intersection(clip) == region, (control.id, region, clip)
            painted_widget = app.screen.get_widget_at(
                region.x + region.width // 2, region.y + region.height // 2
            )[0]
            assert painted_widget is control or control in painted_widget.ancestors

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
            for module_name in (
                "tldw_chatbook.app",
                "tldw_chatbook.config",
                "tldw_chatbook.UI.MCP_Modules.mcp_workbench",
                "tldw_chatbook.UI.MCP_Modules.mcp_audit_mode",
                "tldw_chatbook.UI.MCP_Modules.mcp_inspector",
                "tldw_chatbook.MCP.unified_control_plane_service",
            ):
                path = Path(sys.modules[module_name].__file__).resolve()
                assert path.is_relative_to(repo), (module_name, path)
                result["module_origins"][module_name] = str(path)
            service = app.unified_mcp_service
            log = service.execution_log
            assert log is not None
            for i in range(AUDIT_FIXTURE_COUNT):
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
                    assert table.row_count == AUDIT_FIXTURE_COUNT
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
                    assert table.row_count == AUDIT_FIXTURE_COUNT // 2
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
                    assert table.row_count == AUDIT_FIXTURE_COUNT // 2
                    await capture(stem + "-initiator")
                    await pilot.press("tab")
                    await settle()
                    assert table.has_focus
                    visible(table)
                    await pilot.press("ctrl+end")
                    await settle()
                    assert table.cursor_row == AUDIT_FIXTURE_COUNT // 2 - 1
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
            result["network_attempts"] = blocked_attempts()
            if result["network_attempts"]:
                result["passed"] = False
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
