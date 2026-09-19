"""Native compact inspector access and execution through a real private stdio server.

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
    from textual.widgets import Button, Static
    from textual_image._terminal import probe_terminal

    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
    from tldw_chatbook.UI.MCP_Modules.mcp_rail import MCPRail
    from tldw_chatbook.UI.MCP_Modules.mcp_servers_mode import MCPServersMode
    from tldw_chatbook.UI.MCP_Modules.mcp_tools_mode import MCPToolsMode
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
        for p in ("mcp_servers_mode", "mcp_workbench", "mcp_profile_form")
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
        "Docs/superpowers/qa/2026-09-18-mcp-inspector-reachability/stdio_server.py",
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
            await service.save_local_profile(
                {
                    "profile_id": "execution-review",
                    "command": sys.executable,
                    "args": [
                        str(
                            repo
                            / "Docs/superpowers/qa/2026-09-18-mcp-inspector-reachability/stdio_server.py"
                        ),
                        str(state),
                        str(trace),
                    ],
                    "env_placeholders": {},
                    "env_literals": {},
                }
            )
            await service.connect_local_profile("execution-review")
            client = service.local_service._get_client()
            process = client.sessions["execution-review"].process
            result["fixture_pid"] = process.pid
            workbench._snapshots = await workbench._collect_snapshots()
            await workbench._sync_children()
            workbench.set_mode("tools")
            await settle()
            mode = workbench.query_one(MCPToolsMode)
            assert await mode.select_tool_row("local:execution-review::review_echo")
            await focus("#mcp-tools-table")
            await pilot.press("enter")
            inspector = workbench.query_one(MCPInspector)
            await wait_for(
                lambda: (
                    inspector.current_tool is not None
                    and inspector.current_tool.name == "review_echo"
                ),
                "tool selected",
            )
            await focus("#mcp-inspector-test-tool")
            await pilot.press("enter")
            await wait_for(lambda: inspector._test_preview is not None, "preview ready")

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
                    field = await focus("#mcp-schema-field-0")
                    field.value = ""
                    run = await focus("#mcp-inspector-test-run")
                    before = len(service.execution_log.read_recent(50))
                    await pilot.press("enter")
                    summary = inspector.query_one("#mcp-inspector-test-result", Static)
                    await wait_for(
                        lambda summary=summary: str(summary.renderable).startswith(
                            "Failed"
                        ),
                        "validation failure",
                    )
                    assert len(service.execution_log.read_recent(50)) == before
                    summary.scroll_visible(immediate=True)
                    await settle()
                    visible(summary)
                    await capture(stem + "-invalid")
                    field = await focus("#mcp-schema-field-0")
                    field.value = "corrected"
                    state.write_text(
                        json.dumps(
                            {"content": [{"type": "text", "text": "Echo: corrected"}]}
                        )
                    )
                    await focus("#mcp-inspector-test-run")
                    region = run.region
                    paint = " ".join(
                        app.screen._compositor.render_strips()[y]
                        .crop(region.x, region.right)
                        .text.strip()
                        for y in range(region.y, region.bottom)
                    )
                    assert "Approve & run once" in " ".join(paint.split()), paint
                    await pilot.press("enter")
                    await wait_for(
                        lambda before=before: (
                            len(service.execution_log.read_recent(50)) > before
                            and inspector._test_preview is not None
                        ),
                        "tool execution",
                    )
                    summary.scroll_visible(immediate=True)
                    await settle()
                    visible(summary)
                    assert str(summary.renderable).startswith("OK")
                    audit = service.execution_log.read_recent(1)[0]
                    assert audit["ok"] is True
                    assert client.sessions["execution-review"].process is process
                    assert process.returncode is None
                    await capture(stem + "-executed")
                    await focus("#mcp-inspector-test-close")
                    await pilot.press("enter")
                    await wait_for(
                        lambda: not inspector.query("#mcp-inspector-test-panel"),
                        "panel closed",
                    )
                    await focus("#mcp-inspector-test-tool")
                    await pilot.press("enter")
                    await wait_for(
                        lambda: inspector._test_preview is not None, "reopened preview"
                    )
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": [width, height],
                            "validation_failed_without_execution": True,
                            "approval_label": paint,
                            "executed": audit,
                            "same_connection": True,
                            "closed_and_reopened": True,
                        }
                    )
                    record()
            workbench.set_mode("servers")
            await settle()
            rail = workbench.query_one(MCPRail)
            row = next(
                button
                for button, key in rail._row_targets.items()
                if key == "local:execution-review"
            )
            await focus("#" + row.id)
            await pilot.press("enter")
            await wait_for(
                lambda: bool(
                    inspector.query("#mcp-inspector-action-refresh_discovery")
                ),
                "selected server readiness",
            )
            result["readiness_cells"] = []
            for theme in ("textual-dark", "textual-light"):
                for width, height in ((80, 24), (170, 48)):
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
                        "readiness resize",
                    )
                    labels = []
                    for button in inspector.query(Button):
                        if not button.display or not all(
                            parent.display for parent in button.ancestors
                        ):
                            continue
                        if button.disabled:
                            button.scroll_visible(immediate=True)
                            await settle()
                        else:
                            await focus("#" + button.id)
                        visible(button)
                        region = button.region
                        paint = " ".join(
                            app.screen._compositor.render_strips()[y]
                            .crop(region.x, region.right)
                            .text.strip()
                            for y in range(region.y, region.bottom)
                        )
                        assert " ".join(str(button.label).split()) in " ".join(
                            paint.split()
                        ), paint
                        labels.append(str(button.label))
                    assert "Refresh tools" in labels
                    await capture(f"{theme}-{width}x{height}-readiness")
                    result["readiness_cells"].append(
                        {"theme": theme, "size": [width, height], "labels": labels}
                    )
                    record()
            result["audit"] = service.execution_log.read_recent(50)
            assert len(result["audit"]) == 4
            requests = [json.loads(line) for line in trace.read_text().splitlines()]
            assert sum(row["method"] == "tools/call" for row in requests) == 4
            result["wire_requests"] = requests
            await service.disconnect_local_profile("execution-review")
            result["fixture_returncode"] = process.returncode
            await service.delete_local_profile("execution-review")
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
