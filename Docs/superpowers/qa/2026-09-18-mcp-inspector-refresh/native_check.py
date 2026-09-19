"""Native selected-tool refresh, draft, focus and raw-schema qualification.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
Catalog changes are controlled projections of one real executable tool. No
execution or remote server is involved; permission previews use the real service.
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
        str(here.parent.parent / "2026-09-16-ingest-lifecycle/native_check.py")
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
    from textual.widgets import Button, DataTable, Static, TextArea
    from textual_image._terminal import probe_terminal

    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
    from tldw_chatbook.UI.MCP_Modules.mcp_schema_form import parse_schema
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
        for p in ("mcp_inspector", "mcp_workbench", "mcp_tools_mode", "mcp_schema_form")
    ]
    paths += [
        "tldw_chatbook/css/" + p
        for p in (
            "tldw_cli_modular.tcss",
            "widget_defaults_scoped.tcss",
            "widget_defaults_self.tcss",
            "core/_variables.tcss",
        )
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
            await wait_for(lambda: not app.screen.query("Toast"), "notices clear")
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
                    and bool(workbench.query(MCPToolsMode))
                ),
                "MCP loaded",
            )
            await focus("#mcp-mode-tools")
            await pilot.press("enter")
            await settle()
            original_collect = workbench._collect_hub_tools
            catalog = original_collect()
            choices = [
                t
                for t in catalog
                if t.executable
                and parse_schema(t.input_schema)
                and any(f.kind == "string" for f in parse_schema(t.input_schema))
            ]
            assert choices, [(t.name, t.executable) for t in catalog]
            original = choices[0]
            result["tool"] = original.tool_id
            state = {"tool": original}

            def collect():
                return [
                    state["tool"] if t.tool_id == original.tool_id else t
                    for t in original_collect()
                    if state["tool"] is not None or t.tool_id != original.tool_id
                ]

            workbench._collect_hub_tools = collect
            inspector = workbench.query_one(MCPInspector)
            canvas = workbench.query_one(MCPToolsMode)
            store = app.unified_mcp_service.permission_store
            policy = store.read_snapshot_strict().payload["profiles"]
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
                    state["tool"] = original
                    await workbench._sync_children()
                    await canvas.select_tool_row(original.tool_id)
                    table = canvas.query_one(DataTable)
                    table.focus()
                    await pilot.press("enter")
                    await wait_for(
                        lambda: inspector.current_tool == original, "tool selected"
                    )
                    await focus("#mcp-inspector-test-tool")
                    await pilot.press("enter")
                    await wait_for(
                        lambda: inspector._test_preview is not None, "real preview"
                    )
                    fields = list(inspector.query("#mcp-inspector-test-form Input"))
                    assert fields
                    field = await focus("#" + fields[0].id)
                    await pilot.press("home", "shift+end", "backspace", *"keep draft")
                    await settle()
                    assert field.value == "keep draft"
                    preview = inspector._test_preview
                    state["tool"] = replace(original)
                    await workbench._sync_children()
                    await settle()
                    assert (
                        inspector.query_one("#" + field.id) is field and field.has_focus
                    )
                    assert (
                        field.value == "keep draft"
                        and inspector._test_preview is preview
                    )
                    visible(field)
                    await capture(stem + "-retained-draft")
                    state["tool"] = replace(
                        original,
                        description="Catalog changed. Review these current tool details.",
                        input_schema={"oneOf": [{"type": "object"}]},
                    )
                    await workbench._sync_children()
                    await settle()
                    assert inspector.current_tool == state["tool"]
                    assert inspector._test_preview is None and not inspector.query(
                        "#mcp-inspector-test-panel"
                    )
                    button = inspector.query_one("#mcp-inspector-test-tool", Button)
                    assert button.has_focus
                    visible(button)
                    note = inspector.query_one(
                        "#mcp-inspector-tool-refresh-note", Static
                    )
                    assert "Reopen Test Tool" in str(note.renderable)
                    await capture(stem + "-changed-details")
                    await focus("#mcp-inspector-test-tool")
                    await pilot.press("enter")
                    await wait_for(
                        lambda: inspector._test_preview is not None, "raw preview"
                    )
                    raw = await focus("#mcp-schema-raw")
                    assert isinstance(raw, TextArea)
                    raw.load_text('{"query": "keep raw draft"}')
                    await workbench._sync_children()
                    await settle()
                    assert raw.has_focus and raw.text == '{"query": "keep raw draft"}'
                    await capture(stem + "-raw-draft")
                    state["tool"] = None
                    await workbench._sync_children()
                    await settle()
                    assert (
                        inspector.current_tool is None
                        and inspector._test_preview is None
                    )
                    assert not inspector.query("#mcp-inspector-test-panel")
                    assert not inspector.query_one("#mcp-inspector-tool").display
                    assert store.read_snapshot_strict().payload["profiles"] == policy
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": [width, height],
                            "draft_and_preview_preserved": True,
                            "changed_form_retired": True,
                            "focus_restored": True,
                            "raw_draft_preserved": True,
                            "removed_tool_cleared": True,
                            "policy_unchanged": True,
                        }
                    )
                    record()
            workbench._collect_hub_tools = original_collect
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
