"""Native MCP gate label, keyboard traversal and private toggle verification."""

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
    for key in (
        "NO_COLOR",
        "OPENAI_API_KEY",
        "LLAMA_CPP_API_KEY",
        "SEARX_URL",
        "SERPER_API_KEY",
    ):
        os.environ.pop(key, None)
    sys.path.insert(0, str(repo))
    import tomllib

    from textual.widgets import Button
    from textual_image._terminal import probe_terminal

    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.config import get_cli_setting
    from tldw_chatbook.UI.MCP_Modules.mcp_rail import MCPRail
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.Utils.app_shutdown import claim_process_exit

    evidence = root / "evidence"
    evidence.mkdir(exist_ok=False)
    (root / "launch.json").write_text(json.dumps({"pid": os.getpid()}))
    claim_process_exit()
    probe_terminal()
    app = TldwCli()
    sources = [
        "tldw_chatbook/UI/MCP_Modules/mcp_servers_mode.py",
        "tldw_chatbook/UI/MCP_Modules/mcp_workbench.py",
        "tldw_chatbook/UI/MCP_Modules/mcp_rail.py",
        "tldw_chatbook/Agents/builtin_tool_gate.py",
        "tldw_chatbook/Agents/tool_catalog.py",
        "tldw_chatbook/css/components/_agentic_terminal.tcss",
        "tldw_chatbook/css/tldw_cli_modular.tcss",
        "tldw_chatbook/css/core/_variables.tcss",
        "tldw_chatbook/css/widget_defaults_scoped.tcss",
        "tldw_chatbook/css/widget_defaults_self.tcss",
        str(Path(__file__).relative_to(repo)),
        "Tests/UI/test_mcp_gate_label_layout.py",
    ]
    assert all((repo / p).is_file() for p in sources), sources
    result = {
        "pid": os.getpid(),
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "source_hashes": {
            p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in sources
        },
        "cells": [],
        "fixture_scope": "Real TldwCli and private config/service. Production MCP navigation, keyboard Servers/built-in selection, direct focus of first gate followed by real Tab traversal, and real deep-research gate save/reversal. No tool execution, remote server, permission mutation or broader gate-save concurrency qualification.",
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

    def visible(modal, widget):
        region, clip = modal._compositor.visible_widgets[widget]
        assert region.intersection(clip) == region, (widget.id, region, clip)

    def painted(modal, widget):
        r = widget.content_region
        return "\n".join(
            s.crop(r.x, r.right).text
            for s in modal._compositor.render_strips()[r.y : r.bottom]
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

        async def capture(stem):
            await wait_for(lambda: not app.screen.query("Toast"), "notices clear")
            app.save_screenshot(stem + ".svg", path=str(evidence))
            pane = await tmux("capture-pane", "-p", "-t", session)
            (evidence / (stem + ".txt")).write_text(pane.stdout)

        def read_tools():
            return tomllib.loads((root / "config.toml").read_text())["tools"]

        def assert_label(button):
            visible(app.screen, button)
            assert "".join(button.label.plain.split()) in "".join(
                painted(app.screen, button).split()
            )

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
                    and bool(workbench.query("#mcp-perm-table"))
                ),
                "MCP loaded",
            )
            store = app.unified_mcp_service.permission_store
            policy_before = store.read_snapshot_strict().payload["profiles"]
            app.screen.query_one("#mcp-mode-servers", Button).focus()
            await pilot.press("enter")
            await settle()
            rail = workbench.query_one(MCPRail)
            builtin = next(
                button
                for button, key in rail._row_targets.items()
                if key == "builtin:tldw_chatbook"
            )
            builtin.focus()
            await pilot.press("enter")
            await settle()
            assert workbench._selected_server_key == "builtin:tldw_chatbook"
            assert get_cli_setting("console", "local_tools_enabled", True) is True
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
                    buttons = list(workbench.query("#mcp-detail-tool-gates Button"))
                    assert len(buttons) >= 9 and not any(b.disabled for b in buttons)
                    buttons[0].focus()
                    await settle()
                    labels = []
                    for index, button in enumerate(buttons):
                        assert app.focused is button
                        assert_label(button)
                        labels.append(
                            {
                                "id": button.id,
                                "label": button.label.plain,
                                "region": list(button.region),
                            }
                        )
                        if index + 1 < len(buttons):
                            await pilot.press("tab")
                            await settle()
                    deep_id = "mcp-gate-web_deep_search_enabled"
                    deep = workbench.query_one("#" + deep_id, Button)
                    deep.focus()
                    await settle()
                    before = read_tools()
                    old_value = bool(
                        get_cli_setting("tools", "web_deep_search_enabled", False)
                    )
                    for desired in (not old_value, old_value):
                        previous = workbench.query_one("#" + deep_id, Button)
                        await wait_for(
                            lambda previous=previous: not previous.has_class("-active"),
                            "button ready",
                        )
                        await pilot.press("enter")

                        def saved(desired=desired, previous=previous, deep_id=deep_id):
                            rows = list(workbench.query("#" + deep_id))
                            return (
                                bool(rows)
                                and rows[0] is not previous
                                and rows[0].is_mounted
                                and get_cli_setting(
                                    "tools", "web_deep_search_enabled", False
                                )
                                is desired
                                and app.focused is rows[0]
                            )

                        await wait_for(saved, "gate save and focus restored")
                        expected = {**before, "web_deep_search_enabled": desired}
                        assert read_tools() == expected
                        current = workbench.query_one("#" + deep_id, Button)
                        assert current.label.plain.endswith(
                            ": on ▸" if desired else ": off ▸"
                        )
                        assert_label(current)
                        if desired != old_value:
                            await capture(stem + "-deep-research")
                    ask = workbench.query_one("#mcp-gate-ask_user_enabled", Button)
                    ask.focus()
                    await settle()
                    assert_label(ask)
                    await capture(stem + "-ask-user")
                    assert (
                        store.read_snapshot_strict().payload["profiles"]
                        == policy_before
                    )
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": [width, height],
                            "painted_gate_labels": labels,
                            "real_save_reversal": [not old_value, old_value],
                            "focus_restored": True,
                            "only_requested_tool_setting_changed": True,
                            "permission_profiles_unchanged": True,
                        }
                    )
                    record()
            result["passed"] = True
        except Exception:  # noqa: BLE001 - retain failed native journey evidence
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
