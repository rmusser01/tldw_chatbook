"""Native cancellation progress and retry qualification.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
Runs TldwCli with the real control-plane lifecycle, governance and attempt store.
Only the client's connection call is replaced by held cleanup followed by a
controlled failed retry. No external server or successful transport is qualified.
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
    from textual.widgets import Button, Static
    from textual_image._terminal import probe_terminal

    from tldw_chatbook.app import TldwCli
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
        for p in ("mcp_inspector", "mcp_workbench", "mcp_servers_mode")
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
            await pilot.pause(0.3)
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
            await service.save_local_profile(
                {
                    "profile_id": "cleanup-demo",
                    "command": "/usr/bin/false",
                    "args": [],
                    "env_placeholders": {},
                    "env_literals": {},
                }
            )
            client = service.local_service._get_client()
            original_connect = client.connect_to_server
            inspector = workbench.query_one(MCPInspector)
            key = "local:cleanup-demo"

            async def focus_lifecycle():
                for action in ("connect", "refresh_discovery"):
                    selector = f"#mcp-inspector-action-{action}"
                    if inspector.query(selector):
                        return await focus(selector)
                raise AssertionError("No lifecycle retry action rendered")

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
                    started, cancelling, release = (
                        asyncio.Event(),
                        asyncio.Event(),
                        asyncio.Event(),
                    )
                    calls = []
                    interrupted = []

                    async def connect(
                        profile_id,
                        *_args,
                        calls=calls,
                        started=started,
                        cancelling=cancelling,
                        release=release,
                        interrupted=interrupted,
                        **_kwargs,
                    ):
                        calls.append(profile_id)
                        if len(calls) == 1:
                            started.set()
                            try:
                                await asyncio.Event().wait()
                            except asyncio.CancelledError:
                                cancelling.set()
                                try:
                                    await release.wait()
                                except asyncio.CancelledError:
                                    interrupted.append(True)
                                    raise
                                raise
                        raise RuntimeError(
                            "Fixture connection failed; retry is available."
                        )

                    client.connect_to_server = connect
                    workbench._snapshots = await workbench._collect_snapshots()
                    await workbench._sync_children()
                    await workbench._select_server_key(key)
                    await focus_lifecycle()
                    await pilot.press("enter")
                    await wait_for(started.is_set, "connection started")
                    original_worker = workbench._in_flight[key]
                    await focus("#mcp-inspector-cancel")
                    visible(inspector.query_one("#mcp-inspector-message", Static))
                    await capture(stem + "-working")
                    await pilot.press("enter")
                    await wait_for(cancelling.is_set, "cleanup entered")
                    cancel = inspector.query_one("#mcp-inspector-cancel", Button)
                    assert cancel.disabled and str(cancel.label) == "Cancelling…"
                    visible(cancel)
                    visible(inspector.query_one("#mcp-inspector-message", Static))
                    assert workbench._in_flight[key] is original_worker
                    # The same real toolbar retry must remain blocked during cleanup.
                    await focus("#mcp-detail-connect")
                    await pilot.press("enter")
                    await settle()
                    cancel.press()
                    await settle()
                    assert calls == ["cleanup-demo"] and not interrupted
                    assert workbench._in_flight[key] is original_worker
                    await capture(stem + "-cancelling")
                    release.set()
                    await wait_for(
                        lambda: key not in workbench._in_flight, "cancellation settled"
                    )
                    state = service.local_service.store.get_profile_runtime_state(
                        "cleanup-demo"
                    )
                    assert (
                        state["ok"] is False and state["last_error"] == "Cancelled"
                    ), state
                    await focus_lifecycle()
                    await pilot.press("enter")
                    await wait_for(
                        lambda calls=calls: (
                            len(calls) == 2 and key not in workbench._in_flight
                        ),
                        "retry settled",
                    )
                    state = service.local_service.store.get_profile_runtime_state(
                        "cleanup-demo"
                    )
                    assert (
                        state["ok"] is False
                        and "Fixture connection failed" in state["last_error"]
                    ), state
                    await focus_lifecycle()
                    await capture(stem + "-retry-available")
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": [width, height],
                            "cancel_visible": True,
                            "cleanup_retained_admission": True,
                            "repeat_cancel_did_not_interrupt": not interrupted,
                            "cancellation_persisted_after_settlement": True,
                            "retry_admitted_after_settlement": calls
                            == ["cleanup-demo"] * 2,
                            "retry_error_persisted": state["last_error"],
                        }
                    )
                    record()
            client.connect_to_server = original_connect
            await service.delete_local_profile("cleanup-demo")
            assert await service.local_external_catalog() == []
            result["passed"] = True
        except Exception:  # noqa: BLE001 - preserve failure and shut down the native app
            result.update(passed=False, error=traceback.format_exc())
            app.save_screenshot("failed-state.svg", path=str(evidence))
        finally:
            if "release" in locals():
                release.set()
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
