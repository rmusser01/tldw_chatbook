"""Native server-action ownership and safe confirmation qualification.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
Uses real private local-profile persistence and toolbar keyboard activation.
No external server connection is attempted. Controlled queue/rebuild delays
exercise stale controls and an already-accepted confirmation.
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
    from textual.widgets import Button
    from textual_image._terminal import probe_terminal

    from tldw_chatbook.app import TldwCli
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
                    and bool(workbench.query(MCPServersMode))
                ),
                "MCP loaded",
            )
            canvas = workbench.query_one(MCPServersMode)
            service = app.unified_mcp_service
            payload = lambda name: {
                "profile_id": name,
                "command": "/usr/bin/false",
                "args": [],
                "env_placeholders": {},
                "env_literals": {},
            }
            await service.save_local_profile(payload("beta"))
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
                    await service.save_local_profile(payload("alpha"))
                    workbench._snapshots = await workbench._collect_snapshots()
                    await workbench._sync_children()
                    await workbench._select_server_key("local:alpha")
                    await settle()
                    await focus("#mcp-detail-delete")
                    await pilot.press("enter")
                    await settle()
                    keep = canvas.query_one("#mcp-detail-delete-cancel", Button)
                    assert app.focused is keep
                    visible(keep)
                    visible(canvas.query_one("#mcp-detail-delete-confirm", Button))
                    await capture(stem + "-safe-confirmation")
                    await pilot.press("escape")
                    await settle()
                    assert not canvas._delete_armed
                    assert {
                        r["profile_id"] for r in await service.local_external_catalog()
                    } == {"alpha", "beta"}

                    # Hold a genuine pressed message at the canvas queue boundary.
                    old = await focus("#mcp-detail-connect")
                    pending = []
                    post = canvas.post_message

                    def hold(message, old=old, pending=pending, post=post):
                        if (
                            isinstance(message, Button.Pressed)
                            and message.button is old
                        ):
                            pending.append(message)
                            return True
                        return post(message)

                    canvas.post_message = hold
                    await pilot.press("enter")
                    await wait_for(
                        lambda pending=pending: len(pending) == 1,
                        "queued alpha connect",
                    )
                    await workbench._select_server_key("local:beta")
                    canvas.post_message = post
                    post(pending[0])
                    await settle()
                    assert not workbench._in_flight
                    assert workbench._selected_server_key == "local:beta"
                    records = await service.local_external_catalog()
                    assert not any(r.get("runtime_state") for r in records), records
                    await focus("#mcp-detail-edit")
                    await capture(stem + "-retired-action-ignored")

                    # Accept Alpha's confirmation, then hold only its presentation
                    # rebuild while selection moves to Beta. The real store deletes.
                    await workbench._select_server_key("local:alpha")
                    await focus("#mcp-detail-delete")
                    await pilot.press("enter")
                    await settle()
                    await focus("#mcp-detail-delete-confirm")
                    entered, release = asyncio.Event(), asyncio.Event()
                    rebuild = canvas._rebuild_detail_toolbar
                    calls = 0

                    async def held_rebuild(
                        entered=entered, release=release, rebuild=rebuild
                    ):
                        nonlocal calls
                        calls += 1
                        if calls == 1:
                            entered.set()
                            await release.wait()
                        await rebuild()

                    canvas._rebuild_detail_toolbar = held_rebuild
                    try:
                        await tmux("send-keys", "-t", session, "Enter")
                        await asyncio.wait_for(entered.wait(), 10)
                        await workbench._select_server_key("local:beta")
                    finally:
                        release.set()
                    await wait_for(
                        lambda: (
                            "alpha" not in workbench._catalog_records
                            and not workbench._profile_delete_in_flight
                        ),
                        "delete alpha only",
                    )
                    canvas._rebuild_detail_toolbar = rebuild
                    records = await service.local_external_catalog()
                    assert [r["profile_id"] for r in records] == ["beta"], records
                    assert workbench._selected_server_key == "local:beta"
                    await focus("#mcp-detail-edit")
                    await capture(stem + "-beta-preserved")
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": [width, height],
                            "safe_confirmation_visible": True,
                            "escape_kept_profiles": True,
                            "retired_connect_ignored": True,
                            "accepted_delete_only_alpha": True,
                            "beta_selected_and_persisted": True,
                        }
                    )
                    record()
            await service.delete_local_profile("beta")
            assert await service.local_external_catalog() == []
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
