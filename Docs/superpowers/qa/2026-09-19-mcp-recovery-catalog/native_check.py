"""Native reviewed MCP catalog and guidance in a real disposable restore.

The actual passive catalog reader supplies fresh disconnected definitions.
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
    from loguru import logger
    from textual.css.query import NoMatches, QueryError
    from textual.widgets import Button, DataTable, Static
    from textual_image._terminal import probe_terminal

    from Tests.Backup_Recovery.test_mcp_recovery_review import _SETUP

    logger.remove()
    logger.add(root / "native.log", level="INFO")
    fixture = {}
    # The restore fixture retains real owner bindings; keep unrelated onboarding
    # and catalog refresh out of this disposable visual journey.
    setup = _SETUP.replace(
        r'[general]\nusers_name="Local"\n[paths]',
        r'[general]\nusers_name="Local"\nlog_level="INFO"\n'
        r"[first_run]\nsetup_completed=true\n"
        r"[splash_screen]\nenabled=false\n"
        r"[model_catalog]\nauto_refresh_enabled=false\nrefresh_on_startup=false\n"
        r"[paths]",
    )
    exec(setup, fixture)  # noqa: S102 - fixed repository-owned restore fixture
    activation, witness = fixture["activation"], fixture["witness"]
    historical, history = fixture["user"], fixture["history"]
    from tldw_chatbook.Backup_Recovery.bootstrap import effective_config_path

    assert effective_config_path().is_relative_to(root)
    effects = []

    def forbidden(*args, **kwargs):
        effects.append("MCP connection or execution")
        raise AssertionError("Native review must stay passive")

    client = fixture["MCPClient"]
    for name in ("connect_to_server", "list_tools", "execute_tool"):
        setattr(client, name, forbidden)
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.MCP_Modules.mcp_permissions_mode import MCPPermissionsMode
    from tldw_chatbook.UI.MCP_Modules.mcp_servers_mode import MCPServersMode
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.Utils.app_shutdown import claim_process_exit
    from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

    evidence = root / "evidence"
    evidence.mkdir(exist_ok=False)
    (root / "launch.json").write_text(json.dumps({"pid": os.getpid()}))
    claim_process_exit()
    probe_terminal()
    app = TldwCli()
    paths = [
        "tldw_chatbook/UI/MCP_Modules/" + p + ".py"
        for p in ("mcp_permissions_mode", "mcp_workbench", "mcp_inspector")
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
        "tldw_chatbook/UI/Screens/mcp_screen.py",
        "tldw_chatbook/MCP/recovery_activation.py",
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
            assert service.permission_store.get_global_default() == "ask"
            assert not activation.allowed(witness["generation"], "mcp.local")
            approvals = 0
            notices = []
            notify = app.notify

            def observe(message, **kwargs):
                notices.append(str(message))
                return notify(message, **kwargs)

            app.notify = observe
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
                    await focus("#mcp-mode-permissions")
                    await pilot.press("enter")
                    await settle()
                    assert workbench.active_mode == "permissions"
                    approved_before = activation.allowed(
                        witness["generation"], "mcp.local"
                    )
                    await focus("#mcp-perm-recovery-review")
                    await pilot.press("enter")
                    await wait_for(
                        lambda: isinstance(app.screen, ConfirmationDialog),
                        "review confirmation",
                    )
                    content = app.screen.query_one("#confirmation-dialog")
                    content.scroll_home(animate=False)
                    await settle()
                    # Reading the top moved the already-focused action offscreen.
                    # Real Tab navigation reveals it again before activation.
                    await pilot.press("tab")
                    await settle()
                    await focus("#cancel-button")
                    await pilot.press("enter")
                    await wait_for(
                        lambda: not isinstance(app.screen, ConfirmationDialog),
                        "review cancelled",
                    )
                    assert (
                        activation.allowed(witness["generation"], "mcp.local")
                        == approved_before
                    )
                    assert (
                        not workbench._mcp_recovery_busy
                        and workbench._mcp_recovery_token is None
                    )
                    await focus("#mcp-perm-recovery-review")
                    await pilot.press("enter")
                    await wait_for(
                        lambda: isinstance(app.screen, ConfirmationDialog),
                        "review again",
                    )
                    await focus("#confirm-button")
                    await pilot.press("enter")
                    await wait_for(
                        lambda: (
                            not isinstance(app.screen, ConfirmationDialog)
                            and not workbench._mcp_recovery_busy
                            and any("Fresh MCP roots reviewed" in n for n in notices)
                        ),
                        "fresh defaults displayed",
                    )
                    approvals += 1
                    notices.clear()
                    assert service.permission_store.get_global_default() == "ask"
                    assert workbench._selected_server_key is None
                    assert workbench._source == "local"
                    assert (
                        workbench.query_one("#mcp-servers-table", DataTable).row_count
                        == 2
                    )
                    record_value = workbench._catalog_records["demo"]
                    assert record_value["command"] == "disposable-sentinel"
                    assert not record_value["is_connected"]
                    assert not record_value["discovery_snapshot"]
                    copy = " ".join(
                        str(widget.renderable)
                        for widget in workbench.query_one(MCPPermissionsMode).query(
                            Static
                        )
                    )
                    assert "Historical rules and grants remain inactive" in copy
                    assert "stay inactive until reviewed" not in copy
                    assert (
                        workbench.query_one("#mcp-audit-table", DataTable).row_count
                        == 0
                    )
                    assert not service.local_service.store.list_governance_rules()
                    assert not service.local_service.store.list_approval_requests()
                    assert all(
                        (historical / name).read_bytes() == data
                        for name, data in history.items()
                    )
                    assert all(
                        activation.allowed(witness["generation"], owner)
                        for owner in (
                            "mcp.local",
                            "mcp.permissions",
                            "mcp.context",
                            "mcp.targets",
                        )
                    )
                    assert not activation.allowed(witness["generation"], "config")
                    assert not activation.allowed(witness["generation"], "skills")
                    assert not effects and not fixture["blocked_attempts"]()
                    await capture(stem + "-fresh-defaults")
                    await focus("#mcp-mode-servers")
                    await pilot.press("enter")
                    await settle()
                    table = await focus("#mcp-servers-table")
                    identities = [
                        table.coordinate_to_cell_key((i, 0))[0].value
                        for i in range(table.row_count)
                    ]
                    assert "local:demo" in identities
                    await pilot.press("ctrl+home")
                    for _ in range(identities.index("local:demo")):
                        await pilot.press("down")
                    await pilot.press("enter")
                    await settle()
                    assert workbench._selected_server_key == "local:demo"
                    client = service.local_service.client
                    assert client is None or not client.sessions
                    assert not effects and not fixture["blocked_attempts"]()
                    await capture(stem + "-reviewed-server")
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": [width, height],
                            "cancellation_preserved_activation": True,
                            "fresh_ask_local_defaults": True,
                            "historical_bytes_retained": True,
                            "only_mcp_owners_approved": True,
                            "reviewed_definition_in_servers": True,
                            "historical_rules_still_inactive": True,
                        }
                    )
                    record()
            result.update(approvals=approvals, passive_only=True)
            result["passed"] = True
        except Exception:  # noqa: BLE001 - preserve failure and shut down the native app
            result.update(passed=False, error=traceback.format_exc())
            app.save_screenshot("failed-state.svg", path=str(evidence))
        finally:
            record()
            if isinstance(app.screen, ConfirmationDialog):
                await pilot.press("escape")
                await settle()
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
