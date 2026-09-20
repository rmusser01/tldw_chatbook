"""Real local stdio connect, catalog refresh, failure and retry in native TldwCli.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
Private paths are validated before application imports; no remote server is used.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import runpy
import subprocess
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_chatbook.MCP.unified_control_plane_service import (
        UnifiedMCPControlPlaneService,
    )


async def _save_fixture_profile(
    service: UnifiedMCPControlPlaneService, repo: Path, evidence: Path
) -> str:
    """Create the local fixture only when its ID is unoccupied.

    Args:
        service: Control plane bound to the validated private profile.
        repo: Checkout containing the local JSON-RPC fixture.
        evidence: Newly created exclusive directory for state and wire trace files.

    Returns:
        The successfully created profile ID, owned by this runner.

    Raises:
        ValueError: The fixture ID belongs to an existing profile.
    """
    if service.local_service.store.get_profile("wire-review") is not None:
        raise ValueError("Native fixture profile wire-review already exists")
    await service.save_local_profile(
        {
            "profile_id": "wire-review",
            "command": sys.executable,
            "args": [
                str(repo / "Tests/MCP/fixtures/stdio_catalog_server.py"),
                str(evidence / "fixture-state.json"),
                str(evidence / "fixture-trace.jsonl"),
            ],
            "env_placeholders": {},
            "env_literals": {},
        }
    )
    return "wire-review"


def main() -> None:
    """Qualify real stdio catalog refresh in an unused private terminal profile.

    Raises:
        SystemExit: Zero for success, one for a journey failure, or two for
            invalid native runner arguments before application startup.
    """
    here = Path(__file__).resolve()
    repo = here.parents[5]
    sys.path.insert(0, str(repo))
    args = runpy.run_path(str(repo / "Docs/superpowers/qa/native_runner_args.py"))[
        "parse_native_args"
    ]()
    root, socket, session = args.root, args.tmux_socket, args.session
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
    from textual.widgets import Button

    from Tests.network_guard import blocked_attempts, install

    install()
    logger.remove()
    logger.add(root / "native.log", level="INFO")

    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.MCP_Modules.mcp_servers_mode import MCPServersMode
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.Utils.app_shutdown import claim_process_exit
    from tldw_chatbook.Utils.terminal_utils import warm_up_image_protocol

    evidence = root / "evidence"
    evidence.mkdir(exist_ok=False)
    (root / "launch.json").write_text(json.dumps({"pid": os.getpid()}))
    claim_process_exit()
    warm_up_image_protocol()
    app = TldwCli()
    paths = [
        "tldw_chatbook/UI/MCP_Modules/" + p + ".py"
        for p in (
            "mcp_servers_mode",
            "mcp_workbench",
            "mcp_profile_form",
            "mcp_inspector",
        )
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
        "tldw_chatbook/MCP/local_control_service.py",
        "tldw_chatbook/MCP/client.py",
        "Tests/MCP/fixtures/stdio_catalog_server.py",
    ]
    result = {
        "pid": os.getpid(),
        "cells": [],
        "module_origins": {
            "tldw_chatbook.app": sys.modules["tldw_chatbook.app"].__file__
        },
        "source_hashes": {
            p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in paths
        },
        "runner_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
    }

    def record():
        (evidence / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    async def tmux(*command):
        return await asyncio.to_thread(
            subprocess.run,
            [args.tmux_path, "-L", socket, *command],
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
            result.setdefault("notifications", {})[stem] = [
                {"message": str(notice.message), "severity": str(notice.severity)}
                for notice in app._notifications
            ]
            app.save_screenshot(stem + "-feedback.svg", path=str(evidence))
            await wait_for(
                lambda: not app._notifications and not app.screen.query("Toast"),
                "notices clear",
            )
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
            module = type(workbench).__module__
            result["module_origins"][module] = sys.modules[module].__file__
            await wait_for(
                lambda: (
                    not workbench.is_loading
                    and not workbench._reloading
                    and bool(workbench.query(MCPServersMode))
                ),
                "MCP loaded",
            )
            service = app.unified_mcp_service
            state = evidence / "fixture-state.json"
            trace = evidence / "fixture-trace.jsonl"
            state.write_text(json.dumps({"version": "original"}))
            fixture_id = await _save_fixture_profile(service, repo, evidence)
            client = service.local_service._get_client()
            store = service.local_service.store
            key = "local:wire-review"
            processes = []

            def catalog(version):
                snapshot = store.get_discovery_snapshot("wire-review")
                assert [t["name"] for t in snapshot["tools"]] == [f"{version}_tool"]
                assert [r["uri"] for r in snapshot["resources"]] == [
                    f"fixture://{version}"
                ]
                assert [p["name"] for p in snapshot["prompts"]] == [f"{version}_prompt"]
                return snapshot

            async def activate(selector):
                await focus(selector)
                await pilot.press("enter")
                await wait_for(
                    lambda: key not in workbench._in_flight, "operation settled"
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
                    state.write_text(json.dumps({"version": "original"}))
                    workbench._snapshots = await workbench._collect_snapshots()
                    await workbench._sync_children()
                    await workbench._select_server_key(key)
                    await activate("#mcp-detail-connect")
                    catalog("original")
                    original = client.sessions["wire-review"].process
                    processes.append(original)
                    await focus("#mcp-inspector-action-refresh_discovery")
                    await capture(stem + "-connected")

                    state.write_text(json.dumps({"version": "updated"}))
                    await activate("#mcp-inspector-action-refresh_discovery")
                    catalog("updated")
                    refreshed = client.sessions["wire-review"].process
                    processes.append(refreshed)
                    assert (
                        refreshed.pid != original.pid
                        and original.returncode is not None
                    )
                    assert store.get_profile_runtime_state("wire-review")["ok"] is True
                    await focus("#mcp-inspector-action-refresh_discovery")
                    await capture(stem + "-refreshed")

                    state.write_text(json.dumps({"version": "updated", "fail": True}))
                    await activate("#mcp-inspector-action-refresh_discovery")
                    assert not client.sessions and refreshed.returncode is not None
                    catalog("updated")
                    failed = store.get_profile_runtime_state("wire-review")
                    assert (
                        failed["ok"] is False
                        and "Failed to connect profile" in failed["last_error"]
                    )
                    await focus("#mcp-inspector-action-refresh_discovery")
                    await capture(stem + "-failed-retry-available")

                    state.write_text(json.dumps({"version": "recovered"}))
                    await activate("#mcp-inspector-action-refresh_discovery")
                    catalog("recovered")
                    assert not client.sessions
                    recovered = store.get_profile_runtime_state("wire-review")
                    assert recovered["ok"] is True and not recovered.get("last_error")
                    await focus("#mcp-inspector-action-connect")
                    await capture(stem + "-recovered")
                    await activate("#mcp-detail-connect")
                    processes.append(client.sessions["wire-review"].process)
                    await activate("#mcp-detail-disconnect")
                    assert not client.sessions
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": [width, height],
                            "original_pid": original.pid,
                            "refreshed_pid": refreshed.pid,
                            "fresh_tools_resources_prompts": True,
                            "failure_preserved_catalog": True,
                            "failed_attempt": failed,
                            "retry_restored_disconnected_state": True,
                            "recovered_attempt": recovered,
                            "final_disconnect": True,
                        }
                    )
                    record()
            result["processes"] = [
                {"pid": proc.pid, "returncode": proc.returncode} for proc in processes
            ]
            assert all(proc.returncode is not None for proc in processes)
            requests = [json.loads(line) for line in trace.read_text().splitlines()]
            result["fixture_pids"] = sorted({request["pid"] for request in requests})
            assert len(result["fixture_pids"]) == 20
            assert sum(row["method"] == "initialize" for row in requests) == 20
            for method in ("tools/list", "resources/list", "prompts/list"):
                assert sum(row["method"] == method for row in requests) == 16
            result["wire_requests"] = requests
            await service.delete_local_profile(fixture_id)
            assert store.get_profile(fixture_id) is None
            result["network_attempts"] = list(blocked_attempts())
            assert not result["network_attempts"]
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
