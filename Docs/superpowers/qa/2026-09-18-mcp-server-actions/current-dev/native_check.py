"""Native server-action ownership and safe confirmation qualification.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
Uses real private local-profile persistence and toolbar keyboard activation.
No external server connection is attempted. Controlled queue/rebuild delays
exercise stale controls and an already-accepted confirmation.
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
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_chatbook.MCP.unified_control_plane_service import (
        UnifiedMCPControlPlaneService,
    )


async def _save_fixture_profile(
    service: UnifiedMCPControlPlaneService, profile_id: str
) -> str:
    """Create a fixture without overwriting an existing canonical profile.

    Args:
        service: Control plane for the validated private profile.
        profile_id: Fixture ID to normalize with the real local-profile boundary.

    Returns:
        The successfully saved canonical profile ID, owned by this runner.

    Raises:
        ValueError: The requested profile is invalid or already exists.
    """
    from tldw_chatbook.MCP.local_store import LocalExternalMCPProfile

    profile = LocalExternalMCPProfile.from_input_dict(
        {"profile_id": profile_id, "command": "/usr/bin/false"}
    )
    if service.local_service.store.get_profile(profile.profile_id) is not None:
        raise ValueError("Native fixture profile already exists")
    saved = await service.save_local_profile(profile.to_input_dict())
    return saved["profile_id"]


def main() -> None:
    """Qualify server actions in an unused private terminal profile.

    Raises:
        SystemExit: Zero on success, one on a captured journey failure, or two
            when the native CLI/profile validation rejects the request.
    """
    here = Path(__file__).resolve()
    repo = here.parents[5]
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
        "tldw_chatbook/MCP/local_store.py",
        "tldw_chatbook/MCP/local_control_service.py",
        "tldw_chatbook/MCP/unified_control_plane_service.py",
        "tldw_chatbook/Utils/terminal_utils.py",
        "Docs/superpowers/qa/native_runner_args.py",
        "Docs/superpowers/qa/2026-09-16-ingest-lifecycle/native_check.py",
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
            assert (
                app.screen.get_widget_at(
                    region.x + region.width // 2, region.y + region.height // 2
                )[0]
                is control
            )
            if isinstance(control, Button):
                painted = "".join(
                    app.screen._compositor.render_strips()[row].text
                    for row in range(region.y, region.bottom)
                )
                assert str(control.label) in painted, (control.id, painted)

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

        owned: set[str] = set()
        service = None
        canvas = None
        post = None
        rebuild = None
        release = None
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
            names = {
                type(app).__module__,
                type(workbench).__module__,
                type(canvas).__module__,
                type(service).__module__,
                type(service.local_service).__module__,
                "tldw_chatbook.MCP.local_store",
                "tldw_chatbook.Utils.terminal_utils",
                "tldw_chatbook.Utils.input_validation",
            }
            for name in sorted(names):
                origin = Path(sys.modules[name].__file__).resolve()
                assert origin.is_relative_to(repo), (name, origin)
                result["module_origins"][name] = str(origin)
            baseline_profiles = {
                record["profile_id"]
                for record in await service.local_external_catalog()
            }
            # Reject either occupied ID before creating the other fixture.
            if baseline_profiles.intersection({"alpha", "beta"}):
                raise ValueError("Native fixture profile already exists")
            owned.add(await _save_fixture_profile(service, "beta"))
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
                    owned.add(await _save_fixture_profile(service, "alpha"))
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
                    await pilot.press("enter")
                    await settle()
                    assert not canvas._delete_armed
                    await focus("#mcp-detail-delete")
                    await pilot.press("enter")
                    await settle()
                    await pilot.press("escape")
                    await settle()
                    assert not canvas._delete_armed
                    assert {
                        r["profile_id"] for r in await service.local_external_catalog()
                    } == baseline_profiles | {"alpha", "beta"}

                    # A rapid mode round trip must not revive queued delete consent.
                    await focus("#mcp-detail-delete")
                    await pilot.press("enter")
                    await settle()
                    old = await focus("#mcp-detail-delete-confirm")
                    pending = []
                    post = canvas.post_message

                    def hold_confirm(message, old=old, pending=pending, post=post):
                        if (
                            isinstance(message, Button.Pressed)
                            and message.button is old
                        ):
                            pending.append(message)
                            return True
                        return post(message)

                    canvas.post_message = hold_confirm
                    try:
                        await pilot.press("enter")
                        await wait_for(
                            lambda pending=pending: len(pending) == 1,
                            "queued alpha confirmation",
                        )
                        async with workbench._sync_children_lock:
                            workbench.set_mode("tools")
                            workbench.set_mode("servers")
                            post(pending[0])
                            await settle()
                            assert not workbench._profile_delete_in_flight
                    finally:
                        canvas.post_message = post
                    await wait_for(
                        lambda: bool(canvas.query("#mcp-detail-delete")),
                        "fresh server toolbar",
                    )
                    assert not canvas._delete_armed
                    assert {
                        r["profile_id"] for r in await service.local_external_catalog()
                    } == baseline_profiles | {"alpha", "beta"}
                    await focus("#mcp-detail-delete")
                    await capture(stem + "-round-trip-kept-profiles")

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
                    try:
                        await pilot.press("enter")
                        await wait_for(
                            lambda pending=pending: len(pending) == 1,
                            "queued alpha connect",
                        )
                        await workbench._select_server_key("local:beta")
                    finally:
                        canvas.post_message = post
                    post(pending[0])
                    await settle()
                    assert not workbench._in_flight
                    assert workbench._selected_server_key == "local:beta"
                    records = await service.local_external_catalog()
                    assert not any(
                        r.get("runtime_state")
                        for r in records
                        if r["profile_id"] in owned
                    ), records
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
                        canvas._rebuild_detail_toolbar = rebuild
                    await wait_for(
                        lambda: (
                            "alpha" not in workbench._catalog_records
                            and not workbench._profile_delete_in_flight
                        ),
                        "delete alpha only",
                    )
                    records = await service.local_external_catalog()
                    assert {r["profile_id"] for r in records} == baseline_profiles | {
                        "beta"
                    }, records
                    owned.remove("alpha")
                    assert workbench._selected_server_key == "local:beta"
                    await focus("#mcp-detail-edit")
                    await capture(stem + "-beta-preserved")
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": [width, height],
                            "safe_confirmation_visible": True,
                            "keep_kept_profiles": True,
                            "escape_kept_profiles": True,
                            "round_trip_confirmation_ignored": True,
                            "retired_connect_ignored": True,
                            "accepted_delete_only_alpha": True,
                            "beta_selected_and_persisted": True,
                        }
                    )
                    record()
            await service.delete_local_profile("beta")
            owned.remove("beta")
            assert {
                r["profile_id"] for r in await service.local_external_catalog()
            } == baseline_profiles
            result["passed"] = True
        except Exception:  # noqa: BLE001 - preserve failure and shut down the native app
            result.update(passed=False, error=traceback.format_exc())
            app.save_screenshot("failed-state.svg", path=str(evidence))
        finally:
            if release is not None:
                release.set()
            if canvas is not None:
                if post is not None:
                    canvas.post_message = post
                if rebuild is not None:
                    canvas._rebuild_detail_toolbar = rebuild
            if service is not None:
                for profile_id in sorted(owned):
                    try:
                        await service.disconnect_local_profile(profile_id)
                        await service.delete_local_profile(profile_id)
                    except Exception:  # noqa: BLE001 - preserve cleanup failure
                        result.update(
                            passed=False, cleanup_error=traceback.format_exc()
                        )
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
