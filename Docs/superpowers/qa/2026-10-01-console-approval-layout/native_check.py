"""Native Ask-gated fs_read through the real Console and local executor."""

import asyncio
import hashlib
import json
import os
import runpy
import socket
import subprocess
import sys
import traceback
from pathlib import Path

WORKER_TIMEOUT_SECONDS = 8


def main() -> None:
    """Run ROOT TMUX_SOCKET SESSION in an existing native tmux session.

    ROOT is an unused, prepared private profile under /tmp. Shared validation
    checks arguments, profile paths and tmux before app imports or output writes.
    The journey records keyboard bulk choices, actual controller submission,
    painted controls and normal shutdown in native.log and evidence/.

    Raises:
        SystemExit: Status 0 after success, 1 after a journey or application
            failure, or 2 for invalid command-line input or profile paths.
    """
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    sys.path.insert(0, str(repo))
    args = runpy.run_path(str(here.parent / "native_runner_args.py"))[
        "parse_native_args"
    ]()
    root, tmux_socket, session = args.root, args.tmux_socket, args.session
    tmux_path = args.tmux_path
    os.environ.update(
        HOME=str(root / "home"),
        USERPROFILE=str(root / "home"),
        TLDW_CONFIG_PATH=str(root / "config.toml"),
        XDG_DATA_HOME=str(root / "data"),
        XDG_CONFIG_HOME=str(root / "config"),
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
    )
    os.environ.pop("NO_COLOR", None)
    attempts = []
    original_connect = socket.socket.connect

    def guard_connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            attempts.append("network connect")
            raise RuntimeError("Network is disabled in this disposable UI journey")
        return original_connect(sock, address)

    socket.socket.connect = guard_connect
    from loguru import logger
    from textual.widgets import Button

    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.Utils.app_shutdown import claim_process_exit
    from tldw_chatbook.Utils.terminal_utils import warm_up_image_protocol
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard

    module_origins = {
        name: str(Path(sys.modules[name].__file__).resolve())
        for name in (
            "tldw_chatbook",
            "tldw_chatbook.app",
            "tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card",
            "tldw_chatbook.Utils.input_validation",
            "tldw_chatbook.Utils.path_validation",
        )
    }
    assert all(Path(path).is_relative_to(repo) for path in module_origins.values())
    logger.remove()
    logger.add(root / "native.log", level="INFO")
    evidence = root / "evidence"
    evidence.mkdir(exist_ok=False)
    (root / "launch.json").write_text(json.dumps({"pid": os.getpid()}))
    claim_process_exit()
    warm_up_image_protocol()
    app = TldwCli()
    from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console

    _configure_native_ready_console(app)
    sources = [
        "tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py",
        "tldw_chatbook/Chat/console_chat_controller.py",
        "tldw_chatbook/UI/Screens/chat_screen.py",
        "tldw_chatbook/css/tldw_cli_modular.tcss",
        "tldw_chatbook/css/widget_defaults_scoped.tcss",
        "tldw_chatbook/css/widget_defaults_self.tcss",
        "tldw_chatbook/css/core/_variables.tcss",
        "Docs/superpowers/qa/native_runner_args.py",
    ]
    result = {
        "pid": os.getpid(),
        "cells": [],
        "module_origins": module_origins,
        "source_hashes": {
            p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in sources
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture_scope": "Real Ask-gated fs_read via the production controller-composed LocalToolProvider, real private permission store, native Console decisions and production local executor. The disposable project root is supplied explicitly at the provider-composition seam; no model turn, external MCP server or network request is simulated as successful.",
    }

    def record():
        (evidence / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    async def tmux(*args):
        return await asyncio.to_thread(
            subprocess.run,
            [tmux_path, "-L", tmux_socket, *args],
            check=True,
            text=True,
            capture_output=True,
        )

    async def journey(pilot):
        worker = None
        controller = None

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

        async def focus(selector):
            control = app.screen.query_one(selector)
            control.focus()
            await settle()
            assert control.has_focus
            region, clip = app.screen._compositor.visible_widgets[control]
            assert region.width > 0 and region.height > 0, (selector, region)
            assert region.intersection(clip) == region, (selector, region, clip)
            if isinstance(control, Button):
                hit, _ = app.screen.get_widget_at(
                    region.x + region.width // 2, region.y + region.height // 2
                )
                assert hit is control, (selector, hit)
                painted = "\n".join(
                    strip.crop(region.x, region.right).text
                    for strip in app.screen._compositor.render_strips()[
                        region.y : region.bottom
                    ]
                )
                assert control.label.plain in painted, (selector, painted)
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
            await app.handle_screen_navigation(NavigateToScreen("chat"))
            await wait_for(
                lambda: getattr(app.screen, "screen_name", None) == "chat", "Console"
            )
            controller = app.screen._ensure_console_chat_controller()
            controller_module = type(controller).__module__
            controller_path = Path(sys.modules[controller_module].__file__).resolve()
            assert controller_path.is_relative_to(repo)
            module_origins[controller_module] = str(controller_path)
            chat = controller.new_session(
                title="Approval review fixture", ephemeral=True
            )
            await app.screen._sync_native_console_chat_ui()
            await settle()
            assert app.screen._console_chat_store is controller.store
            assert app.screen._console_chat_store.active_session_id == chat.id
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "f.txt").write_text("native-tool-fixture")
            for width, height, inspect_open in (
                (80, 24, True),
                (80, 24, False),
                (235, 52, True),
            ):
                await tmux(
                    "resize-window", "-t", session, "-x", str(width), "-y", str(height)
                )
                await wait_for(
                    lambda width=width, height=height: app.size == (width, height),
                    "resize",
                )
                app.screen._set_console_rail_preference(
                    left_open=False, right_open=inspect_open
                )
                await settle()
                for mode in ("fast-deny", "deny-all", "approve-once"):
                    provider, _ = controller._compose_local_provider(
                        session_id=chat.id, project_root=workspace
                    )
                    assert provider is not None
                    hub = provider.hub_tool_for("fs_read")
                    app.unified_mcp_service.set_tool_state(
                        hub.server_key, hub.name, "ask", tool=hub
                    )
                    assert app.unified_mcp_service.gate_tool_test(hub).state == "ask"
                    worker = asyncio.create_task(
                        asyncio.to_thread(
                            provider.invoke, "local:fs_read", {"path": "f.txt"}
                        )
                    )
                    card = app.screen.query_one(ChatApprovalCard)
                    await wait_for(
                        lambda card=card: (
                            card.display and len(card._batch_selects) == 1
                        ),
                        "real Ask-gated fs_read card",
                    )
                    round_id = card._batch_round_id
                    assert round_id in controller._pending_approval_rounds
                    app.screen._sync_console_mode_bar()
                    app.screen._sync_console_rail_and_controls()
                    await settle()
                    assert app.screen._console_pending_approval_count() == 1
                    stem = f"{width}x{height}-inspect-{inspect_open}-{mode}"
                    for button in card.query(Button):
                        await focus("#" + button.id)
                    await capture(stem + "-pending")
                    if mode == "deny-all":
                        await focus("#approval-deny-all")
                        await pilot.press("enter")
                        await settle()
                        assert card._batch_selects[0].value == "deny"
                        assert not worker.done()
                        await focus("#approval-submit")
                    elif mode == "fast-deny":
                        await focus("#" + card.query_one(".approval-row-fast-deny").id)
                    else:
                        await focus(
                            "#" + card.query_one(".approval-row-fast-approve").id
                        )
                    await capture(stem + "-decision")
                    await pilot.press("enter")
                    tool_result = await asyncio.wait_for(worker, WORKER_TIMEOUT_SECONDS)
                    worker = None
                    if mode == "approve-once":
                        assert (
                            tool_result.ok
                            and "native-tool-fixture" in tool_result.content
                        ), tool_result
                    else:
                        assert (
                            not tool_result.ok
                            and "denied by the user" in tool_result.error
                        ), tool_result
                    await wait_for(
                        lambda card=card: not card.display, "approval cleared"
                    )
                    assert round_id not in controller._pending_approval_rounds
                    assert not attempts, attempts
                    result["cells"].append(
                        {
                            "size": [width, height],
                            "inspect_open": inspect_open,
                            "mode": mode,
                            "painted_controls": True,
                            "real_ask_gate": True,
                            "real_local_executor_on_approval": tool_result.ok,
                            "result_ok": tool_result.ok,
                            "error": tool_result.error,
                        }
                    )
                    record()
            result["passed"] = True
        except Exception:  # noqa: BLE001 - preserve evidence and shut down the native app
            result.update(passed=False, error=traceback.format_exc())
            app.save_screenshot("failed-state.svg", path=str(evidence))
        finally:
            result["network_attempts"] = attempts
            record()
            if worker is not None and not worker.done() and controller is not None:
                for round_id in tuple(controller._pending_approval_rounds):
                    controller.resolve_pending_approval({}, round_id=round_id)
                await asyncio.wait_for(worker, WORKER_TIMEOUT_SECONDS)
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
