"""Native keyboard approval round trips with the real controller, no tool dispatch."""

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

    from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
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
        "fixture_scope": "Synthetic pending-call metadata enters the real controller request/resolve round trip. Returned decisions are checked and never passed to a provider or tool executor.",
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
            await settle()
            for theme in ("textual-dark", "textual-light"):
                for width, height in ((120, 40), (170, 48)):
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
                    pending = [
                        MCPPendingCall(
                            llm_name=f"mcp__fixture__{name}",
                            server_key="local:fixture",
                            tool_name=name,
                            server_label="Disposable review fixture",
                            arguments={"query": f"Review {name}"},
                            reason="ask",
                            call_id=name,
                        )
                        for name in ("search", "lookup")
                    ]
                    worker = asyncio.create_task(
                        asyncio.to_thread(
                            controller.request_mcp_approvals,
                            pending,
                            session_id=chat.id,
                        )
                    )
                    card = app.screen.query_one(ChatApprovalCard)
                    await wait_for(
                        lambda card=card: (
                            card.display and len(card._batch_selects) == 2
                        ),
                        "approval card",
                    )
                    round_id = card._batch_round_id
                    assert round_id in controller._pending_approval_rounds
                    card.focus_first_decision()
                    await settle()
                    assert app.focused in card._batch_selects
                    await focus("#approval-deny-all")
                    await pilot.press("enter")
                    await settle()
                    assert all(s.value == "deny" for s in card._batch_selects)
                    assert not worker.done()
                    stem = f"{theme}-{width}x{height}"
                    await capture(stem + "-deny-all")
                    await focus("#approval-approve-all")
                    await pilot.press("enter")
                    await settle()
                    assert all(s.value == "approve_once" for s in card._batch_selects)
                    assert not worker.done()
                    await focus("#approval-submit")
                    await capture(stem + "-ready-to-submit")
                    await pilot.press("enter")
                    decisions = await asyncio.wait_for(worker, 8)
                    worker = None
                    assert decisions == {
                        "search": "approve_once",
                        "lookup": "approve_once",
                    }, decisions
                    await wait_for(
                        lambda card=card: not card.display, "approval cleared"
                    )
                    assert round_id not in controller._pending_approval_rounds
                    assert not attempts, attempts
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": [width, height],
                            "bulk_only_edits_choices": True,
                            "real_controller_round_trip": True,
                            "decisions": decisions,
                            "tool_dispatches": 0,
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
                await asyncio.wait_for(worker, 8)
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
