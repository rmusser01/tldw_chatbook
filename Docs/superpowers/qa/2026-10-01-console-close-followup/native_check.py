"""Native Close verification with real decision rounds and a deterministic owning task."""

import asyncio
import hashlib
import json
import os
import runpy
import socket
import subprocess
import sys
import threading
import traceback
from pathlib import Path


def main() -> None:
    """Run ROOT TMUX_SOCKET SESSION in an existing, disposable tmux pane."""
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    sys.path.insert(0, str(repo))
    # Select the profile before the shared argument validator imports app utilities.
    root = Path(sys.argv[1]).resolve()
    os.environ.update(
        HOME=str(root / "home"),
        USERPROFILE=str(root / "home"),
        TLDW_CONFIG_PATH=str(root / "config.toml"),
        XDG_DATA_HOME=str(root / "data"),
        XDG_CONFIG_HOME=str(root / "config"),
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
    )
    args = runpy.run_path(str(here.parent / "native_runner_args.py"))[
        "parse_native_args"
    ]()
    root, tmux_socket, session = args.root, args.tmux_socket, args.session
    attempts = []
    original_connect = socket.socket.connect

    def guard_connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            attempts.append("network connect")
            raise RuntimeError("Network is disabled in this disposable UI journey")
        return original_connect(sock, address)

    socket.socket.connect = guard_connect
    os.environ.pop("NO_COLOR", None)
    from loguru import logger
    from textual.widgets import Button

    from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
    from tldw_chatbook.Agents.run_context import use_run_id
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleLifecycleImpact,
        ConsoleMessageRole,
        ConsoleRunState,
        ConsoleRunStatus,
    )
    from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionCloseImpact
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.Utils.app_shutdown import claim_process_exit
    from tldw_chatbook.Utils.terminal_utils import warm_up_image_protocol
    from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

    logger.remove()
    logger.add(root / "native.log", level="INFO")
    evidence = root / "evidence"
    evidence.mkdir()
    (root / "launch.json").write_text(json.dumps({"pid": os.getpid()}))
    claim_process_exit()
    warm_up_image_protocol()
    app = TldwCli()
    sources = (
        "tldw_chatbook/UI/Console_Modules/session.py",
        "tldw_chatbook/Chat/console_chat_controller.py",
        "tldw_chatbook/UI/Screens/chat_screen.py",
        "tldw_chatbook/Widgets/confirmation_dialog.py",
        "tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py",
        "tldw_chatbook/css/tldw_cli_modular.tcss",
        "Docs/superpowers/qa/native_runner_args.py",
    )
    result = {
        "pid": os.getpid(),
        "cells": [],
        "module_origins": {
            name: str(Path(sys.modules[name].__file__).resolve())
            for name in ("tldw_chatbook", "tldw_chatbook.app")
        },
        "source_hashes": {
            p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in sources
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture_scope": (
            "Actual TldwCli and LinuxDriver. Real request_user_questions and "
            "request_mcp_approvals worker rounds; deterministic registered owning "
            "asyncio task, no provider, tool dispatch or external server. Actual "
            "ConsoleRuntime closes the requested background session. Terminal "
            "SGR mouse input presses both tab close and confirmation buttons."
        ),
    }

    def record():
        (evidence / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    async def tmux(*arguments):
        return await asyncio.to_thread(
            subprocess.run,
            [args.tmux_path, "-L", tmux_socket, *arguments],
            check=True,
            text=True,
            capture_output=True,
        )

    async def journey(pilot):
        workers = []
        tasks = []
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

        def painted(control):
            region, clip = app.screen._compositor.visible_widgets[control]
            assert region.width > 0 and region.height > 0
            assert region.intersection(clip) == region, (control.id, region, clip)
            return "\n".join(
                strip.crop(region.x, region.right).text
                for strip in app.screen._compositor.render_strips()[
                    region.y : region.bottom
                ]
            )

        async def click(selector):
            await wait_for(
                lambda: not app._notifications and not app.screen.query("Toast"),
                "notices clear before click",
            )
            await wait_for(lambda: bool(app.screen.query(selector)), selector)
            button = app.screen.query_one(selector, Button)
            button.scroll_visible(animate=False)
            await settle()
            assert not button.disabled
            assert button.label.plain in painted(button), selector
            region = button.region
            x, y = region.x + region.width // 2, region.y + region.height // 2
            hit, _ = app.screen.get_widget_at(x, y)
            assert hit is button, (selector, hit)
            # Both edges pass through tmux, the native driver's input parser,
            # and the normal Textual button handler.
            await tmux(
                "send-keys",
                "-t",
                session,
                "-l",
                f"\x1b[<0;{x + 1};{y + 1}M\x1b[<0;{x + 1};{y + 1}m",
            )

        async def capture(stem):
            await wait_for(lambda: not app._notifications, "notices clear")
            app.save_screenshot(stem + ".svg", path=str(evidence))
            pane = await tmux("capture-pane", "-p", "-t", session)
            (evidence / (stem + ".txt")).write_text(pane.stdout)
            return pane.stdout

        def arm(kind, session_id):
            def request():
                if kind == "approval":
                    return controller.request_mcp_approvals(
                        [
                            MCPPendingCall(
                                llm_name="close_fixture",
                                server_key="local:fixture",
                                tool_name="search",
                                server_label="Disposable close fixture",
                                arguments={"query": "private close payload"},
                                reason="ask",
                                call_id="close-call",
                            )
                        ],
                        session_id=session_id,
                    )
                return controller.request_user_questions(
                    [
                        {
                            "header": "Choice",
                            "question": "Private choice?",
                            "options": [
                                {"label": "One", "description": "First choice"},
                                {"label": "Two", "description": "Second choice"},
                            ],
                        }
                    ],
                    session_id=session_id,
                )

            run_id = f"native-close-{kind}-{session_id}-{len(workers)}"
            with use_run_id(run_id):
                worker = asyncio.create_task(asyncio.to_thread(request))
            workers.append((kind, session_id, worker, run_id))
            return worker

        def own_run(session_id, worker):
            assistant = controller.store.append_message(
                session_id, role=ConsoleMessageRole.ASSISTANT, content=""
            )
            controller._active_cancel_events[session_id] = threading.Event()

            async def waiting_run():
                controller._active_stream_tasks[session_id] = asyncio.current_task()
                controller._active_assistant_message_ids[session_id] = assistant.id
                controller._set_run_state(
                    ConsoleRunState(ConsoleRunStatus.STREAMING, "Waiting"),
                    session_id=session_id,
                )
                try:
                    await asyncio.shield(worker)
                    await asyncio.Event().wait()
                finally:
                    controller._active_stream_tasks.pop(session_id, None)
                    controller._active_assistant_message_ids.pop(session_id, None)
                    controller._active_cancel_events.pop(session_id, None)

            task = asyncio.create_task(waiting_run())
            tasks.append(task)
            return task

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
            console = app.screen
            controller = console._ensure_console_chat_controller()
            path = Path(sys.modules[type(controller).__module__].__file__).resolve()
            assert path.is_relative_to(repo)
            result["module_origins"][type(controller).__module__] = str(path)
            keeper = controller.store.active_session_id
            controller.store.rename_session(keeper, "Viewed question")
            for width, height in ((235, 52), (80, 24)):
                await tmux(
                    "resize-window", "-t", session, "-x", str(width), "-y", str(height)
                )
                await wait_for(
                    lambda width=width, height=height: app.size == (width, height),
                    "resize",
                )
                for kind in ("approval", "question"):
                    chat = controller.new_session(
                        title="Pending [notes]", ephemeral=True
                    )
                    await settle()
                    controller.switch_session(keeper)
                    await settle()
                    sibling = arm("question", keeper)
                    sibling_run_id = workers[-1][3]
                    await wait_for(
                        lambda: "question" in controller.pending_round_kinds(keeper),
                        "viewed question",
                    )
                    worker = arm(kind, chat.id)
                    owner = own_run(chat.id, worker)
                    await wait_for(
                        lambda kind=kind, sid=chat.id: (
                            kind in controller.pending_round_kinds(sid)
                            and sid in controller._active_stream_tasks
                        ),
                        "background round and owning task",
                    )
                    await console._sync_native_console_chat_ui()
                    await click(f"#console-close-session-tab-{chat.id}")
                    await wait_for(
                        lambda: isinstance(app.screen, ConfirmationDialog),
                        "close dialog",
                    )
                    dialog = app.screen
                    await wait_for(
                        lambda dialog=dialog: (
                            dialog.query_one("#cancel-button").has_focus
                        ),
                        "default Stay focus",
                    )
                    assert 'Close tab "Pending [notes]"?' == dialog.title
                    category = (
                        "Tool approvals: denied"
                        if kind == "approval"
                        else "Questions: cancelled"
                    )
                    assert category in dialog.message
                    assert "private close payload" not in dialog.message
                    for absent in (
                        "Unsent draft:",
                        "Pending attachments:",
                        "Delegated agents:",
                        "Unsent queued prompts:",
                    ):
                        assert absent not in dialog.message
                    stem = f"dark-{width}x{height}-{kind}"
                    frame = await capture(stem + "-confirm")
                    assert "Pending [notes]" in frame and category in frame
                    assert "Stay" in painted(dialog.query_one("#cancel-button"))
                    assert "Close" in painted(dialog.query_one("#confirm-button"))
                    await click("#confirm-button")
                    await wait_for(
                        lambda sid=chat.id: (
                            sid not in {s.id for s in controller.store.sessions()}
                        ),
                        "background tab gone",
                    )
                    await wait_for(
                        lambda worker=worker, owner=owner: (
                            worker.done() and owner.done()
                        ),
                        "owned work terminated",
                    )
                    expected = (
                        {"close-call": "deny"}
                        if kind == "approval"
                        else {"answered": False, "reason": "cancelled"}
                    )
                    actual = await worker
                    assert actual == expected, actual
                    assert owner.cancelled()
                    assert not controller.pending_round_kinds(chat.id)
                    assert not controller._interrupt_host.session_round_payloads(
                        kind, chat.id
                    )
                    assert (
                        controller.store.active_session_id == keeper
                        and not sibling.done()
                    )
                    assert controller.pending_round_kinds(keeper) == {"question"}
                    await capture(stem + "-closed")
                    result["cells"].append(
                        {
                            "size": [width, height],
                            "kind": kind,
                            "named_dialog_painted": True,
                            "default_stay": True,
                            "zero_consequences_absent": True,
                            "terminal_mouse_close": True,
                            "worker_result": actual,
                            "owning_task_cancelled": True,
                            "target_rounds_removed": True,
                            "viewed_sibling_question_pending": True,
                        }
                    )
                    record()
                    controller.revoke_approval_rounds_for_run(sibling_run_id)
                    await wait_for(sibling.done, "release fixture sibling")
                    assert await sibling == {"answered": False, "reason": "cancelled"}
            # Geometry-only snapshot: all possible consequences, sanitizer-cap title.
            chat = controller.new_session(title="A" * 60, ephemeral=True)
            await settle()
            controller.switch_session(keeper)
            await settle()
            impact = ConsoleSessionCloseImpact(
                session_id=chat.id,
                transcript_message_count=1,
                lifecycle=ConsoleLifecycleImpact(
                    revision=1,
                    live_run_count=1,
                    queued_session_count=1,
                    unsent_prompt_count=1,
                    delegated_child_count=1,
                ),
                has_draft=True,
                pending_attachment_count=1,
                pending_round_kinds=frozenset(
                    {"approval", "question", "skill_install", "skill_script"}
                ),
            )
            dialog_worker = console.run_worker(
                console._session._confirm_session_close(impact), exit_on_error=False
            )
            await wait_for(
                lambda: isinstance(app.screen, ConfirmationDialog), "max-risk dialog"
            )
            dialog = app.screen
            try:
                await wait_for(
                    lambda: dialog.query_one("#cancel-button").has_focus,
                    "max-risk Stay focus",
                )
                container = dialog.query_one("#confirmation-dialog")
                assert container.region.intersection(dialog.region) == container.region
                for selector in (".dialog-title", "#cancel-button", "#confirm-button"):
                    control = dialog.query_one(selector)
                    assert control.region.intersection(dialog.region) == control.region
                    assert painted(control)
                await capture("dark-80x24-all-consequences-long-title")
                result["max_risk_dialog"] = {
                    "size": [80, 24],
                    "title_characters": 60,
                    "all_six_nonzero_counts": True,
                    "all_four_pending_kinds": True,
                    "title_and_actions_fully_painted": True,
                    "default_stay": True,
                    "fixture_scope": "Synthetic impact snapshot exercises only real confirmation geometry; not real work in these six categories.",
                }
            finally:
                dialog.dismiss(False)
                assert await dialog_worker.wait() is False
            assert not attempts, attempts
            result["passed"] = True
        except Exception:  # noqa: BLE001 -- keep failure evidence before shutdown
            result.update(passed=False, error=traceback.format_exc())
            app.save_screenshot("failed-state.svg", path=str(evidence))
            pane = await tmux("capture-pane", "-p", "-t", session)
            (evidence / "failed-state.txt").write_text(pane.stdout)
        finally:
            if controller is not None:
                for kind, session_id, worker, run_id in workers:
                    controller.revoke_approval_rounds_for_run(run_id)
                    controller._cancel_pending_decisions_for_session(session_id)
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                await asyncio.wait_for(
                    asyncio.gather(*(w for _, _, w, _ in workers)), 8
                )
            result["network_attempts"] = attempts
            record()
            await tmux("send-keys", "-t", session, "C-q")

    app.run(auto_pilot=journey, size=(235, 52))
    result.update(
        app_run_returned=True,
        app_return_code=app.return_code,
        app_exception=type(app._exception).__name__
        if app._exception is not None
        else None,
    )
    if app.return_code != 0 or app._exception is not None:
        result["passed"] = False
    result["source_hashes_after"] = {
        p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in sources
    }
    record()
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
