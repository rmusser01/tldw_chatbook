"""Actual selected composer, original ACK flow, provider and durable outcome."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time

import pytest

from Tests.Chat.test_console_execution_backend_process import accepted_process_provider  # noqa: F401 -- pytest fixture
from Tests.private_profile import private_profile_test


@private_profile_test
@pytest.mark.asyncio
async def test_selected_original_enter_reaches_provider_and_durable_terminal(
    request, accepted_process_provider  # noqa: F811 -- imported pytest fixture
):
    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Chat.test_console_execution_backend_process import (
        _ACCEPTED_REPLY,
        _ObservedStderr,
    )
    from Tests.Chat.console_execution_mount_wait import wait_for_selected_composer
    from tldw_chatbook import config
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Chat.console_execution_process import ConsoleExecutionClient
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.Constants import TAB_CHAT
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

    assert config.save_settings_to_cli_config(
        {
            "first_run": {"setup_completed": True},
            "splash_screen": {"enabled": False},
            "general": {"default_tab": "home"},
            "chat_defaults": {"provider": "openai", "model": "gpt-4o-mini"},
            "api_settings.openai": {
                "api_key": "task62-local-readiness-only",
                "api_base": accepted_process_provider.base_url,
            },
        }
    )
    parent_runtime_entries, received = [], []
    tail = bytearray()

    def runtime_entry(frame):
        parent_runtime_entries.append(frame.f_locals["self"])

    def submit_return(frame, values):
        if (
            frame.f_locals.get("self") is client
            and frame.f_locals.get("operation") == "ReceiveTurn"
        ):
            # Observe the actual result even when submit delegates to its
            # transport owner without storing a receipt in its own frame.
            received.append(values[-1])

    def created(frame):
        caller = frame.f_back
        if (
            caller.f_code is ConsoleExecutionClient._launch.__code__
            and caller.f_locals.get("self") is client
        ):
            process = frame.f_locals["self"]
            assert type(process) is subprocess.Popen and process.stderr is not None
            process.stderr = _ObservedStderr(process.stderr, tail)

    try:
        with _observe(
            ConsoleRuntime.__init__.__code__,
            sys.monitoring.events.PY_START,
            runtime_entry,
        ):
            app = TldwCli(console_execution="process")
            client = app.console_execution_client
            with _observe(
                subprocess.Popen.__init__.__code__,
                sys.monitoring.events.PY_RETURN,
                created,
            ):
                async with app.run_test(size=(120, 40)) as pilot:
                    await app._initial_screen_setup_task
                    await app._console_execution_start_task
                    async with asyncio.timeout(10):
                        await app.handle_screen_navigation(NavigateToScreen(TAB_CHAT))
                        await wait_for_selected_composer(app)
                    screen = app.screen
                    assert type(screen) is ChatScreen and app.console_runtime is None
                    held_store = screen._console_chat_store
                    held_session = held_store.session(held_store.active_session_id)
                    held_controller = screen._console_chat_controller
                    composer = screen.query_one(
                        "#console-native-composer", ConsoleComposerBar
                    )
                    permission = client.submit(
                        "PermissionAction",
                        {"operation": {"kind": "SetKillSwitch", "value": False}},
                    )
                    await asyncio.shield(permission.applied)
                    assert await asyncio.shield(permission.retired) is True
                    composer.focus_draft_from(None)
                    await pilot.press(*("space" if character == " " else character for character in "original selected enter"))
                    assert composer.draft_text() == "original selected enter"
                    started = time.monotonic()
                    with _observe(
                        ConsoleExecutionClient.submit.__code__,
                        sys.monitoring.events.PY_RETURN,
                        submit_return,
                        values=True,
                    ):
                        try:
                            await pilot.press("enter")
                            async with asyncio.timeout(10):
                                while not received:
                                    await pilot.pause()
                                receipt = received[0]
                                assert await asyncio.shield(receipt.retired) is True
                                while composer.draft_text():
                                    await pilot.pause()
                        except BaseException:
                            # Snapshot before App teardown can settle/cancel work.
                            # Existing receipts and server clocks only: no added
                            # tracing, callbacks, payloads, or successful-path I/O.
                            operations = {
                                "UpdateInputs", "ReceiveTurn", "ReadProjection",
                                "CommitPlanAck", "ArchiveSendCheckpointAck",
                            }
                            events = {
                                "commit_planned", "input_projected", "terminal",
                                "failed", "archive_checkpoint",
                            }
                            pending = tuple(client._requests.values())
                            receipts = received + [
                                item for item in pending
                                if item.operation in operations
                                and all(item is not turn for turn in received)
                            ]
                            diagnostic = {
                                "elapsed_seconds": time.monotonic() - started,
                                "receive_turn_count": len(received),
                                "provider_post_seconds": [
                                    stamp - started for stamp in
                                    accepted_process_provider.post_received_at
                                ],
                                "provider_reply_seconds": [
                                    stamp - started for stamp in
                                    accepted_process_provider.reply_sent_at
                                ],
                                "projection_lock_held": client._projection_lock.locked(),
                                "projection_tasks": len(client._projection_reads),
                                "draft_present": bool(composer.draft_text()),
                                "receipts": [
                                    {
                                        "operation": item.operation,
                                        "futures": {
                                            name: {
                                                "done": getattr(item, name).done(),
                                                "cancelled": getattr(item, name).cancelled(),
                                            }
                                            for name in ("received", "applied", "retired")
                                        },
                                        "events": [
                                            event.kind for event in item.events
                                            if event.kind in events
                                        ],
                                    }
                                    for item in receipts
                                ],
                            }
                            encoded = json.dumps(diagnostic, sort_keys=True)
                            request.node.user_properties.append(
                                ("selected_enter_failure_state", encoded)
                            )
                            print("Selected Enter failure state:", encoded, file=sys.stderr)
                            raise
                    request.node.user_properties.append(
                        (
                            "selected_enter_to_retired_seconds",
                            time.monotonic() - started,
                        )
                    )
                    assert len(received) == 1 and not parent_runtime_entries
                    terminal = [
                        item for item in receipt.events if item.kind == "terminal"
                    ]
                    projected = [
                        item
                        for item in receipt.events
                        if item.kind == "input_projected"
                    ]
                    assert (
                        len(terminal) == 1 and terminal[0].payload["accepted"] is True
                    )
                    assert len(projected) == 1
                    assert receipt.potentially_committed
                    assert len(accepted_process_provider.payloads) == 1
                    row = client.projection_view.session(receipt.identity.session_id)
                    await client.read_projection(row.id, row.incarnation_id)
                    row = client.projection_view.session(row.id)
                    assert row.persisted_conversation_id
                    assert screen._console_chat_store is held_store
                    assert held_controller.store is held_store
                    assert held_session.persisted_conversation_id == row.persisted_conversation_id
                    run_rows = await asyncio.to_thread(
                        held_controller._agent_bridge._db.list_runs,
                        held_session.persisted_conversation_id,
                    )
                    assert len(run_rows) == 1 and run_rows[0]["status"] == "done"
                    assert run_rows[0]["conversation_id"] == row.persisted_conversation_id
                    # Independent original repository observation; this is not a
                    # mirrored controller or a decoded transcript persistence claim.
                    rows = app.chachanotes_db.get_messages_for_conversation(
                        row.persisted_conversation_id, limit=3
                    )
                    assert [(item["role"], item["content"]) for item in rows] == [
                        ("user", "original selected enter"),
                        ("assistant", _ACCEPTED_REPLY),
                    ]
                    assert app.console_runtime is None
                assert client._disposed and client._physically_closed
                assert client._process.returncode == 0
    finally:
        if tail:
            print(
                "Selected Send backend tail:",
                tail.decode("utf-8", errors="replace"),
                file=sys.stderr,
            )
