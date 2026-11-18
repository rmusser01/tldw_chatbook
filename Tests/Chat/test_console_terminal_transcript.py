"""A settled original turn delivers resident rows independently of FULL."""

import asyncio
import io
import sys
import threading
from contextlib import ExitStack
from types import CodeType

import pytest

from Tests.Chat.test_console_execution_backend_process import accepted_process_provider  # noqa: F401
from Tests.private_profile import private_profile_test


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("held", ["queued", "native"])
async def test_retired_original_turn_transcript_does_not_wait_for_full(
    request, accepted_process_provider, held,  # noqa: F811
):
    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Chat.test_console_execution_backend_process import _ACCEPTED_REPLY
    from Tests.Chat.test_console_execution_profile import _settled_profile
    from tldw_chatbook import config
    from tldw_chatbook.Chat.console_execution_backend import ConsoleExecutionBackend
    from tldw_chatbook.Chat.console_execution_profile import open_console_backend_profile
    from tldw_chatbook.Chat.console_execution_protocol import (
        ConsoleWireMessage, WireIdentity, decode_message, encode_message,
    )
    from tldw_chatbook.Chat.console_execution_worker import _BackendChannel, _BackendService
    from tldw_chatbook.Chat.console_display_phases import ConsoleDisplayPhysicalOwner
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings

    assert config.save_settings_to_cli_config({
        "api_settings.openai": {
            "api_key": "task62-local-readiness-only",
            "api_base": accepted_process_provider.base_url,
        },
    })
    path, directory, *_ = _settled_profile()
    profile = open_console_backend_profile(
        expected_config_path=path, expected_data_directory=directory,
    )
    backend = ConsoleExecutionBackend(profile, profile_epoch="profile", service_epoch="terminal")
    tasks, frames = [], []
    locked = False
    entered, release = threading.Event(), threading.Event()
    observations = ExitStack()

    def collect():
        while not channel.outbox.empty():
            frames.append(decode_message(channel.outbox.get_nowait()))

    def start(operation, payload):
        sequence = backend._sequence + 1
        message = ConsoleWireMessage(WireIdentity(
            "profile", "terminal", sequence, f"terminal-{sequence}",
            session.id, session.incarnation_id, backend.controls.source_revision,
        ), "command", operation, payload)
        task = service.accept(message)
        if task is not None:
            tasks.append(task)
        return message, task

    async def finish(message, task):
        if task is not None:
            await asyncio.shield(task)
        collect()
        return [item for item in frames if item.identity.request_id == message.identity.request_id]

    async def command(operation, payload):
        return await finish(*start(operation, payload))

    def applied(result):
        assert [item.kind for item in result] == ["received", "applied", "retired"]
        assert result[-1].payload["retired"] is True
        return encode_message(result[1])["payload"]

    try:
        backend.start()
        session = backend.store.create_session(
            title="Retired resident transcript", assistant_kind="generic",
            settings=ConsoleSessionSettings(
                provider="openai", model="gpt-4o-mini", source="user",
                base_url=accepted_process_provider.base_url, streaming=True,
            ),
        )
        channel = _BackendChannel(io.BytesIO(), io.BytesIO())
        service = _BackendService(backend, channel)
        applied(await command("PermissionAction", {"operation": {"kind": "SetKillSwitch", "value": False}}))
        first = applied(await command("ReadProjection", {"view_checkpoint": {
            "attachment_id": "actual-client", "visit": 1, "action": "mount",
        }}))
        pressed = first["session_view"]["pressed_inputs"]
        applied(await command("UpdateInputs", {
            "draft": "Original terminal transcript", "authored_token": [1, 1],
            "expected_settings_revision": pressed["settings_revision"],
            "expected_settings": pressed["settings"],
            "expected_attachment_ids": pressed["attachment_ids"],
            "remove_attachment_ids": [], "prefill": None,
        }))
        ready = applied(await command("ReadProjection", {}))
        pressed = ready["session_view"]["pressed_inputs"]
        row = next(item for item in ready["session_view"]["sessions"] if item["id"] == session.id)
        turn, work = start("ReceiveTurn", {
            "turn_id": "original-terminal-turn", "draft_revision": row["draft_revision"],
            "expected_settings_revision": pressed["settings_revision"],
            "expected_attachment_revision": pressed["attachment_revision"],
            "expected_prefill_revision": pressed["prefill_revision"],
            "pressed_inputs": pressed,
            "selection": {"source": "backend", "attachment_id": "actual-client", "visit": 1, "rag_source_types": []},
        })
        original = backend.adapter._receipts[turn.identity.request_id][1]
        acknowledged = set()
        async with asyncio.timeout(30):
            while not work.done():
                collect()
                for item in tuple(frames):
                    if item.identity.request_id == turn.identity.request_id and item.kind == "commit_planned":
                        ordinal = item.payload["ordinal"]
                        if ordinal not in acknowledged:
                            acknowledged.add(ordinal)
                            applied(await command("CommitPlanAck", {
                                "received_request_id": turn.identity.request_id,
                                "turn_id": "original-terminal-turn", "ordinal": ordinal,
                                "fingerprint": encode_message(item)["payload"]["fingerprint"],
                            }))
                await asyncio.sleep(0.01)
        result = await finish(turn, work)
        assert await asyncio.shield(original.retired) is True
        assert any(item.kind == "terminal" and item.payload["accepted"] is True for item in result)
        assert len(accepted_process_provider.payloads) == 1
        assert session.persisted_conversation_id is not None
        persisted = backend.domain.sources.chat_database.get_messages_for_conversation(
            session.persisted_conversation_id, limit=3,
        )
        assert persisted[-1]["content"] == _ACCEPTED_REPLY

        # Hold either the original queue or an original native producer's return
        # before its executor Future settles. Never replace the checked function.
        if held == "queued":
            await backend.controls._projection_read_lock.acquire()
            locked = True
        else:
            invoke = next(code for code in ConsoleDisplayPhysicalOwner.native_phase.__code__.co_consts
                          if type(code) is CodeType and code.co_name == "invoke")

            def hold_native(frame):
                full = frame.f_locals.get("self")
                if getattr(getattr(full, "owner", None), "runtime", None) is backend.runtime and not entered.is_set():
                    entered.set()
                    assert release.wait(30), "Original FULL producer was not released"

            observations.enter_context(_observe(invoke, sys.monitoring.events.PY_RETURN, hold_native))
        full_message, full_task = start("ReadProjection", {})
        await asyncio.sleep(0)
        if held == "native":
            async with asyncio.timeout(10):
                while not entered.is_set():
                    await asyncio.sleep(0.01)
        assert not full_task.done()
        readiness_age = backend.presentation.readiness.at
        async with asyncio.timeout(10):
            resident = applied(await command("ReadTerminalTranscript", {
                "received_request_id": turn.identity.request_id,
                "turn_id": "original-terminal-turn", "attachment_id": "actual-client", "visit": 1,
            }))
        assert not full_task.done() and backend.controls._projection_read_lock.locked()
        assert backend.presentation.readiness.at == readiness_age
        assert resident["projection"]["messages"][-1]["content"] == _ACCEPTED_REPLY
        assert resident["session"]["persisted_conversation_id"] == session.persisted_conversation_id
        if locked:
            backend.controls._projection_read_lock.release()
            locked = False
        release.set()
        # The independent original read must still finish and retire normally.
        full = applied(await finish(full_message, full_task))
        assert full["messages"][-1]["content"] == _ACCEPTED_REPLY
    finally:
        release.set()
        if locked:
            backend.controls._projection_read_lock.release()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        observations.close()
        await backend.aclose()
