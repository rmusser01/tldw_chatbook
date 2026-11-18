"""Actual parent archive coordination around a contained backend's saved Send."""

from __future__ import annotations

import asyncio
import sys
import threading
from dataclasses import replace

import pytest

from Tests.Chat.test_console_execution_backend_process import accepted_process_provider  # noqa: F401 -- pytest fixture
from Tests.private_profile import private_profile_test


@private_profile_test
@pytest.mark.asyncio
async def test_saved_process_send_keeps_original_archive_reservation_and_generation(
    request, accepted_process_provider,  # noqa: F811 -- imported pytest fixture
):
    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Chat.test_console_first_send_atomicity import _until
    from tldw_chatbook import config
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Chat import conversation_archive_actions as archive
    from tldw_chatbook.Chat.console_execution_archive import receive_archive_checkpoint
    from tldw_chatbook.Chat.console_execution_process import ConsoleExecutionClient
    from tldw_chatbook.Chat.console_execution_protocol import ConsoleProtocolError, _detach
    from tldw_chatbook.MCP.permission_process import write_message

    assert config.save_settings_to_cli_config({
        "first_run": {"setup_completed": True},
        "splash_screen": {"enabled": False},
        "general": {"default_tab": "home"},
        "chat_defaults": {"provider": "openai", "model": "gpt-4o-mini"},
        "api_settings.openai": {
            "api_key": "task62-local-readiness-only",
            "api_base": accepted_process_provider.base_url,
        },
    })
    app = TldwCli(console_execution="process")
    client = app.console_execution_client
    held_ack, release_ack = threading.Event(), threading.Event()
    held_archive, release_archive = threading.Event(), threading.Event()
    expired = threading.Event()
    checkpoints, sent_acks, custody = [], [], []
    pending_archive = None

    def received(frame):
        if frame.f_locals["self"] is not client:
            return
        message = frame.f_locals["message"]
        if message.operation != "ReceiveTurn":
            return
        if message.kind == "archive_checkpoint":
            checkpoints.append(message)
        if message.kind in {"terminal", "retired"}:
            record = client._archive_sends[message.identity.request_id]
            assert record.reservations is app._conversation_send_inflight
            assert record.reservations[record.conversation_id] == 1
            # Observation is before the original receiver processes the real
            # terminal/retirement frame; neither event alone releases custody.
            custody.append((record.receipt.turn_id, message.kind))

    def writing(frame):
        value = frame.f_locals["value"]
        if (frame.f_locals["stream"] is not client._channel.output
            or value.get("operation") != "ArchiveSendCheckpointAck"):
            return
        payload = value["payload"]
        sent_acks.append(dict(payload))
        if payload["turn_id"] == "archive-generation" and payload["ordinal"] == 1:
            held_ack.set()
            if not release_ack.wait(10):
                expired.set()

    async def joined(receipt):
        result = await asyncio.shield(receipt.applied)
        assert await asyncio.shield(receipt.retired) is True
        return result

    def snapshot(value, name):
        return _detach(_detach(value, name, 0, set()), name, 0, set(), thaw=True)

    async def inputs(draft, token):
        await client.read_projection(sid, incarnation)
        row, pressed = client.projection_view.session(sid), client.pressed_inputs
        await joined(client.submit("UpdateInputs", {
            "draft": draft, "authored_token": [token, 1],
            "expected_settings_revision": row.settings_revision,
            "expected_settings": pressed["settings"],
            "expected_attachment_ids": pressed["attachment_ids"],
            "remove_attachment_ids": [], "prefill": None,
        }, session_id=sid, incarnation_id=incarnation))
        await client.read_projection(sid, incarnation)
        row, pressed = client.projection_view.session(sid), client.pressed_inputs
        assert row.draft == pressed["draft"] == draft
        return {
            "draft_revision": row.draft_revision,
            "expected_settings_revision": pressed["settings_revision"],
            "expected_attachment_revision": pressed["attachment_revision"],
            "expected_prefill_revision": pressed["prefill_revision"],
            "selection": snapshot(client.send_selection, "selection"),
            "pressed_inputs": snapshot(pressed, "pressed_inputs"),
        }

    def send(turn_id, payload):
        receipt = client.submit("ReceiveTurn", {"turn_id": turn_id, **payload},
                                session_id=sid, incarnation_id=incarnation)
        # This executes before any parent-loop turn can receive a child frame.
        record = client._archive_sends[receipt.identity.request_id]
        assert record.receipt is receipt and record.conversation_id == conversation_id
        assert app._conversation_send_inflight == {conversation_id: 1}
        return receipt, record

    async def finish(receipt, record, *, accepted, refusal=None):
        await asyncio.shield(receipt.received)
        assert (await asyncio.shield(receipt.applied))["turn_id"] == receipt.turn_id
        index = 0
        async with asyncio.timeout(10):
            while (event := await client.next_turn_event(receipt, index)) is not None:
                index += 1
                if event.kind == "commit_planned":
                    assert accepted, "An archive-refused turn reached the commit gate"
                    await joined(client.acknowledge_plan(receipt, event))
            assert await asyncio.shield(receipt.retired) is True
            assert await _until(lambda: receipt.identity.request_id not in client._archive_sends)
        terminal = [event for event in receipt.events if event.kind == "terminal"]
        assert len(terminal) == 1 and terminal[0].payload["accepted"] is accepted
        assert record.terminal and not app._conversation_send_inflight
        assert all(ack.applied.result()["applied"] is True and ack.retired.result() is True
                   for ack in record.acknowledgements)
        assert custody[-2:] == [(receipt.turn_id, "terminal"), (receipt.turn_id, "retired")]
        if refusal is not None:
            assert [ack["refusal"] for ack in sent_acks if ack["turn_id"] == receipt.turn_id][-1] == refusal
            await client.read_projection(sid, incarnation)
            assert client.projection_view.session(sid).draft == receipt_drafts[receipt.turn_id]

    try:
        async with app.run_test(size=(120, 40)):
            await app._initial_screen_setup_task
            await app._console_execution_start_task
            assert app.console_runtime is None and client._view is None
            local = archive.local_conversation_service(app)
            conversation_id, other = [await archive.storage_call(
                local, "create_conversation", title=title,
                runtime_backend="local", scope_type="global",
            ) for title in ("Saved selected archive target", "Unrelated archive target")]
            opened = await joined(client.submit("OpenSession", {
                "title": "Saved selected archive target", "workspace_id": None,
                "conversation_id": conversation_id,
            }))
            sid, incarnation = opened["session_id"], opened["incarnation_id"]
            await joined(client.submit("PermissionAction", {
                "operation": {"kind": "SetKillSwitch", "value": False},
            }))
            receipt_drafts = {}
            with (
                _observe(ConsoleExecutionClient._receive.__code__, sys.monitoring.events.PY_START, received),
                _observe(write_message.__code__, sys.monitoring.events.PY_START, writing),
            ):
                # A saved unarchived conversation reaches the real provider;
                # both original archive checkpoints remain on this receipt.
                payload = await inputs("Saved process accepted", 1)
                accepted, record = send("archive-accepted", payload)
                await finish(accepted, record, accepted=True)
                assert [(ack["ordinal"], ack["refusal"]) for ack in sent_acks] == [(1, None), (2, None)]
                assert app._conversation_archive_states[conversation_id] is False
                assert len(accepted_process_provider.payloads) == 1

                # Hold only the actual writer's already-captured prepare ACK.
                # The original unrelated archive action then changes the real
                # global generation before the child's complete native read.
                receipt_drafts["archive-generation"] = "Preserve generation-race draft"
                payload = await inputs(receipt_drafts["archive-generation"], 2)
                generation = getattr(app, "_conversation_archive_generation", 0)
                cache = app._conversation_archive_states
                pending, record = send("archive-generation", payload)
                assert await _until(lambda: held_ack.is_set() or pending.retired.done())
                assert held_ack.is_set() and record.generation == generation
                original = checkpoints[-1]
                for invalid in (original, replace(original, payload={
                    **dict(original.payload), "ordinal": 2, "stage": "finish",
                    "conversation_id": other, "archived": False,
                })):
                    with pytest.raises(ConsoleProtocolError, match="archive_send_checkpoint_changed"):
                        receive_archive_checkpoint(client, pending, invalid)
                assert record.ordinal == 1 and len(record.acknowledgements) == 1
                version = (await archive.storage_call(local, "get_conversation_metadata", conversation_id))["version"]
                refused = await archive.change_conversation_archive(
                    app, [conversation_id], archived=True,
                    expected_versions={conversation_id: version},
                )
                assert refused == {"changed": {}, "failures": {
                    conversation_id: "Wait for the current send to finish before archiving.",
                }}
                other_version = (await archive.storage_call(local, "get_conversation_metadata", other))["version"]
                changed = await archive.change_conversation_archive(
                    app, [other], archived=True, expected_versions={other: other_version},
                )
                assert other in changed["changed"] and not changed["failures"]
                assert app._conversation_archive_generation == generation + 1
                release_ack.set()
                await finish(pending, record, accepted=False, refusal="archive_inflight")
                assert app._conversation_archive_states is cache
                assert cache == {conversation_id: False, other: True}
                assert len(accepted_process_provider.payloads) == 1

                # Hold the actual parent archive producer after its DB method
                # returns but before its original owned call retires/publishes.
                # The prepare checkpoint must refuse without a finish read.
                await inputs("", 3)
                receipt_drafts["archive-inflight"] = "Preserve archive-inflight draft"

                def archived(frame):
                    if frame.f_locals["self"] is local:
                        held_archive.set()
                        if not release_archive.wait(10):
                            expired.set()

                version = (await archive.storage_call(local, "get_conversation_metadata", conversation_id))["version"]
                with _observe(type(local).set_conversations_archived.__code__, sys.monitoring.events.PY_RETURN, archived):
                    pending_archive = asyncio.create_task(archive.change_conversation_archive(
                        app, [conversation_id], archived=True,
                        expected_versions={conversation_id: version},
                    ))
                    assert await _until(held_archive.is_set)
                    assert conversation_id in app._conversation_archive_inflight
                    payload = await inputs(receipt_drafts["archive-inflight"], 4)
                    refused, record = send("archive-inflight", payload)
                    await finish(refused, record, accepted=False, refusal="archive_inflight")
                    assert record.ordinal == 1
                    release_archive.set()
                    changed = await pending_archive
                    assert conversation_id in changed["changed"] and not changed["failures"]

                # A later send reads the actual archived DB state, reaches
                # original cache publication and preserves its new draft too.
                receipt_drafts["archive-durable"] = "Preserve archived draft"
                payload = await inputs(receipt_drafts["archive-durable"], 5)
                refused, record = send("archive-durable", payload)
                await finish(refused, record, accepted=False, refusal="archived")
                assert record.ordinal == 2
                assert app._conversation_archive_states[conversation_id] is True
                assert [message.payload["archived"] for message in checkpoints
                        if message.payload["turn_id"] == "archive-durable"] == [None, True]
                assert len(accepted_process_provider.payloads) == 1
                assert not expired.is_set() and not client._archive_sends
                assert app.console_runtime is None and client._view is None
        assert client._disposed and client._physically_closed
        assert client._process.returncode == 0
    finally:
        release_ack.set()
        release_archive.set()
        if pending_archive is not None:
            await asyncio.gather(pending_archive, return_exceptions=True)
