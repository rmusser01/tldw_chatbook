"""A runtime poll publishes only changes since the UI's last adopted result."""

import asyncio
import time
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_projection_adoption import projection_payload
from tldw_chatbook.Chat.console_execution_protocol import (
    ConsoleProtocolError, ConsoleWireMessage, WireIdentity, decode_message, encode_message,
)
from tldw_chatbook.Chat.console_display_poll import (
    DisplayPublication, expand_projection_delta, poll_response,
)


def full(payload, request_id, sequence):
    identity = WireIdentity(
        "profile", "service", sequence, request_id, payload["session_id"],
        payload["incarnation_id"], payload["source_revision"],
    )
    return ConsoleWireMessage(identity, "applied", "ReadProjection", payload)


def poll(after="first", *, visit=1):
    return {"attachment_id": "ui", "visit": visit, "after": after}


def test_poll_round_trip_omits_unchanged_sections_and_preserves_complete_checks():
    body = projection_payload()
    first = full(body, "first", 4)
    base = DisplayPublication("ui", 1, first)
    body["session_view"]["fingerprints"][0]["last_content_length"] += 1
    second = full(body, "second", 8)
    delta = poll_response(second, poll(), base)
    assert delta.kind == "projection_delta"
    assert set(delta.payload["sections"]) == {"fingerprints"}
    assert not delta.payload["fields"]
    received = decode_message(encode_message(delta))
    assert expand_projection_delta(received, poll(), base) == second
    third = full(body, "third", 12)
    next_base = DisplayPublication("ui", 1, second)
    empty = poll_response(third, poll("second"), next_base)
    assert not empty.payload["fields"] and not empty.payload["sections"]
    assert expand_projection_delta(empty, poll("second"), next_base) == third


@pytest.mark.parametrize("change", ["visit", "attachment", "base", "source", "incarnation", "epoch"])
def test_stale_cursor_gets_a_complete_snapshot_and_cannot_expand_an_old_delta(change):
    body = projection_payload()
    first, second = full(body, "first", 4), full(body, "second", 8)
    current = poll()
    base = DisplayPublication("ui", 1, first)
    delta = poll_response(second, current, base)
    if change == "visit":
        current["visit"] += 1
    elif change == "attachment":
        current["attachment_id"] = "other"
    elif change == "base":
        current["after"] = None
    elif change == "epoch":
        base = replace(base, message=replace(first, identity=replace(first.identity, service_epoch="new"),
                                              payload=encode_message(first)["payload"]))
    else:
        other = deepcopy(body) if change == "source" else projection_payload()
        if change == "source":
            other["source_revision"] += 1
        base = replace(base, message=full(other, "first", 4))
    assert poll_response(second, current, base) is second
    with pytest.raises(ConsoleProtocolError):
        expand_projection_delta(delta, current, base)


def test_delta_cannot_publish_inconsistent_or_unknown_sections():
    body = projection_payload()
    first = full(body, "first", 4)
    base = DisplayPublication("ui", 1, first)
    delta = poll_response(full(body, "second", 8), poll(), base)
    bad = encode_message(delta)
    bad["payload"]["sections"]["active_session_id"] = "absent"
    with pytest.raises(ConsoleProtocolError):
        expand_projection_delta(decode_message(bad), poll(), base)
    bad["payload"]["sections"] = {"permission": True}
    with pytest.raises(ConsoleProtocolError):
        decode_message(bad)


def test_readiness_clock_update_publishes_no_unchanged_inspector_or_history_data():
    body = projection_payload()
    first = full(body, "first", 4)
    base = DisplayPublication("ui", 1, first)
    body["session_view"]["readiness"]["captured_at_monotonic"] += 1.0
    changed = full(body, "second", 8)
    delta = poll_response(changed, poll(), base)
    assert not delta.payload["sections"]
    assert delta.payload["readiness"].keys() == {"captured_at_monotonic"}
    assert expand_projection_delta(delta, poll(), base) == changed


def test_transcript_reference_does_not_copy_the_discarded_full_history(monkeypatch):
    from tldw_chatbook.Chat import console_execution_protocol as codec

    body = projection_payload(message_count=32)
    first = full(body, "first", 4)
    base = DisplayPublication("ui", 1, first)
    body["messages"] = None
    second = full(body, "second", 8)
    delta = poll_response(second, poll(), base)
    previous = first.payload
    visited = []
    detach = codec._detach

    def observe(value, *args, **kwargs):
        if value is previous["messages"]:
            visited.append(value)
        return detach(value, *args, **kwargs)

    monkeypatch.setattr(codec, "_detach", observe)
    assert expand_projection_delta(delta, poll(), base) == second
    assert not visited


def test_encoding_checks_and_exports_one_disposable_full_projection(monkeypatch):
    from tldw_chatbook.Chat import console_execution_protocol as codec

    message = full(projection_payload(), "first", 4)
    copies = []
    detach = codec._detach

    def observe(value, *args, **kwargs):
        result = detach(value, *args, **kwargs)
        if value is message.payload and kwargs.get("thaw"):
            copies.append(result)
        return result

    monkeypatch.setattr(codec, "_detach", observe)
    encoded = encode_message(message)
    assert len(copies) == 1 and copies[0] is encoded["payload"]
    encoded["payload"]["session_view"]["active_session_id"] = "different"
    assert message.payload["session_view"]["active_session_id"] == message.identity.session_id
    with pytest.raises(ConsoleProtocolError):
        decode_message(encoded)


def test_new_visit_reads_the_entire_transcript_even_if_revision_matches():
    from tldw_chatbook.Chat.console_control_dispatch import ReadProjection
    from tldw_chatbook.Chat.console_execution_projection_result import decode_projection_payload
    from tldw_chatbook.Chat.console_execution_worker import _BackendService

    body = projection_payload()
    frame = decode_projection_payload(body)
    adapter = SimpleNamespace(_receipts={})
    backend = SimpleNamespace(_worker_service=None, adapter=adapter, presentation=SimpleNamespace(
        projection_view=lambda identity, result: body["session_view"],
    ))
    wire = []
    worker = _BackendService(backend, SimpleNamespace(send=wire.append))
    for visit in (1, 2):
        identity = full(body, f"poll-{visit}", visit).identity
        request = ConsoleWireMessage(identity, "command", "ReadProjection", {
            "known_display_revision": frame.display_revision, "known_marker_revision": frame.marker_revision,
            "display_poll": poll(None, visit=visit),
        })
        adapter._receipts[identity.request_id] = (request,)
        worker.applied(identity, ReadProjection(identity), frame)
    assert wire[-1].kind == "applied"
    assert len(wire[-1].payload["messages"]) == len(frame.messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("retired", [True, False])
async def test_client_polls_after_last_adopted_result_and_waits_for_retirement(retired):
    from Tests.Chat.test_console_execution_process import _ready_client, _reply
    from tldw_chatbook.Chat.console_execution_process import ConsoleChannelLost

    client = _ready_client()
    client._visit, client._view = 1, object()
    client.require_view = lambda view, visit: (view is client._view and visit == 1) or pytest.fail()
    body = projection_payload()
    body["source_revision"] = client._source_revision
    session, incarnation = body["session_id"], body["incarnation_id"]
    first = None
    published_sections = None
    for index in range(3 if retired else 2):
        task = asyncio.create_task(client._read_projection_locked(session, incarnation))
        await asyncio.sleep(0)
        request = client._channel.messages[-1]
        receipt = client._requests[request.identity.request_id]
        cursor = request.payload["display_poll"]
        assert cursor["after"] == (first.message.identity.request_id if first else None)
        sequence = index * 3 + 1
        client._receive(_reply(client, receipt, sequence, "received", {"sequence": request.identity.sequence}))
        body["session_view"]["readiness"].update(captured_at_monotonic=time.monotonic(), remaining_seconds=1.0)
        if index == 2:
            history = body["session_view"]["readiness"]["prompt_history"]
            history["revision"] += 1
            history["entries"].append({"input": "new prompt", "timestamp": "1"})
        checked = _reply(client, receipt, sequence + 1, "applied", body)
        response = poll_response(checked, cursor, first)
        if index:
            assert response.kind == "projection_delta"
        client._receive(decode_message(encode_message(response)))
        assert receipt.applied.result() == checked.payload
        assert client._display_publication is first
        client._receive(_reply(client, receipt, sequence + 2, "retired", {
            "operation_id": request.identity.request_id, "retired": not index or retired,
        }))
        if index and not retired:
            with pytest.raises(ConsoleChannelLost, match="retirement"):
                await task
            assert client._display_publication is first
        else:
            await task
            assert client._display_publication.message == checked
            first = client._display_publication
            sections = (
                client.workspace_surface, client.session_composition, client.context_summary,
                client.agent_rail, client.prompt_history._entries,
            )
            if index == 1:
                # Only the readiness clock changed; local recall and mounted
                # readers continue to share the previously published sections.
                assert all(new is old for new, old in zip(sections, published_sections, strict=True))
            elif index == 2:
                assert all(new is old for new, old in zip(sections[:-1], published_sections[:-1], strict=True))
                assert client.prompt_history.entry_texts()[-1] == "new prompt"
                assert sections[-1] is not published_sections[-1]
            published_sections = sections
