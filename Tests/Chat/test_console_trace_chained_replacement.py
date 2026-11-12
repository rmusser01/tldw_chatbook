"""A rewritten tail wider than 256 stored sequences is recorded as a chain.

TASK-38298, ADR-337 amendment 2026-11-06 (accepted by the owner 2026-11-12).
After Compact now or Edit & resend in a chat of more than about 128 exchanges,
the next captured send replaces more than 256 stored sequences. One replacement
row may span at most 256, and every reader refuses a wider row. The send is
recorded instead as a chain of ordinary replacement rows: each group of at most
256 stored sequences is replaced by a content-free placeholder, and a final
link replaces the placeholders with the first new row.

Each case reserves, plans, verifies and binds a real captured send through the
trace service, then reads the trace back with a fresh service and with the
native reader that the Inspector and trace export use.
"""

from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import pytest

from Tests.Chat.test_console_trace_rewritten_tail_planner import (
    chat_factory as chat_factory,  # noqa: PLC0414 - pytest fixture re-export
)
from Tests.Chat.test_console_trace_service import _provenance
from tldw_chatbook.Chat import console_trace_service as trace_service_module
from tldw_chatbook.Chat.console_trace_final_values import (
    build_verified_surface_delta,
    verify_provider_request_shadow,
)
from tldw_chatbook.Chat.console_trace_maintenance import TraceGarbageCollector
from tldw_chatbook.Chat.console_trace_models import TraceCallState, new_opaque_id
from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader
from tldw_chatbook.Chat.console_trace_provenance import ConsoleRequestRoute
from tldw_chatbook.Chat.console_trace_service import (
    ConsoleTraceService,
    TraceCallIdentity,
)

_COUNTS = (
    "SELECT count(*) FROM console_trace_surface_nodes",
    "SELECT count(*) FROM console_trace_surface_replacements",
    "SELECT count(*) FROM console_trace_events "
    "WHERE event_type IN ('surface_append', 'surface_replace', 'gap')",
    "SELECT count(*) FROM console_trace_events",
)


def _send(chat, rows, current, *, service):
    """Reserve, plan, verify and bind one captured send, as the runtime does."""

    reserved = service.reserve_call(
        chat.db,
        TraceCallIdentity(
            chat.owner_id,
            chat.segment_id,
            current.message_id,
            str(uuid4()),
            0,
            new_opaque_id(),
            chat.policy.policy_id,
        ),
    )
    preparation_identity = new_opaque_id()
    with chat.db.transaction() as cursor:
        admission, boundary = service.prepare_current_surface_delta(
            cursor,
            owner_id=chat.owner_id,
            segment_id=chat.segment_id,
            route_identity=ConsoleRequestRoute.FRESH.value,
            preparation_identity=preparation_identity,
            provenance=_provenance(tuple(row.descriptor for row in rows)),
            values=tuple(row.value for row in rows),
            current_turn_id=current.message_id,
            current_policy_id=chat.policy.policy_id,
            reserved_call=reserved,
        )
    sent = {
        "api_endpoint": "openai",
        "model": "gpt-test",
        **boundary._provider_request_surface_values(),
    }
    bundle = verify_provider_request_shadow(
        actual_kwargs=sent,
        expected_kwargs=dict(sent),
        provenance=boundary.provenance,
        project_handler_kwargs=lambda supplied: supplied,
        endpoint_identity="https://api.example.invalid/v1",
        preparation_identity=preparation_identity,
        surface_boundary=boundary,
    )
    delta = build_verified_surface_delta(
        boundary.provenance, bundle, admission=admission
    )
    call = service.bind_and_mark_dispatch(
        chat.db,
        call_id=reserved.call_id,
        owner_id=chat.owner_id,
        segment_id=chat.segment_id,
        provenance=boundary.provenance,
        bundle=bundle,
        surface_delta=delta,
        occurred_at="2026-11-12T00:00:00Z",
    )
    with chat.db.transaction() as cursor:
        for state in (TraceCallState.RESPONSE_STARTED, TraceCallState.COMPLETE):
            chat.repository.advance_call_state(
                cursor,
                call_id=call.call_id,
                target=state,
                occurred_at="2026-11-12T00:00:01Z",
            )
        tail = chat.repository.get_event_tail(cursor, chat.segment_id)
        chat.repository.append_event(
            cursor,
            segment_id=chat.segment_id,
            sequence=tail.sequence + 1,
            event_type="call_outcome",
            call_id=call.call_id,
        )
    chat.head = call.surface_node_id
    return admission, call, list(sent["messages_payload"])


def _keys(chat, head_id):
    """Revision ids of one head's surface, read cold by a fresh service."""

    with chat.db.transaction() as cursor:
        head = chat.repository.get_surface_node(cursor, head_id)
        projection = ConsoleTraceService(chat.repository)._surface_projection(
            cursor, chat.segment_id, head
        )
    return [key[2] for _sequence, key in projection.entries]


def _assert_warm_matches_cold(chat, service, head_id):
    """The checkpoint a warm service keeps equals what a restart rebuilds.

    The next send plans on the warm checkpoint, so its sequences must
    account for the placeholders, and its lineage must count them as the
    cold reader does.
    """

    warm = service._surface_ref_cache[chat.segment_id]
    parent = service._parent_capabilities[id(warm.checkpoint)]
    with chat.db.transaction() as cursor:
        head = chat.repository.get_surface_node(cursor, head_id)
        cold = ConsoleTraceService(chat.repository)._surface_projection(
            cursor, chat.segment_id, head
        )
    entries = cold.root.materialize()
    assert warm.head_id == parent.head_id == head_id
    assert warm.root.materialize() == parent.root.materialize() == entries
    assert dict(warm.lineage_domains) == dict(cold.lineage_domains)
    assert parent.lineage_domains == dict(cold.lineage_domains)
    assert [sequence for sequence, _ in parent.descriptors.iter_entries()] == [
        sequence for sequence, _ in entries
    ]
    assert set(parent.active_domains) == {sequence for sequence, _ in entries}
    assert parent.next_sequence == head.sequence + 1


def _revisions(rows):
    return [row.descriptor.revision_id for row in rows]


def _exported(chat, current):
    """The request the native reader (Inspector, trace export) rebuilds."""

    (call,) = ConsoleTraceNativeReader(chat.db).read_calls(current.message_id)
    assert call.verification_status == "verified"
    return call.capture.request["messages_payload"]


def _counts(chat):
    """Surface nodes, replacement rows, surface events and all events."""

    with chat.db.transaction() as cursor:
        return tuple(cursor.execute(query).fetchone()[0] for query in _COUNTS)


def _ranges(admission):
    return [
        (link.start_sequence, link.end_sequence)
        for link in admission.replacement_range.chain
    ]


def _long_chat(chat, trigger):
    """Record 130 exchanges (259 rows), then Compact now or edit the first."""

    rows = chat.chain(260)
    first = chat.record(rows[:259])
    if trigger == "compact":
        chat.remember(rows[-3])
        current = chat.add(rows[-1], "after compaction")
        incoming = [rows[-2], rows[-1], current]
    else:
        edited = chat.add(None, "edited question")
        answer = chat.add(edited, "answer", "assistant")
        current = chat.add(answer, "after the edit")
        incoming = [edited, answer, current]
    return rows[:259], first, incoming, current


@pytest.mark.parametrize("restart", [False, True], ids=["warm", "restarted"])
@pytest.mark.parametrize("trigger", ["compact", "edit-first"])
def test_a_rewrite_over_256_recorded_rows_is_captured_as_a_chain(
    chat_factory, trigger, restart
):
    """AC#2/#3: Compact now, or Edit & resend on the first message, at 130
    exchanges. The send is captured, and its trace is exactly what was sent."""

    chat = chat_factory()
    recorded, first, incoming, current = _long_chat(chat, trigger)
    service = ConsoleTraceService(chat.repository) if restart else chat.recorder
    before = _counts(chat)

    admission, call, sent = _send(chat, incoming, current, service=service)

    assert chat.refusals == []
    assert admission.rewritten_tail is not None
    assert admission.rewritten_tail.memory_anchor is (trigger == "compact")
    assert _ranges(admission) == [(0, 255), (256, 258)]
    assert sent == [row.value for row in incoming]
    assert _exported(chat, current) == sent
    assert _keys(chat, call.surface_node_id) == _revisions(incoming)
    _assert_warm_matches_cold(chat, service, call.surface_node_id)
    # Two placeholder links and the final link: 5 nodes, 3 replacement rows,
    # 3 surface_replace events and 2 surface_append events, and no gap event.
    assert tuple(
        after - prior for after, prior in zip(_counts(chat)[:3], before[:3], strict=True)
    ) == (5, 3, 5)
    with chat.db.transaction() as cursor:
        placeholders = cursor.execute(
            "SELECT component_kind, reference_kind, omission_reason_code "
            "FROM console_trace_surface_nodes WHERE sequence IN (259, 260) "
            "ORDER BY sequence"
        ).fetchall()
        gaps = cursor.execute(
            "SELECT count(*) FROM console_trace_events WHERE event_type = 'gap'"
        ).fetchone()[0]
    assert [tuple(row) for row in placeholders] == [
        ("surface_replacement", "omission", "replacement_chain_link")
    ] * 2
    assert gaps == 0
    # The head recorded before the chain still reconstructs its 259 rows.
    assert _keys(chat, first.surface_head_id) == _revisions(recorded)

    # The following send is a plain append over the chained head.
    following = chat.add(current, "and again")
    later, call, sent = _send(
        chat, [*incoming, following], following, service=service
    )
    assert later.replacement_range is None
    assert later.rewritten_tail is None
    assert _exported(chat, following) == sent
    assert _keys(chat, call.surface_node_id) == _revisions([*incoming, following])
    _assert_warm_matches_cold(chat, service, call.surface_node_id)


def test_a_chain_over_dead_sequences_keeps_each_link_within_256(chat_factory):
    """131 live rows over 261 stored sequences (an earlier edit left 129 dead)."""

    from Tests.Chat.test_console_trace_service import SurfaceReplacementPlan

    chat = chat_factory()
    rows = chat.chain(140)
    first = chat.record(rows)
    edit = chat.add(rows[0], "edited", "assistant")
    surface = [rows[0], edit, *rows[131:]]
    chat.record(
        surface,
        replacement=SurfaceReplacementPlan(
            predecessor_head_id=first.surface_head_id,
            start_node_id=first.appended_nodes[1].node_id,
            end_node_id=first.appended_nodes[130].node_id,
            start_sequence=1,
            end_sequence=130,
            replacement_index=1,
        ),
    )
    later = chat.chain(120, parent=rows[-1], start=140)
    recorded = [*surface, *later]
    third = chat.record(recorded, append_from=len(surface))
    chat.remember(later[-3])
    current = chat.add(later[-1], "after compaction")
    incoming = [later[-2], later[-1], current]

    admission, call, sent = _send(chat, incoming, current, service=chat.recorder)

    assert _ranges(admission) == [(0, 255), (256, 260)]
    assert _exported(chat, current) == sent == [row.value for row in incoming]
    assert _keys(chat, call.surface_node_id) == _revisions(incoming)
    _assert_warm_matches_cold(chat, chat.recorder, call.surface_node_id)
    assert _keys(chat, third.surface_head_id) == _revisions(recorded)


def test_trace_gc_keeps_every_link_of_a_chain(chat_factory):
    chat = chat_factory()
    recorded, first, incoming, current = _long_chat(chat, "compact")
    _send(chat, incoming, current, service=chat.recorder)
    following = chat.add(current, "and again")
    _admission, call, _sent = _send(
        chat, [*incoming, following], following, service=chat.recorder
    )
    before = _counts(chat)

    result = TraceGarbageCollector(chat.db).collect(request_id=new_opaque_id())

    assert result.status == "completed"
    assert _counts(chat) == before
    assert _keys(chat, call.surface_node_id) == _revisions([*incoming, following])
    assert _keys(chat, first.surface_head_id) == _revisions(recorded)
    assert _exported(chat, following)[-1] == following.value


def test_a_rewrite_needing_more_links_than_the_cap_is_refused(
    chat_factory, monkeypatch
):
    """The final link spans one placeholder per link, so the cap is 256 links.

    Recording 65,537 sequences is too slow for a unit test, so the cap is
    lowered to two links here: 512 sequences fit in two, 513 need three.
    """

    assert trace_service_module.MAX_SURFACE_REPLACEMENT_CHAIN_LINKS == 256
    monkeypatch.setattr(trace_service_module, "MAX_SURFACE_REPLACEMENT_CHAIN_LINKS", 2)
    outcomes = {}
    for recorded in (512, 513):
        chat = chat_factory()
        rows = chat.chain(recorded + 1)
        chat.record(rows[:recorded])
        chat.remember(rows[-3])
        current = chat.add(rows[-1], "after compaction")
        incoming = [rows[-2], rows[-1], current]
        if recorded == 512:
            outcomes[recorded] = _ranges(chat.prepare(incoming, current))
        else:
            outcomes[recorded] = chat.refusal(incoming, current)

    assert outcomes == {512: [(0, 255), (256, 511)], 513: ["count_or_span"]}


def test_dispatch_binding_revalidates_every_link(chat_factory):
    """Each link's range is rechecked, and no live row may lie between links."""

    from tldw_chatbook.Chat.console_trace_final_values import (
        SurfaceReplacementLink,
        VerifiedSurfaceDeltaItem,
        VerifiedSurfaceReplacement,
    )

    chat = chat_factory()
    rows = chat.chain(259)
    first = chat.record(rows)
    nodes = first.appended_nodes

    def plan(*bounds, last_end_node=None):
        links = [
            SurfaceReplacementLink(nodes[start].node_id, nodes[end].node_id, start, end)
            for start, end in bounds
        ]
        if last_end_node is not None:
            links[-1] = replace(links[-1], end_node_id=last_end_node.node_id)
        links = tuple(links)
        return VerifiedSurfaceReplacement(
            predecessor_head_id=first.surface_head_id,
            start_node_id=links[0].start_node_id,
            end_node_id=links[-1].end_node_id,
            start_sequence=links[0].start_sequence,
            end_sequence=links[-1].end_sequence,
            current_ordinal=0,
            item=VerifiedSurfaceDeltaItem(
                "messages_payload", 0, rows[0].descriptor
            ),
            chain=links,
        )

    service = ConsoleTraceService(chat.repository)
    outcomes = {}
    with chat.db.transaction() as cursor:
        tail = chat.repository.get_surface_node(cursor, first.surface_head_id)
        for label, candidate in (
            ("whole", plan((0, 255), (256, 258))),
            ("gap", plan((0, 100), (200, 258))),
            ("wrong end node", plan((0, 255), (256, 258), last_end_node=nodes[257])),
        ):
            try:
                service._validate_replacement_range(
                    cursor,
                    segment_id=chat.segment_id,
                    tail=tail,
                    plan=candidate,
                    descriptor_count=1,
                )
                outcomes[label] = "valid"
            except ValueError as exc:
                outcomes[label] = str(exc)

    assert outcomes == {
        "whole": "valid",
        "gap": "replacement_chain_gap",
        "wrong end node": "replacement_range_mismatch",
    }
