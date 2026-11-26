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

from contextlib import contextmanager
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


def _prepare_send(chat, rows, current, *, service):
    """Reserve, plan and verify one captured send, as the runtime does.

    Returns the admission, a callable that binds and dispatches the call, the
    reserved call and the messages the provider receives.
    """

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

    def bind():
        return service.bind_and_mark_dispatch(
            chat.db,
            call_id=reserved.call_id,
            owner_id=chat.owner_id,
            segment_id=chat.segment_id,
            provenance=boundary.provenance,
            bundle=bundle,
            surface_delta=delta,
            occurred_at="2026-11-12T00:00:00Z",
        )

    return admission, bind, reserved, list(sent["messages_payload"])


def _send(chat, rows, current, *, service):
    """Reserve, plan, bind and settle one captured send."""

    admission, bind, _reserved, sent = _prepare_send(
        chat, rows, current, service=service
    )
    return admission, _settle(chat, bind()), sent


def _settle(chat, call):
    """Complete a bound call, as the response does, and advance the head."""

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
    return call


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


def _row_zero_replaced(chat):
    """Rows 0-9, rows 0-5 replaced by one row (sequence 10), rows 11-14 after.

    The projection reads 10, 6-9, 11-14: its first row has a newer sequence
    than the rows after it, and it is not a system slot.
    """
    from Tests.Chat.test_console_trace_service import SurfaceReplacementPlan

    rows = chat.chain(10)
    first = chat.record(rows)
    edited = chat.add(None, "edited")
    surface = [edited, *rows[6:]]
    chat.record(
        surface,
        replacement=SurfaceReplacementPlan(
            predecessor_head_id=first.surface_head_id,
            start_node_id=first.appended_nodes[0].node_id,
            end_node_id=first.appended_nodes[5].node_id,
            start_sequence=0,
            end_sequence=5,
            replacement_index=0,
        ),
    )
    later = chat.chain(4, parent=rows[-1], start=10)
    third = chat.record([*surface, *later], append_from=len(surface))
    return first, third, edited


def test_a_rewrite_never_skips_a_leading_row_that_is_not_a_slot(chat_factory):
    """TASK-42525 round 5: a rewritten tail's links skip only a system slot.

    A restored system slot leads the projection with a newer sequence than
    the rows after it, and a rewrite's links skip it (ADR-337 amendment
    2026-11-25). A replacement row at the front looks the same by sequence;
    a rewrite from after it must still meet the sequence gap.
    """
    chat = chat_factory()
    _first, _third, edited = _row_zero_replaced(chat)
    n1 = chat.add(edited, "n1", "assistant")
    n2 = chat.add(n1, "n2", "user")
    current = chat.add(n2, "current", "assistant")

    assert chat.refusal([edited, n1, n2, current], current) == ["sequence_gap"]


def test_dispatch_binding_skips_only_a_leading_system_slot(chat_factory):
    """TASK-42525 round 5: binding accepts a skipped row only if it is a slot.

    A chain whose links skip the projection's first row is refused when that
    row is not a rendered-system slot, as for a call with no slot at all.
    """
    from tldw_chatbook.Chat.console_trace_final_values import (
        SurfaceReplacementLink,
        VerifiedSurfaceDeltaItem,
        VerifiedSurfaceReplacement,
    )

    chat = chat_factory()
    first, third, edited = _row_zero_replaced(chat)
    by_sequence = {
        node.sequence: node for node in (*first.appended_nodes, *third.appended_nodes)
    }
    links = tuple(
        SurfaceReplacementLink(
            by_sequence[start].node_id, by_sequence[end].node_id, start, end
        )
        for start, end in ((6, 9), (11, 14))
    )
    plan = VerifiedSurfaceReplacement(
        predecessor_head_id=third.surface_head_id,
        start_node_id=links[0].start_node_id,
        end_node_id=links[-1].end_node_id,
        start_sequence=6,
        end_sequence=14,
        current_ordinal=1,
        item=VerifiedSurfaceDeltaItem("messages_payload", 1, edited.descriptor),
        chain=links,
    )
    service = ConsoleTraceService(chat.repository)
    with chat.db.transaction() as cursor:
        tail = chat.repository.get_surface_node(cursor, third.surface_head_id)
        assert [sequence for sequence, _ in service._surface_projection(
            cursor, chat.segment_id, tail
        ).entries] == [10, 6, 7, 8, 9, 11, 12, 13, 14]
        with pytest.raises(ValueError, match="^replacement_chain_gap$"):
            service._validate_replacement_range(
                cursor,
                segment_id=chat.segment_id,
                tail=tail,
                plan=plan,
                descriptor_count=1,
            )


def test_dispatch_binding_refuses_a_link_that_covers_no_live_row(
    chat_factory, monkeypatch
):
    """TASK-40577: a link over dead sequences only is refused before any write.

    An earlier edit left sequences 1..130 dead. The reader folds each link
    over the live rows in its range, so a link holding none would commit and
    then fail every later read of the new head
    (``surface_replacement_target_inactive``). The planner never builds such
    a link; a defective grouping stands in for one here. Preparation runs the
    same check, so it is skipped while the send is prepared, to show that
    dispatch binding refuses the plan by itself. The send is then refused
    with nothing written, and the chat's next send records normally.
    """

    from Tests.Chat.test_console_trace_service import SurfaceReplacementPlan
    from tldw_chatbook.Chat.console_trace_errors import TraceCallPersistenceError

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

    real_links = trace_service_module._replacement_links
    defective = [(0, 0), (1, 130), (131, 255), (256, 260)]

    def defective_links(sequences, **options):
        assert real_links(sequences, **options) == [(0, 255), (256, 260)]
        return defective

    real_persist = ConsoleTraceService.persist_request
    real_validate = ConsoleTraceService._validate_replacement_range
    opened: list[int] = []
    refusals: list[tuple[str, int]] = []

    def persist_request(self, cursor, **kwargs):
        opened.append(cursor.connection.total_changes)
        return real_persist(self, cursor, **kwargs)

    def validate(self, cursor, **kwargs):
        try:
            return real_validate(self, cursor, **kwargs)
        except ValueError as exc:
            # Rows changed in the binding transaction before the refusal.
            refusals.append((str(exc), cursor.connection.total_changes - opened[-1]))
            raise

    with monkeypatch.context() as patch:
        patch.setattr(trace_service_module, "_replacement_links", defective_links)
        with monkeypatch.context() as preparation:
            preparation.setattr(
                ConsoleTraceService,
                "_validate_replacement_range",
                lambda *_args, **_kwargs: None,
            )
            admission, bind, reserved, _sent = _prepare_send(
                chat, incoming, current, service=chat.recorder
            )
        assert _ranges(admission) == defective
        patch.setattr(ConsoleTraceService, "persist_request", persist_request)
        patch.setattr(ConsoleTraceService, "_validate_replacement_range", validate)
        before = _counts(chat)

        with pytest.raises(TraceCallPersistenceError) as refused:
            bind()

        assert refused.value.outcome == "rolled_back"
        assert refusals == [("replacement_chain_link_inactive", 0)]
        assert _counts(chat) == before
        with chat.db.transaction() as cursor:
            call = chat.repository.get_call(cursor, reserved.call_id)
        assert call.state is TraceCallState.RESERVED
        assert call.surface_node_id is None
        assert _keys(chat, third.surface_head_id) == _revisions(recorded)
        # Preparation, which runs the same check, refuses the plan too.
        with pytest.raises(ValueError, match="^replacement_chain_link_inactive$"):
            _prepare_send(chat, incoming, current, service=chat.recorder)

    admission, call, sent = _send(chat, incoming, current, service=chat.recorder)

    assert _ranges(admission) == [(0, 255), (256, 260)]
    assert _exported(chat, current) == sent == [row.value for row in incoming]
    assert _keys(chat, call.surface_node_id) == _revisions(incoming)
    _assert_warm_matches_cold(chat, chat.recorder, call.surface_node_id)


def test_dispatch_binding_accepts_a_final_link_holding_only_its_end_row(
    chat_factory, monkeypatch
):
    """TASK-40577 review: the live-row check counts the range's last row.

    An edit replaced 12..267, so the live rows are 0..11 and the edit at 268.
    The planner's links are (0, 11) and (268, 268): the second link's only
    live row is the replaced range's end sequence. Preparation's copy of the
    check is skipped, so dispatch binding's own check is the one that must
    accept the chain, and the send must read back exactly.
    """

    from Tests.Chat.test_console_trace_service import SurfaceReplacementPlan

    chat = chat_factory()
    rows = chat.chain(268)
    first = chat.record(rows)
    edit = chat.add(rows[11], "edited", "user")
    surface = [*rows[:12], edit]
    chat.record(
        surface,
        replacement=SurfaceReplacementPlan(
            predecessor_head_id=first.surface_head_id,
            start_node_id=first.appended_nodes[12].node_id,
            end_node_id=first.appended_nodes[267].node_id,
            start_sequence=12,
            end_sequence=267,
            replacement_index=12,
        ),
    )
    chat.remember(rows[10])
    current = chat.add(edit, "after compaction")
    incoming = [rows[11], edit, current]
    with monkeypatch.context() as preparation:
        preparation.setattr(
            ConsoleTraceService,
            "_validate_replacement_range",
            lambda *_args, **_kwargs: None,
        )
        admission, bind, _reserved, sent = _prepare_send(
            chat, incoming, current, service=chat.recorder
        )
    assert _ranges(admission) == [(0, 11), (268, 268)]

    call = _settle(chat, bind())

    assert _exported(chat, current) == sent == [row.value for row in incoming]
    assert _keys(chat, call.surface_node_id) == _revisions(incoming)
    _assert_warm_matches_cold(chat, chat.recorder, call.surface_node_id)
    assert _keys(chat, first.surface_head_id) == _revisions(rows)


def test_a_committed_chain_reconciles_to_the_dispatched_call(
    chat_factory, monkeypatch
):
    """A failure after the binding commit still returns the dispatched call.

    Reconciliation reads back every link. Only the first link follows the
    head the send planned on; later links follow the chain's placeholders.
    """

    real_policy = trace_service_module.trace_critical_write_checkpoint_policy

    @contextmanager
    def failing_after_commit(database):
        with real_policy(database):
            yield
        raise RuntimeError("post-commit failure")

    chat = chat_factory()
    _recorded, _first, incoming, current = _long_chat(chat, "compact")
    admission, bind, reserved, _sent = _prepare_send(
        chat, incoming, current, service=chat.recorder
    )
    assert _ranges(admission) == [(0, 255), (256, 258)]
    monkeypatch.setattr(
        trace_service_module,
        "trace_critical_write_checkpoint_policy",
        failing_after_commit,
    )

    call = bind()

    assert call.call_id == reserved.call_id
    assert call.state is TraceCallState.DISPATCH_STARTED
    assert _keys(chat, call.surface_node_id) == _revisions(incoming)


@pytest.mark.parametrize("restart", [False, True], ids=["warm", "restarted"])
def test_a_later_chain_skips_the_dead_run_between_its_links(chat_factory, restart):
    """A second chain leaves the dead run between its links alone.

    An edit replaces sequence 0, so the retained prefix of the first chain
    holds a high-sequence row. The second chain then starts from the root:
    its links are (1..1) and (300..305), and the dead run 2..299 between
    them is not replaced again. The second link spans the first chain's
    placeholders. Every head before and after still reads back, and trace
    GC keeps everything.
    """

    from Tests.Chat.test_console_trace_service import SurfaceReplacementPlan

    chat = chat_factory()
    rows = chat.chain(300)
    first = chat.record(rows)
    edit = chat.add(None, "edited m0")
    surface = [edit, *rows[1:]]
    second = chat.record(
        surface,
        replacement=SurfaceReplacementPlan(
            predecessor_head_id=first.surface_head_id,
            start_node_id=first.appended_nodes[0].node_id,
            end_node_id=first.appended_nodes[0].node_id,
            start_sequence=0,
            end_sequence=0,
            replacement_index=0,
        ),
    )
    assert _keys(chat, second.surface_head_id) == _revisions(surface)
    service = ConsoleTraceService(chat.repository) if restart else chat.recorder
    # The first chain keeps the high-sequence edit in its retained prefix.
    edited = chat.add(rows[1], "edited q2")
    answer = chat.add(edited, "answer", "assistant")
    current = chat.add(answer, "after")
    incoming = [edit, rows[1], edited, answer, current]
    admission, call1, sent = _send(chat, incoming, current, service=service)
    assert _ranges(admission) == [(2, 257), (258, 299)]
    assert _exported(chat, current) == sent
    _assert_warm_matches_cold(chat, service, call1.surface_node_id)
    # The second chain starts from the root.
    root = chat.add(None, "fresh root")
    answer2 = chat.add(root, "answer2", "assistant")
    current2 = chat.add(answer2, "again")
    incoming2 = [root, answer2, current2]
    service2 = ConsoleTraceService(chat.repository) if restart else service
    admission2, call2, sent2 = _send(chat, incoming2, current2, service=service2)
    assert _ranges(admission2) == [(1, 1), (300, 305)]
    assert _exported(chat, current2) == sent2 == [row.value for row in incoming2]
    assert _keys(chat, call2.surface_node_id) == _revisions(incoming2)
    _assert_warm_matches_cold(chat, service2, call2.surface_node_id)
    assert _keys(chat, call1.surface_node_id) == _revisions(incoming)
    assert _keys(chat, second.surface_head_id) == _revisions(surface)
    assert _keys(chat, first.surface_head_id) == _revisions(rows)
    # Then a plain append over the second chain.
    following = chat.add(current2, "and again")
    later, call3, sent3 = _send(
        chat, [*incoming2, following], following, service=service2
    )
    assert later.replacement_range is None
    _assert_warm_matches_cold(chat, service2, call3.surface_node_id)
    before = _counts(chat)

    result = TraceGarbageCollector(chat.db).collect(request_id=new_opaque_id())

    assert result.status == "completed"
    assert _counts(chat) == before
    for head, expected in (
        (call3.surface_node_id, [*incoming2, following]),
        (call2.surface_node_id, incoming2),
        (call1.surface_node_id, incoming),
        (first.surface_head_id, rows),
    ):
        assert _keys(chat, head) == _revisions(expected)
    assert _exported(chat, following) == sent3
