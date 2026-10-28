"""Trace-only continuation round storage, corruption and reachability."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from Tests.Chat.test_console_trace_graph_gc import (
    _conversation,
    _finish_legacy_migration,
    _root_graph,
    _terminal_call,
)
from tldw_chatbook.Chat.console_trace_continuation import (
    CONTINUATION_DELTA_COMPONENT,
    CONTINUATION_ROUND_NORMALIZATION,
    compact_continuation_projection,
    resolve_continuation_node,
)
from tldw_chatbook.Chat.console_trace_maintenance import TraceGarbageCollector
from tldw_chatbook.Chat.console_trace_models import SurfaceReplacement, TraceContentRef
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db():
    database = CharactersRAGDB(":memory:", "native-rounds")
    yield database
    database.close_connection()


def _value(count):
    return {
        "schema_version": 3,
        "checkpoint_revision": count * 3,
        "managed_resume": None,
        "provider": "deepseek",
        "selected_provider": "deepseek",
        "protocol": "chat_completions",
        "model": "deepseek-test",
        "api_base_url": "https://api.deepseek.com/v1",
        "state": "active",
        "rounds": [
            {
                "assistant_content": "",
                "reasoning_blocks": ["reason " + str(index)],
                "calls": [
                    {
                        "call_id": f"call-{index}",
                        "name": "calculator",
                        "arguments": '{"expression":"2+2"}',
                        "state": "completed",
                        "result": "4" * 1024,
                    }
                ],
            }
            for index in range(count)
        ],
    }


def _append(
    db,
    repository,
    segment_id,
    policy_id,
    count,
    *,
    corrupt=None,
    edge=True,
    compact=True,
):
    with db.transaction() as cursor:
        previous = repository.get_surface_tail(cursor, segment_id)
        value = (
            _value(count)
            if count == 1 or not compact
            else compact_continuation_projection(
                _value(count - 1), _value(count), policy_id
            )
        )
        if corrupt:
            corrupt(value)
        artifact = repository.store_sanitized_artifact(
            cursor,
            sanitized_bytes=json.dumps(
                value, sort_keys=True, separators=(",", ":")
            ).encode(),
            media_type="application/json",
            normalization_version="canonical-json-v1"
            if count == 1 or not compact
            else CONTINUATION_ROUND_NORMALIZATION,
        )
        node = repository.append_surface_node(
            cursor,
            segment_id=segment_id,
            sequence=previous.sequence + 1,
            predecessor_node_id=previous.node_id,
            component_kind="continuation"
            if count == 1 or not compact
            else CONTINUATION_DELTA_COMPONENT,
            reference=TraceContentRef(artifact.artifact_id, "trace_artifact"),
        )
        event_sequence = repository.get_event_tail(cursor, segment_id).sequence + 1
        if count == 1:
            repository.append_event(
                cursor,
                segment_id=segment_id,
                sequence=event_sequence,
                event_type="surface_append",
                surface_node_id=node.node_id,
            )
        elif edge:
            replacement = repository.append_surface_replacement(
                cursor,
                segment_id=segment_id,
                replacement=SurfaceReplacement(
                    previous.node_id,
                    previous.node_id,
                    previous.node_id,
                    previous.sequence,
                    previous.sequence,
                    node.node_id,
                ),
            )
            repository.append_event(
                cursor,
                segment_id=segment_id,
                sequence=event_sequence,
                event_type="surface_replace",
                surface_replacement_id=replacement.replacement_id,
            )
        return node


def test_round_chain_keeps_legacy_full_values_and_exact_history(db):
    repository = ConsoleTraceRepository()
    _, _, segment, policy, _ = _root_graph(db, repository)
    nodes = [_append(db, repository, segment, policy, count) for count in range(1, 5)]
    with db.transaction() as cursor:
        for count, node in enumerate(nodes, 1):
            assert resolve_continuation_node(
                repository, cursor, node, policy_id=policy
            ) == _value(count)


@pytest.mark.parametrize(
    "fault",
    [
        "missing_edge",
        "duplicate_edge",
        "wrong_domain",
        "wrong_policy",
        "wrong_count",
        "rounds_in_metadata",
        "changed_schema",
        "missing_artifact",
        "cycle",
    ],
)
def test_round_chain_refuses_corrupt_storage(db, fault):
    repository = ConsoleTraceRepository()
    _, _, segment, policy, _ = _root_graph(db, repository)
    first = _append(db, repository, segment, policy, 1)

    def corrupt(value):
        if fault == "wrong_policy":
            value["policy_id"] = "missing"
        if fault == "wrong_count":
            value["prior_round_count"] = 4
        if fault == "rounds_in_metadata":
            value["checkpoint"]["rounds"] = []
        if fault == "changed_schema":
            value["checkpoint"]["schema_version"] = 999

    node = _append(
        db,
        repository,
        segment,
        policy,
        2,
        corrupt=corrupt,
        edge=fault != "missing_edge",
    )
    with db.transaction() as cursor:
        if fault == "duplicate_edge":
            repository.append_surface_replacement(
                cursor,
                segment_id=segment,
                replacement=SurfaceReplacement(
                    first.node_id,
                    first.node_id,
                    first.node_id,
                    first.sequence,
                    first.sequence,
                    node.node_id,
                ),
            )
        if fault == "wrong_domain":
            node = replace(node, component_kind="message")
        if fault == "missing_artifact":
            node = replace(node, artifact_id="missing")
        if fault == "cycle":
            node = replace(node, predecessor_node_id=node.node_id)
        with pytest.raises(ValueError, match="continuation_trace"):
            resolve_continuation_node(repository, cursor, node, policy_id=policy)


@pytest.mark.parametrize(
    "fault",
    [
        "old_projection",
        "round_limit",
        "call_limit",
        "reasoning_limit",
        "argument_limit",
        "result_limit",
        "depth",
    ],
)
def test_compact_projection_preserves_prefix_and_existing_bounds(fault):
    old, current = _value(1), _value(2)
    if fault == "old_projection":
        current["rounds"][0]["calls"][0]["result"] = "newly redacted"
    if fault == "round_limit":
        current = _value(129)
    if fault == "call_limit":
        current["rounds"][-1]["calls"] *= 128
    if fault == "reasoning_limit":
        current["rounds"][-1]["reasoning_blocks"] = ["x" * (4 * 1024 * 1024 + 1)]
    if fault == "argument_limit":
        current["rounds"][-1]["calls"][0]["arguments"] = "x" * (1024 * 1024 + 1)
    if fault == "result_limit":
        current["rounds"][-1]["calls"][0]["result"] = "x" * 16001
    if fault == "depth":
        nested = "value"
        for _ in range(35):
            nested = [nested]
        current["unexpected"] = nested
    with pytest.raises(ValueError):
        compact_continuation_projection(old, current, "policy")


def test_round_storage_artifact_and_metadata_growth_is_linear(db, tmp_path):
    repository = ConsoleTraceRepository()
    _, _, segment, policy, _ = _root_graph(db, repository)
    samples = {}
    for count in range(1, 33):
        node = _append(db, repository, segment, policy, count)
        if count in {8, 16, 32}:
            with db.transaction() as cursor:
                assert resolve_continuation_node(repository, cursor, node) == _value(
                    count
                )
                artifact_bytes = cursor.execute(
                    "SELECT SUM(byte_length) FROM console_trace_artifacts"
                ).fetchone()[0]
                metadata_bytes = 0
                for table in (
                    "console_trace_surface_nodes",
                    "console_trace_surface_replacements",
                    "console_trace_events",
                    "console_trace_artifacts",
                ):
                    for row in cursor.execute("SELECT * FROM " + table):
                        metadata_bytes += sum(
                            len(str(value).encode())
                            for value in row
                            if value is not None and not isinstance(value, bytes)
                        )
                samples[count] = {
                    "artifact_bytes": artifact_bytes,
                    "metadata_bytes": metadata_bytes,
                }
    for field in ("artifact_bytes", "metadata_bytes"):
        assert 1.5 < samples[32][field] / samples[16][field] < 2.2
        assert 1.5 < samples[16][field] / samples[8][field] < 2.2
    (tmp_path / "growth.json").write_text(json.dumps(samples, sort_keys=True))


@pytest.mark.parametrize(
    "compact", [False, True], ids=["legacy-full", "compact-rounds"]
)
def test_archive_fork_and_final_owner_gc_keep_round_dependencies(db, compact):
    repository = ConsoleTraceRepository()
    conversation, owner, segment, policy, _ = _root_graph(db, repository)
    node = None
    for count in range(1, 5):
        node = _append(db, repository, segment, policy, count, compact=compact)
    with db.transaction() as cursor:
        _terminal_call(
            cursor,
            repository,
            owner_id=owner,
            segment_id=segment,
            policy_id=policy,
            turn_id="native",
            event_sequence=5,
        )
        version = cursor.execute(
            "SELECT version FROM conversations WHERE id = ?", (conversation,)
        ).fetchone()[0]
    assert db.set_conversations_archived(
        [conversation], archived=True, expected_versions={conversation: version}
    )["changed"]
    child_id, _ = _conversation(db, "child")
    with db.transaction() as cursor:
        boundary = repository.capture_fork_boundary(
            cursor, conversation_id=conversation, included_turn_ids=("native",)
        )
        child = repository.attach_fork_owner(
            cursor, conversation_id=child_id, boundary=boundary
        )
        repository.detach_owner(
            cursor, owner_id=owner, detached_at="2026-08-31T12:01:00Z"
        )
    _finish_legacy_migration(db)
    assert (
        TraceGarbageCollector(db).collect(request_id="keep-native-fork").status
        == "completed"
    )
    with db.transaction() as cursor:
        assert resolve_continuation_node(repository, cursor, node) == _value(4)
        assert (
            len(
                repository.read_lineage_surface_nodes(
                    cursor,
                    segment_id=child.root_segment_id,
                    start_sequence=0,
                    end_sequence=4,
                )
            )
            == 5
        )
        repository.detach_owner(
            cursor, owner_id=child.owner_id, detached_at="2026-08-31T12:02:00Z"
        )
    assert (
        TraceGarbageCollector(db).collect(request_id="release-native-fork").status
        == "completed"
    )
    with db.transaction() as cursor:
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM console_trace_surface_replacements"
            ).fetchone()[0]
            == 0
        )
        assert (
            cursor.execute("SELECT COUNT(*) FROM console_trace_artifacts").fetchone()[0]
            == 0
        )


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("fault", ["exception", "orphan"])
def test_sweep_failure_restores_graph_and_foreign_key_mode(
    db, monkeypatch, nested, fault, deferred
):
    from contextlib import nullcontext

    repository = ConsoleTraceRepository()
    _, owner, segment, policy, _ = _root_graph(db, repository)
    for count in range(1, 3):
        _append(db, repository, segment, policy, count)
    with db.transaction() as cursor:
        _terminal_call(
            cursor,
            repository,
            owner_id=owner,
            segment_id=segment,
            policy_id=policy,
            turn_id="kept",
            event_sequence=3,
        )
        if fault == "exception":
            repository.detach_owner(
                cursor, owner_id=owner, detached_at="2026-08-31T12:01:00Z"
            )
    _finish_legacy_migration(db)
    collector = TraceGarbageCollector(db)
    collector.mark(request_id="native-failed-sweep")
    original = TraceGarbageCollector._sweep_unmarked

    def fail(cursor, request_id):
        result = original(cursor, request_id)
        if fault == "exception":
            raise RuntimeError("injected_sweep_failure")
        cursor.execute(
            "DELETE FROM console_trace_policies WHERE policy_id = ?", (policy,)
        )
        return result

    monkeypatch.setattr(TraceGarbageCollector, "_sweep_unmarked", staticmethod(fail))
    connection = db.get_connection()
    tables = (
        "console_trace_artifacts",
        "console_trace_events",
        "console_trace_surface_nodes",
        "console_trace_surface_replacements",
        "console_trace_policies",
    )
    before = {
        table: tuple(map(tuple, connection.execute("SELECT * FROM " + table)))
        for table in tables
    }
    with db.transaction() if nested else nullcontext():
        connection.execute(
            "PRAGMA defer_foreign_keys = " + ("ON" if deferred else "OFF")
        )
        with pytest.raises(
            RuntimeError, match="injected_sweep_failure|trace_gc_foreign_key_dependency"
        ):
            collector.sweep(request_id="native-failed-sweep")
        assert connection.execute("PRAGMA defer_foreign_keys").fetchone()[0] == int(
            nested and deferred
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert {
            table: tuple(map(tuple, connection.execute("SELECT * FROM " + table)))
            for table in tables
        } == before
    assert connection.execute("PRAGMA defer_foreign_keys").fetchone()[0] == 0
