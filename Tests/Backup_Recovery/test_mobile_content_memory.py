"""Native captured memory plans preserve proofs without activating history."""

from dataclasses import replace

import pytest

from Tests.Backup_Recovery.test_mobile_content_graph import _seed_core, content_reads


def _seed_policy(db, purpose):
    from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
    from tldw_chatbook.Chat.console_context_repository import (
        AuxiliaryAttemptStart,
        AuxiliaryAttemptStatus,
        AuxiliaryPricingProvenance,
        ConsoleContextRepository,
    )
    from tldw_chatbook.Chat.provider_usage import ProviderUsage

    repository = ConsoleContextRepository(db)
    repository.save_policy(
        "conversation",
        ConsoleContextPolicyOverrides.from_mapping(
            {
                "budget_mode": "custom",
                "custom_budget_tokens": 4096,
                "compaction_mode": "ask",
                "trigger_ratio": 0.8,
                "target_ratio": 0.5,
                "summary_max_tokens": 256,
                "compaction_representation": "text_summary",
            }
        ),
    )
    repository.start_auxiliary_attempt(
        AuxiliaryAttemptStart(
            "memory-attempt",
            "conversation",
            "conversation_compaction",
            "openai",
            "model",
            256,
            900,
            "2026-10-26T01:02:03Z",
        )
    )
    repository.finish_auxiliary_attempt(
        "memory-attempt",
        status=AuxiliaryAttemptStatus.FAILED,
        finished_at="2026-10-26T01:02:04Z",
        elapsed_ms=1000,
        usage=ProviderUsage(uncached_input=900, output=7),
        pricing=AuxiliaryPricingProvenance(
            catalog_revision="historic", source="provider"
        ),
        failure_reason="summary_did_not_make_progress",
    )


def test_captured_complete_content_preserves_native_context_and_accounting(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
    from tldw_chatbook.Chat.provider_usage import ProviderUsage

    with content_reads(
        tmp_path, monkeypatch, _seed_core, native_seed=_seed_policy
    ) as reads:
        build = getattr(reads, "content_plan", None)
        assert callable(build), "captured reader needs complete C assembly"
        plan = build()
        assert not plan.blockers
        policy = next(
            dict(r.values)
            for r in plan.rows
            if r.table == "console_conversation_context_policy"
        )
        assert policy["conversation_id"] == plan.identities.mapped(
            "conversation", "conversation"
        )
        assert (
            ConsoleContextPolicyOverrides.from_mapping(policy).custom_budget_tokens
            == 4096
        )
        attempt = next(
            dict(r.values) for r in plan.rows if r.table == "console_auxiliary_attempts"
        )
        assert attempt["operation_id"] == plan.identities.mapped(
            "auxiliary_attempt", "memory-attempt"
        )
        assert attempt["status"] == "failed"
        assert attempt["failure_reason"] == "summary_did_not_make_progress"
        assert (
            ProviderUsage.from_json(attempt["provider_usage_json"]).total_tokens == 907
        )
        assert sum(len(columns) for _, columns in plan.field_ownership) == 580
        assert len(plan.field_ownership) == 61
        assert plan.pending_handlers == ("T", "S", "A-runs")
        assert {r.table for r in plan.operational_provenance} >= {
            "character_conversation_search_state",
            "character_conversation_search_revision",
        }
        assert not any(
            r.table.startswith("character_conversation_search_") for r in plan.rows
        )
        for value in (plan, replace(plan, rows=(), blockers=(), pending_handlers=())):
            with pytest.raises(ValueError, match="pending"):
                value.require_complete()


def _seed_memory(db, purpose):
    import json

    from tldw_chatbook.Chat import console_context_repository as native
    from tldw_chatbook.Chat.console_context_compaction import (
        DurableConversationUnit,
        DurableMessageSnapshot,
    )

    _seed_policy(db, purpose)
    with db.transaction() as cursor:
        cursor.execute(
            "UPDATE conversations SET active_leaf_message_id='graph-answer' WHERE id='conversation'"
        )
        states = tuple(
            native._persisted_lineage_state(
                cursor,
                cursor.execute("SELECT * FROM messages WHERE id=?", (mid,)).fetchone(),
            )
            for mid in ("graph-user", "graph-answer")
        )
    snapshots = tuple(
        DurableMessageSnapshot(
            message_id=s.fence.message_id,
            version=s.fence.version,
            role=s.role,
            content=s.content,
            parent_message_id=s.fence.parent_message_id,
            selected_variant_id=s.fence.selected_variant_id,
            selected_variant_index=s.fence.selected_variant_index,
            attachment_digests=s.fence.attachment_digests,
        )
        for s in states
    )
    repository = native.ConsoleContextRepository(db)
    repository.insert_memory(
        native.ConsoleMemoryRecord(
            memory_id="memory-one",
            conversation_id="conversation",
            boundary_message_id="graph-answer",
            captured_leaf_message_id="graph-answer",
            lineage_json=json.dumps([s.fence.message_id for s in states]),
            summary_text="Historical summary",
            provider="openai",
            model="model",
            prompt_id="prompt-opaque",
            prompt_revision=4,
            prompt_digest="a" * 64,
            selected_units_json=json.dumps(
                [DurableConversationUnit(snapshots).provenance_payload()]
            ),
            summarized_prefix_digest=native._prefix_digest_from_persisted(states),
            input_tokens=900,
            output_tokens=30,
            before_tokens=1100,
            after_tokens=200,
            created_at="2026-10-26T01:02:04Z",
        )
    )
    repository.insert_memory_scope(
        native.ConsoleMemoryScopeRecord(
            "memory-one",
            "conversation",
            native.MemoryCoverageKind.PREFIX,
            native.MemoryOriginKind.AUTOMATIC,
        )
    )
    repository.insert_memory_selection(
        native.ConsoleMemorySelectionRecord(
            1,
            "select-one",
            "conversation",
            "graph-answer",
            "memory-one",
            native.MemorySelectionKind.SELECT,
            False,
            "2026-10-26T01:02:04Z",
        )
    )


def test_captured_memory_uses_native_mapped_prefix_and_unit_provenance(
    tmp_path, monkeypatch
):
    import json

    from tldw_chatbook.Chat import console_context_repository as native
    from tldw_chatbook.Chat.console_context_compaction import (
        DurableConversationUnit,
        DurableMessageSnapshot,
    )

    with content_reads(
        tmp_path, monkeypatch, _seed_core, native_seed=_seed_memory
    ) as reads:
        plan = reads.content_plan()
        assert not plan.blockers
        rows = {
            r.table: dict(r.values)
            for r in plan.rows
            if r.table
            in {
                "console_conversation_memories",
                "console_conversation_memory_scopes",
                "console_conversation_memory_selections",
            }
        }
        memory = rows["console_conversation_memories"]
        original = next(
            dict(r.values)
            for r in plan.source_rows
            if r.table == "console_conversation_memories"
        )
        assert memory["prompt_digest"] == original["prompt_digest"]
        assert (
            memory["summarized_prefix_digest"] != original["summarized_prefix_digest"]
        )
        units = json.loads(memory["selected_units_json"])
        assert units[0]["messages"][0]["message_id"] == plan.identities.mapped(
            "message", "graph-user"
        )
        assert units[0]["messages"][1]["selected_variant_id"] == plan.identities.mapped(
            "message", "graph-variant"
        )
        assert (
            units[0]["messages"][0]["attachment_digests"]
            == json.loads(original["selected_units_json"])[0]["messages"][0][
                "attachment_digests"
            ]
        )
        states = []
        snapshots = []
        for proof in units[0]["messages"]:
            message = next(
                dict(r.values)
                for r in plan.rows
                if r.table == "messages" and dict(r.values)["id"] == proof["message_id"]
            )
            content = message["content"]
            if proof["selected_variant_id"]:
                content = next(
                    dict(r.values)["content"]
                    for r in plan.rows
                    if r.table == "messages"
                    and dict(r.values)["id"] == proof["selected_variant_id"]
                )
            fence = native.PersistedLineageFenceRow(
                proof["message_id"],
                proof["parent_message_id"],
                proof["version"],
                proof["deleted"],
                proof["content_digest"],
                proof["selected_variant_id"],
                proof["selected_variant_index"],
                tuple(proof["attachment_digests"]),
            )
            states.append(native._PersistedLineageState(fence, proof["role"], content))
            snapshots.append(
                DurableMessageSnapshot(
                    message_id=proof["message_id"],
                    version=proof["version"],
                    role=proof["role"],
                    content=content,
                    parent_message_id=proof["parent_message_id"],
                    selected_variant_id=proof["selected_variant_id"],
                    selected_variant_index=proof["selected_variant_index"],
                    attachment_digests=tuple(proof["attachment_digests"]),
                )
            )
        assert memory[
            "summarized_prefix_digest"
        ] == native._prefix_digest_from_persisted(tuple(states))
        assert units == [DurableConversationUnit(tuple(snapshots)).provenance_payload()]
        selection = native._memory_selection_from_row(
            rows["console_conversation_memory_selections"]
        )
        assert native._persisted_memory_is_valid(
            memory | rows["console_conversation_memory_scopes"],
            selection,
            tuple(states),
            {s.fence.message_id: i for i, s in enumerate(states)},
        )
        assert selection.sequence > 1
        assert native._memory_from_row(memory).summary_text == "Historical summary"


def _native_memory_inputs(change=None):
    """Native rows and explicit data-only maps; this never claims capture authority."""
    from contextlib import closing
    from types import SimpleNamespace

    from tldw_chatbook.Backup_Recovery.mobile_content_graph import (
        _ContentRow,
        _GraphBuilder,
        _UnprovedReference,
    )
    from tldw_chatbook.Backup_Recovery.mobile_content_memory import _TABLES
    from tldw_chatbook.Backup_Recovery.mobile_plan_manifest import _TABLES as specs
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    class Inputs(_GraphBuilder):
        def __init__(self, source, originals):
            self.specs = {s.table: s for s in specs if s.family == "chat"}
            self.source = source
            self.canonical = {t: [] for t in _TABLES}
            self.rows, self.blockers, self.dependencies = [], [], []
            self.identities = SimpleNamespace(source_manifest_digest="a" * 64)
            self.maps = {
                ("conversation", "conversation"): "mapped-conversation",
                ("auxiliary_attempt", "memory-attempt"): "mapped-attempt",
            }
            for row in originals:
                if row.table == "messages":
                    self.maps["message", row.original[0]] = "mapped-" + row.original[0]
            for row in source["console_conversation_memories"]:
                self.maps["memory", row["id"]] = "mapped-" + row["id"]
            for row in source["console_conversation_memory_selections"]:
                self.maps["memory_selection", row["selection_id"]] = (
                    "mapped-" + row["selection_id"]
                )
                self.maps["memory_sequence", row["sequence"]] = row["sequence"] + 100
            for row in originals:
                values = dict(row.values)
                if row.table == "messages":
                    for column in ("id", "parent_message_id", "variant_of"):
                        if values[column] is not None:
                            values[column] = self.maps["message", values[column]]
                    values["conversation_id"] = "mapped-conversation"
                    self.rows.append(replace(row, values=tuple(values.items())))
                elif row.table == "message_attachments":
                    values["message_id"] = self.maps["message", values["message_id"]]
                    self.rows.append(replace(row, values=tuple(values.items())))

        def ref(self, domain, identity, *, optional=False):
            if identity is None and optional:
                return None
            if (domain, identity) not in self.maps:
                raise _UnprovedReference(domain, identity)
            self.dependencies.append((domain, identity))
            return self.maps[domain, identity]

    with closing(CharactersRAGDB(":memory:", "memory-native-input")) as db:
        db.add_conversation({"id": "conversation", "title": "native memory"})
        with db.transaction() as cursor:
            _seed_core(cursor.connection, "legacy")
        _seed_memory(db, "legacy")
        if change:
            change(db)
        tables = _TABLES + ("messages", "message_attachments")
        source = {}
        manifest = {s.table: s for s in specs if s.family == "chat"}
        for table in tables:
            columns = ",".join(
                'CAST("created_at" AS TEXT) AS "created_at"'
                if column == "created_at"
                and table
                in {
                    "console_conversation_memories",
                    "console_conversation_memory_selections",
                }
                else '"' + column + '"'
                for column in manifest[table].columns
            )
            source[table] = [
                dict(r)
                for r in db.get_connection().execute(
                    "SELECT " + columns + ' FROM "' + table + '"'
                )
            ]
        manifest = {s.table: s for s in specs if s.family == "chat"}
        originals = tuple(
            _ContentRow(
                t,
                tuple(row[c] for c in manifest[t].identity_columns),
                tuple(row.items()),
            )
            for t in ("messages", "message_attachments")
            for row in source[t]
        )
    return Inputs(source, originals), originals


def test_native_memory_scope_failure_closes_memory_and_selection_dependencies():
    from tldw_chatbook.Backup_Recovery import mobile_content_memory as module

    def invalid_scope(db):
        with db.transaction() as cursor:
            cursor.execute(
                "UPDATE console_conversation_memory_scopes SET origin_kind='manual_rewind',selection_anchor_message_id='graph-user'"
            )

    builder, originals = _native_memory_inputs(invalid_scope)
    module._memory_rows(builder, originals)
    close = getattr(module, "_close_memory_dependencies", None)
    assert callable(close), "full C must close invalid scope to memory and selection"
    close(builder)
    assert not any(
        r.table.startswith("console_conversation_memor") for r in builder.rows
    )
    assert {b.table for b in builder.blockers} >= {
        "console_conversation_memories",
        "console_conversation_memory_scopes",
        "console_conversation_memory_selections",
    }


def test_captured_legacy_summary_successor_is_discharged_with_exact_mapped_boundary(
    tmp_path, monkeypatch
):
    def seed(db, purpose):
        with db.transaction() as cursor:
            cursor.execute(
                "UPDATE conversations SET context_summary='Retained legacy summary',summary_boundary_message_id='graph-answer',active_leaf_message_id='graph-answer' WHERE id='conversation'"
            )

    with content_reads(tmp_path, monkeypatch, _seed_core, native_seed=seed) as reads:
        plan = reads.content_plan()
        assert not plan.blockers
        assert not any(s.handler == "C3" for s in plan.successors)
        assert any(
            s.handler == "C3" and s.domain == "conversation_memory"
            for s in plan.discharged
        )
        conversation = next(
            dict(r.values)
            for r in plan.rows
            if r.table == "conversations" and r.original == ("conversation",)
        )
        assert conversation["context_summary"] == "Retained legacy summary"
        assert conversation["summary_boundary_message_id"] == plan.identities.mapped(
            "message", "graph-answer"
        )


def _checkpoint_memory(db, *, runtime=False, expanded=True):
    import json

    from Tests.Chat.test_provider_continuation import _call, _checkpoint
    from tldw_chatbook.Chat import console_context_repository as native
    from tldw_chatbook.Chat.console_context_compaction import (
        DurableConversationUnit,
        DurableMessageSnapshot,
        prefix_digest,
    )
    from tldw_chatbook.Chat.provider_continuation import (
        dump_provider_continuation_json,
        parse_provider_continuation_json,
    )

    checkpoint = parse_provider_continuation_json(
        _checkpoint(
            state="complete",
            rounds=[
                {
                    "assistant_content": "Calling",
                    "reasoning_blocks": [],
                    "calls": [_call(state="completed", result="4")],
                }
            ],
        )
    )
    with db.transaction() as cursor:
        cursor.execute("UPDATE messages SET deleted=1 WHERE id='graph-variant'")
        cursor.execute(
            "UPDATE messages SET provider_continuation_json=?,assistant_generation_state='complete' WHERE id='graph-answer'",
            (dump_provider_continuation_json(checkpoint),),
        )
        states = tuple(
            native._persisted_lineage_state(
                cursor,
                cursor.execute("SELECT * FROM messages WHERE id=?", (mid,)).fetchone(),
            )
            for mid in ("graph-user", "graph-answer")
        )
        raw = tuple(
            DurableMessageSnapshot(
                s.fence.message_id,
                s.fence.version,
                s.role,
                s.content,
                parent_message_id=s.fence.parent_message_id,
                attachment_digests=s.fence.attachment_digests,
            )
            for s in states
        )
        call = checkpoint.rounds[0].calls[0]
        virtual = (
            DurableMessageSnapshot(
                "graph-answer:tool-calls:0",
                raw[1].version,
                "assistant",
                "Calling",
                parent_message_id="graph-user",
                tool_calls=(
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    },
                ),
            ),
            DurableMessageSnapshot(
                "graph-answer:tool-result:0:0",
                raw[1].version,
                "tool",
                call.result.value,
                parent_message_id="graph-answer:tool-calls:0",
                tool_call_id=call.call_id,
            ),
        )
        unit = (
            (raw[0],)
            + virtual
            + (replace(raw[1], parent_message_id=virtual[-1].message_id),)
            if expanded
            else raw
        )
        cursor.execute(
            "UPDATE console_conversation_memories SET selected_units_json=?, summarized_prefix_digest=?",
            (
                json.dumps([DurableConversationUnit(unit).provenance_payload()]),
                prefix_digest(raw)
                if runtime
                else native._prefix_digest_from_persisted(states),
            ),
        )


@pytest.mark.parametrize("expanded", [False, True])
def test_native_checkpoint_memory_reconstructs_fixed_virtual_ids(expanded):
    import json

    from tldw_chatbook.Backup_Recovery.mobile_content_memory import (
        _close_memory_dependencies,
        _memory_rows,
    )

    builder, originals = _native_memory_inputs(
        lambda db: _checkpoint_memory(db, runtime=True, expanded=expanded)
    )
    _memory_rows(builder, originals)
    _close_memory_dependencies(builder)
    assert not builder.blockers
    memory = next(
        dict(r.values)
        for r in builder.rows
        if r.table == "console_conversation_memories"
    )
    proof = json.loads(memory["selected_units_json"])[0]["messages"]
    assert proof[0]["message_id"] == "mapped-graph-user"
    assert proof[-1]["message_id"] == "mapped-graph-answer"
    if expanded:
        assert proof[1]["message_id"] == "mapped-graph-answer:tool-calls:0"
        assert proof[2]["parent_message_id"] == proof[1]["message_id"]
        assert proof[1]["tool_call_ids"] == ["call_1"]
        assert proof[2]["tool_call_id"] == "call_1"
        assert proof[-1]["parent_message_id"] == proof[2]["message_id"]
    assert memory["prompt_digest"] == "a" * 64
    assert (
        memory["summarized_prefix_digest"]
        != builder.source["console_conversation_memories"][0][
            "summarized_prefix_digest"
        ]
    )


@pytest.mark.parametrize(
    "change", ["body", "checkpoint", "unit_order", "unit_duplicate", "attachment"]
)
def test_native_memory_rejects_unrepresentable_historical_inputs(change):
    import json

    from tldw_chatbook.Backup_Recovery.mobile_content_memory import (
        _close_memory_dependencies,
        _memory_rows,
    )

    def mutate(db):
        _checkpoint_memory(db)
        with db.transaction() as cursor:
            if change == "body":
                cursor.execute(
                    "UPDATE messages SET content='new body',version=version+1 WHERE id='graph-answer'"
                )
            elif change == "checkpoint":
                cursor.execute(
                    "UPDATE messages SET provider_continuation_json=NULL WHERE id='graph-answer'"
                )
            elif change == "attachment":
                cursor.execute(
                    "INSERT INTO message_attachments(message_id,position,data,mime_type,display_name) VALUES('graph-user',1,?,'image/png','new')",
                    (b"changed",),
                )
            else:
                row = cursor.execute(
                    "SELECT selected_units_json FROM console_conversation_memories"
                ).fetchone()
                units = json.loads(row[0])
                if change == "unit_order":
                    units[0]["messages"][1:3] = reversed(units[0]["messages"][1:3])
                else:
                    units += units
                cursor.execute(
                    "UPDATE console_conversation_memories SET selected_units_json=?",
                    (json.dumps(units),),
                )

    builder, originals = _native_memory_inputs(mutate)
    _memory_rows(builder, originals)
    _close_memory_dependencies(builder)
    assert any(b.table == "console_conversation_memories" for b in builder.blockers)
    assert not any(
        r.table.startswith("console_conversation_memor") for r in builder.rows
    )
    assert (
        builder.source["console_conversation_memories"][0]["summary_text"]
        == "Historical summary"
    )


def _sealed_memories(db, *, cycle=False):
    import json

    from tldw_chatbook.Chat import console_context_repository as native
    from tldw_chatbook.Chat.console_context_compaction import _sealed_memory_marker

    repository = native.ConsoleContextRepository(db)
    with db.transaction() as cursor:
        cursor.execute(
            "UPDATE console_conversation_memory_scopes SET coverage_kind='range',origin_kind='manual_rewind',selection_anchor_message_id='graph-user'"
        )
        row = dict(
            cursor.execute("SELECT * FROM console_conversation_memories").fetchone()
        )
        scope = native._memory_scope_from_row(
            cursor.execute(
                "SELECT * FROM console_conversation_memory_scopes"
            ).fetchone()
        )
    previous = native._memory_from_row(row)
    record = replace(
        previous,
        memory_id="memory-two",
        selected_units_json=json.dumps([_sealed_memory_marker(previous, scope)]),
    )
    repository.insert_memory(record)
    new_scope = (
        replace(scope, memory_id="memory-two")
        if cycle
        else native.ConsoleMemoryScopeRecord(
            "memory-two",
            "conversation",
            native.MemoryCoverageKind.PREFIX,
            native.MemoryOriginKind.AUTOMATIC,
        )
    )
    repository.insert_memory_scope(new_scope)
    repository.insert_memory_selection(
        native.ConsoleMemorySelectionRecord(
            2,
            "select-two",
            "conversation",
            "graph-answer",
            "memory-two",
            native.MemorySelectionKind.SELECT,
            True,
            "2026-10-26T01:03:00Z",
        )
    )
    repository.insert_memory_selection(
        native.ConsoleMemorySelectionRecord(
            3,
            "reset-three",
            "conversation",
            "graph-answer",
            None,
            native.MemorySelectionKind.RESET,
            True,
            "2026-10-26T01:04:00Z",
        )
    )
    if cycle:
        with db.transaction() as cursor:
            cursor.execute(
                "UPDATE console_conversation_memories SET selected_units_json=? WHERE id='memory-one'",
                (json.dumps([_sealed_memory_marker(record, new_scope)]),),
            )


@pytest.mark.parametrize("cycle", [False, True])
def test_native_sealed_range_history_and_selection_frontier(cycle):
    import json

    from tldw_chatbook.Backup_Recovery.mobile_content_memory import (
        _close_memory_dependencies,
        _memory_rows,
    )
    from tldw_chatbook.Chat.console_context_repository import _memory_selection_from_row

    builder, originals = _native_memory_inputs(
        lambda db: _sealed_memories(db, cycle=cycle)
    )
    _memory_rows(builder, originals)
    _close_memory_dependencies(builder)
    if cycle:
        assert any(
            b.reason == "native_memory_provenance_cycle" for b in builder.blockers
        )
        assert not any(r.table == "console_conversation_memories" for r in builder.rows)
        return
    assert not builder.blockers
    memory = next(
        dict(r.values)
        for r in builder.rows
        if r.table == "console_conversation_memories" and r.original == ("memory-two",)
    )
    assert json.loads(memory["selected_units_json"]) == [
        {
            "kind": "sealed_prior_memory",
            "memory_id": "mapped-memory-one",
            "memory_revision": 1,
            "start_message_id": "mapped-graph-user",
            "end_message_id": "mapped-graph-answer",
        }
    ]
    selections = [
        _memory_selection_from_row(dict(r.values))
        for r in builder.rows
        if r.table == "console_conversation_memory_selections"
    ]
    assert [s.sequence for s in selections] == [101, 102, 103]
    assert selections[-1].event_kind.value == "reset"
    assert selections[-1].selected_memory_id is None


def _seed_search(db, purpose):
    from tldw_chatbook.DB.character_conversation_search import (
        CharacterConversationSearchRepository,
    )

    with db.transaction() as cursor:
        cursor.execute(
            "UPDATE conversations SET assistant_kind='character',assistant_id='1',assistant_authority_id=?,character_id=1,runtime_backend='local',active_leaf_message_id='graph-variant' WHERE id='conversation'",
            (db.get_local_authority_id(),),
        )
    assert (
        CharacterConversationSearchRepository(db).ensure_keyword_index().value
        == "ready"
    )
    with db.transaction() as cursor:
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM character_conversation_search_documents"
            ).fetchone()[0]
            > 0
        )
        cursor.execute("UPDATE messages SET content=content WHERE id='graph-user'")
        if purpose == "canonical":
            cursor.execute(
                "INSERT INTO character_conversation_search_generations(generation_id,data_authority_id,status,policy_version,source_revision,lease_expires_at) VALUES('unfinished-build',?,'building',1,0,DATETIME('now','+5 minutes'))",
                (db.get_local_authority_id(),),
            )


def test_captured_search_preserves_typed_source_provenance_and_refuses_destination_build(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import _sqlite_row_digest
    from tldw_chatbook.Backup_Recovery.mobile_content_memory import _SEARCH_TABLES
    from tldw_chatbook.Backup_Recovery.mobile_plan_manifest import _TABLES

    with content_reads(
        tmp_path, monkeypatch, _seed_core, native_seed=_seed_search
    ) as reads:
        plan = reads.content_plan()
        assert any(
            b.reason == "canonical_search_build_unresolved" for b in plan.blockers
        )
        assert {p.table for p in plan.operational_provenance} == set(_SEARCH_TABLES)
        assert not any(r.table in _SEARCH_TABLES for r in plan.rows)
        columns = {s.table: s.columns for s in _TABLES}
        for proof in plan.operational_provenance:
            original = next(
                dict(r.values)
                for r in plan.source_rows
                if r.table == proof.table and r.original == proof.original
            )
            assert proof.source_digest == _sqlite_row_digest(
                tuple(original[c] for c in columns[proof.table])
            )
            assert (
                proof.source_manifest_digest == plan.identities.source_manifest_digest
            )
        assert plan.authority_slots
        assert plan.pending_handlers == ("T", "S", "A-runs")


@pytest.mark.parametrize("field", ["policy", "usage", "reason"])
def test_native_context_and_accounting_refuse_invalid_captured_values(field):
    from tldw_chatbook.Backup_Recovery.mobile_content_memory import (
        _context_and_accounting,
    )

    def change(db):
        with db.transaction() as cursor:
            if field == "policy":
                cursor.execute(
                    "UPDATE console_conversation_context_policy SET target_ratio=0.75"
                )
            elif field == "usage":
                cursor.execute(
                    "UPDATE console_auxiliary_attempts SET provider_usage_json='{\"uncached_input\":-1}'"
                )
            else:
                cursor.execute(
                    "UPDATE console_auxiliary_attempts SET status='succeeded'"
                )

    if field == "reason":
        # Native SQLite already enforces successful attempts have no failure reason.
        import sqlite3

        with pytest.raises(sqlite3.IntegrityError):
            _native_memory_inputs(change)
        return
    builder, _originals = _native_memory_inputs(change)
    _context_and_accounting(builder)
    table = (
        "console_conversation_context_policy"
        if field == "policy"
        else "console_auxiliary_attempts"
    )
    assert any(b.table == table for b in builder.blockers)
    assert not any(r.table == table for r in builder.rows)


@pytest.mark.parametrize("tail", ["checkpoint", "selected_role"])
def test_native_memory_preserves_prefix_with_independent_tail(tail):
    import json

    from Tests.Chat.test_provider_continuation import _checkpoint
    from tldw_chatbook.Backup_Recovery.mobile_content_memory import _memory_rows
    from tldw_chatbook.Chat import console_context_repository as native

    def change(db):
        with db.transaction() as cursor:
            cursor.execute(
                "INSERT INTO messages(id,conversation_id,parent_message_id,sender,role,content,client_id) "
                "VALUES ('tail','conversation','graph-answer','assistant','assistant','Later message','fixture')"
            )
            if tail == "checkpoint":
                cursor.execute(
                    "UPDATE messages SET provider_continuation_json=? WHERE id='tail'",
                    (json.dumps(_checkpoint()),),
                )
            else:
                cursor.execute(
                    "UPDATE messages SET is_selected_variant=0 WHERE id='tail'"
                )
                cursor.execute(
                    "INSERT INTO messages(id,conversation_id,parent_message_id,sender,role,content,variant_of,variant_number,is_selected_variant,client_id) "
                    "VALUES ('tail-variant','conversation','graph-answer','user','user','Selected user','tail',1,1,'fixture')"
                )
                cursor.execute(
                    "UPDATE console_conversation_memory_scopes SET origin_kind='manual_rewind',selection_anchor_message_id='tail'"
                )
                cursor.execute(
                    "UPDATE console_conversation_memory_selections SET suppresses_legacy=1"
                )
            lineage = ["graph-user", "graph-answer", "tail"]
            cursor.execute(
                "UPDATE console_conversation_memories SET captured_leaf_message_id='tail',lineage_json=?",
                (json.dumps(lineage),),
            )
            cursor.execute(
                "UPDATE console_conversation_memory_selections SET activation_message_id='tail'"
            )
            cursor.execute(
                "UPDATE conversations SET active_leaf_message_id='tail' WHERE id='conversation'"
            )
            states = tuple(
                native._persisted_lineage_state(
                    cursor,
                    cursor.execute(
                        "SELECT * FROM messages WHERE id=?", (mid,)
                    ).fetchone(),
                )
                for mid in lineage
            )
            memory = dict(
                cursor.execute("SELECT * FROM console_conversation_memories").fetchone()
            )
            scope = dict(
                cursor.execute(
                    "SELECT * FROM console_conversation_memory_scopes"
                ).fetchone()
            )
            selection = native._memory_selection_from_row(
                cursor.execute(
                    "SELECT * FROM console_conversation_memory_selections"
                ).fetchone()
            )
            assert native._persisted_memory_is_valid(
                memory | scope,
                selection,
                states,
                {mid: i for i, mid in enumerate(lineage)},
            )

    builder, originals = _native_memory_inputs(change)
    _memory_rows(builder, originals)
    assert not builder.blockers
    assert {r.table for r in builder.rows} >= {
        "console_conversation_memories",
        "console_conversation_memory_scopes",
        "console_conversation_memory_selections",
    }
    # A separately blocked C2 tail owner still prevents the memory proof.
    blocked, originals = _native_memory_inputs(change)
    blocked.rows = [
        r
        for r in blocked.rows
        if not (r.table == "messages" and r.original == ("tail",))
    ]
    _memory_rows(blocked, originals)
    assert any(b.table == "console_conversation_memories" for b in blocked.blockers)


def test_captured_content_schema_subset_preserves_live_metadata_and_retained_fences(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import sqlite_validation

    with content_reads(tmp_path, monkeypatch, _seed_core) as reads:
        subset = getattr(reads, "content_schema_catalog", None)
        assert callable(subset), "C builders need a fixed live C metadata subset"
        selected = ("notes", "console_conversation_memories")
        full = reads.schema_catalog()
        statements = []
        for member in reads._connections.values():
            member[0].set_trace_callback(statements.append)
        actual = subset(selected)
        assert actual == tuple(r for r in full if r[1] == "chat" and r[2] in selected)
        pragmas = [sql for sql in statements if sql.startswith("PRAGMA table_xinfo")]
        assert sorted(pragmas) == sorted(
            [
                'PRAGMA table_xinfo("' + table + '")'
                for table in selected
                for _ in range(2)
            ]
        )
        for member in reads._connections.values():
            member[0].set_trace_callback(None)

        class TupleSubclass(tuple):
            pass

        for invalid in (
            None,
            [],
            (),
            "notes",
            ("notes", "notes"),
            ("notes", 1),
            ("schema_version",),
            ("console_run_checkpoints",),
            ('notes"; DROP TABLE notes;--',),
            TupleSubclass(selected),
        ):
            with pytest.raises(ValueError, match="content_catalog_selector"):
                subset(invalid)
        # Even an unqueried retained family keeps its original resource fence.
        retained = reads._connections["canonical", "runs"][1]
        with monkeypatch.context() as fault:
            fault.setattr(retained, "steps", sqlite_validation._STEP_BUDGET + 1)
            with pytest.raises(ValueError, match="sqlite_resource_limit"):
                subset(selected)
        reads.cancel.set()
        try:
            with pytest.raises(InterruptedError, match="cancelled"):
                subset(selected)
        finally:
            reads.cancel.clear()
        # A non-Chat member altered during the metadata query is caught on return.
        candidate = reads._connections["legacy", "runs"][3]
        changed = []

        def alter_retained(sql):
            if sql.startswith("PRAGMA table_xinfo") and not changed:
                with candidate.open("ab") as stream:
                    stream.write(b"changed retained member")
                changed.append(True)

        connection = reads._connections["canonical", "chat"][0]
        connection.set_trace_callback(alter_retained)
        try:
            with pytest.raises(ValueError, match="candidate_changed"):
                subset(selected)
        finally:
            connection.set_trace_callback(None)
        assert changed
        with pytest.raises(ValueError, match="candidate_changed"):
            subset(selected)
