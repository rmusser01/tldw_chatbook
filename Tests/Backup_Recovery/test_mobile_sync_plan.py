"""Fixed sync frontiers compose native captured plans without owner effects."""

from dataclasses import replace

import pytest

from Tests.Backup_Recovery.test_mobile_content_graph import content_reads


def test_integrated_plan_covers_manifest_and_keeps_owner_gates(tmp_path, monkeypatch):
    with content_reads(
        tmp_path,
        monkeypatch,
        lambda connection, purpose: None,
        native_seed=_version_default,
    ) as reads:
        build = getattr(reads, "integrated_plan", None)
        assert callable(build), "captured scope needs complete R/I/C/T/S composition"
        plan = build()
        assert not plan.blockers
        assert plan.sync.frontiers
        assert plan.manifest.tables
        assert plan.pending_handlers == ("A-runs", "owner-write", "public-cutover")
        for value in (plan, replace(plan, pending_handlers=())):
            with pytest.raises(ValueError, match="pending"):
                value.require_complete()
    with pytest.raises(ValueError, match="inactive"):
        reads.integrated_plan()


def _retired_entries(connection, purpose):
    connection.execute(
        "INSERT INTO world_books(id,name,client_id) VALUES (41,?, 'retired-fixture')",
        (purpose + " book",),
    )
    for identity in (-7, 0, 41):
        connection.execute(
            "INSERT INTO world_book_entries(id,world_book_id,keys,content) VALUES (?,41,'[]','erased native body')",
            (identity,),
        )
        connection.execute("DELETE FROM world_book_entries WHERE id=?", (identity,))
    connection.execute(
        "INSERT INTO world_book_entries(id,world_book_id,keys,content) VALUES (0,41,'[]','live replacement')"
    )


def test_native_retired_entry_ids_join_live_and_canonical_occupancy(
    tmp_path, monkeypatch
):
    import json

    with content_reads(tmp_path, monkeypatch, _retired_entries) as reads:
        columns, records = reads.page("legacy", "chat", "sync_log", limit=1000)
        tombstones = [
            dict(zip(columns, r, strict=True))
            for r in records
            if r[1:4]
            in (
                ("world_book_entries", "-7", "delete"),
                ("world_book_entries", "0", "delete"),
                ("world_book_entries", "41", "delete"),
            )
        ]
        assert len(tombstones) == 3
        assert all(
            set(json.loads(r["payload"])) == {"id", "world_book_id"} for r in tombstones
        )
        assert all("erased native body" not in r["payload"] for r in tombstones)
        plan = reads.identity_plan()
        reservations = [r for r in plan.reservations if r.domain == "world_book_entry"]
        assert {r.original for r in reservations} == {-7, 0, 41}
        assert len(reservations) == 3
        assert all(r.mapped not in {-7, 0, 41} for r in reservations)
        assert all(type(r.mapped) is int for r in reservations)
        sync = reads.sync_plan()
        deleted = [
            f
            for f in sync.frontiers
            if f.entity == "world_book_entries" and f.operation == "delete"
        ]
        assert len(deleted) == 3
        assert {f.original_id for f in deleted} == {-7, 0, 41}
        assert all(
            set(json.loads(f.payload)) == {"id", "world_book_id"} for f in deleted
        )
        assert all(
            json.loads(f.payload)["world_book_id"] == plan.mapped("world_book", 41)
            for f in deleted
        )
        assert all(len(f.source_change_ids) == 1 for f in deleted)
    with pytest.raises(ValueError, match="inactive"):
        reads.identity_plan()


def _version_default(db, purpose):
    if purpose == "legacy":
        assert db.update_character_card(
            1,
            {"name": "Legacy supported assistant"},
            expected_version=db.get_character_card_by_id(1)["version"],
        )


def _nine_entities(db, purpose):
    _version_default(db, purpose)
    db.add_note("Target", "target body", note_id="sync-target")
    db.add_note(
        "Source", "plain sync-target [target](note://sync-target)", note_id="sync-note"
    )
    db.add_note("Deleted", "erased note body", note_id="sync-deleted-note")
    db.soft_delete_note("sync-deleted-note", expected_version=1)
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO keywords(id,keyword) VALUES (51,?)", (purpose + " keyword",)
        )
        for table in ("keyword_collections", "chat_dictionaries", "world_books"):
            cursor.execute(
                "INSERT INTO " + table + "(id,name) VALUES (51,?)", (purpose + " name",)
            )
        cursor.execute(
            "INSERT INTO world_book_entries(id,world_book_id,keys,content) VALUES (51,51,'[]','lore body')"
        )
    for identity in ("sync-live", "sync-deleted"):
        db.add_message(
            {
                "id": identity,
                "conversation_id": "conversation",
                "sender": "User",
                "role": "user",
                "content": "old proof body",
            }
        )
    db.update_message("sync-live", {"content": "new proof body"}, expected_version=1)
    db.soft_delete_message("sync-deleted", expected_version=1)

    import json

    from Tests.Chat.test_provider_continuation import _checkpoint
    from tldw_chatbook.Chat.provider_continuation import (
        dump_provider_continuation_json,
        parse_provider_continuation_json,
    )
    from tldw_chatbook.Chat.thinking_blocks import (
        DisplayableThinkingBlock,
        ThinkingEnvelope,
        dump_thinking_blocks_json,
    )

    private = dump_provider_continuation_json(
        parse_provider_continuation_json(json.dumps(_checkpoint()))
    )
    thinking = dump_thinking_blocks_json(
        ThinkingEnvelope(
            (
                DisplayableThinkingBlock(
                    "opaque",
                    0,
                    "deepseek",
                    "history",
                    "responses",
                    "text",
                    "complete",
                    "private thinking",
                ),
            )
        )
    )
    for identity in ("sync-assistant", "sync-private-deleted"):
        db.add_message(
            {
                "id": identity,
                "conversation_id": "conversation",
                "sender": "Assistant",
                "role": "assistant",
                "content": "private assistant body",
                "provider_continuation_json": private,
                "assistant_generation_state": "continuation_active",
                "thinking_blocks_json": thinking,
            }
        )
    db.soft_delete_message("sync-private-deleted", expected_version=1)


def test_nine_native_frontiers_rebuild_current_and_previous_message_proofs(
    tmp_path, monkeypatch
):
    import json
    from contextlib import closing

    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    with content_reads(
        tmp_path,
        monkeypatch,
        lambda connection, purpose: None,
        native_seed=_nine_entities,
    ) as reads:
        plan = reads.integrated_plan()
        assert not plan.blockers
        assert {f.entity for f in plan.sync.frontiers} == {
            "messages",
            "conversations",
            "notes",
            "character_cards",
            "keywords",
            "keyword_collections",
            "chat_dictionaries",
            "world_books",
            "world_book_entries",
        }
        note = next(
            f
            for f in plan.sync.frontiers
            if f.entity == "notes" and f.original_id == "sync-note"
        )
        assert (
            json.loads(note.payload)["content"]
            == "plain sync-target [target](note://"
            + plan.identities.mapped("note", "sync-target")
            + ")"
        )
        deleted_note = next(
            f
            for f in plan.sync.frontiers
            if f.entity == "notes" and f.original_id == "sync-deleted-note"
        )
        assert "erased note body" not in deleted_note.payload
        cid = plan.identities.mapped("conversation", "conversation")
        with closing(CharactersRAGDB(":memory:", "sync-consumer")) as db:
            db.add_conversation({"id": cid})
            for identity in (
                "sync-live",
                "sync-deleted",
                "sync-assistant",
                "sync-private-deleted",
            ):
                row = next(
                    dict(r.values)
                    for r in plan.rows
                    if r.table == "messages" and r.original == (identity,)
                )
                proof = next(
                    f
                    for f in plan.sync.frontiers
                    if f.entity == "messages" and f.original_id == identity
                )
                with db.transaction() as cursor:
                    cursor.execute(
                        "INSERT INTO messages ("
                        + ",".join(row)
                        + ") VALUES ("
                        + ",".join("?" for _ in row)
                        + ")",
                        tuple(row.values()),
                    )
                    if proof.previous_payload is not None:
                        cursor.execute(
                            "INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload) VALUES ('messages',?,?,?,?,?,?)",
                            (
                                proof.mapped_id,
                                proof.previous_operation,
                                proof.previous_timestamp,
                                proof.previous_client_id,
                                proof.version - 1,
                                proof.previous_payload,
                            ),
                        )
                    if proof.operation == "delete":
                        cursor.execute(
                            "DELETE FROM sync_log WHERE entity='messages' AND entity_id=?",
                            (proof.mapped_id,),
                        )
                        cursor.execute(
                            "INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload) VALUES ('messages',?,?,?,?,?,?)",
                            (
                                proof.mapped_id,
                                proof.operation,
                                proof.timestamp,
                                proof.client_id,
                                proof.version,
                                proof.payload,
                            ),
                        )
                reader = (
                    db.read_committed_chat_delete_intent
                    if proof.operation == "delete"
                    else db.read_committed_chat_sync_intent
                )
                native = reader(
                    message_id=proof.mapped_id,
                    message_version=proof.version,
                    payload_hash=proof.payload_hash,
                )
                assert native is not None
                if identity == "sync-live":
                    assert native.base_payload_hash == proof.previous_hash
                current = db.execute_query(
                    "SELECT payload FROM sync_log WHERE entity='messages' AND entity_id=? AND version=?",
                    (proof.mapped_id, proof.version),
                ).fetchall()
                assert len(current) == 1
                assert json.loads(current[0][0]) == json.loads(proof.payload)
            db.prune_sync_log()
            assert (
                db.read_committed_chat_sync_intent(
                    message_id=plan.identities.mapped("message", "sync-live"),
                    message_version=2,
                    payload_hash=next(
                        f.payload_hash
                        for f in plan.sync.frontiers
                        if f.original_id == "sync-live"
                    ),
                )
                is not None
            )


def test_composed_native_trace_and_sync_share_predecode_allowance(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery.test_mobile_trace_plan import _seed_response_trace
    from tldw_chatbook.Backup_Recovery import mobile_sync_plan, mobile_trace_plan
    from tldw_chatbook.Chat.console_trace_read_budget import (
        _TraceReadBudget,
        _TraceReadLimit,
    )

    marker = "fixed-sync-budget-marker:"

    def seed(db, purpose):
        _version_default(db, purpose)
        _seed_response_trace(db, purpose, "budget-trace-a")
        _seed_response_trace(db, purpose, "budget-trace-b")
        if purpose == "legacy":
            for index in range(8):
                db.add_note(
                    "Budget",
                    marker + "x" * (128 * 1024),
                    note_id="budget-note-" + str(index),
                )

    calls, charged, decoded, incidence = [], set(), [], []
    native_trace, native_json, native_object = (
        mobile_trace_plan._plan_traces,
        _TraceReadBudget.json,
        mobile_sync_plan._object,
    )

    native_remember = mobile_trace_plan._TraceInputClosure.remember

    def remembered(self, key):
        before = self.budget._remaining
        retained = native_remember(self, key)
        if retained:
            assert self.budget._remaining < before
            incidence.append(self.budget)
        return retained

    def traced(reads, **kwargs):
        budget = kwargs.get("_read_budget")
        before = None if budget is None else budget._remaining
        result = native_trace(reads, **kwargs)
        calls.append((budget, before, None if budget is None else budget._remaining))
        return result

    def reserved(self, raw):
        native_json(self, raw)
        if marker in raw:
            charged.add((id(self), raw))

    def decoded_object(raw):
        if marker in raw:
            decoded.append((id(calls[0][0]), raw) in charged)
        return native_object(raw)

    monkeypatch.setattr(mobile_trace_plan._TraceInputClosure, "remember", remembered)
    monkeypatch.setattr(mobile_trace_plan, "_plan_traces", traced)
    monkeypatch.setattr(_TraceReadBudget, "json", reserved)
    monkeypatch.setattr(mobile_sync_plan, "_object", decoded_object)
    with content_reads(
        tmp_path, monkeypatch, lambda connection, purpose: None, native_seed=seed
    ) as reads:
        limited = False
        try:
            reads.sync_plan()
        except _TraceReadLimit:
            limited = True
        assert calls and calls[0][0] is not None, (
            "S must pass the same real allowance into native T"
        )
        assert calls[0][2] < calls[0][1]
        assert incidence and all(budget is calls[0][0] for budget in incidence)
        assert limited, (
            "actual accumulated S expansion must exhaust the unchanged allowance"
        )
        assert decoded and all(decoded), (
            "every reached S decode must follow its reservation"
        )
    with pytest.raises(ValueError, match="inactive"):
        reads.sync_plan()


@pytest.mark.parametrize("raw_references", [False, True])
def test_integrated_native_workspace_and_metadata_semantics(
    tmp_path, monkeypatch, raw_references
):
    import json
    import sqlite3

    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from Tests.Backup_Recovery.test_mobile_trace_plan import _seed_response_trace
    from tldw_chatbook.Backup_Recovery import mobile_workspace_plan
    from tldw_chatbook.Canvas.repository import CanvasRepository
    from tldw_chatbook.Chat.console_chat_fork import (
        encode_console_fork_message_metadata,
        parse_console_fork_message_metadata,
    )
    from tldw_chatbook.Chat.console_project_instructions import (
        ProjectInstructionControlState,
        decode_project_context_json,
        encode_project_context_json,
    )
    from tldw_chatbook.Chat.message_metadata import (
        CanvasCardMetadata,
        CanvasCardOriginMetadata,
        MessageMetadata,
    )
    from tldw_chatbook.Workspaces.registry_service import _workspace_from_row

    native_workspace = mobile_workspace_plan._plan_workspaces
    observed_identities = []

    def observed_workspace(reads, **kwargs):
        observed_identities.append(kwargs.get("_identities"))
        return native_workspace(reads, **kwargs)

    monkeypatch.setattr(mobile_workspace_plan, "_plan_workspaces", observed_workspace)

    canvas_id = "11111111-1111-4111-8111-111111111111"
    revision_id = "22222222-2222-4222-8222-222222222222"
    original_seed = mobile_consolidation_fixture._seed

    def seeded(stores, **kwargs):
        def linked(db, legacy, result):
            if not legacy or not raw_references:
                return
            from tldw_chatbook.Backup_Recovery.inventory import read_profile_config
            from tldw_chatbook.Backup_Recovery.profile_paths import user_data_dir

            data = user_data_dir(
                read_profile_config(tmp_path / "profile" / "config.toml")
            )
            data.mkdir(exist_ok=True)
            raw = {
                "tldw_chatbook_personas.json": {
                    "profiles": [
                        {
                            "id": "persona-1",
                            "name": "Private fixture",
                            "is_active": True,
                            "policy_rules": [],
                        }
                    ]
                },
                "mcp_permissions.json": {
                    "schema_version": 1,
                    "kill_switch": False,
                    "profiles": {
                        "default": {"global_default": "ask", "servers": {}},
                        "profile-1": {"servers": {}},
                    },
                },
            }
            for name, value in raw.items():
                path = data / name
                path.write_text(json.dumps(value))
                path.chmod(0o600)
            with sqlite3.connect(stores.workspace_path) as connection:
                connection.execute(
                    "UPDATE workspace_records SET assistant_defaults=? WHERE workspace_id='conflicting-workspace'",
                    (
                        json.dumps(
                            {
                                "assistant_kind": "persona",
                                "assistant_id": "persona-1",
                                "persona_memory_mode": "read_only",
                                "tool_policy_profile_id": "profile-1",
                            }
                        ),
                    ),
                )
                connection.execute(
                    "INSERT INTO workspace_agent_backfill VALUES ('assistant_defaults','2026-10-25')"
                )
            connection.close()

        return original_seed(stores, **kwargs, linked_chat_seed=linked)

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seeded)

    def native(db, purpose):
        _version_default(db, purpose)
        _seed_response_trace(db, purpose, "semantic")
        db.add_conversation(
            {
                "id": "character-owned",
                "title": "Character",
                "assistant_kind": "character",
                "assistant_id": "1",
                "character_id": 1,
            }
        )
        context = encode_project_context_json(
            ProjectInstructionControlState(
                True, "binding", "original-fingerprint", "original-notice"
            )
        )
        db.set_conversation_console_project_context("conversation", context)
        assert db.get_conversation_console_project_context("conversation") == context
        canvas = CanvasRepository(db).create_canvas(
            "semantic",
            canvas_id=canvas_id,
            revision_id=revision_id,
            title="Exact canvas",
            source="<p>opaque-run is prose</p>",
            runtime_profile="canvas-v1",
            actor_kind="assistant",
            origin_message_id="semantic-answer",
            origin_turn_id="opaque-canvas-token",
            created_at="2026-10-25T00:00:00Z",
        )
        metadata = MessageMetadata(
            canvas_cards=(
                CanvasCardMetadata(
                    canvas_id,
                    revision_id,
                    "Exact canvas",
                    1,
                    canvas.revision.content_sha256,
                    "updated",
                    CanvasCardOriginMetadata("semantic-answer", "opaque-canvas-token"),
                    True,
                ),
            )
        )
        assert db.update_message_metadata_local("semantic-answer", metadata.to_json())
        db.add_message(
            {
                "id": "fork-copy",
                "conversation_id": "semantic",
                "sender": "User",
                "role": "user",
                "content": "fork body",
                "metadata_json": encode_console_fork_message_metadata(
                    "complete", "exact label", "semantic-user"
                ),
            }
        )

    with content_reads(
        tmp_path, monkeypatch, lambda connection, purpose: None, native_seed=native
    ) as reads:
        plan = reads.integrated_plan()
        assert not plan.blockers, plan.blockers
        assert observed_identities == [plan.identities]
        assert observed_identities[0] is plan.identities
        rows = {(r.table, r.original): r for r in plan.rows}
        assert not any(
            p in {"I:binding", "T:canvas_turn", "T:trace_turn", "T:character_authority"}
            for r in plan.rows
            for p in r.required_proofs
        )
        mapped = plan.identities.mapped
        context = decode_project_context_json(
            dict(rows["conversations", ("conversation",)].values)[
                "console_project_context_json"
            ]
        )
        assert context.working_folder_binding_id == mapped("binding", "binding")
        cards = MessageMetadata.from_json(
            dict(rows["messages", ("semantic-answer",)].values)["metadata_json"]
        ).canvas_cards
        assert cards[0].origin.run_id == "opaque-canvas-token"
        assert cards[0].origin.message_id == mapped("message", "semantic-answer")
        assert cards[0].revision_id == mapped("canvas_revision", revision_id)
        fork = parse_console_fork_message_metadata(
            dict(rows["messages", ("fork-copy",)].values)["metadata_json"]
        )
        assert fork == ("complete", "exact label", mapped("message", "semantic-user"))
        assert dict(rows["conversations", ("character-owned",)].values)[
            "assistant_authority_id"
        ]
        if raw_references:
            record = dict(rows["workspace_records", ("conflicting-workspace",)].values)
            assert (
                _workspace_from_row(record).assistant_defaults.assistant_id
                == "persona-1"
            )
            references = plan.workspace.authority_references
            assert {(p.kind, p.reference_id, p.owner_id) for p in references} == {
                ("persona", "persona-1", "personas"),
                ("permission_profile", "profile-1", "mcp.permissions"),
            }
            assert all(
                len(p.member_digest) == len(p.provenance_digest) == 64
                for p in references
            )
        assert any(
            d.requirement == "atomic_exact_binding_hold_and_native_unavailability"
            for d in plan.pending_duties
        )
    with pytest.raises(ValueError, match="inactive"):
        reads.integrated_plan()


def test_native_deleted_predecessor_rejects_foreign_generation_state(
    tmp_path, monkeypatch
):
    import json

    def native(db, purpose):
        _version_default(db, purpose)
        db.add_message(
            {
                "id": "revived",
                "conversation_id": "conversation",
                "sender": "User",
                "role": "user",
                "content": "old body",
            }
        )
        db.soft_delete_message("revived", expected_version=1)
        db.restore_message_subtree((("revived", 2),))
        if purpose == "legacy":
            with db.transaction() as cursor:
                raw = cursor.execute(
                    "SELECT payload FROM sync_log WHERE entity='messages' AND entity_id='revived' AND version=2"
                ).fetchone()[0]
                payload = json.loads(raw)
                payload["assistant_generation_state"] = "complete"
                cursor.execute(
                    "UPDATE sync_log SET payload=? WHERE entity='messages' AND entity_id='revived' AND version=2",
                    (json.dumps(payload),),
                )
            from tldw_chatbook.Sync_Interop.hashing import canonical_payload_hash

            assert (
                db.read_committed_chat_sync_intent(
                    message_id="revived",
                    message_version=3,
                    payload_hash=canonical_payload_hash(
                        {
                            "role": "user",
                            "content": "old body",
                            "assistant_generation_state": None,
                        }
                    ),
                )
                is None
            )

    with content_reads(
        tmp_path, monkeypatch, lambda connection, purpose: None, native_seed=native
    ) as reads:
        plan = reads.sync_plan()
        assert any(
            b.table == "messages"
            and b.original == ("revived",)
            and b.reason == "generation_state_unrepresentable"
            for b in plan.blockers
        )
        assert not any(
            f.entity == "messages" and f.original_id == "revived"
            for f in plan.frontiers
        )


def test_workspace_private_identity_binding_refuses_mismatches(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery.mobile_workspace_plan import _plan_workspaces

    with content_reads(
        tmp_path, monkeypatch, lambda connection, purpose: None
    ) as reads:
        # The ordinary public no-argument entry still constructs its own full map.
        workspace = reads.workspace_plan()
        assert not workspace.blockers
        identity = workspace.identities
        for invalid in (
            object(),
            replace(identity, source_fingerprint="foreign"),
            replace(identity, source_manifest_digest="0" * 64),
            replace(identity, canonical_manifest_digest="0" * 64),
        ):
            with pytest.raises(
                ValueError, match="mobile_workspace_identity_binding_changed"
            ):
                _plan_workspaces(reads, _identities=invalid)
    with pytest.raises(ValueError, match="inactive"):
        _plan_workspaces(reads, _identities=identity)


def test_native_retired_entry_keeps_each_parent_proof_and_refuses_missing_parent(
    tmp_path, monkeypatch
):
    import json

    def seed(connection, purpose):
        for identity in (61, 62, 63):
            connection.execute(
                "INSERT INTO world_books(id,name,client_id) VALUES (?,?, 'history')",
                (identity, purpose + str(identity)),
            )
        for parent in (61, 62):
            connection.execute(
                "INSERT INTO world_book_entries(id,world_book_id,keys,content) VALUES (0,?,'[]','erased')",
                (parent,),
            )
            connection.execute("DELETE FROM world_book_entries WHERE id=0")
        connection.execute(
            "INSERT INTO world_book_entries(id,world_book_id,keys,content) VALUES (0,62,'[]','new live')"
        )
        connection.execute(
            "INSERT INTO world_book_entries(id,world_book_id,keys,content) VALUES (63,63,'[]','erased parent')"
        )
        connection.execute("DELETE FROM world_book_entries WHERE id=63")
        connection.execute("DELETE FROM world_books WHERE id=63")

    with content_reads(
        tmp_path, monkeypatch, seed, native_seed=_version_default
    ) as reads:
        plan = reads.sync_plan()
        deleted = [
            f
            for f in plan.frontiers
            if f.entity == "world_book_entries"
            and f.operation == "delete"
            and f.original_id == 0
        ]
        assert len(deleted) == 2
        assert len({f.source_change_ids for f in deleted}) == 2
        assert {json.loads(f.payload)["world_book_id"] for f in deleted} == {
            plan.identities.mapped("world_book", p) for p in (61, 62)
        }
        live = next(
            f
            for f in plan.frontiers
            if f.entity == "world_book_entries"
            and f.operation == "create"
            and f.original_id == 0
        )
        assert all(f.mapped_id == live.mapped_id for f in deleted)
        assert all("erased" not in f.payload for f in deleted)
        assert any(
            b.table == "sync_log"
            and b.reason == "sync_dependency_unproven"
            and b.dependency_domain == "world_book"
            and b.dependency_id == 63
            for b in plan.blockers
        )
        assert not any(
            f.entity == "world_book_entries" and f.original_id == 63
            for f in plan.frontiers
        )
    with pytest.raises(ValueError, match="inactive"):
        reads.sync_plan()


def test_exact_frontier_and_metadata_refusals_close_dependencies(tmp_path, monkeypatch):
    from tldw_chatbook.Canvas.repository import CanvasRepository
    from tldw_chatbook.Chat.console_chat_fork import (
        encode_console_fork_message_metadata,
    )
    from tldw_chatbook.Chat.message_metadata import (
        CanvasCardMetadata,
        CanvasCardOriginMetadata,
        MessageMetadata,
    )

    def native(db, purpose):
        # The fixture's raw default-card rename deliberately has no version bump.
        db.add_message(
            {
                "id": "bad-previous",
                "conversation_id": "conversation",
                "sender": "User",
                "role": "user",
                "content": "first",
            }
        )
        db.update_message("bad-previous", {"content": "second"}, expected_version=1)
        db.add_message(
            {
                "id": "bad-fork",
                "conversation_id": "conversation",
                "sender": "User",
                "role": "user",
                "content": "fork",
                "metadata_json": encode_console_fork_message_metadata(
                    "complete", "label", "bad-previous"
                ),
            }
        )
        db.add_message(
            {
                "id": "bad-canvas",
                "conversation_id": "conversation",
                "sender": "Assistant",
                "role": "assistant",
                "content": "canvas",
            }
        )
        canvas = CanvasRepository(db).create_canvas(
            "conversation",
            canvas_id="11111111-1111-4111-8111-111111111111",
            revision_id="22222222-2222-4222-8222-222222222222",
            title="Canvas",
            source="<p>body</p>",
            runtime_profile="canvas-v1",
            actor_kind="assistant",
            origin_message_id="bad-canvas",
            origin_turn_id="true-opaque-token",
            created_at="2026-10-25T00:00:00Z",
        )
        card = CanvasCardMetadata(
            canvas.revision.canvas_id,
            canvas.revision.revision_id,
            "Canvas",
            1,
            canvas.revision.content_sha256,
            "updated",
            CanvasCardOriginMetadata("bad-canvas", "wrong-opaque-token"),
            True,
        )
        db.update_message_metadata_local(
            "bad-canvas", MessageMetadata(canvas_cards=(card,)).to_json()
        )
        if purpose == "legacy":
            with db.transaction() as cursor:
                cursor.execute(
                    "DELETE FROM sync_log WHERE entity='messages' AND entity_id='bad-previous' AND version=1"
                )

    with content_reads(
        tmp_path, monkeypatch, lambda connection, purpose: None, native_seed=native
    ) as reads:
        plan = reads.integrated_plan()
        assert any(
            b.table == "character_cards"
            and b.original == (1,)
            and b.reason == "current_proof_ambiguous"
            for b in plan.blockers
        )
        assert any(
            b.table == "messages"
            and b.original == ("bad-previous",)
            and b.reason == "previous_proof_missing_or_ambiguous"
            for b in plan.blockers
        )
        assert any(
            b.table == "messages"
            and b.original == ("bad-fork",)
            and b.reason == "integrated_semantic_dependency_unproven"
            for b in plan.blockers
        )
        assert any(
            b.table == "messages"
            and b.original == ("bad-canvas",)
            and b.reason == "integrated_semantic_dependency_unproven"
            for b in plan.blockers
        )
        assert not any(
            r.table == "messages"
            and r.original in {("bad-previous",), ("bad-fork",), ("bad-canvas",)}
            for r in plan.rows
        )
        assert all(
            "first" not in repr(b) and "true-opaque-token" not in repr(b)
            for b in plan.blockers
        )
    with pytest.raises(ValueError, match="inactive"):
        reads.integrated_plan()


@pytest.mark.parametrize("control", range(9))
def test_integrated_later_sync_loss_closes_native_call_and_canvas_inputs(
    tmp_path, monkeypatch, control
):
    from tldw_chatbook.Backup_Recovery import mobile_sync_plan

    budgets = []
    native_sync = mobile_sync_plan._plan_sync

    def observed_sync(reads, **kwargs):
        budgets.append(kwargs["_read_budget"])
        return native_sync(reads, **kwargs)

    monkeypatch.setattr(mobile_sync_plan, "_plan_sync", observed_sync)

    from Tests.Backup_Recovery.test_mobile_trace_plan import _seed_response_trace
    from tldw_chatbook.Canvas.repository import CanvasRepository

    seeded = {}

    def native(db, purpose):
        _version_default(db, purpose)
        bad = _seed_response_trace(db, purpose, "later-lost")
        good = _seed_response_trace(db, purpose + "-other", "later-retained")
        db.set_message_attachments(
            good[2],
            [
                {
                    "position": 1,
                    "data": b"qualified attachment",
                    "mime_type": "application/octet-stream",
                    "display_name": "historical attachment",
                }
            ],
        )
        canvases = []
        for call, canvas_id, revision_id in (
            (
                bad,
                "11111111-1111-4111-8111-111111111111",
                "22222222-2222-4222-8222-222222222222",
            ),
            (
                good,
                "33333333-3333-4333-8333-333333333333",
                "44444444-4444-4444-8444-444444444444",
            ),
        ):
            document = CanvasRepository(db).create_canvas(
                call[1].removesuffix("-user"),
                canvas_id=canvas_id,
                revision_id=revision_id,
                title="Qualified",
                source="<p>qualified native history</p>",
                runtime_profile="canvas-v1",
                actor_kind="assistant",
                origin_message_id=call[2],
                origin_turn_id="opaque-history",
                created_at="2026-10-25T00:00:00Z",
            )
            canvases.append(document)
        assert db.update_message(
            bad[2], {"content": "changed response"}, expected_version=1
        )
        if purpose == "legacy":
            with db.transaction() as cursor:
                cursor.execute(
                    "DELETE FROM sync_log WHERE entity='messages' AND entity_id=? AND version=1",
                    (bad[2],),
                )
        seeded[purpose] = bad, good, canvases

    with content_reads(
        tmp_path, monkeypatch, lambda connection, purpose: None, native_seed=native
    ) as reads:
        plan = reads.integrated_plan()
        bad, good, canvases = seeded["legacy"]
        assert any(
            b.table == "messages"
            and b.original == (bad[2],)
            and b.reason == "previous_proof_missing_or_ambiguous"
            for b in plan.blockers
        )
        rows = {(r.table, r.original): r for r in plan.rows}
        assert ("console_trace_calls", (good[0],)) in rows
        assert ("console_trace_calls", (bad[0],)) not in rows
        assert ("canvas_documents", (canvases[0].revision.canvas_id,)) not in rows
        assert ("canvas_documents", (canvases[1].revision.canvas_id,)) in rows
        assert ("canvas_revisions", (canvases[1].revision.revision_id,)) in rows
        assert not any(
            f.entity == "messages" and f.original_id == bad[2]
            for f in plan.sync.frontiers
        )
        assert all((d.table, d.original) in rows for d in plan.semantic_discharges)
        from dataclasses import FrozenInstanceError

        from tldw_chatbook.Backup_Recovery.mobile_integrated_closure import (
            _close_qualified_inputs,
        )

        trace = plan.sync.trace
        proofs = {p.original_call_id: p for p in trace.input_incidence}
        assert ("messages", (bad[2],)) in proofs[bad[0]].inputs
        assert (
            proofs[good[0]].source_manifest_digest
            == plan.identities.source_manifest_digest
        )
        assert (
            proofs[good[0]].canonical_manifest_digest
            == plan.identities.canonical_manifest_digest
        )
        with pytest.raises(FrozenInstanceError):
            proofs[good[0]].mapped_call_id = "changed"
        qualified = trace.content.rows + trace.rows
        bad_call = next(
            r
            for r in trace.rows
            if r.table == "console_trace_calls" and r.original == (bad[0],)
        )
        mapped_header = dict(bad_call.values)["request_header_id"]
        header = next(
            r
            for r in trace.rows
            if r.table == "console_trace_request_headers"
            and dict(r.values)["header_id"] == mapped_header
        )
        assert (header.table, header.original) in proofs[bad[0]].inputs
        missing_header = tuple(
            r
            for r in qualified
            if (r.table, r.original) != (header.table, header.original)
        )
        if control == 0:
            blockers = []
            closed = _close_qualified_inputs(
                trace, missing_header, blockers, _read_budget=budgets[0]
            )
            assert not any(
                r.table == "console_trace_calls" and r.original == (bad[0],)
                for r in closed
            )
            assert any(
                r.table == "console_trace_calls" and r.original == (good[0],)
                for r in closed
            )
            assert any(b.reason == "integrated_native_input_lost" for b in blockers)

        good_proof = proofs[good[0]]
        attachments = [
            key for key in good_proof.inputs if key[0] == "message_attachments"
        ]
        assert len(attachments) == 1
        if control == 1:
            blockers = []
            closed = _close_qualified_inputs(
                trace,
                tuple(r for r in qualified if (r.table, r.original) != attachments[0]),
                blockers,
                _read_budget=budgets[0],
            )
            assert not any(
                r.table == "console_trace_calls" and r.original == (good[0],)
                for r in closed
            )
            assert any(
                r.table == "console_trace_calls" and r.original == (bad[0],)
                for r in closed
            )
            assert any(b.reason == "integrated_native_input_lost" for b in blockers)
        for index, invalid in enumerate(
            (
                (),
                (good_proof, good_proof),
                (replace(good_proof, source_manifest_digest="0" * 64),),
                (replace(good_proof, canonical_manifest_digest="0" * 64),),
                (replace(good_proof, mapped_call_id="wrong-call"),),
                (replace(good_proof, inputs=()),),
            )
        ):
            if control != index + 2:
                continue
            malformed = replace(trace, input_incidence=(proofs[bad[0]],) + invalid)
            blockers = []
            closed = _close_qualified_inputs(
                malformed, qualified, blockers, _read_budget=budgets[0]
            )
            assert not any(
                r.table == "console_trace_calls" and r.original == (good[0],)
                for r in closed
            )
            assert any(
                r.table == "console_trace_calls" and r.original == (bad[0],)
                for r in closed
            )
            assert any(
                b.reason == "integrated_native_incidence_unproved" for b in blockers
            )
        if control == 8:
            changed = tuple(
                replace(
                    r,
                    values=tuple(
                        (dict(r.values) | {"content": "unqualified change"}).items()
                    ),
                )
                if r.table == "messages" and r.original == (good[2],)
                else r
                for r in qualified
            )
            blockers = []
            closed = _close_qualified_inputs(
                trace, changed, blockers, _read_budget=budgets[0]
            )
            assert not any(
                r.table == "console_trace_calls" and r.original == (good[0],)
                for r in closed
            )
            assert any(
                r.table == "console_trace_calls" and r.original == (bad[0],)
                for r in closed
            )
            assert any(b.reason == "integrated_native_input_changed" for b in blockers)
    with pytest.raises(ValueError, match="inactive"):
        reads.integrated_plan()
