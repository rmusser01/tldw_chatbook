"""Workspace transformations consume authenticated captures without owner writes."""

import json
import sqlite3
from contextlib import contextmanager
from threading import Event

import pytest

from Tests.Backup_Recovery.test_mobile_captured_sqlite import _fingerprints
from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import _capture_fixture


@contextmanager
def workspace_reads(
    tmp_path,
    monkeypatch,
    *,
    source_complete=False,
    target_complete=False,
    exempt=False,
    audit="{}",
    dependencies=None,
    entities=False,
    pending_citation=False,
    archived=False,
    default_conflict=False,
    malformed_binding=False,
    unknown_marker=False,
    unsupported_study=False,
    reserved_suffix=False,
    default_sidecars=None,
):
    from tldw_chatbook.Backup_Recovery.inventory import discover, read_profile_config
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        _preview_selected_mobile,
    )
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )
    from tldw_chatbook.Backup_Recovery.profile_paths import legacy_mobile_dir

    with _capture_fixture(tmp_path, monkeypatch) as values:
        session, journal, config, stores, _, _, _ = values
        source = legacy_mobile_dir(read_profile_config(config))
        for path, completed in (
            (source / "workspaces.db", source_complete),
            (stores.workspace_path, target_complete),
        ):
            with sqlite3.connect(path) as connection:
                if default_sidecars is not None:
                    state = default_sidecars[
                        0 if path == source / "workspaces.db" else 1
                    ]
                    for table in ("workspace_change_review", "workspace_rag_scopes"):
                        connection.execute(
                            "DELETE FROM "
                            + table
                            + " WHERE workspace_id='workspace-default'"
                        )
                    if state is not None:
                        connection.execute(
                            "INSERT INTO workspace_change_review VALUES ('workspace-default',?,'2026-10-25')",
                            (int(state),),
                        )
                        connection.execute(
                            "INSERT INTO workspace_rag_scopes VALUES ('workspace-default',?,'2026-10-25')",
                            (
                                json.dumps(
                                    {
                                        "version": 2,
                                        "updated_at": "2026-10-25",
                                        "items": [],
                                        "empty_is_scoped": state,
                                    }
                                ),
                            ),
                        )
                if completed:
                    connection.execute(
                        "INSERT INTO workspace_agent_backfill VALUES ('assistant_defaults', '2026-10-25')"
                    )
                if path == source / "workspaces.db":
                    if archived:
                        connection.execute(
                            "UPDATE workspace_records SET archived=1 WHERE workspace_id='conflicting-workspace'"
                        )
                    if default_conflict:
                        connection.execute(
                            "UPDATE workspace_records SET description='source-only Default description' WHERE workspace_id='workspace-default'"
                        )
                        connection.execute(
                            "UPDATE workspace_runtime_bindings SET workspace_id='workspace-default'"
                        )
                    if malformed_binding:
                        connection.execute(
                            "UPDATE workspace_runtime_bindings SET metadata_json=?",
                            (json.dumps(["private invalid metadata"]),),
                        )
                    if unknown_marker:
                        connection.execute(
                            "INSERT INTO workspace_agent_backfill VALUES ('future-marker','2026-10-25')"
                        )
                    if reserved_suffix:
                        connection.execute(
                            "INSERT INTO workspace_records SELECT 'reserved-suffix',' Shared title (2) ',description,authority,sync_status,0,0,created_at,updated_at,assistant_defaults,assistant_defaults_explicit_none FROM workspace_records WHERE workspace_id='conflicting-workspace'"
                        )
                    if exempt:
                        connection.execute(
                            "UPDATE workspace_records SET assistant_defaults_explicit_none=1 WHERE workspace_id='conflicting-workspace'"
                        )
                    connection.execute(
                        "INSERT INTO workspace_handoff_audit VALUES ('audit','conflicting-workspace','export','complete','private summary',?,'2026-10-25')",
                        (audit,),
                    )
            connection.close()
        if unsupported_study:
            with sqlite3.connect(source / "chat.db") as connection:
                connection.execute(
                    "INSERT INTO topics(id,title,content,created_by,last_modified_by) VALUES ('standalone-topic','private topic','private study body','fixture','fixture')"
                )
            connection.close()
        if entities:
            for path in (source / "chat.db", stores.chat_path):
                with sqlite3.connect(path) as connection:
                    for identity in (1, 2) if path == source / "chat.db" else (1,):
                        connection.execute(
                            "INSERT INTO keywords(id,keyword) VALUES (?,?)",
                            (identity, "tag-" + str(identity)),
                        )
                        for table in (
                            "keyword_collections",
                            "chat_dictionaries",
                            "world_books",
                        ):
                            connection.execute(
                                "INSERT INTO " + table + "(id,name) VALUES (?,?)",
                                (identity, "title-" + str(identity)),
                            )
                        connection.execute(
                            "INSERT INTO world_book_entries(id,world_book_id,keys,content) VALUES (?,1,'[]','private lore')",
                            (identity,),
                        )
                    for revision in (1, 2):
                        connection.execute(
                            "INSERT INTO console_response_rule_revisions VALUES ('same-rule',?,'{}',1,'{}','2026-10-25')",
                            (revision,),
                        )
                    connection.execute(
                        "INSERT INTO canvas_documents(id,conversation_id,created_at) VALUES ('30000000-0000-4000-8000-000000000001','conversation','2026-10-25')"
                    )
                connection.close()
        if pending_citation:
            from Tests.Chat.test_citation_trace_repository import _sealed_write

            trace = _sealed_write().trace
            with sqlite3.connect(source / "chat.db") as connection:
                for profile in ("source-profile", "other-profile"):
                    connection.execute(
                        "INSERT INTO rag_citation_traces(profile_id,trace_id,schema_version,request_id,generation_id,origin_scope_id,origin,lifecycle,completeness_at_seal,selected_attempt_id,policy_version,aggregate_json,visibility_state,created_at,sealed_at) VALUES (?,?,1,?,?,?,'local','sealed','complete',?,?,?,'active',?,?)",
                        (
                            profile,
                            trace.trace_id,
                            trace.request_id,
                            trace.generation_id,
                            profile,
                            trace.selected_attempt_id,
                            trace.policy_version,
                            trace.model_dump_json(),
                            trace.created_at.isoformat(),
                            trace.sealed_at.isoformat(),
                        ),
                    )
            connection.close()
        if dependencies is not None:
            for path in (source / "chat.db", stores.chat_path):
                with sqlite3.connect(path) as connection:
                    connection.execute(
                        "INSERT INTO notes(id,title,content) VALUES ('scoped-note','same title',?)",
                        (
                            "source body"
                            if path == source / "chat.db"
                            else "canonical body",
                        ),
                    )
                connection.close()
            with sqlite3.connect(source / "workspaces.db") as connection:
                for kind, identity in (
                    ("conversation", "conversation"),
                    (dependencies, "scoped-note"),
                ):
                    connection.execute(
                        "INSERT INTO workspace_memberships VALUES (?, 'conflicting-workspace', ?, ?, 'source', 'reference', 'private title', '2026-10-25')",
                        ("member-" + kind, kind, identity),
                    )
                connection.execute(
                    "INSERT INTO workspace_rag_scopes VALUES ('conflicting-workspace', ?, '2026-10-25')",
                    (
                        json.dumps(
                            {
                                "version": 2,
                                "updated_at": "2026-10-25",
                                "empty_is_scoped": True,
                                "items": [
                                    {
                                        "source_type": dependencies,
                                        "source_id": "scoped-note",
                                    }
                                ],
                            }
                        ),
                    ),
                )
                connection.execute(
                    "INSERT INTO workspace_change_review VALUES ('conflicting-workspace',0,'2026-10-25')"
                )
            connection.close()
        with session._discovery_reads(private_sqlite=True):
            inventory = discover((config,))
            preview = _preview_selected_mobile(config, config)
        archives = tmp_path / "archives"
        archives.mkdir(mode=0o700)
        with _captured_mobile(
            session,
            journal,
            preview,
            inventory.scope_digest,
            destination=archives,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as captured:
            before = _fingerprints(tmp_path / "profile")
            with captured.sqlite_reads(
                password=b"test-private-mobile-rollback", cancel=Event()
            ) as reads:
                yield reads
            assert _fingerprints(tmp_path / "profile") == before


def test_workspace_plan_native_consumers_preserve_default_selection_and_binding_meaning(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Workspaces.registry_service import (
        _runtime_binding_from_row,
        _workspace_from_row,
    )

    with workspace_reads(tmp_path, monkeypatch, audit=" { } ") as reads:
        plan = reads.workspace_plan()
        assert plan == reads.workspace_plan()
        assert not plan.blockers
        records = [
            dict(row.values) for row in plan.rows if row.table == "workspace_records"
        ]
        assert len(records) == 1  # Canonical Default remains its actual native row.
        workspace = _workspace_from_row(records[0])
        assert workspace.name == "Shared title (2)"
        assert not workspace.active
        assert workspace.assistant_defaults is None
        assert not workspace.assistant_defaults_explicit_none
        binding_row = next(
            dict(row.values)
            for row in plan.rows
            if row.table == "workspace_runtime_bindings"
        )
        binding = _runtime_binding_from_row(binding_row)
        columns, originals = reads.page(
            "legacy", "workspaces", "workspace_runtime_bindings"
        )
        original = dict(zip(columns, originals[0], strict=True))
        assert binding.locator == original["locator"]
        assert binding_row["metadata_json"] == original["metadata_json"]
        assert binding.metadata["exclusions"][0]["path"] == "private"
        assert plan.binding_holds == (binding.binding_id,)
        assert binding.workspace_id == workspace.workspace_id
        audit = next(
            dict(row.values)
            for row in plan.rows
            if row.table == "workspace_handoff_audit"
        )
        assert audit["workspace_id"] == workspace.workspace_id
        assert audit["manifest_json"] == " { } "
        assert plan.names == (
            ("conflicting-workspace", "Shared title", "Shared title (2)"),
        )
        assert all(row.table != "workspace_agent_backfill" for row in plan.rows)


@pytest.mark.parametrize(
    "source_complete,target_complete,exempt,blocked",
    (
        (True, False, False, True),
        (False, True, False, True),
        (True, True, False, False),
        (True, False, True, False),
    ),
)
def test_workspace_backfill_frontier_uses_actual_native_eligibility(
    tmp_path, monkeypatch, source_complete, target_complete, exempt, blocked
):
    with workspace_reads(
        tmp_path,
        monkeypatch,
        source_complete=source_complete,
        target_complete=target_complete,
        exempt=exempt,
    ) as reads:
        plan = reads.workspace_plan()
        assert bool(plan.blockers) is blocked
        if blocked:
            assert [(row.table, row.identity, row.reason) for row in plan.blockers] == [
                (
                    "workspace_records",
                    "conflicting-workspace",
                    "backfill_frontier_conflict",
                )
            ]
        else:
            row = next(
                dict(row.values)
                for row in plan.rows
                if row.table == "workspace_records"
            )
            assert row["assistant_defaults_explicit_none"] == int(exempt)


def test_nonempty_audit_refuses_exact_row_without_guessing_manifest_grammar(
    tmp_path, monkeypatch
):
    with workspace_reads(
        tmp_path,
        monkeypatch,
        audit=json.dumps({"workspace_id": "conflicting-workspace"}),
    ) as reads:
        plan = reads.workspace_plan()
        assert [(row.table, row.identity, row.reason) for row in plan.blockers] == [
            ("workspace_handoff_audit", "audit", "audit_manifest_unsupported")
        ]
        assert plan.blockers[0].workspace_id == "conflicting-workspace"


def test_workspace_membership_and_rag_use_one_typed_mapping(tmp_path, monkeypatch):
    from tldw_chatbook.Chat.rag_scope import parse_scope
    from tldw_chatbook.Workspaces.registry_service import _membership_from_row

    with workspace_reads(tmp_path, monkeypatch, dependencies="note") as reads:
        plan = reads.workspace_plan()
        assert not plan.blockers
        rows = [
            dict(row.values)
            for row in plan.rows
            if row.table == "workspace_memberships"
        ]
        assert len(rows) == 2
        for row in rows:
            member = _membership_from_row(row)
            original = (
                "conversation" if member.item_type == "conversation" else "scoped-note"
            )
            assert member.item_id == plan.identities.mapped(member.item_type, original)
            assert member.item_id != original
            assert member.workspace_id == plan.identities.mapped(
                "workspace", "conflicting-workspace"
            )
        scope = next(
            dict(row.values) for row in plan.rows if row.table == "workspace_rag_scopes"
        )
        native = parse_scope(json.loads(scope["payload"]))
        assert native.empty_is_scoped
        assert native.items[0].source_id == plan.identities.mapped(
            "note", "scoped-note"
        )
        assert (
            next(
                dict(row.values)["enabled"]
                for row in plan.rows
                if row.table == "workspace_change_review"
            )
            == 0
        )


def test_external_media_edges_block_exact_private_typed_reference(
    tmp_path, monkeypatch
):
    with workspace_reads(tmp_path, monkeypatch, dependencies="media") as reads:
        plan = reads.workspace_plan()
        assert {
            (item.table, item.dependency_domain, item.dependency_id)
            for item in plan.blockers
        } == {
            ("workspace_memberships", "media", "scoped-note"),
            ("workspace_rag_scopes", "media", "scoped-note"),
        }
        assert all(
            item.workspace_id == "conflicting-workspace" for item in plan.blockers
        )
        assert all("private title" not in repr(item) for item in plan.blockers)
        # Unrelated Conversation membership remains a valid planned row.
        assert (
            len([row for row in plan.rows if row.table == "workspace_memberships"]) == 1
        )
        assert any(row.table == "workspace_rag_scopes" for row in plan.source_rows)


def test_archived_workspace_is_backfill_exempt_and_not_renamed(tmp_path, monkeypatch):
    with workspace_reads(
        tmp_path, monkeypatch, source_complete=True, archived=True
    ) as reads:
        plan = reads.workspace_plan()
        assert not plan.blockers
        row = next(
            dict(row.values) for row in plan.rows if row.table == "workspace_records"
        )
        assert row["archived"] == 1 and row["name"] == "Shared title"


def test_default_content_and_binding_conflicts_are_exact_prewrite_blockers(
    tmp_path, monkeypatch
):
    with workspace_reads(tmp_path, monkeypatch, default_conflict=True) as reads:
        plan = reads.workspace_plan()
        assert {(row.identity, row.reason) for row in plan.blockers} == {
            ("workspace-default", "default_content_conflict"),
            ("binding", "default_binding_conflict"),
        }
        assert not plan.binding_holds


def test_named_workspace_collision_reserves_all_source_suffix_names(
    tmp_path, monkeypatch
):
    with workspace_reads(tmp_path, monkeypatch, reserved_suffix=True) as reads:
        plan = reads.workspace_plan()
        assert not plan.blockers
        assert (
            "conflicting-workspace",
            "Shared title",
            "Shared title (3)",
        ) in plan.names
        assert (
            "reserved-suffix",
            " Shared title (2) ",
            " Shared title (2) ",
        ) in plan.names


def test_unknown_frontier_and_malformed_binding_do_not_silently_drop_meaning(
    tmp_path, monkeypatch
):
    with workspace_reads(
        tmp_path, monkeypatch, unknown_marker=True, malformed_binding=True
    ) as reads:
        plan = reads.workspace_plan()
        assert {(row.identity, row.reason) for row in plan.blockers} == {
            ("future-marker", "backfill_key_unknown"),
            ("binding", "binding_unrepresentable"),
        }
        assert not plan.binding_holds


@pytest.mark.parametrize(
    "source_state,target_state",
    ((True, None), (None, True), (None, None), (False, None), (None, False)),
)
def test_default_sidecars_compare_both_native_absence_states(
    tmp_path, monkeypatch, source_state, target_state
):
    from types import SimpleNamespace

    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    with workspace_reads(
        tmp_path, monkeypatch, default_sidecars=(source_state, target_state)
    ) as reads:
        meanings = []
        for purpose in ("legacy", "canonical"):
            connection = reads._connections[purpose, "workspaces"][0]

            @contextmanager
            def native_connection(connection=connection):
                original_factory = connection.row_factory
                connection.row_factory = sqlite3.Row
                try:
                    yield connection
                finally:
                    connection.row_factory = original_factory

            registry = LocalWorkspaceRegistryService(
                SimpleNamespace(connection=native_connection)
            )
            enabled = registry.change_review_enabled("workspace-default")
            scope = registry.get_workspace_scope("workspace-default")
            meanings.append((enabled, scope is not None))
        assert meanings == [
            (bool(source_state), source_state is True),
            (bool(target_state), target_state is True),
        ]
        plan = reads.workspace_plan()
        expected = (
            set()
            if meanings[0] == meanings[1]
            else {
                (table, "workspace-default", "workspace-default")
                for table in ("workspace_change_review", "workspace_rag_scopes")
            }
        )
        assert {
            (row.table, row.identity, row.workspace_id) for row in plan.blockers
        } == expected
        assert all(
            dict(row.values).get("workspace_id") != "workspace-default"
            for row in plan.rows
            if row.table in {"workspace_change_review", "workspace_rag_scopes"}
        )
        assert any(row.table == "workspace_records" for row in plan.rows)
