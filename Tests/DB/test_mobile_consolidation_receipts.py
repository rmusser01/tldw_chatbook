"""Committed owner census and historical definition projection, under native DBs."""

import json

import pytest

from Tests.agent_runs_owner import owned_agent_runs_db
from tldw_chatbook.Agents.agent_models import AgentDefinition
from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget


def _codec():
    from tldw_chatbook.DB import mobile_consolidation_receipts

    return mobile_consolidation_receipts


def _receipt(entries, owner="runs", *, command_count=0, source_version=25):
    return _codec()._encode_owner_receipt(
        owner,
        entries
        + (
            (
                {
                    "version": 1,
                    "kind": "provenance",
                    "data": {
                        "family": "snapshot_allocator",
                        "record": {
                            "source_sequence": None,
                            "canonical_sequence": None,
                            "required_frontier": 0,
                        },
                    },
                },
            )
            if owner == "runs"
            else ()
        ),
        source_manifest_sha256="a" * 64,
        canonical_manifest_sha256="b" * 64,
        source_runs_version=source_version if owner == "runs" else None,
        command_history_count=command_count if owner == "runs" else None,
        budget=_TraceReadBudget(),
    )


def _install(db, entries, *, history=(), source_version=25):
    encoded = _receipt(
        entries, command_count=len(history), source_version=source_version
    )
    from tldw_chatbook.DB.console_command_history import (
        _commit_command_history,
        _prepare_command_history,
    )

    passive = _prepare_command_history(
        "operation", "c" * 64, history, _TraceReadBudget()
    )
    with db.transaction() as conn:
        _commit_command_history(db, conn, "operation", "c" * 64, passive)
        for key, raw in encoded:
            conn.execute(
                "INSERT INTO mobile_consolidation_batches VALUES (?,?,?,?,?)",
                ("operation", "c" * 64, key, raw, "2026-10-27T00:00:00Z"),
            )
    return encoded


def _identity(original, mapped, domain="run"):
    return {
        "version": 1,
        "kind": "identity",
        "data": {"domain": domain, "original": original, "mapped": mapped},
    }


def test_committed_receipt_numeric_census_rejects_late_missing_child(tmp_path):
    db = owned_agent_runs_db(tmp_path / "runs.db")
    try:
        _install(db, tuple(_identity(f"old-{i}", f"mapped-{i}") for i in range(12)))
        with db.transaction() as conn:
            master, children = _codec()._read_owner_receipt(
                conn, "operation", "c" * 64, "runs", _TraceReadBudget()
            )
            assert master["entry_count"] == len(children) == 13
            assert {
                child["data"]["mapped"]
                for child in children
                if child["kind"] == "identity"
            } == {f"mapped-{i}" for i in range(12)}
            conn.execute(
                "DELETE FROM mobile_consolidation_batches WHERE batch_key='runs.entry:10'"
            )
        with db.transaction() as conn, pytest.raises(ValueError, match="receipt"):
            _codec()._read_owner_receipt(
                conn, "operation", "c" * 64, "runs", _TraceReadBudget()
            )
    finally:
        db.close()


@pytest.mark.parametrize(
    "bad",
    [
        {
            "version": True,
            "kind": "identity",
            "data": {"domain": "run", "original": "old", "mapped": "new"},
        },
        _identity("old", "new", "agent_definitions"),
        _identity(1, 2, "run"),
        {
            "version": 1,
            "kind": "provenance",
            "data": {"family": "unreviewed", "record": {}},
        },
    ],
)
def test_receipt_encoder_rejects_unadvertised_grammar(bad):
    with pytest.raises(ValueError, match="receipt"):
        _receipt((bad,))


def test_enabled_definition_roster_uses_complete_committed_mapping(tmp_path):
    db = owned_agent_runs_db(tmp_path / "runs.db")
    try:
        imported = db.create_agent_definition(
            AgentDefinition(
                name="imported-enabled",
                instructions="Retained preset instructions",
                provider="openai",
                model="historical-model",
            )
        )
        current = db.create_agent_definition(
            AgentDefinition(
                name="current-enabled", instructions="Current preset instructions"
            )
        )
        _install(db, (_identity("old-definition", imported, "agent_definition"),))
        assert {row["id"] for row in db.list_agent_definitions()} == {imported, current}
        assert db.get_agent_definition(imported)["enabled"]
        assert [row["id"] for row in db.list_agent_definitions(enabled_only=True)] == [
            current
        ]
        with db.transaction() as conn:
            conn.execute(
                "UPDATE mobile_consolidation_batches SET mapping_json=? WHERE batch_key='runs.entry:0'",
                (
                    json.dumps(
                        _identity("old-definition", imported, "agent_definition")
                    ),
                ),
            )
        with pytest.raises(ValueError, match="receipt"):
            db.list_agent_definitions(enabled_only=True)
        assert db.get_agent_definition(imported)["id"] == imported
    finally:
        db.close()


@pytest.mark.parametrize("native_hash", [None, "native", "invalid"])
def test_sync_native_hash_and_raw_digest_are_distinct(native_hash):
    from tldw_chatbook.Sync_Interop.hashing import canonical_payload_hash

    versioned = (
        canonical_payload_hash({"content": "retained"})
        if native_hash == "native"
        else None
    )
    if native_hash == "invalid":
        versioned = "sha512:" + "a" * 64
    record = {
        "entity": "messages",
        "original_id": "old",
        "mapped_id": "new",
        "version": 1,
        "operation": "create",
        "timestamp": "native-time",
        "client_id": "native-client",
        "payload_sha256": "a" * 64,
        "payload_hash": versioned,
        "previous_payload_sha256": None,
        "previous_hash": versioned,
        "previous_operation": None,
        "previous_timestamp": None,
        "previous_client_id": None,
        "current_disposition": "native_insert_current_unique",
        "source_change_ids": [1],
        "owner_table": "messages",
        "owner_original": ["old"],
    }
    child = {
        "version": 1,
        "kind": "provenance",
        "data": {"family": "sync_frontier", "record": record},
    }
    if native_hash == "invalid":
        with pytest.raises(ValueError, match="receipt"):
            _receipt((child,), "chat")
    else:
        raw = dict(_receipt((child,), "chat"))["chat.entry:0"]
        decoded = json.loads(raw)["data"]["record"]
        assert decoded["payload_hash"] == decoded["previous_hash"] == versioned
        assert decoded["payload_sha256"] == "a" * 64


@pytest.mark.parametrize(
    "relative,logical,kind,sha",
    [
        ("", "root", "file", "a" * 64),
        ("", "other", "directory", None),
        (".", "root", "directory", None),
    ],
)
def test_empty_runlog_path_is_only_the_native_directory_root(
    relative, logical, kind, sha
):
    child = {
        "version": 1,
        "kind": "runlog",
        "data": {
            "logical_id": logical,
            "root_id": "root",
            "relative_path": relative,
            "location": "component",
            "size": 0,
            "sha256": sha,
            "owner": None,
            "framed_runs": [],
            "historical_run_ids": [],
            "segment_ids": [],
            "artifact": {
                "path": "/native/component",
                "device": 1,
                "inode": 2,
                "kind": kind,
                "size": 0,
                "sha256": "b" * 64,
            },
        },
    }
    with pytest.raises(ValueError, match="receipt"):
        _codec()._validate_child(child, "runs")


@pytest.mark.parametrize("domain", ["fleet_sequence", "memory_sequence"])
def test_native_ordered_identity_extension_keeps_integer_owner_grammar(domain):
    children = (_identity(1, 2, domain),)
    assert (
        json.loads(dict(_receipt(children, "chat"))["chat.entry:0"])["data"]
        == children[0]["data"]
    )
    for invalid in (
        _identity("1", 2, domain),
        _identity(1, True, domain),
        _identity(1, 2, domain + "s"),
    ):
        with pytest.raises(ValueError, match="receipt"):
            _receipt((invalid,), "chat")
    with pytest.raises(ValueError, match="receipt"):
        _receipt(children, "runs")


@pytest.mark.parametrize("damage", ["missing", "count", "malformed"])
def test_runs_consumer_requires_complete_same_operation_passive_component(
    tmp_path, damage
):
    db = owned_agent_runs_db(tmp_path / "coupled-receipt.db")
    try:
        definition = db.create_agent_definition(
            AgentDefinition(name="retained", instructions="Historical instructions")
        )
        _install(db, (_identity("original", definition, "agent_definition"),))
        with db.transaction() as connection:
            if damage == "missing":
                connection.execute(
                    "DELETE FROM mobile_consolidation_batches WHERE operation_id='operation' AND batch_key='history.command'"
                )
            elif damage == "malformed":
                connection.execute(
                    "UPDATE mobile_consolidation_batches SET mapping_json='{}' WHERE operation_id='operation' AND batch_key='history.command'"
                )
            else:
                raw = connection.execute(
                    "SELECT mapping_json FROM mobile_consolidation_batches WHERE operation_id='operation' AND batch_key='runs'"
                ).fetchone()[0]
                master = json.loads(raw)
                master["command_history_count"] = 1
                connection.execute(
                    "UPDATE mobile_consolidation_batches SET mapping_json=? WHERE operation_id='operation' AND batch_key='runs'",
                    (
                        json.dumps(
                            master,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    ),
                )
        with pytest.raises(ValueError, match="history|receipt"):
            db.list_agent_definitions(enabled_only=True)
        assert (
            db.get_agent_definition(definition)["instructions"]
            == "Historical instructions"
        )
    finally:
        db.close()


@pytest.mark.parametrize(
    "source_version,with_history", [(24, False), (None, False), (25, False), (25, True)]
)
def test_runs_owner_and_passive_component_share_complete_native_snapshot(
    tmp_path, source_version, with_history
):
    from Tests.DB.test_console_command_history import historical

    db = owned_agent_runs_db(tmp_path / "complete-coupling.db")
    try:
        imported = db.create_agent_definition(
            AgentDefinition(name="imported", instructions="Retained")
        )
        current = db.create_agent_definition(
            AgentDefinition(name="current", instructions="Current")
        )
        history = (historical(db),) if with_history else ()
        _install(
            db,
            (_identity("old", imported, "agent_definition"),),
            history=history,
            source_version=source_version,
        )
        assert [row["id"] for row in db.list_agent_definitions(enabled_only=True)] == [
            current
        ]
        page = db.command_receipts.list_imported_history("operation", "c" * 64)
        assert len(page.items) == len(history)
        assert page.next_cursor is None
        if history:
            assert page.items[0].original_fields == history[0].values
            with db.transaction() as connection:
                raw = connection.execute(
                    "SELECT mapping_json FROM mobile_consolidation_batches WHERE batch_key='history.command:historical'"
                ).fetchone()[0]
                payload = json.loads(raw)
                payload["record"].pop("updated_at")
                connection.execute(
                    "UPDATE mobile_consolidation_batches SET mapping_json=? WHERE batch_key='history.command:historical'",
                    (json.dumps(payload),),
                )
            with pytest.raises(ValueError, match="history"):
                db.list_agent_definitions(enabled_only=True)
    finally:
        db.close()
