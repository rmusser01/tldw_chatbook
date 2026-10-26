"""Native citation history, missing keys and forged credential omissions."""

from contextlib import contextmanager

import pytest


@contextmanager
def native_citation_payload(tmp_path, monkeypatch, history=None, damage=None):
    from Tests.RuntimePolicy.test_server_credentials import FakeKeyring
    from tldw_chatbook.Backup_Recovery import credentials
    from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.runtime_policy.server_credentials import (
        KeyringServerCredentialStore,
    )

    root = tmp_path / "stage"
    root.mkdir(mode=0o700)
    path = root / "chat.sqlite"
    backend = FakeKeyring()
    monkeypatch.setattr(
        credentials,
        "_credential_store",
        lambda: KeyringServerCredentialStore(keyring_backend=backend),
    )
    db = CharactersRAGDB(path, "citation-coverage")
    try:
        context = (
            db.get_connection()
            .execute(
                "SELECT profile_id,fingerprint_key_id FROM rag_identity_context WHERE context_name='default'"
            )
            .fetchone()
        )
        profile, key = context
        conversation = db.add_conversation(
            {"title": "Credential history", "character_id": None}
        )
        message = db.add_message(
            {
                "conversation_id": conversation,
                "sender": "assistant",
                "content": "Historical answer",
            }
        )
        with db.transaction() as cursor:
            if history == "tombstone":
                cursor.execute(
                    "INSERT INTO rag_payload_tombstones VALUES (?, 'local_payload_v1','payload','scope','source_revoked','policy','2026-01-01','2027-01-01')",
                    (profile,),
                )
            elif history == "journal":
                cursor.execute(
                    "INSERT INTO rag_legacy_migration_journal VALUES (?,?,'fingerprint','pending',0,'2026-01-01','2026-01-01',NULL,NULL,NULL)",
                    (profile, conversation),
                )
            elif history in {"owner", "imported"}:
                imported = history == "imported"
                cursor.execute(
                    """INSERT INTO rag_citation_traces
                    (profile_id,trace_id,schema_version,request_id,generation_id,origin_scope_id,origin,lifecycle,completeness_at_seal,selected_attempt_id,policy_version,aggregate_json,visibility_state,created_at,sealed_at,import_package_fingerprint,external_trace_id)
                    VALUES (?, 'trace',1,'request','generation',?,?,'sealed','unavailable','attempt','policy','{}','active','2026-01-01','2026-01-01',?,?)""",
                    (
                        profile,
                        "package" if imported else profile,
                        "imported" if imported else "local",
                        "package" if imported else None,
                        "external" if imported else None,
                    ),
                )
                if history == "owner":
                    cursor.execute(
                        "INSERT INTO rag_message_trace_owners VALUES (?,?,0,'trace','active','body-fingerprint','idempotency','2026-01-01','2026-01-01')",
                        (profile, message),
                    )
            elif history in {"content_hash", "comparison_fingerprint"}:
                cursor.execute(
                    """INSERT INTO rag_evidence_snapshots
                    (profile_id,payload_id,governance_scope_id,authority_id,confidentiality_policy_id,revocation_scope_id,origin_namespace,origin_payload_id,storage_mode,redaction_state,retention_class,content_hash,comparison_fingerprint,created_at)
                    VALUES (?,'payload','governance','authority','policy','scope','local','origin','embedded','available','local',?,?,'2026-01-01')""",
                    (
                        profile,
                        "hash" if history == "content_hash" else None,
                        "comparison" if history == "comparison_fingerprint" else None,
                    ),
                )
        from tldw_chatbook.DB.citation_fingerprint_policy import (
            fingerprint_bearing_rows_exist,
        )

        assert fingerprint_bearing_rows_exist(db.get_connection()) is (
            history is not None
        )
        if damage is not None:
            with db.transaction() as cursor:
                cursor.execute(
                    "DELETE FROM rag_identity_context"
                    if damage == "changed_context"
                    else "CREATE TABLE foreign_catalog(id)"
                )
    finally:
        db.close_connection()
    inventory = Inventory(
        (StorageItem("db.chachanotes.primary", "chat", path, "included", ()),),
        True,
        "scope",
        (),
    )
    yield root, path, inventory, backend, key


@pytest.mark.parametrize(
    "history",
    [
        "owner",
        "tombstone",
        "journal",
        "imported",
        "content_hash",
        "comparison_fingerprint",
    ],
)
def test_each_native_history_condition_requires_key_and_rejects_forged_omission(
    tmp_path, monkeypatch, history
):
    from tldw_chatbook.Backup_Recovery import credentials

    with native_citation_payload(tmp_path, monkeypatch, history) as (
        root,
        path,
        inventory,
        backend,
        _,
    ):
        issues = credentials.process_credentials(
            root, inventory, mode="rollback", encrypted=True
        )
        assert len(issues) == 1 and issues[0].startswith("credential_missing:"), issues
        with pytest.raises(ValueError, match="credential_material_missing"):
            credentials.validate_citation_coverage(
                root, (("db.chachanotes.primary", path),), []
            )
        assert backend.values == {} and backend.deleted == []


def test_unused_default_key_omission_is_qualified_without_keyring_write(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import credentials

    with native_citation_payload(tmp_path, monkeypatch) as (
        root,
        path,
        inventory,
        backend,
        _,
    ):
        before = path.read_bytes(), path.stat().st_mode
        assert (
            credentials.process_credentials(
                root, inventory, mode="rollback", encrypted=True
            )
            == ()
        )
        assert credentials._material(root) == []
        credentials.validate_citation_coverage(
            root, (("db.chachanotes.primary", path),), []
        )
        assert before == (path.read_bytes(), path.stat().st_mode)
        assert backend.values == {} and backend.deleted == []


@pytest.mark.parametrize(
    "fault", ["unreadable", "unknown_predicate", "changed_context", "unknown_catalog"]
)
def test_unused_classification_refuses_uncertain_native_evidence(
    tmp_path, monkeypatch, fault
):
    from tldw_chatbook.Backup_Recovery import credentials
    from tldw_chatbook.DB import citation_fingerprint_policy

    with native_citation_payload(
        tmp_path,
        monkeypatch,
        damage=fault if fault in {"changed_context", "unknown_catalog"} else None,
    ) as (root, path, inventory, backend, _):
        if fault == "unreadable":

            def unreadable(*args):
                raise RuntimeError("synthetic secret backend exception")

            monkeypatch.setattr(backend, "get_password", unreadable)
        elif fault == "unknown_predicate":

            def unknown(*args):
                raise ValueError("unknown predicate")

            monkeypatch.setattr(
                citation_fingerprint_policy, "fingerprint_bearing_rows_exist", unknown
            )
        issues = credentials.process_credentials(
            root, inventory, mode="rollback", encrypted=True
        )
        assert issues and all("synthetic secret" not in issue for issue in issues)
        with pytest.raises((ValueError, __import__("sqlite3").Error)):
            credentials.validate_citation_coverage(
                root, (("db.chachanotes.primary", path),), []
            )
        assert backend.values == {} and backend.deleted == []


@pytest.mark.parametrize(
    "change", ["import_package_fingerprint=NULL", "origin='local'"]
)
def test_imported_predicate_arms_cannot_be_separated_in_qualified_native_catalog(
    tmp_path, monkeypatch, change
):
    import sqlite3

    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    with native_citation_payload(tmp_path, monkeypatch, history="imported") as (
        _,
        path,
        _,
        _,
        _,
    ):
        db = CharactersRAGDB(path, "citation-constraint-proof")
        try:
            with pytest.raises(sqlite3.IntegrityError), db.transaction() as cursor:
                # Two fixed test statements, never caller input or imported SQL.
                cursor.execute("UPDATE rag_citation_traces SET " + change)
            row = (
                db.get_connection()
                .execute(
                    "SELECT origin,import_package_fingerprint FROM rag_citation_traces"
                )
                .fetchone()
            )
            assert tuple(row) == ("imported", "package")
        finally:
            db.close_connection()
