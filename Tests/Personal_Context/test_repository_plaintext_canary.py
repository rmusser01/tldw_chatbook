from __future__ import annotations

from tldw_chatbook.Personal_Context.repository import PersonalContextRepository


def test_plaintext_never_appears_in_database_wal_or_sidecars_and_reopens(
    tmp_path, memory_protector, record_factory
) -> None:
    canary = "PROFILE-CANARY-DO-NOT-PERSIST-PLAINTEXT-8c58f6"
    db_path = tmp_path / "personal-context.db"
    repo = PersonalContextRepository(db_path, key_protector=memory_protector)
    manifest = repo.create_provisional_profile()

    keeper = repo._connect()
    keeper.execute("PRAGMA journal_mode=WAL")
    keeper.execute("BEGIN")
    keeper.execute("SELECT COUNT(*) FROM encrypted_objects").fetchone()

    record = record_factory(manifest.profile_id, value=canary)
    repo.commit_record_version(
        record,
        expected_version_id=None,
        outbox_body={"snapshot": canary + "-OUTBOX"},
    )
    repo.commit_runtime_policy("scope-global", {"policy": canary + "-POLICY"})
    repo.commit_scope_binding("scope-global", {"label": canary + "-BINDING"})

    durable_paths = [
        db_path,
        db_path.with_name(db_path.name + "-wal"),
        db_path.with_name(db_path.name + "-shm"),
        db_path.with_name(db_path.name + "-journal"),
    ]
    assert db_path.with_name(db_path.name + "-wal").exists()
    durable = b"".join(path.read_bytes() for path in durable_paths if path.exists())
    assert canary.encode() not in durable

    keeper.rollback()
    keeper.close()
    repo.close()
    reopened = PersonalContextRepository(db_path, key_protector=memory_protector)
    assert reopened.get_record(record.record_id) == record
