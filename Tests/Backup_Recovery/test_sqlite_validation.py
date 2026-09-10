"""Real staged SQLite validation under installed recovery policies."""

import importlib
import sqlite3
from contextlib import closing
from dataclasses import replace
from threading import Event

import pytest

from Tests.Backup_Recovery.test_core_owners import core_store, seed_domain  # noqa: F401
from Tests.Backup_Recovery.test_domain_owners import adapter as domain_adapter
from Tests.Backup_Recovery.test_domain_owners import (  # noqa: F401
    domain_store,
    study_store,
)
from Tests.Backup_Recovery.test_operational_owners import STORES, operational_adapter
from Tests.Backup_Recovery.test_participant_lifetimes import (
    local_root as local_root,  # noqa: PLC0414
)
from tldw_chatbook.Backup_Recovery import sqlite_validation as validation
from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
from tldw_chatbook.DB.private_sqlite import open_recovery_validation
from tldw_chatbook.DB.recovery_core import core_adapters
from tldw_chatbook.Research_Interop.recovery import recovery_adapters


def research_candidate(tmp_path):
    owner = recovery_adapters()[0]
    path = tmp_path / "research.db"
    with closing(sqlite3.connect(path)) as db:
        for sql in owner.schema_policy().schema_sql[0][1]:
            if not sql.startswith("CREATE TABLE sqlite_sequence"):
                db.execute(sql)
        db.execute(
            "INSERT INTO research_runs(id,query,created_at,updated_at) VALUES ('kept','nebula','now','now')"
        )
        db.commit()
    return owner, path


def test_unknown_trigger_is_rejected_before_migration(tmp_path):
    candidate = tmp_path / "hostile.db"
    with sqlite3.connect(candidate) as db:
        db.executescript(
            "CREATE TABLE payload(x); CREATE TRIGGER surprise AFTER INSERT ON payload "
            "BEGIN DELETE FROM payload; END;"
        )
    assert "unsupported_schema" in validate_candidate(
        core_adapters()[0], candidate, Event(), migrate=True
    )


def test_real_installed_core_and_fts_survive(core_store):  # noqa: F811
    name, path, store, _ = core_store
    seed_domain(name, store)
    store.close()
    owner = next(
        a
        for a in core_adapters()
        if a.owner_id
        == "db."
        + name
        + (".primary" if name in {"chachanotes", "media", "prompts"} else "")
    )
    assert validate_candidate(owner, path, Event(), migrate=False) == ()
    fts = {
        "chachanotes": "messages_fts",
        "media": "media_fts",
        "prompts": "prompts_fts",
    }.get(name)
    if fts:
        with open_recovery_validation(owner.owner_id, path, writable=False) as db:
            assert db.execute(
                f"SELECT count(*) FROM {fts} WHERE {fts} MATCH 'nebula'"
            ).fetchone() == (1,)


def test_owned_tts_reference_digest_checked_on_restricted_connection(tmp_path):
    import hashlib

    from tldw_chatbook.TTS.profile_schema import open_profile_store
    from tldw_chatbook.TTS.recovery import recovery_adapters

    path = tmp_path / "tts.db"
    with closing(open_profile_store(path)) as db:
        # Existing constructor supplies the real installed v4 layout.
        db.execute(
            "INSERT INTO tts_generation_profiles VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "p",
                "Voice",
                "voice",
                "openai",
                "tts-1",
                "alloy",
                "wav",
                1.0,
                "{}",
                1,
                "2026-01-01",
                "2026-01-01",
            ),
        )
        db.execute(
            "INSERT INTO tts_profile_clone_references VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "p",
                "r",
                b"voice",
                "Transcript",
                hashlib.sha256(b"voice").hexdigest(),
                5,
                1,
                8000,
                1,
                "pcm_s16le",
                "2026-01-01",
                "2026-01-01",
                None,
                None,
            ),
        )
        db.commit()
    owner = next(a for a in recovery_adapters() if a.owner_id == "tts.profile_store")
    assert validate_candidate(owner, path, Event(), migrate=False) == ()
    with closing(sqlite3.connect(path)) as db:
        db.execute("UPDATE tts_profile_clone_references SET wav_bytes=?", (b"wrong",))
        db.commit()
    assert validate_candidate(owner, path, Event(), migrate=False) == (
        "tts_reference_digest_mismatch",
    )


def test_actual_domain_owners(domain_store):  # noqa: F811
    name, path, service = domain_store
    service.close()
    assert validate_candidate(domain_adapter(name), path, Event(), migrate=False) == ()


@pytest.mark.parametrize("name", tuple(STORES))
def test_actual_operational_owners(tmp_path, name):
    path = tmp_path / "operational.db"
    module, symbol, *_ = STORES[name]
    constructor = getattr(importlib.import_module("tldw_chatbook." + module), symbol)
    # File Notes' unrelated filesystem admission needs the app config selector.
    # Its actual memory constructor gives the identical schema to stage here.
    store = (
        constructor(":memory:")
        if name == "file_notes"
        else constructor(db_path=path)
        if name == "kanban"
        else constructor(path)
    )
    try:
        if name == "file_notes":
            with closing(sqlite3.connect(path)) as destination:
                store._get_connection().backup(destination)
        if name == "receipts":
            with store.transaction():
                pass
    finally:
        if hasattr(store, "close"):
            store.close()
    assert (
        validate_candidate(operational_adapter(name), path, Event(), migrate=False)
        == ()
    )


def test_shared_study_assets_and_memberships(study_store):  # noqa: F811
    path, _ = study_store
    for identity in (
        "study.local",
        "quiz.local",
        "notes.sync_bindings",
        "chat.attachments",
    ):
        assert (
            validate_candidate(
                validation._installed_owner(identity), path, Event(), migrate=False
            )
            == ()
        )
    with closing(sqlite3.connect(path)) as db:
        db.execute("UPDATE flashcard_assets SET content=x'00'")
        db.commit()
    assert validate_candidate(
        validation._installed_owner("study.local"), path, Event(), migrate=False
    ) == ("missing_required_asset",)


def test_recovered_catalog_domain_validation_uses_candidate_only(tmp_path, local_root):
    from tldw_chatbook.Backup_Recovery.recovered_media import RecoveredMedia

    source = tmp_path / "source"
    source.write_bytes(b"opaque media")
    store = RecoveredMedia(tmp_path / "recovered")
    asset = store.retain(
        source, profile="p", message="m", slug="s", media_type="image/png"
    )
    owner = store.recovery_adapter()
    assert validate_candidate(owner, store.db_path, Event(), migrate=False) == ()
    # Cross-owner file validation needs the caller's staging map: this entry
    # checks the catalog, and cannot follow a locator to source bytes.
    store.resolve(asset)[1].unlink()
    assert validate_candidate(owner, store.db_path, Event(), migrate=False) == ()
    with closing(sqlite3.connect(store.db_path)) as db:
        db.execute("INSERT INTO operations VALUES (?,'retain')", (asset,))
        db.commit()
    assert validate_candidate(owner, store.db_path, Event(), migrate=False) == (
        "recovered_operation_pending",
    )


def test_inert_trusted_schema_pragma_is_not_accepted(tmp_path, monkeypatch):
    from tldw_chatbook.DB import private_sqlite

    owner, path = research_candidate(tmp_path)
    connector = private_sqlite._connect_registered_sqlite

    class UnsupportedTrust:
        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def execute(self, query, *args):
            if query == "PRAGMA trusted_schema":
                return self.connection.execute("SELECT 1")
            return self.connection.execute(query, *args)

    monkeypatch.setattr(
        private_sqlite,
        "_connect_registered_sqlite",
        lambda *a, **kw: UnsupportedTrust(connector(*a, **kw)),
    )
    assert validate_candidate(owner, path, Event(), migrate=True) == (
        "sqlite_security_unavailable",
    )


def test_research_older_migration_preserves_committed_rows(tmp_path):
    owner, path = research_candidate(tmp_path)
    assert validate_candidate(owner, path, Event(), migrate=False) == ()
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("PRAGMA user_version").fetchone() == (0,)
    assert validate_candidate(owner, path, Event(), migrate=True) == ()
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("PRAGMA user_version").fetchone() == (1,)
        assert db.execute(
            "SELECT query,lease_attempts FROM research_runs"
        ).fetchall() == [("nebula", 0)]


@pytest.mark.parametrize(
    "sql",
    [
        "CREATE TRIGGER surprise AFTER UPDATE ON research_runs BEGIN SELECT writefile('sentinel','bad'); END",
        "CREATE VIEW surprise AS SELECT load_extension('hostile')",
        "CREATE VIRTUAL TABLE surprise USING fts5(content)",
        "CREATE TABLE surprise(x GENERATED ALWAYS AS (hostile(id)), id)",
    ],
)
def test_altered_valid_catalog_never_executes_schema(tmp_path, sql):
    owner, path = research_candidate(tmp_path)
    sentinel = tmp_path.parent / (tmp_path.name + "-sentinel")
    sentinel.write_bytes(b"preserved")
    with closing(sqlite3.connect(path)) as db:
        db.create_function(
            "hostile",
            1,
            lambda value: sentinel.write_bytes(b"changed"),
            deterministic=True,
        )
        db.execute(sql)
    assert validate_candidate(owner, path, Event(), migrate=True) == (
        "unsupported_schema",
    )
    assert sentinel.read_bytes() == b"preserved"
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("PRAGMA user_version").fetchone() == (0,)


@pytest.mark.parametrize(
    "sql",
    [
        "ATTACH ':memory:' AS surprise",
        "CREATE TABLE surprise(x)",
        "DELETE FROM research_runs",
        "PRAGMA writable_schema=ON",
        "PRAGMA trusted_schema=ON",
        "SELECT load_extension('surprise')",
        "SELECT writefile('outside','payload')",
    ],
)
def test_registered_seam_denies_side_effects_even_on_writable_candidate(tmp_path, sql):
    owner, path = research_candidate(tmp_path)
    with (
        open_recovery_validation(owner.owner_id, path, writable=True) as db,
        pytest.raises(sqlite3.Error),
    ):
        db.execute(sql)


def test_newer_schema_stamp_rejected(tmp_path):
    owner, path = research_candidate(tmp_path)
    with closing(sqlite3.connect(path)) as db:
        db.execute("PRAGMA user_version=999")
    assert validate_candidate(owner, path, Event(), migrate=True) == (
        "unsupported_schema_version",
    )


def test_archive_cannot_supply_migration_authority(tmp_path):
    owner, path = research_candidate(tmp_path)

    class ForeignOwner:
        owner_id = owner.owner_id

        def schema_policy(self):
            return replace(
                owner.schema_policy(),
                migration_steps=((0, 1, ("DELETE FROM research_runs",)),),
            )

    assert validate_candidate(ForeignOwner(), path, Event(), migrate=True) == (
        "unsupported_schema_policy",
    )


def test_failed_installed_migration_rolls_back(tmp_path, monkeypatch):
    owner, path = research_candidate(tmp_path)
    authorize = validation._Restrictions.authorize
    count = 0

    def fail_late(self, action, first, second, database, source):
        nonlocal count
        if action == sqlite3.SQLITE_ALTER_TABLE:
            count += 1
            if count == 3:
                return sqlite3.SQLITE_DENY
        return authorize(self, action, first, second, database, source)

    monkeypatch.setattr(validation._Restrictions, "authorize", fail_late)
    assert validate_candidate(owner, path, Event(), migrate=True)
    assert count == 3
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("PRAGMA user_version").fetchone() == (0,)
        assert "lease_owner" not in {
            r[1] for r in db.execute("PRAGMA table_info(research_runs)")
        }
        assert db.execute("SELECT query FROM research_runs").fetchall() == [("nebula",)]


def test_cancel_during_installed_migration_rolls_back(tmp_path, monkeypatch):
    owner, path = research_candidate(tmp_path)
    cancel = Event()
    authorize = validation._Restrictions.authorize
    alters = 0

    def cancel_late(self, action, first, second, database, source):
        nonlocal alters
        if action == sqlite3.SQLITE_ALTER_TABLE:
            alters += 1
            if alters == 3:
                cancel.set()
        return authorize(self, action, first, second, database, source)

    monkeypatch.setattr(validation._Restrictions, "authorize", cancel_late)
    assert validate_candidate(owner, path, cancel, migrate=True) == ("cancelled",)
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("PRAGMA user_version").fetchone() == (0,)
        assert "lease_owner" not in {
            r[1] for r in db.execute("PRAGMA table_info(research_runs)")
        }


def test_migration_never_reopens_candidate_unrestricted(tmp_path, monkeypatch):
    from tldw_chatbook.DB import private_sqlite

    owner, path = research_candidate(tmp_path)
    connector = private_sqlite._connect_registered_sqlite
    opened = []

    def record(owner_id, candidate, **kwargs):
        connection = connector(owner_id, candidate, **kwargs)
        if candidate != ":memory:":
            opened.append((owner_id, candidate, connection))
        return connection

    monkeypatch.setattr(private_sqlite, "_connect_registered_sqlite", record)
    assert validate_candidate(owner, path, Event(), migrate=True) == ()
    assert len(opened) == 1 and opened[0][:2] == ("recovery.validation", path)
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0][2].execute("SELECT 1")


def test_sql_progress_budget_interrupts(tmp_path, monkeypatch):
    owner, path = research_candidate(tmp_path)
    monkeypatch.setattr(validation, "_STEP_BUDGET", 0)
    monkeypatch.setattr(validation, "_PROGRESS_INTERVAL", 1)
    assert validate_candidate(owner, path, Event(), migrate=True) == (
        "sqlite_resource_limit",
    )


def test_cancelled_query_stops_without_migration(tmp_path, monkeypatch):
    owner, path = research_candidate(tmp_path)
    cancel = Event()
    progress = validation._Restrictions.progress

    def cancel_on_progress(self):
        cancel.set()
        return progress(self)

    monkeypatch.setattr(validation, "_PROGRESS_INTERVAL", 1)
    monkeypatch.setattr(validation._Restrictions, "progress", cancel_on_progress)
    assert validate_candidate(owner, path, cancel, migrate=True) == ("cancelled",)


@pytest.mark.parametrize(
    "missing",
    [
        "enable_load_extension",
        "setlimit",
        "getlimit",
        "set_authorizer",
        "set_progress_handler",
    ],
)
def test_missing_security_primitive_fails_closed(tmp_path, monkeypatch, missing):
    from tldw_chatbook.DB import private_sqlite

    owner, path = research_candidate(tmp_path)
    connector = private_sqlite._connect_registered_sqlite

    class MissingPrimitive:
        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            if name == missing:
                raise AttributeError(name)
            return getattr(self.connection, name)

    monkeypatch.setattr(
        private_sqlite,
        "_connect_registered_sqlite",
        lambda *a, **kw: MissingPrimitive(connector(*a, **kw)),
    )
    assert validate_candidate(owner, path, Event(), migrate=True) == (
        "sqlite_security_unavailable",
    )
