"""Core recovery owner qualification using real local domain stores."""


def test_core_owner_set_is_declared():
    from tldw_chatbook.DB.recovery_core import core_adapters

    names = {adapter.owner_id for adapter in core_adapters()}
    assert {"db.chachanotes.primary", "db.media.primary", "db.prompts.primary"} <= names


from pathlib import Path
import pytest


@pytest.fixture(
    params=[
        "chachanotes",
        "media",
        "prompts",
        "library_collections",
        "library_ingest_jobs",
    ]
)
def core_store(request, tmp_path):
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
    from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
    from tldw_chatbook.DB.Library_Ingest_Jobs_DB import LibraryIngestJobsDB

    classes = dict(
        chachanotes=CharactersRAGDB,
        media=MediaDatabase,
        prompts=PromptsDatabase,
        library_collections=LibraryCollectionsDB,
        library_ingest_jobs=LibraryIngestJobsDB,
    )
    name = request.param
    path = tmp_path / (name + ".db")
    owner = classes[name](path, "recovery-fixture")
    conn = (
        owner.get_connection()
        if name in ("chachanotes", "media", "prompts")
        else (
            owner._held_connection()
            if name == "library_collections"
            else owner._get_connection()
        )
    )
    try:
        yield name, path, owner, conn
    finally:
        owner.close()


def test_schema_policy_matches_installed_store(core_store):
    from tldw_chatbook.DB.recovery_core import core_adapters

    name, path, owner, conn = core_store
    adapter = next(
        a
        for a in core_adapters()
        if a.owner_id
        == "db."
        + name
        + (".primary" if name in ("chachanotes", "media", "prompts") else "")
    )
    actual = tuple(
        row[0]
        for row in conn.execute(
            "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
        )
    )
    policy = adapter.schema_policy()
    assert policy is not None
    assert policy.versions == (owner._CURRENT_SCHEMA_VERSION,)
    assert policy.schema_sql == ((owner._CURRENT_SCHEMA_VERSION, actual),)
    assert policy.migration_steps == ()


def test_native_maintenance_mints_scoped_capture_authority(tmp_path):
    from tldw_chatbook.Backup_Recovery.admission import Admission

    source = tmp_path / "source"
    source.mkdir(mode=0o700)
    authority = Admission(tmp_path / "control")
    authority.register("core", (source,))
    with authority.maintenance(("core",), 1) as session:
        assert session is not None


def adapter_for(name):
    from tldw_chatbook.DB.recovery_core import core_adapters

    return next(
        a
        for a in core_adapters()
        if a.owner_id
        == "db."
        + name
        + (".primary" if name in ("chachanotes", "media", "prompts") else "")
    )


def test_capture_roundtrip_and_validation(core_store, tmp_path, monkeypatch):
    from threading import Event
    from tldw_chatbook.Backup_Recovery.admission import Admission
    from tldw_chatbook.Backup_Recovery.models import StorageItem
    import sqlite3

    name, source, owner, connection = core_store
    adapter = adapter_for(name)
    expected = tuple(connection.iterdump())
    owner.close()
    before = source.read_bytes()
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    authority = application_authority(tmp_path, source, monkeypatch)
    item = StorageItem(
        adapter.owner_id, "profile:test:" + adapter.owner_id, source, "included", ()
    )
    destination = stage / "snapshot.db"
    with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
        with session.capture_scope((source,), stage):
            adapter.capture(item, destination, Event())
            assert adapter.validate(destination) == ()
    assert source.read_bytes() == before
    with sqlite3.connect(destination) as captured:
        assert tuple(captured.iterdump()) == expected


def seed_domain(name, owner):
    if name == "chachanotes":
        character = owner.add_character_card(
            {
                "name": "Recovery character",
                "description": "nebula character",
                "image": b"\x00character\xff",
            }
        )
        conversation = owner.add_conversation(
            {"title": "nebula conversation", "character_id": character}
        )
        message = owner.add_message(
            {
                "conversation_id": conversation,
                "sender": "user",
                "content": "nebula message",
                "image_data": b"\x00image\xff",
                "image_mime_type": "image/png",
            }
        )
        owner.set_message_attachments(
            message,
            [
                {
                    "position": 1,
                    "data": b"\x00attachment\xff",
                    "mime_type": "application/octet-stream",
                    "display_name": "evidence.bin",
                }
            ],
        )
        note = owner.add_note("nebula note", "Soft deleted durable note")
        owner.soft_delete_note(note, 1)
        owner.create_flashcard_asset(
            original_filename="asset.bin",
            mime_type="application/octet-stream",
            content=b"\x00asset\xff",
        )
    elif name == "media":
        kept, _, _ = owner.add_media_with_keywords(
            title="nebula media",
            media_type="document",
            content="searchable nebula content",
            keywords=["nebula"],
        )
        deleted, _, _ = owner.add_media_with_keywords(
            title="deleted media",
            media_type="document",
            content="deleted unique content",
        )
        assert kept and deleted
        owner.soft_delete_media(deleted)
    elif name == "prompts":
        kept, _, _ = owner.add_prompt(
            "nebula prompt",
            "Author",
            "Details",
            user_prompt="nebula instructions",
            keywords=["nebula"],
        )
        deleted, _, _ = owner.add_prompt(
            "deleted prompt", "Author", "Details", user_prompt="retained deletion"
        )
        assert kept and deleted
        owner.soft_delete_prompt(deleted)
    elif name == "library_collections":
        from tldw_chatbook.Library.library_collections_service import (
            LocalLibraryCollectionsService,
        )

        service = LocalLibraryCollectionsService(owner)
        collection = service.create_collection("nebula collection")
        service.add_item_to_collection(
            collection.collection_id,
            source_type="media",
            source_id="1",
            title="nebula item",
        )
        deleted = service.create_collection("deleted collection")
        service.delete_collection(deleted.collection_id)
    else:
        from tldw_chatbook.Library.library_ingest_jobs import (
            LibraryIngestJob,
            IngestJobState,
        )

        owner.upsert_job(
            LibraryIngestJob(
                job_id="job-1",
                source_path="/external/document.txt",
                title="nebula ingestion",
                state=IngestJobState.DONE,
                media_id=1,
            )
        )
        owner.upsert_job(
            LibraryIngestJob(
                job_id="job-2",
                source_path="https://example.invalid",
                title="remote ingestion",
                state=IngestJobState.DONE,
                origin="server",
                remote_media_id="remote-7",
            )
        )


def test_committed_wal_domain_records_blobs_and_soft_deletes(
    core_store, tmp_path, monkeypatch
):
    import sqlite3
    from threading import Event
    from tldw_chatbook.Backup_Recovery.admission import Admission
    from tldw_chatbook.Backup_Recovery.models import StorageItem

    name, source, owner, conn = core_store
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    observer = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
    observer.execute("BEGIN")
    observer.execute("SELECT count(*) FROM sqlite_schema").fetchone()
    try:
        seed_domain(name, owner)
        expected = tuple(conn.iterdump())
        owner.close()
        wal = Path(str(source) + "-wal")
        assert wal.stat().st_size > 32
        before = source.read_bytes(), wal.read_bytes(), source.stat().st_mtime_ns
        stage = tmp_path / "stage"
        stage.mkdir(mode=0o700)
        authority = application_authority(tmp_path, source, monkeypatch)
        adapter = adapter_for(name)
        item = StorageItem(
            adapter.owner_id, "profile:test:" + adapter.owner_id, source, "included", ()
        )
        with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
            with session.capture_scope((source,), stage):
                adapter.capture(item, stage / "captured.db", Event())
        with sqlite3.connect(stage / "captured.db") as captured:
            assert tuple(captured.iterdump()) == expected
            if name == "chachanotes":
                assert (
                    captured.execute("SELECT data FROM message_attachments").fetchone()[
                        0
                    ]
                    == b"\x00attachment\xff"
                )
                assert (
                    captured.execute("SELECT content FROM flashcard_assets").fetchone()[
                        0
                    ]
                    == b"\x00asset\xff"
                )
                assert (
                    captured.execute(
                        "SELECT count(*) FROM messages_fts WHERE messages_fts MATCH 'nebula'"
                    ).fetchone()[0]
                    == 1
                )
                assert captured.execute("SELECT deleted FROM notes").fetchone()[0] == 1
            elif name == "media":
                assert (
                    captured.execute(
                        "SELECT count(*) FROM media_fts WHERE media_fts MATCH 'nebula'"
                    ).fetchone()[0]
                    == 1
                )
            elif name == "prompts":
                assert (
                    captured.execute(
                        "SELECT count(*) FROM prompts_fts WHERE prompts_fts MATCH 'nebula'"
                    ).fetchone()[0]
                    == 1
                )
        assert (
            source.read_bytes(),
            wal.read_bytes(),
            source.stat().st_mtime_ns,
        ) == before
    finally:
        observer.close()


def test_capture_scope_retires_escaped_default_native_handle(tmp_path, monkeypatch):
    import sqlite3
    from tldw_chatbook.Backup_Recovery.admission import Admission
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    source = tmp_path / "source.db"
    with sqlite3.connect(source) as db:
        db.execute("CREATE TABLE entries(value)")
    source.chmod(0o600)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    authority = application_authority(tmp_path, source, monkeypatch)

    with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
        with session.capture_scope((source,), stage):
            escaped = connect_private_sqlite(
                "recovery.core.media", stage / "escaped.db"
            )
            escaped.execute("CREATE TABLE evidence(value)")
            escaped.execute("INSERT INTO evidence VALUES (1)")
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            escaped.execute("INSERT INTO evidence VALUES (2)")
    with authority.normal(("core",)):
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            escaped.execute("INSERT INTO evidence VALUES (3)")


@pytest.fixture
def scoped_files(tmp_path, monkeypatch):
    import sqlite3
    from tldw_chatbook.Backup_Recovery.admission import Admission

    source = tmp_path / "source.db"
    with sqlite3.connect(source) as db:
        db.execute("CREATE TABLE entries(value)")
    source.chmod(0o600)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    authority = application_authority(tmp_path, source, monkeypatch)
    return authority, source, stage


def test_scope_is_directional_revocable_and_cannot_be_copied_or_cross_thread(
    scoped_files, tmp_path
):
    import copy
    from concurrent.futures import ThreadPoolExecutor
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    authority, source, stage = scoped_files
    with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
        with pytest.raises(RecoveryRequired, match="capture_source_outside_scope"):
            with session.capture_scope((tmp_path,), stage):
                pass
        with pytest.raises(RecoveryRequired, match="maintenance_session_inactive"):
            with copy.copy(session).capture_scope((source,), stage):
                pass
        with ThreadPoolExecutor() as pool:

            def foreign_thread():
                with session.capture_scope((source,), stage):
                    pass

            with pytest.raises(RecoveryRequired, match="maintenance_session_inactive"):
                pool.submit(foreign_thread).result()
        with session.capture_scope((source,), stage):
            with pytest.raises(RecoveryRequired, match="capture_path_outside_scope"):
                connect_private_sqlite("recovery.core.media", source)
            with pytest.raises(RecoveryRequired, match="capture_owner_not_registered"):
                connect_private_sqlite("db.media.primary", stage / "ordinary.db")
            with pytest.raises(RecoveryRequired, match="capture_path_outside_scope"):
                connect_private_sqlite("recovery.core.media", tmp_path / "outside.db")
            with pytest.raises(RecoveryRequired, match="capture_target_alias"):
                (stage / "alias.db").hardlink_to(source)
                connect_private_sqlite("recovery.core.media", stage / "alias.db")
            (stage / "escape").symlink_to(tmp_path, target_is_directory=True)
            with pytest.raises(RecoveryRequired, match="capture_path_outside_scope"):
                connect_private_sqlite(
                    "recovery.core.media", stage / "escape" / "outside.db"
                )
        with pytest.raises(
            RecoveryRequired, match="maintenance_requires_owner_capability"
        ):
            connect_private_sqlite("recovery.core.media", source, read_only=True)
    with pytest.raises(RecoveryRequired, match="maintenance_session_inactive"):
        with session.capture_scope((source,), stage):
            pass
    assert not (tmp_path / "outside.db").exists()
    assert not (stage / "ordinary.db").exists()


def test_native_session_holds_other_process_until_capture_resources_retire(
    scoped_files,
):
    import subprocess
    import sys
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    authority, source, stage = scoped_files
    program = """import sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery.admission import Admission
print("ready", flush=True)
with Admission(Path(sys.argv[1])).normal(("core",)):
    print("entered", flush=True)
"""
    child = None
    try:
        with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
            with session.capture_scope((source,), stage):
                escaped = connect_private_sqlite(
                    "recovery.core.media", stage / "capture.db"
                )
                child = subprocess.Popen(
                    [sys.executable, "-c", program, str(authority.control_root)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                assert child.stdout.readline().strip() == "ready"
                with pytest.raises(subprocess.TimeoutExpired):
                    child.communicate(timeout=0.2)
            import sqlite3

            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                escaped.execute("CREATE TABLE late(value)")
        out, err = child.communicate(timeout=10)
        assert child.returncode == 0, err
        assert out.strip() == "entered"
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.communicate()


@pytest.mark.parametrize(
    "damage, expected",
    [
        ("version", "unsupported_schema_version"),
        ("trigger", "unsupported_schema"),
        ("table", "unsupported_schema"),
    ],
)
def test_schema_variants_never_blessed_by_current_version(core_store, damage, expected):
    name, source, owner, conn = core_store
    if damage == "version":
        table = "db_schema_version" if name == "chachanotes" else "schema_version"
        conn.execute("UPDATE " + table + " SET version=999")
    elif damage == "trigger":
        table = "db_schema_version" if name == "chachanotes" else "schema_version"
        conn.execute(
            "CREATE TRIGGER hostile AFTER UPDATE ON "
            + table
            + " BEGIN SELECT load_extension('arbitrary'); END"
        )
    else:
        conn.execute("CREATE TABLE unexpected(payload)")
    conn.commit()
    owner.close()
    assert adapter_for(name).validate(source) == (expected,)


def test_interrupted_capture_never_returns_success_or_changes_source(
    core_store, tmp_path, monkeypatch
):
    from threading import Event
    from tldw_chatbook.Backup_Recovery.admission import Admission
    from tldw_chatbook.Backup_Recovery.models import StorageItem

    name, source, owner, conn = core_store
    owner.close()
    before = source.read_bytes()
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    authority = application_authority(tmp_path, source, monkeypatch)
    adapter = adapter_for(name)
    item = StorageItem(
        adapter.owner_id, "profile:test:" + adapter.owner_id, source, "included", ()
    )

    class ProgressCancellation(Event):
        checks = 0

        def is_set(self):
            self.checks += 1
            return self.checks >= 2

    with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
        with session.capture_scope((source,), stage):
            with pytest.raises(InterruptedError, match="cancelled"):
                adapter.capture(item, stage / "interrupted.db", ProgressCancellation())
            cancelled = Event()
            cancelled.set()
            with pytest.raises(InterruptedError, match="cancelled"):
                adapter.capture(item, stage / "never-created.db", cancelled)
            assert not (stage / "never-created.db").exists()
    assert source.read_bytes() == before


def test_independent_native_authority_cannot_capture_application_owner(
    scoped_files, tmp_path
):
    from tldw_chatbook.Backup_Recovery.admission import Admission
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    _, source, stage = scoped_files
    foreign = Admission(tmp_path / "foreign-control")
    foreign.register("core", (source,))
    with foreign.maintenance(("core",), 1) as session:
        with pytest.raises(RecoveryRequired, match="conflicting_admission_authority"):
            with session.capture_scope((source,), stage):
                pass


def application_authority(tmp_path, source, monkeypatch):
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        bind_profile,
    )

    root = tmp_path / "bootstrap"
    selector = tmp_path / "selected.toml"
    selector.write_text('[general]\nusers_name="fixture"\n')
    authority = admission_authority(root)
    authority.register("core", (source, selector))
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
    bind_profile(root, selector, ("core",), root / "admission")
    return authority


def test_relocation_preserves_owned_blobs_and_inert_external_locators(core_store):
    name, source, owner, conn = core_store
    seed_domain(name, owner)
    before = tuple(conn.iterdump())
    owner.close()
    adapter = adapter_for(name)
    adapter.relocate(source, {"/external": Path("/new/external")})
    import sqlite3

    with sqlite3.connect(source) as captured:
        assert tuple(captured.iterdump()) == before


def test_discovery_custom_paths_and_explicit_core_dependencies(tmp_path):
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
    )
    from tldw_chatbook.Backup_Recovery.inventory import classify_entries
    from tldw_chatbook.DB.recovery_core import core_adapters

    context = DiscoveryContext(tmp_path / "selected.toml", "selected")
    config = {DISCOVERY_CONTEXT_KEY: context, "database": {}}
    for adapter in core_adapters():
        custom = tmp_path / (adapter.owner_id + ".db")
        custom.write_bytes(b"pure discovery never opens this non-SQLite file")
        config["database"][adapter.setting_name] = str(custom)
    for adapter in core_adapters():
        (item,) = adapter.discover(config)
        assert item.path == Path(config["database"][adapter.setting_name])
        assert item.logical_id == "profile:selected:" + adapter.owner_id
        assert item.status == "included"
        assert "profile:selected:config" in item.dependencies
        assert not classify_entries((item,)).complete
    chacha = core_adapters()[0].discover(config)[0]
    assert "profile:selected:persona.assets:unresolved" in chacha.dependencies
    assert "profile:selected:notes.file_notes" in chacha.dependencies


def test_local_cross_store_dependencies_use_exact_profile_ids(tmp_path):
    from tldw_chatbook.DB.Library_Ingest_Jobs_DB import LibraryIngestJobsDB
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    from tldw_chatbook.Backup_Recovery.models import StorageItem

    ingest = LibraryIngestJobsDB(tmp_path / "ingest.db")
    media = MediaDatabase(tmp_path / "media.db", "fixture")
    try:
        seed_domain("library_ingest_jobs", ingest)
        media_id, _, _ = media.add_media_with_keywords(
            title="local record", media_type="document", content="local content"
        )
        assert media_id == 1
    finally:
        ingest.close()
        media.close()
    adapter = adapter_for("library_ingest_jobs")
    dep = "profile:a:db.media.primary"
    item = StorageItem(
        adapter.owner_id,
        "profile:a:" + adapter.owner_id,
        tmp_path / "ingest.db",
        "included",
        (dep,),
    )
    assert adapter.validate_dependencies(
        item, item.path, {"profile:b:db.media.primary": tmp_path / "media.db"}
    ) == ("dependency_unavailable",)
    assert (
        adapter.validate_dependencies(item, item.path, {dep: tmp_path / "media.db"})
        == ()
    )
    import sqlite3

    with sqlite3.connect(tmp_path / "ingest.db") as connection:
        connection.execute("UPDATE ingest_jobs SET media_id=999 WHERE origin='local'")
    assert adapter.validate_dependencies(
        item, item.path, {dep: tmp_path / "media.db"}
    ) == ("invalid_domain_reference",)


def test_capture_refuses_missing_unbound_gate_and_changed_profile_binding(scoped_files):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery import bootstrap

    authority, source, stage = scoped_files
    with authority.maintenance(("core",), 1) as session:
        with pytest.raises(
            RecoveryRequired, match="capture_unbound_admission_required"
        ):
            with session.capture_scope((source,), stage):
                pass
    import os

    Path(os.environ["TLDW_CONFIG_PATH"]).write_text("changed selector")
    with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
        with pytest.raises(RecoveryRequired, match="capture_source_binding_unverified"):
            with session.capture_scope((source,), stage):
                pass


def test_capture_blocks_actual_unbound_writer_to_selected_source(scoped_files):
    import os
    import subprocess
    import sys

    authority, source, stage = scoped_files
    program = """import sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery import bootstrap
bootstrap.default_bootstrap_root = lambda: Path(sys.argv[1])
from tldw_chatbook.DB.private_sqlite import connect_private_sqlite
print("ready", flush=True)
connection = connect_private_sqlite("db.base", Path(sys.argv[2]))
connection.execute("INSERT INTO entries VALUES ('unbound-writer')")
connection.commit()
connection.close()
print("written", flush=True)
"""
    environment = dict(os.environ, TLDW_CONFIG_PATH=str(stage / "unknown.toml"))
    child = None
    try:
        with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
            with session.capture_scope((source,), stage):
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        program,
                        str(authority.control_root.parent),
                        str(source),
                    ],
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                assert child.stdout.readline().strip() == "ready"
                with pytest.raises(subprocess.TimeoutExpired):
                    child.communicate(timeout=0.2)
        out, err = child.communicate(timeout=10)
        assert child.returncode == 0, err
        assert out.strip() == "written"
        import sqlite3

        with sqlite3.connect(source) as connection:
            assert connection.execute("SELECT value FROM entries").fetchall() == [
                ("unbound-writer",)
            ]
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.communicate()


def test_factory_discovery_and_schema_policy_are_runtime_independent(tmp_path):
    import os
    import subprocess
    import sys

    home = tmp_path / "fresh-home"
    program = """import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from tldw_chatbook.DB.recovery_core import core_adapters
from tldw_chatbook.Backup_Recovery.models import DISCOVERY_CONTEXT_KEY, DiscoveryContext
home = Path(sys.argv[2])
config = {DISCOVERY_CONTEXT_KEY: DiscoveryContext(home / "config.toml", "pure")}
for adapter in core_adapters():
    assert adapter.schema_policy().versions
    item, = adapter.discover(config)
    assert item.status == "missing_required"
assert "tldw_chatbook.config" not in sys.modules
assert "tldw_chatbook.DB.private_sqlite" not in sys.modules
assert "tldw_chatbook.DB.ChaChaNotes_DB" not in sys.modules
assert "tldw_chatbook.DB.base_db" not in sys.modules
assert not home.exists()
print("pure")
"""
    environment = dict(
        os.environ,
        HOME=str(home),
        USERPROFILE=str(home),
        TLDW_CONFIG_PATH=str(home / "config.toml"),
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            program,
            str(Path(__file__).resolve().parents[2]),
            str(home),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "pure"
    assert not home.exists()


def test_schema_valid_but_orphaned_domain_reference_is_rejected(tmp_path):
    import sqlite3
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    source = tmp_path / "source.db"
    owner = CharactersRAGDB(source, "fixture")
    owner.close()
    with sqlite3.connect(source) as connection:
        connection.execute(
            "INSERT INTO message_attachments VALUES ('missing-message', 1, ?, 'image/png', 'orphan')",
            (b"orphan-bytes",),
        )
    assert adapter_for("chachanotes").validate(source) == ("invalid_domain_reference",)


def test_staging_substitution_is_refused_before_open(scoped_files, tmp_path):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    authority, source, stage = scoped_files
    with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
        with session.capture_scope((source,), stage):
            stage.rename(tmp_path / "old-stage")
            stage.mkdir(mode=0o700)
            with pytest.raises(RecoveryRequired, match="capture_staging_changed"):
                connect_private_sqlite("recovery.core.media", stage / "never.db")
    assert not (stage / "never.db").exists()


def test_missing_referenced_later_owned_asset_blocks_dependency_validation(tmp_path):
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
    )

    source = tmp_path / "source.db"
    owner = CharactersRAGDB(source, "fixture")
    note = owner.add_note("File note", "Durable note text")
    with owner.transaction() as cursor:
        cursor.execute(
            "UPDATE notes SET file_path_on_disk=?, version=version+1 WHERE id=?",
            (str(tmp_path / "missing.md"), note),
        )
    owner.close()
    adapter = adapter_for("chachanotes")
    (item,) = adapter.discover(
        {
            DISCOVERY_CONTEXT_KEY: DiscoveryContext(tmp_path / "selector", "a"),
            "database": {"chachanotes_db_path": str(source)},
        }
    )
    assert adapter.validate(source) == ()
    assert adapter.validate_dependencies(item, source, {}) == (
        "dependency_unavailable",
    )
    assert not (tmp_path / "missing.md").exists()


def test_capture_allows_actual_disjoint_bound_writer(scoped_files, tmp_path):
    import os
    import sqlite3
    import subprocess
    import sys
    from tldw_chatbook.Backup_Recovery.control_records import bind_profile

    authority, source, stage = scoped_files
    other = tmp_path / "disjoint.db"
    with sqlite3.connect(other) as connection:
        connection.execute("CREATE TABLE entries(value)")
    other.chmod(0o600)
    selector = tmp_path / "disjoint.toml"
    selector.write_text('[general]\nusers_name="disjoint"\n')
    authority.register("disjoint", (other, selector))
    bind_profile(
        authority.control_root.parent, selector, ("disjoint",), authority.control_root
    )
    program = """import sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery import bootstrap
bootstrap.default_bootstrap_root = lambda: Path(sys.argv[1])
from tldw_chatbook.DB.private_sqlite import connect_private_sqlite
connection = connect_private_sqlite("db.base", Path(sys.argv[2]))
connection.execute("INSERT INTO entries VALUES ('disjoint-writer')")
connection.commit()
connection.close()
print("written", flush=True)
"""
    environment = dict(os.environ, TLDW_CONFIG_PATH=str(selector))
    child = None
    try:
        with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
            with session.capture_scope((source,), stage):
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        program,
                        str(authority.control_root.parent),
                        str(other),
                    ],
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                out, err = child.communicate(timeout=3)
                assert child.returncode == 0, err
                assert out.strip() == "written"
        with sqlite3.connect(other) as connection:
            assert connection.execute("SELECT value FROM entries").fetchall() == [
                ("disjoint-writer",)
            ]
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.communicate()


@pytest.mark.parametrize(
    "damage",
    ["lost_authority", "lost_namespace", "replaced_marker", "corrupt_registry"],
)
def test_existing_admission_authority_never_repairs_lost_evidence(
    scoped_files, tmp_path, damage
):
    import json
    from tldw_chatbook.Backup_Recovery.control_records import admission_authority
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    authority, _, _ = scoped_files
    root = authority.control_root.parent
    registry = authority.control_root / "registry.json"
    if damage == "lost_authority":
        authority.control_root.rename(tmp_path / "retained-authority")
    elif damage == "lost_namespace":
        data = json.loads(registry.read_text())
        del data["entries"]["bootstrap.unbound"]
        registry.write_text(json.dumps(data))
    elif damage == "replaced_marker":
        marker = root / "unbound-owner"
        marker.rename(tmp_path / "retained-marker")
        marker.write_bytes(b"local enrollment owner\n")
        marker.chmod(0o600)
    else:
        registry.write_bytes(b"{")
    before = {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    with pytest.raises(RecoveryRequired, match="recovery_scope_uncertain"):
        admission_authority(root)
    after = {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    assert after == before
    if damage == "lost_authority":
        assert not authority.control_root.exists()


@pytest.mark.parametrize("custom_phase", ("new", "init"))
def test_failed_capture_factory_cannot_leave_a_retained_native_handle(
    scoped_files, custom_phase
):
    import sqlite3
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    authority, source, stage = scoped_files
    retained = []
    invoked = []

    class RetainingFailure(sqlite3.Connection):
        def __new__(cls, *args, **kwargs):
            invoked.append("new")
            instance = super().__new__(cls)
            if custom_phase == "new":
                sqlite3.Connection.__init__(instance, *args, **kwargs)
                retained.append(instance)
                raise RuntimeError("partial_initialization_failed")
            return instance

        def __init__(self, *args, **kwargs):
            invoked.append("init")
            super().__init__(*args, **kwargs)
            retained.append(self)
            self.execute("CREATE TABLE factory_evidence(value)")
            self.commit()
            raise RuntimeError("partial_initialization_failed")

    try:
        with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
            with session.capture_scope((source,), stage):
                with pytest.raises(
                    RecoveryRequired, match="capture_factory_not_qualified"
                ):
                    connect_private_sqlite(
                        "recovery.core.media",
                        stage / "partial.db",
                        factory=RetainingFailure,
                    )
                assert invoked == []
                assert not (stage / "partial.db").exists()
        with authority.normal(("core",)):
            assert retained == []
            assert not (stage / "partial.db").exists()
    finally:
        for connection in retained:
            sqlite3.Connection.close(connection)


def test_dependency_validation_scans_each_real_peer_once_per_invocation(tmp_path):
    import sys
    from collections import Counter
    from contextlib import closing
    from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
    from tldw_chatbook.Library.library_collections_service import (
        LocalLibraryCollectionsService,
    )
    from tldw_chatbook.Backup_Recovery.models import StorageItem
    from tldw_chatbook.DB import private_sqlite

    stores = {
        "db.library_collections": LibraryCollectionsDB(tmp_path / "collections.db"),
        "db.media.primary": MediaDatabase(tmp_path / "media.db", "fixture"),
        "db.prompts.primary": PromptsDatabase(tmp_path / "prompts.db", "fixture"),
    }
    paths = {owner: Path(store.db_path) for owner, store in stores.items()}
    try:
        service = LocalLibraryCollectionsService(stores["db.library_collections"])
        collection = service.create_collection("References")
        other = service.create_collection("Other")
        for index in range(4):
            media_id, _, _ = stores["db.media.primary"].add_media_with_keywords(
                title=f"Media {index}",
                media_type="document",
                content=f"Unique content {index}",
            )
            service.add_item_to_collection(
                collection.collection_id, source_type="media", source_id=str(media_id)
            )
        for index in range(3):
            prompt_id, _, _ = stores["db.prompts.primary"].add_prompt(
                f"Prompt {index}", "Author", "Details"
            )
            service.add_item_to_collection(
                collection.collection_id, source_type="prompt", source_id=str(prompt_id)
            )
        for target in (collection, other):
            service.add_item_to_collection(
                collection.collection_id,
                source_type="collection",
                source_id=target.collection_id,
            )
    finally:
        for store in stores.values():
            store.close()
    adapter = adapter_for("library_collections")
    dependencies = ("profile:a:db.media.primary", "profile:a:db.prompts.primary")
    item = StorageItem(
        adapter.owner_id,
        "profile:a:" + adapter.owner_id,
        paths[adapter.owner_id],
        "included",
        dependencies,
    )
    candidates = {"profile:a:" + owner: path for owner, path in paths.items()}
    validations = Counter()
    connections = Counter()
    validate_code = type(adapter).validate.__code__
    connect_code = private_sqlite._connect_registered_sqlite.__code__

    def profile(frame, event, arg):
        if event == "call" and frame.f_code is validate_code:
            validations[frame.f_locals["self"].owner_id] += 1
        if event == "call" and frame.f_code is connect_code:
            connections[frame.f_locals["owner_id"]] += 1

    previous = sys.getprofile()
    try:
        sys.setprofile(profile)
        result = adapter.validate_dependencies(item, item.path, candidates)
    finally:
        sys.setprofile(previous)
    assert result == ()
    assert validations == Counter({owner: 1 for owner in stores})
    assert connections == Counter(
        {
            "recovery.core.library_collections": 2,
            "recovery.core.media": 2,
            "recovery.core.prompts": 2,
        }
    )
    # A later invocation must not reuse a stale process-wide validation cache.
    import sqlite3

    with closing(sqlite3.connect(paths["db.media.primary"])) as connection:
        connection.execute("CREATE TABLE unqualified(payload)")
    assert adapter.validate_dependencies(item, item.path, candidates) == (
        "dependency_unavailable",
    )
