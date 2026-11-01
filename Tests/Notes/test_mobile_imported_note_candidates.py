"""Receipt-backed Notes startup controls; no imported execution authority."""

import asyncio
from dataclasses import replace

import pytest

from Tests.Notes.test_notes_sync_runtime import (
    _Adapter,
    _Coordinator,
    _input,
    _store,
    _Watcher,
)
from tldw_chatbook.Library.library_notes_lasting_sync_state import (
    initial_lasting_sync_snapshot,
)
from tldw_chatbook.Notes.notes_device_state_store import NotesSyncStoreSetting
from tldw_chatbook.Notes.notes_sync_runtime import (
    NotesSyncRuntimeOwner,
    NotesSyncRuntimeSnapshot,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "marker,admitted,sole,expected",
    [
        (True, True, True, 1),
        (False, True, True, 0),
        (True, False, True, 0),
        (True, True, False, 0),
        ("wrong", True, True, 0),
    ],
)
async def test_present_review_failure_preserves_native_startup(
    tmp_path, marker, admitted, sole, expected
):
    store = _store(tmp_path, marker=bool(marker))
    if marker == "wrong":
        store.set_setting(NotesSyncStoreSetting("cutover_marker", "wrong"))
    calls, migrations = [], []

    def review():
        calls.append("review")
        raise ValueError("private receipt body must not be projected")

    watcher = _Watcher()
    owner = NotesSyncRuntimeOwner(
        store=store,
        migrate_legacy=lambda: migrations.append("migration"),
        coordinator=_Coordinator(),
        adapter=_Adapter([_input(file_digest="a" * 64)]),
        watcher_factory=lambda _schedule: watcher,
        cutover_admitted=admitted,
        profile_process_is_sole=sole,
        review_imported_notes=review,
    )
    try:
        await owner.start()
        await asyncio.sleep(0)
        snapshot = owner.snapshot()
        assert calls == ["review"] * expected
        assert snapshot.import_review_required is bool(expected)
        if admitted and sole and marker != "wrong":
            assert snapshot.status == "active"
            assert snapshot.next_action == "sync_now"
            assert snapshot.roots[0].watching
            assert watcher.started.is_set()
        assert migrations == (["migration"] if not marker else [])
        await owner.start()
        assert len(calls) == expected
    finally:
        await owner.shutdown()


def test_warning_is_a_defaulted_exact_boolean():
    assert (
        NotesSyncRuntimeSnapshot("active", "sync_now").import_review_required is False
    )
    assert initial_lasting_sync_snapshot().import_review_required is False
    for bad in (1, None, "receipt body"):
        with pytest.raises(TypeError):
            NotesSyncRuntimeSnapshot("active", "sync_now", import_review_required=bad)
        with pytest.raises(TypeError):
            replace(initial_lasting_sync_snapshot(), import_review_required=bad)


def _projection_fixture(tmp_path, monkeypatch):
    from Tests.Notes.test_notes_sync_legacy_migration import (
        _add_note,
        _legacy_connection,
    )
    from tldw_chatbook.DB import mobile_consolidation_receipts as receipts
    from tldw_chatbook.Notes.notes_device_state_store import NotesDeviceStateStore

    # Unit projection controls only; genuine complete receipts are tested separately.
    monkeypatch.setattr(
        receipts, "_imported_note_ids", lambda connection, budget: {"imported"}
    )
    connection = _legacy_connection(tmp_path)
    # Only the cheap presence gate; the complete reader is replaced in these unit controls.
    connection.execute("CREATE TABLE mobile_consolidation_batches(batch_key TEXT)")
    connection.execute("INSERT INTO mobile_consolidation_batches VALUES ('chat')")
    connection.commit()
    root = tmp_path / "external"
    root.mkdir()
    file = root / "linked.md"
    file.write_text("preserved")
    _add_note(
        connection,
        note_id="imported",
        root=root,
        relative_path="linked.md",
        file_path=file,
    )
    store = NotesDeviceStateStore(tmp_path / "private.sqlite")
    return connection, store, root, file


def _review(connection, store):
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget
    from tldw_chatbook.Notes.notes_sync_legacy import (
        _persist_imported_notes,
        _plan_imported_notes,
    )

    budget = _TraceReadBudget()
    plan, warning = _plan_imported_notes(
        connection, store, budget, note_scope_id="local_note"
    )
    if plan is not None:
        _persist_imported_notes(store, plan, budget, note_scope_id="local_note")
    return warning


def test_projection_adds_only_new_paused_candidate_and_represented_edits_are_noop(
    tmp_path, monkeypatch
):
    connection, store, _root, file = _projection_fixture(tmp_path, monkeypatch)
    try:
        assert _review(connection, store) is False
        with store.transaction() as private:
            before = private.execute("SELECT * FROM notes_sync_bindings").fetchall()
            assert before[0][6] == "candidate"
            assert private.execute("SELECT state FROM notes_sync_roots").fetchone() == (
                "paused",
            )
        connection.execute(
            "UPDATE notes SET version=99,file_path_on_disk='/unavailable' WHERE id='imported'"
        )
        connection.commit()
        monkeypatch.setattr(
            "tldw_chatbook.Notes.notes_sync_legacy._probe_note_file",
            lambda *args: pytest.fail("represented note was probed"),
        )
        assert _review(connection, store) is False
        with store.transaction() as private:
            assert (
                private.execute("SELECT * FROM notes_sync_bindings").fetchall()
                == before
            )
            assert private.execute(
                "SELECT count(*) FROM notes_sync_legacy_migrations"
            ).fetchone() == (1,)
        assert file.read_text() == "preserved"
    finally:
        connection.close()
        store.close()


@pytest.mark.parametrize("state", ["active", "paused"])
@pytest.mark.parametrize("overlap", ["same", "ancestor", "descendant"])
def test_any_id_any_state_overlapping_root_is_not_extended(
    tmp_path, monkeypatch, state, overlap
):
    from tldw_chatbook.Notes.notes_device_state_store import NotesSyncRootRecord
    from tldw_chatbook.Notes.notes_sync_models import (
        NotesSyncDirection,
        NotesSyncRootState,
    )

    connection, store, root, file = _projection_fixture(tmp_path, monkeypatch)
    existing = (
        root
        if overlap == "same"
        else root.parent
        if overlap == "ancestor"
        else root / "child"
    )
    existing.mkdir(exist_ok=True)
    store.create_root(
        NotesSyncRootRecord(
            root_id="arbitrary-existing",
            note_scope_id="local_note",
            logical_folder_id="existing-folder" if state == "active" else None,
            canonical_path=str(existing),
            direction=NotesSyncDirection.BIDIRECTIONAL,
            state=NotesSyncRootState(state),
        )
    )
    try:
        with store.transaction() as private:
            before = private.execute("SELECT * FROM notes_sync_roots").fetchall()
        assert _review(connection, store) is True
        with store.transaction() as private:
            assert (
                private.execute("SELECT * FROM notes_sync_roots").fetchall() == before
            )
            assert private.execute(
                "SELECT count(*) FROM notes_sync_bindings"
            ).fetchone() == (0,)
        assert file.read_text() == "preserved"
    finally:
        connection.close()
        store.close()


def test_private_insert_rechecks_current_root_geometry(tmp_path, monkeypatch):
    from tldw_chatbook.Notes import notes_sync_legacy as legacy
    from tldw_chatbook.Notes.notes_device_state_store import NotesSyncRootRecord
    from tldw_chatbook.Notes.notes_sync_models import (
        NotesSyncDirection,
        NotesSyncRootState,
    )

    connection, store, root, _ = _projection_fixture(tmp_path, monkeypatch)
    native_plan = legacy.plan_legacy_notes_sync_migration

    def plan(snapshot, **kwargs):
        result = native_plan(snapshot, **kwargs)
        store.create_root(
            NotesSyncRootRecord(
                root_id="racing-root",
                note_scope_id="local_note",
                logical_folder_id="racing-folder",
                canonical_path=str(root),
                direction=NotesSyncDirection.BIDIRECTIONAL,
                state=NotesSyncRootState.ACTIVE,
            )
        )
        return result

    monkeypatch.setattr(legacy, "plan_legacy_notes_sync_migration", plan)
    try:
        with pytest.raises(legacy.LegacyNotesSyncMigrationError):
            _review(connection, store)
        with store.transaction() as private:
            assert private.execute(
                "SELECT root_id FROM notes_sync_roots"
            ).fetchall() == [("racing-root",)]
            assert private.execute(
                "SELECT count(*) FROM notes_sync_bindings"
            ).fetchone() == (0,)
    finally:
        connection.close()
        store.close()


@pytest.mark.parametrize("reason", ["represented", "no_linked", "no_receipt"])
def test_identity_noop_does_not_decode_receipts_or_probe_notes(
    tmp_path, monkeypatch, reason
):
    from tldw_chatbook.DB import mobile_consolidation_receipts as receipts
    from tldw_chatbook.Notes import notes_sync_legacy as legacy

    connection, store, _, _ = _projection_fixture(tmp_path, monkeypatch)
    try:
        if reason == "represented":
            assert _review(connection, store) is False
            connection.execute("UPDATE notes SET version=77")
        elif reason == "no_linked":
            connection.execute("DELETE FROM notes")
        else:
            connection.execute("DELETE FROM mobile_consolidation_batches")
        connection.commit()
        monkeypatch.setattr(
            receipts,
            "_imported_note_ids",
            lambda *args: pytest.fail("no-op decoded full receipts"),
        )
        monkeypatch.setattr(
            legacy, "_probe_note_file", lambda *args: pytest.fail("no-op probed a note")
        )
        assert _review(connection, store) is False
    finally:
        connection.close()
        store.close()


@pytest.mark.integration
def test_actual_public_receipt_adds_disjoint_candidate_after_core_reader_closes(
    tmp_path, monkeypatch
):
    import hashlib
    import json
    import sqlite3
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    from types import SimpleNamespace

    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import (
        _capture_fixture,
    )
    from Tests.Backup_Recovery.test_mobile_owning_transactions import (
        _ordinary_imported_owner,
        _small_native_seed,
    )
    from Tests.network_guard import blocked_attempts
    from Tests.real_profile_guard import take_violations
    from tldw_chatbook.Backup_Recovery import (
        credentials,
        profile_paths,
        recovery_service,
    )
    from tldw_chatbook.Backup_Recovery.inventory import read_profile_config
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        require_mobile_cutover,
    )
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.Chat.console_trace_read_budget import (
        _mobile_budget_pair,
        _TraceReadBudget,
    )
    from tldw_chatbook.DB import mobile_consolidation_receipts as receipts
    from tldw_chatbook.Notes import notes_sync_legacy as legacy
    from tldw_chatbook.Notes.notes_device_state_store import NotesDeviceStateStore
    from tldw_chatbook.Notes.notes_sync_runtime import (
        CUTOVER_MARKER,
        build_notes_sync_import_reviewer,
    )
    from tldw_chatbook.runtime_policy.server_credentials import (
        KeyringServerCredentialStore,
    )

    seeded = {}

    def seed(stores, *, legacy, **options):
        value = _small_native_seed(
            stores,
            legacy=legacy,
            source_case="linked_note_capture",
            note_root=tmp_path / "external-legacy-note" if legacy else None,
            **options,
        )
        if legacy:
            seeded.update(value["operator_note"])
        return value

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seed)
    with _capture_fixture(tmp_path, monkeypatch) as values:
        _, journal, _, stores, preview, _, _ = values
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(preview.target_selector))

    class EmptyKeyring:
        def get_password(self, *_):
            return None

    credential_store = KeyringServerCredentialStore(keyring_backend=EmptyKeyring())
    monkeypatch.setattr(credentials, "_credential_store", lambda: credential_store)
    source_before = mobile_consolidation_fixture.fingerprints(preview.source_dir)
    external_before = hashlib.sha256(Path(seeded["path"]).read_bytes()).hexdigest()
    budgets, errors, decoded, closed_handles = [], [], [], []
    native_init, native_issue = _TraceReadBudget.__init__, recovery_service.issue_code

    def counted(actual):
        native_init(actual)
        budgets.append(actual)

    def issue(error, **kwargs):
        errors.append((type(error).__name__, str(error)))
        return native_issue(error, **kwargs)

    monkeypatch.setattr(_TraceReadBudget, "__init__", counted)
    monkeypatch.setattr(recovery_service, "issue_code", issue)
    report = {
        "scope": "actual public receipt/native Notes component; not full PRESENT App startup"
    }
    service = RecoveryService(journal.root.parent)
    try:
        preview = service.preview_mobile_consolidation(
            preview.source_selector, preview.target_selector
        )
        operation = service.start_mobile_consolidation(
            preview,
            destination_confirmed=True,
            rollback_password=b"private-notes-receipt-component",
        )
        state = service.wait(operation)
        report.update(operation=operation, state=state["state"], errors=errors)
        assert state["state"] == "succeeded", report
        assert not service.pending_operations()
    finally:
        service.close()
        report["pairs_after_service_close"] = [
            {
                "coupled": _mobile_budget_pair(row._parent, row),
                "aggregate": row._parent._remaining,
                "native": row._remaining,
            }
            for row in budgets
            if row._parent is not None
        ]
        (tmp_path / "notes-receipt-native.json").write_text(
            json.dumps(report, indent=2)
        )
    store = None
    try:
        assert require_mobile_cutover(preview.source_dir, stores) is None
        state_path = (
            profile_paths.user_data_dir(read_profile_config(preview.target_selector))
            / "tldw_chatbook_notes_sync_state.db"
        )
        store = NotesDeviceStateStore(state_path)
        store.set_setting(NotesSyncStoreSetting("cutover_marker", CUTOVER_MARKER))
        store.close()
        from tldw_chatbook import app_service_wiring as wiring

        monkeypatch.setattr(wiring, "get_notes_sync_state_db_path", lambda: state_path)
        with _ordinary_imported_owner(stores, "db.chachanotes.primary") as core:
            borrowed = core.get_connection()
            app = SimpleNamespace(app_config={}, chachanotes_db=core)
            assert wiring._notes_sync_boot_start_evidence(app) is True
            assert wiring._notes_sync_state_has_imports(state_path, core) is True
            assert core.get_connection() is borrowed
            with core.transaction() as cursor:
                note = cursor.execute(
                    "SELECT id,version,last_synced_disk_file_hash FROM notes WHERE file_path_on_disk=?",
                    (seeded["path"],),
                ).fetchone()
            assert note is not None
            note_id, version, digest = note
            assert digest == seeded["sha256"]
            native_close = core.close_connection

            def close():
                connection = getattr(core._local, "conn", None)
                native_close()
                if connection is not None:
                    with pytest.raises(
                        sqlite3.ProgrammingError, match="closed database"
                    ):
                        sqlite3.Connection.in_transaction.__get__(connection)
                    closed_handles.append(connection)

            monkeypatch.setattr(core, "close_connection", close)
            native_read = receipts._read_owner_receipt

            def read(connection, operation_id, source, owner, budget):
                result = native_read(connection, operation_id, source, owner, budget)
                decoded.append(
                    (operation_id, owner, connection.in_transaction, budget._remaining)
                )
                return result

            monkeypatch.setattr(receipts, "_read_owner_receipt", read)
            native_private_open = NotesDeviceStateStore._open_schema_ready_connection
            private_open_checks = []

            def private_open(actual, *args, **kwargs):
                current = getattr(core._local, "conn", None)
                assert current is None or not current.in_transaction
                private_open_checks.append(True)
                return native_private_open(actual, *args, **kwargs)

            monkeypatch.setattr(
                NotesDeviceStateStore, "_open_schema_ready_connection", private_open
            )
            native_persist = legacy._persist_imported_notes

            def persist(*args, **kwargs):
                assert getattr(core._local, "conn", None) is None
                assert closed_handles
                return native_persist(*args, **kwargs)

            monkeypatch.setattr(legacy, "_persist_imported_notes", persist)
            reviewer = build_notes_sync_import_reviewer(
                database_path=state_path,
                core=core,
                note_scope_id="local_note",
                file_notes_binding=lambda: None,
                private_paths=(state_path, stores.chat_path),
            )
            issued_before = len(budgets)
            with ThreadPoolExecutor(max_workers=1) as worker:
                assert worker.submit(reviewer).result() is False
            assert len(budgets) == issued_before + 1
            assert budgets[-1]._parent is None
            assert decoded and all(
                row[0] == operation and row[1:3] == ("chat", True) for row in decoded
            )
            assert core.get_connection() is borrowed
            with store.transaction() as private:
                bindings = private.execute(
                    "SELECT note_id,note_version,content_digest,state FROM notes_sync_bindings"
                ).fetchall()
                roots = private.execute(
                    "SELECT canonical_path,state,last_status_code FROM notes_sync_roots"
                ).fetchall()
                reviews = private.execute(
                    "SELECT state FROM notes_sync_legacy_migrations"
                ).fetchall()
                assert bindings == [(note_id, version, digest, "candidate")]
                assert roots == [
                    (
                        str(Path(seeded["root"]).resolve()),
                        "paused",
                        "migration_review_required",
                    )
                ]
                assert reviews == [("pending_review",)]
            receipt_count = len(decoded)
            assert core.update_note(
                note_id,
                {"content": "ordinary represented edit"},
                expected_version=version,
            )
            assert reviewer() is False
            assert core.get_connection() is borrowed
            assert len(decoded) == receipt_count
            with store.transaction() as private:
                assert (
                    private.execute(
                        "SELECT note_id,note_version,content_digest,state FROM notes_sync_bindings"
                    ).fetchall()
                    == bindings
                )
                assert (
                    private.execute(
                        "SELECT state FROM notes_sync_legacy_migrations"
                    ).fetchall()
                    == reviews
                )
            report.update(
                note_id=note_id,
                version=version,
                digest=digest,
                roots=roots,
                bindings=bindings,
                reviews=reviews,
                decoded=decoded,
                worker_reader_closed_before_persist=True,
                private_schema_outside_core=private_open_checks,
                ordinary_allowance_remaining=[
                    row._remaining for row in budgets[issued_before:]
                ],
                borrowed_handle_preserved=True,
            )
        assert require_mobile_cutover(preview.source_dir, stores) is None
    finally:
        if store is not None:
            handles = tuple(store._connections)
            store.close()
            for connection in handles:
                with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                    sqlite3.Connection.in_transaction.__get__(connection)
            report["private_handles_closed"] = len(handles)
        report.update(
            source_after=mobile_consolidation_fixture.fingerprints(preview.source_dir),
            source_before=source_before,
            external_after=hashlib.sha256(
                Path(seeded["path"]).read_bytes()
            ).hexdigest(),
            external_before=external_before,
            profile_violations=take_violations(),
            network_violations=blocked_attempts(),
            core_handles_closed=len(closed_handles),
        )
        (tmp_path / "notes-receipt-native.json").write_text(
            json.dumps(report, indent=2)
        )
        assert report["source_after"] == source_before
        assert report["external_after"] == external_before
        assert not report["profile_violations"] and not report["network_violations"]


def _current_core_with_link(tmp_path):
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    core = CharactersRAGDB(tmp_path / "canonical.db", "notes-import-refusal")
    note_id = core.add_note("kept", "kept note")
    root = tmp_path / "external-link"
    root.mkdir()
    file = root / "note.md"
    file.write_text("kept note")
    with core.transaction() as cursor:
        cursor.execute(
            "UPDATE notes SET file_path_on_disk=?,relative_file_path_on_disk=?,sync_root_folder=?,last_synced_disk_file_hash=? WHERE id=?",
            (str(file), file.name, str(root), "a" * 64, note_id),
        )
    return core, note_id


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ["malformed", "oversized"])
async def test_real_receipt_refusal_preserves_existing_active_runtime_and_closes_private_handles(
    tmp_path, monkeypatch, corruption
):
    import sqlite3

    from tldw_chatbook.Notes.notes_device_state_store import NotesDeviceStateStore
    from tldw_chatbook.Notes.notes_sync_runtime import build_notes_sync_import_reviewer

    core, _ = _current_core_with_link(tmp_path)
    with core.transaction() as cursor:
        cursor.execute(
            "INSERT INTO mobile_consolidation_batches VALUES (?,?,?,?,?)",
            (
                "bad-current-receipt",
                "a" * 64,
                "chat",
                "{" if corruption == "malformed" else "x" * (16 * 1024 * 1024 + 1),
                "2026-10-31T00:00:00Z",
            ),
        )
    state_root = tmp_path / "state"
    state_root.mkdir()
    store = _store(state_root)
    callback_handles = []
    native_open = NotesDeviceStateStore._open_schema_ready_connection

    def opened(actual, *args, **kwargs):
        connection = native_open(actual, *args, **kwargs)
        if actual is not store:
            callback_handles.append(connection)
        return connection

    monkeypatch.setattr(NotesDeviceStateStore, "_open_schema_ready_connection", opened)
    reviewer = build_notes_sync_import_reviewer(
        database_path=store.db_path,
        core=core,
        note_scope_id="local_note",
        file_notes_binding=lambda: None,
        private_paths=(store.db_path, core.db_path),
    )
    watcher = _Watcher()
    owner = NotesSyncRuntimeOwner(
        store=store,
        migrate_legacy=lambda: pytest.fail("present marker migrated"),
        coordinator=_Coordinator(),
        adapter=_Adapter([_input(file_digest="a" * 64)]),
        watcher_factory=lambda _: watcher,
        cutover_admitted=True,
        profile_process_is_sole=True,
        review_imported_notes=reviewer,
    )
    try:
        with store.transaction() as private:
            roots = private.execute(
                "SELECT root_id,state,canonical_path FROM notes_sync_roots"
            ).fetchall()
        await owner.start()
        await asyncio.sleep(0)
        state = owner.snapshot()
        assert state.import_review_required and state.status == "active"
        assert state.next_action == "sync_now" and state.roots[0].watching
        with store.transaction() as private:
            assert (
                private.execute(
                    "SELECT root_id,state,canonical_path FROM notes_sync_roots"
                ).fetchall()
                == roots
            )
            assert private.execute(
                "SELECT count(*) FROM notes_sync_bindings"
            ).fetchone() == (0,)
            assert private.execute(
                "SELECT count(*) FROM notes_sync_legacy_migrations"
            ).fetchone() == (0,)
        assert callback_handles
        for connection in callback_handles:
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                sqlite3.Connection.in_transaction.__get__(connection)
    finally:
        await owner.shutdown()
        core.close_connection()


def test_borrowed_core_transaction_refuses_before_private_schema_creation(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Notes.notes_device_state_store import NotesDeviceStateStore
    from tldw_chatbook.Notes.notes_sync_runtime import build_notes_sync_import_reviewer

    core, _ = _current_core_with_link(tmp_path)
    path = tmp_path / "not-created.db"
    with core.transaction() as cursor:
        cursor.execute(
            "INSERT INTO mobile_consolidation_batches VALUES (?,?,?,?,?)",
            ("present", "a" * 64, "chat", "{}", "2026-10-31T00:00:00Z"),
        )
    reviewer = build_notes_sync_import_reviewer(
        database_path=path,
        core=core,
        note_scope_id="local_note",
        file_notes_binding=lambda: None,
        private_paths=(),
    )
    monkeypatch.setattr(
        NotesDeviceStateStore,
        "initialize",
        lambda _: pytest.fail("private schema opened under Core TX"),
    )
    try:
        with core.transaction() as cursor:
            borrowed = cursor.connection
            with pytest.raises(ValueError, match="canonical_snapshot_still_borrowed"):
                reviewer()
            assert borrowed.in_transaction
            assert core.get_connection() is borrowed
        assert not path.exists()
    finally:
        core.close_connection()


@pytest.mark.parametrize("case", ["no_receipt", "no_linked"])
def test_callback_fast_path_never_creates_private_store(tmp_path, monkeypatch, case):
    from tldw_chatbook.DB import mobile_consolidation_receipts as receipts
    from tldw_chatbook.Notes.notes_device_state_store import NotesDeviceStateStore
    from tldw_chatbook.Notes.notes_sync_runtime import build_notes_sync_import_reviewer

    core, note_id = _current_core_with_link(tmp_path)
    if case == "no_linked":
        with core.transaction() as cursor:
            cursor.execute("UPDATE notes SET deleted=1 WHERE id=?", (note_id,))
            cursor.execute(
                "INSERT INTO mobile_consolidation_batches VALUES (?,?,?,?,?)",
                ("malformed-unrelated", "a" * 64, "chat", "{", "2026-10-31T00:00:00Z"),
            )
    path = tmp_path / "not-created.db"
    reviewer = build_notes_sync_import_reviewer(
        database_path=path,
        core=core,
        note_scope_id="local_note",
        file_notes_binding=lambda: None,
        private_paths=(),
    )
    monkeypatch.setattr(
        NotesDeviceStateStore,
        "initialize",
        lambda _: pytest.fail("no-op initialized private store"),
    )
    monkeypatch.setattr(
        receipts, "_imported_note_ids", lambda *_: pytest.fail("no-op decoded receipts")
    )
    try:
        handle = core.get_connection()
        assert reviewer() is False
        assert core.get_connection() is handle
        assert not path.exists()
    finally:
        core.close_connection()


@pytest.mark.asyncio
async def test_closing_during_added_review_does_not_load_roots_or_start_watcher(
    tmp_path,
):
    import threading

    store = _store(tmp_path)
    entered, release = threading.Event(), threading.Event()

    def review():
        entered.set()
        release.wait(10)
        return False

    watcher = _Watcher()
    owner = NotesSyncRuntimeOwner(
        store=store,
        migrate_legacy=lambda: None,
        coordinator=_Coordinator(),
        adapter=_Adapter([_input()]),
        watcher_factory=lambda _: watcher,
        cutover_admitted=True,
        profile_process_is_sole=True,
        review_imported_notes=review,
    )
    task = asyncio.create_task(owner.start())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        closing = asyncio.create_task(owner.shutdown())
        await asyncio.sleep(0)
        release.set()
        await task
        await closing
        assert not watcher.started.is_set()
        assert owner.snapshot().roots == ()
    finally:
        release.set()
        await owner.shutdown()
