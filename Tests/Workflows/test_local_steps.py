"""Bounded source reads and real, captured Local Note effects."""

import asyncio
import json
import logging
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger as loguru_logger

from tldw_chatbook.Agents.builtin_tool_gate import BuiltinToolGate
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_chatbook.MCP.permission_store import MCPPermissionStore
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.runtime_policy import (
    PolicyDeniedError,
    RuntimeSourceState,
    ServicePolicyEnforcer,
)
from tldw_chatbook.Workflows.session_permissions import (
    EffectRequest,
    WorkflowPermissions,
)


def test_source_read_does_not_change_permissions(tmp_path):
    from tldw_chatbook.Workflows.local_steps import read_local_text

    source = tmp_path / "source.txt"
    source.write_text("hello é", encoding="utf-8")
    source.chmod(0o644)
    assert (
        read_local_text(source, protected_paths=(), before_read=lambda: None)
        == "hello é"
    )
    assert stat.S_IMODE(source.stat().st_mode) == 0o644


@pytest.mark.parametrize(
    "payload",
    [b"a" * (1024 * 1024 + 1), b"\xff", b"\x00" * 200000],
    ids=["raw_limit", "utf8", "json_limit"],
)
def test_source_rejects_byte_decode_and_serialized_result_overflow(tmp_path, payload):
    from tldw_chatbook.Workflows.local_steps import read_local_text

    source = tmp_path / "source.txt"
    source.write_bytes(payload)
    with pytest.raises(ValueError):
        read_local_text(source, protected_paths=(), before_read=lambda: None)


def test_source_exact_serialized_result_boundary(tmp_path):
    from tldw_chatbook.Workflows.local_steps import read_local_text

    source = tmp_path / "source.txt"
    # {"text":"..."} occupies 11 bytes besides this ASCII payload.
    content = "a" * (1024 * 1024 - 11)
    source.write_text(content)
    assert (
        read_local_text(source, protected_paths=(), before_read=lambda: None) == content
    )
    source.write_text(content + "a")
    with pytest.raises(ValueError):
        read_local_text(source, protected_paths=(), before_read=lambda: None)


@pytest.mark.parametrize(
    "kind",
    [
        "symlink",
        "hardlink",
        "fifo",
        "directory",
        "extension",
        "parent_link",
        "parent_mode",
        "owner",
    ],
)
def test_unsafe_source_refused_before_raw_file_open(tmp_path, monkeypatch, kind):
    from tldw_chatbook.Workflows.local_steps import read_local_text

    source = tmp_path / "source.txt"
    source.write_text("secret")
    if kind == "symlink":
        target = tmp_path / "target"
        source.rename(target)
        source.symlink_to(target)
    elif kind == "hardlink":
        os.link(source, tmp_path / "alias")
    elif kind in {"fifo", "directory"}:
        source.unlink()
        os.mkfifo(source) if kind == "fifo" else source.mkdir()
    elif kind == "extension":
        source = source.rename(tmp_path / "source.bin")
    elif kind == "parent_link":
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        source = alias / source.name
    elif kind == "parent_mode":
        tmp_path.chmod(0o777)
    elif kind == "owner":
        real_lstat = Path.lstat

        def wrong_owner(path, *args, **kwargs):
            entry = real_lstat(path, *args, **kwargs)
            if path == source:
                fields = list(entry)
                fields[4] = os.geteuid() + 1
                return os.stat_result(fields)
            return entry

        monkeypatch.setattr(Path, "lstat", wrong_owner)
    real_open = os.open
    opened = []

    def observe_open(path, flags, *args, **kwargs):
        if Path(path) == source:
            opened.append(path)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", observe_open)
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {observe_open})
    try:
        with pytest.raises((ValueError, OSError)):
            read_local_text(source, protected_paths=(), before_read=lambda: None)
        assert opened == []
    finally:
        if kind == "parent_mode":
            tmp_path.chmod(0o700)


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
def test_known_database_identity_refused_before_raw_open(tmp_path, monkeypatch, suffix):
    from tldw_chatbook.Workflows.local_steps import read_local_text

    source = tmp_path / "source.txt"
    source.write_text("never open a known database inode")
    database = tmp_path / "notes.db"
    # A visible alias with st_nlink == 1: must compare stat identities too.
    Path(str(database) + suffix).symlink_to(source)
    real_open = os.open
    opened = []

    def observe_open(path, flags, *args, **kwargs):
        if Path(path) == source:
            opened.append(path)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", observe_open)
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {observe_open})
    with pytest.raises(ValueError):
        read_local_text(source, protected_paths=(database,), before_read=lambda: None)
    assert opened == []


@pytest.mark.parametrize("guard", ["O_NOFOLLOW", "O_NONBLOCK", "unverified"])
def test_unverified_platform_refuses_source(tmp_path, monkeypatch, guard):
    from tldw_chatbook.Utils.private_paths import PrivatePathResult, PrivatePathStatus
    from tldw_chatbook.Workflows import local_steps

    source = tmp_path / "source.txt"
    source.write_text("hello")
    if guard == "unverified":
        monkeypatch.setattr(
            local_steps,
            "verify_trusted_directory",
            lambda *a, **kw: PrivatePathResult(
                tmp_path, PrivatePathStatus.UNVERIFIED_PLATFORM
            ),
        )
    else:
        monkeypatch.setattr(os, guard, 0)
    with pytest.raises((ValueError, OSError)):
        local_steps.read_local_text(
            source, protected_paths=(), before_read=lambda: None
        )


def test_before_read_is_last_check_and_open_is_readonly(tmp_path, monkeypatch):
    from tldw_chatbook.Workflows.local_steps import read_local_text

    source = tmp_path / "source.txt"
    source.write_text("hello")
    events = []
    real_open = os.open

    def observe_open(path, flags, *args, **kwargs):
        if Path(path) == source:
            events.append("open")
            assert flags & os.O_ACCMODE == os.O_RDONLY
            assert flags & os.O_NOFOLLOW
            assert flags & os.O_NONBLOCK
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", observe_open)
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {observe_open})
    assert (
        read_local_text(
            source, protected_paths=(), before_read=lambda: events.append("check")
        )
        == "hello"
    )
    assert events == ["check", "open"]


@pytest.fixture
def notes_bridge(tmp_path, monkeypatch):
    from tldw_chatbook.Chat.console_library_policy import ConsoleLibraryMigrationSeed
    from tldw_chatbook.Workflows.local_steps import capture_local_note_destination

    # This bridge exercises real temporary Notes owners, not process-lived config
    # selection. Keep unrelated migration defaults independent of fixture HOME.
    monkeypatch.setattr(
        "tldw_chatbook.Notes.Notes_Library.load_console_library_migration_seed",
        lambda: ConsoleLibraryMigrationSeed(auto_retrieve_on_send=False),
    )
    template = CharactersRAGDB(tmp_path / "notes.db", client_id="template")
    owner = NotesInteropService(tmp_path, "application", global_db_to_use=template)
    policy = {"state": RuntimeSourceState(active_source="local")}
    scope = NotesScopeService(
        owner,
        None,
        policy_enforcer=ServicePolicyEnforcer(state_provider=lambda: policy["state"]),
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        destination = pool.submit(
            capture_local_note_destination, scope, user_id="reader"
        ).result(timeout=10)
        db = destination.db

        def run(function, *args, **kwargs):
            def checked():
                try:
                    return function(*args, **kwargs)
                finally:
                    assert threading.get_ident() != threading.main_thread().ident
                    assert getattr(db._local, "conn", None) is None

            return pool.submit(checked).result(timeout=10)

        try:
            yield SimpleNamespace(
                template=template,
                owner=owner,
                scope=scope,
                policy=policy,
                destination=destination,
                db=db,
                pool=pool,
                run=run,
            )
        finally:
            # Also release deliberately opened failure-test connections on their owner.
            pool.submit(db.close_connection).result(timeout=10)
            db.close_connection()
            owner.close_all_user_connections()
            template.close_connection()


def note_rows(bridge):
    with bridge.template.transaction() as cursor:
        return [
            dict(row)
            for row in cursor.execute(
                "SELECT id, title, content, client_id, deleted FROM notes"
            )
        ]


def create(bridge, **kwargs):
    from tldw_chatbook.Workflows.local_steps import create_local_note

    arguments = {
        "create_note_id": "attempt",
        "title": "  Reviewed  ",
        "content": "exact\n é ",
        "before_write": lambda: None,
    }
    arguments.update(kwargs)
    return bridge.run(create_local_note, bridge.destination, **arguments)


def read(bridge, note_id="attempt"):
    from tldw_chatbook.Workflows.local_steps import read_local_note

    return bridge.run(read_local_note, bridge.destination, note_id=note_id)


def test_capture_retains_actual_cache_and_closes_capture_thread(notes_bridge):
    bridge = notes_bridge
    assert bridge.destination.db is bridge.owner.notes_db("reader")
    assert bridge.destination.db is not bridge.template
    assert bridge.destination.client_id == "reader"
    assert bridge.destination.db_path == str(bridge.template.db_path_str)
    assert (
        bridge.pool.submit(lambda: getattr(bridge.db._local, "conn", None)).result()
        is None
    )
    assert "reader" not in repr(bridge.destination)
    assert str(bridge.template.db_path_str) not in repr(bridge.destination)
    with pytest.raises(FrozenInstanceError):
        bridge.destination.user_id = "other"


def test_real_create_readback_and_duplicate_preserve_one_exact_row(notes_bridge):
    assert create(notes_bridge) == "attempt"
    row = read(notes_bridge)
    assert (row["id"], row["title"], row["content"], row["client_id"]) == (
        "attempt",
        "Reviewed",
        "exact\n é ",
        "reader",
    )
    with pytest.raises(CharactersRAGDBError):
        create(notes_bridge, content="replacement")
    assert note_rows(notes_bridge) == [
        {
            "id": "attempt",
            "title": "Reviewed",
            "content": "exact\n é ",
            "client_id": "reader",
            "deleted": 0,
        }
    ]
    assert notes_bridge.owner.notes_db("reader") is notes_bridge.db


@pytest.mark.parametrize(
    "values",
    [
        {"title": " \n "},
        {"content": None},
        {"create_note_id": " "},
        {"create_note_id": " attempt "},
    ],
)
def test_invalid_note_input_has_no_effect(notes_bridge, values):
    with pytest.raises(ValueError):
        create(notes_bridge, **values)
    assert note_rows(notes_bridge) == []


@pytest.mark.parametrize("operation", ["create", "read"])
def test_actual_policy_denial_prevents_local_effect(notes_bridge, operation):
    if operation == "read":
        create(notes_bridge)
    notes_bridge.policy["state"] = None
    with pytest.raises(PolicyDeniedError):
        create(notes_bridge) if operation == "create" else read(notes_bridge)
    assert len(note_rows(notes_bridge)) == (1 if operation == "read" else 0)


@pytest.mark.parametrize(
    "change", ["service", "cache", "missing", "path", "client", "user"]
)
@pytest.mark.parametrize("operation", ["create", "read"])
def test_changed_destination_refuses_without_replacement(
    notes_bridge, monkeypatch, change, operation
):
    bridge = notes_bridge
    if change == "service":
        bridge.scope.local_notes_service = None
    elif change == "cache":
        bridge.owner._db_instances["reader"] = bridge.template
    elif change == "missing":
        del bridge.owner._db_instances["reader"]
    elif change == "path":
        monkeypatch.setattr(bridge.db, "db_path_str", "different")
    elif change == "client":
        monkeypatch.setattr(bridge.db, "client_id", "different")
    else:
        bridge.destination = replace(bridge.destination, user_id="different")
    before = dict(bridge.owner._db_instances)
    with pytest.raises(ValueError):
        create(bridge) if operation == "create" else read(bridge)
    assert bridge.owner._db_instances == before
    assert note_rows(bridge) == []


def test_mutated_template_does_not_replace_the_actual_cached_destination(notes_bridge):
    notes_bridge.owner.unified_db_template = None
    assert create(notes_bridge) == "attempt"
    assert read(notes_bridge)["title"] == "Reviewed"


def test_bound_context_holds_existing_lock_and_never_manufactures_db(notes_bridge):
    bridge = notes_bridge
    with bridge.owner.bound_notes_db("reader", bridge.db) as actual:
        assert actual is bridge.db
        assert bridge.owner.notes_db("reader") is actual
        assert not bridge.owner._db_lock.acquire(blocking=False)
    assert bridge.owner._db_lock.acquire(blocking=False)
    bridge.owner._db_lock.release()
    with pytest.raises(ValueError), bridge.owner.bound_notes_db("missing", bridge.db):
        pytest.fail("missing cache must refuse")
    assert "missing" not in bridge.owner._db_instances


def test_bound_context_nonblocking_contention_preserves_owner_lock(notes_bridge):
    bridge = notes_bridge
    before = dict(bridge.owner._db_instances)
    with bridge.owner._db_lock:
        with (
            pytest.raises(BlockingIOError, match="note_destination_busy"),
            bridge.owner.bound_notes_db("reader", bridge.db, blocking=False),
        ):
            pytest.fail("a busy route must not be entered")
        assert not bridge.owner._db_lock.acquire(blocking=False)
        assert bridge.owner._db_instances == before
    with bridge.owner.bound_notes_db("reader", bridge.db, blocking=False) as actual:
        assert actual is bridge.db


def test_bound_context_default_worker_waits_for_existing_owner(notes_bridge):
    bridge = notes_bridge
    started = threading.Event()

    def worker():
        started.set()
        with bridge.owner.bound_notes_db("reader", bridge.db) as actual:
            return actual

    with bridge.owner._db_lock:
        future = bridge.pool.submit(worker)
        assert started.wait(5)
        assert not future.done()
    assert future.result(timeout=5) is bridge.db


@pytest.mark.parametrize("blocking", [True, False])
def test_bound_context_releases_on_body_error_and_changed_route(notes_bridge, blocking):
    bridge = notes_bridge
    with (
        pytest.raises(RuntimeError, match="body failed"),
        bridge.owner.bound_notes_db("reader", bridge.db, blocking=blocking),
    ):
        raise RuntimeError("body failed")
    assert bridge.owner._db_lock.acquire(blocking=False)
    bridge.owner._db_lock.release()
    before = dict(bridge.owner._db_instances)
    with (
        pytest.raises(ValueError, match="note_destination_changed"),
        bridge.owner.bound_notes_db("missing", bridge.db, blocking=blocking),
    ):
        pytest.fail("missing route must refuse without replacement")
    assert bridge.owner._db_lock.acquire(blocking=False)
    bridge.owner._db_lock.release()
    assert bridge.owner._db_instances == before


@pytest.mark.parametrize("phase", ["rollback", "before_commit", "after_commit"])
def test_real_transaction_barriers_and_error_readback(notes_bridge, monkeypatch, phase):
    from tldw_chatbook.Workflows.local_steps import create_local_note

    bridge = notes_bridge
    entered, release = threading.Event(), threading.Event()
    original = bridge.owner.note_transaction

    @contextmanager
    def held_transaction(user_id):
        with original(user_id) as cursor:
            yield cursor
            if phase != "after_commit":
                # This is the actual uncommitted row on the writer's connection.
                assert cursor.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 1
                entered.set()
                assert release.wait(10)
                if phase == "rollback":
                    raise CharactersRAGDBError("injected_before_commit")
        if phase == "after_commit":
            entered.set()
            assert release.wait(10)
            raise CharactersRAGDBError("lost_response_after_commit")

    monkeypatch.setattr(bridge.owner, "note_transaction", held_transaction)
    future = bridge.pool.submit(
        create_local_note,
        bridge.destination,
        create_note_id="attempt",
        title="Reviewed",
        content="accepted",
        before_write=lambda: None,
    )
    try:
        assert entered.wait(10)
        assert not future.done()
        assert len(note_rows(bridge)) == (1 if phase == "after_commit" else 0)
    finally:
        release.set()
    if phase == "before_commit":
        assert future.result(timeout=10) == "attempt"
    else:
        with pytest.raises(CharactersRAGDBError):
            future.result(timeout=10)
    assert (
        bridge.pool.submit(lambda: getattr(bridge.db._local, "conn", None)).result()
        is None
    )
    row = read(bridge)
    if phase == "rollback":
        assert row is None
        assert note_rows(bridge) == []
    else:
        assert (row["id"], row["title"], row["content"]) == (
            "attempt",
            "Reviewed",
            "accepted",
        )
        assert len(note_rows(bridge)) == 1


def test_readback_reports_changed_content_and_refuses_deleted_row(notes_bridge):
    create(notes_bridge)
    notes_bridge.owner.update_note(
        "reader", "attempt", {"title": "Edited", "content": "different"}, 1
    )
    assert (read(notes_bridge)["title"], read(notes_bridge)["content"]) == (
        "Edited",
        "different",
    )
    notes_bridge.owner.soft_delete_note("reader", "attempt", 2)
    assert read(notes_bridge) is None
    assert note_rows(notes_bridge)[0]["deleted"] == 1


@pytest.mark.parametrize("kind", ["null_id", "db_error"])
def test_real_bridge_error_logs_exclude_private_title_and_content(
    notes_bridge, monkeypatch, caplog, kind
):
    bridge = notes_bridge
    canary = "PRIVATE_WORKFLOW_CANARY_7h4"

    def fail_insert(*args, **kwargs):
        if kind == "db_error":
            raise CharactersRAGDBError("injected_storage_failure")

    # Real scope -> transaction -> interop -> DB.add_note error handling.
    monkeypatch.setattr(bridge.db, "_add_note_with_cursor", fail_insert)
    records = []
    sink = loguru_logger.add(records.append, level="DEBUG", format="{message}")
    try:
        with caplog.at_level(logging.DEBUG), pytest.raises(CharactersRAGDBError):
            create(bridge, title=canary, content=canary)
    finally:
        loguru_logger.remove(sink)
    assert note_rows(bridge) == []
    assert canary not in caplog.text + "".join(records)


@pytest.mark.parametrize(
    ("note_id", "note_ref"),
    [(None, "generated"), ("PRIVATE_ATTEMPT_CANARY_7h4", "759916bcadcd")],
)
def test_null_note_failure_keeps_safe_correlation(
    notes_bridge, monkeypatch, caplog, note_id, note_ref
):
    # Exercise the real interop error path and logging; only the DB's invalid
    # return is injected. Losing the references must fail without exposing input.
    monkeypatch.setattr(notes_bridge.db, "add_note", lambda **kwargs: None)
    with caplog.at_level(logging.ERROR), pytest.raises(CharactersRAGDBError):
        notes_bridge.owner.add_note(
            "reader", "PRIVATE_TITLE", "PRIVATE_CONTENT", note_id=note_id
        )
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "tldw_chatbook.Notes.Notes_Library"
    ]
    assert len(messages) == 1
    assert "user_ref=3d0941964aa3" in messages[0]
    assert f"note_ref={note_ref}" in messages[0]
    assert all(value not in messages[0] for value in ("PRIVATE_", "reader"))
    assert note_rows(notes_bridge) == []


@pytest.fixture
def effect_authority(tmp_path):
    store = MCPPermissionStore(tmp_path / "permissions.json")
    payload = store.load()
    payload["profiles"]["default"]["servers"] = {
        "agent:builtin": {
            "tools": {
                name: {"state": "allow"}
                for name in ("create_note", "workflow_read_file")
            }
        }
    }
    store.save(payload)
    permissions = WorkflowPermissions(
        BuiltinToolGate(SimpleNamespace(permission_store=store))
    )

    def check(kind):
        decision = permissions.check(EffectRequest("run", "step", kind, "{}"))
        if decision.refusal is not None:
            raise ValueError(decision.refusal_code)

    return store, check


@pytest.mark.parametrize("kind", ["file", "note"])
def test_effect_time_revocation_prevents_io(
    notes_bridge, effect_authority, tmp_path, kind
):
    from tldw_chatbook.Workflows.local_steps import read_local_text

    store, check = effect_authority
    check(kind)
    payload = json.loads(store.path.read_text())
    payload["kill_switch"] = True
    store.save(payload)
    with pytest.raises(ValueError, match="kill_switch"):
        if kind == "file":
            source = tmp_path / "input.txt"
            source.write_text("hello")
            read_local_text(source, protected_paths=(), before_read=lambda: check(kind))
        else:
            create(notes_bridge, before_write=lambda: check(kind))
    assert note_rows(notes_bridge) == []


def test_notes_bridge_refuses_running_event_loop(notes_bridge):
    from tldw_chatbook.Workflows.local_steps import create_local_note

    async def on_loop():
        with pytest.raises(RuntimeError, match="worker"):
            create_local_note(
                notes_bridge.destination,
                create_note_id="attempt",
                title="Title",
                content="Body",
                before_write=lambda: None,
            )

    asyncio.run(on_loop())
    assert note_rows(notes_bridge) == []


@pytest.mark.parametrize("failure", ["fstat", "fdopen", "decode", "read"])
def test_source_descriptor_closed_on_failure(tmp_path, monkeypatch, failure):
    from tldw_chatbook.Workflows.local_steps import read_local_text

    source = tmp_path / "source.txt"
    source.write_bytes(b"\xff" if failure == "decode" else b"hello")
    descriptors = []
    real_open, real_fstat, real_fdopen = os.open, os.fstat, os.fdopen

    def observe_open(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags, *args, **kwargs)
        if Path(path) == source:
            descriptors.append(descriptor)
        return descriptor

    def fail_fstat(descriptor):
        if descriptor in descriptors:
            raise OSError("injected_fstat")
        return real_fstat(descriptor)

    def fail_fdopen(descriptor, mode):
        raise OSError("injected_fdopen")

    class BrokenRead:
        def __init__(self, descriptor, mode):
            self.stream = real_fdopen(descriptor, mode)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, limit):
            raise OSError("injected_read")

    monkeypatch.setattr(os, "open", observe_open)
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {observe_open})
    if failure == "fstat":
        monkeypatch.setattr(os, "fstat", fail_fstat)
    elif failure == "fdopen":
        monkeypatch.setattr(os, "fdopen", fail_fdopen)
    elif failure == "read":
        monkeypatch.setattr(os, "fdopen", BrokenRead)
    with pytest.raises((ValueError, OSError)):
        read_local_text(source, protected_paths=(), before_read=lambda: None)
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        real_fstat(descriptors[0])


def test_refused_file_effect_never_opens_payload(tmp_path, monkeypatch):
    from tldw_chatbook.Workflows.local_steps import read_local_text

    source = tmp_path / "source.txt"
    source.write_text("private")
    real_open = os.open

    def observe_open(path, flags, *args, **kwargs):
        assert Path(path) != source
        return real_open(path, flags, *args, **kwargs)

    def denied():
        raise ValueError("effect_denied")

    monkeypatch.setattr(os, "open", observe_open)
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {observe_open})
    with pytest.raises(ValueError, match="effect_denied"):
        read_local_text(source, protected_paths=(), before_read=denied)


def test_readback_rejects_actual_other_client_row(notes_bridge):
    # Same file, another Notes client: an ID match cannot establish ownership.
    notes_bridge.template.add_note("Other", "unrelated", note_id="attempt")
    with pytest.raises(ValueError, match="note_readback_mismatch"):
        read(notes_bridge)
    assert len(note_rows(notes_bridge)) == 1


def test_capture_failure_closes_actual_worker_connection(notes_bridge, monkeypatch):
    from tldw_chatbook.Workflows.local_steps import capture_local_note_destination

    bridge = notes_bridge
    original = bridge.owner.notes_db

    def changed_owner(user_id):
        db = original(user_id)
        db.get_connection()
        bridge.scope.local_notes_service = None
        return db

    monkeypatch.setattr(bridge.owner, "notes_db", changed_owner)
    with pytest.raises(ValueError, match="note_destination_changed"):
        bridge.run(capture_local_note_destination, bridge.scope, user_id="reader")
    assert bridge.owner._db_instances["reader"] is bridge.db


def test_note_effect_check_runs_under_bound_owner_lock(notes_bridge):
    def check():
        assert not notes_bridge.owner._db_lock.acquire(blocking=False)
        assert notes_bridge.owner.notes_db("reader") is notes_bridge.db

    assert create(notes_bridge, before_write=check) == "attempt"
    assert len(note_rows(notes_bridge)) == 1


def test_readback_error_closes_actual_worker_connection(notes_bridge, monkeypatch):
    create(notes_bridge)
    original = notes_bridge.db.get_note_by_id

    def lost_readback(note_id):
        assert original(note_id)["content"] == "exact\n é "
        raise CharactersRAGDBError("lost_readback")

    monkeypatch.setattr(notes_bridge.db, "get_note_by_id", lost_readback)
    with pytest.raises(CharactersRAGDBError, match="lost_readback"):
        read(notes_bridge)
    assert len(note_rows(notes_bridge)) == 1


def test_live_notes_file_alias_refused_without_raw_open(
    notes_bridge, tmp_path, monkeypatch
):
    from tldw_chatbook.Workflows.local_steps import read_local_text

    source = tmp_path / "notes.txt"
    os.link(notes_bridge.destination.db_path, source)
    real_open = os.open

    def observe_open(path, flags, *args, **kwargs):
        assert Path(path) != source
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", observe_open)
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {observe_open})
    try:
        with pytest.raises(ValueError, match="source_unsafe"):
            read_local_text(
                source,
                protected_paths=(Path(notes_bridge.destination.db_path),),
                before_read=lambda: None,
            )
    finally:
        # Remove only this test-created alias before the SQLite owner's cleanup.
        source.unlink()


def test_local_create_never_dispatches_sync_profile(notes_bridge):
    def forbidden(**kwargs):
        pytest.fail("Workflow Local Note creation must not enqueue remote sync")

    notes_bridge.scope.sync_v2_notes_producer = SimpleNamespace(
        enqueue_note_upsert=forbidden
    )
    assert create(notes_bridge) == "attempt"
    assert len(note_rows(notes_bridge)) == 1


def test_worker_cleanup_preserves_other_thread_handle_and_cached_owner(notes_bridge):
    bridge = notes_bridge
    main_connection = bridge.db.get_connection()
    assert create(bridge) == "attempt"
    assert read(bridge)["content"] == "exact\n é "
    assert bridge.db.get_connection() is main_connection
    assert main_connection.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 1
    assert bridge.owner.notes_db("reader") is bridge.db


@pytest.mark.parametrize("operation", ["capture", "create", "read"])
@pytest.mark.parametrize("closed_first", [False, True])
def test_escaping_cleanup_has_a_distinct_payload_free_signal(
    notes_bridge, monkeypatch, operation, closed_first
):
    """Task 4 can distinguish observable cleanup errors from ordinary save errors."""
    from tldw_chatbook.Workflows.local_steps import (
        LocalNoteCleanupError,
        capture_local_note_destination,
        create_local_note,
        read_local_note,
    )

    bridge = notes_bridge
    if operation == "read":
        create(bridge)
    original_close = bridge.db.close_connection
    cleanup_failure = CharactersRAGDBError("injected_cleanup_failure")

    def fail_close():
        assert threading.get_ident() != threading.main_thread().ident
        if closed_first:
            original_close()
        raise cleanup_failure

    def operation_on_worker():
        if operation == "capture":
            # Capture may encounter an already-open cached current-thread handle.
            bridge.db.get_connection()
            return capture_local_note_destination(bridge.scope, user_id="reader")
        if operation == "read":
            return read_local_note(bridge.destination, note_id="attempt")
        return create_local_note(
            bridge.destination,
            create_note_id="attempt",
            title="Reviewed",
            content="accepted",
            before_write=lambda: None,
        )

    monkeypatch.setattr(bridge.db, "close_connection", fail_close)
    try:
        future = bridge.pool.submit(operation_on_worker)
        with pytest.raises(LocalNoteCleanupError) as raised:
            future.result(timeout=10)
        assert str(raised.value) == "note_cleanup_failed"
        assert "injected_cleanup_failure" not in repr(raised.value)
        # The signal reports an escaping close error, not physical handle state.
        connection_retained = bridge.pool.submit(
            lambda: getattr(bridge.db._local, "conn", None) is not None
        ).result(timeout=10)
        assert connection_retained is not closed_first
        assert len(note_rows(bridge)) == (0 if operation == "capture" else 1)
        assert bridge.owner._db_instances["reader"] is bridge.db
    finally:
        monkeypatch.setattr(bridge.db, "close_connection", original_close)
        bridge.pool.submit(original_close).result(timeout=10)
    # A later successful readback proves the row, not earlier physical cleanup.
    if operation != "capture":
        assert read(bridge)["id"] == "attempt"
