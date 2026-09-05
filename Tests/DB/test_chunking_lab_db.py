"""Real recovery, CAS, retention, and process-crash evidence for the Lab owner."""

import os
import select
import sqlite3
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from tldw_chatbook.Chunking.lab_models import ExecutionReport, RunResult
from tldw_chatbook.Chunking.lab_state import (
    accept_result,
    capture_batch,
    edit_control,
    edit_json,
    install_batch,
    new_session,
    pin_baseline,
    replace_sample,
    undo_edit,
    update_view,
)
from tldw_chatbook.DB.Chunking_Lab_DB import (
    CheckpointConflict,
    CheckpointStore,
    RecoverySchemaError,
)


def completed_session():
    session = replace_sample(
        new_session("profile"), "exact sample\n", {"path": "deleted.txt"}
    )
    requests = capture_batch(session, tuple(session.candidates))
    session = install_batch(session, requests)
    result = RunResult(
        request=requests[0],
        status="completed",
        report=ExecutionReport(
            chunks=(
                {
                    "text": "exact sample",
                    "metadata": {"x": 1},
                    "provenance": {},
                    "span": None,
                },
            ),
            transformed_text="exact sample\n",
        ),
        started_at="start",
        finished_at="finish",
        elapsed_ms=1,
        error=None,
    )
    return accept_result(session, result)


def test_replace_preserves_unsaved_displaced_content_and_undo_across_views(tmp_path):
    store = CheckpointStore(tmp_path / "lab.sqlite3", "profile")
    old = new_session("profile")
    token = store.save(old, expected=None)
    displaced = edit_json(old, next(iter(old.candidates)), '{"unsaved":')
    restored, replacement_token = store.replace(
        new_session("other"), displaced, expected=token
    )
    assert restored.profile_key == "profile"
    assert restored.epoch != old.epoch
    with pytest.raises(CheckpointConflict):
        store.save(displaced, expected=token)
    for index in range(4):
        restored = update_view(restored, {"tab": str(index)})
        replacement_token = store.save(restored, expected=replacement_token)
    undone, undo_token = store.undo_restore(expected=replacement_token)
    assert undone.candidates == displaced.candidates
    assert undone.epoch not in {old.epoch, restored.epoch}
    assert store.load()[1] == undo_token
    with pytest.raises(ValueError, match="undo"):
        store.undo_restore(expected=undo_token)
    store.close()


def test_failed_replace_rolls_back_both_checkpoints_and_keeps_retry_authority(tmp_path):
    store = CheckpointStore(tmp_path / "lab.sqlite3", "profile")
    old = new_session("profile")
    token = store.save(old, expected=None)
    conn = store._connection()
    conn.execute(
        "CREATE TRIGGER refuse_replace BEFORE UPDATE OF epoch ON lab_state WHEN NEW.epoch != OLD.epoch BEGIN SELECT RAISE(ABORT, 'replacement refused'); END"
    )
    displaced = edit_json(old, next(iter(old.candidates)), '{"unsaved":')
    with pytest.raises(sqlite3.IntegrityError):
        store.replace(
            replace_sample(
                new_session("other"), "new imported blob", {"kind": "paste"}
            ),
            displaced,
            expected=token,
        )
    assert store.load() == (old, token)
    assert conn.execute("SELECT COUNT(*) FROM lab_checkpoints").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM lab_blobs").fetchone()[0] == 1
    conn.execute("DROP TRIGGER refuse_replace")
    retry = store.save(displaced, expected=token)
    assert store.load() == (displaced, retry)
    store.close()


def test_content_save_releases_restore_undo_and_clear_removes_all_refs(tmp_path):
    store = CheckpointStore(tmp_path / "lab.sqlite3", "profile")
    old = completed_session()
    token = store.save(old, expected=None)
    restored, token = store.replace(new_session("other"), old, expected=token)
    restored = edit_json(restored, next(iter(restored.candidates)), '{"changed":')
    token = store.save(restored, expected=token)
    with pytest.raises(ValueError, match="undo"):
        store.undo_restore(expected=token)
    restored, token = store.replace(new_session("third"), restored, expected=token)
    store.clear(expected=token)
    assert (
        store._connection().execute("SELECT COUNT(*) FROM lab_blobs").fetchone()[0] == 0
    )
    assert (
        store._connection()
        .execute("SELECT COUNT(*) FROM lab_checkpoints")
        .fetchone()[0]
        == 0
    )
    with pytest.raises(CheckpointConflict):
        store.save(restored, expected=token)
    store.close()


def test_coalesced_edit_and_undo_still_consume_restore_undo(tmp_path):
    store = CheckpointStore(tmp_path / "lab.sqlite3", "profile")
    old = new_session("profile")
    token = store.save(old, expected=None)
    restored, token = store.replace(new_session("other"), old, expected=token)
    changed = undo_edit(
        edit_json(restored, next(iter(restored.candidates)), '{"temporary":')
    )
    assert changed.candidates == restored.candidates and changed.undo == restored.undo
    token = store.save(changed, expected=token)
    with pytest.raises(ValueError, match="undo"):
        store.undo_restore(expected=token)
    store.close()


def test_replacement_retires_manifest_without_rewriting_run_provenance(tmp_path):
    store = CheckpointStore(tmp_path / "lab.sqlite3", "profile")
    old = completed_session()
    old_results = old.results
    token = store.save(old, expected=None)
    restored, token = store.replace(completed_session(), old, expected=token)
    assert restored.batch is None
    assert all(
        result["request"]["epoch"] != restored.epoch
        for result in restored.results.values()
    )
    undone, _ = store.undo_restore(expected=token)
    assert undone.results == old_results
    assert undone.batch is None
    store.close()


def test_old_checkpoint_without_content_revision_loads_with_zero(tmp_path):
    import json

    store = CheckpointStore(tmp_path / "lab.sqlite3", "profile")
    old = new_session("profile")
    store.save(old, expected=None)
    conn = store._connection()
    checkpoint_id, raw = conn.execute(
        "SELECT id,document FROM lab_checkpoints"
    ).fetchone()
    document = json.loads(raw)
    del document["session"]["content_revision"]
    conn.execute(
        "UPDATE lab_checkpoints SET document=? WHERE id=?",
        (json.dumps(document), checkpoint_id),
    )
    loaded, token = store.load()
    assert loaded.content_revision == 0
    store.save(update_view(loaded, {"tab": "Draft"}), expected=token)
    store.close()


def test_invalid_draft_survives_store_reopen(tmp_path):
    session = new_session("test-profile")
    candidate_id = next(iter(session.candidates))
    session = edit_json(session, candidate_id, '{"chunking":')
    path = tmp_path / "lab.sqlite3"
    store = CheckpointStore(path, "test-profile")
    store.save(session, expected=None)
    store.close()
    reopened = CheckpointStore(path, "test-profile")
    restored, token = reopened.load()
    assert restored.model_dump() == session.model_dump()
    assert token.revision == session.revision
    reopened.close()


def test_two_connections_require_epoch_and_generation_cas(tmp_path):
    path = tmp_path / "lab.sqlite3"
    first, second = CheckpointStore(path, "profile"), CheckpointStore(path, "profile")
    session = new_session("profile")
    assert first.load() is None
    assert second.load() is None
    token = first.save(session, expected=None)
    with pytest.raises(CheckpointConflict):
        second.save(new_session("profile"), expected=None)
    loaded, peer_token = second.load()
    changed = update_view(session, {"tab": "Compare"})
    committed = first.save(changed, expected=token)
    with pytest.raises(CheckpointConflict):
        second.save(update_view(loaded, {"tab": "Draft"}), expected=peer_token)
    assert second.load()[1] == committed
    first.close()
    second.close()


@pytest.mark.parametrize("phase", ["before", "after"])
def test_killed_publication_is_old_or_new_complete_checkpoint(tmp_path, phase):
    path = tmp_path / "lab.sqlite3"
    store = CheckpointStore(path, "profile")
    old = new_session("profile")
    store.save(old, expected=None)
    store.close()
    # Child inherits the repository test harness's isolated config AND data env.
    # The connection wrapper gates immediately BEFORE real COMMIT. For AFTER it
    # first calls real execute(COMMIT), then signals: no mocked successful commit.
    script = r"""
import os, sqlite3, sys
from Tests.DB.test_chunking_lab_db import completed_session
import tldw_chatbook.DB.Chunking_Lab_DB as module
real_connect = module.connect_private_sqlite
phase = sys.argv[2]
class Gate(sqlite3.Connection):
    armed = False
    def execute(self, sql, *args):
        if self.armed and sql.strip().upper() == "COMMIT":
            if phase == "before":
                os.write(1, b"READY\n")
                os.read(0, 1)
            result = super().execute(sql, *args)
            if phase == "after":
                os.write(1, b"READY\n")
                os.read(0, 1)
            return result
        return super().execute(sql, *args)
def connect(*args, **kwargs):
    return real_connect(*args, factory=Gate, **kwargs)
module.connect_private_sqlite = connect
store = module.CheckpointStore(sys.argv[1], "profile")
session, token = store.load()
new = completed_session().model_copy(update={"epoch": session.epoch, "batch": None})
store._connection().armed = True
store.save(new, expected=token)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(path), phase],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
    )
    try:
        assert select.select([child.stdout], [], [], 15)[0], (
            "Child never reached real COMMIT"
        )
        assert child.stdout.readline() == b"READY\n"
        child.kill()
        child.wait(timeout=5)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
    reopened = CheckpointStore(path, "profile")
    recovered, token = reopened.load()
    if phase == "before":
        assert recovered.model_dump() == old.model_dump()
        assert token.generation == 1
    else:
        assert len(recovered.results) == 1
        result = next(iter(recovered.results.values()))
        assert result["report"]["chunks"][0]["text"] == "exact sample"
        assert (
            recovered.samples[recovered.view["sample_hash"]]["text"] == "exact sample\n"
        )
        assert token.generation == 2
    reopened.close()


def test_exact_pending_controls_results_view_and_undo_do_not_need_source(tmp_path):
    session = completed_session()
    candidate_id = next(iter(session.candidates))
    session = pin_baseline(session)
    session = edit_control(session, candidate_id, "chunking.config.max_size", "12e")
    session = update_view(session, {"tab": "Compare", "selected_chunk": 0})
    session = replace_sample(session, "replacement", {"path": "gone"})
    store = CheckpointStore(tmp_path / "lab.sqlite3", "profile")
    token = store.save(session, expected=None)
    for i in range(3):
        session = update_view(session, {"selected_chunk": i})
        token = store.save(session, expected=token)
    store.close()
    store = CheckpointStore(tmp_path / "lab.sqlite3", "profile")
    restored, _ = store.load()
    assert restored.model_dump() == session.model_dump()
    undone = undo_edit(restored)
    assert undone.samples[undone.view["sample_hash"]]["text"] == "exact sample\n"
    assert restored.candidates[candidate_id]["draft"]["pending_controls"] == {
        "chunking.config.max_size": "12e"
    }
    store.close()


def test_clear_tombstone_removes_content_and_rejects_late_epoch(tmp_path):
    path = tmp_path / "lab.sqlite3"
    store = CheckpointStore(path, "profile")
    session = completed_session()
    token = store.save(session, expected=None)
    cleared = store.clear(expected=token)
    assert cleared.epoch != token.epoch
    assert cleared.generation > token.generation
    with pytest.raises(CheckpointConflict):
        store.save(session, expected=token)
    with pytest.raises(CheckpointConflict):
        store.save(session, expected=cleared)
    store.close()
    with sqlite3.connect(path) as raw:
        assert raw.execute("SELECT count(*) FROM lab_blobs").fetchone()[0] == 0
        assert raw.execute("SELECT count(*) FROM lab_checkpoints").fetchone()[0] == 0
    reopened = CheckpointStore(path, "profile")
    fresh, fresh_token = reopened.load()
    assert fresh_token == cleared
    assert fresh.epoch == cleared.epoch and not fresh.results and not fresh.undo
    assert fresh.samples[fresh.view["sample_hash"]]["text"] == ""
    reopened.close()


def test_newer_schema_and_wrong_profile_preserve_file(tmp_path):
    path = tmp_path / "lab.sqlite3"
    store = CheckpointStore(path, "profile")
    token = store.save(new_session("profile"), expected=None)
    store.close()
    wrong = CheckpointStore(path, "other")
    with pytest.raises(RecoverySchemaError):
        wrong.load()
    wrong.close()
    with sqlite3.connect(path) as raw:
        raw.execute("PRAGMA user_version=2")
    store = CheckpointStore(path, "profile")
    with pytest.raises(RecoverySchemaError):
        store.load()
    with pytest.raises(RecoverySchemaError):
        store.save(new_session("profile"), expected=token)
    store.close()
    with sqlite3.connect(path) as raw:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == 2
        assert raw.execute("SELECT generation FROM lab_state").fetchone()[0] == 1


def test_malformed_current_falls_back_with_warning_without_reset(tmp_path):
    path = tmp_path / "lab.sqlite3"
    store = CheckpointStore(path, "profile")
    old = new_session("profile")
    token = store.save(old, expected=None)
    token = store.save(update_view(old, {"tab": "Compare"}), expected=token)
    store.close()
    with sqlite3.connect(path) as raw:
        raw.execute(
            "UPDATE lab_checkpoints SET document='broken' WHERE id=(SELECT current_checkpoint FROM lab_state)"
        )
    reopened = CheckpointStore(path, "profile")
    restored, actual = reopened.load()
    assert restored.model_dump() == old.model_dump()
    assert actual.generation == token.generation and actual.revision == old.revision
    assert "previous" in reopened.recovery_warning.lower()
    reopened.close()


def test_recovery_interrupts_unfinished_batch_retaining_previous_output(tmp_path):
    session = completed_session()
    candidate_id = next(iter(session.candidates))
    requests = capture_batch(session, (candidate_id,))
    session = install_batch(session, requests)
    store = CheckpointStore(tmp_path / "lab.sqlite3", "profile")
    token = store.save(session, expected=None)
    store.close()
    reopened = CheckpointStore(tmp_path / "lab.sqlite3", "profile")
    restored, acknowledgment = reopened.load()
    assert acknowledgment == token
    assert restored.revision > token.revision
    assert restored.batch["outcomes"][requests[0].run_id] == "interrupted"
    assert restored.results[requests[0].run_id]["status"] == "interrupted"
    assert restored.candidates[candidate_id]["previous_run_id"] in restored.results
    reopened.close()


def test_cached_blobs_are_private_captures_and_small_edits_do_not_revalidate_reports(
    tmp_path, monkeypatch
):
    session = completed_session()
    path = tmp_path / "lab.sqlite3"
    store = CheckpointStore(path, "profile")
    token = store.save(session, expected=None)
    result = next(iter(session.results.values()))
    result["report"]["chunks"][0]["text"] = "valid-looking mutation"
    session.samples[session.view["sample_hash"]]["source"]["path"] = "mutated"
    statements = []
    store._connection().set_trace_callback(statements.append)

    def forbidden_validation(*args, **kwargs):
        raise AssertionError("A small edit revalidated an immutable report")

    with monkeypatch.context() as scoped:
        scoped.setattr(RunResult, "model_validate", forbidden_validation)
        token = store.save(update_view(session, {"tab": "Compare"}), expected=token)
    assert not any("INSERT OR IGNORE INTO lab_blobs" in sql for sql in statements)
    store.close()
    store = CheckpointStore(path, "profile")
    restored, _ = store.load()
    assert (
        next(iter(restored.results.values()))["report"]["chunks"][0]["text"]
        == "exact sample"
    )
    assert (
        restored.samples[restored.view["sample_hash"]]["source"]["path"]
        == "deleted.txt"
    )
    store.close()


def test_failed_graph_validation_does_not_poison_next_blob_publication(tmp_path):
    store = CheckpointStore(tmp_path / "lab.sqlite3", "profile")
    session = completed_session()
    invalid = session.model_copy(update={"view": {"sample_hash": "missing"}})
    with pytest.raises(ValueError):
        store.save(invalid, expected=None)
    store.save(session, expected=None)
    store.close()
    reopened = CheckpointStore(tmp_path / "lab.sqlite3", "profile")
    assert reopened.load()[0].model_dump() == session.model_dump()
    reopened.close()


def test_disk_full_rollback_and_retry_preserve_atomic_content(tmp_path):
    store = CheckpointStore(tmp_path / "lab.sqlite3", "profile")
    session = new_session("profile")
    token = store.save(session, expected=None)
    conn = store._connection()
    pages = conn.execute("PRAGMA page_count").fetchone()[0]
    conn.execute(f"PRAGMA max_page_count={pages}")
    latest = replace_sample(session, "large new sample " * 10000, {})
    with pytest.raises(sqlite3.OperationalError, match="full"):
        store.save(latest, expected=token)
    assert store.load()[0].model_dump() == session.model_dump()
    conn.execute("PRAGMA max_page_count=1000000")
    store.save(latest, expected=token)
    assert store.load()[0].model_dump() == latest.model_dump()
    store.close()


def test_save_after_fallback_retains_previous_valid_checkpoint(tmp_path):
    path = tmp_path / "lab.sqlite3"
    store = CheckpointStore(path, "profile")
    old = new_session("profile")
    token = store.save(old, expected=None)
    store.save(update_view(old, {"tab": "Compare"}), expected=token)
    store.close()
    with sqlite3.connect(path) as raw:
        raw.execute(
            "UPDATE lab_checkpoints SET document='broken' WHERE id=(SELECT current_checkpoint FROM lab_state)"
        )
    store = CheckpointStore(path, "profile")
    recovered, token = store.load()
    store.save(update_view(recovered, {"tab": "Draft"}), expected=token)
    store.close()
    with sqlite3.connect(path) as raw:
        raw.execute(
            "UPDATE lab_checkpoints SET document='also broken' WHERE id=(SELECT current_checkpoint FROM lab_state)"
        )
    store = CheckpointStore(path, "profile")
    assert store.load()[0].model_dump() == old.model_dump()
    store.close()


def test_gc_retains_restore_undo_across_view_checkpoints_then_clear_removes_it(
    tmp_path,
):
    path = tmp_path / "lab.sqlite3"
    store = CheckpointStore(path, "profile")
    original = completed_session()
    token = store.save(original, expected=None)
    conn = store._connection()
    fresh, token = store.replace(new_session("profile"), original, expected=token)
    for index in range(4):
        fresh = update_view(fresh, {"selected_chunk": index})
        token = store.save(fresh, expected=token)
    assert conn.execute("SELECT count(*) FROM lab_checkpoints").fetchone()[0] == 3
    assert (
        conn.execute("SELECT count(*) FROM lab_blobs WHERE kind='result'").fetchone()[0]
        == 1
    )
    store.clear(expected=token)
    assert conn.execute("SELECT count(*) FROM lab_blobs").fetchone()[0] == 0
    store.close()


def test_gc_drops_unreferenced_payload_only_after_rolling_previous_expires(tmp_path):
    store = CheckpointStore(tmp_path / "lab.sqlite3", "profile")
    original = completed_session()
    token = store.save(original, expected=None)
    fresh = new_session("profile").model_copy(
        update={"epoch": token.epoch, "revision": original.revision + 1}
    )
    token = store.save(fresh, expected=token)
    conn = store._connection()
    assert (
        conn.execute("SELECT count(*) FROM lab_blobs WHERE kind='result'").fetchone()[0]
        == 1
    )
    store.save(update_view(fresh, {"tab": "Compare"}), expected=token)
    assert (
        conn.execute("SELECT count(*) FROM lab_blobs WHERE kind='result'").fetchone()[0]
        == 0
    )
    assert conn.execute("SELECT count(*) FROM lab_checkpoints").fetchone()[0] == 2
    store.close()


@pytest.mark.parametrize("damage", ["future", "bad-json", "blob-hash", "wrong-shape"])
def test_unrecoverable_checkpoint_damage_never_resets_store(tmp_path, damage):
    path = tmp_path / "lab.sqlite3"
    store = CheckpointStore(path, "profile")
    store.save(new_session("profile"), expected=None)
    store.close()
    with sqlite3.connect(path) as raw:
        if damage == "blob-hash":
            raw.execute("UPDATE lab_blobs SET payload='{}'")
        else:
            document = {
                "future": '{"schema_version":2}',
                "bad-json": "invalid",
                "wrong-shape": "[]",
            }[damage]
            raw.execute("UPDATE lab_checkpoints SET document=?", (document,))
    store = CheckpointStore(path, "profile")
    with pytest.raises(RecoverySchemaError):
        store.load()
    with pytest.raises(CheckpointConflict):
        store.save(new_session("profile"), expected=None)
    assert (
        store._connection().execute("SELECT generation FROM lab_state").fetchone()[0]
        == 1
    )
    store.close()


def test_owner_secures_database_wal_and_shm_and_isolates_profiles(tmp_path):
    paths = [tmp_path / "one.sqlite3", tmp_path / "two.sqlite3"]
    stores = [
        CheckpointStore(path, profile)
        for path, profile in zip(paths, ("one", "two"), strict=True)
    ]
    try:
        for store, profile in zip(stores, ("one", "two"), strict=True):
            store.save(new_session(profile), expected=None)
            assert store.load()[0].profile_key == profile
            conn = store._connection()
            assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2
            assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        if os.name == "posix":
            for path in paths:
                for suffix in ("", "-wal", "-shm"):
                    assert (
                        stat.S_IMODE(type(path)(str(path) + suffix).stat().st_mode)
                        == 0o600
                    )
    finally:
        for store in stores:
            store.close()


def test_simultaneous_initial_publications_choose_one_owner(tmp_path):
    path = tmp_path / "lab.sqlite3"
    # The shared private-path helper intentionally fails closed if two opens
    # race exclusive first-file creation. This test isolates SQLite publication
    # contention after that existing security boundary has acquired the file.
    empty = CheckpointStore(path, "profile")
    assert empty.load() is None
    empty.close()
    ready = Barrier(2)

    def publish():
        session = new_session("profile")
        store = CheckpointStore(path, "profile")
        ready.wait(timeout=5)
        try:
            return store.save(session, expected=None)
        except CheckpointConflict:
            return None
        finally:
            store.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(publish) for _ in range(2)]
        tokens = [future.result(timeout=10) for future in futures]
    assert sum(token is not None for token in tokens) == 1


def test_store_load_invalidates_cache_assumptions_after_another_instance_clears(
    tmp_path,
):
    path = tmp_path / "lab.sqlite3"
    first, second = CheckpointStore(path, "profile"), CheckpointStore(path, "profile")
    original = completed_session()
    token = first.save(original, expected=None)
    second.clear(expected=token)
    fresh, token = first.load()
    # Explicitly reuse old immutable data under freshly acquired authority.
    # Its old on-disk blobs disappeared with the other writer's Clear.
    retained = original.model_copy(update={"epoch": fresh.epoch, "batch": None})
    first.save(retained, expected=token)
    first.close()
    restored, _ = second.load()
    assert len(restored.results) == 1
    second.close()


@pytest.mark.parametrize("damage", ["missing-state", "false-tombstone"])
def test_malformed_state_is_never_mistaken_for_a_fresh_session(tmp_path, damage):
    path = tmp_path / "lab.sqlite3"
    store = CheckpointStore(path, "profile")
    session = new_session("profile")
    token = store.save(session, expected=None)
    store.save(update_view(session, {"tab": "Compare"}), expected=token)
    store.close()
    with sqlite3.connect(path) as raw:
        if damage == "missing-state":
            raw.execute("DELETE FROM lab_state")
        else:
            raw.execute("UPDATE lab_state SET current_checkpoint=NULL")
    store = CheckpointStore(path, "profile")
    with pytest.raises(RecoverySchemaError):
        store.load()
    store.close()
