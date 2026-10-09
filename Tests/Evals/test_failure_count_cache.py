"""B6: run-group failure counts are cached between eval_results mutations.

The evals rail re-composes on every selection change and
``run_group_cell_failure_counts`` answered with a fresh ``json_extract``
scan over all ``eval_results`` rows each time. The counts now memoize on
the DB instance and are invalidated by ``store_result`` -- the only
production writer of ``eval_results`` (verified by grepping
``INSERT INTO``/``DELETE FROM``/``UPDATE`` ``eval_results`` across
``tldw_chatbook/``).
"""

import threading
from contextlib import contextmanager

import pytest

import tldw_chatbook.DB.Evals_DB as evals_module
from tldw_chatbook.DB.Evals_DB import EvalsDB


@pytest.fixture
def db():
    db = EvalsDB(db_path=":memory:", client_id="b6_test")
    yield db
    db.close()


def _seed_group(db: EvalsDB, name: str) -> str:
    """Create one run carrying a run_group_id; return (run_group_id, run_id)."""
    task_id = db.create_task(
        name=f"task-{name}",
        description="B6 fixture",
        task_type="question_answer",
        config_format="custom",
        config_data={"name": f"task-{name}"},
    )
    model_id = db.create_model(name=f"model-{name}", provider="mock", model_id="m")
    run_id = db.create_run(name=f"run-{name}", task_id=task_id, model_id=model_id)
    group_id = f"group-{name}"
    db.update_run(run_id, {"run_group_id": group_id})
    return group_id, run_id


def _count_result_scans(db: EvalsDB) -> int:
    """Run the failure-count query once, counting SELECTs touching eval_results."""
    counts = {"n": 0}

    def trace(statement: str) -> None:
        stripped = statement.lstrip().upper()
        if stripped.startswith("SELECT") and "FROM EVAL_RESULTS" in stripped:
            counts["n"] += 1

    conn = db.get_connection()
    conn.set_trace_callback(trace)
    try:
        db.run_group_cell_failure_counts()
    finally:
        conn.set_trace_callback(None)
    return counts["n"]


def _store_cell(db: EvalsDB, run_id: str, *, errored: bool) -> str:
    logprobs = (
        {"schema": "cell-error", "error": {"message": "boom"}}
        if errored
        else {"schema": "cell-capture", "text": "ok"}
    )
    return db.store_result(
        run_id=run_id,
        sample_id=f"s-{errored}-{id(logprobs)}",
        input_data={"input": "prompt"},
        actual_output="out",
        expected_output="out",
        logprobs=logprobs,
        metrics={},
        metadata={},
    )


def test_second_call_serves_from_cache(db):
    group_id, run_id = _seed_group(db, "a")
    _store_cell(db, run_id, errored=True)

    first = _count_result_scans(db)
    second = _count_result_scans(db)

    assert first >= 1, "first call must scan eval_results at least once"
    assert second == 0, (
        f"second call re-scanned eval_results ({second} SELECTs); expected cached memo"
    )
    assert db.run_group_cell_failure_counts() == {group_id: (1, 1)}


def test_store_result_invalidates_and_values_refresh(db):
    group_id, run_id = _seed_group(db, "b")
    _store_cell(db, run_id, errored=True)
    assert db.run_group_cell_failure_counts() == {group_id: (1, 1)}

    _store_cell(db, run_id, errored=False)

    scans = _count_result_scans(db)
    assert scans >= 1, "cache must be invalidated by store_result (a re-scan is required)"
    assert db.run_group_cell_failure_counts() == {group_id: (2, 1)}


def test_invalidate_public_api_forces_rescan(db):
    group_id, run_id = _seed_group(db, "c")
    _store_cell(db, run_id, errored=False)
    db.run_group_cell_failure_counts()  # populate the memo

    db.invalidate_run_group_failure_cache()

    scans = _count_result_scans(db)
    assert scans >= 1, "invalidate_run_group_failure_cache must force the next call to re-scan"
    assert db.run_group_cell_failure_counts() == {group_id: (1, 0)}


def test_result_commit_invalidates_a_rail_read_during_the_write(tmp_path, monkeypatch):
    db = EvalsDB(tmp_path / "overlap.sqlite3", client_id="b6_overlap")
    group_id, run_id = _seed_group(db, "overlap")
    _store_cell(db, run_id, errored=True)
    assert db.run_group_cell_failure_counts() == {group_id: (1, 1)}
    started = threading.Event()
    finish = threading.Event()
    errors = []
    original_counter = evals_module.log_counter

    def hold_writer(name, *args, **kwargs):
        if (
            name == "eval_db_operation_success"
            and kwargs.get("labels", {}).get("operation") == "store_result"
        ):
            # The result and sample-counter writes are still uncommitted.
            started.set()
            assert finish.wait(5)
        return original_counter(name, *args, **kwargs)

    monkeypatch.setattr(evals_module, "log_counter", hold_writer)

    def write_result():
        try:
            _store_cell(db, run_id, errored=False)
        except Exception as error:  # noqa: BLE001 - report worker failures to the asserting thread
            errors.append(error)
        finally:
            db.close()

    writer = threading.Thread(target=write_result)
    try:
        writer.start()
        assert started.wait(5)
        assert db.run_group_cell_failure_counts() == {group_id: (1, 1)}
        finish.set()
        writer.join(5)
        assert not writer.is_alive() and not errors
        assert (
            db.get_connection()
            .execute("SELECT COUNT(*) FROM eval_results")
            .fetchone()[0]
            == 2
        )
        assert db.run_group_cell_failure_counts() == {group_id: (2, 1)}
    finally:
        finish.set()
        writer.join(5)
        db.close()


def test_reader_cannot_publish_a_snapshot_from_before_a_result_commit(
    tmp_path, monkeypatch
):
    db = EvalsDB(tmp_path / "publication.sqlite3", client_id="b6_publication")
    group_id, run_id = _seed_group(db, "publication")
    _store_cell(db, run_id, errored=True)
    read_snapshot = threading.Event()
    publish = threading.Event()
    errors = []
    observed = []
    real_connection = db.connection

    class HeldCursor:
        def __init__(self, cursor):
            self.cursor = cursor

        def fetchall(self):
            rows = self.cursor.fetchall()
            read_snapshot.set()
            assert publish.wait(5)
            return rows

    class HeldConnection:
        def __init__(self, conn):
            self.conn = conn

        def execute(self, sql, *args):
            cursor = self.conn.execute(sql, *args)
            if "FROM eval_results" in sql:
                return HeldCursor(cursor)
            return cursor

    @contextmanager
    def hold_reader():
        with real_connection() as conn:
            yield HeldConnection(conn) if threading.current_thread() is reader else conn

    def read_counts():
        try:
            observed.append(db.run_group_cell_failure_counts())
        except Exception as error:  # noqa: BLE001 - report worker failures to the asserting thread
            errors.append(error)
        finally:
            db.close()

    reader = threading.Thread(target=read_counts)
    monkeypatch.setattr(db, "connection", hold_reader)
    try:
        reader.start()
        assert read_snapshot.wait(5)
        _store_cell(db, run_id, errored=False)
        publish.set()
        reader.join(5)
        assert not reader.is_alive() and not errors
        assert observed == [{group_id: (1, 1)}]
        assert db.run_group_cell_failure_counts() == {group_id: (2, 1)}
    finally:
        publish.set()
        reader.join(5)
        db.close()


def test_run_group_change_invalidates_its_result_counts(db):
    group_id, run_id = _seed_group(db, "move")
    _store_cell(db, run_id, errored=True)
    assert db.run_group_cell_failure_counts() == {group_id: (1, 1)}
    db.update_run(run_id, {"run_group_id": "new-group"})
    assert db.run_group_cell_failure_counts() == {"new-group": (1, 1)}


def test_cached_counts_keep_the_repository_admission_fence(tmp_path):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant

    db = EvalsDB(tmp_path / "admission.sqlite3", client_id="b6_admission")
    group_id, run_id = _seed_group(db, "admission")
    _store_cell(db, run_id, errored=True)
    assert db.run_group_cell_failure_counts() == {group_id: (1, 1)}
    participant = _repository_participant(db)
    try:
        participant.close_admission()
        with pytest.raises(RecoveryRequired, match="storage_locally_paused"):
            db.run_group_cell_failure_counts()
    finally:
        db.close()


def test_result_and_completed_samples_rollback_together(db, monkeypatch):
    group_id, run_id = _seed_group(db, "rollback")
    _store_cell(db, run_id, errored=True)
    assert db.run_group_cell_failure_counts() == {group_id: (1, 1)}
    original_counter = evals_module.log_counter

    def fail_before_commit(name, *args, **kwargs):
        if (
            name == "eval_db_operation_success"
            and kwargs.get("labels", {}).get("operation") == "store_result"
        ):
            raise RuntimeError("before_commit")
        return original_counter(name, *args, **kwargs)

    monkeypatch.setattr(evals_module, "log_counter", fail_before_commit)
    with pytest.raises(RuntimeError, match="before_commit"):
        _store_cell(db, run_id, errored=False)
    assert len(db.get_run_results(run_id)) == 1
    assert db.get_run(run_id)["completed_samples"] == 1
    assert db.run_group_cell_failure_counts() == {group_id: (1, 1)}
