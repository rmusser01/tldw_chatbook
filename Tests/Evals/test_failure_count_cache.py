"""B6: run-group failure counts are cached between eval_results mutations.

The evals rail re-composes on every selection change and
``run_group_cell_failure_counts`` answered with a fresh ``json_extract``
scan over all ``eval_results`` rows each time. The counts now memoize on
the DB instance and are invalidated by ``store_result`` -- the only
production writer of ``eval_results`` (verified by grepping
``INSERT INTO``/``DELETE FROM``/``UPDATE`` ``eval_results`` across
``tldw_chatbook/``).
"""

import pytest

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
