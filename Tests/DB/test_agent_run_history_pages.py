"""Bounded, metadata-only history pages across a changing conversation."""

import pytest

from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


@pytest.fixture
def db(tmp_path):
    database = AgentRunsDB(tmp_path / "runs.db", client_id="history")
    try:
        yield database
    finally:
        database.close()


def _seed(
    db,
    run_id,
    *,
    conversation="c",
    kind="subagent",
    stamp="2026-09-07T10:00:00",
    status="done",
):
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO agent_runs (id,conversation_id,agent_kind,task,status,steps,result,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                conversation,
                kind,
                "check logs",
                status,
                "deliberately not JSON",
                "large body",
                stamp,
                stamp,
            ),
        )


def test_history_pages_are_narrow_scoped_and_stable_when_new_runs_arrive(db):
    for name in ("a", "b", "c", "d", "e"):
        _seed(db, name, status="superseded" if name == "b" else "done")
    _seed(db, "foreign", conversation="other")
    _seed(db, "primary", kind="primary")
    first = db.list_subagent_run_headers("c", limit=2)
    assert [r["id"] for r in first] == ["e", "d"]
    assert not {"steps", "result", "budget"} & first[0].keys()
    _seed(db, "new", stamp="2026-09-07T11:00:00")
    cursor = (first[-1]["created_at"], first[-1]["id"])
    second = db.list_subagent_run_headers("c", before=cursor, limit=2)
    assert [r["id"] for r in second] == ["c", "b"]
    assert second[-1]["status"] == "superseded"
    third = db.list_subagent_run_headers(
        "c", before=(second[-1]["created_at"], "b"), limit=2
    )
    assert [r["id"] for r in third] == ["a"]
    assert db.list_subagent_run_headers("missing") == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"limit": 0},
        {"limit": -1},
        {"limit": True},
        {"limit": 102},
        {"limit": "10"},
        {"before": ("only-one",)},
        {"before": (2, "id")},
        {"before": "bad"},
    ],
)
def test_invalid_page_arguments_are_rejected(db, kwargs):
    with pytest.raises(ValueError):
        db.list_subagent_run_headers("c", **kwargs)


def test_history_metadata_bounds_large_task_text(db):
    _seed(db, "child")
    with db.transaction() as conn:
        conn.execute("UPDATE agent_runs SET task=? WHERE id='child'", ("x" * 100_000,))
    assert len(db.list_subagent_run_headers("c")[0]["task"]) == 200
