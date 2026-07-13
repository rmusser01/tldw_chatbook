"""AgentRunsDB against a real on-disk SQLite file."""
import pytest

from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


@pytest.fixture()
def db(tmp_path):
    return AgentRunsDB(tmp_path / "agent_runs.db", client_id="test")


def test_create_and_get_run(db):
    run_id = db.create_run(conversation_id="conv1", agent_kind="primary",
                           budget={"max_steps": 8})
    run = db.get_run(run_id)
    assert run["conversation_id"] == "conv1"
    assert run["agent_kind"] == "primary"
    assert run["status"] == "running"
    assert run["steps"] == [] and run["parent_run_id"] is None
    assert run["budget"] == {"max_steps": 8}
    assert run["created_at"] and run["updated_at"]


def test_get_missing_run_returns_none(db):
    assert db.get_run("nope") is None


def test_append_steps_accumulates_and_parses(db):
    run_id = db.create_run(conversation_id="c", agent_kind="primary")
    db.append_steps(run_id, [{"index": 0, "kind": "model", "summary": "hi"}])
    db.append_steps(run_id, [{"index": 1, "kind": "tool_call",
                              "tool_name": "calculator"}])
    steps = db.get_run(run_id)["steps"]
    assert [s["index"] for s in steps] == [0, 1]
    assert steps[1]["tool_name"] == "calculator"


def test_set_status_and_result(db):
    run_id = db.create_run(conversation_id="c", agent_kind="subagent",
                           task="do x", parent_run_id="p1")
    db.set_status(run_id, "done", result="the answer")
    run = db.get_run(run_id)
    assert run["status"] == "done" and run["result"] == "the answer"
    assert run["task"] == "do x" and run["parent_run_id"] == "p1"


def test_count_subagents_counts_only_subagent_kind(db):
    db.create_run(conversation_id="c", agent_kind="primary")
    parent = db.create_run(conversation_id="c", agent_kind="primary")
    for i in range(3):
        db.create_run(conversation_id="c", agent_kind="subagent",
                      task=f"t{i}", parent_run_id=parent)
    db.create_run(conversation_id="other", agent_kind="subagent", task="x",
                  parent_run_id="zzz")
    assert db.count_subagent_runs("c") == 3


def test_supersede_run_tree_marks_run_and_children(db):
    parent = db.create_run(conversation_id="c", agent_kind="primary")
    child = db.create_run(conversation_id="c", agent_kind="subagent",
                          task="t", parent_run_id=parent)
    other = db.create_run(conversation_id="c", agent_kind="primary")
    changed = db.supersede_run_tree(parent)
    assert changed == 2
    assert db.get_run(parent)["status"] == "superseded"
    assert db.get_run(child)["status"] == "superseded"
    assert db.get_run(other)["status"] == "running"


def test_list_runs_filters_superseded_when_asked(db):
    a = db.create_run(conversation_id="c", agent_kind="primary")
    db.create_run(conversation_id="c", agent_kind="primary")
    db.supersede_run_tree(a)
    assert len(db.list_runs("c")) == 2
    live = db.list_runs("c", include_superseded=False)
    assert len(live) == 1 and live[0]["status"] == "running"


def test_sql_is_parameterized_against_quotes(db):
    run_id = db.create_run(conversation_id="c''; DROP TABLE agent_runs;--",
                           agent_kind="primary", task="a 'quoted' task")
    assert db.get_run(run_id)["task"] == "a 'quoted' task"
