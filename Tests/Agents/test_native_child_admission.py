"""Native child admission and cleanup do not depend on workflow introspection."""

import functools
import threading
from dataclasses import replace

import pytest

from Tests.Agents.conftest import join_fleet_children, pin_turn_scoped_children
from Tests.Agents.test_agent_service import fence
from Tests.Agents.test_fleet_runtime import FLEET_CFG, make_fleet_service
from tldw_chatbook.Agents import agent_service
from tldw_chatbook.Agents.agent_models import RUN_DONE, RUN_ERROR, SPAWN_TOOL_NAME
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def db(tmp_path):
    database = AgentRunsDB(tmp_path / "runs.db")
    yield database
    database.close()


def test_ordinary_native_launch_accepts_a_transparent_wrapper(db, monkeypatch):
    pin_turn_scoped_children(monkeypatch)
    service, _, fleet = make_fleet_service(
        db,
        [fence(SPAWN_TOOL_NAME, {"task": "work"}), "done"],
        {"work": ["child result"]},
        allow_unconsumed=True,
    )
    launch = service._launch_prepared_child_result

    @functools.wraps(launch)
    def wrapped(*args, **kwargs):
        return launch(*args, **kwargs)

    monkeypatch.setattr(service, "_launch_prepared_child_result", wrapped)
    _, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    join_fleet_children(service)
    assert outcome.status == RUN_DONE
    assert [(item.status, item.result) for item in fleet.snapshot()] == [
        (RUN_DONE, "child result")
    ]


@pytest.mark.parametrize("failure", ["factory", "entry"])
def test_child_guard_setup_failure_is_terminal_and_releases_native_capacity(
    db, monkeypatch, failure
):
    pin_turn_scoped_children(monkeypatch)
    original_guard = agent_service.worker_guard
    child_attempted = threading.Event()

    def failing_guard(service):
        decorate = original_guard(service)

        def wrapper(function):
            if failure == "factory" and function.__name__ == "run_child":
                child_attempted.set()
                raise RuntimeError("child guard factory failed")
            guarded = decorate(function)

            @functools.wraps(function)
            def invoke(*args, **kwargs):
                if threading.current_thread().name.startswith("fleet-"):
                    child_attempted.set()
                    raise RuntimeError("child guard setup failed")
                return guarded(*args, **kwargs)

            return invoke

        return wrapper

    monkeypatch.setattr(agent_service, "worker_guard", failing_guard)
    service, _, fleet = make_fleet_service(
        db,
        [fence(SPAWN_TOOL_NAME, {"task": "work"}), "done"],
        {"work": [lambda: pytest.fail("refused child reached provider")]},
        allow_unconsumed=True,
    )
    config = replace(FLEET_CFG, budget=replace(FLEET_CFG.budget, max_wall_seconds=0.5))
    _, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=config,
        api_endpoint="llama_cpp",
    )
    join_fleet_children(service)
    assert child_attempted.is_set()
    assert outcome.status == RUN_DONE
    children = [row for row in db.list_runs("c") if row["agent_kind"] == "subagent"]
    assert len(children) == 1
    assert children[0]["status"] == RUN_ERROR
    assert fleet.snapshot()[0].status == RUN_ERROR
    assert service.runtime_capacity.snapshot().executions == ()


def test_start_error_after_worker_entry_keeps_capacity_and_child_budget(
    db, monkeypatch
):
    pin_turn_scoped_children(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    real_start = threading.Thread.start
    failed = False

    def child():
        entered.set()
        assert release.wait(5)
        return "child result"

    def fail_after_start(thread):
        nonlocal failed
        real_start(thread)
        if thread.name.startswith("fleet-") and not failed:
            failed = True
            assert entered.wait(3)
            raise RuntimeError("start returned an error after worker entry")

    def finish_parent():
        assert service.runtime_capacity.snapshot().child_executions == 1
        release.set()
        return "done"

    service, _, fleet = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "work"}),
            fence(SPAWN_TOOL_NAME, {"task": "second"}),
            finish_parent,
        ],
        {"work": [child], "second": [lambda: pytest.fail("child budget refilled")]},
        allow_unconsumed=True,
    )
    monkeypatch.setattr(threading.Thread, "start", fail_after_start)
    try:
        _, outcome = service.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=replace(
                FLEET_CFG,
                budget=replace(FLEET_CFG.budget, max_subagents=1, max_wall_seconds=10),
            ),
            api_endpoint="llama_cpp",
        )
        join_fleet_children(service)
        assert outcome.status == RUN_DONE
        assert db.count_subagent_runs("c") == 1
        assert fleet.all_finished()
        assert service.runtime_capacity.snapshot().executions == ()
    finally:
        release.set()
