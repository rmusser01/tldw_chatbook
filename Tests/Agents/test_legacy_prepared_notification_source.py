"""Legacy hooks fired under an agent admission keep their source (TASK-41943).

The Console prepares a legacy Stop notification under ``worker_guard(bridge)``
and a SubagentStop one inside the agent service's admission, and delivers
both later through ``RunHooksEngine.notify(_prepared=...)``, whose check runs
on the notify worker with no admission. Agent-run PreToolUse hooks fire under
the service's admission too. Every one of them must see the same legacy hook
sources at capture and at check, so only a real source change refuses them.
"""

import threading

import pytest

from Tests.Agents.test_hook_permissions import hook_file as _hook_file
from Tests.Agents.test_native_legacy_hooks import native_hooks, wait_until
from tldw_chatbook.Chat.console_workflow_runs import native_workflow_qualified

hook_file, native_hooks = _hook_file, native_hooks

pytestmark = [
    pytest.mark.bootstrap_profile,
    pytest.mark.skipif(
        not native_workflow_qualified(), reason="native POSIX group custody required"
    ),
]


@pytest.fixture
def bridge(native_hooks):
    """The Console's agent bridge over a file-backed AgentRuns database."""
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    db = AgentRunsDB(native_hooks.tmp_path / "agent_runs.sqlite", "test")
    try:
        yield ConsoleAgentBridge(
            agent_runs_db=db, store=native_hooks.store, provider_gateway=None
        )
    finally:
        db.close()


def _prepare_stop(engine, bridge, session_id):
    """The controller's ``hold_legacy_stop``: prepare under the bridge's guard."""
    from tldw_chatbook.Agents.activation import worker_guard

    return worker_guard(bridge)(
        lambda: engine.prepare_notification("Stop", session_id)
    )()


def _prepare_subagent_stop(engine, bridge, session_id):
    """The spawn tool's prepare, inside the agent service's own admission."""
    from tldw_chatbook.Agents.activation import execution

    with execution(bridge):
        return engine.prepare_notification("SubagentStop", session_id)


def _on_thread(function, *args):
    box = []
    thread = threading.Thread(target=lambda: box.append(function(*args)))
    thread.start()
    thread.join(10)
    assert not thread.is_alive()
    return box[0]


def _marker_hook(h, event):
    marker = h.tmp_path / "ran"
    h.configure(
        f"import pathlib; pathlib.Path({str(marker)!r}).write_text('ran')",
        event=event,
    )
    return marker


@pytest.mark.parametrize(
    "event,prepare,data",
    [
        ("Stop", _prepare_stop, {"status": "completed"}),
        (
            "SubagentStop",
            _prepare_subagent_stop,
            {"child_run_id": "child-1", "status": "completed"},
        ),
    ],
)
def test_prepared_legacy_notification_runs_its_hook(
    native_hooks, bridge, event, prepare, data
):
    h = native_hooks
    marker = _marker_hook(h, event)
    custody = _on_thread(prepare, h.engine, bridge, h.session.id)
    assert custody is not None, "the notification was refused when prepared"
    h.engine.notify(event, _prepared=custody, session_id=h.session.id, data=data)
    wait_until(marker.exists, seconds=30)


def test_prepared_notification_refuses_a_source_changed_after_capture(
    native_hooks, bridge, monkeypatch
):
    """Another workspace database installed before delivery refuses the hook."""
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Tools import workspace_file_roots
    from tldw_chatbook.Workspaces.registry_service import (
        LocalWorkspaceRegistryService,
    )

    h = native_hooks
    marker = _marker_hook(h, "Stop")
    custody = _on_thread(_prepare_stop, h.engine, bridge, h.session.id)
    assert custody is not None, "the notification was refused when prepared"
    other = WorkspaceDB(h.tmp_path / "other-workspaces.sqlite", client_id="test")
    monkeypatch.setattr(
        workspace_file_roots,
        "_default_registry_instance",
        LocalWorkspaceRegistryService(other),
    )
    done = []
    original = h.engine._notification_done

    def recorded(future):
        done.append(future.exception())
        original(future)

    monkeypatch.setattr(h.engine, "_notification_done", recorded)
    try:
        h.engine.notify(
            "Stop",
            _prepared=custody,
            session_id=h.session.id,
            data={"status": "completed"},
        )
        wait_until(lambda: done, seconds=30)
        assert isinstance(done[0], RuntimeError)
        assert str(done[0]) == "native hook source changed"
        wait_until(
            lambda: not h.runtime.execution_capacity.snapshot().executions, seconds=30
        )
        assert not marker.exists()
    finally:
        other.close()


def test_pre_tool_use_under_an_agent_admission_runs_its_hook(native_hooks, bridge):
    """An agent run's PreToolUse guard fires under the service's admission."""
    from tldw_chatbook.Agents.activation import execution

    h = native_hooks
    marker = _marker_hook(h, "PreToolUse")

    def fire():
        with execution(bridge):
            return h.engine.fire(
                "PreToolUse", session_id=h.session.id, data={"tool_name": "read"}
            )

    outcome = _on_thread(fire)
    assert not outcome.blocked, outcome.reason
    assert marker.exists()
