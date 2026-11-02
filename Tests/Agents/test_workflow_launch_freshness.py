"""Final native freshness reads retain complete host-only activity custody."""

import asyncio
import hashlib
import json
import shlex
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests.Chat import test_console_native_preparation as native_tests
from Tests.Chat.test_console_native_preparation import (
    native_host as native_host,  # noqa: PLC0414
)
from tldw_chatbook.Agents import workflow_run_service as evidence
from tldw_chatbook.Agents.execution_capacity import current_execution_owner
from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint, WorktreeScope

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


@pytest.fixture(autouse=True)
def native_input(monkeypatch):
    original = native_tests.plan_data

    def with_input(root):
        data = original(root)
        data["phases"][0]["tasks"][0]["inputs"] = ["source.py"]
        return data

    monkeypatch.setattr(native_tests, "plan_data", with_input)


@pytest.mark.parametrize("drift", [False, True])
async def test_final_collector_is_owned_before_rows_and_drains_on_refusal(
    native_host, monkeypatch, drift
):
    h = native_host
    prepared = await h.bridge._prepare_workflow_reply(
        phase=h.phase, task_id="task-1", binding=h.binding, resolution=h.selected
    )
    auth = prepared.authorization
    entered, release = threading.Event(), threading.Event()
    observed = {}
    original = evidence._manifest
    contender = WorktreeFootprint(
        writes=(
            WorktreeScope(h.root / "source.py", False, capture_directory_chain(h.root)),
        )
    )

    def held(root, paths, **kwargs):
        if kwargs.get("authorization") is auth:
            observed["owner"] = current_execution_owner()
            observed["claims"] = tuple(auth._native_preparation[0].activity_claims)
            entered.set()
            assert release.wait(5)
            try:
                return original(root, paths, **kwargs)
            finally:
                observed["returned"] = True
        return original(root, paths, **kwargs)

    monkeypatch.setattr(evidence, "_manifest", held)
    if drift:
        (h.root / "source.py").write_text("changed after preparation\n")
    launch = asyncio.create_task(native_tests._launch_native_for_test(h, prepared))
    receipt = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert h.db.list_runs(h.conversation_id) == []
        assert h.calls == []
        with h.db.connection() as conn:
            assert (
                conn.execute(
                    "SELECT state FROM automatic_work_reservations WHERE id=?",
                    (auth.claim.reservation_id,),
                ).fetchone()[0]
                == "reserved"
            )
        with h.runtime.worktree_activity.prelaunch(footprint=contender) as claim:
            assert claim is None, "final collector read has no physical claim"
        assert observed["owner"] is not None
        assert observed["owner"].run_id is None
        assert observed["owner"].child
        observed["owner"].on_drained(
            lambda drained: observed.update(
                drained_after_read=observed.get("returned", False)
            )
        )
        h.runtime.worktree_activity.validate_claim(
            observed["claims"][0],
            owner=observed["owner"],
            footprint=auth.physical_footprint,
        )
    finally:
        release.set()
        if drift:
            with pytest.raises(ValueError, match="required_inputs_changed"):
                await launch
            await prepared.close()
        else:
            receipt = await launch
            await native_tests._settle_native_for_test(h, prepared, receipt)
    assert h.runtime.execution_capacity.snapshot().executions == ()
    assert observed["drained_after_read"] is True
    with h.runtime.worktree_activity.prelaunch(footprint=contender) as claim:
        assert claim is not None
    if drift:
        assert h.db.list_runs(h.conversation_id) == []
        assert auth._native_launch == []
        assert auth._native_preparation[0].activity_claims == []
        with h.db.connection() as conn:
            assert (
                conn.execute(
                    "SELECT state FROM automatic_work_reservations WHERE id=?",
                    (auth.claim.reservation_id,),
                ).fetchone()[0]
                == "reserved"
            )


def _dependency_plan(data, root):
    wrapper = root / "verify.sh"
    wrapper.write_text("#!/bin/bash\nprintf verified\n")
    wrapper.chmod(0o700)
    (root / "artifact.txt").write_text("native predecessor output\n")
    (root / "checks").mkdir()
    (root / "checks" / "data.txt").write_text("verifier-only input\n")
    task = data["phases"][0]["tasks"][0]
    data["tools"] = ["shell_exec", "fs_read"]
    task.update(
        execution_mode="exclusive",
        allowed_tools=["shell_exec"],
        outputs=["artifact.txt"],
        footprint=[
            {"path": "source.py", "access": "read"},
            {"path": "artifact.txt", "access": "write"},
            {"path": "verify.sh", "access": "read"},
            {"path": "checks", "access": "read", "subtree": True},
        ],
        expected_checks=["native-check"],
        verifiers=[
            {
                "id": "native-check",
                "executor_tool_id": "shell_exec",
                "verifier_path": str(wrapper),
                "verifier_sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
                "input_paths": ["checks"],
            }
        ],
    )
    data["phases"][0]["tasks"].append(
        {
            "id": "task-2",
            "role": "worker",
            "instructions": "Use the native handoff",
            "target": task["target"],
            "allowed_tools": ["fs_read"],
            "dependencies": ["task-1"],
            "inputs": ["task-1:artifact.txt"],
            "footprint": [{"path": "artifact.txt", "access": "read"}],
        }
    )


@pytest.fixture
async def dependency_native(native_host, monkeypatch):
    from Tests.Agents.test_agent_service import fence
    from Tests.Agents.test_local_tool_provider import make_provider
    from Tests.Chat.test_automatic_provider_budget import response
    from tldw_chatbook.Agents.raw_shell_tool_provider import RawShellToolProvider
    from tldw_chatbook.Chat.console_raw_cli import RawCliRuntime

    h = native_host
    raw = RawCliRuntime(lambda: True)
    raw.arm()
    raw.grant_model_session(h.session.id)
    h.controller.app.raw_cli_runtime = raw
    provider = RawShellToolProvider(
        runtime=raw, console_session_id=h.session.id, initial_directory=lambda: h.root
    )
    replies = iter(
        [
            fence(
                "shell_exec",
                {"command": shlex.join((str(h.root / "verify.sh"),)), "shell": "bash"},
            ),
            "verified",
            fence("fs_read", {"path": "source.py"}),
            "consumer done",
        ]
    )

    def chat(**kwargs):
        h.calls.append(kwargs)
        result = response()
        result["choices"][0]["message"]["content"] = next(replies)
        return result

    monkeypatch.setattr(h.gateway, "_chat_api_call_fn", chat)
    prepared = await h.bridge._prepare_workflow_reply(
        phase=h.phase,
        task_id="task-1",
        binding=h.binding,
        resolution=h.selected,
        raw_shell_provider=provider,
        native_tools_enabled=False,
    )
    try:
        receipt = await native_tests._launch_native_for_test(h, prepared)
        await native_tests._settle_native_for_test(h, prepared, receipt)
        saved = await h.service.call(h.service.get, h.workflow.id)
        assert saved.results[0].checks[0].status == "passed"
        checkpoint = h.service._checkpoint(h.workflow.id, receipt.attempt_id)
        local = make_provider(
            root=h.root, admitted_roots=(h.binding,), allow_write=True
        )
        yield SimpleNamespace(
            host=h, shell=Path(checkpoint.verifiers[0].shell_path), local=local
        )
    finally:
        raw.shutdown()


@pytest.mark.parametrize("native_host", [_dependency_plan], indirect=True)
async def test_preparation_and_launch_cover_real_dependency_shell_and_sources(
    dependency_native, monkeypatch
):
    d = dependency_native
    h = d.host
    entered, release = threading.Event(), threading.Event()
    original = evidence._shell_manifest
    owners = []

    def held(path):
        owners.append(current_execution_owner())
        entered.set()
        assert release.wait(5)
        return original(path)

    monkeypatch.setattr(evidence, "_shell_manifest", held)
    preparation = asyncio.create_task(
        h.bridge._prepare_workflow_reply(
            phase=h.phase,
            task_id="task-2",
            binding=h.binding,
            resolution=h.selected,
            local_provider=d.local,
        )
    )
    shell_writer = WorktreeFootprint(
        writes=(WorktreeScope(d.shell, False, capture_directory_chain(d.shell.parent)),)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        with h.runtime.worktree_activity.prelaunch(footprint=shell_writer) as claim:
            assert claim is None, "preparation reads the native shell without its claim"
        assert owners[-1] is not None
    finally:
        release.set()
        prepared = await preparation
    auth = prepared.authorization
    paths = [
        h.root / name
        for name in ("source.py", "artifact.txt", "checks/data.txt", "verify.sh")
    ]
    paths.append(d.shell)
    assert {scope.path for scope in auth.footprint.reads} == {h.root / "artifact.txt"}
    for path in paths:
        writer = WorktreeFootprint(
            writes=(WorktreeScope(path, False, capture_directory_chain(path.parent)),)
        )
        assert auth.physical_footprint._conflicts(writer), path
    entered.clear()
    release.clear()
    before_rows = h.db.list_runs(h.conversation_id)
    before_calls = len(h.calls)
    launch = asyncio.create_task(native_tests._launch_native_for_test(h, prepared))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert h.db.list_runs(h.conversation_id) == before_rows
        assert len(h.calls) == before_calls
        for path in paths:
            writer = WorktreeFootprint(
                writes=(
                    WorktreeScope(path, False, capture_directory_chain(path.parent)),
                )
            )
            with h.runtime.worktree_activity.prelaunch(footprint=writer) as claim:
                assert claim is None, path
        owner = owners[-1]
        assert owner.child and owner.run_id is None
        h.runtime.worktree_activity.validate_claim(
            auth._native_preparation[0].activity_claims[0],
            owner=owner,
            footprint=auth.physical_footprint,
        )
    finally:
        release.set()
        receipt = await launch
        await native_tests._settle_native_for_test(h, prepared, receipt)
    assert "workflow task scope refused" in json.dumps(h.calls[-1])


async def test_contention_retry_keeps_original_reservation_and_duplicate_skips_reads(
    native_host, monkeypatch
):
    from tldw_chatbook.Agents.agent_service import _WorkflowChildWait
    from tldw_chatbook.Agents.execution_capacity import WorkOrigin

    h = native_host
    prepared = await h.bridge._prepare_workflow_reply(
        phase=h.phase, task_id="task-1", binding=h.binding, resolution=h.selected
    )
    auth = prepared.authorization
    reservation = auth.claim.reservation_id
    original = evidence._manifest
    calls = []

    def counted(*args, **kwargs):
        calls.append(current_execution_owner())
        return original(*args, **kwargs)

    monkeypatch.setattr(evidence, "_manifest", counted)
    writer = WorktreeFootprint(
        writes=(
            WorktreeScope(h.root / "source.py", False, capture_directory_chain(h.root)),
        )
    )
    owner = h.runtime.execution_capacity.begin_execution(
        origin=WorkOrigin.MANUAL, conversation_id="contender"
    )
    claim = h.runtime.worktree_activity.try_acquire(owner=owner, footprint=writer)
    assert claim is not None
    try:
        with pytest.raises(_WorkflowChildWait, match="activity busy"):
            await native_tests._launch_native_for_test(h, prepared)
        assert calls == []
        assert auth._native_launch == []
        assert h.db.list_runs(h.conversation_id) == []
        with h.db.connection() as conn:
            assert (
                conn.execute(
                    "SELECT state FROM automatic_work_reservations WHERE id=?",
                    (reservation,),
                ).fetchone()[0]
                == "reserved"
            )
    finally:
        owner.finish_root()
        claim.settle(durable=True)
    receipt = await native_tests._launch_native_for_test(h, prepared)
    try:
        for thread in auth._native_preparation[0].threads.values():
            await asyncio.to_thread(thread.join, 5)
            assert not thread.is_alive()
        assert calls and all(owner is receipt.owner for owner in calls)
        reads = len(calls)
        assert await native_tests._launch_native_for_test(h, prepared) is receipt
        assert len(calls) == reads
        assert auth.claim.reservation_id == reservation
    finally:
        await native_tests._settle_native_for_test(h, prepared, receipt)


@pytest.mark.parametrize("native_host", [_dependency_plan], indirect=True)
@pytest.mark.parametrize("mutation", ["replace", "alias", "excluded", "shell", "body"])
async def test_dependency_source_change_refuses_before_child_commit(
    dependency_native, monkeypatch, mutation
):
    from tldw_chatbook.Tools import raw_cli_executor

    d = dependency_native
    h = d.host
    prepared = await h.bridge._prepare_workflow_reply(
        phase=h.phase,
        task_id="task-2",
        binding=h.binding,
        resolution=h.selected,
        local_provider=d.local,
    )
    auth = prepared.authorization
    source = h.root / "source.py"
    if mutation == "replace":
        replacement = h.root / "replacement.py"
        replacement.write_bytes(source.read_bytes())
        replacement.replace(source)
    elif mutation == "alias":
        original = h.root / "original.py"
        source.rename(original)
        source.symlink_to(original)
    elif mutation == "excluded":
        h.registry.add_binding_exclusion("w", "b", "source.py")
    elif mutation == "shell":
        # Current executable selection is metadata; it grants no new shell authority.
        monkeypatch.setattr(
            raw_cli_executor, "resolve_shell_argv", lambda *a: (str(source),)
        )
    else:
        source.write_text("external predecessor edit\n")
    original_manifest = evidence.capture_manifest
    reads = []

    def observed(*args, **kwargs):
        reads.append(current_execution_owner())
        return original_manifest(*args, **kwargs)

    monkeypatch.setattr(evidence, "capture_manifest", observed)
    before_rows, before_calls = h.db.list_runs(h.conversation_id), len(h.calls)
    try:
        with pytest.raises((ValueError, PermissionError)):
            await native_tests._launch_native_for_test(h, prepared)
        assert h.db.list_runs(h.conversation_id) == before_rows
        assert len(h.calls) == before_calls
        assert auth._native_launch == []
        assert auth._native_preparation[0].activity_claims == []
        assert h.runtime.execution_capacity.snapshot().executions == ()
        with h.db.connection() as conn:
            assert (
                conn.execute(
                    "SELECT state FROM automatic_work_reservations WHERE id=?",
                    (auth.claim.reservation_id,),
                ).fetchone()[0]
                == "reserved"
            )
        if mutation == "body":
            assert reads and all(owner is not None and owner.child for owner in reads)
        else:
            assert reads == [], "identity drift must refuse before the first body read"
    finally:
        await prepared.close()
