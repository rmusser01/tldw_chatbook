"""Final native freshness reads retain complete host-only activity custody."""

import asyncio
import hashlib
import json
import shlex
import sys
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

    def held(frame, event, _arg):
        if (
            frame.f_code is not original.__code__
            or frame.f_locals.get("authorization") is not auth
        ):
            return
        if event == "call":
            observed["owner"] = current_execution_owner()
            observed["claims"] = tuple(auth._native_preparation[0].activity_claims)
            entered.set()
            assert release.wait(5)
        elif event == "return":
            observed["returned"] = True

    previous_thread_profile = threading.getprofile()
    previous_profile = sys.getprofile()
    threading.setprofile_all_threads(held)
    try:
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

    finally:
        threading.setprofile_all_threads(previous_thread_profile)
        sys.setprofile(previous_profile)


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
        original_owner = receipt.owner
        original_launch_operation = prepared._launch_operation
        original_native = prepared.authorization._native_preparation[0]
        original_threads = tuple(original_native.threads.values())
        try:
            await native_tests._settle_native_for_test(h, prepared, receipt)
        except ValueError as error:
            if str(error) == "native_retirement_unproven":
                try:
                    state = "unavailable"
                    capacity = original_owner.capacity
                    if capacity._lock.acquire(blocking=False):
                        try:
                            state = (
                                original_owner._root_finished,
                                original_owner._drain_outcome,
                                original_owner._cleanup_unproven,
                                capacity._owners.get(original_owner.execution_id)
                                is original_owner,
                                tuple(
                                    (id(op), op.kind, op is original_launch_operation)
                                    for op in original_owner._operations
                                ),
                            )
                        finally:
                            capacity._lock.release()
                    tracker = h.service._tasks.get(receipt.attempt_id)
                    associated = (
                        tracker is not None
                        and tracker.authorization is prepared.authorization
                        and tracker.receipt is receipt
                    )
                    print(
                        "G_DIRECT_DEPENDENCY_ORIGINAL_RETIREMENT",
                        {
                            "owner": id(original_owner),
                            "launch_operation": id(original_launch_operation),
                            "launch_still_installed": prepared._launch_operation
                            is original_launch_operation,
                            "close_operation": id(prepared._close_operation)
                            if prepared._close_operation is not None
                            else None,
                            "state": state,
                            "tracker": (tracker.drain, len(tracker.pending))
                            if associated
                            else "unassociated",
                            "receipt_failure": receipt.failure,
                            "original_receipt": len(
                                prepared.authorization._native_launch
                            )
                            == 1
                            and prepared.authorization._native_launch[0] is receipt,
                            "original_native": prepared.authorization._native_preparation[
                                0
                            ]
                            is original_native,
                            "threads": [
                                (id(t), t.is_alive()) for t in original_threads
                            ],
                        },
                        flush=True,
                    )
                except BaseException as diagnostic_error:  # noqa: BLE001 - preserve original setup and teardown errors.
                    print(
                        "G_DIRECT_DEPENDENCY_DIAGNOSTIC_ERROR",
                        type(diagnostic_error).__name__,
                        flush=True,
                    )
            raise
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

    def held(frame, event, _arg):
        if event == "call" and frame.f_code is original.__code__:
            owners.append(current_execution_owner())
            entered.set()
            assert release.wait(5)

    previous_thread_profile = threading.getprofile()
    previous_profile = sys.getprofile()
    threading.setprofile_all_threads(held)
    try:
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
            writes=(
                WorktreeScope(d.shell, False, capture_directory_chain(d.shell.parent)),
            )
        )
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            with h.runtime.worktree_activity.prelaunch(footprint=shell_writer) as claim:
                assert claim is None, (
                    "preparation reads the native shell without its claim"
                )
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
                writes=(
                    WorktreeScope(path, False, capture_directory_chain(path.parent)),
                )
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
                        WorktreeScope(
                            path, False, capture_directory_chain(path.parent)
                        ),
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

    finally:
        threading.setprofile_all_threads(previous_thread_profile)
        sys.setprofile(previous_profile)


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

    def counted(frame, event, _arg):
        if event == "call" and frame.f_code is original.__code__:
            calls.append(current_execution_owner())

    previous_thread_profile = threading.getprofile()
    previous_profile = sys.getprofile()
    threading.setprofile_all_threads(counted)
    try:
        writer = WorktreeFootprint(
            writes=(
                WorktreeScope(
                    h.root / "source.py", False, capture_directory_chain(h.root)
                ),
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

    finally:
        threading.setprofile_all_threads(previous_thread_profile)
        sys.setprofile(previous_profile)


@pytest.mark.parametrize("native_host", [_dependency_plan], indirect=True)
@pytest.mark.parametrize("mutation", ["replace", "alias", "shell", "body"])
async def test_dependency_source_change_refuses_before_child_commit(
    dependency_native, monkeypatch, mutation
):
    import dis

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
    validator_returns = []
    validator_code = evidence.WorkflowRunService._task_input_result.__wrapped__.__code__

    def observed(frame, event, _arg):
        if event == "call" and frame.f_code is original_manifest.__code__:
            reads.append(current_execution_owner())
        if (
            frame.f_code is validator_code
            and frame.f_locals.get("authorization") is auth
            and event == "return"
            and dis.opname[frame.f_code.co_code[frame.f_lasti]]
            in {"RETURN_VALUE", "RETURN_CONST"}
        ):
            validator_returns.append(_arg)

    previous_thread_profile = threading.getprofile()
    previous_profile = sys.getprofile()
    threading.setprofile_all_threads(observed)
    try:
        before_rows, before_calls = h.db.list_runs(h.conversation_id), len(h.calls)
        try:
            with pytest.raises((ValueError, PermissionError)) as caught:
                await native_tests._launch_native_for_test(h, prepared)
            original_error = caught.value
            original_stack = []
            tb = original_error.__traceback__
            while tb is not None:
                original_stack.append(
                    (
                        tb.tb_frame.f_code.co_filename,
                        tb.tb_frame.f_code.co_name,
                        tb.tb_lineno,
                    )
                )
                tb = tb.tb_next
            print(
                "G_FRESHNESS_ORIGINAL_REFUSAL",
                mutation,
                type(original_error).__name__,
                original_stack,
                [
                    (
                        type(value).__name__,
                        value.conclusive,
                        value.error is original_error,
                    )
                    for value in validator_returns
                ],
                flush=True,
            )
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
                assert reads and all(
                    owner is not None and owner.child for owner in reads
                )
            else:
                assert reads == [], (
                    "identity drift must refuse before the first body read"
                )
        finally:
            await prepared.close()

    finally:
        threading.setprofile_all_threads(previous_thread_profile)
        sys.setprofile(previous_profile)


async def excluded_prepared_path_journey(tmp_path, _unused):
    """A real new exclusion retains the original unreturned native launch."""
    import dis
    import os
    from types import CodeType

    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Tools.local_tool_impls import (
        LocalToolError,
        resolve_workspace_path,
    )

    saved, errors, validator_returns = {}, [], []
    generators = []
    primary = None
    previous_thread_profile, previous_profile = threading.getprofile(), sys.getprofile()
    with pytest.MonkeyPatch.context() as monkeypatch:
        try:
            native_input.__wrapped__(monkeypatch)
            host_generator = native_tests.native_host.__wrapped__(
                tmp_path, monkeypatch, SimpleNamespace(param=_dependency_plan)
            )
            generators.append(host_generator)
            h = await anext(host_generator)
            dependency_generator = dependency_native.__wrapped__(h, monkeypatch)
            generators.append(dependency_generator)
            d = await anext(dependency_generator)
            prepared = await h.bridge._prepare_workflow_reply(
                phase=h.phase,
                task_id="task-2",
                binding=h.binding,
                resolution=h.selected,
                local_provider=d.local,
            )
            auth = prepared.authorization
            native = auth._native_preparation[0]
            source, lifetime, source_scope = (
                prepared.source,
                prepared._lifetime,
                prepared._source_scope,
            )
            validator_code = (
                evidence.WorkflowRunService._task_input_result.__wrapped__.__code__
            )
            launch_codes = [
                value
                for value in AgentService._workflow_launch_result.__code__.co_consts
                if isinstance(value, CodeType) and value.co_name == "launch"
            ]
            assert len(launch_codes) == 1

            def observe(frame, event, result):
                if (
                    frame.f_code is not validator_code
                    or frame.f_locals.get("authorization") is not auth
                ):
                    return
                try:
                    if event == "call":
                        parent = frame.f_back
                        while (
                            parent is not None and parent.f_code is not launch_codes[0]
                        ):
                            parent = parent.f_back
                        assert parent is not None
                        entry = parent.f_locals["entry"]
                        owner, claim = (
                            parent.f_locals["owner"],
                            parent.f_locals["claim"],
                        )
                        assert (
                            entry.preparation is prepared
                            and entry.authorization is auth
                        )
                        assert entry.native is native and entry.mode == "D"
                        assert current_execution_owner() is owner
                        assert native.activity_claims == [claim]
                        assert parent.f_locals["operation"] is None
                        assert parent.f_locals["close_operation"] is None
                        assert not saved
                        saved.update(entry=entry, owner=owner, claim=claim)
                    elif event == "return" and dis.opname[
                        frame.f_code.co_code[frame.f_lasti]
                    ] in {"RETURN_VALUE", "RETURN_CONST"}:
                        validator_returns.append(result)
                except BaseException as error:  # noqa: BLE001 - never throw from native observation.
                    errors.append(error)

            def charges():
                with h.db.connection() as conn:
                    return tuple(
                        tuple(row)
                        for row in conn.execute(
                            "SELECT id,chain_id,owner_id,kind,amount,state,actual_amount "
                            "FROM automatic_work_reservations ORDER BY id"
                        )
                    )

            before_rows, before_calls = h.db.list_runs(h.conversation_id), len(h.calls)
            before_charges = charges()
            before_accounting = h.db.automatic_work.snapshot(auth.claim.chain_id)
            assert before_calls == 2 and len(before_rows) == 1
            assert before_accounting.used["generation"] == 1
            assert before_accounting.used["child_launch"] == 1
            assert before_accounting.used["model_call"] == 2
            assert before_accounting.reserved["child_launch"] == 1
            h.registry.add_binding_exclusion("w", "b", "source.py")
            threading.setprofile_all_threads(observe)
            try:
                with pytest.raises(LocalToolError, match="protected path") as refused:
                    await native_tests._launch_native_for_test(h, prepared)
            finally:
                threading.setprofile_all_threads(previous_thread_profile)
                sys.setprofile(previous_profile)
            if errors:
                raise errors[0]
            assert saved and validator_returns == []
            original_error = refused.value
            tb = original_error.__traceback__
            resolver_frames = []
            while tb is not None:
                if tb.tb_frame.f_code is resolve_workspace_path.__code__:
                    resolver_frames.append(tb.tb_frame)
                tb = tb.tb_next
            assert len(resolver_frames) == 1
            assert auth._native_launch == [None]
            assert native.activity_claims == [saved["claim"]]
            assert saved["entry"].marker is auth._native_launch
            owner = saved["owner"]
            assert (
                owner.is_admitted and not owner._root_finished and owner.run_id is None
            )
            h.runtime.worktree_activity.validate_claim(
                saved["claim"], owner=owner, footprint=auth.physical_footprint
            )
            assert (
                prepared._launch_operation is None and prepared._close_operation is None
            )
            assert not native.threads and not native.uncertain_start
            for _ in range(2):
                with pytest.raises(
                    PermissionError, match="launch return has not completed"
                ):
                    await prepared.close()
                assert prepared in h.runtime._workflow_preparations
                assert prepared.authorization is auth and prepared.phase is h.phase
                assert auth._native_preparation[0] is native
                assert auth._bridge_preparation[0] is prepared
                assert h.phase._task_scopes["task-2"] is auth
                assert prepared.source is source and prepared._lifetime is lifetime
                assert prepared._source_scope is source_scope
                assert not prepared._closed and not prepared._closing
                assert auth._native_launch == [None]
                assert native.activity_claims == [saved["claim"]]
                assert owner.is_admitted and not owner._root_finished
                assert h.db.list_runs(h.conversation_id) == before_rows
                assert len(h.calls) == before_calls and charges() == before_charges
            after_accounting = h.db.automatic_work.snapshot(auth.claim.chain_id)
            assert dict(after_accounting.used) == dict(before_accounting.used)
            assert dict(after_accounting.reserved) == dict(before_accounting.reserved)
            print("G_EXCLUDED_PATH_RETAINED_ORIGINAL_CUSTODY", flush=True)
            os._exit(17)  # Deliberate isolated retained exit; no creator/drain claim.
        except BaseException as error:
            primary = error
            raise
        finally:
            threading.setprofile_all_threads(previous_thread_profile)
            sys.setprofile(previous_profile)
            cleanup_errors = []
            for generator in reversed(generators):
                try:
                    await generator.aclose()
                except BaseException as error:  # noqa: BLE001 - attempt every original fixture finally.
                    cleanup_errors.append(error)
            if primary is not None:
                for error in cleanup_errors:
                    primary.add_note("original fixture cleanup: " + repr(error))
            elif cleanup_errors:
                raise cleanup_errors[0]


async def test_exclusion_after_native_preparation_retains_original_unknown_custody(
    tmp_path,
):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    with pytest.raises(
        AssertionError, match="G_EXCLUDED_PATH_RETAINED_ORIGINAL_CUSTODY"
    ):
        _run(
            tmp_path,
            "excluded-prepared",
            "native",
            script=Path(__file__)
            .parents[1]
            .joinpath("Chat/workflow_coordinator_probe.py")
            .read_text(),
            timeout=60,
        )
