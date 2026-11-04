"""Native invocation returns exclude competing launch and retirement writers."""

import asyncio
import contextlib
import dis
import sys
import threading

import pytest

from Tests.Chat import test_console_native_preparation as native_tests
from Tests.Chat.test_console_native_preparation import (
    native_host as native_host,  # noqa: PLC0414
)
from tldw_chatbook.Agents.agent_service import (
    _WORKFLOW_LAUNCH_CODE,
    AgentService,
    _WorkflowChildWait,
    _WorkflowLaunchReturn,
)
from tldw_chatbook.Agents.execution_capacity import OwnedOperation, WorkOrigin
from tldw_chatbook.Backup_Recovery.participants import _core_operation
from tldw_chatbook.Chat.console_workflow_wave import _WorkflowWave, run_workflow_wave
from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint, WorktreeScope

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


async def consent_unknown_return_journey(tmp_path, _unused):
    import inspect
    import os
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_project_instructions import (
        ProjectInstructionControlState,
    )
    from tldw_chatbook.Chat.console_workflow_preparation import (
        _finish_native_preparation,
        _PreparedWorkflowReply,
    )

    mode = sys.argv[2]
    assert mode in {"release-return", "source-shadow", "owning-exit"}
    generator = primary = None
    saved, errors, confirmations, replacements, finishes = {}, [], [], [], []
    sentinel = RuntimeError("original consent boundary failed")
    previous_profile, previous_thread_profile = sys.getprofile(), threading.getprofile()
    previous_trace, previous_thread_trace = sys.gettrace(), threading.gettrace()
    release_code = _PreparedWorkflowReply._release_configuration.__code__
    source_code = _PreparedWorkflowReply.activate_source.__wrapped__.__code__
    own_code = next(
        code
        for code in _PreparedWorkflowReply._read.__code__.co_consts
        if inspect.iscode(code) and code.co_name == "invoke_owned"
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        try:
            generator = native_tests.native_host.__wrapped__(
                tmp_path, monkeypatch, SimpleNamespace(param=None)
            )
            h = await anext(generator)
            (h.root / "AGENTS.md").write_text("Original retained consent source\n")
            h.session.project_instruction_state = ProjectInstructionControlState(
                project_instructions_enabled=True,
                working_folder_binding_id=h.binding.binding_id,
                working_folder_locator_fingerprint=h.binding.locator_fingerprint,
            )
            before = h.db.automatic_work.snapshot(h.phase.attempt.chain_id)
            issuer, host = asyncio.current_task(), threading.current_thread()

            def replacement(*_args):
                replacements.append(True)
                raise AssertionError("replacement scope exit must not execute")

            def confirmation(snapshot):
                p = saved["prepared"]
                assert snapshot is p._snapshot and mode == "owning-exit"
                confirmations.append(snapshot)
                record = p._finite_reads[-1]
                assert record.prepared is p and record.issuing is issuer
                assert record.configuration_return is p._consent_configuration_return
                assert (
                    record.owner is not None
                    and record.operation is not None
                    and record.claim is None
                )
                saved.update(
                    record=record, owner=record.owner, operation=record.operation
                )
                raise sentinel

            def profile(frame, event, result):
                if event == "call" and frame.f_code is OwnedOperation.finish.__code__:
                    if (
                        saved.get("operation") is frame.f_locals.get("self")
                        and "operation" in saved
                    ):
                        finishes.append(frame)
                    return
                if frame.f_code is not release_code or event not in {"call", "return"}:
                    return
                parent = frame.f_back
                if (
                    parent is None
                    or parent.f_code is not _finish_native_preparation.__code__
                ):
                    return
                inject = shadow = False
                try:
                    p = frame.f_locals["self"]
                    assert p.phase is h.phase and parent.f_locals["prepared"] is p
                    assert (
                        threading.current_thread() is host
                        and asyncio.current_task() is issuer
                    )
                    if event == "call":
                        assert not saved
                        scope = p._source_scope
                        assert scope is not None and scope.gen.gi_frame is not None
                        assert p in h.runtime._workflow_preparations
                        assert h.phase._task_scopes["task-1"] is p.authorization
                        assert p.authorization._bridge_preparation[0] is p
                        for record in p._finite_reads:
                            p._finite_read_return(record)
                            assert record.positive
                        saved.update(
                            prepared=p,
                            scope=scope,
                            auth=p.authorization,
                            source=p.source,
                            native=p.native,
                            lifetime=p._lifetime,
                            prefix=tuple(p._finite_reads),
                        )
                        shadow = mode == "source-shadow"
                    elif dis.opname[frame.f_code.co_code[frame.f_lasti]] in {
                        "RETURN_VALUE",
                        "RETURN_CONST",
                    }:
                        assert p is saved["prepared"]
                        scope, registry = (
                            saved["scope"],
                            h.runtime._workflow_preparations,
                        )
                        assert type(result) is tuple and len(result) == 3
                        assert all(a is b for a, b in zip(result, (p, scope, registry)))
                        assert (
                            scope.gen.gi_frame is None
                            and p._source_scope is None
                            and p not in registry
                        )
                        assert p._configuration_transition[:3] == (p, scope, registry)
                        saved.update(
                            returned=result, transition=p._configuration_transition
                        )
                        inject = mode == "release-return"
                except BaseException as error:  # noqa: BLE001 - observation cannot change native disposition.
                    errors.append(error)
                if shadow:
                    saved["scope"].__exit__ = replacement
                    saved["shadowed"] = True
                if inject:
                    sys.setprofile(previous_profile)
                    raise sentinel

            def trace(frame, event, _value):
                if frame.f_code is not source_code:
                    return None
                if event != "return" or dis.opname[
                    frame.f_code.co_code[frame.f_lasti]
                ] not in {"RETURN_VALUE", "RETURN_CONST"}:
                    return trace
                inject = False
                try:
                    record = saved.get("record")
                    if (
                        record is None
                        or frame.f_locals.get("self") is not saved["prepared"]
                    ):
                        return trace
                    parent = frame.f_back
                    while parent is not None and parent.f_code is not own_code:
                        parent = parent.f_back
                    assert parent is not None and parent.f_locals["record"] is record
                    assert parent.f_locals["self"] is saved["prepared"]
                    assert (
                        record.body_error is sentinel and record.owner is saved["owner"]
                    )
                    assert (
                        record.operation is saved["operation"]
                        and not record.scopes_returned
                    )
                    saved["source_exit"] = frame
                    inject = True
                except BaseException as error:  # noqa: BLE001 - native source fault occurs only after exact associations.
                    errors.append(error)
                if inject:
                    threading.settrace_all_threads(previous_thread_trace)
                    sys.settrace(previous_trace)
                    raise sentinel
                return trace

            threading.setprofile_all_threads(profile)
            if mode == "owning-exit":
                threading.settrace_all_threads(trace)
            with pytest.raises(RuntimeError) as caught:
                await h.bridge._prepare_workflow_reply(
                    phase=h.phase,
                    task_id="task-1",
                    binding=h.binding,
                    resolution=h.selected,
                    confirm_project_instruction_dispatch=confirmation,
                )
            threading.setprofile_all_threads(previous_thread_profile)
            sys.setprofile(previous_profile)
            threading.settrace_all_threads(previous_thread_trace)
            sys.settrace(previous_trace)
            assert errors == [] and replacements == [] and finishes == []
            p, scope, auth = saved["prepared"], saved["scope"], saved["auth"]
            records = tuple(p._finite_reads)
            assert p.authorization is auth and auth._bridge_preparation[0] is p
            assert h.phase._task_scopes["task-1"] is auth
            assert (
                p.source is saved["source"]
                and p.native is saved["native"]
                and p._lifetime is saved["lifetime"]
            )
            assert p._terminal_configuration_return is None and not p._closed
            if mode == "owning-exit":
                assert len(confirmations) == 1 and saved.get("source_exit") is not None
                record = saved["record"]
                assert record is records[-1] and record.before_issue is None
                assert record.body_error is record.wrapper_error is sentinel
                assert (
                    record.proof[1].done() and record.proof[1].exception() is sentinel
                )
                assert record.proof[2].done() and record.proof[2].result() is None
                assert not record.scopes_returned and not record.positive
                assert p._finite == (saved["owner"], None)
                assert (
                    saved["operation"] in saved["owner"]._operations
                    and saved["owner"].is_admitted
                )
                assert p._consent_configuration_return[0] is saved["returned"]
                assert str(caught.value) == "original workflow reader return unproven"
            else:
                assert confirmations == [] and records == saved["prefix"]
                assert p._consent_configuration_return is None
                if mode == "release-return":
                    assert p._configuration_transition is saved["transition"]
                    assert (
                        p._configuration_transition_error is sentinel
                        and caught.value.__cause__ is sentinel
                    )
                else:
                    assert saved["shadowed"] and p._configuration_transition is None
                    assert isinstance(
                        p._configuration_transition_error, PermissionError
                    )
                    assert (
                        str(p._configuration_transition_error)
                        == "original configuration source changed"
                    )
            assert (
                (p._source_scope is scope and p in h.runtime._workflow_preparations)
                if mode == "source-shadow"
                else (
                    p._source_scope is None
                    and p not in h.runtime._workflow_preparations
                )
            )
            claim = auth.claim

            def reservation():
                with h.db.connection() as conn:
                    return tuple(
                        conn.execute(
                            "SELECT id,chain_id,owner_id,kind,amount,state FROM automatic_work_reservations WHERE id=?",
                            (claim.reservation_id,),
                        ).fetchone()
                    )

            row = reservation()
            assert row == (
                claim.reservation_id,
                claim.chain_id,
                claim.owner_id,
                "child_launch",
                1,
                "reserved",
            )
            after = h.db.automatic_work.snapshot(claim.chain_id)
            expected = dict(before.reserved)
            expected["child_launch"] += 1
            assert after.used == before.used and after.reserved == expected
            for _ in range(2):
                with pytest.raises((RuntimeError, PermissionError)):
                    await p.close()
                assert (
                    tuple(p._finite_reads) == records
                    and p._terminal_configuration_return is None
                )
                assert p.authorization is auth and auth._bridge_preparation[0] is p
                assert replacements == [] and finishes == []
                if mode == "owning-exit":
                    assert p._finite == (saved["owner"], None)
                    assert saved["operation"] in saved["owner"]._operations
                    assert saved["owner"].is_admitted
                    assert record.wrapper_error is record.body_error is sentinel
                assert reservation() == row
                current = h.db.automatic_work.snapshot(claim.chain_id)
                assert current.used == after.used and current.reserved == after.reserved
            assert h.calls == [] and h.db.list_runs(h.conversation_id) == []
            print("G_CONSENT_UNKNOWN_RETAINED; no fixture/creator drain", flush=True)
            os._exit(17)
        except BaseException as error:
            primary = error
            raise
        finally:
            threading.setprofile_all_threads(previous_thread_profile)
            sys.setprofile(previous_profile)
            threading.settrace_all_threads(previous_thread_trace)
            sys.settrace(previous_trace)
            if generator is not None:
                try:
                    await generator.aclose()
                except BaseException as error:
                    if primary is None:
                        raise
                    primary.add_note(
                        "Original fixture cleanup also failed: " + type(error).__name__
                    )
                    primary.cleanup_error = error


@pytest.mark.parametrize("mode", ["release-return", "source-shadow", "owning-exit"])
async def test_original_consent_unknown_return_retains_same_custody(tmp_path, mode):
    import subprocess
    from pathlib import Path

    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    route = "consent-unknown"
    script = Path(__file__).with_name("workflow_coordinator_probe.py").read_text()
    with pytest.raises(AssertionError, match="G_CONSENT_UNKNOWN_RETAINED") as caught:
        await asyncio.to_thread(_run, tmp_path, route, mode, script=script, timeout=60)
    frames, tb = [], caught.value.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code is _run.__code__:
            frames.append(tb.tb_frame)
        tb = tb.tb_next
    assert len(frames) == 1
    actual = frames[0].f_locals["result"]
    assert type(actual) is subprocess.CompletedProcess and actual.returncode == 17
    assert actual.args[-2:] == [route, mode]


async def _original_wave_release_receiver_control(h, fail_return):
    from tldw_chatbook.Chat.console_workflow_wave import _workflow_wave_return

    prepared = await h.bridge._prepare_workflow_reply(
        phase=h.phase, task_id="task-1", binding=h.binding, resolution=h.selected
    )
    scope, auth, source = (
        prepared._source_scope,
        prepared.authorization,
        prepared.source,
    )
    native = auth._native_preparation[0]
    capacity = h.runtime.execution_capacity
    capacity._refresh_limits()
    allowance = capacity.max_child_executions - capacity.reserved_manual_children
    assert 0 < allowance <= capacity.max_child_executions
    before = h.db.automatic_work.snapshot(auth.claim.chain_id)

    def reservation():
        with h.db.connection() as conn:
            return tuple(
                conn.execute(
                    "SELECT id,chain_id,owner_id,kind,amount,state FROM automatic_work_reservations WHERE id=?",
                    (auth.claim.reservation_id,),
                ).fetchone()
            )

    original_reservation = reservation()
    saved, errors, competitors, returns = {}, [], [], []
    primary = None
    failure = RuntimeError("original final wave release return failed")
    previous = sys.getprofile()
    host = threading.current_thread()
    codes = (
        _WorkflowWave.__init__.__code__,
        _WorkflowWave._open.__code__,
        _WorkflowWave._release.__code__,
    )

    def observe(frame, event, result):
        if event != "return" or frame.f_code not in codes:
            return
        if dis.opname[frame.f_code.co_code[frame.f_lasti]] not in {
            "RETURN_VALUE",
            "RETURN_CONST",
        }:
            return
        wave = frame.f_locals["self"]
        if wave.prepared != (prepared,):
            return
        inject = False
        try:
            assert threading.current_thread() is host and wave.phase is h.phase
            if frame.f_code is codes[0]:
                assert not saved
                saved["wave"] = wave
            elif frame.f_code is codes[1]:
                assert wave is saved["wave"] and asyncio.current_task() is wave.driver
                assert type(result.error) is _WorkflowChildWait and not result.retain
                assert result.source is result.footprint is None
                saved["opening"] = result
            else:
                assert wave is saved["wave"] and asyncio.current_task() is wave.driver
                assert wave.driver.get_coro() is wave._driver_coroutine
                assert wave.closed and wave not in h.runtime._workflow_waves
                assert wave._release_return is None
                terminal = prepared._terminal_configuration_return
                returned, closing, records = terminal
                assert returned[0] is prepared and returned[1] is scope
                assert returned[2] is h.runtime._workflow_preparations
                assert records[-1] is closing and closing.positive
                assert closing.wrapper_error is closing.body_error is None
                proof = prepared._finite_read_return(closing)
                assert proof is closing.proof[1].result()
                assert closing.proof[2].done() and closing.proof[2].result() is None
                assert scope.gen.gi_frame is None and prepared._source_scope is None
                assert (
                    prepared._closed
                    and prepared not in h.runtime._workflow_preparations
                )
                expected = (
                    wave,
                    wave.driver,
                    wave._driver_coroutine,
                    wave.source,
                    wave.reservation,
                    wave.fence,
                    wave.baseline_return,
                    wave.ending_return,
                    wave.prepared,
                )
                assert type(result) is tuple and len(result) == 12
                assert all(a is b for a, b in zip(result[:9], expected))
                assert result[9:11] == ((), ()) and len(result[11]) == 1
                assert result[11][0][0] is prepared and result[11][0][1] is terminal
                saved.update(
                    returned=result,
                    terminal=terminal,
                    closing=closing,
                    proof=proof,
                    records=records,
                )
                inject = fail_return
        except BaseException as error:  # noqa: BLE001 - passive observations preserve native errors.
            errors.append(error)
        if inject:
            sys.setprofile(previous)
            raise failure

    try:
        for _ in range(allowance):
            owner = capacity.begin_execution(
                origin=WorkOrigin.AUTOMATIC,
                child=True,
                conversation_id="original-release-contender",
            )
            competitors.append(owner)
            owner.on_drained(
                lambda proven, owner=owner: returns.append((owner, proven))
            )
        sys.setprofile(observe)
        with pytest.raises(BaseException) as caught:
            await run_workflow_wave(h.phase, (prepared,))
        sys.setprofile(previous)
        assert errors == []
        wave = saved["wave"]
        assert wave.driver.done() and not wave.driver.cancelled()
        assert wave.driver.exception() is caught.value
        assert wave._driver_coroutine.cr_frame is None
        assert caught.value is (failure if fail_return else saved["opening"].error)
        assert prepared._wave is wave and prepared.authorization is auth
        assert (
            auth._bridge_preparation[0] is prepared
            and h.phase._task_scopes["task-1"] is auth
        )
        assert prepared.source is source and auth._native_preparation[0] is native
        assert wave.source is wave.reservation is wave.fence is wave.publication is None
        assert not wave.baseline_return.done() and not wave.ending_return.done()
        assert not wave.receipts and not wave.unproven_launches
        assert auth._native_launch == [] and not native.threads
        for _ in range(2):
            if fail_return:
                assert wave._release_return is None
                with pytest.raises(
                    RuntimeError, match="original workflow release return unproven"
                ):
                    await prepared.close()
            else:
                assert wave._release_return is saved["returned"]
                await prepared.close()
            assert prepared._terminal_configuration_return is saved["terminal"]
            assert tuple(prepared._finite_reads) == saved["records"]
            assert prepared._finite_read_return(saved["closing"]) is saved["proof"]
        with pytest.raises(
            RuntimeError, match="original workflow wave return unproven"
        ):
            _workflow_wave_return(wave)
        after = h.db.automatic_work.snapshot(auth.claim.chain_id)
        assert after.used == before.used and after.reserved == before.reserved
        assert reservation() == original_reservation
        assert h.calls == [] and h.db.list_runs(h.conversation_id) == []
    except BaseException as error:
        primary = error
        raise
    finally:
        sys.setprofile(previous)
        cleanup_errors = []
        for owner in competitors:
            try:
                owner.finish_root()
            except BaseException as error:  # noqa: BLE001 - preserve actual original cleanup failures.
                cleanup_errors.append(error)
        try:
            assert len(returns) == len(competitors)
            assert all(
                proven is True and not owner.is_admitted for owner, proven in returns
            )
        except BaseException as error:  # noqa: BLE001 - preserve primary and return failures separately.
            cleanup_errors.append(error)
        if cleanup_errors:
            if primary is None:
                raise cleanup_errors[0]
            primary.cleanup_errors = tuple(cleanup_errors)
            for error in cleanup_errors:
                primary.add_note(
                    "Original competitor cleanup also failed: " + type(error).__name__
                )
    assert capacity.snapshot().executions == ()


async def test_original_wave_refusal_keeps_returned_member_close_inert(native_host):
    await _original_wave_release_receiver_control(native_host, False)


async def wave_release_receiver_failure_journey(tmp_path, _unused):
    import os
    from types import SimpleNamespace

    generator = primary = None
    with pytest.MonkeyPatch.context() as monkeypatch:
        try:
            generator = native_tests.native_host.__wrapped__(
                tmp_path, monkeypatch, SimpleNamespace(param=None)
            )
            h = await anext(generator)
            await _original_wave_release_receiver_control(h, True)
            print(
                "G_WAVE_RELEASE_RECEIVER_RETAINED; no fixture/creator drain", flush=True
            )
            os._exit(17)
        except BaseException as error:
            primary = error
            raise
        finally:
            if generator is not None:
                try:
                    await generator.aclose()
                except BaseException as error:
                    if primary is None:
                        raise
                    primary.add_note(
                        "Original fixture cleanup also failed: " + type(error).__name__
                    )
                    primary.cleanup_error = error


async def test_original_wave_release_receiver_failure_refuses_member_noop(tmp_path):
    import subprocess
    from pathlib import Path

    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    route = "wave-release-receiver"
    script = Path(__file__).with_name("workflow_coordinator_probe.py").read_text()
    with pytest.raises(
        AssertionError, match="G_WAVE_RELEASE_RECEIVER_RETAINED"
    ) as caught:
        await asyncio.to_thread(
            _run, tmp_path, route, "retained", script=script, timeout=60
        )
    frames, tb = [], caught.value.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code is _run.__code__:
            frames.append(tb.tb_frame)
        tb = tb.tb_next
    assert len(frames) == 1
    actual = frames[0].f_locals["result"]
    assert type(actual) is subprocess.CompletedProcess and actual.returncode == 17
    assert actual.args[-2:] == [route, "retained"]


@pytest.mark.parametrize("consent", ["none", "proceed", "cancel", "error"])
async def test_original_configuration_consent_and_terminal_returns(
    native_host, consent
):
    from tldw_chatbook.Chat.console_project_instructions import (
        ProjectInstructionControlState,
    )
    from tldw_chatbook.Chat.console_workflow_preparation import _PreparedWorkflowReply

    h = native_host
    if consent != "none":
        (h.root / "AGENTS.md").write_text("Original consent lifecycle instruction\n")
        h.session.project_instruction_state = ProjectInstructionControlState(
            project_instructions_enabled=True,
            working_folder_binding_id=h.binding.binding_id,
            working_folder_locator_fingerprint=h.binding.locator_fingerprint,
        )
    saved, errors, entries, releases, confirmations = {}, [], [], [], []
    returned_prepared = primary = None
    close_attempted = False
    sentinel = RuntimeError("original consent callback body refused")
    issuer, host = asyncio.current_task(), threading.current_thread()
    previous = sys.getprofile()
    initial = h.db.automatic_work.snapshot(h.phase.attempt.chain_id)

    def reservation(claim):
        with h.db.connection() as conn:
            row = conn.execute(
                "SELECT id,chain_id,owner_id,kind,amount,state FROM automatic_work_reservations WHERE id=?",
                (claim.reservation_id,),
            ).fetchone()
        return tuple(row)

    def confirmation(snapshot):
        p = saved["prepared"]
        assert snapshot is p._snapshot and p._source_scope is None
        assert p.authorization._bridge_preparation[0] is p
        confirmations.append(snapshot)
        claim = p.authorization.claim
        saved.update(
            claim=claim,
            baseline=h.db.automatic_work.snapshot(claim.chain_id),
            reservation=reservation(claim),
        )
        if consent == "error":
            raise sentinel
        return consent

    def observe(frame, event, result):
        if (
            frame.f_code
            not in {
                _PreparedWorkflowReply._retain_configuration.__code__,
                _PreparedWorkflowReply._release_configuration.__code__,
            }
            or event != "return"
            or dis.opname[frame.f_code.co_code[frame.f_lasti]]
            not in {"RETURN_VALUE", "RETURN_CONST"}
        ):
            return
        p = frame.f_locals["self"]
        if p.phase is not h.phase:
            return
        try:
            assert (
                threading.current_thread() is host and asyncio.current_task() is issuer
            )
            assert saved.get("prepared", p) is p
            saved["prepared"] = p
            scope = frame.f_locals["scope"]
            if frame.f_code is _PreparedWorkflowReply._retain_configuration.__code__:
                assert (
                    p._source_scope is scope and p in h.runtime._workflow_preparations
                )
                assert scope.gen.gi_frame is not None
                entries.append(scope)
            else:
                assert type(result) is tuple and len(result) == 3
                assert (
                    result[0] is p
                    and result[1] is scope
                    and result[2] is h.runtime._workflow_preparations
                )
                assert scope.gen.gi_frame is None and p._source_scope is None
                assert p not in h.runtime._workflow_preparations
                releases.append(result)
        except BaseException as error:  # noqa: BLE001 - observation never changes native entry/exit.
            errors.append(error)

    sys.setprofile(observe)
    try:
        kwargs = (
            {}
            if consent == "none"
            else {"confirm_project_instruction_dispatch": confirmation}
        )
        if consent in {"none", "proceed"}:
            prepared = await h.bridge._prepare_workflow_reply(
                phase=h.phase,
                task_id="task-1",
                binding=h.binding,
                resolution=h.selected,
                **kwargs,
            )
            returned_prepared = prepared
            assert prepared is saved["prepared"] and not prepared._closed
            if consent == "none":
                claim = prepared.authorization.claim
                saved.update(
                    claim=claim,
                    baseline=h.db.automatic_work.snapshot(claim.chain_id),
                    reservation=reservation(claim),
                )
            else:
                assert len(entries) == 2 and entries[0] is not entries[1]
                assert prepared._source_scope is entries[1]
                assert prepared._consent_configuration_return is None
            close_attempted = True
            await prepared.close()
        else:
            with pytest.raises(BaseException) as caught:
                await h.bridge._prepare_workflow_reply(
                    phase=h.phase,
                    task_id="task-1",
                    binding=h.binding,
                    resolution=h.selected,
                    **kwargs,
                )
            prepared = saved["prepared"]
            if consent == "error":
                assert caught.value is sentinel
            else:
                assert type(caught.value) is PermissionError
                assert str(caught.value) == "workflow project instructions declined"
        assert errors == []
        assert len(confirmations) == (0 if consent == "none" else 1)
        assert len(entries) == len(releases) == (2 if consent == "proceed" else 1)
        assert all(read.before_issue is None for read in prepared._finite_reads)
        for read in prepared._finite_reads:
            prepared._finite_read_return(read)
        assert prepared._closed and prepared._source_scope is None
        returned, closing, records = prepared._terminal_configuration_return
        assert returned is releases[-1] and returned[1] is entries[-1]
        assert records[-1] is closing and closing.positive
        assert closing.wrapper_error is closing.body_error is None
        assert closing.proof[1].done() and closing.proof[1].exception() is None
        assert closing.proof[2].done() and closing.proof[2].result() is None
        assert prepared not in h.runtime._workflow_preparations
        claim = saved["claim"]
        assert prepared.authorization.claim is claim
        assert saved["reservation"] == (
            claim.reservation_id,
            claim.chain_id,
            claim.owner_id,
            "child_launch",
            1,
            "reserved",
        )
        after_first = h.db.automatic_work.snapshot(claim.chain_id)
        assert after_first.used == saved["baseline"].used == initial.used
        assert after_first.reserved == saved["baseline"].reserved
        assert reservation(claim) == saved["reservation"]
        await prepared.close()
        assert tuple(prepared._finite_reads) == records
        assert len(releases) == len(entries)
        after_second = h.db.automatic_work.snapshot(claim.chain_id)
        assert (
            after_second.used == after_first.used
            and after_second.reserved == after_first.reserved
        )
        assert reservation(claim) == saved["reservation"]
        assert h.calls == [] and h.db.list_runs(h.conversation_id) == []
    except BaseException as error:
        primary = error
        raise
    finally:
        sys.setprofile(previous)
        if returned_prepared is not None and not close_attempted:
            try:
                await returned_prepared.close()
            except BaseException as error:
                if primary is None:
                    raise
                primary.add_note(
                    "Original returned preparation cleanup also failed: "
                    + type(error).__name__
                )
                primary.cleanup_error = error


async def configuration_entry_failure_journey(tmp_path, _unused):
    import os
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_workflow_preparation import _PreparedWorkflowReply

    generator = primary = None
    saved, errors = {}, []
    sentinel = RuntimeError("original initial configuration entry return failed")
    previous = sys.getprofile()
    with pytest.MonkeyPatch.context() as monkeypatch:
        try:
            generator = native_tests.native_host.__wrapped__(
                tmp_path, monkeypatch, SimpleNamespace(param=None)
            )
            h = await anext(generator)
            registry = h.runtime._workflow_preparations
            before = h.db.automatic_work.snapshot(h.phase.attempt.chain_id)
            count = h.runtime._first_send_sources
            issuer, host = asyncio.current_task(), threading.current_thread()

            def observe(frame, event, _result):
                if (
                    frame.f_code
                    is not contextlib._GeneratorContextManager.__enter__.__code__
                    or event != "return"
                ):
                    return
                parent = frame.f_back
                if (
                    parent is None
                    or parent.f_code
                    is not _PreparedWorkflowReply._retain_configuration.__code__
                ):
                    return
                if dis.opname[frame.f_code.co_code[frame.f_lasti]] not in {
                    "RETURN_VALUE",
                    "RETURN_CONST",
                }:
                    return
                inject = False
                try:
                    p, scope = parent.f_locals["self"], frame.f_locals["self"]
                    assert p.phase is h.phase and p.runtime is h.runtime
                    assert (
                        threading.current_thread() is host
                        and asyncio.current_task() is issuer
                    )
                    assert p._configuration_transition[:3] == (p, scope, registry)
                    assert p._source_scope is None and p not in registry
                    assert scope.gen.gi_frame.f_locals["self"] is h.runtime
                    assert h.runtime._first_send_sources == count + 1
                    saved.update(
                        prepared=p, scope=scope, transition=p._configuration_transition
                    )
                    inject = True
                except BaseException as error:  # noqa: BLE001 - passive observation failures do not enter native code.
                    errors.append(error)
                if inject:
                    sys.setprofile(previous)
                    raise sentinel

            sys.setprofile(observe)
            with pytest.raises(RuntimeError) as caught:
                await h.bridge._prepare_workflow_reply(
                    phase=h.phase,
                    task_id="task-1",
                    binding=h.binding,
                    resolution=h.selected,
                )
            sys.setprofile(previous)
            assert errors == [] and saved
            prepared, scope = saved["prepared"], saved["scope"]
            assert caught.value.__cause__ is sentinel
            assert prepared._configuration_transition_error is sentinel
            assert prepared._configuration_transition is saved["transition"]
            assert prepared in registry and h.runtime._workflow_preparations is registry
            assert prepared._source_scope is None and prepared._finite_reads == []
            assert (
                prepared.source
                is prepared.authorization
                is prepared.native
                is prepared._finite
                is None
            )
            assert not prepared._closed and not h.phase._task_scopes
            for _ in range(2):
                with pytest.raises(RuntimeError) as repeated:
                    await prepared.close()
                assert repeated.value.__cause__ is sentinel
                assert prepared._finite_reads == [] and prepared in registry
                assert prepared._configuration_transition is saved["transition"]
                assert scope.gen.gi_frame is not None
            after = h.db.automatic_work.snapshot(before.chain_id)
            assert after.used == before.used and after.reserved == before.reserved
            assert h.calls == [] and h.db.list_runs(h.conversation_id) == []
            print(
                "G_INITIAL_CONFIGURATION_ENTRY_RETAINED; no fixture/creator drain",
                flush=True,
            )
            os._exit(17)
        except BaseException as error:
            primary = error
            raise
        finally:
            sys.setprofile(previous)
            if generator is not None:
                try:
                    await generator.aclose()
                except BaseException as error:
                    if primary is None:
                        raise
                    primary.add_note(
                        "Original fixture cleanup also failed: " + type(error).__name__
                    )
                    primary.cleanup_error = error


async def test_original_unreturned_initial_configuration_entry_keeps_existing_root(
    tmp_path,
):
    import subprocess
    from pathlib import Path

    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    route = "configuration-entry"
    script = Path(__file__).with_name("workflow_coordinator_probe.py").read_text()
    with pytest.raises(
        AssertionError, match="G_INITIAL_CONFIGURATION_ENTRY_RETAINED"
    ) as caught:
        await asyncio.to_thread(
            _run, tmp_path, route, "retained", script=script, timeout=60
        )
    frames = []
    tb = caught.value.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code is _run.__code__:
            frames.append(tb.tb_frame)
        tb = tb.tb_next
    assert len(frames) == 1
    result = frames[0].f_locals["result"]
    assert type(result) is subprocess.CompletedProcess and result.returncode == 17
    assert result.args[-2:] == [route, "retained"]


async def normal_terminal_configuration_exit_journey(tmp_path, _unused):
    """Normal earlier reads cannot certify a failed terminal configuration exit."""
    import os
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.Chat.console_workflow_preparation import _PreparedWorkflowReply

    generator = primary = None
    old_trace = sys.gettrace()
    saved, errors = {}, []
    with pytest.MonkeyPatch.context() as monkeypatch:
        try:
            generator = native_tests.native_host.__wrapped__(
                tmp_path, monkeypatch, SimpleNamespace(param=None)
            )
            h = await anext(generator)
            prepared = await h.bridge._prepare_workflow_reply(
                phase=h.phase,
                task_id="task-1",
                binding=h.binding,
                resolution=h.selected,
            )
            scope, registry = prepared._source_scope, h.runtime._workflow_preparations
            auth, source, native = (
                prepared.authorization,
                prepared.source,
                prepared.native,
            )
            scope_frame = scope.gen.gi_frame
            assert (
                scope_frame.f_code
                is ConsoleRuntime._first_send_source_scope.__wrapped__.__code__
            )
            assert all(read.before_issue is None for read in prepared._finite_reads)
            before = h.db.automatic_work.snapshot(auth.claim.chain_id)
            sentinel = RuntimeError("original normal-P terminal scope exit failed")
            issuer, host = asyncio.current_task(), threading.current_thread()

            def trace(frame, event, _arg):
                if (
                    frame is not scope_frame
                    or event != "return"
                    or dis.opname[frame.f_code.co_code[frame.f_lasti]]
                    not in {"RETURN_VALUE", "RETURN_CONST"}
                ):
                    return trace
                inject = False
                try:
                    assert (
                        threading.current_thread() is host
                        and asyncio.current_task() is issuer
                    )
                    context_exit = frame.f_back
                    assert (
                        context_exit.f_code
                        is contextlib._GeneratorContextManager.__exit__.__code__
                    )
                    assert context_exit.f_locals["self"] is scope
                    release_frame = context_exit.f_back
                    assert (
                        release_frame.f_code
                        is _PreparedWorkflowReply._release_configuration.__code__
                    )
                    assert release_frame.f_locals["self"] is prepared
                    assert release_frame.f_locals["scope"] is scope
                    assert release_frame.f_locals["registry"] is registry
                    assert (
                        release_frame.f_back.f_code
                        is _PreparedWorkflowReply.close.__code__
                    )
                    closing = prepared._finite_reads[-1]
                    assert (
                        closing.positive
                        and closing.wrapper_error is closing.body_error is None
                    )
                    actual = prepared._finite_read_return(closing)
                    assert actual is closing.proof[1].result()
                    assert closing.proof[2].done() and closing.proof[2].result() is None
                    saved.update(closing=closing, actual=actual, injected=True)
                    inject = True
                except BaseException as error:  # noqa: BLE001 - passive facts do not alter native cleanup.
                    errors.append(error)
                if inject:
                    sys.settrace(old_trace)
                    raise sentinel
                return trace

            sys.settrace(trace)
            with pytest.raises(RuntimeError) as first:
                await prepared.close()
            sys.settrace(old_trace)
            assert first.value is sentinel and saved.get("injected") and errors == []
            records = tuple(prepared._finite_reads)
            assert prepared._closed and prepared._source_scope is None
            assert (
                prepared in registry and prepared._terminal_configuration_return is None
            )
            assert all(read.before_issue is None for read in records)
            for _ in range(2):
                with pytest.raises(RuntimeError):
                    await prepared.close()
                assert tuple(prepared._finite_reads) == records
                assert prepared._finite_read_return(saved["closing"]) is saved["actual"]
                assert prepared.source is source and prepared.native is native
                assert (
                    prepared.authorization is auth
                    and auth._bridge_preparation[0] is prepared
                )
                assert (
                    prepared in registry
                    and prepared._terminal_configuration_return is None
                )
            after = h.db.automatic_work.snapshot(before.chain_id)
            assert after.used == before.used and after.reserved == before.reserved
            assert h.calls == [] and h.db.list_runs(h.conversation_id) == []
            print(
                "G_NORMAL_TERMINAL_EXIT_RETAINED; no fixture/creator drain", flush=True
            )
            os._exit(17)
        except BaseException as error:
            primary = error
            raise
        finally:
            sys.settrace(old_trace)
            if generator is not None:
                try:
                    await generator.aclose()
                except BaseException as error:
                    if primary is None:
                        raise
                    primary.add_note(
                        "Original fixture cleanup also failed: " + type(error).__name__
                    )
                    primary.cleanup_error = error


async def test_normal_preparation_terminal_exit_failure_never_becomes_inert(tmp_path):
    import subprocess
    from pathlib import Path

    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    route = "normal-terminal-config-exit"
    script = Path(__file__).with_name("workflow_coordinator_probe.py").read_text()
    with pytest.raises(
        AssertionError, match="G_NORMAL_TERMINAL_EXIT_RETAINED"
    ) as caught:
        await asyncio.to_thread(
            _run, tmp_path, route, "retained", script=script, timeout=60
        )
    frames = []
    tb = caught.value.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code is _run.__code__:
            frames.append(tb.tb_frame)
        tb = tb.tb_next
    assert len(frames) == 1
    result = frames[0].f_locals["result"]
    assert type(result) is subprocess.CompletedProcess
    assert result.returncode == 17 and result.args[-2:] == [route, "retained"]


async def test_consent_first_validator_preserves_original_error_and_unused_close(
    native_host,
):
    import inspect
    from types import CoroutineType, FunctionType, MethodType

    from tldw_chatbook.Chat.console_preparation_reads import (
        ConsolePreparationRead,
        _HookSourceBinding,
        run_preparation_read,
    )
    from tldw_chatbook.Chat.console_project_instructions import (
        ProjectInstructionControlState,
    )
    from tldw_chatbook.Chat.console_workflow_preparation import (
        _finish_native_preparation,
        _PlanningPreparationRead,
        _PreparedWorkflowReply,
    )

    h = native_host
    (h.root / "AGENTS.md").write_text("Original native consent instruction sentinel\n")
    h.session.project_instruction_state = ProjectInstructionControlState(
        project_instructions_enabled=True,
        working_folder_binding_id=h.binding.binding_id,
        working_folder_locator_fingerprint=h.binding.locator_fingerprint,
    )
    cancel, release = threading.Event(), threading.Event()
    ready = asyncio.Event()
    confirmation_calls, errors, saved = [], [], {}
    host, loop = threading.current_thread(), asyncio.get_running_loop()
    previous, previous_thread = sys.getprofile(), threading.getprofile()
    before = h.db.automatic_work.snapshot(h.phase.attempt.chain_id)
    invoke_code = next(
        code
        for code in _PreparedWorkflowReply._read.__code__.co_consts
        if inspect.iscode(code) and code.co_name == "invoke"
    )
    release_code = _PreparedWorkflowReply._release_configuration.__code__
    worker_invoke_code = next(
        code
        for code in run_preparation_read.__code__.co_consts
        if inspect.iscode(code) and code.co_name == "invoke"
    )
    preparation_task = primary = None

    def diagnostic_error(label, error):
        try:
            locations, traceback = [], error.__traceback__
            for _ in range(32):
                if traceback is None:
                    break
                code = traceback.tb_frame.f_code
                locations.append((code.co_filename, code.co_name, traceback.tb_lineno))
                traceback = traceback.tb_next
            print(label, type(error).__name__, locations, error is primary, flush=True)
        except BaseException:  # noqa: BLE001,S110 - diagnostics cannot replace the original failure.
            pass

    def confirmation(snapshot):
        confirmation_calls.append(snapshot)
        return "proceed"

    def rows(claim):
        with h.db.connection() as conn:
            task = conn.execute(
                "SELECT t.id,t.run_id,t.status,c.child_reservation_id,p.native_attempt_id,c.launch_run_id "
                "FROM workflow_task_attempts t JOIN workflow_task_claims c ON c.task_attempt_id=t.id "
                "JOIN workflow_phase_attempts p ON p.id=c.phase_attempt_id WHERE t.id=?",
                (claim.id,),
            ).fetchone()
            reservation = conn.execute(
                "SELECT id,chain_id,owner_id,kind,amount,state FROM automatic_work_reservations WHERE id=?",
                (claim.reservation_id,),
            ).fetchone()
        return tuple(task), tuple(reservation)

    def observe(frame, event, result):
        if event != "return" or (
            frame.f_code is not invoke_code and frame.f_code is not release_code
        ):
            return
        if dis.opname[frame.f_code.co_code[frame.f_lasti]] not in {
            "RETURN_VALUE",
            "RETURN_CONST",
        }:
            return
        if frame.f_code is invoke_code:
            prepared, callback = frame.f_locals["self"], frame.f_locals["callback"]
            if (
                prepared.phase is not h.phase
                or not inspect.ismethod(callback)
                or callback.__self__ is not prepared
                or callback.__func__ is not _PreparedWorkflowReply._enter
            ):
                return
            try:
                record = frame.f_locals["record"]
                assert record.issuing is preparation_task and record.scopes_returned
                assert type(result) is tuple and len(result) == 10
                assert all(
                    a is b
                    for a, b in zip(
                        result[:8],
                        (
                            record,
                            prepared,
                            preparation_task,
                            record.callback,
                            record.proof,
                            record.owner,
                            record.claim,
                            record.operation,
                        ),
                    )
                )
                assert result[9] is None and prepared._finite is None
                assert (
                    not saved
                    and prepared.authorization is h.phase._task_scopes["task-1"]
                )
                saved.update(
                    prepared=prepared,
                    auth=prepared.authorization,
                    claim=prepared.authorization.claim,
                    entry_record=record,
                    entry_return=result,
                )
                loop.call_soon_threadsafe(ready.set)
                if not release.wait(5):
                    errors.append(
                        TimeoutError("original pre-consent receiving hold expired")
                    )
            except BaseException as error:  # noqa: BLE001 - observer errors never enter native callbacks.
                errors.append(error)
            return
        parent = frame.f_back
        if parent is None or parent.f_code is not _finish_native_preparation.__code__:
            return
        try:
            prepared = frame.f_locals["self"]
            assert (
                prepared is saved["prepared"]
                and parent.f_locals["prepared"] is prepared
            )
            assert (
                threading.current_thread() is host
                and asyncio.current_task() is preparation_task
            )
            scope, registry = frame.f_locals["scope"], frame.f_locals["registry"]
            assert type(result) is tuple and all(
                a is b for a, b in zip(result, (prepared, scope, registry))
            )
            assert scope.gen.gi_frame is None and prepared._source_scope is None
            assert (
                prepared not in registry
                and prepared.authorization._bridge_preparation[0] is prepared
            )
            saved.update(returned=result, prefix=tuple(prepared._finite_reads))
            threading.setprofile_all_threads(previous_thread)
            sys.setprofile(previous)
            cancel.set()
        except BaseException as error:  # noqa: BLE001 - native callback records observation only.
            errors.append(error)

    threading.setprofile_all_threads(observe)
    try:
        preparation_task = asyncio.create_task(
            h.bridge._prepare_workflow_reply(
                phase=h.phase,
                task_id="task-1",
                binding=h.binding,
                resolution=h.selected,
                should_cancel=cancel.is_set,
                confirm_project_instruction_dispatch=confirmation,
            )
        )
        await asyncio.wait_for(ready.wait(), 10)
        assert errors == []
        claim, auth, prepared = saved["claim"], saved["auth"], saved["prepared"]
        assert auth.claim is claim and auth.phase is h.phase
        assert (
            auth._bridge_preparation[0] is prepared and prepared.authorization is auth
        )
        original_rows = rows(claim)
        assert original_rows == (
            (
                claim.id,
                None,
                "accepted",
                claim.reservation_id,
                claim.phase_attempt_id,
                claim.launch_run_id,
            ),
            (
                claim.reservation_id,
                claim.chain_id,
                claim.owner_id,
                "child_launch",
                1,
                "reserved",
            ),
        )
        preconsent = h.db.automatic_work.snapshot(claim.chain_id)
        expected_reserved = dict(before.reserved)
        assert expected_reserved["child_launch"] == 0
        expected_reserved["child_launch"] += 1
        assert (
            preconsent.used == before.used and preconsent.reserved == expected_reserved
        )
        release.set()
        with pytest.raises(BaseException) as caught:
            await preparation_task
        assert errors == [] and confirmation_calls == []
        failed = prepared._finite_reads[len(saved["prefix"])]
        assert failed.before_issue is not None and failed.issue[12] is None
        assert isinstance(failed.wrapper_error, PermissionError)
        assert caught.value is failed.wrapper_error
        assert failed.proof is failed.owner is failed.claim is failed.operation is None
        assert failed.issue[0].cr_frame is None and not failed.issue[0].cr_running
        assert prepared._closed and prepared._source_scope is None
        terminal, closing, records = prepared._terminal_configuration_return
        assert terminal is saved["returned"] and records[-1] is closing
        assert closing.positive and closing.wrapper_error is closing.body_error is None
        actual = prepared._finite_read_return(closing)
        assert actual is closing.proof[1].result()
        assert closing.proof[2].done() and closing.proof[2].result() is None
        assert prepared not in h.runtime._workflow_preparations
        after_first = h.db.automatic_work.snapshot(claim.chain_id)
        assert rows(claim) == original_rows
        assert (
            after_first.used == preconsent.used
            and after_first.reserved == preconsent.reserved
        )
        await prepared.close()
        assert tuple(prepared._finite_reads) == records
        assert h.calls == [] and h.db.list_runs(h.conversation_id) == []
        after = h.db.automatic_work.snapshot(claim.chain_id)
        assert after.used == after_first.used and after.reserved == after_first.reserved
        assert rows(claim) == original_rows
    except BaseException as error:
        primary = error
        try:
            task = preparation_task
            if type(task) is asyncio.Task:
                print(
                    "G_CONSENT_TASK_BOUNDARY",
                    task.done(),
                    task.cancelled(),
                    "entry_record" in saved,
                    "entry_return" in saved,
                    "returned" in saved,
                    len(errors),
                    flush=True,
                )
                chain = []
                current = task.get_coro() if not task.done() else None
                for _ in range(16):
                    if type(current) is not CoroutineType or any(
                        current is item for item in chain
                    ):
                        break
                    chain.append(current)
                    frame, code = current.cr_frame, current.cr_code
                    print(
                        "G_CONSENT_TASK_AWAIT",
                        code.co_filename,
                        code.co_name,
                        frame.f_lineno if frame is not None else None,
                        frame is None,
                        current.cr_running,
                        flush=True,
                    )
                    current = current.cr_await
                worker_snapshot_taken = False
                for coroutine in chain:
                    frame = coroutine.cr_frame
                    if (
                        frame is None
                        or frame.f_code is not _PreparedWorkflowReply._read.__code__
                    ):
                        continue
                    p, record = frame.f_locals.get("self"), frame.f_locals.get("record")
                    if (
                        type(p) is not _PreparedWorkflowReply
                        or type(record) is not _PlanningPreparationRead
                        or record.prepared is not p
                        or record.issuing is not task
                        or p.phase is not h.phase
                        or p.runtime is not h.runtime
                        or not any(record is item for item in p._finite_reads)
                    ):
                        continue
                    callback = frame.f_locals.get("callback")
                    function = (
                        callback.__func__ if type(callback) is MethodType else callback
                    )
                    callback_name = (
                        function.__code__.co_name
                        if type(function) is FunctionType
                        else "unavailable"
                    )
                    print(
                        "G_CONSENT_ORIGINAL_READ_STAGE",
                        callback_name,
                        record.scopes_returned,
                        record.positive,
                        type(record.body_error).__name__,
                        type(record.wrapper_error).__name__,
                        record.issue is not None,
                        record.proof is not None,
                        flush=True,
                    )
                    issue = record.issue
                    child = coroutine.cr_await
                    if (
                        type(issue) is not tuple
                        or len(issue) != 15
                        or type(child) is not CoroutineType
                        or issue[0] is not child
                        or child.cr_code is not run_preparation_read.__code__
                        or child.cr_frame is None
                        or issue[1] is not child.cr_frame
                    ):
                        continue
                    read = child.cr_frame.f_locals.get("read")
                    if (
                        type(read) is not ConsolePreparationRead
                        or read.source is not p
                        or read.task is not task
                        or read.callback is not record.callback
                        or read.creator is not p.controller
                    ):
                        continue
                    for label, future in (
                        ("producer", read._producer),
                        ("retired", read.retired),
                    ):
                        facts = (
                            (future.done(), future.cancelled())
                            if type(future) is asyncio.Future
                            else "unavailable"
                        )
                        print(
                            "G_CONSENT_ORIGINAL_READ_FUTURE", label, facts, flush=True
                        )
                    proof = record.proof
                    worker = child.cr_frame.f_locals.get("invoke")
                    if (
                        worker_snapshot_taken
                        or type(proof) is not tuple
                        or len(proof) != 4
                        or any(
                            a is not b
                            for a, b in zip(
                                proof,
                                (read, read._producer, read.retired, record.callback),
                            )
                        )
                        or type(worker) is not FunctionType
                        or worker.__code__ is not worker_invoke_code
                    ):
                        print("G_CONSENT_WORKER_SNAPSHOT", "unavailable", flush=True)
                        continue
                    closed = dict(
                        zip(
                            worker.__code__.co_freevars,
                            (cell.cell_contents for cell in worker.__closure__ or ()),
                        )
                    )
                    if (
                        closed.get("read") is not read
                        or closed.get("callback") is not record.callback
                    ):
                        print("G_CONSENT_WORKER_SNAPSHOT", "unavailable", flush=True)
                        continue
                    worker_snapshot_taken = True
                    matches = []
                    for ident, top in sys._current_frames().items():
                        frames, cursor = [], top
                        for _ in range(64):
                            if cursor is None:
                                break
                            frames.append(cursor)
                            if (
                                cursor.f_code is worker_invoke_code
                                and cursor.f_locals.get("read") is read
                                and cursor.f_locals.get("callback") is record.callback
                            ):
                                binding = cursor.f_locals.get("binding")
                                if (
                                    type(binding) is _HookSourceBinding
                                    and binding.read is read
                                    and type(binding.thread) is threading.Thread
                                    and binding.thread.ident == ident
                                    and binding.task is None
                                ):
                                    matches.append((tuple(frames), binding.active))
                                break
                            cursor = cursor.f_back
                    print("G_CONSENT_WORKER_SNAPSHOT_MATCHES", len(matches), flush=True)
                    for frames, active in matches:
                        print(
                            "G_CONSENT_WORKER_STAGE",
                            active,
                            any(
                                item.f_code is invoke_code
                                and item.f_locals.get("self") is p
                                and item.f_locals.get("record") is record
                                and item.f_locals.get("issuing") is task
                                for item in frames
                            ),
                            any(
                                item.f_code is _PreparedWorkflowReply._enter.__code__
                                and item.f_locals.get("self") is p
                                for item in frames
                            ),
                            [
                                (
                                    item.f_code.co_filename,
                                    item.f_code.co_name,
                                    item.f_lineno,
                                )
                                for item in frames[:32]
                            ],
                            flush=True,
                        )
            else:
                print("G_CONSENT_TASK_BOUNDARY", "unavailable", flush=True)
        except BaseException as diagnostic:  # noqa: BLE001 - diagnostic failure remains separate.
            diagnostic_error("G_CONSENT_DIAGNOSTIC_ERROR", diagnostic)
        for observed in errors:
            locations = []
            traceback = observed.__traceback__
            while traceback is not None:
                locations.append(
                    (
                        traceback.tb_frame.f_code.co_filename,
                        traceback.tb_frame.f_code.co_name,
                        traceback.tb_lineno,
                    )
                )
                traceback = traceback.tb_next
            print(
                "G_CONSENT_OBSERVER_ERROR",
                type(observed).__name__,
                locations,
                flush=True,
            )
        raise
    finally:
        release.set()
        threading.setprofile_all_threads(previous_thread)
        sys.setprofile(previous)
        if preparation_task is not None and (
            primary is not None or not preparation_task.done()
        ):
            cleanup_stage = "original-task-await"
            try:
                pending = await preparation_task
                try:
                    print(
                        "G_CONSENT_CLEANUP_TASK_RETURN",
                        type(pending) is _PreparedWorkflowReply,
                        pending is saved["prepared"]
                        if "prepared" in saved
                        else "unavailable",
                        flush=True,
                    )
                except BaseException as diagnostic:  # noqa: BLE001 - diagnostic failure remains separate.
                    diagnostic_error("G_CONSENT_DIAGNOSTIC_ERROR", diagnostic)
                cleanup_stage = "original-returned-preparation-close"
                await pending.close()
                try:
                    print("G_CONSENT_CLEANUP_CLOSE_RETURN", flush=True)
                except BaseException as diagnostic:  # noqa: BLE001 - diagnostic failure remains separate.
                    diagnostic_error("G_CONSENT_DIAGNOSTIC_ERROR", diagnostic)
            except BaseException as error:
                diagnostic_error("G_CONSENT_CLEANUP_ERROR " + cleanup_stage, error)
                if primary is None:
                    raise
                primary.add_note(
                    "Original preparation cleanup also failed: " + type(error).__name__
                )
                primary.cleanup_error = error


@pytest.mark.parametrize("contention", [False, True])
async def test_original_direct_scope_excludes_wave_close_and_duplicate_finish(
    native_host, contention
):
    h = native_host
    prepared = await h.bridge._prepare_workflow_reply(
        phase=h.phase, task_id="task-1", binding=h.binding, resolution=h.selected
    )
    auth = prepared.authorization
    entered, release = threading.Event(), threading.Event()
    saved, errors, order = {}, [], []
    competitor = claim = None
    competitor_returns = []
    if contention:
        competitor = h.runtime.execution_capacity.begin_execution(
            origin=WorkOrigin.MANUAL, conversation_id="original-contender"
        )
        competitor.on_drained(competitor_returns.append)
        claim = h.runtime.worktree_activity.try_acquire(
            owner=competitor,
            footprint=WorktreeFootprint(
                writes=(WorktreeScope(h.root, True, capture_directory_chain(h.root)),)
            ),
        )
        assert claim is not None

    core_code = _core_operation.__wrapped__.__code__
    inner_code = next(
        code
        for code in contextlib.ContextDecorator.__call__.__code__.co_consts
        if isinstance(code, type(core_code)) and code.co_name == "inner"
    )
    codes = {
        _WORKFLOW_LAUNCH_CODE,
        core_code,
        AgentService._workflow_launch_result.__code__,
        OwnedOperation.finish.__code__,
    }

    def observe(frame, event, result):
        if (
            frame.f_code not in codes
            or event != "return"
            or dis.opname[frame.f_code.co_code[frame.f_lasti]]
            not in {"RETURN_VALUE", "RETURN_CONST"}
        ):
            return
        try:
            if frame.f_code is _WORKFLOW_LAUNCH_CODE:
                if (
                    type(result) is not _WorkflowLaunchReturn
                    or result.entry.preparation is not prepared
                ):
                    return
                assert not saved
                saved.update(
                    result=result, entry=result.entry, operation=result.operation
                )
                order.append("body")
            elif frame.f_code is core_code and saved and not entered.is_set():
                exit_frame = frame.f_back
                wrapper = exit_frame.f_back if exit_frame is not None else None
                if wrapper is None or wrapper.f_code is not inner_code:
                    return
                function = wrapper.f_locals.get("func")
                if function is None or function.__code__ is not _WORKFLOW_LAUNCH_CODE:
                    return
                cells = dict(
                    zip(
                        function.__code__.co_freevars,
                        (cell.cell_contents for cell in function.__closure__ or ()),
                    )
                )
                if cells.get("entry") is not saved["entry"]:
                    return
                assert frame.f_locals["repository"] is prepared.native.service.db
                assert exit_frame.f_locals["self"].gen.gi_frame is frame
                saved.update(core=frame, wrapper=wrapper, function=function)
                order.append("core-return-held")
                entered.set()
                if not release.wait(5):
                    errors.append(
                        AssertionError("original core return hold exceeded 5s")
                    )
            elif frame.f_code is AgentService._workflow_launch_result.__code__:
                if frame.f_locals.get("authorization") is auth:
                    if result.duplicate:
                        assert result.operation is None
                        assert result.receipt is saved["result"].receipt
                        saved["duplicate"] = result
                        return
                    assert result is saved["result"]
                    order.append("own-return")
            elif frame.f_code is OwnedOperation.finish.__code__ and saved:
                if frame.f_locals.get("self") is saved["operation"]:
                    order.append("operation-return")
        except BaseException as error:  # noqa: BLE001 - observation must not replace native errors.
            errors.append(error)
            entered.set()

    previous_thread_profile, previous_profile = threading.getprofile(), sys.getprofile()
    threading.setprofile_all_threads(observe)
    launch = asyncio.create_task(native_tests._launch_native_for_test(h, prepared))
    receipt = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        if errors:
            raise errors[0]
        result = saved["result"]
        assert result.entry is saved["entry"]
        assert prepared._wave is None and h.runtime._workflow_waves == {}
        if contention:
            assert (
                result.owner is None
                and result.operation is None
                and result.receipt is None
            )
            assert result.wait and isinstance(result.error, _WorkflowChildWait)
            assert auth._native_launch == [None]
            with pytest.raises(PermissionError, match="already claimed"):
                await native_tests._launch_native_for_test(h, prepared)
        else:
            operation = result.operation
            assert operation is prepared._launch_operation
            assert (
                operation.owner is result.owner
                and operation in result.owner._operations
            )
            assert (
                await native_tests._launch_native_for_test(h, prepared)
                is result.receipt
            )
            assert prepared._launch_operation is operation
            assert operation in result.owner._operations
            assert "operation-return" not in order
            assert saved["duplicate"].operation is None
        with pytest.raises(PermissionError, match="preparation already claimed"):
            await run_workflow_wave(h.phase, (prepared,))
        assert prepared._wave is None and h.runtime._workflow_waves == {}
        with pytest.raises(PermissionError, match="launch return has not completed"):
            await prepared.close()
        assert not prepared._closed and not prepared._closing
        assert not launch.done()
    finally:
        release.set()
        try:
            if contention:
                with pytest.raises(_WorkflowChildWait) as refused:
                    await launch
                assert refused.value is saved["result"].error
            else:
                receipt = await launch
        finally:
            threading.setprofile_all_threads(previous_thread_profile)
            sys.setprofile(previous_profile)
            if competitor is not None:
                competitor.finish_root()
                claim.settle(durable=True)
                assert competitor_returns == [True] and not competitor.is_admitted
    if errors:
        raise errors[0]
    assert order[:2] == ["body", "core-return-held"]
    assert "own-return" in order
    if contention:
        assert auth._native_launch == [] and prepared._launch_operation is None
        assert h.calls == []
        receipt = await native_tests._launch_native_for_test(h, prepared)
    else:
        assert order.index("own-return") < order.index("operation-return")
        assert prepared._launch_operation is None and receipt is saved["result"].receipt
    await native_tests._settle_native_for_test(h, prepared, receipt)
    assert h.runtime.execution_capacity.snapshot().executions == ()


async def test_cancelled_caller_receives_original_positive_wait_unlink(native_host):
    h = native_host
    prepared = await h.bridge._prepare_workflow_reply(
        phase=h.phase, task_id="task-1", binding=h.binding, resolution=h.selected
    )
    capacity = h.runtime.execution_capacity
    capacity._refresh_limits()
    allowance = capacity.max_child_executions - capacity.reserved_manual_children
    assert 0 < allowance <= capacity.max_child_executions
    competitors, returns, saved, errors = [], [], {}, []
    issuing = None
    caught = []
    message = "original-positive-wait-cancellation"

    def trace(frame, event, value):
        if frame.f_code is not run_workflow_wave.__code__:
            return None
        try:
            if event == "exception" and isinstance(value[1], asyncio.CancelledError):
                assert asyncio.current_task() is issuing
                assert frame is issuing.get_coro().cr_frame
                caught.append(value[1])
        except BaseException as error:  # noqa: BLE001 - observation cannot replace native outcome.
            errors.append(error)
        return trace

    def observe(frame, event, result):
        if (
            event != "return"
            or frame.f_code is not _WorkflowWave._release.__code__
            or dis.opname[frame.f_code.co_code[frame.f_lasti]]
            not in {"RETURN_VALUE", "RETURN_CONST"}
        ):
            return
        wave = frame.f_locals["self"]
        if wave.prepared != (prepared,):
            return
        try:
            assert type(result) is tuple and len(result) == 12 and wave.closed
            assert all(
                first is second
                for first, second in zip(
                    result[:9],
                    (
                        wave,
                        wave.driver,
                        wave._driver_coroutine,
                        wave.source,
                        wave.reservation,
                        wave.fence,
                        wave.baseline_return,
                        wave.ending_return,
                        wave.prepared,
                    ),
                )
            )
            assert result[9:] == ((), (), ())
            assert wave not in h.runtime._workflow_waves and prepared._wave is wave
            assert wave.driver is asyncio.current_task()
            assert (
                not wave.driver.done()
                and wave.reservation is None
                and wave.source is None
            )
            assert tuple(wave.unlaunched) == (prepared,)
            assert not wave.receipts and not wave.unproven_launches
            saved.update(wave=wave, driver=wave.driver, release_return=result)
            assert issuing.cancel(message)
        except BaseException as error:  # noqa: BLE001 - preserve original native return.
            errors.append(error)

    previous_thread_profile, previous_profile = threading.getprofile(), sys.getprofile()
    previous_trace = sys.gettrace()
    try:
        for _ in range(allowance):
            owner = capacity.begin_execution(
                origin=WorkOrigin.AUTOMATIC,
                conversation_id="original-competitor",
                child=True,
            )
            competitors.append(owner)
            owner.on_drained(
                lambda proven, owner=owner: returns.append((owner, proven))
            )
        threading.setprofile_all_threads(observe)
        sys.settrace(trace)
        issuing = asyncio.create_task(
            run_workflow_wave(h.phase, (prepared,), _retain_waits=True)
        )
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await issuing
        assert caught and cancelled.value is caught[0]
        assert cancelled.value.args == (message,)
        assert all(error is caught[0] for error in caught)
        if errors:
            raise errors[0]
        wave, driver = saved["wave"], saved["driver"]
        assert wave._release_return is saved["release_return"]
        assert driver is wave.driver and driver.done() and not driver.cancelled()
        assert driver.exception() is None and driver.result() is None
        assert prepared._wave is None and not prepared._closed and not prepared._closing
        assert wave.closed and wave not in h.runtime._workflow_waves
        assert prepared.authorization._native_launch == []
        assert prepared.authorization._native_preparation[0].threads == {}
        assert h.calls == []
    finally:
        threading.setprofile_all_threads(previous_thread_profile)
        sys.setprofile(previous_profile)
        sys.settrace(previous_trace)
        for owner in competitors:
            owner.finish_root()
        assert len(returns) == len(competitors)
        assert all(
            proven is True and not owner.is_admitted for owner, proven in returns
        )
    await prepared.close()
    record = prepared._finite_reads[-1]
    prepared._finite_read_return(record)
    assert record.positive and record.wrapper_error is None
    assert prepared._closed and prepared._source_scope is None
    assert capacity.snapshot().executions == ()


async def test_repeated_cancellation_retains_first_before_original_driver_error(
    native_host,
):
    from tldw_chatbook.Chat.console_workflow_preparation import _PreparedWorkflowReply
    from tldw_chatbook.Chat.console_workflow_wave import _WaveOpenRefusal

    h = native_host
    prepared = await h.bridge._prepare_workflow_reply(
        phase=h.phase, task_id="task-1", binding=h.binding, resolution=h.selected
    )
    capacity = h.runtime.execution_capacity
    capacity._refresh_limits()
    allowance = capacity.max_child_executions - capacity.reserved_manual_children
    assert 0 < allowance <= capacity.max_child_executions
    competitors, returns, caught, errors = [], [], [], []
    saved = {}
    entered, release = threading.Event(), threading.Event()
    first_saved, second_saved = asyncio.Event(), asyncio.Event()
    issuing = None
    finish_code = next(
        code
        for code in _PreparedWorkflowReply.close.__code__.co_consts
        if isinstance(code, type(run_workflow_wave.__code__))
        and code.co_name == "finish"
    )
    invoke_code = next(
        code
        for code in _PreparedWorkflowReply._read.__code__.co_consts
        if isinstance(code, type(finish_code)) and code.co_name == "invoke"
    )

    def trace(frame, event, value):
        if frame.f_code is not run_workflow_wave.__code__:
            return None
        try:
            assert asyncio.current_task() is issuing
            if event == "exception" and isinstance(value[1], asyncio.CancelledError):
                caught.append(value[1])
            if event == "line" and caught:
                cancellation = frame.f_locals.get("cancellation")
                if cancellation is not None:
                    assert cancellation is caught[0]
                    if len(caught) == 1:
                        first_saved.set()
                    elif len(caught) == 2:
                        second_saved.set()
        except BaseException as error:  # noqa: BLE001 - preserve original outcome.
            errors.append(error)
            first_saved.set()
            second_saved.set()
        return trace

    def observe(frame, event, result):
        if (
            event != "return"
            or frame.f_code not in {_WorkflowWave._open.__code__, finish_code}
            or dis.opname[frame.f_code.co_code[frame.f_lasti]]
            not in {"RETURN_VALUE", "RETURN_CONST"}
        ):
            return
        try:
            if frame.f_code is _WorkflowWave._open.__code__:
                wave = frame.f_locals["self"]
                if wave.prepared != (prepared,):
                    return
                assert type(result) is _WaveOpenRefusal and not result.retain
                assert result.source is None and result.footprint is None
                assert isinstance(result.error, _WorkflowChildWait)
                assert result.error.reason == "runtime_capacity"
                assert asyncio.current_task() is wave.driver
                saved.update(wave=wave, driver=wave.driver, refusal=result)
                return
            if frame.f_locals.get("self") is not prepared:
                return
            parent = frame.f_back
            assert parent.f_code is invoke_code
            callback = parent.f_locals["callback"]
            record = parent.f_locals["record"]
            assert callback.__code__ is finish_code
            assert record.prepared is prepared and record.issuing is saved["driver"]
            assert record.callback.__code__ is invoke_code
            assert prepared._wave is saved["wave"]
            assert result is None and prepared._closed and not prepared._closing
            assert prepared._native_closed and not record.positive
            assert not saved["driver"].done()
            saved.update(record=record, callback=callback, finish_frame=frame)
            entered.set()
            if not release.wait(5):
                errors.append(AssertionError("original close return hold exceeded 5s"))
        except BaseException as error:  # noqa: BLE001 - preserve original return.
            errors.append(error)
            entered.set()

    previous_thread_profile, previous_profile = threading.getprofile(), sys.getprofile()
    previous_trace = sys.gettrace()
    try:
        for _ in range(allowance):
            owner = capacity.begin_execution(
                origin=WorkOrigin.AUTOMATIC,
                conversation_id="original-competitor",
                child=True,
            )
            competitors.append(owner)
            owner.on_drained(
                lambda proven, owner=owner: returns.append((owner, proven))
            )
        threading.setprofile_all_threads(observe)
        sys.settrace(trace)
        issuing = asyncio.create_task(run_workflow_wave(h.phase, (prepared,)))
        async with asyncio.timeout(5):
            assert await asyncio.to_thread(entered.wait, 5)
            if errors:
                raise errors[0]
            assert issuing.cancel("first-original-cancellation")
            await first_saved.wait()
            assert issuing.cancel("second-original-cancellation")
            await second_saved.wait()
        if errors:
            raise errors[0]
        assert len(caught) == 2 and caught[0] is not caught[1]
        assert caught[0].args == ("first-original-cancellation",)
        assert caught[1].args == ("second-original-cancellation",)
        assert not saved["driver"].done() and not saved["record"].positive
        release.set()
        with pytest.raises(_WorkflowChildWait) as refused:
            await issuing
        assert refused.value is saved["refusal"].error
        driver, wave, record = saved["driver"], saved["wave"], saved["record"]
        assert driver.done() and not driver.cancelled()
        assert driver.exception() is refused.value
        prepared._finite_read_return(record)
        assert record.positive and record.wrapper_error is None
        assert prepared._closed and prepared._source_scope is None
        assert wave.closed and wave not in h.runtime._workflow_waves
        assert not wave.receipts and not wave.unlaunched and not wave.unproven_launches
        assert wave.source is None and wave.reservation is None
        assert prepared.authorization._native_launch == []
        assert prepared.authorization._native_preparation[0].threads == {}
        assert h.calls == []
    finally:
        release.set()
        try:
            if issuing is not None and not issuing.done():
                with pytest.raises(_WorkflowChildWait):
                    await issuing
        finally:
            threading.setprofile_all_threads(previous_thread_profile)
            sys.setprofile(previous_profile)
            sys.settrace(previous_trace)
            for owner in competitors:
                owner.finish_root()
            assert len(returns) == len(competitors)
            assert all(
                proven is True and not owner.is_admitted for owner, proven in returns
            )
    if errors:
        raise errors[0]
    assert capacity.snapshot().executions == ()


async def test_strict_constructor_failure_returns_native_owner_and_reserved_charge(
    native_host,
):
    from tldw_chatbook.Agents.agent_service import _WorkflowChildLaunchFailed

    h = native_host
    prepared = await h.bridge._prepare_workflow_reply(
        phase=h.phase, task_id="task-1", binding=h.binding, resolution=h.selected
    )
    auth = prepared.authorization
    native = auth._native_preparation[0]
    saved, errors = {}, []
    injected = RuntimeError("original constructor-entry test failure")

    def observe(frame, event, result):
        if event != "call" or frame.f_code is not threading.Thread.__init__.__code__:
            return
        parent = frame.f_back
        if parent.f_code is not AgentService._launch_prepared_child_result.__code__:
            return
        entry = parent.f_locals.get("entry")
        if entry is None or entry.preparation is not prepared:
            return
        try:
            assert (
                entry.mode == "D"
                and entry.authorization is auth
                and entry.native is native
            )
            assert parent.f_locals["thread"] is None
            assert parent.f_locals["start_entered"] is False
            assert frame.f_locals["target"] is parent.f_locals["run_child_owned"]
            saved.update(
                entry=entry,
                owner=parent.f_locals["child_owner"],
                claim=parent.f_locals["child_kwargs"]["workflow_activity_claim"],
                launch=prepared._launch_operation,
                close=prepared._close_operation,
            )
        except BaseException as error:  # noqa: BLE001 - only explicit injection changes native outcome.
            errors.append(error)
            return
        threading.setprofile_all_threads(None)
        raise injected

    from tldw_chatbook.Agents import activation as agent_activation
    from tldw_chatbook.Backup_Recovery import activation, admission_runtime

    trace_rows, trace_errors = [], []
    guard_code = admission_runtime.RecoveryAdmissionGuard.execution.__wrapped__.__code__
    scope_code = activation.execution_scope.__wrapped__.__code__
    close_code = type(prepared)._close_on_native_owner.__code__
    source_code = type(prepared.source).activate.__wrapped__.__code__
    trace_codes = {
        guard_code,
        scope_code,
        activation._source_scope_admitted.__code__,
        activation.activation_permission.__code__,
    }

    def trace(frame, event, result):
        if frame.f_code is close_code and frame.f_locals.get("self") is prepared:
            if event == "return":
                threading.settrace_all_threads(None)
                sys.settrace(None)
            return trace
        if frame.f_code not in trace_codes or not saved:
            return None
        try:
            ancestors = []
            current = frame.f_back
            while current is not None:
                ancestors.append(current)
                current = current.f_back
            closes = [
                item
                for item in ancestors
                if item.f_code is close_code
                and item.f_locals.get("self") is prepared
                and item.f_locals.get("owner") is saved["owner"]
            ]
            sources = [
                item
                for item in ancestors
                if item.f_code is source_code
                and item.f_locals.get("self") is prepared.source
            ]
            if len(closes) != 1 or len(sources) != 1:
                return trace
            if event not in {"exception", "return"}:
                return trace
            row = {
                "function": frame.f_code.co_name,
                "line": frame.f_lineno,
                "event": event,
                "same_close_owner": closes[0].f_locals["owner"] is saved["owner"],
                "same_source": sources[0].f_locals["self"] is prepared.source,
                "same_claim": prepared.authorization._native_preparation[
                    0
                ].activity_claims[0]
                is saved["claim"],
            }
            path = frame.f_locals.get("path")
            row["source_membership"] = [
                index
                for index, (_owner, value) in enumerate(prepared.source.sources)
                if path is not None and path == value
            ]
            if frame.f_code is guard_code:
                row["same_guard"] = (
                    frame.f_locals.get("self") is agent_activation._guard
                )
                state = frame.f_locals.get("active")
                row["state_id"] = id(state)
                row["current_identity"] = (
                    state is not None
                    and state.identity == admission_runtime.execution_identity()
                )
            if event == "exception":
                error_type, _error, tb = result
                row["exception_type"] = error_type.__name__
                stack = []
                while tb is not None:
                    stack.append(
                        (
                            tb.tb_frame.f_code.co_filename,
                            tb.tb_frame.f_code.co_name,
                            tb.tb_lineno,
                        )
                    )
                    tb = tb.tb_next
                row["traceback"] = stack
            else:
                row["bool_result"] = result if type(result) is bool else None
                row["allowed"] = (
                    frame.f_locals.get("allowed")
                    if frame.f_code is scope_code
                    else None
                )
            trace_rows.append(row)
        except BaseException as error:  # noqa: BLE001 - preserve original admission failure.
            trace_errors.append(type(error).__name__)
        return trace

    previous_thread_trace, previous_trace = threading.gettrace(), sys.gettrace()
    previous_thread_profile, previous_profile = threading.getprofile(), sys.getprofile()
    try:
        threading.settrace_all_threads(trace)
        threading.setprofile_all_threads(observe)
        with pytest.raises(_WorkflowChildLaunchFailed) as failed:
            await native_tests._launch_native_for_test(h, prepared)
    finally:
        threading.setprofile_all_threads(previous_thread_profile)
        sys.setprofile(previous_profile)
        threading.settrace_all_threads(previous_thread_trace)
        sys.settrace(previous_trace)
        print(
            "G_CONSTRUCTOR_EXACT_ADMISSION_TRACE", trace_rows, trace_errors, flush=True
        )
    if errors:
        raise errors[0]
    assert saved
    receipt = failed.value.receipt
    assert auth._native_launch == [receipt] and receipt.failure == "thread_start"
    assert receipt.owner is saved["owner"] and receipt.activity_claim is saved["claim"]
    assert h.db.get_run(receipt.run_id)["status"] == "error"
    with h.db.connection() as conn:
        reservation = conn.execute(
            "SELECT state FROM automatic_work_reservations WHERE id=?",
            (auth.claim.reservation_id,),
        ).fetchone()
    assert reservation[0] == "released"
    drains = []
    receipt.owner.on_drained(drains.append)
    assert drains == [True] and not receipt.owner.is_admitted
    assert saved["launch"] is not None and saved["close"] is not None
    assert prepared._launch_operation is None and prepared._close_operation is None
    assert not native.uncertain_start and not native.threads and h.calls == []
    assert native.fleet.live_count() == 0
    activity = h.runtime.worktree_activity
    with activity._lock:
        state = activity._claims[receipt.activity_claim.id]
        member = state.members[receipt.activity_claim.id]
        assert state.custody is saved["claim"]
        assert activity._members[receipt.activity_claim.id] is member
        assert member.owner is saved["owner"]
        assert member.drained is True and member.durable is False
    await native_tests._settle_native_for_test(h, prepared, receipt, status="failed")
    checkpoint = h.service._checkpoint(h.workflow.id, receipt.attempt_id)
    result = h.db.workflow_runs.read_result(h.workflow.id, receipt.attempt_id)
    assert checkpoint.run_id == result.run_id == receipt.run_id
    assert result.status == "failed"
    with activity._lock:
        assert member.drained is True and member.durable is True
        assert receipt.activity_claim.id not in activity._claims
        assert receipt.activity_claim.id not in activity._members
    assert prepared._closed and prepared._source_scope is None
    assert h.runtime.execution_capacity.snapshot().executions == ()


async def native_return_journey(tmp_path, monkeypatch):
    import os

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig
    from tldw_chatbook.Agents.agent_service import _WorkflowChildLaunchFailed
    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Workspaces.change_review_finalization import (
        ChangeReviewCompletion,
    )

    mode = sys.argv[2]
    assert mode in {"unknown-entry", "unknown-return"}
    rig = await build_workflow_rig(
        tmp_path, monkeypatch, WorkflowProvider(), change_review=True
    )
    saved, errors, target_returns = {}, [], []
    injected = RuntimeError("original native start boundary test failure")
    previous_thread_profile, previous_profile = threading.getprofile(), sys.getprofile()
    original_start = threading.Thread.start.__code__
    original_core = AgentService._launch_prepared_child_result.__code__

    def observe(frame, event, result):
        if saved and frame.f_code is saved["target"].__code__:
            if (
                event == "return"
                and dis.opname[frame.f_code.co_code[frame.f_lasti]]
                in {"RETURN_VALUE", "RETURN_CONST"}
                and threading.current_thread() is saved["thread"]
            ):
                target_returns.append((saved["thread"], frame, result))
            return
        if frame.f_code is not original_start or event not in {"call", "return"}:
            return
        parent = frame.f_back
        if parent.f_code is not original_core:
            return
        entry = parent.f_locals.get("entry")
        if entry is None or entry.mode != "W":
            return
        should_inject = False
        try:
            run = rig.coordinator._runs[rig.workflow.id]
            assert (
                entry.wave.phase is run.phase and entry.preparation.phase is run.phase
            )
            if event == "call":
                assert not saved
                thread = frame.f_locals["self"]
                target = parent.f_locals["run_child_owned"]
                child_kwargs = parent.f_locals["child_kwargs"]
                assert parent.f_locals["thread"] is thread
                assert thread._target is target
                assert parent.f_locals["start_entered"] is True
                assert parent.f_locals["native_thread_started"] is False
                assert (
                    child_kwargs["workflow_task_authorization"] is entry.authorization
                )
                saved.update(
                    thread=thread,
                    target=target,
                    child_kwargs=child_kwargs,
                    entry=entry,
                    prepared=entry.preparation,
                    native=entry.native,
                    owner=parent.f_locals["child_owner"],
                    claim=child_kwargs["workflow_activity_claim"],
                    launch=entry.preparation._launch_operation,
                    close=entry.preparation._close_operation,
                    run=run,
                    intake=run.intake,
                    token=run.token,
                    source=entry.preparation.source,
                    lifetime=entry.preparation._lifetime,
                )
                should_inject = mode == "unknown-entry"
            elif mode == "unknown-return" and dis.opname[
                frame.f_code.co_code[frame.f_lasti]
            ] in {"RETURN_VALUE", "RETURN_CONST"}:
                assert frame.f_locals["self"] is saved["thread"]
                assert parent.f_locals["entry"] is saved["entry"]
                assert (
                    result is None and parent.f_locals["native_thread_started"] is False
                )
                saved["thread"].join(5)
                assert not saved["thread"].is_alive()
                assert len(target_returns) == 1 and target_returns[0][2] is None
                saved["target_return"] = target_returns[0]
                should_inject = True
        except BaseException as error:  # noqa: BLE001 - observer error is not native failure evidence.
            errors.append(error)
            return
        if should_inject:
            threading.setprofile_all_threads(None)
            saved["injected"] = True
            raise injected

    primary_error = None
    try:
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        await asyncio.shield(rig.coordinator._runs[rig.workflow.id].driver)
        await rig.wait_for_status("awaiting_phase_review")
        decision = await rig.service.call(
            rig.service.prepare_decision, rig.workflow.id, phase_id="phase-1"
        )
        threading.setprofile_all_threads(observe)
        await rig.coordinator.approve(decision, command_id="strict-original-start")
        run = rig.coordinator._runs[rig.workflow.id]
        await asyncio.shield(run.driver)
        threading.setprofile_all_threads(previous_thread_profile)
        sys.setprofile(previous_profile)
        if errors:
            raise errors[0]
        assert saved.get("injected") and saved["run"] is run
        entry, prepared, native = saved["entry"], saved["prepared"], saved["native"]
        auth, wave, owner = entry.authorization, entry.wave, saved["owner"]
        assert len(native.uncertain_start) == 1
        held = native.uncertain_start[0]
        assert held[0] is saved["thread"] and held[1] is saved["target"]
        assert held[2] is saved["child_kwargs"] and held[3] is injected
        returned = entry.caller.f_locals["result"]
        assert type(returned) is _WorkflowLaunchReturn and returned.entry is entry
        assert returned.owner is owner and returned.operation is saved["launch"]
        assert not returned.known and isinstance(
            returned.error, _WorkflowChildLaunchFailed
        )
        receipt = returned.receipt
        assert auth._native_launch == [receipt] and receipt.failure == "unknown"
        assert receipt.owner is owner and receipt.activity_claim is saved["claim"]
        assert wave.driver.done() and wave.driver.exception() is returned.error
        assert (
            wave.unproven_launches == {prepared}
            and not wave.receipts
            and not wave.unlaunched
        )
        assert not wave.closed and not prepared._closed and prepared._wave is wave
        assert (
            prepared._launch_operation is saved["launch"]
            and saved["launch"] is not None
        )
        assert (
            prepared.source is saved["source"]
            and prepared._lifetime is saved["lifetime"]
        )
        assert native.threads == {}
        assert wave.baseline_return.result() is ChangeReviewCompletion.OBSERVED
        assert not wave.ending_return.done()
        assert owner.is_admitted
        with owner.capacity._lock:
            assert owner._cleanup_unproven and saved["launch"] in owner._operations
        with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
            state = conn.execute(
                "SELECT state FROM automatic_work_reservations WHERE id=?",
                (auth.claim.reservation_id,),
            ).fetchone()[0]
            row = rig.service.db.get_run(receipt.run_id)
        snapshot = await rig.coordinator.get(rig.workflow.id)
        assert snapshot.status == "recovery_required" and snapshot.results == ()
        assert snapshot.accounting.used["generation"] == 2
        if mode == "unknown-entry":
            assert state == "reserved" and row["status"] == "running"
            assert (
                prepared._close_operation is saved["close"]
                and saved["close"] is not None
            )
            assert [key for key, _ in rig.provider.calls] == ["planning"]
            assert snapshot.accounting.used["child_launch"] == 0
            assert snapshot.accounting.reserved["child_launch"] == 1
            assert snapshot.accounting.used["model_call"] == 1
            assert not target_returns
        else:
            assert state == "committed" and row["status"] == "done"
            assert (
                prepared._close_operation is None
                and saved["target_return"] is target_returns[0]
            )
            assert [key for key, _ in rig.provider.calls] == ["planning", "task-1"]
            assert snapshot.accounting.used["child_launch"] == 1
            assert snapshot.accounting.used["model_call"] == 2
            assert not saved["thread"].is_alive()
        for _ in range(2):
            with pytest.raises(
                RuntimeError, match="workflow_original_shutdown_unresolved"
            ):
                await rig.runtime.dispose()
            assert rig.coordinator._runs[rig.workflow.id] is run
            assert run.intake is saved["intake"] and run.token is saved["token"]
            assert (
                prepared._launch_operation is saved["launch"]
                and native.uncertain_start[0] is held
            )
            assert not rig.runtime.execution_capacity.snapshot().closed
        print(
            "G_STRICT_START_REQUIRED_CUSTODY_RETAINED",
            mode,
            "; no creator drain",
            flush=True,
        )
        os._exit(17)
    except BaseException as error:  # Original failure must survive cleanup refusal.
        primary_error = error
        raise
    finally:
        threading.setprofile_all_threads(previous_thread_profile)
        sys.setprofile(previous_profile)
        rig.provider_release()
        try:
            await rig.close()
        except (
            BaseException
        ) as cleanup_error:  # Report actual native cleanup separately.
            if primary_error is None:
                raise
            import traceback

            traceback.print_exception(cleanup_error)
            primary_error.add_note(
                "Original native cleanup also failed; exact exception printed above"
            )


@pytest.mark.parametrize("mode", ["unknown-entry", "unknown-return"])
async def test_strict_start_unknown_retains_original_custody(tmp_path, mode):
    from pathlib import Path

    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    with pytest.raises(
        AssertionError, match="G_STRICT_START_REQUIRED_CUSTODY_RETAINED"
    ):
        await asyncio.to_thread(
            _run,
            tmp_path,
            "native-return",
            mode,
            script=Path(__file__)
            .with_name("workflow_coordinator_probe.py")
            .read_text(),
            timeout=60,
        )


async def test_original_finisher_callback_keeps_launch_exclusion_after_removal(
    native_host,
):
    import time
    from concurrent.futures import Future

    from tldw_chatbook.Agents import activation
    from tldw_chatbook.Chat import console_agent_bridge as bridge_module

    h = native_host
    prepared = await h.bridge._prepare_workflow_reply(
        phase=h.phase, task_id="task-1", binding=h.binding, resolution=h.selected
    )
    auth = prepared.authorization
    own_entered, own_release = threading.Event(), threading.Event()
    callback_entered, callback_release = threading.Event(), threading.Event()
    saved, errors, target_returns, order = {}, [], [], []
    original_core = AgentService._launch_prepared_child_result.__code__
    original_result = AgentService._workflow_launch_result.__code__
    original_finish = AgentService._finish_workflow_launch.__code__
    original_start = threading.Thread.start.__code__
    consumer, consumer_errors = {}, []
    lifeline_start = bridge_module._ModelCallLifeline.start.__code__
    child_scope = (
        bridge_module._StreamingModelAdapter.child_lifeline.__wrapped__.__code__
    )
    chat_code = bridge_module._StreamingModelAdapter._chat_call_impl.__code__
    submit_code = asyncio.run_coroutine_threadsafe.__code__
    decorate_code = next(
        code
        for code in activation.async_worker_guard.__code__.co_consts
        if isinstance(code, type(chat_code)) and code.co_name == "decorate"
    )
    guarded_call = next(
        code
        for code in decorate_code.co_consts
        if isinstance(code, type(chat_code)) and code.co_name == "call"
    )
    consume_code = next(
        code
        for code in chat_code.co_consts
        if isinstance(code, type(chat_code)) and code.co_name == "_consume"
    )
    consumer_codes = {lifeline_start, submit_code, guarded_call}

    def observe_consumer(frame, event, result):
        # Diagnostic errors never become native callback/control errors.
        try:
            thread = saved.get("thread")
            normal_return = event == "return" and dis.opname[
                frame.f_code.co_code[frame.f_lasti]
            ] in {"RETURN_VALUE", "RETURN_CONST"}
            if frame.f_code is lifeline_start:
                if not normal_return or threading.current_thread() is not thread:
                    return
                parent = frame.f_back
                if parent is None or parent.f_code is not child_scope:
                    return
                lifeline = frame.f_locals["self"]
                assert lifeline is parent.f_locals["lifeline"]
                operation = lifeline._operation
                assert operation is not None and operation.owner is saved["child_owner"]
                consumer.update(
                    lifeline=lifeline,
                    adapter=parent.f_locals["self"],
                    loop=lifeline.loop,
                    driver=lifeline._thread,
                    operation=operation,
                )
            elif frame.f_code is submit_code:
                parent = frame.f_back
                if (
                    threading.current_thread() is not thread
                    or parent is None
                    or parent.f_code is not chat_code
                    or parent.f_locals.get("workflow_preparation") is not prepared
                ):
                    return
                assert parent.f_locals["self"] is consumer["adapter"]
                assert frame.f_locals["loop"] is consumer["loop"]
                if event == "call":
                    coroutine = frame.f_locals["coro"]
                    decorated = parent.f_locals["_consume"]
                    original = decorated.__wrapped__
                    assert coroutine.cr_code is decorated.__code__ is guarded_call
                    assert original.__code__ is consume_code
                    closure = dict(
                        zip(
                            original.__code__.co_freevars,
                            (cell.cell_contents for cell in original.__closure__ or ()),
                        )
                    )
                    assert closure["workflow_preparation"] is prepared
                    consumer.update(
                        coroutine=coroutine,
                        decorated=decorated,
                        original=original,
                    )
                elif normal_return:
                    assert frame.f_locals["coro"] is consumer["coroutine"]
                    assert type(result) is Future
                    consumer["future"] = result
            elif event == "call" and frame.f_locals.get("function") is consumer.get(
                "original"
            ):
                assert consumer.get("original") is not None
                assert frame.f_locals["service"] is consumer["adapter"]
                task, loop = asyncio.current_task(), asyncio.get_running_loop()
                assert type(task) is asyncio.Task
                assert task.get_coro() is consumer["coroutine"]
                assert loop is consumer["loop"]
                assert threading.current_thread() is consumer["driver"]
                if "task" in consumer:
                    assert consumer["task"] is task
                consumer["task"] = task
        except BaseException as diagnostic_error:  # noqa: BLE001 - observations cannot change native outcomes.
            consumer_errors.append(type(diagnostic_error).__name__)

    def consumer_snapshot():
        # Non-atomic metadata only. No Future condition lock/result consumption.
        row = {"diagnostic_error_types": list(consumer_errors)}
        lifeline = consumer.get("lifeline")
        if lifeline is None:
            row["association"] = "unavailable"
            return row
        operation, driver, loop = (
            consumer["operation"],
            consumer["driver"],
            consumer["loop"],
        )
        same = (
            lifeline._operation is operation
            and lifeline._thread is driver
            and lifeline.loop is loop
            and operation.owner is saved.get("child_owner")
        )
        if not same:
            row["association"] = "changed/unknown"
            return row
        row.update(
            association="same at sample",
            loop_running=loop.is_running(),
            loop_closed=loop.is_closed(),
            driver_alive=driver.is_alive(),
            cleanup_returned=lifeline._cleanup_returned,
            driver_returned=lifeline._driver_returned,
        )
        future = consumer.get("future")
        row["future_state"] = future._state if type(future) is Future else "unavailable"
        task = consumer.get("task")
        if type(task) is asyncio.Task and task.get_coro() is consumer.get("coroutine"):
            row.update(task_done=task.done(), task_cancelled=task.cancelled())
            stack, current = [], task.get_coro()
            while current is not None:
                frame = getattr(current, "cr_frame", None)
                if frame is not None:
                    stack.append(
                        (frame.f_code.co_filename, frame.f_code.co_name, frame.f_lineno)
                    )
                current = getattr(current, "cr_await", None)
            row["consumer_stack"] = stack or "unavailable"
        else:
            row["consumer_task"] = "unavailable"
        identity = driver.ident
        frame = sys._current_frames().get(identity)
        stack = []
        if driver.ident == identity and driver.is_alive():
            while frame is not None:
                stack.append(
                    (frame.f_code.co_filename, frame.f_code.co_name, frame.f_lineno)
                )
                frame = frame.f_back
        row["driver_stack"] = stack or "unavailable"
        row["association_still_same"] = (
            lifeline._operation is operation
            and lifeline._thread is driver
            and lifeline.loop is loop
            and operation.owner is saved.get("child_owner")
        )
        return row

    def callback(outcome):
        try:
            if errors:
                raise errors[0]
            assert outcome is True
            assert threading.current_thread() is saved["worker"]
            saved["callback_outcome"] = outcome
            callback_entered.set()
            if not callback_release.wait(5):
                errors.append(AssertionError("original drain callback hold exceeded5s"))
        except BaseException as error:  # noqa: BLE001 - observer errors must not alter native callbacks.
            errors.append(error)
            callback_entered.set()

    codes = {
        original_core,
        original_result,
        original_finish,
        original_start,
        callback.__code__,
    }

    def observe(frame, event, result):
        if frame.f_code not in codes and frame.f_code not in consumer_codes:
            return
        if frame.f_code in consumer_codes:
            observe_consumer(frame, event, result)
            return
        target = saved.get("target")
        if target is not None and frame.f_code is target.__code__:
            if (
                event == "return"
                and dis.opname[frame.f_code.co_code[frame.f_lasti]]
                in {"RETURN_VALUE", "RETURN_CONST"}
                and threading.current_thread() is saved["thread"]
            ):
                target_returns.append((frame, result))
            return
        if frame.f_code not in codes:
            return
        try:
            if frame.f_code is original_start and event == "call":
                parent = frame.f_back
                if parent.f_code is not original_core:
                    return
                entry = parent.f_locals.get("entry")
                if entry is None or entry.preparation is not prepared:
                    return
                assert entry.mode == "D" and entry.authorization is auth
                thread, target = (
                    frame.f_locals["self"],
                    parent.f_locals["run_child_owned"],
                )
                assert thread is parent.f_locals["thread"] and thread._target is target
                codes.add(target.__code__)
                saved.update(
                    thread=thread,
                    target=target,
                    entry=entry,
                    child_owner=parent.f_locals["child_owner"],
                )
                return
            if event != "return" or dis.opname[
                frame.f_code.co_code[frame.f_lasti]
            ] not in {"RETURN_VALUE", "RETURN_CONST"}:
                return
            if frame.f_code is original_core and saved:
                if frame.f_locals.get("entry") is not saved["entry"]:
                    return
                assert frame.f_locals["thread"] is saved["thread"]
                assert frame.f_locals["native_thread_started"] is True
                saved["native_return"] = result
            elif frame.f_code is original_result:
                if (
                    type(result) is not _WorkflowLaunchReturn
                    or result.entry.preparation is not prepared
                ):
                    return
                if result.duplicate:
                    assert (
                        result.operation is None
                        and result.receipt is saved["result"].receipt
                    )
                    saved["duplicate"] = result
                    return
                assert (
                    result.entry is saved["entry"]
                    and saved["native_return"] is not None
                )
                saved.update(result=result, worker=threading.current_thread())
                saved["thread"].join(max(0, setup_deadline - time.monotonic()))
                try:
                    assert not saved["thread"].is_alive()
                except AssertionError:
                    try:
                        thread = saved["thread"]
                        identity = thread.ident
                        frame = sys._current_frames().get(identity)
                        current = (
                            thread is saved["thread"]
                            and thread.ident == identity
                            and thread.is_alive()
                        )
                        stack = []
                        if current and frame is not None:
                            while frame is not None:
                                stack.append(
                                    (
                                        frame.f_code.co_filename,
                                        frame.f_code.co_name,
                                        frame.f_lineno,
                                    )
                                )
                                frame = frame.f_back
                        print(
                            "G_CALLBACK_JOIN_BOUNDARY",
                            {
                                "same_live_thread": current,
                                "stack": stack or "unknown",
                                "own_entered": own_entered.is_set(),
                                "own_release": own_release.is_set(),
                                "launch_done": launch.done(),
                                "launch_cancelled": launch.cancelled()
                                if launch.done()
                                else None,
                                "target_returns": len(target_returns),
                                "submitted_consumer": consumer_snapshot(),
                                "observer_error_types": [
                                    type(error).__name__ for error in errors
                                ],
                            },
                            flush=True,
                        )
                    except BaseException as diagnostic_error:  # noqa: BLE001 - retain original assertion and cleanup.
                        print(
                            "G_CALLBACK_DIAGNOSTIC_ERROR",
                            type(diagnostic_error).__name__,
                            flush=True,
                        )
                    raise
                assert len(target_returns) == 1 and target_returns[0][1] is None
                own_entered.set()
                if not own_release.wait(5):
                    errors.append(AssertionError("original own result hold exceeded5s"))
            elif frame.f_code is callback.__code__:
                assert threading.current_thread() is saved["worker"]
                order.append("callback-return")
            elif frame.f_code is original_finish and saved:
                if frame.f_locals.get("result") is saved["result"]:
                    order.append("finisher-return")
        except BaseException as error:  # noqa: BLE001 - preserve native outcome on observer failure.
            errors.append(error)
            own_entered.set()
            callback_entered.set()

    previous_thread_profile, previous_profile = threading.getprofile(), sys.getprofile()
    setup_deadline = time.monotonic() + 30
    threading.setprofile_all_threads(observe)
    launch = asyncio.create_task(native_tests._launch_native_for_test(h, prepared))
    receipt = primary = None
    try:
        assert await asyncio.to_thread(
            own_entered.wait, max(0, setup_deadline - time.monotonic())
        )
        if errors:
            raise errors[0]
        returned = saved["result"]
        owner, operation = returned.owner, returned.operation
        assert (
            returned.receipt.owner is owner and prepared._launch_operation is operation
        )
        assert prepared._close_operation is None
        with owner.capacity._lock:
            assert owner._root_finished and owner._operations == {operation}
            assert owner.capacity._owners.get(owner.execution_id) is owner
        if errors:
            raise errors[0]
        owner.on_drained(callback)
        own_release.set()
        assert await asyncio.to_thread(callback_entered.wait, 5)
        if errors:
            raise errors[0]
        assert saved["callback_outcome"] is True
        with owner.capacity._lock:
            assert owner.execution_id not in owner.capacity._owners
            assert operation not in owner._operations
        assert prepared._launch_operation is operation and not launch.done()
        assert (
            await native_tests._launch_native_for_test(h, prepared) is returned.receipt
        )
        assert saved["duplicate"].operation is None
        with pytest.raises(PermissionError, match="preparation already claimed"):
            await run_workflow_wave(h.phase, (prepared,))
        assert prepared._wave is None and h.runtime._workflow_waves == {}
        with pytest.raises(PermissionError, match="launch return has not completed"):
            await prepared.close()
        assert not prepared._closed and not prepared._closing
        assert prepared._launch_operation is operation and not launch.done()
    except BaseException as error:
        primary = error
        raise
    finally:
        own_release.set()
        callback_release.set()
        cleanup_errors = []
        try:
            try:
                receipt = await launch
            except BaseException as error:  # noqa: BLE001 - retain the original error without throwing into native observation or masking cleanup.
                cleanup_errors.append(error)
            thread = saved.get("thread")
            if thread is not None:
                try:
                    await asyncio.to_thread(thread.join, 5)
                    assert not thread.is_alive()
                except BaseException as error:  # noqa: BLE001 - retain the original error without throwing into native observation or masking cleanup.
                    cleanup_errors.append(error)
        finally:
            threading.setprofile_all_threads(previous_thread_profile)
            sys.setprofile(previous_profile)
        if cleanup_errors:
            if primary is None:
                raise cleanup_errors[0]
            primary.add_note(
                "Original callback cleanup also failed; see retained errors"
            )
            primary.cleanup_errors = tuple(cleanup_errors)
    if errors:
        raise errors[0]
    assert receipt is saved["result"].receipt
    assert order == ["callback-return", "finisher-return"]
    assert prepared._launch_operation is None and not receipt.owner.is_admitted
    await native_tests._settle_native_for_test(h, prepared, receipt)
    assert prepared._closed and h.runtime.execution_capacity.snapshot().executions == ()


async def storage_source_journey(tmp_path, _unused):
    """Retain original storage custody when receiving proof is unavailable."""
    import os
    from types import SimpleNamespace

    from tldw_chatbook.Agents import activation
    from tldw_chatbook.Backup_Recovery.admission_runtime import RecoveryAdmissionGuard

    mode = sys.argv[2]
    assert mode in {"copy", "method", "guard-exit"}
    generator = None
    primary = None
    competitor = contender_claim = None
    competitor_drains = []
    previous_thread_profile, previous_profile = threading.getprofile(), sys.getprofile()
    with pytest.MonkeyPatch.context() as monkeypatch:
        try:
            generator = native_tests.native_host.__wrapped__(
                tmp_path, monkeypatch, SimpleNamespace(param=None)
            )
            h = await anext(generator)
            prepared = await h.bridge._prepare_workflow_reply(
                phase=h.phase,
                task_id="task-1",
                binding=h.binding,
                resolution=h.selected,
            )
            auth = prepared.authorization
            native = auth._native_preparation[0]
            observation = native.source_observation
            source = prepared.source
            lifetime, source_scope = prepared._lifetime, prepared._source_scope
            assert observation is prepared._configured_source_observation
            assert observation[0] is prepared and observation[1] is source
            assert observation[2] is source.sources
            before = h.db.automatic_work.snapshot(auth.claim.chain_id)
            replacements, errors, saved = [], [], {}
            if mode == "copy":
                source.sources = tuple(list(observation[2]))  # noqa: C414 - force equal, distinct tuple.
                assert source.sources == observation[2]
                assert source.sources is not observation[2]
                expected = "original workflow storage observation changed"
            elif mode == "method":

                def replacement():
                    replacements.append(True)
                    raise AssertionError("replacement storage method executed")

                source.check_memory = replacement
                expected = "original workflow storage source changed"
            else:
                competitor = h.runtime.execution_capacity.begin_execution(
                    origin=WorkOrigin.MANUAL, conversation_id="original-guard-contender"
                )
                competitor.on_drained(competitor_drains.append)
                contender_claim = h.runtime.worktree_activity.try_acquire(
                    owner=competitor,
                    footprint=WorktreeFootprint(
                        writes=(
                            WorktreeScope(
                                h.root, True, capture_directory_chain(h.root)
                            ),
                        )
                    ),
                )
                assert contender_claim is not None
                guard_code = RecoveryAdmissionGuard.execution.__wrapped__.__code__
                result_code = AgentService._workflow_launch_result.__code__

                def observe(frame, event, result):
                    if (
                        frame.f_code is not guard_code
                        or frame.f_locals.get("self") is not activation._guard
                        or frame.f_locals.get("service") is not prepared.native.service
                        or event != "return"
                        or dis.opname[frame.f_code.co_code[frame.f_lasti]]
                        not in {"RETURN_VALUE", "RETURN_CONST"}
                    ):
                        return
                    parent = frame.f_back
                    while parent is not None and parent.f_code is not result_code:
                        parent = parent.f_back
                    if parent is None or parent.f_locals.get("entry", None) is None:
                        return
                    entry = parent.f_locals["entry"]
                    if entry.preparation is not prepared:
                        return
                    try:
                        value = parent.f_locals["result"]
                        assert (
                            type(value) is _WorkflowLaunchReturn
                            and value.entry is entry
                        )
                        assert entry.mode == "D" and entry.authorization is auth
                        assert entry.native is native and entry.wave is None
                        assert value.wait and isinstance(
                            value.error, _WorkflowChildWait
                        )
                        assert (
                            value.owner is None
                            and value.operation is None
                            and value.receipt is None
                        )
                        assert native.wait_return == [None] and auth._native_launch == [
                            None
                        ]
                        assert frame.f_locals["sources"] is observation[2]
                        assert not saved
                        saved.update(
                            entry=entry, result=value, error=value.error, frame=frame
                        )
                    except BaseException as error:  # noqa: BLE001 - fail observation without substituting outcome.
                        errors.append(error)
                        return
                    threading.setprofile_all_threads(None)
                    raise value.error

                threading.setprofile_all_threads(observe)
                expected = "workflow worktree activity busy"
            try:
                with pytest.raises(
                    PermissionError if mode != "guard-exit" else _WorkflowChildWait,
                    match=expected,
                ) as caught:
                    await native_tests._launch_native_for_test(h, prepared)
            finally:
                threading.setprofile_all_threads(previous_thread_profile)
                sys.setprofile(previous_profile)
                if competitor is not None:
                    competitor.finish_root()
                    contender_claim.settle(durable=True)
            if errors:
                raise errors[0]
            if mode == "guard-exit":
                assert saved and caught.value is saved["error"]
                assert saved["result"].error is caught.value
                assert saved["entry"].marker is auth._native_launch
                assert competitor_drains == [True] and not competitor.is_admitted
                with h.runtime.worktree_activity._lock:
                    assert contender_claim.id not in h.runtime.worktree_activity._claims
                assert auth._native_launch == [None] and native.wait_return == [None]
                close_error = "workflow launch return has not completed"
            else:
                assert auth._native_launch == [] and native.wait_return == []
                close_error = expected
            assert replacements == []
            assert native.source_observation is observation
            assert prepared._configured_source_observation is observation
            assert (
                not native.activity_claims
                and not native.threads
                and not native.uncertain_start
            )
            assert h.db.list_runs(h.conversation_id) == [] and h.calls == []
            assert (
                prepared._launch_operation is None and prepared._close_operation is None
            )
            assert h.runtime.execution_capacity.snapshot().executions == ()
            for _ in range(2):
                with pytest.raises(PermissionError, match=close_error):
                    await prepared.close()
                assert prepared in h.runtime._workflow_preparations
                assert prepared.authorization is auth and prepared.phase is h.phase
                assert prepared.source is source and prepared._lifetime is lifetime
                assert prepared._source_scope is source_scope
                assert not prepared._closed and not prepared._closing
                assert native.source_observation is observation and replacements == []
                assert h.db.list_runs(h.conversation_id) == [] and h.calls == []
                current = h.db.automatic_work.snapshot(auth.claim.chain_id)
                assert dict(current.used) == dict(before.used)
                assert dict(current.reserved) == dict(before.reserved)
            print("G_STORAGE_SOURCE_RETAINED_" + mode, flush=True)
            os._exit(17)  # Retained original preparation is not creator retirement.
        except BaseException as error:
            primary = error
            raise
        finally:
            threading.setprofile_all_threads(previous_thread_profile)
            sys.setprofile(previous_profile)
            if generator is not None:
                try:
                    await generator.aclose()
                except BaseException as error:
                    if primary is None:
                        raise
                    primary.add_note("original fixture cleanup: " + repr(error))


@pytest.mark.parametrize("mode", ["copy", "method", "guard-exit"])
async def test_original_storage_observation_and_guard_refuse_unproven_return(
    tmp_path, mode
):
    from pathlib import Path

    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    with pytest.raises(AssertionError, match="G_STORAGE_SOURCE_RETAINED_" + mode):
        _run(
            tmp_path,
            "storage-source",
            mode,
            script=Path(__file__)
            .with_name("workflow_coordinator_probe.py")
            .read_text(),
            timeout=60,
        )


async def test_known_constructor_return_keeps_callback_exclusion_until_finish(
    native_host,
):
    from tldw_chatbook.Agents.agent_service import (
        _NativeWorkflowReturn,
        _WorkflowChildLaunchFailed,
    )

    h = native_host
    prepared = await h.bridge._prepare_workflow_reply(
        phase=h.phase, task_id="task-1", binding=h.binding, resolution=h.selected
    )
    auth = prepared.authorization
    native = auth._native_preparation[0]
    source, lifetime, source_scope = (
        prepared.source,
        prepared._lifetime,
        prepared._source_scope,
    )
    original_counts = h.db.automatic_work.snapshot(auth.claim.chain_id)
    own_entered, own_release = threading.Event(), threading.Event()
    callback_entered, callback_release = threading.Event(), threading.Event()
    saved, errors, order = {}, [], []
    injected = RuntimeError("known constructor callback control")
    native_code = AgentService._launch_prepared_child_result.__code__
    own_code = AgentService._workflow_launch_result.__code__
    finish_code = AgentService._finish_workflow_launch.__code__
    constructor_code = threading.Thread.__init__.__code__

    def inject(frame, event, result):
        if event != "call" or frame.f_code is not constructor_code:
            return
        parent = frame.f_back
        if parent is None or parent.f_code is not native_code:
            return
        entry = parent.f_locals.get("entry")
        if entry is None or entry.preparation is not prepared:
            return
        try:
            assert (
                entry.mode == "D"
                and entry.authorization is auth
                and entry.native is native
            )
            assert (
                parent.f_locals["thread"] is None
                and parent.f_locals["start_entered"] is False
            )
            assert parent.f_locals["native_thread_started"] is False
            assert frame.f_locals["target"] is parent.f_locals["run_child_owned"]
            saved.update(
                entry=entry,
                owner=parent.f_locals["child_owner"],
                claim=parent.f_locals["child_kwargs"]["workflow_activity_claim"],
                launch=prepared._launch_operation,
                close=prepared._close_operation,
                injected=injected,
            )
        except BaseException as error:  # noqa: BLE001 - only the exact constructor fault is injected.
            errors.append(error)
            return
        threading.settrace_all_threads(None)
        sys.settrace(None)
        raise injected

    def callback(outcome):
        try:
            assert outcome is True and threading.current_thread() is saved["worker"]
            saved["callback_outcome"] = outcome
            callback_entered.set()
            if not callback_release.wait(5):
                errors.append(
                    AssertionError("original known-error callback hold exceeded5s")
                )
        except BaseException as error:  # noqa: BLE001 - native callbacks retain their original outcome.
            errors.append(error)
            callback_entered.set()

    codes = {native_code, own_code, finish_code, callback.__code__}

    def observe(frame, event, result):
        if (
            frame.f_code not in codes
            or event != "return"
            or dis.opname[frame.f_code.co_code[frame.f_lasti]]
            not in {"RETURN_VALUE", "RETURN_CONST"}
        ):
            return
        try:
            if frame.f_code is native_code:
                if frame.f_locals.get("entry") is not saved.get("entry") or not saved:
                    return
                assert type(result) is _NativeWorkflowReturn and result.known
                assert frame.f_locals["thread"] is None
                assert frame.f_locals["start_entered"] is False
                assert frame.f_locals["native_thread_started"] is False
                assert result.receipt.failure == "thread_start"
                assert result.receipt.owner is saved["owner"]
                assert result.receipt.activity_claim is saved["claim"]
                saved["native_return"] = result
            elif frame.f_code is own_code:
                if (
                    type(result) is not _WorkflowLaunchReturn
                    or result.entry.preparation is not prepared
                ):
                    return
                if result.duplicate:
                    assert (
                        result.operation is None
                        and result.receipt is saved["result"].receipt
                    )
                    saved["duplicate"] = result
                    return
                actual = saved["native_return"]
                assert (
                    result.entry is saved["entry"] and result.known and not result.wait
                )
                assert result.receipt is actual.receipt and result.error is actual.error
                assert result.owner is saved["owner"] and result.claim is saved["claim"]
                assert (
                    result.operation is saved["launch"]
                    and prepared._launch_operation is result.operation
                )
                assert prepared._native_closed and prepared._close_operation is None
                assert not native.threads and not native.uncertain_start
                saved.update(result=result, worker=threading.current_thread())
                own_entered.set()
                if not own_release.wait(5):
                    errors.append(
                        AssertionError(
                            "original known-error own-return hold exceeded5s"
                        )
                    )
            elif frame.f_code is callback.__code__:
                assert threading.current_thread() is saved["worker"]
                order.append("callback-return")
            elif frame.f_locals.get("result") is saved.get("result") and saved:
                order.append("finisher-return")
        except BaseException as error:  # noqa: BLE001 - observer assertions cannot escape into native code.
            errors.append(error)
            own_entered.set()
            callback_entered.set()

    previous_thread_trace, previous_trace = threading.gettrace(), sys.gettrace()
    previous_thread_profile, previous_profile = threading.getprofile(), sys.getprofile()
    threading.settrace_all_threads(inject)
    threading.setprofile_all_threads(observe)
    launch = asyncio.create_task(native_tests._launch_native_for_test(h, prepared))
    primary = actual_error = None
    try:
        assert await asyncio.to_thread(own_entered.wait, 5)
        if errors:
            raise errors[0]
        returned = saved["result"]
        owner, operation = returned.owner, returned.operation
        assert (
            saved["injected"] is injected and returned.error.receipt is returned.receipt
        )
        assert operation is not None and prepared._launch_operation is operation
        with owner.capacity._lock:
            assert owner._root_finished and owner._operations == {operation}
            assert owner.capacity._owners.get(owner.execution_id) is owner
        owner.on_drained(callback)
        own_release.set()
        assert await asyncio.to_thread(callback_entered.wait, 5)
        if errors:
            raise errors[0]
        assert saved["callback_outcome"] is True and not launch.done()
        with owner.capacity._lock:
            assert owner.execution_id not in owner.capacity._owners
            assert operation not in owner._operations
        assert (
            prepared._launch_operation is operation
            and prepared._close_operation is None
        )
        assert prepared.source is source and prepared._lifetime is lifetime
        assert prepared._source_scope is source_scope
        assert h.db.get_run(returned.receipt.run_id)["status"] == "error"
        with h.db.connection() as conn:
            reservation = conn.execute(
                "SELECT state FROM automatic_work_reservations WHERE id=?",
                (auth.claim.reservation_id,),
            ).fetchone()
        assert reservation[0] == "released" and h.calls == []
        with pytest.raises(_WorkflowChildLaunchFailed) as duplicate:
            await native_tests._launch_native_for_test(h, prepared)
        assert duplicate.value.receipt is returned.receipt
        assert (
            duplicate.value is saved["duplicate"].error
            and duplicate.value is not returned.error
        )
        assert (
            saved["duplicate"].operation is None
            and prepared._launch_operation is operation
        )
        with pytest.raises(PermissionError, match="preparation already claimed"):
            await run_workflow_wave(h.phase, (prepared,))
        assert prepared._wave is None and h.runtime._workflow_waves == {}
        with pytest.raises(PermissionError, match="launch return has not completed"):
            await prepared.close()
        assert not prepared._closed and not prepared._closing
        assert prepared._launch_operation is operation and not launch.done()
    except BaseException as error:
        primary = error
        raise
    finally:
        own_release.set()
        callback_release.set()
        try:
            try:
                await launch
            except _WorkflowChildLaunchFailed as error:
                actual_error = error
            except BaseException as error:
                if primary is None:
                    raise
                primary.add_note("original launch finally: " + repr(error))
        finally:
            threading.setprofile_all_threads(previous_thread_profile)
            sys.setprofile(previous_profile)
            threading.settrace_all_threads(previous_thread_trace)
            sys.settrace(previous_trace)
    if errors:
        raise errors[0]
    returned = saved["result"]
    assert actual_error is returned.error is saved["native_return"].error
    assert order == ["callback-return", "finisher-return"]
    assert prepared._launch_operation is None and prepared._close_operation is None
    drains = []
    returned.owner.on_drained(drains.append)
    assert drains == [True] and not returned.owner.is_admitted
    activity = h.runtime.worktree_activity
    with activity._lock:
        state = activity._claims[returned.claim.id]
        member = state.members[returned.claim.id]
        assert (
            state.custody is saved["claim"]
            and activity._members[returned.claim.id] is member
        )
        assert (
            member.owner is saved["owner"]
            and member.drained is True
            and member.durable is False
        )
    await native_tests._settle_native_for_test(
        h, prepared, returned.receipt, status="failed"
    )
    checkpoint = h.service._checkpoint(h.workflow.id, returned.receipt.attempt_id)
    result = h.db.workflow_runs.read_result(h.workflow.id, returned.receipt.attempt_id)
    assert (
        checkpoint.run_id == result.run_id == returned.receipt.run_id
        and result.status == "failed"
    )
    with activity._lock:
        assert member.drained is True and member.durable is True
        assert (
            returned.claim.id not in activity._claims
            and returned.claim.id not in activity._members
        )
    counts = h.db.automatic_work.snapshot(auth.claim.chain_id)
    assert dict(counts.used) == dict(original_counts.used)
    assert (
        counts.reserved["child_launch"] == 0
        and original_counts.reserved["child_launch"] == 1
    )
    assert not native.threads and not native.uncertain_start and h.calls == []
    assert prepared._closed and prepared._source_scope is None
    assert h.runtime.execution_capacity.snapshot().executions == ()


async def test_original_initial_close_returns_empty_source_and_its_same_read(
    native_host,
):
    from tldw_chatbook.Chat import console_workflow_preparation as preparation

    h = native_host
    before = h.db.automatic_work.snapshot(h.phase.attempt.chain_id)
    contender = h.runtime.execution_capacity.begin_execution(
        origin=WorkOrigin.MANUAL, conversation_id="original-initial-close-contender"
    )
    drains = []
    contender.on_drained(drains.append)
    claim = h.runtime.worktree_activity.try_acquire(
        owner=contender,
        footprint=WorktreeFootprint(
            writes=(WorktreeScope(h.root, True, capture_directory_chain(h.root)),)
        ),
    )
    assert claim is not None
    cls = preparation._PreparedWorkflowReply
    read_code, close_code = cls._read.__code__, cls.close.__code__
    finish_code = next(
        code
        for code in close_code.co_consts
        if isinstance(code, type(close_code)) and code.co_name == "finish"
    )
    check_code = preparation._storage_source_current.__code__
    prepare_code = preparation.prepare_workflow_reply.__code__
    configure_code = cls._configure.__code__
    release_code = cls._release_configuration.__code__
    codes = {read_code, close_code, check_code, configure_code, release_code}
    issuer = asyncio.current_task()
    saved, errors, checks, helper_checks = {}, [], [], []

    def observe(frame, event, result):
        if frame.f_code not in codes:
            return
        try:
            if frame.f_code is read_code and event == "call" and not saved:
                owner, callback = frame.f_locals["self"], frame.f_locals["callback"]
                if owner.phase is not h.phase or owner.bridge is not h.bridge:
                    return
                parent = frame.f_back
                if parent.f_code is not preparation._finish_native_preparation.__code__:
                    return
                assert parent.f_locals["prepared"] is owner
                assert (
                    callback.__self__ is owner and callback.__func__ is cls._configure
                )
                assert asyncio.current_task() is issuer
                saved.update(
                    prepared=owner,
                    read_frame=frame,
                    scope=owner._source_scope,
                    configure_entries=0,
                )
                return
            owner = saved.get("prepared")
            if owner is None:
                return
            if (
                frame.f_code is configure_code
                and frame.f_locals.get("self") is owner
                and event == "call"
            ):
                saved["configure_entries"] += 1
            if (
                frame.f_code is close_code
                and frame.f_locals.get("self") is owner
                and event == "call"
                and "close_frame" not in saved
            ):
                parent = frame.f_back
                assert (
                    parent.f_code is prepare_code
                    and parent.f_locals["prepared"] is owner
                )
                assert (
                    parent.f_locals["bridge"] is h.bridge
                    and parent.f_locals["phase"] is h.phase
                )
                assert (
                    parent.f_locals["binding"] is h.binding
                    and parent.f_locals["resolution"] is h.selected
                )
                assert (
                    asyncio.current_task() is issuer
                    and owner._loop is asyncio.get_running_loop()
                )
                assert (
                    owner.source
                    is owner.authorization
                    is owner.native
                    is owner._lifetime
                    is owner._configured_source_observation
                    is None
                )
                assert owner._wave is None and owner in h.runtime._workflow_preparations
                saved.update(
                    close_frame=frame,
                    failed_record=saved["read_frame"].f_locals["record"],
                )
            if event != "return" or dis.opname[
                frame.f_code.co_code[frame.f_lasti]
            ] not in {"RETURN_VALUE", "RETURN_CONST"}:
                return
            if (
                frame.f_code is check_code
                and frame.f_locals.get("preparation") is owner
            ):
                parent = frame.f_back
                assert parent.f_locals.get("self") is owner
                if parent.f_code is cls._execution_reads_source.__code__:
                    helper_checks.append(threading.current_thread())
                else:
                    assert parent.f_code in {close_code, finish_code}
                    checks.append((parent.f_code, threading.current_thread()))
            elif (
                frame.f_code is read_code
                and frame.f_locals.get("self") is owner
                and frame.f_locals.get("closing")
            ):
                callback = frame.f_locals["callback"]
                assert callback.__code__ is finish_code
                cells = dict(
                    zip(
                        callback.__code__.co_freevars,
                        (cell.cell_contents for cell in callback.__closure__ or ()),
                    )
                )
                assert cells["self"] is owner
                record = frame.f_locals["record"]
                assert record.prepared is owner and record.issuing is issuer
                saved["close_record"] = record
            elif frame.f_code is release_code and frame.f_locals.get("self") is owner:
                assert frame.f_locals["scope"] is saved["scope"]
                saved["released"] = True
            elif frame is saved.get("close_frame"):
                saved["close_returned"] = True
        except BaseException as error:  # noqa: BLE001 - observation never changes native cleanup.
            errors.append(error)

    previous_thread_profile, previous_profile = threading.getprofile(), sys.getprofile()
    threading.setprofile_all_threads(observe)
    try:
        with pytest.raises(_WorkflowChildWait) as caught:
            await h.bridge._prepare_workflow_reply(
                phase=h.phase,
                task_id="task-1",
                binding=h.binding,
                resolution=h.selected,
            )
        if errors:
            raise errors[0]
        first = saved["prepared"]
        assert (
            caught.value.reason == "worktree_activity"
            and saved["configure_entries"] == 0
        )
        failed = saved["failed_record"]
        read, producer, retired, callback = failed.proof
        assert (
            read.source is first
            and read.creator is h.controller
            and read.task is issuer
        )
        assert (
            read.callback is callback is failed.callback
            and read._producer is producer
            and read.retired is retired
        )
        assert (
            producer.done()
            and not producer.cancelled()
            and producer.exception() is None
        )
        assert retired.done() and not retired.cancelled() and retired.result() is None
        assert (
            failed.owner is failed.claim is failed.operation is None
            and failed.positive
            and failed.scopes_returned
            and failed.wrapper_error is None
            and failed.body_error is caught.value
        )
        actual_wait = producer.result()
        expected_wait = (
            failed,
            first,
            issuer,
            callback,
            failed.proof,
            None,
            None,
            None,
            None,
            caught.value,
        )
        assert type(actual_wait) is tuple and len(actual_wait) == len(expected_wait)
        assert all(a is b for a, b in zip(actual_wait, expected_wait))
        record = saved["close_record"]
        read, producer, retired, callback = record.proof
        assert (
            read.source is first
            and read.creator is h.controller
            and read.task is issuer
        )
        assert (
            read.callback is callback is record.callback
            and read._producer is producer
            and read.retired is retired
        )
        assert (
            producer.done()
            and not producer.cancelled()
            and producer.exception() is None
        )
        assert retired.done() and not retired.cancelled() and retired.result() is None
        actual = producer.result()
        expected = (
            record,
            first,
            issuer,
            callback,
            record.proof,
            None,
            None,
            None,
            None,
            None,
        )
        assert (
            type(actual) is tuple
            and len(actual) == len(expected)
            and all(a is b for a, b in zip(actual, expected))
        )
        assert record.scopes_returned and record.positive
        assert [code for code, _thread in checks] == [close_code, finish_code]
        assert threading.current_thread() in helper_checks
        assert any(thread is not threading.current_thread() for thread in helper_checks)
        assert (
            checks[0][1] is threading.current_thread()
            and checks[1][1] is not checks[0][1]
        )
        assert (
            saved["close_returned"]
            and saved["released"]
            and saved["scope"].gen.gi_frame is None
        )
        assert first._closed and not first._closing and first._source_scope is None
        assert first not in h.runtime._workflow_preparations and first._finite is None
        assert (
            not h.phase._task_scopes
            and not h.calls
            and h.db.list_runs(h.conversation_id) == []
        )
        after = h.db.automatic_work.snapshot(h.phase.attempt.chain_id)
        assert dict(after.used) == dict(before.used) and dict(after.reserved) == dict(
            before.reserved
        )
        assert contender.is_admitted and drains == []
    finally:
        threading.setprofile_all_threads(previous_thread_profile)
        sys.setprofile(previous_profile)
        contender.finish_root()
        claim.settle(durable=True)
    assert drains == [True] and not contender.is_admitted
    with h.runtime.worktree_activity._lock:
        assert (
            claim.id not in h.runtime.worktree_activity._claims
            and claim.id not in h.runtime.worktree_activity._members
        )
    retry = await h.bridge._prepare_workflow_reply(
        phase=h.phase, task_id="task-1", binding=h.binding, resolution=h.selected
    )
    assert (
        retry is not saved["prepared"]
        and retry.source is not None
        and retry.authorization is not None
    )
    await retry.close()
    assert (
        retry._closed
        and retry._source_scope is None
        and retry._finite_reads[-1].positive
    )
    assert h.runtime.execution_capacity.snapshot().executions == ()


async def external_core_journey(tmp_path, _unused):
    import os
    from types import SimpleNamespace

    from tldw_chatbook.Agents import activation
    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Backup_Recovery import (
        admission_runtime,
    )
    from tldw_chatbook.Backup_Recovery import (
        storage_admission as storage,
    )
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.DB.base_db import run_owned_db_call

    generator = None
    native_fixture_returned = False
    primary = None
    previous_thread_profile, previous_profile = threading.getprofile(), sys.getprofile()
    previous_thread_trace, previous_trace = threading.gettrace(), sys.gettrace()
    with pytest.MonkeyPatch.context() as monkeypatch:
        try:
            generator = native_tests.native_host.__wrapped__(
                tmp_path, monkeypatch, SimpleNamespace(param=None)
            )
            h = await anext(generator)
            prepared = await h.bridge._prepare_workflow_reply(
                phase=h.phase,
                task_id="task-1",
                binding=h.binding,
                resolution=h.selected,
            )
            auth = prepared.authorization
            native = auth._native_preparation[0]
            source, observation = (
                prepared.source,
                prepared._configured_source_observation,
            )
            lifetime, source_scope = prepared._lifetime, prepared._source_scope
            counts = h.db.automatic_work.snapshot(auth.claim.chain_id)
            with h.db.connection() as conn:
                reservation = tuple(
                    conn.execute(
                        "SELECT id,chain_id,owner_id,kind,amount,state FROM automatic_work_reservations WHERE id=?",
                        (auth.claim.reservation_id,),
                    ).fetchone()
                )
            original_marker, original_waits = auth._native_launch, native.wait_return
            saved, errors, path_errors = {}, [], []
            diagnostic_guards, diagnostic_paths, diagnostic_errors = [], [], []
            public_code = AgentService.launch_workflow_task.__code__
            issuer = asyncio.current_task()
            repository_code = storage._repository_operation.__wrapped__.__code__
            core_code = _core_operation.__wrapped__.__code__
            lease_code = storage.StorageLease.close.__code__
            guard_code = (
                admission_runtime.RecoveryAdmissionGuard.execution.__wrapped__.__code__
            )
            check_code = storage._Operation.check.__code__
            result_code = AgentService._workflow_launch_result.__code__
            native_code = AgentService._launch_prepared_child_result.__code__
            finisher_code = AgentService._finish_workflow_launch.__code__
            close_method = type(prepared).close.__code__
            close_finish_code = next(
                code
                for code in close_method.co_consts
                if isinstance(code, type(core_code)) and code.co_name == "finish"
            )
            read_code = type(prepared)._read.__code__
            invoke_code = next(
                code
                for code in run_owned_db_call.__code__.co_consts
                if isinstance(code, type(core_code)) and code.co_name == "invoke"
            )

            def callback():
                outer = saved["outer"]
                assert threading.current_thread() is saved["worker"]
                assert storage._operation_local.operation is outer
                with storage._lock:
                    assert (
                        outer in storage._operations
                        and outer.lease in storage._live_leases
                    )
                    assert outer.participant is h.db._maintenance_participant
                    assert outer.participant.repository() is h.db
                    assert outer.thread is saved["worker"] and outer.task is None
                    assert outer.pid == os.getpid() and outer.path == h.db.db_path
                try:
                    with owned_goal_storage(h.service), h.phase.context.scope():
                        prepared.native.service.launch_workflow_task(
                            auth.task,
                            authorization=auth,
                            should_cancel=prepared.should_cancel,
                        )
                except activation.AgentActivationRequired as error:
                    saved["public_error"] = error
                    outcome = (outer, error)
                    saved["outcome"] = outcome
                    return outcome
                raise AssertionError(
                    "original narrow core did not refuse native admission"
                )

            def profile(frame, event, result):
                code = frame.f_code
                if code not in {
                    repository_code,
                    core_code,
                    lease_code,
                    invoke_code,
                    run_owned_db_call.__code__,
                    callback.__code__,
                    _WORKFLOW_LAUNCH_CODE,
                    native_code,
                    finisher_code,
                    close_method,
                    close_finish_code,
                    read_code,
                }:
                    return
                try:
                    if (
                        code in {_WORKFLOW_LAUNCH_CODE, native_code, finisher_code}
                        and event == "call"
                    ):
                        saved.setdefault("unexpected_bodies", []).append(code)
                    if (
                        code is close_finish_code
                        and event == "call"
                        and frame.f_locals.get("self") is prepared
                    ):
                        parent = frame.f_back
                        record = parent.f_locals["record"]
                        assert parent.f_locals["self"] is prepared
                        assert parent.f_locals["callback"].__code__ is close_finish_code
                        assert record.prepared is prepared and record.issuing is issuer
                        assert record.callback is parent.f_locals["invoke"]
                        assert record is prepared._finite_reads[-1]
                        saved.update(
                            close_read=record,
                            close_callback_frame=frame,
                            close_worker=threading.current_thread(),
                        )
                    if (
                        code is repository_code
                        and event == "return"
                        and dis.opname[code.co_code[frame.f_lasti]] == "YIELD_VALUE"
                    ):
                        participant = frame.f_locals["participant"]
                        if participant.repository() is not h.db or "outer" in saved:
                            return
                        ancestors = []
                        parent = frame.f_back
                        while parent is not None:
                            ancestors.append(parent)
                            parent = parent.f_back
                        invokes = [
                            item
                            for item in ancestors
                            if item.f_code is invoke_code
                            and item.f_locals.get("database") is h.db
                            and item.f_locals.get("operation") is callback
                        ]
                        cores = [
                            item
                            for item in ancestors
                            if item.f_code is core_code
                            and item.f_locals.get("repository") is h.db
                        ]
                        if len(invokes) != 1 or len(cores) != 1:
                            return
                        assert (
                            result is frame.f_locals["operation"]
                            and frame.f_locals["previous"] is None
                        )
                        assert result is storage._operation_local.operation
                        saved.update(
                            outer=result,
                            lease=result.lease,
                            worker=threading.current_thread(),
                            repository_frame=frame,
                            core_frame=cores[0],
                            invoke_frame=invokes[0],
                        )
                    elif code is callback.__code__ and event == "call":
                        assert frame.f_back is saved["invoke_frame"]
                        assert (
                            frame.f_back.f_locals["database"] is h.db
                            and frame.f_back.f_locals["operation"] is callback
                        )
                        assert storage._operation_local.operation is saved["outer"]
                    if event != "return" or dis.opname[
                        code.co_code[frame.f_lasti]
                    ] not in {"RETURN_VALUE", "RETURN_CONST"}:
                        return
                    if frame is saved.get("close_callback_frame"):
                        assert threading.current_thread() is saved["close_worker"]
                        saved["close_callback_returned"] = True
                    elif (
                        code is read_code
                        and frame.f_locals.get("record") is saved.get("close_read")
                        and "close_read" in saved
                    ):
                        assert (
                            frame.f_locals["self"] is prepared
                            and saved["close_read"].positive
                        )
                        assert asyncio.current_task() is issuer
                        saved["close_read_returned"] = True
                    elif (
                        code is close_method
                        and "close_coroutine" in saved
                        and saved["close_coroutine"].cr_frame is frame
                    ):
                        assert (
                            frame.f_locals["self"] is prepared
                            and asyncio.current_task() is issuer
                        )
                        assert (
                            prepared._source_scope is None
                            and source_scope.gen.gi_frame is None
                        )
                        saved["close_returned"] = True
                    elif frame is saved.get("repository_frame"):
                        saved["repository_returned"] = True
                    elif frame is saved.get("core_frame"):
                        saved["core_returned"] = True
                    elif (
                        code is lease_code
                        and frame.f_locals.get("self") is saved.get("lease")
                        and "lease" in saved
                    ):
                        saved["lease_returned"] = True
                    elif frame is saved.get("invoke_frame"):
                        assert (
                            storage._operation_local.operation is None
                            and result is saved["outcome"]
                        )
                        saved["invoke_returned"] = True
                    elif (
                        code is run_owned_db_call.__code__
                        and frame.f_locals.get("database") is h.db
                        and frame.f_locals.get("operation") is callback
                    ):
                        assert (
                            asyncio.current_task() is issuer
                            and result is saved["outcome"]
                        )
                        saved["host_returned"] = True
                except BaseException as error:  # noqa: BLE001 - observation cannot change native operations.
                    errors.append(error)

            def trace(frame, event, value):
                if frame.f_code not in {guard_code, check_code}:
                    return None
                try:
                    if frame.f_code is guard_code and event == "call":
                        ancestors = []
                        parent = frame.f_back
                        while parent is not None:
                            ancestors.append(parent)
                            parent = parent.f_back
                        if (
                            frame.f_locals.get("self") is activation._guard
                            and threading.current_thread() is saved.get("worker")
                            and storage._operation_local.operation is saved.get("outer")
                            and any(
                                item.f_code is public_code
                                and item.f_locals.get("self") is prepared.native.service
                                and item.f_locals.get("authorization") is auth
                                for item in ancestors
                            )
                            and not any(item[0] is frame for item in diagnostic_guards)
                        ):
                            receiver = frame.f_locals.get("service")
                            kind = (
                                "original-ledger"
                                if receiver is h.phase._ledger
                                else "original-native-service"
                                if receiver is prepared.native.service
                                else "other"
                            )
                            diagnostic_guards.append((frame, kind, receiver))
                    elif (
                        frame.f_code is check_code
                        and event == "exception"
                        and "outer" in saved
                        and frame.f_locals.get("self") is saved["outer"]
                    ):
                        ancestors = []
                        parent = frame.f_back
                        while parent is not None:
                            ancestors.append(parent)
                            parent = parent.f_back
                        guards = [
                            item
                            for item in diagnostic_guards
                            if any(item[0] is ancestor for ancestor in ancestors)
                        ]
                        error_type, error, traceback = value
                        diagnostic_paths.append(
                            (
                                frame,
                                error,
                                guards,
                                {
                                    "type": error_type.__name__,
                                    "reason": str(error)
                                    if str(error)
                                    in {
                                        "operation_path_outside_scope",
                                        "operation_provenance_invalid",
                                    }
                                    else "other",
                                    "guard_kinds": [item[1] for item in guards],
                                    "same_outer": frame.f_locals.get("self")
                                    is saved["outer"],
                                    "same_worker": threading.current_thread()
                                    is saved["worker"],
                                    "same_tls": storage._operation_local.operation
                                    is saved["outer"],
                                    "same_traceback_frame": traceback.tb_frame is frame,
                                    "selected_differs": frame.f_locals.get("selected")
                                    != saved["outer"].path,
                                    "original_observation_path": any(
                                        frame.f_locals.get("path") is row[1]
                                        for row in observation[2]
                                    ),
                                    "stack": [
                                        (
                                            item.f_code.co_filename,
                                            item.f_code.co_name,
                                            item.f_lineno,
                                        )
                                        for item in (frame, *ancestors)
                                    ],
                                },
                            )
                        )
                except BaseException as error:  # noqa: BLE001 - passive diagnostic cannot affect native calls.
                    diagnostic_errors.append(error)
                try:
                    if frame.f_code is guard_code and event == "call":
                        if (
                            frame.f_locals.get("self") is not activation._guard
                            or frame.f_locals.get("service") is not h.phase._ledger
                        ):
                            return trace
                        ancestors = []
                        parent = frame.f_back
                        while parent is not None:
                            ancestors.append(parent)
                            parent = parent.f_back
                        launches = [
                            item
                            for item in ancestors
                            if item.f_code is result_code
                            and item.f_locals.get("self") is prepared.native.service
                        ]
                        assert len(launches) == 1
                        assert "entry" not in launches[0].f_locals
                        assert launches[0].f_locals["authorization"] is auth
                        assert any(
                            item.f_code is type(h.phase).check.__code__
                            and item.f_locals.get("self") is h.phase
                            for item in ancestors
                        )
                        assert any(
                            item.f_code is type(auth).check_preparation.__code__
                            and item.f_locals.get("self") is auth
                            for item in ancestors
                        )
                        assert any(
                            item.f_code
                            is AgentService._check_workflow_launch_authorization.__code__
                            and item.f_locals.get("self") is prepared.native.service
                            and item.f_locals.get("authorization") is auth
                            for item in ancestors
                        )
                        assert (
                            threading.current_thread() is saved["worker"]
                            and storage._operation_local.operation is saved["outer"]
                        )
                        assert auth._native_launch == [] and native.wait_return == []
                        saved.update(guard_frame=frame, preentry_frame=launches[0])
                    elif (
                        frame.f_code is check_code
                        and event == "exception"
                        and frame.f_locals.get("self") is saved.get("outer")
                        and "outer" in saved
                    ):
                        error_type, error, traceback = value
                        guard = saved.get("guard_frame")
                        if guard is None:
                            return trace
                        ancestors = []
                        parent = frame.f_back
                        while parent is not None:
                            ancestors.append(parent)
                            parent = parent.f_back
                        assert any(item is guard for item in ancestors)
                        assert (
                            error_type is RecoveryRequired
                            and str(error) == "operation_path_outside_scope"
                        )
                        assert traceback.tb_frame is frame
                        requested = guard.f_locals["path"]
                        assert any(
                            requested is row[1] for row in guard.f_locals["observed"]
                        )
                        check_operation = frame.f_back
                        acquisition_check = check_operation.f_back
                        acquisition = acquisition_check.f_locals["self"]
                        assert (
                            check_operation.f_code is storage._check_operation.__code__
                        )
                        assert check_operation.f_locals["operation"] is saved["outer"]
                        assert (
                            acquisition_check.f_code
                            is storage._Acquisition.check.__code__
                        )
                        assert acquisition.operation is saved["outer"]
                        original_check = acquisition_check.f_back
                        acquire = original_check.f_back
                        wrapper = acquire.f_back
                        assert original_check.f_locals["attempt"] is acquisition
                        assert (
                            acquire.f_code is storage._acquire_storage.__code__
                            and acquire.f_locals["attempt"] is acquisition
                        )
                        assert (
                            wrapper.f_code is storage.acquire_storage.__code__
                            and wrapper.f_locals["attempt"] is acquisition
                        )
                        assert wrapper.f_back is guard
                        assert (
                            dis.opname[frame.f_code.co_code[traceback.tb_lasti]]
                            == "RAISE_VARARGS"
                        )
                        assert frame.f_locals["path"] == requested
                        assert frame.f_locals["selected"] != saved["outer"].path
                        assert (
                            threading.current_thread() is saved["worker"]
                            and storage._operation_local.operation is saved["outer"]
                        )
                        path_errors.append((error, frame, requested, saved["outer"]))
                except BaseException as error:  # noqa: BLE001 - no guard mutation or exception injection.
                    errors.append(error)
                return trace

            threading.setprofile_all_threads(profile)
            threading.settrace_all_threads(trace)
            try:
                outcome = await run_owned_db_call(h.db, callback)
            finally:
                threading.setprofile_all_threads(previous_thread_profile)
                sys.setprofile(previous_profile)
                threading.settrace_all_threads(previous_thread_trace)
                sys.settrace(previous_trace)
            if errors:
                raise errors[0]
            assert outcome is saved["outcome"] and outcome == (
                saved["outer"],
                saved["public_error"],
            )
            public_frames = []
            traceback = saved["public_error"].__traceback__
            while traceback is not None:
                public_frames.append(
                    (
                        traceback.tb_frame.f_code.co_filename,
                        traceback.tb_frame.f_code.co_name,
                        traceback.tb_lineno,
                    )
                )
                traceback = traceback.tb_next
            print(
                "G_EXTERNAL_CORE_DISCRIMINATOR",
                {
                    "path_errors": len(path_errors),
                    "guards": [item[1] for item in diagnostic_guards[:8]],
                    "paths": [item[3] for item in diagnostic_paths[:8]],
                    "diagnostic_errors": [
                        type(item).__name__ for item in diagnostic_errors
                    ],
                    "public_error": type(saved["public_error"]).__name__,
                    "public_stack": public_frames,
                    "D_entry_observed": "entry" in saved,
                    "marker_count": len(original_marker),
                    "wait_count": len(original_waits),
                    "outer_returns": {
                        name: saved.get(name)
                        for name in (
                            "repository_returned",
                            "core_returned",
                            "lease_returned",
                            "invoke_returned",
                            "host_returned",
                        )
                    },
                },
                flush=True,
            )
            assert len(path_errors) == 1 and path_errors[0][3] is saved["outer"]
            assert not saved.get("unexpected_bodies")
            assert (
                saved["repository_returned"]
                and saved["core_returned"]
                and saved["lease_returned"]
                and saved["invoke_returned"]
                and saved["host_returned"]
            )
            with storage._lock:
                assert (
                    saved["outer"] not in storage._operations
                    and saved["lease"] not in storage._live_leases
                )
            assert auth._native_launch is original_marker and original_marker == []
            assert native.wait_return is original_waits and original_waits == []
            assert (
                native.source_observation is observation
                and prepared._configured_source_observation is observation
            )
            assert source.sources is observation[2] and prepared.source is source
            assert (
                not native.activity_claims
                and not native.threads
                and not native.uncertain_start
            )
            assert prepared._launch_operation is prepared._close_operation is None
            assert h.calls == [] and h.db.list_runs(h.conversation_id) == []
            assert h.runtime.execution_capacity.snapshot().executions == ()
            assert "entry" not in saved["preentry_frame"].f_locals
            assert public_frames[-1][1] == "execution"
            actual_public = saved["public_error"].__traceback__
            while actual_public.tb_next is not None:
                actual_public = actual_public.tb_next
            assert actual_public.tb_frame is saved["guard_frame"]
            assert actual_public.tb_frame.f_code is guard_code
            before_close = h.db.automatic_work.snapshot(auth.claim.chain_id)
            assert dict(before_close.used) == dict(counts.used)
            assert dict(before_close.reserved) == dict(counts.reserved)
            with h.db.connection() as conn:
                assert (
                    tuple(
                        conn.execute(
                            "SELECT id,chain_id,owner_id,kind,amount,state FROM automatic_work_reservations WHERE id=?",
                            (auth.claim.reservation_id,),
                        ).fetchone()
                    )
                    == reservation
                )
            assert (
                prepared._lifetime is lifetime
                and prepared._source_scope is source_scope
            )
            reads = prepared._finite_reads
            count = len(reads)
            closing = prepared.close()
            saved["close_coroutine"] = closing
            threading.setprofile_all_threads(profile)
            try:
                await closing
            finally:
                threading.setprofile_all_threads(previous_thread_profile)
                sys.setprofile(previous_profile)
            if errors:
                raise errors[0]
            record = saved["close_read"]
            assert prepared._finite_reads is reads and len(reads) == count + 1
            assert reads[-1] is record and record.positive
            assert (
                saved["close_callback_returned"]
                and saved["close_read_returned"]
                and saved["close_returned"]
            )
            assert closing.cr_frame is None
            assert prepared._finite_read_return(record)[0] is record
            assert (
                prepared._closed and prepared._native_closed and not prepared._closing
            )
            assert prepared._source_scope is None and source_scope.gen.gi_frame is None
            assert lifetime.gen.gi_frame is None
            assert prepared not in h.runtime._workflow_preparations
            await prepared.close()
            assert prepared._finite_reads is reads and len(reads) == count + 1
            assert prepared.source is source and source.sources is observation[2]
            assert (
                prepared._configured_source_observation
                is native.source_observation
                is observation
            )
            assert auth._native_launch is original_marker and original_marker == []
            assert native.wait_return is original_waits and original_waits == []
            after = h.db.automatic_work.snapshot(auth.claim.chain_id)
            assert dict(after.used) == dict(counts.used) and dict(
                after.reserved
            ) == dict(counts.reserved)
            with h.db.connection() as conn:
                assert (
                    tuple(
                        conn.execute(
                            "SELECT id,chain_id,owner_id,kind,amount,state FROM automatic_work_reservations WHERE id=?",
                            (auth.claim.reservation_id,),
                        ).fetchone()
                    )
                    == reservation
                )
            assert h.calls == [] and h.db.list_runs(h.conversation_id) == []
            assert h.runtime.execution_capacity.snapshot().executions == ()
        except BaseException as error:
            primary = error
            raise
        finally:
            threading.setprofile_all_threads(previous_thread_profile)
            sys.setprofile(previous_profile)
            threading.settrace_all_threads(previous_thread_trace)
            sys.settrace(previous_trace)
            if generator is not None:
                try:
                    await generator.aclose()
                    native_fixture_returned = True
                except BaseException as error:
                    if primary is None:
                        raise
                    primary.add_note("original fixture cleanup: " + repr(error))
    assert native_fixture_returned and h.runtime._disposed
    assert h.runtime.execution_capacity.snapshot().closed
    assert h.runtime.execution_capacity.snapshot().executions == ()
    assert not h.runtime._workflow_preparations and not h.runtime._preparation_reads


async def test_original_external_counted_core_refuses_before_native_entry_and_closes(
    tmp_path,
):
    await external_core_journey(tmp_path, None)


def _native_two_independent_tasks(data, root):
    from copy import deepcopy

    second = deepcopy(data["phases"][0]["tasks"][0])
    second.update(
        id="task-2",
        instructions="Only the second approved task",
        footprint=[{"path": "second.txt", "access": "write", "subtree": False}],
    )
    (root / "second.txt").write_text("second original\n")
    data["phases"][0]["tasks"].append(second)


@pytest.mark.parametrize("native_host", [_native_two_independent_tasks], indirect=True)
async def test_actual_second_member_refusal_rolls_back_first_before_wave_driver(
    native_host,
):
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Chat.console_workflow_preparation import _PreparedWorkflowReply

    h = native_host
    first = await h.bridge._prepare_workflow_reply(
        phase=h.phase, task_id="task-1", binding=h.binding, resolution=h.selected
    )
    second = await h.bridge._prepare_workflow_reply(
        phase=h.phase, task_id="task-2", binding=h.binding, resolution=h.selected
    )
    items = (first, second)
    originals = tuple(
        (
            item,
            item.authorization,
            item.authorization._native_preparation[0],
            item.source,
            item._configured_source_observation,
            item._lifetime,
            item._source_scope,
        )
        for item in items
    )
    before = h.db.automatic_work.snapshot(h.phase.attempt.chain_id)
    with h.db.connection() as conn:
        reservations = tuple(
            tuple(
                conn.execute(
                    "SELECT id,state,amount FROM automatic_work_reservations WHERE id=?",
                    (item.authorization.claim.reservation_id,),
                ).fetchone()
            )
            for item in items
        )
    competitor = h.runtime.execution_capacity.begin_execution(
        origin=WorkOrigin.MANUAL, conversation_id="partial-wave-contender"
    )
    drains = []
    competitor.on_drained(drains.append)
    claim = h.runtime.worktree_activity.try_acquire(
        owner=competitor,
        footprint=WorktreeFootprint(
            writes=(WorktreeScope(h.root, True, capture_directory_chain(h.root)),)
        ),
    )
    assert claim is not None
    entered, release = threading.Event(), threading.Event()
    saved, errors, close_reads = {}, [], {}
    auth, native = second.authorization, second.authorization._native_preparation[0]
    marker, waits = auth._native_launch, native.wait_return
    own_code = AgentService._workflow_launch_result.__code__
    finish_code = AgentService._finish_workflow_launch.__code__
    wave_code = run_workflow_wave.__code__
    constructor_code = _WorkflowWave.__init__.__code__
    close_code = next(
        code
        for code in _PreparedWorkflowReply.close.__code__.co_consts
        if isinstance(code, type(own_code)) and code.co_name == "finish"
    )
    host = asyncio.current_task()
    diagnostic_errors = []
    helper_code = next(
        code
        for code in native_tests._launch_native_for_test.__code__.co_consts
        if isinstance(code, type(own_code)) and code.co_name == "launch"
    )
    public_code = AgentService.launch_workflow_task.__code__
    observed_codes = frozenset(
        (
            own_code,
            finish_code,
            constructor_code,
            close_code,
            _WORKFLOW_LAUNCH_CODE,
            helper_code,
            public_code,
        )
    )

    def diagnostic(stage):
        try:
            worker = saved.get("helper_worker")
            stack = []
            if worker is not None and worker.is_alive():
                frame = sys._current_frames().get(worker.ident)
                while frame is not None and len(stack) < 30:
                    stack.append(
                        (frame.f_code.co_filename, frame.f_code.co_name, frame.f_lineno)
                    )
                    frame = frame.f_back
                if not worker.is_alive():
                    stack = ["worker-returned-during-observation"]
            awaiting = launch.get_coro()
            await_chain = []
            for _ in range(12):
                frame = getattr(awaiting, "cr_frame", None)
                if frame is None:
                    break
                await_chain.append(
                    (frame.f_code.co_filename, frame.f_code.co_name, frame.f_lineno)
                )
                awaiting = getattr(awaiting, "cr_await", None)
            task_error = (
                launch.exception() if launch.done() and not launch.cancelled() else None
            )
            traceback = task_error.__traceback__ if task_error is not None else None
            error_stack = []
            while traceback is not None and len(error_stack) < 30:
                error_stack.append(
                    (
                        traceback.tb_frame.f_code.co_filename,
                        traceback.tb_frame.f_code.co_name,
                        traceback.tb_lineno,
                    )
                )
                traceback = traceback.tb_next
            result = saved.get("result")
            print(
                "G_PARTIAL_W_READINESS",
                {
                    "stage": stage,
                    "entered": entered.is_set(),
                    "release": release.is_set(),
                    "task_done": launch.done(),
                    "task_cancelled": launch.cancelled(),
                    "task_error": type(task_error).__name__
                    if task_error is not None
                    else None,
                    "task_error_reason": task_error.reason
                    if type(task_error) is _WorkflowChildWait
                    else None,
                    "same_wait_error": task_error is actual_wait
                    if actual_wait is not None
                    else None,
                    "same_own_error": result.error is task_error
                    if result is not None
                    else None,
                    "helper_worker_observed": worker is not None,
                    "public_call": saved.get("public_call", False),
                    "result_call": saved.get("result_call", False),
                    "body_entry": "body_frame" in saved,
                    "own_result": result is not None,
                    "finisher_return": saved.get("finisher_returned", False),
                    "marker_count": len(marker),
                    "wait_count": len(waits),
                    "observer_errors": [type(item).__name__ for item in errors],
                    "diagnostic_errors": [
                        type(item).__name__ for item in diagnostic_errors
                    ],
                    "worker_stack": stack,
                    "task_await": await_chain,
                    "error_stack": error_stack,
                },
                flush=True,
            )
        except BaseException as error:  # noqa: BLE001 - the original assertion/finally must survive diagnostic failure.
            diagnostic_errors.append(error)
            print("G_PARTIAL_W_DIAGNOSTIC_ERROR", type(error).__name__, flush=True)

    def profile(frame, event, result):
        if frame.f_code not in observed_codes:
            return
        try:
            if event == "call" and frame.f_code in {helper_code, public_code, own_code}:
                try:
                    if (
                        frame.f_code is helper_code
                        and frame.f_locals.get("h") is h
                        and frame.f_locals.get("prepared") is second
                    ):
                        coroutine = launch.get_coro()
                        original = coroutine.cr_frame.f_locals["launch"]
                        assert (
                            coroutine.cr_code
                            is native_tests._launch_native_for_test.__code__
                        )
                        assert original.__code__ is helper_code
                        cells = dict(
                            zip(
                                original.__code__.co_freevars,
                                (cell.cell_contents for cell in original.__closure__),
                            )
                        )
                        assert cells["h"] is h and cells["prepared"] is second
                        saved.update(
                            helper_worker=threading.current_thread(), helper_frame=frame
                        )
                    elif (
                        frame.f_locals.get("self") is second.native.service
                        and frame.f_locals.get("authorization") is auth
                    ):
                        saved[
                            "public_call"
                            if frame.f_code is public_code
                            else "result_call"
                        ] = True
                except BaseException as error:  # noqa: BLE001 - passive associations cannot alter the original call.
                    diagnostic_errors.append(error)
            if frame.f_code is _WORKFLOW_LAUNCH_CODE and event == "call":
                entry = frame.f_locals["entry"]
                if entry.preparation is not second:
                    return
                wrapper = frame.f_back
                original = wrapper.f_locals["func"]
                own = wrapper.f_back
                assert original.__code__ is _WORKFLOW_LAUNCH_CODE
                cells = dict(
                    zip(
                        original.__code__.co_freevars,
                        (cell.cell_contents for cell in original.__closure__),
                    )
                )
                assert cells["entry"] is entry and own.f_code is own_code
                assert (
                    own.f_locals["entry"] is entry
                    and own.f_locals["self"] is second.native.service
                )
                assert (
                    entry.authorization is auth
                    and entry.native is native
                    and entry.mode == "D"
                )
                assert (
                    entry.caller is own.f_back
                    and entry.caller.f_locals["authorization"] is auth
                )
                assert entry.caller.f_code is AgentService.launch_workflow_task.__code__
                assert auth._native_launch is marker and marker == [None]
                assert native.wait_return is waits and waits == []
                assert not auth._lock.locked()
                assert second._launch_operation is second._close_operation is None
                outer = storage._operation_local.operation
                with storage._lock:
                    assert (
                        outer in storage._operations
                        and outer.lease in storage._live_leases
                    )
                    assert outer.participant.repository() is second.native.service.db
                    assert (
                        outer.thread is threading.current_thread()
                        and outer.task is None
                    )
                saved.update(
                    entry=entry,
                    worker=threading.current_thread(),
                    body_frame=frame,
                    own_frame=own,
                    core=outer,
                )
                entered.set()
                if not release.wait(5):
                    errors.append(AssertionError("original body-entry hold exceeded5s"))
                return
            if frame.f_code is close_code and event == "call":
                item = frame.f_locals.get("self")
                if not any(item is original for original in items):
                    return
                parent = frame.f_back
                assert parent.f_locals["self"] is item
                record = parent.f_locals["record"]
                assert record.prepared is item and record.issuing is host
                assert frame is not parent and record is item._finite_reads[-1]
                close_reads[item] = record
                return
            if event != "return" or dis.opname[
                frame.f_code.co_code[frame.f_lasti]
            ] not in {"RETURN_VALUE", "RETURN_CONST"}:
                return
            if frame.f_code is constructor_code:
                wave = frame.f_locals["self"]
                if wave.prepared == items and wave.phase is h.phase:
                    assert "wave" not in saved
                    saved["wave"] = wave
            elif frame.f_code is own_code:
                if (
                    type(result) is not _WorkflowLaunchReturn
                    or result.entry.preparation is not second
                ):
                    return
                assert (
                    result.entry.authorization is auth and result.entry.native is native
                )
                assert result.entry.mode == "D" and result.known and result.wait
                assert (
                    result.receipt
                    is result.owner
                    is result.claim
                    is result.operation
                    is None
                )
                assert (
                    type(result.error) is _WorkflowChildWait
                    and result.error.reason == "worktree_activity"
                )
                assert auth._native_launch is marker and marker == [None]
                assert native.wait_return is waits and waits == [None]
                assert result.entry is saved["entry"] and frame is saved["own_frame"]
                assert (
                    threading.current_thread() is saved["worker"] and release.is_set()
                )
                saved["result"] = result
            elif (
                frame.f_locals.get("result") is saved.get("result")
                and "result" in saved
            ):
                assert threading.current_thread() is saved["worker"]
                saved["finisher_returned"] = True
        except BaseException as error:  # noqa: BLE001 - observations never escape into native callbacks.
            errors.append(error)
            entered.set()

    def trace(frame, event, value):
        if frame.f_code is not wave_code:
            return None
        if event == "exception":
            try:
                error_type, error, _traceback = value
                if frame.f_locals.get("item") is second:
                    wave = saved["wave"]
                    assert error_type is PermissionError
                    assert str(error) == "workflow preparation already claimed"
                    assert (
                        asyncio.current_task() is host
                        and frame.f_locals["wave"] is wave
                    )
                    assert (
                        frame.f_locals["auth"] is auth
                        and frame.f_locals["native"] is native
                    )
                    assert frame.f_locals["bound"] == [first] and first._wave is wave
                    assert (
                        second._wave is None
                        and auth._native_launch is marker
                        and marker == [None]
                    )
                    assert (
                        wave.driver
                        is wave.source
                        is wave.fence
                        is wave.reservation
                        is None
                    )
                    assert (
                        wave._driver_coroutine is None
                        and wave not in h.runtime._workflow_waves
                    )
                    saved["refusal"] = error
            except BaseException as error:  # noqa: BLE001 - preserve original native refusal.
                errors.append(error)
        return trace

    previous_profile, previous_thread_profile = sys.getprofile(), threading.getprofile()
    previous_trace = sys.gettrace()
    threading.setprofile_all_threads(profile)
    try:
        launch = asyncio.create_task(native_tests._launch_native_for_test(h, second))
        primary = actual_wait = None
        try:
            ready = await asyncio.to_thread(entered.wait, 30)
            if not ready:
                diagnostic("original-failed-setup30")
            assert ready
            if errors:
                raise errors[0]
            sys.settrace(trace)
            try:
                with pytest.raises(
                    PermissionError, match="workflow preparation already claimed"
                ) as refused:
                    await run_workflow_wave(h.phase, items)
            finally:
                sys.settrace(previous_trace)
            if errors:
                raise errors[0]
            assert refused.value is saved["refusal"]
            wave = saved["wave"]
            assert first._wave is second._wave is None
            assert (
                wave.driver
                is wave._driver_coroutine
                is wave.source
                is wave.fence
                is wave.reservation
                is wave.publication
                is None
            )
            assert not wave.baseline_return.done() and not wave.ending_return.done()
            assert not wave.receipts and wave not in h.runtime._workflow_waves
            assert marker == [None] and waits == [] and not launch.done()
            assert h.calls == [] and h.db.list_runs(h.conversation_id) == []
        except BaseException as error:
            primary = error
            raise
        finally:
            release.set()
            try:
                try:
                    await launch
                except _WorkflowChildWait as error:
                    actual_wait = error
                except BaseException as error:
                    if primary is None:
                        raise
                    primary.add_note("original launch cleanup: " + repr(error))
            finally:
                diagnostic("original-launch-finally")
                sys.settrace(previous_trace)
                competitor.finish_root()
                claim.settle(durable=True)
        try:
            if errors:
                raise errors[0]
            assert actual_wait is saved["result"].error and saved["finisher_returned"]
            assert marker == [] and waits == [(actual_wait, None, None)]
            assert drains == [True] and not competitor.is_admitted
            with h.runtime.worktree_activity._lock:
                assert claim.id not in h.runtime.worktree_activity._claims
                assert claim.id not in h.runtime.worktree_activity._members
            for (
                item,
                authorization,
                actual_native,
                source,
                observation,
                lifetime,
                scope,
            ) in originals:
                assert (
                    item.authorization is authorization
                    and authorization._native_preparation[0] is actual_native
                )
                assert (
                    item.source is source
                    and item._configured_source_observation is observation
                )
                assert item._lifetime is lifetime and item._source_scope is scope
                assert item._launch_operation is item._close_operation is None
                await item.close()
                record = close_reads[item]
                assert record is item._finite_reads[-1] and record.positive
                assert item._finite_read_return(record)[0] is record
                assert (
                    item._closed
                    and item._source_scope is None
                    and scope.gen.gi_frame is None
                )
                assert item not in h.runtime._workflow_preparations
            after = h.db.automatic_work.snapshot(h.phase.attempt.chain_id)
            assert dict(after.used) == dict(before.used) and dict(
                after.reserved
            ) == dict(before.reserved)
            with h.db.connection() as conn:
                assert (
                    tuple(
                        tuple(
                            conn.execute(
                                "SELECT id,state,amount FROM automatic_work_reservations WHERE id=?",
                                (item.authorization.claim.reservation_id,),
                            ).fetchone()
                        )
                        for item in items
                    )
                    == reservations
                )
            assert h.calls == [] and h.db.list_runs(h.conversation_id) == []
            assert h.runtime.execution_capacity.snapshot().executions == ()
            if errors:
                raise errors[0]
        finally:
            threading.setprofile_all_threads(previous_thread_profile)
            sys.setprofile(previous_profile)
    finally:
        threading.setprofile_all_threads(previous_thread_profile)
        sys.setprofile(previous_profile)
        sys.settrace(previous_trace)


async def wave_capture_close_journey(tmp_path, monkeypatch):
    import json
    import os
    from pathlib import Path

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig
    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Agents.workflow_models import WorkflowPlan
    from tldw_chatbook.Chat.console_chat_controller import (
        capture_run_admitted_workspace_roots,
    )
    from tldw_chatbook.Chat.console_workflow_preparation import _PreparedWorkflowReply
    from tldw_chatbook.Chat.console_workflow_runs import _accept_workflow_phase

    mode = sys.argv[2]
    assert mode in {"unborrowed-close", "capture-exit"}
    rig = await build_workflow_rig(
        tmp_path,
        monkeypatch,
        WorkflowProvider(),
        change_review=True,
        public_coordinator=False,
    )
    primary = None
    entered, release = threading.Event(), threading.Event()
    saved, errors = {}, []
    competitor = claim = wave_task = None
    drains = []
    previous_profile, previous_thread_profile = sys.getprofile(), threading.getprofile()
    injected = RuntimeError("original workflow capture callback exit failure")
    try:
        (session,) = rig.store.sessions()
        registry = rig.store.persistence.workspace_registry
        binding = capture_run_admitted_workspace_roots(
            session=session, registry=registry
        )[0]
        selected = await rig.gateway.resolve_for_send(
            rig.controller._provider_selection_for_session(session.id)
        )
        data = json.loads(json.dumps(rig.provider.plan))
        data["tools"] = []
        data["phases"][0]["tasks"][0].update(
            inputs=[], outputs=[], allowed_tools=[], expected_checks=[]
        )
        db, service = rig.service.db, rig.service
        with owned_goal_storage(service):
            workflow = db.workflow_runs.create(
                rig.workflow.request, launch_id="actual-wave-capture"
            )
            workflow = db.workflow_runs.save_plan(
                workflow.id,
                WorkflowPlan.model_validate_json(json.dumps(data)),
                expected_control_revision=workflow.control_revision,
            )
            attempt = db.automatic_work.prepare_workflow_phase(
                workflow.id,
                revision_id=1,
                phase_id="phase-1",
                owner_id=rig.controller._fleet_wake.runtime_owner_id,
                session_id=session.id,
                command_id="actual-wave-capture-approval",
            )
            phase = _accept_workflow_phase(
                rig.controller, attempt.id, session_id=session.id
            )
        prepared = await rig.runtime.agent_bridge._prepare_workflow_reply(
            phase=phase, task_id="task-1", binding=binding, resolution=selected
        )
        auth = prepared.authorization
        source, lifetime, scope = (
            prepared.source,
            prepared._lifetime,
            prepared._source_scope,
        )
        with owned_goal_storage(service), db.connection() as conn:
            reservation = tuple(
                conn.execute(
                    "SELECT id,chain_id,owner_id,kind,amount,state FROM automatic_work_reservations WHERE id=?",
                    (auth.claim.reservation_id,),
                ).fetchone()
            )
            before = db.automatic_work.snapshot(phase.attempt.chain_id)
        capture_code = next(
            code
            for code in _WorkflowWave._open.__code__.co_consts
            if isinstance(code, type(_WorkflowWave._open.__code__))
            and code.co_name == "capture"
        )
        read_code = _PreparedWorkflowReply._read.__code__
        invoke_code = next(
            code
            for code in read_code.co_consts
            if isinstance(code, type(read_code)) and code.co_name == "invoke"
        )
        finish_code = next(
            code
            for code in _PreparedWorkflowReply.close.__code__.co_consts
            if isinstance(code, type(read_code)) and code.co_name == "finish"
        )
        open_code = _WorkflowWave._open.__code__
        constructor_code, release_code = (
            _WorkflowWave.__init__.__code__,
            _WorkflowWave._release.__code__,
        )
        codes = frozenset(
            (
                capture_code,
                invoke_code,
                finish_code,
                constructor_code,
                release_code,
                open_code,
            )
        )

        def observe(frame, event, result):
            if frame.f_code not in codes:
                return
            inject = False
            try:
                normal = event == "return" and dis.opname[
                    frame.f_code.co_code[frame.f_lasti]
                ] in {"RETURN_VALUE", "RETURN_CONST"}
                if frame.f_code is constructor_code and normal:
                    wave = frame.f_locals["self"]
                    if wave.prepared == (prepared,):
                        assert wave.phase is phase and wave.capture_enabled
                        saved["wave"] = wave
                elif frame.f_code is capture_code and normal:
                    wave = saved["wave"]
                    assert (
                        frame.f_locals["self"] is wave
                        and frame.f_locals["first"] is prepared
                    )
                    record = prepared._finite_reads[-1]
                    assert record.prepared is prepared and record.issuing is wave.driver
                    assert type(result) is tuple and result[0] is wave.source
                    assert result[0].issuer is wave.coordinator._native
                    assert (
                        result[0].workflow_wave is wave
                        and result[0].bridge is prepared.bridge
                    )
                    assert (
                        result[0].ordinary_tree
                        is result[0].owner
                        is result[0].claim
                        is None
                    )
                    assert not result[0]._holds and not result[0]._closed.is_set()
                    saved.update(
                        captured=result,
                        capture_read=record,
                        capture_worker=threading.current_thread(),
                    )
                    inject = mode == "capture-exit"
                elif (
                    frame.f_code is finish_code
                    and event == "call"
                    and frame.f_locals.get("self") is prepared
                ):
                    parent = frame.f_back
                    assert (
                        parent.f_code is invoke_code
                        and parent.f_locals["self"] is prepared
                    )
                    record = parent.f_locals["record"]
                    assert record.issuing is saved["wave"].driver
                    saved["close_read"] = record
                elif frame.f_code is invoke_code and normal:
                    record = frame.f_locals["record"]
                    if record is saved.get("capture_read"):
                        assert result[0] is record and result[8] is saved["captured"]
                        saved["capture_return"] = result
                        if mode == "unborrowed-close":
                            expected = (
                                record,
                                prepared,
                                record.issuing,
                                record.callback,
                                record.proof,
                                record.owner,
                                record.claim,
                                record.operation,
                                saved["captured"],
                                None,
                            )
                            assert type(result) is tuple and len(result) == len(
                                expected
                            )
                            assert all(a is b for a, b in zip(result, expected))
                            assert frame.f_locals["self"] is prepared
                            assert record.callback.__code__ is invoke_code
                            owner, actual_claim, operation = result[5:8]
                            assert record.drains == [True] and record.scopes_returned
                            assert record.body_error is None
                            assert threading.current_thread() is saved["capture_worker"]
                            with owner.capacity._lock:
                                assert (
                                    owner.capacity._owners.get(owner.execution_id)
                                    is not owner
                                )
                                assert operation not in owner._operations
                            with prepared.runtime.worktree_activity._lock:
                                assert (
                                    actual_claim.id
                                    not in prepared.runtime.worktree_activity._claims
                                )
                                assert (
                                    actual_claim.id
                                    not in prepared.runtime.worktree_activity._members
                                )
                            saved["capture_retirement"] = (
                                owner,
                                actual_claim,
                                operation,
                            )
                            entered.set()
                            if not release.wait(5):
                                errors.append(
                                    AssertionError(
                                        "original capture retirement hold exceeded5"
                                    )
                                )
                    elif record is saved.get("close_read"):
                        assert result[0] is record
                        saved["close_return"] = result
                elif frame.f_code is open_code and normal:
                    if frame.f_locals["self"] is saved.get("wave"):
                        assert result.source is saved["captured"][0]
                        assert result.footprint is saved["captured"][1]
                        assert not result.retain
                        saved["opening_return"] = result
                elif (
                    frame.f_code is release_code
                    and normal
                    and frame.f_locals["self"] is saved.get("wave")
                ):
                    wave = saved["wave"]
                    assert asyncio.current_task() is wave.driver
                    assert wave.closed and wave not in rig.runtime._workflow_waves
                    saved["release_return"] = True
            except BaseException as error:  # noqa: BLE001 - passive observations cannot alter native callbacks.
                errors.append(error)
                entered.set()
                return
            if inject:
                threading.setprofile_all_threads(None)
                saved["injected"] = True
                raise injected

        threading.setprofile_all_threads(observe)
        wave_task = asyncio.create_task(run_workflow_wave(phase, (prepared,)))
        if mode == "unborrowed-close":
            assert await asyncio.to_thread(entered.wait, 10)
            if errors:
                raise errors[0]
            wave = saved["wave"]
            record = saved["capture_read"]
            read_count = len(prepared._finite_reads)
            assert prepared._wave is wave and rig.runtime._workflow_waves[wave] is wave
            assert not record.proof[1].done()
            with pytest.raises(
                RuntimeError, match="original workflow reader return unproven"
            ):
                await prepared.close()
            assert (
                len(prepared._finite_reads) == read_count
                and not prepared._closing
                and not prepared._closed
            )
            assert (
                prepared.source is source
                and prepared._lifetime is lifetime
                and prepared._source_scope is scope
            )
            competitor = rig.runtime.execution_capacity.begin_execution(
                origin=WorkOrigin.MANUAL, conversation_id="actual-capture-contender"
            )
            competitor.on_drained(drains.append)
            root = Path(workflow.request.binding.locator)
            claim = rig.runtime.worktree_activity.try_acquire(
                owner=competitor,
                footprint=WorktreeFootprint(
                    writes=(WorktreeScope(root, True, capture_directory_chain(root)),)
                ),
            )
            assert claim is not None
            release.set()
            with pytest.raises(
                PermissionError, match="workflow wave activity busy"
            ) as caught:
                await wave_task
            assert wave.driver.done() and wave.driver.exception() is caught.value
            assert caught.value is saved["opening_return"].error
            assert prepared._finite_read_return(record) is saved["capture_return"]
            assert record.positive and record.result is saved["captured"]
            close_read = saved["close_read"]
            assert (
                prepared._finite_read_return(close_read) is saved["close_return"]
                and close_read.positive
            )
            assert (
                saved["release_return"]
                and wave.closed
                and wave not in rig.runtime._workflow_waves
            )
            assert (
                prepared._closed
                and not prepared._closing
                and prepared not in rig.runtime._workflow_preparations
            )
            assert prepared._source_scope is None and scope.gen.gi_frame is None
        else:
            with pytest.raises(
                RuntimeError, match="original workflow capture callback exit failure"
            ) as caught:
                await wave_task
            assert caught.value is injected and saved["injected"]
            wave, record = saved["wave"], saved["capture_read"]
            read, producer, retired, callback = record.proof
            assert read is record.proof[0] and callback is record.callback
            assert (
                read.callback is callback
                and read.source is prepared
                and read.task is wave.driver
            )
            assert (
                producer.done()
                and not producer.cancelled()
                and producer.exception() is None
            )
            assert (
                retired.done() and not retired.cancelled() and retired.result() is None
            )
            assert record.wrapper_error is None and record.positive
            assert record.scopes_returned and record.body_error is injected
            actual = producer.result()
            expected = (
                record,
                prepared,
                wave.driver,
                callback,
                record.proof,
                record.owner,
                record.claim,
                record.operation,
                None,
                injected,
            )
            assert type(actual) is tuple and len(actual) == len(expected)
            assert all(a is b for a, b in zip(actual, expected))
            assert prepared._finite_read_return(record) is actual
            assert wave.driver.done() and wave.driver.exception() is injected
            assert rig.runtime._workflow_waves[wave] is wave and prepared._wave is wave
            assert not wave.closed and not prepared._closed and not prepared._closing
            assert (
                prepared.source is source
                and prepared._lifetime is lifetime
                and prepared._source_scope is scope
            )
            read_count = len(prepared._finite_reads)
            for _ in range(2):
                with pytest.raises(
                    PermissionError, match="original wave close invocation required"
                ):
                    await prepared.close()
                assert (
                    len(prepared._finite_reads) == read_count
                    and prepared._finite_reads[-1] is record
                )
                assert prepared._wave is wave and not prepared._closing
        if errors:
            raise errors[0]
        assert wave.source is saved["captured"][0]
        assert (
            wave.source.ordinary_tree is wave.source.owner is wave.source.claim is None
        )
        with wave.source._holds_lock:
            assert not wave.source._holds and not wave.source._closed.is_set()
        assert wave.fence is wave.reservation is wave.publication is wave.anchor is None
        assert (
            not wave.baseline_return.done()
            and not wave.ending_return.done()
            and not wave.receipts
        )
        assert rig.provider.calls == []
        with owned_goal_storage(service), db.connection() as conn:
            assert (
                tuple(
                    conn.execute(
                        "SELECT id,chain_id,owner_id,kind,amount,state FROM automatic_work_reservations WHERE id=?",
                        (auth.claim.reservation_id,),
                    ).fetchone()
                )
                == reservation
            )
            after = db.automatic_work.snapshot(phase.attempt.chain_id)
            assert dict(after.used) == dict(before.used) and dict(
                after.reserved
            ) == dict(before.reserved)
            assert db.list_runs(workflow.request.conversation_id) == []
        if mode == "capture-exit":
            print(
                "G_CAPTURE_SOURCE_RETAINED_WITH_CALLBACK_ERROR; no creator drain",
                flush=True,
            )
            os._exit(17)
        competitor.finish_root()
        claim.settle(durable=True)
        assert drains == [True] and not competitor.is_admitted
        with rig.runtime.worktree_activity._lock:
            assert (
                claim.id not in rig.runtime.worktree_activity._claims
                and claim.id not in rig.runtime.worktree_activity._members
            )
        competitor = claim = None
    except BaseException as error:
        primary = error
        raise
    finally:
        release.set()
        if wave_task is not None and not wave_task.done():
            try:
                await wave_task
            except BaseException as error:
                if primary is None:
                    raise
                else:
                    primary.add_note("original wave cleanup: " + type(error).__name__)
        threading.setprofile_all_threads(previous_thread_profile)
        sys.setprofile(previous_profile)
        if competitor is not None:
            competitor.finish_root()
        if claim is not None:
            claim.settle(durable=True)
        try:
            await rig.close()
        except BaseException as cleanup:
            if primary is None:
                raise
            primary.add_note("original rig cleanup: " + type(cleanup).__name__)


@pytest.mark.parametrize("mode", ["unborrowed-close", "capture-exit"])
async def test_original_wave_capture_retains_source_until_exact_disposition(
    tmp_path, mode
):
    import subprocess
    from pathlib import Path

    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    script = Path(__file__).with_name("workflow_coordinator_probe.py").read_text()
    if mode == "capture-exit":
        with pytest.raises(
            AssertionError, match="G_CAPTURE_SOURCE_RETAINED_WITH_CALLBACK_ERROR"
        ) as caught:
            await asyncio.to_thread(
                _run,
                tmp_path,
                "wave-capture",
                mode,
                script=script,
                timeout=60,
            )
        frames = []
        tb = caught.value.__traceback__
        while tb is not None:
            if tb.tb_frame.f_code is _run.__code__:
                frames.append(tb.tb_frame)
            tb = tb.tb_next
        assert len(frames) == 1
        original = frames[0].f_locals
        result = original["result"]
        assert original["script"] is script and original["tmp_path"] is tmp_path
        assert type(result) is subprocess.CompletedProcess
        assert result.returncode == 17 and result.args[-2:] == ["wave-capture", mode]
    else:
        await asyncio.to_thread(
            _run,
            tmp_path,
            "wave-capture",
            mode,
            script=Path(__file__)
            .with_name("workflow_coordinator_probe.py")
            .read_text(),
            timeout=60,
        )


async def live_bind_source_journey(tmp_path, monkeypatch):
    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig
    from tldw_chatbook.Agents import agent_service as service_module
    from tldw_chatbook.Agents import workflow_task_scope as task_scope
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
    from tldw_chatbook.Agents.execution_capacity import ExecutionOwner
    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Agents.workflow_models import WorkflowSnapshot
    from tldw_chatbook.Chat.console_workflow_preparation import _PreparedWorkflowReply
    from tldw_chatbook.Chat.console_workflow_wave import _workflow_wave_return
    from tldw_chatbook.Workspaces.change_review_finalization import (
        ChangeReviewCompletion,
    )

    mode = sys.argv[2]
    assert mode in {"before-lookup", "after-helper"}
    rig = await build_workflow_rig(
        tmp_path, monkeypatch, WorkflowProvider(), change_review=True
    )
    saved, errors, replacement_calls, binding_calls = {}, [], [], []
    primary = None
    native_code = AgentService._launch_prepared_child_result.__code__
    start_code = threading.Thread.start.__code__
    run_wrapper = AgentService._run_one
    run_body = run_wrapper.__wrapped__
    bind_run_code = ExecutionOwner.bind_run.__code__
    bind_record, helper_record, checker_record = (
        task_scope._BIND_METHOD,
        task_scope._BIND_SOURCE,
        task_scope._BIND_CHECKER,
    )
    close_code = _PreparedWorkflowReply._close_on_native_owner.__code__
    root_code = AgentService._finish_execution.__code__
    codes = {
        start_code,
        bind_run_code,
        helper_record[1],
        bind_record[1],
        close_code,
        root_code,
    }
    previous_profile, previous_thread_profile = sys.getprofile(), threading.getprofile()
    previous_trace, previous_thread_trace = sys.gettrace(), threading.gettrace()

    def replacement(self, *, owner, activity_claim):
        replacement_calls.append((self, owner, activity_claim))
        raise AssertionError("changed bind receiver executed")

    def original_caller(body):
        assert (
            body.f_code is run_body.__code__
            and body.f_locals["self"] is saved["service"]
        )
        wrapper = body.f_back
        assert (
            wrapper.f_code is run_wrapper.__code__
            and wrapper.f_locals["function"] is run_body
        )
        outer = wrapper.f_back
        assert outer.f_code is AgentService._run_owned.__code__
        assert outer.f_locals["self"] is saved["service"]
        assert outer.f_locals["workflow_task_authorization"] is saved["auth"]
        assert outer.f_locals["workflow_activity_claim"] is saved["claim"]
        assert task_scope._CURRENT.get() is saved["auth"]
        assert threading.current_thread() is saved["thread"]
        assert body.f_locals["owner"] is saved["owner"]
        assert body.f_locals["run_id"] == saved["auth"].claim.launch_run_id
        assert not saved["auth"]._binding_state
        assert (
            not saved["auth"]._lock.locked()
            and not saved["owner"].capacity._lock.locked()
        )
        saved["body"] = body

    def observe(frame, event, result):
        if frame.f_code not in codes and frame.f_code is not saved.get("target_code"):
            return
        try:
            normal = event == "return" and dis.opname[
                frame.f_code.co_code[frame.f_lasti]
            ] in {"RETURN_VALUE", "RETURN_CONST"}
            if (
                frame.f_code is start_code
                and event == "call"
                and frame.f_back.f_code is native_code
            ):
                parent = frame.f_back
                entry = parent.f_locals["entry"]
                if entry is None or entry.mode != "W":
                    return
                run = rig.coordinator._runs[rig.workflow.id]
                thread, target = (
                    frame.f_locals["self"],
                    parent.f_locals["run_child_owned"],
                )
                assert thread._target is target and parent.f_locals["thread"] is thread
                assert (
                    entry.wave.phase is run.phase
                    and entry.preparation.phase is run.phase
                )
                kwargs = parent.f_locals["child_kwargs"]
                assert kwargs["workflow_task_authorization"] is entry.authorization
                saved.update(
                    entry=entry,
                    prepared=entry.preparation,
                    auth=entry.authorization,
                    native=entry.native,
                    service=entry.service,
                    wave=entry.wave,
                    run=run,
                    intake=run.intake,
                    token=run.token,
                    thread=thread,
                    target=target,
                    target_code=target.__code__,
                    child_code=parent.f_locals["run_child"].__code__,
                    owner=parent.f_locals["child_owner"],
                    claim=kwargs["workflow_activity_claim"],
                    source=entry.preparation.source,
                    lifetime=entry.preparation._lifetime,
                    launch=entry.preparation._launch_operation,
                )
            elif (
                frame.f_code is bind_run_code
                and normal
                and frame.f_locals["self"] is saved.get("owner")
            ):
                body = frame.f_back
                if body.f_code is not run_body.__code__:
                    return
                original_caller(body)
                assert saved["owner"].run_id == saved["auth"].claim.launch_run_id
                saved["bound_owner"] = True
                if mode == "before-lookup":
                    assert (
                        task_scope.WorkflowTaskAuthorization.bind_native
                        is bind_record[0]
                    )
                    task_scope.WorkflowTaskAuthorization.bind_native = replacement
                    saved["changed"] = True
            elif (
                frame.f_code is helper_record[1]
                and normal
                and frame.f_back.f_code is run_body.__code__
            ):
                body = frame.f_back
                original_caller(body)
                assert body.f_locals["checked_helper"] is False
                assert body.f_locals["task_authorization"] is saved["auth"]
                assert body.f_locals["bind_native"] is bind_record[0]
                assert body.f_locals["source"] is helper_record[0]
                assert body.f_locals["checker"] is checker_record[0]
                assert (
                    body.f_locals["method_record"] is bind_record
                    and body.f_locals["source_record"] is helper_record
                    and body.f_locals["checker_record"] is checker_record
                )
                assert service_module._WORKFLOW_BINDING_SOURCE[0] is task_scope
                assert (
                    mode == "after-helper"
                    and task_scope.WorkflowTaskAuthorization.bind_native
                    is bind_record[0]
                )
                task_scope.WorkflowTaskAuthorization.bind_native = replacement
                saved["changed"] = True
            elif (
                frame.f_code is bind_record[1]
                and event == "call"
                and frame.f_locals.get("self") is saved.get("auth")
            ):
                binding_calls.append(frame)
            elif (
                frame.f_code is close_code
                and normal
                and frame.f_locals["self"] is saved.get("prepared")
            ):
                assert frame.f_locals["owner"] is saved["owner"]
                saved["native_close_return"] = True
            elif (
                frame.f_code is root_code
                and normal
                and frame.f_locals["owner"] is saved.get("owner")
            ):
                assert frame.f_locals["self"] is saved["service"]
                saved["root_return"] = True
            elif (
                frame.f_code is saved.get("target_code")
                and normal
                and threading.current_thread() is saved["thread"]
            ):
                saved["target_return"] = (frame, result)
        except BaseException as error:  # noqa: BLE001 - observations cannot alter native callbacks.
            errors.append(error)

    def trace(frame, event, value):
        if frame.f_code is not run_body.__code__ and frame.f_code is not saved.get(
            "child_code"
        ):
            return None
        if event == "exception" and saved:
            try:
                error_type, error, _tb = value
                if (
                    frame.f_code is run_body.__code__
                    and frame.f_locals.get("self") is saved["service"]
                ):
                    assert (
                        frame is saved["body"]
                        and threading.current_thread() is saved["thread"]
                    )
                    assert frame.f_locals["task_authorization"] is saved["auth"]
                    assert error_type is AutomaticWorkRefused
                    assert (
                        str(error)
                        == "workflow_task_original_native_binding_source_changed"
                    )
                    saved["source_error"] = error
                elif (
                    frame.f_code is saved["child_code"]
                    and threading.current_thread() is saved["thread"]
                ):
                    assert error is saved["source_error"]
                    saved["child_error"] = error
            except BaseException as error:  # noqa: BLE001 - trace errors remain separate from native outcome.
                errors.append(error)
        return trace

    try:
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        await asyncio.shield(rig.coordinator._runs[rig.workflow.id].driver)
        await rig.wait_for_status("awaiting_phase_review")
        decision = await rig.service.call(
            rig.service.prepare_decision, rig.workflow.id, phase_id="phase-1"
        )
        threading.setprofile_all_threads(observe)
        threading.settrace(trace)
        await rig.coordinator.approve(decision, command_id="original-bind-source")
        run = rig.coordinator._runs[rig.workflow.id]
        await asyncio.shield(run.driver)
        if errors:
            raise errors[0]
        assert saved["changed"] and saved["bound_owner"] and saved["run"] is run
        saved["thread"].join(5)
        assert not saved["thread"].is_alive() and saved["target_return"][1] is None
        assert saved["native_close_return"] and saved["root_return"]
        assert (
            saved["source_error"] is saved["child_error"]
            and not replacement_calls
            and not binding_calls
        )
        prepared, auth, wave, owner = (
            saved["prepared"],
            saved["auth"],
            saved["wave"],
            saved["owner"],
        )
        assert (
            not auth._binding_state
            and prepared._native_closed
            and prepared._close_operation is None
        )
        assert prepared._launch_operation is None and not owner.is_admitted
        assert (
            prepared.source is saved["source"]
            and prepared._lifetime is saved["lifetime"]
        )
        assert len(wave.receipts) == 1 and wave.receipts[0][0] is prepared
        receipt = wave.receipts[0][1]
        assert (
            receipt is auth._native_launch[0]
            and receipt.owner is owner
            and receipt.activity_claim is saved["claim"]
        )
        assert receipt.failure is None and receipt.run_id == auth.claim.launch_run_id
        tracker = rig.service._tasks[auth.claim.id]
        assert tracker.receipt is receipt and tracker.drain is True
        assert wave.driver.done() and not wave.driver.cancelled()
        assert wave.driver.exception() is None
        assert wave.driver.result() is wave.reservation
        assert wave.closed and wave not in rig.runtime._workflow_waves
        assert run.driver.done() and not run.driver.cancelled()
        assert run.driver.exception() is None
        assert len(run.waves) == 1
        returned_wave = run.waves[0]
        assert returned_wave[0] is wave and returned_wave[1] is wave.driver
        assert returned_wave[6][0][0] is prepared and returned_wave[6][0][1] is receipt
        assert _workflow_wave_return(wave, saved=returned_wave) is returned_wave
        assert wave.baseline_return.result() is ChangeReviewCompletion.OBSERVED
        assert wave.anchor is None and not wave.ending_return.done()
        assert wave.reservation.completion is ChangeReviewCompletion.CANCELLED
        assert prepared._closed and prepared._source_scope is None
        close_read = prepared._finite_reads[-1]
        prepared._finite_read_return(close_read)
        assert close_read.positive and close_read.wrapper_error is None
        assert close_read.body_error is None
        activity = rig.runtime.worktree_activity
        with activity._lock:
            assert saved["claim"].id not in activity._claims
            assert saved["claim"].id not in activity._members
        with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
            state = conn.execute(
                "SELECT state FROM automatic_work_reservations WHERE id=?",
                (auth.claim.reservation_id,),
            ).fetchone()[0]
            task = conn.execute(
                "SELECT run_id,status,retirement FROM workflow_task_attempts WHERE id=?",
                (auth.claim.id,),
            ).fetchone()
            assert state == "committed" and tuple(task) == (
                receipt.run_id,
                "failed",
                "drained",
            )
            result = conn.execute(
                "SELECT json_extract(payload_json,'$.run_id'),json_extract(payload_json,'$.status') "
                "FROM workflow_results WHERE attempt_id=?",
                (auth.claim.id,),
            ).fetchall()
            checkpoint = conn.execute(
                "SELECT json_extract(payload_json,'$.run_id') FROM workflow_checkpoints WHERE attempt_id=?",
                (auth.claim.id,),
            ).fetchall()
            assert [tuple(row) for row in result] == [(receipt.run_id, "failed")]
            assert [tuple(row) for row in checkpoint] == [(receipt.run_id,)]
            assert auth.phase is run.phase
            assert (
                auth.claim.phase_attempt_id
                == auth.phase.attempt.id
                == run.phase.attempt.id
            )
            phase = conn.execute(
                "SELECT status,retirement FROM workflow_phase_attempts WHERE native_attempt_id=?",
                (auth.claim.phase_attempt_id,),
            ).fetchone()
            assert tuple(phase) == ("failed", "drained")
            window = conn.execute(
                "SELECT state,baseline,ending,snapshot_count FROM workflow_change_windows WHERE id=?",
                (wave.reservation.id,),
            ).fetchone()
            assert tuple(window) == ("unused", "completed", "unused", 0)
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM workflow_change_window_refs WHERE window_id=?",
                    (wave.reservation.id,),
                ).fetchone()[0]
                == 0
            )
            assert rig.service.db.get_run(receipt.run_id)["status"] == "error"
        snapshot = await rig.coordinator.get(rig.workflow.id)
        assert type(snapshot) is WorkflowSnapshot and snapshot.status == "paused"
        assert [
            (result.workflow_id, result.attempt_id, result.run_id, result.status)
            for result in snapshot.results
        ] == [(rig.workflow.id, auth.claim.id, receipt.run_id, "failed")]
        durable = await rig.service.call(rig.service.get, rig.workflow.id)
        assert durable.status == "awaiting_resume"
        assert snapshot.accounting.used["generation"] == 2
        assert (
            snapshot.accounting.used["child_launch"] == 1
            and snapshot.accounting.used["model_call"] == 1
        )
        assert [key for key, _ in rig.provider.calls] == ["planning"]
        assert rig.workflow.id not in rig.coordinator._runs
        assert run.intake is saved["intake"] and run.token is saved["token"]
        with rig.store._preparation_lock:
            assert (
                rig.store._preparations_by_session.get(run.session_id)
                is not saved["intake"]
            )
        wake = rig.controller._fleet_wake
        with wake._registry_lock:
            assert (
                wake._automatic_primary_claims.get(run.session_id) is not saved["token"]
            )
        assert rig.runtime.execution_capacity.snapshot().executions == ()
    except BaseException as error:
        primary = error
        raise
    finally:
        threading.setprofile_all_threads(previous_thread_profile)
        sys.setprofile(previous_profile)
        threading.settrace(previous_thread_trace)
        sys.settrace(previous_trace)
        rig.provider_release()
        if "thread" in saved:
            saved["thread"].join(5)
        try:
            await rig.close()
        except BaseException as cleanup:
            if primary is None:
                raise
            primary.add_note(
                "original live-bind rig cleanup: " + type(cleanup).__name__
            )


@pytest.mark.parametrize("mode", ["before-lookup", "after-helper"])
async def test_original_live_bind_receiver_refuses_replacement_before_invocation(
    tmp_path, mode
):
    from pathlib import Path

    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    await asyncio.to_thread(
        _run,
        tmp_path,
        "live-bind",
        mode,
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


async def callback_baseexception_journey(tmp_path, _unused):
    import os
    from types import SimpleNamespace

    from tldw_chatbook.Agents.agent_service import _NativeWorkflowReturn
    from tldw_chatbook.Agents.execution_capacity import ExecutionOwner

    class CallbackReturnUnproven(BaseException):
        pass

    generator = launch = None
    primary = None
    entered, release = threading.Event(), threading.Event()
    saved, errors, normal_finishes = {}, [], []
    constructor_error = RuntimeError("original constructor before callback failure")
    callback_error = CallbackReturnUnproven("original callback did not return")
    native_code = AgentService._launch_prepared_child_result.__code__
    own_code = AgentService._workflow_launch_result.__code__
    finish_code = AgentService._finish_workflow_launch.__code__
    constructor_code = threading.Thread.__init__.__code__
    previous_thread_trace, previous_trace = threading.gettrace(), sys.gettrace()
    previous_thread_profile, previous_profile = threading.getprofile(), sys.getprofile()
    with pytest.MonkeyPatch.context() as monkeypatch:
        try:
            generator = native_tests.native_host.__wrapped__(
                tmp_path, monkeypatch, SimpleNamespace(param=None)
            )
            h = await anext(generator)
            prepared = await h.bridge._prepare_workflow_reply(
                phase=h.phase,
                task_id="task-1",
                binding=h.binding,
                resolution=h.selected,
            )
            auth = prepared.authorization
            native = auth._native_preparation[0]
            source, lifetime, source_scope = (
                prepared.source,
                prepared._lifetime,
                prepared._source_scope,
            )
            before = h.db.automatic_work.snapshot(auth.claim.chain_id)

            def inject(frame, event, value):
                if event != "call" or frame.f_code is not constructor_code:
                    return
                parent = frame.f_back
                if parent is None or parent.f_code is not native_code:
                    return
                entry = parent.f_locals.get("entry")
                if entry is None or entry.preparation is not prepared:
                    return
                try:
                    assert (
                        entry.mode == "D"
                        and entry.authorization is auth
                        and entry.native is native
                    )
                    assert parent.f_locals["thread"] is None
                    assert parent.f_locals["start_entered"] is False
                    assert parent.f_locals["native_thread_started"] is False
                    assert (
                        frame.f_locals["target"] is parent.f_locals["run_child_owned"]
                    )
                    saved.update(
                        entry=entry,
                        owner=parent.f_locals["child_owner"],
                        claim=parent.f_locals["child_kwargs"][
                            "workflow_activity_claim"
                        ],
                        operation=prepared._launch_operation,
                    )
                except BaseException as error:  # noqa: BLE001 - observations never supply the deliberate fault.
                    errors.append(error)
                    return
                threading.settrace_all_threads(None)
                sys.settrace(None)
                raise constructor_error

            def callback(outcome):
                try:
                    owner, operation = saved["owner"], saved["operation"]
                    assert (
                        outcome is True
                        and threading.current_thread() is saved["worker"]
                    )
                    with owner.capacity._lock:
                        assert owner.execution_id not in owner.capacity._owners
                        assert (
                            operation not in owner._operations and owner._root_finished
                        )
                    assert not auth._lock.locked()
                    assert prepared._launch_operation is operation
                    saved["callback"] = (
                        threading.current_thread(),
                        owner,
                        operation,
                        outcome,
                    )
                except BaseException as error:  # noqa: BLE001 - observer errors cannot substitute for native failure.
                    errors.append(error)
                    return
                raise callback_error

            codes = {native_code, own_code, finish_code}

            def observe(frame, event, result):
                if (
                    frame.f_code not in codes
                    or event != "return"
                    or dis.opname[frame.f_code.co_code[frame.f_lasti]]
                    not in {"RETURN_VALUE", "RETURN_CONST"}
                ):
                    return
                try:
                    if frame.f_code is native_code:
                        if (
                            not saved
                            or frame.f_locals.get("entry") is not saved["entry"]
                        ):
                            return
                        assert type(result) is _NativeWorkflowReturn and result.known
                        assert frame.f_locals["thread"] is None
                        assert frame.f_locals["start_entered"] is False
                        assert frame.f_locals["native_thread_started"] is False
                        assert result.receipt.failure == "thread_start"
                        assert result.receipt.owner is saved["owner"]
                        assert result.receipt.activity_claim is saved["claim"]
                        saved["native_return"] = result
                    elif frame.f_code is own_code:
                        if (
                            type(result) is not _WorkflowLaunchReturn
                            or result.entry.preparation is not prepared
                        ):
                            return
                        assert (
                            result.entry is saved["entry"]
                            and result.known
                            and not result.wait
                            and not result.duplicate
                        )
                        actual = saved["native_return"]
                        assert (
                            result.error is actual.error
                            and result.receipt is actual.receipt
                        )
                        assert (
                            result.operation
                            is saved["operation"]
                            is prepared._launch_operation
                        )
                        assert (
                            result.owner is saved["owner"]
                            and result.claim is saved["claim"]
                        )
                        assert (
                            prepared._native_closed
                            and prepared._close_operation is None
                        )
                        assert not native.threads and not native.uncertain_start
                        saved.update(result=result, worker=threading.current_thread())
                        entered.set()
                        if not release.wait(5):
                            errors.append(
                                AssertionError("original own-return hold exceeded5s")
                            )
                    elif saved and frame.f_locals.get("result") is saved.get("result"):
                        normal_finishes.append(frame)
                except BaseException as error:  # noqa: BLE001 - native callbacks retain their original outcome.
                    errors.append(error)
                    entered.set()

            threading.settrace_all_threads(inject)
            threading.setprofile_all_threads(observe)
            launch = asyncio.create_task(
                native_tests._launch_native_for_test(h, prepared)
            )
            assert await asyncio.to_thread(entered.wait, 5)
            if errors:
                raise errors[0]
            result = saved["result"]
            owner, operation, claim = result.owner, result.operation, result.claim
            with owner.capacity._lock:
                assert owner._root_finished and owner._operations == {operation}
                assert owner.capacity._owners.get(owner.execution_id) is owner
            owner.on_drained(callback)
            release.set()
            with pytest.raises(CallbackReturnUnproven) as caught:
                await launch
            assert (
                caught.value is callback_error and launch.exception() is callback_error
            )
            if errors:
                raise errors[0]
            assert not normal_finishes
            assert saved["callback"][0] is saved["worker"]
            assert saved["callback"][1] is owner and saved["callback"][2] is operation
            assert saved["callback"][3] is True
            frames = []
            tb = caught.value.__traceback__
            while tb is not None:
                frames.append(tb.tb_frame)
                tb = tb.tb_next
            assert any(
                frame.f_code is callback.__code__ and frame.f_locals["outcome"] is True
                for frame in frames
            )
            assert any(
                frame.f_code is ExecutionOwner._invoke_callbacks.__code__
                for frame in frames
            )
            assert any(
                frame.f_code is OwnedOperation.finish.__code__
                and frame.f_locals["self"] is operation
                for frame in frames
            )
            assert any(
                frame.f_code is finish_code and frame.f_locals["result"] is result
                for frame in frames
            )
            threading.setprofile_all_threads(previous_thread_profile)
            sys.setprofile(previous_profile)
            threading.settrace_all_threads(previous_thread_trace)
            sys.settrace(previous_trace)
            assert (
                result.error is saved["native_return"].error
                and result.error is not callback_error
            )
            assert result.error.receipt is result.receipt
            assert result.receipt is auth._native_launch[0]
            assert (
                prepared._launch_operation is operation
                and prepared._close_operation is None
            )
            assert (
                prepared._native_closed
                and not native.threads
                and not native.uncertain_start
            )
            with owner.capacity._lock:
                assert (
                    owner.execution_id not in owner.capacity._owners
                    and operation not in owner._operations
                )
            activity = h.runtime.worktree_activity
            with activity._lock:
                state = activity._claims[claim.id]
                member = activity._members[claim.id]
                assert state.custody is claim and state.members[claim.id] is member
                assert (
                    member.owner is owner
                    and member.drained is True
                    and member.durable is False
                )
            assert h.db.get_run(result.receipt.run_id)["status"] == "error"
            with h.db.connection() as conn:
                assert (
                    conn.execute(
                        "SELECT state FROM automatic_work_reservations WHERE id=?",
                        (auth.claim.reservation_id,),
                    ).fetchone()[0]
                    == "released"
                )
                assert (
                    conn.execute(
                        "SELECT 1 FROM workflow_results WHERE attempt_id=? UNION ALL SELECT 1 FROM workflow_checkpoints WHERE attempt_id=?",
                        (auth.claim.id, auth.claim.id),
                    ).fetchall()
                    == []
                )
            current = h.db.automatic_work.snapshot(auth.claim.chain_id)
            assert dict(current.used) == dict(before.used)
            assert (
                current.reserved["child_launch"] == 0
                and before.reserved["child_launch"] == 1
            )
            assert h.calls == []
            for _ in range(2):
                count = len(prepared._finite_reads)
                with pytest.raises(
                    PermissionError, match="workflow launch return has not completed"
                ) as closing:
                    await prepared.close()
                assert len(prepared._finite_reads) == count + 1
                record = prepared._finite_reads[-1]
                assert (
                    record.prepared is prepared
                    and record.issuing is asyncio.current_task()
                )
                read, producer, retired, original_callback = record.proof
                assert record.callback is original_callback is read.callback
                assert (
                    read.source is prepared
                    and read.task is record.issuing
                    and read.creator is h.controller
                )
                assert read._producer is producer and read.retired is retired
                assert (
                    producer.done()
                    and not producer.cancelled()
                    and producer.exception() is None
                )
                assert (
                    record.wrapper_error is None
                    and record.body_error is closing.value
                    and record.positive
                    and record.scopes_returned
                )
                actual_close = producer.result()
                expected_close = (
                    record,
                    prepared,
                    record.issuing,
                    original_callback,
                    record.proof,
                    None,
                    None,
                    None,
                    None,
                    closing.value,
                )
                assert type(actual_close) is tuple and len(actual_close) == len(
                    expected_close
                )
                assert all(a is b for a, b in zip(actual_close, expected_close))
                assert prepared in h.runtime._workflow_preparations
                assert (
                    prepared.source is source
                    and prepared._lifetime is lifetime
                    and prepared._source_scope is source_scope
                )
                assert (
                    prepared._launch_operation is operation
                    and not prepared._closed
                    and not prepared._closing
                )
                assert (
                    len(auth._native_launch) == 1
                    and auth._native_launch[0] is result.receipt
                    and not native.threads
                    and not native.uncertain_start
                )
                assert prepared._native_closed and prepared._close_operation is None
                assert h.calls == []
                counts = h.db.automatic_work.snapshot(auth.claim.chain_id)
                assert dict(counts.used) == dict(current.used) and dict(
                    counts.reserved
                ) == dict(current.reserved)
                with activity._lock:
                    assert (
                        activity._claims[claim.id] is state
                        and activity._members[claim.id] is member
                    )
                    assert member.drained is True and member.durable is False
                with h.db.connection() as conn:
                    assert (
                        conn.execute(
                            "SELECT 1 FROM workflow_results WHERE attempt_id=? UNION ALL SELECT 1 FROM workflow_checkpoints WHERE attempt_id=?",
                            (auth.claim.id, auth.claim.id),
                        ).fetchall()
                        == []
                    )
            print(
                "G_POST_REMOVAL_CALLBACK_RETAINED; no fixture or creator return",
                flush=True,
            )
            os._exit(17)
        except BaseException as error:
            primary = error
            raise
        finally:
            release.set()
            try:
                if launch is not None:
                    try:
                        await launch
                    except BaseException as error:
                        if primary is None:
                            raise
                        primary.add_note(
                            "original launch cleanup: " + type(error).__name__
                        )
            finally:
                threading.setprofile_all_threads(previous_thread_profile)
                sys.setprofile(previous_profile)
                threading.settrace_all_threads(previous_thread_trace)
                sys.settrace(previous_trace)
                if generator is not None:
                    try:
                        await generator.aclose()
                    except BaseException as error:
                        if primary is None:
                            raise
                        primary.add_note(
                            "original fixture cleanup: " + type(error).__name__
                        )


async def test_original_callback_baseexception_retains_removed_operation_custody(
    tmp_path,
):
    import subprocess
    from pathlib import Path

    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    route, outcome = "callback-baseexception", "retained"
    script = Path(__file__).with_name("workflow_coordinator_probe.py").read_text()
    with pytest.raises(
        AssertionError, match="G_POST_REMOVAL_CALLBACK_RETAINED"
    ) as caught:
        await asyncio.to_thread(
            _run, tmp_path, route, outcome, script=script, timeout=60
        )
    frames = []
    tb = caught.value.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code is _run.__code__:
            frames.append(tb.tb_frame)
        tb = tb.tb_next
    assert len(frames) == 1
    original = frames[0].f_locals
    assert original["tmp_path"] is tmp_path and original["script"] is script
    assert original["route"] == route and original["outcome"] == outcome
    process_result = original["result"]
    assert type(process_result) is subprocess.CompletedProcess
    assert process_result.returncode == 17 and process_result.args[-2:] == [
        route,
        outcome,
    ]


@pytest.mark.parametrize(
    "mode", ["value", "error", "async-error", "cancel", "async-cancel", "stop"]
)
async def test_original_finite_callback_outcome_crosses_all_scopes(native_host, mode):
    from contextvars import ContextVar

    from tldw_chatbook.Agents import activation
    from tldw_chatbook.Agents.automatic_work_runtime import current_automatic_work
    from tldw_chatbook.Agents.execution_capacity import current_execution_owner
    from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree
    from tldw_chatbook.Chat.console_workflow_preparation import (
        _PreparedWorkflowReply,
        current_workflow_preparation,
    )

    h = native_host
    prepared = await h.bridge._prepare_workflow_reply(
        phase=h.phase, task_id="task-1", binding=h.binding, resolution=h.selected
    )
    before = h.db.automatic_work.snapshot(prepared.authorization.claim.chain_id)
    value = object()
    error = (
        asyncio.CancelledError("original callback cancellation")
        if "cancel" in mode
        else StopIteration("original callback stop")
        if mode == "stop"
        else RuntimeError("original finite callback failure")
    )
    marker = ContextVar("finite_test_caller", default=None)
    token = marker.set(value)
    task = asyncio.current_task()
    saved, notices, drained, observations = {}, [], [], []
    invoke_code = next(
        code
        for code in _PreparedWorkflowReply._read.__code__.co_consts
        if isinstance(code, type(_PreparedWorkflowReply._read.__code__))
        and code.co_name == "invoke"
    )

    def context():
        return (
            current_execution_owner(),
            current_automatic_work(),
            current_workflow_preparation(),
            current_ordinary_tree(),
            activation._guard.context.get(),
            marker.get(),
        )

    def observe(frame, event, _result):
        if frame.f_code is not invoke_code or event != "call":
            return
        try:
            if frame.f_locals.get("self") is prepared:
                assert not saved
                saved.update(
                    record=frame.f_locals["record"],
                    ambient=context(),
                    worker=threading.current_thread(),
                )
        except BaseException as failure:  # noqa: BLE001 - retain the original error without throwing into native observation or masking cleanup.
            observations.append(failure)

    def available():
        try:
            caller = sys._getframe(1).f_back
            record = saved.get("record")
            if (
                record is not None
                and caller.f_code is OwnedOperation.finish.__code__
                and caller.f_locals.get("self") is record.operation
            ):
                notices.append((context(), threading.current_thread()))
        except BaseException as failure:  # noqa: BLE001 - retain the original error without throwing into native observation or masking cleanup.
            observations.append(failure)

    def on_drained(proven):
        drained.append((proven, context(), threading.current_thread()))

    def enter_body():
        owner, claim = prepared._finite
        saved.update(owner=owner, claim=claim, body=context())
        assert owner is current_execution_owner()
        assert current_workflow_preparation() is prepared
        assert current_automatic_work() is h.phase.context
        owner.on_drained(on_drained)

    async def pending():
        enter_body()
        raise error

    def callback():
        if mode.startswith("async-"):
            return pending()
        enter_body()
        if mode != "value":
            raise error
        return value

    previous_thread_profile, previous_profile = threading.getprofile(), sys.getprofile()
    unsubscribe = h.runtime.execution_capacity.subscribe_tool_availability(available)
    threading.setprofile_all_threads(observe)
    try:
        if mode == "value":
            assert await prepared._read(callback) is value
        elif mode == "stop":
            with pytest.raises(
                RuntimeError, match="coroutine raised StopIteration"
            ) as caught:
                await prepared._read(callback)
            assert caught.value.__cause__ is error
        else:
            with pytest.raises(type(error)) as caught:
                await prepared._read(callback)
            assert caught.value is error
        assert asyncio.current_task() is task and marker.get() is value
    finally:
        threading.setprofile_all_threads(previous_thread_profile)
        sys.setprofile(previous_profile)
        unsubscribe()
        marker.reset(token)
    if observations:
        raise observations[0]
    record = saved["record"]
    assert record is prepared._finite_reads[-1]
    assert record.positive and record.scopes_returned and record.wrapper_error is None
    assert record.body_error is (None if mode == "value" else error)
    actual = record.proof[1].result()
    assert actual[:8] == (
        record,
        prepared,
        task,
        record.callback,
        record.proof,
        saved["owner"],
        saved["claim"],
        record.operation,
    )
    assert actual[8] is (value if mode == "value" else None)
    assert actual[9] is record.body_error
    assert record.proof[1].done() and record.proof[1].exception() is None
    assert record.proof[2].done() and record.proof[2].result() is None
    assert notices == [(saved["ambient"], saved["worker"])]
    assert drained == [(True, saved["ambient"], saved["worker"])]
    assert saved["body"][:4] != saved["ambient"][:4]
    assert not saved["owner"].is_admitted and prepared._finite is None
    with h.runtime.worktree_activity._lock:
        assert saved["claim"].id not in h.runtime.worktree_activity._claims
        assert saved["claim"].id not in h.runtime.worktree_activity._members
    assert h.calls == [] and h.db.list_runs(h.conversation_id) == []
    after = h.db.automatic_work.snapshot(before.chain_id)
    assert after.used == before.used and after.reserved == before.reserved
    await prepared.close()
    assert prepared._closed and prepared._source_scope is None
    assert prepared not in h.runtime._workflow_preparations


async def terminal_configuration_exit_journey(tmp_path, _unused):
    """A closed native frame does not certify its later configuration exit."""
    import os
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.Chat.console_workflow_preparation import _PreparedWorkflowReply

    generator = primary = None
    saved, errors, releases = {}, [], []
    old_trace, old_sys_trace = threading.gettrace(), sys.gettrace()
    old_profile, old_sys_profile = threading.getprofile(), sys.getprofile()
    with pytest.MonkeyPatch.context() as monkeypatch:
        try:
            generator = native_tests.native_host.__wrapped__(
                tmp_path, monkeypatch, SimpleNamespace(param=None)
            )
            h = await anext(generator)
            cancel = threading.Event()
            prepared = await h.bridge._prepare_workflow_reply(
                phase=h.phase,
                task_id="task-1",
                binding=h.binding,
                resolution=h.selected,
                should_cancel=cancel.is_set,
            )
            scope, registry = prepared._source_scope, h.runtime._workflow_preparations
            source, native, lifetime, auth = (
                prepared.source,
                prepared.native,
                prepared._lifetime,
                prepared.authorization,
            )
            scope_frame = scope.gen.gi_frame
            assert (
                scope_frame.f_code
                is ConsoleRuntime._first_send_source_scope.__wrapped__.__code__
            )
            before = h.db.automatic_work.snapshot(auth.claim.chain_id)
            cancel.set()
            with pytest.raises(
                PermissionError, match="workflow native source changed"
            ) as refused:
                await prepared._read(lambda: pytest.fail("unissued callback ran"))
            first = prepared._finite_reads[-1]
            assert (
                first.wrapper_error is refused.value and first.before_issue is not None
            )
            assert first.proof is first.owner is first.claim is first.operation is None
            assert first.before_issue[0] is first and first.before_issue[1] is prepared
            assert first.issue[0].cr_frame is None and not first.issue[0].cr_running
            assert not any(
                name in first.issue[1].f_locals
                for name in ("read", "producer", "submit_resolved")
            )
            assert prepared._source_scope is scope and prepared in registry
            task, host = asyncio.current_task(), threading.current_thread()

            def observe(frame, event, _arg):
                if (
                    event == "call"
                    and frame.f_code
                    is _PreparedWorkflowReply._release_configuration.__code__
                    and frame.f_locals.get("self") is prepared
                ):
                    releases.append(frame)

            def trace(frame, event, _arg):
                if frame is not scope_frame or saved.get("injected"):
                    return trace
                if event != "return" or dis.opname[
                    frame.f_code.co_code[frame.f_lasti]
                ] not in {"RETURN_VALUE", "RETURN_CONST"}:
                    return trace
                inject = False
                try:
                    assert (
                        threading.current_thread() is host
                        and asyncio.current_task() is task
                    )
                    assert frame.f_locals["self"] is h.runtime
                    context_exit = frame.f_back
                    assert (
                        context_exit.f_code
                        is contextlib._GeneratorContextManager.__exit__.__code__
                    )
                    assert context_exit.f_locals["self"] is scope
                    release_frame = context_exit.f_back
                    assert (
                        release_frame.f_code
                        is _PreparedWorkflowReply._release_configuration.__code__
                    )
                    assert release_frame.f_locals["self"] is prepared
                    assert release_frame.f_locals["scope"] is scope
                    assert release_frame.f_locals["registry"] is registry
                    close_frame = release_frame.f_back
                    assert close_frame.f_code is _PreparedWorkflowReply.close.__code__
                    assert close_frame.f_locals["self"] is prepared
                    closing = prepared._finite_reads[-1]
                    assert (
                        closing is not first
                        and closing.positive
                        and closing.scopes_returned
                    )
                    assert closing.wrapper_error is closing.body_error is None
                    actual = prepared._finite_read_return(closing)
                    assert actual is closing.proof[1].result() and actual[0] is closing
                    assert closing.proof[2].done() and closing.proof[2].result() is None
                    assert prepared._closed and prepared._native_closed
                    assert prepared._source_scope is None and prepared in registry
                    assert prepared._terminal_configuration_return is None
                    saved.update(injected=True, closing=closing, actual=actual)
                    inject = True
                except BaseException as error:  # noqa: BLE001 - observation cannot change native configuration cleanup.
                    errors.append(error)
                if inject:
                    threading.settrace_all_threads(None)
                    sys.settrace(None)
                    raise refused.value
                return trace

            threading.setprofile_all_threads(observe)
            threading.settrace_all_threads(trace)
            with pytest.raises(PermissionError) as closing_error:
                await prepared.close()
            assert closing_error.value is refused.value
            assert saved.get("injected") and errors == []
            assert len(releases) == 1 and scope.gen.gi_frame is None
            records = tuple(prepared._finite_reads)
            for _ in range(2):
                with pytest.raises(
                    RuntimeError, match="original workflow read issue unproven"
                ):
                    await prepared.close()
                assert tuple(prepared._finite_reads) == records
                assert prepared._finite_reads[-1] is saved["closing"]
                assert prepared._finite_read_return(saved["closing"]) is saved["actual"]
                assert prepared._terminal_configuration_return is None
                assert prepared._source_scope is None and prepared in registry
                assert h.runtime._workflow_preparations is registry
                assert prepared.source is source and prepared.native is native
                assert prepared._lifetime is lifetime and prepared.authorization is auth
                assert prepared._closed and prepared._native_closed
                assert len(releases) == 1
            assert h.calls == [] and h.db.list_runs(h.conversation_id) == []
            after = h.db.automatic_work.snapshot(before.chain_id)
            assert after.used == before.used and after.reserved == before.reserved
            print(
                "G_TERMINAL_CONFIGURATION_RETURN_RETAINED; no fixture/creator drain",
                flush=True,
            )
            os._exit(17)
        except BaseException as error:
            primary = error
            raise
        finally:
            threading.settrace_all_threads(old_trace)
            sys.settrace(old_sys_trace)
            threading.setprofile_all_threads(old_profile)
            sys.setprofile(old_sys_profile)
            if generator is not None:
                try:
                    await generator.aclose()
                except BaseException as error:
                    if primary is None:
                        raise
                    primary.add_note(
                        "Original fixture cleanup also failed: " + type(error).__name__
                    )
                    primary.cleanup_error = error


async def test_original_configuration_exit_error_cannot_certify_terminal_close(
    tmp_path,
):
    import subprocess
    from pathlib import Path

    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    route = "terminal-config-exit"
    script = Path(__file__).with_name("workflow_coordinator_probe.py").read_text()
    with pytest.raises(
        AssertionError, match="G_TERMINAL_CONFIGURATION_RETURN_RETAINED"
    ) as caught:
        await asyncio.to_thread(
            _run, tmp_path, route, "retained", script=script, timeout=60
        )
    frames = []
    tb = caught.value.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code is _run.__code__:
            frames.append(tb.tb_frame)
        tb = tb.tb_next
    assert len(frames) == 1
    original = frames[0].f_locals
    result = original["result"]
    assert original["script"] is script and original["tmp_path"] is tmp_path
    assert type(result) is subprocess.CompletedProcess
    assert result.returncode == 17 and result.args[-2:] == [route, "retained"]


async def finite_preentry_failure_journey(tmp_path, _unused):
    """The original configuration root survives an unreturned first capture."""
    import os
    from types import SimpleNamespace

    from tldw_chatbook.Chat import console_preparation_reads as native_reads
    from tldw_chatbook.Chat import console_workflow_runs as runs
    from tldw_chatbook.Chat.console_workflow_preparation import _PreparedWorkflowReply

    saved, errors = {}, []
    sentinel = RuntimeError("original first preparation capture failure")
    previous_thread_trace, previous_trace = threading.gettrace(), sys.gettrace()
    invoke_code = next(
        code
        for code in _PreparedWorkflowReply._read.__code__.co_consts
        if isinstance(code, type(runs._capture_workflow_read.__code__))
        and code.co_name == "invoke"
    )
    generator = primary = None
    with pytest.MonkeyPatch.context() as monkeypatch:
        try:
            generator = native_tests.native_host.__wrapped__(
                tmp_path, monkeypatch, SimpleNamespace(param=None)
            )
            h = await anext(generator)
            before = h.db.automatic_work.snapshot(h.phase.attempt.chain_id)

            def trace(frame, event, _arg):
                if (
                    event != "call"
                    or frame.f_code is not runs._capture_workflow_read.__code__
                    or saved.get("injected")
                ):
                    return trace
                inject = False
                try:
                    prepared = frame.f_locals.get("source")
                    if (
                        type(prepared) is not _PreparedWorkflowReply
                        or prepared.phase is not h.phase
                        or prepared.controller is not h.controller
                        or prepared.runtime is not h.runtime
                        or prepared.task_id != "task-1"
                    ):
                        return trace
                    caller = frame.f_back
                    assert caller.f_code is invoke_code
                    record = caller.f_locals["record"]
                    assert prepared._finite_reads == [record]
                    assert record.prepared is prepared
                    assert record.callback is caller.f_locals["invoke"]
                    assert frame.f_locals["callback"] is record.callback
                    assert frame.f_locals["issuing"] is record.issuing
                    binding = native_reads._source_binding.get()
                    assert (
                        binding.active and binding.thread is threading.current_thread()
                    )
                    read = binding.read
                    assert type(read) is native_reads.ConsolePreparationRead
                    assert read.source is prepared and read.creator is h.controller
                    assert (
                        read.callback is record.callback and read.task is record.issuing
                    )
                    assert prepared._source_scope is not None
                    assert prepared in h.runtime._workflow_preparations
                    assert (
                        prepared.source is prepared.native is prepared._lifetime is None
                    )
                    assert prepared.authorization is None
                    saved.update(
                        prepared=prepared,
                        record=record,
                        read=read,
                        producer=read._producer,
                        retired=read.retired,
                        scope=prepared._source_scope,
                        registry=h.runtime._workflow_preparations,
                        injected=True,
                    )
                    inject = True
                except BaseException as error:  # noqa: BLE001 - retain the original error without throwing into native observation or masking cleanup.
                    errors.append(error)
                if inject:
                    threading.settrace_all_threads(None)
                    sys.settrace(None)
                    raise sentinel
                return trace

            threading.settrace_all_threads(trace)
            with pytest.raises(
                RuntimeError, match="original workflow read return missing"
            ):
                await h.bridge._prepare_workflow_reply(
                    phase=h.phase,
                    task_id="task-1",
                    binding=h.binding,
                    resolution=h.selected,
                )
            assert errors == [] and saved.get("injected") is True
            prepared, record, read = saved["prepared"], saved["record"], saved["read"]
            assert prepared._finite_reads == [record]
            assert record.wrapper_error is sentinel
            assert record.proof is record.before_issue is record.owner is None
            assert record.claim is record.operation is record.body_error is None
            assert not record.scopes_returned and not record.positive
            assert prepared._finite is None
            assert read.creator is h.controller and read.source is prepared
            assert read.callback is record.callback and read.task is record.issuing
            assert (
                read._producer is saved["producer"] and read.retired is saved["retired"]
            )
            assert read._producer.done() and not read._producer.cancelled()
            assert read._producer.exception() is sentinel
            assert read.retired.done() and read.retired.result() is None
            for _ in range(2):
                with pytest.raises(
                    RuntimeError, match="original workflow read return missing"
                ):
                    await prepared.close()
                assert prepared._finite_reads == [record]
                assert prepared._source_scope is saved["scope"]
                assert saved["scope"].gen.gi_frame is not None
                assert h.runtime._workflow_preparations is saved["registry"]
                assert prepared in saved["registry"] and not prepared._closed
                assert prepared.source is prepared.native is prepared._lifetime is None
                assert prepared.authorization is None
            assert h.calls == [] and h.db.list_runs(h.conversation_id) == []
            after = h.db.automatic_work.snapshot(before.chain_id)
            assert after.used == before.used and after.reserved == before.reserved
            print(
                "G_FINITE_PREENTRY_RETAINED; original configuration custody, no drain",
                flush=True,
            )
            os._exit(17)
        except BaseException as error:
            primary = error
            raise
        finally:
            threading.settrace_all_threads(previous_thread_trace)
            sys.settrace(previous_trace)
            if generator is not None:
                try:
                    await generator.aclose()
                except BaseException as error:
                    if primary is None:
                        raise
                    primary.add_note(
                        "Original fixture cleanup also failed: " + type(error).__name__
                    )
                    primary.cleanup_error = error


async def test_original_preentry_unknown_read_retains_configuration_before_auto_close(
    tmp_path,
):
    import subprocess
    from pathlib import Path

    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    route = "finite-preentry"
    script = Path(__file__).with_name("workflow_coordinator_probe.py").read_text()
    with pytest.raises(AssertionError, match="G_FINITE_PREENTRY_RETAINED") as caught:
        await asyncio.to_thread(
            _run, tmp_path, route, "retained", script=script, timeout=60
        )
    frames = []
    tb = caught.value.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code is _run.__code__:
            frames.append(tb.tb_frame)
        tb = tb.tb_next
    assert len(frames) == 1
    original = frames[0].f_locals
    result = original["result"]
    assert original["script"] is script and original["tmp_path"] is tmp_path
    assert type(result) is subprocess.CompletedProcess
    assert result.returncode == 17 and result.args[-2:] == [route, "retained"]


async def finite_scope_failure_journey(tmp_path, _unused):
    """An actual failed owning exit cannot borrow its callback's error identity."""
    import os
    from types import SimpleNamespace

    from tldw_chatbook.Agents import activation
    from tldw_chatbook.Agents.execution_capacity import ExecutionOwner
    from tldw_chatbook.Backup_Recovery.admission_runtime import RecoveryAdmissionGuard
    from tldw_chatbook.Backup_Recovery.storage_admission import (
        StorageLease,
        _live_leases,
    )
    from tldw_chatbook.Chat import console_preparation_reads as native_reads
    from tldw_chatbook.Chat import console_workflow_runs as runs
    from tldw_chatbook.Chat.console_workflow_preparation import _PreparedWorkflowReply
    from tldw_chatbook.Workspaces.worktree_activity import WorktreeActivity

    mode = sys.argv[2]
    generator = primary = None
    saved, finish_calls, observer_errors = {}, [], []
    old_trace, old_sys_trace = threading.gettrace(), sys.gettrace()
    old_profile, old_sys_profile = threading.getprofile(), sys.getprofile()
    sentinel = RuntimeError("original callback and owning exit failure")
    codes = {
        "owner": ExecutionOwner.activate.__wrapped__.__code__,
        "source": _PreparedWorkflowReply.activate_source.__wrapped__.__code__,
        "activity": WorktreeActivity.prelaunch.__wrapped__.__code__,
        "outer-guard": RecoveryAdmissionGuard.execution.__wrapped__.__code__,
        "outer-storage": StorageLease.close.__code__,
        "async-storage": StorageLease.close.__code__,
        "async-guard": RecoveryAdmissionGuard.execution.__wrapped__.__code__,
        "finish": OwnedOperation.finish.__code__,
        "read-missing": runs._capture_workflow_read.__code__,
    }
    with pytest.MonkeyPatch.context() as monkeypatch:
        try:
            generator = native_tests.native_host.__wrapped__(
                tmp_path, monkeypatch, SimpleNamespace(param=None)
            )
            h = await anext(generator)
            prepared = await h.bridge._prepare_workflow_reply(
                phase=h.phase,
                task_id="task-1",
                binding=h.binding,
                resolution=h.selected,
            )
            source, scope, lifetime = (
                prepared.source,
                prepared._source_scope,
                prepared._lifetime,
            )
            before = h.db.automatic_work.snapshot(prepared.authorization.claim.chain_id)
            invoke_code = next(
                code
                for code in _PreparedWorkflowReply._read.__code__.co_consts
                if isinstance(code, type(codes["owner"])) and code.co_name == "invoke"
            )
            own_code = next(
                code
                for code in _PreparedWorkflowReply._read.__code__.co_consts
                if isinstance(code, type(codes["owner"]))
                and code.co_name == "invoke_owned"
            )

            def body():
                record = prepared._finite_reads[-1]
                saved.update(
                    record=record,
                    owner=record.owner,
                    claim=record.claim,
                    operation=record.operation,
                )
                assert prepared._finite == (record.owner, record.claim)
                raise sentinel

            async def pending():
                body()

            def callback():
                return pending() if mode.startswith("async-") else body()

            def profile(frame, event, _result):
                if frame.f_code is OwnedOperation.finish.__code__ and event == "call":  # noqa: SIM102 - keep native-frame selection separate from the exact retained receiver.
                    if frame.f_locals.get("self") is saved.get("operation"):
                        finish_calls.append(frame)

            def trace(frame, event, _arg):
                if frame.f_code is not codes[mode] or saved.get("injected"):
                    return trace
                normal = event == "return" and dis.opname[
                    frame.f_code.co_code[frame.f_lasti]
                ] in {"RETURN_VALUE", "RETURN_CONST"}
                if mode in {"outer-storage", "async-storage", "read-missing"}:
                    if event != "call":
                        return trace
                elif not normal:
                    return trace
                inject = False
                try:
                    record = prepared._finite_reads[-1]
                    ancestors = []
                    caller = frame.f_back
                    while caller is not None:
                        ancestors.append(caller)
                        caller = caller.f_back
                    invoke = next(
                        (
                            f
                            for f in ancestors
                            if f.f_code is invoke_code
                            and f.f_locals.get("record") is record
                        ),
                        None,
                    )
                    own = next(
                        (
                            f
                            for f in ancestors
                            if f.f_code is own_code
                            and f.f_locals.get("record") is record
                        ),
                        None,
                    )
                    if mode == "read-missing":
                        if (
                            invoke is None
                            or frame.f_locals.get("source") is not prepared
                        ):
                            return trace
                        binding = native_reads._source_binding.get()
                        assert binding.active and binding.read.source is prepared
                        assert binding.read.callback is record.callback
                        saved.update(record=record, read=binding.read)
                        inject = True
                    elif (
                        record is saved.get("record") and record.body_error is sentinel
                    ):
                        if mode == "owner":
                            inject = (
                                own is not None
                                and frame.f_locals.get("self") is saved["owner"]
                            )
                        elif mode == "source":
                            inject = (
                                own is not None
                                and frame.f_locals.get("self") is prepared
                            )
                        elif mode == "activity":
                            inject = (
                                invoke is not None
                                and frame.f_locals.get("self")
                                is h.runtime.worktree_activity
                                and frame.f_locals.get("claim") is saved["claim"]
                            )
                        elif mode == "outer-guard":
                            inject = (
                                invoke is not None
                                and own is None
                                and frame.f_locals.get("self") is activation._guard
                            )
                        elif mode == "outer-storage":
                            inject = invoke is not None and own is None
                        elif mode == "finish":
                            inject = (
                                invoke is not None
                                and frame.f_locals.get("self") is saved["operation"]
                            )
                        elif mode.startswith("async-"):
                            task = asyncio.current_task()
                            inject = (
                                task is not None
                                and task.get_coro() is record.completion
                            )
                            if mode == "async-guard":
                                # Only the actual outer async-worker guard return,
                                # after complete's inner contexts have returned.
                                inject = (
                                    inject
                                    and frame.f_locals.get("self") is activation._guard
                                    and any(
                                        f.f_locals.get("function") is not None
                                        and getattr(
                                            f.f_locals["function"], "__code__", None
                                        )
                                        is not None
                                        and f.f_locals["function"].__code__.co_name
                                        == "complete"
                                        for f in ancestors
                                    )
                                    and not any(
                                        f.f_code.co_name == "complete"
                                        for f in ancestors
                                    )
                                )
                            saved["async_task"] = task
                    if inject:
                        saved.update(
                            injected=True,
                            frame=frame,
                            receiver=frame.f_locals.get("self"),
                        )
                        if "storage" in mode:
                            assert type(saved["receiver"]) is StorageLease
                            saved["lease"] = saved["receiver"]
                except BaseException as observation_error:  # noqa: BLE001 - retain the original error without throwing into native observation or masking cleanup.
                    observer_errors.append(observation_error)
                    return trace
                if inject:
                    # Intentional fault at the SAME native boundary; observers
                    # above only retain facts and cannot replace the fault.
                    threading.settrace_all_threads(None)
                    sys.settrace(None)
                    raise sentinel
                return trace

            threading.setprofile_all_threads(profile)
            threading.settrace_all_threads(trace)
            try:
                with pytest.raises(RuntimeError) as caught:
                    await prepared._read(callback)
                assert caught.value is sentinel
                assert saved.get("injected") is True
                assert observer_errors == []
                record = saved["record"]
                assert record is prepared._finite_reads[-1]
                assert record.wrapper_error is sentinel and not record.positive
                assert not record.scopes_returned
                if mode == "read-missing":
                    assert (
                        record.proof
                        is record.owner
                        is record.claim
                        is record.operation
                        is None
                    )
                    read = saved["read"]
                    assert record.body_error is None and prepared._finite is None
                    assert record.before_issue is None
                else:
                    read, producer, retired, original = record.proof  # noqa: RUF059 - retain the complete original issue/proof fields for identity inspection.
                    assert read.callback is original is record.callback
                    assert record.body_error is sentinel
                    assert (
                        record.owner is saved["owner"]
                        and record.claim is saved["claim"]
                    )
                    assert record.operation is saved["operation"]
                    assert prepared._finite == (saved["owner"], saved["claim"])
                    with h.runtime.execution_capacity._lock:
                        assert (
                            h.runtime.execution_capacity._owners.get(
                                saved["owner"].execution_id
                            )
                            is saved["owner"]
                        )
                        assert not saved["owner"]._root_finished
                        assert (saved["operation"] in saved["owner"]._operations) is (
                            mode != "finish"
                        )
                    with h.runtime.worktree_activity._lock:
                        assert saved["claim"].id in h.runtime.worktree_activity._claims
                        assert (
                            h.runtime.worktree_activity._members[
                                saved["claim"].id
                            ].owner
                            is saved["owner"]
                        )
                    if "storage" in mode:
                        assert saved["lease"] in _live_leases
                assert read.creator is h.controller and read.source is prepared
                assert read.task is record.issuing
                assert read._producer.done() and read._producer.exception() is sentinel
                assert read.retired.done() and read.retired.result() is None
                assert len(finish_calls) == (1 if mode == "finish" else 0)
                count = len(prepared._finite_reads)
                for _ in range(2):
                    with pytest.raises(RuntimeError, match="original workflow read"):
                        await prepared.close()
                    assert len(prepared._finite_reads) == count
                    assert prepared.source is source and prepared._source_scope is scope
                    assert prepared._lifetime is lifetime and not prepared._closed
                    assert prepared in h.runtime._workflow_preparations
                    assert len(finish_calls) == (1 if mode == "finish" else 0)
                assert h.calls == [] and h.db.list_runs(h.conversation_id) == []
                after = h.db.automatic_work.snapshot(before.chain_id)
                assert after.used == before.used and after.reserved == before.reserved
                print(
                    "G_FINITE_SCOPE_RETAINED",
                    mode,
                    "; actual original custody, no fixture/creator drain",
                    flush=True,
                )
                os._exit(17)
            finally:
                threading.settrace_all_threads(old_trace)
                sys.settrace(old_sys_trace)
                threading.setprofile_all_threads(old_profile)
                sys.setprofile(old_sys_profile)
        except BaseException as error:
            primary = error
            raise
        finally:
            if generator is not None:
                try:
                    await generator.aclose()
                except BaseException as error:
                    if primary is None:
                        raise
                    primary.add_note(
                        "Original fixture cleanup also failed: " + type(error).__name__
                    )
                    primary.cleanup_error = error


@pytest.mark.parametrize(
    "mode",
    [
        "owner",
        "source",
        "activity",
        "outer-guard",
        "outer-storage",
        "async-storage",
        "async-guard",
        "finish",
        "read-missing",
    ],
)
async def test_same_callback_error_does_not_certify_failed_native_scope(tmp_path, mode):
    import subprocess
    from pathlib import Path

    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    route = "finite-scope"
    script = Path(__file__).with_name("workflow_coordinator_probe.py").read_text()
    with pytest.raises(AssertionError, match="G_FINITE_SCOPE_RETAINED") as caught:
        await asyncio.to_thread(_run, tmp_path, route, mode, script=script, timeout=60)
    frames = []
    tb = caught.value.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code is _run.__code__:
            frames.append(tb.tb_frame)
        tb = tb.tb_next
    assert len(frames) == 1
    original = frames[0].f_locals
    result = original["result"]
    assert original["script"] is script and original["tmp_path"] is tmp_path
    assert type(result) is subprocess.CompletedProcess
    assert result.returncode == 17 and result.args[-2:] == [route, mode]


async def test_original_nonissued_finite_read_allows_exact_unused_cleanup(native_host):
    cancel = threading.Event()
    h = native_host
    prepared = await h.bridge._prepare_workflow_reply(
        phase=h.phase,
        task_id="task-1",
        binding=h.binding,
        resolution=h.selected,
        should_cancel=cancel.is_set,
    )
    scope = prepared._source_scope
    source = prepared.source
    calls = []
    before = h.db.automatic_work.snapshot(prepared.authorization.claim.chain_id)
    cancel.set()
    with pytest.raises(
        PermissionError, match="workflow native source changed"
    ) as caught:
        await prepared._read(lambda: calls.append(True))
    record = prepared._finite_reads[-1]
    (
        coroutine,
        frame,
        checker,
        validator,
        thread,
        reads,  # noqa: RUF059 - retain the complete original issue/proof fields for identity inspection.
        observers,  # noqa: RUF059 - retain the complete original issue/proof fields for identity inspection.
        session_id,  # noqa: RUF059 - retain the complete original issue/proof fields for identity inspection.
        runtime,
        controller,  # noqa: RUF059 - retain the complete original issue/proof fields for identity inspection.
        service,  # noqa: RUF059 - retain the complete original issue/proof fields for identity inspection.
        original_source,
        original_scope,
        registry,
        closing,  # noqa: RUF059 - retain the complete original issue/proof fields for identity inspection.
    ) = record.issue
    assert record.before_issue == (
        record,
        prepared,
        asyncio.current_task(),
        record.issue,
        caught.value,
    )
    assert record.wrapper_error is caught.value and record.proof is None
    assert record.owner is record.claim is record.operation is None
    assert (
        record.body_error is None and not record.scopes_returned and not record.positive
    )
    assert coroutine.cr_frame is None and not coroutine.cr_running
    assert not any(
        name in frame.f_locals for name in ("read", "producer", "submit_resolved")
    )
    assert frame.f_locals["require_current"] is checker
    assert validator.__self__ is prepared and thread is threading.current_thread()
    assert original_source is source and original_scope is scope
    assert registry is runtime._workflow_preparations and prepared in registry
    assert calls == [] and h.calls == [] and h.db.list_runs(h.conversation_id) == []
    assert prepared._finite is None
    await prepared.close()
    closing_read = prepared._finite_reads[-1]
    assert closing_read is not record and closing_read.positive
    assert closing_read.proof[1].exception() is None
    assert closing_read.proof[2].done() and closing_read.proof[2].result() is None
    assert prepared._closed and prepared._native_closed
    assert prepared._source_scope is None and prepared not in registry
    after = h.db.automatic_work.snapshot(before.chain_id)
    assert after.used == before.used and after.reserved == before.reserved
    original_reads = tuple(prepared._finite_reads)
    original_proof = closing_read.proof
    original_lifetime = prepared._lifetime
    await prepared.close()
    assert len(prepared._finite_reads) == len(original_reads)
    assert all(a is b for a, b in zip(prepared._finite_reads, original_reads))
    assert prepared._finite_reads[-1] is closing_read
    assert closing_read.proof is original_proof
    assert original_proof[1].exception() is None
    assert original_proof[2].done() and original_proof[2].result() is None
    assert prepared._lifetime is original_lifetime and prepared._finite is None
    assert prepared.source is source and prepared._source_scope is None
    assert prepared not in registry and prepared._closed and prepared._native_closed
    assert h.calls == [] and h.db.list_runs(h.conversation_id) == []
    repeated = h.db.automatic_work.snapshot(before.chain_id)
    assert repeated.used == before.used and repeated.reserved == before.reserved


async def test_original_no_owner_preparation_wait_returns_without_invented_owner(
    native_host,
):
    h = native_host
    prepared = await h.bridge._prepare_workflow_reply(
        phase=h.phase, task_id="task-1", binding=h.binding, resolution=h.selected
    )
    before = h.db.automatic_work.snapshot(prepared.authorization.claim.chain_id)
    competitor = h.runtime.execution_capacity.begin_execution(
        origin=WorkOrigin.MANUAL, conversation_id="actual-preparation-contender"
    )
    drains, calls = [], []
    competitor.on_drained(drains.append)
    claim = h.runtime.worktree_activity.try_acquire(
        owner=competitor, footprint=WorktreeFootprint(opaque=True)
    )
    assert claim is not None
    try:
        with pytest.raises(_WorkflowChildWait) as caught:
            await prepared._read(lambda: calls.append(True))
        record = prepared._finite_reads[-1]
        assert record.body_error is caught.value and record.wrapper_error is None
        assert record.positive and record.scopes_returned and record.drains == []
        assert record.owner is record.claim is record.operation is None
        assert record.proof[1].result()[5:8] == (None, None, None)
        assert record.proof[1].result()[9] is caught.value
        assert record.proof[2].done() and record.proof[2].result() is None
        assert prepared._finite is None and competitor.is_admitted
        assert calls == [] and h.calls == []
    finally:
        competitor.finish_root()
        claim.settle(durable=True)
    assert drains == [True] and not competitor.is_admitted
    await prepared.close()
    assert prepared._closed and prepared not in h.runtime._workflow_preparations
    after = h.db.automatic_work.snapshot(before.chain_id)
    assert after.used == before.used and after.reserved == before.reserved


@pytest.mark.parametrize("body_fails", [False, True])
async def test_original_finite_caller_cancellation_waits_and_keeps_body_outcome(
    native_host, body_fails
):
    h = native_host
    prepared = await h.bridge._prepare_workflow_reply(
        phase=h.phase, task_id="task-1", binding=h.binding, resolution=h.selected
    )
    before = h.db.automatic_work.snapshot(prepared.authorization.claim.chain_id)
    entered, release = threading.Event(), threading.Event()
    value, failure = object(), RuntimeError("original held finite callback failed")
    saved = {}

    def callback():
        record = prepared._finite_reads[-1]
        saved.update(
            record=record,
            owner=record.owner,
            operation=record.operation,
            claim=record.claim,
        )
        entered.set()
        assert release.wait(5), "original finite callback hold exceeded5s"
        if body_fails:
            raise failure
        return value

    task = asyncio.create_task(prepared._read(callback))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel("first original caller cancellation")
        await asyncio.sleep(0)
        task.cancel("second original caller cancellation")
        await asyncio.sleep(0)
        assert not task.done() and saved["owner"].is_admitted
        assert prepared._finite == (saved["owner"], saved["claim"])
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
    record = saved["record"]
    assert record.issuing is task and record.wrapper_error is caught.value
    assert record.body_error is (failure if body_fails else None)
    assert record.scopes_returned and not record.positive
    assert record.proof[1].exception() is None
    actual = record.proof[1].result()
    assert actual[8] is (None if body_fails else value)
    assert actual[9] is record.body_error
    assert record.proof[2].done() and record.proof[2].result() is None
    assert record.drains == [True] and not saved["owner"].is_admitted
    assert prepared._finite is None
    await prepared.close()
    assert prepared._closed and prepared._source_scope is None
    assert prepared not in h.runtime._workflow_preparations
    assert h.calls == [] and h.db.list_runs(h.conversation_id) == []
    after = h.db.automatic_work.snapshot(before.chain_id)
    assert after.used == before.used and after.reserved == before.reserved


async def finite_planning_scope_journey(tmp_path, monkeypatch):
    import os

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig
    from tldw_chatbook.Agents import activation
    from tldw_chatbook.Backup_Recovery.admission_runtime import RecoveryAdmissionGuard
    from tldw_chatbook.Chat.console_chat_controller import (
        capture_run_admitted_workspace_roots,
    )
    from tldw_chatbook.Chat.console_workflow_preparation import (
        _PreparedWorkflowReply,
        prepare_workflow_planning,
    )

    mode = sys.argv[2]
    negative = mode.startswith("async-")
    host_thread, host_loop = threading.current_thread(), asyncio.get_running_loop()
    rig = await build_workflow_rig(
        tmp_path, monkeypatch, WorkflowProvider(), public_coordinator=False
    )
    original_error = RuntimeError("original planning finite callback failed")
    old_trace, old_sys_trace = threading.gettrace(), sys.gettrace()
    saved, errors = {}, []
    primary = None
    try:
        assert not hasattr(rig.runtime, "_workflow_coordinator")
        session = rig.store.sessions()[0]
        binding = capture_run_admitted_workspace_roots(
            session=session, registry=rig.service.persistence.workspace_registry
        )[0]
        request = rig.workflow.request
        prepared = await prepare_workflow_planning(
            rig.controller,
            session_id=session.id,
            binding=binding,
            command_id="finite-planning-return",
            objective=request.objective,
            policy=request.policy,
            _request=request,
        )
        assert prepared.phase.attempt.purpose == "planning"
        assert prepared._loop is host_loop
        assert prepared.primary_owner is None and rig.provider.calls == []
        source, scope = prepared.source, prepared._source_scope
        intake, token = prepared._planning_intake, prepared._planning_token
        before = rig.service.db.automatic_work.snapshot(prepared.phase.context.chain_id)
        source_code = _PreparedWorkflowReply.activate_source.__wrapped__.__code__
        guard_code = RecoveryAdmissionGuard.execution.__wrapped__.__code__

        def body():
            record = prepared._planning_reads[-1]
            saved.update(
                record=record,
                owner=record.owner,
                claim=record.claim,
                operation=record.operation,
            )
            raise original_error

        async def pending():
            body()

        def callback():
            if mode == "synchronous":
                body()
            return pending()

        def trace(frame, event, _arg):
            wanted = source_code if mode == "async-source" else guard_code
            if (
                not negative
                or frame.f_code is not wanted
                or event != "return"
                or saved.get("injected")
            ):
                return trace
            if dis.opname[frame.f_code.co_code[frame.f_lasti]] not in {
                "RETURN_VALUE",
                "RETURN_CONST",
            }:
                return trace
            inject = False
            try:
                record = saved.get("record")
                if (
                    record is None
                    or record.body_error is not original_error
                    or record.completion is None
                    or threading.current_thread() is not host_thread
                ):
                    return trace
                task = asyncio.current_task()
                if (
                    asyncio.get_running_loop() is not host_loop
                    or task is None
                    or task.get_coro() is not record.completion
                ):
                    return trace
                ancestors, caller = [], frame.f_back
                while caller is not None:
                    ancestors.append(caller)
                    caller = caller.f_back
                if mode == "async-source":
                    inject = frame.f_locals.get("self") is prepared and any(
                        f.f_code.co_name == "complete"
                        and f.f_locals.get("record") is record
                        for f in ancestors
                    )
                else:
                    inject = (
                        frame.f_locals.get("self") is activation._guard
                        and not any(f.f_code.co_name == "complete" for f in ancestors)
                        and any(
                            getattr(f.f_locals.get("function"), "__code__", None)
                            is not None
                            and f.f_locals["function"].__code__.co_name == "complete"
                            for f in ancestors
                        )
                    )
                if inject:
                    saved.update(injected=True, frame=frame, task=task)
            except BaseException as error:  # noqa: BLE001 - retain the original error without throwing into native observation or masking cleanup.
                errors.append(error)
                return trace
            if inject:
                threading.settrace_all_threads(None)
                sys.settrace(None)
                raise original_error
            return trace

        if negative:
            threading.settrace_all_threads(trace)
        try:
            with pytest.raises(RuntimeError) as caught:
                await prepared._read(callback)
        finally:
            threading.settrace_all_threads(old_trace)
            sys.settrace(old_sys_trace)
        record = saved["record"]
        assert (
            record is prepared._planning_reads[-1]
            and record.body_error is original_error
        )
        if errors:
            details = []
            for error in errors[:3]:
                locations, tb = [], error.__traceback__
                while tb is not None and len(locations) < 8:
                    locations.append(
                        (
                            tb.tb_frame.f_code.co_filename,
                            tb.tb_frame.f_code.co_name,
                            tb.tb_lineno,
                        )
                    )
                    tb = tb.tb_next
                details.append((type(error).__name__, locations))
            print("G_PLANNING_OBSERVER_ERRORS", len(errors), details, flush=True)
        assert errors == []
        read, producer, retired, original_callback = record.proof
        assert (
            read.source is prepared
            and read.callback is original_callback is record.callback
        )
        assert read.task is record.issuing is asyncio.current_task()
        assert read._producer is producer and read.retired is retired
        assert retired.done() and retired.result() is None
        if negative:
            assert saved.get("injected") is True
            assert str(caught.value) == "original workflow reader return unproven"
            assert (
                record.wrapper_error is original_error
                and producer.exception() is original_error
            )
            assert record.completion_future.exception() is original_error
            assert not record.positive and not record.scopes_returned
            assert prepared._finite == (saved["owner"], saved["claim"])
            with rig.runtime.execution_capacity._lock:
                assert (
                    rig.runtime.execution_capacity._owners.get(
                        saved["owner"].execution_id
                    )
                    is saved["owner"]
                )
                assert saved["operation"] in saved["owner"]._operations
                assert not saved["owner"]._root_finished
            with rig.runtime.worktree_activity._lock:
                assert saved["claim"].id in rig.runtime.worktree_activity._claims
            count = len(prepared._planning_reads)
            for _ in range(2):
                # Original F close conservatively returns while retaining; it
                # does not share execution's new no-issue alternative.
                assert await prepared.close() is None
                assert len(prepared._planning_reads) == count
                assert prepared.source is source and prepared._source_scope is scope
                assert prepared in rig.runtime._workflow_preparations
                assert not prepared._closed and not prepared._planning_settled
                assert (
                    prepared._planning_intake is intake
                    and prepared._planning_token is token
                )
            after = rig.service.db.automatic_work.snapshot(before.chain_id)
            assert after.used == before.used and after.reserved == before.reserved
            assert rig.provider.calls == [] and not hasattr(
                rig.runtime, "_workflow_coordinator"
            )
            print(
                "G_PLANNING_FINITE_RETAINED",
                mode,
                "; no constructor retirement",
                flush=True,
            )
            os._exit(17)
        assert caught.value is original_error and record.wrapper_error is None
        assert producer.exception() is None and producer.result()[9] is original_error
        assert record.positive and record.scopes_returned and record.drains == [True]
        assert not saved["owner"].is_admitted and prepared._finite is None
        await prepared.close()
        assert prepared._closed and prepared._source_scope is None
        assert prepared not in rig.runtime._workflow_preparations
        assert prepared._planning_intake is prepared._planning_token is None
        assert rig.runtime.execution_capacity.snapshot().executions == ()
        assert rig.provider.calls == [] and not hasattr(
            rig.runtime, "_workflow_coordinator"
        )
        after = rig.service.db.automatic_work.snapshot(before.chain_id)
        assert after.used == before.used and after.reserved == before.reserved
    except BaseException as error:
        primary = error
        raise
    finally:
        threading.settrace_all_threads(old_trace)
        sys.settrace(old_sys_trace)
        try:
            await rig.close()
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note(
                "Original planning fixture cleanup also failed: " + type(error).__name__
            )
            primary.cleanup_error = error


@pytest.mark.parametrize(
    "mode", ["synchronous", "awaitable", "async-source", "async-guard"]
)
async def test_original_planning_finite_error_requires_its_actual_async_scope_return(
    tmp_path, mode
):
    import subprocess
    from pathlib import Path

    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    route = "finite-planning"
    script = Path(__file__).with_name("workflow_coordinator_probe.py").read_text()
    if not mode.startswith("async-"):
        await asyncio.to_thread(_run, tmp_path, route, mode, script=script, timeout=60)
        return
    with pytest.raises(AssertionError, match="G_PLANNING_FINITE_RETAINED") as caught:
        await asyncio.to_thread(_run, tmp_path, route, mode, script=script, timeout=60)
    frames, tb = [], caught.value.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code is _run.__code__:
            frames.append(tb.tb_frame)
        tb = tb.tb_next
    assert len(frames) == 1
    actual = frames[0].f_locals
    assert actual["script"] is script and actual["tmp_path"] is tmp_path
    assert type(actual["result"]) is subprocess.CompletedProcess
    assert actual["result"].returncode == 17
    assert actual["result"].args[-2:] == [route, mode]
