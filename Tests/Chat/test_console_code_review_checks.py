"""Check feedback uses actual scoped launch and its fixed runtime mode."""

import json

import pytest

from Tests.Agents import code_review_helpers, test_code_review_dispatch
from Tests.Agents.test_code_review_dispatch import model_pages
from Tests.Agents.test_code_review_reports import coverage, finding
from tldw_chatbook.Agents.run_context import current_run_actor, use_run_actor

dispatch_case = test_code_review_dispatch.dispatch_case
review_case = code_review_helpers.review_case
source_case = code_review_helpers.source_case

pytestmark = pytest.mark.bootstrap_profile


def seed_original(c):
    with c.participant(c.prepared) as provider:
        service = provider.service
        service.submit(
            c.prepared.review_id, "findings", [finding(c, c.prepared, "one")], 0, "seed"
        )
        service.submit(
            c.prepared.review_id,
            "reviewer_complete",
            [coverage(c, c.prepared, empty=False)],
            1,
            "complete",
        )
        service.capture_review_output(
            service.actor.run_id, "Original raw text", terminal=True
        )
        service.settle_review_run(
            service.actor.run_id, terminal_status="done", receipt_id="reviewer-terminal"
        )
    with use_run_actor(c.actor):
        c.primary.read(c.prepared.review_id, "summary")


@pytest.mark.parametrize("native", [False, True])
def test_inline_proposal_resolves_without_wait_or_execution(dispatch_case, native):
    c = dispatch_case
    observed = {}

    def read_batch():
        return c.reply(
            "code_review_read", {"review_id": c.prepared.review_id, "view": "batch"}
        )

    def propose():
        page = model_pages(
            [request for actor, request in c.calls if actor.kind == "subagent"][-1]
        )[-1]
        observed["unit"] = page["records"][0]["verification_unit_id"]
        return c.reply(
            "code_review_submit",
            {
                "review_id": c.prepared.review_id,
                "kind": "check_proposals",
                "records": [
                    {
                        "schema_version": 1,
                        "argv": ["python", "-m", "pytest"],
                        "intent": "Test the captured defect",
                        "verification_unit_ids": [observed["unit"]],
                    }
                ],
                "expected_revision": 0,
                "submission_id": "proposal",
            },
        )

    def read_result():
        rows = [
            request
            for actor, request in c.calls
            if actor.kind
            == ("subagent" if current_run_actor().kind == "subagent" else "primary")
        ][-1]["messages_payload"]
        for row in reversed(rows):
            text = row.get("content") or ""
            if '"accepted_ids"' in text:
                ack = json.loads(text if native else text.partition(": ")[2])
                observed["proposal"] = ack["accepted_ids"][0]
                break
        return c.reply(
            "code_review_read",
            {
                "review_id": c.prepared.review_id,
                "view": "verification_batch",
                "document_ref": observed["proposal"],
            },
        )

    def finish():
        observed["page"] = model_pages(
            [request for actor, request in c.calls if actor.kind == "subagent"][-1]
        )[-1]
        return "Source-based verification complete; executable check unavailable"

    c.run(
        verify=True,
        inline=True,
        native=native,
        prepare_hook=seed_original,
        child_script=[
            lambda: c.reply(
                "load_tools",
                {
                    "ids": [
                        "code_review:code_review_read",
                        "code_review:code_review_submit",
                    ]
                },
            ),
            read_batch,
            propose,
            read_result,
            finish,
        ],
    )
    receipt = observed["page"]["records"][0]
    assert receipt["outcome"] == "unavailable"
    assert receipt["evidence"] == "check_feedback_unavailable_inline"
    assert receipt["proposal_id"] == observed["proposal"]
    assert receipt["native_execution_receipt_id"] is None
    assert len({actor.run_id for actor, _ in c.calls if actor.kind == "subagent"}) == 1


def test_stock_raw_runtime_has_no_aggregate_review_write_boundary():
    from tldw_chatbook.Chat.console_raw_cli import RawCliRuntime

    runtime = RawCliRuntime(lambda: True)
    assert runtime.review_check_boundary_factory() is None


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("one_worker", [False, True])
def test_wait_yields_live_verifier_and_decline_reaches_same_run(
    dispatch_case, native, one_worker
):
    import threading

    c = dispatch_case
    waiting = threading.Event()
    observed = {}
    if one_worker:
        from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity

        c.runtime_capacity = RuntimeCapacity(
            max_tool_workers=1, reserved_manual_tool_workers=0
        )

    def batch():
        return c.reply(
            "code_review_read", {"review_id": c.prepared.review_id, "view": "batch"}
        )

    def proposal():
        assert waiting.wait(5)
        unit = model_pages(
            [request for actor, request in c.calls if actor.kind == "subagent"][-1]
        )[-1]["records"][0]["verification_unit_id"]
        return c.reply(
            "code_review_submit",
            {
                "review_id": c.prepared.review_id,
                "kind": "check_proposals",
                "records": [
                    {
                        "schema_version": 1,
                        "argv": ["python", "-m", "pytest"],
                        "intent": "Test captured source",
                        "verification_unit_ids": [unit],
                    }
                ],
                "expected_revision": 0,
                "submission_id": "proposal",
            },
        )

    def child_wait():
        with c.store.lease(c.prepared.review_id) as context:
            observed["proposal"] = next(iter(context.checks))
        return c.reply(
            "code_review_read",
            {
                "review_id": c.prepared.review_id,
                "view": "verification_batch",
                "document_ref": observed["proposal"],
            },
        )

    def child_finish():
        observed["receipt"] = model_pages(
            [request for actor, request in c.calls if actor.kind == "subagent"][-1]
        )[-1]["records"][0]
        return "Finished with an unresolved check"

    def parent_wait():
        waiting.set()
        return c.reply("wait_agents", {})

    def decline():
        if one_worker:
            import time

            deadline = time.monotonic() + 5
            while (
                c.runtime_capacity.snapshot().tool_workers == 0
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            assert c.runtime_capacity.snapshot().tool_workers == 1
        rows = [
            request
            for actor, request in c.calls
            if actor.kind
            == ("subagent" if current_run_actor().kind == "subagent" else "primary")
        ][-1]["messages_payload"]
        observed["notice"] = str(rows)
        with c.store.lease(c.prepared.review_id) as context:
            check = next(iter(context.checks.values()))
            observed["live"] = (
                context.streams[
                    (check.binding.slot_id, check.binding.attempt_id)
                ].terminal_status
                is None
            )
        return c.reply(
            "code_review_submit",
            {
                "review_id": c.prepared.review_id,
                "kind": "check_actions",
                "records": [
                    {
                        "schema_version": 1,
                        "proposal_id": check.proposal_id,
                        "action": "decline",
                        "reason": "User requested source-only evidence",
                    }
                ],
                "expected_revision": 0,
                "submission_id": "decline",
            },
        )

    def parent_script(c):
        return [
            c.reply(
                "load_tools",
                {
                    "ids": [
                        "code_review:code_review_submit",
                        "code_review:code_review_read",
                    ]
                },
            ),
            c.reply(
                "spawn_subagent",
                {"review_id": c.prepared.review_id, "review_slot_id": "V1"},
            ),
            parent_wait,
            decline,
        ]

    c.run(
        verify=True,
        native=native,
        prepare_hook=seed_original,
        parent_script=parent_script,
        child_script=[
            lambda: c.reply(
                "load_tools",
                {
                    "ids": [
                        "code_review:code_review_read",
                        "code_review:code_review_submit",
                    ]
                },
            ),
            batch,
            proposal,
            child_wait,
            child_finish,
        ],
    )
    assert observed["live"]
    assert "check_requested" in observed["notice"]
    assert observed["receipt"]["outcome"] == (
        "unavailable" if one_worker else "declined"
    )
    assert (
        ("tool_capacity" in observed["receipt"]["evidence"])
        if one_worker
        else observed["receipt"]["evidence"] == "User requested source-only evidence"
    )


class FaultBoundedExecutor:
    """Finite scripted external boundary; this is not a host subprocess sandbox."""

    def __init__(self, outcome="executed"):
        self.outcome = outcome
        self.launches = []
        self.root = None
        self.cleaned = False

    def reserve_review_boundary(self, *, parent, sources, max_bytes, max_files):
        import shutil
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        assert sum(map(len, sources.values())) <= max_bytes
        assert len(sources) <= max_files
        root = Path(tempfile.mkdtemp(dir=parent))
        self.root = root
        for name, body in sources.items():
            path = root / name
            assert path.is_relative_to(root)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)

        def release():
            if self.outcome == "uncertain":
                return False
            shutil.rmtree(root)
            self.cleaned = True
            return True

        return SimpleNamespace(root=root, release=release)

    def execute(self, request, *, cancel_event, on_event, admit_worker):
        import time
        from types import SimpleNamespace

        from tldw_chatbook.Tools.raw_cli_executor import RawCliResult

        assert request.initial_directory == self.root
        assert admit_worker(
            SimpleNamespace(admit=lambda: None), lambda: time.monotonic()
        )
        self.launches.append(request.invocation_id)
        if getattr(self, "on_accepted", None):
            self.on_accepted()
        return RawCliResult(
            invocation_id=request.invocation_id,
            caller=request.caller,
            resolved_shell="bash",
            initial_directory=request.initial_directory,
            elapsed_seconds=0.01,
            stdout_preview=(
                "Captured-source check evidence" * 200
                if self.outcome == "paged"
                else "Captured-source check evidence"
            ),
            stderr_preview="",
            record_output="Captured-source check evidence",
            exit_code=0,
            terminal_state="completed",
            truncated=False,
            cleanup_proven=self.outcome != "uncertain",
        )


@pytest.mark.parametrize("native", [False, True])
def test_claim_executes_once_through_native_runtime_and_returns_host_receipt(
    dispatch_case, native
):
    import threading

    from tldw_chatbook.Agents.raw_shell_tool_provider import RawShellToolProvider
    from tldw_chatbook.Chat.console_raw_cli import RawCliRuntime
    from tldw_chatbook.MCP.permission_store import EffectiveToolState

    c = dispatch_case
    state = {}
    ready = threading.Event()
    executor = FaultBoundedExecutor()
    runtime = RawCliRuntime(lambda: True, executor=executor)
    assert runtime.arm().armed
    runtime.grant_model_session("chat")
    c.process_provider = RawShellToolProvider(
        runtime=runtime,
        console_session_id="chat",
        initial_directory=lambda: c.root,
        resolve_state=lambda _: EffectiveToolState(
            state="ask", origin="global_default"
        ),
    )

    def batch():
        return c.reply(
            "code_review_read", {"review_id": c.prepared.review_id, "view": "batch"}
        )

    def propose():
        assert ready.wait(5)
        children = [
            (actor, request) for actor, request in c.calls if actor.kind == "subagent"
        ]
        unit = model_pages(children[-1][1])[-1]["records"][0]["verification_unit_id"]
        return c.reply(
            "code_review_submit",
            {
                "review_id": c.prepared.review_id,
                "kind": "check_proposals",
                "records": [
                    {
                        "schema_version": 1,
                        "argv": ["python", "-m", "pytest"],
                        "intent": "Test captured source",
                        "verification_unit_ids": [unit],
                    }
                ],
                "expected_revision": 0,
                "submission_id": "proposal",
            },
        )

    def read_result():
        with c.store.lease(c.prepared.review_id) as context:
            proposal_id = next(iter(context.checks))
        return c.reply(
            "code_review_read",
            {
                "review_id": c.prepared.review_id,
                "view": "verification_batch",
                "document_ref": proposal_id,
            },
        )

    def child_finish():
        children = [
            (actor, request) for actor, request in c.calls if actor.kind == "subagent"
        ]
        state["receipt"] = model_pages(children[-1][1])[-1]["records"][0]
        return "Verification complete"

    def wait():
        ready.set()
        return c.reply("wait_agents", {})

    def claim():
        with c.store.lease(c.prepared.review_id) as context:
            state["proposal"] = next(iter(context.checks))
        return c.reply(
            "code_review_submit",
            {
                "review_id": c.prepared.review_id,
                "kind": "check_actions",
                "records": [
                    {
                        "schema_version": 1,
                        "proposal_id": state["proposal"],
                        "action": "claim",
                    }
                ],
                "expected_revision": 0,
                "submission_id": "claim",
            },
        )

    def process():
        with c.store.lease(c.prepared.review_id) as context:
            check = context.checks[state["proposal"]]
            assert check.claim.tool_name == "shell_exec"
            assert check.claim.arguments["initial_directory"] != str(c.root)
            return c.reply(check.claim.tool_name, check.claim.arguments)

    def parent(c):
        return [
            c.reply(
                "load_tools",
                {"ids": ["code_review:code_review_submit", "raw_shell:shell_exec"]},
            ),
            c.reply(
                "spawn_subagent",
                {"review_id": c.prepared.review_id, "review_slot_id": "V1"},
            ),
            wait,
            claim,
            claim,
            process,
        ]

    c.run(
        verify=True,
        native=native,
        prepare_hook=seed_original,
        parent_script=parent,
        child_script=[
            lambda: c.reply(
                "load_tools",
                {
                    "ids": [
                        "code_review:code_review_read",
                        "code_review:code_review_submit",
                    ]
                },
            ),
            batch,
            propose,
            read_result,
            child_finish,
        ],
    )
    assert len(executor.launches) == 1
    assert state["receipt"]["outcome"] == "executed"
    assert state["receipt"]["native_execution_receipt_id"] == executor.launches[0]
    assert "Captured-source check evidence" in state["receipt"]["evidence"]
    assert executor.cleaned


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "outcome",
    ["executed", "uncertain", "stock", "declined", "permission_denied", "paged"],
)
def test_check_handoff_lifecycle_matrix(dispatch_case, native, outcome):
    c = dispatch_case
    observed = c.run_check_handoff(
        outcome=outcome,
        native_tools=native,
        duplicate_actions=True,
        result_limit=1800 if outcome == "paged" else 16000,
    )
    assert observed.verifier_launch_count == 1
    assert observed.wait_yielded_while_verifier_live
    assert observed.result_seen_by_same_verifier
    expected = (
        "executed"
        if outcome == "paged"
        else "unavailable"
        if outcome in {"stock", "permission_denied"}
        else outcome
    )
    assert c.check_observation_state["receipt"]["outcome"] == expected
    assert observed.execution_launch_count == int(
        outcome in {"executed", "paged", "uncertain"}
    )
    assert observed.accepted_unit_dispositions[0]["verdict"] == (
        "confirmed" if expected == "executed" else "unresolved"
    )
    if outcome == "executed":
        assert observed.check_events == [
            "proposal",
            "wait_yield",
            "claim",
            "execution_accepted",
            "receipt",
            "verifier_read",
            "disposition",
        ]
    if outcome == "uncertain":
        assert observed.reserved_bytes > 1024 * 1024
        assert not c.check_executor.cleaned
        # The external owner later proves deletion; retain the uncertain verdict.
        with c.store.lease(c.prepared.review_id) as context:
            execution = next(iter(context.checks.values())).execution
        c.check_executor.outcome = "executed"
        assert execution.release()
        assert c.store.usage().reserved_bytes == 0
    if outcome == "paged":
        assert len(observed.receipts) > 1
        assert all(
            len(json.dumps(page, separators=(",", ":"))) <= 1800
            for page in observed.receipts
        )
        assert (
            len({page["records"][0]["receipt_id"] for page in observed.receipts}) == 1
        )


@pytest.mark.parametrize("prepare,launch", [(1, 1), (3, 1), (1, 3)])
def test_actual_factory_binds_check_mode_at_launch(dispatch_case, prepare, launch):
    observed = dispatch_case.run_inline_check(
        max_live_at_prepare=prepare, max_live_at_launch=launch, source_only=True
    )
    assert observed.feedback_mode == (
        "unavailable_inline" if launch == 1 else "primary_handoff"
    )
    assert observed.result_seen_by_same_verifier
    assert observed.accepted_unit_dispositions[0]["verdict"] == "confirmed"
    if launch == 1:
        assert observed.execution_launch_count == 0
        assert not observed.wait_yielded_while_verifier_live
        assert observed.host_check_wait_count == 0
        assert observed.primary_check_notice_count == 0


@pytest.mark.parametrize("forged", [False, True])
def test_unavailable_check_allows_source_decision_but_never_check_evidence(
    dispatch_case, forged
):
    observed = dispatch_case.run_check_handoff(
        outcome="stock", source_only=not forged, forged_evidence=forged
    )
    assert observed.accepted_unit_dispositions[0]["verdict"] == (
        "unresolved" if forged else "confirmed"
    )
    if forged:
        assert "invalid_executed_check_evidence" in str(observed.requests)


def test_proposal_before_wait_entry_is_detected(dispatch_case):
    observed = dispatch_case.run_check_handoff(primary_waiting=False)
    assert observed.wait_yielded_while_verifier_live
    assert observed.result_seen_by_same_verifier


@pytest.mark.parametrize(
    "outcome", ["primary_ended", "verifier_cancelled", "source_fence", "close"]
)
def test_terminal_fences_release_pending_verifier_without_replacement(
    dispatch_case, outcome
):
    observed = dispatch_case.run_check_handoff(outcome=outcome)
    assert observed.verifier_launch_count == 1
    assert observed.execution_launch_count == 0
    assert observed.wait_yielded_while_verifier_live
    if outcome != "close":
        receipt = dispatch_case.check_observation_state["terminal_receipt"]
        assert receipt.outcome in {"cancelled", "unavailable"}
    assert dispatch_case.store.usage().reserved_bytes == 0


def test_closed_verifier_and_forged_native_receipts_are_rejected(dispatch_case):
    from tldw_chatbook.Agents.code_review_models import ReviewRefused
    from tldw_chatbook.Agents.run_context import CurrentRunActor

    c = dispatch_case
    c.run_check_handoff(outcome="executed")
    state = c.check_observation_state
    actor = CurrentRunActor("subagent", state["verifier"], "primary")
    child = c.service(actor, grant=c.primary.verification_grant)
    with use_run_actor(actor):
        with pytest.raises(ReviewRefused, match="verification_attempt_closed"):
            child.submit(c.prepared.review_id, "dispositions", [], 2, "late")
    for review_id, proposal, receipt in [
        (c.prepared.review_id, state["proposal"], "forged"),
        (
            c.prepared.review_id,
            "different-proposal",
            state["receipt"]["native_execution_receipt_id"],
        ),
    ]:
        with pytest.raises(ReviewRefused, match="invalid_native_check_receipt"):
            c.primary.register_check_result(review_id, proposal, receipt)
    with use_run_actor(c.actor):
        with pytest.raises(ReviewRefused, match="verification_attempt_closed"):
            c.primary.submit(
                c.prepared.review_id,
                "check_actions",
                [
                    {
                        "schema_version": 1,
                        "proposal_id": state["proposal"],
                        "action": "claim",
                    }
                ],
                1,
                "late-action",
            )


@pytest.mark.parametrize("native", [False, True])
def test_minimal_check_receipt_refusal_retains_host_fact_and_unresolved_claim(
    dispatch_case, native
):
    observed = dispatch_case.run_check_handoff(
        outcome="transport_unavailable", native_tools=native
    )
    receipt = dispatch_case.check_observation_state["terminal_receipt"]
    assert receipt.outcome == "executed"
    assert observed.execution_launch_count == 1
    assert not observed.result_seen_by_same_verifier
    assert observed.accepted_unit_dispositions[0]["verdict"] == "unresolved"
    assert (
        observed.accepted_unit_dispositions[0]["reason"]
        == "check_result_transport_unavailable"
    )


def test_verifier_deadline_expires_pending_check_without_launch(dispatch_case):
    observed = dispatch_case.run_check_handoff(outcome="deadline")
    receipt = dispatch_case.check_observation_state["terminal_receipt"]
    assert receipt.outcome == "expired"
    assert receipt.evidence == "verifier_deadline_expired"
    assert observed.execution_launch_count == 0
    assert observed.verifier_launch_count == 1
    assert observed.accepted_unit_dispositions[0]["verdict"] == "unresolved"


def test_accepted_process_keeps_verifier_wait_bounded(dispatch_case, monkeypatch):
    import time

    from tldw_chatbook.Agents.code_review_checks import current_check_execution

    native_execute = FaultBoundedExecutor.execute
    waits = []

    def execute(executor, *args, **kwargs):
        accepted = executor.on_accepted

        def observe_wait():
            accepted()
            check = current_check_execution().check
            before = check.wait_count
            time.sleep(0.2)  # Fault-controlled external process remains accepted.
            waits.append(check.wait_count - before)

        executor.on_accepted = observe_wait
        return native_execute(executor, *args, **kwargs)

    monkeypatch.setattr(FaultBoundedExecutor, "execute", execute)
    observed = dispatch_case.run_check_handoff(outcome="executed")
    assert observed.result_seen_by_same_verifier
    assert waits and 0 < waits[0] < 15


def test_concurrent_primary_action_replays_keep_one_native_claim(
    dispatch_case, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context

    from tldw_chatbook.Agents.code_review_checks import current_check_execution
    from tldw_chatbook.Agents.code_review_models import ReviewRefused

    native_execute = FaultBoundedExecutor.execute
    replayed = []

    def execute(executor, *args, **kwargs):
        execution = current_check_execution()
        service, check = execution.service, execution.check
        action = {
            "schema_version": 1,
            "proposal_id": check.proposal_id,
            "action": "claim",
        }

        def replay():
            ack = service.submit(
                check.binding.review_id, "check_actions", [action], 0, "check-action"
            )
            claim = service.claim_check(check.binding.review_id, check.proposal_id)
            return ack, claim

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = [pool.submit(copy_context().run, replay) for _ in range(8)]
            replayed.extend(future.result(timeout=5) for future in results)
        assert all(claim is check.claim for _, claim in replayed)
        assert all(ack == replayed[0][0] for ack, _ in replayed)
        with pytest.raises(ReviewRefused, match="submission_conflict"):
            service.submit(
                check.binding.review_id,
                "check_actions",
                [{**action, "reason": "changed replay"}],
                0,
                "check-action",
            )
        with service.store.lease(check.binding.review_id) as context:
            assert (
                context.streams[
                    (check.binding.slot_id, check.binding.attempt_id)
                ].revision
                == 1
            )
            assert (
                context.check_streams[
                    (check.primary_run_id, check.binding.attempt_id)
                ].revision
                == 1
            )
        return native_execute(executor, *args, **kwargs)

    monkeypatch.setattr(FaultBoundedExecutor, "execute", execute)
    observed = dispatch_case.run_check_handoff(outcome="executed")
    assert len(replayed) == 8
    assert observed.execution_launch_count == 1
    assert observed.result_seen_by_same_verifier


def test_close_after_claim_releases_unstarted_disposable_owner(
    dispatch_case, monkeypatch
):
    from tldw_chatbook.Agents.code_review_models import ReviewRefused
    from tldw_chatbook.Agents.code_review_service import CodeReviewService

    native_submit = CodeReviewService._submit
    executions = []

    def submitted(service, review_id, kind, records, revision, submission_id):
        result = native_submit(
            service, review_id, kind, records, revision, submission_id
        )
        if kind == "check_actions" and records[0]["action"] == "claim":
            with service.store.lease(review_id) as context:
                executions.append(context.checks[records[0]["proposal_id"]].execution)
            service.store.close()
        return result

    monkeypatch.setattr(CodeReviewService, "_submit", submitted)
    with pytest.raises(ReviewRefused, match="review_expired"):
        dispatch_case.run_check_handoff(outcome="executed")
    assert len(executions) == 1
    assert not executions[0].started
    assert executions[0].released
    assert not executions[0].boundary.root.exists()
    assert dispatch_case.store.usage().reserved_bytes == 0


@pytest.mark.parametrize("winner", ["close", "accept", "uncertain_accept"])
def test_native_admission_serializes_close_and_preserves_process_owner(
    dispatch_case, monkeypatch, winner
):
    import threading

    from tldw_chatbook.Agents.code_review_checks import current_check_execution
    from tldw_chatbook.Agents.code_review_models import ReviewRefused

    native_execute = FaultBoundedExecutor.execute
    observed = {}

    def execute(executor, request, *, cancel_event, on_event, admit_worker):
        execution = current_check_execution()
        observed.update(execution=execution, executor=executor)
        close_started, close_done = threading.Event(), threading.Event()

        def close():
            close_started.set()
            execution.service.store.close()
            close_done.set()

        def admit(tree, commit_launch):
            if winner == "close":
                close()
                assert not execution.released
                assert execution.service.store.usage().reserved_bytes > 0
                return admit_worker(tree, commit_launch)

            def commit():
                closer = threading.Thread(target=close)
                observed["closer"] = closer
                closer.start()
                assert close_started.wait(1)
                assert close_done.wait(1)
                assert execution.service.store._tab.closed.is_set()
                assert execution.accepted_id is None
                return commit_launch()

            accepted = admit_worker(tree, commit)
            assert accepted
            assert close_done.wait(2)
            observed["closer"].join()
            assert execution.accepted_id == request.invocation_id
            assert not execution.released
            assert execution.service.store.usage().reserved_bytes > 0
            return accepted

        return native_execute(
            executor,
            request,
            cancel_event=cancel_event,
            on_event=on_event,
            admit_worker=admit,
        )

    monkeypatch.setattr(FaultBoundedExecutor, "execute", execute)
    with pytest.raises(ReviewRefused, match="review_expired"):
        dispatch_case.run_check_handoff(
            outcome="uncertain" if winner == "uncertain_accept" else "executed"
        )
    execution, executor = observed["execution"], observed["executor"]
    assert len(executor.launches) == int(winner != "close")
    if winner == "uncertain_accept":
        assert not execution.released
        assert dispatch_case.store.usage().reserved_bytes > 0
        executor.outcome = "executed"  # External owner now proves deletion.
        assert execution.release()
    assert execution.released
    assert dispatch_case.store.usage().reserved_bytes == 0
    assert not execution.boundary.root.exists()


@pytest.mark.parametrize("native", [False, True])
def test_published_decline_excludes_opposing_claim_and_native_launch(
    dispatch_case, monkeypatch, native
):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context
    from types import SimpleNamespace

    from tldw_chatbook.Agents.code_review_service import CodeReviewService

    publish = CodeReviewService._submit
    wait = CodeReviewService.await_check_result
    raced = threading.Event()
    observed = []

    def retained_wait(service, *args):
        receipt = wait(service, *args)
        # Keep the same actual verifier alive until the opposing action finishes.
        assert raced.wait(5)
        return receipt

    def interleave(service, review_id, kind, records, revision, submission_id):
        ack = publish(service, review_id, kind, records, revision, submission_id)
        if kind == "check_actions" and records[0]["action"] == "decline":
            proposal = records[0]["proposal_id"]

            def claim_and_execute():
                service.submit(
                    review_id,
                    "check_actions",
                    [{"schema_version": 1, "proposal_id": proposal, "action": "claim"}],
                    ack.submission_revision,
                    "opposing-claim",
                )
                claim = service.claim_check(review_id, proposal)
                if claim.tool_name:
                    call = SimpleNamespace(name=claim.tool_name, args=claim.arguments)
                    with service.prepare_check_process(call):
                        service.check_process_provider.invoke(
                            "raw_shell:shell_exec", claim.arguments
                        )
                return claim

            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    observed.append(
                        pool.submit(copy_context().run, claim_and_execute).result(
                            timeout=5
                        )
                    )
            finally:
                raced.set()
        return ack

    monkeypatch.setattr(CodeReviewService, "await_check_result", retained_wait)
    monkeypatch.setattr(CodeReviewService, "_submit", interleave)
    run = dispatch_case.run_check_handoff(outcome="declined", native_tools=native)
    assert run.execution_launch_count == 0, (
        f"published decline allowed {run.execution_launch_count} native launches"
    )
    assert observed and observed[0].tool_name is None
    assert observed[0].receipt.outcome == "declined"
    assert run.result_seen_by_same_verifier


@pytest.mark.parametrize("outcome", ["executed", "uncertain"])
def test_native_acceptance_wins_opposing_decline_and_claim_replay(
    dispatch_case, monkeypatch, outcome
):
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context

    from tldw_chatbook.Agents.code_review_checks import current_check_execution
    from tldw_chatbook.Agents.code_review_models import ReviewRefused

    execute = FaultBoundedExecutor.execute
    observed = []

    def accepted_first(executor, *args, **kwargs):
        execution = current_check_execution()
        service, check = execution.service, execution.check
        accepted = executor.on_accepted

        def oppose():
            accepted()

            def decline():
                with pytest.raises(
                    ReviewRefused, match="check_execution_already_accepted"
                ):
                    service.submit(
                        check.binding.review_id,
                        "check_actions",
                        [
                            {
                                "schema_version": 1,
                                "proposal_id": check.proposal_id,
                                "action": "decline",
                            }
                        ],
                        1,
                        "too-late-decline",
                    )
                ack = service.submit(
                    check.binding.review_id,
                    "check_actions",
                    [
                        {
                            "schema_version": 1,
                            "proposal_id": check.proposal_id,
                            "action": "claim",
                        }
                    ],
                    0,
                    "check-action",
                )
                assert ack.submission_revision == 1
                assert (
                    service.claim_check(check.binding.review_id, check.proposal_id)
                    is check.claim
                )

            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(copy_context().run, decline).result(timeout=5)
            assert execution.accepted_id == check.native_execution_receipt_id
            assert not execution.released and service.store.usage().reserved_bytes > 0
            observed.append(execution)

        executor.on_accepted = oppose
        return execute(executor, *args, **kwargs)

    monkeypatch.setattr(FaultBoundedExecutor, "execute", accepted_first)
    run = dispatch_case.run_check_handoff(outcome=outcome)
    assert run.execution_launch_count == 1
    assert run.result_seen_by_same_verifier
    assert len(observed) == 1
    execution = observed[0]
    if outcome == "uncertain":
        assert not execution.released and dispatch_case.store.usage().reserved_bytes > 0
        dispatch_case.check_executor.outcome = "executed"
        assert execution.release()
    assert execution.released


def test_decline_write_failure_preserves_claim_then_replay_commits_once(
    dispatch_case, monkeypatch
):
    import json

    from tldw_chatbook.Agents.code_review_models import ReviewRefused
    from tldw_chatbook.Agents.code_review_service import CodeReviewService
    from tldw_chatbook.Agents.code_review_store import ReviewReservation

    publish, write = CodeReviewService._submit, ReviewReservation.write
    fail = False
    tested = []

    def fail_decline(reservation, data, **kwargs):
        nonlocal fail
        try:
            row = json.loads(data)
        except (ValueError, TypeError):
            row = {}
        if (
            fail
            and isinstance(row, dict)
            and row.get("kind") == "check_actions"
            and row["records"][0]["action"] == "decline"
        ):
            fail = False
            raise OSError("fault-controlled decline chunk failure")
        return write(reservation, data, **kwargs)

    def after_claim(service, review_id, kind, records, revision, submission_id):
        nonlocal fail
        ack = publish(service, review_id, kind, records, revision, submission_id)
        if kind != "check_actions" or records[0]["action"] != "claim" or tested:
            return ack
        with service.store.lease(review_id) as context:
            check = context.checks[records[0]["proposal_id"]]
            execution = check.execution
            assert execution is not None and not execution.started
            action = [
                {
                    "schema_version": 1,
                    "proposal_id": check.proposal_id,
                    "action": "decline",
                }
            ]
            reserved = service.store.usage().reserved_bytes
            fail = True
            with pytest.raises(ReviewRefused, match="publication_failed"):
                service.submit(
                    review_id,
                    "check_actions",
                    action,
                    ack.submission_revision,
                    "decline-fault",
                )
            stream = context.check_streams[
                (check.primary_run_id, check.binding.attempt_id)
            ]
            assert stream.revision == 1 and "decline-fault" not in stream.submissions
            assert check.receipt is None and check.claim is not None
            assert not execution.released and execution.boundary.root.exists()
            assert service.store.usage().reserved_bytes == reserved
            declined = service.submit(
                review_id, "check_actions", action, 1, "decline-fault"
            )
            receipt = check.receipt
            assert receipt.outcome == "declined" and execution.released
            assert check.result_reservation is None
            assert service.store.usage().reserved_bytes == 0
            assert (
                service.submit(review_id, "check_actions", action, 1, "decline-fault")
                == declined
            )
            assert check.receipt is receipt and stream.revision == 2
            chunk = json.loads(
                service.store.read_file(
                    context, context.bundle_path / context._head_chunk
                )
            )
            assert chunk["host_check_result"]["receipt_id"] == receipt.receipt_id
            tested.append(execution)
        return ack

    monkeypatch.setattr(ReviewReservation, "write", fail_decline)
    monkeypatch.setattr(CodeReviewService, "_submit", after_claim)
    observed = dispatch_case.run_check_handoff(outcome="executed")
    assert len(tested) == 1 and observed.execution_launch_count == 0
    assert observed.result_seen_by_same_verifier


def test_failed_claim_preparation_returns_no_ack_and_same_action_can_retry(
    dispatch_case, monkeypatch
):
    from tldw_chatbook.Agents.code_review_models import ReviewRefused
    from tldw_chatbook.Agents.code_review_service import CodeReviewService

    publish = CodeReviewService._submit
    reserve = FaultBoundedExecutor.reserve_review_boundary
    fail = True
    recovered = []

    def failing_reserve(executor, **kwargs):
        nonlocal fail
        if fail:
            fail = False
            raise OSError("fault-controlled boundary preparation failure")
        return reserve(executor, **kwargs)

    def retry(service, review_id, kind, records, revision, submission_id):
        try:
            return publish(service, review_id, kind, records, revision, submission_id)
        except ReviewRefused as exc:
            assert exc.reason == "source_authority_unavailable"
            assert kind == "check_actions" and records[0]["action"] == "claim"
            with service.store.lease(review_id) as context:
                check = context.checks[records[0]["proposal_id"]]
                stream = context.check_streams[
                    (check.primary_run_id, check.binding.attempt_id)
                ]
                assert stream.revision == 1 and submission_id in stream.submissions
                assert (
                    check.claim is None
                    and check.execution is None
                    and check.receipt is None
                )
                assert (
                    service.store.usage().reserved_bytes == 8192
                )  # proposal result owner only
            ack = publish(service, review_id, kind, records, revision, submission_id)
            recovered.append(ack)
            assert ack.submission_revision == 1
            return ack

    monkeypatch.setattr(
        FaultBoundedExecutor, "reserve_review_boundary", failing_reserve
    )
    monkeypatch.setattr(CodeReviewService, "_submit", retry)
    run = dispatch_case.run_check_handoff(outcome="executed")
    assert len(recovered) == 1 and run.execution_launch_count == 1
    assert run.result_seen_by_same_verifier
