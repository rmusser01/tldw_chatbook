"""Public review authority and preparation with real source/store/preset owners."""

from contextlib import contextmanager
from dataclasses import dataclass, field

import pytest

from Tests.Agents import test_code_review_source as source_fixtures
from tldw_chatbook.Agents.agent_models import AgentDefinition
from tldw_chatbook.Agents.agent_routing import AgentsRoutingConfig
from tldw_chatbook.Agents.code_review_models import (
    ReviewerChoice,
    ReviewRunBinding,
    ReviewTeam,
)
from tldw_chatbook.Agents.code_review_service import (
    CodeReviewService,
    ReviewCapacity,
    ReviewStageIssuer,
    request_fingerprints,
)
from tldw_chatbook.Agents.code_review_tool_provider import CodeReviewToolProvider
from tldw_chatbook.Agents.run_context import CurrentRunActor, use_run_actor
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

source_case = source_fixtures.source_case


def finding_payload() -> dict:
    return {
        "schema_version": 1,
        "local_key": "empty-input",
        "title": "Empty input raises before validation",
        "locations": [{"source_ref": "src-1", "start_line": 4, "end_line": 4}],
        "trigger": "Call parse with an empty string",
        "impact": "IndexError replaces the documented validation result",
        "evidence": [
            {
                "kind": "source",
                "source_ref": "src-1",
                "start_line": 4,
                "end_line": 4,
                "digest": "a" * 64,
            }
        ],
        "severity": "medium",
        "uncertainty": "No execution performed",
    }


def coverage_payload() -> dict:
    return {
        "schema_version": 1,
        "inspected_refs": ["src-1"],
        "gaps": [],
        "empty_findings": False,
    }


@pytest.fixture
def review_case(source_case):
    c = source_case
    db = AgentRunsDB(":memory:", client_id="review-test")
    preset = db.create_agent_definition(
        AgentDefinition(
            name="reviewer",
            instructions="Review independently",
            provider="openai",
            model="test",
        )
    )
    verifier = db.create_agent_definition(
        AgentDefinition(
            name="verifier",
            instructions="Verify independently",
            provider="openai",
            model="verify",
        )
    )
    c.team = ReviewTeam(
        reviewers=(ReviewerChoice(preset_id=preset),), verifier_preset_id=verifier
    )
    c.owner = object()
    c.issuer = ReviewStageIssuer(c.owner)
    c.actor = CurrentRunActor("primary", "primary", None)
    c.capacity = ReviewCapacity(
        remaining_children=2,
        children_per_turn=2,
        remaining_continuations=1,
        remaining_input_tokens=10000,
    )
    c.app_config = {"tools": {"code_review_enabled": True}, "api_settings": {}}
    c.routing = AgentsRoutingConfig(
        spawn_override_enabled=True, spawn_override_allowlist=("openai/*",)
    )
    c.db = db

    def service(
        actor=None,
        *,
        limit=16000,
        reviewers=None,
        verifier_choice=None,
        grant=None,
        store=None,
    ):
        store = store or c.store
        fingerprints = request_fingerprints(
            c.working_tree_target, c.team, reviewers, verifier_choice
        )
        grant = grant or c.issuer.issue(
            c.owner,
            session_id=store._session_id,
            acceptance_id="accepted",
            stage="review",
            source_epoch=0,
            **fingerprints,
        )
        return CodeReviewService(
            store=store,
            issuer=c.issuer,
            grant=grant,
            actor=actor or c.actor,
            source_authority=c.authority,
            preset_db=c.db,
            app_config=c.app_config,
            routing=c.routing,
            team=c.team,
            primary_provider="openai",
            primary_model="parent",
            capacity=lambda: c.capacity,
            destination_check=lambda target: None,
            max_tool_result_chars=limit,
        )

    c.service = service
    c.primary = service()
    c.provider = CodeReviewToolProvider(c.primary)

    def prepare(**kwargs):
        with use_run_actor(c.actor):
            return c.primary.prepare(c.working_tree_target, **kwargs)

    c.prepare = prepare

    @contextmanager
    def participant(prepared, role="reviewer", slot="R1", limit=16000):
        actor = CurrentRunActor("subagent", role + slot, "primary")
        with c.store.lease(prepared.review_id) as context:
            binding = ReviewRunBinding(
                review_id=prepared.review_id,
                slot_id=slot,
                attempt_id="attempt",
                role=role,
                target_digest=context.target_digest,
                source_epoch=0,
                check_feedback_mode="unavailable_inline"
                if role == "verifier"
                else None,
            )
        c.store.bind_attempt(actor.run_id, binding)
        grant = c.issuer.issue(
            c.owner,
            session_id="chat",
            acceptance_id="accepted",
            stage="verify" if role == "verifier" else "review",
            source_epoch=0,
            **request_fingerprints(c.working_tree_target, c.team),
        )
        provider = CodeReviewToolProvider(service(actor, limit=limit, grant=grant))
        with use_run_actor(actor):
            yield provider

    c.participant = participant
    yield c
    c.issuer.close(c.owner)
    db.close()


review_harness = review_case


@dataclass
class ReviewRunObservation:
    verifier_launch_count: int
    history_reductions: int
    all_provider_requests_fit: bool
    accepted_unit_dispositions: tuple[dict, ...]
    replayed_tool_executions: int
    requests: tuple[dict, ...]
    expanded: tuple[dict, ...]
    wait_yielded_while_verifier_live: bool = False
    execution_launch_count: int = 0
    result_seen_by_same_verifier: bool = False
    check_events: list[str] = field(default_factory=list)
    host_check_wait_count: int = 0
    primary_check_notice_count: int = 0
    feedback_mode: str | None = None
    receipts: tuple[dict, ...] = ()
    reserved_bytes: int = 0


def install_batched_verifier(c, monkeypatch):
    """Script only the external gateway; retain real dispatch/tool/store owners."""

    def run_batched_verifier(
        *,
        units,
        context_window,
        native_tools,
        run_log_enabled,
        max_tool_result_chars=16000,
        oversized=False,
    ):
        return _run_batched_verifier(
            c,
            monkeypatch,
            units=units,
            context_window=context_window,
            native_tools=native_tools,
            run_log_enabled=run_log_enabled,
            max_tool_result_chars=max_tool_result_chars,
            oversized=oversized,
        )

    c.run_batched_verifier = run_batched_verifier


def _run_batched_verifier(
    c,
    monkeypatch,
    *,
    units,
    context_window,
    native_tools,
    run_log_enabled,
    max_tool_result_chars,
    oversized,
):
    from Tests.Agents import test_code_review_dispatch
    from Tests.Agents.test_code_review_reports import coverage, finding
    from tldw_chatbook.Agents.agent_models import AgentConfig
    from tldw_chatbook.Agents.agent_service import ModelRequest
    from tldw_chatbook.Utils import token_counter
    from tldw_chatbook.Utils.token_counter import ContextWindowResolution

    monkeypatch.setattr(
        token_counter,
        "resolve_context_window",
        lambda *a, **kw: ContextWindowResolution(
            context_window, "scripted model metadata", True
        ),
    )
    c.app_config.setdefault("agents", {})["run_log_enabled"] = run_log_enabled
    state = {"cursor": None, "revision": 0, "phase": "read", "accepted": []}

    def prepare(c):
        with c.participant(c.prepared) as provider:
            service = provider.service
            for chunk in range((units + 7) // 8):
                rows = [
                    finding(
                        c,
                        c.prepared,
                        str(i),
                        trigger=str(i)
                        + " unusual trigger"
                        * (
                            200
                            if oversized and i == units - 1
                            else 90
                            if context_window <= 8192
                            else 1
                        )
                        + '\n"é漢字"',
                    )
                    for i in range(chunk * 8, min(units, chunk * 8 + 8))
                ]
                service.submit(
                    c.prepared.review_id, "findings", rows, chunk, "seed" + str(chunk)
                )
            service.submit(
                c.prepared.review_id,
                "reviewer_complete",
                [coverage(c, c.prepared, empty=False)],
                (units + 7) // 8,
                "complete",
            )
            service.capture_review_output(service.actor.run_id, "", terminal=True)
            service.settle_review_run(
                service.actor.run_id, terminal_status="done", receipt_id="terminal"
            )
        with use_run_actor(c.actor):
            c.primary.read(c.prepared.review_id, "summary")

    def step():
        if state["phase"] == "done":
            return "Verification complete"
        if state["phase"] == "read":
            state["phase"] = "submit"
            return c.reply(
                "code_review_read",
                {
                    "review_id": c.prepared.review_id,
                    "view": "batch",
                    "cursor": state["cursor"],
                },
            )
        pages = test_code_review_dispatch.model_pages(c.calls[-1][1])
        assert pages, c.calls[-1][1]["messages_payload"][-1]
        page = pages[-1]
        state["cursor"] = page["cursor"]
        rows = [
            {
                "schema_version": 1,
                "verification_unit_id": row["verification_unit_id"],
                "verdict": "confirmed",
                "reason": "Captured source demonstrates the trigger",
                "evidence": row["claim"]["evidence"],
            }
            for row in page["records"]
        ]
        if not rows and page["cursor"] is None and page["gaps"]:
            state["phase"] = "done"
            return "Verification complete with explicit gaps"
        assert rows, page
        state["accepted"].extend(row["verification_unit_id"] for row in rows)
        state["phase"] = "read" if page["cursor"] else "done"
        revision = state["revision"]
        state["revision"] += 1
        return c.reply(
            "code_review_submit",
            {
                "review_id": c.prepared.review_id,
                "kind": "dispositions",
                "records": rows,
                "expected_revision": revision,
                "submission_id": "batch" + str(revision),
            },
        )

    c.run(
        verify=True,
        native=native_tools,
        run_log_enabled=run_log_enabled,
        prepare_hook=prepare,
        limit=max_tool_result_chars,
        child_script=[
            lambda: c.reply(
                "load_tools",
                {
                    "ids": [
                        "code_review:code_review_read",
                        "code_review:code_review_submit",
                    ]
                },
            )
        ]
        + [step] * 75,
    )
    children = [
        (actor, request) for actor, request in c.calls if actor.kind == "subagent"
    ]
    with use_run_actor(c.actor):
        expanded = c.primary.expand_dispositions(c.prepared.review_id)
    with c.store.lease(c.prepared.review_id) as context:
        binding = next(
            value
            for value in context.active_attempts.values()
            if value.role == "verifier"
        )
        stream = context.streams[(binding.slot_id, binding.attempt_id)]
        accepted = tuple(stream.verification.dispositions.values())
        submissions = len(stream.submissions)
    runs = [row for row in c.db.list_runs("chat") if row["agent_kind"] == "subagent"]
    executions = sum(
        step["kind"] == "tool_result" and step["tool_name"] == "code_review_submit"
        for row in runs
        for step in c.db.get_run(row["id"])["steps"]
    )
    return ReviewRunObservation(
        verifier_launch_count=len({actor.run_id for actor, _ in children}),
        history_reductions=sum(
            "Accepted verification checkpoint" in str(request["messages_payload"])
            for _, request in children
        ),
        all_provider_requests_fit=all(
            c.runtime.review_input_capacity(
                AgentConfig(model=request["model"], system_prompt=""),
                request["api_endpoint"],
                ModelRequest(
                    messages=tuple(request["messages_payload"]),
                    tools=tuple(request.get("tools", ())),
                ),
            )
            >= 0
            for _, request in children
        ),
        accepted_unit_dispositions=accepted,
        replayed_tool_executions=max(0, executions - submissions),
        requests=tuple(request for _, request in children),
        expanded=expanded,
    )


def install_check_handoff(c):
    """Expose the slice-nine scenarios using the actual turn/executor factory."""
    c.run_check_handoff = lambda **kwargs: _run_check(c, **kwargs)
    c.run_inline_check = lambda **kwargs: _run_check(c, inline=True, **kwargs)


def _run_check(
    c,
    *,
    outcome="executed",
    primary_waiting=True,
    duplicate_actions=False,
    native_tools=True,
    inline=False,
    max_live_at_prepare=1,
    max_live_at_launch=1,
    result_limit=16000,
    source_only=False,
    forged_evidence=False,
):
    import threading

    from Tests.Agents.test_code_review_dispatch import model_pages
    from Tests.Chat.test_console_code_review_checks import (
        FaultBoundedExecutor,
        seed_original,
    )
    from tldw_chatbook.Agents.agent_models import AgentConfig
    from tldw_chatbook.Agents.agent_service import ModelRequest
    from tldw_chatbook.Agents.raw_shell_tool_provider import RawShellToolProvider
    from tldw_chatbook.Agents.run_context import current_run_actor
    from tldw_chatbook.Chat.console_raw_cli import RawCliRuntime
    from tldw_chatbook.MCP.permission_store import EffectiveToolState

    c.use_factory = True
    launch_inline = inline and max_live_at_launch == 1
    ready = threading.Event()
    if not primary_waiting:
        ready.set()
    state = {"pages": [], "events": [], "receipt": None, "live": False}
    executor = FaultBoundedExecutor(outcome=outcome)
    runtime = RawCliRuntime(
        lambda: True, executor=None if outcome == "stock" else executor
    )
    runtime.arm()
    if outcome != "permission_denied":
        runtime.grant_model_session("chat")
    c.process_provider = RawShellToolProvider(
        runtime=runtime,
        console_session_id="chat",
        initial_directory=lambda: c.root,
        resolve_state=lambda _: EffectiveToolState(
            state="deny" if outcome == "permission_denied" else "ask",
            origin="global_default",
        ),
    )

    def last_child_page():
        return model_pages(
            [request for actor, request in c.calls if actor.kind == "subagent"][-1]
        )[-1]

    def batch():
        state["verifier"] = current_run_actor().run_id
        return c.reply(
            "code_review_read", {"review_id": c.prepared.review_id, "view": "batch"}
        )

    def proposal():
        if not launch_inline:
            assert ready.wait(5)
        state["unit"] = last_child_page()["records"][0]
        return c.reply(
            "code_review_submit",
            {
                "review_id": c.prepared.review_id,
                "kind": "check_proposals",
                "records": [
                    {
                        "schema_version": 1,
                        "argv": ["python", "-m", "pytest"],
                        "intent": "Check captured source",
                        "verification_unit_ids": [
                            state["unit"]["verification_unit_id"]
                        ],
                    }
                ],
                "expected_revision": 0,
                "submission_id": "check-proposal",
            },
        )

    def read_result():
        if outcome == "transport_unavailable":
            c.review_adapters[current_run_actor().run_id].result_limit = 400
        if outcome == "deadline":
            import time

            c.review_adapters[current_run_actor().run_id].check_deadline = (
                time.monotonic() - 1
            )
        with c.store.lease(c.prepared.review_id) as context:
            check = next(iter(context.checks.values()))
            state["proposal"] = check.proposal_id
            state["mode"] = check.binding.check_feedback_mode
            if "proposal" not in state["events"]:
                state["events"].append("proposal")
        return c.reply(
            "code_review_read",
            {
                "review_id": c.prepared.review_id,
                "view": "verification_batch",
                "document_ref": check.proposal_id,
            },
        )

    def decide():
        if outcome == "transport_unavailable":
            return "Check result cannot be transported; leave the claim unresolved"
        page = last_child_page()
        state["pages"].append(page)
        if page["cursor"]:
            c.child_scripts[current_run_actor().run_id].insert(0, decide)
            return c.reply(
                "code_review_read",
                {
                    "review_id": c.prepared.review_id,
                    "view": "verification_batch",
                    "document_ref": state["proposal"],
                    "cursor": page["cursor"],
                },
            )
        with c.store.lease(c.prepared.review_id) as context:
            check = context.checks[state["proposal"]]
            state["receipt"] = check.receipt.model_dump(mode="json")
            state["same"] = (
                check.result_read
                and current_run_actor().run_id
                == check.verifier_run_id
                == state["verifier"]
            )
            if "receipt" not in state["events"]:
                state["events"].append("receipt")
            state["events"].append("verifier_read")
        receipt = state["receipt"]
        evidence = (
            state["unit"]["claim"]["evidence"]
            if source_only
            else (
                [{"kind": "check", "receipt_id": receipt["receipt_id"]}]
                if receipt["outcome"] == "executed" or forged_evidence
                else []
            )
        )
        return c.reply(
            "code_review_submit",
            {
                "review_id": c.prepared.review_id,
                "kind": "dispositions",
                "records": [
                    {
                        "schema_version": 1,
                        "verification_unit_id": state["unit"]["verification_unit_id"],
                        "verdict": "confirmed" if evidence else "unresolved",
                        "reason": "Captured evidence"
                        if evidence
                        else receipt["evidence"],
                        "evidence": evidence,
                        "next_check": None
                        if evidence
                        else "Run the proposed test in an authorized bounded environment",
                    }
                ],
                "expected_revision": 1,
                "submission_id": "decision",
            },
        )

    def done():
        with c.store.lease(c.prepared.review_id) as context:
            check = context.checks[state["proposal"]]
            ledger = c.store.verification_ledger(context, check.binding)
            if ledger.dispositions:
                state["events"].append("disposition")
        return "Verification finished"

    def wait():
        ready.set()
        if not primary_waiting:
            import time

            until = time.monotonic() + 5
            while time.monotonic() < until:
                with c.store.lease(c.prepared.review_id) as context:
                    if context.checks:
                        break
                time.sleep(0.01)
        return c.reply("wait_agents", {})

    def action():
        with c.store.lease(c.prepared.review_id) as context:
            check = next(iter(context.checks.values()))
            state["proposal"] = check.proposal_id
            if "wait_yield" not in state["events"]:
                state["live"] = context.streams[
                    (check.binding.slot_id, check.binding.attempt_id)
                ].terminal_status is None and "check_requested" in str(
                    [request for actor, request in c.calls if actor.kind == "primary"][
                        -1
                    ]["messages_payload"]
                )
                if "proposal" not in state["events"]:
                    state["events"].append("proposal")
                state["events"].append("wait_yield")
            state["check"] = check
        if outcome == "primary_ended":
            return "Primary ended before deciding the proposal"
        if outcome == "verifier_cancelled":
            for cancel in c.runtime._fleet_cancels.values():
                cancel.set()
            return "Verifier was cancelled"
        if outcome == "source_fence":
            object.__setattr__(c.authority.admitted, "guard", lambda write: False)
            return "Source authority was revoked"
        if outcome == "close":
            c.store.close()
            return "Chat closed"
        return c.reply(
            "code_review_submit",
            {
                "review_id": c.prepared.review_id,
                "kind": "check_actions",
                "records": [
                    {
                        "schema_version": 1,
                        "proposal_id": check.proposal_id,
                        "action": "decline" if outcome == "declined" else "claim",
                    }
                ],
                "expected_revision": 0,
                "submission_id": "check-action",
            },
        )

    def summary():
        with c.store.lease(c.prepared.review_id) as context:
            check = context.checks[state["proposal"]]
            if check.claim:
                state["events"].append("claim")
        return c.reply(
            "code_review_read",
            {
                "review_id": c.prepared.review_id,
                "view": "summary",
                "document_ref": state["proposal"],
            },
        )

    def process():
        with c.store.lease(c.prepared.review_id) as context:
            check = context.checks[state["proposal"]]
            if check.claim and check.claim.tool_name:
                return c.reply(check.claim.tool_name, check.claim.arguments)
        return "No executable check available"

    def parent(c):
        rows = [
            c.reply(
                "load_tools",
                {
                    "ids": [
                        "code_review:code_review_read",
                        "code_review:code_review_submit",
                        "raw_shell:shell_exec",
                    ]
                },
            ),
            c.reply(
                "spawn_subagent",
                {"review_id": c.prepared.review_id, "review_slot_id": "V1"},
            ),
        ]
        if not launch_inline:
            rows += (
                [wait, action]
                + ([action] if duplicate_actions else [])
                + [summary, process]
            )
        return rows

    executor.on_accepted = lambda: state["events"].append("execution_accepted")
    c.run(
        verify=True,
        inline=launch_inline,
        preparation_inline=max_live_at_prepare == 1 if inline else False,
        native=native_tools,
        limit=result_limit,
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
            proposal,
            read_result,
            decide,
            done,
        ],
    )
    from tldw_chatbook.Agents.code_review_models import ReviewRefused

    try:
        with use_run_actor(c.actor):
            expanded = c.primary.expand_dispositions(c.prepared.review_id)
        with c.store.lease(c.prepared.review_id) as context:
            check = next(iter(context.checks.values()))
    except ReviewRefused:
        if outcome not in {"source_fence", "close"}:
            raise
        expanded = ()
        check = state["check"]
    state["closed_binding"] = check.binding
    state["terminal_receipt"] = check.receipt
    c.check_observation_state = state
    c.check_executor = executor
    return ReviewRunObservation(
        verifier_launch_count=len(
            {actor.run_id for actor, _ in c.calls if actor.kind == "subagent"}
        ),
        history_reductions=sum(
            "Accepted verification checkpoint" in str(request["messages_payload"])
            for _, request in c.calls
        ),
        all_provider_requests_fit=all(
            c.runtime.review_input_capacity(
                AgentConfig(model=request["model"], system_prompt=""),
                request["api_endpoint"],
                ModelRequest(
                    messages=tuple(request["messages_payload"]),
                    tools=tuple(request.get("tools", ())),
                ),
            )
            >= 0
            for actor, request in c.calls
            if actor.kind == "subagent"
        ),
        accepted_unit_dispositions=tuple(row["disposition"] for row in expanded),
        replayed_tool_executions=max(0, len(executor.launches) - 1),
        requests=tuple(request for _, request in c.calls),
        expanded=tuple(expanded),
        wait_yielded_while_verifier_live=state["live"],
        execution_launch_count=len(executor.launches),
        result_seen_by_same_verifier=state.get("same", False),
        check_events=state["events"],
        host_check_wait_count=check.wait_count,
        primary_check_notice_count=check.notice_count,
        feedback_mode=state.get("mode"),
        receipts=tuple(state["pages"]),
        reserved_bytes=c.store.usage().reserved_bytes,
    )
