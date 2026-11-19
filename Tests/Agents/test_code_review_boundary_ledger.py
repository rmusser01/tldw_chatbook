"""Real native receipts cross the existing SQLite wake lifecycle atomically."""

from dataclasses import replace

import pytest

from Tests.Chat import test_console_code_review_authority as native_helpers
from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused

source_case = native_helpers.source_case
review_case = native_helpers.review_case
dispatch_case = native_helpers.dispatch_case
native_dispatch = native_helpers.native_dispatch
pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def ledger_case(native_dispatch):
    from Tests.Agents.test_code_review_reports import coverage

    c = native_dispatch

    def complete():
        return c.reply(
            "code_review_submit",
            dict(
                review_id=c.prepared.review_id,
                kind="reviewer_complete",
                records=[coverage(c, c.prepared)],
                expected_revision=0,
                submission_id="done",
            ),
        )

    c.run(reviewers=3, child_script=[complete, "finished"], work_chain_id=c.chain_id)
    c.live_boundary = c.coordinator.seal_next_boundary(c.prepared.review_id)
    assert c.coordinator.validate_boundary(c.live_boundary)
    c.ledger = c.db.automatic_work
    c.owner_id = "runtime"
    c.ledger.register_review_boundary_authority(
        c.owner_id,
        validate=c.coordinator.validate_boundary,
        transition=c.coordinator.boundary_transition,
        acceptance_scope=c.coordinator.boundary_acceptance_scope,
    )
    c.claim = lambda attempt="attempt-1", receipt=c.live_boundary: c.ledger.claim_wake(
        c.chain_id,
        attempt_id=attempt,
        owner_id=c.owner_id,
        session_id="chat",
        run_ids=(),
        review_boundary=receipt,
    )
    return c


def links(c):
    with c.db.connection() as conn:
        return [
            dict(row)
            for row in conn.execute("SELECT * FROM automatic_review_boundary_claims")
        ]


def test_empty_ordinary_wake_remains_invalid(ledger_case):
    c = ledger_case
    with pytest.raises(ValueError, match="must not be empty"):
        c.ledger.claim_wake(
            c.chain_id,
            attempt_id="empty",
            owner_id=c.owner_id,
            session_id="chat",
            run_ids=(),
        )


def test_valid_boundary_claim_replays_one_exact_reservation(ledger_case):
    c = ledger_case
    first = c.claim()
    assert first.state == "prepared"
    assert first.review_boundary_id == c.live_boundary.boundary_id
    assert c.claim() == first
    assert len(links(c)) == 1
    assert c.ledger.snapshot(c.chain_id).reserved["generation"] == 1
    with pytest.raises(AutomaticWorkRefused, match="review_boundary_already_claimed"):
        c.claim("another")


def test_forged_foreign_and_revoked_receipts_cannot_claim(ledger_case):
    c = ledger_case
    for receipt in (
        replace(c.live_boundary),
        replace(c.live_boundary, chain_id="foreign"),
        replace(c.live_boundary, source_epoch=1),
    ):
        with pytest.raises((ValueError, AutomaticWorkRefused), match="review_boundary"):
            c.claim(receipt=receipt)
    c.coordinator.fence_session("chat")
    with pytest.raises(AutomaticWorkRefused, match="review_boundary_unavailable"):
        c.claim()
    assert links(c) == []
    assert c.ledger.snapshot(c.chain_id).reserved["generation"] == 0


def test_proven_preacceptance_abort_releases_only_exact_link_and_reclaims(ledger_case):
    c = ledger_case
    first = c.claim()
    assert c.ledger.abort_wake(first.id, owner_id=c.owner_id)
    assert links(c) == []
    assert c.ledger.snapshot(c.chain_id).reserved["generation"] == 0
    assert c.claim().state == "aborted"
    second = c.claim("second")
    assert second.state == "prepared"
    assert [link["attempt_id"] for link in links(c)] == [second.id]
    assert not c.ledger.abort_wake(first.id, owner_id=c.owner_id)
    assert len(links(c)) == 1


@pytest.mark.parametrize("complete", [False, True])
def test_accepted_and_completed_keep_link_and_charge_without_reclaim(
    ledger_case, complete
):
    c = ledger_case
    first = c.claim()
    assert c.ledger.accept_wake(first.id, owner_id=c.owner_id)
    assert not c.coordinator.validate_boundary(c.live_boundary)
    if complete:
        assert c.ledger.complete_wake(first.id, owner_id=c.owner_id)
        assert not c.ledger.complete_wake(first.id, owner_id=c.owner_id)
    assert not c.ledger.accept_wake(first.id, owner_id=c.owner_id)
    assert not c.ledger.abort_wake(first.id, owner_id=c.owner_id)
    assert len(links(c)) == 1
    assert c.ledger.snapshot(c.chain_id).used["generation"] == 1
    assert c.ledger.snapshot(c.chain_id).reserved["generation"] == 0
    with pytest.raises(AutomaticWorkRefused, match="review_boundary"):
        c.claim("again")


def test_accept_revalidates_live_grant_before_consuming_generation(ledger_case):
    c = ledger_case
    first = c.claim()
    c.coordinator.fence_session("chat")
    with pytest.raises(AutomaticWorkRefused, match="review_boundary_unavailable"):
        c.ledger.accept_wake(first.id, owner_id=c.owner_id)
    assert c.ledger.read_attempt(first.id, owner_id=c.owner_id).state == "prepared"
    assert c.ledger.abort_wake(first.id, owner_id=c.owner_id)
    assert not links(c)


def test_foreign_runtime_recovery_retains_uncertain_evidence(ledger_case):
    c = ledger_case
    first = c.claim()
    c.ledger.accept_wake(first.id, owner_id=c.owner_id)
    c.ledger.recover(current_owner_id="replacement-runtime")
    assert (
        c.ledger.read_attempt(first.id, owner_id=c.owner_id).state == "review_required"
    )
    assert len(links(c)) == 1
    assert c.ledger.snapshot(c.chain_id).used["generation"] == 1
    assert not c.ledger.abort_wake(first.id, owner_id=c.owner_id)


def test_v24_upgrade_and_reopen_preserve_ordinary_history(tmp_path):
    import sqlite3

    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    path = tmp_path / "upgrade.db"
    db = AgentRunsDB(path, client_id="migration")
    parent = db.create_run(conversation_id="chat", agent_kind="primary")
    db.close()
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE automatic_review_boundary_claims")
        connection.execute(
            "ALTER TABLE automatic_wake_attempts DROP COLUMN review_boundary_id"
        )
        connection.execute("DELETE FROM schema_version WHERE version=25")
    for _ in range(2):
        db = AgentRunsDB(path, client_id="migration")
        assert db.get_run(parent)["agent_kind"] == "primary"
        with db.connection() as connection:
            assert (
                connection.execute(
                    "SELECT max(version) FROM schema_version"
                ).fetchone()[0]
                == 25
            )
            assert (
                connection.execute(
                    "SELECT count(*) FROM automatic_review_boundary_claims"
                ).fetchone()[0]
                == 0
            )
            assert "review_boundary_id" in {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(automatic_wake_attempts)"
                )
            }
            ddl = connection.execute(
                "SELECT sql FROM sqlite_master WHERE name='automatic_wake_attempts'"
            ).fetchone()[0]
            assert "'completion', 'progress', 'mixed'" in ddl
            assert "'fleet_wake', 'goal_iteration'" in ddl
        db.close()


def test_stale_sql_predecessor_cannot_replace_live_native_settlement(ledger_case):
    c = ledger_case
    run_id = c.live_boundary.predecessor_run_ids[0]
    with c.db.transaction() as connection:
        connection.execute("DELETE FROM agent_run_steps WHERE run_id=?", (run_id,))
    with pytest.raises(
        ValueError, match="review_boundary predecessor settlement missing"
    ):
        c.claim()
    assert links(c) == []
    assert c.ledger.snapshot(c.chain_id).reserved["generation"] == 0


def test_accept_and_abort_serialize_without_double_refund(ledger_case):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from tldw_chatbook.DB.base_db import operation_owned_connection

    c = ledger_case
    first = c.claim()
    start = threading.Barrier(2)

    def action(method):
        start.wait(5)
        with operation_owned_connection(c.db):
            return method(first.id, owner_id=c.owner_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        accepted = pool.submit(action, c.ledger.accept_wake)
        aborted = pool.submit(action, c.ledger.abort_wake)
        outcomes = accepted.result(10), aborted.result(10)
    assert outcomes in ((True, False), (False, True))
    snapshot = c.ledger.snapshot(c.chain_id)
    assert snapshot.reserved["generation"] == 0
    assert snapshot.used["generation"] == int(outcomes[0])
    assert len(links(c)) == int(outcomes[0])


def test_native_close_during_claim_commit_cannot_accept_or_reactivate(ledger_case):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from tldw_chatbook.DB.base_db import operation_owned_connection

    c = ledger_case
    validated, release = threading.Event(), threading.Event()

    def validate(receipt):
        result = c.coordinator.validate_boundary(receipt)
        validated.set()
        assert release.wait(5)
        return result

    c.ledger.unregister_review_boundary_authority(c.owner_id)
    c.ledger.register_review_boundary_authority(
        c.owner_id,
        validate=validate,
        transition=c.coordinator.boundary_transition,
        acceptance_scope=c.coordinator.boundary_acceptance_scope,
    )

    def claim():
        with operation_owned_connection(c.db):
            return c.claim()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(claim)
        assert validated.wait(5)
        c.coordinator.fence_session("chat")
        release.set()
        attempt = future.result(5)
    with pytest.raises(AutomaticWorkRefused, match="review_boundary_unavailable"):
        c.ledger.accept_wake(attempt.id, owner_id=c.owner_id)
    assert c.ledger.abort_wake(attempt.id, owner_id=c.owner_id)
    assert not links(c)
    assert c.ledger.snapshot(c.chain_id).reserved["generation"] == 0
    assert c.ledger.snapshot(c.chain_id).used["generation"] == 0
    assert not c.coordinator.validate_boundary(c.live_boundary)


@pytest.mark.parametrize("state", ["ready", "prepared", "accepted", "completed"])
def test_manual_resume_preserves_old_boundary_evidence(ledger_case, state):
    from tldw_chatbook.Agents.code_review_models import ReviewRefused
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

    c = ledger_case
    if state != "ready":
        attempt = c.claim()
        if state in {"accepted", "completed"}:
            c.ledger.accept_wake(attempt.id, owner_id=c.owner_id)
        if state == "completed":
            c.ledger.complete_wake(attempt.id, owner_id=c.owner_id)
    old_links = links(c)
    old_budget = c.ledger.snapshot(c.chain_id)
    message = c.chat_store.append_message(
        "chat", role=ConsoleMessageRole.USER, content="Resume review"
    )
    c.coordinator.bind_accepted_request(message.id, session_id="chat", source_epoch=0)
    c.coordinator.bind_stage_admission(
        message.id,
        capacity=lambda: c.capacity,
        routing=c.routing,
        app_config=c.app_config,
        destination_check=lambda target: None,
    )
    if state == "accepted":
        with pytest.raises(ReviewRefused, match="review_attempt_outcome_uncertain"):
            c.coordinator.accept_stage(
                c.prepared.review_id, "resume_review", accepted_message_id=message.id
            )
        assert message.id not in c.coordinator._accepted_stages
    else:
        admitted = c.coordinator.accept_stage(
            c.prepared.review_id, "resume_review", accepted_message_id=message.id
        )
        assert admitted.slots == ("R3",)
        assert not c.coordinator.validate_boundary(c.live_boundary)
    assert links(c) == old_links
    new_budget = c.ledger.snapshot(c.chain_id)
    assert (new_budget.used, new_budget.reserved, new_budget.status) == (
        old_budget.used,
        old_budget.reserved,
        old_budget.status,
    )
    if state == "prepared":
        with pytest.raises(AutomaticWorkRefused, match="review_boundary_unavailable"):
            c.ledger.accept_wake(attempt.id, owner_id=c.owner_id)
        assert c.ledger.abort_wake(attempt.id, owner_id=c.owner_id)
        assert not links(c)


def _manual_resume(c, destination_check):
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

    message = c.chat_store.append_message(
        "chat", role=ConsoleMessageRole.USER, content="Resume review"
    )
    c.coordinator.bind_accepted_request(message.id, session_id="chat", source_epoch=0)
    c.coordinator.bind_stage_admission(
        message.id,
        capacity=lambda: c.capacity,
        routing=c.routing,
        app_config=c.app_config,
        destination_check=destination_check,
    )
    return message.id, lambda: c.coordinator.accept_stage(
        c.prepared.review_id, "resume_review", accepted_message_id=message.id
    )


def test_manual_resume_rechecks_acceptance_after_destination_confirmation(ledger_case):
    from tldw_chatbook.Agents.code_review_models import ReviewRefused

    c = ledger_case
    attempt = c.claim()

    def destination_check(target):
        assert c.ledger.accept_wake(attempt.id, owner_id=c.owner_id)
        return None

    message_id, resume = _manual_resume(c, destination_check)
    with pytest.raises(ReviewRefused, match="review_attempt_outcome_uncertain"):
        resume()
    assert message_id not in c.coordinator._accepted_stages
    assert c.ledger.read_attempt(attempt.id, owner_id=c.owner_id).state == "accepted"
    assert c.coordinator._boundaries[c.live_boundary.boundary_id][2] == "accepted"
    assert len(links(c)) == 1
    assert c.ledger.snapshot(c.chain_id).used["generation"] == 1
    with c.store.lease(c.prepared.review_id) as context:
        assert not context.scheduled_attempts


@pytest.mark.parametrize("phase", ["validated", "committed"])
def test_manual_resume_waits_for_acceptance_commit_and_transition(
    ledger_case, monkeypatch, phase
):
    import threading
    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    from tldw_chatbook.Agents.code_review_models import ReviewRefused
    from tldw_chatbook.DB.base_db import operation_owned_connection

    c = ledger_case
    attempt = c.claim()
    paused, release, consented = threading.Event(), threading.Event(), threading.Event()
    method = (
        "_validate_review_boundary"
        if phase == "validated"
        else "_transition_review_boundary"
    )
    original = getattr(c.ledger, method)

    def pause(*args, **kwargs):
        if phase == "validated":
            original(*args, **kwargs)
        paused.set()
        assert release.wait(5)
        if phase == "committed":
            original(*args, **kwargs)

    monkeypatch.setattr(c.ledger, method, pause)
    message_id, resume = _manual_resume(c, lambda target: consented.set())

    def accept():
        with operation_owned_connection(c.db):
            return c.ledger.accept_wake(attempt.id, owner_id=c.owner_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        accepted = pool.submit(accept)
        try:
            assert paused.wait(5)
            resumed = pool.submit(resume)
            assert consented.wait(5)
            with pytest.raises(TimeoutError):
                resumed.result(0.15)
        finally:
            release.set()
        assert accepted.result(5)
        with pytest.raises(ReviewRefused, match="review_attempt_outcome_uncertain"):
            resumed.result(5)
    assert message_id not in c.coordinator._accepted_stages
    assert c.ledger.read_attempt(attempt.id, owner_id=c.owner_id).state == "accepted"
    assert len(links(c)) == 1
    assert c.ledger.snapshot(c.chain_id).used["generation"] == 1


def test_manual_publication_wins_before_old_acceptance(ledger_case, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    from tldw_chatbook.DB.base_db import operation_owned_connection

    c = ledger_case
    attempt = c.claim()
    message_id, resume = _manual_resume(c, lambda target: None)
    publishing, release = threading.Event(), threading.Event()
    issue = c.coordinator.issuer.issue

    def pause_publication(*args, **kwargs):
        grant = issue(*args, **kwargs)
        publishing.set()
        assert release.wait(5)
        return grant

    monkeypatch.setattr(c.coordinator.issuer, "issue", pause_publication)

    def accept():
        with operation_owned_connection(c.db):
            return c.ledger.accept_wake(attempt.id, owner_id=c.owner_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        resumed = pool.submit(resume)
        try:
            assert publishing.wait(5)
            accepted = pool.submit(accept)
            with pytest.raises(TimeoutError):
                accepted.result(0.15)
        finally:
            release.set()
        assert resumed.result(5).slots == ("R3",)
        with pytest.raises(AutomaticWorkRefused, match="review_boundary_unavailable"):
            accepted.result(5)
    assert message_id in c.coordinator._accepted_stages
    assert c.ledger.read_attempt(attempt.id, owner_id=c.owner_id).state == "prepared"
    assert c.ledger.snapshot(c.chain_id).used["generation"] == 0
    assert c.ledger.snapshot(c.chain_id).reserved["generation"] == 1
    assert len(links(c)) == 1
    assert c.ledger.abort_wake(attempt.id, owner_id=c.owner_id)
    assert not links(c)


def test_claim_validation_never_waits_on_native_acceptance_scope(ledger_case):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    from tldw_chatbook.DB.base_db import operation_owned_connection

    c = ledger_case

    def claim():
        with operation_owned_connection(c.db):
            return c.claim()

    def accept(attempt):
        with operation_owned_connection(c.db):
            return c.ledger.accept_wake(attempt.id, owner_id=c.owner_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        with c.coordinator.boundary_acceptance_scope(c.live_boundary):
            # A real claim takes SQLite and validates the native receipt while
            # another owner holds admission. It must finish, not wait on scope.
            attempt = pool.submit(claim).result(5)
            accepted = pool.submit(accept, attempt)
            with pytest.raises(TimeoutError):
                accepted.result(0.15)
        assert accepted.result(5)
    assert c.ledger.snapshot(c.chain_id).used["generation"] == 1
    assert len(links(c)) == 1


def test_manual_resume_rechecks_selected_claim_after_confirmation(ledger_case):
    from tldw_chatbook.Agents.code_review_models import ReviewRefused
    from tldw_chatbook.Agents.run_context import use_run_actor

    c = ledger_case
    original = c.primary.for_run(c.primary.actor, capacity=lambda: c.capacity)
    claims = []

    def destination_check(target):
        with use_run_actor(original.actor):
            claims.append(original.claim_review_slot(c.prepared.review_id, "R3"))
        return None

    message_id, resume = _manual_resume(c, destination_check)
    with pytest.raises(ReviewRefused, match="review_attempt_state_changed"):
        resume()
    assert message_id not in c.coordinator._accepted_stages
    with c.store.lease(c.prepared.review_id) as context:
        assert context.launch_claims["R3"] is claims[0]
        assert not context.scheduled_attempts


@pytest.mark.parametrize(
    "change", ["copy", "source", "revision", "fence", "grant", "retired"]
)
def test_accepted_boundary_service_rechecks_retained_authority(ledger_case, change):
    from tldw_chatbook.Agents.code_review_models import ReviewRefused

    c = ledger_case
    c.claim()
    assert c.ledger.accept_wake("attempt-1", owner_id=c.owner_id)
    receipt = c.live_boundary
    with c.store.lease(receipt.review_id) as context:
        if change == "source":
            context.source_epoch += 1
        if change == "revision":
            context.revision += 1
        if change == "retired":
            context.state = "released"
    if change == "copy":
        receipt = replace(receipt)
    if change == "fence":
        c.coordinator.fence_session("chat")
    if change == "grant":
        c.coordinator.issuer.close(c.coordinator)
    with pytest.raises(ReviewRefused):
        c.coordinator.service_for_boundary(
            receipt, capacity=lambda: c.capacity, destination_check=lambda target: None
        )


@pytest.mark.parametrize(
    "tokens,verified,children,enabled",
    [
        (32000, False, 2, True),
        (4096, True, 2, True),
        (32000, True, 0, True),
        (32000, True, 2, False),
        (32000, True, 2, True),
    ],
)
def test_accepted_boundary_requires_actual_stage_schema_reachability(
    ledger_case, tokens, verified, children, enabled
):
    from types import SimpleNamespace

    from tldw_chatbook.Agents.agent_models import RunBudget
    from tldw_chatbook.Agents.code_review_models import ReviewRefused
    from tldw_chatbook.Agents.code_review_tool_provider import CodeReviewToolProvider
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.Chat.console_agent_bridge import build_console_first_request_plan
    from tldw_chatbook.Utils.token_counter import ContextWindowResolution

    c = ledger_case
    c.claim()
    assert c.ledger.accept_wake("attempt-1", owner_id=c.owner_id)
    service = c.coordinator.service_for_boundary(
        c.live_boundary,
        capacity=lambda: c.capacity,
        destination_check=lambda target: None,
    )
    service.app_config["tools"]["code_review_enabled"] = enabled

    def plan():
        return build_console_first_request_plan(
            shared_registry=ToolCatalogRegistry(),
            shared_allowed_tools=(),
            context={},
            skills_present=False,
            mcp_provider=None,
            builtin_gate=None,
            local_provider=None,
            library_provider=None,
            library_authority=None,
            workspace_id=None,
            ephemeral=False,
            diff_sink=None,
            scratch_root=None,
            scratch_lease=None,
            resolution=SimpleNamespace(
                model="synthetic",
                execution_key="custom-hosted",
                provider="custom-hosted",
                max_tokens=1024,
                context_window=ContextWindowResolution(tokens, "fixture", verified),
            ),
            fallback_model="synthetic",
            session_system_prompt="BASE",
            native_tools=False,
            turn_skill_bindings=(),
            turn_bundle_block="",
            install_skill_enabled=False,
            run_skill_script_enabled=False,
            agent_messages=[
                {"role": "user", "content": "Review these changes. " * 1200},
                {"role": "system", "content": "Accepted review continuation"},
            ],
            code_review_provider=CodeReviewToolProvider(service),
            run_budget=RunBudget(max_subagents=children),
        )

    if verified and tokens == 32000 and children and enabled:
        result = plan()
        assert "spawn_subagent" in {s.name for s in result.schemas.runtime_schemas}
    else:
        with pytest.raises(ReviewRefused, match="review_stage_tools_unreachable"):
            plan()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    ["none", "stop", "priority", "source", "grant", "owner", "unknown", "timeout"],
)
async def test_accepted_review_capacity_rechecks_after_held_metadata(
    ledger_case, change
):
    import asyncio
    import threading
    from types import SimpleNamespace

    import httpx

    from Tests.Chat.test_console_review_wake_transport import MODEL, profile, selection
    from tldw_chatbook.Agents.automatic_work_runtime import AutomaticWorkContext
    from tldw_chatbook.Agents.code_review_models import ReviewRefused
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_fleet_wake import (
        _WAKE_AUTHORIZATION_KEY,
        AgentWakeAuthorization,
        ConsoleFleetWakeCoordinator,
        _WakeDelivery,
    )
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway

    c = ledger_case
    c.claim()
    assert c.ledger.accept_wake("attempt-1", owner_id=c.owner_id)
    context = AutomaticWorkContext(c.ledger, c.chain_id, c.owner_id, "attempt-1")
    context.mark_accepted()
    entered, release = asyncio.Event(), asyncio.Event()
    requests = []

    async def metadata(request):
        requests.append(str(request.url))
        entered.set()
        await release.wait()
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": MODEL,
                        **({} if change == "unknown" else {"context_length": 32000}),
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(metadata)) as client:
        gateway = ConsoleProviderGateway(
            config_provider=profile, environ={}, http_client=client
        )
        controller = SimpleNamespace(
            store=c.chat_store,
            code_review=c.coordinator,
            _code_review=c.coordinator,
            provider_gateway=gateway,
            _disposed=False,
            _shutdown_requested=threading.Event(),
            _session_close_generations={},
            max_parallel_runs=3,
            PROVIDER_VALIDATION_TIMEOUT_SECONDS=0.01 if change == "timeout" else 1,
        )
        fleet = controller._fleet_wake = ConsoleFleetWakeCoordinator(controller)
        fleet._owner_id = c.owner_id
        authorization = AgentWakeAuthorization(
            fleet,
            "chat",
            _key=_WAKE_AUTHORIZATION_KEY,
            conversation_id="chat",
            work_chain_id=c.chain_id,
            attempt_id="attempt-1",
            owner_id=c.owner_id,
            context=context,
            review_boundary=c.live_boundary,
        )
        authorization.accepted = True
        fleet._active["chat"] = _WakeDelivery("chat", authorization=authorization)
        try:
            resolution = await gateway.resolve_for_send(selection())
            await entered.wait()
            before = c.ledger.snapshot(c.chain_id).used["generation"]
            task = asyncio.create_task(
                ConsoleChatController._resolve_accepted_review_capacity(
                    controller,
                    c.chat_store._session_or_raise("chat"),
                    authorization,
                    resolution,
                    SimpleNamespace(
                        resolved_destination=resolution.resolved_destination
                    ),
                )
            )
            # Let both real authority checks finish and the cache follower wait.
            for _ in range(30):
                await asyncio.sleep(0.001)
                if change == "timeout" or not task.done():
                    continue
            if change == "stop":
                c.coordinator.fence_session("chat")
            elif change == "priority":
                controller.wake_user_priority_probe = lambda _: True
            elif change == "source":
                with c.store.lease(c.live_boundary.review_id) as review:
                    review.source_epoch += 1
            elif change == "grant":
                c.coordinator.issuer.close(c.coordinator)
            elif change == "owner":
                fleet._active.clear()
            release.set()
            if change == "none":
                result = await task
                assert (
                    result.context_window.verified
                    and result.context_window.tokens == 32000
                )
                assert result.resolved_destination is resolution.resolved_destination
                assert result.base_url == resolution.base_url
            else:
                with pytest.raises((ReviewRefused, TimeoutError)):
                    await task
            assert requests == ["http://127.0.0.1:8010/v1/models"]
            assert (
                c.ledger.read_attempt("attempt-1", owner_id=c.owner_id).state
                == "accepted"
            )
            assert c.ledger.snapshot(c.chain_id).used["generation"] == before
        finally:
            release.set()
            await gateway.aclose()
