"""Current native assessment authority and real SQLite acceptance boundaries."""

import sqlite3
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from Tests.ChaChaNotesDB.test_console_dispatch_checkpoint_repository import (
    _acceptance,
    _insert,
)
from Tests.Chat.response_rules_fixtures import inputs, revision, source
from Tests.Chat.response_rules_store_fixtures import (
    rule_store as rule_store,  # noqa: PLC0414 - pytest fixture registration
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_dispatch_repository import ConsoleDispatchRepository
from tldw_chatbook.Chat.console_prompt_queue_coordinator import (
    _PromptChain,
)
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnCustodyRequest
from tldw_chatbook.Chat.response_rules.corrections import (
    MachineFollowupReceipt,
    NativeCorrectionProposal,
)
from tldw_chatbook.Chat.response_rules.evaluator import (
    aggregate_checks,
    check_deterministic,
)

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]


def rig():
    from Tests.Chat.test_console_prompt_queue_coordinator import (
        SequencedGateway,
        _arm_controller,
    )

    # Queue ownership uses its native registry; configuration comes from a real controller.
    controller, store, session_id = _arm_controller(SequencedGateway())
    session = next(s for s in store.sessions() if s.id == session_id)
    queue = controller.prompt_queue_coordinator
    queue.registry.begin_chain(
        session.id,
        context_epoch=store.conversation_context_epoch(session.id),
        expected_revision=queue.registry.snapshot(session.id).revision,
    )
    request = ConsoleTurnCustodyRequest(
        turn_id="turn",
        session_id=session.id,
        draft="Explain result",
        configuration=controller.resolve_turn_configuration_snapshot(session.id),
    )
    queue._chains[session.id] = _PromptChain(
        request=request, continuation_started=time.monotonic()
    )
    origin = source(
        session_id=session.id,
        conversation_id=None,
        message_id="answer",
        parent_turn_id=request.turn_id,
    )
    rule = revision()
    assessment = aggregate_checks(
        origin,
        (check_deterministic(rule, inputs("Missing proof")),),
        inputs=inputs("Missing proof"),
    )
    queue.bind_native_assessment_lookup(
        lambda s, key: (
            assessment if s == origin and key == assessment.assessment_id else None
        ),
        rules=lambda s: (rule,) if s == origin else (),
    )
    return queue, session.id, origin, assessment


@pytest.mark.asyncio
async def test_unregistered_or_historical_assessment_cannot_admit_repair():
    queue, session, origin, assessment = rig()
    assert (
        await queue.admit_machine_followup(
            session, source=origin, native=NativeCorrectionProposal(origin, "missing")
        )
        is None
    )
    queue.bind_native_assessment_lookup(
        lambda *_: replace(assessment, state="stale"), rules=lambda _: (revision(),)
    )
    assert (
        await queue.admit_machine_followup(
            session,
            source=origin,
            native=NativeCorrectionProposal(origin, assessment.assessment_id),
        )
        is None
    )
    assert queue.registry.snapshot(session).total_count == 0


@pytest.mark.asyncio
async def test_native_only_needs_no_hook_event_and_duplicate_callback_admits_once():
    queue, session, origin, assessment = rig()
    proposal = NativeCorrectionProposal(origin, assessment.assessment_id)
    first = await queue.admit_machine_followup(session, source=origin, native=proposal)
    assert first is not None
    assert (
        await queue.admit_machine_followup(session, source=origin, native=proposal)
        is None
    )
    snapshot = queue.registry.snapshot(session)
    assert snapshot.total_count == 1 and not queue._stop_parents
    receipt = queue._chains[session].machine_receipt
    assert receipt.admitted_turns == receipt.native_turns == 1
    assert receipt.contributors == ("native",)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "block",
    [
        "foreground",
        "sealed",
        "recovery",
        "third_native",
        "shared_cap",
        "wall",
        "authority",
    ],
)
async def test_native_followup_respects_existing_owner_limits(block):
    queue, session, origin, assessment = rig()
    if block == "foreground":
        queue.registry.admit(
            session,
            text="Human next task",
            expected_revision=queue.registry.snapshot(session).revision,
        )
    if block == "sealed":
        queue._sealed_continuations.add(session)
    if block == "recovery":
        queue._dispatch_recoveries[session] = object()
    if block in {"third_native", "shared_cap"}:
        queue._chains[session].machine_receipt = MachineFollowupReceipt(
            "previous",
            "p",
            "s",
            "a",
            "chain",
            3 if block == "shared_cap" else 2,
            2,
            ("native",),
            1,
        )
    if block == "wall":
        queue._chains[session].continuation_started = time.monotonic() - 121
    if block == "authority":
        queue.bind_continuation_admission(lambda _: False)
    assert (
        await queue.admit_machine_followup(
            session,
            source=origin,
            native=NativeCorrectionProposal(origin, assessment.assessment_id),
        )
        is None
    )


@pytest.mark.asyncio
async def test_native_and_hook_feedback_share_one_atomic_acceptance():
    from tldw_chatbook.Agents.hooks_v2.models import HookResult

    queue, session, origin, assessment = rig()
    request = queue._chains[session].request
    event = SimpleNamespace(event_id="stop", turn_id=request.turn_id)
    outcome = SimpleNamespace(
        allowed=True,
        outstanding_cleanup=False,
        accepted=(
            (
                "handler",
                HookResult(continuation={"message": "Genuine hook contribution"}),
            ),
        ),
    )
    lifecycle = SimpleNamespace(
        live=True,
        current=lambda: True,
        engine=SimpleNamespace(effects_current=lambda *_: True),
        terminal_budgets={},
        terminal_budget_times={},
        inherited_budgets={},
    )
    key = (request.turn_id, event.event_id)
    queue._stop_parents[key] = (
        session,
        (lifecycle, "scope", event, origin.message_id, request),
    )
    queue._stop_outcomes[key] = outcome
    queue._chains[session].pending_stop_key = key
    turn = await queue.admit_machine_followup(
        session,
        source=origin,
        native=NativeCorrectionProposal(origin, assessment.assessment_id),
    )
    assert turn is not None
    receipt = queue._chains[session].machine_receipt
    assert (
        receipt.contributors == ("native", "hook")
        and receipt.native_turns == receipt.admitted_turns == 1
    )
    issued = next(iter(queue._machine_entries.values()))
    assert "Native response-rule" in str(
        issued[4]
    ) and "Genuine hook contribution" in str(issued[4])
    assert queue.registry.snapshot(session).total_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("veto", ["deny", "stop", "unsettled", "revoked", "overflow"])
async def test_controlling_hook_outcome_cannot_be_bypassed_by_native_feedback(veto):
    from tldw_chatbook.Agents.hooks_v2.models import HookResult

    queue, session, origin, assessment = rig()
    request = queue._chains[session].request
    key = (request.turn_id, "stop")
    result = HookResult(
        decision="deny" if veto == "deny" else "pass",
        stop_continuations=veto == "stop",
        continuation={"message": "x" * (8192 if veto == "overflow" else 2)},
    )
    event = SimpleNamespace(event_id="stop")
    lifecycle = SimpleNamespace(
        live=True,
        current=lambda: veto != "revoked",
        engine=SimpleNamespace(effects_current=lambda *_: True),
    )
    queue._stop_parents[key] = (
        session,
        (lifecycle, "scope", event, origin.message_id, request),
    )
    queue._stop_outcomes[key] = SimpleNamespace(
        allowed=True,
        outstanding_cleanup=veto == "unsettled",
        accepted=(("handler", result),),
    )
    queue._chains[session].pending_stop_key = key
    assert (
        await queue.admit_machine_followup(
            session,
            source=origin,
            native=NativeCorrectionProposal(origin, assessment.assessment_id),
        )
        is None
    )


def acceptance(origin):
    receipt = MachineFollowupReceipt(
        origin.operation_id,
        origin.parent_turn_id,
        origin.settlement_id,
        origin.message_id,
        "chain",
        1,
        1,
        ("native",),
        origin.message_version,
    )
    return replace(
        _acceptance(origin.conversation_id, suffix="repair"),
        origin="queued",
        queue_entry_id="entry",
        parent_message_id=origin.message_id,
        machine_followup_receipt=receipt,
    )


def test_failure_after_feedback_insert_rolls_back_all_acceptance_rows(rule_store):
    _store, db, origin = rule_store
    with db.transaction() as cursor:
        cursor.execute(
            "CREATE TEMP TRIGGER refuse_checkpoint BEFORE INSERT ON console_dispatch_checkpoints BEGIN SELECT RAISE(ABORT,'injected checkpoint failure'); END"
        )
    with pytest.raises(sqlite3.DatabaseError):
        _insert(db, ConsoleDispatchRepository(db), acceptance(origin))
    with db.transaction() as cursor:
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM messages WHERE id IN ('user-repair','assistant-repair')"
            ).fetchone()[0]
            == 0
        )
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM console_dispatch_checkpoints"
            ).fetchone()[0]
            == 0
        )
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM console_machine_followup_receipts"
            ).fetchone()[0]
            == 0
        )


def test_shared_receipt_is_unique_and_does_not_replace_old_hook_provenance(rule_store):
    _store, db, origin = rule_store
    repo = ConsoleDispatchRepository(db)
    item = acceptance(origin)
    _insert(db, repo, item)
    with db.transaction() as cursor:
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM console_machine_followup_receipts"
            ).fetchone()[0]
            == 1
        )
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM console_hook_continuation_receipts"
            ).fetchone()[0]
            == 0
        )
        cursor.execute("DELETE FROM console_dispatch_checkpoints")
    with pytest.raises((ValueError, sqlite3.DatabaseError)):
        _insert(
            db,
            repo,
            replace(
                item,
                user_message_id="duplicate-user",
                assistant_message_id="duplicate-answer",
            ),
        )
    with db.transaction() as cursor:
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM messages WHERE id='duplicate-user'"
            ).fetchone()[0]
            == 0
        )


def test_receipt_change_invalidates_staged_acceptance_fingerprint(rule_store):
    _rules, _db, origin = rule_store
    item = acceptance(origin)
    changed = replace(
        item,
        machine_followup_receipt=replace(
            item.machine_followup_receipt, settlement_id="changed"
        ),
    )
    owners = SimpleNamespace(
        user_message_id=item.user_message_id,
        assistant_message_id=item.assistant_message_id,
    )
    identity = SimpleNamespace(conversation_id=origin.conversation_id, title="Rules")
    preparation = SimpleNamespace(session_id=origin.session_id)

    def fingerprint(value):
        return ConsoleChatStore._durable_acceptance_fingerprint(
            value, preparation, identity, owners, None, {}, None
        )

    assert fingerprint(item) != fingerprint(changed)


def test_combined_receipt_requires_hook_provenance_and_commits_both_atomically(
    rule_store,
):
    from tldw_chatbook.Agents.hooks_v2.continuations import ContinuationReceipt

    _rules, db, origin = rule_store
    repo = ConsoleDispatchRepository(db)
    item = acceptance(origin)
    item = replace(
        item,
        machine_followup_receipt=replace(
            item.machine_followup_receipt, contributors=("native", "hook")
        ),
    )
    with pytest.raises(ValueError, match="hook"):
        _insert(db, repo, item)
    item = replace(
        item,
        continuation_receipt=ContinuationReceipt(
            origin.parent_turn_id, "real-stop-event", origin.message_id, "chain", 1
        ),
    )
    _insert(db, repo, item)
    with db.transaction() as cursor:
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM console_machine_followup_receipts"
            ).fetchone()[0]
            == 1
        )
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM console_hook_continuation_receipts"
            ).fetchone()[0]
            == 1
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("native_turns", 2),
        ("admitted_turns", True),
        ("contributors", ("forged",)),
        ("source_message_version", 9),
    ],
)
def test_caller_forged_receipt_counters_or_source_are_refused(rule_store, field, value):
    _rules, db, origin = rule_store
    item = acceptance(origin)
    with pytest.raises((ValueError, sqlite3.DatabaseError)):
        receipt = replace(item.machine_followup_receipt, **{field: value})
        _insert(
            db,
            ConsoleDispatchRepository(db),
            replace(item, machine_followup_receipt=receipt),
        )


@pytest.mark.asyncio
async def test_hook_after_native_repair_keeps_shared_chain_and_counters():
    from tldw_chatbook.Agents.hooks_v2.engine import HookEventOutcome
    from tldw_chatbook.Agents.hooks_v2.models import HookResult

    queue, session, origin, assessment = rig()
    admitted_turn = await queue.admit_machine_followup(
        session,
        source=origin,
        native=NativeCorrectionProposal(origin, assessment.assessment_id),
    )
    first = queue._chains[session].machine_receipt
    claim = queue.registry.claim_next(
        session, expected_revision=queue.registry.snapshot(session).revision
    ).claim
    queue.registry.settle_claim(
        session,
        entry_id=claim.prompt.entry_id,
        expected_revision=queue.registry.snapshot(session).revision,
    )
    queue._retire_machine_entry(claim.prompt.entry_id)
    request = replace(queue._chains[session].request, turn_id=admitted_turn)
    queue._chains[session].request = request
    event = SimpleNamespace(event_id="next-stop")
    lifecycle = SimpleNamespace(
        live=True,
        current=lambda: True,
        engine=SimpleNamespace(effects_current=lambda *_: True),
        terminal_budgets={},
        terminal_budget_times={},
        inherited_budgets={},
    )
    key = (request.turn_id, event.event_id)
    queue._stop_parents[key] = (
        session,
        (lifecycle, "scope", event, "next-answer", request),
    )
    queue._stop_outcomes[key] = HookEventOutcome(
        accepted=(
            (
                "handler",
                HookResult(continuation={"message": "Actual owned hook feedback"}),
            ),
        )
    )
    turn = await queue.schedule_continuation(
        request.turn_id,
        event.event_id,
        (HookResult(continuation={"message": "FORGED-CALLER-FEEDBACK"}),),
    )
    assert turn is not None
    receipt = queue._chains[session].machine_receipt
    assert (
        receipt.chain_id == first.chain_id
        and receipt.admitted_turns == 2
        and receipt.native_turns == 1
    )
    assert "FORGED-CALLER-FEEDBACK" not in str(
        next(iter(queue._machine_entries.values()))[4]
    )


@pytest.mark.asyncio
async def test_foreground_admission_invalidates_native_transaction_gate():
    from tldw_chatbook.Agents.hooks_v2.continuations import ContinuationAdmissionRefused

    queue, session, origin, assessment = rig()
    await queue.admit_machine_followup(
        session,
        source=origin,
        native=NativeCorrectionProposal(origin, assessment.assessment_id),
    )
    claim = queue.registry.claim_next(
        session, expected_revision=queue.registry.snapshot(session).revision
    ).claim
    queue._chains[session].current_entry_id = claim.prompt.entry_id
    gate = queue.continuation_contribution(session, claim.prompt.entry_id)
    assert queue.continuation_receipt(session, claim.prompt.entry_id) is None
    assert queue.machine_followup_receipt(
        session, claim.prompt.entry_id
    ).contributors == ("native",)
    queue.admit(
        session,
        text="New user request",
        expected_revision=queue.registry.snapshot(session).revision,
    )
    with pytest.raises(ContinuationAdmissionRefused):
        gate.consume()
