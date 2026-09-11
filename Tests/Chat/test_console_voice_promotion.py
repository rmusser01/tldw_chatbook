import asyncio
import json
from dataclasses import FrozenInstanceError, fields, replace
from types import SimpleNamespace
from uuid import uuid4

import pytest

from tldw_chatbook.Chat.console_voice_promotion import (
    CompletedVoicePairCommit,
    ConsoleVoicePromotionClaim,
    ConsoleVoicePromotionClaimStatus,
    ConsoleVoicePromotionLease,
    ConsoleVoicePromotionRecovery,
    ConsoleSessionBindingOrigin,
    ResolvedVoicePromotionDestination,
    VoicePromotionContext,
    VoicePromotionClaimStatus,
    VoicePromotionIdentitySet,
    VoicePromotionOutcomeStatus,
    VoicePromotionOwner,
    VoiceWinningPromotion,
    derive_voice_promotion_identities,
    new_voice_promotion_id,
)
from tldw_chatbook.Chat.console_voice_attempts import (
    VoiceAttemptSnapshot,
    VoiceAttemptToolRequest,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_exchange_capture import (
    freeze_provisional_capture_eligibility,
)
from tldw_chatbook.Chat.console_trace_models import (
    FrozenTracePolicy,
    TraceCallState,
    new_opaque_id,
)
from tldw_chatbook.Chat.console_voice_trace_gateway import (
    ProvisionalTraceRegistry,
    ProvisionalTraceUnavailable,
    VoiceTraceImportContext,
)
from tldw_chatbook.Chat.console_voice_trace_promotion import (
    PostDispatchTraceCall,
    PostDispatchTraceResponse,
    PostDispatchTraceSurfaceComponent,
    derive_post_dispatch_trace_ids,
    derive_post_dispatch_trace_node_id,
)
from tldw_chatbook.Chat.provider_usage import ProviderUsage


_CANONICAL_USAGE_JSON = ProviderUsage(
    uncached_input=7,
    output=11,
    provider="openai",
    model="gpt-voice-test",
).to_json()


def _origin() -> ConsoleSessionBindingOrigin:
    return ConsoleSessionBindingOrigin(
        session_id="session-a",
        session_incarnation=1,
        persisted_conversation_id="conversation-a",
        conversation_binding_revision=0,
    )


def _context(**changes: object) -> VoicePromotionContext:
    values: dict[str, object] = {
        "promotion_id": "promotion-a",
        "attempt_id": "attempt-a",
        "origin": _origin(),
        "expected_native_leaf_id": "native-leaf-a",
        "expected_persisted_leaf_id": "persisted-leaf-a",
        "user_text": "private user transcript",
        "assistant_text": "private assistant reply",
        "usage_json": _CANONICAL_USAGE_JSON,
        "terminal_boundary_id": "terminal-boundary-a",
        "capture_eligible_at_dispatch": True,
    }
    values.update(changes)
    return VoicePromotionContext(**values)  # type: ignore[arg-type]


def _identity_values(identities: VoicePromotionIdentitySet) -> tuple[str, ...]:
    return tuple(getattr(identities, item.name) for item in fields(identities))


@pytest.mark.parametrize(
    "value",
    (
        _origin(),
        _context(),
        ResolvedVoicePromotionDestination(
            session_id="session-a",
            session_incarnation=1,
            persisted_conversation_id="conversation-a",
            expected_persisted_leaf_id="persisted-leaf-a",
            capture_eligible_at_dispatch=True,
        ),
        CompletedVoicePairCommit(
            conversation_id="conversation-a",
            user_message_id="user-a",
            assistant_message_id="assistant-a",
            terminal_receipt_id="receipt-a",
            active_leaf_message_id="assistant-a",
        ),
    ),
)
def test_promotion_contracts_are_frozen_and_slotted(value: object) -> None:
    assert not hasattr(value, "__dict__")
    with pytest.raises(FrozenInstanceError):
        setattr(value, fields(value)[0].name, "changed")


def test_context_repr_does_not_disclose_transcript_or_provider_payload() -> None:
    context = _context()

    diagnostic = repr(context)

    assert context.user_text not in diagnostic
    assert context.assistant_text not in diagnostic
    assert context.usage_json not in diagnostic


@pytest.mark.parametrize(
    "raw_usage",
    (
        pytest.param("{", id="malformed-json"),
        pytest.param(
            '{"input_tokens": 7, "output_tokens": 11}',
            id="raw-provider-keys",
        ),
        pytest.param(
            json.dumps(
                {**json.loads(_CANONICAL_USAGE_JSON), "private_provider_field": 1},
                sort_keys=True,
            ),
            id="unknown-field",
        ),
        pytest.param(
            json.dumps(
                {**json.loads(_CANONICAL_USAGE_JSON), "uncached_input": -7},
                sort_keys=True,
            ),
            id="clamped-value",
        ),
        pytest.param(
            json.dumps(
                json.loads(_CANONICAL_USAGE_JSON),
                sort_keys=True,
                separators=(",", ":"),
            ),
            id="noncanonical-serialization",
        ),
    ),
)
def test_context_rejects_noncanonical_provider_usage(raw_usage: str) -> None:
    with pytest.raises(ValueError, match="usage_json"):
        _context(usage_json=raw_usage)


def test_promotion_identity_derivation_is_stable_and_domain_separated() -> None:
    first = derive_voice_promotion_identities("promotion-a")
    retry = derive_voice_promotion_identities("promotion-a")

    assert first == retry
    assert len(set(_identity_values(first))) == len(fields(VoicePromotionIdentitySet))


def test_different_promotion_ids_never_reuse_identities_for_identical_text() -> None:
    first_context = _context(promotion_id="promotion-a")
    second_context = _context(promotion_id="promotion-b")

    first = derive_voice_promotion_identities(first_context.promotion_id)
    second = derive_voice_promotion_identities(second_context.promotion_id)

    first_values = _identity_values(first)
    second_values = _identity_values(second)
    assert set(first_values).isdisjoint(second_values)
    for protected in (
        first_context.user_text,
        first_context.assistant_text,
        first_context.usage_json,
    ):
        assert protected is not None
        assert all(
            protected not in identity for identity in (*first_values, *second_values)
        )


def test_new_promotion_ids_are_opaque_and_unique() -> None:
    first = new_voice_promotion_id()
    second = new_voice_promotion_id()

    assert first != second
    assert derive_voice_promotion_identities(
        first
    ) != derive_voice_promotion_identities(second)


def test_claim_outcome_and_lease_are_opaque_frozen_values() -> None:
    destination = ResolvedVoicePromotionDestination(
        session_id="session-a",
        session_incarnation=1,
        persisted_conversation_id="conversation-a",
        expected_persisted_leaf_id="persisted-leaf-a",
        capture_eligible_at_dispatch=True,
    )
    lease = ConsoleVoicePromotionLease(
        lease_id="lease-a",
        session_id="session-a",
        session_incarnation=1,
        lease_revision=1,
        promotion_id="promotion-a",
        expected_native_leaf_id="native-leaf-a",
        destination=destination,
    )
    claim = ConsoleVoicePromotionClaim.claimed(lease)

    assert claim.status is ConsoleVoicePromotionClaimStatus.CLAIMED
    assert claim.lease is lease
    assert not hasattr(lease, "__dict__")
    assert "private user transcript" not in repr(lease)
    with pytest.raises(FrozenInstanceError):
        lease.lease_id = "changed"

    recovery = ConsoleVoicePromotionRecovery(lease=lease, context=_context())
    assert not hasattr(recovery, "__dict__")
    assert "private user transcript" not in repr(recovery)
    with pytest.raises(FrozenInstanceError):
        recovery.lease = lease


@pytest.mark.parametrize(
    "status",
    (
        ConsoleVoicePromotionClaimStatus.TRANSIENT_CONTENTION,
        ConsoleVoicePromotionClaimStatus.CONFLICT,
    ),
)
def test_non_claim_claim_outcomes_carry_no_capability(
    status: ConsoleVoicePromotionClaimStatus,
) -> None:
    claim = ConsoleVoicePromotionClaim(status=status)

    assert claim.lease is None


@pytest.mark.parametrize(
    ("factory", "match"),
    (
        (
            lambda: ConsoleSessionBindingOrigin(
                session_id="",
                session_incarnation=1,
                persisted_conversation_id=None,
                conversation_binding_revision=0,
            ),
            "session_id",
        ),
        (
            lambda: ConsoleSessionBindingOrigin(
                session_id="session-a",
                session_incarnation=True,
                persisted_conversation_id=None,
                conversation_binding_revision=0,
            ),
            "session_incarnation",
        ),
        (
            lambda: _context(origin=object()),
            "origin",
        ),
        (
            lambda: _context(user_text=""),
            "user_text",
        ),
        (
            lambda: _context(capture_eligible_at_dispatch=1),
            "capture_eligible_at_dispatch",
        ),
        (
            lambda: ResolvedVoicePromotionDestination(
                session_id="session-a",
                session_incarnation=1,
                persisted_conversation_id="",
                expected_persisted_leaf_id=None,
                capture_eligible_at_dispatch=False,
            ),
            "persisted_conversation_id",
        ),
        (
            lambda: CompletedVoicePairCommit(
                conversation_id="conversation-a",
                user_message_id="user-a",
                assistant_message_id="assistant-a",
                terminal_receipt_id="receipt-a",
                active_leaf_message_id="assistant-a",
                already_committed=1,
            ),
            "already_committed",
        ),
        (
            lambda: derive_voice_promotion_identities("x" * 513),
            "promotion_id",
        ),
    ),
)
def test_promotion_contracts_fail_closed_at_their_bounded_boundary(
    factory, match: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        factory()


def _post_dispatch_call(promotion_id: str) -> PostDispatchTraceCall:
    return PostDispatchTraceCall(
        call_id=derive_post_dispatch_trace_ids(
            promotion_id,
            call_count=1,
        ).call_ids[0],
        idempotency_key="voice-call-0",
        call_sequence=0,
        provider_name="test-provider",
        model_name="test-model",
        route_identity="chat_completions",
        endpoint_identity="https://provider.invalid/v1",
        generation_parameters_json="{}",
        adapter_defaults_json="{}",
        response_format_json="{}",
        reasoning_controls_json="{}",
        dispatch_started_at="2026-08-31T12:00:00Z",
        response_started_at="2026-08-31T12:00:01Z",
        settled_at="2026-08-31T12:00:02Z",
        request_surface=(
            PostDispatchTraceSurfaceComponent.revision(
                node_id=derive_post_dispatch_trace_node_id(promotion_id, 0, 0),
                component_kind="provider_message",
                revision_id=new_opaque_id(),
            ),
        ),
        response=PostDispatchTraceResponse.committed_revision(new_opaque_id()),
        sealed_payload_bytes=8,
        terminal_state=TraceCallState.COMPLETE,
    )


def _winning_case():
    promotion_id = str(uuid4())
    attempt_id = str(uuid4())
    store = ConsoleChatStore()
    session = store.create_session()
    conversation_id = str(uuid4())
    session.persisted_conversation_id = conversation_id
    context = VoicePromotionContext(
        promotion_id=promotion_id,
        attempt_id=attempt_id,
        origin=ConsoleSessionBindingOrigin(
            session_id=session.id,
            session_incarnation=store._settings_session_incarnations[session.id],
            persisted_conversation_id=conversation_id,
            conversation_binding_revision=session.conversation_binding_revision,
        ),
        expected_native_leaf_id=store.active_leaf(session.id),
        expected_persisted_leaf_id=None,
        user_text="protected user text",
        assistant_text="protected assistant text",
        usage_json=None,
        terminal_boundary_id=str(uuid4()),
        capture_eligible_at_dispatch=True,
    )
    identities = derive_voice_promotion_identities(promotion_id)
    commit = CompletedVoicePairCommit(
        conversation_id=conversation_id,
        user_message_id=identities.user_message_id,
        assistant_message_id=identities.assistant_message_id,
        terminal_receipt_id=identities.terminal_receipt_id,
        active_leaf_message_id=identities.assistant_message_id,
        user_revision_id=str(uuid4()),
        assistant_revision_id=str(uuid4()),
    )
    events: list[str] = []

    def commit_completed_voice_pair(*, destination, context):
        del destination, context
        events.append("pair")
        return commit

    store.persistence = SimpleNamespace(
        commit_completed_voice_pair=commit_completed_voice_pair
    )
    registry = ProvisionalTraceRegistry()
    trace_attempt = registry.begin_attempt(
        promotion_id=promotion_id,
        attempt_id=attempt_id,
        policy=FrozenTracePolicy(promotion_id, "credentials-v1", False, None),
        eligibility=freeze_provisional_capture_eligibility(
            capture_enabled=True,
            session_is_saved=True,
        ),
    )
    assert trace_attempt is not None
    envelope = registry._retain_gateway_call(
        trace_attempt,
        _post_dispatch_call(promotion_id),
    )
    assert envelope is not None
    manifest = registry.seal_attempt(trace_attempt, expected_call_count=1)
    snapshot = VoiceAttemptSnapshot(
        attempt_epoch=1,
        response_text=context.assistant_text,
        trace_manifest=manifest,
        trace_envelopes=(envelope,),
    )
    return store, session.id, context, commit, registry, snapshot, events


async def _inline(call):
    return call()


@pytest.mark.asyncio
async def test_exact_claim_callback_precedes_settlement_and_survives_observer_death():
    store, _, context, commit, registry, snapshot, events = _winning_case()
    release = asyncio.Event()

    async def delayed(call):
        await release.wait()
        return call()

    owner = VoicePromotionOwner(lambda: store, sync_runner=delayed)
    adapter = VoiceWinningPromotion(owner, trace_registry=registry)
    claims = []
    observer = adapter.promote(context, snapshot, on_claim=claims.append)
    assert len(claims) == 1
    assert claims[0].status is VoicePromotionClaimStatus.CLAIMED
    assert events == []
    observer.cancel()
    release.set()
    assert await owner.wait_for_session(context.origin.session_id, 1)
    assert events == ["pair"]


@pytest.mark.asyncio
async def test_refused_winner_never_reports_a_claim():
    store, _, context, _, registry, snapshot, _ = _winning_case()
    claims = []
    adapter = VoiceWinningPromotion(
        VoicePromotionOwner(lambda: store), trace_registry=registry
    )
    outcome = await adapter.promote(
        context, replace(snapshot, response_text="different"), on_claim=claims.append
    )
    assert claims == []
    assert outcome.status is VoicePromotionOutcomeStatus.RECOVERY


@pytest.mark.asyncio
@pytest.mark.parametrize("callback_raises", [False, True])
async def test_claim_callback_covers_failed_settlement_task_factory(
    monkeypatch, callback_raises
):
    store, _, context, _, registry, snapshot, events = _winning_case()
    release = asyncio.Event()

    async def delayed(call):
        await release.wait()
        return call()

    owner = VoicePromotionOwner(lambda: store, sync_runner=delayed)
    adapter = VoiceWinningPromotion(owner, trace_registry=registry)
    loop = asyncio.get_running_loop()
    create = loop.create_task

    def fail_settlement(coro, **kwargs):
        if kwargs.get("name", "").startswith("voice-winning-settlement-"):
            raise RuntimeError("private task factory failure")
        return create(coro, **kwargs)

    monkeypatch.setattr(loop, "create_task", fail_settlement)
    claims = []

    def claimed(claim):
        claims.append(claim)
        if callback_raises:
            raise ValueError("private observer failure")

    observer = None
    try:
        if callback_raises:
            with pytest.raises(ValueError):
                adapter.promote(context, snapshot, on_claim=claimed)
        else:
            observer = asyncio.ensure_future(
                adapter.promote(context, snapshot, on_claim=claimed)
            )
        assert len(claims) == 1
        assert claims[0].status is VoicePromotionClaimStatus.CLAIMED
        assert events == []
        if observer is not None:
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
    finally:
        release.set()
    assert await owner.wait_for_session(context.origin.session_id, 1)
    assert events == ["pair"]


@pytest.mark.asyncio
async def test_winning_promotion_commits_pair_before_best_effort_trace_import() -> None:
    store, session_id, context, commit, registry, snapshot, events = _winning_case()
    owner = VoicePromotionOwner(lambda: store, sync_runner=_inline)

    def trace_context_factory(_context, completed):
        assert events == ["pair"]
        assert completed is commit
        assert completed.user_revision_id is not None
        assert completed.assistant_revision_id is not None
        return VoiceTraceImportContext(
            import_id=context.promotion_id,
            conversation_id=commit.conversation_id,
            user_message_id=commit.user_message_id,
            user_revision_id=completed.user_revision_id,
            assistant_message_id=commit.assistant_message_id,
            assistant_revision_id=completed.assistant_revision_id,
            turn_id=str(uuid4()),
            run_id="voice-run",
            policy=FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None),
        )

    def import_trace(manifest, envelopes, trace_context):
        del trace_context
        events.append("trace")
        claim = registry.claim(manifest, envelopes)
        registry._consume_claim(claim)

    adapter = VoiceWinningPromotion(
        owner,
        trace_registry=registry,
        trace_context_factory=trace_context_factory,
        trace_importer=import_trace,
        sync_runner=_inline,
    )

    outcome = await adapter.promote(context, snapshot)

    assert outcome.status is VoicePromotionOutcomeStatus.PROMOTED
    assert events == ["pair", "trace"]
    assert store.active_leaf(session_id) == commit.assistant_message_id
    assert registry.retained_bytes == 0


@pytest.mark.asyncio
async def test_cancelled_view_waiter_cannot_orphan_claimed_trace_settlement() -> None:
    store, session_id, context, commit, registry, snapshot, events = _winning_case()
    pair_started = asyncio.Event()
    release_pair = asyncio.Event()

    async def blocked_pair_runner(call):
        pair_started.set()
        await release_pair.wait()
        return call()

    owner = VoicePromotionOwner(lambda: store, sync_runner=blocked_pair_runner)

    def trace_context_factory(_context, completed):
        assert completed.user_revision_id is not None
        assert completed.assistant_revision_id is not None
        return VoiceTraceImportContext(
            import_id=context.promotion_id,
            conversation_id=commit.conversation_id,
            user_message_id=commit.user_message_id,
            user_revision_id=completed.user_revision_id,
            assistant_message_id=commit.assistant_message_id,
            assistant_revision_id=completed.assistant_revision_id,
            turn_id=str(uuid4()),
            run_id="voice-run",
            policy=FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None),
        )

    def import_trace(manifest, envelopes, _trace_context):
        events.append("trace")
        trace_claim = registry.claim(manifest, envelopes)
        registry._consume_claim(trace_claim)

    adapter = VoiceWinningPromotion(
        owner,
        trace_registry=registry,
        trace_context_factory=trace_context_factory,
        trace_importer=import_trace,
        sync_runner=_inline,
    )
    caller = asyncio.ensure_future(adapter.promote(context, snapshot))
    await pair_started.wait()

    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    release_pair.set()
    assert await owner.wait_for_session(session_id, timeout=1) is True
    for _ in range(10):
        if registry.retained_bytes == 0:
            break
        await asyncio.sleep(0)

    assert events == ["pair", "trace"]
    assert store.active_leaf(session_id) == commit.assistant_message_id
    assert registry.retained_bytes == 0


@pytest.mark.asyncio
async def test_trace_import_failure_never_rolls_back_committed_pair() -> None:
    store, session_id, context, commit, registry, snapshot, events = _winning_case()
    owner = VoicePromotionOwner(lambda: store, sync_runner=_inline)

    def trace_context_factory(_context, completed):
        assert completed.user_revision_id is not None
        assert completed.assistant_revision_id is not None
        return VoiceTraceImportContext(
            import_id=context.promotion_id,
            conversation_id=commit.conversation_id,
            user_message_id=commit.user_message_id,
            user_revision_id=completed.user_revision_id,
            assistant_message_id=commit.assistant_message_id,
            assistant_revision_id=completed.assistant_revision_id,
            turn_id=str(uuid4()),
            run_id="voice-run",
            policy=FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None),
        )

    def fail_trace(*_args):
        events.append("trace_failed")
        raise RuntimeError("trace unavailable")

    adapter = VoiceWinningPromotion(
        owner,
        trace_registry=registry,
        trace_context_factory=trace_context_factory,
        trace_importer=fail_trace,
        sync_runner=_inline,
    )

    outcome = await adapter.promote(context, snapshot)

    assert outcome.status is VoicePromotionOutcomeStatus.PROMOTED
    assert events == ["pair", "trace_failed"]
    assert store.active_leaf(session_id) == commit.assistant_message_id
    assert registry.retained_bytes == 0
    with pytest.raises(ProvisionalTraceUnavailable):
        registry.claim(snapshot.trace_manifest, snapshot.trace_envelopes)


@pytest.mark.asyncio
async def test_durable_commit_trace_import_survives_native_publication_recovery(
    monkeypatch,
) -> None:
    store, session_id, context, commit, registry, snapshot, events = _winning_case()

    def fail_native_publication(*_args):
        raise RuntimeError("native publication unavailable")

    monkeypatch.setattr(store, "publish_durable_voice_pair", fail_native_publication)
    owner = VoicePromotionOwner(lambda: store, sync_runner=_inline)

    def trace_context_factory(_context, completed):
        assert completed is commit
        assert completed.user_revision_id is not None
        assert completed.assistant_revision_id is not None
        return VoiceTraceImportContext(
            import_id=context.promotion_id,
            conversation_id=commit.conversation_id,
            user_message_id=commit.user_message_id,
            user_revision_id=completed.user_revision_id,
            assistant_message_id=commit.assistant_message_id,
            assistant_revision_id=completed.assistant_revision_id,
            turn_id=str(uuid4()),
            run_id="voice-run",
            policy=FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None),
        )

    def import_trace(manifest, envelopes, _trace_context):
        events.append("trace")
        trace_claim = registry.claim(manifest, envelopes)
        registry._consume_claim(trace_claim)

    outcome = await VoiceWinningPromotion(
        owner,
        trace_registry=registry,
        trace_context_factory=trace_context_factory,
        trace_importer=import_trace,
        sync_runner=_inline,
    ).promote(context, snapshot)

    assert outcome.status is VoicePromotionOutcomeStatus.RECOVERY
    assert outcome._commit is commit
    assert events == ["pair", "trace"]
    assert registry.retained_bytes == 0
    recovery = owner.recovery_for_session(session_id)
    assert recovery is not None
    assert owner.discard_recovery(recovery) is True


@pytest.mark.asyncio
async def test_tool_winner_is_rejected_and_destroys_trace_without_pair_commit() -> None:
    store, _session_id, context, _commit, registry, snapshot, events = _winning_case()
    owner = VoicePromotionOwner(lambda: store, sync_runner=_inline)
    tool_snapshot = replace(
        snapshot,
        tool_request=VoiceAttemptToolRequest(
            attempt_epoch=snapshot.attempt_epoch,
            tool_calls=({"id": "tool-call", "name": "lookup"},),
        ),
    )

    outcome = await VoiceWinningPromotion(
        owner,
        trace_registry=registry,
    ).promote(context, tool_snapshot)

    assert outcome.status is VoicePromotionOutcomeStatus.RECOVERY
    assert outcome.failure_code == "invalid_winning_snapshot"
    assert events == []
    assert registry.retained_bytes == 0


@pytest.mark.asyncio
async def test_incomplete_trace_is_discarded_but_valid_pair_still_commits() -> None:
    store, session_id, context, commit, registry, snapshot, events = _winning_case()
    owner = VoicePromotionOwner(lambda: store, sync_runner=_inline)
    incomplete_snapshot = replace(snapshot, trace_envelopes=())

    outcome = await VoiceWinningPromotion(
        owner,
        trace_registry=registry,
    ).promote(context, incomplete_snapshot)

    assert outcome.status is VoicePromotionOutcomeStatus.PROMOTED
    assert events == ["pair"]
    assert store.active_leaf(session_id) == commit.assistant_message_id
    assert registry.retained_bytes == 0


@pytest.mark.asyncio
async def test_close_fence_refuses_pair_and_destroys_winning_trace() -> None:
    store, session_id, context, _commit, registry, snapshot, events = _winning_case()
    owner = VoicePromotionOwner(lambda: store, sync_runner=_inline)
    close_token = owner.begin_session_close(session_id)
    try:
        outcome = await VoiceWinningPromotion(
            owner,
            trace_registry=registry,
        ).promote(context, snapshot)
    finally:
        owner.abort_session_close(close_token)

    assert outcome.status is VoicePromotionOutcomeStatus.RECOVERY
    assert outcome.failure_code == "claim_close_fenced"
    assert events == []
    assert registry.retained_bytes == 0
