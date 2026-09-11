"""Gateway-owned capability tests for promoted speculative voice traces."""

from __future__ import annotations

from dataclasses import replace
import inspect
from threading import Event, Thread

import pytest

from tldw_chatbook.Chat.console_chat_controller import (
    _build_speculative_voice_capture_request,
)
from tldw_chatbook.Chat.console_exchange_capture import (
    freeze_provisional_capture_eligibility,
)
from tldw_chatbook.Chat.console_trace_models import (
    FrozenTracePolicy,
    TraceCallState,
    new_opaque_id,
)
from tldw_chatbook.Chat.console_trace_provenance import (
    ProviderArtifactTraceProvenance,
    SavedRevisionTraceProvenance,
    TraceProvenanceSource,
)
from tldw_chatbook.Chat.console_trace_service import (
    ConsoleTraceService,
    ProvisionalTraceImportExpiredError,
    ProvisionalTraceImportRetryableError,
)
from tldw_chatbook.Chat.console_voice_trace_gateway import (
    MAX_PROVISIONAL_TRACE_APP_BYTES,
    PROVISIONAL_TRACE_TTL_SECONDS,
    ProvisionalTraceEnvelope,
    ProvisionalTraceManifest,
    ProvisionalTraceRegistry,
    ProvisionalTraceUnavailable,
    VoiceTraceImportContext,
)
from tldw_chatbook.Chat.console_voice_trace_promotion import (
    ConfirmedPreCommitTraceImportError,
    PostDispatchTraceArtifact,
    PostDispatchTraceCall,
    PostDispatchTraceImport,
    PostDispatchTraceImportResult,
    PostDispatchTraceResponse,
    PostDispatchTraceSurfaceComponent,
    derive_post_dispatch_trace_ids,
    derive_post_dispatch_trace_node_id,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now


_FROZEN_POLICY = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)


def _policy() -> FrozenTracePolicy:
    return _FROZEN_POLICY


def test_speculative_voice_capture_request_has_complete_frozen_provenance() -> None:
    saved = SavedRevisionTraceProvenance(new_opaque_id())
    request = _build_speculative_voice_capture_request(
        messages=(
            {"role": "system", "content": "rendered system"},
            {
                "role": "user",
                "content": "saved history",
                "_native_message_id": "native-history",
            },
            {"role": "user", "content": "rolling transcript"},
        ),
        tools=({"type": "function", "function": {"name": "clock"}},),
        capture_policy=_policy(),
        saved_by_owner={"native-history": saved},
    )

    assert request.capture_durability == "durable"
    assert request.provenance is not None
    assert isinstance(request.provenance.system[0], ProviderArtifactTraceProvenance)
    assert request.provenance.system[0].source is TraceProvenanceSource.RENDERED_SYSTEM
    assert request.provenance.compactable[0].messages == (saved,)
    assert (
        request.provenance.active_request[0].source
        is TraceProvenanceSource.ACTIVE_REQUEST
    )
    assert request.provenance.tools[0].source is TraceProvenanceSource.TOOL_DEFINITION
    assert len(request.provenance.metadata) == 1


def _context(promotion_id: str) -> VoiceTraceImportContext:
    return VoiceTraceImportContext(
        import_id=promotion_id,
        conversation_id=new_opaque_id(),
        user_message_id=new_opaque_id(),
        user_revision_id=new_opaque_id(),
        assistant_message_id=new_opaque_id(),
        assistant_revision_id=new_opaque_id(),
        turn_id=new_opaque_id(),
        run_id="voice-run",
        policy=_policy(),
    )


def _call(
    promotion_id: str,
    sequence: int,
    *,
    artifact_bytes: bytes = b"",
    artifact_id: str | None = None,
    usage_json: str | None = None,
) -> PostDispatchTraceCall:
    artifact = (
        None
        if not artifact_bytes
        else PostDispatchTraceArtifact(
            artifact_id=artifact_id or new_opaque_id(),
            media_type="application/json",
            normalization_version="canonical-json-v1",
            sanitized_bytes=artifact_bytes,
        )
    )
    request_surface = (
        PostDispatchTraceSurfaceComponent.revision(
            node_id=derive_post_dispatch_trace_node_id(promotion_id, sequence, 0),
            component_kind="provider_message",
            revision_id=new_opaque_id(),
        ),
    )
    response = (
        PostDispatchTraceResponse.no_response("provider_error_no_response")
        if artifact is None
        else PostDispatchTraceResponse.artifact(artifact)
    )
    json_bytes = 8
    retained_bytes = (
        json_bytes
        + len(artifact_bytes)
        + (0 if usage_json is None else len(usage_json.encode("utf-8")))
    )
    return PostDispatchTraceCall(
        call_id=derive_post_dispatch_trace_ids(
            promotion_id, call_count=sequence + 1
        ).call_ids[sequence],
        idempotency_key=f"voice-call-{sequence}",
        call_sequence=sequence,
        provider_name="test-provider",
        model_name="test-model",
        route_identity="chat_completions",
        endpoint_identity="https://provider.invalid/v1",
        generation_parameters_json="{}",
        adapter_defaults_json="{}",
        response_format_json="{}",
        reasoning_controls_json="{}",
        dispatch_started_at=f"2026-08-31T12:00:0{sequence}Z",
        response_started_at=(
            None if artifact is None else f"2026-08-31T12:00:0{sequence + 1}Z"
        ),
        settled_at=f"2026-08-31T12:00:0{sequence + 2}Z",
        usage_json=usage_json,
        request_surface=request_surface,
        response=response,
        sealed_payload_bytes=retained_bytes,
        terminal_state=(
            TraceCallState.ERROR if artifact is None else TraceCallState.COMPLETE
        ),
    )


def _eligible():
    return freeze_provisional_capture_eligibility(
        capture_enabled=True,
        session_is_saved=True,
    )


def _sealed(
    registry: ProvisionalTraceRegistry,
    *,
    promotion_id: str | None = None,
    attempt_id: str | None = None,
    call_count: int = 2,
    artifact_bytes: bytes = b"x",
):
    promotion_id = promotion_id or new_opaque_id()
    attempt_id = attempt_id or new_opaque_id()
    attempt = registry.begin_attempt(
        policy=_policy(),
        promotion_id=promotion_id,
        attempt_id=attempt_id,
        eligibility=_eligible(),
    )
    assert attempt is not None
    envelopes = tuple(
        registry._retain_gateway_call(
            attempt,
            _call(promotion_id, sequence, artifact_bytes=artifact_bytes),
        )
        for sequence in range(call_count)
    )
    assert all(envelope is not None for envelope in envelopes)
    manifest = registry.seal_attempt(attempt, expected_call_count=call_count)
    return promotion_id, attempt, envelopes, manifest


def test_gateway_seals_ordered_multi_call_manifest_and_hides_payloads() -> None:
    registry = ProvisionalTraceRegistry()
    promotion_id, _attempt, envelopes, manifest = _sealed(registry, call_count=3)

    assert tuple(envelope.call_sequence for envelope in envelopes) == (0, 1, 2)
    assert manifest.promotion_id == promotion_id
    assert manifest.expected_call_count == 3
    assert manifest.envelope_ids == tuple(
        envelope.envelope_id for envelope in envelopes
    )
    assert len(manifest.observed_chronology) == 3
    assert manifest.aggregate_payload_bytes == 27
    rendered = repr((envelopes, manifest, registry))
    assert "provider.invalid" not in rendered
    assert "test-provider" not in rendered
    assert "sanitized_bytes" not in rendered


def test_opaque_capabilities_are_private_construction_and_forgery_fails() -> None:
    registry = ProvisionalTraceRegistry()
    _, _, envelopes, manifest = _sealed(registry)

    with pytest.raises(TypeError):
        ProvisionalTraceEnvelope()
    with pytest.raises(TypeError):
        ProvisionalTraceManifest()

    forged_envelope = object.__new__(ProvisionalTraceEnvelope)
    with pytest.raises(ProvisionalTraceUnavailable):
        registry.claim(manifest, (envelopes[0], forged_envelope))


def test_opaque_values_are_immutable_and_bypass_mutation_is_rejected() -> None:
    registry = ProvisionalTraceRegistry()
    _, _, envelopes, manifest = _sealed(registry)

    with pytest.raises(AttributeError):
        envelopes[0].call_sequence = 7
    object.__setattr__(envelopes[0], "call_sequence", 7)
    with pytest.raises(ProvisionalTraceUnavailable):
        registry.claim(manifest, envelopes)
    assert registry.retained_bytes == 0


@pytest.mark.parametrize("mode", ["missing", "substitution", "cross_attempt"])
def test_claim_rejects_incomplete_or_substituted_manifest_as_one_unit(
    mode: str,
) -> None:
    registry = ProvisionalTraceRegistry()
    _, _, envelopes, manifest = _sealed(registry)
    _, _, other_envelopes, _other_manifest = _sealed(registry)
    supplied = {
        "missing": envelopes[:1],
        "substitution": (envelopes[0], envelopes[0]),
        "cross_attempt": (envelopes[0], other_envelopes[1]),
    }[mode]

    with pytest.raises(ProvisionalTraceUnavailable):
        registry.claim(manifest, supplied)


def test_second_claim_rejects_without_destroying_legitimate_claim() -> None:
    registry = ProvisionalTraceRegistry()
    _, _, envelopes, manifest = _sealed(registry)
    expected_bytes = registry.retained_bytes
    first = registry.claim(manifest, envelopes)

    with pytest.raises(ProvisionalTraceUnavailable):
        registry.claim(manifest, envelopes)

    assert registry.retained_bytes == expected_bytes
    registry._release_claim(first)
    second = registry.claim(manifest, envelopes)
    registry._consume_claim(second)
    assert registry.retained_bytes == 0


def test_sequence_gap_and_partial_seal_destroy_the_attempt() -> None:
    registry = ProvisionalTraceRegistry()
    promotion_id = new_opaque_id()
    attempt = registry.begin_attempt(
        policy=_policy(),
        promotion_id=promotion_id,
        attempt_id=new_opaque_id(),
        eligibility=_eligible(),
    )
    assert attempt is not None

    with pytest.raises(ProvisionalTraceUnavailable):
        registry._retain_gateway_call(attempt, _call(promotion_id, 1))
    assert registry.retained_bytes == 0


def test_ninth_call_rejects_and_destroys_the_whole_attempt() -> None:
    registry = ProvisionalTraceRegistry()
    promotion_id = new_opaque_id()
    attempt = registry.begin_attempt(
        policy=_policy(),
        promotion_id=promotion_id,
        attempt_id=new_opaque_id(),
        eligibility=_eligible(),
    )
    assert attempt is not None
    for sequence in range(8):
        registry._retain_gateway_call(attempt, _call(promotion_id, sequence))

    ninth = replace(
        _call(promotion_id, 0),
        call_id=new_opaque_id(),
        idempotency_key="voice-call-8",
        call_sequence=8,
    )
    with pytest.raises(ProvisionalTraceUnavailable):
        registry._retain_gateway_call(attempt, ninth)
    assert registry.retained_bytes == 0

    attempt = registry.begin_attempt(
        policy=_policy(),
        promotion_id=new_opaque_id(),
        attempt_id=new_opaque_id(),
        eligibility=_eligible(),
    )
    assert attempt is not None
    with pytest.raises(ProvisionalTraceUnavailable):
        registry.seal_attempt(attempt, expected_call_count=2)
    assert registry.retained_bytes == 0


def test_per_attempt_and_app_wide_overflow_release_exact_budget() -> None:
    attempt_registry = ProvisionalTraceRegistry(
        attempt_byte_limit=10,
        app_byte_limit=20,
    )
    promotion_id = new_opaque_id()
    attempt = attempt_registry.begin_attempt(
        policy=_policy(),
        promotion_id=promotion_id,
        attempt_id=new_opaque_id(),
        eligibility=_eligible(),
    )
    assert attempt is not None
    with pytest.raises(ProvisionalTraceUnavailable):
        attempt_registry._retain_gateway_call(
            attempt,
            _call(promotion_id, 0, artifact_bytes=b"123"),
        )
    assert attempt_registry.retained_bytes == 0

    app_registry = ProvisionalTraceRegistry(
        attempt_byte_limit=16,
        app_byte_limit=16,
    )
    _sealed(app_registry, call_count=1, artifact_bytes=b"")
    assert app_registry.retained_bytes == 8
    second_id = new_opaque_id()
    second = app_registry.begin_attempt(
        policy=_policy(),
        promotion_id=second_id,
        attempt_id=new_opaque_id(),
        eligibility=_eligible(),
    )
    assert second is not None
    with pytest.raises(ProvisionalTraceUnavailable):
        app_registry._retain_gateway_call(
            second, _call(second_id, 0, artifact_bytes=b"x")
        )
    assert app_registry.retained_bytes == 8
    assert MAX_PROVISIONAL_TRACE_APP_BYTES == 128 * 1024 * 1024


def test_usage_and_shared_artifacts_have_exact_canonical_memory_accounting() -> None:
    artifact_id = new_opaque_id()
    first_bytes = bytes(bytearray(b"shared"))
    second_bytes = bytes(bytearray(b"shared"))
    assert first_bytes is not second_bytes
    usage_json = '{"input_tokens":1}'
    inline_bytes = 8 + len(usage_json.encode("utf-8"))
    per_attempt_bytes = inline_bytes * 2 + len(first_bytes)
    registry = ProvisionalTraceRegistry(
        attempt_byte_limit=per_attempt_bytes,
        app_byte_limit=per_attempt_bytes,
    )
    promotion_id = new_opaque_id()
    attempt = registry.begin_attempt(
        policy=_policy(),
        promotion_id=promotion_id,
        attempt_id=new_opaque_id(),
        eligibility=_eligible(),
    )
    assert attempt is not None

    registry._retain_gateway_call(
        attempt,
        _call(
            promotion_id,
            0,
            artifact_bytes=first_bytes,
            artifact_id=artifact_id,
            usage_json=usage_json,
        ),
    )
    registry._retain_gateway_call(
        attempt,
        _call(
            promotion_id,
            1,
            artifact_bytes=second_bytes,
            artifact_id=artifact_id,
            usage_json=usage_json,
        ),
    )
    manifest = registry.seal_attempt(attempt, expected_call_count=2)

    state = next(iter(registry._states.values()))
    first_artifact = state.calls[0].response.artifact_value
    second_artifact = state.calls[1].response.artifact_value
    assert first_artifact is second_artifact
    assert first_artifact.sanitized_bytes is second_artifact.sanitized_bytes
    assert manifest.aggregate_payload_bytes == per_attempt_bytes
    assert registry.retained_bytes == per_attempt_bytes

    too_small = ProvisionalTraceRegistry(
        attempt_byte_limit=inline_bytes + len(first_bytes) - 1,
        app_byte_limit=inline_bytes + len(first_bytes) - 1,
    )
    too_small_id = new_opaque_id()
    too_small_attempt = too_small.begin_attempt(
        policy=_policy(),
        promotion_id=too_small_id,
        attempt_id=new_opaque_id(),
        eligibility=_eligible(),
    )
    assert too_small_attempt is not None
    with pytest.raises(ProvisionalTraceUnavailable):
        too_small._retain_gateway_call(
            too_small_attempt,
            _call(
                too_small_id,
                0,
                artifact_bytes=first_bytes,
                artifact_id=artifact_id,
                usage_json=usage_json,
            ),
        )
    assert too_small.retained_bytes == 0


def test_app_budget_canonicalizes_shared_artifact_and_releases_by_reference() -> None:
    artifact_id = new_opaque_id()
    artifact_bytes = b"shared"
    one_attempt_bytes = 8 + len(artifact_bytes)
    app_bytes = 16 + len(artifact_bytes)
    registry = ProvisionalTraceRegistry(
        attempt_byte_limit=one_attempt_bytes,
        app_byte_limit=app_bytes,
    )
    retained = []
    for _ in range(2):
        promotion_id = new_opaque_id()
        attempt = registry.begin_attempt(
            policy=_policy(),
            promotion_id=promotion_id,
            attempt_id=new_opaque_id(),
            eligibility=_eligible(),
        )
        assert attempt is not None
        registry._retain_gateway_call(
            attempt,
            _call(
                promotion_id,
                0,
                artifact_bytes=bytes(bytearray(artifact_bytes)),
                artifact_id=artifact_id,
            ),
        )
        manifest = registry.seal_attempt(attempt, expected_call_count=1)
        assert manifest.aggregate_payload_bytes == one_attempt_bytes
        retained.append((attempt, manifest))

    assert registry.retained_bytes == app_bytes
    states = tuple(registry._states.values())
    assert states[0].artifacts[artifact_id] is states[1].artifacts[artifact_id]
    registry.abandon_manifest(retained[0][1])
    assert registry.retained_bytes == one_attempt_bytes
    registry.abandon_manifest(retained[1][1])
    assert registry.retained_bytes == 0


def test_shared_artifact_identity_with_different_content_rejects_new_attempt() -> None:
    artifact_id = new_opaque_id()
    registry = ProvisionalTraceRegistry(attempt_byte_limit=32, app_byte_limit=64)
    first_id = new_opaque_id()
    first_attempt = registry.begin_attempt(
        policy=_policy(),
        promotion_id=first_id,
        attempt_id=new_opaque_id(),
        eligibility=_eligible(),
    )
    assert first_attempt is not None
    registry._retain_gateway_call(
        first_attempt,
        _call(
            first_id,
            0,
            artifact_bytes=b"first",
            artifact_id=artifact_id,
        ),
    )
    registry.seal_attempt(first_attempt, expected_call_count=1)
    retained_bytes = registry.retained_bytes
    second_id = new_opaque_id()
    second_attempt = registry.begin_attempt(
        policy=_policy(),
        promotion_id=second_id,
        attempt_id=new_opaque_id(),
        eligibility=_eligible(),
    )
    assert second_attempt is not None
    with pytest.raises(ProvisionalTraceUnavailable):
        registry._retain_gateway_call(
            second_attempt,
            _call(
                second_id,
                0,
                artifact_bytes=b"other",
                artifact_id=artifact_id,
            ),
        )
    assert registry.retained_bytes == retained_bytes
    registry.abandon_attempt(first_attempt)
    assert registry.retained_bytes == 0


def test_expiry_destroys_payload_and_rejects_redemption() -> None:
    clock = _Clock()
    registry = ProvisionalTraceRegistry(clock=clock)
    _, _, envelopes, manifest = _sealed(registry)

    clock.now += PROVISIONAL_TRACE_TTL_SECONDS + 0.001
    with pytest.raises(ProvisionalTraceUnavailable):
        registry.claim(manifest, envelopes)
    assert registry.retained_bytes == 0


def test_temporary_dispatch_never_issues_even_if_saved_later() -> None:
    registry = ProvisionalTraceRegistry()
    promotion_id = new_opaque_id()
    attempt_id = new_opaque_id()
    temporary = freeze_provisional_capture_eligibility(
        capture_enabled=True,
        session_is_saved=False,
    )

    assert (
        registry.begin_attempt(
            policy=_policy(),
            promotion_id=promotion_id,
            attempt_id=attempt_id,
            eligibility=temporary,
        )
        is None
    )
    assert (
        registry.begin_attempt(
            policy=_policy(),
            promotion_id=promotion_id,
            attempt_id=attempt_id,
            eligibility=_eligible(),
        )
        is None
    )


def test_losing_or_cancelled_attempt_destroys_all_semantic_payloads() -> None:
    registry = ProvisionalTraceRegistry()
    _, attempt, _envelopes, _manifest = _sealed(registry)
    assert registry.retained_bytes > 0

    registry.abandon_attempt(attempt)
    assert registry.retained_bytes == 0


def test_ineligible_overflow_and_unsealed_attempts_expire_without_aba() -> None:
    clock = _Clock()
    registry = ProvisionalTraceRegistry(clock=clock)
    ineligible = freeze_provisional_capture_eligibility(
        capture_enabled=True,
        session_is_saved=False,
    )
    for _ in range(4_096):
        assert (
            registry.begin_attempt(
                policy=_policy(),
                promotion_id=new_opaque_id(),
                attempt_id=new_opaque_id(),
                eligibility=ineligible,
            )
            is None
        )
    assert len(registry._ineligible) == 4_096
    assert registry._ineligible_overflow_until is None
    ordinary = registry.begin_attempt(
        policy=_policy(),
        promotion_id=new_opaque_id(),
        attempt_id=new_opaque_id(),
        eligibility=_eligible(),
    )
    assert ordinary is not None
    registry.abandon_attempt(ordinary)

    denied_identity = (new_opaque_id(), new_opaque_id())
    assert (
        registry.begin_attempt(
            policy=_policy(),
            promotion_id=denied_identity[0],
            attempt_id=denied_identity[1],
            eligibility=ineligible,
        )
        is None
    )
    assert len(registry._ineligible) == 4_096
    assert (
        registry._ineligible_overflow_until == clock.now + PROVISIONAL_TRACE_TTL_SECONDS
    )
    assert (
        registry.begin_attempt(
            policy=_policy(),
            promotion_id=denied_identity[0],
            attempt_id=denied_identity[1],
            eligibility=_eligible(),
        )
        is None
    )
    assert (
        registry.begin_attempt(
            policy=_policy(),
            promotion_id=new_opaque_id(),
            attempt_id=new_opaque_id(),
            eligibility=_eligible(),
        )
        is None
    )

    clock.now += PROVISIONAL_TRACE_TTL_SECONDS + 0.001
    assert registry.reap_expired() == 0
    assert len(registry._ineligible) == 0
    assert registry._ineligible_overflow_until is None
    recovered = registry.begin_attempt(
        policy=_policy(),
        promotion_id=denied_identity[0],
        attempt_id=denied_identity[1],
        eligibility=_eligible(),
    )
    assert recovered is not None
    registry.abandon_attempt(recovered)

    promotion_id = new_opaque_id()
    attempt_id = new_opaque_id()
    stale = registry.begin_attempt(
        policy=_policy(),
        promotion_id=promotion_id,
        attempt_id=attempt_id,
        eligibility=_eligible(),
    )
    assert stale is not None
    registry._retain_gateway_call(
        stale,
        _call(promotion_id, 0, artifact_bytes=b"unsealed"),
    )
    retained_bytes = registry.retained_bytes
    assert retained_bytes > 0

    clock.now += PROVISIONAL_TRACE_TTL_SECONDS + 0.001
    assert registry.reap_expired() == 1
    assert registry.retained_bytes == 0
    replacement = registry.begin_attempt(
        policy=_policy(),
        promotion_id=promotion_id,
        attempt_id=attempt_id,
        eligibility=_eligible(),
    )
    assert replacement is not None
    assert replacement is not stale
    with pytest.raises(ProvisionalTraceUnavailable):
        registry._retain_gateway_call(stale, _call(promotion_id, 0))
    registry.abandon_attempt(replacement)
    assert registry.reap_expired() == 0
    assert registry.retained_bytes == 0


class _BlockingRepository:
    def __init__(self, outcome: object) -> None:
        self.entered = Event()
        self.proceed = Event()
        self.outcome = outcome

    def import_post_dispatch_trace(
        self,
        database: object,
        request: PostDispatchTraceImport,
    ) -> PostDispatchTraceImportResult:
        del database, request
        self.entered.set()
        assert self.proceed.wait(timeout=5)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        assert isinstance(self.outcome, PostDispatchTraceImportResult)
        return self.outcome


@pytest.mark.parametrize("confirmed_failure", [False, True])
def test_active_claim_is_pinned_against_reap_and_abandon_until_settlement(
    confirmed_failure: bool,
) -> None:
    clock = _Clock()
    registry = ProvisionalTraceRegistry(clock=clock)
    promotion_id, _attempt, envelopes, manifest = _sealed(registry)
    context = _context(promotion_id)
    outcome = (
        ConfirmedPreCommitTraceImportError() if confirmed_failure else _result(context)
    )
    repository = _BlockingRepository(outcome)
    service = ConsoleTraceService(repository=repository)  # type: ignore[arg-type]
    result: dict[str, object] = {}

    def import_trace() -> None:
        try:
            result["value"] = service.import_provisional_voice_trace(
                object(), registry, manifest, envelopes, context
            )
        except Exception as exc:
            result["error"] = exc

    worker = Thread(target=import_trace)
    worker.start()
    assert repository.entered.wait(timeout=5)
    retained_bytes = registry.retained_bytes
    clock.now += PROVISIONAL_TRACE_TTL_SECONDS + 0.001

    assert registry.reap_expired() == 0
    with pytest.raises(ProvisionalTraceUnavailable):
        registry.abandon_manifest(manifest)
    assert registry.retained_bytes == retained_bytes
    repository.proceed.set()
    worker.join(timeout=5)
    assert worker.is_alive() is False

    if confirmed_failure:
        error = result.get("error")
        assert isinstance(error, ProvisionalTraceImportExpiredError)
        assert isinstance(error.__cause__, ConfirmedPreCommitTraceImportError)
    else:
        assert isinstance(result.get("value"), PostDispatchTraceImportResult)
    assert registry.retained_bytes == 0


class _RecordingRepository:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.requests: list[PostDispatchTraceImport] = []

    def import_post_dispatch_trace(
        self,
        database: object,
        request: PostDispatchTraceImport,
    ) -> PostDispatchTraceImportResult:
        del database
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, PostDispatchTraceImportResult)
        return outcome


def _result(context: VoiceTraceImportContext, call_count: int = 2):
    return PostDispatchTraceImportResult(
        conversation_id=context.conversation_id,
        owner_id=new_opaque_id(),
        segment_id=new_opaque_id(),
        call_ids=derive_post_dispatch_trace_ids(
            context.import_id, call_count=call_count
        ).call_ids,
        already_imported=False,
    )


def test_import_materializes_winning_pair_revision_ids_before_repository_call() -> None:
    registry = ProvisionalTraceRegistry()
    promotion_id = new_opaque_id()
    attempt = registry.begin_attempt(
        policy=_policy(),
        promotion_id=promotion_id,
        attempt_id=new_opaque_id(),
        eligibility=_eligible(),
    )
    assert attempt is not None
    first = replace(
        _call(promotion_id, 0),
        request_surface=(
            PostDispatchTraceSurfaceComponent.omission(
                node_id=derive_post_dispatch_trace_node_id(promotion_id, 0, 0),
                component_kind="provider_message",
                reason_code="voice_user_revision_pending",
            ),
        ),
    )
    final = replace(
        _call(promotion_id, 1),
        response_started_at="2026-08-31T12:00:02Z",
        request_surface=(
            PostDispatchTraceSurfaceComponent.omission(
                node_id=derive_post_dispatch_trace_node_id(promotion_id, 1, 0),
                component_kind="provider_message",
                reason_code="voice_user_revision_pending",
            ),
        ),
        response=PostDispatchTraceResponse.no_response(
            "voice_assistant_revision_pending"
        ),
        terminal_state=TraceCallState.COMPLETE,
    )
    envelopes = (
        registry._retain_gateway_call(attempt, first),
        registry._retain_gateway_call(attempt, final),
    )
    assert all(envelope is not None for envelope in envelopes)
    manifest = registry.seal_attempt(attempt, expected_call_count=2)
    context = _context(promotion_id)
    repository = _RecordingRepository([_result(context, call_count=2)])

    ConsoleTraceService(repository=repository).import_provisional_voice_trace(
        object(),
        registry,
        manifest,
        envelopes,
        context,
    )

    imported_calls = repository.requests[0].calls
    assert all(
        call.request_surface[0].reference_kind == "revision"
        and call.request_surface[0].revision_id == context.user_revision_id
        for call in imported_calls
    )
    assert imported_calls[0].response == PostDispatchTraceResponse.no_response(
        "provider_error_no_response"
    )
    assert imported_calls[1].response == PostDispatchTraceResponse.committed_revision(
        context.assistant_revision_id
    )
    assert registry.retained_bytes == 0


def test_confirmed_precommit_failure_releases_same_manifest_for_retry() -> None:
    registry = ProvisionalTraceRegistry()
    promotion_id, _, envelopes, manifest = _sealed(registry)
    context = _context(promotion_id)
    repository = _RecordingRepository(
        [ConfirmedPreCommitTraceImportError(), _result(context)]
    )
    service = ConsoleTraceService(repository=repository)  # type: ignore[arg-type]

    with pytest.raises(ProvisionalTraceImportRetryableError):
        service.import_provisional_voice_trace(
            object(), registry, manifest, envelopes, context
        )
    result = service.import_provisional_voice_trace(
        object(), registry, manifest, envelopes, context
    )

    assert result.conversation_id == context.conversation_id
    assert repository.requests[0] == repository.requests[1]
    assert registry.retained_bytes == 0
    with pytest.raises(ProvisionalTraceUnavailable):
        registry.claim(manifest, envelopes)


def test_exception_after_commit_reconciles_before_consuming_once() -> None:
    registry = ProvisionalTraceRegistry()
    promotion_id, _, envelopes, manifest = _sealed(registry)
    context = _context(promotion_id)
    imported = _result(context)
    reconciled = replace(imported, already_imported=True)
    repository = _RecordingRepository([RuntimeError("after commit"), reconciled])
    service = ConsoleTraceService(repository=repository)  # type: ignore[arg-type]

    result = service.import_provisional_voice_trace(
        object(), registry, manifest, envelopes, context
    )

    assert result.already_imported is True
    assert len(repository.requests) == 2
    assert repository.requests[0] == repository.requests[1]
    assert registry.retained_bytes == 0
    with pytest.raises(ProvisionalTraceUnavailable):
        service.import_provisional_voice_trace(
            object(), registry, manifest, envelopes, context
        )


def test_gateway_api_never_exposes_raw_capture_tuples_or_any_annotations() -> None:
    methods = (
        ProvisionalTraceRegistry.begin_attempt,
        ProvisionalTraceRegistry._retain_gateway_call,
        ProvisionalTraceRegistry.seal_attempt,
        ProvisionalTraceRegistry.claim,
        ConsoleTraceService.import_provisional_voice_trace,
    )
    rendered = " ".join(str(inspect.signature(method)) for method in methods)
    assert "Any" not in rendered
    assert "ExchangeCapture" not in rendered
    assert "tuple[object" not in rendered


def test_public_registry_callables_expose_only_opaque_or_content_free_values() -> None:
    public_callables = (
        member
        for name, member in inspect.getmembers(ProvisionalTraceRegistry, callable)
        if not name.startswith("_")
    )
    rendered = " ".join(str(inspect.signature(member)) for member in public_callables)

    assert "PostDispatchTraceCall" not in rendered
    assert "PostDispatchTraceArtifact" not in rendered
    assert "sanitized_bytes" not in rendered
