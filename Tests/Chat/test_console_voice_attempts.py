from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError, replace
import gc
import inspect
import sys
import threading
import time
from typing import Any
from uuid import uuid4
import weakref

import pytest

from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderCallPurpose,
    ConsoleProviderGateway,
    ConsoleProviderResolution,
    ConsoleProviderStreamSignals,
    ProviderToolCalls,
)
from tldw_chatbook.Chat.console_prepared_request import build_console_request
from tldw_chatbook.Chat.console_exchange_capture import (
    CaptureDetail,
    freeze_provisional_capture_eligibility,
)
from tldw_chatbook.Chat.console_trace_models import (
    FrozenTracePolicy,
    TraceCallState,
    new_opaque_id,
)
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestRoute,
    ConsoleTraceCaptureMode,
    ProviderArtifactTraceProvenance,
    TraceProvenanceSource,
    request_route_provenance,
)
from tldw_chatbook.Chat.console_voice_attempts import (
    AttemptCleanupManager,
    AttemptCleanupOutcome,
    ProviderAttemptFailed,
    VoiceAttempt,
    VoiceAttemptDelta,
    VoiceAttemptRequest,
    VoiceAttemptSnapshot,
    VoiceAttemptToolRequest,
    VoiceCleanupCapacityExceeded,
)
from tldw_chatbook.Chat.console_voice_trace_gateway import (
    ProvisionalTraceEnvelope,
    ProvisionalTraceManifest,
    ProvisionalTraceUnavailable,
)
from tldw_chatbook.Chat.console_voice_trace_promotion import (
    PostDispatchTraceCall,
    PostDispatchTraceResponse,
    PostDispatchTraceSurfaceComponent,
    derive_post_dispatch_trace_ids,
    derive_post_dispatch_trace_node_id,
)
from tldw_chatbook.Chat.console_voice_supervisor import (
    VoiceDispatchKind,
    VoiceDispatchQuarantined,
    VoiceDispatchSupervisor,
)
from tldw_chatbook.LLM_Calls.hosted_chat import HostedChatTurn


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Read weather",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


@pytest.mark.asyncio
async def test_awaited_visible_sink_blocks_pulls_and_cancellation_closes_iterator():
    entered, release = asyncio.Event(), asyncio.Event()
    closed = asyncio.Event()
    pulls = []
    observed = []

    class Gateway:
        async def stream_chat(self, *args, **kwargs):
            try:
                for index in range(3):
                    pulls.append(index)
                    yield "😀" * 2049
            finally:
                closed.set()

    async def sink(delta):
        observed.append(delta.text)
        entered.set()
        await release.wait()

    attempt = VoiceAttempt(
        request=_request(),
        gateway=Gateway(),
        is_epoch_current=lambda epoch: True,
        visible_delta_sink=sink,
    )
    attempt.start()
    await entered.wait()
    assert pulls == [0]
    assert len(observed[0].encode()) == 4096
    attempt.request_cancellation()
    with pytest.raises(asyncio.CancelledError):
        await attempt.wait()
    assert closed.is_set()


def _request(*, epoch: int = 1, tools: list[dict[str, Any]] | None = None):
    resolution = ConsoleProviderResolution(
        provider="openai",
        base_url="",
        model="gpt-test",
        ready=True,
        execution_key="openai",
    )
    builder = ConsoleProviderGateway(http_client=object())  # type: ignore[arg-type]
    prepared = builder.prepare_chat_request(
        resolution,
        [{"role": "user", "content": "immutable prompt"}],
        tools=tools,
    )
    return VoiceAttemptRequest(
        attempt_epoch=epoch,
        resolution=resolution,
        prepared=prepared,
        exchange_capture_enabled=True,
    )


_CAPTURE_POLICY = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)


@pytest.mark.parametrize(
    "forgery", ["issuer", "bundle", "provenance", "preparation", "replay"]
)
def test_provisional_verifier_proof_is_exact_and_single_use(forgery):
    from tldw_chatbook.Chat.console_trace_final_values import (
        _SURFACE_VERIFICATION_ISSUER,
    )
    from tldw_chatbook.Chat.console_provider_gateway import (
        reconstruct_provider_gateway_kwargs,
    )

    gateway = ConsoleProviderGateway(http_client=object())
    attempt = gateway.begin_provisional_voice_trace(
        promotion_id=str(uuid4()),
        attempt_id=str(uuid4()),
        policy=_CAPTURE_POLICY,
        eligibility=freeze_provisional_capture_eligibility(
            capture_enabled=True, session_is_saved=True
        ),
    )
    registry = gateway.provisional_trace_registry
    boundary = registry._begin_gateway_call(
        attempt, gateway._retain_provisional_voice_trace_call
    )
    boundary.reserve()
    resolution = ConsoleProviderResolution(
        provider="openai",
        base_url="",
        model="fixture",
        ready=True,
        execution_key="openai",
    )
    request = _capture_on_request(gateway, resolution=resolution, trace_attempt=attempt)
    provenance = request.prepared.provenance
    bundle = gateway._verify_trace_shadow(
        resolution,
        request.prepared,
        reconstruct_provider_gateway_kwargs(resolution, request.prepared),
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
        trace_call_boundary=boundary,
    )
    assert bundle.available
    with pytest.raises(ProvisionalTraceUnavailable):
        if forgery in {"issuer", "preparation"}:
            foreign = registry._begin_gateway_call(
                attempt, gateway._retain_provisional_voice_trace_call
            )
            foreign.reserve()
            candidate = (
                replace(bundle, preparation_identity=foreign.preparation_identity)
                if forgery == "issuer"
                else bundle
            )
            foreign._bind_verified_bundle(
                provenance,
                candidate,
                object() if forgery == "issuer" else _SURFACE_VERIFICATION_ISSUER,
            )
        elif forgery == "bundle":
            boundary.mark_dispatch_started(replace(bundle), provenance)
        elif forgery == "provenance":
            boundary.mark_dispatch_started(bundle, replace(provenance))
        else:
            boundary.mark_dispatch_started(bundle, provenance)
            boundary.mark_dispatch_started(bundle, provenance)
    gateway.abandon_provisional_voice_trace(attempt)
    assert registry.retained_bytes == 0


@pytest.mark.asyncio
async def test_unavailable_real_verification_keeps_reply_but_never_seals_trace(
    monkeypatch,
):
    from tldw_chatbook.Chat import console_trace_final_values as final_values

    gateway = ConsoleProviderGateway(
        http_client=object(),
        chat_api_call_fn=lambda **_kwargs: {
            "choices": [{"message": {"content": "spoken reply"}}]
        },
    )
    trace_attempt = gateway.begin_provisional_voice_trace(
        promotion_id=str(uuid4()),
        attempt_id=str(uuid4()),
        policy=_CAPTURE_POLICY,
        eligibility=freeze_provisional_capture_eligibility(
            capture_enabled=True, session_is_saved=True
        ),
    )
    resolution = ConsoleProviderResolution(
        provider="openai",
        base_url="",
        model="fixture",
        ready=True,
        execution_key="openai",
    )
    request = _capture_on_request(
        gateway, resolution=resolution, trace_attempt=trace_attempt
    )

    def failed_normalization(_value):
        raise ValueError("fixture sanitizer failure")

    monkeypatch.setattr(
        final_values, "_normalize_provider_continuations", failed_normalization
    )
    failures = []
    attempt = VoiceAttempt(
        request=request,
        gateway=gateway,
        is_epoch_current=lambda _epoch: True,
        on_failed=failures.append,
    )
    await attempt.start()
    assert failures == []
    assert attempt.snapshot.response_text == "spoken reply"
    assert attempt.snapshot.trace_manifest is None
    assert attempt.snapshot.trace_envelopes == ()
    assert gateway.provisional_trace_registry.retained_bytes == 0


def _capture_on_request(
    gateway: ConsoleProviderGateway,
    *,
    resolution: ConsoleProviderResolution,
    trace_attempt,
    system: str | None = None,
) -> VoiceAttemptRequest:
    policy = _CAPTURE_POLICY
    active = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.ACTIVE_REQUEST,
        policy,
    )
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": "immutable prompt"}
    ]
    semantic = build_console_request(
        messages,
        message_provenance=tuple(
            ProviderArtifactTraceProvenance(
                TraceProvenanceSource.RENDERED_SYSTEM, policy
            )
            if message["role"] == "system"
            else active
            for message in messages
        ),
        memory_provenance=(),
        mandatory_provenance=(),
        tool_provenance=(),
        metadata_provenance=(request_route_provenance(ConsoleRequestRoute.FRESH),),
        capture_policy=policy,
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
    )
    prepared = gateway.prepare_chat_request(
        resolution,
        replace(semantic, capture_durability="durable"),
        route=ConsoleRequestRoute.FRESH,
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
    )
    return VoiceAttemptRequest(
        attempt_epoch=1,
        resolution=resolution,
        prepared=prepared,
        exchange_capture_enabled=True,
        provisional_trace_attempt=trace_attempt,
    )


def _trace_call(promotion_id: str) -> PostDispatchTraceCall:
    request_surface = (
        PostDispatchTraceSurfaceComponent.revision(
            node_id=derive_post_dispatch_trace_node_id(promotion_id, 0, 0),
            component_kind="provider_message",
            revision_id=str(uuid4()),
        ),
    )
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
        response_started_at=None,
        settled_at="2026-08-31T12:00:01Z",
        request_surface=request_surface,
        response=PostDispatchTraceResponse.no_response("provider_error_no_response"),
        sealed_payload_bytes=8,
        terminal_state=TraceCallState.ERROR,
    )


def test_voice_attempt_freezes_capture_detail_with_dispatch_request() -> None:
    request = replace(_request(), capture_detail=CaptureDetail.FULL)
    attempt = VoiceAttempt(
        request=request,
        gateway=_ScriptedGateway([]),
        is_epoch_current=lambda epoch: epoch == request.attempt_epoch,
    )

    assert attempt.signals.capture_detail is CaptureDetail.FULL


class _ScriptedGateway:
    def __init__(self, *items: object) -> None:
        self.items = items
        self.calls: list[dict[str, object]] = []

    async def stream_chat(
        self,
        resolution: object,
        prepared: object,
        *,
        tools: object,
        signals: ConsoleProviderStreamSignals,
    ):
        self.calls.append(
            {
                "resolution": resolution,
                "prepared": prepared,
                "tools": tools,
                "signals": signals,
            }
        )
        for item in self.items:
            if isinstance(item, BaseException):
                raise item
            yield item


async def _wait_for_thread_event(event: threading.Event) -> None:
    deadline = asyncio.get_running_loop().time() + 1.0
    while not event.is_set():
        if asyncio.get_running_loop().time() >= deadline:
            pytest.fail("provider thread event did not arrive")
        await asyncio.sleep(0.001)


@pytest.mark.asyncio
async def test_request_snapshot_is_immutable_and_tools_are_forwarded_directly() -> None:
    mutable_tools = [dict(TOOLS[0])]
    request = _request(tools=mutable_tools)
    mutable_tools[0]["type"] = "mutated"
    gateway = _ScriptedGateway("hello")
    current_epoch = 1
    attempt = VoiceAttempt(
        request=request,
        gateway=gateway,
        is_epoch_current=lambda epoch: epoch == current_epoch,
    )

    await attempt.start()

    with pytest.raises(FrozenInstanceError):
        request.attempt_epoch = 9  # type: ignore[misc]
    with pytest.raises(TypeError):
        request.prepared.tools[0]["type"] = "mutated"  # type: ignore[index]
    assert gateway.calls == [
        {
            "resolution": request.resolution,
            "prepared": request.prepared,
            "tools": request.prepared.tools,
            "signals": attempt.signals,
        }
    ]
    assert request.prepared.tools[0]["type"] == "function"


def test_request_repr_redacts_provider_resolution_credentials() -> None:
    resolution = ConsoleProviderResolution(
        provider="openai",
        base_url="https://user:password@example.test/v1?token=query-secret",
        model="gpt-test",
        ready=True,
        execution_key="openai",
    )
    prepared = ConsoleProviderGateway(  # type: ignore[arg-type]
        http_client=object()
    ).prepare_chat_request(
        resolution,
        [{"role": "user", "content": "private prompt"}],
    )
    request = VoiceAttemptRequest(
        attempt_epoch=1,
        resolution=resolution,
        prepared=prepared,
    )

    rendered = repr(request)

    assert "user" not in rendered
    assert "password" not in rendered
    assert "query-secret" not in rendered
    assert "example.test" not in rendered


@pytest.mark.asyncio
async def test_gateway_accepts_exact_frozen_tools_with_a_prepared_request() -> None:
    calls: list[dict[str, object]] = []

    def provider(**kwargs: object):
        calls.append(dict(kwargs))
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "weather",
                                    "arguments": "{}",
                                },
                            }
                        ]
                    }
                }
            ]
        }

    request = _request(tools=TOOLS)
    gateway = ConsoleProviderGateway(
        http_client=object(),  # type: ignore[arg-type]
        chat_api_call_fn=provider,
    )

    output = [
        item
        async for item in gateway.stream_chat(
            request.resolution,
            request.prepared,
            tools=request.prepared.tools,
            signals=ConsoleProviderStreamSignals(exchange_capture_enabled=False),
            dispatch_purpose=ConsoleProviderCallPurpose.VOICE_PROVISIONAL,
        )
    ]

    assert len(calls) == 1
    assert calls[0]["tools"] == TOOLS
    assert len(output) == 1
    assert isinstance(output[0], ProviderToolCalls)

    with pytest.raises(ValueError, match="do not match"):
        _ = [
            item
            async for item in gateway.stream_chat(
                request.resolution,
                request.prepared,
                tools=[{"type": "function", "function": {"name": "wrong"}}],
                signals=ConsoleProviderStreamSignals(exchange_capture_enabled=False),
                dispatch_purpose=ConsoleProviderCallPurpose.VOICE_PROVISIONAL,
            )
        ]


@pytest.mark.asyncio
async def test_real_gateway_empty_terminal_tool_metadata_does_not_freeze_speech() -> (
    None
):
    class _TerminalResponse(dict):
        terminal_turn = HostedChatTurn(
            text="hello",
            tool_calls=(),
            assistant_message={"role": "assistant", "content": "hello"},
            finish_reason="stop",
        )

    gateway = ConsoleProviderGateway(
        http_client=object(),  # type: ignore[arg-type]
        chat_api_call_fn=lambda **_kwargs: _TerminalResponse(
            {"choices": [{"message": {"content": "hello"}}]}
        ),
    )
    tool_requests: list[VoiceAttemptToolRequest] = []
    deltas: list[VoiceAttemptDelta] = []
    attempt = VoiceAttempt(
        request=_request(tools=TOOLS),
        gateway=gateway,
        is_epoch_current=lambda epoch: epoch == 1,
        on_delta=deltas.append,
        on_tool_request=tool_requests.append,
    )

    await attempt.start()

    assert [event.text for event in deltas] == ["hello"]
    assert tool_requests == []
    assert attempt.snapshot.response_text == "hello"
    assert attempt.snapshot.tool_request is None
    assert attempt.snapshot.speech_frozen is False


def test_attempt_has_no_store_executor_approval_or_citation_sink() -> None:
    parameters = inspect.signature(VoiceAttempt).parameters

    assert not {
        "store",
        "tool_executor",
        "approval_hook",
        "citation_sink",
        "agent_service",
    }.intersection(parameters)
    with pytest.raises(TypeError):
        VoiceAttempt(  # type: ignore[call-arg]
            request=_request(),
            gateway=_ScriptedGateway(),
            is_epoch_current=lambda _epoch: True,
            store=object(),
        )


@pytest.mark.asyncio
async def test_first_complete_tool_request_freezes_later_provisional_speech() -> None:
    first = ProviderToolCalls(
        (
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "weather", "arguments": "{}"},
            },
        )
    )
    second = ProviderToolCalls(
        (
            {
                "id": "call-2",
                "type": "function",
                "function": {"name": "later", "arguments": "{}"},
            },
        )
    )
    deltas: list[VoiceAttemptDelta] = []
    tool_requests: list[VoiceAttemptToolRequest] = []
    attempt = VoiceAttempt(
        request=_request(tools=TOOLS),
        gateway=_ScriptedGateway("safe preamble", first, "must not speak", second),
        is_epoch_current=lambda epoch: epoch == 1,
        on_delta=deltas.append,
        on_tool_request=tool_requests.append,
    )

    await attempt.start()

    assert [event.text for event in deltas] == ["safe preamble"]
    assert len(tool_requests) == 1
    assert tool_requests[0].tool_calls[0]["id"] == "call-1"
    assert attempt.snapshot.response_text == "safe preamble"
    assert attempt.snapshot.speech_frozen is True


@pytest.mark.asyncio
async def test_tool_request_immediately_destroys_provisional_trace_capability() -> None:
    promotion_id = str(uuid4())
    gateway = ConsoleProviderGateway(
        http_client=object(),  # type: ignore[arg-type]
        chat_api_call_fn=lambda **_kwargs: {
            "choices": [
                {
                    "message": {
                        "content": "safe preamble",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "weather",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    }
                }
            ]
        },
    )
    trace_attempt = gateway.begin_provisional_voice_trace(
        policy=_CAPTURE_POLICY,
        promotion_id=promotion_id,
        attempt_id=str(uuid4()),
        eligibility=freeze_provisional_capture_eligibility(
            capture_enabled=True,
            session_is_saved=True,
        ),
    )
    assert trace_attempt is not None
    gateway._retain_provisional_voice_trace_call(
        trace_attempt,
        _trace_call(promotion_id),
    )
    request = _request(tools=TOOLS)
    attempt = VoiceAttempt(
        request=VoiceAttemptRequest(
            attempt_epoch=request.attempt_epoch,
            resolution=request.resolution,
            prepared=request.prepared,
            exchange_capture_enabled=True,
            provisional_trace_attempt=trace_attempt,
        ),
        gateway=gateway,
        is_epoch_current=lambda epoch: epoch == 1,
    )

    await attempt.start()

    assert attempt.snapshot.response_text == "safe preamble"
    assert attempt.snapshot.tool_request is not None
    assert attempt.snapshot.trace_manifest is None
    assert attempt.snapshot.trace_envelopes == ()
    with pytest.raises(ProvisionalTraceUnavailable):
        gateway.seal_provisional_voice_trace(trace_attempt)


@pytest.mark.asyncio
async def test_failed_trace_seal_does_not_discard_winning_response() -> None:
    gateway = ConsoleProviderGateway(
        http_client=object(),  # type: ignore[arg-type]
        chat_api_call_fn=lambda **_kwargs: {
            "choices": [{"message": {"content": "exact assistant"}}]
        },
    )
    trace_attempt = gateway.begin_provisional_voice_trace(
        policy=_CAPTURE_POLICY,
        promotion_id=str(uuid4()),
        attempt_id=str(uuid4()),
        eligibility=freeze_provisional_capture_eligibility(
            capture_enabled=True,
            session_is_saved=True,
        ),
    )
    assert trace_attempt is not None
    request = _request()
    attempt = VoiceAttempt(
        request=VoiceAttemptRequest(
            attempt_epoch=request.attempt_epoch,
            resolution=request.resolution,
            prepared=request.prepared,
            exchange_capture_enabled=True,
            provisional_trace_attempt=trace_attempt,
        ),
        gateway=gateway,
        is_epoch_current=lambda epoch: epoch == 1,
    )

    await attempt.start()

    assert attempt.snapshot.response_text == "exact assistant"
    assert attempt.snapshot.trace_manifest is None
    assert attempt.snapshot.trace_envelopes == ()
    with pytest.raises(ProvisionalTraceUnavailable):
        gateway.seal_provisional_voice_trace(trace_attempt)


@pytest.mark.asyncio
async def test_provider_failure_is_typed_and_contains_no_exception_or_body() -> None:
    failures: list[ProviderAttemptFailed] = []
    attempt = VoiceAttempt(
        request=_request(epoch=7),
        gateway=_ScriptedGateway(RuntimeError("secret prompt and response body")),
        is_epoch_current=lambda epoch: epoch == 7,
        on_failed=failures.append,
    )

    await attempt.start()

    assert failures == [ProviderAttemptFailed(7, "RuntimeError")]
    assert "secret" not in repr(failures[0])
    assert not hasattr(failures[0], "exception")
    assert attempt.snapshot.response_text == ""


@pytest.mark.asyncio
async def test_synchronous_invalidation_rejects_late_delta_usage_and_capture() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    seen_signals: list[ConsoleProviderStreamSignals] = []

    class _CancellationResistantGateway:
        async def stream_chat(self, _resolution, _prepared, *, tools, signals):
            del tools
            seen_signals.append(signals)
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            call = signals.new_usage_call()
            call.record_usage_payload({"completion_tokens": 999})
            call.begin_exchange(
                provider="openai",
                model="gpt-test",
                endpoint=None,
                request={"body": "late secret"},
                omitted_keys=(),
            )
            call.record_exchange_content("late secret")
            call.close_exchange(status="complete")
            call.close_usage_call()
            yield "late secret"

    current_epoch = 3
    deltas: list[VoiceAttemptDelta] = []
    attempt = VoiceAttempt(
        request=_request(epoch=3),
        gateway=_CancellationResistantGateway(),
        is_epoch_current=lambda epoch: epoch == current_epoch,
        on_delta=deltas.append,
    )
    runner = attempt.start()
    await started.wait()

    attempt.invalidate()
    current_epoch = 4
    release.set()
    await runner

    assert seen_signals == [attempt.signals]
    assert deltas == []
    assert attempt.snapshot.response_text == ""
    assert attempt.snapshot.usage_payloads == ()
    assert attempt.snapshot.trace_manifest is None
    assert attempt.snapshot.trace_envelopes == ()


@pytest.mark.asyncio
async def test_winning_snapshot_carries_only_gateway_sealed_trace_capabilities() -> (
    None
):
    promotion_id = str(uuid4())
    attempt_id = str(uuid4())
    gateway = ConsoleProviderGateway(
        http_client=object(),  # type: ignore[arg-type]
        chat_api_call_fn=lambda **_kwargs: {
            "choices": [{"message": {"content": "exact assistant"}}]
        },
    )
    trace_attempt = gateway.begin_provisional_voice_trace(
        policy=_CAPTURE_POLICY,
        promotion_id=promotion_id,
        attempt_id=attempt_id,
        eligibility=freeze_provisional_capture_eligibility(
            capture_enabled=True,
            session_is_saved=True,
        ),
    )
    assert trace_attempt is not None
    resolution = ConsoleProviderResolution(
        provider="openai",
        base_url="",
        model="gpt-test",
        ready=True,
        execution_key="openai",
    )
    request = _capture_on_request(
        gateway,
        resolution=resolution,
        trace_attempt=trace_attempt,
    )
    failures = []
    attempt = VoiceAttempt(
        request=request,
        gateway=gateway,
        is_epoch_current=lambda epoch: epoch == 1,
        on_failed=failures.append,
    )

    await attempt.start()
    snapshot = attempt.snapshot

    assert failures == []
    assert snapshot.response_text == "exact assistant"
    assert type(snapshot.trace_manifest) is ProvisionalTraceManifest
    assert len(snapshot.trace_envelopes) == 1
    assert type(snapshot.trace_envelopes[0]) is ProvisionalTraceEnvelope
    assert snapshot.trace_manifest.promotion_id == promotion_id
    assert snapshot.trace_manifest.attempt_id == attempt_id
    rendered_annotations = repr(VoiceAttemptSnapshot.__annotations__)
    assert "Any" not in rendered_annotations
    assert "exchange_captures" not in rendered_annotations


@pytest.mark.asyncio
async def test_real_gateway_settles_provisional_before_manifest_seal(
    monkeypatch,
) -> None:
    from tldw_chatbook.Chat import console_voice_trace_gateway as trace_gateway

    original_artifact = trace_gateway._artifact
    projection_threads = []

    def record_artifact(**kwargs):
        projection_threads.append(threading.get_ident())
        return original_artifact(**kwargs)

    monkeypatch.setattr(trace_gateway, "_artifact", record_artifact)
    gateway = ConsoleProviderGateway(
        http_client=object(),
        chat_api_call_fn=lambda **kwargs: {
            "choices": [{"message": {"content": "answer"}}]
        },
    )
    trace_attempt = gateway.begin_provisional_voice_trace(
        policy=_CAPTURE_POLICY,
        promotion_id=str(uuid4()),
        attempt_id=str(uuid4()),
        eligibility=freeze_provisional_capture_eligibility(
            capture_enabled=True,
            session_is_saved=True,
        ),
    )
    resolution = ConsoleProviderResolution(
        provider="openai",
        base_url="",
        model="gpt-test",
        ready=True,
        execution_key="openai",
    )
    request = _capture_on_request(
        gateway,
        resolution=resolution,
        trace_attempt=trace_attempt,
        system="provider-only context",
    )
    output = [
        item
        async for item in gateway.stream_chat(
            resolution,
            request.prepared,
            route=ConsoleRequestRoute.FRESH,
            capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
            dispatch_purpose=ConsoleProviderCallPurpose.VOICE_PROVISIONAL,
            provisional_trace_attempt=trace_attempt,
        )
    ]
    assert output == ["answer"]
    assert projection_threads and threading.get_ident() not in projection_threads
    manifest, envelopes = gateway.seal_provisional_voice_trace(trace_attempt)
    assert len(envelopes) == 1
    assert manifest.promotion_id == trace_attempt.promotion_id


@pytest.mark.asyncio
async def test_cancelled_projection_keeps_thread_custody_and_discards_late_artifact(
    monkeypatch,
):
    from tldw_chatbook.Chat import console_voice_trace_gateway as trace_gateway

    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    original = trace_gateway._artifact

    def blocked_artifact(**kwargs):
        entered.set()
        assert release.wait(2.0)
        try:
            return original(**kwargs)
        finally:
            returned.set()

    monkeypatch.setattr(trace_gateway, "_artifact", blocked_artifact)
    gateway = ConsoleProviderGateway(
        http_client=object(),
        chat_api_call_fn=lambda **_kwargs: {
            "choices": [{"message": {"content": "answer"}}]
        },
    )
    trace_attempt = gateway.begin_provisional_voice_trace(
        policy=_CAPTURE_POLICY,
        promotion_id=str(uuid4()),
        attempt_id=str(uuid4()),
        eligibility=freeze_provisional_capture_eligibility(
            capture_enabled=True, session_is_saved=True
        ),
    )
    resolution = ConsoleProviderResolution(
        provider="openai",
        base_url="",
        model="gpt-test",
        ready=True,
        execution_key="openai",
    )
    request = _capture_on_request(
        gateway,
        resolution=resolution,
        trace_attempt=trace_attempt,
        system="provider-only context",
    )
    attempt = VoiceAttempt(
        request=request, gateway=gateway, is_epoch_current=lambda epoch: epoch == 1
    )
    supervisor = VoiceDispatchSupervisor()

    async def observe_completion(task, _timeout):
        await asyncio.sleep(0)
        return task.done()

    manager = AttemptCleanupManager(supervisor, wait_for_exit=observe_completion)
    attempt.start()
    try:
        await _wait_for_thread_event(entered)
        assert await manager.cancel(attempt) is AttemptCleanupOutcome.DETACHED
        assert not returned.is_set() and supervisor.orphan_count == 1
        assert gateway.provisional_trace_registry.retained_bytes == 0
    finally:
        release.set()
        await _wait_for_thread_event(returned)
        await asyncio.wait_for(
            asyncio.gather(attempt.provider_cleanup_task, return_exceptions=True),
            timeout=1.0,
        )
        for _ in range(20):
            if supervisor.orphan_count == 0:
                break
            await asyncio.sleep(0)
    assert supervisor.orphan_count == 0
    assert gateway.provisional_trace_registry.retained_bytes == 0
    with pytest.raises(ProvisionalTraceUnavailable):
        gateway.seal_provisional_voice_trace(trace_attempt)


def test_invalidated_attempt_destroys_gateway_trace_capability() -> None:
    promotion_id = str(uuid4())
    gateway = ConsoleProviderGateway(http_client=object())  # type: ignore[arg-type]
    trace_attempt = gateway.begin_provisional_voice_trace(
        policy=_CAPTURE_POLICY,
        promotion_id=promotion_id,
        attempt_id=str(uuid4()),
        eligibility=freeze_provisional_capture_eligibility(
            capture_enabled=True,
            session_is_saved=True,
        ),
    )
    assert trace_attempt is not None
    gateway._retain_provisional_voice_trace_call(
        trace_attempt,
        _trace_call(promotion_id),
    )
    request = _request()
    attempt = VoiceAttempt(
        request=VoiceAttemptRequest(
            attempt_epoch=request.attempt_epoch,
            resolution=request.resolution,
            prepared=request.prepared,
            exchange_capture_enabled=True,
            provisional_trace_attempt=trace_attempt,
        ),
        gateway=gateway,
        is_epoch_current=lambda epoch: epoch == 1,
    )

    attempt.invalidate()

    assert attempt.snapshot.trace_manifest is None
    assert attempt.snapshot.trace_envelopes == ()
    with pytest.raises(ProvisionalTraceUnavailable):
        gateway.seal_provisional_voice_trace(trace_attempt)


@pytest.mark.asyncio
async def test_cancelled_waiter_immediately_invalidates_attempt_and_trace() -> None:
    started = asyncio.Event()

    class _BlockingGateway:
        def __init__(self, owner: ConsoleProviderGateway) -> None:
            self._owner = owner

        async def stream_chat(self, _resolution, _prepared, *, tools, signals):
            del tools, signals
            started.set()
            await asyncio.Future()
            yield "unreachable"

        def abandon_provisional_voice_trace(self, trace_attempt) -> None:
            self._owner.abandon_provisional_voice_trace(trace_attempt)

    promotion_id = str(uuid4())
    gateway = ConsoleProviderGateway(http_client=object())  # type: ignore[arg-type]
    trace_attempt = gateway.begin_provisional_voice_trace(
        policy=_CAPTURE_POLICY,
        promotion_id=promotion_id,
        attempt_id=str(uuid4()),
        eligibility=freeze_provisional_capture_eligibility(
            capture_enabled=True,
            session_is_saved=True,
        ),
    )
    assert trace_attempt is not None
    gateway._retain_provisional_voice_trace_call(
        trace_attempt,
        _trace_call(promotion_id),
    )
    request = _request()
    attempt = VoiceAttempt(
        request=VoiceAttemptRequest(
            attempt_epoch=request.attempt_epoch,
            resolution=request.resolution,
            prepared=request.prepared,
            exchange_capture_enabled=True,
            provisional_trace_attempt=trace_attempt,
        ),
        gateway=_BlockingGateway(gateway),
        is_epoch_current=lambda epoch: epoch == 1,
    )
    attempt.start()
    await started.wait()
    waiter = asyncio.create_task(attempt.wait())
    await asyncio.sleep(0)

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert attempt.snapshot.response_text == ""
    assert attempt.snapshot.trace_manifest is None
    assert attempt.snapshot.trace_envelopes == ()
    assert gateway.provisional_trace_registry.retained_bytes == 0
    with pytest.raises(ProvisionalTraceUnavailable):
        gateway.seal_provisional_voice_trace(trace_attempt)


def test_invalidation_scrubs_open_trace_and_rejects_late_provider_content() -> None:
    from tldw_chatbook.Chat.console_provider_gateway import _TraceResponseAccumulator

    attempt = VoiceAttempt(
        request=_request(),
        gateway=_ScriptedGateway(),
        is_epoch_current=lambda epoch: epoch == 1,
    )
    accumulator = _TraceResponseAccumulator()
    attempt.signals.observe_trace_response(accumulator, "early secret", synthetic=False)
    assert accumulator.items == ("early secret",)

    attempt.invalidate()
    attempt.signals.observe_trace_response(accumulator, "late secret", synthetic=False)

    assert accumulator.items == ()
    assert accumulator._retained_bytes == 0
    assert accumulator.omission_reason == "voice_attempt_discarded"
    assert attempt.signals._trace_accumulator is None


@pytest.mark.asyncio
async def test_tool_barrier_scrubs_open_and_late_trace_observations() -> None:
    from tldw_chatbook.Chat.console_provider_gateway import _TraceResponseAccumulator

    class _ToolGateway:
        accumulator = _TraceResponseAccumulator()

        async def stream_chat(self, _resolution, _prepared, *, tools, signals):
            del tools
            signals.observe_trace_response(
                self.accumulator, "early secret", synthetic=False
            )
            yield ProviderToolCalls(
                (
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": "{}"},
                    },
                )
            )
            signals.observe_trace_response(
                self.accumulator, "late secret", synthetic=False
            )

    gateway = _ToolGateway()
    attempt = VoiceAttempt(
        request=_request(tools=TOOLS),
        gateway=gateway,
        is_epoch_current=lambda epoch: epoch == 1,
    )

    await attempt.start()

    assert gateway.accumulator.items == ()
    assert gateway.accumulator._retained_bytes == 0
    assert gateway.accumulator.omission_reason == "voice_attempt_discarded"
    assert attempt.signals._trace_accumulator is None


def test_exchange_begin_rechecks_epoch_inside_capture_lock() -> None:
    """Invalidation between the fast-path check and mutation cannot re-open capture."""

    current = True
    second_check_entered = threading.Event()
    release_second_check = threading.Event()
    checks = 0

    def is_current(_epoch: int) -> bool:
        nonlocal checks
        checks += 1
        accepted = current
        if checks == 2:
            second_check_entered.set()
            assert release_second_check.wait(1.0)
        return accepted

    attempt = VoiceAttempt(
        request=_request(),
        gateway=_ScriptedGateway(),
        is_epoch_current=is_current,
    )
    signals = attempt.signals
    call = signals.new_usage_call()

    worker = threading.Thread(
        target=lambda: call.begin_exchange(
            provider="openai",
            model="gpt-test",
            endpoint=None,
            request={"messages": []},
            omitted_keys=(),
        )
    )
    worker.start()
    assert second_check_entered.wait(1.0)
    current = False
    signals.discard_exchange_captures()
    release_second_check.set()
    worker.join(1.0)

    assert not worker.is_alive()
    assert signals.exchange_captures() == []


@pytest.mark.asyncio
async def test_cancellation_fences_before_closing_attempt_owned_tts_handle() -> None:
    attempt = VoiceAttempt(
        request=_request(),
        gateway=_ScriptedGateway(),
        is_epoch_current=lambda epoch: epoch == 1,
    )
    fence_states: list[bool] = []
    closed = asyncio.Event()

    async def close_tts() -> None:
        fence_states.append(attempt.signals.accepts_events())
        closed.set()

    attempt.register_tts_canceller(close_tts)
    attempt.request_cancellation()
    await closed.wait()

    assert fence_states == [False]


@pytest.mark.asyncio
async def test_tts_cancellation_owns_custom_awaitable_future_and_existing_task() -> (
    None
):
    release = asyncio.Event()
    future = asyncio.get_running_loop().create_future()
    completed: list[str] = []

    class _CustomAwaitable:
        def __await__(self):
            async def run() -> None:
                await release.wait()
                completed.append("custom")

            return run().__await__()

    async def existing_work() -> None:
        await release.wait()
        completed.append("task")

    existing_task = asyncio.create_task(existing_work())
    attempt = VoiceAttempt(
        request=_request(),
        gateway=_ScriptedGateway(),
        is_epoch_current=lambda epoch: epoch == 1,
    )
    attempt.register_tts_canceller(_CustomAwaitable)
    attempt.register_tts_canceller(lambda: future)
    attempt.register_tts_canceller(lambda: existing_task)
    await attempt.start()

    attempt.request_cancellation()
    completion = attempt.cleanup_task
    await asyncio.sleep(0)

    assert completion.done() is False
    future.set_result(None)
    release.set()
    await completion
    assert completed == ["task", "custom"]


class _UncooperativeStream:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.force_close_calls = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.started.set()
        while not self.release.is_set():
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                continue
        raise StopAsyncIteration

    async def force_close(self) -> None:
        self.force_close_calls += 1


class _StreamGateway:
    def __init__(self, stream: _UncooperativeStream) -> None:
        self.stream = stream

    def stream_chat(self, _resolution, _prepared, *, tools, signals):
        del tools, signals
        return self.stream


@pytest.mark.asyncio
async def test_cooperative_cancellation_exits_without_conservative_transition() -> None:
    started = asyncio.Event()
    never = asyncio.Event()

    class _CooperativeGateway:
        async def stream_chat(self, _resolution, _prepared, *, tools, signals):
            del tools, signals
            started.set()
            await never.wait()
            yield "unreachable"

    attempt = VoiceAttempt(
        request=_request(),
        gateway=_CooperativeGateway(),
        is_epoch_current=lambda epoch: epoch == 1,
    )
    attempt.start()
    await started.wait()
    conservative: list[int] = []
    manager = AttemptCleanupManager(
        VoiceDispatchSupervisor(),
        on_conservative=conservative.append,
    )

    outcome = await manager.cancel(attempt)

    assert outcome is AttemptCleanupOutcome.CLEAN
    assert conservative == []
    assert manager.obsolete_cleanup_count == 0


@pytest.mark.asyncio
async def test_real_gateway_provider_thread_remains_owned_through_detach() -> None:
    entered = threading.Event()
    release = threading.Event()
    provider_returned = threading.Event()
    response_closed = threading.Event()

    class _ClosableResponse(dict):
        def close(self) -> None:
            response_closed.set()

    def blocking_provider(**_kwargs: object) -> object:
        entered.set()
        release.wait(2.0)
        provider_returned.set()
        return _ClosableResponse(
            {"choices": [{"message": {"content": "must stay fenced"}}]}
        )

    gateway = ConsoleProviderGateway(
        http_client=object(),  # type: ignore[arg-type]
        chat_api_call_fn=blocking_provider,
    )
    attempt = VoiceAttempt(
        request=_request(),
        gateway=gateway,
        is_epoch_current=lambda epoch: epoch == 1,
    )
    attempt.start()
    await _wait_for_thread_event(entered)
    waits: list[float] = []

    async def observe_completion(task: asyncio.Task[object], timeout: float) -> bool:
        waits.append(timeout)
        await asyncio.sleep(0)
        return task.done()

    supervisor = VoiceDispatchSupervisor()
    manager = AttemptCleanupManager(supervisor, wait_for_exit=observe_completion)
    try:
        outcome = await manager.cancel(attempt)
        still_running_at_outcome = not provider_returned.is_set()
        orphan_count_at_outcome = supervisor.orphan_count
    finally:
        release.set()
        await _wait_for_thread_event(provider_returned)
        await _wait_for_thread_event(response_closed)
        for _ in range(20):
            if supervisor.orphan_count == 0:
                break
            await asyncio.sleep(0)

    assert outcome is AttemptCleanupOutcome.DETACHED
    assert waits == pytest.approx([2.0, 5.0, 5.5], abs=0.02)
    assert still_running_at_outcome is True
    assert orphan_count_at_outcome == 1
    assert supervisor.orphan_count == 0


@pytest.mark.asyncio
async def test_real_gateway_blocking_response_close_is_off_loop_and_owned() -> None:
    next_blocked = threading.Event()
    release_next = threading.Event()
    close_entered = threading.Event()
    release_close = threading.Event()
    close_finished = threading.Event()

    class _BlockingCloseResponse:
        def __init__(self) -> None:
            self._first = True

        def __iter__(self):
            return self

        def __next__(self) -> object:
            if self._first:
                self._first = False
                return {"choices": [{"delta": {"content": "partial"}}]}
            next_blocked.set()
            release_next.wait(2.0)
            raise StopIteration

        def close(self) -> None:
            close_entered.set()
            release_next.set()
            release_close.wait(2.0)
            close_finished.set()

    response = _BlockingCloseResponse()
    gateway = ConsoleProviderGateway(
        http_client=object(),  # type: ignore[arg-type]
        chat_api_call_fn=lambda _response=response, **_kwargs: _response,
    )
    attempt = VoiceAttempt(
        request=_request(),
        gateway=gateway,
        is_epoch_current=lambda epoch: epoch == 1,
    )
    attempt.start()
    await _wait_for_thread_event(next_blocked)
    waits: list[float] = []

    async def observe_completion(task: asyncio.Task[object], timeout: float) -> bool:
        waits.append(timeout)
        await asyncio.sleep(0)
        return task.done()

    supervisor = VoiceDispatchSupervisor()
    manager = AttemptCleanupManager(supervisor, wait_for_exit=observe_completion)
    delayed_release = threading.Timer(0.5, release_close.set)
    delayed_release.start()
    try:
        outcome = await manager.cancel(attempt)
        released_at_outcome = release_close.is_set()
        close_finished_at_outcome = close_finished.is_set()
        orphan_count_at_outcome = supervisor.orphan_count
    finally:
        release_next.set()
        release_close.set()
        delayed_release.cancel()
        delayed_release.join(1.0)
        await _wait_for_thread_event(close_entered)
        await _wait_for_thread_event(close_finished)
        for _ in range(20):
            if supervisor.orphan_count == 0:
                break
            await asyncio.sleep(0)

    assert outcome is AttemptCleanupOutcome.DETACHED
    assert waits == pytest.approx([2.0, 5.0, 5.5], abs=0.02)
    assert released_at_outcome is False
    assert close_finished_at_outcome is False
    assert orphan_count_at_outcome == 1
    assert supervisor.orphan_count == 0


@pytest.mark.asyncio
async def test_detached_real_close_does_not_retain_coordinator_content_owner() -> None:
    release_next = threading.Event()
    next_blocked = threading.Event()
    close_entered = threading.Event()
    release_close = threading.Event()
    close_finished = threading.Event()

    class _Coordinator:
        def __init__(self) -> None:
            self.private_transcript = "private transcript body"

        def is_epoch_current(self, epoch: int) -> bool:
            return epoch == 1

    class _BlockingCloseResponse:
        def __init__(self) -> None:
            self._first = True

        def __iter__(self):
            return self

        def __next__(self) -> object:
            if self._first:
                self._first = False
                return {"choices": [{"delta": {"content": "private reply body"}}]}
            next_blocked.set()
            release_next.wait(2.0)
            raise StopIteration

        def close(self) -> None:
            close_entered.set()
            release_next.set()
            release_close.wait(2.0)
            close_finished.set()

    coordinator = _Coordinator()
    coordinator_ref = weakref.ref(coordinator)
    response = _BlockingCloseResponse()
    gateway = ConsoleProviderGateway(
        http_client=object(),  # type: ignore[arg-type]
        chat_api_call_fn=lambda _response=response, **_kwargs: _response,
    )
    attempt = VoiceAttempt(
        request=_request(),
        gateway=gateway,
        is_epoch_current=coordinator.is_epoch_current,
    )
    attempt.start()
    await _wait_for_thread_event(next_blocked)

    async def never_complete(_task: asyncio.Task[object], _timeout: float) -> bool:
        await asyncio.sleep(0)
        return False

    supervisor = VoiceDispatchSupervisor()
    manager = AttemptCleanupManager(supervisor, wait_for_exit=never_complete)
    cleanup = manager.cancel(attempt)
    try:
        assert await cleanup is AttemptCleanupOutcome.DETACHED
        await _wait_for_thread_event(close_entered)
        assert close_finished.is_set() is False
        assert supervisor.orphan_count == 1

        del cleanup
        del attempt
        del manager
        del gateway
        del response
        del coordinator
        gc.collect()

        assert coordinator_ref() is None
        assert supervisor.orphan_count == 1
    finally:
        release_next.set()
        release_close.set()
        await _wait_for_thread_event(close_finished)
        for _ in range(20):
            if supervisor.orphan_count == 0:
                break
            await asyncio.sleep(0)

    assert supervisor.orphan_count == 0


@pytest.mark.asyncio
async def test_cleanup_deadlines_transition_force_close_then_detach() -> None:
    stream = _UncooperativeStream()
    attempt = VoiceAttempt(
        request=_request(epoch=8),
        gateway=_StreamGateway(stream),
        is_epoch_current=lambda epoch: epoch == 8,
    )
    attempt.start()
    await stream.started.wait()
    waits: list[float] = []

    async def never_exits(_task: asyncio.Task[object], timeout: float) -> bool:
        waits.append(timeout)
        await asyncio.sleep(0)
        return False

    conservative: list[int] = []
    detached: list[int] = []
    supervisor = VoiceDispatchSupervisor()
    manager = AttemptCleanupManager(
        supervisor,
        wait_for_exit=never_exits,
        on_conservative=conservative.append,
        on_detached=detached.append,
    )

    outcome = await manager.cancel(attempt)

    assert outcome is AttemptCleanupOutcome.DETACHED
    assert waits == pytest.approx([2.0, 5.0, 5.5], abs=0.02)
    assert conservative == [8]
    assert stream.force_close_calls == 1
    assert detached == [8]
    assert supervisor.orphan_count == 1
    assert manager.obsolete_cleanup_count == 0

    stream.release.set()
    await attempt.wait()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert supervisor.orphan_count == 0


@pytest.mark.asyncio
async def test_cancelling_cleanup_observer_does_not_cancel_owned_escalation() -> None:
    stream = _UncooperativeStream()
    attempt = VoiceAttempt(
        request=_request(epoch=11),
        gateway=_StreamGateway(stream),
        is_epoch_current=lambda epoch: epoch == 11,
    )
    attempt.start()
    await stream.started.wait()
    allow_escalation = asyncio.Event()

    async def blocked_wait(_task: asyncio.Task[object], _timeout: float) -> bool:
        await allow_escalation.wait()
        return False

    conservative: list[int] = []
    detached: list[int] = []
    supervisor = VoiceDispatchSupervisor()
    manager = AttemptCleanupManager(
        supervisor,
        wait_for_exit=blocked_wait,
        on_conservative=conservative.append,
        on_detached=detached.append,
    )
    observer = manager.cancel(attempt)
    try:
        await asyncio.sleep(0)

        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        allow_escalation.set()
        for _ in range(20):
            if supervisor.orphan_count == 1 and manager.obsolete_cleanup_count == 0:
                break
            await asyncio.sleep(0)

        assert conservative == [11]
        assert detached == [11]
        assert stream.force_close_calls == 1
        assert supervisor.orphan_count == 1
        assert manager.obsolete_cleanup_count == 0
    finally:
        allow_escalation.set()
        stream.release.set()
        await attempt.wait()
        for _ in range(20):
            if supervisor.orphan_count == 0:
                break
            await asyncio.sleep(0)
    assert supervisor.orphan_count == 0


@pytest.mark.asyncio
async def test_cancelled_error_from_sync_tts_canceller_cannot_escape_cleanup() -> None:
    stream = _UncooperativeStream()
    attempt = VoiceAttempt(
        request=_request(epoch=21),
        gateway=_StreamGateway(stream),
        is_epoch_current=lambda epoch: epoch == 21,
    )
    attempt.start()
    await stream.started.wait()
    fence_states: list[bool] = []

    def misbehaving_canceller() -> None:
        fence_states.append(attempt.signals.accepts_events())
        raise asyncio.CancelledError

    attempt.register_tts_canceller(misbehaving_canceller)

    async def never_exits(_task: asyncio.Task[object], _timeout: float) -> bool:
        await asyncio.sleep(0)
        return False

    supervisor = VoiceDispatchSupervisor()
    manager = AttemptCleanupManager(supervisor, wait_for_exit=never_exits)
    try:
        observer = manager.cancel(attempt)
        assert await observer is AttemptCleanupOutcome.DETACHED
        assert fence_states == [False]
        assert manager.obsolete_cleanup_count == 0
        assert supervisor.orphan_count == 1
    finally:
        stream.release.set()
        await attempt.wait()
        for _ in range(20):
            if supervisor.orphan_count == 0:
                break
            await asyncio.sleep(0)
    assert supervisor.orphan_count == 0


@pytest.mark.asyncio
async def test_reentrant_cancel_from_tts_canceller_reuses_reserved_cleanup() -> None:
    stream = _UncooperativeStream()
    attempt = VoiceAttempt(
        request=_request(epoch=22),
        gateway=_StreamGateway(stream),
        is_epoch_current=lambda epoch: epoch == 22,
    )
    attempt.start()
    await stream.started.wait()

    async def never_exits(_task: asyncio.Task[object], _timeout: float) -> bool:
        await asyncio.sleep(0)
        return False

    conservative: list[int] = []
    detached: list[int] = []
    supervisor = VoiceDispatchSupervisor()
    manager = AttemptCleanupManager(
        supervisor,
        wait_for_exit=never_exits,
        on_conservative=conservative.append,
        on_detached=detached.append,
    )
    nested_observers: list[asyncio.Future[object]] = []

    def cancel_reentrantly() -> asyncio.Future[object]:
        nested = manager.cancel(attempt)
        nested_observers.append(nested)
        return nested

    attempt.register_tts_canceller(cancel_reentrantly)
    try:
        outer = manager.cancel(attempt)

        assert manager.obsolete_cleanup_count == 1
        assert len(nested_observers) == 1
        assert nested_observers[0] is not outer
        assert await asyncio.gather(outer, nested_observers[0]) == [
            AttemptCleanupOutcome.DETACHED,
            AttemptCleanupOutcome.DETACHED,
        ]
        assert conservative == [22]
        assert detached == [22]
        assert stream.force_close_calls == 1
        assert supervisor.orphan_count == 1
    finally:
        stream.release.set()
        await attempt.wait()
        for _ in range(20):
            if supervisor.orphan_count == 0:
                break
            await asyncio.sleep(0)
    assert supervisor.orphan_count == 0


@pytest.mark.asyncio
async def test_async_tts_wrapper_reentrant_cancel_cannot_depend_on_cleanup() -> None:
    class _CooperativeStream:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.force_close_calls = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.started.set()
            await asyncio.Event().wait()
            raise StopAsyncIteration

        async def force_close(self) -> None:
            self.force_close_calls += 1

    stream = _CooperativeStream()
    attempt = VoiceAttempt(
        request=_request(epoch=23),
        gateway=_StreamGateway(stream),  # type: ignore[arg-type]
        is_epoch_current=lambda epoch: epoch == 23,
    )
    attempt.start()
    await stream.started.wait()

    async def observe_cooperative_exit(
        task: asyncio.Task[object], _timeout: float
    ) -> bool:
        for _ in range(20):
            if task.done():
                return True
            await asyncio.sleep(0)
        return task.done()

    supervisor = VoiceDispatchSupervisor()
    manager = AttemptCleanupManager(
        supervisor,
        wait_for_exit=observe_cooperative_exit,
    )
    nested_results: list[object] = []
    nested_started = asyncio.Event()
    nested_acknowledged = asyncio.Event()

    class _ReentrantAwaitable:
        def __await__(self):
            async def run() -> None:
                nested_started.set()
                nested_results.append(await manager.cancel(attempt))
                nested_acknowledged.set()

            return run().__await__()

    attempt.register_tts_canceller(_ReentrantAwaitable)

    observer = manager.cancel(attempt)
    await asyncio.wait_for(nested_started.wait(), timeout=1.0)
    await asyncio.wait_for(nested_acknowledged.wait(), timeout=1.0)
    outcome = await asyncio.wait_for(asyncio.shield(observer), timeout=1.0)

    assert outcome is AttemptCleanupOutcome.CLEAN
    assert nested_results == [AttemptCleanupOutcome.CLEAN]
    assert stream.force_close_calls == 0
    for _ in range(20):
        if supervisor.orphan_count == 0:
            break
        await asyncio.sleep(0)
    assert supervisor.orphan_count == 0
    assert manager.obsolete_cleanup_count == 0


@pytest.mark.asyncio
async def test_spawned_tts_descendants_cannot_inherit_cancellation_authority() -> None:
    class _CooperativeStream:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.force_close_calls = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.started.set()
            await asyncio.Event().wait()
            raise StopAsyncIteration

        async def force_close(self) -> None:
            self.force_close_calls += 1

    stream = _CooperativeStream()
    attempt = VoiceAttempt(
        request=_request(epoch=24),
        gateway=_StreamGateway(stream),  # type: ignore[arg-type]
        is_epoch_current=lambda epoch: epoch == 24,
    )
    attempt.start()
    await stream.started.wait()

    supervisor = VoiceDispatchSupervisor()
    manager = AttemptCleanupManager(supervisor)
    descendant_gate = asyncio.Event()
    provider_hold = asyncio.get_running_loop().create_future()
    assert attempt.signals.register_provider_work(provider_hold, lambda: None)
    wrapper_finished = asyncio.Event()
    descendant_handles_created = asyncio.Event()
    direct_results: list[object] = []
    descendant_done_at_creation: list[bool] = []
    descendant_results: list[object] = []
    descendants: list[asyncio.Task[None]] = []

    async def observe_cleanup_from_descendant() -> None:
        await descendant_gate.wait()
        observer = manager.cancel(attempt)
        descendant_done_at_creation.append(observer.done())
        if len(descendant_done_at_creation) == 2:
            descendant_handles_created.set()
        descendant_results.append(await observer)

    async def spawn_grandchild() -> None:
        await descendant_gate.wait()
        grandchild = asyncio.create_task(observe_cleanup_from_descendant())
        descendants.append(grandchild)
        await grandchild

    class _CancellationWrapper:
        def __await__(self):
            async def run() -> None:
                child = asyncio.create_task(observe_cleanup_from_descendant())
                child_parent = asyncio.create_task(spawn_grandchild())
                descendants.extend((child, child_parent))
                direct_results.append(await manager.cancel(attempt))
                wrapper_finished.set()

            return run().__await__()

    attempt.register_tts_canceller(_CancellationWrapper)
    outer = manager.cancel(attempt)

    for _ in range(20):
        if len(descendants) == 2:
            break
        await asyncio.sleep(0)
    assert len(descendants) == 2
    descendant_gate.set()
    await asyncio.wait_for(descendant_handles_created.wait(), timeout=1.0)

    assert direct_results == []
    assert descendant_done_at_creation == [False, False]
    assert outer.done() is False

    provider_hold.set_result(None)
    assert await asyncio.wait_for(asyncio.shield(outer), timeout=1.0) is (
        AttemptCleanupOutcome.CLEAN
    )
    await asyncio.wait_for(wrapper_finished.wait(), timeout=1.0)
    await asyncio.gather(*descendants)
    assert direct_results == [AttemptCleanupOutcome.CLEAN]
    assert descendant_results == [
        AttemptCleanupOutcome.CLEAN,
        AttemptCleanupOutcome.CLEAN,
    ]
    for _ in range(20):
        if manager.obsolete_cleanup_count == 0:
            break
        await asyncio.sleep(0)
    assert stream.force_close_calls == 0
    for _ in range(20):
        if supervisor.orphan_count == 0:
            break
        await asyncio.sleep(0)
    assert supervisor.orphan_count == 0
    assert manager.obsolete_cleanup_count == 0


@pytest.mark.asyncio
async def test_preexisting_registered_tts_task_uses_exact_task_identity() -> None:
    attempt = VoiceAttempt(
        request=_request(epoch=25),
        gateway=_ScriptedGateway(),
        is_epoch_current=lambda epoch: epoch == 25,
    )
    await attempt.start()
    supervisor = VoiceDispatchSupervisor()
    manager = AttemptCleanupManager(supervisor)
    begin_reentry = asyncio.Event()
    nested_results: list[object] = []

    async def preexisting_work() -> None:
        await begin_reentry.wait()
        nested_results.append(await manager.cancel(attempt))

    preexisting_task = asyncio.create_task(preexisting_work())
    attempt.register_tts_canceller(lambda: preexisting_task)
    outer = manager.cancel(attempt)
    begin_reentry.set()

    assert await asyncio.wait_for(asyncio.shield(outer), timeout=1.0) is (
        AttemptCleanupOutcome.CLEAN
    )
    assert nested_results == [AttemptCleanupOutcome.CLEAN]
    for _ in range(20):
        if supervisor.orphan_count == 0:
            break
        await asyncio.sleep(0)
    assert supervisor.orphan_count == 0
    for _ in range(20):
        if manager.obsolete_cleanup_count == 0:
            break
        await asyncio.sleep(0)
    assert manager.obsolete_cleanup_count == 0


@pytest.mark.asyncio
async def test_structured_tts_wrapper_can_await_child_cleanup_observer() -> None:
    class _CooperativeStream:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.force_close_calls = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.started.set()
            await asyncio.Event().wait()
            raise StopAsyncIteration

        async def force_close(self) -> None:
            self.force_close_calls += 1

    stream = _CooperativeStream()
    attempt = VoiceAttempt(
        request=_request(epoch=28),
        gateway=_StreamGateway(stream),  # type: ignore[arg-type]
        is_epoch_current=lambda epoch: epoch == 28,
    )
    attempt.start()
    await stream.started.wait()

    async def observe_cooperative_exit(
        task: asyncio.Task[object], _timeout: float
    ) -> bool:
        for _ in range(20):
            if task.done():
                return True
            await asyncio.sleep(0)
        return task.done()

    conservative: list[int] = []
    detached: list[int] = []
    supervisor = VoiceDispatchSupervisor()
    manager = AttemptCleanupManager(
        supervisor,
        wait_for_exit=observe_cooperative_exit,
        on_conservative=conservative.append,
        on_detached=detached.append,
    )
    child_results: list[object] = []
    child_done_at_creation: list[bool] = []
    wrapper_finished = asyncio.Event()

    async def child() -> None:
        observer = manager.cancel(attempt)
        child_done_at_creation.append(observer.done())
        child_results.append(await observer)

    class _StructuredWrapper:
        def __await__(self):
            async def run() -> None:
                async with asyncio.TaskGroup() as group:
                    group.create_task(child())
                wrapper_finished.set()

            return run().__await__()

    attempt.register_tts_canceller(_StructuredWrapper)
    outcome = await asyncio.wait_for(
        asyncio.shield(manager.cancel(attempt)),
        timeout=1.0,
    )
    await asyncio.wait_for(wrapper_finished.wait(), timeout=1.0)
    for _ in range(20):
        if manager.obsolete_cleanup_count == 0:
            break
        await asyncio.sleep(0)

    assert outcome is AttemptCleanupOutcome.CLEAN
    assert child_done_at_creation == [False]
    assert child_results == [AttemptCleanupOutcome.CLEAN]
    assert conservative == []
    assert detached == [28]
    assert stream.force_close_calls == 0
    for _ in range(20):
        if supervisor.orphan_count == 0:
            break
        await asyncio.sleep(0)
    assert supervisor.orphan_count == 0
    assert manager.obsolete_cleanup_count == 0


@pytest.mark.asyncio
async def test_preexisting_structured_tts_task_can_await_grandchild_observer() -> None:
    class _CooperativeStream:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.force_close_calls = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.started.set()
            await asyncio.Event().wait()
            raise StopAsyncIteration

        async def force_close(self) -> None:
            self.force_close_calls += 1

    stream = _CooperativeStream()
    attempt = VoiceAttempt(
        request=_request(epoch=29),
        gateway=_StreamGateway(stream),  # type: ignore[arg-type]
        is_epoch_current=lambda epoch: epoch == 29,
    )
    attempt.start()
    await stream.started.wait()

    async def observe_cooperative_exit(
        task: asyncio.Task[object], _timeout: float
    ) -> bool:
        for _ in range(20):
            if task.done():
                return True
            await asyncio.sleep(0)
        return task.done()

    conservative: list[int] = []
    supervisor = VoiceDispatchSupervisor()
    manager = AttemptCleanupManager(
        supervisor,
        wait_for_exit=observe_cooperative_exit,
        on_conservative=conservative.append,
    )
    begin_cancellation = asyncio.Event()
    structured_finished = asyncio.Event()
    grandchild_results: list[object] = []
    grandchild_done_at_creation: list[bool] = []

    async def grandchild() -> None:
        observer = manager.cancel(attempt)
        grandchild_done_at_creation.append(observer.done())
        grandchild_results.append(await observer)

    async def child() -> None:
        async with asyncio.TaskGroup() as group:
            group.create_task(grandchild())

    async def preexisting_work() -> None:
        await begin_cancellation.wait()
        async with asyncio.TaskGroup() as group:
            group.create_task(child())
        structured_finished.set()

    preexisting_task = asyncio.create_task(preexisting_work())
    attempt.register_tts_canceller(lambda: preexisting_task)
    outer = manager.cancel(attempt)
    begin_cancellation.set()
    outcome = await asyncio.wait_for(asyncio.shield(outer), timeout=1.0)
    await asyncio.wait_for(structured_finished.wait(), timeout=1.0)
    for _ in range(20):
        if manager.obsolete_cleanup_count == 0:
            break
        await asyncio.sleep(0)

    assert outcome is AttemptCleanupOutcome.CLEAN
    assert grandchild_done_at_creation == [False]
    assert grandchild_results == [AttemptCleanupOutcome.CLEAN]
    assert conservative == []
    assert stream.force_close_calls == 0
    for _ in range(20):
        if supervisor.orphan_count == 0:
            break
        await asyncio.sleep(0)
    assert supervisor.orphan_count == 0
    assert manager.obsolete_cleanup_count == 0


@pytest.mark.asyncio
async def test_uncooperative_provider_and_structured_tts_share_one_orphan() -> None:
    stream = _UncooperativeStream()
    attempt = VoiceAttempt(
        request=_request(epoch=30),
        gateway=_StreamGateway(stream),
        is_epoch_current=lambda epoch: epoch == 30,
    )
    attempt.start()
    await stream.started.wait()

    async def never_exits(_task: asyncio.Task[object], _timeout: float) -> bool:
        await asyncio.sleep(0)
        return False

    supervisor = VoiceDispatchSupervisor()
    manager = AttemptCleanupManager(supervisor, wait_for_exit=never_exits)
    child_results: list[object] = []
    wrapper_finished = asyncio.Event()

    async def child() -> None:
        child_results.append(await manager.cancel(attempt))

    class _StructuredWrapper:
        def __await__(self):
            async def run() -> None:
                async with asyncio.TaskGroup() as group:
                    group.create_task(child())
                wrapper_finished.set()

            return run().__await__()

    attempt.register_tts_canceller(_StructuredWrapper)
    try:
        assert await manager.cancel(attempt) is AttemptCleanupOutcome.DETACHED
        await asyncio.wait_for(wrapper_finished.wait(), timeout=1.0)
        for _ in range(20):
            if manager.obsolete_cleanup_count == 0:
                break
            await asyncio.sleep(0)

        assert child_results == [AttemptCleanupOutcome.DETACHED]
        assert stream.force_close_calls == 1
        assert supervisor.orphan_count == 1
        assert manager.obsolete_cleanup_count == 0
    finally:
        stream.release.set()
        await attempt.wait()
        for _ in range(20):
            if supervisor.orphan_count == 0:
                break
            await asyncio.sleep(0)
    assert supervisor.orphan_count == 0


@pytest.mark.asyncio
async def test_provider_detach_supervises_owned_tts_in_combined_orphan() -> None:
    stream = _UncooperativeStream()
    attempt = VoiceAttempt(
        request=_request(epoch=32),
        gateway=_StreamGateway(stream),
        is_epoch_current=lambda epoch: epoch == 32,
    )
    attempt.start()
    await stream.started.wait()

    release_tts = asyncio.Event()
    tts_cancelled = asyncio.Event()
    tts_completed = asyncio.Event()

    async def tts_finalizer() -> None:
        try:
            await release_tts.wait()
        except asyncio.CancelledError:
            tts_cancelled.set()
            raise
        tts_completed.set()

    tts_task = asyncio.create_task(tts_finalizer())
    attempt.register_tts_canceller(lambda: tts_task)

    async def never_exits(_task: asyncio.Task[object], _timeout: float) -> bool:
        await asyncio.sleep(0)
        return False

    supervisor = VoiceDispatchSupervisor()
    manager = AttemptCleanupManager(supervisor, wait_for_exit=never_exits)
    try:
        assert await manager.cancel(attempt) is AttemptCleanupOutcome.DETACHED
        assert tts_cancelled.is_set() is False
        assert tts_task.done() is False
        assert supervisor.orphan_count == 1

        release_tts.set()
        await asyncio.wait_for(tts_completed.wait(), timeout=1.0)
        await tts_task
        assert supervisor.orphan_count == 1

        stream.release.set()
        await attempt.wait()
        for _ in range(20):
            if supervisor.orphan_count == 0:
                break
            await asyncio.sleep(0)
        assert supervisor.orphan_count == 0
    finally:
        release_tts.set()
        stream.release.set()
        if not tts_task.done():
            tts_task.cancel()
        await asyncio.gather(attempt.wait(), tts_task, return_exceptions=True)
        for _ in range(20):
            if supervisor.orphan_count == 0:
                break
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_ordinary_duplicate_waits_for_cooperative_tts_quarantine() -> None:
    class _CooperativeStream:
        def __init__(self) -> None:
            self.started = asyncio.Event()

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.started.set()
            await asyncio.Event().wait()
            raise StopAsyncIteration

    stream = _CooperativeStream()
    attempt = VoiceAttempt(
        request=_request(epoch=34),
        gateway=_StreamGateway(stream),  # type: ignore[arg-type]
        is_epoch_current=lambda epoch: epoch == 34,
    )
    attempt.start()
    await stream.started.wait()
    provider_hold = asyncio.get_running_loop().create_future()
    assert attempt.signals.register_provider_work(provider_hold, lambda: None)
    provider_completion = attempt.provider_cleanup_task

    release_tts = asyncio.Event()
    tts_cancelled = asyncio.Event()

    async def resistant_tts() -> None:
        while not release_tts.is_set():
            try:
                await release_tts.wait()
            except asyncio.CancelledError:
                tts_cancelled.set()

    tts_task = asyncio.create_task(resistant_tts())
    attempt.register_tts_canceller(lambda: tts_task)
    allow_provider = asyncio.Event()

    async def controlled_wait(task: asyncio.Task[object], _timeout: float) -> bool:
        if task is provider_completion:
            await allow_provider.wait()
            while not task.done():
                await asyncio.sleep(0)
            return True
        return task.done()

    supervisor = VoiceDispatchSupervisor()
    manager = AttemptCleanupManager(supervisor, wait_for_exit=controlled_wait)
    first = manager.cancel(attempt)
    duplicate = manager.cancel(attempt)
    try:
        await asyncio.sleep(0)
        assert duplicate.done() is False
        assert first.done() is False
        assert supervisor.orphan_count == 0

        provider_hold.set_result(None)
        allow_provider.set()
        assert await asyncio.gather(first, duplicate) == [
            AttemptCleanupOutcome.CLEAN,
            AttemptCleanupOutcome.CLEAN,
        ]
        assert tts_cancelled.is_set() is False
        assert tts_task.done() is False
        assert supervisor.orphan_count == 1
        with pytest.raises(VoiceDispatchQuarantined):
            supervisor.ensure_dispatch_allowed(VoiceDispatchKind.HANDS_FREE)

        release_tts.set()
        await asyncio.wait_for(tts_task, timeout=1.0)
        for _ in range(20):
            if supervisor.orphan_count == 0:
                break
            await asyncio.sleep(0)
        assert supervisor.orphan_count == 0
    finally:
        if not provider_hold.done():
            provider_hold.set_result(None)
        allow_provider.set()
        release_tts.set()
        if not tts_task.done():
            tts_task.cancel()
        await asyncio.gather(attempt.wait(), tts_task, return_exceptions=True)
        for _ in range(20):
            if supervisor.orphan_count == 0:
                break
            await asyncio.sleep(0)
    assert supervisor.orphan_count == 0


@pytest.mark.asyncio
async def test_ordinary_duplicate_waits_for_uncooperative_combined_orphan() -> None:
    stream = _UncooperativeStream()
    attempt = VoiceAttempt(
        request=_request(epoch=35),
        gateway=_StreamGateway(stream),
        is_epoch_current=lambda epoch: epoch == 35,
    )
    attempt.start()
    await stream.started.wait()

    release_tts = asyncio.Event()
    tts_cancelled = asyncio.Event()

    async def resistant_tts() -> None:
        while not release_tts.is_set():
            try:
                await release_tts.wait()
            except asyncio.CancelledError:
                tts_cancelled.set()

    tts_task = asyncio.create_task(resistant_tts())
    attempt.register_tts_canceller(lambda: tts_task)

    async def never_exits(_task: asyncio.Task[object], _timeout: float) -> bool:
        await asyncio.sleep(0)
        return False

    supervisor = VoiceDispatchSupervisor()
    manager = AttemptCleanupManager(supervisor, wait_for_exit=never_exits)
    first = manager.cancel(attempt)
    duplicate = manager.cancel(attempt)
    try:
        assert await duplicate is AttemptCleanupOutcome.DETACHED
        assert first.done() is True
        assert await first is AttemptCleanupOutcome.DETACHED
        assert tts_cancelled.is_set() is False
        assert tts_task.done() is False
        assert supervisor.orphan_count == 1
        with pytest.raises(VoiceDispatchQuarantined):
            supervisor.ensure_dispatch_allowed(VoiceDispatchKind.HANDS_FREE)

        release_tts.set()
        await asyncio.wait_for(tts_task, timeout=1.0)
        assert supervisor.orphan_count == 1

        stream.release.set()
        await attempt.wait()
        for _ in range(20):
            if supervisor.orphan_count == 0:
                break
            await asyncio.sleep(0)
        assert supervisor.orphan_count == 0
    finally:
        stream.release.set()
        release_tts.set()
        if not tts_task.done():
            tts_task.cancel()
        await asyncio.gather(attempt.wait(), tts_task, return_exceptions=True)
        for _ in range(20):
            if supervisor.orphan_count == 0:
                break
            await asyncio.sleep(0)
    assert supervisor.orphan_count == 0


@pytest.mark.asyncio
async def test_cleanup_failure_settles_all_shared_outcome_observers() -> None:
    stream = _UncooperativeStream()
    attempt = VoiceAttempt(
        request=_request(epoch=33),
        gateway=_StreamGateway(stream),
        is_epoch_current=lambda epoch: epoch == 33,
    )
    attempt.start()
    await stream.started.wait()

    async def fail_wait(_task: asyncio.Task[object], _timeout: float) -> bool:
        raise RuntimeError("content-free cleanup failure")

    manager = AttemptCleanupManager(
        VoiceDispatchSupervisor(),
        wait_for_exit=fail_wait,
    )
    first = manager.cancel(attempt)
    dependent = manager.cancel(attempt)
    try:
        results = await asyncio.wait_for(
            asyncio.gather(first, dependent, return_exceptions=True),
            timeout=0.2,
        )
        assert [type(result) for result in results] == [RuntimeError, RuntimeError]
    finally:
        stream.release.set()
        await attempt.wait()


@pytest.mark.asyncio
async def test_unused_shared_outcome_failure_never_reaches_exception_handler() -> None:
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    reports: list[dict[str, Any]] = []
    loop.set_exception_handler(lambda _loop, context: reports.append(context))
    stream = _UncooperativeStream()
    attempt = VoiceAttempt(
        request=_request(epoch=36),
        gateway=_StreamGateway(stream),
        is_epoch_current=lambda epoch: epoch == 36,
    )
    attempt.start()
    await stream.started.wait()

    async def fail_wait(_task: asyncio.Task[object], _timeout: float) -> bool:
        raise RuntimeError("content-free cleanup failure")

    manager = AttemptCleanupManager(
        VoiceDispatchSupervisor(),
        wait_for_exit=fail_wait,
    )
    try:
        with pytest.raises(RuntimeError, match="content-free cleanup failure"):
            await manager.cancel(attempt)
        stream.release.set()
        await attempt.wait()
        del manager
        del attempt
        gc.collect()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert [
            report
            for report in reports
            if report.get("message") == "Future exception was never retrieved"
        ] == []
    finally:
        stream.release.set()
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
async def test_cancelled_public_observer_consumes_later_cleanup_failure() -> None:
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    reports: list[dict[str, Any]] = []
    loop.set_exception_handler(lambda _loop, context: reports.append(context))
    stream = _UncooperativeStream()
    attempt = VoiceAttempt(
        request=_request(epoch=37),
        gateway=_StreamGateway(stream),
        is_epoch_current=lambda epoch: epoch == 37,
    )
    attempt.start()
    await stream.started.wait()
    fail_cleanup = asyncio.Event()

    async def fail_after_cancel(_task: asyncio.Task[object], _timeout: float) -> bool:
        await fail_cleanup.wait()
        raise RuntimeError("content-free cleanup failure")

    manager = AttemptCleanupManager(
        VoiceDispatchSupervisor(),
        wait_for_exit=fail_after_cancel,
    )
    observer = manager.cancel(attempt)
    observer.cancel()
    try:
        fail_cleanup.set()
        for _ in range(20):
            if manager.obsolete_cleanup_count == 0:
                break
            await asyncio.sleep(0)
        assert manager.obsolete_cleanup_count == 0
        stream.release.set()
        await attempt.wait()
        del manager
        del attempt
        del observer
        gc.collect()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert [
            report
            for report in reports
            if report.get("message") == "Future exception was never retrieved"
        ] == []
    finally:
        fail_cleanup.set()
        stream.release.set()
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
async def test_stuck_tts_finalizer_is_sanitized_and_strongly_quarantined() -> None:
    class _Coordinator:
        def __init__(self) -> None:
            self.private_transcript = "private transcript body"

        def is_epoch_current(self, epoch: int) -> bool:
            return epoch == 31

    class _CooperativeStream:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.force_close_calls = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.started.set()
            await asyncio.Event().wait()
            raise StopAsyncIteration

        async def force_close(self) -> None:
            self.force_close_calls += 1

    coordinator = _Coordinator()
    coordinator_ref = weakref.ref(coordinator)
    stream = _CooperativeStream()
    attempt = VoiceAttempt(
        request=_request(epoch=31),
        gateway=_StreamGateway(stream),  # type: ignore[arg-type]
        is_epoch_current=coordinator.is_epoch_current,
    )
    attempt.start()
    await stream.started.wait()
    release_tts = asyncio.Event()

    async def cancellation_resistant_tts() -> None:
        while not release_tts.is_set():
            try:
                await release_tts.wait()
            except asyncio.CancelledError:
                continue

    resistant_tasks = [asyncio.create_task(cancellation_resistant_tts())]
    attempt.register_tts_canceller(resistant_tasks.pop)

    async def observe_exit(task: asyncio.Task[object], _timeout: float) -> bool:
        for _ in range(20):
            if task.done():
                return True
            await asyncio.sleep(0)
        return task.done()

    detached: list[int] = []
    supervisor = VoiceDispatchSupervisor()
    manager = AttemptCleanupManager(
        supervisor,
        wait_for_exit=observe_exit,
        on_detached=detached.append,
    )
    assert await manager.cancel(attempt) is AttemptCleanupOutcome.CLEAN
    try:
        for _ in range(100):
            if supervisor.orphan_count == 1 and manager.obsolete_cleanup_count == 0:
                break
            await asyncio.sleep(0)

        assert detached == [31]
        assert stream.force_close_calls == 0
        assert supervisor.orphan_count == 1
        assert manager.obsolete_cleanup_count == 0

        del attempt
        del manager
        del coordinator
        gc.collect()
        assert coordinator_ref() is None
        assert supervisor.orphan_count == 1
    finally:
        release_tts.set()
        for _ in range(100):
            if supervisor.orphan_count == 0:
                break
            await asyncio.sleep(0)
    assert supervisor.orphan_count == 0


@pytest.mark.asyncio
async def test_repeated_cancel_reuses_cleanup_force_close_and_orphan_slot() -> None:
    stream = _UncooperativeStream()
    attempt = VoiceAttempt(
        request=_request(epoch=12),
        gateway=_StreamGateway(stream),
        is_epoch_current=lambda epoch: epoch == 12,
    )
    attempt.start()
    await stream.started.wait()
    allow_escalation = asyncio.Event()

    async def blocked_wait(_task: asyncio.Task[object], _timeout: float) -> bool:
        await allow_escalation.wait()
        return False

    supervisor = VoiceDispatchSupervisor()
    manager = AttemptCleanupManager(supervisor, wait_for_exit=blocked_wait)
    first = manager.cancel(attempt)
    second = manager.cancel(attempt)
    try:
        assert manager.obsolete_cleanup_count == 1
        allow_escalation.set()
        assert await asyncio.gather(first, second) == [
            AttemptCleanupOutcome.DETACHED,
            AttemptCleanupOutcome.DETACHED,
        ]
        assert stream.force_close_calls == 1
        assert supervisor.orphan_count == 1
    finally:
        allow_escalation.set()
        stream.release.set()
        await attempt.wait()
        for task in (first, second):
            if not task.done():
                task.cancel()
        await asyncio.gather(first, second, return_exceptions=True)
        for _ in range(20):
            if supervisor.orphan_count == 0:
                break
            await asyncio.sleep(0)
    assert supervisor.orphan_count == 0


@pytest.mark.asyncio
async def test_cleanup_deadlines_are_absolute_from_synchronous_cancel() -> None:
    stream = _UncooperativeStream()
    attempt = VoiceAttempt(
        request=_request(epoch=13),
        gateway=_StreamGateway(stream),
        is_epoch_current=lambda epoch: epoch == 13,
    )
    attempt.start()
    await stream.started.wait()
    waits: list[float] = []

    async def record_wait(_task: asyncio.Task[object], timeout: float) -> bool:
        waits.append(timeout)
        await asyncio.sleep(0)
        return False

    supervisor = VoiceDispatchSupervisor()
    manager = AttemptCleanupManager(supervisor, wait_for_exit=record_wait)
    observer = manager.cancel(attempt)
    try:
        time.sleep(0.08)
        assert await observer is AttemptCleanupOutcome.DETACHED

        assert 1.80 < waits[0] < 1.99
        assert 4.80 < waits[1] < 4.99
        assert 5.30 < waits[2] < 5.49
    finally:
        stream.release.set()
        await attempt.wait()
        if not observer.done():
            observer.cancel()
        await asyncio.gather(observer, return_exceptions=True)
        for _ in range(20):
            if supervisor.orphan_count == 0:
                break
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_completion_winning_detach_boundary_is_not_falsely_quarantined() -> None:
    stream = _UncooperativeStream()
    attempt = VoiceAttempt(
        request=_request(epoch=14),
        gateway=_StreamGateway(stream),
        is_epoch_current=lambda epoch: epoch == 14,
    )
    attempt.start()
    await stream.started.wait()
    call_count = 0

    async def boundary_race(task: asyncio.Task[object], _timeout: float) -> bool:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return False
        stream.release.set()
        while not task.done():
            await asyncio.sleep(0)
        return False

    supervisor = VoiceDispatchSupervisor()
    manager = AttemptCleanupManager(supervisor, wait_for_exit=boundary_race)

    assert await manager.cancel(attempt) is AttemptCleanupOutcome.FORCE_CLOSED
    assert supervisor.orphan_count == 0
    assert stream.force_close_calls == 1


@pytest.mark.asyncio
async def test_two_detached_cleanups_recover_only_as_each_runner_exits() -> None:
    streams = [_UncooperativeStream(), _UncooperativeStream()]
    attempts = [
        VoiceAttempt(
            request=_request(epoch=epoch),
            gateway=_StreamGateway(stream),
            is_epoch_current=lambda _epoch: True,
        )
        for epoch, stream in enumerate(streams, start=1)
    ]
    for attempt in attempts:
        attempt.start()
    await asyncio.gather(*(stream.started.wait() for stream in streams))

    async def never_exits(_task: asyncio.Task[object], _timeout: float) -> bool:
        await asyncio.sleep(0)
        return False

    supervisor = VoiceDispatchSupervisor()
    manager = AttemptCleanupManager(supervisor, wait_for_exit=never_exits)

    outcomes = await asyncio.gather(*(manager.cancel(attempt) for attempt in attempts))

    assert outcomes == [
        AttemptCleanupOutcome.DETACHED,
        AttemptCleanupOutcome.DETACHED,
    ]
    assert supervisor.orphan_count == 2
    streams[0].release.set()
    await attempts[0].wait()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert supervisor.orphan_count == 1
    streams[1].release.set()
    await attempts[1].wait()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert supervisor.orphan_count == 0


@pytest.mark.asyncio
async def test_detached_outcome_unblocks_sequential_tts_descendants() -> None:
    stream = _UncooperativeStream()
    attempt = VoiceAttempt(
        request=_request(epoch=49),
        gateway=_StreamGateway(stream),
        is_epoch_current=lambda epoch: epoch == 49,
    )
    attempt.start()
    await stream.started.wait()

    async def never_exits(_task: asyncio.Task[object], _timeout: float) -> bool:
        await asyncio.sleep(0)
        return False

    supervisor = VoiceDispatchSupervisor()
    manager = AttemptCleanupManager(supervisor, wait_for_exit=never_exits)
    descendant_results: list[object] = []
    wrapper_finished = asyncio.Event()

    async def observe_cleanup() -> object:
        return await manager.cancel(attempt)

    async def sequential_wrapper() -> None:
        descendant_results.append(await asyncio.create_task(observe_cleanup()))
        descendant_results.append(await asyncio.create_task(observe_cleanup()))
        wrapper_finished.set()

    attempt.register_tts_canceller(sequential_wrapper)
    try:
        assert await manager.cancel(attempt) is AttemptCleanupOutcome.DETACHED
        await asyncio.wait_for(wrapper_finished.wait(), timeout=1.0)

        assert descendant_results == [
            AttemptCleanupOutcome.DETACHED,
            AttemptCleanupOutcome.DETACHED,
        ]
        assert supervisor.orphan_count == 1
    finally:
        stream.release.set()
        await attempt.wait()
        for _ in range(50):
            if supervisor.orphan_count == 0:
                break
            await asyncio.sleep(0)
    assert supervisor.orphan_count == 0


@pytest.mark.asyncio
async def test_same_tts_task_can_be_owned_by_two_attempt_cleanups() -> None:
    attempts = [
        VoiceAttempt(
            request=_request(epoch=epoch),
            gateway=_ScriptedGateway(),
            is_epoch_current=lambda _epoch: True,
        )
        for epoch in (50, 51)
    ]
    await asyncio.gather(*(attempt.start() for attempt in attempts))

    async def observe_exit(task: asyncio.Task[object], _timeout: float) -> bool:
        for _ in range(20):
            if task.done():
                return True
            await asyncio.sleep(0)
        return task.done()

    supervisor = VoiceDispatchSupervisor()
    manager = AttemptCleanupManager(supervisor, wait_for_exit=observe_exit)
    begin = asyncio.Event()
    shared_finished = asyncio.Event()
    descendant_results: list[object] = []

    async def observe_cleanup(attempt: VoiceAttempt) -> object:
        return await manager.cancel(attempt)

    async def shared_tts_work() -> None:
        await begin.wait()
        descendant_results.extend(
            await asyncio.gather(
                *(asyncio.create_task(observe_cleanup(attempt)) for attempt in attempts)
            )
        )
        shared_finished.set()

    shared_task = asyncio.create_task(shared_tts_work())
    for attempt in attempts:
        attempt.register_tts_canceller(lambda: shared_task)

    first = manager.cancel(attempts[0])
    second = manager.cancel(attempts[1])
    begin.set()
    try:
        assert await asyncio.gather(first, second) == [
            AttemptCleanupOutcome.CLEAN,
            AttemptCleanupOutcome.CLEAN,
        ]
        await asyncio.wait_for(shared_finished.wait(), timeout=1.0)

        assert descendant_results == [
            AttemptCleanupOutcome.CLEAN,
            AttemptCleanupOutcome.CLEAN,
        ]
        for _ in range(50):
            if supervisor.orphan_count == 0:
                break
            await asyncio.sleep(0)
        assert supervisor.orphan_count == 0
    finally:
        if not shared_task.done():
            shared_task.cancel()
        await asyncio.gather(shared_task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_seconds", [0.02, 0.5])
async def test_cooperative_tts_cleanup_is_supervised_without_cancellation(
    cleanup_seconds: float,
) -> None:
    attempt = VoiceAttempt(
        request=_request(epoch=52),
        gateway=_ScriptedGateway(),
        is_epoch_current=lambda epoch: epoch == 52,
    )
    await attempt.start()

    cancelled = asyncio.Event()
    completed = asyncio.Event()
    finished = asyncio.Event()

    async def cooperative_tts_cleanup() -> None:
        try:
            await asyncio.sleep(cleanup_seconds)
            completed.set()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        finally:
            finished.set()

    attempt.register_tts_canceller(cooperative_tts_cleanup)
    supervisor = VoiceDispatchSupervisor()
    manager = AttemptCleanupManager(supervisor)

    assert await manager.cancel(attempt) is AttemptCleanupOutcome.CLEAN
    assert finished.is_set() is False
    assert supervisor.orphan_count == 1
    with pytest.raises(VoiceDispatchQuarantined):
        supervisor.ensure_dispatch_allowed(VoiceDispatchKind.HANDS_FREE)

    await asyncio.wait_for(finished.wait(), timeout=1.0)
    for _ in range(20):
        if supervisor.orphan_count == 0:
            break
        await asyncio.sleep(0)

    assert completed.is_set()
    assert cancelled.is_set() is False
    assert supervisor.orphan_count == 0


@pytest.mark.asyncio
async def test_staggered_attempts_do_not_cancel_shared_tts_cleanup() -> None:
    attempts = [
        VoiceAttempt(
            request=_request(epoch=epoch),
            gateway=_ScriptedGateway(),
            is_epoch_current=lambda _epoch: True,
        )
        for epoch in (53, 54)
    ]
    await asyncio.gather(*(attempt.start() for attempt in attempts))

    provider_holds = [asyncio.get_running_loop().create_future() for _ in attempts]
    for attempt, provider_hold in zip(attempts, provider_holds):
        assert attempt.signals.register_provider_work(provider_hold, lambda: None)
    provider_completions = [attempt.provider_cleanup_task for attempt in attempts]
    provider_releases = [asyncio.Event(), asyncio.Event()]

    async def staggered_wait(task: asyncio.Task[object], _timeout: float) -> bool:
        index = provider_completions.index(task)
        await provider_releases[index].wait()
        while not task.done():
            await asyncio.sleep(0)
        return True

    supervisor = VoiceDispatchSupervisor()
    manager = AttemptCleanupManager(supervisor, wait_for_exit=staggered_wait)
    begin = asyncio.Event()
    release_shared_work = asyncio.Event()
    shared_observed_outcomes = asyncio.Event()
    shared_completed = asyncio.Event()
    shared_cancelled = asyncio.Event()
    shared_results: list[AttemptCleanupOutcome] = []

    async def shared_tts_cleanup() -> None:
        try:
            await begin.wait()
            shared_results.extend(
                await asyncio.gather(*(manager.cancel(a) for a in attempts))
            )
            shared_observed_outcomes.set()
            await release_shared_work.wait()
            shared_completed.set()
        except asyncio.CancelledError:
            shared_cancelled.set()
            raise

    shared_task = asyncio.create_task(shared_tts_cleanup())
    for attempt in attempts:
        attempt.register_tts_canceller(lambda: shared_task)

    observers = [manager.cancel(attempt) for attempt in attempts]
    begin.set()
    try:
        provider_holds[0].set_result(None)
        provider_releases[0].set()
        assert await observers[0] is AttemptCleanupOutcome.CLEAN
        for _ in range(20):
            await asyncio.sleep(0)

        assert shared_task.done() is False
        assert shared_cancelled.is_set() is False
        assert supervisor.orphan_count == 1

        provider_holds[1].set_result(None)
        provider_releases[1].set()
        assert await observers[1] is AttemptCleanupOutcome.CLEAN
        await asyncio.wait_for(shared_observed_outcomes.wait(), timeout=1.0)
        assert shared_results == [
            AttemptCleanupOutcome.CLEAN,
            AttemptCleanupOutcome.CLEAN,
        ]
        assert shared_task.done() is False
        assert supervisor.orphan_count == 2

        release_shared_work.set()
        await asyncio.wait_for(shared_completed.wait(), timeout=1.0)
        await shared_task
        for _ in range(20):
            if supervisor.orphan_count == 0:
                break
            await asyncio.sleep(0)
        assert supervisor.orphan_count == 0
    finally:
        for provider_hold in provider_holds:
            if not provider_hold.done():
                provider_hold.set_result(None)
        for provider_release in provider_releases:
            provider_release.set()
        release_shared_work.set()
        if not shared_task.done():
            shared_task.cancel()
        await asyncio.gather(shared_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_tts_cleanup_never_interposes_the_loop_task_factory() -> None:
    attempt = VoiceAttempt(
        request=_request(epoch=52),
        gateway=_ScriptedGateway(),
        is_epoch_current=lambda epoch: epoch == 52,
    )
    await attempt.start()
    provider_hold = asyncio.get_running_loop().create_future()
    assert attempt.signals.register_provider_work(provider_hold, lambda: None)
    provider_completion = attempt.provider_cleanup_task

    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    factory_calls: list[dict[str, object]] = []

    def application_factory(
        task_loop: asyncio.AbstractEventLoop,
        coroutine: Any,
        **kwargs: object,
    ) -> asyncio.Task[Any]:
        factory_calls.append(dict(kwargs))
        return asyncio.Task(coroutine, loop=task_loop, **kwargs)

    loop.set_task_factory(application_factory)
    root_started = asyncio.Event()
    release_root = asyncio.Event()
    allow_provider = asyncio.Event()

    async def tts_cleanup() -> None:
        root_started.set()
        await release_root.wait()

    async def wait_for_provider(task: asyncio.Task[object], _timeout: float) -> bool:
        if task is provider_completion:
            await allow_provider.wait()
            while not task.done():
                await asyncio.sleep(0)
            return True
        return task.done()

    manager = AttemptCleanupManager(
        VoiceDispatchSupervisor(),
        wait_for_exit=wait_for_provider,
    )
    cleanup: asyncio.Future[object] | None = None
    try:
        attempt.register_tts_canceller(tts_cleanup)
        cleanup = manager.cancel(attempt)
        await asyncio.wait_for(root_started.wait(), timeout=1.0)

        assert loop.get_task_factory() is application_factory
        task_kwargs: dict[str, object] = {"name": "unrelated-background"}
        if sys.version_info >= (3, 14):
            task_kwargs["eager_start"] = False
        unrelated = loop.create_task(
            asyncio.sleep(0, result="unrelated"),
            **task_kwargs,
        )
        assert await unrelated == "unrelated"
        assert unrelated.get_name() == "unrelated-background"
        assert factory_calls
    finally:
        release_root.set()
        if not provider_hold.done():
            provider_hold.set_result(None)
        allow_provider.set()
        if cleanup is not None:
            await asyncio.gather(cleanup, return_exceptions=True)
        loop.set_task_factory(previous_factory)


@pytest.mark.asyncio
async def test_external_factory_replacement_cannot_break_structured_tts_cleanup() -> (
    None
):
    attempt = VoiceAttempt(
        request=_request(epoch=53),
        gateway=_ScriptedGateway(),
        is_epoch_current=lambda epoch: epoch == 53,
    )
    await attempt.start()
    provider_hold = asyncio.get_running_loop().create_future()
    assert attempt.signals.register_provider_work(provider_hold, lambda: None)
    provider_completion = attempt.provider_cleanup_task

    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()

    def first_factory(
        task_loop: asyncio.AbstractEventLoop,
        coroutine: Any,
        **kwargs: object,
    ) -> asyncio.Task[Any]:
        return asyncio.Task(coroutine, loop=task_loop, **kwargs)

    def replacement_factory(
        task_loop: asyncio.AbstractEventLoop,
        coroutine: Any,
        **kwargs: object,
    ) -> asyncio.Task[Any]:
        return asyncio.Task(coroutine, loop=task_loop, **kwargs)

    loop.set_task_factory(first_factory)
    root_started = asyncio.Event()
    create_descendant = asyncio.Event()
    allow_provider = asyncio.Event()
    wrapper_finished = asyncio.Event()
    descendant_results: list[object] = []

    async def descendant() -> None:
        descendant_results.append(await manager.cancel(attempt))

    async def tts_cleanup() -> None:
        root_started.set()
        await create_descendant.wait()
        async with asyncio.TaskGroup() as group:
            group.create_task(descendant())
        wrapper_finished.set()

    async def wait_for_provider(task: asyncio.Task[object], _timeout: float) -> bool:
        if task is provider_completion:
            await allow_provider.wait()
            while not task.done():
                await asyncio.sleep(0)
            return True
        return task.done()

    supervisor = VoiceDispatchSupervisor()
    manager = AttemptCleanupManager(supervisor, wait_for_exit=wait_for_provider)
    cleanup: asyncio.Future[object] | None = None
    try:
        attempt.register_tts_canceller(tts_cleanup)
        cleanup = manager.cancel(attempt)
        await asyncio.wait_for(root_started.wait(), timeout=1.0)
        loop.set_task_factory(replacement_factory)
        create_descendant.set()
        await asyncio.sleep(0)
        provider_hold.set_result(None)
        allow_provider.set()

        assert await cleanup is AttemptCleanupOutcome.CLEAN
        await asyncio.wait_for(wrapper_finished.wait(), timeout=1.0)
        assert descendant_results == [AttemptCleanupOutcome.CLEAN]
        assert loop.get_task_factory() is replacement_factory
        for _ in range(20):
            if supervisor.orphan_count == 0:
                break
            await asyncio.sleep(0)
        assert supervisor.orphan_count == 0
    finally:
        create_descendant.set()
        if not provider_hold.done():
            provider_hold.set_result(None)
        allow_provider.set()
        if cleanup is not None and not cleanup.done():
            cleanup.cancel()
        if cleanup is not None:
            await asyncio.gather(cleanup, return_exceptions=True)
        loop.set_task_factory(previous_factory)


@pytest.mark.asyncio
@pytest.mark.parametrize("separate_managers", [False, True])
async def test_obsolete_cleanup_count_is_capped_at_two(separate_managers) -> None:
    gate = asyncio.Event()
    streams = [_UncooperativeStream() for _ in range(3)]
    attempts = [
        VoiceAttempt(
            request=_request(epoch=index),
            gateway=_StreamGateway(stream),
            is_epoch_current=lambda _epoch: True,
        )
        for index, stream in enumerate(streams, start=1)
    ]
    for attempt in attempts:
        attempt.start()
    await asyncio.gather(*(stream.started.wait() for stream in streams))

    async def blocked_wait(_task: asyncio.Task[object], _timeout: float) -> bool:
        await gate.wait()
        return False

    supervisor = VoiceDispatchSupervisor()
    manager = AttemptCleanupManager(supervisor, wait_for_exit=blocked_wait)
    managers = [
        AttemptCleanupManager(supervisor, wait_for_exit=blocked_wait)
        if separate_managers
        else manager
        for _ in attempts
    ]
    first = managers[0].cancel(attempts[0])
    second = managers[1].cancel(attempts[1])
    await asyncio.sleep(0)

    try:
        assert sum(owner.obsolete_cleanup_count for owner in set(managers)) == 2
        with pytest.raises(VoiceCleanupCapacityExceeded):
            managers[2].cancel(attempts[2])
    finally:
        for stream in streams:
            stream.release.set()
        gate.set()
        await asyncio.gather(first, second)
        await asyncio.gather(*(attempt.wait() for attempt in attempts))
