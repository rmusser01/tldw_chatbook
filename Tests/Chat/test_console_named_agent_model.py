"""Console adapter contracts for named-agent model overrides."""

import asyncio
import threading

import tldw_chatbook.Chat.console_agent_bridge as bridge_module
from tldw_chatbook.Agents.agent_service import SUBAGENT_SYSTEM_PROMPT
from tldw_chatbook.Chat.console_agent_bridge import _StreamingModelAdapter
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_prepared_request import build_console_request
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
    ProviderToolCalls,
    ProviderTurnMetadata,
)
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy, new_opaque_id
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestRoute,
    ConsoleTraceCaptureMode,
    ProviderArtifactTraceProvenance,
    TraceProvenanceSource,
    request_route_provenance,
)
from tldw_chatbook.Chat.provider_continuation import ContinuationRestoreTarget


class _ConcurrentRecordingGateway:
    def __init__(self) -> None:
        self._arrived = asyncio.Event()
        self._arrival_count = 0
        self._lock = threading.Lock()
        self.prepared: list[tuple[ConsoleProviderResolution, tuple[object, ...]]] = []
        self.dispatched: list[tuple[ConsoleProviderResolution, object]] = []

    def prepare_chat_request(
        self,
        resolution,
        messages,
        *,
        continuation_sidecar,
        **_kwargs,
    ):
        with self._lock:
            self.prepared.append((resolution, tuple(continuation_sidecar)))
        return {"prepared_for": resolution.model, "messages": messages}

    async def stream_chat(self, resolution, messages, **_kwargs):
        with self._lock:
            self.dispatched.append((resolution, messages))
            self._arrival_count += 1
            if self._arrival_count == 2:
                self._arrived.set()
        await asyncio.wait_for(self._arrived.wait(), timeout=5.0)
        yield ProviderToolCalls(
            (),
            metadata=ProviderTurnMetadata(
                finish_reason="stop",
                usage={"prompt_tokens": 2, "completion_tokens": 1},
            ),
        )


def test_concurrent_worker_override_is_call_local_and_keeps_parent_continuation(
    monkeypatch,
):
    loop = asyncio.new_event_loop()
    driver = threading.Thread(target=loop.run_forever, daemon=True)
    driver.start()
    gateway = _ConcurrentRecordingGateway()
    parent_resolution = ConsoleProviderResolution(
        provider="OpenAI",
        base_url="https://example.invalid/v1",
        model="parent-model",
        ready=True,
        readiness_key="openai",
        execution_key="openai",
        continuation_protocol="responses",
    )
    store = ConsoleChatStore()
    session = store.ensure_session()
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    adapter = _StreamingModelAdapter(
        store=store,
        provider_gateway=gateway,
        resolution=parent_resolution,
        assistant_message_id=assistant.id,
        should_cancel=lambda: False,
        loop=loop,
        native_tools=False,
        continuation_sidecar=("parent-sidecar",),
        continuation_target=ContinuationRestoreTarget(
            provider="openai",
            model="parent-model",
            protocol="responses",
            api_base_url="https://example.invalid/v1",
        ),
        continuation_owner_key="_owner",
    )
    usage_identities: list[tuple[str, str]] = []
    real_usage = bridge_module._openai_usage_from_provider_call

    def recording_usage(payload, *, provider, model):
        with gateway._lock:
            usage_identities.append((provider, model))
        return real_usage(payload, provider=provider, model=model)

    monkeypatch.setattr(
        bridge_module, "_openai_usage_from_provider_call", recording_usage
    )
    responses: dict[str, dict] = {}
    failures: list[BaseException] = []

    def call(label: str, model: str, system: str) -> None:
        try:
            responses[label] = adapter.chat_call(
                messages_payload=[{"role": "system", "content": system}],
                model=model,
                api_endpoint="openai",
            )
        except Exception as exc:  # noqa: BLE001 - assert on the main thread
            failures.append(exc)

    parent = threading.Thread(
        target=call, args=("parent", "parent-model", "Primary system prompt")
    )
    child = threading.Thread(
        target=call,
        args=("child", "budget-reader-model", SUBAGENT_SYSTEM_PROMPT),
    )
    try:
        parent.start()
        child.start()
        parent.join(timeout=10)
        child.join(timeout=10)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        driver.join(timeout=5)
        loop.close()

    assert not parent.is_alive() and not child.is_alive()
    assert failures == []
    assert set(responses) == {"parent", "child"}
    by_model = {
        resolution.model: (resolution, messages)
        for resolution, messages in gateway.dispatched
    }
    assert set(by_model) == {"parent-model", "budget-reader-model"}
    worker_resolution = by_model["budget-reader-model"][0]
    assert worker_resolution.provider == parent_resolution.provider
    assert worker_resolution.base_url == parent_resolution.base_url
    assert adapter._resolution is parent_resolution
    assert gateway.prepared == [(parent_resolution, ("parent-sidecar",))]
    assert by_model["parent-model"][1]["prepared_for"] == "parent-model"
    assert isinstance(by_model["budget-reader-model"][1], list)
    assert sorted(usage_identities) == [
        ("OpenAI", "budget-reader-model"),
        ("OpenAI", "parent-model"),
    ]
    assert responses["parent"]["usage"]["total_tokens"] == 3
    assert responses["child"]["usage"]["total_tokens"] == 3


def test_capture_on_worker_uses_its_model_and_keeps_parent_private_history(monkeypatch):
    async def exercise():
        resolution = ConsoleProviderResolution(
            provider="OpenAI",
            base_url="https://example.invalid/v1",
            model="parent-model",
            ready=True,
            readiness_key="openai",
            execution_key="openai",
            continuation_protocol="responses",
        )
        target = ContinuationRestoreTarget(
            provider="openai",
            model="parent-model",
            protocol="responses",
            api_base_url=resolution.base_url,
        )
        policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
        messages = [
            {"role": "system", "content": SUBAGENT_SYSTEM_PROMPT},
            {"role": "user", "content": "Read the supplied synthetic source."},
        ]
        admitted = build_console_request(
            messages,
            message_provenance=(
                ProviderArtifactTraceProvenance(
                    TraceProvenanceSource.RENDERED_SYSTEM, policy
                ),
                ProviderArtifactTraceProvenance(
                    TraceProvenanceSource.ACTIVE_REQUEST, policy
                ),
            ),
            memory_provenance=(),
            mandatory_provenance=(),
            tool_provenance=(),
            metadata_provenance=(request_route_provenance(ConsoleRequestRoute.FRESH),),
            capture_policy=policy,
            capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
        )
        store = ConsoleChatStore()
        session = store.ensure_session()
        assistant = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content=""
        )
        gateway = ConsoleProviderGateway(environ={})
        prepared_options = []
        dispatched = []
        prepare = gateway.prepare_chat_request

        def record_prepare(call_resolution, request, **kwargs):
            prepared_options.append(kwargs)
            return prepare(call_resolution, request, **kwargs)

        async def record_stream(call_resolution, request, **kwargs):
            dispatched.append((call_resolution, request, kwargs))
            yield "Worker evidence."

        monkeypatch.setattr(gateway, "prepare_chat_request", record_prepare)
        monkeypatch.setattr(gateway, "stream_chat", record_stream)
        adapter = _StreamingModelAdapter(
            store=store,
            provider_gateway=gateway,
            resolution=resolution,
            assistant_message_id=assistant.id,
            should_cancel=lambda: False,
            loop=asyncio.get_running_loop(),
            native_tools=False,
            continuation_target=target,
            continuation_sidecar=("parent-continuation",),
            continuation_owner_key="_owner",
            thinking_sidecar=("parent-thinking",),
            thinking_owner_key="_thinking_owner",
            capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
            trace_request=admitted,
        )
        try:
            response = await asyncio.to_thread(
                adapter.chat_call,
                messages_payload=messages,
                model="budget-reader-model",
            )
        finally:
            await gateway.aclose()
        assert response["choices"][0]["message"]["content"] == "Worker evidence."
        assert len(dispatched) == 1
        call_resolution, request, options = dispatched[0]
        assert call_resolution.model == request.model == "budget-reader-model"
        assert options["capture_mode"] is ConsoleTraceCaptureMode.CAPTURE_ON
        assert options["route"] is ConsoleRequestRoute.AGENT_FIRST
        assert prepared_options[0]["thinking_sidecar"] == ()
        assert not prepared_options[0].get("continuation_sidecar")
        assert prepared_options[0]["continuation_target"].model == "budget-reader-model"
        assert adapter._resolution is resolution
        assert adapter._continuation_target is target
        assert target.model == "parent-model"

    asyncio.run(exercise())
