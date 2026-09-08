"""Trace privacy projections must never replace provider-visible context."""

import json
from copy import deepcopy

import pytest

from Tests.Chat import test_console_trace_runtime as runtime_fixtures
from Tests.Chat.test_console_trace_runtime import (
    _saved_message,
    _semantic_request,
)
from tldw_chatbook.Chat.console_prepared_request import thaw_json
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderResolution,
    ConsoleProviderStreamSignals,
)
from tldw_chatbook.Chat.console_trace_errors import TraceCallPersistenceError
from tldw_chatbook.Chat.console_trace_models import (
    FrozenTracePolicy,
    TraceCallState,
    new_opaque_id,
)
from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestRoute,
    ConsoleTraceCaptureMode,
    ProviderArtifactTraceProvenance,
    TraceProvenanceSource,
)
from tldw_chatbook.Chat.console_trace_redaction import BUILTIN_PII_RULESET_REVISION_ID
from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory

make_database = runtime_fixtures.make_database
make_gateway = runtime_fixtures.make_gateway


@pytest.mark.parametrize(
    "cold_factory, pii_enabled, retained_tamper, owned_retry",
    [
        (False, False, None, False),
        (True, False, None, False),
        (False, True, None, False),
        (True, True, None, False),
        (False, True, "context", False),
        (False, True, "saved", False),
        (False, True, "inactive", False),
        (False, False, None, True),
        (True, False, None, True),
        (False, True, None, True),
        (True, True, None, True),
    ],
)
async def test_generic_tool_request_retains_raw_project_context(
    tmp_path,
    make_database,
    make_gateway,
    monkeypatch,
    cold_factory,
    pii_enabled,
    retained_tamper,
    owned_retry,
):
    database = make_database(tmp_path / "provider-input.sqlite", "provider-input")
    conversation_id = database.add_conversation({"title": "provider input"})
    user_id, user = _saved_message(database, conversation_id, "calculate")
    policy = FrozenTracePolicy(
        new_opaque_id(),
        "credentials-v1",
        pii_enabled,
        BUILTIN_PII_RULESET_REVISION_ID if pii_enabled else None,
    )
    credential = "sk-" + "a" * 48
    runtime_credential = "local-provider-password-31976"
    contact = "elise@example.test"
    context = {
        "role": "user",
        "content": f"Project guidance: {credential}; {runtime_credential}; {contact}",
        "_chatbook_ephemeral_origin": "project_instructions",
    }
    messages = [{"role": "user", "content": "calculate"}, context]
    descriptors = [
        user,
        ProviderArtifactTraceProvenance(
            TraceProvenanceSource.PROJECT_INSTRUCTION, policy
        ),
    ]
    actor, chain = new_opaque_id(), new_opaque_id()
    factory = ConsoleTraceBoundaryFactory(database)
    adapter_inputs = []

    def adapter(**kwargs):
        adapter_inputs.append(deepcopy(thaw_json(kwargs["messages_payload"])))
        return {"choices": [{"message": {"content": "ok"}}]}

    def boundary(request, resolution, route):
        return factory(request, resolution, route)

    gateway = make_gateway(
        chat_api_call_fn=adapter, trace_call_boundary_factory=boundary
    )
    resolution = ConsoleProviderResolution(
        ready=True,
        provider="openai",
        model="gpt-test",
        execution_key="openai",
        base_url="https://api.openai.com/v1",
        streaming=False,
        api_key=runtime_credential,
    )
    routes = (ConsoleRequestRoute.AGENT_FIRST, ConsoleRequestRoute.TOOL_LOOP)
    if retained_tamper:
        # The third request has a warm checkpoint in the same route; cold
        # bootstrap independently reconstructs and verifies the incoming rows.
        routes += (ConsoleRequestRoute.TOOL_LOOP,)
    for call_index, route in enumerate(routes):
        if route is ConsoleRequestRoute.TOOL_LOOP:
            if cold_factory:
                factory = ConsoleTraceBoundaryFactory(database)
            messages.append({"role": "tool", "content": "42", "tool_call_id": "call-1"})
            descriptors.append(
                ProviderArtifactTraceProvenance(
                    TraceProvenanceSource.TOOL_RESULT, policy
                )
            )
            if retained_tamper and call_index == 2:
                original = type(factory.service).prepare_surface_provenance

                def tampered(*args, _original=original, **kwargs):
                    retained = dict(kwargs["retained_artifact_values"])
                    sequence = next(iter(retained))
                    if retained_tamper == "context":
                        retained[sequence] = {
                            **context,
                            "content": "unadmitted guidance",
                        }
                    else:
                        retained[0 if retained_tamper == "saved" else 999] = context
                    kwargs["retained_artifact_values"] = retained
                    return _original(*args, **kwargs)

                monkeypatch.setattr(
                    type(factory.service), "prepare_surface_provenance", tampered
                )
        semantic = _semantic_request(
            messages,
            descriptors,
            policy,
            route=route,
            actor_id=actor,
            chain_id=chain,
        )
        prepared = gateway.prepare_chat_request(
            resolution,
            semantic,
            route=route,
            route_actor_id=actor,
            route_chain_id=chain,
            capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
        )

        async def dispatch(request=prepared, dispatch_route=route, call_signals=None):
            return [
                item
                async for item in gateway.stream_chat(
                    resolution,
                    request,
                    route=dispatch_route,
                    route_actor_id=actor,
                    route_chain_id=chain,
                    capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
                    signals=call_signals,
                )
            ]

        if retained_tamper and call_index == 2:
            with pytest.raises(TraceCallPersistenceError):
                await dispatch()
            assert len(adapter_inputs) == 2
            return
        if owned_retry and call_index == 1:
            accepted_owner = object()
            signals = ConsoleProviderStreamSignals()
            gateway._bind_trace_preparation(signals, accepted_owner)
            original_bind = type(factory.repository).bind_call

            def fail_bind(
                instance,
                *args,
                _original=original_bind,
                _repository=factory.repository,
                **kwargs,
            ):
                result = _original(instance, *args, **kwargs)
                if instance is _repository:
                    raise RuntimeError("synthetic trace bind rollback")
                return result

            with monkeypatch.context() as patch:
                patch.setattr(type(factory.repository), "bind_call", fail_bind)
                with pytest.raises(TraceCallPersistenceError) as failure:
                    await dispatch(call_signals=signals)
            failed = failure.value.boundary
            assert failed.dispatch_outcome == "rolled_back"
            reserved = failed.reserve()
            assert len(adapter_inputs) == 1
            gateway._bind_trace_preparation(signals, accepted_owner, boundary=failed)
            assert await dispatch(call_signals=signals) == ["ok"]
            assert len(adapter_inputs) == 2
            with database.transaction() as cursor:
                recovered = factory.repository.get_call(cursor, reserved.call_id)
                assert recovered.idempotency_key == reserved.idempotency_key
                assert recovered.state is TraceCallState.COMPLETE
                assert (
                    cursor.execute(
                        "SELECT COUNT(*) FROM console_trace_calls"
                    ).fetchone()[0]
                    == 2
                )
        else:
            assert await dispatch() == ["ok"]
        assert adapter_inputs[-1] == messages

    captures = ConsoleTraceNativeReader(database).read_calls(user_id)
    assert len(captures) == 2
    for call in captures:
        captured = json.dumps(call.capture.request)
        assert credential not in captured
        assert runtime_credential not in captured
        assert (contact in captured) is not pii_enabled
