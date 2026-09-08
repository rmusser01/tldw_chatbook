"""Per-call system headers preserve discovery, retries and immutable history."""

import json
from copy import deepcopy
from dataclasses import replace

import httpx
import pytest

from Tests.Chat.test_console_trace_runtime import (
    _saved_message,
    _semantic_request,
)
from Tests.Chat.test_console_trace_runtime import (
    make_database as make_database,  # noqa: PLC0414 - re-export the pytest fixture
)
from Tests.Chat.test_console_trace_runtime import (
    make_gateway as make_gateway,  # noqa: PLC0414 - re-export the pytest fixture
)
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


@pytest.mark.parametrize("cold_factory", [False, True])
@pytest.mark.parametrize("pii_enabled", [False, True])
@pytest.mark.parametrize("owned_retry", [False, True])
async def test_system_header_preserves_wire_input_history_and_owned_retry(
    tmp_path,
    make_database,
    make_gateway,
    monkeypatch,
    cold_factory,
    pii_enabled,
    owned_retry,
):
    await _exercise_system_header(
        tmp_path,
        make_database,
        make_gateway,
        monkeypatch,
        cold_factory=cold_factory,
        pii_enabled=pii_enabled,
        owned_retry=owned_retry,
    )


@pytest.mark.parametrize(
    "tamper",
    [
        "saved_system",
        "saved_user",
        "tool_prefix",
        "moved_system",
        "system_source",
        "policy",
    ],
)
@pytest.mark.parametrize("cold_factory", [False, True])
async def test_system_header_does_not_authorize_changed_history(
    tmp_path,
    make_database,
    make_gateway,
    monkeypatch,
    tamper,
    cold_factory,
):
    await _exercise_system_header(
        tmp_path,
        make_database,
        make_gateway,
        monkeypatch,
        cold_factory=cold_factory,
        tamper=tamper,
    )


@pytest.mark.parametrize(
    "tamper", ["missing", "duplicate", "ordinal", "role", "marker"]
)
async def test_reader_rejects_malformed_system_header(
    tmp_path,
    make_database,
    make_gateway,
    monkeypatch,
    tamper,
):
    database, factory, reader, capture = await _exercise_system_header(
        tmp_path,
        make_database,
        make_gateway,
        monkeypatch,
    )
    original = type(reader.service).reconstruct_header

    def malformed(instance, cursor, header_id):
        header = original(instance, cursor, header_id)
        if instance is not reader.service:
            return header
        components = list(header.components)
        system = next(
            item for item in components if item.component_kind == "rendered_system_row"
        )
        defaults = dict(header.adapter_defaults)
        if tamper == "missing":
            components.remove(system)
        elif tamper == "duplicate":
            components.append(system)
        elif tamper == "ordinal":
            components[components.index(system)] = replace(system, ordinal=1)
        elif tamper == "role":
            components[components.index(system)] = replace(
                system, value={"role": "user", "content": "invalid"}
            )
        else:
            defaults.pop("rendered_system_slot")
        return replace(header, components=tuple(components), adapter_defaults=defaults)

    monkeypatch.setattr(type(reader.service), "reconstruct_header", malformed)
    with database.transaction() as cursor:
        call = factory.repository.get_call(cursor, capture.call_id)
        with pytest.raises(ValueError, match="rendered_system_slot"):
            reader._reconstruct_request(cursor, call)


@pytest.mark.parametrize("cold_factory", [False, True])
@pytest.mark.parametrize("mismatch", [False, True])
async def test_system_only_continuation_verifies_the_final_row(
    tmp_path,
    make_database,
    make_gateway,
    monkeypatch,
    cold_factory,
    mismatch,
):
    await _exercise_system_header(
        tmp_path,
        make_database,
        make_gateway,
        monkeypatch,
        cold_factory=cold_factory,
        system_only=True,
        probe_final_mismatch=mismatch,
    )


async def _exercise_system_header(
    tmp_path,
    make_database,
    make_gateway,
    monkeypatch,
    *,
    cold_factory=False,
    pii_enabled=False,
    owned_retry=False,
    tamper=None,
    system_only=False,
    probe_final_mismatch=False,
):
    database = make_database(tmp_path / "system.sqlite", "system-header")
    conversation = database.add_conversation({"title": "System headers"})
    user_id, user = _saved_message(database, conversation, "Calculate")
    policy = FrozenTracePolicy(
        new_opaque_id(),
        "credentials-v1",
        pii_enabled,
        BUILTIN_PII_RULESET_REVISION_ID if pii_enabled else None,
    )
    system = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.RENDERED_SYSTEM, policy
    )
    if tamper == "saved_system":
        _, system = _saved_message(
            database, conversation, "Initial system", sender="system"
        )
    secret = "runtime-password-for-header-test"
    contact = "elise@example.test"
    messages = [
        {"role": "system", "content": "Initial system"},
        {"role": "user", "content": "Calculate"},
    ]
    descriptors = [system, user]
    actor, chain = new_opaque_id(), new_opaque_id()
    factory = ConsoleTraceBoundaryFactory(database)
    sent = []

    def respond(request):
        sent.append(json.loads(request.content)["messages"])
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        gateway = make_gateway(
            http_client=client,
            trace_call_boundary_factory=lambda request, resolution, route: factory(
                request, resolution, route
            ),
        )
        resolution = ConsoleProviderResolution(
            ready=True,
            provider="llama_cpp",
            execution_key="llama_cpp",
            model="qwen3.7-27b",
            base_url="http://localhost:8080",
            api_key=secret,
            streaming=False,
        )
        original = ()
        reader = ConsoleTraceNativeReader(database)
        for index in range(3):
            route = (
                ConsoleRequestRoute.AGENT_FIRST
                if index == 0
                else ConsoleRequestRoute.TOOL_LOOP
            )
            if index:
                if cold_factory:
                    factory = ConsoleTraceBoundaryFactory(database)
                if not system_only:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": f"call-{index}",
                            "content": str(index),
                        }
                    )
                    descriptors.append(
                        ProviderArtifactTraceProvenance(
                            TraceProvenanceSource.TOOL_RESULT, policy
                        )
                    )
            if index == 2:
                messages[0] = {
                    "role": "system",
                    "content": f'Loaded calculator. Credential: "{secret}"; contact: {contact}',
                }
                if tamper == "saved_user":
                    messages[1] = {"role": "user", "content": "Forged question"}
                elif tamper == "tool_prefix":
                    messages[2] = {**messages[2], "content": "Forged result"}
                elif tamper == "moved_system":
                    messages[0], messages[1] = messages[1], messages[0]
                    descriptors[0], descriptors[1] = descriptors[1], descriptors[0]
                elif tamper == "system_source":
                    descriptors[0] = ProviderArtifactTraceProvenance(
                        TraceProvenanceSource.ACTIVE_REQUEST, policy
                    )
                elif tamper == "policy":
                    policy = FrozenTracePolicy(
                        new_opaque_id(),
                        "credentials-v1",
                        True,
                        BUILTIN_PII_RULESET_REVISION_ID,
                    )
                    descriptors = [
                        replace(item, policy=policy)
                        if type(item) is ProviderArtifactTraceProvenance
                        else item
                        for item in descriptors
                    ]

            def prepare(descriptors=descriptors, policy=policy, route=route):
                return gateway.prepare_chat_request(
                    resolution,
                    _semantic_request(
                        messages,
                        descriptors,
                        policy,
                        route=route,
                        actor_id=actor,
                        chain_id=chain,
                    ),
                    route=route,
                    route_actor_id=actor,
                    route_chain_id=chain,
                    capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
                )

            if index == 2 and tamper in {"moved_system", "system_source"}:
                with pytest.raises(
                    ValueError, match="trace provenance category mismatch"
                ):
                    prepare()
                assert len(sent) == 2
                assert reader.read_calls(user_id) == original
                return
            request = prepare()
            verifications = []
            if index == 2 and system_only:
                original_verify = type(factory.service)._verify_prepared_boundary

                def verify_system_only(
                    instance,
                    boundary,
                    provenance,
                    actual,
                    expected,
                    issuer,
                    original_verify=original_verify,
                    verifications=verifications,
                ):
                    prepared = instance._prepared_capabilities[id(provenance)]
                    assert not prepared.items
                    if probe_final_mismatch:
                        changed = dict(actual)
                        changed["messages_payload"] = (
                            {"role": "system", "content": "Changed after preparation"},
                            *actual["messages_payload"][1:],
                        )
                        assert not original_verify(
                            instance, boundary, provenance, changed, expected, issuer
                        )
                        assert id(provenance) not in instance._prepared_capabilities
                        # Restore the test-owned preparation after proving
                        # rejection, then exercise the real unmodified call.
                        instance._prepared_capabilities[id(provenance)] = prepared
                    result = original_verify(
                        instance, boundary, provenance, actual, expected, issuer
                    )
                    verifications.append(result)
                    return result

                monkeypatch.setattr(
                    type(factory.service),
                    "_verify_prepared_boundary",
                    verify_system_only,
                )

            async def dispatch(signals=None, request=request, route=route):
                return [
                    chunk
                    async for chunk in gateway.stream_chat(
                        resolution,
                        request,
                        route=route,
                        route_actor_id=actor,
                        route_chain_id=chain,
                        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
                        signals=signals,
                    )
                ]

            if index == 2 and tamper:
                with pytest.raises(TraceCallPersistenceError):
                    await dispatch()
                assert len(sent) == 2
                assert reader.read_calls(user_id) == original
                return
            if index == 2 and owned_retry:
                owner = object()
                signals = ConsoleProviderStreamSignals()
                gateway._bind_trace_preparation(signals, owner)
                original_bind = type(factory.repository).bind_call

                def fail_bind(
                    instance,
                    *args,
                    original_bind=original_bind,
                    factory=factory,
                    **kwargs,
                ):
                    result = original_bind(instance, *args, **kwargs)
                    if instance is factory.repository:
                        raise RuntimeError(
                            "synthetic rollback after header persistence"
                        )
                    return result

                with monkeypatch.context() as patch:
                    patch.setattr(type(factory.repository), "bind_call", fail_bind)
                    with pytest.raises(TraceCallPersistenceError) as failure:
                        await dispatch(signals)
                boundary = failure.value.boundary
                assert boundary.dispatch_outcome == "rolled_back"
                reserved = boundary.reserve()
                assert len(sent) == 2
                gateway._bind_trace_preparation(signals, owner, boundary=boundary)
                assert await dispatch(signals) == ["ok"]
                with database.transaction() as cursor:
                    recovered = factory.repository.get_call(cursor, reserved.call_id)
                    assert recovered.state is TraceCallState.COMPLETE
                    assert recovered.idempotency_key == reserved.idempotency_key
                    assert (
                        cursor.execute(
                            "SELECT COUNT(*) FROM console_trace_calls"
                        ).fetchone()[0]
                        == 3
                    )
            else:
                assert await dispatch() == ["ok"]
            if index == 2 and system_only:
                assert verifications == [True]
            assert sent[-1] == messages
            captures = reader.read_calls(user_id)
            assert len(captures) == index + 1
            assert captures[: len(original)] == original
            original = deepcopy(captures)

        captured = captures[-1].capture.request["messages_payload"]
        assert captured[0]["role"] == "system"
        assert "Loaded calculator" in captured[0]["content"]
        assert secret not in json.dumps(captured)
        assert (contact in json.dumps(captured)) is not pii_enabled
        assert len(sent) == 3
        return database, factory, reader, captures[-1]
