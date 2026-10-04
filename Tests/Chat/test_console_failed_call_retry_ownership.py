"""A failed call authorizes an unchanged retry, never arbitrary tool history."""

from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_trace_runtime import (
    _saved_message,
    _semantic_request,
    make_database as _make_database_fixture,
    make_gateway as _make_gateway_fixture,
)
from tldw_chatbook.Chat.Chat_Deps import ChatProviderError, ChatRateLimitError
from tldw_chatbook.Chat.console_prepared_request import prepare_provider_request
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution
from tldw_chatbook.Chat.console_trace_errors import TraceCallPersistenceError
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy, new_opaque_id
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestRoute,
    ConsoleTraceCaptureMode,
    ProviderArtifactTraceProvenance,
    TraceProvenanceSource,
)
from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory

pytestmark = pytest.mark.bootstrap_profile
make_database = _make_database_fixture
make_gateway = _make_gateway_fixture


@pytest.mark.parametrize("wire_style", ["single_preamble", "distinct_roles"])
@pytest.mark.parametrize(
    "scenario",
    [
        "valid",
        "cold",
        "system_valid",
        "updated_system",
        "changed_system",
        "changed_surface",
        "changed_actor",
        "changed_chain",
        "changed_policy",
        "foreign_turn",
        "dispatch_unknown",
        "dispatch_started",
        "settlement_failed",
    ],
)
async def test_failed_call_retry_requires_exact_durable_chain(
    tmp_path, monkeypatch, make_database, make_gateway, scenario, wire_style
):
    database = make_database(tmp_path / "retry.sqlite", "retry")
    conversation = database.add_conversation({"title": "retry"})
    _, revision = _saved_message(database, conversation, "hello")
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    actor, chain = new_opaque_id(), new_opaque_id()
    factory = ConsoleTraceBoundaryFactory(database)
    unfinished = scenario in {
        "dispatch_unknown",
        "dispatch_started",
        "settlement_failed",
    }
    if unfinished:
        # Fail the settlement write before it commits. Terminal settlements
        # cannot be rewritten to manufacture an open call: SQLite enforces it.
        monkeypatch.setattr(
            type(factory.service),
            "prepare_settlement_handoff",
            lambda *args, **kwargs: SimpleNamespace(settle=lambda _owner: False),
        )
    calls = []
    errors = []

    def safe_error(_provider, exc):
        errors.append(exc)
        return "fixture provider failure"

    def provider(**kwargs):
        calls.append(kwargs)
        if len(calls) == (2 if scenario == "updated_system" else 1):
            raise ChatRateLimitError("fixture", provider="deepseek")
        return {"choices": [{"message": {"content": "recovered"}}]}

    gateway = make_gateway(
        chat_api_call_fn=provider,
        trace_call_boundary_factory=factory,
        safe_error_copy=safe_error,
    )
    prepare = gateway.prepare_chat_request

    def prepare_request(*args, **kwargs):
        prepared = prepare(*args, **kwargs)
        if wire_style == "distinct_roles":
            return prepare_provider_request(
                prepared.semantic,
                wire_style=wire_style,
                provider=prepared.provider,
                model=prepared.model,
                capacity=prepared.capacity,
            )
        return prepared

    monkeypatch.setattr(gateway, "prepare_chat_request", prepare_request)
    resolution = ConsoleProviderResolution(
        ready=True,
        provider="deepseek",
        execution_key="deepseek",
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key="fixture",
        streaming=False,
    )
    messages, descriptors = [{"role": "user", "content": "hello"}], [revision]
    if scenario in {"system_valid", "updated_system", "changed_system"}:
        messages.insert(0, {"role": "system", "content": "Original instructions"})
        descriptors.insert(
            0,
            ProviderArtifactTraceProvenance(
                TraceProvenanceSource.RENDERED_SYSTEM, policy
            ),
        )

    async def send(route):
        request = _semantic_request(
            messages, descriptors, policy, route=route, actor_id=actor, chain_id=chain
        )
        return [
            item
            async for item in gateway.stream_chat(
                resolution,
                request,
                route=route,
                route_actor_id=actor,
                route_chain_id=chain,
                capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
            )
        ]

    if scenario == "updated_system":
        assert await send(ConsoleRequestRoute.AGENT_FIRST) == ["recovered"]
        messages[0] = {"role": "system", "content": "Updated after successful call"}
    with pytest.raises(ChatProviderError) as failure:
        await send(
            ConsoleRequestRoute.TOOL_LOOP
            if scenario == "updated_system"
            else ConsoleRequestRoute.AGENT_FIRST
        )
    assert failure.value.status_code == 429, errors
    failed_attempts = 2 if scenario == "updated_system" else 1
    assert len(calls) == failed_attempts
    with database.transaction() as cursor:
        assert cursor.execute(
            "SELECT state FROM console_trace_calls ORDER BY call_sequence DESC LIMIT 1"
        ).fetchone()[0] == ("dispatch_started" if unfinished else "error")
    if scenario == "cold":
        gateway._trace_call_boundary_factory = ConsoleTraceBoundaryFactory(database)
    elif scenario == "changed_system":
        messages[0] = {"role": "system", "content": "Unobserved changed instructions"}
    elif scenario == "changed_surface":
        messages = messages + [
            {"role": "tool", "content": "unobserved addition", "tool_call_id": "forged"}
        ]
        descriptors = descriptors + [
            ProviderArtifactTraceProvenance(TraceProvenanceSource.TOOL_RESULT, policy)
        ]
    elif scenario == "foreign_turn":
        other = database.add_conversation({"title": "other"})
        _, revision = _saved_message(database, other, "hello")
        descriptors = [revision]
    elif scenario == "changed_actor":
        actor = new_opaque_id()
    elif scenario == "changed_chain":
        chain = new_opaque_id()
    elif scenario == "changed_policy":
        policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    elif scenario == "dispatch_unknown":
        with database.transaction() as cursor:
            cursor.execute(
                "UPDATE console_trace_calls SET state = 'dispatch_unknown', settled_at = dispatch_started_at"
            )
    if scenario in {"valid", "cold", "system_valid", "updated_system"}:
        assert await send(ConsoleRequestRoute.TOOL_LOOP) == ["recovered"]
        assert len(calls) == failed_attempts + 1
    else:
        with pytest.raises(TraceCallPersistenceError):
            await send(ConsoleRequestRoute.TOOL_LOOP)
        assert len(calls) == failed_attempts
