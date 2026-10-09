"""Failed retry settings use JSON identity, including nested scalar types."""

from copy import deepcopy
from dataclasses import replace

import pytest

from Tests.Chat.test_console_trace_runtime import _saved_message, _semantic_request
from Tests.Chat.test_console_trace_runtime import (
    make_database as _make_database_fixture,
)
from Tests.Chat.test_console_trace_runtime import make_gateway as _make_gateway_fixture
from tldw_chatbook.Chat.Chat_Deps import ChatRateLimitError
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution
from tldw_chatbook.Chat.console_trace_errors import TraceCallPersistenceError
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy, new_opaque_id
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestRoute,
    ConsoleTraceCaptureMode,
)
from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory

pytestmark = pytest.mark.bootstrap_profile
make_database = _make_database_fixture
make_gateway = _make_gateway_fixture


def _response_format(value, *, arrays=False):
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "retry_result",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "flag": {"enum": [value]} if arrays else {"const": value}
                },
                "additionalProperties": False,
            },
        },
    }


@pytest.mark.parametrize("provider", ["deepseek", "openai"])
@pytest.mark.parametrize(
    "field",
    ["response_format", "response_format_arrays", "seed", "thinking_budget_tokens"],
)
@pytest.mark.parametrize(
    ("original", "retried", "allowed"),
    [
        (True, 1, False),
        (False, 0, False),
        (1, 1.0, False),
        (1, 1, True),
    ],
    ids=["true-to-one", "false-to-zero", "integer-to-float", "unchanged"],
)
async def test_failed_retry_rejects_python_equal_json_changes(
    tmp_path,
    monkeypatch,
    make_database,
    make_gateway,
    provider,
    field,
    original,
    retried,
    allowed,
):
    database = make_database(tmp_path / "retry-json.sqlite", "retry-json")
    conversation = database.add_conversation({"title": "retry JSON"})
    _, revision = _saved_message(database, conversation, "hello")
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    actor, chain = new_opaque_id(), new_opaque_id()
    factory = ConsoleTraceBoundaryFactory(database)
    calls = []

    def adapter(**kwargs):
        calls.append(deepcopy(kwargs))
        if len(calls) == 1:
            raise ChatRateLimitError("fixture rejected request", provider=provider)
        return {"choices": [{"message": {"content": "recovered"}}]}

    gateway = make_gateway(
        chat_api_call_fn=adapter,
        trace_call_boundary_factory=factory,
        safe_error_copy=lambda _provider, _error: "fixture rejection",
    )
    prepare = gateway.prepare_chat_request
    setting_name = "response_format" if field.startswith("response_format") else field
    response_format = (
        _response_format(original, arrays=field == "response_format_arrays")
        if setting_name == "response_format"
        else None
    )

    def prepare_request(*args, **kwargs):
        return replace(prepare(*args, **kwargs), response_format=response_format)

    monkeypatch.setattr(gateway, "prepare_chat_request", prepare_request)
    resolution = ConsoleProviderResolution(
        ready=True,
        provider=provider,
        execution_key=provider,
        model="deepseek-chat" if provider == "deepseek" else "gpt-test",
        base_url="https://api.deepseek.com"
        if provider == "deepseek"
        else "https://api.openai.com/v1",
        api_key="fixture",
        streaming=False,
        **({field: original} if setting_name != "response_format" else {}),
    )

    async def send(route):
        request = _semantic_request(
            [{"role": "user", "content": "hello"}],
            [revision],
            policy,
            route=route,
            actor_id=actor,
            chain_id=chain,
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

    with pytest.raises(ChatRateLimitError) as rejection:
        await send(ConsoleRequestRoute.AGENT_FIRST)
    assert type(rejection.value) is ChatRateLimitError
    assert rejection.value.provider == provider
    assert rejection.value.status_code == 429
    assert len(calls) == 1
    submitted = calls[0][setting_name]
    if setting_name == "response_format":
        property_schema = submitted["json_schema"]["schema"]["properties"]["flag"]
        submitted = (
            property_schema["enum"][0]
            if "enum" in property_schema
            else property_schema["const"]
        )
    assert type(submitted) is type(original)
    with database.transaction() as cursor:
        failed = cursor.execute(
            "SELECT state, request_header_id FROM console_trace_calls"
        ).fetchone()
        assert failed[0] == "error"
        components = cursor.execute(
            "SELECT component_kind FROM console_trace_header_components WHERE header_id = ?",
            (failed[1],),
        ).fetchall()
        # These generic gateway branches have no literal-envelope fallback fence.
        assert "provider_literal_envelope" not in {row[0] for row in components}

    if setting_name == "response_format":
        response_format = _response_format(
            retried, arrays=field == "response_format_arrays"
        )
        if allowed:
            response_format = dict(reversed(list(response_format.items())))
            response_format["json_schema"] = dict(
                reversed(list(response_format["json_schema"].items()))
            )
    else:
        resolution = replace(resolution, **{field: retried})
    if allowed:
        assert await send(ConsoleRequestRoute.TOOL_LOOP) == ["recovered"]
        assert len(calls) == 2
        with database.transaction() as cursor:
            rows = cursor.execute(
                "SELECT call_sequence, state FROM console_trace_calls ORDER BY call_sequence"
            ).fetchall()
        assert [tuple(row) for row in rows] == [(0, "error"), (1, "complete")]
    else:
        with pytest.raises(TraceCallPersistenceError):
            await send(ConsoleRequestRoute.TOOL_LOOP)
        assert len(calls) == 1
