"""Retry JSON identity is enforced before actual hosted and OpenAI HTTP posts."""

import json as json_module
from copy import deepcopy

import pytest
import requests

from Tests.Chat.test_console_retry_json_identity import _response_format
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
    ProviderArtifactTraceProvenance,
    TraceProvenanceSource,
)
from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory
from tldw_chatbook.Utils.sensitive_llm_logging import sensitive_llm_request

pytestmark = pytest.mark.bootstrap_profile
make_database = _make_database_fixture
make_gateway = _make_gateway_fixture


@pytest.mark.parametrize("provider", ["groq", "together", "openai"])
@pytest.mark.parametrize("changed", [False, True], ids=["unchanged", "changed-type"])
async def test_json_schema_retry_fence_precedes_actual_http(
    tmp_path,
    monkeypatch,
    make_database,
    make_gateway,
    provider,
    changed,
):
    database = make_database(tmp_path / "retry-json-wire.sqlite", "retry-json-wire")
    conversation = database.add_conversation({"title": "retry JSON wire"})
    _, revision = _saved_message(database, conversation, "hello")
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    actor, chain = new_opaque_id(), new_opaque_id()
    posts = []
    models = {
        "groq": "llama-3.3-70b-versatile",
        "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "openai": "gpt-4o",
    }
    model = models[provider]

    def post(_session, url, *, json, **_kwargs):
        posts.append((url, deepcopy(json)))
        response = requests.Response()
        response.url = url
        response.status_code = 429 if len(posts) == 1 else 200
        body = (
            {"error": {"message": "fixture rejection"}}
            if len(posts) == 1
            else {
                "id": "test",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "recovered"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "total_tokens": 4,
                },
            }
        )
        response._content = json_module.dumps(body).encode("utf-8")
        response._content_consumed = True
        return response

    monkeypatch.setattr(requests.Session, "post", post)
    gateway = make_gateway(
        trace_call_boundary_factory=ConsoleTraceBoundaryFactory(database)
    )
    resolution = ConsoleProviderResolution(
        ready=True,
        provider=provider,
        execution_key=provider,
        model=model,
        base_url={
            "groq": "https://api.groq.com/openai/v1",
            "together": "https://api.together.xyz/v1",
            "openai": "https://api.openai.com/v1",
        }[provider],
        api_key="fixture-key",
        streaming=False,
    )

    async def send(route, value):
        request = _semantic_request(
            [
                {"role": "system", "content": "Fixture instructions"},
                {"role": "user", "content": "hello"},
            ],
            [
                ProviderArtifactTraceProvenance(
                    TraceProvenanceSource.RENDERED_SYSTEM, policy
                ),
                revision,
            ],
            policy,
            route=route,
            actor_id=actor,
            chain_id=chain,
        )
        prepared = gateway.prepare_chat_request(
            resolution,
            request,
            route=route,
            route_actor_id=actor,
            route_chain_id=chain,
            response_format=_response_format(value, arrays=True),
            capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
        )
        # The real sensitive one-shot context disables internal HTTP retries;
        # this test exercises two independently trace-owned gateway attempts.
        with sensitive_llm_request():
            return [
                item
                async for item in gateway.stream_chat(
                    resolution,
                    prepared,
                    route=route,
                    route_actor_id=actor,
                    route_chain_id=chain,
                    capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
                )
            ]

    with pytest.raises(ChatRateLimitError) as rejected:
        await send(ConsoleRequestRoute.AGENT_FIRST, True)
    assert type(rejected.value) is ChatRateLimitError
    assert rejected.value.provider == provider
    assert rejected.value.status_code == 429
    assert len(posts) == 1
    schema = posts[0][1]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["flag"]["enum"] == [True]
    assert schema["properties"]["flag"]["enum"][0] is True
    with database.transaction() as cursor:
        initial = cursor.execute(
            "SELECT state, provider_inactive_at, outcome FROM console_trace_calls"
        ).fetchone()
        assert initial[0] == "error", tuple(initial)
        header = cursor.execute(
            "SELECT header.provider_name, header.endpoint_identity FROM console_trace_request_headers header JOIN console_trace_calls call ON call.request_header_id = header.header_id"
        ).fetchone()
        assert header[0] == provider, tuple(header)
        surfaces = cursor.execute(
            "SELECT reference_kind FROM console_trace_surface_nodes"
        ).fetchall()
        assert "revision" in {row[0] for row in surfaces}
        assert "omission" not in {row[0] for row in surfaces}
    if changed:
        with pytest.raises(TraceCallPersistenceError):
            await send(ConsoleRequestRoute.TOOL_LOOP, 1)
        assert len(posts) == 1
    else:
        assert await send(ConsoleRequestRoute.TOOL_LOOP, True) == ["recovered"]
        assert len(posts) == 2
        assert json_module.dumps(
            posts[1][1]["response_format"], sort_keys=True
        ) == json_module.dumps(
            posts[0][1]["response_format"],
            sort_keys=True,
        )
