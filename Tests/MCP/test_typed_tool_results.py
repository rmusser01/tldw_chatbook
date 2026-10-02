"""Complete typed MCP results at the protocol and display boundaries."""


def test_structured_error_survives_client_normalization():
    from tldw_chatbook.MCP.tool_results import parse_tool_result

    raw = {
        "content": [],
        "structuredContent": {"version": 2, "decision": "pass"},
        "isError": True,
        "_meta": {"trace": "t"},
    }
    result = parse_tool_result(raw)
    assert result.is_error
    assert result.structured_content == raw["structuredContent"]
    assert result.metadata == raw["_meta"]


import json
import sys
from contextlib import asynccontextmanager

import pytest

from tldw_chatbook.MCP.client import MCPClient

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]


@asynccontextmanager
async def controlled_stdio_client(tmp_path, *, tool_schema=None):
    """An inert stdlib peer echoes owned fixtures, never importing the app."""
    peer = tmp_path / "peer.py"
    peer.write_text(
        """import json, sys, time
calls = 0
held = None
schema = json.loads(sys.argv[1])
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    method = request["method"]
    if method == "initialize":
        result = {"protocolVersion": "2025-03-26", "capabilities": {}, "serverInfo": {"name": "owned-fixture"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "fixture", "inputSchema": schema}]}
    elif method == "resources/list":
        result = {"resources": []}
    elif method == "prompts/list":
        result = {"prompts": []}
    else:
        calls += 1
        args = request["params"]["arguments"]
        if args.get("hold"):
            held = {"jsonrpc": "2.0", "id": request["id"], "result": args["payload"]}
            continue
        if "wire_result" in args:
            message = '{"jsonrpc":"2.0","id":' + str(request["id"]) + ',"result":' + args["wire_result"] + '}'
            if args.get("batch"):
                message = '[{"jsonrpc":"2.0","method":"notifications/test"},' + message + ']'
            print(message, flush=True)
            continue
        if "rpc_error" in args:
            print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "error": {"code": -32603, "message": args["rpc_error"]}}), flush=True)
            continue
        if args.get("disconnect"):
            sys.exit(0)
        if args.get("pause"):
            time.sleep(10)
        if args.get("delay"):
            time.sleep(args["delay"])
        result = {"content": [{"type": "text", "text": str(calls)}]} if args.get("counter") else args.get("payload", {"content": []})
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
    if held is not None:
        print(json.dumps(held), flush=True)
        held = None
""",
        encoding="utf-8",
    )
    client = MCPClient(name="typed-results-test")
    process = None
    session = None
    try:
        assert await client.connect_to_server(
            "fixture",
            sys.executable,
            ["-I", str(peer), json.dumps(tool_schema or {"type": "object"})],
        )
        session = client.sessions["fixture"]
        process = session.process
        yield client
    finally:
        await client.disconnect_from_server("fixture")
        if session is not None:
            await session.close()
        if process is not None:
            assert process.returncode is not None


async def reconnect_owned_peer(client, tmp_path):
    """A new explicit connection follows actual settlement of the prior child."""
    previous = client.sessions.get("fixture")
    if previous is not None:
        await client.disconnect_from_server("fixture")
        assert previous.process.returncode is not None
    assert await client.connect_to_server(
        "fixture",
        sys.executable,
        ["-I", str(tmp_path / "peer.py"), json.dumps({"type": "object"})],
    )


@pytest.mark.asyncio
async def test_actual_stdio_preserves_complete_tool_response(tmp_path):
    """Catches the first field loss, with legacy display on the same connection."""
    async with controlled_stdio_client(tmp_path) as client:
        control = await client.call_tool(
            "fixture",
            "fixture",
            {"payload": {"content": [{"type": "text", "text": "ok"}]}},
        )
        assert control == {"result": [{"type": "text", "text": "ok"}]}
        raw = {
            "content": [],
            "structuredContent": {"version": 2, "decision": "pass"},
            "isError": True,
            "_meta": {"trace": "t"},
        }
        result = await client.sessions["fixture"].call_tool("fixture", {"payload": raw})
        assert getattr(result, "structured_content", None) == raw["structuredContent"]
        assert result.is_error is True
        assert result.metadata == {"trace": "t"}


@pytest.mark.parametrize("flag", [None, 0, 1, "false", "true", [], {}])
def test_parser_rejects_non_boolean_error_flag(flag):
    from tldw_chatbook.MCP.tool_results import parse_tool_result

    with pytest.raises(ValueError, match="mcp_error_flag_invalid"):
        parse_tool_result({"content": [], "isError": flag})


@pytest.mark.parametrize(
    "payload", [{"content": []}, {"content": [], "isError": False}]
)
def test_empty_success_is_distinct_from_transport_failure(payload):
    from tldw_chatbook.MCP.tool_results import parse_tool_result, project_tool_result

    result = parse_tool_result(payload)
    assert result.is_error is False
    assert result.transport_error is None
    assert result.structured_content is None
    assert project_tool_result(result) == {"result": []}


def test_complete_blocks_and_metadata_are_owned_copies_with_legacy_projection():
    from tldw_chatbook.MCP.tool_results import parse_tool_result, project_tool_result

    blocks = [
        {"type": "image", "mimeType": "image/png", "data": "YWJj", "_meta": {"a": 1}},
        {"type": "resource", "resource": {"uri": "note://one", "text": "body"}},
    ]
    payload = {"content": blocks, "_meta": {"trace": ["t"]}}
    result = parse_tool_result(payload)
    blocks[0]["data"] = "changed"
    payload["_meta"]["trace"].append("changed")
    assert result.content[0]["data"] == "YWJj"
    assert result.content[1]["resource"]["text"] == "body"
    assert result.metadata == {"trace": ["t"]}
    projected = project_tool_result(result)
    assert projected == {"result": list(result.content)}
    projected["result"][0]["data"] = "display changed"
    assert result.content[0]["data"] == "YWJj"


@pytest.mark.parametrize("case", ["metadata_size", "depth", "cycle", "nan", "object"])
def test_parser_bounds_complete_payload_before_projection(case):
    from tldw_chatbook.MCP.tool_results import parse_tool_result

    payload = {"content": []}
    if case == "metadata_size":
        payload["_meta"] = {"ignored_by_display": "x" * 786_432}
    elif case == "depth":
        nested = {}
        payload["_meta"] = nested
        for _ in range(66):
            nested["next"] = {}
            nested = nested["next"]
    elif case == "cycle":
        payload["_meta"] = payload
    elif case == "nan":
        payload["_meta"] = {"number": float("nan")}
    else:
        payload["_meta"] = {"object": object()}
    with pytest.raises(ValueError):
        parse_tool_result(payload)


def test_protocol_result_cap_does_not_apply_hook_cap_to_ordinary_results():
    from tldw_chatbook.MCP.tool_results import parse_tool_result, project_tool_result

    text = "é" * 20_000
    result = parse_tool_result({"content": [{"type": "text", "text": text}]})
    assert project_tool_result(result) == {"result": [{"type": "text", "text": text}]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wire",
    [
        '{"content":[],"isError":true,"isError":false}',
        '{"content":[],"structuredContent":{"decision":"deny","decision":"pass"}}',
        '{"content":[],"_meta":{"a":1,"a":2}}',
        '{"content":[],"_meta":{"value":NaN}}',
        '{"content":[],"isError":"false"}',
    ],
)
async def test_actual_stdio_rejects_ambiguous_protocol_result(tmp_path, wire):
    async with controlled_stdio_client(tmp_path) as client:
        control = await client.call_tool(
            "fixture", "fixture", {"payload": {"content": []}}
        )
        assert control == {"result": []}
        result = await client.call_tool_result(
            "fixture", "fixture", {"wire_result": wire}
        )
        assert result.transport_error is not None
        assert result.structured_content is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["rpc_error", "disconnect"])
async def test_actual_stdio_failure_never_interpolates_remote_error_text(
    tmp_path, mode
):
    sentinel = "credential:https://private.invalid/?api_key=SECRET"
    async with controlled_stdio_client(tmp_path) as client:
        control = await client.call_tool(
            "fixture", "fixture", {"payload": {"content": []}}
        )
        assert control == {"result": []}
        result = await client.call_tool_result("fixture", "fixture", {mode: sentinel})
        assert result.transport_error is not None
        assert sentinel not in result.transport_error
        assert result.metadata == {}
        assert result.content == ()
        assert result.structured_content is None


async def controlled_service(tmp_path, client):
    from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
    from tldw_chatbook.MCP.local_store import LocalExternalMCPProfile, LocalMCPStore
    from tldw_chatbook.MCP.unified_control_plane_service import (
        UnifiedMCPControlPlaneService,
    )

    store = LocalMCPStore(tmp_path / "store.json")
    store.save_profile(
        LocalExternalMCPProfile(profile_id="fixture", command=sys.executable)
    )
    store.save_discovery_snapshot("fixture", await client.describe_server("fixture"))
    local = LocalMCPControlService(store=store, client=client, manifest_provider=dict)
    return UnifiedMCPControlPlaneService(
        local_service=local, server_service=None, target_store=None, context_store=None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [False, True])
async def test_actual_local_and_unified_services_keep_typed_result_and_audit(
    tmp_path, error
):
    async with controlled_stdio_client(tmp_path) as client:
        service = await controlled_service(tmp_path, client)
        payload = {
            "content": [],
            "structuredContent": {"decision": "pass"},
            "isError": error,
            "_meta": {"trace": "t"},
        }
        local = await service.local_service.execute_external_tool_result(
            "fixture", "fixture", {"payload": payload}
        )
        assert local.structured_content == {"decision": "pass"}
        assert local.is_error is error
        result = await service.execute_hub_tool_result(
            "local:fixture", "fixture", {"payload": payload}
        )
        assert result.structured_content == {"decision": "pass"}
        assert result.is_error is error
        assert result.metadata == {"trace": "t"}
        assert result.duplicate_keys_checked is True
        assert result.dispatch_state == "settled"
        records = service.execution_log.read_recent()
        assert len(records) == 1
        assert records[0]["status"] == ("error" if error else "success")
        if error:
            with pytest.raises(RuntimeError, match="mcp_tool_error"):
                await service.execute_hub_tool(
                    "local:fixture", "fixture", {"payload": payload}
                )
        else:
            assert await service.execute_hub_tool(
                "local:fixture", "fixture", {"payload": payload}
            ) == {"result": []}
        assert len(service.execution_log.read_recent()) == 2


@pytest.mark.asyncio
async def test_actual_provider_rejects_error_with_pass_and_keeps_success_display(
    tmp_path,
):
    import asyncio

    from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider
    from tldw_chatbook.MCP.hub_tool_catalog import HubTool

    async with controlled_stdio_client(tmp_path) as client:
        service = await controlled_service(tmp_path, client)
        provider = MCPToolProvider(
            service=service, main_loop=asyncio.get_running_loop()
        )
        await provider.compose_catalog()
        tool = HubTool(
            server_key="local:fixture",
            server_label="fixture",
            source="local",
            name="fixture",
            description="",
            input_schema={"type": "object"},
            tags=(),
            stale=False,
            executable=True,
        )
        service.set_tool_state("local:fixture", "fixture", "allow", tool=tool)
        tool_id = provider.list_catalog()[0].id
        control = await asyncio.to_thread(
            provider.invoke,
            tool_id,
            {"payload": {"content": [{"type": "text", "text": "ok"}]}},
        )
        assert control.ok is True
        assert json.loads(control.content) == {
            "result": [{"type": "text", "text": "ok"}]
        }
        result = await asyncio.to_thread(
            provider.invoke,
            tool_id,
            {
                "payload": {
                    "content": [],
                    "structuredContent": {"decision": "pass"},
                    "isError": True,
                }
            },
        )
        assert result.ok is False
        assert result.dispatch_state == "settled"
        assert len(service.execution_log.read_recent()) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("batch", [False, True])
async def test_wire_provenance_keeps_exact_result_span_excluding_envelope(
    tmp_path, batch
):
    wire = '{ "content" : [], "structuredContent": {"decision":"pass"}, "_meta": {"trace":"é"}, "unknown" : 1 }'
    async with controlled_stdio_client(tmp_path) as client:
        result = await client.call_tool_result(
            "fixture", "fixture", {"wire_result": wire, "batch": batch}
        )
        assert result.transport_error is None
        assert result.encoded_payload == wire.encode("utf-8")
        assert result.duplicate_keys_checked is True
        assert result.dispatch_state == "settled"


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [786_431, 786_432, 786_433])
async def test_original_result_size_bound_is_separate_from_rpc_envelope(tmp_path, size):
    prefix, suffix = '{"content":[{"type":"text","text":"', '"}]}'
    wire = prefix + "x" * (size - len(prefix) - len(suffix)) + suffix
    async with controlled_stdio_client(tmp_path) as client:
        result = await client.call_tool_result(
            "fixture", "fixture", {"wire_result": wire}
        )
        if size > 786_432:
            assert result.transport_error is not None
            assert result.dispatch_state == "uncertain"
        else:
            assert result.transport_error is None
            assert result.encoded_payload == wire.encode()
            assert result.dispatch_state == "settled"


@pytest.mark.asyncio
async def test_mutation_and_model_copy_cannot_borrow_original_provenance(tmp_path):
    import copy

    async with controlled_stdio_client(tmp_path) as client:
        result = await client.call_tool_result(
            "fixture",
            "fixture",
            {"payload": {"content": [], "structuredContent": {"decision": "pass"}}},
        )
        assert result.duplicate_keys_checked is True
        for copied in (
            result.model_copy(),
            result.model_copy(deep=True),
            copy.copy(result),
            copy.deepcopy(result),
            result.model_copy(update={"is_error": True}),
        ):
            assert copied.encoded_payload is None
            assert copied.duplicate_keys_checked is False
        result.structured_content["decision"] = "deny"
        assert result.encoded_payload is None
        assert result.duplicate_keys_checked is False


def test_plain_model_and_spoofed_payload_never_acquire_wire_provenance():
    from types import SimpleNamespace

    from pydantic import BaseModel

    from tldw_chatbook.MCP.tool_results import parse_tool_result

    class CompatibleResult(BaseModel):
        content: list[dict]
        isError: bool

    payloads = [
        {
            "content": [],
            "isError": False,
            "_encoded_payload": "fake",
            "duplicate_keys_checked": True,
            "dispatch_state": "settled",
            "_meta": {"dispatch_state": "settled"},
        },
        SimpleNamespace(content=[], isError=False),
        CompatibleResult(content=[], isError=False),
    ]
    for payload in payloads:
        result = parse_tool_result(payload)
        assert result.encoded_payload is None
        assert result.duplicate_keys_checked is False
        assert result.dispatch_state == "not_started"


def test_reparsing_transport_failure_cannot_turn_it_into_empty_success():
    from tldw_chatbook.MCP.tool_results import (
        MCPDispatchObservation,
        parse_tool_result,
        transport_failure,
    )

    result = parse_tool_result(
        transport_failure(
            "mcp_transport_unavailable", MCPDispatchObservation("uncertain")
        )
    )
    assert result.transport_error == "mcp_transport_unavailable"
    assert result.dispatch_state == "uncertain"


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before_write", "after_write"])
async def test_service_timeout_preserves_actual_write_observation(tmp_path, phase):
    import asyncio

    from tldw_chatbook.MCP.tool_results import MCPDispatchObservation, observe_dispatch

    async with controlled_stdio_client(tmp_path) as client:
        service = await controlled_service(tmp_path, client)
        assert await service.execute_hub_tool(
            "local:fixture", "fixture", {"payload": {"content": []}}
        ) == {"result": []}
        session = client.sessions["fixture"]
        observation = MCPDispatchObservation()
        if phase == "before_write":
            await session._write_lock.acquire()
        try:
            with observe_dispatch(observation):
                result = await service.execute_hub_tool_result(
                    "local:fixture", "fixture", {"pause": True}, timeout_seconds=0.03
                )
        finally:
            if phase == "before_write":
                session._write_lock.release()
        assert result.transport_error == "mcp_tool_timeout"
        assert result.dispatch_state == (
            "not_started" if phase == "before_write" else "uncertain"
        )
        assert observation.state == result.dispatch_state
        assert session._pending_requests == {}
        records = service.execution_log.read_recent()
        assert len(records) == 2
        assert records[0]["status"] == "timeout"
        if phase == "before_write":
            # Current recovery custody settles the child before returning timeout.
            assert session.process.returncode is not None
            assert "fixture" not in client.sessions
            await reconnect_owned_peer(client, tmp_path)
            assert await asyncio.wait_for(
                client.call_tool("fixture", "fixture", {"payload": {"content": []}}), 1
            ) == {"result": []}


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before_write", "after_write"])
async def test_confirmed_service_cancellation_retains_observation_and_one_audit(
    tmp_path, phase
):
    import asyncio

    from tldw_chatbook.MCP.tool_results import MCPDispatchObservation, observe_dispatch

    async with controlled_stdio_client(tmp_path) as client:
        service = await controlled_service(tmp_path, client)
        session = client.sessions["fixture"]
        observation = MCPDispatchObservation()
        if phase == "before_write":
            await session._write_lock.acquire()
        with observe_dispatch(observation):
            task = asyncio.create_task(
                service.execute_hub_tool_result(
                    "local:fixture", "fixture", {"pause": True}
                )
            )
        try:
            for _ in range(100):
                if session._pending_requests and (
                    phase == "before_write" or observation.state == "uncertain"
                ):
                    break
                await asyncio.sleep(0.001)
            assert session._pending_requests
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            if phase == "before_write":
                session._write_lock.release()
        assert observation.state == (
            "not_started" if phase == "before_write" else "uncertain"
        )
        assert session._pending_requests == {}
        assert len(service.execution_log.read_recent()) == 1
        assert service.execution_log.read_recent()[0]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_unified_governance_refusal_never_writes_a_tool_request(tmp_path):
    from tldw_chatbook.MCP.tool_results import MCPDispatchObservation, observe_dispatch

    async with controlled_stdio_client(tmp_path) as client:
        service = await controlled_service(tmp_path, client)
        assert await service.execute_hub_tool(
            "local:fixture", "fixture", {"payload": {"content": []}}
        ) == {"result": []}

        class RefuseTrigger:
            def require_allowed(self, action_id, **_kwargs):
                raise PermissionError("fixture refusal")

        # The real local service's governance entry is the controlled dependency.
        service.local_service.policy_enforcer = RefuseTrigger()
        observation = MCPDispatchObservation()
        with observe_dispatch(observation), pytest.raises(PermissionError):
            await service.execute_hub_tool_result(
                "local:fixture", "fixture", {"pause": True}
            )
        assert observation.state == "not_started"
        assert len(service.execution_log.read_recent()) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["malformed", "duplicate", "oversized_meta", "rpc_error"]
)
async def test_unified_failure_status_and_no_remote_diagnostics(tmp_path, case):
    async with controlled_stdio_client(tmp_path) as client:
        service = await controlled_service(tmp_path, client)
        sentinel = "secret-credential-DO-NOT-LOG"
        args = {
            "malformed": {"payload": {"content": [], "isError": "false"}},
            "duplicate": {
                "wire_result": '{"content":[],"_meta":{"trace":1,"trace":2}}'
            },
            "oversized_meta": {
                "payload": {"content": [], "_meta": {"ignored": "x" * 786_432}}
            },
            "rpc_error": {"rpc_error": sentinel},
        }[case]
        result = await service.execute_hub_tool_result("local:fixture", "fixture", args)
        assert result.transport_error is not None
        assert result.dispatch_state == (
            "settled" if case == "rpc_error" else "uncertain"
        )
        records = service.execution_log.read_recent()
        assert len(records) == 1
        assert records[0]["status"] == "error"
        assert sentinel not in json.dumps(records)
        assert result.encoded_payload is None


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["structured", "mirror", "resource", "image"])
async def test_unified_complete_result_and_legacy_projection_share_one_boundary(
    tmp_path, shape
):
    structured = {"version": 2, "decision": "pass"}
    payloads = {
        "structured": {"structuredContent": structured},
        "mirror": {
            "structuredContent": structured,
            "content": [{"type": "text", "text": json.dumps(structured)}],
        },
        "resource": {
            "content": [
                {"type": "resource", "resource": {"uri": "note://one", "text": "body"}}
            ]
        },
        "image": {
            "content": [{"type": "image", "mimeType": "image/png", "data": "YWJj"}]
        },
    }
    payload = payloads[shape]
    async with controlled_stdio_client(tmp_path) as client:
        service = await controlled_service(tmp_path, client)
        typed = await service.execute_hub_tool_result(
            "local:fixture", "fixture", {"payload": payload}
        )
        assert json.loads(typed.encoded_payload) == payload
        assert typed.structured_content == payload.get("structuredContent")
        display = await service.execute_hub_tool(
            "local:fixture", "fixture", {"payload": payload}
        )
        assert display == {"result": payload.get("content", [])}
        assert len(service.execution_log.read_recent()) == 2


@pytest.mark.parametrize("original,changed", [(True, 1), (False, 0), (1, True)])
def test_changed_json_types_cannot_borrow_wire_provenance(original, changed):
    from tldw_chatbook.MCP.tool_results import decode_protocol_frame, parse_tool_result

    frame = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [], "structuredContent": {"value": original}},
        }
    ).encode()
    decoded = decode_protocol_frame(frame)["result"]
    assert parse_tool_result(decoded).duplicate_keys_checked is True
    decoded["structuredContent"]["value"] = changed
    result = parse_tool_result(decoded)
    assert result.duplicate_keys_checked is False
    assert result.encoded_payload is None


@pytest.mark.asyncio
async def test_concurrent_calls_keep_exact_matching_results_and_observations(tmp_path):
    import asyncio

    from tldw_chatbook.MCP.tool_results import MCPDispatchObservation, observe_dispatch

    async with controlled_stdio_client(tmp_path) as client:
        observations = [MCPDispatchObservation(), MCPDispatchObservation()]

        async def call(index):
            with observe_dispatch(observations[index]):
                return await client.call_tool_result(
                    "fixture",
                    "fixture",
                    {
                        "hold": index == 0,
                        "payload": {
                            "content": [],
                            "structuredContent": {"index": index},
                            "isError": bool(index),
                        },
                    },
                )

        first = asyncio.create_task(call(0))
        for _ in range(100):
            if observations[0].state == "uncertain":
                break
            await asyncio.sleep(0.001)
        results = await asyncio.wait_for(asyncio.gather(first, call(1)), 1)
        assert [result.structured_content for result in results] == [
            {"index": 0},
            {"index": 1},
        ]
        assert [result.is_error for result in results] == [False, True]
        assert [item.state for item in observations] == ["settled", "settled"]
        assert all(result.duplicate_keys_checked for result in results)


@pytest.mark.asyncio
async def test_late_response_after_service_timeout_is_not_replayed_or_audited_again(
    tmp_path,
):
    import asyncio

    async with controlled_stdio_client(tmp_path) as client:
        service = await controlled_service(tmp_path, client)
        result = await service.execute_hub_tool_result(
            "local:fixture", "fixture", {"delay": 0.08}, timeout_seconds=0.02
        )
        assert result.dispatch_state == "uncertain"
        assert result.transport_error == "mcp_tool_timeout"
        assert "fixture" not in client.sessions
        await reconnect_owned_peer(client, tmp_path)
        control = await asyncio.wait_for(
            client.call_tool_result("fixture", "fixture", {"counter": True}), 1
        )
        assert control.content == ({"type": "text", "text": "1"},)
        assert control.dispatch_state == "settled"
        assert len(service.execution_log.read_recent()) == 1
        assert service.execution_log.read_recent()[0]["status"] == "timeout"


@pytest.mark.asyncio
async def test_builtin_typed_entry_retains_arbitrary_legacy_mapping(tmp_path):
    async with controlled_stdio_client(tmp_path) as client:
        service = await controlled_service(tmp_path, client)

        class Delegate:
            async def execute_tool(self, tool_name, arguments):
                return {"answer": 42, "application_fields": ["retained"]}

        service.local_service.runtime_delegate = Delegate()
        result = await service.execute_hub_tool_result(
            "builtin:tldw_chatbook", "calculator", {}
        )
        assert isinstance(result, dict)
        assert result["result"] == {"answer": 42, "application_fields": ["retained"]}
        assert len(service.execution_log.read_recent()) == 1


@pytest.mark.asyncio
async def test_provider_unacknowledged_cancellation_is_uncertain_even_before_write(
    tmp_path, monkeypatch
):
    import asyncio
    import threading

    import tldw_chatbook.Agents.mcp_tool_provider as provider_module
    from tldw_chatbook.MCP.hub_tool_catalog import HubTool

    async with controlled_stdio_client(tmp_path) as client:
        service = await controlled_service(tmp_path, client)
        provider = provider_module.MCPToolProvider(
            service=service, main_loop=asyncio.get_running_loop()
        )
        await provider.compose_catalog()
        tool = HubTool(
            server_key="local:fixture",
            server_label="fixture",
            source="local",
            name="fixture",
            description="",
            input_schema={"type": "object"},
            tags=(),
            stale=False,
            executable=True,
        )
        service.set_tool_state("local:fixture", "fixture", "allow", tool=tool)
        tool_id = provider.list_catalog()[0].id
        control = await asyncio.to_thread(
            provider.invoke, tool_id, {"payload": {"content": []}}
        )
        assert control.ok is True
        monkeypatch.setattr(service, "_tool_call_timeout", lambda: 0.5)
        monkeypatch.setattr(provider_module, "_RESULT_WAIT_SLACK_SECONDS", -0.49)
        outcomes = []
        audit_threads = []
        start_thread = threading.Thread.start

        def tracked_start(thread):
            if thread.name == "mcp-bridge-audit":
                audit_threads.append(thread)
            start_thread(thread)

        monkeypatch.setattr(threading.Thread, "start", tracked_start)
        thread = threading.Thread(
            target=lambda: outcomes.append(provider.invoke(tool_id, {"delay": 0.08}))
        )
        thread.start()
        # Deliberately withhold the loop: Future cancellation is not coroutine acknowledgement.
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert outcomes[0].ok is False
        assert outcomes[0].dispatch_state == "uncertain"
        for audit_thread in audit_threads:
            await asyncio.to_thread(audit_thread.join, 1)
            assert not audit_thread.is_alive()
        await asyncio.sleep(0.03)
        await reconnect_owned_peer(client, tmp_path)
        control = await asyncio.wait_for(
            client.call_tool_result("fixture", "fixture", {"counter": True}), 1
        )
        # A fresh, explicitly connected child receives only the new control.
        assert control.content[0]["text"] == "1"
        rows = service.execution_log.read_recent()
        assert len(rows) == 2
        assert rows[0]["status"] == "error"
        assert rows[0]["error_category"] == "completion_uncertain"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reference",
    ["#/$defs/value", "https://example.invalid/schema", "file:///not-accessed.json"],
)
async def test_raw_client_and_provider_preserve_schema_reference_constraints(
    tmp_path, reference
):
    import asyncio

    from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider

    schema = {
        "type": "object",
        "properties": {"value": {"$ref": reference}},
        "$defs": {"value": {"type": "integer", "minimum": 2}},
        "required": ["value"],
    }
    async with controlled_stdio_client(tmp_path, tool_schema=schema) as client:
        assert client.get_server_tools("fixture")[0]["inputSchema"] == schema
        service = await controlled_service(tmp_path, client)
        provider = MCPToolProvider(
            service=service, main_loop=asyncio.get_running_loop()
        )
        await provider.compose_catalog()
        assert provider.load_schema(provider.list_catalog()[0].id).parameters == schema


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [786_432, 786_433])
async def test_original_internal_whitespace_is_charged_before_reserialization(
    tmp_path, size
):
    prefix, suffix = '{"content":[', "]}"
    wire = prefix + " " * (size - len(prefix) - len(suffix)) + suffix
    async with controlled_stdio_client(tmp_path) as client:
        result = await client.call_tool_result(
            "fixture", "fixture", {"wire_result": wire}
        )
        if size == 786_432:
            assert result.transport_error is None
            assert result.encoded_payload == wire.encode()
        else:
            assert result.transport_error == "mcp_result_invalid"
            assert result.encoded_payload is None


@pytest.mark.asyncio
async def test_unqualified_session_failure_cannot_claim_dispatch_never_started():
    from types import SimpleNamespace

    class CompatibleSession:
        fail = False

        async def call_tool(self, tool_name, arguments):
            if self.fail:
                raise RuntimeError("remote-secret-must-not-be-diagnostic")
            return SimpleNamespace(content=[{"type": "text", "text": "ok"}])

    client = MCPClient(name="compatible-result-control")
    session = CompatibleSession()
    client.sessions["fixture"] = session
    control = await client.call_tool_result("fixture", "tool", {})
    assert control.content == ({"type": "text", "text": "ok"},)
    assert control.dispatch_state == "settled"
    assert control.encoded_payload is None
    session.fail = True
    result = await client.call_tool_result("fixture", "tool", {})
    assert result.transport_error == "mcp_transport_unavailable"
    assert result.dispatch_state == "uncertain"
    assert result.encoded_payload is None


@pytest.mark.asyncio
async def test_builtin_delegate_settlement_preserves_host_truth_without_wire_evidence(
    tmp_path,
):
    from tldw_chatbook.MCP.tool_results import MCPDispatchObservation, observe_dispatch

    async with controlled_stdio_client(tmp_path) as client:
        service = await controlled_service(tmp_path, client)

        class Delegate:
            fail = False

            async def execute_tool(self, tool_name, arguments):
                if self.fail:
                    raise RuntimeError("owned-delegate-error")
                return {"answer": 42}

        delegate = Delegate()
        service.local_service.runtime_delegate = delegate
        observation = MCPDispatchObservation()
        with observe_dispatch(observation):
            result = await service.execute_hub_tool_result(
                "builtin:tldw_chatbook", "calculator", {}
            )
        assert result["result"] == {"answer": 42}
        assert observation.state == "settled"
        delegate.fail = True
        observation = MCPDispatchObservation()
        with (
            observe_dispatch(observation),
            pytest.raises(RuntimeError, match="owned-delegate-error"),
        ):
            await service.execute_hub_tool_result(
                "builtin:tldw_chatbook", "calculator", {}
            )
        assert observation.state == "settled"
        assert len(service.execution_log.read_recent()) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("stall_before_append", [False, True])
@pytest.mark.parametrize("audit_fails", [False, True])
async def test_provider_audit_publication_race_records_one_truthful_outcome(
    tmp_path, monkeypatch, stall_before_append, audit_fails
):
    import asyncio
    import time

    import tldw_chatbook.Agents.mcp_tool_provider as provider_module
    from tldw_chatbook.MCP.hub_tool_catalog import HubTool

    async with controlled_stdio_client(tmp_path) as client:
        service = await controlled_service(tmp_path, client)
        provider = provider_module.MCPToolProvider(
            service=service, main_loop=asyncio.get_running_loop()
        )
        await provider.compose_catalog()
        tool = HubTool(
            server_key="local:fixture",
            server_label="fixture",
            source="local",
            name="fixture",
            description="",
            input_schema={"type": "object"},
            tags=(),
            stale=False,
            executable=True,
        )
        service.set_tool_state("local:fixture", "fixture", "allow", tool=tool)
        tool_id = provider.list_catalog()[0].id
        control = await asyncio.to_thread(provider.invoke, tool_id, {})
        assert control.ok is True
        assert [row["status"] for row in service.execution_log.read_recent()] == [
            "success"
        ]
        monkeypatch.setattr(service, "_tool_call_timeout", lambda: 2.0)
        monkeypatch.setattr(provider_module, "_RESULT_WAIT_SLACK_SECONDS", -1.5)
        append = service.execution_log.append
        import threading

        entered = threading.Event()

        def slow_append(record):
            entered.set()
            if stall_before_append:
                time.sleep(0.75)
            if not audit_fails:
                append(record)
            if not stall_before_append:
                time.sleep(0.75)
            if audit_fails:
                raise OSError("PRIVATE_AUDIT_ERROR")

        monkeypatch.setattr(service.execution_log, "append", slow_append)
        result = await asyncio.to_thread(provider.invoke, tool_id, {})
        assert entered.is_set(), "the real audit publication window was not reached"
        assert result.ok is False
        assert result.dispatch_state == "uncertain"
        await asyncio.sleep(0.02)
        rows = service.execution_log.read_recent()
        assert [row["status"] for row in rows] == (
            ["success"] if audit_fails else ["success", "success"]
        )
        counter = await client.call_tool_result("fixture", "fixture", {"counter": True})
        assert counter.content[0]["text"] == "3"


@pytest.mark.asyncio
@pytest.mark.parametrize("bridge_failure", ["closed_loop", "cancelled_future"])
@pytest.mark.parametrize("audit_fails", [False, True])
async def test_provider_bridge_audit_is_bounded_and_best_effort(
    tmp_path, monkeypatch, bridge_failure, audit_fails
):
    import asyncio
    import concurrent.futures
    import threading

    import tldw_chatbook.Agents.mcp_tool_provider as provider_module
    from tldw_chatbook.MCP.hub_tool_catalog import HubTool

    async with controlled_stdio_client(tmp_path) as client:
        service = await controlled_service(tmp_path, client)
        provider = provider_module.MCPToolProvider(
            service=service, main_loop=asyncio.get_running_loop()
        )
        await provider.compose_catalog()
        tool = HubTool(
            server_key="local:fixture",
            server_label="fixture",
            source="local",
            name="fixture",
            description="",
            input_schema={"type": "object"},
            tags=(),
            stale=False,
            executable=True,
        )
        service.set_tool_state("local:fixture", "fixture", "allow", tool=tool)
        tool_id = provider.list_catalog()[0].id
        assert (await asyncio.to_thread(provider.invoke, tool_id, {})).ok
        append = service.execution_log.append
        entered, release, finished = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        attempts = []
        audit_threads = []
        start_thread = threading.Thread.start

        def tracked_start(thread):
            if thread.name == "mcp-bridge-audit":
                audit_threads.append(thread)
            start_thread(thread)

        monkeypatch.setattr(threading.Thread, "start", tracked_start)

        def stalled_append(record):
            attempts.append(record)
            entered.set()
            try:
                assert release.wait(2)
                if audit_fails:
                    raise OSError("PRIVATE_AUDIT_ERROR")
                append(record)
            finally:
                finished.set()

        monkeypatch.setattr(service.execution_log, "append", stalled_append)
        if bridge_failure == "closed_loop":
            closed_loop = asyncio.new_event_loop()
            closed_loop.close()
            provider._main_loop = closed_loop
        else:

            def cancelled_submission(coro, loop):
                coro.close()
                future = concurrent.futures.Future()
                future.cancel()
                return future

            monkeypatch.setattr(
                asyncio, "run_coroutine_threadsafe", cancelled_submission
            )
        outcomes = []
        worker = threading.Thread(
            target=lambda: outcomes.append(provider.invoke(tool_id, {}))
        )
        worker.start()
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            await asyncio.to_thread(worker.join, 0.2)
            assert not worker.is_alive(), "bridge return waited for audit I/O"
            assert outcomes[0].ok is False
            state = "not_started" if bridge_failure == "closed_loop" else "uncertain"
            assert outcomes[0].dispatch_state == state
            # Saturation drops another best-effort audit without a queue or wait.
            saturated = await asyncio.wait_for(
                asyncio.to_thread(provider.invoke, tool_id, {}), 0.2
            )
            assert saturated.ok is False
            assert saturated.dispatch_state == state
            assert len(attempts) == 1
        finally:
            release.set()
            await asyncio.to_thread(worker.join, 1)
            assert await asyncio.to_thread(finished.wait, 1)
            for audit_thread in audit_threads:
                await asyncio.to_thread(audit_thread.join, 1)
                assert not audit_thread.is_alive()
        assert len(audit_threads) == 1
        assert len(attempts) == 1
        assert attempts[0].status == "error"
        assert attempts[0].error_category == (
            "bridge_not_started"
            if bridge_failure == "closed_loop"
            else "completion_uncertain"
        )
        assert len(service.execution_log.read_recent()) == (1 if audit_fails else 2)
        # The actual worker exited, so a new fallback can use the released slot.
        assert not (await asyncio.to_thread(provider.invoke, tool_id, {})).ok
        for audit_thread in audit_threads:
            await asyncio.to_thread(audit_thread.join, 1)
            assert not audit_thread.is_alive()
        assert len(audit_threads) == 2
        assert len(attempts) == 2
        assert len(service.execution_log.read_recent()) == (1 if audit_fails else 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("reject_thread_start", [False, True])
async def test_provider_bridge_first_audit_and_unscheduled_fallback(
    tmp_path, monkeypatch, reject_thread_start
):
    import asyncio
    import threading
    import time

    import tldw_chatbook.Agents.mcp_tool_provider as provider_module
    from tldw_chatbook.MCP.hub_tool_catalog import HubTool

    async with controlled_stdio_client(tmp_path) as client:
        service = await controlled_service(tmp_path, client)
        provider = provider_module.MCPToolProvider(
            service=service, main_loop=asyncio.get_running_loop()
        )
        await provider.compose_catalog()
        tool = HubTool(
            server_key="local:fixture",
            server_label="fixture",
            source="local",
            name="fixture",
            description="",
            input_schema={"type": "object"},
            tags=(),
            stale=False,
            executable=True,
        )
        service.set_tool_state("local:fixture", "fixture", "allow", tool=tool)
        tool_id = provider.list_catalog()[0].id
        assert (await asyncio.to_thread(provider.invoke, tool_id, {})).ok
        monkeypatch.setattr(service, "_tool_call_timeout", lambda: 2.0)
        monkeypatch.setattr(provider_module, "_RESULT_WAIT_SLACK_SECONDS", -1.5)
        record = service._record_tool_execution
        start_thread = threading.Thread.start
        audit_threads = []

        def delayed_service_record(*args, **kwargs):
            if threading.current_thread().name != "mcp-bridge-audit":
                time.sleep(0.75)
            return record(*args, **kwargs)

        def controlled_start(thread):
            if thread.name == "mcp-bridge-audit":
                if reject_thread_start:
                    raise RuntimeError("PRIVATE_THREAD_START_ERROR")
                audit_threads.append(thread)
            start_thread(thread)

        monkeypatch.setattr(service, "_record_tool_execution", delayed_service_record)
        monkeypatch.setattr(threading.Thread, "start", controlled_start)
        try:
            result = await asyncio.to_thread(provider.invoke, tool_id, {})
            assert result.ok is False
            assert result.dispatch_state == "uncertain"
        finally:
            for thread in audit_threads:
                await asyncio.to_thread(thread.join, 1)
                assert not thread.is_alive()
        rows = service.execution_log.read_recent()
        assert len(rows) == 2
        assert rows[0]["status"] == ("success" if reject_thread_start else "error")
        if not reject_thread_start:
            assert rows[0]["error_category"] == "completion_uncertain"
        # Start failure also releases capacity without claiming publication.
        reject_thread_start = False
        monkeypatch.setattr(service, "_record_tool_execution", record)
        closed_loop = asyncio.new_event_loop()
        closed_loop.close()
        provider._main_loop = closed_loop
        assert not (await asyncio.to_thread(provider.invoke, tool_id, {})).ok
        for thread in audit_threads:
            await asyncio.to_thread(thread.join, 1)
            assert not thread.is_alive()
        rows = service.execution_log.read_recent()
        assert len(rows) == 3
        assert rows[0]["error_category"] == "bridge_not_started"


@pytest.mark.asyncio
async def test_saturated_bridge_writer_does_not_suppress_later_service_audit(
    tmp_path, monkeypatch
):
    import asyncio
    import threading
    import time

    import tldw_chatbook.Agents.mcp_tool_provider as provider_module
    from tldw_chatbook.MCP.hub_tool_catalog import HubTool

    async with controlled_stdio_client(tmp_path) as client:
        service = await controlled_service(tmp_path, client)
        loop = asyncio.get_running_loop()
        provider = provider_module.MCPToolProvider(service=service, main_loop=loop)
        await provider.compose_catalog()
        tool = HubTool(
            server_key="local:fixture",
            server_label="fixture",
            source="local",
            name="fixture",
            description="",
            input_schema={"type": "object"},
            tags=(),
            stale=False,
            executable=True,
        )
        service.set_tool_state("local:fixture", "fixture", "allow", tool=tool)
        tool_id = provider.list_catalog()[0].id
        assert (await asyncio.to_thread(provider.invoke, tool_id, {})).ok
        append = service.execution_log.append
        record = service._record_tool_execution
        entered, release = threading.Event(), threading.Event()
        audit_threads = []

        def blocked_bridge_append(row):
            if threading.current_thread().name == "mcp-bridge-audit":
                audit_threads.append(threading.current_thread())
                entered.set()
                assert release.wait(2)
            append(row)

        def delayed_service_record(*args, **kwargs):
            if threading.current_thread().name != "mcp-bridge-audit":
                time.sleep(0.75)
            return record(*args, **kwargs)

        monkeypatch.setattr(service.execution_log, "append", blocked_bridge_append)
        closed_loop = asyncio.new_event_loop()
        closed_loop.close()
        provider._main_loop = closed_loop
        try:
            assert not (await asyncio.to_thread(provider.invoke, tool_id, {})).ok
            assert await asyncio.to_thread(entered.wait, 1)
            provider._main_loop = loop
            monkeypatch.setattr(service, "_tool_call_timeout", lambda: 2.0)
            monkeypatch.setattr(provider_module, "_RESULT_WAIT_SLACK_SECONDS", -1.5)
            monkeypatch.setattr(
                service, "_record_tool_execution", delayed_service_record
            )
            result = await asyncio.to_thread(provider.invoke, tool_id, {})
            assert result.ok is False
            assert result.dispatch_state == "uncertain"
            assert [row["status"] for row in service.execution_log.read_recent()] == [
                "success",
                "success",
            ]
        finally:
            release.set()
            for thread in audit_threads:
                await asyncio.to_thread(thread.join, 1)
                assert not thread.is_alive()
        rows = service.execution_log.read_recent()
        assert len(audit_threads) == 1
        assert len(rows) == 3
        assert rows[0]["error_category"] == "bridge_not_started"


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["client", "local", "plane"])
async def test_typed_entries_keep_actual_producer_and_storage_admission(
    tmp_path, owner
):
    from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission
    from tldw_chatbook.MCP.activation import MCPActivationRequired

    async with controlled_stdio_client(tmp_path) as client:
        plane = await controlled_service(tmp_path, client)
        if owner == "client":
            producer = client
            invoke = lambda: client.call_tool_result(
                "fixture", "fixture", {"counter": True}
            )
        elif owner == "local":
            producer = plane.local_service
            invoke = lambda: producer.execute_external_tool_result(
                "fixture", "fixture", {"counter": True}
            )
        else:
            producer = plane
            invoke = lambda: plane.execute_hub_tool_result(
                "local:fixture", "fixture", {"counter": True}
            )
        producer._producer_lifetime.close()
        with pytest.raises(bootstrap.RecoveryRequired, match="runtime_producer_paused"):
            await invoke()
        producer._producer_lifetime.resume()
        pause = storage_admission._begin_local_pause()
        try:
            with pytest.raises(MCPActivationRequired, match="mcp_activation_required"):
                await invoke()
        finally:
            pause.resume()
        result = await invoke()
        assert result.content[0]["text"] == "1"
