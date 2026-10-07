"""Exact wire-era controls, including actual owned stdio subprocesses.

Primary fixture pins are recorded in protocol_profiles.py and the M2 report.
These peers are original fixtures, not a third-party conformance server.
"""

from __future__ import annotations

import json
import sys

import pytest

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]

from tldw_chatbook.MCP.local_store import TransportProfile
from tldw_chatbook.MCP.protocol_profiles import (
    header_value,
    protocol_profile,
    tool_header_paths,
)

VERSIONS = ("2026-07-28", "2025-11-25", "2025-03-26")


@pytest.mark.parametrize("value", ["2099-01-01", "", None, 7])
def test_closed_versions(value):
    with pytest.raises(ValueError, match="mcp_protocol_unsupported"):
        protocol_profile(value)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("region", "region"),
        (True, "true"),
        (42, "42"),
        ("", ""),
        (" a ", "=?base64?IGEg?="),
        ("a\nb", "=?base64?YQpi?="),
        ("é", "=?base64?w6k=?="),
        ("=?base64?literal?=", "=?base64?PT9iYXNlNjQ/bGl0ZXJhbD89?="),
    ],
)
def test_routing_header_encoding(value, expected):
    assert header_value(value) == expected


@pytest.mark.parametrize(
    "schema",
    [
        {"x-mcp-header": "Root", "type": "string"},
        {"properties": {"x": {"type": "number", "x-mcp-header": "Bad"}}},
        {"properties": {"x": {"type": "string", "x-mcp-header": "Bad\nHeader"}}},
        {
            "properties": {
                "x": {"type": "string", "x-mcp-header": "Same"},
                "y": {"type": "boolean", "x-mcp-header": "same"},
            }
        },
        {"allOf": [{"properties": {"x": {"type": "string", "x-mcp-header": "Bad"}}}]},
        {"properties": {"x": {"items": {"type": "string", "x-mcp-header": "Bad"}}}},
    ],
)
def test_invalid_header_annotations(schema):
    with pytest.raises(ValueError, match="mcp_header_schema_invalid"):
        tool_header_paths(schema)


def test_internal_reference_constraints_are_preserved():
    schema = {
        "type": "object",
        "$defs": {"entry": {"type": "string"}},
        "properties": {
            "x": {"$ref": "#/$defs/entry"},
            "nested": {
                "type": "object",
                "properties": {"region": {"type": "string", "x-mcp-header": "Region"}},
            },
        },
    }
    assert tool_header_paths(schema) == ((("nested", "region"), "Region", "string"),)
    assert schema["properties"]["x"]["$ref"] == "#/$defs/entry"


@pytest.mark.parametrize(
    "url, loopback",
    [
        ("http://example.com/mcp", True),
        ("http://127.0.0.1/mcp", False),
        ("https://user:pass@example.com/mcp", False),
        ("https://@example.com/mcp", False),
        ("https://example.com/mcp#", False),
        ("http://localhost/mcp", True),
        ("ftp://127.0.0.1/mcp", True),
    ],
)
def test_origin_authority_is_explicit(url, loopback):
    with pytest.raises(ValueError, match="mcp_origin_invalid"):
        TransportProfile(
            profile_id="peer",
            transport="streamable_http",
            url=url,
            development_loopback=loopback,
        )


def test_resolved_stdio_tokens_are_literal():
    profile = TransportProfile(
        profile_id="peer",
        command="python",
        args=("", " padded ", "${UNCHANGED}"),
        env={"EXACT": " padded "},
    )
    assert profile.args == ("", " padded ", "${UNCHANGED}")
    assert profile.env == {"EXACT": " padded "}


@pytest.mark.asyncio
@pytest.mark.parametrize("version", VERSIONS)
async def test_actual_stdio_profile_exchange_and_literal_arguments(tmp_path, version):
    from tldw_chatbook.MCP.client import MCPClient

    log = tmp_path / "wire.jsonl"
    program = tmp_path / "peer.py"
    program.write_text("""import json, sys
from pathlib import Path
log = Path(sys.argv[1])
assert sys.argv[2:] == ["", " padded "]
for line in sys.stdin:
    request = json.loads(line)
    with log.open("a") as out:
        out.write(json.dumps(request) + "\\n")
    if "id" not in request:
        continue
    method = request["method"]
    if method == "initialize":
        result = {"protocolVersion": request["params"]["protocolVersion"], "capabilities": {"tools": {}}, "serverInfo": {"name": "owned", "version": "1"}}
    elif method == "server/discover":
        result = {"supportedVersions": ["2026-07-28"], "capabilities": {"tools": {}}, "serverInfo": {"name": "owned", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "echo", "inputSchema": {"type": "object"}}]}
    elif method == "tools/call":
        result = {"content": [{"type": "text", "text": "ok"}], "structuredContent": {"pass": True}}
    else:
        result = {method.split("/")[0]: []}
    response = {"jsonrpc": "2.0", "id": request["id"], "result": result}
    if method == "tools/call" and request["params"].get("arguments", {}).get("batch"):
        response = [response]
    print(json.dumps(response), flush=True)
""")
    client = MCPClient()
    profile = TransportProfile(
        profile_id="peer",
        command=sys.executable,
        args=("-I", str(program), str(log), "", " padded "),
        protocol_version=version,
    )
    try:
        assert await client.connect_profile(profile)
        process = client.sessions["peer"].process
        before = [json.loads(line) for line in log.read_text().splitlines()]
        assert all(message["method"] != "tools/call" for message in before)
        result = await client.call_tool_result("peer", "echo", {})
        assert result.structured_content == {"pass": True}
        assert result.duplicate_keys_checked and result.dispatch_state == "settled"
        batch = await client.call_tool_result("peer", "echo", {"batch": True})
        assert (batch.transport_error is None) == (version == "2025-03-26")
        messages = [json.loads(line) for line in log.read_text().splitlines()]
        if version == VERSIONS[0]:
            assert messages[0]["method"] == "server/discover"
            assert all(
                message["params"]["_meta"]["io.modelcontextprotocol/protocolVersion"]
                == version
                for message in messages
            )
        else:
            assert messages[0]["method"] == "initialize"
            assert messages[0]["params"]["protocolVersion"] == version
            assert messages[1]["method"] == "notifications/initialized"
    finally:
        await client.disconnect_all()
    assert process.returncode is not None


@pytest.mark.parametrize(
    "query",
    [
        "",
        "?route=alpha&tag=one&tag=two&empty=&encoded=a%2Fb%26c%3Dd",
        "?token=visible-routing-value",
        "?route=${M2_ENDPOINT_ROUTE}",
    ],
)
def test_endpoint_query_is_literal_visible_configuration(monkeypatch, query):
    monkeypatch.setenv("M2_ENDPOINT_ROUTE", "host-value-must-not-appear")
    url = "https://example.invalid/mcp" + query
    profile = TransportProfile(profile_id="peer", transport="streamable_http", url=url)
    assert profile.url == url
    assert "host-value-must-not-appear" not in profile.url
