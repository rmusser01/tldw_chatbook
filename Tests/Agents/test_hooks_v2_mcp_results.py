"""Strict original-wire MCP hook result qualification."""

import json

import pytest

from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers
from tldw_chatbook.MCP.tool_results import decode_protocol_frame, parse_tool_result


@pytest.fixture
def handler():
    return parse_handlers(
        [
            {
                "id": "mcp-hook",
                "event": "PreToolUse",
                "type": "mcp_tool",
                "server": "local:test",
                "tool": "check",
                "effects": ["deny"],
                "input": {},
            }
        ]
    )[0]


@pytest.fixture
def typed_result():
    def make(*, is_error=False, structured=None, content=None, metadata=None):
        payload = {
            "isError": is_error,
            "content": content or [],
            "_meta": metadata or {},
        }
        if structured is not None:
            payload["structuredContent"] = structured
        frame = json.dumps({"jsonrpc": "2.0", "id": 1, "result": payload}).encode()
        return parse_tool_result(decode_protocol_frame(frame)["result"])

    return make


def test_strict_wire_positive_precondition(typed_result):
    result = typed_result(structured={"version": 2, "decision": "pass"})
    assert result.duplicate_keys_checked
    assert json.loads(result.encoded_payload)["structuredContent"]["decision"] == "pass"
    assert not result.model_copy().duplicate_keys_checked


def test_error_payload_cannot_return_a_pass(handler, typed_result):
    from tldw_chatbook.Agents.hooks_v2.mcp_results import normalize_hook_result

    error = typed_result(is_error=True, structured={"version": 2, "decision": "pass"})
    with pytest.raises(ValueError):
        normalize_hook_result(error, handler)
    good = typed_result(is_error=False, structured={"version": 2, "decision": "pass"})
    assert normalize_hook_result(good, handler).decision == "pass"


@pytest.mark.parametrize("form", ["structured", "text", "mirror", "empty"])
def test_documented_success_forms(handler, typed_result, form):
    from tldw_chatbook.Agents.hooks_v2.mcp_results import normalize_hook_result

    body = {"version": 2, "decision": "pass"}
    result = typed_result(
        structured=body if form in {"structured", "mirror"} else None,
        content=(
            [{"type": "text", "text": " \n" + json.dumps(body) + "\n "}]
            if form in {"text", "mirror"}
            else []
        ),
    )
    assert normalize_hook_result(result, handler).decision == "pass"


@pytest.mark.parametrize(
    "content",
    [
        [{"type": "text", "text": ""}],
        [{"type": "text", "text": "   "}],
        [{"type": "image", "data": "unused"}],
        [{"type": "text", "text": '{"version":2,"decision":"pass","decision":"deny"}'}],
        [{"type": "text", "text": '```json\n{"version":2,"decision":"pass"}\n```'}],
        [{"type": "text", "text": "{}{}"}],
        [{"type": "text", "text": "[]"}],
        [{"type": "text", "text": '{"version":2,"decision":"pass"}'}] * 2,
    ],
)
def test_unsupported_text_shapes_refused(handler, typed_result, content):
    from tldw_chatbook.Agents.hooks_v2.mcp_results import normalize_hook_result

    with pytest.raises(ValueError):
        normalize_hook_result(typed_result(content=content), handler)


@pytest.mark.parametrize(
    "structured",
    [{}, {"version": 2, "decision": "deny"}, {"version": True, "decision": "pass"}],
)
def test_invalid_or_conflicting_structured_cannot_fall_back(
    handler, typed_result, structured
):
    from tldw_chatbook.Agents.hooks_v2.mcp_results import normalize_hook_result

    with pytest.raises(ValueError):
        normalize_hook_result(
            typed_result(
                structured=structured,
                content=[{"type": "text", "text": '{"version":2,"decision":"pass"}'}],
            ),
            handler,
        )


@pytest.mark.parametrize("flag", [1, 0, "false", None])
def test_nonboolean_error_is_refused_by_actual_decode(typed_result, flag):
    with pytest.raises(ValueError):
        typed_result(is_error=flag, structured={"version": 2, "decision": "pass"})


def test_duplicate_protocol_metadata_refused():
    with pytest.raises(ValueError):
        decode_protocol_frame(
            b'{"jsonrpc":"2.0","id":1,"result":{"_meta":{"x":1,"x":2}}}'
        )


@pytest.mark.parametrize("mutation", ["copy", "nested", "constructor", "predecoded"])
def test_missing_original_provenance_refused(handler, typed_result, mutation):
    from tldw_chatbook.Agents.hooks_v2.mcp_results import normalize_hook_result
    from tldw_chatbook.MCP.tool_results import MCPToolResult

    result = typed_result(structured={"version": 2, "decision": "pass"})
    if mutation == "copy":
        result = result.model_copy(deep=True)
    elif mutation == "nested":
        result.metadata["changed"] = True
    elif mutation == "constructor":
        result = MCPToolResult(structured_content={"version": 2, "decision": "pass"})
    else:
        result = parse_tool_result(json.loads(result.encoded_payload))
    with pytest.raises(ValueError, match="unqualified"):
        normalize_hook_result(result, handler)


def test_full_payload_includes_metadata_and_original_whitespace(handler, typed_result):
    from tldw_chatbook.Agents.hooks_v2.mcp_results import normalize_hook_result

    with pytest.raises(ValueError, match="overflow"):
        normalize_hook_result(
            typed_result(
                structured={"version": 2, "decision": "pass"},
                metadata={"x": "x" * 16384},
            ),
            handler,
        )
    body = b'{"structuredContent":{"version":2,"decision":"pass"}}'
    for length in (16384, 16385):
        raw = body[:-1] + b" " * (length - len(body)) + b"}"
        result = parse_tool_result(
            decode_protocol_frame(b'{"jsonrpc":"2.0","id":1,"result":' + raw + b"}")[
                "result"
            ]
        )
        if length == 16384:
            assert normalize_hook_result(result, handler).decision == "pass"
        else:
            with pytest.raises(ValueError, match="overflow"):
                normalize_hook_result(result, handler)


def test_required_context_cannot_accept_empty(typed_result):
    from tldw_chatbook.Agents.hooks_v2.mcp_results import normalize_hook_result

    handler = parse_handlers(
        [
            {
                "id": "context",
                "event": "SessionStart",
                "type": "mcp_tool",
                "server": "local:test",
                "tool": "check",
                "effects": ["context"],
                "require_context": True,
            }
        ]
    )[0]
    with pytest.raises(ValueError):
        normalize_hook_result(typed_result(), handler)
    result = typed_result(
        structured={
            "version": 2,
            "decision": "pass",
            "context": [{"text": "qualified", "lifetime": "runtime"}],
        }
    )
    assert normalize_hook_result(result, handler).context[0].text == "qualified"
