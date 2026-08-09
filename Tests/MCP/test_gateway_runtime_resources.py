"""Strict resource routing and continuation tests for the MCP gateway adapter."""

from __future__ import annotations

import base64
import copy
import json
import re
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

import pytest

gateway = pytest.importorskip(
    "mcp_unified.gateway", reason="mcp-unified extra not installed"
)
GatewayApplicationError = gateway.GatewayApplicationError
GatewayRequestContext = gateway.GatewayRequestContext

from tldw_chatbook.MCP.gateway_runtime import (  # noqa: E402
    CONTINUATION_QUERY_KEY,
    MAX_RESOURCE_CHUNK_BYTES,
    ChatbookGatewayRuntime,
)


def _context() -> GatewayRequestContext:
    return GatewayRequestContext(request_id="resource-test")


def _runtime() -> ChatbookGatewayRuntime:
    return ChatbookGatewayRuntime(
        name="tldw_chatbook",
        version="0.1.0",
        tool_descriptors=[],
    )


def _resource_result(
    scheme: str,
    identifier: str,
    content: str,
    metadata: object | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        # The existing real handlers interpolate the decoded identifier.
        "uri": f"{scheme}://{identifier}",
        "name": f"{scheme} resource",
        "mimeType": "text/plain",
        "content": content,
    }
    if metadata is not None:
        result["metadata"] = metadata
    return result


def _register_resources(
    runtime: ChatbookGatewayRuntime,
    *,
    content: dict[str, str] | None = None,
    metadata: dict[str, object | None] | None = None,
    calls: list[tuple[str, str]] | None = None,
) -> None:
    texts = content if content is not None else {}
    metadata_by_scheme = metadata if metadata is not None else {}

    def result(scheme: str, identifier: str) -> dict[str, Any]:
        if calls is not None:
            calls.append((scheme, identifier))
        return _resource_result(
            scheme,
            identifier,
            texts.get(scheme, f"{scheme}:{identifier}"),
            metadata_by_scheme.get(scheme),
        )

    @runtime.resource("conversation://{conversation_id}")
    async def get_conversation(conversation_id: str) -> dict[str, Any]:
        """Get a conversation by ID."""
        return result("conversation", conversation_id)

    @runtime.resource("note://{note_id}")
    async def get_note(note_id: str) -> dict[str, Any]:
        """Get a note by ID."""
        return result("note", note_id)

    @runtime.resource("character://{character_id}")
    async def get_character(character_id: str) -> dict[str, Any]:
        """Get a character profile by ID."""
        return result("character", character_id)

    @runtime.resource("media://{media_id}")
    async def get_media(media_id: str) -> dict[str, Any]:
        """Get media content by ID."""
        return result("media", media_id)

    @runtime.resource("rag-chunk://{chunk_uuid}")
    async def get_rag_chunk(chunk_uuid: str) -> dict[str, Any]:
        """Get a RAG chunk by UUID."""
        return result("rag-chunk", chunk_uuid)


def _ready_runtime(
    *,
    content: dict[str, str] | None = None,
    metadata: dict[str, object | None] | None = None,
    calls: list[tuple[str, str]] | None = None,
) -> ChatbookGatewayRuntime:
    runtime = _runtime()
    _register_resources(runtime, content=content, metadata=metadata, calls=calls)

    @runtime.list_resources()
    async def list_resources() -> list[dict[str, Any]]:
        return []

    runtime.finalize()
    return runtime


def _token(next_uri: str) -> str:
    values = parse_qs(urlsplit(next_uri).query, keep_blank_values=True)
    assert list(values) == [CONTINUATION_QUERY_KEY]
    assert len(values[CONTINUATION_QUERY_KEY]) == 1
    return values[CONTINUATION_QUERY_KEY][0]


def _decode_token(token: str) -> dict[str, Any]:
    padding = "=" * (-len(token) % 4)
    return json.loads(base64.urlsafe_b64decode(token + padding))


def _encode_token(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _with_token(base_uri: str, token: str) -> str:
    parsed = urlsplit(base_uri)
    return urlunsplit(parsed._replace(query=urlencode({CONTINUATION_QUERY_KEY: token})))


@pytest.mark.asyncio
async def test_lists_exact_ordered_templates_and_routes_decoded_identifiers() -> None:
    calls: list[tuple[str, str]] = []
    runtime = _ready_runtime(calls=calls)

    templates = await runtime.list_resource_templates(_context())

    assert templates == [
        {
            "uriTemplate": "conversation://{conversation_id}",
            "name": "get_conversation",
            "description": "Get a conversation by ID.",
        },
        {
            "uriTemplate": "note://{note_id}",
            "name": "get_note",
            "description": "Get a note by ID.",
        },
        {
            "uriTemplate": "character://{character_id}",
            "name": "get_character",
            "description": "Get a character profile by ID.",
        },
        {
            "uriTemplate": "media://{media_id}",
            "name": "get_media",
            "description": "Get media content by ID.",
        },
        {
            "uriTemplate": "rag-chunk://{chunk_uuid}",
            "name": "get_rag_chunk",
            "description": "Get a RAG chunk by UUID.",
        },
    ]

    uris = [
        "conversation://conv-1",
        "note://note%20one",
        "character://42",
        "media://7",
        "rag-chunk://chunk-uuid",
    ]
    for uri in uris:
        result = await runtime.read_resource(uri, _context())
        assert result["contents"][0]["uri"] == uri
        assert result["contents"][0]["text"]

    assert calls == [
        ("conversation", "conv-1"),
        ("note", "note one"),
        ("character", "42"),
        ("media", "7"),
        ("rag-chunk", "chunk-uuid"),
    ]


def test_resource_registration_rejects_duplicate_unknown_and_mismatched_templates() -> (
    None
):
    runtime = _runtime()

    @runtime.resource("conversation://{conversation_id}")
    async def get_conversation(conversation_id: str) -> dict[str, Any]:
        return _resource_result("conversation", conversation_id, "ok", None)

    with pytest.raises(ValueError, match="duplicate resource template"):

        @runtime.resource("conversation://{conversation_id}")
        async def duplicate(conversation_id: str) -> dict[str, Any]:
            return _resource_result("conversation", conversation_id, "ok", None)

    with pytest.raises(ValueError, match="resource template"):
        runtime.resource("unknown://{unknown_id}")

    with pytest.raises(ValueError, match="resource template"):
        runtime.resource("conversation://{note_id}")

    with pytest.raises(ValueError, match="identifier"):

        @runtime.resource("note://{note_id}")
        async def mismatched_identifier(conversation_id: str) -> dict[str, Any]:
            return _resource_result("note", conversation_id, "ok", None)


def test_partial_resource_template_set_fails_finalization() -> None:
    runtime = _runtime()

    @runtime.resource("conversation://{conversation_id}")
    async def get_conversation(conversation_id: str) -> dict[str, Any]:
        return _resource_result("conversation", conversation_id, "ok", None)

    with pytest.raises(ValueError, match="resource template"):
        runtime.finalize()


def test_all_resource_templates_without_dynamic_catalog_fail_finalization() -> None:
    runtime = _runtime()
    _register_resources(runtime)

    with pytest.raises(ValueError, match="resource.*catalog|catalog.*resource"):
        runtime.finalize()


def test_dynamic_catalog_without_resource_templates_fails_finalization() -> None:
    runtime = _runtime()

    @runtime.list_resources()
    async def list_resources() -> list[dict[str, Any]]:
        return []

    with pytest.raises(ValueError, match="resource.*template|template.*resource"):
        runtime.finalize()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "uri",
    [
        pytest.param("unknown://value", id="unknown-scheme"),
        pytest.param("conversation://value#fragment", id="fragment"),
        pytest.param("conversation://value#", id="empty-fragment"),
        pytest.param("conversation://value/extra", id="extra-path"),
        pytest.param("conversation://", id="empty-identifier"),
        pytest.param("conversation:///value", id="identifier-in-path"),
        pytest.param("conversation://value%", id="trailing-percent"),
        pytest.param("conversation://value%2", id="short-percent"),
        pytest.param("conversation://value%GG", id="nonhex-percent"),
        pytest.param("conversation://value?unknown=1", id="unknown-query"),
        pytest.param(
            "conversation://value?tldw_continue=a&tldw_continue=b",
            id="duplicate-query",
        ),
        pytest.param("conversation://value?tldw_continue=", id="empty-token"),
        pytest.param("conversation://note://value", id="template-mismatch"),
    ],
)
async def test_invalid_uri_is_rejected_before_handler_invocation(uri: str) -> None:
    calls: list[tuple[str, str]] = []
    runtime = _ready_runtime(calls=calls)

    with pytest.raises(GatewayApplicationError) as exc_info:
        await runtime.read_resource(uri, _context())

    assert exc_info.value.kind == "resource"
    assert len(exc_info.value.public_message) <= 512
    assert uri not in exc_info.value.public_message
    assert calls == []


@pytest.mark.asyncio
async def test_dynamic_catalog_preserves_fields_order_and_defensive_copies() -> None:
    runtime = _runtime()
    _register_resources(runtime)
    source = [
        {
            "uri": "conversation://first",
            "name": "First",
            "description": "First conversation",
            "mimeType": "text/markdown",
            "ignored": "not a canonical resource field",
        },
        {
            "uri": "note://second",
            "name": "Second",
            "mimeType": "text/plain",
        },
    ]

    @runtime.list_resources()
    async def list_resources() -> list[dict[str, Any]]:
        return source

    runtime.finalize()

    first = await runtime.list_resources(_context())
    first[0]["name"] = "mutated"
    templates = await runtime.list_resource_templates(_context())
    original_templates = copy.deepcopy(templates)
    templates[0]["name"] = "mutated"
    templates[0]["description"] = "mutated"
    templates.append({"uriTemplate": "bad://{id}", "name": "bad"})

    assert await runtime.list_resources(_context()) == [
        {
            "uri": "conversation://first",
            "name": "First",
            "description": "First conversation",
            "mimeType": "text/markdown",
        },
        {
            "uri": "note://second",
            "name": "Second",
            "mimeType": "text/plain",
        },
    ]
    assert await runtime.list_resource_templates(_context()) == original_templates
    assert source[0]["name"] == "First"


def test_duplicate_dynamic_catalog_registration_is_rejected() -> None:
    runtime = _runtime()

    @runtime.list_resources()
    async def first() -> list[dict[str, Any]]:
        return []

    with pytest.raises(ValueError, match="resource list handler"):

        @runtime.list_resources()
        async def second() -> list[dict[str, Any]]:
            return []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "expected_text", "expected_more"),
    [
        pytest.param("small", "small", False, id="ascii-small"),
        pytest.param(
            "a" * MAX_RESOURCE_CHUNK_BYTES,
            "a" * MAX_RESOURCE_CHUNK_BYTES,
            False,
            id="exact-limit",
        ),
        pytest.param(
            "a" * (MAX_RESOURCE_CHUNK_BYTES - 1) + "😀tail",
            "a" * (MAX_RESOURCE_CHUNK_BYTES - 1),
            True,
            id="multibyte-boundary",
        ),
        pytest.param(
            "a" * MAX_RESOURCE_CHUNK_BYTES + "tail",
            "a" * MAX_RESOURCE_CHUNK_BYTES,
            True,
            id="over-limit",
        ),
    ],
)
async def test_resource_result_is_one_bounded_text_block_with_exact_counts(
    text: str,
    expected_text: str,
    expected_more: bool,
) -> None:
    runtime = _ready_runtime(content={"conversation": text})

    result = await runtime.read_resource("conversation://one", _context())

    assert result["contents"] == [
        {
            "uri": "conversation://one",
            "mimeType": "text/plain",
            "text": expected_text,
        }
    ]
    assert len(expected_text.encode("utf-8")) <= MAX_RESOURCE_CHUNK_BYTES
    continuation = result["_meta"]["tldw.chatbook/continuation"]
    assert continuation == {
        "startChar": 0,
        "endChar": len(expected_text),
        "totalChars": len(text),
        "totalBytes": len(text.encode("utf-8")),
        "returnedBytes": len(expected_text.encode("utf-8")),
        "hasMore": expected_more,
        "nextUri": continuation["nextUri"] if expected_more else None,
    }
    if expected_more:
        assert continuation["nextUri"].startswith(
            f"conversation://one?{CONTINUATION_QUERY_KEY}="
        )


@pytest.mark.asyncio
async def test_continuation_token_is_bounded_url_safe_state_not_a_mac() -> None:
    text = "a" * (MAX_RESOURCE_CHUNK_BYTES + 20)
    runtime = _ready_runtime(content={"conversation": text})
    first = await runtime.read_resource("conversation://one", _context())
    next_uri = first["_meta"]["tldw.chatbook/continuation"]["nextUri"]
    token = _token(next_uri)

    assert len(token) <= 512
    assert re.fullmatch(r"[A-Za-z0-9_-]+", token)
    payload = _decode_token(token)
    assert payload.keys() == {"v", "o", "b", "c"}
    assert payload["v"] == 1
    assert payload["o"] == MAX_RESOURCE_CHUNK_BYTES
    assert re.fullmatch(r"[0-9a-f]{64}", payload["b"])
    assert re.fullmatch(r"[0-9a-f]{64}", payload["c"])

    # The cursor is integrity/version/state, explicitly not HMAC authorization:
    # a client can re-encode another in-range offset with the same public state.
    payload["o"] += 1
    forged_uri = _with_token("conversation://one", _encode_token(payload))
    forged = await runtime.read_resource(forged_uri, _context())
    assert forged["_meta"]["tldw.chatbook/continuation"]["startChar"] == payload["o"]
    assert forged["contents"][0]["text"] == text[payload["o"] :]


@pytest.mark.asyncio
async def test_continuations_reconstruct_exact_multibyte_text() -> None:
    text = ("é😀" * 70_000) + "the end"
    runtime = _ready_runtime(content={"conversation": text})
    uri: str | None = "conversation://one"
    chunks: list[str] = []
    starts: list[int] = []

    while uri is not None:
        result = await runtime.read_resource(uri, _context())
        chunks.append(result["contents"][0]["text"])
        continuation = result["_meta"]["tldw.chatbook/continuation"]
        starts.append(continuation["startChar"])
        uri = continuation["nextUri"]

    assert len(chunks) >= 2
    assert starts == sorted(starts)
    assert "".join(chunks) == text


@pytest.mark.asyncio
async def test_wrong_base_continuation_fails_before_other_handler_call() -> None:
    calls: list[tuple[str, str]] = []
    text = "a" * (MAX_RESOURCE_CHUNK_BYTES + 1)
    runtime = _ready_runtime(content={"conversation": text, "note": text}, calls=calls)
    first = await runtime.read_resource("conversation://one", _context())
    calls.clear()
    wrong_base_uri = _with_token(
        "note://one",
        _token(first["_meta"]["tldw.chatbook/continuation"]["nextUri"]),
    )

    with pytest.raises(GatewayApplicationError) as exc_info:
        await runtime.read_resource(wrong_base_uri, _context())

    assert exc_info.value.reason_code == "invalid_resource_uri"
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "token",
    [
        pytest.param("not*url-safe", id="invalid-alphabet"),
        pytest.param("e30", id="missing-state"),
        pytest.param("A" * 513, id="over-bound"),
    ],
)
async def test_malformed_continuation_fails_closed(token: str) -> None:
    calls: list[tuple[str, str]] = []
    runtime = _ready_runtime(calls=calls)

    with pytest.raises(GatewayApplicationError) as exc_info:
        await runtime.read_resource(
            _with_token("conversation://one", token), _context()
        )

    assert exc_info.value.reason_code == "invalid_resource_uri"
    assert calls == []


@pytest.mark.asyncio
async def test_out_of_range_continuation_fails_closed() -> None:
    text = "a" * (MAX_RESOURCE_CHUNK_BYTES + 2)
    runtime = _ready_runtime(content={"conversation": text})
    first = await runtime.read_resource("conversation://one", _context())
    payload = _decode_token(
        _token(first["_meta"]["tldw.chatbook/continuation"]["nextUri"])
    )
    payload["o"] = len(text)

    with pytest.raises(GatewayApplicationError) as exc_info:
        await runtime.read_resource(
            _with_token("conversation://one", _encode_token(payload)), _context()
        )

    assert exc_info.value.reason_code == "invalid_resource_uri"


@pytest.mark.asyncio
@pytest.mark.parametrize("unsupported_version", [1.0, 2])
async def test_unsupported_continuation_version_fails_closed(
    unsupported_version: object,
) -> None:
    text = "a" * (MAX_RESOURCE_CHUNK_BYTES + 2)
    runtime = _ready_runtime(content={"conversation": text})
    first = await runtime.read_resource("conversation://one", _context())
    payload = _decode_token(
        _token(first["_meta"]["tldw.chatbook/continuation"]["nextUri"])
    )
    payload["v"] = unsupported_version

    with pytest.raises(GatewayApplicationError) as exc_info:
        await runtime.read_resource(
            _with_token("conversation://one", _encode_token(payload)), _context()
        )

    assert exc_info.value.reason_code == "invalid_resource_uri"


@pytest.mark.asyncio
async def test_duplicate_continuation_state_key_fails_closed() -> None:
    text = "a" * (MAX_RESOURCE_CHUNK_BYTES + 2)
    runtime = _ready_runtime(content={"conversation": text})
    first = await runtime.read_resource("conversation://one", _context())
    payload = _decode_token(
        _token(first["_meta"]["tldw.chatbook/continuation"]["nextUri"])
    )
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))[:-1]
    duplicate_key_token = (
        base64.urlsafe_b64encode(f'{raw},"v":1}}'.encode()).decode().rstrip("=")
    )

    with pytest.raises(GatewayApplicationError) as exc_info:
        await runtime.read_resource(
            _with_token("conversation://one", duplicate_key_token), _context()
        )

    assert exc_info.value.reason_code == "invalid_resource_uri"


@pytest.mark.asyncio
async def test_changed_content_continuation_requires_restart() -> None:
    state = {"conversation": "a" * (MAX_RESOURCE_CHUNK_BYTES + 2)}
    runtime = _ready_runtime(content=state)
    first = await runtime.read_resource("conversation://one", _context())
    state["conversation"] += "changed"

    with pytest.raises(GatewayApplicationError) as exc_info:
        await runtime.read_resource(
            first["_meta"]["tldw.chatbook/continuation"]["nextUri"], _context()
        )

    assert exc_info.value.reason_code == "resource_changed"
    assert "restart" in exc_info.value.public_message.lower()
    assert len(exc_info.value.public_message) <= 512


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata", "expected_resource_metadata"),
    [
        pytest.param(
            {"count": 2, "nested": {"ok": True}},
            {"count": 2, "nested": {"ok": True}},
            id="non-empty",
        ),
        pytest.param({}, None, id="empty"),
        pytest.param(None, None, id="absent"),
    ],
)
async def test_handler_metadata_is_only_namespaced_when_non_empty(
    metadata: object | None,
    expected_resource_metadata: dict[str, Any] | None,
) -> None:
    runtime = _ready_runtime(metadata={"conversation": metadata})

    result = await runtime.read_resource("conversation://one", _context())

    expected_meta = {
        "tldw.chatbook/continuation": {
            "startChar": 0,
            "endChar": len("conversation:one"),
            "totalChars": len("conversation:one"),
            "totalBytes": len("conversation:one"),
            "returnedBytes": len("conversation:one"),
            "hasMore": False,
            "nextUri": None,
        }
    }
    if expected_resource_metadata is not None:
        expected_meta["tldw.chatbook/resource"] = expected_resource_metadata
    assert result["_meta"] == expected_meta
    assert "count" not in result["_meta"]


def _deep_metadata() -> dict[str, Any]:
    root: dict[str, Any] = {}
    current = root
    for _ in range(70):
        child: dict[str, Any] = {}
        current["next"] = child
        current = child
    return root


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param(["not", "a", "mapping"], id="not-mapping"),
        pytest.param({"value": object()}, id="non-json"),
        pytest.param({"value": float("nan")}, id="non-finite"),
        pytest.param(_deep_metadata(), id="too-deep"),
    ],
)
async def test_invalid_handler_metadata_fails_closed(metadata: object) -> None:
    sentinel = "/private/secret/resource-metadata"
    invalid = copy.deepcopy(metadata)
    if isinstance(invalid, dict) and "value" in invalid:
        invalid["sentinel"] = sentinel
    runtime = _ready_runtime(metadata={"conversation": invalid})

    with pytest.raises(GatewayApplicationError) as exc_info:
        await runtime.read_resource("conversation://one", _context())

    assert exc_info.value.reason_code == "invalid_resource_result"
    assert sentinel not in exc_info.value.public_message
    assert len(exc_info.value.public_message) <= 512
