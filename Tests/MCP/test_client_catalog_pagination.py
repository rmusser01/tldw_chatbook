"""Bounded catalog pagination and exact resource metadata client contracts."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_chatbook.MCP import client as client_module


CATALOG_RESPONSES = [
    pytest.param("list_tools", "tools", "tools/list", "name", id="tools"),
    pytest.param(
        "list_resources", "resources", "resources/list", "uri", id="resources"
    ),
    pytest.param("list_prompts", "prompts", "prompts/list", "name", id="prompts"),
]


def _item(item_key: str, value: str) -> dict[str, Any]:
    if item_key == "resources":
        return {"uri": value, "name": value}
    return {"name": value}


def _scripted_connection(
    responder: Callable[[int, str, dict[str, Any]], dict[str, Any]],
) -> tuple[client_module._StdioJSONRPCConnection, list[tuple[str, dict[str, Any]]]]:
    connection = client_module._StdioJSONRPCConnection.__new__(
        client_module._StdioJSONRPCConnection
    )
    requests: list[tuple[str, dict[str, Any]]] = []

    async def request(method: str, params: dict[str, Any]) -> dict[str, Any]:
        copied_params = dict(params)
        requests.append((method, copied_params))
        return responder(len(requests), method, copied_params)

    connection.request = request  # type: ignore[method-assign]
    return connection, requests


async def _catalog_values(
    connection: client_module._StdioJSONRPCConnection,
    list_method: str,
    item_key: str,
    value_field: str,
) -> list[str]:
    response = await getattr(connection, list_method)()
    return [getattr(item, value_field) for item in getattr(response, item_key)]


def _assert_client_error(error: BaseException, expected_message: str) -> None:
    assert type(error).__name__ == "MCPClientError"
    assert str(error) == expected_message


@pytest.mark.parametrize(
    ("list_method", "item_key", "request_method", "value_field"),
    CATALOG_RESPONSES,
)
@pytest.mark.asyncio
async def test_catalogs_omit_first_cursor_forward_exact_cursor_and_preserve_order(
    list_method: str,
    item_key: str,
    request_method: str,
    value_field: str,
) -> None:
    pages = [
        {
            item_key: [_item(item_key, "first"), _item(item_key, "second")],
            "nextCursor": "cursor-exact",
        },
        {item_key: [_item(item_key, "third")]},
    ]
    connection, requests = _scripted_connection(
        lambda index, _method, _params: pages[index - 1]
    )

    assert await _catalog_values(connection, list_method, item_key, value_field) == [
        "first",
        "second",
        "third",
    ]
    assert requests == [
        (request_method, {}),
        (request_method, {"cursor": "cursor-exact"}),
    ]


@pytest.mark.parametrize(
    ("list_method", "item_key", "request_method", "value_field"),
    CATALOG_RESPONSES,
)
@pytest.mark.asyncio
async def test_catalog_null_cursor_terminates_without_another_request(
    list_method: str,
    item_key: str,
    request_method: str,
    value_field: str,
) -> None:
    connection, requests = _scripted_connection(
        lambda _index, _method, _params: {
            item_key: [_item(item_key, "only")],
            "nextCursor": None,
        }
    )

    assert await _catalog_values(connection, list_method, item_key, value_field) == [
        "only"
    ]
    assert requests == [(request_method, {})]


@pytest.mark.parametrize("cursor", ["", 7, False, [], {}])
@pytest.mark.asyncio
async def test_catalog_rejects_empty_or_non_string_cursor_without_payload_leakage(
    cursor: object,
) -> None:
    sentinel = "private-cursor-payload"
    connection, _requests = _scripted_connection(
        lambda _index, _method, _params: {
            "tools": [{"name": sentinel}],
            "nextCursor": cursor,
        }
    )

    with pytest.raises(Exception) as exc_info:
        await connection.list_tools()

    _assert_client_error(exc_info.value, "Invalid MCP catalog cursor")
    assert sentinel not in str(exc_info.value)
    assert repr(cursor) not in str(exc_info.value)


@pytest.mark.asyncio
async def test_catalog_rejects_repeated_cursor_instead_of_returning_partial_items() -> (
    None
):
    pages = [
        {"tools": [{"name": "first"}], "nextCursor": "repeat"},
        {"tools": [{"name": "private-second"}], "nextCursor": "repeat"},
    ]
    connection, requests = _scripted_connection(
        lambda index, _method, _params: pages[index - 1]
    )

    with pytest.raises(Exception) as exc_info:
        await connection.list_tools()

    _assert_client_error(exc_info.value, "Repeated MCP catalog cursor")
    assert "private-second" not in str(exc_info.value)
    assert requests == [
        ("tools/list", {}),
        ("tools/list", {"cursor": "repeat"}),
    ]


@pytest.mark.parametrize("items", [None, "private-items", {"private": "items"}])
@pytest.mark.asyncio
async def test_catalog_rejects_non_list_item_array_without_payload_leakage(
    items: object,
) -> None:
    connection, _requests = _scripted_connection(
        lambda _index, _method, _params: {"tools": items}
    )

    with pytest.raises(Exception) as exc_info:
        await connection.list_tools()

    _assert_client_error(exc_info.value, "Invalid MCP catalog items")
    assert "private" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_catalog_accepts_exactly_100_pages() -> None:
    def respond(index: int, _method: str, _params: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {"tools": [{"name": f"tool-{index}"}]}
        if index < 100:
            result["nextCursor"] = f"page-{index + 1}"
        return result

    connection, requests = _scripted_connection(respond)

    values = await _catalog_values(connection, "list_tools", "tools", "name")

    assert len(values) == 100
    assert values == [f"tool-{index}" for index in range(1, 101)]
    assert len(requests) == 100
    assert requests[0] == ("tools/list", {})
    assert requests[-1] == ("tools/list", {"cursor": "page-100"})


@pytest.mark.asyncio
async def test_catalog_rejects_page_101_instead_of_returning_100_page_partial() -> None:
    def respond(index: int, _method: str, _params: dict[str, Any]) -> dict[str, Any]:
        return {
            "tools": [{"name": f"tool-{index}"}],
            "nextCursor": f"page-{index + 1}",
        }

    connection, requests = _scripted_connection(respond)

    with pytest.raises(Exception) as exc_info:
        await connection.list_tools()

    _assert_client_error(exc_info.value, "MCP catalog page limit exceeded")
    assert len(requests) == 100


@pytest.mark.asyncio
async def test_catalog_accepts_exactly_10_000_items() -> None:
    items = [{"name": f"tool-{index}"} for index in range(10_000)]
    connection, _requests = _scripted_connection(
        lambda _index, _method, _params: {"tools": items}
    )

    values = await _catalog_values(connection, "list_tools", "tools", "name")

    assert len(values) == 10_000
    assert values[0] == "tool-0"
    assert values[-1] == "tool-9999"


@pytest.mark.asyncio
async def test_catalog_rejects_item_10_001_instead_of_returning_partial_items() -> None:
    first_page = [{"name": f"tool-{index}"} for index in range(10_000)]
    pages = [
        {"tools": first_page, "nextCursor": "more"},
        {"tools": [{"name": "private-item-10001"}]},
    ]
    connection, requests = _scripted_connection(
        lambda index, _method, _params: pages[index - 1]
    )

    with pytest.raises(Exception) as exc_info:
        await connection.list_tools()

    _assert_client_error(exc_info.value, "MCP catalog item limit exceeded")
    assert "private-item-10001" not in str(exc_info.value)
    assert requests == [
        ("tools/list", {}),
        ("tools/list", {"cursor": "more"}),
    ]


@pytest.mark.asyncio
async def test_low_level_resource_read_copies_exact_result_metadata() -> None:
    metadata = {
        "tldw.chatbook/continuation": {"hasMore": True, "nextUri": "note://2"},
        "tldw.chatbook/resource": {"kind": "note"},
    }
    connection, _requests = _scripted_connection(
        lambda _index, _method, _params: {
            "contents": [{"uri": "note://1", "mimeType": "text/plain", "text": "body"}],
            "_meta": metadata,
        }
    )

    result = await connection.read_resource("note://1")

    assert result._meta == metadata
    assert result._meta is not metadata
    metadata["late-mutation"] = True
    assert "late-mutation" not in result._meta


@pytest.mark.parametrize("metadata", [None, pytest.param("absent", id="absent")])
@pytest.mark.asyncio
async def test_low_level_resource_read_defaults_missing_or_null_metadata_to_empty(
    metadata: object,
) -> None:
    payload: dict[str, Any] = {"contents": []}
    if metadata is None:
        payload["_meta"] = None
    connection, _requests = _scripted_connection(
        lambda _index, _method, _params: payload
    )

    result = await connection.read_resource("note://1")

    assert result._meta == {}


@pytest.mark.parametrize("metadata", ["private-metadata", [], 7, True])
@pytest.mark.asyncio
async def test_low_level_resource_read_rejects_invalid_metadata_without_payload_leakage(
    metadata: object,
) -> None:
    connection, _requests = _scripted_connection(
        lambda _index, _method, _params: {"contents": [], "_meta": metadata}
    )

    with pytest.raises(Exception) as exc_info:
        await connection.read_resource("note://1")

    _assert_client_error(exc_info.value, "Invalid MCP resource metadata")
    assert "private-metadata" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_high_level_resource_read_preserves_exact_metadata_key_and_copies_it() -> (
    None
):
    metadata = {
        "tldw.chatbook/continuation": {"hasMore": False, "nextUri": None},
        "tldw.chatbook/resource": {"kind": "note"},
    }

    class Session:
        async def read_resource(self, resource_uri: str) -> SimpleNamespace:
            assert resource_uri == "note://1"
            return SimpleNamespace(
                contents=[SimpleNamespace(text="body", mimeType="text/markdown")],
                _meta=metadata,
            )

    client = client_module.MCPClient.__new__(client_module.MCPClient)
    client.sessions = {"server": Session()}  # type: ignore[dict-item]

    result = await client.read_resource("server", "note://1")

    assert result == {
        "uri": "note://1",
        "content": "body",
        "mimeType": "text/markdown",
        "_meta": metadata,
    }
    assert result["_meta"] is not metadata
    result["_meta"]["late-mutation"] = True
    assert "late-mutation" not in metadata


@pytest.mark.asyncio
async def test_high_level_resource_read_rejects_invalid_metadata_without_payload_leakage() -> (
    None
):
    class Session:
        async def read_resource(self, resource_uri: str) -> SimpleNamespace:
            assert resource_uri == "note://1"
            return SimpleNamespace(contents=[], _meta="private-metadata")

    client = client_module.MCPClient.__new__(client_module.MCPClient)
    client.sessions = {"server": Session()}  # type: ignore[dict-item]

    result = await client.read_resource("server", "note://1")

    assert result == {"error": "Invalid MCP resource metadata"}
    assert "private-metadata" not in str(result)
