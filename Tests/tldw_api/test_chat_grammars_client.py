from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    ChatGrammarCreate,
    ChatGrammarListResponse,
    ChatGrammarResponse,
    ChatGrammarUpdate,
    TLDWAPIClient,
)


NOW = "2026-04-25T12:00:00Z"


def _grammar_payload(**overrides) -> dict:
    payload = {
        "id": "grammar-1",
        "name": "JSON object",
        "description": "Constrain output to a JSON object.",
        "grammar_text": "root ::= object",
        "validation_status": "unchecked",
        "validation_error": None,
        "last_validated_at": None,
        "is_archived": False,
        "created_at": NOW,
        "updated_at": NOW,
        "version": 1,
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_chat_grammars_client_routes_crud_and_soft_delete(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            _grammar_payload(),
            {"items": [_grammar_payload()], "total": 1},
            _grammar_payload(),
            _grammar_payload(name="Strict JSON", version=2),
            {},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    created = await client.create_chat_grammar(
        ChatGrammarCreate(
            name="JSON object",
            description="Constrain output to a JSON object.",
            grammar_text="root ::= object",
        )
    )
    listed = await client.list_chat_grammars(include_archived=True, limit=25, offset=5)
    fetched = await client.get_chat_grammar("grammar-1", include_archived=True)
    updated = await client.update_chat_grammar("grammar-1", ChatGrammarUpdate(name="Strict JSON", version=1))
    deleted = await client.delete_chat_grammar("grammar-1", hard_delete=True)

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/grammars")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "name": "JSON object",
        "description": "Constrain output to a JSON object.",
        "grammar_text": "root ::= object",
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/grammars")
    assert mocked.await_args_list[1].kwargs["params"] == {
        "include_archived": True,
        "limit": 25,
        "offset": 5,
    }
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/grammars/grammar-1")
    assert mocked.await_args_list[2].kwargs["params"] == {"include_archived": True}
    assert mocked.await_args_list[3].args[:2] == ("PATCH", "/api/v1/grammars/grammar-1")
    assert mocked.await_args_list[3].kwargs["json_data"] == {"version": 1, "name": "Strict JSON"}
    assert mocked.await_args_list[4].args[:2] == ("DELETE", "/api/v1/grammars/grammar-1")
    assert mocked.await_args_list[4].kwargs["params"] == {"hard_delete": True}

    assert isinstance(created, ChatGrammarResponse)
    assert isinstance(listed, ChatGrammarListResponse)
    assert isinstance(fetched, ChatGrammarResponse)
    assert updated.version == 2
    assert deleted is True
