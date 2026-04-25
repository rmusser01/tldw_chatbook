from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api.chat_dictionary_schemas import (
    BulkDictionaryEntryOperationRequest,
    ChatDictionaryCreateRequest,
    ChatDictionaryUpdateRequest,
    DictionaryEntryCreateRequest,
    DictionaryEntryReorderRequest,
    ImportDictionaryJSONRequest,
    ImportDictionaryMarkdownRequest,
    ProcessChatDictionariesRequest,
)
from tldw_chatbook.tldw_api.client import TLDWAPIClient


@pytest.mark.asyncio
async def test_chat_dictionary_client_routes_dictionary_crud(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"id": 7, "name": "Lore"})
    monkeypatch.setattr(client, "_request", mocked)

    await client.list_chat_dictionaries(include_inactive=True, include_usage=True)
    await client.create_chat_dictionary(ChatDictionaryCreateRequest(name="Lore"))
    await client.get_chat_dictionary(7)
    await client.update_chat_dictionary(7, ChatDictionaryUpdateRequest(name="Lore v2", version=2))
    await client.delete_chat_dictionary(7, hard_delete=True)

    assert [call.args[:2] for call in mocked.await_args_list] == [
        ("GET", "/api/v1/chat/dictionaries"),
        ("POST", "/api/v1/chat/dictionaries"),
        ("GET", "/api/v1/chat/dictionaries/7"),
        ("PUT", "/api/v1/chat/dictionaries/7"),
        ("DELETE", "/api/v1/chat/dictionaries/7"),
    ]
    assert mocked.await_args_list[0].kwargs["params"] == {
        "include_inactive": "true",
        "include_usage": "true",
    }
    assert mocked.await_args_list[3].kwargs["json_data"] == {"name": "Lore v2", "version": 2}
    assert mocked.await_args_list[4].kwargs["params"] == {"hard_delete": "true"}


@pytest.mark.asyncio
async def test_chat_dictionary_client_routes_entries_and_processing(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(client, "_request", mocked)

    await client.add_chat_dictionary_entry(7, DictionaryEntryCreateRequest(pattern="Ada", replacement="Dr. Ada"))
    await client.list_chat_dictionary_entries(7, group="people")
    await client.update_chat_dictionary_entry(12, {"enabled": False})
    await client.delete_chat_dictionary_entry(12)
    await client.bulk_chat_dictionary_entry_operations(
        BulkDictionaryEntryOperationRequest(operation="deactivate", entry_ids=[12])
    )
    await client.reorder_chat_dictionary_entries(7, DictionaryEntryReorderRequest(entry_ids=[12, 13]))
    await client.process_chat_dictionaries(ProcessChatDictionariesRequest(text="Ada", dictionary_id=7))

    assert [call.args[:2] for call in mocked.await_args_list] == [
        ("POST", "/api/v1/chat/dictionaries/7/entries"),
        ("GET", "/api/v1/chat/dictionaries/7/entries"),
        ("PUT", "/api/v1/chat/dictionaries/entries/12"),
        ("DELETE", "/api/v1/chat/dictionaries/entries/12"),
        ("POST", "/api/v1/chat/dictionaries/entries/bulk"),
        ("PUT", "/api/v1/chat/dictionaries/7/entries/reorder"),
        ("POST", "/api/v1/chat/dictionaries/process"),
    ]
    assert mocked.await_args_list[1].kwargs["params"] == {"group": "people"}


@pytest.mark.asyncio
async def test_chat_dictionary_client_routes_import_export_history_and_stats(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(client, "_request", mocked)

    await client.import_chat_dictionary_markdown(
        ImportDictionaryMarkdownRequest(name="Lore", content="# Lore", activate=True)
    )
    await client.export_chat_dictionary_markdown(7)
    await client.export_chat_dictionary_json(7)
    await client.import_chat_dictionary_json(ImportDictionaryJSONRequest(data={"name": "Lore"}))
    await client.list_chat_dictionary_activity(7, limit=5, offset=10)
    await client.list_chat_dictionary_versions(7, limit=5, offset=10)
    await client.get_chat_dictionary_version(7, 3)
    await client.revert_chat_dictionary_version(7, 3)
    await client.get_chat_dictionary_statistics(7)

    assert [call.args[:2] for call in mocked.await_args_list] == [
        ("POST", "/api/v1/chat/dictionaries/import"),
        ("GET", "/api/v1/chat/dictionaries/7/export"),
        ("GET", "/api/v1/chat/dictionaries/7/export/json"),
        ("POST", "/api/v1/chat/dictionaries/import/json"),
        ("GET", "/api/v1/chat/dictionaries/7/activity"),
        ("GET", "/api/v1/chat/dictionaries/7/versions"),
        ("GET", "/api/v1/chat/dictionaries/7/versions/3"),
        ("POST", "/api/v1/chat/dictionaries/7/versions/3/revert"),
        ("GET", "/api/v1/chat/dictionaries/7/statistics"),
    ]
    assert mocked.await_args_list[4].kwargs["params"] == {"limit": 5, "offset": 10}
    assert mocked.await_args_list[5].kwargs["params"] == {"limit": 5, "offset": 10}
