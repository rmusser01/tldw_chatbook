from unittest.mock import Mock

import pytest

from tldw_chatbook.Character_Chat.chat_dictionary_scope_service import ChatDictionaryScopeService


class FakeLocalChatDictionaryService:
    def __init__(self):
        self.calls = []

    def list_dictionaries(self, **kwargs):
        self.calls.append(("list_dictionaries", kwargs))
        return {"dictionaries": [{"id": 1, "source": "local"}]}

    def create_dictionary(self, request_data):
        self.calls.append(("create_dictionary", request_data))
        return {"id": 2, "source": "local"}

    def list_entries(self, dictionary_id, **kwargs):
        self.calls.append(("list_entries", dictionary_id, kwargs))
        return {"entries": [{"id": f"local:chat_dictionary_entry:{dictionary_id}:0"}]}

    def get_statistics(self, dictionary_id):
        self.calls.append(("get_statistics", dictionary_id))
        return {"dictionary_id": dictionary_id, "entry_count": 1, "source": "local"}

class FakeLocalChatDictionaryHistoryService(FakeLocalChatDictionaryService):
    def list_activity(self, dictionary_id, **kwargs):
        self.calls.append(("list_activity", dictionary_id, kwargs))
        return {"dictionary_id": dictionary_id, "activity": [{"action": "create"}], "source": "local"}

    def list_versions(self, dictionary_id, **kwargs):
        self.calls.append(("list_versions", dictionary_id, kwargs))
        return {"dictionary_id": dictionary_id, "versions": [{"revision": 1}], "source": "local"}

    def get_version(self, dictionary_id, revision):
        self.calls.append(("get_version", dictionary_id, revision))
        return {"dictionary_id": dictionary_id, "revision": revision, "source": "local"}

    def revert_version(self, dictionary_id, revision):
        self.calls.append(("revert_version", dictionary_id, revision))
        return {"dictionary_id": dictionary_id, "reverted_to_revision": revision, "source": "local"}


class FakeServerChatDictionaryService:
    def __init__(self):
        self.calls = []

    async def list_dictionaries(self, **kwargs):
        self.calls.append(("list_dictionaries", kwargs))
        return {"dictionaries": [{"id": 7, "source": "server"}]}

    async def create_dictionary(self, request_data):
        self.calls.append(("create_dictionary", request_data))
        return {"id": 8, "source": "server"}

    async def list_entries(self, dictionary_id, **kwargs):
        self.calls.append(("list_entries", dictionary_id, kwargs))
        return {"entries": [{"id": 12}]}

    async def get_statistics(self, dictionary_id):
        self.calls.append(("get_statistics", dictionary_id))
        return {"dictionary_id": dictionary_id, "entry_count": 2, "source": "server"}


@pytest.mark.asyncio
async def test_chat_dictionary_scope_routes_local_and_server_modes_with_policy():
    local_service = FakeLocalChatDictionaryService()
    server_service = FakeServerChatDictionaryService()
    policy = Mock()
    scope = ChatDictionaryScopeService(
        local_service=local_service,
        server_service=server_service,
        policy_enforcer=policy,
    )

    local_result = await scope.list_dictionaries(mode="local", include_inactive=True)
    server_result = await scope.list_dictionaries(mode="server", include_usage=True)
    local_created = await scope.create_dictionary({"name": "Local"}, mode="local")
    server_created = await scope.create_dictionary({"name": "Server"}, mode="server")
    local_entries = await scope.list_entries(2, mode="local", group="people")
    server_entries = await scope.list_entries(8, mode="server")
    default_statistics = await scope.get_statistics(2)
    local_statistics = await scope.get_statistics(2, mode="local")
    server_statistics = await scope.get_statistics(8, mode="server")

    assert local_result["dictionaries"][0]["source"] == "local"
    assert server_result["dictionaries"][0]["source"] == "server"
    assert local_created["source"] == "local"
    assert server_created["source"] == "server"
    assert local_entries["entries"][0]["id"] == "local:chat_dictionary_entry:2:0"
    assert server_entries["entries"][0]["id"] == 12
    assert default_statistics["source"] == "local"
    assert local_statistics["source"] == "local"
    assert server_statistics["source"] == "server"
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "chat.dictionaries.list.local",
        "chat.dictionaries.list.server",
        "chat.dictionaries.create.local",
        "chat.dictionaries.create.server",
        "chat.dictionary.entries.list.local",
        "chat.dictionary.entries.list.server",
        "chat.dictionary.statistics.detail.local",
        "chat.dictionary.statistics.detail.local",
        "chat.dictionary.statistics.detail.server",
    ]


@pytest.mark.asyncio
async def test_chat_dictionary_scope_reports_and_blocks_local_history_gaps():
    scope = ChatDictionaryScopeService(
        local_service=FakeLocalChatDictionaryService(),
        server_service=FakeServerChatDictionaryService(),
    )

    report = scope.list_unsupported_capabilities(mode="local")

    assert report == [
        {
            "operation_id": "chat.dictionary.activity.local",
            "source": "local",
            "supported": False,
            "reason_code": "local_scope_missing",
            "user_message": "Local chat dictionary activity history is not available.",
            "affected_action_ids": ["chat.dictionary.activity.list.local"],
        },
        {
            "operation_id": "chat.dictionary.versions.local",
            "source": "local",
            "supported": False,
            "reason_code": "local_scope_missing",
            "user_message": "Local chat dictionary version history is not available.",
            "affected_action_ids": [
                "chat.dictionary.versions.detail.local",
                "chat.dictionary.versions.list.local",
                "chat.dictionary.versions.restore.local",
            ],
        },
    ]
    with pytest.raises(ValueError, match="Local chat dictionary activity history is not available"):
        await scope.list_activity(2, mode="local")


@pytest.mark.asyncio
async def test_chat_dictionary_scope_routes_local_history_when_backend_supports_it():
    local_service = FakeLocalChatDictionaryHistoryService()
    scope = ChatDictionaryScopeService(
        local_service=local_service,
        server_service=FakeServerChatDictionaryService(),
    )

    report = scope.list_unsupported_capabilities(mode="local")
    activity = await scope.list_activity(2, mode="local", limit=5)
    versions = await scope.list_versions(2, mode="local", limit=5)
    version = await scope.get_version(2, 1, mode="local")
    reverted = await scope.revert_version(2, 1, mode="local")

    assert report == []
    assert activity["source"] == "local"
    assert versions["versions"][0]["revision"] == 1
    assert version["revision"] == 1
    assert reverted["reverted_to_revision"] == 1
    assert local_service.calls == [
        ("list_activity", 2, {"limit": 5}),
        ("list_versions", 2, {"limit": 5}),
        ("get_version", 2, 1),
        ("revert_version", 2, 1),
    ]


@pytest.mark.asyncio
async def test_scope_service_character_attach_roundtrip(tmp_path):
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Character_Chat.local_chat_dictionary_service import LocalChatDictionaryService
    from tldw_chatbook.Character_Chat.chat_dictionary_scope_service import ChatDictionaryScopeService

    db = CharactersRAGDB(tmp_path / "scope.db", "test-client")
    local = LocalChatDictionaryService(db)
    scope = ChatDictionaryScopeService(local_service=local, server_service=None)
    dict_id = local.create_dictionary({"name": "Slang", "entries": [{"pattern": "x", "replacement": "y"}]})["id"]
    char_id = db.add_character_card({"name": "Noir"})

    await scope.attach_to_character(dict_id, char_id, mode="local")
    listing = await scope.list_character_dictionaries(char_id, mode="local")
    assert [d["name"] for d in listing["dictionaries"]] == ["Slang"]

    await scope.detach_from_character(char_id, "Slang", mode="local")
    listing = await scope.list_character_dictionaries(char_id, mode="local")
    assert listing["dictionaries"] == []


@pytest.mark.asyncio
async def test_scope_service_summarize_active_dictionaries(tmp_path):
    import json as _json
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Character_Chat.local_chat_dictionary_service import LocalChatDictionaryService
    from tldw_chatbook.Character_Chat.chat_dictionary_scope_service import ChatDictionaryScopeService

    db = CharactersRAGDB(tmp_path / "sum.db", "test-client")
    local = LocalChatDictionaryService(db)
    scope = ChatDictionaryScopeService(local_service=local, server_service=None)
    conv_id = db.add_conversation({"title": "c"})
    did = local.create_dictionary({"name": "Conv", "entries": [{"pattern": "x", "replacement": "y"}]})["id"]
    conv = db.get_conversation_by_id(conv_id)
    meta = _json.loads(conv.get("metadata") or "{}"); meta["active_dictionaries"] = [did]
    db.update_conversation(conv_id, {"metadata": _json.dumps(meta)}, expected_version=conv["version"])
    char_id = db.add_character_card({"name": "N"})
    local.attach_to_character(local.create_dictionary({"name": "Char"})["id"], char_id)

    out = await scope.summarize_active_dictionaries(conv_id, char_id, mode="local")
    names = {(d["name"], d["source"]) for d in out["dictionaries"]}
    assert ("Conv", "conversation") in names and ("Char", "character") in names
