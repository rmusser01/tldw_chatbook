from unittest.mock import Mock

import pytest

from tldw_chatbook.Character_Chat.server_chat_dictionary_service import ServerChatDictionaryService
from tldw_chatbook.runtime_policy import PolicyDecision, PolicyDeniedError


class FakeChatDictionaryClient:
    def __init__(self):
        self.calls = []

    async def list_chat_dictionaries(self, **kwargs):
        self.calls.append(("list_chat_dictionaries", kwargs))
        return {"dictionaries": [{"id": 7, "name": "Lore"}]}

    async def create_chat_dictionary(self, request_data):
        self.calls.append(("create_chat_dictionary", request_data))
        return {"id": 7, "name": request_data["name"]}

    async def get_chat_dictionary(self, dictionary_id):
        self.calls.append(("get_chat_dictionary", dictionary_id))
        return {"id": dictionary_id, "name": "Lore", "entries": []}

    async def update_chat_dictionary(self, dictionary_id, request_data):
        self.calls.append(("update_chat_dictionary", dictionary_id, request_data))
        return {"id": dictionary_id, "name": request_data["name"]}

    async def delete_chat_dictionary(self, dictionary_id, **kwargs):
        self.calls.append(("delete_chat_dictionary", dictionary_id, kwargs))
        return {}

    async def add_chat_dictionary_entry(self, dictionary_id, request_data):
        self.calls.append(("add_chat_dictionary_entry", dictionary_id, request_data))
        return {"id": 12, "dictionary_id": dictionary_id}

    async def list_chat_dictionary_entries(self, dictionary_id, **kwargs):
        self.calls.append(("list_chat_dictionary_entries", dictionary_id, kwargs))
        return {"entries": [{"id": 12, "dictionary_id": dictionary_id}]}

    async def update_chat_dictionary_entry(self, entry_id, request_data):
        self.calls.append(("update_chat_dictionary_entry", entry_id, request_data))
        return {"id": entry_id, **request_data}

    async def delete_chat_dictionary_entry(self, entry_id):
        self.calls.append(("delete_chat_dictionary_entry", entry_id))
        return {}

    async def reorder_chat_dictionary_entries(self, dictionary_id, request_data):
        self.calls.append(("reorder_chat_dictionary_entries", dictionary_id, request_data))
        return {"dictionary_id": dictionary_id, "entry_ids": request_data["entry_ids"]}

    async def process_chat_dictionaries(self, request_data):
        self.calls.append(("process_chat_dictionaries", request_data))
        return {"processed_text": request_data["text"]}

    async def import_chat_dictionary_markdown(self, request_data):
        self.calls.append(("import_chat_dictionary_markdown", request_data))
        return {"dictionary_id": 7}

    async def export_chat_dictionary_markdown(self, dictionary_id):
        self.calls.append(("export_chat_dictionary_markdown", dictionary_id))
        return {"name": "Lore", "content": "# Lore"}

    async def export_chat_dictionary_json(self, dictionary_id):
        self.calls.append(("export_chat_dictionary_json", dictionary_id))
        return {"name": "Lore", "entries": []}

    async def import_chat_dictionary_json(self, request_data):
        self.calls.append(("import_chat_dictionary_json", request_data))
        return {"dictionary_id": 7}

    async def list_chat_dictionary_activity(self, dictionary_id, **kwargs):
        self.calls.append(("list_chat_dictionary_activity", dictionary_id, kwargs))
        return {"dictionary_id": dictionary_id, "events": []}

    async def list_chat_dictionary_versions(self, dictionary_id, **kwargs):
        self.calls.append(("list_chat_dictionary_versions", dictionary_id, kwargs))
        return {"dictionary_id": dictionary_id, "versions": []}

    async def get_chat_dictionary_version(self, dictionary_id, revision):
        self.calls.append(("get_chat_dictionary_version", dictionary_id, revision))
        return {"dictionary_id": dictionary_id, "revision": revision}

    async def revert_chat_dictionary_version(self, dictionary_id, revision):
        self.calls.append(("revert_chat_dictionary_version", dictionary_id, revision))
        return {"dictionary_id": dictionary_id, "reverted_to_revision": revision}

    async def get_chat_dictionary_statistics(self, dictionary_id):
        self.calls.append(("get_chat_dictionary_statistics", dictionary_id))
        return {"dictionary_id": dictionary_id, "entry_count": 0}


class FakeCachingProvider:
    def __init__(self, client_factory):
        self.client_factory = client_factory
        self.client = None
        self.build_calls = 0
        self.constructed_clients = 0

    def build_client(self):
        self.build_calls += 1
        if self.client is None:
            self.client = self.client_factory()
            self.constructed_clients += 1
        return self.client


class ExplodingProvider:
    def __init__(self):
        self.calls = 0

    def build_client(self):
        self.calls += 1
        raise AssertionError("provider should not be used")


@pytest.mark.asyncio
async def test_server_chat_dictionary_service_reuses_provider_cached_client_across_operations():
    provider = FakeCachingProvider(FakeChatDictionaryClient)
    service = ServerChatDictionaryService.from_server_context_provider(provider)

    await service.list_dictionaries(include_inactive=True)
    await service.get_dictionary(7)

    assert provider.build_calls == 2
    assert provider.constructed_clients == 1
    assert provider.client.calls == [
        ("list_chat_dictionaries", {"include_inactive": True}),
        ("get_chat_dictionary", 7),
    ]


@pytest.mark.asyncio
async def test_server_chat_dictionary_service_direct_client_takes_precedence_over_provider():
    client = FakeChatDictionaryClient()
    provider = ExplodingProvider()
    service = ServerChatDictionaryService(client=client, client_provider=provider)

    await service.get_statistics(7)

    assert provider.calls == 0
    assert client.calls == [("get_chat_dictionary_statistics", 7)]


@pytest.mark.asyncio
async def test_server_chat_dictionary_service_denied_policy_does_not_build_provider_client():
    policy = Mock()
    policy.require_allowed = None
    policy.require_ui_action_allowed = Mock(
        return_value=PolicyDecision(
            allowed=False,
            reason_code="wrong_source",
            user_message="Blocked.",
            effective_source="server",
            authority_owner="server",
        )
    )
    provider = ExplodingProvider()
    service = ServerChatDictionaryService.from_server_context_provider(provider, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError):
        await service.list_dictionaries()

    assert provider.calls == 0


@pytest.mark.asyncio
async def test_server_chat_dictionary_service_routes_core_actions_with_policy():
    client = FakeChatDictionaryClient()
    policy = Mock()
    service = ServerChatDictionaryService(client=client, policy_enforcer=policy)

    listed = await service.list_dictionaries(include_inactive=True)
    created = await service.create_dictionary({"name": "Lore"})
    detail = await service.get_dictionary(7)
    updated = await service.update_dictionary(7, {"name": "Lore v2"})
    deleted = await service.delete_dictionary(7, hard_delete=True)

    assert listed["dictionaries"][0]["name"] == "Lore"
    assert created["id"] == 7
    assert detail["entries"] == []
    assert updated["name"] == "Lore v2"
    assert deleted is True
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "chat.dictionaries.list.server",
        "chat.dictionaries.create.server",
        "chat.dictionaries.detail.server",
        "chat.dictionaries.update.server",
        "chat.dictionaries.delete.server",
    ]


@pytest.mark.asyncio
async def test_server_chat_dictionary_service_routes_entries_processing_import_export_and_history():
    client = FakeChatDictionaryClient()
    policy = Mock()
    service = ServerChatDictionaryService(client=client, policy_enforcer=policy)

    await service.add_entry(7, {"pattern": "Ada", "replacement": "Dr. Ada"})
    await service.list_entries(7, group="people")
    await service.update_entry(12, {"enabled": False})
    deleted = await service.delete_entry(12)
    await service.reorder_entries(7, {"entry_ids": [12]})
    await service.process_text({"text": "Ada"})
    await service.import_markdown({"name": "Lore", "content": "# Lore"})
    await service.export_markdown(7)
    await service.import_json({"data": {"name": "Lore"}})
    await service.export_json(7)
    await service.list_activity(7, limit=5)
    await service.list_versions(7, limit=5)
    await service.get_version(7, 3)
    await service.revert_version(7, 3)
    await service.get_statistics(7)

    assert deleted is True
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "chat.dictionary.entries.create.server",
        "chat.dictionary.entries.list.server",
        "chat.dictionary.entries.update.server",
        "chat.dictionary.entries.delete.server",
        "chat.dictionary.entries.reorder.server",
        "chat.dictionaries.process.server",
        "chat.dictionaries.import.server",
        "chat.dictionaries.export.server",
        "chat.dictionaries.import.server",
        "chat.dictionaries.export.server",
        "chat.dictionary.activity.list.server",
        "chat.dictionary.versions.list.server",
        "chat.dictionary.versions.detail.server",
        "chat.dictionary.versions.restore.server",
        "chat.dictionary.statistics.detail.server",
    ]


@pytest.mark.asyncio
async def test_server_chat_dictionary_service_hard_stops_denied_ui_policy_decision():
    policy = Mock()
    policy.require_allowed = None
    policy.require_ui_action_allowed = Mock(
        return_value=PolicyDecision(
            allowed=False,
            reason_code="wrong_source",
            user_message="Blocked.",
            effective_source="server",
            authority_owner="server",
        )
    )
    client = FakeChatDictionaryClient()
    service = ServerChatDictionaryService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.list_dictionaries()

    assert exc.value.reason_code == "wrong_source"
    assert client.calls == []
