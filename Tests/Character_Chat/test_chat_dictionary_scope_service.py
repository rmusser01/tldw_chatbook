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

    def list_activity(self, dictionary_id, **kwargs):
        raise AssertionError("local activity should be blocked before backend invocation")


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
