import pytest

from tldw_chatbook.Chat_Grammars_Interop.chat_grammars_scope_service import ChatGrammarsScopeService
from tldw_chatbook.Chat_Grammars_Interop.local_chat_grammars_service import LocalChatGrammarsService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeChatGrammarsService:
    def __init__(self):
        self.calls = []

    async def create_grammar(self, **kwargs):
        self.calls.append(("create_grammar", kwargs))
        return {"id": "grammar-1", "name": kwargs["name"]}

    async def list_grammars(self, **kwargs):
        self.calls.append(("list_grammars", kwargs))
        return {"items": [{"id": "grammar-1", "name": "JSON object"}], "total": 1}

    async def get_grammar(self, grammar_id, **kwargs):
        self.calls.append(("get_grammar", grammar_id, kwargs))
        return {"id": grammar_id, "name": "JSON object"}

    async def update_grammar(self, grammar_id, **kwargs):
        self.calls.append(("update_grammar", grammar_id, kwargs))
        return {"id": grammar_id, "name": kwargs.get("name", "JSON object")}

    async def delete_grammar(self, grammar_id, **kwargs):
        self.calls.append(("delete_grammar", grammar_id, kwargs))
        return True


class FakePolicyEnforcer:
    def __init__(self, denied_reason=None):
        self.denied_reason = denied_reason
        self.calls = []

    def require_allowed(self, *, action_id):
        self.calls.append(action_id)
        if self.denied_reason:
            raise PolicyDeniedError(
                action_id=action_id,
                reason_code=self.denied_reason,
                user_message=f"{action_id} denied",
                effective_source="server",
                authority_owner="server",
            )


@pytest.mark.asyncio
async def test_chat_grammars_scope_service_routes_server_crud_and_normalizes_records():
    server = FakeChatGrammarsService()
    policy = FakePolicyEnforcer()
    scope = ChatGrammarsScopeService(server_service=server, policy_enforcer=policy)

    listed = await scope.list_grammars(mode="server", include_archived=True)
    created = await scope.create_grammar(mode="server", name="JSON object", grammar_text="root ::= object")
    fetched = await scope.get_grammar("grammar-1", mode="server")
    updated = await scope.update_grammar("grammar-1", mode="server", name="Strict JSON")
    deleted = await scope.delete_grammar("grammar-1", mode="server", hard_delete=True)

    assert listed["items"][0]["record_id"] == "server:chat_grammar:grammar-1"
    assert created["record_id"] == "server:chat_grammar:grammar-1"
    assert fetched["record_id"] == "server:chat_grammar:grammar-1"
    assert updated["record_id"] == "server:chat_grammar:grammar-1"
    assert deleted["record_id"] == "server:chat_grammar:grammar-1"
    assert server.calls == [
        ("list_grammars", {"include_archived": True}),
        ("create_grammar", {"name": "JSON object", "grammar_text": "root ::= object"}),
        ("get_grammar", "grammar-1", {}),
        ("update_grammar", "grammar-1", {"name": "Strict JSON"}),
        ("delete_grammar", "grammar-1", {"hard_delete": True}),
    ]
    assert policy.calls == [
        "chat.grammars.list.server",
        "chat.grammars.create.server",
        "chat.grammars.detail.server",
        "chat.grammars.update.server",
        "chat.grammars.delete.server",
    ]


@pytest.mark.asyncio
async def test_chat_grammars_scope_service_routes_local_crud_and_normalizes_records(tmp_path):
    local = LocalChatGrammarsService(store_path=tmp_path / "grammars.json")
    policy = FakePolicyEnforcer()
    scope = ChatGrammarsScopeService(local_service=local, server_service=None, policy_enforcer=policy)

    created = await scope.create_grammar(mode="local", name="JSON object", grammar_text="root ::= object")
    listed = await scope.list_grammars(mode="local")
    fetched = await scope.get_grammar("local-grammar-1", mode="local")
    updated = await scope.update_grammar("local-grammar-1", mode="local", version=1, name="Strict JSON")
    deleted = await scope.delete_grammar("local-grammar-1", mode="local")

    assert created["record_id"] == "local:chat_grammar:local-grammar-1"
    assert listed["items"][0]["record_id"] == "local:chat_grammar:local-grammar-1"
    assert fetched["record_id"] == "local:chat_grammar:local-grammar-1"
    assert updated["record_id"] == "local:chat_grammar:local-grammar-1"
    assert deleted["record_id"] == "local:chat_grammar:local-grammar-1"
    assert policy.calls == [
        "chat.grammars.create.local",
        "chat.grammars.list.local",
        "chat.grammars.detail.local",
        "chat.grammars.update.local",
        "chat.grammars.delete.local",
    ]


@pytest.mark.asyncio
async def test_chat_grammars_scope_service_honestly_rejects_missing_local_service():
    server = FakeChatGrammarsService()
    scope = ChatGrammarsScopeService(local_service=None, server_service=server, policy_enforcer=FakePolicyEnforcer())

    with pytest.raises(ValueError, match="Local chat grammars backend is unavailable"):
        await scope.list_grammars(mode="local")

    assert server.calls == []


@pytest.mark.asyncio
async def test_chat_grammars_scope_service_blocks_denied_server_action_before_dispatch():
    server = FakeChatGrammarsService()
    scope = ChatGrammarsScopeService(server_service=server, policy_enforcer=FakePolicyEnforcer("server_auth_required"))

    with pytest.raises(PolicyDeniedError) as exc:
        await scope.list_grammars(mode="server")

    assert exc.value.reason_code == "server_auth_required"
    assert server.calls == []


def test_chat_grammars_scope_service_reports_known_unsupported_capabilities():
    scope = ChatGrammarsScopeService(server_service=FakeChatGrammarsService())

    assert scope.list_unsupported_capabilities(mode="server") == []
    assert scope.list_unsupported_capabilities(mode="local") == []
