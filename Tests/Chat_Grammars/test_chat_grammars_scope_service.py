import pytest

from tldw_chatbook.Chat_Grammars_Interop.chat_grammars_scope_service import ChatGrammarsScopeService
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
async def test_chat_grammars_scope_service_honestly_rejects_local_mode_as_remote_only():
    server = FakeChatGrammarsService()
    scope = ChatGrammarsScopeService(server_service=server, policy_enforcer=FakePolicyEnforcer())

    with pytest.raises(ValueError, match="Saved chat grammars are server-only"):
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
    assert scope.list_unsupported_capabilities(mode="local") == [
        {
            "operation_id": "chat.grammars.remote_only.local",
            "source": "local",
            "supported": False,
            "reason_code": "remote_only_surface",
            "user_message": "Saved chat grammars are unavailable in local/offline mode.",
            "affected_action_ids": [],
        }
    ]
