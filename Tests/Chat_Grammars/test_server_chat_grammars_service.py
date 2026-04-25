from unittest.mock import Mock

import pytest

from tldw_chatbook.Chat_Grammars_Interop import ServerChatGrammarsService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError


def _grammar_payload(**overrides):
    payload = {
        "id": "grammar-1",
        "name": "JSON object",
        "description": "Constrain output to JSON.",
        "grammar_text": "root ::= object",
        "validation_status": "unchecked",
        "is_archived": False,
        "created_at": "2026-04-25T12:00:00Z",
        "updated_at": "2026-04-25T12:00:00Z",
        "version": 1,
    }
    payload.update(overrides)
    return payload


class FakeChatGrammarsClient:
    def __init__(self):
        self.calls = []

    async def create_chat_grammar(self, request_data):
        self.calls.append(("create_chat_grammar", request_data.model_dump(exclude_none=True, mode="json")))
        return _grammar_payload()

    async def list_chat_grammars(self, **kwargs):
        self.calls.append(("list_chat_grammars", kwargs))
        return {"items": [_grammar_payload()], "total": 1}

    async def get_chat_grammar(self, grammar_id, **kwargs):
        self.calls.append(("get_chat_grammar", grammar_id, kwargs))
        return _grammar_payload(id=grammar_id)

    async def update_chat_grammar(self, grammar_id, request_data):
        self.calls.append(("update_chat_grammar", grammar_id, request_data.model_dump(exclude_none=True, mode="json")))
        return _grammar_payload(id=grammar_id, name="Strict JSON", version=2)

    async def delete_chat_grammar(self, grammar_id, **kwargs):
        self.calls.append(("delete_chat_grammar", grammar_id, kwargs))
        return True


@pytest.mark.asyncio
async def test_server_chat_grammars_service_routes_crud_with_policy_actions():
    client = FakeChatGrammarsClient()
    policy = Mock()
    service = ServerChatGrammarsService(client=client, policy_enforcer=policy)

    created = await service.create_grammar(name="JSON object", description="Constrain output to JSON.", grammar_text="root ::= object")
    listed = await service.list_grammars(include_archived=True, limit=25, offset=5)
    fetched = await service.get_grammar("grammar-1", include_archived=True)
    updated = await service.update_grammar("grammar-1", name="Strict JSON", version=1)
    deleted = await service.delete_grammar("grammar-1", hard_delete=True)

    assert created["id"] == "grammar-1"
    assert listed["total"] == 1
    assert fetched["id"] == "grammar-1"
    assert updated["version"] == 2
    assert deleted is True
    assert client.calls == [
        (
            "create_chat_grammar",
            {
                "name": "JSON object",
                "description": "Constrain output to JSON.",
                "grammar_text": "root ::= object",
            },
        ),
        ("list_chat_grammars", {"include_archived": True, "limit": 25, "offset": 5}),
        ("get_chat_grammar", "grammar-1", {"include_archived": True}),
        ("update_chat_grammar", "grammar-1", {"version": 1, "name": "Strict JSON"}),
        ("delete_chat_grammar", "grammar-1", {"hard_delete": True}),
    ]
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "chat.grammars.create.server",
        "chat.grammars.list.server",
        "chat.grammars.detail.server",
        "chat.grammars.update.server",
        "chat.grammars.delete.server",
    ]


@pytest.mark.asyncio
async def test_server_chat_grammars_service_hard_stops_denied_ui_policy_decision():
    policy = Mock()
    policy.require_allowed = None
    policy.require_ui_action_allowed = Mock(
        return_value=PolicyDecision(
            allowed=False,
            reason_code="server_auth_required",
            user_message="Blocked.",
            effective_source="server",
            authority_owner="server",
        )
    )
    client = FakeChatGrammarsClient()
    service = ServerChatGrammarsService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.list_grammars()

    assert exc.value.reason_code == "server_auth_required"
    assert client.calls == []
