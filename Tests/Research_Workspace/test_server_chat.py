import asyncio
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import pytest

from tldw_chatbook.Research_Workspace.chat_models import (
    ResearchChatSessionRequest,
    ResearchChatTurnRequest,
    ResearchRetrievalSettings,
    ResearchSourceVersionRef,
)
from tldw_chatbook.Research_Workspace.contracts import QualifiedWorkspaceRef
from tldw_chatbook.Research_Workspace.layout_state import ResearchPanePreferences
from tldw_chatbook.Research_Workspace.overlay_store import (
    ResearchPresentationOverlayStore,
)
from tldw_chatbook.Research_Workspace.server_chat import ServerResearchChat


REF = QualifiedWorkspaceRef("server", "ws-1", "profile-1", "principal-1")
SOURCE = ResearchSourceVersionRef(
    REF, "association-1", "42", "article", association_version=2, ready_modes=("fts",)
)


class _Server:
    base_url = "https://research.example"

    def __init__(self):
        self.rows = []
        self.calls = []
        self.fail_context = False
        self.fail_assistant = False
        self.receipts = {}
        self.prompt = ()
        self.conversation = {
            "id": "chat-1",
            "title": "Reading group",
            "scope_type": "workspace",
            "workspace_id": REF.workspace_id,
            "topic_label": "research-chat",
        }

    async def create_research_chat_workspace_session(self, request, *, idempotency_key):
        assert request.workspace_id == REF.workspace_id
        assert request.workspace_assistant_selection == "none"
        assert idempotency_key
        return self.conversation

    async def list_chat_conversations(self, **kwargs):
        assert (
            kwargs["scope_type"] == "workspace"
            and kwargs["workspace_id"] == REF.workspace_id
        )
        return {
            "items": [self.conversation],
            "pagination": {"total": 1, "has_more": False},
        }

    async def get_chat_conversation(self, conversation_id, **kwargs):
        assert (
            conversation_id == "chat-1" and kwargs["workspace_id"] == REF.workspace_id
        )
        return {**self.conversation, "message_count": len(self.rows)}

    async def get_chat_conversation_messages_with_context(
        self, conversation_id, **kwargs
    ):
        assert (
            conversation_id == "chat-1" and kwargs["workspace_id"] == REF.workspace_id
        )
        return list(self.rows)

    async def search_research_chat(self, request):
        assert request.workspace_id == REF.workspace_id
        assert request.include_media_ids == (42,)
        self.calls.append("rag")
        return {
            "documents": [
                {
                    "id": "42",
                    "source": "media_db",
                    "title": "Study",
                    "content": "Selected server excerpt",
                },
                {
                    "id": "999",
                    "source": "media_db",
                    "title": "Other",
                    "content": "Unselected server text",
                },
            ]
        }

    async def stream_chat_completion(self, request, *, workspace_id):
        assert workspace_id == REF.workspace_id and request.conversation_id == "chat-1"
        assert request.tool_choice == "none" and not request.save_to_db
        prompt = "\n".join(message.content for message in request.messages)
        assert "Unselected server text" not in prompt
        if "rag" in self.calls:
            assert "Selected server excerpt" in prompt
        self.prompt = tuple(message.content for message in request.messages)
        self.calls.append("completion")
        yield "The study found a result [S1]."

    async def create_character_message(
        self, chat_id, request, *, scope_type, workspace_id, idempotency_key
    ):
        assert (
            chat_id == "chat-1"
            and scope_type == "workspace"
            and workspace_id == REF.workspace_id
        )
        if idempotency_key in self.receipts:
            return self.receipts[idempotency_key]
        if request.role == "assistant" and self.fail_assistant:
            raise RuntimeError("fixture append failure")
        row = {
            "id": f"message-{len(self.rows) + 1}",
            "conversation_id": chat_id,
            "sender": request.role,
            "content": request.content,
            "parent_message_id": request.parent_message_id,
        }
        self.rows.append(row)
        result = SimpleNamespace(**row)
        self.receipts[idempotency_key] = result
        self.calls.append(request.role)
        return result

    async def persist_chat_message_rag_context(
        self, message_id, context, *, workspace_id
    ):
        assert workspace_id == REF.workspace_id
        if self.fail_context:
            raise RuntimeError("fixture storage failure")
        row = next(item for item in self.rows if item["id"] == message_id)
        row["rag_context"] = context.model_dump(mode="json")
        self.calls.append("context")
        return {"success": True, "message_id": message_id}

    async def get_character_message(self, message_id, **scope):
        assert scope == {"scope_type": "workspace", "workspace_id": REF.workspace_id}
        return SimpleNamespace(
            **next(row for row in self.rows if row["id"] == message_id)
        )

    async def update_character_message(self, message_id, payload, **scope):
        row = next(row for row in self.rows if row["id"] == message_id)
        assert scope == {
            "scope_type": "workspace",
            "workspace_id": REF.workspace_id,
            "expected_version": row["version"],
        }
        row.update(payload, version=row["version"] + 1)

    async def delete_character_message(self, message_id, **scope):
        row = next(row for row in self.rows if row["id"] == message_id)
        assert scope == {
            "scope_type": "workspace",
            "workspace_id": REF.workspace_id,
            "expected_version": row["version"],
        }
        self.rows.remove(row)


@pytest.fixture
def server_chat(tmp_path):
    server = _Server()
    overlay = ResearchPresentationOverlayStore(tmp_path / "research" / "overlay.json")

    def bind(ref):
        if ref != REF:
            raise ValueError("Server identity changed")
        return server

    async def sources(ref):
        assert ref == REF
        return (SOURCE,)

    return ServerResearchChat(bind, overlay), server, overlay, sources


@pytest.mark.asyncio
async def test_server_message_actions_use_the_current_scoped_version(server_chat):
    chat, server, _, _ = server_chat
    selected = {
        "id": "message-1",
        "conversation_id": "chat-1",
        "version": 1,
        "sender": "user",
        "content": "Original question",
    }
    server.rows = [dict(selected)]
    await chat.edit(REF, "chat-1", selected, "Revised question")
    assert server.rows[0]["content"] == "Revised question"
    with pytest.raises(ValueError, match="selected message changed"):
        await chat.delete(REF, "chat-1", selected)
    await chat.delete(REF, "chat-1", server.rows[0])
    assert not server.rows


@pytest.mark.asyncio
async def test_server_treats_a_slash_question_as_literal_text(server_chat):
    chat, server, _, sources = server_chat
    text = "  /notes read this title"
    request = ResearchChatTurnRequest(
        REF,
        "chat-1",
        text,
        "general",
        (),
        ResearchRetrievalSettings(),
        chat.resolve_route(REF, "openai", "model"),
        0,
    )
    outcome = await chat.answer(request, lambda _chunk: None, asyncio.Event(), sources)
    assert outcome.status == "complete"
    assert server.prompt[-1].startswith("Question:\n")
    assert server.rows[0]["content"] == text


@pytest.mark.parametrize("mode", ["general", "auto"])
@pytest.mark.asyncio
async def test_server_workspace_chat_stays_on_one_server_and_reloads_context(
    server_chat, mode
):
    chat, server, _, sources = server_chat
    session = await chat.create_session(
        REF, ResearchChatSessionRequest("Reading group")
    )
    assert (await chat.list_sessions(REF)).items == (session,)
    route = chat.resolve_route(REF, "", "fixture")
    assert route.endpoint_fingerprint == sha256(server.base_url.encode()).hexdigest()
    request = ResearchChatTurnRequest(
        REF,
        session.conversation_id,
        "What did the study find?",
        mode,
        (SOURCE,),
        ResearchRetrievalSettings(),
        route,
        0,
    )
    received = []
    result = await chat.answer(request, received.append, asyncio.Event(), sources)
    assert result.status == "complete" and result.retrieved == (mode == "auto")
    assert server.calls == (["rag"] if mode == "auto" else []) + [
        "completion",
        "user",
        "assistant",
        "context",
    ]
    messages = await chat.load_messages(REF, session.conversation_id)
    assert [row["content"] for row in messages] == [request.user_text, received[0]]
    assert messages[1]["parent_message_id"] == messages[0]["id"]
    assert (
        messages[1]["rag_context"]["settings_snapshot"]["retrieved"] is result.retrieved
    )
    with pytest.raises(ValueError, match="identity"):
        await chat.load_messages(
            replace(REF, principal_id="other"), session.conversation_id
        )


@pytest.mark.asyncio
async def test_server_retry_finishes_partial_receipt_without_duplicate_messages(
    server_chat,
):
    chat, server, overlay, sources = server_chat
    session = await chat.create_session(REF, ResearchChatSessionRequest())
    request = ResearchChatTurnRequest(
        REF,
        session.conversation_id,
        "What did the study find?",
        "rag",
        (SOURCE,),
        ResearchRetrievalSettings(),
        chat.resolve_route(REF, "", "fixture"),
        0,
    )
    overlay.save(
        REF,
        ResearchPanePreferences(),
        expected_revision=0,
        chat_draft=request.user_text,
    )
    server.fail_context = True
    with pytest.raises(RuntimeError, match="storage"):
        await chat.answer(request, lambda _: None, asyncio.Event(), sources)
    saved = overlay.load(REF)
    assert saved.chat_append_receipts[-1].stage == "assistant_committed"
    assert "The study found" not in overlay.path.read_text()
    server.fail_context = False
    outcome = await chat.answer(request, lambda _: None, asyncio.Event(), sources)
    assert outcome.assistant_message_id == server.rows[-1]["id"]
    assert len(server.rows) == 2
    assert server.calls.count("completion") == 1
    assert overlay.load(REF).chat_draft == ""


@pytest.mark.asyncio
async def test_server_restart_regenerates_only_against_the_committed_user_message(
    server_chat,
):
    chat, server, overlay, sources = server_chat
    session = await chat.create_session(REF, ResearchChatSessionRequest())
    request = ResearchChatTurnRequest(
        REF,
        session.conversation_id,
        "What did the study find?",
        "general",
        (),
        ResearchRetrievalSettings(),
        chat.resolve_route(REF, "", "fixture"),
        0,
    )
    server.fail_assistant = True
    with pytest.raises(RuntimeError, match="append"):
        await chat.answer(request, lambda _: None, asyncio.Event(), sources)
    [receipt] = overlay.load(REF).chat_append_receipts
    assert receipt.stage == "user_committed"
    assert len(server.rows) == 1

    restarted = ServerResearchChat(chat.client_for, overlay)
    server.fail_assistant = False
    regenerated = replace(
        request, turn_id="regenerate-1", retry_user_message_id=receipt.user_message_id
    )
    result = await restarted.answer(
        regenerated, lambda _: None, asyncio.Event(), sources
    )
    assert result.user_message_id == receipt.user_message_id
    assert len(server.rows) == 2
    assert server.rows[-1]["parent_message_id"] == receipt.user_message_id
    assert server.calls.count("user") == 1
    assert [row.stage for row in overlay.load(REF).chat_append_receipts] == ["complete"]
