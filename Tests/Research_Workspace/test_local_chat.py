import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from Tests.Chat.test_citation_trace_repository import _repository as citation_repository
from Tests.Research_Workspace.test_chat_models import LOCAL, SOURCE, turn
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Research_Workspace.chat_models import ResearchChatSessionRequest
from tldw_chatbook.Research_Workspace.contracts import QualifiedWorkspaceRef
from tldw_chatbook.Research_Workspace.egress_consent import ResearchEgressConsentService
from tldw_chatbook.Research_Workspace.layout_state import ResearchPanePreferences
from tldw_chatbook.Research_Workspace.local_chat import LocalResearchChat
from tldw_chatbook.Research_Workspace.overlay_store import (
    ResearchPresentationOverlayStore,
)
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService

pytestmark = pytest.mark.bootstrap_profile


class _Provider:
    def __init__(self):
        self.messages = []
        self.fail = False
        self.egress = "on_device"
        self.endpoint = "http://localhost:1234"

    async def resolve_for_send(self, selection):
        return SimpleNamespace(
            ready=True,
            provider=selection.provider,
            model=selection.explicit_model,
            resolved_destination=SimpleNamespace(
                egress_class=self.egress,
                endpoint_identity=self.endpoint,
            ),
        )

    async def stream_chat(self, _resolution, messages, *, signals):
        self.messages.append(messages)
        if self.fail:
            raise ValueError("fixture-private-failure")
        signals.record_usage_payload({"input_tokens": 12, "output_tokens": 4})
        yield "The study supports the result [S1]."


@pytest.fixture
def local_chat(tmp_path):
    db = CharactersRAGDB(tmp_path / "chat.sqlite", "research-test")
    workspace_db = WorkspaceDB(tmp_path / "workspaces.sqlite")
    registry = LocalWorkspaceRegistryService(workspace_db)
    registry.create_workspace(workspace_id=LOCAL.workspace_id, name="Reading")
    registry.create_workspace(workspace_id="other", name="Other")
    provider = _Provider()
    overlay = ResearchPresentationOverlayStore(tmp_path / "research" / "overlay.json")
    searches = []

    async def search(request, scope):
        searches.append((request.effective_retrieval, scope))
        return [
            {
                "source": "media",
                "id": "42",
                "title": "Study",
                "content": "Selected study evidence",
                "score": 0.9,
                "metadata": {},
            },
            {
                "source": "media",
                "id": "999",
                "title": "Unselected",
                "content": "Unselected private text",
                "score": 1.0,
                "metadata": {},
            },
        ]

    async def sources(ref):
        return (SOURCE,) if ref == LOCAL else ()

    chat = LocalResearchChat(
        ChatPersistenceService(
            db,
            workspace_registry=registry,
            citation_repository=citation_repository(db),
        ),
        ChatConversationService(db),
        lambda: provider,
        search,
        overlay,
    )
    try:
        yield chat, db, registry, provider, overlay, searches, sources
    finally:
        db.close_connection()
        workspace_db.close()


@pytest.mark.parametrize("method", ["fts", "semantic", "hybrid"])
@pytest.mark.asyncio
async def test_local_workspace_chat_persists_exact_conversation_turn_and_evidence(
    local_chat, method
):
    chat, db, registry, provider, _, searches, _ = local_chat
    session = await chat.create_session(
        LOCAL, ResearchChatSessionRequest("Article discussion")
    )
    assert (
        db.get_conversation_by_id(session.conversation_id)["workspace_id"]
        == LOCAL.workspace_id
    )
    assert [
        m.item_id for m in registry.list_workspace_conversations(LOCAL.workspace_id)
    ] == [session.conversation_id]
    assert (await chat.list_sessions(LOCAL)).items == (session,)
    route = await chat.resolve_route("fixture", "model")
    source = replace(SOURCE, ready_modes=(method,))
    request = turn(
        conversation_id=session.conversation_id,
        mode="rag",
        processing_route=route,
        source_snapshot=(source,),
        retrieval=replace(turn().retrieval, mode=method),
    )
    chunks = []

    async def selected(_ref):
        return (source,)

    outcome = await chat.answer(request, chunks.append, asyncio.Event(), selected)
    assert outcome.status == "complete"
    assert outcome.retrieved
    [(used_mode, scope)] = searches
    assert used_mode.value == method
    assert scope.state == "scoped" and scope.allowlist == {"media": frozenset({"42"})}
    assert "Selected study evidence" in repr(provider.messages)
    assert "Unselected private text" not in repr(provider.messages)
    rows = await chat.load_messages(LOCAL, session.conversation_id)
    assert [row["content"] for row in rows] == [request.user_text, chunks[0]]
    assert rows[-1]["parent_message_id"] == rows[0]["id"]
    assert rows[-1]["assistant_generation_state"] == "complete"
    assert rows[-1]["usage_json"] is not None
    assert [citation.source for citation in outcome.citations] == [source]
    assert chat.persistence.citation_repository.active_owner_candidate_message_ids(
        [outcome.assistant_message_id]
    ) == {outcome.assistant_message_id}
    metadata = json.loads(rows[-1]["metadata_json"])["research_chat"]
    assert metadata["sources"][0]["catalog_item_version"] == SOURCE.catalog_item_version
    with pytest.raises(ValueError, match="workspace"):
        await chat.load_messages(
            QualifiedWorkspaceRef("local", "other"), session.conversation_id
        )


@pytest.mark.asyncio
async def test_remote_source_transfer_requires_exact_consent_and_failure_keeps_draft(
    local_chat,
):
    chat, db, _, provider, overlay, searches, sources = local_chat
    session = await chat.create_session(LOCAL, ResearchChatSessionRequest())
    provider.egress = "public_network"
    provider.endpoint = "https://remote.example"
    request = turn(
        conversation_id=session.conversation_id,
        processing_route=await chat.resolve_route("fixture", "model"),
    )
    with pytest.raises(ValueError, match="consent"):
        await chat.answer(request, lambda _: None, asyncio.Event(), sources)
    assert searches == provider.messages == []

    consent = ResearchEgressConsentService()
    approved = consent.approve(consent.preflight(request), ())
    overlay.save(
        LOCAL,
        ResearchPanePreferences(),
        expected_revision=0,
        chat_egress_consents=approved,
        chat_draft=request.user_text,
    )
    provider.endpoint = "https://different.example"
    with pytest.raises(ValueError, match="route"):
        await chat.answer(request, lambda _: None, asyncio.Event(), sources)
    assert searches == provider.messages == []

    provider.endpoint = "https://remote.example"
    provider.fail = True
    outcome = await chat.answer(request, lambda _: None, asyncio.Event(), sources)
    assert outcome.status == "failed"
    assert db.get_messages_for_conversation(session.conversation_id) == []
    assert overlay.load(LOCAL).chat_draft == request.user_text
