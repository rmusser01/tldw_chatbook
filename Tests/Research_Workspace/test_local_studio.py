from dataclasses import replace
import asyncio
from types import SimpleNamespace

import json

import pytest

from Tests.Research_Workspace.test_studio_generation import request
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chatbooks.local_chatbook_service import LocalChatbookService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Research_Workspace.local_studio import (
    LocalStudioAdapter,
    StudioOutputConflict,
)
from tldw_chatbook.Media.local_media_reading_service import LocalMediaReadingService
from tldw_chatbook.Media.media_reading_scope_service import MediaReadingScopeService
from tldw_chatbook.Research_Workspace.local_adapter import LocalResearchWorkspaceAdapter
from tldw_chatbook.Research_Workspace.local_chat import LocalResearchChat
from tldw_chatbook.Research_Workspace.overlay_store import (
    ResearchPresentationOverlayStore,
)
from tldw_chatbook.Research_Workspace.studio_generation import (
    GeneratedFlashcard,
    GeneratedQuizQuestion,
    StudioCitation,
    StudioGeneratedOutput,
)
from tldw_chatbook.Study_Interop.local_quiz_service import LocalQuizService
from tldw_chatbook.Study_Interop.local_study_service import LocalStudyService
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.fixture
def local_studio(tmp_path):
    registry_db = WorkspaceDB(tmp_path / "workspaces.sqlite")
    canonical_db = CharactersRAGDB(tmp_path / "learning.sqlite", "research-studio")
    registry = LocalWorkspaceRegistryService(registry_db)
    registry.create_workspace(workspace_id="study", name="Reading group")
    study = LocalStudyService(canonical_db)
    quizzes = LocalQuizService(canonical_db)
    artifacts = LocalChatbookService(registry_path=tmp_path / "chatbooks.json")
    try:
        yield SimpleNamespace(
            adapter=LocalStudioAdapter(registry, artifacts, study, quizzes),
            registry=registry,
            artifacts=artifacts,
            study=study,
            quizzes=quizzes,
        )
    finally:
        canonical_db.close_connection()
        registry_db.close()


@pytest.mark.parametrize(
    "kind", ("summary", "report", "compare_sources", "flashcards", "quiz")
)
async def test_local_studio_saves_to_the_canonical_owner_and_retries_once(
    local_studio, kind
):
    turn = request(kind)
    body = "The observatory is at the northern station [S1]."
    citations = (StudioCitation("S1", turn.source_snapshot[0]),)
    if kind == "flashcards":
        output = StudioGeneratedOutput(
            "complete",
            cards=(
                GeneratedFlashcard(
                    front="Where?", back="North station", citations=["S1"]
                ),
            ),
            citations=citations,
        )
    elif kind == "quiz":
        output = StudioGeneratedOutput(
            "complete",
            questions=(
                GeneratedQuizQuestion(
                    question="Where?",
                    options=["North", "South"],
                    correct_index=0,
                    citations=["S1"],
                ),
            ),
            citations=citations,
        )
    else:
        output = StudioGeneratedOutput("complete", text=body, citations=citations)

    saved = await local_studio.adapter.save(turn, output)
    assert saved.status == "complete" and saved.owner_id
    assert await local_studio.adapter.save(turn, output) == saved
    assert (await local_studio.adapter.list_outputs(turn.workspace)).items == (saved,)
    [link] = local_studio.registry.list_workspace_memberships(
        turn.workspace.workspace_id
    )
    assert link.role == "output" and link.item_id == saved.owner_id
    provenance = link.metadata["research_generation"]
    assert provenance["output_kind"] == kind
    assert body not in repr(link.metadata)
    assert (await local_studio.adapter.resolve_output(saved))["id"] == saved.owner_id

    if kind == "flashcards":
        assert saved.owner_kind == "local_study_deck"
        [card] = local_studio.study.list_flashcards(deck_id=saved.owner_id)
        assert card["front"] == "Where?"
        extra = json.loads(card["metadata"])["extra"]
        assert json.loads(extra)["source_citations"][0]["marker"] == "S1"
    elif kind == "quiz":
        assert saved.owner_kind == "local_quiz"
        [question] = local_studio.quizzes.list_questions(
            saved.owner_id, include_answers=True
        )["items"]
        assert question["question_text"] == "Where?"
        assert question["source_citations"][0]["marker"] == "S1"
    else:
        assert saved.owner_kind == "local_chatbook"
        owner = await local_studio.artifacts.get_chatbook(saved.owner_id)
        assert owner["metadata"]["content"] == body
    await local_studio.adapter.delete(saved)
    assert await local_studio.adapter.resolve_output(saved) is None
    assert (
        local_studio.registry.list_workspace_memberships(turn.workspace.workspace_id)
        == ()
    )


async def test_local_chatbook_replacement_rejects_stale_owners_and_new_version_remains_independent(
    local_studio,
):
    turn = request()
    first = await local_studio.adapter.save(
        turn, StudioGeneratedOutput("complete", text="First report")
    )
    second = await local_studio.adapter.save(
        replace(turn, operation_id="replace-1"),
        StudioGeneratedOutput("complete", text="Replacement"),
        replace_output=first,
    )
    assert (
        second.owner_id == first.owner_id and second.owner_version > first.owner_version
    )
    with pytest.raises(StudioOutputConflict):
        await local_studio.adapter.delete(first)

    third = await local_studio.adapter.save(
        replace(turn, operation_id="version-2"),
        StudioGeneratedOutput("complete", text="New version"),
        supersedes=second,
    )
    assert third.owner_id != second.owner_id
    assert (await local_studio.adapter.resolve_output(second))["metadata"][
        "content"
    ] == "Replacement"
    assert await local_studio.adapter.export_output(third, "markdown") == b"New version"


async def test_local_flashcard_child_failure_leaves_no_visible_partial_owner(
    local_studio, monkeypatch
):
    turn = request("flashcards")
    original = local_studio.study.create_flashcards_bulk

    def fail(cards):
        original(cards)
        raise RuntimeError("simulated child failure")

    monkeypatch.setattr(local_studio.study, "create_flashcards_bulk", fail)
    output = StudioGeneratedOutput(
        "complete", cards=(GeneratedFlashcard(front="Where?", back="North"),)
    )
    with pytest.raises(RuntimeError, match="output could not be saved"):
        await local_studio.adapter.save(turn, output)
    assert not local_studio.study.list_decks()
    assert (await local_studio.adapter.list_outputs(turn.workspace)).items == ()


async def test_local_chatbook_replace_retry_recovers_the_committed_owner_revision(
    local_studio, monkeypatch
):
    owner = local_studio.adapter
    turn = request()
    first = await owner.save(
        turn, StudioGeneratedOutput("complete", text="First output")
    )
    updated = replace(turn, operation_id="link-recovery")
    result = StudioGeneratedOutput("complete", text="New output")
    persist = local_studio.registry.upsert_membership_metadata
    failed = False

    def fail_once(*args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("simulated membership write failure")
        return persist(*args, **kwargs)

    monkeypatch.setattr(local_studio.registry, "upsert_membership_metadata", fail_once)
    with pytest.raises(RuntimeError, match="needs recovery"):
        await owner.save(updated, result, replace_output=first)
    recovered = await owner.save(updated, result, replace_output=first)
    assert recovered.owner_id == first.owner_id
    assert recovered.owner_version == first.owner_version + 1
    assert (await owner.list_outputs(turn.workspace)).items == (recovered,)
    assert (await owner.resolve_output(recovered))["metadata"][
        "content"
    ] == "New output"


async def test_local_workspace_port_generates_from_real_selected_library_media(
    local_studio, tmp_path
):
    store = MediaDatabase(tmp_path / "library.sqlite", client_id="studio-port")
    body = "Rowan works from the northern observatory."
    recorded = []

    class Provider:
        async def resolve_for_send(self, selection):
            return SimpleNamespace(
                ready=True,
                provider=selection.provider,
                model=selection.explicit_model,
                resolved_destination=SimpleNamespace(
                    egress_class="on_device", endpoint_identity="fixture"
                ),
            )

        async def stream_chat(self, resolution, messages, *, signals):
            recorded.extend(messages)
            yield "The northern observatory is Rowan's station [S1]."

    async def unexpected_query(*_args):
        pytest.fail("A default Studio output should use selected excerpts")

    try:
        media_id, _, _ = store.add_media_with_keywords(
            title="Observatory log",
            media_type="document",
            content=body,
            chunks=[{"text": body, "start_char": 0, "end_char": len(body)}],
        )
        assert media_id is not None
        local_studio.registry.link_membership(
            "study", item_type="media", item_id=str(media_id)
        )
        database = local_studio.study.db
        provider = Provider()
        chat = LocalResearchChat(
            ChatPersistenceService(database),
            ChatConversationService(database),
            lambda: provider,
            unexpected_query,
            ResearchPresentationOverlayStore(tmp_path / "profile" / "overlay.json"),
        )
        port = LocalResearchWorkspaceAdapter(
            local_studio.registry,
            media_scope_service=MediaReadingScopeService(
                LocalMediaReadingService(store), None
            ),
            chat=chat,
            studio=local_studio.adapter,
        )
        default_request = request("summary")
        ref = default_request.workspace
        sources = await port.capture_chat_sources(ref)
        route = await port.resolve_chat_route(ref, "fixture", "model")
        turn = replace(default_request, source_snapshot=sources, processing_route=route)
        output = await port.generate_output(turn, lambda _chunk: None, asyncio.Event())
        assert output.status == "complete"
        assert body in repr(recorded)
        assert (await port.list_outputs(ref)).items == (output,)
        owner = await port.resolve_output(output)
        assert owner["metadata"]["citations"][0]["catalog_item_id"] == str(media_id)
        assert owner["metadata"]["content"].endswith("[S1].")
    finally:
        store.close()
