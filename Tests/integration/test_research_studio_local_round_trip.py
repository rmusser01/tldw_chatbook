"""Generate all five primary outputs through their real Local stores."""

import asyncio
from types import SimpleNamespace

import pytest

from Tests.Research_Workspace.test_local_studio import local_studio as local_studio
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chatbooks.local_chatbook_service import LocalChatbookService
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Media.local_media_reading_service import LocalMediaReadingService
from tldw_chatbook.Media.media_reading_scope_service import MediaReadingScopeService
from tldw_chatbook.Research_Workspace.contracts import QualifiedWorkspaceRef
from tldw_chatbook.Research_Workspace.local_adapter import LocalResearchWorkspaceAdapter
from tldw_chatbook.Research_Workspace.local_chat import LocalResearchChat
from tldw_chatbook.Research_Workspace.local_studio import LocalStudioAdapter
from tldw_chatbook.Research_Workspace.overlay_store import (
    ResearchPresentationOverlayStore,
)
from tldw_chatbook.Research_Workspace.studio_models import (
    ResearchGenerationConfig,
    ResearchStudioRequest,
)
from tldw_chatbook.UI.Navigation.research_output_handoffs import output_handoff


pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


async def test_selected_library_excerpts_generate_and_reopen_every_primary_local_output(
    local_studio, tmp_path
):
    ref = QualifiedWorkspaceRef("local", "study")
    media = MediaDatabase(tmp_path / "library.sqlite", client_id="studio-round-trip")
    database = local_studio.study.db
    approved_texts = (
        "Rowan works at the northern station.",
        "The northern station maps the night sky.",
    )
    other = "Unselected source contains different station data."

    class Provider:
        response = ""
        prompts = []

        async def resolve_for_send(self, selection):
            return SimpleNamespace(
                ready=True,
                provider=selection.provider,
                model=selection.explicit_model,
                resolved_destination=SimpleNamespace(
                    egress_class="on_device", endpoint_identity="test"
                ),
            )

        async def stream_chat(self, _resolution, messages, *, signals):
            self.prompts.append(messages)
            yield self.response

    async def unexpected_retrieval(*_args):
        pytest.fail("Default FTS Studio reads the explicitly selected excerpts")

    provider = Provider()
    chat = LocalResearchChat(
        ChatPersistenceService(database),
        ChatConversationService(database),
        lambda: provider,
        unexpected_retrieval,
        ResearchPresentationOverlayStore(tmp_path / "profile" / "research.json"),
    )
    port = LocalResearchWorkspaceAdapter(
        local_studio.registry,
        media_scope_service=MediaReadingScopeService(
            LocalMediaReadingService(media), None
        ),
        chat=chat,
        studio=local_studio.adapter,
    )
    try:
        selection = []
        for index, body in enumerate((*approved_texts, other)):
            media_id, _, _ = media.add_media_with_keywords(
                title=f"Library source {index}",
                media_type="document",
                content=body,
                chunks=[{"text": body, "start_char": 0, "end_char": len(body)}],
            )
            link = local_studio.registry.link_membership(
                ref.workspace_id, item_type="media", item_id=str(media_id)
            )
            selection.append(link.item_id)
        await port.set_selected_scope(ref, tuple(selection[:2]))
        sources = await port.capture_chat_sources(ref)
        assert {row.catalog_item_id for row in sources} == set(selection[:2])
        route = await port.resolve_chat_route(ref, "fixture", "model")
        outputs = []
        replies = {
            "summary": "Rowan maps the sky [S1].",
            "flashcards": '{"items":[{"front":"Who?","back":"Rowan","citations":["S1"]}]}',
            "quiz": '{"items":[{"question":"Who?","options":["Rowan","Sam"],"correct_index":0,"citations":["S1"]}]}',
            "report": "The station maps the sky [S2].",
            "compare_sources": "The sources describe personnel [S1] and work [S2].",
        }
        for kind, response in replies.items():
            config = ResearchGenerationConfig(title=f"Observatory {kind}")
            request = ResearchStudioRequest(
                f"generate-{kind}", ref, kind, sources, config, route, 0
            )
            provider.response = response
            output = await port.generate_output(
                request, lambda _chunk: None, asyncio.Event()
            )
            assert output.status == "complete" and output.source_snapshot == sources
            outputs.append(output)
            assert other not in repr(provider.prompts[-1])
            assert all(body in repr(provider.prompts[-1]) for body in approved_texts)

        reopened = LocalStudioAdapter(
            local_studio.registry,
            LocalChatbookService(registry_path=tmp_path / "chatbooks.json"),
            local_studio.study,
            local_studio.quizzes,
        )
        history = await reopened.list_outputs(ref)
        assert set(history.items) == set(outputs)
        for output in history.items:
            record = await reopened.resolve_output(output)
            assert (
                record
                and record.get("title", record.get("name"))
                == f"Observatory {output.output_kind}"
            )
            _, target = output_handoff(output)
            assert (
                target.endswith(output.owner_id)
                if isinstance(target, str)
                else target.owner_id == output.owner_id
            )
            if output.output_kind == "flashcards":
                assert [
                    card["front"]
                    for card in local_studio.study.list_flashcards(
                        deck_id=output.owner_id
                    )
                ] == ["Who?"]
            elif output.output_kind == "quiz":
                assert (
                    local_studio.quizzes.list_questions(output.owner_id)["items"][0][
                        "question_text"
                    ]
                    == "Who?"
                )
            else:
                assert record["metadata"]["content"] == replies[output.output_kind]

        report = next(output for output in outputs if output.output_kind == "report")
        await port.set_selected_scope(ref, (selection[0],))
        current = await port.capture_chat_sources(ref)
        assert current != report.source_snapshot
        refresh = ResearchStudioRequest(
            "report-next", ref, "report", current, ResearchGenerationConfig(), route, 0
        )
        provider.response = "Updated from the first source [S1]."
        fresh = await port.generate_output(
            refresh, lambda _chunk: None, asyncio.Event(), supersedes=report
        )
        assert fresh.owner_id != report.owner_id and fresh.source_snapshot == current
        assert await reopened.resolve_output(report)
        await reopened.delete(fresh)
        assert await reopened.resolve_output(fresh) is None
        assert fresh not in (await reopened.list_outputs(ref)).items
    finally:
        media.close()
