from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from textual.worker import WorkerState

from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Chat.citation_evidence_models import EvidenceBundle, EvidenceReference
from tldw_chatbook.Event_Handlers.worker_events import handle_api_call_worker_state_changed


@pytest.mark.asyncio
async def test_non_streaming_worker_attaches_answer_citation_validation() -> None:
    """A direct worker response validates citation labels against staged evidence."""
    markdown_widget = MagicMock()
    header_widget = MagicMock()

    def query_message(selector: str, *_args):
        if selector == ".message-text":
            return markdown_widget
        if selector == ".message-header":
            return header_widget
        raise AssertionError(f"Unexpected message query: {selector}")

    ai_widget = SimpleNamespace(
        is_mounted=True,
        message_text="",
        role="AI",
        parent=None,
        query_one=query_message,
        mark_generation_complete=MagicMock(),
    )
    chat_container = SimpleNamespace(is_mounted=True, scroll_end=MagicMock())
    screen = SimpleNamespace(query_one=MagicMock(return_value=chat_container))
    evidence_bundle = EvidenceBundle(
        bundle_id="library-rag:incident",
        query="Why did the incident happen?",
        references=(
            EvidenceReference(
                evidence_id="S1",
                source_id="note-1",
                source_type="note",
                title="Incident Review",
                snippet="Expired credential caused the incident.",
                authority_label="Source authority: local",
            ),
        ),
    )
    app = SimpleNamespace(
        loguru_logger=MagicMock(),
        screen=screen,
        current_ai_message_widget=ai_widget,
        current_chat_is_ephemeral=True,
        current_chat_conversation_id=None,
        current_chat_active_character_data=None,
        current_chat_is_streaming=False,
        chachanotes_db=None,
        app_config={"chat_defaults": {}},
        is_mounted=False,
        _current_chat_handoff_payload=ChatHandoffPayload(
            source="search-rag",
            item_type="rag-result",
            title="Incident Review",
            body="Retrieved content",
            metadata={"evidence_bundle": evidence_bundle.to_payload()},
        ).to_dict(),
    )
    event = SimpleNamespace(
        state=WorkerState.SUCCESS,
        worker=SimpleNamespace(
            name="API_Call_chat",
            result="The credential expired [S1].",
        ),
    )

    with patch("tldw_chatbook.config.get_cli_setting", return_value=False):
        await handle_api_call_worker_state_changed(app, event)

    assert ai_widget.citation_validation.status == "validated"
    assert ai_widget.citation_refs[0].evidence_id == "S1"
    assert app._current_chat_answer_citation_validation["status"] == "validated"
