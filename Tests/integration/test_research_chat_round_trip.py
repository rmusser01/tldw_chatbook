"""Exercise a Local answer against canonical Media FTS, Chat, and Notes."""

import json

import pytest
from textual.widgets import Button, Input, ListView, Select, Static, TextArea

from Tests.Chat.test_citation_trace_repository import _repository
from Tests.UI.app_factory import swap_app_aliases
from Tests.UI.library_quit_guard_support import _settings_without_splash, _until
from Tests.UI.test_research_chat_region import build_research_app, open_chat
from Tests.app_module_patches import patch_app_global
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Media.local_media_reading_service import LocalMediaReadingService
from tldw_chatbook.Media.media_reading_scope_service import MediaReadingScopeService
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


async def test_selected_library_media_grounded_answer_reopens_with_citations_and_note(
    tmp_path, monkeypatch
):
    app, db, provider = build_research_app(tmp_path, monkeypatch)
    media = MediaDatabase(tmp_path / "library.sqlite", client_id="research-integration")
    citation_repository = _repository(db)
    notes = NotesInteropService(
        base_db_directory=tmp_path,
        api_client_id="test-client",
        global_db_to_use=db,
    )
    swap_app_aliases(
        app,
        media_db=media,
        media_reading_scope_service=MediaReadingScopeService(
            LocalMediaReadingService(media), None
        ),
        citation_trace_repository=citation_repository,
        notes_scope_service=NotesScopeService(
            local_notes_service=notes, server_service=None
        ),
    )
    provider.answer = "The observatory is north [S1]."
    selected = "The observatory is at the northern station."
    excluded = "The observatory records control data at the excluded station."
    registry = app.workspace_registry_service
    membership_ids = []
    for title, body in (("Selected source", selected), ("Unselected source", excluded)):
        media_id, _, _ = media.add_media_with_keywords(
            title=title,
            media_type="document",
            content=body,
            chunks=[{"text": body, "start_char": 0, "end_char": len(body)}],
        )
        assert media_id is not None
        membership_ids.append(
            registry.link_membership(
                "study", item_type="media", item_id=str(media_id), title=title
            )
        )
    selection = membership_ids[0]

    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(160, 40)) as pilot:
            screen, composer = await open_chat(app, pilot)
            ref = screen.controller.selected_ref
            port = screen.controller.port_for_data_source(ref.data_source)
            await port.set_selected_scope(ref, (selection.item_id,))
            screen.query_one("#research-chat-provider", Input).value = "fixture"
            screen.query_one("#research-chat-model", Input).value = "model"
            screen.query_one("#research-chat-mode", Select).value = "rag"
            composer.load_text("observatory")
            screen.query_one("#research-chat-send", Button).press()
            status = screen.query_one("#research-chat-status", Static)
            try:
                await _until(
                    pilot,
                    lambda: "Answer saved" in str(status.render()),
                    "grounded answer persisted",
                )
            except AssertionError as exc:
                raise AssertionError(
                    f"Grounded Chat displayed {status.render()}"
                ) from exc
            session = screen.query_one("#research-chat-session", Select).value
            assert len(provider.requests) == 1
            assert selected in repr(provider.requests[0])
            assert excluded not in repr(provider.requests[0])
            assert [row["role"] for row in provider.requests[0]] == ["system", "user"]
            answer = db.get_messages_for_conversation(session)[-1]
            details = json.loads(answer["metadata_json"])["research_chat"]
            assert details["effective_retrieval"] == "fts"
            assert {cite["source_id"] for cite in details["citations"]} == {
                selection.membership_id
            }
            assert citation_repository.active_owner_candidate_message_ids(
                [answer["id"]]
            ) == {answer["id"]}

            await pilot.press("ctrl+1")
            await _until(pilot, lambda: app.current_tab == "home", "left Research")
            await pilot.press("f5")
            screen, _ = await open_chat(app, pilot)
            assert screen.query_one("#research-chat-session", Select).value == session
            screen.query_one("#research-chat-citations", Button).press()
            await _until(pilot, lambda: app.screen is not screen, "citations reloaded")
            assert f"Library media {selection.item_id}" in " ".join(
                str(label.render()) for label in app.screen.query(Static)
            )
            await pilot.press("escape")
            screen.query_one("#research-chat-note", Button).press()
            editor = screen.query_one("#research-quick-notes-section")
            body = editor.query_one("#research-quick-note-body", TextArea)
            await _until(
                pilot, lambda: body.text == provider.answer, "captured Quick Note"
            )
            _, request = editor.capture_save_request()
            assert request.source_ids == (selection.membership_id,)
            assert request.message_ids == (answer["id"],)
            screen.query_one("#research-quick-note-save", Button).press()
            await _until(
                pilot, lambda: not editor.is_dirty, "canonical Quick Note saved"
            )
            _, saved_note = editor.capture_save_request()
            assert saved_note.note_id

    media.close()
    reopened_app, reopened_db, _ = build_research_app(tmp_path, monkeypatch)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with reopened_app.run_test(size=(160, 40)) as pilot:
            screen, _ = await open_chat(reopened_app, pilot)
            assert screen.query_one("#research-chat-session", Select).value == session
            assert (
                len(screen.query_one("#research-chat-transcript", ListView).children)
                == 2
            )
            assert (
                reopened_db.get_message_by_id(answer["id"])["metadata_json"]
                == answer["metadata_json"]
            )
            assert (
                reopened_db.get_note_by_id(saved_note.note_id)["content"]
                == provider.answer
            )
