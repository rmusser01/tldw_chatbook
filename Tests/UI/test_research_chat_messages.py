from types import SimpleNamespace

import pytest
from textual.widgets import Button, Input, ListView, Select, Static, TextArea

from Tests.app_module_patches import patch_app_global
from Tests.UI.library_quit_guard_support import _settings_without_splash, _until
from Tests.UI.test_research_chat_region import build_research_app, open_chat
from tldw_chatbook.UI.Research_Workspace_Modules.chat_message import ResearchChatMessage


pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


async def test_selected_answer_opens_existing_quick_note_editor_and_carries_provenance(
    tmp_path, monkeypatch
):
    app, db, _ = build_research_app(tmp_path, monkeypatch)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(160, 40)) as pilot:
            screen, composer = await open_chat(app, pilot)
            await send_question(screen, composer, pilot)
            session_id = screen.query_one("#research-chat-session", Select).value
            saved = db.get_messages_for_conversation(session_id)[-1]
            capture = screen.query_one("#research-quick-note-capture-message", Button)
            await _until(
                pilot,
                lambda: not capture.disabled,
                "selected Chat message available in Studio",
            )
            capture.press()
            note = screen.query_one("#research-quick-note-body", TextArea)
            await _until(
                pilot,
                lambda: note.text == saved["content"],
                "answer in the Quick Note editor",
            )
            editor = screen.query_one("#research-quick-notes-section")
            ref, request = editor.capture_save_request()
            assert ref.workspace_id == "study"
            assert request.message_ids == (saved["id"],)


async def test_local_message_delete_undo_and_branch_use_the_saved_transcript(
    tmp_path, monkeypatch
):
    app, db, _ = build_research_app(tmp_path, monkeypatch)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(160, 40)) as pilot:
            screen, composer = await open_chat(app, pilot)
            await send_question(screen, composer, pilot)
            chooser = screen.query_one("#research-chat-session", Select)
            session_id = chooser.value
            saved = db.get_messages_for_conversation(session_id)[-1]

            screen.query_one("#research-chat-delete", Button).press()
            await _until(
                pilot,
                lambda: (
                    app.screen is not screen
                    and bool(list(app.screen.query("#confirm-button")))
                ),
                "delete confirmation",
            )
            app.screen.query_one("#confirm-button", Button).press()
            await _until(
                pilot,
                lambda: len(db.get_messages_for_conversation(session_id)) == 1,
                "canonical delete",
            )
            undo = screen.query_one("#research-chat-undo", Button)
            await _until(pilot, lambda: not undo.disabled, "undo available")
            undo.press()
            await _until(
                pilot,
                lambda: len(db.get_messages_for_conversation(session_id)) == 2,
                "canonical restore",
            )
            branch = screen.query_one("#research-chat-branch", Button)
            await _until(pilot, lambda: not branch.disabled, "branch available")
            branch.press()
            await _until(
                pilot, lambda: chooser.value != session_id, "canonical branch selected"
            )
            branch_id = chooser.value
            conversation = db.get_conversation_by_id(branch_id)
            assert conversation["parent_conversation_id"] == session_id
            assert conversation["forked_from_message_id"] == saved["id"]
            assert (
                db.get_messages_for_conversation(branch_id)[-1]["content"]
                == saved["content"]
            )


async def test_local_edit_and_regenerate_preserve_the_existing_draft(
    tmp_path, monkeypatch
):
    app, db, _ = build_research_app(tmp_path, monkeypatch)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(120, 30)) as pilot:
            screen, composer = await open_chat(app, pilot)
            await send_question(screen, composer, pilot)
            transcript = screen.query_one("#research-chat-transcript", ListView)
            chooser = screen.query_one("#research-chat-session", Select)
            original_id = chooser.value
            transcript.index = 0
            await _until(
                pilot,
                lambda: transcript.highlighted_child.row["sender"] == "user",
                "question selected",
            )
            screen.query_one("#research-chat-edit", Button).press()
            await _until(
                pilot,
                lambda: (
                    app.screen is not screen
                    and bool(list(app.screen.query("#research-chat-edit-save")))
                ),
                "edit modal",
            )
            app.screen.query_one("#research-chat-edit-body", TextArea).load_text(
                "Revised question"
            )
            app.screen.query_one("#research-chat-edit-save", Button).press()
            regenerate = screen.query_one("#research-chat-regenerate", Button)
            status = screen.query_one("#research-chat-status", Static)
            await _until(
                pilot,
                lambda: "Message saved." in str(status.render()),
                "saved revision",
            )
            assert [
                row["content"] for row in db.get_messages_for_conversation(original_id)
            ][0] == "Revised question"

            composer.load_text("An unsent draft")
            regenerate.press()
            await _until(
                pilot,
                lambda: chooser.value != original_id and not regenerate.disabled,
                "regenerated answer",
            )
            branch_id = chooser.value
            assert (
                db.get_conversation_by_id(branch_id)["parent_conversation_id"]
                == original_id
            )
            rows = db.get_messages_for_conversation(branch_id)
            assert [row["sender"] for row in rows] == ["user", "assistant"]
            assert rows[0]["content"] == "Revised question"
            assert composer.text == "An unsent draft"


async def test_read_aloud_uses_the_confirmed_local_speech_destination(
    tmp_path, monkeypatch
):
    app, _, provider = build_research_app(tmp_path, monkeypatch)
    spoken = []
    destination = SimpleNamespace(
        provider_label="Test voice",
        sanitized_destination="on device",
        fingerprint="speech-fixture",
    )

    class Speech:
        async def resolve_console_speech_destination(
            self, assistant_kind, character_ref
        ):
            assert assistant_kind is character_ref is None
            return destination

        async def speak_guarded_utterance(
            self, text, *, validator, expected_destination_fingerprint, **_identity
        ):
            assert validator()
            assert expected_destination_fingerprint == destination.fingerprint
            spoken.append(text)
            return True

    async def speech_handler():
        return Speech()

    monkeypatch.setattr(app, "_ensure_tts_handler", speech_handler)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(160, 40)) as pilot:
            screen, composer = await open_chat(app, pilot)
            await send_question(screen, composer, pilot)
            button = screen.query_one("#research-chat-speak", Button)
            await _until(
                pilot,
                lambda: button.display and not button.disabled,
                "speech route resolved",
            )
            button.press()
            await _until(
                pilot,
                lambda: (
                    app.screen is not screen
                    and bool(list(app.screen.query("#confirm-button")))
                ),
                "speech confirmation",
            )
            assert not spoken
            app.screen.query_one("#confirm-button", Button).press()
            await _until(
                pilot, lambda: spoken == [provider.answer], "selected answer spoken"
            )


async def send_question(screen, composer, pilot):
    screen.query_one("#research-chat-provider", Input).value = "fixture"
    screen.query_one("#research-chat-model", Input).value = "model"
    screen.query_one("#research-chat-mode", Select).value = "general"
    composer.load_text("Please answer briefly")
    screen.query_one("#research-chat-send", Button).press()
    transcript = screen.query_one("#research-chat-transcript", ListView)
    button = screen.query_one("#research-chat-note", Button)
    await _until(
        pilot,
        lambda: len(transcript.children) == 2 and not button.disabled,
        "saved transcript and actions",
    )
    assert isinstance(transcript.highlighted_child, ResearchChatMessage)
