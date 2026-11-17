"""Original composer attachment reads over actual finite published values."""

import base64
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_execution_readers import _publication, _reader_client
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
from tldw_chatbook.Chat.console_execution_projection_view import (
    ConsoleProjectionUnavailable,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar


def _inputs(session, attachments):
    return {
        "session_id": session.id,
        "incarnation_id": session.incarnation_id,
        "draft": session.draft,
        "authored_token": None,
        "settings": None,
        "settings_revision": session.settings_revision,
        "attachment_revision": 0,
        "attachment_ids": [item["attachment_id"] for item in attachments],
        "attachments": attachments,
        "one_shot_prefill": None,
        "prefill_revision": 0,
    }


def _attachment(identity, *, file_type="image", data=b"image"):
    return {
        "attachment_id": identity,
        "file_path": "selected",
        "display_name": "Selected",
        "file_type": file_type,
        "insert_mode": "attachment",
        "data": None if data is None else base64.b64encode(data).decode("ascii"),
        "mime_type": "image/png" if file_type == "image" else None,
        "text_content": None,
        "original_size": 5,
        "processed_size": 5,
    }


def test_original_composer_load_reads_empty_and_qualifying_published_attachments():
    session = ConsoleChatSession()
    client = _reader_client(_publication(session))
    screen = SimpleNamespace(_console_chat_store=client.store_reader)
    client.pressed_inputs = _inputs(session, [])
    composer = ConsoleComposerBar(
        sendable_attachment_provider=lambda: ChatScreen._console_pending_image_attachment(
            screen
        )
        is not None,
    )
    # This is the original failing Compose call, without mounting a fake App.
    composer.load_draft("")
    assert not composer.has_sendable_payload()
    client.pressed_inputs = _inputs(
        session,
        [
            _attachment("document", file_type="document"),
            _attachment("missing-image", data=None),
            _attachment("image"),
        ],
    )
    composer.load_draft("")
    assert composer.has_sendable_payload()
    image = ChatScreen._console_pending_image_attachment(screen)
    assert image.attachment_id == "image" and image.data == b"image"
    image.data = b"presentation mutation"
    assert ChatScreen._console_pending_image_attachment(screen).data == b"image"
    assert len(client.store_reader.pending_attachments(session.id)) == 3
    client.pressed_inputs = _inputs(session, [])
    composer.load_draft("")
    assert not composer.has_sendable_payload()


def test_staged_attachment_reader_refuses_mixed_session_incarnation_and_ids():
    session = ConsoleChatSession()
    client = _reader_client(_publication(session))
    body = _inputs(session, [_attachment("image")])
    for changed in (
        dict(body, session_id="other"),
        dict(body, incarnation_id="other"),
        dict(body, attachment_ids=["other"]),
    ):
        client.pressed_inputs = changed
        with pytest.raises(
            ConsoleProjectionUnavailable, match="attachment publication"
        ):
            client.store_reader.pending_attachments(session.id)
