"""TASK-33620.15.22: one rule decides whether a staged attachment makes Send ready.

Send's readiness is synced from two places: the screen's sync
(``ChatScreen._sync_console_composer_action_state``) and the composer's own
resync on keystrokes and after a Send's acceptance
(``ConsoleComposerBar._sync_current_action_state``). On dev 3be493413a the
screen counted a staged image that carries bytes, the one staged item its send
gate (``_send_console_message_from_visible_action_observed``) sends with an
empty draft, while the composer counted any attachment label. Whichever ran
last decided Send's state, so a staged item the gate refuses flipped Send
between ready and disabled, and a label the screen had not yet cleared kept
Send ready after its image had left the staging list.

Both sites now read the composer's ``has_sendable_payload``, and the screen
answers its attachment half with the send gate's own predicate. A ready Send
is one whose press sends.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.test_console_dictation import _mounted_console, _ready_host
from tldw_chatbook.Chat import attachment_core
from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Widgets.Console import ConsoleComposerBar

# The harness mounts the real ChatScreen, controller and store.
pytestmark = pytest.mark.bootstrap_profile

#: Every kind of item the store accepts for staging, and whether the send gate
#: sends it with an empty draft. Only the image is produced by today's staging
#: paths (file picker, clipboard, refused-echo resend); a text, code, data, PDF
#: or document file is inserted into the draft instead. The others are kinds
#: the store would hold, so Send must not claim to send them either.
_STAGED_KINDS = (
    (
        "image",
        {"file_type": "image", "insert_mode": "attachment", "data": b"\x89PNG"},
        True,
    ),
    (
        "document-with-bytes",
        {"file_type": "document", "insert_mode": "attachment", "data": b"%PDF-1.4"},
        False,
    ),
    (
        "image-without-bytes",
        {"file_type": "image", "insert_mode": "attachment", "data": None},
        False,
    ),
    (
        "inline-text",
        {
            "file_type": "text",
            "insert_mode": "inline",
            "data": None,
            "text_content": "hello",
        },
        False,
    ),
)


def _attachment(name: str, fields: dict) -> PendingAttachment:
    return PendingAttachment(
        file_path=f"/tmp/{name}",
        display_name=name,
        original_size=8,
        processed_size=8,
        **fields,
    )


def _send_state(send: Button) -> tuple[bool, bool]:
    """Send's readiness as both sites paint it: (disabled, inactive)."""
    return send.disabled, send.has_class("console-send-inactive")


async def _ready_console(host, pilot, monkeypatch):
    # A staged image must not be refused for vision here, so its readiness is
    # decided by the payload rule alone.
    monkeypatch.setattr(attachment_core, "vision_block_reason", lambda *_a, **_k: None)
    console = await _mounted_console(host, pilot)
    composer = console.query_one("#console-native-composer", ConsoleComposerBar)
    send = composer.query_one("#console-send-message", Button)
    store = console._ensure_console_chat_store()
    session_id = store.ensure_session().id
    composer.load_draft("")
    await pilot.pause()
    return console, composer, send, store, session_id


@pytest.mark.asyncio
async def test_a_staged_attachment_gets_the_same_send_readiness_from_either_sync_site(
    monkeypatch,
):
    _app, host = _ready_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console, composer, send, store, session_id = await _ready_console(
            host, pilot, monkeypatch
        )
        for name, fields, sendable in _STAGED_KINDS:
            composer.load_draft("")
            store.clear_pending_attachments(session_id)
            assert store.add_pending_attachment(session_id, _attachment(name, fields))

            composer._sync_current_action_state()
            console._sync_console_control_bar()
            screen_last = _send_state(send)
            composer._sync_current_action_state()
            composer_last = _send_state(send)

            assert composer._pending_attachment_label is not None, name
            assert screen_last == composer_last, name
            assert composer_last == (not sendable, not sendable), name
            if sendable:
                continue
            # A disabled Send is what pressing it does: nothing is sent and
            # the item stays staged.
            before = len(store.messages_for_session(session_id))
            assert await console._send_console_message_from_visible_action() is False
            assert len(store.messages_for_session(session_id)) == before, name
            assert len(store.pending_attachments(session_id)) == 1, name


@pytest.mark.asyncio
async def test_the_composer_resync_reads_what_is_staged_not_the_painted_label(
    monkeypatch,
):
    """A Send's custody takes the image before the screen repaints its label."""
    _app, host = _ready_host()
    async with host.run_test(size=(140, 42)) as pilot:
        console, composer, send, store, session_id = await _ready_console(
            host, pilot, monkeypatch
        )
        image = _attachment("photo.png", _STAGED_KINDS[0][1])
        assert store.add_pending_attachment(session_id, image)
        console._sync_console_control_bar()
        assert _send_state(send) == (False, False)
        assert composer._pending_attachment_label is not None

        store.transfer_pending_attachments_to_turn(
            session_id, "turn-1", (image.attachment_id,)
        )
        composer._sync_current_action_state()

        assert _send_state(send) == (True, True)
        assert await console._send_console_message_from_visible_action() is False
