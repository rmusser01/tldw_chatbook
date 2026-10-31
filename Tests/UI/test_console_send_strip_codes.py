"""TASK-33621.11 AC#2: a staged image the model can't take names the image.

Live on dev (claude-haiku-4-5, a staged PNG): the composer read "Send
blocked — choose a model to continue ›", a link into the first-run setup
wizard, though a model was selected. The refusal rode the setup slot and
the strip matched the word "model" in it.

These drive the real ChatScreen, store, readiness and composer. The model
is the harness's llama.cpp ``local-model``, which the real capability tables
do not list, so the real classifier refuses the image: nothing is stubbed.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_console_dictation import _mounted_console, _ready_host
from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Widgets.Console import ConsoleComposerBar

pytestmark = pytest.mark.bootstrap_profile

MODEL = "local-model"


def _image() -> PendingAttachment:
    return PendingAttachment(
        file_path="/tmp/leaf.png",
        display_name="leaf.png",
        file_type="image",
        insert_mode="attachment",
        data=b"\x89PNG-bytes",
        mime_type="image/png",
        original_size=116,
        processed_size=116,
    )


@pytest.mark.asyncio
async def test_a_staged_image_the_model_cannot_take_names_the_image_not_setup():
    _app, host = _ready_host()
    async with host.run_test(size=(235, 52)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        store = console._ensure_console_chat_store()
        session_id = store.ensure_session().id
        composer.load_draft("what colour is this leaf?")
        assert store.add_pending_attachment(session_id, _image())
        console._sync_console_control_bar()
        await pilot.pause()

        # A model IS selected: readiness is ready; only the image blocks.
        assert console._console_setup_blocked_reason() == ""
        verbatim = console._console_attachment_blocked_reason()
        assert verbatim.startswith(f"Console send blocked: {MODEL} can't accept images.")

        strip = composer.query_one("#console-send-disabled-reason", Static)
        assert strip.content.plain == f"Send blocked — {MODEL} can't accept images"
        assert "choose a model" not in strip.content.plain
        assert "run_setup_wizard" not in strip.content.markup
        assert not composer.has_class("console-composer-setup-blocked")
        send = composer.query_one("#console-send-message", Button)
        assert send.disabled
        assert send.tooltip == verbatim
