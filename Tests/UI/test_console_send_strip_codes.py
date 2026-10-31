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


# --- Review of TASK-33621.11: the code rides beside the sentence -----------

import copy  # noqa: E402

from Tests.UI.app_factory import attach_chachanotes_db  # noqa: E402
from Tests.UI.test_console_native_chat_flow import (  # noqa: E402
    _configure_native_ready_console,
)
from Tests.UI.test_destination_shells import _build_test_app  # noqa: E402
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (  # noqa: E402
    ConsoleHarness,
)
from tldw_chatbook.Chat.citation_evidence_models import (  # noqa: E402
    EvidenceBundle,
    EvidenceReference,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole  # noqa: E402
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch  # noqa: E402
from tldw_chatbook.Chat.console_session_settings import (  # noqa: E402
    ConsoleSessionSettings,
)

#: The refused Enter's row for an OpenAI session with no key (the setup
#: sentence, which stands in for the gate's own "Console send blocked: ...").
NO_KEY_ROW = "Add API key in Settings > Providers & Models before sending."


def _system_rows(store, session_id) -> list:
    return [
        message
        for message in store.messages_for_session(session_id)
        if message.role is ConsoleMessageRole.SYSTEM
    ]


async def _refused_enter(console, composer, pilot) -> None:
    send = composer.query_one("#console-send-message", Button)
    await console.handle_console_send_message(Button.Pressed(send))
    await pilot.pause(0.1)


def _openai_without_a_key(store, session_id) -> None:
    store.replace_session_settings(
        session_id,
        ConsoleSessionSettings(provider="openai", model="gpt-4.1-2025-04-14"),
    )


@pytest.mark.asyncio
async def test_a_refused_enter_writes_plain_sentences_the_context_view_can_copy(
    monkeypatch,
):
    """A coded blocker never reaches the transcript as anything but a str.

    The first cut made the blocker a ``str`` subclass whose ``__new__``
    required a keyword-only code, so ``copy.deepcopy`` raised TypeError on
    it. A refused Enter wrote it as a System row's content, and the Inspect
    Context / Next Send view deep-copies every row -- so staging a PNG the
    model can't take, pressing Enter, then opening that view raised.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
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
        await _refused_enter(console, composer, pilot)  # the image blocks

        _openai_without_a_key(store, session_id)
        await console._sync_native_console_chat_ui()
        composer.load_draft("and this one?")  # a new press, not a repeat
        await _refused_enter(console, composer, pilot)  # setup blocks

        rows = _system_rows(store, session_id)
        assert [row.content[:40] for row in rows] == [
            f"Console send blocked: {MODEL} can't accept images."[:40],
            NO_KEY_ROW[:40],
        ]
        # What the Inspect Context / Next Send view does to every row.
        copy.deepcopy(store.messages_for_session(session_id))
        controller = console._ensure_console_chat_controller()
        snapshot = await controller.build_context_snapshot("", session_id=session_id)
        assert snapshot is not None
        assert [type(row.content) for row in rows] == [str, str]
        assert rows[1].content == NO_KEY_ROW
        assert composer.draft_text() == "and this one?"


@pytest.mark.asyncio
async def test_a_refused_enter_names_the_blocker_the_gate_stopped_on(monkeypatch):
    """TASK-38331, folded in at review: the row follows the gate's code.

    The send gate stops on missing Library evidence before provider setup.
    prompt_queue let the setup sentence stand in for the gate's unless the
    gate's sentence began "Console send blocked: Library Search/RAG" -- a
    prefix the evidence copy has not had since it was renamed "Library
    search", so a provider gap hid the evidence blocker the gate stopped on.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _build_test_app()
    _configure_native_ready_console(app)
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)
    blocked = EvidenceReference(
        evidence_id="S1",
        source_id="media-1",
        source_type="media",
        title="Source 1",
        snippet="Body 1",
        authority_label="local",
        status="blocked",
        source_owner="local",
        score=None,
        metadata={},
    )
    bundle = EvidenceBundle(
        bundle_id="bundle-blocked",
        query="question",
        source="Library Search/RAG",
        references=(blocked,),
    )
    launch = ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title="Library Search/RAG retrieval",
        payload={"query": "question", "evidence_bundle": bundle.to_payload()},
        status="staged",
    )
    async with host.run_test(size=(235, 52)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        store = console._ensure_console_chat_store()
        session_id = store.ensure_session().id
        _openai_without_a_key(store, session_id)
        console._retrieval._stage_console_library_rag_launch(launch)
        await console._sync_native_console_chat_ui()
        assert console._console_setup_blocked_reason() == NO_KEY_ROW
        composer.load_draft("answer from the staged evidence")
        await _refused_enter(console, composer, pilot)

        rows = _system_rows(store, session_id)
        assert [row.content for row in rows] == [
            "Console send blocked: Library search has no available evidence. "
            "Review source authority before sending."
        ]
