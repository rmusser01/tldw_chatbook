import threading

import pytest

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole


class _Gateway:
    async def resolve_for_send(self, _selection):
        class _R:
            ready = True
            visible_copy = ""
        return _R()

    async def stream_chat(self, _resolution, _messages):
        if False:
            yield ""


def _controller(applier):
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_Gateway(),
        provider="llama_cpp",
        model="test-model",
        chat_dictionary_applier=applier,
    )
    return controller, store


def _session_with_conv(store, conv_id="conv-1"):
    session = store.create_session(title="t")
    session.persisted_conversation_id = conv_id
    return session


def _warden(conv_id, text):
    return text.replace("Warden", "grim jailer")


@pytest.mark.asyncio
async def test_substitutes_final_user_string(warden_applier=_warden):
    controller, store = _controller(warden_applier)
    session = _session_with_conv(store)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": ConsoleMessageRole.USER.value, "content": "The Warden nods."},
    ]
    out = await controller._apply_chat_dictionaries(messages, session.id)
    assert out[-1]["content"] == "The grim jailer nods."
    # Input list/dicts untouched (fresh copies only).
    assert messages[-1]["content"] == "The Warden nods."


@pytest.mark.asyncio
async def test_substitutes_text_part_of_parts_list_leaving_images():
    controller, store = _controller(_warden)
    session = _session_with_conv(store)
    image_part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    messages = [
        {
            "role": ConsoleMessageRole.USER.value,
            "content": [{"type": "text", "text": "The Warden nods."}, image_part],
        }
    ]
    out = await controller._apply_chat_dictionaries(messages, session.id)
    parts = out[-1]["content"]
    assert parts[0] == {"type": "text", "text": "The grim jailer nods."}
    assert parts[1] is image_part


@pytest.mark.asyncio
async def test_skips_skill_command_message():
    controller, store = _controller(_warden)
    session = _session_with_conv(store)
    messages = [{"role": ConsoleMessageRole.USER.value, "content": "/Warden do a thing"}]
    out = await controller._apply_chat_dictionaries(messages, session.id)
    assert out[-1]["content"] == "/Warden do a thing"


@pytest.mark.asyncio
async def test_only_final_user_message_substituted():
    controller, store = _controller(_warden)
    session = _session_with_conv(store)
    messages = [
        {"role": ConsoleMessageRole.USER.value, "content": "The Warden earlier."},
        {"role": "assistant", "content": "ok"},
        {"role": ConsoleMessageRole.USER.value, "content": "The Warden now."},
    ]
    out = await controller._apply_chat_dictionaries(messages, session.id)
    assert out[0]["content"] == "The Warden earlier."
    assert out[-1]["content"] == "The grim jailer now."


@pytest.mark.asyncio
async def test_no_applier_returns_input_unchanged():
    controller, store = _controller(None)
    session = _session_with_conv(store)
    messages = [{"role": ConsoleMessageRole.USER.value, "content": "The Warden nods."}]
    out = await controller._apply_chat_dictionaries(messages, session.id)
    assert out[-1]["content"] == "The Warden nods."


@pytest.mark.asyncio
async def test_unsaved_session_returns_input_unchanged():
    controller, store = _controller(_warden)
    session = store.create_session(title="t")  # persisted_conversation_id stays None
    messages = [{"role": ConsoleMessageRole.USER.value, "content": "The Warden nods."}]
    out = await controller._apply_chat_dictionaries(messages, session.id)
    assert out[-1]["content"] == "The Warden nods."


@pytest.mark.asyncio
async def test_applier_runs_off_the_event_loop():
    loop_thread = threading.get_ident()
    seen = {}

    def _recording_applier(conv_id, text):
        seen["thread"] = threading.get_ident()
        return text.replace("Warden", "grim jailer")

    controller, store = _controller(_recording_applier)
    session = _session_with_conv(store)
    messages = [{"role": ConsoleMessageRole.USER.value, "content": "The Warden nods."}]
    out = await controller._apply_chat_dictionaries(messages, session.id)
    assert out[-1]["content"] == "The grim jailer nods."
    assert seen["thread"] != loop_thread  # offloaded via asyncio.to_thread


@pytest.mark.asyncio
async def test_applier_exception_returns_input_unchanged():
    def _boom(conv_id, text):
        raise RuntimeError("applier exploded")

    controller, store = _controller(_boom)
    session = _session_with_conv(store)
    messages = [{"role": ConsoleMessageRole.USER.value, "content": "The Warden nods."}]
    out = await controller._apply_chat_dictionaries(messages, session.id)
    assert out[-1]["content"] == "The Warden nods."
