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
        world_info_applier=applier,
    )
    return controller, store


def _session_with_conv(store, conv_id="conv-1"):
    session = store.create_session(title="t")
    session.persisted_conversation_id = conv_id
    return session


def _stub_wi(conv_id, text, history):
    return f"[WI]\n\n{text}"


@pytest.mark.asyncio
async def test_apply_world_info_wraps_final_user_message():
    controller, store = _controller(_stub_wi)
    session = _session_with_conv(store)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "earlier reply"},
        {"role": ConsoleMessageRole.USER.value, "content": "a dragon appears"},
    ]
    out = await controller._apply_world_info(messages, session.id)
    assert out[-1]["content"] == "[WI]\n\na dragon appears"
    # Earlier messages untouched.
    assert out[0] == {"role": "system", "content": "sys"}
    assert out[1] == {"role": "assistant", "content": "earlier reply"}
    # Input list/dicts untouched (fresh copies only).
    assert messages[-1]["content"] == "a dragon appears"


@pytest.mark.asyncio
async def test_apply_world_info_noop_without_conversation():
    controller, store = _controller(_stub_wi)
    session = store.create_session(title="t")  # persisted_conversation_id stays None
    messages = [{"role": ConsoleMessageRole.USER.value, "content": "a dragon appears"}]
    out = await controller._apply_world_info(messages, session.id)
    assert out[-1]["content"] == "a dragon appears"


@pytest.mark.asyncio
async def test_apply_world_info_multimodal_history_and_message():
    captured = {}

    def _recording_wi(conv_id, text, history):
        captured["message_text"] = text
        captured["history"] = history
        assert isinstance(text, str)
        assert all(isinstance(h["content"], str) for h in history)
        return f"[WI]\n\n{text}"

    controller, store = _controller(_recording_wi)
    session = _session_with_conv(store)
    image_part = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAAA"},
    }
    history_image_part = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,BBBB"},
    }
    messages = [
        {
            "role": ConsoleMessageRole.USER.value,
            "content": [
                {"type": "text", "text": "earlier turn"},
                history_image_part,
            ],
        },
        {"role": "assistant", "content": "ok"},
        {
            "role": ConsoleMessageRole.USER.value,
            "content": [{"type": "text", "text": "a dragon"}, image_part],
        },
    ]
    out = await controller._apply_world_info(messages, session.id)

    # Stub received string types, proving normalization happened.
    assert captured["message_text"] == "a dragon"
    assert captured["history"][0]["content"] == "earlier turn"

    parts = out[-1]["content"]
    assert parts[0] == {"type": "text", "text": "[WI]\n\na dragon"}
    assert parts[1] is image_part


@pytest.mark.asyncio
async def test_apply_world_info_command_message_skipped():
    controller, store = _controller(_stub_wi)
    session = _session_with_conv(store)
    messages = [
        {"role": ConsoleMessageRole.USER.value, "content": "/do a thing"}
    ]
    out = await controller._apply_world_info(messages, session.id)
    assert out[-1]["content"] == "/do a thing"


@pytest.mark.asyncio
async def test_apply_world_info_applier_none():
    controller, store = _controller(None)
    session = _session_with_conv(store)
    messages = [{"role": ConsoleMessageRole.USER.value, "content": "a dragon appears"}]
    out = await controller._apply_world_info(messages, session.id)
    assert out[-1]["content"] == "a dragon appears"
