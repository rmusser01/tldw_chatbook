"""The bridge half of TASK-26000: the redirect abort cuts ONLY the primary's
in-flight stream, and a cut prose turn gets a separator so the re-asked
turn's chunks don't glue onto the partial in the same transcript message
(review F2)."""

from __future__ import annotations

import asyncio
import threading

from tldw_chatbook.Agents.agent_service import SUBAGENT_SYSTEM_PROMPT
from tldw_chatbook.Chat.console_agent_bridge import _StreamingModelAdapter
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore


def _adapter_harness(abort, chunks=("Looking at ", "the JSON parser", ", the rest")):
    emitted = []

    def chat_api_call(**kwargs):
        def stream():
            for chunk in chunks:
                emitted.append(chunk)
                yield {"choices": [{"delta": {"content": chunk}}]}
            yield {"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]}

        return stream()

    gateway = ConsoleProviderGateway(chat_api_call_fn=chat_api_call)
    store = ConsoleChatStore()
    session = store.ensure_session()
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    resolution = ConsoleProviderResolution(
        provider="QwenCloud",
        base_url="https://workspace.example.test/compatible-mode/v1",
        model="qwen3.8-max",
        ready=True,
        readiness_key="qwencloud",
        execution_key="qwencloud",
        api_key="qwen-test-key",
        streaming=True,
        api_mode="responses",
    )
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()
    adapter = _StreamingModelAdapter(
        store=store,
        provider_gateway=gateway,
        resolution=resolution,
        assistant_message_id=assistant.id,
        should_cancel=lambda: False,
        loop=loop,
        native_tools=False,
    )
    adapter._primary_stream_abort = abort
    return adapter, store, assistant.id, loop, emitted


def _shutdown(loop):
    loop.call_soon_threadsafe(loop.stop)


def test_redirect_cut_returns_the_partial_and_separates_the_row():
    emitted: list[str] = []

    def abort():
        # arm once the fake provider has emitted its second content chunk;
        # the consumer polls after processing each one, so the cut lands
        # between chunk 2 and chunk 3
        return len(emitted) >= 2

    adapter, store, message_id, loop, emitted = _adapter_harness(abort)
    try:
        response = adapter.chat_call(
            messages_payload=[
                {"role": "system", "content": "you are the primary console agent"},
                {"role": "user", "content": "analyze"},
            ]
        )
    finally:
        _shutdown(loop)

    content = response["choices"][0]["message"]["content"]
    # The gateway pre-fetches, so the exact cut chunk is timing-dependent;
    # the CONTRACT is truncation: the stream never completes.
    full = "Looking at the JSON parser, the rest"
    assert content in ("Looking at ", "Looking at the JSON parser"), (
        f"stream was not cut: {content!r}"
    )
    assert content != full
    row = store.get_message(message_id).content
    assert row.endswith("\n\n"), (
        f"review F2: no separator -- the re-run would glue onto {row!r}"
    )
    assert row[:-2] == content, f"row {row!r} vs partial {content!r}"


def test_a_subagent_stream_is_never_cut_by_the_primary_abort():
    adapter, store, message_id, loop, _emitted = _adapter_harness(lambda: True)
    try:
        response = adapter.chat_call(
            messages_payload=[
                {"role": "system", "content": SUBAGENT_SYSTEM_PROMPT + "\nchild"},
                {"role": "user", "content": "analyze"},
            ]
        )
    finally:
        _shutdown(loop)

    content = response["choices"][0]["message"]["content"]
    assert content == "Looking at the JSON parser, the rest", (
        f"child stream was cut by the primary's abort probe: {content!r}"
    )


def test_an_uncut_primary_stream_gets_no_separator():
    adapter, store, message_id, loop, _emitted = _adapter_harness(lambda: False)
    try:
        adapter.chat_call(
            messages_payload=[
                {"role": "system", "content": "you are the primary console agent"},
                {"role": "user", "content": "analyze"},
            ]
        )
    finally:
        _shutdown(loop)

    row = store.get_message(message_id).content
    assert not row.endswith("\n\n"), f"spurious separator on a normal turn: {row!r}"
