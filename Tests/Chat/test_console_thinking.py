"""Live thinking must remain separate from answers and private replay."""

import pytest

from tldw_chatbook.Chat.console_thinking import thinking_display, publish_thinking
from tldw_chatbook.Chat.llamacpp_think_filter import StartAnchoredThinkFilter
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway


def test_reasoning_mapping_is_displayed_without_becoming_answer_text():
    seen = []
    events = iter(
        [
            {"choices": [{"delta": {"reasoning_content": "Consider this"}}]},
            {"choices": [{"delta": {"content": "Answer"}}]},
        ]
    )
    with thinking_display(seen.append, provider="deepseek", model="deepseek-chat"):
        answer = list(ConsoleProviderGateway.normalize_provider_response(events))
    assert seen == ["Consider this"]
    assert answer == ["Answer"]


@pytest.mark.parametrize(
    "provider,model", [("moonshot", "kimi-k3"), ("openrouter", "moonshotai/kimi-k3")]
)
def test_private_kimi_reasoning_never_reaches_display(provider, model):
    seen = []
    with thinking_display(seen.append, provider=provider, model=model):
        publish_thinking("private reasoning")
    assert seen == []


def test_display_callback_does_not_escape_request_scope():
    seen = []
    with thinking_display(seen.append, provider="zai", model="glm"):
        publish_thinking("live")
    publish_thinking("auxiliary")
    assert seen == ["live"]


def test_split_think_tags_stream_live_and_preserve_literal_answer_tags():
    seen = []
    splitter = StartAnchoredThinkFilter(on_thinking=seen.append)
    assert splitter.feed("<thi") == ""
    assert splitter.feed("nk>Working it out") == ""
    assert "".join(seen) == "Working it out"
    assert splitter.feed("</thi") == ""
    assert (
        splitter.feed("nk>\nAnswer with <think>literal</think>")
        == "Answer with <think>literal</think>"
    )
    assert "".join(seen) == "Working it out"


@pytest.mark.asyncio
async def test_controller_retains_thinking_even_when_display_is_hidden():
    from Tests.Chat.test_console_chat_controller import StreamingGateway
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    import asyncio

    class ThinkingGateway(StreamingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            callback = kwargs["signals"].thinking_callback
            if callback:
                callback("Think through the question")
            await asyncio.sleep(0)
            yield "Answer"

    config = {"console": {}}
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=ThinkingGateway(), provider_config=lambda: config
    )
    await controller.submit_draft("hello")
    messages = store.messages_for_session(store.active_session_id)
    first = [m for m in messages if m.role is ConsoleMessageRole.ASSISTANT][-1]
    assert first.thinking_text == "Think through the question"
    assert first.content == "Answer"
    assert "Think through" not in repr(first)
    assert "Think through" not in str(
        controller._provider_messages_for_session(store.active_session_id)
    )
    config["console"]["show_thinking"] = False
    await controller.submit_draft("again")
    second = store.messages_for_session(store.active_session_id)[-1]
    assert second.thinking_text == "Think through the question"
    assert second.content == "Answer"


@pytest.mark.asyncio
@pytest.mark.parametrize("markdown", [True, False])
async def test_live_block_is_expanded_and_collapse_survives_updates(markdown):
    from textual.app import App
    from textual.widgets import Collapsible, Static
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleChatMessage,
        ConsoleMessageRole,
    )
    from tldw_chatbook.Widgets.Console.console_transcript import (
        ConsoleMarkdownMessage,
        ConsoleTranscriptMessage,
    )

    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="", status="streaming"
    )
    message.thinking_text = "Working it out"
    row_type = ConsoleMarkdownMessage if markdown else ConsoleTranscriptMessage

    class Host(App):
        def compose(self):
            yield row_type(message)

    host = Host()
    async with host.run_test() as pilot:
        block = host.query_one(Collapsible)
        assert block.display
        assert block.collapsed is False
        assert (
            str(block.query_one(".console-thinking-text", Static).render())
            == "Working it out"
        )
        await pilot.click("CollapsibleTitle")
        assert block.collapsed
        message.thinking_text += " step by step"
        host.query_one(row_type).sync_message(message)
        await pilot.pause()
        assert block.collapsed
        assert "step by step" in str(
            block.query_one(".console-thinking-text", Static).render()
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("split_fields", [True, False])
async def test_real_llamacpp_gateway_shows_thinking_with_unchanged_answer(split_fields):
    import httpx
    import json
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderResolution,
        ConsoleProviderStreamSignals,
    )

    seen = []
    delta = (
        {"reasoning_content": "reasoning"}
        if split_fields
        else {"content": "<think>reasoning</think>"}
    )
    wire = "".join(
        "data: " + json.dumps({"choices": [{"delta": d}]}) + "\n\n"
        for d in [delta, {"content": "answer"}]
    )

    async def handler(request):
        return httpx.Response(
            200, text=wire, headers={"content-type": "text/event-stream"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = ConsoleProviderGateway(http_client=client)
        resolution = ConsoleProviderResolution(
            provider="llama_cpp",
            model="qwen3.8-27b",
            base_url="http://localhost:9099",
            ready=True,
            execution_key="llama_cpp",
        )
        result = [
            chunk
            async for chunk in gateway.stream_chat(
                resolution,
                [{"role": "user", "content": "hi"}],
                signals=ConsoleProviderStreamSignals(thinking_callback=seen.append),
            )
        ]
    assert "".join(result) == "answer"
    assert "".join(seen) == "reasoning"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["local_vllm", "local_ollama"])
async def test_generic_local_stream_retains_wire_reasoning_for_display(provider):
    from Tests.Chat.test_console_provider_gateway import _bare_resolution
    from dataclasses import replace
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderStreamSignals

    seen = []
    gateway = ConsoleProviderGateway(
        chat_api_call_fn=lambda **kw: iter(
            [
                {"choices": [{"delta": {"reasoning_content": "thinking"}}]},
                {"choices": [{"delta": {"content": "answer"}}]},
            ]
        )
    )
    try:
        resolution = replace(
            _bare_resolution(provider),
            provider=provider,
            model="qwen3.6",
            streaming=True,
        )
        answer = [
            chunk
            async for chunk in gateway.stream_chat(
                resolution,
                [{"role": "user", "content": "hi"}],
                signals=ConsoleProviderStreamSignals(thinking_callback=seen.append),
            )
        ]
        assert answer == ["answer"]
        assert seen == ["thinking"]
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_transcript_reconciles_thinking_only_changes_with_production_css():
    from dataclasses import replace
    from textual.app import App
    from textual.widgets import Static
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleChatMessage,
        ConsoleMessageRole,
    )
    from tldw_chatbook.Widgets.Console.console_transcript import (
        ConsoleTranscript,
        ConsoleThinkingBlock,
    )

    class Host(App):
        CSS_PATH = TldwCli.CSS_PATH

        def compose(self):
            yield ConsoleTranscript()

    host = Host()
    async with host.run_test(size=(120, 40)) as pilot:
        transcript = host.query_one(ConsoleTranscript)
        message = ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT, content="", status="streaming"
        )
        transcript.set_messages([message])
        await transcript.refresh_messages()
        await pilot.pause()
        assert not host.query_one(ConsoleThinkingBlock).display
        message = replace(message)
        message.thinking_text = "first live thought"
        transcript.set_messages([message])
        await transcript.refresh_messages()
        await pilot.pause()
        block = host.query_one(ConsoleThinkingBlock)
        assert block.display and block.region.height > 1
        assert "first live thought" in str(
            block.query_one(".console-thinking-text", Static).render()
        )
        message = replace(message)
        message.thinking_text = "first live thought, then another"
        transcript.set_messages([message])
        await transcript.refresh_messages()
        await pilot.pause()
        assert "then another" in str(
            block.query_one(".console-thinking-text", Static).render()
        )


def test_thinking_does_not_follow_a_different_or_abandoned_variant():
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    store = ConsoleChatStore()
    session = store.create_session()
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="original"
    )
    store.begin_variant_stream(message.id)
    store.append_thinking_chunk(message.id, "new reasoning")
    store.append_stream_chunk(message.id, "new answer")
    store.finalize_variant_stream(message.id)
    assert store.select_variant(message.id, 0).thinking_text == ""
    assert store.select_variant(message.id, 1).thinking_text == "new reasoning"
    store.begin_variant_stream(message.id)
    store.append_thinking_chunk(message.id, "abandoned reasoning")
    assert store.mark_message_stopped(message.id).thinking_text == "new reasoning"


def test_thinking_trace_is_retained_without_display_truncation():
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    store = ConsoleChatStore()
    session = store.create_session()
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    store.append_thinking_chunk(message.id, "x" * 100_000)
    store.append_thinking_chunk(message.id, "tail")
    assert store.get_message(message.id).thinking_text == "x" * 100_000 + "tail"


def test_zai_display_keeps_existing_private_terminal_reasoning():
    import json
    from tldw_chatbook.LLM_Calls.hosted_chat import HostedChatStream
    from tldw_chatbook.LLM_Calls.hosted_chat_streaming import SSERecord
    from tldw_chatbook.LLM_Calls.zai import ZAIStream, ZAIFinishPolicy

    events = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": "private-continuation"},
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {"index": 0, "delta": {"content": "answer"}, "finish_reason": None}
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": 4}},
    ]
    response = ZAIStream(
        HostedChatStream(
            iter(
                [SSERecord(event=None, data=json.dumps(event)) for event in events]
                + [SSERecord(event=None, data="[DONE]")]
            ),
            finish_policy=ZAIFinishPolicy(),
        )
    )
    seen = []
    with thinking_display(seen.append, provider="zai", model="glm"):
        answer = list(ConsoleProviderGateway.normalize_provider_response(response))
    assert "".join(answer) == "answer"
    assert seen == ["private-continuation"]
    assert response.terminal_turn.reasoning_content == "private-continuation"


def test_openai_public_reasoning_summary_reaches_display_without_answer_changes():
    import json
    from unittest.mock import Mock
    from tldw_chatbook.LLM_Calls.LLM_API_Calls import _responses_stream_to_chat_sse

    response = Mock()
    response.iter_lines.return_value = iter(
        [
            "data: "
            + json.dumps(
                {"type": "response.reasoning_summary_text.delta", "delta": "summary"}
            ),
            "data: "
            + json.dumps({"type": "response.output_text.delta", "delta": "answer"}),
            "data: " + json.dumps({"type": "response.completed", "response": {}}),
        ]
    )
    seen = []
    with thinking_display(seen.append, provider="openai", model="test-model"):
        answer = list(
            ConsoleProviderGateway.normalize_provider_response(
                _responses_stream_to_chat_sse(response, model="test-model")
            )
        )
    assert "".join(answer) == "answer"
    assert seen == ["summary"]


def test_anthropic_thinking_delta_is_display_only(monkeypatch):
    from unittest.mock import Mock
    from Tests.Chat.test_anthropic_streaming_usage import ANTHROPIC_STREAM_LINES, _sse
    from tldw_chatbook.Chat.Chat_Functions import chat_api_call

    response = Mock()
    response.status_code = 200
    response.iter_lines.return_value = iter(
        [
            ANTHROPIC_STREAM_LINES[0],
            _sse(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "consider"},
                }
            ),
            *ANTHROPIC_STREAM_LINES[1:],
        ]
    )
    monkeypatch.setattr("requests.Session.post", lambda *a, **k: response)
    seen = []
    with thinking_display(seen.append, provider="anthropic", model="claude-sonnet-4-6"):
        answer = list(
            ConsoleProviderGateway.normalize_provider_response(
                chat_api_call(
                    "anthropic",
                    messages_payload=[{"role": "user", "content": "hi"}],
                    api_key="test-key",
                    model="claude-sonnet-4-6",
                    streaming=True,
                )
            )
        )
    assert "".join(answer) == "Hello"
    assert seen == ["consider"]


@pytest.mark.parametrize("answer", ["ANSWER_SENTINEL_DO_PERSIST", ""])
def test_thinking_survives_real_database_reopen_without_polluting_answer(
    tmp_path, answer
):
    import sqlite3
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db_path = tmp_path / "thinking.db"
    db = CharactersRAGDB(db_path, "thinking-test")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.create_session()
        store.persist_session_if_needed(session.id)
        message = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="", persist=True
        )
        store.append_thinking_chunk(message.id, "THINKING_SENTINEL_RETAINED")
        if answer:
            store.append_stream_chunk(message.id, answer)
        store.mark_message_complete(message.id)
        with sqlite3.connect(db_path) as connection:
            assert (
                connection.execute(
                    "SELECT count(*) FROM messages_fts WHERE messages_fts MATCH ?",
                    ('"THINKING_SENTINEL_RETAINED"',),
                ).fetchone()[0]
                == 0
            )
            content, raw_metadata = connection.execute(
                "SELECT content, metadata_json FROM messages WHERE role = 'assistant'"
            ).fetchone()
        from tldw_chatbook.Chat.message_metadata import MessageMetadata

        assert content == answer
        metadata = MessageMetadata.from_json(raw_metadata)
        assert metadata.thinking_text == "THINKING_SENTINEL_RETAINED"
        assert "THINKING_SENTINEL_RETAINED" not in repr(metadata)
        from tldw_chatbook.Chat.console_conversation_hydration import (
            console_messages_from_conversation_tree,
        )

        restored = console_messages_from_conversation_tree(
            {
                "root_threads": [
                    {
                        "id": "saved",
                        "sender": "Assistant",
                        "content": content,
                        "metadata_json": raw_metadata,
                    }
                ]
            }
        )
        assert restored[0].thinking_text == "THINKING_SENTINEL_RETAINED"
        from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

        fresh_store = ConsoleChatStore()
        fresh_session = fresh_store.restore_persisted_session(
            title="saved",
            workspace_id=None,
            persisted_conversation_id="saved-conv",
            all_nodes=restored,
        )
        assert (
            fresh_store.messages_for_session(fresh_session.id)[0].thinking_text
            == "THINKING_SENTINEL_RETAINED"
        )
        if answer:
            store.add_variant(message.id, "A variant without thinking")
            with sqlite3.connect(db_path) as connection:
                cleared = connection.execute(
                    "SELECT metadata_json FROM messages WHERE role = 'assistant'"
                ).fetchone()[0]
            assert MessageMetadata.from_json(cleared).thinking_text == ""
            assert (
                store.select_variant(message.id, 0).thinking_text
                == "THINKING_SENTINEL_RETAINED"
            )
    finally:
        db.close_connection()


def test_thinking_exports_and_portable_chatbook_round_trip(tmp_path):
    import json
    from tldw_chatbook.Chat.message_metadata import MessageMetadata
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
        export_conversation_to_json,
        export_conversation_to_text,
    )
    from tldw_chatbook.Chat.Chat_Functions import generate_chat_history_content
    from tldw_chatbook.Chat.trajectory_export import (
        build_trajectory_export,
        validate_trajectory_export,
    )
    from Tests.Chatbooks.test_provider_continuation_roundtrip import _create_export
    from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter
    from tldw_chatbook.Chatbooks.conflict_resolver import ConflictResolution

    source_path = tmp_path / "source.db"
    source = CharactersRAGDB(source_path, "thinking-export")
    destination_path = tmp_path / "destination.db"
    destination = CharactersRAGDB(destination_path, "thinking-import")
    try:
        conversation = source.add_conversation({"title": "Thinking round trip"})
        from tldw_chatbook.Chat.local_reasoning import LocalReasoningReplay

        replay = LocalReasoningReplay.capture(
            model="Qwen3.8-27B", content="Answer", reasoning="Exact final call"
        )
        metadata = MessageMetadata(
            thinking_text="Reasoned before answering", local_reasoning_replay=replay
        )
        message = {
            "conversation_id": conversation,
            "sender": "assistant",
            "role": "assistant",
            "content": "Answer",
            "metadata_json": metadata.to_json(),
        }
        source.add_message(message)
        exported = json.loads(export_conversation_to_json(source, conversation))
        assert exported["messages"][0]["thinking_text"] == metadata.thinking_text
        assert exported["messages"][0]["content"] == "Answer"
        assert exported["messages"][0]["local_reasoning_replay"] == replay.to_dict()
        text = export_conversation_to_text(source, conversation)
        assert "Thinking:\nReasoned before answering\n\nAnswer:\nAnswer" in text
        history, _ = generate_chat_history_content([message], None, None)
        assert (
            json.loads(history)["history"][0]["thinking_text"] == metadata.thinking_text
        )
        import io
        from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
            load_chat_history_from_file_and_save_to_db,
        )

        imported_id, _ = load_chat_history_from_file_and_save_to_db(
            destination, io.BytesIO(history.encode())
        )
        assert imported_id is not None
        imported_history = destination.get_messages_for_conversation(imported_id)
        assert (
            MessageMetadata.from_json(
                imported_history[0]["metadata_json"]
            ).thinking_text
            == metadata.thinking_text
        )
        assert (
            MessageMetadata.from_json(
                imported_history[0]["metadata_json"]
            ).local_reasoning_replay
            == replay
        )
        trajectory = build_trajectory_export(source, conversation)
        validate_trajectory_export(trajectory)
        assert trajectory["messages"][0]["thinking_text"] == metadata.thinking_text
        exported_path = _create_export(
            tmp_path, {"ChaChaNotes": str(source_path)}, conversation
        )
        importer = ChatbookImporter({"ChaChaNotes": str(destination_path)})
        importer.temp_dir = tmp_path / "imports"
        importer.temp_dir.mkdir()
        success, detail = importer.import_chatbook(
            exported_path, conflict_resolution=ConflictResolution.RENAME
        )
        assert success, detail
        row = destination.execute_query(
            "SELECT content, metadata_json FROM messages WHERE role = 'assistant'"
        ).fetchall()
        assert len(row) == 2
        assert all(
            MessageMetadata.from_json(item["metadata_json"]).local_reasoning_replay
            == replay
            for item in row
        )
        row = row[-1]
        assert row["content"] == "Answer"
        assert (
            MessageMetadata.from_json(row["metadata_json"]).thinking_text
            == metadata.thinking_text
        )
    finally:
        source.close_connection()
        destination.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["Replacement answer", ""])
async def test_retry_captures_thinking_before_the_first_answer_chunk(answer):
    from Tests.Chat.test_console_chat_controller import (
        StreamingGateway,
        FailingStreamingGateway,
        _last_failed_assistant,
    )
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    class ThinkingGateway(StreamingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            kwargs["signals"].thinking_callback("Replacement thinking")
            if answer:
                yield answer

    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=FailingStreamingGateway()
    )
    await controller.submit_draft("hello")
    failed = _last_failed_assistant(store)
    controller.provider_gateway = ThinkingGateway()
    await controller.retry_message(failed.id)
    result = store.get_message(failed.id)
    assert result.thinking_text == "Replacement thinking"
    assert result.content == answer
    assert result.status == ("complete" if answer else "failed")


def test_tool_round_cleanup_keeps_thinking_for_the_final_answer():
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    store = ConsoleChatStore()
    session = store.create_session()
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    store.append_thinking_chunk(message.id, "Need a tool. ")
    store.append_stream_chunk(message.id, "Provisional tool-call prose")
    store.reset_stream_content(message.id)
    store.append_thinking_chunk(message.id, "Use the result.")
    store.append_stream_chunk(message.id, "Final answer")
    result = store.mark_message_complete(message.id)
    assert result.thinking_text == "Need a tool. Use the result."
    assert result.content == "Final answer"
    assert (
        store.issue_tts_message_speech_snapshot(message.id).raw_content
        == "Final answer"
    )


@pytest.mark.parametrize("value", [None, 4, ["thought"], {"text": "thought"}])
def test_malformed_stored_thinking_is_not_stringified(value):
    import json
    from tldw_chatbook.Chat.message_metadata import MessageMetadata

    restored = MessageMetadata.from_json(
        json.dumps({"provider": "ollama", "thinking_text": value})
    )
    assert restored.provider == "ollama"
    assert restored.thinking_text == ""


@pytest.mark.asyncio
async def test_late_retry_thinking_cannot_clear_the_retained_failed_reply():
    from Tests.Chat.test_console_chat_controller import (
        StreamingGateway,
        FailingStreamingGateway,
        _last_failed_assistant,
    )
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    callbacks = []

    class EmptyGateway(StreamingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            callbacks.append(kwargs["signals"].thinking_callback)
            if False:
                yield ""

    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=FailingStreamingGateway()
    )
    await controller.submit_draft("hello")
    failed = _last_failed_assistant(store)
    controller.provider_gateway = EmptyGateway()
    await controller.retry_message(failed.id)
    callbacks[0]("Too late")
    result = store.get_message(failed.id)
    assert result.status == "failed"
    assert result.content == failed.content
    assert result.thinking_text == failed.thinking_text


def test_nonstream_responses_retains_only_the_returned_summary():
    from tldw_chatbook.LLM_Calls.LLM_API_Calls import (
        _normalize_openai_responses_payload,
    )

    seen = []
    with thinking_display(seen.append, provider="openai", model="test-model"):
        response = _normalize_openai_responses_payload(
            {
                "output_text": "Answer",
                "output": [
                    {
                        "type": "reasoning",
                        "encrypted_content": "PRIVATE",
                        "summary": [{"type": "summary_text", "text": "Public summary"}],
                    }
                ],
            },
            model="test-model",
        )
    assert seen == ["Public summary"]
    assert response["choices"][0]["message"]["content"] == "Answer"


def test_nonstream_anthropic_retains_returned_thinking(monkeypatch):
    from unittest.mock import Mock
    from tldw_chatbook.Chat.Chat_Functions import chat_api_call

    response = Mock()
    response.status_code = 200
    response.json.return_value = {
        "id": "message",
        "content": [
            {"type": "thinking", "thinking": "Consider this", "signature": "PRIVATE"},
            {"type": "text", "text": "Answer"},
        ],
        "stop_reason": "end_turn",
        "usage": {},
    }
    monkeypatch.setattr("requests.Session.post", lambda *a, **k: response)
    seen = []
    with thinking_display(seen.append, provider="anthropic", model="claude-sonnet-4-6"):
        result = chat_api_call(
            "anthropic",
            messages_payload=[{"role": "user", "content": "hi"}],
            api_key="test-key",
            model="claude-sonnet-4-6",
            streaming=False,
        )
    assert seen == ["Consider this"]
    assert result["choices"][0]["message"]["content"] == "Answer"
