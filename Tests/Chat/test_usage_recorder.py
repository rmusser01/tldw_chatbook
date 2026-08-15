"""LLM usage recorder at the chat_api_call seam (task-16329).

Context-scoped token accounting so the research budget ledger can enforce
``max_tokens``: a recorder is activated around LLM-bearing pipeline calls,
``chat_api_call`` records prompt + completion token estimates into it when
one is active (zero overhead otherwise), and the engine settles the totals
into the ledger. Estimates are labeled as estimates until providers expose
real usage.
"""

import asyncio
from unittest.mock import MagicMock

from tldw_chatbook.Chat import Chat_Functions
from tldw_chatbook.Chat.Chat_Functions import chat_api_call
from tldw_chatbook.Chat.usage_recorder import (
    UsageTokenRecorder,
    active_recorder,
    estimate_tokens,
    usage_scope,
)


def test_estimate_tokens_is_four_chars_per_token_minimum_one():
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 1  # floor
    assert estimate_tokens("a" * 400) == 100


def test_recorder_records_exchange_and_real_usage():
    recorder = UsageTokenRecorder()
    recorder.record_exchange(prompt_text="a" * 40, completion_text="b" * 80)
    assert recorder.total_tokens() == 10 + 20

    recorder.record_usage(prompt_tokens=5, completion_tokens=7)
    assert recorder.total_tokens() == 42


def test_usage_scope_activates_and_isolates_the_recorder():
    assert active_recorder() is None

    with usage_scope() as recorder:
        assert active_recorder() is recorder
        recorder.record_usage(prompt_tokens=3, completion_tokens=4)

    assert active_recorder() is None
    assert recorder.total_tokens() == 7


def test_usage_scope_survives_await_points_within_it():
    async def main():
        with usage_scope() as recorder:
            await asyncio.sleep(0)
            assert active_recorder() is recorder
            recorder.record_usage(prompt_tokens=1, completion_tokens=1)
        return recorder

    recorder = asyncio.run(main())
    assert recorder.total_tokens() == 2


def _fake_handler(response_text="hello world"):
    handler = MagicMock(return_value=response_text)
    handler.__name__ = "fake_handler"  # chat_api_call logs handler.__name__
    return handler


def test_chat_api_call_records_estimates_when_recorder_active(monkeypatch):
    handler = _fake_handler("response text here")
    monkeypatch.setitem(Chat_Functions.API_CALL_HANDLERS, "openai", handler)

    with usage_scope() as recorder:
        chat_api_call(
            api_endpoint="openai",
            messages_payload=[{"role": "user", "content": "prompt text here"}],
            api_key=None,
            temp=0.5,
            system_message=None,
            streaming=False,
            minp=None,
            maxp=None,
            model=None,
            topk=None,
            topp=None,
        )

    assert recorder.prompt_tokens() == estimate_tokens("prompt text here")
    assert recorder.completion_tokens() == estimate_tokens("response text here")
    assert recorder.total_tokens() > 0


def test_chat_api_call_records_nothing_without_active_recorder(monkeypatch):
    handler = _fake_handler()
    monkeypatch.setitem(Chat_Functions.API_CALL_HANDLERS, "openai", handler)

    # No scope: must not raise and must not record anywhere observable.
    result = chat_api_call(
        api_endpoint="openai",
        messages_payload=[{"role": "user", "content": "prompt"}],
        api_key=None,
        temp=0.5,
        system_message=None,
        streaming=False,
        minp=None,
        maxp=None,
        model=None,
        topk=None,
        topp=None,
    )

    assert result == "hello world"
    assert active_recorder() is None


# --- OpenAI-shaped dict normalization (task-16330 live-baseline unblock) --------

def test_chat_api_call_normalizes_openai_dict_to_content_string(monkeypatch):
    payload = {
        "choices": [
            {"index": 0, "finish_reason": "stop",
             "message": {"role": "assistant", "content": "the answer"}}
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }
    monkeypatch.setitem(
        Chat_Functions.API_CALL_HANDLERS, "llama_cpp", _fake_handler(payload)
    )

    result = chat_api_call(
        api_endpoint="llama_cpp",
        messages_payload=[{"role": "user", "content": "prompt"}],
        api_key=None, temp=0.5, system_message=None, streaming=False,
        minp=None, maxp=None, model=None, topk=None, topp=None,
    )

    assert result == "the answer"


def test_chat_api_call_records_real_usage_from_dict_responses(monkeypatch):
    payload = {
        "choices": [
            {"message": {"role": "assistant", "content": "answer text"}}
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }
    handler = _fake_handler(payload)
    monkeypatch.setitem(Chat_Functions.API_CALL_HANDLERS, "llama_cpp", handler)

    with usage_scope() as recorder:
        chat_api_call(
            api_endpoint="llama_cpp",
            messages_payload=[{"role": "user", "content": "a much longer prompt text here"}],
            api_key=None, temp=0.5, system_message=None, streaming=False,
            minp=None, maxp=None, model=None, topk=None, topp=None,
        )

    # EXACT counts from the provider, not character estimates.
    assert recorder.prompt_tokens() == 11
    assert recorder.completion_tokens() == 7


def test_chat_api_call_dict_without_usage_falls_back_to_estimates(monkeypatch):
    payload = {"choices": [{"message": {"role": "assistant", "content": "abcde"}}]}
    monkeypatch.setitem(
        Chat_Functions.API_CALL_HANDLERS, "llama_cpp", _fake_handler(payload)
    )

    with usage_scope() as recorder:
        chat_api_call(
            api_endpoint="llama_cpp",
            messages_payload=[{"role": "user", "content": "abcdefg"}],
            api_key=None, temp=0.5, system_message=None, streaming=False,
            minp=None, maxp=None, model=None, topk=None, topp=None,
        )

    assert recorder.prompt_tokens() == estimate_tokens("abcdefg")
    assert recorder.completion_tokens() == estimate_tokens("abcde")


def test_chat_api_call_unknown_dict_shape_passes_through(monkeypatch):
    payload = {"error": "odd provider payload"}
    monkeypatch.setitem(
        Chat_Functions.API_CALL_HANDLERS, "llama_cpp", _fake_handler(payload)
    )

    result = chat_api_call(
        api_endpoint="llama_cpp",
        messages_payload=[{"role": "user", "content": "prompt"}],
        api_key=None, temp=0.5, system_message=None, streaming=False,
        minp=None, maxp=None, model=None, topk=None, topp=None,
    )

    assert result == {"error": "odd provider payload"}
