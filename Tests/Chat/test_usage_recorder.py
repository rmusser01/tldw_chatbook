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

def test_chat_api_call_passes_provider_dicts_through_unchanged(monkeypatch):
    # The Console gateway parses tool_calls/finish_reason/usage from these
    # dicts -- chat_api_call must NOT normalize them to content strings
    # (task-16331 correction). String consumers use chat_reply_text.
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

    assert result == payload


def test_chat_reply_text_extracts_known_shapes_and_empty_for_unknown():
    from tldw_chatbook.Chat.Chat_Functions import chat_reply_text

    assert chat_reply_text("plain") == "plain"
    assert chat_reply_text(
        {"choices": [{"message": {"role": "assistant", "content": "duck"}}]}
    ) == "duck"
    assert chat_reply_text({"choices": [{"text": "legacy"}]}) == "legacy"
    assert chat_reply_text({"choices": [{"message": {"content": None}}]}) == ""
    assert chat_reply_text({"error": "odd"}) == ""
    assert chat_reply_text(None) == ""


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

    result = None
    with usage_scope() as recorder:
        result = chat_api_call(
            api_endpoint="llama_cpp",
            messages_payload=[{"role": "user", "content": "abcdefg"}],
            api_key=None, temp=0.5, system_message=None, streaming=False,
            minp=None, maxp=None, model=None, topk=None, topp=None,
        )

    assert result == payload  # passthrough preserved
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


# --- cloud provider usage key variants (task-16335) -------------------------------

def test_chat_api_call_records_anthropic_style_usage_keys(monkeypatch):
    # Anthropic normalizes its response to the OpenAI chat shape but keeps
    # its own usage field names (input_tokens/output_tokens).
    payload = {
        "choices": [{"message": {"role": "assistant", "content": "answer"}}],
        "usage": {"input_tokens": 31, "output_tokens": 13},
    }
    monkeypatch.setitem(
        Chat_Functions.API_CALL_HANDLERS, "anthropic", _fake_handler(payload)
    )

    with usage_scope() as recorder:
        chat_api_call(
            api_endpoint="anthropic",
            messages_payload=[{"role": "user", "content": "a long enough prompt"}],
            api_key=None, temp=0.5, system_message=None, streaming=False,
            minp=None, maxp=None, model=None, topk=None, topp=None,
        )

    assert recorder.prompt_tokens() == 31
    assert recorder.completion_tokens() == 13


def test_chat_api_call_openai_style_usage_still_exact(monkeypatch):
    payload = {
        "choices": [{"message": {"role": "assistant", "content": "answer"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 6},
    }
    monkeypatch.setitem(
        Chat_Functions.API_CALL_HANDLERS, "anthropic", _fake_handler(payload)
    )

    with usage_scope() as recorder:
        chat_api_call(
            api_endpoint="anthropic",
            messages_payload=[{"role": "user", "content": "prompt"}],
            api_key=None, temp=0.5, system_message=None, streaming=False,
            minp=None, maxp=None, model=None, topk=None, topp=None,
        )

    assert (recorder.prompt_tokens(), recorder.completion_tokens()) == (5, 6)


def test_chat_api_call_mixed_usage_keys_prefer_openai_names(monkeypatch):
    payload = {
        "choices": [{"message": {"role": "assistant", "content": "answer"}}],
        "usage": {"prompt_tokens": 8, "completion_tokens": 9,
                  "input_tokens": 100, "output_tokens": 100},
    }
    monkeypatch.setitem(
        Chat_Functions.API_CALL_HANDLERS, "anthropic", _fake_handler(payload)
    )

    with usage_scope() as recorder:
        chat_api_call(
            api_endpoint="anthropic",
            messages_payload=[{"role": "user", "content": "prompt"}],
            api_key=None, temp=0.5, system_message=None, streaming=False,
            minp=None, maxp=None, model=None, topk=None, topp=None,
        )

    assert (recorder.prompt_tokens(), recorder.completion_tokens()) == (8, 9)


# --- Qodo remediation (task-16814) ------------------------------------------------

def test_estimate_includes_system_message():
    handler = _fake_handler("answer")
    Chat_Functions.API_CALL_HANDLERS["llama_cpp"] = handler
    with usage_scope() as with_system:
        chat_api_call(
            api_endpoint="llama_cpp",
            messages_payload=[{"role": "user", "content": "tiny"}],
            api_key=None, temp=0.5, system_message="A LONG SYSTEM PROMPT " * 10,
            streaming=False, minp=None, maxp=None, model=None, topk=None, topp=None,
        )
    Chat_Functions.API_CALL_HANDLERS["llama_cpp"] = _fake_handler("answer")
    with usage_scope() as without_system:
        chat_api_call(
            api_endpoint="llama_cpp",
            messages_payload=[{"role": "user", "content": "tiny"}],
            api_key=None, temp=0.5, system_message=None,
            streaming=False, minp=None, maxp=None, model=None, topk=None, topp=None,
        )

    assert with_system.prompt_tokens() > without_system.prompt_tokens()


def test_estimate_ignores_non_text_multimodal_content():
    huge_b64 = "data:image/png;base64," + "A" * 100_000
    handler = _fake_handler("answer")
    Chat_Functions.API_CALL_HANDLERS["llama_cpp"] = handler
    with usage_scope() as recorder:
        chat_api_call(
            api_endpoint="llama_cpp",
            messages_payload=[{"role": "user", "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": huge_b64}},
            ]}],
            api_key=None, temp=0.5, system_message=None,
            streaming=False, minp=None, maxp=None, model=None, topk=None, topp=None,
        )

    # Base64 payloads must not explode the estimate: only the text part counts.
    assert recorder.prompt_tokens() < 100


def test_partial_provider_usage_is_not_marked_exact(monkeypatch):
    # Only the prompt side reported: the exchange must not count as exact.
    payload = {
        "choices": [{"message": {"role": "assistant", "content": "answer"}}],
        "usage": {"prompt_tokens": 11},  # completion_tokens absent
    }
    monkeypatch.setitem(
        Chat_Functions.API_CALL_HANDLERS, "llama_cpp", _fake_handler(payload)
    )

    with usage_scope() as recorder:
        chat_api_call(
            api_endpoint="llama_cpp",
            messages_payload=[{"role": "user", "content": "prompt text"}],
            api_key=None, temp=0.5, system_message=None, streaming=False,
            minp=None, maxp=None, model=None, topk=None, topp=None,
        )

    assert recorder.prompt_tokens() == 11
    assert recorder.exact_tokens() == 0  # partial report: never exact


def test_fully_reported_usage_is_exact():
    recorder = UsageTokenRecorder()
    recorder.record_usage(prompt_tokens=5, completion_tokens=7)
    assert recorder.exact_tokens() == 12
    assert recorder.total_tokens() == 12
