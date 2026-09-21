"""Stopping a stream mid-flight must close the provider response, for all handlers.

TASK-32805.1. Seven of the eight hosted streaming handlers in
`LLM_API_Calls.py` yielded the synthetic `data: [DONE]` sentinel inside a
`finally`. When a consumer stops the stream (`generator.close()` sends
`GeneratorExit` at the suspended `yield`), yielding again from `finally`
raises `RuntimeError: generator ignored GeneratorExit`, and the
`response.close()` after it never runs -- the HTTP socket stays open until
the frame is garbage-collected. Only OpenAI had been fixed and pinned. The
local-provider generator closed first but still yielded inside `finally`,
so a Stop there raised the same RuntimeError.

The consumer path swallows that RuntimeError (`contextlib.suppress` in the
provider gateway), which is exactly why it went unnoticed; this reproduces
it at the handler by asserting `response.close()` actually runs on a Stop.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from tldw_chatbook.Chat.Chat_Functions import chat_api_call


def _streaming_response(lines):
    response = Mock()
    response.status_code = 200
    response.raise_for_status = Mock()
    response.iter_lines.return_value = iter(lines)
    response.close = Mock()
    return response


# A minimal two-chunk stream in each provider's SSE shape. The passthrough
# providers (anthropic/deepseek/groq/mistral/openrouter) yield any non-empty
# line, but the cohere and google TRANSLATORS parse their native schema and
# yield nothing for an OpenAI-style chunk -- so a shared OpenAI fixture made
# next(generator) exhaust (StopIteration) before generator.close() ran, and
# their cancellation cleanup was never exercised (Qodo #1). Give those two a
# fixture in their own schema so the first next() suspends at a real yield.
_OPENAI_STYLE = [
    'data: {"choices": [{"delta": {"content": "first"}}]}',
    'data: {"choices": [{"delta": {"content": "second"}}]}',
]
_CHUNKS_BY_PROVIDER = {
    "cohere": [
        'data: {"type": "content-delta", "delta": {"message": {"content": {"text": "first"}}}}',
        'data: {"type": "content-delta", "delta": {"message": {"content": {"text": "second"}}}}',
    ],
    "google": [
        'data: {"candidates": [{"content": {"parts": [{"text": "first"}], "role": "model"}, "index": 0}]}',
        'data: {"candidates": [{"content": {"parts": [{"text": "second"}], "role": "model"}, "index": 0}]}',
    ],
}


def _chunks_for(provider):
    return list(_CHUNKS_BY_PROVIDER.get(provider, _OPENAI_STYLE))


@pytest.mark.parametrize(
    "provider",
    ["anthropic", "cohere", "deepseek", "google", "groq", "mistral", "openrouter"],
)
def test_stopping_a_hosted_stream_closes_its_response(provider):
    response = _streaming_response(_chunks_for(provider))
    with patch("requests.Session.post", return_value=response), patch(
        "requests.post", return_value=response
    ):
        generator = chat_api_call(
            provider,
            messages_payload=[{"role": "user", "content": "hi"}],
            api_key="sk-test",
            model="test-model",
            streaming=True,
        )
        # Pull one chunk so the generator is suspended at a yield, then stop.
        next(generator)
        generator.close()  # must not raise "generator ignored GeneratorExit"

    response.close.assert_called_once_with()


def test_no_streaming_handler_yields_inside_a_finally():
    """AC#2, gate-free: no `yield` inside any `finally` in the LLM call files.

    The runtime tests above drive `chat_api_call`, which touches config/DB
    that this worktree's storage-admission gate blocks (the shipped OpenAI
    stop test is gated the same way and runs only in CI). This AST pin
    encodes the same invariant and runs anywhere: a `yield` reached while a
    generator is handling `GeneratorExit` raises `generator ignored
    GeneratorExit`, so the sentinel must live after the `finally`, never in
    it.
    """
    import ast
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    offenders: list[str] = []
    for rel in (
        "tldw_chatbook/LLM_Calls/LLM_API_Calls.py",
        "tldw_chatbook/LLM_Calls/LLM_API_Calls_Local.py",
    ):
        tree = ast.parse((repo / rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for stmt in node.finalbody:
                    for sub in ast.walk(stmt):
                        if isinstance(sub, (ast.Yield, ast.YieldFrom)):
                            offenders.append(f"{rel}:{sub.lineno}")
    assert not offenders, f"yield inside a finally: {offenders}"
