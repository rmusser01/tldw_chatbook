"""Tests for the headless console side-chat service.

Covers: streaming delta join + final outcome, provider-error and cancellation
outcomes, prompt rendering rules, reply buffer capping, model-resolution
precedence (qualified / bare / empty / missing session selection), and the
messages/streaming flags handed to the provider gateway.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from tldw_chatbook.Chat.Chat_Deps import ChatProviderError
from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_side_chat import (
    DEFAULT_CONSOLE_SIDECHAT_PROMPT_TEMPLATE,
    SIDE_CHAT_BUFFER_CAP,
    SIDE_CHAT_SYSTEM_PROMPT,
    ConsoleSideChatService,
    SideChatOutcome,
    cap_reply_buffer,
    render_prompt,
)
from tldw_chatbook.config import (
    DEFAULT_CONSOLE_SIDECHAT_PROMPT_TEMPLATE as CONFIG_DEFAULT_TEMPLATE,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeResolution:
    """Stand-in for ConsoleProviderResolution; service reads provider/model."""

    provider: str = "openai"
    model: str = "gpt-test"
    ready: bool = True
    visible_copy: str = ""


class FakeGateway:
    """Records selections/messages; replays chunks, errors, or blocking."""

    def __init__(
        self,
        chunks: list[Any] | None = None,
        error: Exception | None = None,
        resolution: FakeResolution | None = None,
        block_after_first_chunk: bool = False,
        error_after_chunks: bool = False,
    ) -> None:
        self.chunks = chunks if chunks is not None else ["Hello", " world"]
        self.error = error
        self.resolution = resolution or FakeResolution()
        self.block_after_first_chunk = block_after_first_chunk
        # When False (default) the error fires before any chunk; when True
        # all chunks stream first and the error lands mid-stream afterwards.
        self.error_after_chunks = error_after_chunks
        self.first_chunk_sent = asyncio.Event()
        self.selections: list[ConsoleProviderSelection] = []
        self.messages: list[list[dict[str, str]]] = []
        self.stream_calls = 0

    @property
    def call_count(self) -> int:
        return len(self.selections)

    async def resolve_for_send(
        self, selection: ConsoleProviderSelection
    ) -> FakeResolution:
        self.selections.append(selection)
        return self.resolution

    async def stream_chat(
        self,
        resolution: FakeResolution,
        messages: list[dict[str, str]],
        *,
        route=None,
    ):
        self.stream_calls += 1
        self.messages.append(messages)
        if self.error is not None and not self.error_after_chunks:
            raise self.error
        for index, chunk in enumerate(self.chunks):
            yield chunk
            if self.block_after_first_chunk and index == 0:
                self.first_chunk_sent.set()
                await asyncio.Event().wait()  # never set: holds the stream open
        if self.error is not None:
            raise self.error


async def collect(service: ConsoleSideChatService, **kwargs: Any) -> list[Any]:
    items: list[Any] = []
    async for item in service.run(**kwargs):
        items.append(item)
    return items


def default_run_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "selection_quote": "the quoted transcript text",
        "prompt": "Explain this",
        "provider_selection": ConsoleProviderSelection(provider="openai"),
        "sidechat_model": "",
    }
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# Streaming semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_yields_deltas_then_complete_outcome_with_identity() -> None:
    gateway = FakeGateway(
        chunks=["Hel", "lo ", "world"],
        resolution=FakeResolution(provider="zai", model="glm-test"),
    )
    service = ConsoleSideChatService(gateway)

    items = await collect(service, **default_run_kwargs())

    deltas = [item for item in items if isinstance(item, str)]
    assert deltas == ["Hel", "lo ", "world"]
    assert items[-1] == SideChatOutcome(
        text="Hello world", provider="zai", model="glm-test", status="complete"
    )


@pytest.mark.asyncio
async def test_non_str_stream_items_are_ignored() -> None:
    tool_calls = object()  # stand-in for ProviderToolCalls
    gateway = FakeGateway(chunks=["Hel", tool_calls, "lo"])
    service = ConsoleSideChatService(gateway)

    items = await collect(service, **default_run_kwargs())

    deltas = [item for item in items if isinstance(item, str)]
    assert deltas == ["Hel", "lo"]
    assert items[-1].text == "Hello"


@pytest.mark.asyncio
async def test_messages_shape_system_then_user_with_selection() -> None:
    gateway = FakeGateway()
    service = ConsoleSideChatService(gateway)

    await collect(
        service, **default_run_kwargs(prompt="Summarize", selection_quote="QUOTE")
    )

    assert gateway.messages == [
        [
            {"role": "system", "content": SIDE_CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": "Summarize\n\nSelected text:\nQUOTE"},
        ]
    ]


@pytest.mark.asyncio
async def test_provider_error_yields_error_outcome_with_empty_text() -> None:
    gateway = FakeGateway(error=ChatProviderError("safe copy"))
    service = ConsoleSideChatService(gateway)

    items = await collect(service, **default_run_kwargs())

    assert items == [
        SideChatOutcome(
            text="",
            provider="openai",
            model="gpt-test",
            status="provider_error",
            error="safe copy",
        )
    ]


@pytest.mark.asyncio
async def test_provider_error_mid_stream_preserves_partial_text() -> None:
    """A provider error after deltas must not wipe the streamed partial reply."""
    gateway = FakeGateway(
        chunks=["Hel", "lo ", "wor"],
        error=ChatProviderError("mid-stream boom"),
        error_after_chunks=True,
    )
    service = ConsoleSideChatService(gateway)

    items = await collect(service, **default_run_kwargs())

    assert items == [
        "Hel",
        "lo ",
        "wor",
        SideChatOutcome(
            text="Hello wor",
            provider="openai",
            model="gpt-test",
            status="provider_error",
            error="mid-stream boom",
        ),
    ]


@pytest.mark.asyncio
async def test_blocked_resolution_yields_provider_error_without_stream_chat() -> None:
    gateway = FakeGateway(
        resolution=FakeResolution(
            provider="openai",
            model="",
            ready=False,
            visible_copy="API key missing for openai",
        )
    )
    service = ConsoleSideChatService(gateway)

    items = await collect(service, **default_run_kwargs())

    assert gateway.call_count == 1
    assert gateway.stream_calls == 0
    assert items == [
        SideChatOutcome(
            text="",
            provider="openai",
            model="",
            status="provider_error",
            error="API key missing for openai",
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["", "   ", None])
async def test_ready_resolution_with_blank_model_yields_provider_error_with_fallback_copy(
    model: str | None,
) -> None:
    gateway = FakeGateway(
        resolution=FakeResolution(
            provider="mistral", model=model, ready=True, visible_copy=""
        )
    )
    service = ConsoleSideChatService(gateway)

    items = await collect(service, **default_run_kwargs())

    assert gateway.stream_calls == 0
    assert items == [
        SideChatOutcome(
            text="",
            provider="mistral",
            model="",
            status="provider_error",
            error="Choose a ready provider and model, then reopen the side chat.",
        )
    ]


@pytest.mark.asyncio
async def test_cancellation_yields_cancelled_outcome_then_reraises() -> None:
    gateway = FakeGateway(
        chunks=["partial "],
        resolution=FakeResolution(provider="anthropic", model="claude-x"),
        block_after_first_chunk=True,
    )
    service = ConsoleSideChatService(gateway)
    collected: list[Any] = []

    async def consume() -> None:
        async for item in service.run(**default_run_kwargs()):
            collected.append(item)

    task = asyncio.create_task(consume())
    await asyncio.wait_for(gateway.first_chunk_sent.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert collected == [
        "partial ",
        SideChatOutcome(
            text="partial ",
            provider="anthropic",
            model="claude-x",
            status="cancelled",
        ),
    ]


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_qualified_model_overrides_provider_and_model() -> None:
    gateway = FakeGateway(
        resolution=FakeResolution(provider="mistral", model="mistral-large")
    )
    service = ConsoleSideChatService(gateway)

    await collect(
        service,
        **default_run_kwargs(
            provider_selection=ConsoleProviderSelection(
                provider="openai", base_url="https://session"
            ),
            sidechat_model="mistral/mistral-large-latest",
        ),
    )

    assert gateway.call_count == 1
    recorded = gateway.selections[0]
    assert recorded.provider == "mistral"
    assert recorded.explicit_model == "mistral-large-latest"
    assert recorded.streaming is True


@pytest.mark.asyncio
async def test_qualified_model_works_without_session_selection() -> None:
    gateway = FakeGateway()
    service = ConsoleSideChatService(gateway)

    items = await collect(
        service,
        **default_run_kwargs(provider_selection=None, sidechat_model="openai/gpt-5"),
    )

    assert gateway.call_count == 1
    recorded = gateway.selections[0]
    assert recorded.provider == "openai"
    assert recorded.explicit_model == "gpt-5"
    assert recorded.streaming is True
    assert items[-1].status == "complete"


@pytest.mark.asyncio
async def test_bare_model_keeps_session_provider_and_overrides_model() -> None:
    gateway = FakeGateway()
    service = ConsoleSideChatService(gateway)

    await collect(
        service,
        **default_run_kwargs(
            provider_selection=ConsoleProviderSelection(
                provider="openai", base_url="https://session", explicit_model="gpt-3"
            ),
            sidechat_model="gpt-4o",
        ),
    )

    recorded = gateway.selections[0]
    assert recorded.provider == "openai"
    assert recorded.base_url == "https://session"
    assert recorded.explicit_model == "gpt-4o"
    assert recorded.streaming is True


@pytest.mark.asyncio
async def test_empty_model_uses_session_selection_with_streaming_forced_on() -> None:
    gateway = FakeGateway()
    service = ConsoleSideChatService(gateway)

    await collect(
        service,
        **default_run_kwargs(
            provider_selection=ConsoleProviderSelection(
                provider="anthropic", explicit_model="claude-3", streaming=False
            ),
            sidechat_model="",
        ),
    )

    recorded = gateway.selections[0]
    assert recorded.provider == "anthropic"
    assert recorded.explicit_model == "claude-3"
    assert recorded.streaming is True


@pytest.mark.asyncio
@pytest.mark.parametrize("sidechat_model", ["", "gpt-4o", "   "])
async def test_missing_session_selection_without_qualified_model_errors_before_gateway(
    sidechat_model: str,
) -> None:
    gateway = FakeGateway()
    service = ConsoleSideChatService(gateway)

    items = await collect(
        service,
        **default_run_kwargs(provider_selection=None, sidechat_model=sidechat_model),
    )

    assert gateway.call_count == 0
    assert items == [
        SideChatOutcome(
            text="",
            provider="",
            model="",
            status="provider_error",
            error="No provider available for the side chat.",
        )
    ]


# ---------------------------------------------------------------------------
# render_prompt
# ---------------------------------------------------------------------------


def test_render_prompt_substitutes_selection() -> None:
    assert render_prompt("Summarize: {selection}", "abc") == "Summarize: abc"


def test_render_prompt_leaves_other_braces_literal() -> None:
    assert (
        render_prompt("Use {foo} and {selection} here", "x") == "Use {foo} and x here"
    )


def test_render_prompt_blank_template_falls_back_to_default() -> None:
    assert render_prompt("   ", "sel") == CONFIG_DEFAULT_TEMPLATE.format(
        selection="sel"
    )


def test_render_prompt_missing_placeholder_appends_selection_on_new_line() -> None:
    assert render_prompt("Explain this.", "sel") == "Explain this.\nsel"


# ---------------------------------------------------------------------------
# cap_reply_buffer
# ---------------------------------------------------------------------------


def test_cap_reply_buffer_keeps_short_text_untouched() -> None:
    assert cap_reply_buffer("abc") == "abc"


def test_cap_reply_buffer_keeps_tail_of_oversized_reply() -> None:
    text = "a" * (SIDE_CHAT_BUFFER_CAP + 10)
    capped = cap_reply_buffer(text)
    assert capped == "…\n" + text[-SIDE_CHAT_BUFFER_CAP:]
    assert len(capped) == SIDE_CHAT_BUFFER_CAP + 2


# ---------------------------------------------------------------------------
# Constants re-export sanity
# ---------------------------------------------------------------------------


def test_default_template_matches_config_constant() -> None:
    assert DEFAULT_CONSOLE_SIDECHAT_PROMPT_TEMPLATE == CONFIG_DEFAULT_TEMPLATE
