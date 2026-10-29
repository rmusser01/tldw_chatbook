"""A streamed Console request the server may have refused for streaming.

TASK-33620.10 made every Console provider stream by default. A server that
cannot stream answers ``"stream": true`` with a client error before any text
arrives, or opens the stream and sends no text. Two consequences follow, and
they share the one classification below:

* TASK-37421 (owner decision 2026-10-29): a self-hosted or Custom
  OpenAI-compatible server is asked once more without streaming, the way the
  llama.cpp path always has. A retry there costs nothing.
* TASK-37180: a provider that is not retried (a cloud provider, where a
  silent second paid call is not wanted) names the Streaming setting in its
  failure copy, beside the model advice.

Classification only: nothing here reaches the wire or the provider's text.
Imported lazily by the gateway (ADR-097 boot ratchet).
"""

from __future__ import annotations

from typing import Any

from tldw_chatbook.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatModelUnavailableError,
    ChatRateLimitError,
)

#: Statuses a server uses to refuse a request shape it does not support:
#: Bad Request, Not Acceptable (``text/event-stream``), Unsupported Media
#: Type, Unprocessable Entity (a validating server's ``stream`` field) and Not
#: Implemented. Auth (401/403), payment (402), not-found (404), timeout
#: (408), size (413) and rate-limit (429) failures are left out: the same
#: request without streaming fails the same way, and their own copy already
#: names the fix.
REFUSED_STREAM_STATUSES = frozenset({400, 406, 415, 422, 501})

#: Recovery copy for a refused stream that was not retried (TASK-37180).
REFUSED_STREAM_ADVICE = (
    "If this server cannot stream replies, turn Streaming off and send again: "
    "in Chat settings for this chat or this model, or in config.toml for this "
    "provider."
)


def is_refused_stream(exc: BaseException) -> bool:
    """Whether a failed streamed request may be a refusal to stream.

    The caller decides the rest: the request was streamed and no text, thinking
    or tool call arrived before the failure.

    Args:
        exc: The adapter's failure, before any projection (its cause chain
            still holds the HTTP response a context-length check reads).

    Returns:
        True for a refusal status from :data:`REFUSED_STREAM_STATUSES` that is
        not a model-unavailable or context-length failure.
    """
    if isinstance(
        exc, (ChatAuthenticationError, ChatRateLimitError, ChatModelUnavailableError)
    ):
        return False
    status = getattr(exc, "status_code", None)
    if type(status) is not int or status not in REFUSED_STREAM_STATUSES:
        return False
    from tldw_chatbook.Chat.provider_error_reason import is_context_overflow_error

    return not is_context_overflow_error(exc)


def is_unreadable_stream(exc: BaseException) -> bool:
    """Whether a streamed reply failed because its stream could not be read.

    A Custom endpoint (``custom-ep:<slug>``) streams through the hosted
    engine, which raises a protocol error, not a status, for a stream that
    ends with no terminal record or is not SSE at all: what a server that
    sends no text, or ignores ``stream`` and answers whole, looks like there.
    The caller decides the rest: the request was streamed, to a server that
    is retried, and nothing arrived before the failure.

    Args:
        exc: The adapter's failure, before any projection.

    Returns:
        True for the hosted engine's stream protocol error.
    """
    from tldw_chatbook.LLM_Calls.hosted_chat import HostedChatProtocolError

    return isinstance(exc, HostedChatProtocolError)


def retries_refused_stream(resolution: Any) -> bool:
    """Whether a refused or empty stream is retried once without streaming.

    Every self-hosted provider and every Custom OpenAI-compatible endpoint
    (``custom``, ``custom_2``, a ``custom-ep:<slug>`` entry and the engine key
    it executes through). Cloud providers never are.

    Args:
        resolution: The Console provider resolution about to stream.

    Returns:
        True when the resolution streams to a self-hosted endpoint.
    """
    if getattr(resolution, "streaming", False) is not True:
        return False
    from tldw_chatbook.Chat.provider_readiness import is_self_hosted_provider

    return any(
        is_self_hosted_provider(key)
        for key in (
            getattr(resolution, "provider", ""),
            getattr(resolution, "readiness_key", ""),
            getattr(resolution, "execution_key", ""),
            getattr(resolution, "selected_provider", ""),
        )
        if key
    )
