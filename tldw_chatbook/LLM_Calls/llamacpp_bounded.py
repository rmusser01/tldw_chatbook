"""Opt-in, captured, keyless llama.cpp completion for session-bound workflows."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from ipaddress import ip_address
from typing import TypedDict
from urllib.parse import SplitResult, urlsplit, urlunsplit

import httpx

from tldw_chatbook.Chat.llamacpp_think_filter import split_start_anchored_thinking
from tldw_chatbook.Chat.local_server_discovery import (
    UnsupportedModelResponseEncoding,
    read_bounded_model_response,
)
from tldw_chatbook.Chat.provider_endpoint_contract import resolve_provider_endpoint
from tldw_chatbook.Chat.sampling_params import validate_sampling_params
from tldw_chatbook.Chat.thinking_blocks import ThinkingEnvelopeValidationError
from tldw_chatbook.Utils.sensitive_llm_logging import sensitive_llm_request

_SAMPLING_KEYS = frozenset({"temperature", "top_p", "top_k", "min_p", "seed"})
_PRIVATE_TRANSPORT: ContextVar[bool] = ContextVar(
    "bounded_llama_transport", default=False
)


class _TransportLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not _PRIVATE_TRANSPORT.get()


_TRANSPORT_LOG_FILTER = _TransportLogFilter()


@contextmanager
def _private_transport_logs() -> Iterator[None]:
    # Logger filters do not propagate to child loggers. These are the emitting
    # loggers for the closed HTTP/1 transport (HTTP/2 and proxies are disabled).
    # Install one idempotent filter; only this request's context suppresses it.
    for name in ("httpx", "httpcore", "httpcore.connection", "httpcore.http11"):
        logging.getLogger(name).addFilter(_TRANSPORT_LOG_FILTER)
    token = _PRIVATE_TRANSPORT.set(True)
    try:
        with sensitive_llm_request():
            yield
    finally:
        _PRIVATE_TRANSPORT.reset(token)


@dataclass(frozen=True, slots=True)
class BoundedLlamaRequest:
    """Detached run settings; selected provider identity is not transport identity."""

    provider_id: str
    selected_url: str = field(repr=False)
    dispatch_url: str = field(repr=False)
    model: str
    prompt: str = field(repr=False)
    max_tokens: int
    request_timeout_seconds: float
    sampling: tuple[tuple[str, int | float], ...] = ()


class LlamaUsage(TypedDict):
    """Validated nonnegative provider token counts."""

    input_tokens: int
    output_tokens: int


class LlamaResult(TypedDict):
    """Visible complete answer and optional trustworthy usage."""

    text: str
    usage: LlamaUsage | None


class BoundedLlamaError(ValueError):
    """Failure identified by a payload-free code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _utf8_bytes(text: str) -> bytes | None:
    try:
        return text.encode("utf-8")
    except UnicodeError:
        return None


def estimate_llama_reservation(prompt: str, *, max_tokens: int) -> int:
    """Reserve conservative input units plus the full output allowance.

    Args:
        prompt: Captured user prompt.
        max_tokens: Explicit output allowance.

    Returns:
        UTF-8 byte length plus 64 envelope units and the output allowance.

    Raises:
        BoundedLlamaError: The prompt or output allowance is invalid.
    """
    if type(prompt) is not str or type(max_tokens) is not int or max_tokens < 1:
        raise BoundedLlamaError("invalid_request")
    encoded = _utf8_bytes(prompt)
    if encoded is None:
        raise BoundedLlamaError("invalid_request")
    return len(encoded) + 64 + max_tokens


def _numeric_loopback(host: str) -> bool:
    if "%" in host:
        return False
    try:
        address = ip_address(host)
        return address.is_loopback if address.version == 4 else str(address) == "::1"
    except ValueError:
        return False


def _endpoint(selected_url: str) -> SplitResult:
    resolution = resolve_provider_endpoint("llama_cpp", selected_url)
    if resolution.errors or resolution.chat_url is None:
        raise BoundedLlamaError("invalid_endpoint")
    parsed = urlsplit(resolution.chat_url)
    if parsed.hostname != "localhost" and not _numeric_loopback(parsed.hostname or ""):
        raise BoundedLlamaError("invalid_endpoint")
    return parsed


def resolve_llama_loopback_url(selected_url: str) -> str:
    """Resolve setup's selected endpoint to one numeric loopback chat URL.

    Call only in the retained setup worker, before displaying and approving the
    pinned destination. Dispatch never performs localhost resolution or fallback.

    Args:
        selected_url: Selected llama.cpp endpoint, including an optional prefix.

    Returns:
        Full numeric-loopback chat URL, retaining scheme, port and prefix.

    Raises:
        BoundedLlamaError: Invalid URL, failed DNS or any non-loopback DNS answer.
    """
    parsed = _endpoint(selected_url)
    if parsed.hostname != "localhost":
        return parsed.geturl()
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        answers = socket.getaddrinfo("localhost", port, type=socket.SOCK_STREAM)
    except OSError:
        answers = []
    addresses = [answer[4][0] for answer in answers]
    if not addresses or not all(_numeric_loopback(address) for address in addresses):
        raise BoundedLlamaError("invalid_endpoint")
    host = str(ip_address(addresses[0]))
    authority = f"[{host}]" if ":" in host else host
    if parsed.port is not None:
        authority += f":{parsed.port}"
    return urlunsplit(parsed._replace(netloc=authority))


def _positive_finite(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and value > 0
    except OverflowError:
        return False


def _validate_request(request: BoundedLlamaRequest, deadline_at: float) -> None:
    if not isinstance(request, BoundedLlamaRequest):
        raise BoundedLlamaError("invalid_request")
    estimate_llama_reservation(request.prompt, max_tokens=request.max_tokens)
    for value in (request.provider_id, request.model):
        if type(value) is not str or not value.strip():
            raise BoundedLlamaError("invalid_request")
        if _utf8_bytes(value) is None:
            raise BoundedLlamaError("invalid_request")
    if not _positive_finite(request.request_timeout_seconds) or not _positive_finite(
        deadline_at
    ):
        raise BoundedLlamaError("invalid_request")
    if type(request.sampling) is not tuple:
        raise BoundedLlamaError("invalid_request")
    sampling = {}
    for pair in request.sampling:
        if (
            type(pair) is not tuple
            or len(pair) != 2
            or type(pair[0]) is not str
            or pair[0] not in _SAMPLING_KEYS
            or pair[0] in sampling
            or type(pair[1]) not in (int, float)
        ):
            raise BoundedLlamaError("invalid_request")
        sampling[pair[0]] = pair[1]
    try:
        invalid_sampling = bool(validate_sampling_params(sampling))
    except OverflowError:
        invalid_sampling = True
    if invalid_sampling:
        raise BoundedLlamaError("invalid_request")
    selected = _endpoint(request.selected_url)
    dispatch = _endpoint(request.dispatch_url)
    if (
        not _numeric_loopback(dispatch.hostname or "")
        or dispatch.geturl() != request.dispatch_url
        or (selected.scheme, selected.port, selected.path)
        != (dispatch.scheme, dispatch.port, dispatch.path)
        or (selected.hostname != "localhost" and selected.hostname != dispatch.hostname)
    ):
        raise BoundedLlamaError("invalid_endpoint")
    if deadline_at <= asyncio.get_running_loop().time():
        raise BoundedLlamaError("deadline")


async def _close_request(
    response: httpx.Response | None, client: httpx.AsyncClient
) -> None:
    try:
        if response is not None:
            await response.aclose()
    finally:
        await client.aclose()


async def _settle_cleanup(
    response: httpx.Response | None, client: httpx.AsyncClient
) -> None:
    # A strong reference and repeated shielding keep cancellation of the waiter
    # from freeing its caller's run slot while physical cleanup still runs.
    cleanup = asyncio.gather(_close_request(response, client), return_exceptions=True)
    cancelled = False
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            cancelled = True
    if isinstance(cleanup.result()[0], BaseException):
        raise BoundedLlamaError("cleanup")
    if cancelled:
        raise asyncio.CancelledError


def _parse_answer(body: bytes) -> LlamaResult:
    try:
        document = json.loads(body.decode("utf-8"))
    except (ValueError, RecursionError):
        document = None
    if not isinstance(document, dict):
        raise BoundedLlamaError("malformed_response")
    choices = document.get("choices")
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
    ):
        raise BoundedLlamaError("malformed_response")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise BoundedLlamaError("malformed_response")
    if choice.get("finish_reason") != "stop":
        raise BoundedLlamaError("incomplete_response")
    if any(message.get(key) is not None for key in ("tool_calls", "function_call")):
        raise BoundedLlamaError("unsupported_response")
    content = message["content"]
    if _utf8_bytes(content) is None:
        raise BoundedLlamaError("malformed_response")
    try:
        visible = split_start_anchored_thinking(content)
    except ThinkingEnvelopeValidationError:
        visible = None
    if visible is None:
        raise BoundedLlamaError("malformed_response")
    if visible.status != "complete" or not visible.content.strip():
        raise BoundedLlamaError("incomplete_response")
    usage = document.get("usage")
    validated_usage: LlamaUsage | None = None
    if isinstance(usage, dict) and all(
        type(usage.get(key)) is int and usage[key] >= 0
        for key in ("prompt_tokens", "completion_tokens")
    ):
        validated_usage = {
            "input_tokens": usage["prompt_tokens"],
            "output_tokens": usage["completion_tokens"],
        }
    return {"text": visible.content, "usage": validated_usage}


async def complete_llama_bounded(
    request: BoundedLlamaRequest, *, deadline_at: float
) -> LlamaResult:
    """Send the captured request once and return a bounded chat answer.

    Args:
        request: Immutable settings captured before effect approval.
        deadline_at: Absolute attempt deadline on the running loop's clock.

    Returns:
        Visible answer and usage, when supplied by the peer.

    Raises:
        BoundedLlamaError: Validation, transport, bounds or answer failure.
        asyncio.CancelledError: Cancellation, after owned cleanup settles.

    Missing usage does not refund admission: the caller retains the reservation
    after dispatch, including failure. Closing the connection does not prove that
    the peer stopped generating. Cleanup deliberately outlives request deadlines.
    """
    _validate_request(request, deadline_at)
    payload = {
        "model": request.model,
        "messages": [{"role": "user", "content": request.prompt}],
        "max_tokens": request.max_tokens,
        "stream": False,
        **dict(request.sampling),
    }
    deadline = min(
        deadline_at,
        asyncio.get_running_loop().time() + request.request_timeout_seconds,
    )
    # JSON encoding is still validation, before opening any transport resources.
    try:
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(
            "utf-8"
        )
    except (ValueError, UnicodeError):
        encoded = None
    if encoded is None:
        raise BoundedLlamaError("invalid_request")
    with _private_transport_logs():
        transport = httpx.AsyncHTTPTransport(
            retries=0,
            trust_env=False,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
        )
        client = httpx.AsyncClient(
            transport=transport, trust_env=False, follow_redirects=False
        )
        response = None
        error_code = None
        try:
            async with asyncio.timeout_at(deadline):
                outgoing = client.build_request(
                    "POST",
                    request.dispatch_url,
                    content=encoded,
                    headers={
                        "Accept-Encoding": "identity",
                        "Content-Type": "application/json",
                        # Keep the connection owned by the pool until its close
                        # settles, including an interrupted EOF auto-close.
                        "Connection": "close",
                    },
                    timeout=None,
                )
                response = await client.send(outgoing, stream=True)
                if not 200 <= response.status_code < 300:
                    raise BoundedLlamaError("http_status")
                body = await read_bounded_model_response(response)
                if body is None:
                    raise BoundedLlamaError("response_too_large")
        except TimeoutError:
            error_code = "deadline"
        except UnsupportedModelResponseEncoding:
            error_code = "response_encoding"
        except (httpx.HTTPError, OSError):
            error_code = "transport"
        finally:
            await _settle_cleanup(response, client)
        if error_code:
            raise BoundedLlamaError(error_code)
        return _parse_answer(body)
