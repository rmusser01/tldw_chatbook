"""Provider-neutral hosted Chat-Completions transport and normalization."""

from __future__ import annotations

from tldw_chatbook.LLM_Calls import recovery_review as _provider_recovery
from tldw_chatbook.Utils.input_validation import (
    StrictJSONError,
    strict_json_loads,
)

import re
import json
import math
import time
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, Never, Protocol, cast
from urllib.parse import unquote, urlsplit

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException, Timeout as RequestsTimeout
from urllib3.util import Retry

from tldw_chatbook.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
    ChatRateLimitError,
)
from tldw_chatbook.LLM_Calls.hosted_chat_streaming import OwnedSSEStream, SSERecord
from tldw_chatbook.Utils.egress import create_default_session
from tldw_chatbook.Utils.sensitive_llm_logging import llm_retry_count


_MAX_URL_LENGTH = 2_000
_MAX_PATH_DECODE_PASSES = 3
_PERCENT_ESCAPE_RE = re.compile(r"%[0-9A-Fa-f]{2}")
_ENCODED_PATH_SEPARATOR_RE = re.compile(r"%(?:2[fF]|5[cC])")
_MAX_JSON_DEPTH = 128
_MAX_JSON_NODES = 1_000_000
_MAX_OUTPUT_CHARS = 32 * 1024 * 1024
_MAX_METADATA_CHARS = 4 * 1024
_MAX_TOOL_CALLS = 128
_JSON_DECODE_FAILED = object()
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
# Auth schemes the shared transport accepts (ADR-179 engine). ``"bearer"``
# hard-requires a non-empty key; ``"bearer_optional"`` (Phase 2) admits an
# empty key and then sends no Authorization header at all. ``"api_key_header"``
# is the Phase 3 engine scheme and is deliberately absent until it has a
# header contract, so an unknown scheme fails closed as invalid transport
# configuration.
_SUPPORTED_AUTH_SCHEMES = frozenset({"bearer", "bearer_optional"})
# Upper bound on a provider-named Retry-After sleep (seconds). The engine's
# workers run on threads Stop cannot interrupt, and every hosted provider's
# api_base_url is user-configurable -- a hostile endpoint naming
# "Retry-After: 99999999" must never pin the worker for that long.
_MAX_RETRY_AFTER_SECONDS = 60.0

ReasoningDisposition = Literal["displayable", "proprietary", "ignored"]


class HostedChatBaseURLValidationError(ValueError):
    """Raised when a hosted Chat API base URL is malformed or ambiguous."""


class HostedChatProtocolError(ValueError):
    """Raised when a hosted provider returns malformed Chat data."""


def _same_json_value(left: object, right: object) -> bool:
    return json.dumps(
        left,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ) == json.dumps(
        right,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class HostedChatFinishPolicy(Protocol):
    """Provider-owned finish and reasoning validation policy."""

    reasoning_disposition: ReasoningDisposition

    def validate_finish(
        self,
        *,
        finish_reason: object,
        has_text: bool,
        has_calls: bool,
    ) -> str: ...

    def validate_reasoning_content(self, value: object) -> str | None: ...


@dataclass(frozen=True)
class HostedHTTPTransportConfig:
    """Resolved immutable values for one hosted HTTP transport."""

    provider: str
    base_url: str
    api_key: str = field(repr=False, compare=False)
    timeout: float
    retries: int
    retry_delay: float
    # Provider-requested additional request headers (TASK-32851). The
    # engine's Authorization/Content-Type pair is not overridable; any other
    # header a provider's wire contract requires rides through here.
    extra_headers: Mapping[str, str] = field(default_factory=dict, compare=False)
    # Auth contract (engine, ADR-179 Phase 2): ``"bearer"`` requires a
    # non-empty key; ``"bearer_optional"`` admits an empty key (keyless
    # endpoint, ADR-146) and then sends no Authorization header at all.
    # ``"api_key_header"`` is the Phase 3 engine scheme and is rejected by
    # validation until it has a header contract. Legacy adapters construct
    # without the field and keep the byte-identical ``"bearer"`` default.
    auth_scheme: str = "bearer"


@dataclass(frozen=True)
class HostedChatTurn:
    """One normalized hosted Chat assistant turn."""

    text: str
    tool_calls: tuple[dict[str, Any], ...]
    assistant_message: dict[str, Any] | None = field(repr=False)
    finish_reason: str
    reasoning_content: str | None = field(default=None, repr=False)
    usage: dict[str, Any] | None = field(default=None, repr=False)


@dataclass
class _StreamToolState:
    call_id: str
    name: str
    arguments: list[str]


class HostedChatStream(Iterator[dict[str, Any]]):
    """Normalize one stream of OpenAI-shaped hosted Chat SSE records."""

    def __init__(
        self,
        records: Iterator[SSERecord],
        *,
        finish_policy: HostedChatFinishPolicy,
        allowed_extra_keys: frozenset[str] = frozenset(),
    ) -> None:
        self._records = records
        self._finish_policy = finish_policy
        self._allowed_extra_keys = allowed_extra_keys
        self._text_segments: list[str] = []
        self._reasoning_segments: list[str] = []
        self._tools: dict[int, _StreamToolState] = {}
        self._call_ids: set[str] = set()
        self._finish_reason: str | None = None
        self._usage: dict[str, Any] | None = None
        self._usage_from_choice = False
        self._trailing_usage_seen = False
        self._terminal_turn: HostedChatTurn | None = None
        self._output_chars = 0
        self._closed = False

    def __iter__(self) -> HostedChatStream:
        return self

    def __next__(self) -> dict[str, Any]:
        if self._closed:
            raise StopIteration
        try:
            record = next(self._records)
        except StopIteration:
            self.close()
            raise HostedChatProtocolError(
                "Hosted Chat stream ended before a clean terminal record."
            ) from None
        except Exception:
            self.close()
            raise HostedChatProtocolError("Hosted Chat stream read failed.") from None
        if record.data == "[DONE]":
            if self._finish_reason is None or self._usage is None:
                self.close()
                raise HostedChatProtocolError(
                    "Hosted Chat stream terminated before required metadata."
                )
            self._terminal_turn = _build_turn(
                text="".join(self._text_segments),
                reasoning_content="".join(self._reasoning_segments)
                if self._reasoning_segments
                else None,
                tool_calls=_finish_stream_tools(self._tools),
                finish_reason=self._finish_reason,
                usage=self._usage,
            )
            self.close()
            raise StopIteration

        event = _strict_json_loads(record.data)
        if event is _JSON_DECODE_FAILED or not isinstance(event, Mapping):
            self.close()
            raise HostedChatProtocolError("Hosted Chat stream JSON is malformed.")
        try:
            safe_event = self._consume_event(cast(Mapping[str, Any], event))
        except (HostedChatProtocolError, ChatProviderError):
            self.close()
            raise
        except Exception:
            self.close()
            raise HostedChatProtocolError(
                "Hosted Chat stream data is malformed."
            ) from None
        return safe_event

    @property
    def terminal_turn(self) -> HostedChatTurn:
        """Return terminal metadata after clean stream exhaustion."""
        if self._terminal_turn is None:
            raise HostedChatProtocolError("Hosted Chat stream metadata is incomplete.")
        return self._terminal_turn

    def close(self) -> None:
        """Close this stream."""
        if self._closed:
            return
        self._closed = True
        close = getattr(self._records, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def _consume_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        if set(event) - {
            "id",
            "object",
            "created",
            "model",
            "system_fingerprint",
            "choices",
            "usage",
        } - self._allowed_extra_keys:
            raise HostedChatProtocolError("Hosted Chat stream event is malformed.")
        fingerprint = event.get("system_fingerprint")
        if fingerprint is not None:
            _required_metadata(fingerprint, "system fingerprint")
        choices = event.get("choices")
        usage = event.get("usage")
        usage_from_choice = False
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
            raise HostedChatProtocolError("Hosted Chat stream choices are malformed.")
        if usage is not None and not isinstance(usage, Mapping):
            raise HostedChatProtocolError("Hosted Chat stream usage is malformed.")
        if not choices:
            if self._finish_reason is None or usage is None:
                raise HostedChatProtocolError("Hosted Chat stream usage is misplaced.")
            normalized_usage = deepcopy(dict(usage))
            if self._usage is None:
                self._usage = normalized_usage
            elif not (
                self._usage_from_choice
                and not self._trailing_usage_seen
                and _same_json_value(self._usage, normalized_usage)
            ):
                raise HostedChatProtocolError("Hosted Chat stream usage is malformed.")
            else:
                self._trailing_usage_seen = True
            return deepcopy(dict(event))
        if self._finish_reason is not None:
            raise HostedChatProtocolError(
                "Hosted Chat stream data followed terminal state."
            )
        if len(choices) != 1:
            raise HostedChatProtocolError(
                "Hosted Chat stream choice count is unsupported."
            )
        choice = choices[0]
        if not isinstance(choice, Mapping) or set(choice) - {
            "index",
            "delta",
            "finish_reason",
            "usage",
        }:
            raise HostedChatProtocolError("Hosted Chat stream choice is malformed.")
        if "usage" in choice:
            choice_usage = choice.get("usage")
            if "usage" in event or not isinstance(choice_usage, Mapping):
                raise HostedChatProtocolError("Hosted Chat stream usage is malformed.")
            usage = choice_usage
            usage_from_choice = True
        if type(choice.get("index")) is not int or choice.get("index") != 0:
            raise HostedChatProtocolError(
                "Hosted Chat stream choice index is malformed."
            )
        delta = choice.get("delta")
        if not isinstance(delta, Mapping) or set(delta) - {
            "role",
            "content",
            "reasoning_content",
            "tool_calls",
        }:
            raise HostedChatProtocolError("Hosted Chat stream delta is malformed.")
        if "role" in delta and delta.get("role") != "assistant":
            raise HostedChatProtocolError("Hosted Chat stream role is malformed.")
        content = delta.get("content")
        if content is not None and not isinstance(content, str):
            raise HostedChatProtocolError("Hosted Chat stream content is malformed.")
        if isinstance(content, str):
            self._reserve_output(len(content))
            self._text_segments.append(content)
        if "reasoning_content" in delta:
            reasoning = _validate_reasoning(
                self._finish_policy, delta.get("reasoning_content")
            )
            if reasoning is not None:
                self._reserve_output(len(reasoning))
                self._reasoning_segments.append(reasoning)
        if delta.get("tool_calls") is not None:
            self._consume_tool_deltas(delta["tool_calls"])

        finish_reason = choice.get("finish_reason")
        if finish_reason is not None:
            calls = _finish_stream_tools(self._tools)
            self._finish_reason = _validate_finish(
                self._finish_policy,
                finish_reason=finish_reason,
                has_text=bool("".join(self._text_segments)),
                has_calls=bool(calls),
            )
            if usage is not None:
                self._usage = deepcopy(dict(usage))
                self._usage_from_choice = usage_from_choice
        elif usage is not None:
            raise HostedChatProtocolError(
                "Hosted Chat stream usage preceded terminal state."
            )
        return deepcopy(dict(event))

    def _consume_tool_deltas(self, value: object) -> None:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise HostedChatProtocolError("Hosted Chat stream tools are malformed.")
        event_indexes: set[int] = set()
        for raw_tool in value:
            if not isinstance(raw_tool, Mapping) or set(raw_tool) - {
                "index",
                "id",
                "type",
                "function",
            }:
                raise HostedChatProtocolError("Hosted Chat stream tool is malformed.")
            index = raw_tool.get("index")
            if type(index) is not int or index < 0 or index >= _MAX_TOOL_CALLS:
                raise HostedChatProtocolError(
                    "Hosted Chat stream tool index is malformed."
                )
            if index in event_indexes:
                raise HostedChatProtocolError(
                    "Hosted Chat stream tool index was duplicated."
                )
            event_indexes.add(index)
            function = raw_tool.get("function")
            if not isinstance(function, Mapping) or set(function) - {
                "name",
                "arguments",
            }:
                raise HostedChatProtocolError(
                    "Hosted Chat stream function is malformed."
                )
            arguments = function.get("arguments", "")
            if not isinstance(arguments, str):
                raise HostedChatProtocolError(
                    "Hosted Chat stream arguments are malformed."
                )
            state = self._tools.get(index)
            if state is None:
                call_id = _required_metadata(raw_tool.get("id"), "tool ID")
                name = _required_metadata(function.get("name"), "tool name")
                if raw_tool.get("type") != "function" or call_id in self._call_ids:
                    raise HostedChatProtocolError(
                        "Hosted Chat stream tool identity is malformed."
                    )
                self._call_ids.add(call_id)
                state = _StreamToolState(call_id=call_id, name=name, arguments=[])
                self._tools[index] = state
                self._reserve_output(len(call_id) + len(name))
            else:
                if "id" in raw_tool and raw_tool.get("id") != state.call_id:
                    raise HostedChatProtocolError(
                        "Hosted Chat stream tool identity changed."
                    )
                if "type" in raw_tool and raw_tool.get("type") != "function":
                    raise HostedChatProtocolError(
                        "Hosted Chat stream tool type changed."
                    )
                if "name" in function and function.get("name") != state.name:
                    raise HostedChatProtocolError(
                        "Hosted Chat stream tool name changed."
                    )
            self._reserve_output(len(arguments))
            state.arguments.append(arguments)

    def _reserve_output(self, characters: int) -> None:
        if characters < 0 or self._output_chars + characters > _MAX_OUTPUT_CHARS:
            raise HostedChatProtocolError(
                "Hosted Chat stream output limit was exceeded."
            )
        self._output_chars += characters


def normalize_hosted_chat_response(
    response: object,
    *,
    finish_policy: HostedChatFinishPolicy,
    allowed_extra_keys: frozenset[str] = frozenset(),
) -> HostedChatTurn:
    """Normalize one non-streaming OpenAI-shaped Chat response."""
    if not _json_shape_is_safe(response) or not isinstance(response, Mapping):
        raise HostedChatProtocolError("Hosted Chat response JSON is malformed.")
    if set(response) - {
        "id",
        "object",
        "created",
        "model",
        "system_fingerprint",
        "choices",
        "usage",
    } - allowed_extra_keys:
        raise HostedChatProtocolError("Hosted Chat response is malformed.")
    choices = response.get("choices")
    if (
        not isinstance(choices, Sequence)
        or isinstance(choices, (str, bytes))
        or len(choices) != 1
    ):
        raise HostedChatProtocolError("Hosted Chat response choices are malformed.")
    choice = choices[0]
    if not isinstance(choice, Mapping) or set(choice) - {
        "index",
        "message",
        "finish_reason",
    }:
        raise HostedChatProtocolError("Hosted Chat response choice is malformed.")
    if type(choice.get("index")) is not int or choice.get("index") != 0:
        raise HostedChatProtocolError("Hosted Chat response choice index is malformed.")
    message = choice.get("message")
    if not isinstance(message, Mapping) or set(message) - {
        "role",
        "content",
        "reasoning_content",
        "tool_calls",
    }:
        raise HostedChatProtocolError("Hosted Chat response message is malformed.")
    if message.get("role") != "assistant":
        raise HostedChatProtocolError("Hosted Chat response role is malformed.")
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise HostedChatProtocolError("Hosted Chat response content is malformed.")
    text = content or ""
    reasoning = _validate_reasoning(finish_policy, message.get("reasoning_content"))
    tool_calls = _normalize_tool_calls(message.get("tool_calls", ()))
    if (
        len(text) + len(reasoning or "") + _tool_character_count(tool_calls)
        > _MAX_OUTPUT_CHARS
    ):
        raise HostedChatProtocolError("Hosted Chat response output limit was exceeded.")
    finish_reason = _validate_finish(
        finish_policy,
        finish_reason=choice.get("finish_reason"),
        has_text=bool(text),
        has_calls=bool(tool_calls),
    )
    usage = response.get("usage")
    if usage is not None and not isinstance(usage, Mapping):
        raise HostedChatProtocolError("Hosted Chat response usage is malformed.")
    return _build_turn(
        text=text,
        reasoning_content=reasoning,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=deepcopy(dict(usage)) if isinstance(usage, Mapping) else None,
    )


def hosted_chat_request(
    *,
    config: HostedHTTPTransportConfig,
    payload: Mapping[str, Any],
    streaming: bool,
    finish_policy: HostedChatFinishPolicy,
    allowed_extra_keys: frozenset[str] = frozenset(),
) -> HostedChatTurn | HostedChatStream:
    """Run one hosted Chat-Completions request through the shared boundary."""
    response = owned_json_post(
        config=config,
        route="chat/completions",
        payload=payload,
        streaming=streaming,
    )
    if streaming:
        return HostedChatStream(
            cast(Iterator[SSERecord], response),
            finish_policy=finish_policy,
            allowed_extra_keys=allowed_extra_keys,
        )
    return normalize_hosted_chat_response(
        response,
        finish_policy=finish_policy,
        allowed_extra_keys=allowed_extra_keys,
    )


@_provider_recovery.unqualified
def owned_json_post(
    *,
    config: HostedHTTPTransportConfig,
    route: Literal["chat/completions", "responses"],
    payload: Mapping[str, Any],
    streaming: bool,
) -> dict[str, Any] | OwnedSSEStream:
    """POST JSON with bounded retries and explicit response/session ownership.

    Args:
        config: Fully resolved provider transport values.
        route: Exact hosted generation route.
        payload: Already-built provider request body.
        streaming: Transfer resource ownership to an SSE stream when true.

    Returns:
        A decoded object response or an owned SSE record stream.

    Raises:
        ChatAuthenticationError: For authentication failures.
        ChatRateLimitError: For exhausted rate limiting.
        ChatBadRequestError: For other client request failures.
        ChatProviderError: For network, service, or malformed response failures.
    """
    if route not in {"chat/completions", "responses"}:
        raise _transport_error(config.provider, "request route is invalid")
    try:
        base_url = normalize_hosted_chat_base_url(
            config.base_url,
            default=config.base_url,
        )
    except HostedChatBaseURLValidationError:
        raise _transport_error(
            config.provider,
            "transport configuration is invalid",
        ) from None
    if (
        not isinstance(config.provider, str)
        or not config.provider
        or not isinstance(config.api_key, str)
        or config.auth_scheme not in _SUPPORTED_AUTH_SCHEMES
        or (config.auth_scheme == "bearer" and not config.api_key)
        or isinstance(config.timeout, bool)
        or not isinstance(config.timeout, (int, float))
        or not math.isfinite(float(config.timeout))
        or config.timeout <= 0
        or isinstance(config.retries, bool)
        or not isinstance(config.retries, int)
        or isinstance(config.retry_delay, bool)
        or not isinstance(config.retry_delay, (int, float))
        or not math.isfinite(float(config.retry_delay))
        or config.retry_delay < 0
        or not isinstance(payload, Mapping)
        or not isinstance(config.extra_headers, Mapping)
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(value, str)
            or not value
            or name.lower() in {"authorization", "content-type"}
            for name, value in config.extra_headers.items()
        )
    ):
        raise _transport_error(config.provider, "transport configuration is invalid")

    retries = llm_retry_count(max(0, config.retries))
    url = f"{base_url}/{route}"
    if config.api_key:
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            **config.extra_headers,
        }
    else:
        # bearer_optional with no resolved key: no Authorization header at
        # all -- an empty "Bearer " challenge must never hit the wire under
        # any scheme.
        headers = {
            "Content-Type": "application/json",
            **config.extra_headers,
        }
    session = create_default_session()
    response: requests.Response | None = None
    stream_owns_session = False
    try:
        retry_policy = Retry(
            total=0,
            connect=0,
            read=0,
            redirect=0,
            status=0,
            other=0,
            allowed_methods=frozenset({"POST"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_policy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        for attempt in range(retries + 1):
            response = None
            try:
                response = session.post(
                    url,
                    headers=headers,
                    json=deepcopy(dict(payload)),
                    timeout=float(config.timeout),
                    stream=True,
                )
                status = int(response.status_code)
                if status in _RETRYABLE_STATUS_CODES and attempt < retries:
                    delay = _retry_delay(
                        response,
                        attempt=attempt,
                        retry_delay=float(config.retry_delay),
                    )
                    _best_effort_close(response)
                    response = None
                    if delay > 0:
                        time.sleep(delay)
                    continue
                if status >= 400:
                    _raise_http_error(config.provider, status)
                if streaming:
                    stream = OwnedSSEStream(response=response, session=session)
                    response = None
                    stream_owns_session = True
                    return stream
                try:
                    result = response.json()
                except Exception:
                    raise _transport_error(
                        config.provider, "returned malformed provider JSON"
                    ) from None
                if not isinstance(result, Mapping):
                    raise _transport_error(
                        config.provider, "response envelope must be an object"
                    )
                return deepcopy(dict(result))
            except (ChatAuthenticationError, ChatRateLimitError, ChatBadRequestError):
                raise
            except ChatProviderError:
                raise
            except (requests.ConnectionError, RequestsTimeout) as exc:
                if attempt < retries:
                    if response is not None:
                        _best_effort_close(response)
                        response = None
                    delay = float(config.retry_delay) * (2**attempt)
                    if delay > 0:
                        time.sleep(delay)
                    continue
                raise _transport_error(
                    config.provider,
                    "network request failed",
                    status_code=504 if isinstance(exc, RequestsTimeout) else 502,
                ) from None
            except RequestException:
                raise _transport_error(
                    config.provider, "network request failed"
                ) from None
            finally:
                if response is not None:
                    _best_effort_close(response)
                    response = None
    finally:
        if not stream_owns_session:
            _best_effort_close(session)
    raise _transport_error(config.provider, "request attempts were exhausted")


def _json_shape_is_safe(value: object) -> bool:
    stack: list[tuple[object, int]] = [(value, 1)]
    scheduled_nodes = 1
    try:
        while stack:
            node, depth = stack.pop()
            if depth > _MAX_JSON_DEPTH:
                return False
            if type(node) is dict:
                for key, child in cast(dict[object, object], node).items():
                    if not isinstance(key, str):
                        return False
                    scheduled_nodes += 1
                    if scheduled_nodes > _MAX_JSON_NODES:
                        return False
                    stack.append((child, depth + 1))
                continue
            if type(node) is list:
                for child in cast(list[object], node):
                    scheduled_nodes += 1
                    if scheduled_nodes > _MAX_JSON_NODES:
                        return False
                    stack.append((child, depth + 1))
                continue
            if node is None or isinstance(node, (str, bool)):
                continue
            if isinstance(node, int) and not isinstance(node, bool):
                continue
            if isinstance(node, float) and math.isfinite(node):
                continue
            return False
    except (RecursionError, TypeError, ValueError):
        return False
    return True


def _strict_json_loads(value: str) -> object:
    # task-32805.5: delegate to the one shared strict loader. This family
    # used to accept duplicate keys (last-wins) while the storage family
    # (provider_continuation) rejected them, so a dup-key tool argument
    # accepted here raised an uncaught error when its checkpoint was built.
    # Both now reject duplicate keys.
    try:
        return strict_json_loads(
            value, max_depth=_MAX_JSON_DEPTH, max_nodes=_MAX_JSON_NODES
        )
    except StrictJSONError:
        return _JSON_DECODE_FAILED


_PAYLOAD_MAX_JSON_DEPTH = 64
_PAYLOAD_MAX_JSON_NODES = 50_000
_PAYLOAD_MAX_JSON_STRING_CHARS = 16 * 1024 * 1024
TOOL_FUNCTION_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{2,63}$")


class ProviderPayloadValidators:
    """Shared request-payload validators for hosted Chat-Completions providers.

    Moonshot and Z.ai carried these as two label-only copies; ADR-175 homes
    the mechanics here, parameterized by the provider wire identity. Payload
    builders stay with their providers (ADR-062) and credential resolution
    stays per provider (ADR-012/ADR-062: the neutral layer never imports
    configuration).
    """

    def __init__(self, provider: str, label: str):
        self._provider = provider
        self._label = label

    def bad_request(self, message: str) -> ChatBadRequestError:
        """Build the family bad-request error.

        Args:
            message: Human-readable failure detail.

        Returns:
            The provider-identified ``ChatBadRequestError``.
        """
        return ChatBadRequestError(provider=self._provider, message=message)

    def configuration_error(self, message: str) -> ChatConfigurationError:
        """Build the family configuration error.

        Args:
            message: Human-readable failure detail.

        Returns:
            The provider-identified ``ChatConfigurationError``.
        """
        return ChatConfigurationError(provider=self._provider, message=message)

    def nonnegative_integer(self, 
        settings: Mapping[str, object], name: str, default: int
    ) -> int:
        """Coerce a non-negative integer setting.

        Args:
            settings: Provider settings mapping.
            name: Setting key.
            default: Value used when the key is absent.

        Returns:
            The validated integer.

        Raises:
            ChatConfigurationError: On a negative or non-integer value.
        """
        value = settings.get(name, default)
        if type(value) is not int or value < 0:
            raise self.configuration_error(f"{self._label} {name} must be a non-negative integer.")
        return value

    def nonnegative_number(self, 
        settings: Mapping[str, object], name: str, default: float
    ) -> float:
        """Coerce a finite non-negative numeric setting.

        Args:
            settings: Provider settings mapping.
            name: Setting key.
            default: Value used when the key is absent.

        Returns:
            The validated float.

        Raises:
            ChatConfigurationError: On a non-numeric, negative, or non-finite value.
        """
        value = settings.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise self.configuration_error(f"{self._label} {name} must be numeric.")
        normalized = float(value)
        if not math.isfinite(normalized) or normalized < 0:
            raise self.configuration_error(f"{self._label} {name} must be non-negative.")
        return normalized

    def positive_integer(self, name: str, value: object) -> int:
        """Coerce a positive integer payload value.

        Args:
            name: Setting name, for the error message.
            value: The candidate integer.

        Returns:
            The validated integer.

        Raises:
            ChatBadRequestError: On a zero, negative, or non-integer value.
        """
        if type(value) is not int or value <= 0:
            raise self.bad_request(f"{self._label} {name} is invalid.")
        return value

    def positive_number(self, 
        settings: Mapping[str, object], name: str, default: float
    ) -> float:
        """Coerce a finite positive numeric setting.

        Args:
            settings: Provider settings mapping.
            name: Setting key.
            default: Value used when the key is absent.

        Returns:
            The validated float.

        Raises:
            ChatConfigurationError: On a non-numeric, non-positive, or non-finite value.
        """
        value = settings.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise self.configuration_error(f"{self._label} {name} must be numeric.")
        normalized = float(value)
        if not math.isfinite(normalized) or normalized <= 0:
            raise self.configuration_error(f"{self._label} {name} must be positive and finite.")
        return normalized

    def normalize_stop(self, value: object) -> object:
        """Normalize a stop/stop-sequence value.

        Args:
            value: A non-empty string, or a 1-4 element sequence of them.

        Returns:
            The string, or the sequence as a list.

        Raises:
            ChatBadRequestError: On any other shape.
        """
        if isinstance(value, str) and value:
            return value
        if (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and 1 <= len(value) <= 4
            and all(isinstance(item, str) and item for item in value)
        ):
            return list(value)
        raise self.bad_request(f"{self._label} stop is invalid.")

    def normalize_response_format(self, value: object) -> dict[str, Any]:
        """Normalize a response_format payload.

        Args:
            value: A ``text``/``json_object``/``json_schema`` mapping.

        Returns:
            A deep copy of the validated mapping.

        Raises:
            ChatBadRequestError: On an unknown type or unsupported keys.
        """
        if not isinstance(value, Mapping) or not self.json_shape_is_bounded(value):
            raise self.bad_request(f"{self._label} response format is invalid.")
        format_type = value.get("type")
        if format_type in {"text", "json_object"}:
            if set(value) != {"type"}:
                raise self.bad_request(f"{self._label} response format is invalid.")
        elif format_type == "json_schema":
            if set(value) != {"type", "json_schema"} or not isinstance(
                value.get("json_schema"), Mapping
            ):
                raise self.bad_request(f"{self._label} response format is invalid.")
        else:
            raise self.bad_request(f"{self._label} response format is invalid.")
        return deepcopy(dict(value))

    def normalize_call_batch(self, 
        value: object,
        prior_ids: set[str],
    ) -> tuple[dict[str, Any], ...]:
        """Normalize an assistant tool-call batch.

        Args:
            value: Sequence of raw ``{id, type, function}`` call mappings.
            prior_ids: Mutable set of already-seen call ids, extended here.

        Returns:
            The validated calls as a tuple of deep-copied mappings.

        Raises:
            ChatBadRequestError: On a malformed batch, duplicate id, invalid
                function name, or unparseable arguments.
        """
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
            raise self.bad_request(f"{self._label} tool call batch is invalid.")
        calls: list[dict[str, Any]] = []
        for raw_call in value:
            if not isinstance(raw_call, Mapping) or set(raw_call) != {
                "id",
                "type",
                "function",
            }:
                raise self.bad_request(f"{self._label} tool call is malformed.")
            call_id = raw_call.get("id")
            function = raw_call.get("function")
            if (
                not isinstance(call_id, str)
                or not call_id
                or call_id in prior_ids
                or raw_call.get("type") != "function"
                or not isinstance(function, Mapping)
                or set(function) != {"name", "arguments"}
                or not isinstance(function.get("name"), str)
                or not TOOL_FUNCTION_NAME.fullmatch(cast(str, function.get("name")))
                or not isinstance(function.get("arguments"), str)
            ):
                raise self.bad_request(f"{self._label} tool call is malformed.")
            try:
                arguments = json.loads(cast(str, function.get("arguments")))
            except (TypeError, ValueError):
                raise self.bad_request(f"{self._label} tool call arguments are invalid.") from None
            if not isinstance(arguments, dict) or not self.json_shape_is_bounded(arguments):
                raise self.bad_request(f"{self._label} tool call arguments are invalid.")
            prior_ids.add(call_id)
            calls.append(deepcopy(dict(raw_call)))
        return tuple(calls)

    def json_shape_is_bounded(self, value: object) -> bool:
        """Check a decoded value against the payload shape caps.

        Args:
            value: An already-decoded JSON value.

        Returns:
            True when depth, node count, string sizes, key types, and float
            finiteness are all within the shared payload caps.
        """
        stack: list[tuple[object, int]] = [(value, 1)]
        nodes = 0
        while stack:
            current, depth = stack.pop()
            nodes += 1
            if nodes > _PAYLOAD_MAX_JSON_NODES or depth > _PAYLOAD_MAX_JSON_DEPTH:
                return False
            if current is None or type(current) in {bool, int}:
                continue
            if isinstance(current, float):
                if not math.isfinite(current):
                    return False
                continue
            if isinstance(current, str):
                if len(current) > _PAYLOAD_MAX_JSON_STRING_CHARS:
                    return False
                continue
            if isinstance(current, Mapping):
                for key, item in current.items():
                    if not isinstance(key, str) or len(key) > _PAYLOAD_MAX_JSON_STRING_CHARS:
                        return False
                    stack.append((item, depth + 1))
                continue
            if isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
                stack.extend((item, depth + 1) for item in current)
                continue
            return False
        return True




def _required_metadata(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > _MAX_METADATA_CHARS:
        raise HostedChatProtocolError(f"Hosted Chat {label} is malformed.")
    return value


def _validate_reasoning(
    policy: HostedChatFinishPolicy,
    value: object,
) -> str | None:
    try:
        reasoning = policy.validate_reasoning_content(value)
    except (HostedChatProtocolError, ChatProviderError):
        raise
    except Exception:
        raise HostedChatProtocolError(
            "Hosted Chat reasoning metadata is malformed."
        ) from None
    if reasoning is not None and not isinstance(reasoning, str):
        raise HostedChatProtocolError("Hosted Chat reasoning metadata is malformed.")
    return reasoning


def _validate_finish(
    policy: HostedChatFinishPolicy,
    *,
    finish_reason: object,
    has_text: bool,
    has_calls: bool,
) -> str:
    try:
        normalized = policy.validate_finish(
            finish_reason=finish_reason,
            has_text=has_text,
            has_calls=has_calls,
        )
    except (HostedChatProtocolError, ChatProviderError):
        raise
    except Exception:
        raise HostedChatProtocolError(
            "Hosted Chat finish state is malformed."
        ) from None
    if not isinstance(normalized, str) or not normalized:
        raise HostedChatProtocolError("Hosted Chat finish state is malformed.")
    return normalized


def _normalize_tool_calls(value: object) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise HostedChatProtocolError("Hosted Chat tool calls are malformed.")
    if len(value) > _MAX_TOOL_CALLS:
        raise HostedChatProtocolError("Hosted Chat tool call limit was exceeded.")
    calls: list[dict[str, Any]] = []
    call_ids: set[str] = set()
    for raw_call in value:
        if not isinstance(raw_call, Mapping) or set(raw_call) != {
            "id",
            "type",
            "function",
        }:
            raise HostedChatProtocolError("Hosted Chat tool call is malformed.")
        call_id = _required_metadata(raw_call.get("id"), "tool ID")
        if raw_call.get("type") != "function" or call_id in call_ids:
            raise HostedChatProtocolError("Hosted Chat tool identity is malformed.")
        function = raw_call.get("function")
        if not isinstance(function, Mapping) or set(function) != {"name", "arguments"}:
            raise HostedChatProtocolError("Hosted Chat tool function is malformed.")
        name = _required_metadata(function.get("name"), "tool name")
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            raise HostedChatProtocolError("Hosted Chat tool arguments are malformed.")
        decoded_arguments = _strict_json_loads(arguments)
        if decoded_arguments is _JSON_DECODE_FAILED or not isinstance(
            decoded_arguments, Mapping
        ):
            raise HostedChatProtocolError("Hosted Chat tool arguments are malformed.")
        call_ids.add(call_id)
        calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return tuple(calls)


def _finish_stream_tools(
    states: Mapping[int, _StreamToolState],
) -> tuple[dict[str, Any], ...]:
    if not states:
        return ()
    if tuple(sorted(states)) != tuple(range(len(states))):
        raise HostedChatProtocolError("Hosted Chat stream tool indexes are incomplete.")
    return _normalize_tool_calls(
        [
            {
                "id": states[index].call_id,
                "type": "function",
                "function": {
                    "name": states[index].name,
                    "arguments": "".join(states[index].arguments),
                },
            }
            for index in range(len(states))
        ]
    )


def _tool_character_count(tool_calls: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        len(cast(str, call["id"]))
        + len(cast(str, cast(Mapping[str, Any], call["function"])["name"]))
        + len(cast(str, cast(Mapping[str, Any], call["function"])["arguments"]))
        for call in tool_calls
    )


def _build_turn(
    *,
    text: str,
    reasoning_content: str | None,
    tool_calls: tuple[dict[str, Any], ...],
    finish_reason: str,
    usage: dict[str, Any] | None,
) -> HostedChatTurn:
    assistant_message: dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning_content is not None:
        assistant_message["reasoning_content"] = reasoning_content
    if tool_calls:
        assistant_message["tool_calls"] = deepcopy(list(tool_calls))
    return HostedChatTurn(
        text=text,
        tool_calls=deepcopy(tool_calls),
        assistant_message=assistant_message,
        finish_reason=finish_reason,
        reasoning_content=reasoning_content,
        usage=deepcopy(usage),
    )


def _best_effort_close(resource: object) -> None:
    try:
        cast(Any, resource).close()
    except Exception:
        pass


def _transport_error(
    provider: str,
    detail: str,
    *,
    status_code: int = 502,
) -> ChatProviderError:
    return ChatProviderError(
        provider=provider,
        message=f"{provider} {detail}.",
        status_code=status_code,
    )


def _raise_http_error(provider: str, status: int) -> Never:
    if status in {401, 403}:
        raise ChatAuthenticationError(
            provider=provider,
            message=f"{provider} authentication failed. Check the API key.",
        ) from None
    if status == 429:
        raise ChatRateLimitError(
            provider=provider,
            message=f"{provider} rate limit exceeded. Retry later.",
        ) from None
    if 400 <= status < 500:
        raise ChatBadRequestError(
            provider=provider,
            message=f"{provider} rejected the request (status {status}).",
        ) from None
    raise _transport_error(
        provider,
        f"service failed (status {status})",
        status_code=status,
    ) from None


#: task-32805.4: an upper bound on a provider-supplied Retry-After. api_base_url
#: is user-configurable for these providers and time.sleep() runs on the
#: gateway worker thread that Stop cannot interrupt, so an unbounded value
#: (e.g. Retry-After: 99999999) pins the worker for an arbitrary time.
_MAX_RETRY_AFTER_SECONDS = 60.0


def _retry_delay(
    response: requests.Response,
    *,
    attempt: int,
    retry_delay: float,
) -> float:
    raw_value = response.headers.get("Retry-After")
    if raw_value is not None:
        try:
            if raw_value.strip().isdigit():
                # Bounded (TASK-32853): api_base_url is user-configurable for
                # every hosted provider, so a misbehaving endpoint can name
                # any delay -- and Stop cancels the task but cannot interrupt
                # this worker thread's sleep. Honour the header only up to
                # the cap; a longer demand still retries, after the cap.
                return min(
                    max(0.0, float(int(raw_value.strip()))),
                    _MAX_RETRY_AFTER_SECONDS,
                )
            from email.utils import parsedate_to_datetime

            parsed = parsedate_to_datetime(raw_value)
            return min(
                max(0.0, parsed.timestamp() - time.time()),
                _MAX_RETRY_AFTER_SECONDS,
            )
        except (OverflowError, TypeError, ValueError):
            pass
    return min(retry_delay * (2**attempt), _MAX_RETRY_AFTER_SECONDS)


def _invalid_base_url() -> HostedChatBaseURLValidationError:
    return HostedChatBaseURLValidationError("Hosted Chat API base URL is malformed.")


def _endpoint_markers(path: str) -> frozenset[tuple[int, str]]:
    segments = tuple(segment.casefold() for segment in path.strip("/").split("/"))
    markers = {
        (index, segment)
        for index, segment in enumerate(segments)
        if segment == "responses"
        or (segment == "models" and index == len(segments) - 1)
    }
    markers.update(
        (index, "chat/completions")
        for index in range(len(segments) - 1)
        if segments[index : index + 2] == ("chat", "completions")
    )
    return frozenset(markers)


def _terminal_chat_suffix_is_valid(path: str) -> bool:
    segments = tuple(path.strip("/").split("/"))
    markers = _endpoint_markers(path)
    if not markers:
        return True
    return markers == {(len(segments) - 2, "chat/completions")} and segments[-2:] == (
        "chat",
        "completions",
    )


def _validate_percent_encoded_path(path: str) -> None:
    validation_path = path
    markers = _endpoint_markers(validation_path)
    for _pass in range(_MAX_PATH_DECODE_PASSES):
        if _ENCODED_PATH_SEPARATOR_RE.search(validation_path):
            raise _invalid_base_url()
        try:
            decoded_path = unquote(validation_path, errors="strict")
        except UnicodeDecodeError as exc:
            raise _invalid_base_url() from exc
        if decoded_path == validation_path:
            break
        decoded_markers = _endpoint_markers(decoded_path)
        if (
            any(
                ord(character) < 32 or ord(character) == 127
                for character in decoded_path
            )
            or "\\" in decoded_path
            or any(segment in {".", ".."} for segment in decoded_path.split("/"))
            or decoded_markers - markers
        ):
            raise _invalid_base_url()
        validation_path = decoded_path
        markers = decoded_markers
    if _PERCENT_ESCAPE_RE.search(validation_path) is not None:
        raise _invalid_base_url()


def normalize_hosted_chat_base_url(value: object, *, default: str) -> str:
    """Return one structural hosted Chat API base without a request suffix.

    Args:
        value: Configured base URL. ``None`` selects ``default``.
        default: Provider-owned default base URL.

    Returns:
        The validated base URL without a trailing slash or terminal exact
        lowercase ``/chat/completions`` suffix.

    Raises:
        HostedChatBaseURLValidationError: If the URL is malformed, unsafe, or
            already names an ambiguous request/discovery endpoint.
    """
    candidate = default if value is None else value
    if (
        not isinstance(candidate, str)
        or not candidate
        or len(candidate) > _MAX_URL_LENGTH
        or candidate != candidate.strip()
        or any(character.isspace() for character in candidate)
        or any(ord(character) < 32 or ord(character) == 127 for character in candidate)
        or "\\" in candidate
        or "?" in candidate
        or "#" in candidate
    ):
        raise _invalid_base_url()

    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError as exc:
        raise _invalid_base_url() from exc
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        raise _invalid_base_url()
    if parsed.username is not None or parsed.password is not None:
        raise _invalid_base_url()
    if any(character in parsed.netloc for character in '\\%|^{}<>"`'):
        raise _invalid_base_url()
    if parsed.query or parsed.fragment:
        raise _invalid_base_url()
    if parsed.netloc.endswith(":") or (port is not None and not 0 < port < 65_536):
        raise _invalid_base_url()

    path = parsed.path
    if (
        "//" in path
        or re.search(r"%(?![0-9A-Fa-f]{2})", path) is not None
        or any(segment in {".", ".."} for segment in path.split("/"))
        or not _terminal_chat_suffix_is_valid(path)
    ):
        raise _invalid_base_url()
    _validate_percent_encoded_path(path)

    path = path.rstrip("/")
    suffix = "/chat/completions"
    if path.endswith(suffix):
        path = path[: -len(suffix)]
    return f"{parsed.scheme.casefold()}://{parsed.netloc}{path}".rstrip("/")
