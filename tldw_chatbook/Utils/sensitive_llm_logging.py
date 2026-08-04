"""Request-scoped logging policy for sensitive LLM operations.

The policy is deliberately carried by a :class:`contextvars.ContextVar` so a
single auxiliary request can redact its final adapter diagnostics without
changing process-wide logger configuration or affecting concurrent chat work.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from urllib.parse import urlsplit


_SENSITIVE_LLM_REQUEST: ContextVar[bool] = ContextVar(
    "sensitive_llm_request", default=False
)
SENSITIVE_CONTENT_REDACTION = "<sensitive-content-redacted>"
SENSITIVE_ERROR_REDACTION = "<sensitive-error-detail-redacted>"

# task-2117 Qodo round: the fixed set of provider request-payload fields that
# are safe to write to a debug log verbatim. This is an ALLOWLIST, not a
# denylist -- a denylist ("everything except messages/contents") has already
# failed twice on this exact logging path: first when only messages/contents
# were recognized as content-bearing, then again when providers turned out
# to carry prompt content under OTHER keys (OpenAI Responses API ``input``,
# Anthropic ``system``, Google ``system_instruction``). Any payload key not
# listed here is dropped by ``safe_llm_request_payload_summary`` -- an
# unrecognized/future key is safe by default instead of exposed by default.
SAFE_LLM_PAYLOAD_SCALAR_KEYS: tuple[str, ...] = (
    "model",
    "stream",
    "streaming",
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "temperature",
    "top_p",
    "top_k",
    # Google Gemini's generationConfig uses camelCase field names; callers
    # flatten generationConfig to the top level before summarizing.
    "topP",
    "topK",
    "maxOutputTokens",
)


def is_sensitive_llm_request() -> bool:
    """Return whether the current execution context handles sensitive LLM data."""

    return _SENSITIVE_LLM_REQUEST.get()


@contextmanager
def sensitive_llm_request() -> Iterator[None]:
    """Mark the current request sensitive and restore the prior policy on exit."""

    token = _SENSITIVE_LLM_REQUEST.set(True)
    try:
        yield
    finally:
        _SENSITIVE_LLM_REQUEST.reset(token)


def safe_llm_log_value(value: object) -> object:
    """Return a value safe to interpolate into a diagnostic message.

    Call this before slicing, formatting, previewing, or serializing a value
    that may contain request or response content.
    """

    if is_sensitive_llm_request():
        return SENSITIVE_CONTENT_REDACTION
    return value


def safe_llm_error_detail(value: object) -> object:
    """Return error detail or a stable redaction for a sensitive request."""

    if is_sensitive_llm_request():
        return SENSITIVE_ERROR_REDACTION
    return value


def safe_llm_url_host(value: object) -> str:
    """Return only a URL host while sensitive.

    Args:
        value: URL-like diagnostic value to sanitize.

    Returns:
        The original diagnostic outside sensitive requests, otherwise only
        its hostname or ``"unknown"`` when no safe hostname can be parsed.
    """

    raw = str(value or "")
    if not is_sensitive_llm_request():
        return raw
    try:
        parsed = urlsplit(raw if "://" in raw else f"//{raw}")
        return parsed.hostname or "unknown"
    except ValueError:
        return "unknown"


def llm_content_byte_count(value: object) -> int:
    """Count UTF-8 content bytes without building a body preview or JSON string."""

    if value is None:
        return 0
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, Mapping):
        return sum(
            llm_content_byte_count(key) + llm_content_byte_count(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return sum(llm_content_byte_count(item) for item in value)
    return len(str(value).encode("utf-8"))


def llm_retry_count(configured: int) -> int:
    """Disable transport retries for a sensitive one-shot LLM request."""

    return 0 if is_sensitive_llm_request() else configured


def safe_llm_exception_message(exc: BaseException) -> str:
    """Return an exception description safe for logs and surfaced errors."""

    if is_sensitive_llm_request():
        return type(exc).__name__
    return str(exc)


def safe_llm_tool_names(tools: object) -> list[str]:
    """Return only the callable tool NAMES from a request's tool definitions.

    Provider tool entries also carry descriptions, JSON-schema parameters,
    and (on tool-call messages) arguments -- none of that is safe to write
    to a log. This understands OpenAI-style ``{"function": {"name": ...}}``
    entries, Anthropic's converted ``{"name": ...}`` entries, and Google's
    ``functionDeclarations``/``function_declarations`` wrapper; any other
    shape is skipped rather than serialized.

    Args:
        tools: The raw or provider-converted ``tools`` payload value.

    Returns:
        A list of tool name strings, in payload order. Empty when ``tools``
        is falsy or none of its entries have a recognizable name.
    """

    names: list[str] = []
    for entry in tools or []:
        if not isinstance(entry, Mapping):
            continue
        function = entry.get("function")
        if isinstance(function, Mapping) and isinstance(function.get("name"), str):
            names.append(function["name"])
            continue
        if isinstance(entry.get("name"), str):
            names.append(entry["name"])
            continue
        declarations = entry.get("functionDeclarations") or entry.get(
            "function_declarations"
        )
        if isinstance(declarations, list):
            for declaration in declarations:
                if isinstance(declaration, Mapping) and isinstance(
                    declaration.get("name"), str
                ):
                    names.append(declaration["name"])
    return names


def safe_llm_request_payload_summary(
    payload: Mapping[str, object],
    *,
    content_keys: Sequence[str] = ("messages",),
    system_keys: Sequence[str] = (),
    tools_key: str = "tools",
) -> dict[str, object]:
    """Build an allowlisted, content-free summary of an outgoing LLM request payload.

    Only the fixed scalar keys in :data:`SAFE_LLM_PAYLOAD_SCALAR_KEYS` are
    copied through verbatim. Everything else about the request is reduced to
    non-content facts: a message COUNT (never the messages themselves),
    whether a system prompt is PRESENT (a boolean, never its text), and tool
    NAMES (never schemas or arguments, via :func:`safe_llm_tool_names`). Any
    other key in ``payload`` -- including ones no current provider happens
    to use -- is dropped. See ``Tests/Chat/test_sensitive_llm_logging.py``
    for the regression coverage, including the "unknown payload key" property
    test this shape exists to satisfy.

    Args:
        payload: The outgoing provider request payload, or a flattened view
            of it. Callers whose sampling parameters live in a nested dict
            (e.g. Google's ``generationConfig``) should merge that dict to
            the top level before calling this.
        content_keys: Keys to check, in order, for the conversation/messages
            list. The first one present in ``payload`` is summarized as
            ``message_count`` and scanned for a ``role: "system"`` entry.
        system_keys: Keys holding a system prompt/instruction as a separate
            top-level field (str, content-block list, or dict, depending on
            provider). Any truthy value marks ``has_system_prompt`` True.
        tools_key: Key holding the tool/function definitions list.

    Returns:
        A new dict containing only allowlisted scalar fields plus whichever
        of ``message_count``, ``has_system_prompt``, and ``tool_names``
        apply to this payload.
    """

    summary: dict[str, object] = {
        key: payload[key] for key in SAFE_LLM_PAYLOAD_SCALAR_KEYS if key in payload
    }

    message_list: object = None
    for content_key in content_keys:
        if content_key in payload:
            message_list = payload[content_key]
            break

    has_system_role = False
    if isinstance(message_list, (list, tuple)):
        summary["message_count"] = len(message_list)
        has_system_role = any(
            isinstance(entry, Mapping) and entry.get("role") == "system"
            for entry in message_list
        )

    has_system_field = any(bool(payload.get(key)) for key in system_keys)
    if content_keys or system_keys:
        summary["has_system_prompt"] = bool(has_system_role or has_system_field)

    tools_value = payload.get(tools_key)
    if tools_value:
        tool_names = safe_llm_tool_names(tools_value)
        if tool_names:
            summary["tool_names"] = tool_names

    return summary
