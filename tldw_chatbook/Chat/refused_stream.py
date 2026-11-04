"""A streamed Console request the server may have refused for streaming.

TASK-33620.10 made every Console provider stream by default. A server that
cannot stream answers ``"stream": true`` with a client error before any text
arrives, or opens the stream and sends no text. Two consequences follow, and
they share the one classification below:

* TASK-37421 (owner decision 2026-10-29): a self-hosted server or a Custom
  OpenAI-compatible slot is asked once more without streaming, the way the
  llama.cpp path already supports. Each physical retry retains its own
  admission, usage and trace; missing usage is not proof of zero spend.
* TASK-37180: a provider that is not retried (a cloud provider, where a
  silent second paid call is not wanted, and for now a named custom endpoint
  run by the hosted engine) names the Streaming setting in its failure copy,
  beside the model advice.

Classification and recovery copy only: nothing here reaches the wire.
Imported lazily by the gateway (ADR-097 boot ratchet).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tldw_chatbook.Provider_Contract import (
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
    if type(status) is not int or status not in REFUSED_STREAM_STATUSES:  # noqa: E721 - reject booleans and coerced statuses
        return False
    from tldw_chatbook.Chat.provider_error_reason import is_context_overflow_error

    return not is_context_overflow_error(exc)


def retries_refused_stream(resolution: Any) -> bool:
    """Whether a refused or empty stream is retried once without streaming.

    Every self-hosted provider and the Custom OpenAI-compatible slots
    (``custom``, ``custom_2``). Cloud providers never are. Nor, for now, is a
    named ``custom-ep:<slug>`` entry that runs through the hosted engine
    (``custom-hosted``): its traced request never aligns with the prepared
    one (each call's surface is an alignment-mismatch omission), so the
    retry's own trace call cannot be reserved and, with capture on, the run
    would block (measured live 2026-10-29, TASK-37762). Its refused stream
    names the Streaming setting instead, as a cloud provider's does.

    Args:
        resolution: The Console provider resolution about to stream.

    Returns:
        True when the resolution streams to a self-hosted endpoint.
    """
    if getattr(resolution, "streaming", False) is not True:
        return False
    if getattr(resolution, "execution_key", None) == "custom-hosted":
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


def _tool_definition_rejection_copy(
    _owner: Any,
    provider_message: str,
    tools: Sequence[Mapping[str, Any]] | None,
) -> str | None:
    """Return recovery copy when a provider 400 blames a sent tool definition.

    TASK-33621.1: OpenAI and Anthropic validate every tool schema before the
    model runs, so a rejected tool definition fails the request on EVERY
    model -- the model-picker advice below cannot help and must not be shown.
    Only a tool name the request actually sent is ever echoed; the provider's
    own text never reaches the copy.

    The advice follows the tool's source, which its sent name tells apart:
    every MCP-bridged tool is named ``mcp__<server>__<tool>``
    (``MCP.tool_naming.llm_tool_name``), and every other tool is Chatbook's
    own -- its rejection is a Chatbook bug to report, and not every such
    tool has a switch to promise.

    Args:
        provider_message: The adapter exception's text (untrusted).
        tools: The OpenAI-shape tool definitions this request sent.

    Returns:
        Copy naming the rejected tool, generic tool-definition copy when the
        message blames a tool that cannot be named, or None otherwise.
    """
    if not tools:
        return None
    from tldw_chatbook.Agents.native_tools import (
        blames_tool_definition,
        rejected_tool_name,
    )

    name = rejected_tool_name(provider_message, tools)
    if name is None:
        if not blames_tool_definition(provider_message):
            return None
        # Blamed only through a marker in the provider's text: say what is
        # likely, not what is certain.
        return (
            "The provider rejected one of the tool definitions sent with this "
            "request before the model ran, so choosing another model is "
            "unlikely to help. A tool from an MCP server is the likeliest "
            "cause: turn MCP servers off on the MCP screen, then send again."
        )
    rejected = (
        f"The provider rejected the tool definition for {_owner.escape_markup(name)} "
        "before the model ran, so choosing another model will not help."
    )
    if name.startswith("mcp__"):
        return (
            f"{rejected} Turn off the MCP server that provides it on the MCP "
            "screen, then send again."
        )
    return (
        f"{rejected} It is one of Chatbook's own tools, so please report it. "
        "If its tool group has a switch on the MCP screen, turning that off "
        "lets you send meanwhile."
    )


def _provider_error_copy_with_model_recovery(
    _owner: Any,
    copy: str,
    *,
    model: str | None,
    status_code: int | None,
    provider_message: str = "",
    tools: Sequence[Mapping[str, Any]] | None = None,
    refused_stream: bool = False,
    model_unavailable: bool = False,
) -> str:
    """Add safe recovery to provider bad-request copy.

    A rejected tool definition gets tool copy (see
    ``_tool_definition_rejection_copy``). A 404, or an explicit
    ``model_not_found`` (``model_unavailable``) on any status, names the model
    and says what to check: Fireworks, SambaNova, Nous and GMI answer an
    unknown model, or one the key cannot use, that way (no-key probes
    2026-09-30, TASK-33640). Any other 400 gets no model advice: it was never
    evidence about the model (TASK-33621.11 AC#3).
    """
    if refused_stream and not _owner._tool_definition_rejection_copy(
        provider_message, tools
    ):
        from tldw_chatbook.Chat.refused_stream import REFUSED_STREAM_ADVICE

        copy = _owner._provider_error_copy_with_model_recovery(
            copy,
            model=model,
            status_code=status_code,
            provider_message=provider_message,
            tools=tools,
            model_unavailable=model_unavailable,
        )
        return f"{copy} {REFUSED_STREAM_ADVICE}"
    if status_code == 404 or model_unavailable:
        model_id = _owner._safe_model_id(model)
        named = (
            f" Selected model: {_owner.escape_markup(model_id)}." if model_id else ""
        )
        return (
            f"{copy}{named} The provider could not find this model or "
            "endpoint, or this key cannot use it. Check the model name and "
            "the key, or choose another model from the model picker "
            "(Alt+M: Switch model)."
        )
    if status_code != 400:
        return copy
    tool_copy = _owner._tool_definition_rejection_copy(provider_message, tools)
    return copy if tool_copy is None else f"{copy} {tool_copy}"


def _safe_model_id(_owner: Any, model: str | None) -> str:
    """The model id fit for user copy, or "" when it is absent or redacted.

    Args:
        model: The selected model identifier.

    Returns:
        A printable, bounded model id; empty when it cannot be shown.
    """
    model_result = _owner.CredentialSanitizer().sanitize(model or "")
    if (
        not model_result.available
        or model_result.redacted
        or type(model_result.value) is not str
    ):
        return ""
    return "".join(
        character for character in model_result.value.strip() if character.isprintable()
    )[: _owner.PROVIDER_ERROR_MODEL_ID_MAX_CHARS]
