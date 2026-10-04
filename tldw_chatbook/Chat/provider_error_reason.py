"""The provider's own reason for a failed request, made safe to show.

TASK-34100.5 AC#4 (gap-06). A failed first reply used to keep only a category
and a status ("authentication failed. Status: 401."), dropping the one
sentence the provider wrote for the user -- OpenRouter's "API key expired.",
Google's "This model ... is no longer available ... use models/<newer>".

Exactly one field is allowlisted: ``error.message`` of a JSON error body
(the OpenAI, OpenRouter, Anthropic and Google shapes all carry it). Nothing
else in the body -- headers echoed in metadata, request ids, types -- is
read. The text is capped at :data:`PROVIDER_REASON_MAX_CHARS`, stripped of
anything that looks like a key (including the masked ``sk-ab***cd`` echo
OpenAI returns), and escaped for Rich markup. Callers prefix it with the
provider's display name.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from rich.markup import escape as escape_markup

#: Hard cap on the provider's own sentence in user copy.
PROVIDER_REASON_MAX_CHARS = 200

#: Read at most this much of an error body; error JSON is small.
_MAX_BODY_BYTES = 65536

#: A key-shaped token: a known prefix (``-`` or ``_`` after it: Groq's
#: ``gsk_``, Hugging Face's ``hf_``), or any run with a masked middle.
_KEY_SHAPED = re.compile(
    r"(?:\b(?:sk|pk|rk|xai|gsk|hf|sk-ant|sk-or)[-_][A-Za-z0-9_\-*]{4,}"
    r"|\bAIza[0-9A-Za-z_\-]{10,}"
    r"|\b[A-Za-z0-9_\-]{2,}\*{3,}[A-Za-z0-9_\-*]*"
    r"|\b(?:Bearer|bearer)\s+[A-Za-z0-9_\-.=]{8,})"
)
_CONTROL = re.compile(r"[\x00-\x1f\x7f]+")


def reason_from_error_body(body: object) -> str:
    """Return the allowlisted ``error.message`` from a provider error body.

    Args:
        body: The response body as text or bytes.

    Returns:
        The provider's message, or ``""`` when the body is not the expected
        JSON shape.
    """
    if isinstance(body, (bytes, bytearray)):
        body = bytes(body[:_MAX_BODY_BYTES]).decode("utf-8", "replace")
    if not isinstance(body, str) or not body.strip():
        return ""
    try:
        payload = json.loads(body[:_MAX_BODY_BYTES])
    except (TypeError, ValueError):
        return ""
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        payload = payload[0]  # Google can wrap the error object in a list.
    error = payload.get("error") if isinstance(payload, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    return message if isinstance(message, str) else ""


def bounded_response_body(response: object) -> str:
    """Read at most :data:`_MAX_BODY_BYTES` of an HTTP error response body."""
    if response is None:
        return ""
    try:
        content = getattr(response, "_content", None)
        if isinstance(content, (bytes, bytearray)) and content:
            return bytes(content[:_MAX_BODY_BYTES]).decode("utf-8", "replace")
        iter_content = getattr(response, "iter_content", None)
        if callable(iter_content):
            chunks: list[bytes] = []
            size = 0
            for chunk in iter_content(chunk_size=8192):
                if not chunk:
                    continue
                chunks.append(chunk)
                size += len(chunk)
                if size >= _MAX_BODY_BYTES:
                    break
            return b"".join(chunks)[:_MAX_BODY_BYTES].decode("utf-8", "replace")
        text = getattr(response, "text", "")
        return text[:_MAX_BODY_BYTES] if isinstance(text, str) else ""
    except Exception:  # noqa: BLE001 -- the reason is optional copy
        return ""


def safe_provider_reason(
    reason: str, *, known_credentials: Iterable[str] = ()
) -> str:
    """Cap, scrub and markup-escape one provider sentence.

    Args:
        reason: Raw provider text.
        known_credentials: Exact secrets that must never appear.

    Returns:
        Display-safe text, or ``""`` when nothing usable remains.
    """
    text = _CONTROL.sub(" ", reason or "")
    for credential in known_credentials:
        if credential and len(credential) >= 6:
            text = text.replace(credential, "(key hidden)")
    text = _KEY_SHAPED.sub("(key hidden)", text)
    text = " ".join(text.split())
    if len(text) > PROVIDER_REASON_MAX_CHARS:
        text = text[: PROVIDER_REASON_MAX_CHARS - 1].rstrip() + "…"
    return escape_markup(text) if text else ""


def provider_reason_for_exception(
    exc: BaseException, *, known_credentials: Iterable[str] = ()
) -> str:
    """Find the provider's own reason on an exception, made safe to show.

    An adapter can attach it as ``provider_reason``; otherwise the HTTP error
    the exception was raised from (``__cause__``, or an unsuppressed
    ``__context__``) is read for its body. ``raise ... from None`` is
    respected: a deliberately severed chain (the sensitive-request policy)
    yields nothing.

    Args:
        exc: The provider failure.
        known_credentials: Exact secrets to scrub.

    Returns:
        Display-safe provider text, or ``""``.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        explicit = getattr(current, "provider_reason", None)
        if isinstance(explicit, str) and explicit.strip():
            return safe_provider_reason(explicit, known_credentials=known_credentials)
        response: Any = getattr(current, "response", None)
        if response is not None:
            reason = reason_from_error_body(bounded_response_body(response))
            if reason:
                return safe_provider_reason(
                    reason, known_credentials=known_credentials
                )
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None
    return ""


def attach_provider_reason(
    exc: BaseException,
    response: object,
    *,
    known_credentials: Iterable[str] = (),
) -> BaseException:
    """Record a response's allowlisted reason on ``exc`` (returns ``exc``).

    Nothing is recorded for a sensitive request (the policy every other
    error-detail path honours), and the exact credentials the request
    carried are replaced before the reason is kept.
    """
    from tldw_chatbook.Utils.sensitive_llm_logging import is_sensitive_llm_request

    if is_sensitive_llm_request():
        return exc
    reason = reason_from_error_body(bounded_response_body(response))
    for credential in known_credentials:
        if credential and len(credential) >= 6:
            reason = reason.replace(credential, "(key hidden)")
    if reason:
        try:
            exc.provider_reason = reason  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 -- slotted exception types
            pass
    return exc
