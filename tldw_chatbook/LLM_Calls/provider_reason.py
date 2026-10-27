"""A provider's own reason for a rate limit, from its JSON error body.

TASK-34100.2 AC#18. A 429 reached the Console as 'rate limit exceeded' at
best -- the provider's explanation (which quota, when to retry) stayed in the
log. The handlers that read the error body (Google's, Anthropic's, Cohere's,
Hugging Face's and the shared ``chat_api_call`` dispatcher used by OpenAI and
the other requests-based handlers) pass this reason on
``ChatRateLimitError.provider_reason``. Console reads it through the shared
``Chat/provider_error_reason.provider_reason_for_exception`` (TASK-34100.5),
which caps, scrubs and markup-escapes every provider sentence, and quotes it
after the category. The hosted-engine providers (OpenRouter, Groq, Mistral,
...) attach their body's ``error.message`` the same way
(``hosted_chat._raise_http_error``, TASK-34100.5); this module adds Google's
quota and retry delay, which that allowlisted field does not carry.

Only the provider's ``error.message`` text is read -- never the request --
and Google's structured ``QuotaFailure``/``RetryInfo`` replace its 400
characters of help links with the quota and the delay. The result is
whitespace-collapsed, printable, bounded, and never echoes a key the caller
names; Console's own credential sanitizer runs on it again before display.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

#: Google quota-id fragments -> the period words a reader understands.
_GOOGLE_QUOTA_PERIODS = (
    ("PerMinute", "per minute"),
    ("PerHour", "per hour"),
    ("PerDay", "per day"),
    ("PerSecond", "per second"),
)
RATE_LIMIT_REASON_MAX_CHARS = 240


def _google_quota_line(details: list[Any]) -> tuple[str, str]:
    """The quota that was hit and the retry delay, from Google's details."""
    quota = ""
    retry = ""
    for detail in details:
        if not isinstance(detail, Mapping):
            continue
        kind = str(detail.get("@type", ""))
        if kind.endswith("QuotaFailure") and not quota:
            violations = detail.get("violations")
            violation = (
                violations[0] if isinstance(violations, list) and violations else None
            )
            if not isinstance(violation, Mapping):
                continue
            quota_id = str(violation.get("quotaId", ""))
            value = str(violation.get("quotaValue", "")).strip()
            period = next(
                (words for marker, words in _GOOGLE_QUOTA_PERIODS if marker in quota_id),
                "",
            )
            if value.isdigit() and period:
                unit = "tokens" if "Token" in quota_id else "requests"
                tier = "Free-tier limit" if "FreeTier" in quota_id else "Limit"
                quota = f"{tier}: {value} {unit} {period}"
                dimensions = violation.get("quotaDimensions")
                model = dimensions.get("model") if isinstance(dimensions, Mapping) else None
                if isinstance(model, str) and model.strip():
                    quota += f" for {model.strip()}"
                quota += "."
        elif kind.endswith("RetryInfo"):
            delay = detail.get("retryDelay")
            if isinstance(delay, str) and delay.strip():
                retry = f"Retry in {delay.strip()}."
    return quota, retry


def rate_limit_reason(body: object, *, secrets: Iterable[object] = ()) -> str:
    """The provider's own reason for a 429, compact and safe to show.

    Args:
        body: The error response body (text, bytes or an already-parsed
            mapping).
        secrets: Credentials the request carried; any echo is masked.

    Returns:
        For Google's free tier, for example: 'You exceeded your current
        quota, please check your plan and billing details. Free-tier limit:
        5 requests per minute for gemini-3.8-flash. Retry in 59s.'; ``""``
        when the body carries no message.
    """
    payload: object = body
    if isinstance(body, (bytes, bytearray)):
        try:
            payload = body.decode("utf-8")
        except UnicodeDecodeError:
            return ""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (RecursionError, ValueError):
            return ""
    if not isinstance(payload, Mapping):
        return ""
    error = payload.get("error")
    message: object = None
    details: object = None
    if isinstance(error, Mapping):
        message = error.get("message")
        details = error.get("details")
    elif isinstance(error, str):
        message = error
    elif isinstance(payload.get("message"), str):
        message = payload["message"]
    if not isinstance(message, str) or not message.strip():
        return ""
    text = " ".join(message.split())
    quota, retry = _google_quota_line(details) if isinstance(details, list) else ("", "")
    if quota or retry:
        # Google: the first sentence; the quota and the delay follow.
        cut = text.find(". ")
        text = " ".join(
            part for part in (text[: cut + 1] if cut != -1 else text, quota, retry) if part
        )
    for secret in secrets:
        if isinstance(secret, str) and secret.strip():
            text = text.replace(secret.strip(), "[redacted]")
    text = "".join(character for character in text if character.isprintable())
    if len(text) > RATE_LIMIT_REASON_MAX_CHARS:
        text = text[: RATE_LIMIT_REASON_MAX_CHARS - 1].rstrip() + "…"
    return text
