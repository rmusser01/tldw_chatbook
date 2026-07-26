"""Provider response -> a normalized top-K distribution.

Pinned to payloads captured from a live llama.cpp server, not to
documentation. The spec originally predicted two different shapes -- a
modern content[] form for chat and a legacy token->logprob dict for raw
completions. Observation showed both endpoints return the modern form and
carry a token id the spec had not anticipated.

A provider whose shape is not pinned by a fixture is not supported. Shapes
are never inferred.
"""

from __future__ import annotations

import re
from typing import Any

from .models import TokenProb

#: Tokens shaped like <|foo|>, <|foo>, or <foo_bar> -- chat-template markers.
_BRACKETED = re.compile(r"^<\|?[A-Za-z0-9_\-]+\|?>$")

#: A control token is near-deterministic. A bracket-shaped token the model is
#: genuinely uncertain about is content (markup, code), not template.
_CONTROL_LOGPROB_CEILING = -0.05

#: How many positions to search for a content token before giving up.
CONTENT_TOKEN_WINDOW = 8


class NormalizerError(Exception):
    """The response shape was unrecognized, or held no usable distribution."""


def is_control_token(token: str, logprob: float) -> bool:
    """Structural control-token test.

    Identified by shape plus near-certainty rather than a hardcoded list,
    because every chat template uses different markers.
    """
    return bool(_BRACKETED.match(token)) and logprob >= _CONTROL_LOGPROB_CEILING


def _content_positions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        choices = payload["choices"]
        logprobs = choices[0]["logprobs"]
    except (KeyError, IndexError, TypeError) as exc:
        raise NormalizerError(
            f"response carries no logprobs; got keys {list(payload)!r}"
        ) from exc
    if not isinstance(logprobs, dict) or "content" not in logprobs:
        raise NormalizerError(
            "unrecognized logprobs shape: expected a 'content' array "
            f"(got {list(logprobs) if isinstance(logprobs, dict) else type(logprobs)!r}). "
            "Capture a fixture for this provider before claiming support."
        )
    content = logprobs["content"]
    if not content:
        raise NormalizerError("logprobs.content was empty")
    return content


def _to_token_probs(entry: dict[str, Any]) -> list[TokenProb]:
    raw = entry.get("top_logprobs") or []
    out = [
        TokenProb(
            token=item["token"],
            logprob=float(item["logprob"]),
            bytes_=tuple(item.get("bytes") or ()),
            token_id=item.get("id"),
        )
        for item in raw
    ]
    out.sort(key=lambda t: t.logprob, reverse=True)
    return out


def normalize_logprobs(
    payload: dict[str, Any], *, want_content_token: bool
) -> tuple[list[TokenProb], int]:
    """Return ``(top_k, content_offset)``.

    Args:
        payload: The provider's decoded JSON response.
        want_content_token: When True (chat mode), skip leading control
            tokens and measure the first content position. When False (raw
            mode), measure position 0.

    Raises:
        NormalizerError: shape unrecognized, or no content token in window.
    """
    content = _content_positions(payload)

    if not want_content_token:
        return _to_token_probs(content[0]), 0

    for offset, entry in enumerate(content[:CONTENT_TOKEN_WINDOW]):
        if not is_control_token(entry.get("token", ""), float(entry.get("logprob", 0.0))):
            return _to_token_probs(entry), offset

    raise NormalizerError(
        "no_content_token: every position within the first "
        f"{CONTENT_TOKEN_WINDOW} was a control token. This target's template "
        "emits only control tokens in the measured window."
    )
