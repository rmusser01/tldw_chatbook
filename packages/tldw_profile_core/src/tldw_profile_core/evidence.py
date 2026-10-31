"""Pure exact-text digests; source authority and support remain caller-owned."""

from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True, slots=True)
class ExactTextSpanDigests:
    """Exact UTF-8 identities; neither field establishes evidence support."""

    representation_sha256: str
    span_sha256: str


def exact_text_span_digests(
    representation: str, start: int, end: int
) -> ExactTextSpanDigests:
    """Hash an exact representation and its half-open codepoint span.

    Args:
        representation: Already-owned built-in string, without normalization.
        start: Inclusive codepoint offset as a built-in integer.
        end: Exclusive codepoint offset as a built-in integer.

    Returns:
        Named lowercase SHA-256 hex digests, without source text.

    Raises:
        TypeError: Text or offsets have unsupported scalar types.
        ValueError: Bounds do not satisfy the half-open range convention.
        UnicodeEncodeError: Any source codepoint cannot be encoded as UTF-8.
    """
    if type(representation) is not str:
        raise TypeError("representation must be a built-in string")
    if type(start) is not int or type(end) is not int:
        raise TypeError("span offsets must be built-in integers")
    if not 0 <= start <= end <= len(representation):
        raise ValueError("span must satisfy 0 <= start <= end <= text length")
    representation_utf8 = representation.encode("utf-8", errors="strict")
    span_utf8 = representation[start:end].encode("utf-8", errors="strict")
    return ExactTextSpanDigests(
        representation_sha256=sha256(representation_utf8).hexdigest(),
        span_sha256=sha256(span_utf8).hexdigest(),
    )
