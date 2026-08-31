"""Mandatory, content-free-failure credential filtering for trace storage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
import re
from urllib.parse import SplitResult, urlsplit, urlunsplit

from tldw_chatbook.Utils.log_sanitizer import REDACTION_MARKER, sanitize_string


CREDENTIAL_FILTER_VERSION = "credentials-v1"
CREDENTIAL_SANITIZER_UNAVAILABLE = "credential_sanitizer_unavailable"
DEFAULT_CREDENTIAL_MAX_NODES = 100_000
DEFAULT_CREDENTIAL_MAX_DEPTH = 64
DEFAULT_CREDENTIAL_MAX_TEXT_CODEPOINTS = 1_000_000
DEFAULT_CREDENTIAL_MAX_KNOWN_VALUES = 256
PII_DETECTOR_VERSION = "builtin-pii-v1"
BUILTIN_PII_RULESET_VERSION = "builtin-pii-rules-v1"
# Opaque provenance identity for the immutable built-in v1 ruleset. This is
# intentionally not derived from detector source or matched content.
BUILTIN_PII_RULESET_REVISION_ID = "8c312a6d-17d4-4f24-a806-3a990e160550"
PII_DETECTOR_UNAVAILABLE = "pii_detector_unavailable"
DEFAULT_PII_MAX_TEXT_CODEPOINTS = 1_000_000
DEFAULT_PII_MAX_MATCHES = 10_000
PII_OMISSION_MARKER = "[PII omitted]"
_OMITTED_TEXT = "[credential omitted]"
_CREDENTIAL_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "auth_token",
        "client_secret",
        "cookie",
        "credential",
        "credentials",
        "passphrase",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "token",
    }
)
_SECRET_TEXT = re.compile(
    r"(?i)(?:authorization\s*:\s*\S+(?:\s+\S+)?|bearer\s+\S+|"
    r"(?:set-)?cookie\s*:\s*\S+(?:\s*;[^\r\n]*)?|"
    r"sk-(?:live-)?[a-z0-9_-]{8,}|"
    r"(?:api[_-]?key|token|password|secret)\s*[=:]\s*\S+)"
)
_PRIVATE_KEY_TEXT = re.compile(r"-----BEGIN(?: [A-Z0-9]+)* PRIVATE KEY-----")
_URL_TEXT = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s<>\"']+")

_BUILTIN_PII_RULES = (
    (
        "builtin-email",
        "email",
        re.compile(r"(?i)(?<![\w.+-])[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+"),
    ),
    (
        "builtin-ssn",
        "government_id",
        re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    ),
    (
        "builtin-phone",
        "phone",
        re.compile(r"(?<!\w)\+?\d(?:[\d .()-]{6,}\d)(?!\w)"),
    ),
    (
        "builtin-ipv4",
        "network_address",
        re.compile(
            r"(?<![\d.])(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\."
            r"(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\d.])"
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class CredentialSanitizationResult:
    """Sanitized value or a content-free unavailable marker."""

    available: bool
    value: object | None = field(repr=False)
    omission_reason_code: str | None
    redacted: bool = False
    detector_version: str = CREDENTIAL_FILTER_VERSION


@dataclass(frozen=True, slots=True)
class PIIRedactionSpan:
    """Content-free Unicode-codepoint range produced by one PII rule."""

    start_codepoint: int
    end_codepoint: int
    category: str
    rule_id: str
    detector_version: str = PII_DETECTOR_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.start_codepoint) is not int
            or type(self.end_codepoint) is not int
            or self.start_codepoint < 0
            or self.end_codepoint <= self.start_codepoint
        ):
            raise ValueError("codepoint_range")
        if not all(
            type(value) is str and 0 < len(value) <= 128
            for value in (self.category, self.rule_id, self.detector_version)
        ):
            raise ValueError("span_metadata")


@dataclass(frozen=True, slots=True)
class PIIDetectionResult:
    """PII spans or a content-free fail-closed detector outcome."""

    available: bool
    spans: tuple[PIIRedactionSpan, ...]
    omission_reason_code: str | None
    detector_version: str = PII_DETECTOR_VERSION


@dataclass(frozen=True, slots=True)
class PIIFieldRedaction:
    """One content-free PII span located within a structured value."""

    field_path: str
    span: PIIRedactionSpan


@dataclass(frozen=True, slots=True)
class PIIValueRedactionResult:
    """Irreversibly masked value plus content-free provenance spans."""

    available: bool
    value: object | None = field(repr=False)
    field_redactions: tuple[PIIFieldRedaction, ...]
    omission_reason_code: str | None
    detector_version: str = PII_DETECTOR_VERSION


def merge_pii_spans(
    spans: Sequence[PIIRedactionSpan],
) -> tuple[PIIRedactionSpan, ...]:
    """Sort and union overlapping PII spans without retaining matched text.

    Args:
        spans: Content-free detector ranges to normalize.

    Returns:
        Deterministically ordered, non-overlapping ranges. Overlap between
        categories is represented as ``mixed``.

    Raises:
        TypeError: If any item is not a :class:`PIIRedactionSpan`.
        ValueError: If merged provenance cannot satisfy span metadata bounds.
    """

    normalized = tuple(spans)
    if any(not isinstance(span, PIIRedactionSpan) for span in normalized):
        raise TypeError("spans")
    if not normalized:
        return ()
    ordered = sorted(
        normalized,
        key=lambda span: (
            span.start_codepoint,
            span.end_codepoint,
            span.category,
            span.rule_id,
            span.detector_version,
        ),
    )
    groups: list[list[PIIRedactionSpan]] = [[ordered[0]]]
    group_end = ordered[0].end_codepoint
    for span in ordered[1:]:
        if span.start_codepoint < group_end:
            groups[-1].append(span)
            group_end = max(group_end, span.end_codepoint)
        else:
            groups.append([span])
            group_end = span.end_codepoint
    result: list[PIIRedactionSpan] = []
    for group in groups:
        categories = sorted({span.category for span in group})
        rule_ids = sorted({span.rule_id for span in group})
        detector_versions = sorted({span.detector_version for span in group})
        result.append(
            PIIRedactionSpan(
                min(span.start_codepoint for span in group),
                max(span.end_codepoint for span in group),
                categories[0] if len(categories) == 1 else "mixed",
                "+".join(rule_ids),
                "+".join(detector_versions),
            )
        )
    return tuple(result)


def apply_pii_mask(
    value: str,
    spans: Sequence[PIIRedactionSpan],
    *,
    marker: str = PII_OMISSION_MARKER,
) -> str:
    """Return a masked projection while leaving the source string unchanged.

    Args:
        value: Original string indexed by Unicode codepoint offsets.
        spans: Content-free ranges to replace.
        marker: Non-empty replacement text used for every merged range.

    Returns:
        A new string with each merged range replaced by ``marker``.

    Raises:
        TypeError: If ``value`` or ``marker`` is not a valid string.
        ValueError: If a range extends beyond ``value`` or has invalid metadata.
    """

    if type(value) is not str or type(marker) is not str or not marker:
        raise TypeError("mask_input")
    merged = merge_pii_spans(spans)
    if any(span.end_codepoint > len(value) for span in merged):
        raise ValueError("codepoint_range")
    parts: list[str] = []
    cursor = 0
    for span in merged:
        parts.append(value[cursor : span.start_codepoint])
        parts.append(marker)
        cursor = span.end_codepoint
    parts.append(value[cursor:])
    return "".join(parts)


class BuiltInPIIDetector:
    """Deterministic bounded built-in PII detector for trace masking."""

    __slots__ = ("_max_matches", "_max_text_codepoints")

    def __init__(
        self,
        *,
        max_text_codepoints: int = DEFAULT_PII_MAX_TEXT_CODEPOINTS,
        max_matches: int = DEFAULT_PII_MAX_MATCHES,
    ) -> None:
        """Create a detector with structural work limits.

        Args:
            max_text_codepoints: Maximum code points inspected per field.
            max_matches: Maximum candidate matches accepted per field.

        Raises:
            ValueError: If either bound is not a positive integer.
        """

        for value, name in (
            (max_text_codepoints, "max_text_codepoints"),
            (max_matches, "max_matches"),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(name)
        self._max_text_codepoints = max_text_codepoints
        self._max_matches = max_matches

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def detect(self, value: str) -> PIIDetectionResult:
        """Return deterministic merged spans or a content-free failure.

        Args:
            value: Text to inspect under this detector's work limits.

        Returns:
            Merged spans when detection succeeds, otherwise a content-free
            unavailable result. Source text and matches are never retained.
        """

        try:
            if type(value) is not str or len(value) > self._max_text_codepoints:
                raise ValueError("work_limit")
            candidates: list[PIIRedactionSpan] = []
            for rule_id, category, pattern in _BUILTIN_PII_RULES:
                for match in pattern.finditer(value):
                    candidates.append(
                        PIIRedactionSpan(
                            match.start(),
                            match.end(),
                            category,
                            rule_id,
                        )
                    )
                    if len(candidates) > self._max_matches:
                        raise ValueError("work_limit")
            spans = merge_pii_spans(candidates)
        except Exception:  # noqa: BLE001 - failures must not retain source content
            return PIIDetectionResult(
                available=False,
                spans=(),
                omission_reason_code=PII_DETECTOR_UNAVAILABLE,
            )
        return PIIDetectionResult(
            available=True,
            spans=spans,
            omission_reason_code=None,
        )


def redact_pii_value(
    value: object,
    *,
    detector: BuiltInPIIDetector | None = None,
    max_nodes: int = DEFAULT_CREDENTIAL_MAX_NODES,
    max_depth: int = DEFAULT_CREDENTIAL_MAX_DEPTH,
    max_total_spans: int = DEFAULT_PII_MAX_MATCHES,
) -> PIIValueRedactionResult:
    """Mask all strings in a bounded JSON-like provider-only value.

    Structured paths use mapping ordinals instead of raw keys so a PII-bearing
    user-authored key can never be copied into mask metadata.

    Args:
        value: JSON-like provider-only value to mask without mutation.
        detector: Optional bounded detector; the built-in detector is the default.
        max_nodes: Maximum aggregate structured nodes inspected.
        max_depth: Maximum recursive container depth inspected.
        max_total_spans: Maximum aggregate ranges retained in metadata.

    Returns:
        The masked value and content-free field ranges, or a content-free
        unavailable result when any component cannot be processed safely.

    Raises:
        ValueError: If a supplied structural work limit is not positive.
    """

    active_detector = detector or BuiltInPIIDetector()
    for limit, name in (
        (max_nodes, "max_nodes"),
        (max_depth, "max_depth"),
        (max_total_spans, "max_total_spans"),
    ):
        if type(limit) is not int or limit <= 0:
            raise ValueError(name)
    field_redactions: list[PIIFieldRedaction] = []
    budget = [0]
    active: set[int] = set()

    def visit(item: object, *, path: str, depth: int) -> object:
        if depth > max_depth:
            raise ValueError("work_limit")
        budget[0] += 1
        if budget[0] > max_nodes:
            raise ValueError("work_limit")
        if item is None or type(item) in {bool, int}:
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("unsupported")
            return item
        if type(item) is str:
            detection = active_detector.detect(item)
            if not detection.available:
                raise ValueError("detector_unavailable")
            if len(field_redactions) + len(detection.spans) > max_total_spans:
                raise ValueError("work_limit")
            field_redactions.extend(
                PIIFieldRedaction(path, span) for span in detection.spans
            )
            return apply_pii_mask(item, detection.spans)
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in active:
                raise ValueError("recursive")
            active.add(identity)
            try:
                if any(type(key) is not str for key in item):
                    raise TypeError("unsupported")
                ordered = sorted(item.items(), key=lambda pair: pair[0])
                result: dict[str, object] = {}
                for ordinal, (key, child) in enumerate(ordered):
                    key_path = f"{path}/@{ordinal}#key"
                    key_detection = active_detector.detect(key)
                    if not key_detection.available:
                        raise ValueError("detector_unavailable")
                    if (
                        len(field_redactions)
                        + len(key_detection.spans)
                        > max_total_spans
                    ):
                        raise ValueError("work_limit")
                    field_redactions.extend(
                        PIIFieldRedaction(key_path, span)
                        for span in key_detection.spans
                    )
                    masked_key = apply_pii_mask(key, key_detection.spans)
                    if masked_key in result:
                        raise ValueError("masked key collision")
                    result[masked_key] = visit(
                        child,
                        path=f"{path}/@{ordinal}",
                        depth=depth + 1,
                    )
                return result
            finally:
                active.remove(identity)
        if isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            identity = id(item)
            if identity in active:
                raise ValueError("recursive")
            active.add(identity)
            try:
                return [
                    visit(child, path=f"{path}/{index}", depth=depth + 1)
                    for index, child in enumerate(item)
                ]
            finally:
                active.remove(identity)
        raise TypeError("unsupported")

    try:
        masked = visit(value, path="$", depth=0)
    except Exception:  # noqa: BLE001 - failures must not retain source content
        return PIIValueRedactionResult(
            available=False,
            value=None,
            field_redactions=(),
            omission_reason_code=PII_DETECTOR_UNAVAILABLE,
        )
    return PIIValueRedactionResult(
        available=True,
        value=masked,
        field_redactions=tuple(field_redactions),
        omission_reason_code=None,
    )


def apply_frozen_pii_masks(
    value: object,
    field_masks: Mapping[str, Sequence[PIIRedactionSpan]],
) -> object:
    """Apply persisted masks to a fresh projection without rerunning detection.

    Args:
        value: Fresh JSON-like projection of the original frozen source.
        field_masks: Content-free structured paths and immutable codepoint ranges.

    Returns:
        A new projection with every persisted mask applied.

    Raises:
        TypeError: If the source structure contains unsupported mapping keys.
        ValueError: If paths, ranges, key collisions, or the source shape do
            not exactly match the frozen mask metadata.
    """

    masks = {path: tuple(spans) for path, spans in field_masks.items()}
    if any(type(path) is not str or not path for path in masks):
        raise ValueError("field_path")
    used: set[str] = set()

    def visit(item: object, *, path: str) -> object:
        if type(item) is str:
            spans = masks.get(path)
            if spans is None:
                return item
            used.add(path)
            return apply_pii_mask(item, spans)
        if isinstance(item, Mapping):
            if any(type(key) is not str for key in item):
                raise TypeError("unsupported")
            ordered = sorted(item.items(), key=lambda pair: pair[0])
            result: dict[str, object] = {}
            for ordinal, (key, child) in enumerate(ordered):
                key_path = f"{path}/@{ordinal}#key"
                key_spans = masks.get(key_path)
                if key_spans is not None:
                    used.add(key_path)
                    masked_key = apply_pii_mask(key, key_spans)
                else:
                    masked_key = key
                if masked_key in result:
                    raise ValueError("masked key collision")
                result[masked_key] = visit(child, path=f"{path}/@{ordinal}")
            return result
        if isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            return [
                visit(child, path=f"{path}/{index}")
                for index, child in enumerate(item)
            ]
        return item

    projected = visit(value, path="$")
    if used != set(masks):
        raise ValueError("frozen_mask_source_mismatch")
    return projected


class CredentialSanitizer:
    """Remove recognized credentials without retaining findings or failures."""

    __slots__ = (
        "_known_credentials",
        "_max_depth",
        "_max_nodes",
        "_max_text_codepoints",
    )

    def __init__(
        self,
        *,
        known_credentials: tuple[str, ...] = (),
        max_nodes: int = DEFAULT_CREDENTIAL_MAX_NODES,
        max_depth: int = DEFAULT_CREDENTIAL_MAX_DEPTH,
        max_text_codepoints: int = DEFAULT_CREDENTIAL_MAX_TEXT_CODEPOINTS,
    ) -> None:
        """Create a bounded credential sanitizer.

        Args:
            known_credentials: Runtime credential values that must also be removed.
            max_nodes: Maximum aggregate container, key, and scalar nodes inspected.
            max_depth: Maximum recursive JSON-like value depth inspected.
            max_text_codepoints: Maximum code points inspected in any one string.

        Raises:
            ValueError: If a work limit is invalid or known values exceed the
                bounded detector budget.
        """

        for limit, name in (
            (max_nodes, "max_nodes"),
            (max_depth, "max_depth"),
            (max_text_codepoints, "max_text_codepoints"),
        ):
            if type(limit) is not int or limit <= 0:
                raise ValueError(name)
        filtered_credentials = tuple(
            value for value in known_credentials if isinstance(value, str) and value
        )
        if (
            len(filtered_credentials) > DEFAULT_CREDENTIAL_MAX_KNOWN_VALUES
            or any(len(value) > max_text_codepoints for value in filtered_credentials)
        ):
            raise ValueError("known_credentials")
        self._known_credentials = filtered_credentials
        self._max_nodes = max_nodes
        self._max_depth = max_depth
        self._max_text_codepoints = max_text_codepoints

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def sanitize(self, value: object) -> CredentialSanitizationResult:
        """Return a sanitized JSON-like value, failing closed without content."""

        try:
            sanitized, redacted = self._sanitize(
                value,
                active=set(),
                budget=[0],
                depth=0,
            )
        except Exception:  # noqa: BLE001 - failure details may contain credentials
            return CredentialSanitizationResult(
                available=False,
                value=None,
                omission_reason_code=CREDENTIAL_SANITIZER_UNAVAILABLE,
            )
        return CredentialSanitizationResult(
            available=True,
            value=sanitized,
            omission_reason_code=None,
            redacted=redacted,
        )

    def _sanitize(
        self,
        value: object,
        *,
        active: set[int],
        budget: list[int],
        depth: int,
    ) -> tuple[object, bool]:
        if depth > self._max_depth:
            raise ValueError("work_limit")
        budget[0] += 1
        if budget[0] > self._max_nodes:
            raise ValueError("work_limit")
        if value is None or type(value) in {bool, int}:
            return value, False
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError("unsupported")
            return value, False
        if type(value) is str:
            self._check_text(value)
            sanitized = self._sanitize_text(value)
            return sanitized, sanitized != value
        if type(value) is bytes:
            raise TypeError("unsupported")
        if isinstance(value, Mapping):
            identity = id(value)
            if identity in active:
                raise ValueError("recursive")
            active.add(identity)
            try:
                result: dict[str, object] = {}
                redacted = False
                for key, item in value.items():
                    if type(key) is not str:
                        raise TypeError("unsupported")
                    budget[0] += 1
                    if budget[0] > self._max_nodes:
                        raise ValueError("work_limit")
                    self._check_text(key)
                    if self._credential_key(key):
                        redacted = True
                        continue
                    if self._known_credential_key(key):
                        sanitized_key = _OMITTED_TEXT
                    else:
                        sanitized_key = self._sanitize_text(key, include_known=False)
                    redacted = redacted or sanitized_key != key
                    if sanitized_key in result:
                        raise ValueError("sanitized key collision")
                    sanitized_item, item_redacted = self._sanitize(
                        item,
                        active=active,
                        budget=budget,
                        depth=depth + 1,
                    )
                    redacted = redacted or item_redacted
                    result[sanitized_key] = sanitized_item
                return result, redacted
            finally:
                active.remove(identity)
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            identity = id(value)
            if identity in active:
                raise ValueError("recursive")
            active.add(identity)
            try:
                sequence_result: list[object] = []
                redacted = False
                for item in value:
                    sanitized_item, item_redacted = self._sanitize(
                        item,
                        active=active,
                        budget=budget,
                        depth=depth + 1,
                    )
                    sequence_result.append(sanitized_item)
                    redacted = redacted or item_redacted
                return sequence_result, redacted
            finally:
                active.remove(identity)
        raise TypeError("unsupported")

    def _check_text(self, value: str) -> None:
        if len(value) > self._max_text_codepoints:
            raise ValueError("work_limit")

    @staticmethod
    def _credential_key(key: str) -> bool:
        if "://" in key:
            return False
        normalized = re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")
        components = tuple(part for part in normalized.split("_") if part)
        for credential in _CREDENTIAL_KEYS:
            credential_components = tuple(credential.split("_"))
            width = len(credential_components)
            if any(
                components[index : index + width] == credential_components
                for index in range(len(components) - width + 1)
            ):
                return True
        return False

    def _known_credential_key(self, key: str) -> bool:
        return any(credential in key for credential in self._known_credentials)

    def _sanitize_text(self, value: str, *, include_known: bool = True) -> str:
        if _PRIVATE_KEY_TEXT.search(value):
            return _OMITTED_TEXT
        value = _URL_TEXT.sub(lambda match: self._sanitize_url(match.group()), value)
        if include_known:
            for credential in self._known_credentials:
                value = value.replace(credential, _OMITTED_TEXT)
        if _SECRET_TEXT.search(value):
            value = _SECRET_TEXT.sub(_OMITTED_TEXT, value)
        return sanitize_string(value).replace(REDACTION_MARKER, _OMITTED_TEXT)

    @staticmethod
    def _sanitize_url(value: str) -> str:
        try:
            parsed = urlsplit(value)
        except ValueError:
            parsed = SplitResult("", "", "", "", "")
        if parsed.scheme and parsed.netloc:
            hostname = parsed.hostname
            if hostname is None:
                return _OMITTED_TEXT
            host = f"[{hostname}]" if ":" in hostname else hostname
            try:
                port = parsed.port
            except ValueError:
                return _OMITTED_TEXT
            netloc = f"{host}:{port}" if port is not None else host
            value = urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
        return value
