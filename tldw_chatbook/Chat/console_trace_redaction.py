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


@dataclass(frozen=True, slots=True)
class CredentialSanitizationResult:
    """Sanitized value or a content-free unavailable marker."""

    available: bool
    value: object | None = field(repr=False)
    omission_reason_code: str | None
    redacted: bool = False
    detector_version: str = CREDENTIAL_FILTER_VERSION


class CredentialSanitizer:
    """Remove recognized credentials without retaining findings or failures."""

    __slots__ = ("_known_credentials",)

    def __init__(self, *, known_credentials: tuple[str, ...] = ()) -> None:
        self._known_credentials = tuple(
            value for value in known_credentials if isinstance(value, str) and value
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def sanitize(self, value: object) -> CredentialSanitizationResult:
        """Return a sanitized JSON-like value, failing closed without content."""

        try:
            sanitized, redacted = self._sanitize(value, active=set())
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

    def _sanitize(self, value: object, *, active: set[int]) -> tuple[object, bool]:
        if value is None or type(value) in {bool, int}:
            return value, False
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError("unsupported")
            return value, False
        if type(value) is str:
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
                    sanitized_item, item_redacted = self._sanitize(item, active=active)
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
                    sanitized_item, item_redacted = self._sanitize(item, active=active)
                    sequence_result.append(sanitized_item)
                    redacted = redacted or item_redacted
                return sequence_result, redacted
            finally:
                active.remove(identity)
        raise TypeError("unsupported")

    @staticmethod
    def _credential_key(key: str) -> bool:
        normalized = re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")
        return any(
            normalized == credential or normalized.endswith(f"_{credential}")
            for credential in _CREDENTIAL_KEYS
        )

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
