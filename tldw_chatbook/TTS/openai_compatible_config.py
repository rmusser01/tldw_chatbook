from __future__ import annotations

import hashlib
import ipaddress
import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import SplitResult, quote, unquote_to_bytes, urlsplit, urlunsplit

import idna


class OpenAIAuthenticationMode(StrEnum):
    API_KEY = "api_key"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class OpenAICompatibleEndpoint:
    speech_url: str
    origin: str
    catalog_url: str | None
    official: bool


_KNOWN_PATHS = {
    "": ("/v1/audio/speech", "/v1/models"),
    "/": ("/v1/audio/speech", "/v1/models"),
    "/v1": ("/v1/audio/speech", "/v1/models"),
    "/v1/models": ("/v1/audio/speech", "/v1/models"),
    "/v1/chat/completions": ("/v1/audio/speech", "/v1/models"),
    "/chat/completions": ("/audio/speech", "/models"),
    "/v1/audio/speech": ("/v1/audio/speech", "/v1/models"),
}
_SCHEME_PATTERN = re.compile(r"(?i)https?://")
_NUMERIC_IPV4_COMPONENT = re.compile(r"(?i)(?:[0-9]+|0x[0-9a-f]+)\Z")
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_PATH_SAFE_CHARACTERS = "/:@!$&'()*+,;=-._~%"
_UNICODE_DOTS = str.maketrans({"\u3002": ".", "\uff0e": ".", "\uff61": "."})


def _invalid_endpoint() -> ValueError:
    return ValueError(
        "OpenAI-compatible endpoint must be one unambiguous absolute HTTP(S) URL"
    )


def _canonical_hostname(parsed: SplitResult) -> tuple[str, bool]:
    raw_hostname = parsed.hostname
    if not raw_hostname:
        raise _invalid_endpoint()

    hostname = raw_hostname.translate(_UNICODE_DOTS)
    if parsed.netloc.startswith("["):
        if "%" in hostname:
            raise _invalid_endpoint()
        try:
            return ipaddress.IPv6Address(hostname).compressed, True
        except ipaddress.AddressValueError as error:
            raise _invalid_endpoint() from error

    hostname = hostname.rstrip(".")
    if not hostname:
        raise _invalid_endpoint()

    numeric_labels = hostname.split(".")
    if len(numeric_labels) == 4 and all(
        label.isascii() and label.isdecimal() for label in numeric_labels
    ):
        try:
            return str(ipaddress.IPv4Address(hostname)), False
        except ipaddress.AddressValueError as error:
            raise _invalid_endpoint() from error

    if all(_NUMERIC_IPV4_COMPONENT.fullmatch(label) for label in numeric_labels):
        raise _invalid_endpoint()

    try:
        canonical = idna.encode(
            hostname.lower(),
            uts46=True,
            std3_rules=True,
        ).decode("ascii")
    except (UnicodeError, idna.IDNAError) as error:
        raise _invalid_endpoint() from error
    return canonical, False


def _canonical_path(path: str) -> str:
    if "\\" in path:
        raise _invalid_endpoint()

    canonical_parts: list[str] = []
    index = 0
    while index < len(path):
        character = path[index]
        if character != "%":
            canonical_parts.append(character)
            index += 1
            continue
        if (
            index + 2 >= len(path)
            or path[index + 1] not in _HEX_DIGITS
            or path[index + 2] not in _HEX_DIGITS
        ):
            raise _invalid_endpoint()
        canonical_parts.append("%" + path[index + 1 : index + 3].upper())
        index += 3

    canonical = "".join(canonical_parts)
    for component in canonical.split("/"):
        try:
            decoded = unquote_to_bytes(component).decode("utf-8")
        except UnicodeDecodeError as error:
            raise _invalid_endpoint() from error
        if decoded in {".", ".."} or any(
            character in {"/", "\\"} or unicodedata.category(character) == "Cc"
            for character in decoded
        ):
            raise _invalid_endpoint()

    return quote(canonical, safe=_PATH_SAFE_CHARACTERS)


def _normalized_authority(parsed: SplitResult) -> tuple[str, str, int | None]:
    scheme = parsed.scheme.lower()
    try:
        port = parsed.port
    except ValueError as error:
        raise _invalid_endpoint() from error
    if port == 0:
        raise _invalid_endpoint()

    hostname, ipv6 = _canonical_hostname(parsed)
    default_port = 80 if scheme == "http" else 443
    normalized_port = None if port in (None, default_port) else port
    rendered_host = f"[{hostname}]" if ipv6 else hostname
    authority = (
        rendered_host
        if normalized_port is None
        else f"{rendered_host}:{normalized_port}"
    )
    return authority, hostname, normalized_port


def normalize_openai_compatible_endpoint(raw: str) -> OpenAICompatibleEndpoint:
    if not isinstance(raw, str) or not raw:
        raise _invalid_endpoint()
    if raw != raw.strip() or any(
        character.isspace() or unicodedata.category(character) == "Cc"
        for character in raw
    ):
        raise _invalid_endpoint()
    if "?" in raw or "#" in raw:
        raise _invalid_endpoint()
    scheme_matches = list(_SCHEME_PATTERN.finditer(raw))
    if len(scheme_matches) != 1 or scheme_matches[0].start() != 0:
        raise _invalid_endpoint()

    try:
        parsed = urlsplit(raw)
        if (
            parsed.scheme.lower() not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or "//" in parsed.path
        ):
            raise _invalid_endpoint()
        path = _canonical_path(parsed.path)
        authority, hostname, port = _normalized_authority(parsed)
    except ValueError as error:
        if str(error).startswith("OpenAI-compatible endpoint"):
            raise
        raise _invalid_endpoint() from error

    scheme = parsed.scheme.lower()
    origin = urlunsplit((scheme, authority, "", "", ""))
    path_key = (path[:-1] if path != "/" and path.endswith("/") else path).lower()
    mapped = _KNOWN_PATHS.get(path_key)
    if mapped is None:
        speech_path = path
        catalog_url = None
    else:
        speech_path, catalog_path = mapped
        catalog_url = urlunsplit((scheme, authority, catalog_path, "", ""))

    official = scheme == "https" and hostname == "api.openai.com" and port is None
    return OpenAICompatibleEndpoint(
        speech_url=urlunsplit((scheme, authority, speech_path, "", "")),
        origin=origin,
        catalog_url=catalog_url,
        official=official,
    )


def normalize_openai_authentication_mode(
    raw: object,
    *,
    endpoint: OpenAICompatibleEndpoint,
) -> OpenAIAuthenticationMode:
    mode = (
        OpenAIAuthenticationMode.NONE
        if raw == OpenAIAuthenticationMode.NONE.value
        else OpenAIAuthenticationMode.API_KEY
    )
    if endpoint.official and mode is OpenAIAuthenticationMode.NONE:
        raise ValueError("Official OpenAI requires an API key")
    return mode


def is_loopback_openai_compatible_endpoint(
    endpoint: OpenAICompatibleEndpoint,
) -> bool:
    """Return whether a normalized endpoint targets localhost or a loopback IP."""

    if type(endpoint) is not OpenAICompatibleEndpoint:
        return False
    hostname = urlsplit(endpoint.origin).hostname
    if hostname is None:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def openai_destination_fingerprint(
    provider_id: object,
    endpoint: OpenAICompatibleEndpoint,
) -> str:
    provider = str(provider_id).strip().lower()
    payload = f"{provider}\0{endpoint.origin}".encode()
    return hashlib.sha256(payload).hexdigest()
