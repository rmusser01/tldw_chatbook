"""Dependency-free QwenCloud base URL normalization."""

from __future__ import annotations

import re
from urllib.parse import unquote, urlsplit


DEFAULT_QWENCLOUD_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
_MAX_URL_LENGTH = 2000
_MAX_PATH_DECODE_PASSES = 3
_ENDPOINT_TAILS = (("models",), ("responses",), ("chat", "completions"))
_REQUEST_ENDPOINT_TAILS = _ENDPOINT_TAILS[1:]
_PERCENT_ESCAPE_RE = re.compile(r"%[0-9A-Fa-f]{2}")
_ENCODED_PATH_SEPARATOR_RE = re.compile(r"%(?:2[fF]|5[cC])")


class QwenCloudBaseURLValidationError(ValueError):
    """Raised when a QwenCloud base URL is unsafe or malformed."""


def _invalid(message: str) -> QwenCloudBaseURLValidationError:
    return QwenCloudBaseURLValidationError(message)


def _has_unsafe_endpoint_tail_structure(path: str) -> bool:
    segments = tuple(segment for segment in path.strip("/").split("/") if segment)
    request_tails = [
        (tail, index + len(tail))
        for tail in _REQUEST_ENDPOINT_TAILS
        for index in range(len(segments) - len(tail) + 1)
        if segments[index : index + len(tail)] == tail
    ]
    if len(request_tails) > 1 or any(
        end != len(segments) for _tail, end in request_tails
    ):
        return True
    return any(
        segments[-len(first + second) :] == first + second
        for first in _ENDPOINT_TAILS
        for second in _ENDPOINT_TAILS
    )


def _reserved_endpoint_markers(path: str) -> frozenset[tuple[int, str]]:
    segments = tuple(segment.lower() for segment in path.strip("/").split("/"))
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


def _validate_percent_encoded_path(path: str) -> None:
    validation_path = path
    markers = _reserved_endpoint_markers(validation_path)
    for _pass in range(_MAX_PATH_DECODE_PASSES):
        if _ENCODED_PATH_SEPARATOR_RE.search(validation_path):
            raise _invalid("QwenCloud API base URL path is malformed.")
        try:
            decoded_path = unquote(validation_path, errors="strict")
        except UnicodeDecodeError as exc:
            raise _invalid("QwenCloud API base URL path is malformed.") from exc
        if decoded_path == validation_path:
            break
        decoded_markers = _reserved_endpoint_markers(decoded_path)
        if (
            any(
                ord(character) < 32 or ord(character) == 127
                for character in decoded_path
            )
            or any(segment in {".", ".."} for segment in decoded_path.split("/"))
            or _has_unsafe_endpoint_tail_structure(decoded_path)
            or decoded_markers - markers
        ):
            raise _invalid("QwenCloud API base URL path is malformed.")
        validation_path = decoded_path
        markers = decoded_markers
    if _PERCENT_ESCAPE_RE.search(validation_path) is not None:
        raise _invalid("QwenCloud API base URL path is malformed.")


def normalize_qwencloud_base_url(api_base_url: str | None) -> str:
    """Return one validated QwenCloud base URL without a request suffix.

    Args:
        api_base_url: Configured base URL or a pasted request endpoint. ``None``
            selects the QwenCloud default.

    Returns:
        The validated base URL with a terminal lowercase request path removed.

    Raises:
        QwenCloudBaseURLValidationError: If the URL is malformed or unsafe.
    """
    candidate = DEFAULT_QWENCLOUD_BASE_URL if api_base_url is None else api_base_url
    if not isinstance(candidate, str) or not candidate.strip():
        raise _invalid("QwenCloud API base URL is required.")
    if len(candidate) > _MAX_URL_LENGTH:
        raise _invalid("QwenCloud API base URL is malformed.")
    candidate = candidate.strip().rstrip("/")
    if (
        any(character.isspace() for character in candidate)
        or any(ord(character) < 32 or ord(character) == 127 for character in candidate)
        or "?" in candidate
        or "#" in candidate
    ):
        raise _invalid("QwenCloud API base URL is malformed.")

    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError as exc:
        raise _invalid("QwenCloud API base URL is malformed.") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise _invalid("QwenCloud API base URL must be an absolute HTTP(S) URL.")
    if parsed.username is not None or parsed.password is not None:
        raise _invalid("QwenCloud API base URL must not contain credentials.")
    if any(character in parsed.netloc for character in '\\%|^{}<>"`'):
        raise _invalid("QwenCloud API base URL authority is malformed.")
    if parsed.query or parsed.fragment:
        raise _invalid("QwenCloud API base URL must not contain a query or fragment.")
    if parsed.netloc.endswith(":") or (port is not None and not 0 < port < 65536):
        raise _invalid("QwenCloud API base URL is malformed.")
    if (
        "\\" in parsed.path
        or "//" in parsed.path
        or re.search(r"%(?![0-9A-Fa-f]{2})", parsed.path) is not None
        or any(character.isspace() for character in parsed.path)
        or any(segment in {".", ".."} for segment in parsed.path.split("/"))
        or _has_unsafe_endpoint_tail_structure(parsed.path)
    ):
        raise _invalid("QwenCloud API base URL path is malformed.")
    _validate_percent_encoded_path(parsed.path)

    path = parsed.path.rstrip("/")
    if path.endswith("/models"):
        raise _invalid("QwenCloud API base URL must not use the models endpoint.")
    for suffix in ("/chat/completions", "/responses"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return f"{parsed.scheme.lower()}://{parsed.netloc}{path}".rstrip("/")
