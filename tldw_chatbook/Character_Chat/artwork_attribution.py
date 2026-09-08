"""Bounded, inert public artwork credits shared by editors and Actor Packs.

This allowlist never projects arbitrary profile source context or fetches URLs.
Native visual manifests and rendering authority are deliberately unchanged.
"""

from __future__ import annotations

import ipaddress
import json
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

ARTWORK_NAMESPACE = "tldw/artwork"
ARTWORK_MEMBER = "shared-visual-identity/attribution.json"
ARTWORK_FEATURE = "visual-artwork-attribution/v1"
MAX_ARTWORK_BYTES = 512 * 1024
MAX_NOTICE_BYTES = 64 * 1024
MAX_ARTWORK_ASSETS = 128
_FIELDS = {"version", "creator", "license", "source_url", "notices"}
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_EXPRESSION = re.compile(r"(?:[a-z][a-z0-9_]*|custom:[a-z0-9_]+)\Z")


def _invalid() -> ValueError:
    return ValueError("artwork_attribution_invalid")


def _text(value: object, limit: int, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if type(value) is not str or len(value) > limit:
        raise _invalid()
    if any(
        (ord(char) < 32 and char not in "\n\r\t") or 127 <= ord(char) <= 159
        for char in value
    ):
        raise _invalid()
    try:
        if len(value.encode("utf-8")) > limit:
            raise _invalid()
    except UnicodeError:
        raise _invalid() from None
    return value


def _url(value: object) -> str | None:
    text = _text(value, 2048, nullable=True)
    if text is None:
        return None
    if "\\" in text or any(char.isspace() for char in text):
        raise _invalid()
    try:
        parsed = urlsplit(text)
        host = parsed.hostname
        if (
            parsed.scheme != "https"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port is not None
            or ":" in parsed.netloc
            or parsed.query
            or parsed.fragment
            or "?" in text
            or "#" in text
            or not host
            or "." not in host
            or host.endswith(
                (".", ".local", ".localhost", ".internal", ".lan", ".home")
            )
            or not re.fullmatch(r"[a-z0-9.-]+", host)
            or not re.fullmatch(r"[a-z][a-z0-9-]*", host.rsplit(".", 1)[-1])
            or any(
                not label or label.startswith("-") or label.endswith("-")
                for label in host.split(".")
            )
        ):
            raise _invalid()
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            raise _invalid()
    except (UnicodeError, ValueError):
        raise _invalid() from None
    return text


def _record(value: object, expected_sha256: str | None) -> dict[str, Any]:
    fields = _FIELDS | ({"output_sha256"} if expected_sha256 is not None else set())
    if type(value) is not dict or set(value) != fields:
        raise _invalid()
    if type(value["version"]) is not int or value["version"] != 1:
        raise _invalid()
    result = {
        "version": 1,
        "creator": _text(value["creator"], 512, nullable=True),
        "license": _text(value["license"], 4096, nullable=True),
        "source_url": _url(value["source_url"]),
        "notices": _text(value["notices"], MAX_NOTICE_BYTES),
    }
    if expected_sha256 is not None:
        if type(expected_sha256) is not str or not _DIGEST.fullmatch(expected_sha256):
            raise _invalid()
        if value["output_sha256"] != expected_sha256:
            raise _invalid()
        result["output_sha256"] = expected_sha256
    return result


def artwork_context(
    context: object, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    """Project only validated public attribution from a local source context.

    Args:
        context: Local context mapping, including arbitrary private bookkeeping.
        expected_sha256: Required image digest for an asset; omit for pack credits.

    Raises:
        ValueError: A present known record is invalid or belongs to different bytes.
    """
    if not isinstance(context, Mapping):
        raise _invalid()
    if ARTWORK_NAMESPACE not in context:
        return {}
    return {ARTWORK_NAMESPACE: _record(context[ARTWORK_NAMESPACE], expected_sha256)}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def encode_artwork_attribution(
    pack_context: Mapping[str, Any],
    assets: Mapping[str, tuple[str, Mapping[str, Any]]],
) -> bytes | None:
    """Build a canonical public carrier, or omit it when no attribution exists."""
    if len(assets) > MAX_ARTWORK_ASSETS:
        raise _invalid()
    pack = artwork_context(pack_context).get(ARTWORK_NAMESPACE)
    records = {}
    size = len(_canonical(pack))
    for key, (digest, context) in assets.items():
        if type(key) is not str or len(key) > 128 or not _EXPRESSION.fullmatch(key):
            raise _invalid()
        record = artwork_context(context, expected_sha256=digest).get(ARTWORK_NAMESPACE)
        if record is not None:
            size += len(_canonical(record)) + len(key) + 4
            if size > MAX_ARTWORK_BYTES:
                raise _invalid()
            records[key] = record
    if pack is None and not records:
        return None
    encoded = _canonical({"version": 1, "pack": pack, "assets": records})
    if len(encoded) > MAX_ARTWORK_BYTES:
        raise _invalid()
    return encoded


def decode_artwork_attribution(
    data: bytes, asset_hashes: Mapping[str, str]
) -> dict[str, Any]:
    """Read bounded canonical attribution tied to the section's image hashes."""
    if type(data) is not bytes or len(data) > MAX_ARTWORK_BYTES:
        raise _invalid()
    try:
        value = json.loads(data)
        if type(value) is not dict or set(value) != {"version", "pack", "assets"}:
            raise _invalid()
        if type(value["version"]) is not int or value["version"] != 1:
            raise _invalid()
        assets = value["assets"]
        if type(assets) is not dict or not assets.keys() <= asset_hashes.keys():
            raise _invalid()
        encoded = encode_artwork_attribution(
            {} if value["pack"] is None else {ARTWORK_NAMESPACE: value["pack"]},
            {
                key: (asset_hashes[key], {ARTWORK_NAMESPACE: record})
                for key, record in assets.items()
            },
        )
        if encoded != data:
            raise _invalid()
        return value
    except (KeyError, TypeError, ValueError, UnicodeError, RecursionError):
        raise _invalid() from None
