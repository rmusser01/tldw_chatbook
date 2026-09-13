"""Bounded public artwork metadata, adapted from the reviewed Buddy import work."""

from __future__ import annotations

import ipaddress
import json
import re
from typing import Any
from urllib.parse import urlsplit

MAX_ARTWORK_BYTES = 512 * 1024
MAX_NOTICE_BYTES = 64 * 1024
_FIELDS = {"version", "creator", "license", "source_url", "notices"}
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


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


def encode_native_artwork(record: object) -> str:
    return json.dumps(
        _record(record, None),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def decode_native_artwork(value: object) -> dict[str, Any]:
    if type(value) is not str or len(value.encode("utf-8")) > MAX_ARTWORK_BYTES:
        raise _invalid()
    record = json.loads(value)
    if encode_native_artwork(record) != value:
        raise _invalid()
    return record


def artwork_from_pack(pack: dict) -> dict[str, Any]:
    context = pack.get("source_context", {})
    if not isinstance(context, dict):
        raise _invalid()
    if "artwork" in context:
        return decode_native_artwork(context["artwork"])
    for source in (context, pack):
        if "tldw/artwork" in source:
            return _record(source["tldw/artwork"], None)
        if "artwork" in source:
            value = source["artwork"]
            return (
                decode_native_artwork(value)
                if isinstance(value, str)
                else _record(value, None)
            )
    return _record(
        {
            "version": 1,
            "creator": pack.get("creator"),
            "license": pack.get("license", context.get("license")),
            "source_url": pack.get("source_url"),
            "notices": pack.get("notices", ""),
        },
        None,
    )
