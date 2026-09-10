"""Strict public Petdex manifest adapter; no CLI or authenticated API required."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import PurePosixPath
from urllib.parse import urljoin, urlsplit

from .network import PetdexNetworkError, fetch_bytes, validate_url
from .sources import PetdexSource, _declared_image, read_metadata, source_from_bytes

MAX_REGISTRY_BYTES = 10 * 1024 * 1024
MAX_METADATA_BYTES = 2 * 1024 * 1024
MAX_IMAGE_BYTES = 25 * 1024 * 1024
_COMPACT_FIELDS = (
    "slug",
    "displayName",
    "kind",
    "submittedBy",
    "spritesheet",
    "petJson",
    "zip",
    "spriteVersionNumber",
)
_SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*", re.ASCII)
_BASE = "https://petdex.dev"


def _string(value: object, name: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise ValueError(f"Invalid Petdex manifest {name}")
    return value


def _slug(value: object) -> str:
    if not isinstance(value, str) or len(value) > 128 or not _SLUG.fullmatch(value):
        raise ValueError("Invalid Petdex slug")
    return value


def _asset_url(value: object, base: str | None = None) -> str:
    raw = _string(value, "asset URL")
    if any(ord(char) <= 32 or ord(char) >= 127 for char in raw) or "\\" in raw:
        raise ValueError("Invalid Petdex asset URL")
    url = urljoin(base.rstrip("/") + "/", raw) if base else raw
    parsed = validate_url(url)
    if parsed.hostname != "assets.petdex.dev":
        raise ValueError("Untrusted Petdex asset host")
    return url


def _invalid_number(value: str) -> None:
    raise ValueError("Invalid Petdex JSON number")


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Petdex JSON has duplicate keys")
        result[key] = value
    return result


def parse_manifest(payload: bytes) -> list[dict]:
    """Normalize public object or compact-v2 entries; reject all ambiguity.

    Returns:
        Fresh upstream-style entries plus a canonical public ``sourceUrl``.
    """
    if not isinstance(payload, bytes) or len(payload) > MAX_REGISTRY_BYTES:
        raise ValueError("Petdex registry size limit")
    try:
        data = json.loads(
            payload, object_pairs_hook=_unique_object, parse_constant=_invalid_number
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise ValueError("Invalid Petdex registry JSON") from None
    if not isinstance(data, dict) or not isinstance(data.get("pets"), list):
        raise ValueError("Invalid Petdex manifest")  # noqa: TRY004 - malformed JSON is a value error
    version = data.get("v")
    if "v" in data and (type(version) is not int or version not in (1, 2)):
        raise ValueError("Unsupported Petdex registry version")
    compact = version == 2
    if compact and data.get("fields") != list(_COMPACT_FIELDS):
        raise ValueError("Invalid Petdex compact fields")
    base = _asset_url(data.get("assetBase")) if compact else None
    if "total" in data and (
        type(data["total"]) is not int or data["total"] != len(data["pets"])
    ):
        raise ValueError("Petdex manifest total mismatch")
    entries = []
    seen = set()
    for raw in data["pets"]:
        if compact:
            if not isinstance(raw, list) or len(raw) != len(_COMPACT_FIELDS):
                raise ValueError("Invalid Petdex compact entry")
            raw = dict(zip(_COMPACT_FIELDS, raw, strict=True))
            raw = {
                **raw,
                "spritesheetUrl": raw["spritesheet"],
                "petJsonUrl": raw["petJson"],
                "zipUrl": raw["zip"],
            }
        if not isinstance(raw, dict):
            raise ValueError("Invalid Petdex manifest entry")  # noqa: TRY004 - malformed JSON is a value error
        slug = _slug(raw.get("slug"))
        if slug in seen:
            raise ValueError("Petdex manifest has duplicate slugs")
        seen.add(slug)
        sprite_version = raw.get("spriteVersionNumber", 1)
        if type(sprite_version) is not int or sprite_version not in (1, 2):
            raise ValueError("Unsupported Petdex sprite version")
        if "submittedBy" not in raw or "zipUrl" not in raw:
            raise ValueError("Incomplete Petdex manifest entry")
        entry = {
            "slug": slug,
            "displayName": _string(raw.get("displayName"), "display name"),
            "kind": _string(raw.get("kind"), "kind"),
            "submittedBy": _string(raw.get("submittedBy"), "creator", nullable=True),
            "spritesheetUrl": _asset_url(raw.get("spritesheetUrl"), base),
            "petJsonUrl": _asset_url(raw.get("petJsonUrl"), base),
            "zipUrl": _asset_url(raw["zipUrl"], base)
            if raw["zipUrl"] is not None
            else None,
            "spriteVersionNumber": sprite_version,
            "sourceUrl": f"{_BASE}/pets/{slug}",
        }
        # Public legacy payloads may carry explicit artwork terms. Preserve only
        # declared strings; the repository's software license implies nothing.
        for key in ("creator", "license", "terms", "notices"):
            if key in raw:
                entry[key] = _string(raw[key], key)
        entries.append(entry)
    return entries


def fetch_petdex_source(
    value: str, *, cancel_requested: Callable[[], bool] = lambda: False
) -> PetdexSource:
    """Resolve an exact public slug/page, then acquire its bounded immutable bytes."""
    if isinstance(value, str) and value.startswith("https://"):
        parsed = validate_url(value)
        if parsed.hostname != "petdex.dev" or parsed.query:
            raise ValueError("Invalid Petdex page URL")
        match = re.fullmatch(r"/pets/([^/]+)/?", parsed.path)
        if not match:
            raise ValueError("Invalid Petdex page URL")
        value = match.group(1)
    slug = _slug(value)

    def check() -> None:
        if cancel_requested():
            raise PetdexNetworkError("cancelled")

    def fetch(url: str, limit: int) -> bytes:
        check()
        result = fetch_bytes(url, max_bytes=limit, cancel_requested=cancel_requested)
        check()
        return result

    try:
        payload = fetch(f"{_BASE}/api/manifest/v2", MAX_REGISTRY_BYTES)
    except PetdexNetworkError as exc:
        if exc.status_code not in (404, 405, 410, 501):
            raise
        payload = fetch(f"{_BASE}/api/manifest", MAX_REGISTRY_BYTES)
    entries = parse_manifest(payload)
    entry = next((entry for entry in entries if entry["slug"] == slug), None)
    if entry is None:
        raise ValueError("Petdex pet not found")
    metadata = fetch(entry["petJsonUrl"], MAX_METADATA_BYTES)
    # A CDN object basename need not equal the metadata's package-local filename.
    # This declaration names the pinned bytes only; it never selects a fetch URL.
    image_name = (
        _declared_image(read_metadata(metadata))
        or PurePosixPath(urlsplit(entry["spritesheetUrl"]).path).name
    )
    if not image_name or "%" in image_name:
        raise ValueError("Invalid Petdex sprite filename")
    image = fetch(entry["spritesheetUrl"], MAX_IMAGE_BYTES)
    source = source_from_bytes(
        metadata,
        image,
        image_name,
        registry_entry=dict(entry),
        notices="",
        guard=lambda: True,
    )
    check()
    return source
