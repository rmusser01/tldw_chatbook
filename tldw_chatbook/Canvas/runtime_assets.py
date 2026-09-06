"""Integrity-checked access to the packaged Canvas QuickJS runtime bundle."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from types import MappingProxyType
from typing import Any

RUNTIME_DISABLED_DIAGNOSTIC = (
    "Canvas scripting is disabled because packaged runtime verification failed."
)
RUNTIME_MANIFEST_BYTES = 256 * 1024
_JAVASCRIPT_BYTES = 8 * 1024 * 1024
_TRUSTED_JAVASCRIPT_BYTES = 512 * 1024
_NOTICE_BYTES = 256 * 1024
_LIBRARY_BYTES = 256 * 1024
_SAFE_RESOURCE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_OUTPUT_NAMES = frozenset(
    {
        "quickjs-runtime.js",
        "canvas_runtime_worker.js",
        "canvas_renderer.js",
        "THIRD_PARTY_LICENSES.txt",
    }
)


@dataclass(frozen=True)
class CanvasRuntimeAssets:
    """Verified runtime bytes, or a content-free reason Canvas scripting is disabled."""

    enabled: bool
    javascript: bytes | None
    worker_javascript: bytes | None
    renderer_javascript: bytes | None
    manifest: Mapping[str, Any] | None
    manifest_bytes: bytes | None
    diagnostic: str | None


@dataclass(frozen=True, slots=True)
class CanvasProfileRuntimeAssets:
    """Exact immutable manifest and executable bytes owned for one profile."""

    profile_id: str
    manifest_name: str
    manifest: Mapping[str, Any]
    manifest_bytes: bytes
    javascript: bytes
    worker_javascript: bytes
    renderer_javascript: bytes
    library_files: Mapping[str, bytes]


def _read_bounded(resource: Any, limit: int) -> bytes:
    with resource.open("rb") as handle:
        value = handle.read(limit + 1)
    if len(value) > limit:
        raise ValueError("resource exceeds its packaged byte limit")
    return value


def _strict_json(data: bytes) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    value = json.loads(data.decode("utf-8"), object_pairs_hook=object_pairs)
    if not isinstance(value, dict):
        raise TypeError("manifest must be an object")
    return value


def _valid_output_metadata(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {"bytes", "sha256"}:
        return False
    size = value.get("bytes")
    digest = value.get("sha256")
    return (
        isinstance(size, int)
        and not isinstance(size, bool)
        and size >= 0
        and isinstance(digest, str)
        and len(digest) == 64
        and digest == digest.lower()
        and all(character in "0123456789abcdef" for character in digest)
    )


def _freeze_json(value: Any) -> Any:
    """Return an immutable ownership copy of already-validated JSON data."""

    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _load_profile_verified(
    *,
    profile_id: str,
    manifest_name: str,
    manifest_sha256: str | None,
    library_inventory: Mapping[str, Any],
) -> CanvasProfileRuntimeAssets:
    static = files("tldw_chatbook.Canvas").joinpath("static")
    manifest_bytes = _read_bounded(
        static.joinpath(manifest_name), RUNTIME_MANIFEST_BYTES
    )
    if (
        manifest_sha256 is not None
        and hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256
    ):
        raise ValueError("Canvas runtime manifest integrity mismatch")
    manifest = _strict_json(manifest_bytes)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("runtime_profile") != profile_id
    ):
        raise ValueError("unsupported Canvas runtime manifest")
    if manifest.get("runtime_layout") != {
        "javascript": "quickjs-runtime.js",
        "renderer": "canvas_renderer.js",
        "worker": "canvas_runtime_worker.js",
        "wasm": "embedded",
        "wasm_fetch_required": False,
    }:
        raise ValueError("unsupported Canvas runtime layout")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != _OUTPUT_NAMES:
        raise ValueError("unexpected Canvas runtime output inventory")

    loaded: dict[str, bytes] = {}
    for name, limit in (
        ("quickjs-runtime.js", _JAVASCRIPT_BYTES),
        ("canvas_runtime_worker.js", _TRUSTED_JAVASCRIPT_BYTES),
        ("canvas_renderer.js", _TRUSTED_JAVASCRIPT_BYTES),
        ("THIRD_PARTY_LICENSES.txt", _NOTICE_BYTES),
    ):
        metadata = outputs.get(name)
        if not _valid_output_metadata(metadata):
            raise ValueError("invalid Canvas runtime output metadata")
        contents = _read_bounded(static.joinpath(name), limit)
        actual = {
            "bytes": len(contents),
            "sha256": hashlib.sha256(contents).hexdigest(),
        }
        if actual != metadata:
            raise ValueError("Canvas runtime output integrity mismatch")
        loaded[name] = contents
    if (
        not isinstance(library_inventory, Mapping)
        or set(library_inventory) != {"bytes", "files"}
        or type(library_inventory["bytes"]) is not int
        or not 0 <= library_inventory["bytes"] <= _LIBRARY_BYTES
        or not isinstance(library_inventory["files"], Mapping)
    ):
        raise ValueError("invalid Canvas runtime library inventory")
    library_files: dict[str, bytes] = {}
    for name, metadata in library_inventory["files"].items():
        if (
            type(name) is not str
            or _SAFE_RESOURCE_NAME.fullmatch(name) is None
            or not _valid_output_metadata(metadata)
        ):
            raise ValueError("invalid Canvas runtime library metadata")
        contents = _read_bounded(static.joinpath(name), _LIBRARY_BYTES)
        actual = {
            "bytes": len(contents),
            "sha256": hashlib.sha256(contents).hexdigest(),
        }
        if actual != metadata:
            raise ValueError("Canvas runtime library integrity mismatch")
        library_files[name] = contents
    if sum(map(len, library_files.values())) != library_inventory["bytes"]:
        raise ValueError("Canvas runtime library byte inventory mismatch")
    return CanvasProfileRuntimeAssets(
        profile_id=profile_id,
        manifest_name=manifest_name,
        manifest=_freeze_json(manifest),
        manifest_bytes=manifest_bytes,
        javascript=loaded["quickjs-runtime.js"],
        worker_javascript=loaded["canvas_runtime_worker.js"],
        renderer_javascript=loaded["canvas_renderer.js"],
        library_files=MappingProxyType(library_files),
    )


def load_canvas_profile_runtime_assets(
    *,
    profile_id: str,
    manifest_name: str,
    manifest_sha256: str,
    library_inventory: Mapping[str, Any],
) -> CanvasProfileRuntimeAssets:
    """Load and retain one exact profile's verified manifest and byte closure."""

    if type(profile_id) is not str or _SAFE_RESOURCE_NAME.fullmatch(profile_id) is None:
        raise ValueError("invalid Canvas runtime profile identity")
    if (
        type(manifest_name) is not str
        or _SAFE_RESOURCE_NAME.fullmatch(manifest_name) is None
        or not manifest_name.endswith(".json")
    ):
        raise ValueError("invalid Canvas runtime manifest filename")
    if type(manifest_sha256) is not str or _SHA256.fullmatch(manifest_sha256) is None:
        raise ValueError("invalid Canvas runtime manifest identity")
    return _load_profile_verified(
        profile_id=profile_id,
        manifest_name=manifest_name,
        manifest_sha256=manifest_sha256,
        library_inventory=library_inventory,
    )


def _load_verified() -> CanvasRuntimeAssets:
    owned = _load_profile_verified(
        profile_id="canvas-v1",
        manifest_name="runtime-manifest.json",
        manifest_sha256=None,
        library_inventory={"bytes": 0, "files": {}},
    )
    return CanvasRuntimeAssets(
        enabled=True,
        javascript=owned.javascript,
        worker_javascript=owned.worker_javascript,
        renderer_javascript=owned.renderer_javascript,
        manifest=owned.manifest,
        manifest_bytes=owned.manifest_bytes,
        diagnostic=None,
    )


def load_canvas_runtime_assets() -> CanvasRuntimeAssets:
    """Load verified packaged assets, failing closed without exposing asset content."""

    try:
        return _load_verified()
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return CanvasRuntimeAssets(
            enabled=False,
            javascript=None,
            worker_javascript=None,
            renderer_javascript=None,
            manifest=None,
            manifest_bytes=None,
            diagnostic=RUNTIME_DISABLED_DIAGNOSTIC,
        )


__all__ = [
    "RUNTIME_DISABLED_DIAGNOSTIC",
    "RUNTIME_MANIFEST_BYTES",
    "CanvasProfileRuntimeAssets",
    "CanvasRuntimeAssets",
    "load_canvas_profile_runtime_assets",
    "load_canvas_runtime_assets",
]
