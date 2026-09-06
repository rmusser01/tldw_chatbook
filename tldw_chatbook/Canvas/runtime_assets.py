"""Integrity-checked access to the packaged Canvas QuickJS runtime bundle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.resources import files
import json
from typing import Any, Mapping


RUNTIME_DISABLED_DIAGNOSTIC = (
    "Canvas scripting is disabled because packaged runtime verification failed."
)
_MANIFEST_BYTES = 256 * 1024
_JAVASCRIPT_BYTES = 8 * 1024 * 1024
_TRUSTED_JAVASCRIPT_BYTES = 512 * 1024
_NOTICE_BYTES = 256 * 1024
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
    diagnostic: str | None


def _read_bounded(resource: Any, limit: int) -> bytes:
    with resource.open("rb") as handle:
        value = handle.read(limit + 1)
    if len(value) > limit:
        raise ValueError("resource exceeds its packaged byte limit")
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


def _load_verified() -> CanvasRuntimeAssets:
    static = files("tldw_chatbook.Canvas").joinpath("static")
    manifest_bytes = _read_bounded(
        static.joinpath("runtime-manifest.json"), _MANIFEST_BYTES
    )
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("runtime_profile") != "canvas-v1"
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
    return CanvasRuntimeAssets(
        enabled=True,
        javascript=loaded["quickjs-runtime.js"],
        worker_javascript=loaded["canvas_runtime_worker.js"],
        renderer_javascript=loaded["canvas_renderer.js"],
        manifest=manifest,
        diagnostic=None,
    )


def load_canvas_runtime_assets() -> CanvasRuntimeAssets:
    """Load verified packaged assets, failing closed without exposing asset content."""

    try:
        return _load_verified()
    except Exception:
        return CanvasRuntimeAssets(
            enabled=False,
            javascript=None,
            worker_javascript=None,
            renderer_javascript=None,
            manifest=None,
            diagnostic=RUNTIME_DISABLED_DIAGNOSTIC,
        )


__all__ = [
    "CanvasRuntimeAssets",
    "RUNTIME_DISABLED_DIAGNOSTIC",
    "load_canvas_runtime_assets",
]
