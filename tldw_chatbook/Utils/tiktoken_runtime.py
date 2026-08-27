"""Install Chatbook's immutable, offline tiktoken table reader."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any

_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "tiktoken_cache"
_MANIFEST_PATH = _ASSET_DIR / "manifest.json"
_OVERRIDE_KEYS = ("TIKTOKEN_CACHE_DIR", "DATA_GYM_CACHE_DIR")


class BundledTiktokenAssetError(RuntimeError):
    """A requested tiktoken table is absent from or invalid in the bundle."""


@lru_cache(maxsize=1)
def _manifest_by_url() -> dict[str, dict[str, Any]]:
    """Load the reviewed asset manifest once, indexed by source URL."""
    try:
        manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
        return {entry["url"]: entry for entry in manifest["files"]}
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise BundledTiktokenAssetError(
            f"Unable to load bundled tiktoken manifest {_MANIFEST_PATH}"
        ) from error


def _read_bundled_file(blobpath: str, expected_hash: str | None = None) -> bytes:
    """Read one manifest-approved table without fetching or mutating files."""
    try:
        entry = _manifest_by_url()[blobpath]
    except BundledTiktokenAssetError:
        raise
    except KeyError as error:
        raise BundledTiktokenAssetError(
            f"Tiktoken URL is not in the manifest: {blobpath}"
        ) from error

    try:
        cache_key = hashlib.sha1(blobpath.encode()).hexdigest()  # nosec B324
        manifest_key = entry["cache_key"]
        manifest_hash = entry["sha256"]
    except (AttributeError, KeyError, TypeError, UnicodeError) as error:
        raise BundledTiktokenAssetError(
            f"Invalid bundled tiktoken manifest entry for {blobpath}"
        ) from error

    if manifest_key != cache_key:
        raise BundledTiktokenAssetError(
            f"Bundled tiktoken cache key mismatch for {blobpath}"
        )
    if expected_hash != manifest_hash:
        raise BundledTiktokenAssetError(
            f"Tiktoken expected hash does not match the manifest for {blobpath}"
        )

    try:
        data = (_ASSET_DIR / cache_key).read_bytes()
    except OSError as error:
        raise BundledTiktokenAssetError(
            f"Unable to read bundled tiktoken asset {cache_key}"
        ) from error
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise BundledTiktokenAssetError(
            f"Bundled tiktoken asset hash mismatch for {cache_key}"
        )
    return data


def install_tiktoken_runtime() -> None:
    """Select the bundle unless the caller supplied an upstream cache override."""
    if any(key in os.environ for key in _OVERRIDE_KEYS):
        return

    try:
        import tiktoken.load
    except ImportError:
        return

    parameters = inspect.signature(tiktoken.load.read_file_cached).parameters
    positional = inspect.Parameter.POSITIONAL_OR_KEYWORD
    if (
        tuple(parameters) != ("blobpath", "expected_hash")
        or parameters["blobpath"].kind is not positional
        or parameters["blobpath"].default is not inspect.Parameter.empty
        or parameters["expected_hash"].kind is not positional
        or parameters["expected_hash"].default is not None
    ):
        raise RuntimeError(
            "Unsupported tiktoken read_file_cached parameters; expected "
            "(blobpath, expected_hash)"
        )
    os.environ["TIKTOKEN_CACHE_DIR"] = str(_ASSET_DIR)
    tiktoken.load.read_file_cached = _read_bundled_file
