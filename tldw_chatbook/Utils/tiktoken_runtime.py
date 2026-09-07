"""Install Chatbook's immutable, offline tiktoken table reader."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import inspect
import os
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "tiktoken_cache"
_MANIFEST_PATH = _ASSET_DIR / "manifest.json"
_OVERRIDE_KEYS = ("TIKTOKEN_CACHE_DIR", "DATA_GYM_CACHE_DIR")


class BundledTiktokenAssetError(RuntimeError):
    """A requested tiktoken table is absent from or invalid in the bundle."""


class _ManifestFile(BaseModel):
    """One reviewed tiktoken source and its immutable cache identity."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    encoding: Literal["gpt2", "r50k_base", "p50k_base", "cl100k_base", "o200k_base"]
    url: str = Field(
        pattern=r"^https://openaipublic\.blob\.core\.windows\.net/",
        min_length=1,
    )
    cache_key: str = Field(pattern=r"^[0-9a-f]{40}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _ManifestLicense(BaseModel):
    """Redistribution evidence recorded with the reviewed assets."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    spdx: Literal["MIT"]
    source: str = Field(min_length=1)
    clarification: str = Field(pattern=r"^https://", min_length=1)
    gpt2_additional_source: str = Field(pattern=r"^https://", min_length=1)


class _TiktokenManifest(BaseModel):
    """Complete schema for the package-owned tiktoken manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    tiktoken_version: Literal["0.14.0"]
    constructor_module: Literal["tiktoken_ext.openai_public"]
    constructor_path: Literal["tiktoken_ext/openai_public.py"]
    read_file_cached_signature: Literal[
        "read_file_cached(blobpath: str, expected_hash: str | None = None) -> bytes"
    ]
    cache_key_algorithm: Literal["sha1(source_url UTF-8 bytes)"]
    model_to_encoding_coverage: dict[
        str,
        Literal["gpt2", "r50k_base", "p50k_base", "cl100k_base", "o200k_base"],
    ] = Field(min_length=1)
    license: _ManifestLicense
    update_procedure: list[str] = Field(min_length=1)
    files: list[_ManifestFile] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_file_identities(self) -> Self:
        """Reject entries that would be silently replaced in the URL index."""
        urls = [entry.url for entry in self.files]
        cache_keys = [entry.cache_key for entry in self.files]
        if len(set(urls)) != len(urls):
            raise ValueError("manifest contains duplicate source URLs")
        if len(set(cache_keys)) != len(cache_keys):
            raise ValueError("manifest contains duplicate cache keys")
        return self


@lru_cache(maxsize=1)
def _manifest_by_url() -> dict[str, _ManifestFile]:
    """Load the reviewed asset manifest once, indexed by source URL."""
    try:
        manifest = _TiktokenManifest.model_validate_json(_MANIFEST_PATH.read_bytes())
        return {entry.url: entry for entry in manifest.files}
    except (OSError, ValidationError) as error:
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
    except (AttributeError, UnicodeError) as error:
        raise BundledTiktokenAssetError(
            f"Invalid bundled tiktoken manifest entry for {blobpath}"
        ) from error
    manifest_key = entry.cache_key
    manifest_hash = entry.sha256

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
    """Select the bundle unless the caller supplied an upstream cache override.

    Raises:
        RuntimeError: If tiktoken's cache-reader signature differs from the
            reviewed 0.14.0 compatibility seam.
    """
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
