"""Install Chatbook's immutable, offline tiktoken table reader."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Literal, Self

# No module-scope pydantic import and no eager tiktoken import either: this
# module is imported (and its armer registered) by the package ``__init__``,
# which every ``tldw_chatbook`` import runs — including the pinned workspace
# worker, whose import closure must stay stdlib-only (Phase 0c). Pydantic
# loads on first manifest read (inside ``_manifest_by_url``); tiktoken loads
# only when something actually imports it, at which point the registered
# armer selects the bundled assets before first use.

_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "tiktoken_cache"
_MANIFEST_PATH = _ASSET_DIR / "manifest.json"
_OVERRIDE_KEYS = ("TIKTOKEN_CACHE_DIR", "DATA_GYM_CACHE_DIR")


class BundledTiktokenAssetError(RuntimeError):
    """A requested tiktoken table is absent from or invalid in the bundle."""


@lru_cache(maxsize=1)
def _manifest_by_url() -> dict[str, Any]:
    """Load the reviewed asset manifest once, indexed by source URL."""
    from pydantic import (
        BaseModel,
        ConfigDict,
        Field,
        ValidationError,
        model_validator,
    )

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


class _TiktokenRuntimeArmingLoader:
    """Loader wrapper that arms the bundled runtime after tiktoken execs.

    The import machinery has fully initialized ``tiktoken`` (or failed)
    by the time control returns from :meth:`exec_module`, so arming there
    is strictly before any caller's first use of the library.
    """

    def __init__(self, loader: Any, finder: "_TiktokenRuntimeArmingFinder") -> None:
        self._loader = loader
        self._finder = finder

    def create_module(self, spec: Any) -> Any:
        return self._loader.create_module(spec)

    def exec_module(self, module: Any) -> None:
        try:
            self._loader.exec_module(module)
        except BaseException:
            self._finder.restore()
            raise
        try:
            install_tiktoken_runtime()
        finally:
            self._finder.discard()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)


class _TiktokenRuntimeArmingFinder:
    """Meta-path finder arming the bundled runtime at tiktoken's first import."""

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: Any = None,
    ) -> Any:
        if fullname != "tiktoken" or "tiktoken" in sys.modules:
            return None
        try:
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
        except BaseException:
            self.restore()
            raise
        if (
            spec is None
            or spec.loader is None
            or not hasattr(spec.loader, "exec_module")
        ):
            # tiktoken absent or not wrappable: leave the machinery untouched.
            self.restore()
            return spec
        spec.loader = _TiktokenRuntimeArmingLoader(spec.loader, self)
        return spec

    def restore(self) -> None:
        if not any(finder is self for finder in sys.meta_path):
            sys.meta_path.insert(0, self)

    def discard(self) -> None:
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass


def arm_bundled_tiktoken_runtime() -> None:
    """Arm :func:`install_tiktoken_runtime` without importing tiktoken now.

    Phase 0c: the package ``__init__`` used to call
    ``install_tiktoken_runtime()`` eagerly, which imported ``tiktoken``
    (third-party) into every process that touched any ``tldw_chatbook``
    module — including the pinned workspace worker, whose import closure
    must stay stdlib-only. Registering this armer instead keeps the
    offline-asset guarantee identical: if tiktoken is already imported the
    installer runs immediately (exactly the old behavior); otherwise a
    meta-path finder arms it the moment ``tiktoken`` is first imported —
    which is, by definition, before any caller can use it. A pre-existing
    ``TIKTOKEN_CACHE_DIR``/``DATA_GYM_CACHE_DIR`` override still wins, and
    the finder removes itself once the arming decision has been made.

    Raises:
        RuntimeError: If tiktoken is already imported and its cache-reader
            signature differs from the reviewed 0.14.0 seam (deferred to
            tiktoken's first import otherwise).
    """
    if "tiktoken" in sys.modules:
        install_tiktoken_runtime()
        return
    if not any(
        isinstance(finder, _TiktokenRuntimeArmingFinder)
        for finder in sys.meta_path
    ):
        sys.meta_path.insert(0, _TiktokenRuntimeArmingFinder())
