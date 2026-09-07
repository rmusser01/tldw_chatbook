"""Network-free loading for the pinned audio.cpp artifact-source manifest."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import ipaddress
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import TYPE_CHECKING, Any
import unicodedata
from urllib.parse import quote, urlsplit

if TYPE_CHECKING:
    from tldw_chatbook.Model_Artifacts.service import ArtifactDescriptor
    from tldw_chatbook.TTS.audio_cpp_recipes import (
        AudioCppPackageRecipe,
        AudioCppRecipeRegistry,
    )


AUDIO_CPP_ARTIFACT_REPOSITORY = "audio-cpp/audio.cpp-gguf"
AUDIO_CPP_ARTIFACT_COMMIT = "597048d9a920592808d7d4e2acd7b9c4596a143a"
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_FIELDS = {"repository", "commit", "packages"}
_PACKAGE_FIELDS = {
    "recipe_id",
    "recipe_revision",
    "package_variant",
    "artifact_id",
    "license_id",
    "license_url",
    "usage_notice",
    "files",
}
_FILE_FIELDS = {"source_path", "managed_path", "size_bytes", "sha256"}
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_PACKAGES = 67
_MAX_FILES_PER_PACKAGE = 256
_MAX_TOTAL_FILES = 4096
_MAX_TOKEN_BYTES = 256
_MAX_PATH_BYTES = 1024
_MAX_LICENSE_URL_BYTES = 2048
_MAX_USAGE_NOTICE_BYTES = 4096
_URL_HOST_LABEL = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\Z",
    re.ASCII,
)
_URL_PATH = re.compile(r"(?:/[A-Za-z0-9._~!$&'()*+,;=:@%\-]*)*\Z", re.ASCII)
_INVALID_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})", re.ASCII)


@dataclass(frozen=True, slots=True)
class AudioCppArtifactSourceFile:
    """One immutable upstream file and its managed-package destination."""

    source_path: str
    managed_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class AudioCppArtifactPackage:
    """Reviewed source facts for one exact recipe package."""

    recipe_id: str
    recipe_revision: int
    package_variant: str
    artifact_id: str
    license_id: str
    license_url: str
    usage_notice: str
    files: tuple[AudioCppArtifactSourceFile, ...]

    @property
    def key(self) -> tuple[str, int, str]:
        """Return the stable recipe package key."""

        return (self.recipe_id, self.recipe_revision, self.package_variant)


@dataclass(frozen=True, slots=True)
class AudioCppArtifactSourceManifest:
    """Pinned source-only audio.cpp package catalog."""

    repository: str
    commit: str
    packages: tuple[AudioCppArtifactPackage, ...]


def _object(value: object, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{label} must be an object")
    return value


def _array(value: object, label: str) -> list[Any]:
    if type(value) is not list:
        raise TypeError(f"{label} must be an array")
    return value


def _fields(value: dict[str, Any], expected: set[str], label: str) -> None:
    missing = expected - value.keys()
    extra = value.keys() - expected
    if missing:
        raise ValueError(f"{label} missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise ValueError(f"{label} has unknown fields: {', '.join(sorted(extra))}")


def _unsafe_text(value: str) -> bool:
    return any(
        character in {"\x00", "\r", "\n"}
        or unicodedata.category(character) in {"Cc", "Cf", "Cs"}
        for character in value
    )


def _string(value: object, label: str, *, max_bytes: int = _MAX_TOKEN_BYTES) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    if (
        not value
        or value != value.strip()
        or _unsafe_text(value)
        or len(value.encode("utf-8")) > max_bytes
    ):
        raise ValueError(f"{label} must be bounded safe text")
    return value


def _positive_integer(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise TypeError(f"{label} must be a positive integer")
    return value


def _relative_path(value: object, label: str) -> str:
    path = _string(value, label, max_bytes=_MAX_PATH_BYTES)
    parts = path.split("/")
    pure_path = PurePosixPath(path)
    windows_path = PureWindowsPath(path)
    if (
        "\\" in path
        or "$" in path
        or "%" in path
        or pure_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or any(part in {"", ".", ".."} for part in parts)
        or pure_path.as_posix() != path
    ):
        raise ValueError(f"{label} must be a normalized relative POSIX path")
    return path


def _valid_url_hostname(hostname: str, *, bracketed: bool) -> bool:
    if bracketed:
        try:
            ipaddress.IPv6Address(hostname)
        except ValueError:
            return False
        return True
    if ":" in hostname:
        return False
    try:
        ascii_hostname = hostname.encode("idna").decode("ascii")
    except UnicodeError:
        return False
    labels = ascii_hostname.removesuffix(".").split(".")
    return (
        bool(labels)
        and all(_URL_HOST_LABEL.fullmatch(label) is not None for label in labels)
        and len(ascii_hostname) <= 254
    )


def _valid_url_authority(authority: str, hostname: str) -> bool:
    bracketed = authority.startswith("[")
    if bracketed:
        closing_bracket = authority.find("]")
        if closing_bracket < 0:
            return False
        raw_hostname = authority[1:closing_bracket]
        suffix = authority[closing_bracket + 1 :]
        if suffix and (not suffix.startswith(":") or not suffix[1:].isdigit()):
            return False
    else:
        if "[" in authority or "]" in authority or authority.count(":") > 1:
            return False
        raw_hostname, separator, port_text = authority.rpartition(":")
        if not separator:
            raw_hostname = authority
        elif not port_text.isdigit():
            return False
    return raw_hostname.casefold() == hostname.casefold() and _valid_url_hostname(
        hostname,
        bracketed=bracketed,
    )


def _validate_canonical_https_url(value: str, label: str) -> None:
    if (
        "?" in value
        or "#" in value
        or "\\" in value
        or any(character.isspace() for character in value)
        or _INVALID_PERCENT_ESCAPE.search(value) is not None
    ):
        raise ValueError(f"{label} must be a canonical credential-free https URL")
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(
            f"{label} must be a canonical credential-free https URL"
        ) from exc
    if (
        parsed.scheme != "https"
        or not value.startswith("https://")
        or parsed.hostname is None
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not _valid_url_authority(parsed.netloc, parsed.hostname)
        or _URL_PATH.fullmatch(parsed.path) is None
    ):
        raise ValueError(f"{label} must be a canonical credential-free https URL")


def _source_file(raw: object, label: str) -> AudioCppArtifactSourceFile:
    item = _object(raw, label)
    _fields(item, _FILE_FIELDS, label)
    sha256 = _string(item["sha256"], f"{label}.sha256")
    if _SHA256_RE.fullmatch(sha256) is None:
        raise ValueError(f"{label}.sha256 must be 64 lowercase hexadecimal characters")
    return AudioCppArtifactSourceFile(
        source_path=_relative_path(item["source_path"], f"{label}.source_path"),
        managed_path=_relative_path(item["managed_path"], f"{label}.managed_path"),
        size_bytes=_positive_integer(item["size_bytes"], f"{label}.size_bytes"),
        sha256=sha256,
    )


def _package(raw: object, index: int) -> AudioCppArtifactPackage:
    label = f"packages[{index}]"
    item = _object(raw, label)
    _fields(item, _PACKAGE_FIELDS, label)
    files_raw = _array(item["files"], f"{label}.files")
    if not files_raw:
        raise ValueError(f"{label}.files must not be empty")
    if len(files_raw) > _MAX_FILES_PER_PACKAGE:
        raise ValueError(f"{label}.files exceeds {_MAX_FILES_PER_PACKAGE}")
    files = tuple(
        _source_file(file_raw, f"{label}.files[{file_index}]")
        for file_index, file_raw in enumerate(files_raw)
    )
    for attribute in ("source_path", "managed_path"):
        paths = [getattr(file, attribute) for file in files]
        if len(paths) != len(set(paths)):
            raise ValueError(f"{label} has duplicate {attribute}")

    license_url = _string(
        item["license_url"],
        f"{label}.license_url",
        max_bytes=_MAX_LICENSE_URL_BYTES,
    )
    _validate_canonical_https_url(license_url, f"{label}.license_url")
    return AudioCppArtifactPackage(
        recipe_id=_string(item["recipe_id"], f"{label}.recipe_id"),
        recipe_revision=_positive_integer(
            item["recipe_revision"], f"{label}.recipe_revision"
        ),
        package_variant=_string(item["package_variant"], f"{label}.package_variant"),
        artifact_id=_string(item["artifact_id"], f"{label}.artifact_id"),
        license_id=_string(item["license_id"], f"{label}.license_id"),
        license_url=license_url,
        usage_notice=_string(
            item["usage_notice"],
            f"{label}.usage_notice",
            max_bytes=_MAX_USAGE_NOTICE_BYTES,
        ),
        files=files,
    )


def parse_audio_cpp_artifact_source_manifest(
    raw: object,
    *,
    expected_commit: str | None = AUDIO_CPP_ARTIFACT_COMMIT,
) -> AudioCppArtifactSourceManifest:
    """Validate decoded JSON into immutable source-only records.

    Args:
        raw: Decoded JSON value.
        expected_commit: Exact required revision, or ``None`` for a maintainer
            refresh input that may be advanced to another immutable commit.

    Returns:
        The validated immutable manifest.

    Raises:
        TypeError: If a value has the wrong exact JSON type.
        ValueError: If a value violates the bounded manifest contract.
    """

    root = _object(raw, "manifest")
    _fields(root, _MANIFEST_FIELDS, "manifest")
    repository = _string(root["repository"], "repository")
    if repository != AUDIO_CPP_ARTIFACT_REPOSITORY:
        raise ValueError(f"repository must be {AUDIO_CPP_ARTIFACT_REPOSITORY!r}")
    commit = _string(root["commit"], "commit")
    if _COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("commit must be exactly 40 lowercase hexadecimal characters")
    if expected_commit is not None and commit != expected_commit:
        raise ValueError(f"commit must be the pinned revision {expected_commit}")

    packages_raw = _array(root["packages"], "packages")
    if len(packages_raw) > _MAX_PACKAGES:
        raise ValueError(f"packages exceeds {_MAX_PACKAGES}")
    packages_list: list[AudioCppArtifactPackage] = []
    total_files = 0
    for index, item in enumerate(packages_raw):
        package = _package(item, index)
        total_files += len(package.files)
        if total_files > _MAX_TOTAL_FILES:
            raise ValueError(f"total files exceeds {_MAX_TOTAL_FILES}")
        packages_list.append(package)
    packages = tuple(packages_list)
    keys = [package.key for package in packages]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate package key")
    artifact_ids = [package.artifact_id for package in packages]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError("duplicate artifact_id")
    return AudioCppArtifactSourceManifest(repository, commit, packages)


def load_audio_cpp_artifact_source_manifest(
    path: Path | None = None,
    *,
    expected_commit: str | None = AUDIO_CPP_ARTIFACT_COMMIT,
) -> AudioCppArtifactSourceManifest:
    """Load the checked-in pinned manifest without performing network work."""

    manifest_path = path or Path(__file__).with_name("audio_cpp_artifact_manifest.json")
    with manifest_path.open("rb") as handle:
        content = handle.read(_MAX_MANIFEST_BYTES + 1)
    if len(content) > _MAX_MANIFEST_BYTES:
        raise ValueError(f"manifest exceeds {_MAX_MANIFEST_BYTES} bytes")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    try:
        manifest_text = content.decode("utf-8")
        raw = json.loads(manifest_text, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("manifest is not valid UTF-8 JSON") from exc
    return parse_audio_cpp_artifact_source_manifest(
        raw,
        expected_commit=expected_commit,
    )


@cache
def _cached_audio_cpp_artifact_source_manifest() -> AudioCppArtifactSourceManifest:
    return load_audio_cpp_artifact_source_manifest()


def audio_cpp_artifact_identity_matches_recipe(
    *,
    recipe_id: str,
    recipe_revision: int,
    package_variant: str,
    recipe_artifact_ids: tuple[str, ...],
    recipe_precision: str,
    artifact_id: str,
    revision: str,
    variant: str,
) -> bool:
    """Match scalar recipe facts to one exact pinned artifact package.

    Args:
        recipe_id: Exact reviewed recipe identifier.
        recipe_revision: Exact reviewed recipe revision.
        package_variant: Exact reviewed package variant.
        recipe_artifact_ids: Artifact IDs assigned to the recipe.
        recipe_precision: Precision declared by the recipe.
        artifact_id: Persisted managed artifact identifier.
        revision: Persisted managed artifact revision.
        variant: Persisted managed artifact variant.

    Returns:
        ``True`` only when every scalar matches one package in the checked-in
        pinned manifest. Returns ``False`` when the manifest cannot be loaded
        or validated, or when any identity or mapping fact does not match.
    """

    try:
        manifest = _cached_audio_cpp_artifact_source_manifest()
    except (OSError, TypeError, ValueError):
        return False
    return bool(
        revision == manifest.commit == AUDIO_CPP_ARTIFACT_COMMIT
        and variant == recipe_precision
        and recipe_artifact_ids == (artifact_id,)
        and any(
            package.key == (recipe_id, recipe_revision, package_variant)
            and package.artifact_id == artifact_id
            for package in manifest.packages
        )
    )


def audio_cpp_artifact_source_matches_descriptor(
    descriptor: ArtifactDescriptor,
) -> bool:
    """Match one descriptor to the complete checked-in pinned source facts."""

    from tldw_chatbook.Model_Artifacts.service import ArtifactDescriptor, ArtifactFile

    if type(descriptor) is not ArtifactDescriptor:
        return False
    try:
        manifest = _cached_audio_cpp_artifact_source_manifest()
    except (OSError, TypeError, ValueError):
        return False
    package = next(
        (
            item
            for item in manifest.packages
            if item.artifact_id == descriptor.reference.artifact_id
        ),
        None,
    )
    if package is None:
        return False
    files = tuple(
        ArtifactFile(item.managed_path, item.size_bytes, item.sha256)
        for item in sorted(package.files, key=lambda item: item.managed_path)
    )
    first_source = min(package.files, key=lambda item: item.managed_path).source_path
    return bool(
        descriptor.reference.revision == manifest.commit
        and descriptor.upstream_repository == manifest.repository
        and descriptor.upstream_revision == manifest.commit
        and descriptor.source_url == _pinned_source_url(manifest, first_source)
        and descriptor.files == files
        and descriptor.expected_installed_bytes
        == sum(item.size_bytes for item in files)
    )


def _pinned_source_url(manifest: AudioCppArtifactSourceManifest, path: str) -> str:
    return (
        f"https://huggingface.co/{manifest.repository}/resolve/"
        f"{manifest.commit}/{quote(path, safe='/')}"
    )


def _manifest_recipe(
    package: AudioCppArtifactPackage,
    registry: AudioCppRecipeRegistry,
) -> AudioCppPackageRecipe:
    matches = tuple(
        recipe for recipe in registry.recipes if recipe.recipe_id == package.recipe_id
    )
    if len(matches) != 1:
        raise ValueError("audio.cpp manifest recipe is unknown or duplicated")
    recipe = matches[0]
    artifact_owners = tuple(
        candidate
        for candidate in registry.recipes
        if package.artifact_id in candidate.model_library_artifact_ids
    )
    if (
        artifact_owners != (recipe,)
        or recipe.recipe_revision != package.recipe_revision
        or recipe.package_variant != package.package_variant
        or recipe.model_library_artifact_ids != (package.artifact_id,)
    ):
        raise ValueError("audio.cpp manifest recipe facts do not match")
    return recipe


def _curated_entry(
    manifest: AudioCppArtifactSourceManifest,
    package: AudioCppArtifactPackage,
    recipe: AudioCppPackageRecipe,
) -> tuple[ArtifactDescriptor, dict[str, str]]:
    from tldw_chatbook.Model_Artifacts.service import (
        ArtifactDescriptor,
        ArtifactFile,
        ArtifactFormat,
        ArtifactRef,
        ArtifactRole,
        ProvenanceClass,
    )
    from tldw_chatbook.TTS.audio_cpp_recipes import AudioCppFileKind

    required = {signal.relative_path: signal for signal in recipe.required_files}
    optional = {signal.relative_path: signal for signal in recipe.optional_files}
    signals = required | optional
    files_by_path = {file.managed_path: file for file in package.files}
    manifest_paths = files_by_path.keys()
    if not required.keys() <= manifest_paths <= signals.keys() or any(
        files_by_path[path].size_bytes < signals[path].minimum_size_bytes
        for path in manifest_paths
    ):
        raise ValueError("audio.cpp manifest recipe file closure does not match")
    if recipe.package_format is not AudioCppFileKind.GGUF:
        raise ValueError("audio.cpp manifest recipe format is unsupported")
    if not {"tts", "clone"}.intersection(recipe.capabilities):
        raise ValueError("audio.cpp manifest recipe is not a speech package")

    files = tuple(
        ArtifactFile(file.managed_path, file.size_bytes, file.sha256)
        for file in sorted(package.files, key=lambda item: item.managed_path)
    )
    sources = {
        file.managed_path: _pinned_source_url(manifest, file.source_path)
        for file in sorted(package.files, key=lambda item: item.managed_path)
    }
    return (
        ArtifactDescriptor(
            reference=ArtifactRef(
                package.artifact_id,
                manifest.commit,
                recipe.precision,
            ),
            model_id=recipe.default_public_model_id,
            role=ArtifactRole.ROOT,
            format=ArtifactFormat.GGUF,
            consumer="audio_cpp",
            model_family=recipe.family,
            upstream_repository=manifest.repository,
            upstream_revision=manifest.commit,
            source_url=next(iter(sources.values())),
            precision=recipe.precision,
            expected_installed_bytes=sum(file.size_bytes for file in files),
            license_id=package.license_id,
            license_url=package.license_url,
            usage_notice=package.usage_notice,
            runtime_name="audio.cpp",
            runtime_version_constraint=(
                f"{recipe.audio_cpp_release}@{recipe.audio_cpp_commit}"
            ),
            supported_os=("unassigned",),
            supported_architectures=("unassigned",),
            provenance=(
                ProvenanceClass.CHATBOOK_CURATED,
                ProvenanceClass.INTEGRITY_VERIFIED,
            ),
            files=files,
        ),
        sources,
    )


def audio_cpp_curated_entries(
    registry: AudioCppRecipeRegistry | None = None,
) -> tuple[tuple[ArtifactDescriptor, dict[str, str]], ...]:
    """Join reviewed pinned sources to their exact approved recipes."""

    if registry is None:
        from tldw_chatbook.TTS.audio_cpp_recipes import AUDIO_CPP_RECIPE_REGISTRY

        registry = AUDIO_CPP_RECIPE_REGISTRY
    manifest = load_audio_cpp_artifact_source_manifest()
    entries = tuple(
        _curated_entry(manifest, package, _manifest_recipe(package, registry))
        for package in manifest.packages
    )
    references = [descriptor.reference for descriptor, _sources in entries]
    if len(references) != len(set(references)):
        raise ValueError("audio.cpp manifest produces duplicate artifact references")
    return entries
