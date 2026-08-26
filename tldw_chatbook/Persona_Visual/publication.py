"""Synchronous immutable publication for profile-owned Persona Visual packs."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import sqlite3
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDBError
from tldw_chatbook.Utils.private_paths import secure_private_directory

from . import assets as asset_boundary
from .assets import (
    PersonaVisualAssetMetadata,
    validate_persona_visual_asset_set,
)
from .repository import (
    PersonaVisualIdentity,
    PersonaVisualRepository,
)
from .validation import validate_persona_visual_manifest


_ERROR_PREFIX = "persona_visual_"
_READ_CHUNK_BYTES = 64 * 1024
_MANIFEST_LIMIT = 2 * 1024 * 1024
_SOURCE_CONTEXT_KEYS = frozenset(
    {"source_id", "provenance", "license", "source_server_commit"}
)
_SOURCE_KINDS = frozenset({"imported", "manual"})
_SUFFIXES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
_CLEANUP_PATH = re.compile(
    r"\Apersona_visual/packs/[0-9a-f]{32}/versions/"
    r"(?:[0-9a-f]{32}|\.staging-[0-9a-f]{32})\Z"
)
_CLEANUP_CAPABILITY = re.compile(
    r"\Apv1:([0-9a-f]{64}):(persona_visual/packs/[0-9a-f]{32}/versions/"
    r"(?:[0-9a-f]{32}|\.staging-[0-9a-f]{32}))\Z"
)
_CLEANUP_MARKER_NAME = ".persona-visual-cleanup"


class PersonaVisualPublicationError(ValueError):
    """A stable path-free publication failure with an optional cleanup capability."""

    __slots__ = ("category", "cleanup_candidate")

    def __init__(self, category: str, *, cleanup_candidate: str | None = None) -> None:
        if type(category) is not str or not category.startswith(_ERROR_PREFIX):
            category = "persona_visual_publication_failed"
        self.category = category
        self.cleanup_candidate = cleanup_candidate
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class PersonaVisualPublicationAssetSource:
    """One immutable source reference plus its validated path-free metadata."""

    source_storage_key: str
    metadata: PersonaVisualAssetMetadata


@dataclass(frozen=True, slots=True)
class PersonaVisualPublicationSnapshot:
    """One complete immutable publication request; it is not an editable model."""

    persona_id: str
    persona_revision: int
    title: str
    manifest_json: str
    assets: tuple[PersonaVisualPublicationAssetSource, ...]
    expected_identity: PersonaVisualIdentity | None = None
    description: str = ""
    source_kind: str = "manual"
    source_context: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class PersonaVisualPublicationResult:
    """Exact identity transition plus an opaque orphan-cleanup capability."""

    old_identity: PersonaVisualIdentity | None
    new_identity: PersonaVisualIdentity
    cleanup_candidate: str | None


@dataclass(frozen=True, slots=True)
class _PinnedSource:
    parts: tuple[str, ...]
    directory_identities: tuple[tuple[int, int, int, int, int], ...]
    identity: tuple[int, int, int, int, int]
    byte_count: int
    sha256: str
    data: bytes


@dataclass(frozen=True, slots=True)
class _PinnedPublicationFile:
    parent_fd: int
    name: str
    identity: tuple[int, int, int, int, int]
    byte_count: int
    sha256: str


def publish_persona_visual(
    repository: PersonaVisualRepository,
    snapshot: PersonaVisualPublicationSnapshot,
    *,
    source_root: os.PathLike[str] | str,
    profile_root: os.PathLike[str] | str,
    authority_guard: Callable[[], bool],
    atomic_replace: Callable[..., None] = os.replace,
) -> PersonaVisualPublicationResult:
    """Publish one complete immutable directory and activate it atomically in SQLite.

    Args:
        repository: Idle Persona Visual repository that owns the activation.
        snapshot: Fully bounded manifest, provenance, asset, and authority snapshot.
        source_root: Absolute root containing the snapshotted source assets.
        profile_root: Absolute profile root receiving immutable pack storage.
        authority_guard: Side-effect-free callback checked before and inside the
            repository transaction.
        atomic_replace: Atomic rename operation, injectable for failure testing.

    Returns:
        The old and new identities plus any cleanup capability.

    Raises:
        PersonaVisualPublicationError: If validation, authority, storage, or
            activation fails. The exception message is always a stable category.
    """

    if type(repository) is not PersonaVisualRepository:
        raise PersonaVisualPublicationError("persona_visual_candidate_invalid")
    _require_idle_repository(repository)
    manifest, context, assets = _validate_snapshot(snapshot)
    if not callable(authority_guard) or not callable(atomic_replace):
        raise PersonaVisualPublicationError("persona_visual_candidate_invalid")
    if not _posix_guards_available():
        raise PersonaVisualPublicationError("persona_visual_publication_denied")

    source_path, profile_path = _publication_roots(source_root, profile_root)
    current = _preflight_identity(repository, snapshot)

    pack_token = uuid4().hex
    version_token = uuid4().hex
    cleanup_secret = os.urandom(32).hex()
    versions_relpath = f"persona_visual/packs/{pack_token}/versions"
    staging_name = f".staging-{version_token}"
    final_name = version_token
    staging_token = f"{versions_relpath}/{staging_name}"
    final_token = f"{versions_relpath}/{final_name}"
    versions_path = profile_path / versions_relpath

    source_chain: list[int] = []
    pinned_sources: list[_PinnedSource] = []
    profile_chain: list[tuple[Path, int]] = []
    versions_fd = -1
    staging_fd = -1
    materialized_assets_fd = -1
    materialized_files: list[_PinnedPublicationFile] = []
    renamed = False
    staging_cleanup_attempted = False

    def retained_candidate() -> str | None:
        nonlocal staging_cleanup_attempted
        if (
            versions_fd >= 0
            and staging_fd >= 0
            and _entry_matches_fd(versions_fd, final_name, staging_fd)
        ):
            return _cleanup_capability(final_token, cleanup_secret)
        if (
            versions_fd >= 0
            and staging_fd >= 0
            and _entry_matches_fd(versions_fd, staging_name, staging_fd)
        ):
            staging_cleanup_attempted = True
            if not _delete_pinned_directory(versions_fd, staging_name, staging_fd):
                return _cleanup_capability(staging_token, cleanup_secret)
        return None

    try:
        source_chain = _open_absolute_directory_chain(source_path)
        source_fd = source_chain[-1]
        pinned_sources = [
            _pin_source_asset(source_fd, source, metadata)
            for source, metadata in assets
        ]

        for directory in (
            profile_path,
            profile_path / "persona_visual",
            profile_path / "persona_visual/packs",
            profile_path / f"persona_visual/packs/{pack_token}",
            versions_path,
        ):
            privacy = secure_private_directory(
                directory, create=True, application_owned=True
            )
            if not privacy.verified_private:
                raise PermissionError
        opened_profile = _open_absolute_directory_chain(versions_path)
        profile_chain = _identify_directory_chain(versions_path, opened_profile)
        versions_fd = opened_profile[-1]
        # Ownership moves to profile_chain; it closes the same descriptor list.

        os.mkdir(staging_name, mode=0o700, dir_fd=versions_fd)
        staging_fd = os.open(
            staging_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=versions_fd,
        )
        staging_identity = os.fstat(staging_fd)
        marker_raw = _cleanup_marker(
            cleanup_secret,
            final_token,
            (staging_identity.st_dev, staging_identity.st_ino),
        )
        _write_private_file(staging_fd, _CLEANUP_MARKER_NAME, marker_raw)
        assets_fd = -1
        try:
            os.mkdir("assets", mode=0o700, dir_fd=staging_fd)
            assets_fd = os.open(
                "assets",
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=staging_fd,
            )
            asset_rows: list[dict[str, Any]] = []
            for index, ((_, metadata), pinned) in enumerate(
                zip(assets, pinned_sources)
            ):
                filename = f"{index:03d}{_SUFFIXES[metadata.mime_type]}"
                _write_private_file(assets_fd, filename, pinned.data)
                asset_rows.append(
                    _repository_asset_row(
                        metadata,
                        storage_relpath=f"{final_token}/assets/{filename}",
                    )
                )
            _sync_directory(assets_fd)
        finally:
            if assets_fd >= 0:
                os.close(assets_fd)

        manifest_raw = snapshot.manifest_json.encode("utf-8")
        _write_private_file(staging_fd, "manifest.json", manifest_raw)
        _sync_directory(staging_fd)
        materialized_assets_fd = os.open(
            "assets",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=staging_fd,
        )
        materialized_files.append(
            _pin_publication_file(
                staging_fd,
                _CLEANUP_MARKER_NAME,
                marker_raw,
            )
        )
        materialized_files.append(
            _pin_publication_file(staging_fd, "manifest.json", manifest_raw)
        )
        for index, pinned in enumerate(pinned_sources):
            filename = f"{index:03d}{_SUFFIXES[assets[index][1].mime_type]}"
            materialized_files.append(
                _pin_publication_file(materialized_assets_fd, filename, pinned.data)
            )
        if not _all_source_entries_current(
            source_path, source_chain[-1], pinned_sources
        ):
            raise PermissionError
        if not _directory_chain_current(profile_chain):
            raise PermissionError
        atomic_replace(
            staging_name,
            final_name,
            src_dir_fd=versions_fd,
            dst_dir_fd=versions_fd,
        )
        renamed = True
        if not _entry_matches_fd(versions_fd, final_name, staging_fd):
            raise PermissionError
        _sync_directory(versions_fd)

        try:
            caller_current = authority_guard() is True
        except Exception:
            caller_current = False
        if not caller_current:
            raise PersonaVisualPublicationError(
                "persona_visual_authority_changed",
                cleanup_candidate=retained_candidate(),
            )

        guard_failure: str | None = None

        def final_guard() -> bool:
            nonlocal guard_failure
            filesystem_current = bool(
                _all_source_entries_current(
                    source_path, source_chain[-1], pinned_sources
                )
                and _directory_chain_current(profile_chain)
                and _entry_matches_fd(versions_fd, final_name, staging_fd)
                and _entry_matches_fd(staging_fd, "assets", materialized_assets_fd)
                and _publication_files_current(materialized_files)
            )
            try:
                caller_current = authority_guard() is True
            except Exception:
                caller_current = False
            if not filesystem_current:
                guard_failure = "persona_visual_publication_denied"
                return False
            if not caller_current:
                guard_failure = "persona_visual_authority_changed"
                return False
            guard_failure = None
            return True

        manifest_storage = f"{final_token}/manifest.json"
        try:
            if snapshot.expected_identity is None:
                graph = repository.activate_new_pack(
                    persona_id=snapshot.persona_id,
                    title=snapshot.title,
                    description=snapshot.description,
                    source_kind=snapshot.source_kind,
                    source_context=context,
                    manifest=manifest,
                    manifest_storage_relpath=manifest_storage,
                    assets=asset_rows,
                    expected_persona_revision=snapshot.persona_revision,
                    authority_guard=final_guard,
                )
            else:
                graph = repository.publish_version(
                    persona_id=snapshot.persona_id,
                    manifest=manifest,
                    manifest_storage_relpath=manifest_storage,
                    assets=asset_rows,
                    expected_identity=snapshot.expected_identity,
                    expected_persona_revision=snapshot.persona_revision,
                    authority_guard=final_guard,
                )
        except ValueError as error:
            category = (
                guard_failure
                if str(error) == "persona_visual_authority_changed"
                and guard_failure is not None
                else _repository_error_category(str(error))
            )
            raise PersonaVisualPublicationError(
                category, cleanup_candidate=retained_candidate()
            ) from None
        except (CharactersRAGDBError, sqlite3.Error, OSError, RuntimeError, TypeError):
            raise PersonaVisualPublicationError(
                "persona_visual_database_failed",
                cleanup_candidate=retained_candidate(),
            ) from None
        return PersonaVisualPublicationResult(
            old_identity=None if current is None else current.identity,
            new_identity=graph.identity,
            cleanup_candidate=None,
        )
    except KeyboardInterrupt as interruption:
        if renamed and _entry_matches_fd(versions_fd, final_name, staging_fd):
            cleanup_candidate = _cleanup_capability(final_token, cleanup_secret)
            try:
                deleted = cleanup_persona_visual_publication_candidate(
                    repository,
                    cleanup_candidate,
                    profile_root=profile_path,
                )
            except BaseException:
                deleted = False
            if not deleted:
                interruption.cleanup_candidate = cleanup_candidate
        raise
    except PersonaVisualPublicationError:
        raise
    except PermissionError:
        raise PersonaVisualPublicationError(
            "persona_visual_publication_denied",
            cleanup_candidate=retained_candidate(),
        ) from None
    except (OSError, TypeError, ValueError, OverflowError):
        raise PersonaVisualPublicationError(
            "persona_visual_publication_failed",
            cleanup_candidate=retained_candidate(),
        ) from None
    finally:
        if (
            not renamed
            and not staging_cleanup_attempted
            and versions_fd >= 0
            and staging_fd >= 0
        ):
            _delete_pinned_directory(versions_fd, staging_name, staging_fd)
        if staging_fd >= 0:
            os.close(staging_fd)
        if materialized_assets_fd >= 0:
            os.close(materialized_assets_fd)
        _close_descriptors(source_chain)
        _close_descriptors([descriptor for _path, descriptor in profile_chain])


def cleanup_persona_visual_publication_candidate(
    repository: PersonaVisualRepository,
    cleanup_candidate: str,
    *,
    profile_root: os.PathLike[str] | str,
) -> bool:
    """Use one issued capability to delete an unreferenced pinned directory."""

    if (
        type(repository) is not PersonaVisualRepository
        or type(cleanup_candidate) is not str
        or not _posix_guards_available()
    ):
        raise PersonaVisualPublicationError("persona_visual_cleanup_denied")
    capability = _parse_cleanup_capability(cleanup_candidate)
    if capability is None:
        raise PersonaVisualPublicationError("persona_visual_cleanup_denied")
    cleanup_path, cleanup_secret = capability
    _require_idle_repository(repository)
    try:
        profile_path = _canonical_root(profile_root, must_exist=True)
    except (OSError, TypeError, ValueError):
        raise PersonaVisualPublicationError("persona_visual_cleanup_denied") from None
    candidate_path = profile_path / cleanup_path
    versions_path = candidate_path.parent
    chain: list[int] = []
    candidate_fd = -1
    connection: sqlite3.Connection | None = None
    reservation = False
    try:
        if not versions_path.is_relative_to(profile_path):
            raise PermissionError
        directory = profile_path
        for component in versions_path.relative_to(profile_path).parts:
            privacy = secure_private_directory(
                directory, create=False, application_owned=True
            )
            if not privacy.verified_private:
                raise PermissionError
            directory /= component
        privacy = secure_private_directory(
            versions_path, create=False, application_owned=True
        )
        if not privacy.verified_private:
            raise PermissionError
        chain = _open_absolute_directory_chain(versions_path)
        versions_fd = chain[-1]
        candidate_fd = os.open(
            candidate_path.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=versions_fd,
        )
        if not _owned_private_directory(candidate_fd) or not _entry_matches_fd(
            versions_fd, candidate_path.name, candidate_fd
        ):
            raise PermissionError
        if not _cleanup_marker_current(
            candidate_fd,
            cleanup_path,
            cleanup_secret,
        ):
            raise PermissionError
        connection = repository.db.get_connection()
        connection.execute("BEGIN IMMEDIATE")
        reservation = True
        if _storage_reference_exists(connection, cleanup_path):
            raise PersonaVisualPublicationError("persona_visual_cleanup_referenced")
        if not _entry_matches_fd(versions_fd, candidate_path.name, candidate_fd):
            raise PermissionError
        if not _delete_pinned_directory(versions_fd, candidate_path.name, candidate_fd):
            raise PermissionError
        _sync_directory(versions_fd)
        if _storage_reference_exists(connection, cleanup_path):
            raise PersonaVisualPublicationError("persona_visual_cleanup_referenced")
        connection.commit()
        reservation = False
        return True
    except PersonaVisualPublicationError:
        raise
    except sqlite3.Error:
        raise PersonaVisualPublicationError("persona_visual_database_failed") from None
    except (OSError, PermissionError, RuntimeError, TypeError, ValueError):
        raise PersonaVisualPublicationError("persona_visual_cleanup_denied") from None
    finally:
        if reservation and connection is not None:
            connection.rollback()
        if candidate_fd >= 0:
            os.close(candidate_fd)
        _close_descriptors(chain)


def _validate_snapshot(
    snapshot: object,
) -> tuple[
    object,
    dict[str, str],
    tuple[tuple[str, PersonaVisualAssetMetadata], ...],
]:
    try:
        if type(snapshot) is not PersonaVisualPublicationSnapshot:
            raise ValueError
        if (
            type(snapshot.persona_id) is not str
            or not snapshot.persona_id
            or len(snapshot.persona_id) > 200
            or type(snapshot.persona_revision) is not int
            or snapshot.persona_revision < 0
            or type(snapshot.title) is not str
            or not snapshot.title
            or len(snapshot.title) > 256
            or type(snapshot.description) is not str
            or len(snapshot.description) > 4096
            or type(snapshot.source_kind) is not str
            or snapshot.source_kind not in _SOURCE_KINDS
            or type(snapshot.manifest_json) is not str
            or len(snapshot.manifest_json.encode("utf-8")) > _MANIFEST_LIMIT
            or type(snapshot.assets) is not tuple
            or (
                snapshot.expected_identity is not None
                and type(snapshot.expected_identity) is not PersonaVisualIdentity
            )
        ):
            raise ValueError
        for value in (
            snapshot.persona_id,
            snapshot.title,
            snapshot.description,
            snapshot.manifest_json,
        ):
            value.encode("utf-8")
        manifest = json.loads(
            snapshot.manifest_json,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
        canonical = json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if canonical != snapshot.manifest_json:
            raise ValueError
        if type(snapshot.source_context) is not tuple:
            raise ValueError
        context: dict[str, str] = {}
        for item in snapshot.source_context:
            if type(item) is not tuple or len(item) != 2:
                raise ValueError
            key, value = item
            if (
                type(key) is not str
                or key not in _SOURCE_CONTEXT_KEYS
                or key in context
                or type(value) is not str
                or not value
                or len(value) > 256
            ):
                raise ValueError
            value.encode("utf-8")
            stripped = value.strip()
            if (
                "/" in value
                or "\\" in value
                or any(ord(character) < 32 for character in value)
                or stripped in {".", ".."}
                or stripped.startswith(("{", "[", "~"))
            ):
                raise ValueError
            context[key] = value
        sources: list[tuple[str, PersonaVisualAssetMetadata]] = []
        metadata_items: list[PersonaVisualAssetMetadata] = []
        source_keys: set[str] = set()
        for item in snapshot.assets:
            if type(item) is not PersonaVisualPublicationAssetSource:
                raise ValueError
            source_key = "/".join(
                asset_boundary._storage_parts(
                    item.source_storage_key, item.metadata.mime_type
                )
            )
            if source_key in source_keys:
                raise ValueError
            source_keys.add(source_key)
            sources.append((source_key, item.metadata))
            metadata_items.append(item.metadata)
        normalized = validate_persona_visual_asset_set(tuple(metadata_items))
        if tuple(metadata_items) != normalized:
            raise ValueError
        validate_persona_visual_manifest(
            snapshot.manifest_json,
            {
                metadata.asset_key: (metadata.width, metadata.height)
                for metadata in normalized
            },
        )
        return manifest, context, tuple(sources)
    except PersonaVisualPublicationError:
        raise
    except Exception:
        raise PersonaVisualPublicationError(
            "persona_visual_candidate_invalid"
        ) from None


def _preflight_identity(
    repository: PersonaVisualRepository,
    snapshot: PersonaVisualPublicationSnapshot,
):
    expected = snapshot.expected_identity
    try:
        current = repository.get_active_persona_pack(snapshot.persona_id)
    except ValueError:
        raise PersonaVisualPublicationError(
            "persona_visual_identity_changed"
            if expected is not None
            else "persona_visual_database_failed"
        ) from None
    except (
        CharactersRAGDBError,
        sqlite3.Error,
        OSError,
        RuntimeError,
        TypeError,
    ):
        raise PersonaVisualPublicationError("persona_visual_database_failed") from None
    if expected is None:
        if current is not None:
            raise PersonaVisualPublicationError("persona_visual_identity_changed")
        return None
    if (
        current is None
        or current.identity != expected
        or expected.persona_id != snapshot.persona_id
        or expected.persona_revision != snapshot.persona_revision
    ):
        raise PersonaVisualPublicationError("persona_visual_identity_changed")
    return current


def _repository_error_category(category: str) -> str:
    if category == "persona_visual_authority_changed":
        return category
    if category in {
        "persona_visual_identity_changed",
        "persona_visual_binding_changed",
        "persona_visual_persona_revision_changed",
    }:
        return "persona_visual_identity_changed"
    if category in {
        "persona_visual_manifest_invalid",
        "persona_visual_asset_invalid",
        "persona_visual_pack_invalid",
    }:
        return "persona_visual_candidate_invalid"
    return "persona_visual_database_failed"


def _cleanup_capability(cleanup_path: str, secret: str) -> str:
    if (
        _CLEANUP_PATH.fullmatch(cleanup_path) is None
        or re.fullmatch(r"[0-9a-f]{64}", secret) is None
    ):
        raise ValueError
    return f"pv1:{secret}:{cleanup_path}"


def _parse_cleanup_capability(value: str) -> tuple[str, str] | None:
    match = _CLEANUP_CAPABILITY.fullmatch(value)
    if match is None:
        return None
    secret, cleanup_path = match.groups()
    return cleanup_path, secret


def _cleanup_marker(
    secret: str,
    cleanup_path: str,
    directory_identity: tuple[int, int],
) -> bytes:
    parent, name = cleanup_path.rsplit("/", 1)
    final_path = f"{parent}/{name.removeprefix('.staging-')}"
    if (
        _CLEANUP_PATH.fullmatch(final_path) is None
        or len(directory_identity) != 2
        or any(type(value) is not int or value < 0 for value in directory_identity)
    ):
        raise ValueError
    device, inode = directory_identity
    return (
        hashlib.blake2b(
            f"persona-visual-cleanup-v1\0{final_path}\0{device}\0{inode}".encode(
                "ascii"
            ),
            key=bytes.fromhex(secret),
            digest_size=32,
        )
        .hexdigest()
        .encode("ascii")
    )


def _cleanup_marker_current(
    candidate_fd: int,
    cleanup_path: str,
    secret: str,
) -> bool:
    marker_fd = -1
    try:
        candidate = os.fstat(candidate_fd)
        expected = _cleanup_marker(
            secret,
            cleanup_path,
            (candidate.st_dev, candidate.st_ino),
        )
        marker_fd = os.open(
            _CLEANUP_MARKER_NAME,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0),
            dir_fd=candidate_fd,
        )
        named = os.stat(
            _CLEANUP_MARKER_NAME,
            dir_fd=candidate_fd,
            follow_symlinks=False,
        )
        opened = os.fstat(marker_fd)
        identity = _file_identity(opened)
        if (
            not stat.S_ISREG(named.st_mode)
            or _file_identity(named) != identity
            or opened.st_size != len(expected)
            or _read_bounded(marker_fd, len(expected)) != expected
        ):
            return False
        after = os.fstat(marker_fd)
        final = os.stat(
            _CLEANUP_MARKER_NAME,
            dir_fd=candidate_fd,
            follow_symlinks=False,
        )
        return _file_identity(after) == identity == _file_identity(final)
    except (OSError, ValueError):
        return False
    finally:
        if marker_fd >= 0:
            os.close(marker_fd)


def _publication_roots(
    source_root: os.PathLike[str] | str,
    profile_root: os.PathLike[str] | str,
) -> tuple[Path, Path]:
    try:
        source = _canonical_root(source_root, must_exist=True)
        profile = _canonical_root(profile_root, must_exist=True)
        if (
            source == profile
            or source.is_relative_to(profile)
            or profile.is_relative_to(source)
        ):
            raise ValueError
        return source, profile
    except Exception:
        raise PersonaVisualPublicationError(
            "persona_visual_publication_denied"
        ) from None


def _canonical_root(value: os.PathLike[str] | str, *, must_exist: bool) -> Path:
    raw = os.fspath(value)
    if type(raw) is not str or not raw or "\x00" in raw:
        raise ValueError
    path = Path(raw)
    if not path.is_absolute() or str(path) != raw:
        raise ValueError
    resolved = path.resolve(strict=must_exist)
    if resolved != path:
        raise ValueError
    metadata = os.lstat(path)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError
    return path


def _open_absolute_directory_chain(path: Path) -> list[int]:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptors = [os.open(os.sep, flags)]
    try:
        for component in path.parts[1:]:
            descriptors.append(os.open(component, flags, dir_fd=descriptors[-1]))
        return descriptors
    except BaseException:
        _close_descriptors(descriptors)
        raise


def _identify_directory_chain(
    path: Path, descriptors: list[int]
) -> list[tuple[Path, int]]:
    paths: list[Path] = [Path(os.sep)]
    current = Path(os.sep)
    for component in path.parts[1:]:
        current /= component
        paths.append(current)
    if len(paths) != len(descriptors):
        raise OSError
    return list(zip(paths, descriptors))


def _directory_chain_current(chain: list[tuple[Path, int]]) -> bool:
    try:
        for path, descriptor in chain:
            named = os.lstat(path)
            opened = os.fstat(descriptor)
            if not stat.S_ISDIR(named.st_mode) or (named.st_dev, named.st_ino) != (
                opened.st_dev,
                opened.st_ino,
            ):
                return False
        return True
    except OSError:
        return False


def _pin_source_asset(
    source_root_fd: int,
    source: str,
    metadata: PersonaVisualAssetMetadata,
) -> _PinnedSource:
    parts = tuple(source.split("/"))
    current = source_root_fd
    opened_directories: list[int] = []
    directory_identities: list[tuple[int, int, int, int, int]] = []
    file_fd = -1
    try:
        for component in parts[:-1]:
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=current,
            )
            opened_directories.append(child)
            named = os.stat(component, dir_fd=current, follow_symlinks=False)
            opened = os.fstat(child)
            identity = _file_identity(opened)
            if not stat.S_ISDIR(named.st_mode) or _file_identity(named) != identity:
                raise ValueError
            directory_identities.append(identity)
            current = child
        file_fd = os.open(
            parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0),
            dir_fd=current,
        )
        opened = os.fstat(file_fd)
        named = os.stat(parts[-1], dir_fd=current, follow_symlinks=False)
        identity = _file_identity(opened)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _file_identity(named) != identity
            or opened.st_size != metadata.byte_count
        ):
            raise ValueError
        data = _read_bounded(file_fd, metadata.byte_count)
        after = os.fstat(file_fd)
        final = os.stat(parts[-1], dir_fd=current, follow_symlinks=False)
        if (
            _file_identity(after) != identity
            or _file_identity(final) != identity
            or hashlib.sha256(data).hexdigest() != metadata.sha256
        ):
            raise ValueError
        _validate_asset_bytes(data, metadata)
        return _PinnedSource(
            parts=parts,
            directory_identities=tuple(directory_identities),
            identity=identity,
            byte_count=metadata.byte_count,
            sha256=metadata.sha256,
            data=data,
        )
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        _close_descriptors(opened_directories)


def _all_source_entries_current(
    source_root: Path,
    source_root_fd: int,
    sources: list[_PinnedSource],
) -> bool:
    try:
        named_root = os.lstat(source_root)
        opened_root = os.fstat(source_root_fd)
        if not stat.S_ISDIR(named_root.st_mode) or (
            named_root.st_dev,
            named_root.st_ino,
        ) != (opened_root.st_dev, opened_root.st_ino):
            return False
        for source in sources:
            if not _source_entry_current(source_root_fd, source):
                return False
        return True
    except OSError:
        return False


def _source_entry_current(source_root_fd: int, source: _PinnedSource) -> bool:
    current = source_root_fd
    opened_directories: list[int] = []
    file_fd = -1
    try:
        for component, expected in zip(source.parts[:-1], source.directory_identities):
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=current,
            )
            opened_directories.append(child)
            named = os.stat(component, dir_fd=current, follow_symlinks=False)
            opened = os.fstat(child)
            if (
                not stat.S_ISDIR(named.st_mode)
                or _file_identity(named) != expected
                or _file_identity(opened) != expected
            ):
                return False
            current = child
        file_fd = os.open(
            source.parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0),
            dir_fd=current,
        )
        named = os.stat(source.parts[-1], dir_fd=current, follow_symlinks=False)
        opened = os.fstat(file_fd)
        if (
            not stat.S_ISREG(named.st_mode)
            or _file_identity(named) != source.identity
            or _file_identity(opened) != source.identity
        ):
            return False
        data = _read_bounded(file_fd, source.byte_count)
        after = os.fstat(file_fd)
        final = os.stat(source.parts[-1], dir_fd=current, follow_symlinks=False)
        return bool(
            data == source.data
            and hashlib.sha256(data).hexdigest() == source.sha256
            and _file_identity(after) == source.identity
            and _file_identity(final) == source.identity
        )
    except (OSError, ValueError):
        return False
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        _close_descriptors(opened_directories)


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _pin_publication_file(
    parent_fd: int, name: str, expected: bytes
) -> _PinnedPublicationFile:
    file_fd = os.open(
        name,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0),
        dir_fd=parent_fd,
    )
    try:
        opened = os.fstat(file_fd)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        identity = _file_identity(opened)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _file_identity(named) != identity
            or opened.st_size != len(expected)
            or _read_bounded(file_fd, len(expected)) != expected
        ):
            raise ValueError
        os.lseek(file_fd, 0, os.SEEK_SET)
        return _PinnedPublicationFile(
            parent_fd=parent_fd,
            name=name,
            identity=identity,
            byte_count=len(expected),
            sha256=hashlib.sha256(expected).hexdigest(),
        )
    finally:
        os.close(file_fd)


def _publication_files_current(files: list[_PinnedPublicationFile]) -> bool:
    try:
        for pinned in files:
            file_fd = os.open(
                pinned.name,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0),
                dir_fd=pinned.parent_fd,
            )
            try:
                named = os.stat(
                    pinned.name,
                    dir_fd=pinned.parent_fd,
                    follow_symlinks=False,
                )
                opened = os.fstat(file_fd)
                if (
                    not stat.S_ISREG(named.st_mode)
                    or _file_identity(named) != pinned.identity
                    or _file_identity(opened) != pinned.identity
                ):
                    return False
                data = _read_bounded(file_fd, pinned.byte_count)
                after = os.fstat(file_fd)
                final = os.stat(
                    pinned.name,
                    dir_fd=pinned.parent_fd,
                    follow_symlinks=False,
                )
                if (
                    hashlib.sha256(data).hexdigest() != pinned.sha256
                    or _file_identity(after) != pinned.identity
                    or _file_identity(final) != pinned.identity
                ):
                    return False
            finally:
                os.close(file_fd)
        return True
    except (OSError, ValueError):
        return False


def _read_bounded(descriptor: int, expected_bytes: int) -> bytes:
    data: list[bytes] = []
    remaining = expected_bytes + 1
    while remaining:
        chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
        if not chunk:
            break
        data.append(chunk)
        remaining -= len(chunk)
    result = b"".join(data)
    if len(result) != expected_bytes:
        raise ValueError
    return result


def _validate_asset_bytes(data: bytes, metadata: PersonaVisualAssetMetadata) -> None:
    if hashlib.sha256(data).hexdigest() != metadata.sha256:
        raise ValueError
    asset_boundary._decode_selected_frame(data, metadata, 0)


def _repository_asset_row(
    metadata: PersonaVisualAssetMetadata, *, storage_relpath: str
) -> dict[str, Any]:
    return {
        "asset_key": metadata.asset_key,
        "role": metadata.role,
        "storage_relpath": storage_relpath,
        "mime_type": metadata.mime_type,
        "bytes": metadata.byte_count,
        "sha256": metadata.sha256,
        "width": metadata.width,
        "height": metadata.height,
        "frame_count": metadata.frame_count,
        "duration_ms": metadata.duration_ms,
    }


def _write_private_file(directory_fd: int, name: str, data: bytes) -> None:
    if type(name) is not str or "/" in name or "\\" in name or name in {"", ".", ".."}:
        raise ValueError
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        view = memoryview(data)
        written = 0
        while written < len(data):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_directory(descriptor: int) -> None:
    try:
        os.fsync(descriptor)
    except OSError as error:
        unsupported = {
            errno.EBADF,
            errno.EINVAL,
            getattr(errno, "ENOTSUP", errno.EINVAL),
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        }
        if error.errno not in unsupported:
            raise


def _entry_matches_fd(parent_fd: int, name: str, descriptor: int) -> bool:
    try:
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        opened = os.fstat(descriptor)
        return stat.S_ISDIR(named.st_mode) and (named.st_dev, named.st_ino) == (
            opened.st_dev,
            opened.st_ino,
        )
    except OSError:
        return False


def _delete_pinned_directory(parent_fd: int, name: str, pinned_fd: int) -> bool:
    try:
        if not _entry_matches_fd(parent_fd, name, pinned_fd) or not (
            _owned_private_directory(pinned_fd)
        ):
            return False
        for child in os.listdir(pinned_fd):
            child_stat = os.stat(child, dir_fd=pinned_fd, follow_symlinks=False)
            if stat.S_ISDIR(child_stat.st_mode):
                child_fd = os.open(
                    child,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=pinned_fd,
                )
                try:
                    if not _owned_private_directory(child_fd):
                        return False
                    if not _delete_pinned_directory(pinned_fd, child, child_fd):
                        return False
                finally:
                    os.close(child_fd)
            else:
                if not stat.S_ISREG(child_stat.st_mode):
                    return False
                os.unlink(child, dir_fd=pinned_fd)
        if not _entry_matches_fd(parent_fd, name, pinned_fd):
            return False
        os.rmdir(name, dir_fd=parent_fd)
        return True
    except OSError:
        return False


def _owned_private_directory(descriptor: int) -> bool:
    metadata = os.fstat(descriptor)
    return bool(
        stat.S_ISDIR(metadata.st_mode)
        and stat.S_IMODE(metadata.st_mode) & 0o077 == 0
        and metadata.st_uid == os.geteuid()
    )


def _storage_reference_exists(
    connection: sqlite3.Connection, cleanup_path: str
) -> bool:
    prefix = f"{cleanup_path}/%"
    return (
        connection.execute(
            """
            SELECT 1 FROM persona_visual_pack_versions
             WHERE storage_relpath = ? OR storage_relpath LIKE ?
            UNION ALL
            SELECT 1 FROM persona_visual_assets
             WHERE storage_relpath = ? OR storage_relpath LIKE ?
            LIMIT 1
            """,
            (cleanup_path, prefix, cleanup_path, prefix),
        ).fetchone()
        is not None
    )


def _require_idle_repository(repository: PersonaVisualRepository) -> None:
    try:
        connection = repository.db.get_connection()
        if connection.in_transaction or getattr(
            repository.db._local, "transaction_depth", 0
        ):
            raise PersonaVisualPublicationError("persona_visual_transaction_active")
    except PersonaVisualPublicationError:
        raise
    except (AttributeError, CharactersRAGDBError, sqlite3.Error, RuntimeError):
        raise PersonaVisualPublicationError("persona_visual_database_failed") from None


def _posix_guards_available() -> bool:
    return (
        os.name == "posix"
        and getattr(os, "O_DIRECTORY", 0) != 0
        and getattr(os, "O_NOFOLLOW", 0) != 0
        and {os.open, os.stat, os.rename, os.unlink}.issubset(os.supports_dir_fd)
        and os.stat in os.supports_follow_symlinks
    )


def _close_descriptors(descriptors: list[int]) -> None:
    for descriptor in reversed(descriptors):
        try:
            os.close(descriptor)
        except OSError:
            pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise ValueError


__all__ = [
    "PersonaVisualPublicationAssetSource",
    "PersonaVisualPublicationError",
    "PersonaVisualPublicationResult",
    "PersonaVisualPublicationSnapshot",
    "cleanup_persona_visual_publication_candidate",
    "publish_persona_visual",
]
