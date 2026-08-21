"""Representation-safe filesystem primitives for lasting Database Notes sync."""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tldw_chatbook.Notes.note_import_windows_fs import (
    OS_WINDOWS_FILESYSTEM,
    WindowsReadOnlyFilesystem,
    discover_import_sources,
    read_discovered_source,
)
from tldw_chatbook.Notes.note_import_plan_models import ImportBounds
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncFileIdentity,
    NotesSyncFileObservation,
    NotesSyncSerializationProfile,
)
from tldw_chatbook.Notes.sync_paths import (
    PinnedSyncRoot,
    SafeSyncBytes,
    SyncPathError,
    SyncPathPartialError,
    guarded_rename_available,
)

_UTF8_BOM = b"\xef\xbb\xbf"


def _windows_stable_identity_digest(identity: Any) -> str:
    payload = f"{identity.device}\0{identity.inode}"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


class NotesSyncFilesystemError(OSError):
    """A bounded lasting-sync filesystem refusal."""

    def __init__(self, reason_code: str):
        self.reason_code = reason_code
        super().__init__(reason_code)


class NotesSyncFilesystemPartialError(NotesSyncFilesystemError):
    """A committed mutation requires durable cleanup or review."""

    def __init__(
        self,
        reason_code: str,
        cleanup_handle: "NotesSyncPrivateCleanupHandle",
    ):
        self.cleanup_handle = cleanup_handle
        super().__init__(reason_code)


@dataclass(frozen=True, slots=True, repr=False)
class NotesSyncPrivateCleanupHandle:
    """Owner-private authority needed to journal a partial filesystem result."""

    private_relative_path: str | None

    def __repr__(self) -> str:
        return "NotesSyncPrivateCleanupHandle(<private>)"


@dataclass(frozen=True, slots=True, repr=False)
class NotesSyncFileSnapshot:
    """Private exact-byte and logical-text observation used by review/apply."""

    observation: NotesSyncFileObservation
    text: str
    raw_bytes: bytes
    reviewed_state: SafeSyncBytes
    representation_digest: str
    recovery_bytes: bytes | None = None

    def __repr__(self) -> str:
        return "NotesSyncFileSnapshot(<private>)"


def _canonical_directory(path: Path | str) -> Path:
    selected = Path(path)
    try:
        lexical = selected.lstat()
        canonical = selected.resolve(strict=True)
    except OSError:
        raise NotesSyncFilesystemError("root_unavailable") from None
    if not selected.is_dir():
        raise NotesSyncFilesystemError("root_not_directory")
    if os.path.islink(selected) or getattr(lexical, "st_reparse_tag", 0):
        raise NotesSyncFilesystemError("root_link_or_reparse")
    return canonical


def _overlaps(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def validate_sync_root_admission(
    candidate: Path | str,
    *,
    sync_roots: Iterable[Path | str] = (),
    file_notes_roots: Iterable[Path | str] = (),
    private_paths: Iterable[Path | str] = (),
) -> Path:
    """Return one canonical root after rejecting every owned overlap."""

    canonical = _canonical_directory(candidate)
    for roots, reason in (
        (sync_roots, "root_overlap"),
        (file_notes_roots, "file_notes_overlap"),
        (private_paths, "private_path_overlap"),
    ):
        for other in roots:
            try:
                other_canonical = Path(other).resolve(strict=True)
            except OSError:
                raise NotesSyncFilesystemError("comparison_root_unavailable") from None
            if _overlaps(canonical, other_canonical):
                raise NotesSyncFilesystemError(reason)
    return canonical


def _parse_supported_text(payload: bytes) -> tuple[str, bool, str, bool]:
    bom = payload.startswith(_UTF8_BOM)
    encoded = payload[len(_UTF8_BOM) :] if bom else payload
    try:
        text = encoded.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise NotesSyncFilesystemError("unsupported_encoding") from None

    without_crlf = text.replace("\r\n", "")
    has_crlf = "\r\n" in text
    has_lf = "\n" in without_crlf
    has_cr = "\r" in without_crlf
    if has_cr and (has_crlf or has_lf):
        raise NotesSyncFilesystemError("mixed_newlines")
    if has_cr:
        raise NotesSyncFilesystemError("unsupported_newline")
    if has_crlf and has_lf:
        raise NotesSyncFilesystemError("mixed_newlines")
    newline = "crlf" if has_crlf else "lf"
    logical = text.replace("\r\n", "\n")
    final_newline = logical.endswith("\n")
    return logical, bom, newline, final_newline


class PosixNotesSyncFilesystem:
    """Writable POSIX adapter composed from one descriptor-pinned root."""

    def __init__(
        self,
        selected_root: Path | str,
        *,
        max_file_bytes: int = 10 * 1024 * 1024,
    ):
        if not self.supports_writes():
            raise NotesSyncFilesystemError("writable_adapter_unavailable")
        if type(max_file_bytes) is not int or max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be a positive integer.")
        self.canonical_root = _canonical_directory(selected_root)
        self._max_file_bytes = max_file_bytes
        self._root = PinnedSyncRoot(self.canonical_root)

    @staticmethod
    def supports_writes(*, platform: str | None = None) -> bool:
        """Return whether guarded replacement is available on this platform."""

        selected = sys.platform if platform is None else platform
        return os.name == "posix" and guarded_rename_available(platform=selected)

    def __enter__(self) -> "PosixNotesSyncFilesystem":
        self._root.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._root.__exit__(exc_type, exc, traceback)

    @staticmethod
    def _metadata_issue(snapshot: SafeSyncBytes) -> str | None:
        if snapshot.flags or snapshot.has_extended_acl:
            return "unsupported_metadata"
        if snapshot.owner_user != os.geteuid() or snapshot.owner_group != os.getegid():
            return "unsupported_metadata"
        return None

    def observe(
        self,
        relative_path: Path | str,
        *,
        require_writable: bool = True,
    ) -> NotesSyncFileSnapshot:
        """Capture exact representation and stable identity without mutation."""

        try:
            reviewed = self._root.read_bytes(
                relative_path,
                max_bytes=self._max_file_bytes,
            )
        except SyncPathError as exc:
            raise NotesSyncFilesystemError(exc.reason) from None
        metadata_issue = self._metadata_issue(reviewed)
        if require_writable and metadata_issue is not None:
            raise NotesSyncFilesystemError(metadata_issue)
        text, bom, newline, final_newline = _parse_supported_text(reviewed.content)
        observation = NotesSyncFileObservation(
            relative_path=reviewed.relative_path.as_posix(),
            identity=NotesSyncFileIdentity(
                device=reviewed.identity.device,
                inode=reviewed.identity.inode,
                link_count=reviewed.identity.link_count,
            ),
            content_digest=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            size_bytes=len(reviewed.content),
            serialization=NotesSyncSerializationProfile(
                utf8_bom=bom,
                newline=newline,
                final_newline=final_newline,
                mode=reviewed.mode,
            ),
        )
        return NotesSyncFileSnapshot(
            observation=observation,
            text=text,
            raw_bytes=reviewed.content,
            reviewed_state=reviewed,
            representation_digest=hashlib.sha256(reviewed.content).hexdigest(),
        )

    @staticmethod
    def serialize(text: str, profile: NotesSyncSerializationProfile) -> bytes:
        """Apply one captured profile after normalizing note-side newlines."""

        if type(text) is not str:
            raise TypeError("text must be a string.")
        logical = text.replace("\r\n", "\n").replace("\r", "\n")
        if profile.final_newline and not logical.endswith("\n"):
            logical += "\n"
        elif not profile.final_newline:
            logical = logical.rstrip("\n")
        represented = (
            logical.replace("\n", "\r\n") if profile.newline == "crlf" else logical
        )
        encoded = represented.encode("utf-8")
        return (_UTF8_BOM + encoded) if profile.utf8_bom else encoded

    def replace(
        self,
        relative_path: Path | str,
        text: str,
        *,
        expected: NotesSyncFileSnapshot,
    ) -> NotesSyncFileSnapshot:
        """Replace only the exact state presented for review."""

        if type(expected) is not NotesSyncFileSnapshot:
            raise TypeError("expected must be a NotesSyncFileSnapshot.")
        metadata_issue = self._metadata_issue(expected.reviewed_state)
        if metadata_issue is not None:
            raise NotesSyncFilesystemError(metadata_issue)
        profile = expected.observation.serialization
        payload = self.serialize(text, profile)
        try:
            self._root.replace_bytes(
                relative_path,
                payload,
                expected=expected.reviewed_state,
                mode=profile.mode,
            )
        except SyncPathPartialError as exc:
            raise NotesSyncFilesystemPartialError(
                exc.reason,
                NotesSyncPrivateCleanupHandle(
                    exc.cleanup_leaf,
                ),
            ) from None
        except SyncPathError as exc:
            raise NotesSyncFilesystemError(exc.reason) from None
        try:
            observed = self.observe(relative_path)
        except NotesSyncFilesystemError:
            raise NotesSyncFilesystemPartialError(
                "replacement_postcondition_failed",
                NotesSyncPrivateCleanupHandle(Path(relative_path).as_posix()),
            ) from None
        if (
            observed.raw_bytes != payload
            or observed.observation.serialization != profile
        ):
            raise NotesSyncFilesystemPartialError(
                "replacement_postcondition_failed",
                NotesSyncPrivateCleanupHandle(Path(relative_path).as_posix()),
            )
        return NotesSyncFileSnapshot(
            observation=observed.observation,
            text=observed.text,
            raw_bytes=observed.raw_bytes,
            reviewed_state=observed.reviewed_state,
            representation_digest=observed.representation_digest,
            recovery_bytes=expected.raw_bytes,
        )

    def move(
        self,
        destination_path: Path | str,
        *,
        expected: NotesSyncFileSnapshot,
    ) -> NotesSyncFileSnapshot:
        """Move one exact reviewed file without copy/delete fallback."""

        if type(expected) is not NotesSyncFileSnapshot:
            raise TypeError("expected must be a NotesSyncFileSnapshot.")
        metadata_issue = self._metadata_issue(expected.reviewed_state)
        if metadata_issue is not None:
            raise NotesSyncFilesystemError(metadata_issue)
        try:
            self._root.move_file(
                expected.reviewed_state.relative_path,
                destination_path,
                expected=expected.reviewed_state,
            )
        except SyncPathPartialError as exc:
            raise NotesSyncFilesystemPartialError(
                exc.reason,
                NotesSyncPrivateCleanupHandle(
                    exc.cleanup_leaf,
                ),
            ) from None
        except SyncPathError as exc:
            raise NotesSyncFilesystemError(exc.reason) from None
        try:
            moved = self.observe(destination_path)
        except NotesSyncFilesystemError:
            raise NotesSyncFilesystemPartialError(
                "move_postcondition_failed",
                NotesSyncPrivateCleanupHandle(Path(destination_path).as_posix()),
            ) from None
        if (
            moved.raw_bytes != expected.raw_bytes
            or moved.observation.identity != expected.observation.identity
            or moved.observation.serialization != expected.observation.serialization
        ):
            raise NotesSyncFilesystemPartialError(
                "move_postcondition_failed",
                NotesSyncPrivateCleanupHandle(Path(destination_path).as_posix()),
            )
        return moved


@dataclass(frozen=True, slots=True, repr=False)
class WindowsNotesSyncObservation:
    """Private observation produced by the existing native Windows reader."""

    relative_path: str
    text: str
    content_digest: str
    representation_digest: str
    stable_identity_digest: str
    freshness_digest: str
    size_bytes: int
    serialization: NotesSyncSerializationProfile

    def __repr__(self) -> str:
        return "WindowsNotesSyncObservation(<private>)"


class WindowsNotesSyncObservationFilesystem:
    """Native no-reparse Windows observation without write authority."""

    def __init__(
        self,
        selected_root: Path | str,
        *,
        bounds: ImportBounds,
        filesystem: WindowsReadOnlyFilesystem = OS_WINDOWS_FILESYSTEM,
    ) -> None:
        self._root = filesystem.absolute(Path(selected_root))
        self._bounds = bounds
        self._filesystem = filesystem

    @staticmethod
    def supports_writes() -> bool:
        return False

    def observe(self) -> tuple[WindowsNotesSyncObservation, ...]:
        """Discover and read through the existing native read-only adapter."""

        discovery = discover_import_sources(
            [self._root],
            self._bounds,
            filesystem=self._filesystem,
        )
        observations: list[WindowsNotesSyncObservation] = []
        identities: set[str] = set()
        for candidate in discovery.candidates:
            if candidate.source.source_path.suffix.lower() not in {
                ".md",
                ".markdown",
                ".txt",
            }:
                continue
            payload = read_discovered_source(
                candidate,
                self._bounds,
                filesystem=self._filesystem,
            )
            text, bom, newline, final_newline = _parse_supported_text(payload)
            relative = candidate.source.source_path.relative_to(self._root).as_posix()
            stable_identity_digest = _windows_stable_identity_digest(candidate.identity)
            if stable_identity_digest in identities:
                raise NotesSyncFilesystemError("duplicate_stable_identity")
            identities.add(stable_identity_digest)
            freshness_payload = "\0".join(
                str(value)
                for value in (
                    candidate.identity.mode,
                    candidate.identity.size,
                    candidate.identity.modified_ns,
                    candidate.identity.changed_ns,
                    hashlib.sha256(payload).hexdigest(),
                )
            )
            observations.append(
                WindowsNotesSyncObservation(
                    relative_path=relative,
                    text=text,
                    content_digest=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    representation_digest=hashlib.sha256(payload).hexdigest(),
                    stable_identity_digest=stable_identity_digest,
                    freshness_digest=hashlib.sha256(
                        freshness_payload.encode("ascii")
                    ).hexdigest(),
                    size_bytes=candidate.size_bytes,
                    serialization=NotesSyncSerializationProfile(
                        utf8_bom=bom,
                        newline=newline,
                        final_newline=final_newline,
                        mode=candidate.identity.mode & 0o7777,
                    ),
                )
            )
        return tuple(observations)


__all__ = [
    "NotesSyncFileSnapshot",
    "NotesSyncFilesystemError",
    "NotesSyncFilesystemPartialError",
    "NotesSyncPrivateCleanupHandle",
    "PosixNotesSyncFilesystem",
    "WindowsNotesSyncObservationFilesystem",
    "WindowsNotesSyncObservation",
    "validate_sync_root_admission",
]
