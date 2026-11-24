"""Publish and reconcile profile-private, source-owned article audio."""

from __future__ import annotations

import errno
import os as stdlib_os
import sqlite3
import stat
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from tldw_chatbook.Backup_Recovery.native_files import (
    create_private_directory,
    create_private_file,
    pinned_directory,
    publish_new,
)
from tldw_chatbook.Backup_Recovery.native_platform import flush_directory
from tldw_chatbook.TTS.controlled_wav_profile import RetainedWav
from tldw_chatbook.Utils.path_validation import validate_recovery_relative_path
from tldw_chatbook.Utils.platform_files import os

from .article_listening_lease import ListeningLease
from .article_listening_models import (
    ArticleAudio,
    AudioMetadata,
    AudioReservation,
    GeneratedArticleAudio,
    ListeningConflict,
    _KEY,
    _int,
    validate_article_audio,
)
from .article_listening_store import ArticleListeningStore

MAX_UNREFERENCED_BYTES = 512 * 1024 * 1024
UNREFERENCED_TTL = 7 * 86400


class ArticleAudioError(ValueError):
    """Bounded file lifecycle outcome safe to report outside the storage owner."""


@dataclass(slots=True)
class ArticleAudioCleanup:
    interrupted: int = 0
    missing: int = 0
    removed: int = 0
    retry: int = 0
    uncertain: int = 0


def _names(key):
    if type(key) is not str or not _KEY.fullmatch(key):
        raise ArticleAudioError("unsafe_article_audio_key")
    validate_recovery_relative_path(key)
    return key, "." + key + ".stage"


def _identity(info):
    return info.st_dev, info.st_ino


def _exact(parent, name):
    """Case-insensitive native lookup must never claim an unrecognized spelling."""
    names = os.listdir(parent)
    if name in names:
        return True
    if any(other.casefold() == name.casefold() for other in names):
        raise ArticleAudioError("article_audio_name_collision")
    return False


@contextmanager
def _leaf(parent, name):
    if parent is None or not _exact(parent, name):
        raise FileNotFoundError
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_nlink != 1
            or info.st_mode & 0o077
        ):
            raise ArticleAudioError("unsafe_article_audio_file")
        if _identity(os.stat(name, dir_fd=parent, follow_symlinks=False)) != _identity(
            info
        ):
            raise ArticleAudioError("article_audio_identity_changed")
        yield fd, info
    finally:
        os.close(fd)


def _metadata(row):
    return AudioMetadata(
        *(
            row[k]
            for k in (
                "sha256",
                "sample_rate",
                "content_frames",
                "file_frames",
                "byte_size",
            )
        )
    )


class ArticleAudioAssets:
    """Synchronous storage operations; physical custody ends on worker return."""

    def __init__(
        self,
        store: ArticleListeningStore,
        *,
        profile_data_root: Path,
        reservation: AudioReservation | None = None,
        lease: ListeningLease | None = None,
    ):
        if (
            not isinstance(profile_data_root, Path)
            or not profile_data_root.is_absolute()
            or ".." in profile_data_root.parts
        ):
            raise ArticleAudioError("invalid_article_audio_root")
        self.store = store
        self.root = profile_data_root / "article_audio"
        self.reservation = reservation
        self.lease = lease

    @contextmanager
    def _hold(self, keys):
        if self.store.db.conn.in_transaction or getattr(
            self.store.db._local, "transaction_depth", 0
        ):
            raise ListeningConflict("article_audio_requires_committed_state")
        paths = (self.root, *(self.root / name for key in keys for name in _names(key)))
        guard = (
            self.store.mutation_lease(related_paths=paths)
            if self.lease is None
            else self.lease.borrow(
                self.store.database,
                profile=self.store._authority().profile,
                related_paths=paths,
            )
        )
        try:
            with guard as lease, lease.asset_operation():
                yield lease
        except sqlite3.Error as exc:
            if getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_FULL:
                raise ArticleAudioError("article_audio_disk_full") from None
            raise ArticleAudioError("article_audio_database_unavailable") from None
        except OSError as exc:
            if exc.errno == errno.ENOSPC or getattr(exc, "winerror", None) == 112:
                raise ArticleAudioError("article_audio_disk_full") from None
            raise ArticleAudioError("article_audio_storage_unavailable") from None

    @contextmanager
    def _directory(self, *, create=False):
        with ExitStack() as stack:
            try:
                parent = stack.enter_context(pinned_directory(self.root.parent))
            except FileNotFoundError:
                if create:
                    raise
                yield None
                return
            if not _exact(parent, self.root.name):
                if not create:
                    yield None
                    return
                create_private_directory(self.root)
            with pinned_directory(self.root) as directory:
                info = os.fstat(directory)
                if info.st_uid != os.geteuid() or info.st_mode & 0o077:
                    raise ArticleAudioError("unsafe_article_audio_directory")
                yield directory

    def _check_directory(self, directory):
        with pinned_directory(self.root) as current:
            if _identity(os.fstat(current)) != _identity(os.fstat(directory)):
                raise ArticleAudioError("article_audio_identity_changed")

    def _verify(self, directory, name, expected, *, identity=None):
        with _leaf(directory, name) as (fd, before):
            if identity is not None and _identity(before) != identity:
                raise ArticleAudioError("article_audio_identity_changed")
            with stdlib_os.fdopen(fd, "rb", closefd=False) as stream:
                validate_article_audio(stream, expected)
            after = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if (
                _identity(after) != _identity(before)
                or after.st_nlink != 1
                or after.st_size != before.st_size
                or after.st_mtime_ns != before.st_mtime_ns
            ):
                raise ArticleAudioError("article_audio_identity_changed")
            self._check_directory(directory)
        return _identity(before)

    def _audio(self, generation, item, key, m):
        return ArticleAudio(
            generation,
            item,
            RetainedWav(
                self.root / key,
                m.sha256,
                m.sample_rate,
                m.content_frames,
                m.file_frames,
            ),
            m.byte_size,
        )

    def _verify_issued(self, directory, name, expected, identity, claim, lease):
        try:
            self._verify(directory, name, expected, identity=identity)
        except (ArticleAudioError, FileNotFoundError):
            self.store.relinquish_audio(claim, lease=lease)
            raise ArticleAudioError("article_audio_custody_changed") from None
        except OSError as exc:
            if exc.errno != errno.ELOOP:
                raise
            self.store.relinquish_audio(claim, lease=lease)
            raise ArticleAudioError("article_audio_custody_changed") from None
        except ValueError:
            # The exclusively created file is ours, but its WAV is invalid.
            self.store.cancel_audio(claim, lease=lease)
            raise ArticleAudioError("invalid_article_audio") from None

    def publish(
        self, candidate: GeneratedArticleAudio, *, source_item_id: int, generation: int
    ) -> ArticleAudio:
        """Validate and atomically attach a new immutable, reserved generation."""
        claim = self.reservation
        if (
            type(candidate) is not GeneratedArticleAudio
            or type(claim) is not AudioReservation
            or _int(generation, 1) != claim.generation
            or _int(source_item_id, 1) != claim.item_id
        ):
            raise ListeningConflict("article_audio_reservation_changed")
        expected = candidate.metadata()
        key, stage = _names(claim.file_key)
        with self._hold((key,)) as lease:
            self.store.check_reservation(claim, lease=lease)
            with self._directory(create=True) as directory:
                try:
                    if _exact(directory, key) or _exact(directory, stage):
                        raise ArticleAudioError("article_audio_name_collision")
                except ArticleAudioError:
                    self.store.relinquish_audio(claim, lease=lease)
                    raise
                try:
                    with create_private_file(self.root / stage) as fd:
                        staged_identity = _identity(os.fstat(fd))
                        with stdlib_os.fdopen(fd, "wb", closefd=False) as stream:
                            stream.write(candidate.wav_bytes)
                except FileExistsError:
                    self.store.relinquish_audio(claim, lease=lease)
                    raise ArticleAudioError("article_audio_name_collision") from None
                self._verify_issued(
                    directory, stage, expected, staged_identity, claim, lease
                )
                self.store.check_reservation(claim, lease=lease)
                parent = _identity(os.fstat(directory))
                try:
                    publish_new(
                        self.root / stage,
                        self.root / key,
                        parent_identities=(parent, parent),
                    )
                except FileExistsError:
                    if not _exact(directory, stage) or not _exact(directory, key):
                        # A post-rename error is ambiguous even if it reports
                        # EEXIST. Keep the journal and let reconciliation own it.
                        raise ArticleAudioError(
                            "article_audio_publication_uncertain"
                        ) from None
                    self.store.relinquish_audio(claim, lease=lease)
                    # Our stage was created exclusively. The collided target is foreign.
                    with _leaf(directory, stage) as (_, info):
                        if _identity(info) == staged_identity:
                            self._unlink(directory, stage)
                    raise ArticleAudioError("article_audio_name_collision") from None
                # Flush failure after rename and final SQLite failures leave a pending
                # journal row. A later physical owner reconciles the actual names.
                self._verify_issued(
                    directory, key, expected, staged_identity, claim, lease
                )
                self.store.complete_audio(claim, expected, lease=lease)
                return self._audio(generation, source_item_id, key, expected)

    def _read(self, directory, row, lease):
        g, key = row["generation"], row["file_key"]
        if row["status"] != "ready":
            raise ArticleAudioError("article_audio_missing")
        m = _metadata(row)
        try:
            self._verify(directory, key, m)
        except (ValueError, FileNotFoundError) as exc:
            self.store.mark_audio_missing(g, lease=lease)
            code = (
                "article_audio_missing"
                if isinstance(exc, FileNotFoundError)
                else "article_audio_corrupt"
            )
            raise ArticleAudioError(code) from None
        return self._audio(g, row["item_id"], key, m)

    def read_verified(self, generation: int) -> ArticleAudio:
        """Read and verify one ready receipt; preserve the old bookmark on failure."""
        rows = self.store.audio_inventory(generation)
        if not rows:
            raise ArticleAudioError("article_audio_missing")
        key = rows[0]["file_key"]
        with self._hold((key,)) as lease, self._directory() as directory:
            rows = self.store.audio_inventory(generation)
            if not rows or rows[0]["file_key"] != key:
                raise ListeningConflict("article_audio_inventory_changed")
            return self._read(directory, rows[0], lease)

    def reuse(self, entry_id: str, *, expected_revision: int) -> ArticleAudio | None:
        """Attach only a verified ready generation with this exact entry identity."""
        for generation in self.store.find_reusable(entry_id):
            rows = self.store.audio_inventory(generation)
            if not rows:
                continue
            key = rows[0]["file_key"]
            with self._hold((key,)) as lease, self._directory() as directory:
                rows = self.store.audio_inventory(generation)
                if not rows or rows[0]["file_key"] != key:
                    raise ListeningConflict("article_audio_inventory_changed")
                try:
                    audio = self._read(directory, rows[0], lease)
                except ArticleAudioError as exc:
                    if str(exc) not in (
                        "article_audio_missing",
                        "article_audio_corrupt",
                    ):
                        raise
                    continue
                self.store.attach_audio(
                    entry_id,
                    generation,
                    expected_revision=expected_revision,
                    lease=lease,
                )
                return audio
        return None

    def _unlink(self, directory, name):
        if directory is None or not _exact(directory, name):
            return
        with _leaf(directory, name) as (_, expected):
            self._check_directory(directory)
            observed = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if _identity(observed) != _identity(expected) or observed.st_nlink != 1:
                raise ArticleAudioError("article_audio_identity_changed")
            os.unlink(name, dir_fd=directory)
            flush_directory(directory)

    def _cleanup(self, lease, directory, receipt, rows):
        admitted = {(r["generation"], r["file_key"]) for r in rows}
        for gen, key in self.store.pending_deletions():
            if (gen, key) not in admitted:
                continue
            try:
                names = _names(key)
                if directory is not None and all(
                    _exact(directory, name) for name in names
                ):
                    # An atomic move cannot leave both names. The tombstone
                    # alone cannot decide which object belonged to the writer.
                    receipt.retry += 1
                    receipt.uncertain += 1
                    continue
                for name in names:
                    self._unlink(directory, name)
            except (OSError, ArticleAudioError):
                receipt.retry += 1
                continue
            self.store.finish_audio_deletion(gen, key, lease=lease)
            receipt.removed += 1

    @contextmanager
    def _inventory(self):
        rows = self.store.audio_inventory()
        keys = tuple(row["file_key"] for row in rows)
        with self._hold(keys) as lease:
            rows = self.store.audio_inventory()
            if keys != tuple(row["file_key"] for row in rows):
                raise ListeningConflict("article_audio_inventory_changed")
            with self._directory() as directory:
                yield rows, lease, directory

    def reconcile(self) -> ArticleAudioCleanup:
        """Recover interrupted claims and missing artifacts under physical ownership."""
        receipt = ArticleAudioCleanup()
        with self._inventory() as (rows, lease, directory):
            pending = [r["generation"] for r in rows if r["status"] == "pending"]
            if pending:
                self.store.retire_audio(pending, lease=lease)
                receipt.interrupted = len(pending)
            owned = {name for row in rows for name in _names(row["file_key"])}
            if directory is not None:
                receipt.uncertain = sum(
                    name not in owned for name in os.listdir(directory)
                )
            for row in rows:
                if row["status"] == "ready":
                    try:
                        self._read(directory, row, lease)
                    except ArticleAudioError as exc:
                        if str(exc) not in (
                            "article_audio_missing",
                            "article_audio_corrupt",
                        ):
                            raise
                        receipt.missing += 1
            self._cleanup(lease, directory, receipt, rows)
        return receipt

    def prune(self, now: int) -> ArticleAudioCleanup:
        """Retire expired and least-recently used unreferenced private bytes."""
        _int(now)
        receipt = ArticleAudioCleanup()
        with self._inventory() as (rows, lease, directory):
            eligible = sorted(
                (
                    r
                    for r in rows
                    if r["status"] in ("missing", "ready") and not r["referenced"]
                ),
                key=lambda r: (r["last_used"], r["generation"]),
                reverse=True,
            )
            total = 0
            retired = []
            over_limit = False
            for row in eligible:
                try:
                    with _leaf(directory, row["file_key"]) as (_, info):
                        size = info.st_size
                except FileNotFoundError:
                    size = 0
                except (OSError, ArticleAudioError):
                    receipt.retry += 1
                    continue
                expired = now - row["last_used"] >= UNREFERENCED_TTL
                if not expired:
                    total += size
                    over_limit = total > MAX_UNREFERENCED_BYTES
                if expired or over_limit:
                    retired.append(row["generation"])
            if retired:
                self.store.retire_audio(retired, lease=lease)
            self._cleanup(lease, directory, receipt, rows)
        return receipt
