"""Publish and reconcile profile-private, source-owned article audio."""

from __future__ import annotations

import errno
import hashlib
import json
import os as stdlib_os
import sqlite3
import stat
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from tldw_chatbook.Backup_Recovery.native_files import (
    create_private_directory,
    flush_regular_parent,
    held_regular_file,
    pinned_directory,
    require_regular_file_effects,
)
from tldw_chatbook.TTS.audio_limits import MAX_BUFFERED_AUDIO_BYTES
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


class _Census:
    """Case-insensitive native lookup must never claim an unrecognized spelling."""

    def __init__(self, parent):
        self.names = set(os.listdir(parent)) if parent is not None else set()
        self.folded = {name.casefold() for name in self.names}

    def contains(self, name):
        if name in self.names:
            return True
        if name.casefold() in self.folded:
            raise ArticleAudioError("article_audio_name_collision")
        return False


def _confirm_absent(parent, name):
    try:
        os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise ArticleAudioError("article_audio_custody_uncertain")


def _witness(image, generation, key, physical, digest, size, *, rejected_stage=False):
    fields = (
        "article-audio-rejected-stage-v1"
        if rejected_stage
        else "article-audio-custody-v1",
        image,
        generation,
        key,
        physical,
        digest,
        size,
    )
    return hashlib.sha256(
        json.dumps(fields, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _measure(held):
    size = held.stream.seek(0, 2)
    if not 0 <= size <= MAX_BUFFERED_AUDIO_BYTES:
        raise ArticleAudioError("article_audio_custody_uncertain")
    held.stream.seek(0)
    return hashlib.file_digest(held.stream, "sha256").hexdigest(), size


@contextmanager
def _leaf(parent, name, census=None):
    if parent is None or not (census or _Census(parent)).contains(name):
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
        authorize_commit: Callable[[], None] | None = None,
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
        self._commit_options = (
            {} if authorize_commit is None else {"authorize_commit": authorize_commit}
        )

    @contextmanager
    def _hold(self, keys):
        if self.lease is not None and type(self.lease) is not ListeningLease:
            raise ValueError("invalid_listening_lease")
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
            if exc.args == ("regular_file_effects_unqualified",):
                raise ArticleAudioError("article_audio_effects_unqualified") from None
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
            if create:
                require_regular_file_effects(parent)
            if not _Census(parent).contains(self.root.name):
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

    def _verify(self, directory, name, expected, census=None):
        with _leaf(directory, name, census) as (fd, before):
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

    def _seal_stage(self, directory, held, claim, data, expected, lease):
        physical = held.identity
        identity = claim.image, claim.generation, claim.file_key, physical
        try:
            if held.stream.write(data) != expected.byte_size:
                raise ArticleAudioError("article_audio_write_incomplete")
            held.flush_file()
            try:
                validate_article_audio(held.stream, expected)
            except ValueError:
                raise ArticleAudioError("invalid_article_audio") from None
            self._check_directory(directory)
            witness = _witness(*identity, expected.sha256, expected.byte_size)
            self.store.seal_audio_custody(claim, witness, lease=lease)
            return witness
        except (ValueError, OSError, sqlite3.Error):
            # Only this live, exclusively created stage can acquire a rejection
            # receipt. A later sweep cannot derive custody from unsealed bytes.
            self._check_directory(directory)
            rejected = _witness(*identity, *_measure(held), rejected_stage=True)
            if self.store.reject_audio_stage(claim, rejected, lease=lease):
                held.delete()
            raise

    def publish(
        self, candidate: GeneratedArticleAudio, *, source_item_id: int, generation: int
    ) -> ArticleAudio:
        """Validate and attach a reserved generation while the same native object is held."""
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
            if self.store.audio_inventory(generation)[0]["custody_witness"] is not None:
                raise ArticleAudioError("article_audio_publication_uncertain")
            with self._directory(create=True) as directory, ExitStack() as stack:
                names = _Census(directory)
                try:
                    if names.contains(key) or names.contains(stage):
                        raise ArticleAudioError("article_audio_name_collision")
                    held = stack.enter_context(
                        held_regular_file(directory, stage, create=True)
                    )
                except (FileExistsError, ArticleAudioError):
                    self.store.relinquish_audio(claim, lease=lease)
                    raise ArticleAudioError("article_audio_name_collision") from None
                witness = self._seal_stage(
                    directory, held, claim, candidate.wav_bytes, expected, lease
                )
                try:
                    held.rename_new(key)
                except FileExistsError:
                    # The native barrier can fail after the rename has happened.
                    names = _Census(directory)
                    if not (names.contains(stage) and names.contains(key)):
                        raise ArticleAudioError(
                            "article_audio_publication_uncertain"
                        ) from None
                    self.store.relinquish_audio(claim, lease=lease)
                    held.delete()
                    raise ArticleAudioError("article_audio_name_collision") from None
                self._check_directory(directory)
                self.store.complete_audio(
                    claim,
                    expected,
                    lease=lease,
                    custody_witness=witness,
                    **self._commit_options,
                )
                return self._audio(generation, source_item_id, key, expected)

    def _read(self, directory, row, lease, census=None):
        g, key = row["generation"], row["file_key"]
        if row["status"] != "ready":
            raise ArticleAudioError("article_audio_missing")
        m = _metadata(row)
        try:
            self._verify(directory, key, m, census)
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
        rows = self.store.audio_inventory(_int(generation, 1))
        if not rows:
            raise ArticleAudioError("article_audio_missing")
        key = rows[0]["file_key"]
        with self._hold((key,)) as lease, self._directory() as directory:
            rows = self.store.audio_inventory(generation)
            if not rows or rows[0]["file_key"] != key:
                raise ListeningConflict("article_audio_inventory_changed")
            audio = self._read(directory, rows[0], lease)
            self.store.record_audio_use(generation, lease=lease)
            return audio

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
                    **self._commit_options,
                )
                return audio
        return None

    def _clean_names(self, directory, census, row):
        """Confirm absence, or remove one independently proven local object."""
        final, stage = _names(row["file_key"])
        present = [name for name in (final, stage) if census.contains(name)]
        if len(present) > 1:
            raise ArticleAudioError("article_audio_custody_uncertain")
        if directory is None:
            # The profile directory is the immediate parent of an absent asset
            # tree. If it too is unavailable, retain the tombstone for retry.
            with pinned_directory(self.root.parent) as parent:
                _confirm_absent(parent, self.root.name)
                flush_regular_parent(parent)
            return
        self._check_directory(directory)
        if not present:
            for name in (final, stage):
                _confirm_absent(directory, name)
            flush_regular_parent(directory)
            return
        saved = row["custody_witness"]
        if saved is None:
            raise ArticleAudioError("article_audio_custody_uncertain")
        with held_regular_file(directory, present[0]) as held:
            identity = (
                self.store.db.listening_image,
                row["generation"],
                final,
                held.identity,
                *_measure(held),
            )
            if _witness(*identity) != saved and not (
                present[0] == stage
                and _witness(*identity, rejected_stage=True) == saved
            ):
                raise ArticleAudioError("article_audio_custody_uncertain")
            # Refuse a second leaf born since the planning census. The native
            # operation itself acts only on the verified, share-frozen handle.
            _confirm_absent(directory, stage if present[0] == final else final)
            self._check_directory(directory)
            held.delete()

    def _cleanup(self, lease, directory, census, receipt, rows):
        admitted = {(r["generation"], r["file_key"]): r for r in rows}
        for gen, key in self.store.pending_deletions():
            row = admitted.get((gen, key))
            if row is None:
                continue
            try:
                self._clean_names(directory, census, row)
            except ArticleAudioError:
                receipt.uncertain += 1
                receipt.retry += 1
                continue
            except OSError:
                receipt.retry += 1
                continue
            self.store.finish_audio_deletion(
                gen, key, row["custody_witness"], lease=lease
            )
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
                yield rows, lease, directory, _Census(directory)

    def reconcile(self) -> ArticleAudioCleanup:
        """Recover interrupted claims and missing artifacts under physical ownership."""
        receipt = ArticleAudioCleanup()
        with self._inventory() as (rows, lease, directory, census):
            pending = [r["generation"] for r in rows if r["status"] == "pending"]
            if self.lease is None and pending:
                self.store.retire_audio(pending, lease=lease)
                receipt.interrupted = len(pending)
            owned = {name for row in rows for name in _names(row["file_key"])}
            receipt.uncertain = len(census.names - owned)
            for row in rows:
                if row["status"] == "ready":
                    try:
                        self._read(directory, row, lease, census)
                    except ArticleAudioError as exc:
                        if str(exc) not in (
                            "article_audio_missing",
                            "article_audio_corrupt",
                        ):
                            raise
                        receipt.missing += 1
            self._cleanup(lease, directory, census, receipt, rows)
        return receipt

    def prune(self, now: int) -> ArticleAudioCleanup:
        """Retire expired and least-recently used unreferenced private bytes."""
        _int(now)
        receipt = ArticleAudioCleanup()
        with self._inventory() as (rows, lease, directory, census):
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
                    with _leaf(directory, row["file_key"], census) as (_, info):
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
                if directory is None:
                    with pinned_directory(self.root.parent) as parent:
                        _confirm_absent(parent, self.root.name)
                        require_regular_file_effects(parent)
                else:
                    self._check_directory(directory)
                    require_regular_file_effects(directory)
                self.store.retire_audio(retired, lease=lease)
            self._cleanup(lease, directory, census, receipt, rows)
        return receipt
