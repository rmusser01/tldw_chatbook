"""Physical mutation custody shared by article listening and source deletion."""

from __future__ import annotations

import errno
import re
import stat
import threading
from contextlib import ExitStack, contextmanager
from pathlib import Path

from tldw_chatbook.Backup_Recovery.native_files import (
    create_private_file,
    pinned_directory,
)
from tldw_chatbook.Backup_Recovery.runtime_producer_lifetime import ProducerLifetime
from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage
from tldw_chatbook.Utils.platform_files import fcntl, os
from .article_playback_receipt import RECEIPT_SIZE, decode_predecessor, encode_predecessor


class ListeningBusy(RuntimeError):
    """The DB has another physical listening owner or the holder is retiring."""


class ListeningProducer:
    """Cheap installed gate for the physical lifetime, including delayed borrowers."""

    def __init__(self):
        self._lifetime = ProducerLifetime()

    def _maintenance_close_admission(self):
        self._lifetime.close()

    async def _maintenance_drain(self, deadline):
        return await self._lifetime.drain(deadline)

    def _maintenance_resume(self):
        self._lifetime.resume()


listening_producer = ListeningProducer()


def lock_path(database: Path) -> Path:
    canonical = database.resolve()
    return canonical.with_name(canonical.name + ".article-listening.lock")


_ISSUER = object()


class ListeningLease:
    """Checked capability for exactly one live hold, including native worker borrowing."""

    def __init__(self, database: Path, profile: str | None, storage, owner, *, issuer):
        if issuer is not _ISSUER:
            raise TypeError("listening_lease_requires_physical_owner")
        self._database = database
        self._profile = profile
        self._database_identity = None
        self._related_paths = frozenset()
        self._session_asset_root = None
        self._storage = storage
        self._owner = owner
        self._condition = threading.Condition()
        self._open = True
        self._borrowers = 0
        self._asset_active = False
        self._receipt_lock = threading.Lock()

    def _read_predecessor(self):
        os.lseek(self._owner, 0, os.SEEK_SET)
        return decode_predecessor(os.read(self._owner, RECEIPT_SIZE + 1))

    def recover_predecessor(self):
        """Check a previous exact process and its post-reap settle before reuse."""
        with self.borrow(self._database, profile=self._profile), self._receipt_lock:
            previous = self._read_predecessor()
            if previous is None:
                return
            from tldw_chatbook.TTS.controlled_wav_profile import (
                ControlledWavError,
                wait_for_predecessor,
            )

            try:
                wait_for_predecessor(previous)
            except ControlledWavError as exc:
                raise ListeningBusy("previous_player_unconfirmed") from exc
            self._clear_predecessor(previous)

    def _clear_predecessor(self, previous):
        if previous != self._read_predecessor():
            raise ListeningBusy("player_predecessor_changed")
        os.ftruncate(self._owner, 0)
        os.fsync(self._owner)

    def seal_predecessor(self, pid, created):
        with self.borrow(self._database, profile=self._profile), self._receipt_lock:
            if self._read_predecessor() is not None:
                raise ListeningBusy("previous_player_unconfirmed")
            data = encode_predecessor(pid, created)
            os.lseek(self._owner, 0, os.SEEK_SET)
            if os.write(self._owner, data) != len(data):
                raise OSError("player_predecessor_short_write")
            os.fsync(self._owner)
            if self._read_predecessor() != (pid, created):
                raise OSError("player_predecessor_unconfirmed")

    def clear_predecessor(self, pid, created):
        with self.borrow(self._database, profile=self._profile), self._receipt_lock:
            self._clear_predecessor((pid, created))

    def __copy__(self):
        raise TypeError("listening_lease_cannot_be_copied")

    def __deepcopy__(self, memo):
        raise TypeError("listening_lease_cannot_be_copied")

    @contextmanager
    def borrow(
        self,
        database: Path,
        *,
        profile: str | None,
        related_paths: tuple[Path, ...] = (),
    ):
        with self._condition:
            if (
                not self._open
                or self._owner is None
                or database.resolve() != self._database
                or (profile is not None and self._profile != profile)
                or not self._related_paths.issuperset(related_paths)
            ):
                raise ListeningBusy("listening_lease_unavailable")
            if self._database_identity is not None:
                info = os.stat(self._database, follow_symlinks=False)
                if (
                    info.st_dev,
                    info.st_ino,
                ) != self._database_identity or info.st_nlink != 1:
                    raise ListeningBusy("listening_database_changed")
            self._storage.execution_context(self._database)
            self._borrowers += 1
        try:
            yield self
        finally:
            with self._condition:
                self._borrowers -= 1
                self._condition.notify_all()

    def revoke(self):
        with self._condition:
            self._open = False

    @contextmanager
    def admit_assets(self, database, *, profile, related_paths):
        """Hold the DB owner and independently admit each finite asset path."""
        if self._session_asset_root is None:
            with self.borrow(database, profile=profile, related_paths=related_paths):
                yield self
            return
        if (
            not related_paths
            or related_paths[0] != self._session_asset_root
            or any(
                path.parent != self._session_asset_root for path in related_paths[1:]
            )
        ):
            raise ListeningBusy("listening_asset_scope_changed")
        with self.borrow(database, profile=profile):
            with acquire_storage(database, related_paths=related_paths):
                yield self

    @contextmanager
    def asset_operation(self):
        """Admit one finite file operation per native holder without waiting on SQL."""
        with self._condition:
            if not self._open or self._asset_active:
                raise ListeningBusy("listening_asset_busy")
            self._asset_active = True
        try:
            yield
        finally:
            with self._condition:
                self._asset_active = False

    def _retire(self):
        with self._condition:
            self._open = False
            self._condition.wait_for(lambda: self._borrowers == 0)
            self._owner = None


@contextmanager
def hold_listening(
    database: Path,
    *,
    profile: str | None,
    related_paths: tuple[Path, ...] = (),
    memory: bool = False,
    session_asset_root: Path | None = None,
):
    """Hold ordinary storage and the same private lock inode through physical return."""
    canonical = database.resolve()
    lock = lock_path(canonical)
    with listening_producer._lifetime.operation(), ExitStack() as stack:
        storage = stack.enter_context(
            acquire_storage(canonical, related_paths=(lock, *related_paths))
        )
        identity = None
        if memory:
            if (
                not re.fullmatch(r"memory-[0-9a-f]{32}", canonical.name)
                or canonical.exists()
            ):
                raise ValueError("invalid_memory_listening_binding")
        else:
            db_info = os.stat(canonical, follow_symlinks=False)
            if not stat.S_ISREG(db_info.st_mode) or db_info.st_nlink != 1:
                raise ValueError("unsafe_listening_database")
            identity = db_info.st_dev, db_info.st_ino
        parent = stack.enter_context(pinned_directory(lock.parent))
        try:
            with create_private_file(lock):
                pass
        except FileExistsError:
            pass
        fd = os.open(
            lock.name, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
        stack.callback(os.close, fd)
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_mode & 0o077
            or info.st_uid != os.geteuid()
        ):
            raise ValueError("unsafe_listening_lock")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if (
                exc.errno in {errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES}
                or getattr(exc, "winerror", None) == 33
            ):
                raise ListeningBusy("listening_busy") from None
            raise
        named = os.stat(lock.name, dir_fd=parent, follow_symlinks=False)
        if (named.st_dev, named.st_ino) != (info.st_dev, info.st_ino):
            raise ValueError("listening_lock_changed")
        lease = ListeningLease(canonical, profile, storage, fd, issuer=_ISSUER)
        lease._database_identity = identity
        lease._related_paths = frozenset(related_paths)
        lease._session_asset_root = session_asset_root
        try:
            # A dead parent's OS lock alone cannot authorize a destructive
            # writer, reconciliation or a new output over a still-live player.
            lease.recover_predecessor()
            yield lease
        finally:
            lease._retire()
