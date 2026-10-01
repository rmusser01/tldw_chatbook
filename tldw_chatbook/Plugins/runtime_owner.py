"""One qualified local-profile mutation owner; never executes or kills processes."""

from __future__ import annotations

import ctypes
import errno
import json
import os
import stat
import sys
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

import portalocker

from tldw_chatbook.Plugins.registry import PluginRegistry, validate_page

MNT_LOCAL = 0x1000
_SYNC_NAMES = (
    "dropbox",
    "onedrive",
    "google drive",
    "googledrive",
    "cloudstorage",
    "mobile documents",
)


class _DarwinStatFS(ctypes.Structure):
    # macOS SDK sys/mount.h __DARWIN_STRUCT_STATFS64 (64-bit ABI).
    _fields_ = [
        ("bsize", ctypes.c_uint32),
        ("iosize", ctypes.c_int32),
        ("blocks", ctypes.c_uint64),
        ("bfree", ctypes.c_uint64),
        ("bavail", ctypes.c_uint64),
        ("files", ctypes.c_uint64),
        ("ffree", ctypes.c_uint64),
        ("fsid", ctypes.c_int32 * 2),
        ("owner", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("subtype", ctypes.c_uint32),
        ("fstypename", ctypes.c_char * 16),
        ("mntonname", ctypes.c_char * 1024),
        ("mntfromname", ctypes.c_char * 1024),
        ("flags_ext", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 7),
    ]


def _darwin_filesystem(path: Path) -> tuple[int, str]:
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    probe = library.statfs64
    probe.argtypes = [ctypes.c_char_p, ctypes.POINTER(_DarwinStatFS)]
    probe.restype = ctypes.c_int
    result = _DarwinStatFS()
    if probe(os.fsencode(path), ctypes.byref(result)) != 0:
        raise OSError(ctypes.get_errno(), "plugin filesystem probe failed")
    return result.flags, result.fstypename.decode("ascii", errors="strict")


def _qualify_root(root: Path) -> None:
    if sys.platform != "darwin" or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise PermissionError("plugin runtime platform is unqualified")
    if any(part.lower().startswith(_SYNC_NAMES) for part in root.parts):
        raise PermissionError("synchronized plugin roots are unsupported")
    ancestor = root
    while not ancestor.exists():
        ancestor = ancestor.parent
    try:
        flags, filesystem = _darwin_filesystem(ancestor)
    except (OSError, AttributeError, ValueError) as error:
        raise PermissionError("plugin filesystem could not be qualified") from error
    if not flags & MNT_LOCAL or filesystem != "apfs":
        raise PermissionError("plugin root requires a qualified local filesystem")


def _identity(path: Path) -> tuple[int, int]:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise PermissionError("plugin root must be a real directory")
    return info.st_dev, info.st_ino


def _text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 4096:
        raise ValueError(f"invalid {field}")
    return value


class PluginRuntimeOwner:
    """Acquire a stable OS lock under the resolved profile's plugin directory.

    The caller derives ``root`` from its actual resolved profile, not a config-only
    override. This synchronous worker-owned object is not thread safe. Acquiring
    the lock does not establish that a prior owner's children have stopped.
    """

    def __init__(self, root: Path) -> None:
        # Resolve parent aliases (/tmp), but never silently follow a replaced root.
        absolute = root.absolute()
        self.root = absolute.parent.resolve() / absolute.name
        self._handle: BinaryIO | None = None
        self._root_identity: tuple[int, int] | None = None
        self._pid: int | None = None
        self._registry: PluginRegistry | None = None
        self.session_id = uuid4().hex

    def try_acquire(self) -> bool:
        """Return false only for contention; fail closed on qualification/OS errors."""
        if self._handle is not None:
            self.require_owner(self.root)
            return True
        _qualify_root(self.root)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root_identity = _identity(self.root)
        path = self.root / "runtime.lock"
        descriptor = os.open(
            path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600
        )
        handle = os.fdopen(descriptor, "r+b")
        try:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise PermissionError("plugin lock must be a private regular file")
            os.fchmod(handle.fileno(), 0o600)
            try:
                portalocker.lock(handle, portalocker.LOCK_EX | portalocker.LOCK_NB)
            except portalocker.exceptions.LockException as error:
                cause = error.__cause__
                if isinstance(cause, OSError) and cause.errno in (
                    errno.EAGAIN,
                    errno.EACCES,
                ):
                    handle.close()
                    return False
                raise PermissionError("plugin lock mechanism failed") from error
            named = path.lstat()
            if (named.st_dev, named.st_ino) != (info.st_dev, info.st_ino) or _identity(
                self.root
            ) != root_identity:
                raise PermissionError("plugin lock/root changed during acquisition")
            self._handle = handle
            self._root_identity = root_identity
            self._pid = os.getpid()
            self.session_id = uuid4().hex
            return True
        except BaseException:
            handle.close()
            raise

    def require_owner(self, root: Path) -> None:
        """Check current process and exact root/lock identity at mutation boundaries."""
        if (
            self._handle is None
            or self._handle.closed
            or self._pid != os.getpid()
            or root != self.root
        ):
            raise PermissionError("plugin runtime ownership required")
        if _identity(self.root) != self._root_identity:
            raise PermissionError("plugin root changed")
        named, opened = (
            (self.root / "runtime.lock").lstat(),
            os.fstat(self._handle.fileno()),
        )
        if not stat.S_ISREG(named.st_mode) or (named.st_dev, named.st_ino) != (
            opened.st_dev,
            opened.st_ino,
        ):
            raise PermissionError("plugin lock changed")

    def _store(self) -> PluginRegistry:
        self.require_owner(self.root)
        if self._registry is None:
            self._registry = PluginRegistry(self.root / "registry.sqlite3", owner=self)
        return self._registry

    def reserve_launch(
        self,
        operation_id: str,
        installation_id: str,
        workspace_id: str | None,
        revision_digest: str,
    ) -> str:
        """Durably reserve identity before spawn; refuse unreconciled old ownership."""
        values = tuple(
            _text(value, field)
            for value, field in (
                (operation_id, "operation_id"),
                (installation_id, "installation_id"),
                (revision_digest, "revision_digest"),
            )
        )
        if workspace_id is not None:
            _text(workspace_id, "workspace_id")
        token = uuid4().hex
        with self._store().transaction() as cursor:
            blocked = cursor.execute(
                "SELECT 1 FROM processes WHERE installation_id=? AND state!='settled' AND (owner_session!=? OR state='unresolved') LIMIT 1",
                (installation_id, self.session_id),
            ).fetchone()
            if blocked:
                raise PermissionError("plugin runtime recovery required")
            cursor.execute(
                "INSERT INTO processes(token, operation_id, installation_id, workspace_id, revision_digest, owner_session, state, kind) VALUES (?, ?, ?, ?, ?, ?, 'pending', 'pending_launch')",
                (token, values[0], values[1], workspace_id, values[2], self.session_id),
            )
        return token

    def publish_process(self, token: str, provenance: dict) -> None:
        """Persist exact non-secret process identity; never infer identity from PID."""
        if not isinstance(provenance, dict) or not provenance:
            raise ValueError("process provenance must be a nonempty object")
        serialized = json.dumps(
            provenance,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(serialized.encode("utf-8")) > 65536:
            raise ValueError("process provenance exceeds 64 KiB")
        with self._store().transaction() as cursor:
            cursor.execute(
                "UPDATE processes SET provenance_json=?, state='published', kind='active_run' WHERE token=? AND state='pending' AND owner_session=?",
                (serialized, token, self.session_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("launch is not pending for this owner")

    def settle_process(self, token: str, confirmed: bool) -> None:
        """Record trusted host reconciliation, never a stale PID liveness guess.

        ``confirmed=True`` is a host assertion that all owned writers stopped.
        An OS lock or a reused PID is not such confirmation. False preserves both
        provenance and recovery blocking; no signal or cleanup is performed here.
        """
        if type(confirmed) is not bool:
            raise ValueError("confirmation must be boolean")
        with self._store().transaction() as cursor:
            cursor.execute(
                "UPDATE processes SET state=? WHERE token=? AND state!='settled'",
                ("settled" if confirmed else "unresolved", token),
            )
            if cursor.rowcount != 1:
                raise ValueError("unknown or already settled launch")

    def set_process_kind(self, token: str, kind: str) -> None:
        """Distinguish active revision leases from idle/history references."""
        if kind not in ("active_run", "idle_connection", "archived_history"):
            raise ValueError("invalid published process kind")
        with self._store().transaction() as cursor:
            cursor.execute(
                "UPDATE processes SET kind=? WHERE token=? AND state='published' AND owner_session=?",
                (kind, token, self.session_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("process is not published for this owner")

    def active_revision_leases(self, installation_id: str, revision_digest: str) -> int:
        """Count actual/pending active work; unresolved active work stays counted."""
        with self._store().transaction() as cursor:
            return cursor.execute(
                "SELECT count(*) FROM processes WHERE installation_id=? AND revision_digest=? AND state!='settled' AND kind IN ('pending_launch', 'active_run')",
                (installation_id, revision_digest),
            ).fetchone()[0]

    def unsettled_tokens(
        self, installation_id: str, workspace_id: str | None = None
    ) -> tuple[str, ...]:
        """Retain unknown/idle owners too; lack of an active lease proves no drain."""
        with self._store().transaction() as cursor:
            return tuple(
                row[0]
                for row in cursor.execute(
                    "SELECT token FROM processes WHERE installation_id=? AND state!='settled' AND (? IS NULL OR workspace_id=?)",
                    (installation_id, workspace_id, workspace_id),
                )
            )

    def list_processes(self, *, limit: int, offset: int) -> tuple[dict, ...]:
        """Expose bounded exact recovery evidence, including settled history."""
        validate_page(limit, offset)
        with self._store().transaction() as cursor:
            rows = cursor.execute(
                "SELECT * FROM processes ORDER BY rowid LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            raw = item.pop("provenance_json")
            item["provenance"] = json.loads(raw) if raw is not None else None
            result.append(item)
        return tuple(result)

    def close(self) -> None:
        """Close owned handles, retaining the stable lock file and child evidence."""
        try:
            if self._registry is not None:
                self._registry.close()
                self._registry = None
        finally:
            handle, self._handle = self._handle, None
            self._pid = None
            if handle is not None:
                # Closing a fork-inherited descriptor must not explicitly unlock
                # the parent's flock. Closing alone releases this process's handle.
                handle.close()
