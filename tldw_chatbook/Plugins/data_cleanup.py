"""Reviewed saved-root custody on the existing storage and lifecycle owners.

Native APFS identities are boot-scoped. Host terminal assertions never imply
containment of external writers or deliberately escaped descendants.
"""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import inspect
import json
import os
import re
import stat
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from uuid import uuid4

from .authority import DataRoot, PhysicalIdentity, RootBinding
from .package_files import canonical_json
from .review import OperationReceipt

DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


class _Timespec(ctypes.Structure):
    _fields_ = [("seconds", ctypes.c_int64), ("nanoseconds", ctypes.c_int64)]


class _DarwinStat(ctypes.Structure):
    # SDK sys/stat.h __DARWIN_STRUCT_STAT64, 64-bit Darwin ABI (144 bytes).
    _fields_ = [
        ("device", ctypes.c_int32),
        ("mode", ctypes.c_uint16),
        ("links", ctypes.c_uint16),
        ("inode", ctypes.c_uint64),
        ("uid", ctypes.c_uint32),
        ("gid", ctypes.c_uint32),
        ("rdev", ctypes.c_int32),
        ("atime", _Timespec),
        ("mtime", _Timespec),
        ("ctime", _Timespec),
        ("birth", _Timespec),
        ("size", ctypes.c_int64),
        ("blocks", ctypes.c_int64),
        ("blocksize", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
        ("generation", ctypes.c_uint32),
        ("spare", ctypes.c_int32),
        ("qspare", ctypes.c_int64 * 2),
    ]


def boot_identity() -> str:
    """Read and recheck the actual bounded Darwin boot-session UUID."""
    if sys.platform != "darwin" or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise PermissionError("root_platform_unqualified")
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    probe = library.sysctlbyname
    probe.argtypes = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    probe.restype = ctypes.c_int
    values = []
    for _ in range(2):
        buffer = ctypes.create_string_buffer(64)
        length = ctypes.c_size_t(len(buffer))
        if probe(b"kern.bootsessionuuid", buffer, ctypes.byref(length), None, 0) != 0:
            raise PermissionError("root_boot_identity_unavailable")
        if not 1 <= length.value <= len(buffer):
            raise PermissionError("root_boot_identity_unavailable")
        value = buffer.raw[: length.value].rstrip(b"\0").decode("ascii").lower()
        if re.fullmatch(r"[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}", value) is None:
            raise PermissionError("root_boot_identity_unavailable")
        values.append(value)
    if values[0] != values[1]:
        raise PermissionError("root_boot_identity_changed")
    return values[0]


def physical_identity(fd: int) -> dict:
    """Read exact native birth precision; never synthesize nanoseconds from float."""
    if sys.platform != "darwin" or ctypes.sizeof(_DarwinStat) != 144:
        raise PermissionError("root_platform_unqualified")
    probe = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True).fstat
    probe.argtypes = [ctypes.c_int, ctypes.POINTER(_DarwinStat)]
    probe.restype = ctypes.c_int
    native = _DarwinStat()
    if probe(fd, ctypes.byref(native)) != 0:
        raise PermissionError("root_identity_unavailable")
    observed = os.fstat(fd)
    if not stat.S_ISDIR(native.mode) or (native.device, native.inode, native.mode) != (
        observed.st_dev,
        observed.st_ino,
        observed.st_mode,
    ):
        raise PermissionError("root_identity_unqualified")
    return PhysicalIdentity(
        device=native.device,
        inode=native.inode,
        birth_seconds=native.birth.seconds,
        birth_nanoseconds=native.birth.nanoseconds,
    ).model_dump()


@dataclass(frozen=True)
class DataRootRef:
    installation_id: str
    root_id: str
    generation: int
    path: Path


def root_ref(row) -> DataRootRef:
    return DataRootRef(
        row["installation_id"], row["root_id"], row["generation"], Path(row["path"])
    )


def member(row):
    """Immutable reviewed member, excluding this operation's changing fence/phase."""
    return {
        key: value
        for key, value in row.items()
        if key not in {"cleanup", "deletion_fenced"}
    }


@dataclass(frozen=True)
class RootReview:
    installation_id: str
    authority_marker: object
    authority_json: str
    token: str
    expires_at: float
    targets_json: str
    phase: str
    group_id: str
    target_digest: str
    operation_id: str = ""
    kind: str = "root_data"
    # Exact mutation after drain; not permission from plugin text.
    attachment: str | None = None
    action: str = "delete"

    @property
    def targets(self):
        return json.loads(self.targets_json)

    @property
    def result(self):
        return {
            "group_id": self.group_id,
            "phase": self.phase,
            "target_digest": self.target_digest,
            "root_ids": sorted(row["root_id"] for row in self.targets),
        }


@dataclass
class DataOperation:
    review: RootReview
    fence_token: str
    receipt: OperationReceipt
    task: asyncio.Task | None = None
    cancel_requested: bool = False
    cleanup_tasks: set = field(default_factory=set)
    cleanup_errors: list[str] = field(default_factory=list)


@contextmanager
def open_root(owner, row, *, absent=False, rebind=False):
    """Pin owner/data/leaf without following links; refuse missing ancestry."""
    owner.require_owner(owner.root)
    path = Path(row["path"])
    if path != owner.root / "data" / row["root_id"]:
        raise PermissionError("root_path_unowned")
    binding = (row.get("custody") or {}).get("binding")
    completed_absence = (row.get("custody") or {}).get("state") == "cleaned_absent"
    boot = boot_identity()
    if binding is not None and binding["boot_id"] != boot and not rebind:
        raise PermissionError("root_boot_identity_changed")
    descriptors = []
    try:
        descriptors.append(os.open(owner.root, DIR_FLAGS))
        owner.require_owner(owner.root)
        if (
            os.fstat(descriptors[0]).st_dev,
            os.fstat(descriptors[0]).st_ino,
        ) != owner._root_identity:
            raise PermissionError("root_anchor_changed")
        descriptors.append(os.open("data", DIR_FLAGS, dir_fd=descriptors[-1]))
        anchors = [physical_identity(fd) for fd in descriptors]
        if binding is not None and anchors != binding["anchors"]:
            raise PermissionError("root_anchor_changed")
        try:
            descriptors.append(
                os.open(row["root_id"], DIR_FLAGS, dir_fd=descriptors[-1])
            )
        except FileNotFoundError:
            if not (absent or completed_absence) or binding is None:
                raise PermissionError("root_missing_without_deleting_intent") from None
            # A tombstone retains historical leaf identity; only its observed
            # ancestry and boot scope are rebound, never an unexpected leaf.
            yield (
                descriptors,
                (dict(binding, boot_id=boot) if completed_absence else None),
            )
            return
        if completed_absence:
            raise PermissionError("root_unexpected_after_cleanup")
        leaf = physical_identity(descriptors[-1])
        if binding is not None and leaf != binding["leaf"]:
            raise PermissionError("root_incarnation_changed")
        observed = RootBinding(
            platform="darwin-apfs-boot-v1", boot_id=boot, anchors=anchors, leaf=leaf
        ).model_dump()
        yield descriptors, observed
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


class RootUsage:
    """Exact-root live claims and durable joins owned by one coordinator worker."""

    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.fences = coordinator.fences
        self.pending = {}
        self.live_grants = {}
        self.grant_epochs = {}
        self.root_fences = {}
        self.epochs = self.fences.root_epochs
        self.recovery = {}
        self.closed = False
        self.initialized = False
        self.dirty = False
        self.operations = {}
        self.proofs = {}
        self.committing = set()
        self.cancel_callbacks = {}

    def initialize(self, snapshot, *, fresh=False, reconstructed=False):
        if self.initialized:
            if reconstructed:
                self.recovery.update(
                    {
                        row["root_id"]: "root_owner_evidence_reconstructed"
                        for row in snapshot["data_roots"]
                    }
                )
            return
        self.initialized = True
        store = self.coordinator.authority
        roots = snapshot["data_roots"]
        try:
            if fresh:
                if (
                    roots
                    or self.coordinator.registry._connection.execute(
                        "SELECT 1 FROM data_roots LIMIT 1"
                    ).fetchone()
                ):
                    raise PermissionError("retained root custody at bootstrap")
                store.save_runtime_checkpoint(
                    self.coordinator.owner.session_id, "clean"
                )
            checkpoint = store.load_runtime_checkpoint()
            clean = (
                checkpoint is not None
                and checkpoint.phase == "clean"
                and checkpoint.namespace_id == store.namespace_id()
                and checkpoint.marker == store.load_marker()
            )
        # backend failures quarantine, never infer clean.
        except Exception:  # noqa: BLE001
            clean = False
        if clean:
            # A qualified clean checkpoint proves prior host quiescence even if
            # stale rows survived an older coherent SQLite backup.
            with self.coordinator.registry.transaction() as cursor:
                cursor.execute(
                    "UPDATE root_users SET state='released' WHERE owner_token IN (SELECT token FROM processes WHERE owner_session!=?)",
                    (self.coordinator.owner.session_id,),
                )
                cursor.execute(
                    "UPDATE processes SET state='settled' WHERE owner_session!=?",
                    (self.coordinator.owner.session_id,),
                )
        else:
            self.recovery.update(
                {row["root_id"]: "root_prior_session_unqualified" for row in roots}
            )
        for row in roots:
            if (row.get("custody") or {}).get("state") in {"present", "cleaned_absent"}:
                try:
                    with open_root(self.coordinator.owner, row):
                        pass
                except (OSError, ValueError) as error:
                    self.recovery[row["root_id"]] = (
                        str(error)
                        if isinstance(error, PermissionError)
                        else "root_identity_unqualified"
                    )

    def require_dirty(self):
        self.coordinator._require_worker()
        with self.fences.live_lock:
            if self.closed:
                raise PermissionError("root_admission_closed")
        if not self.dirty:
            try:
                self.coordinator.authority.save_runtime_checkpoint(
                    self.coordinator.owner.session_id, "dirty"
                )
            except BaseException:
                with self.fences.live_lock:
                    self.closed = True
                raise
            self.dirty = True

    def fence(self, roots: tuple[DataRootRef, ...]) -> str:
        token = uuid4().hex
        with self.fences.live_lock:
            for ref in roots:
                if ref.root_id in self.root_fences:
                    raise PermissionError("root_already_fenced")
            for ref in roots:
                self.root_fences[ref.root_id] = token
                self.epochs[ref.root_id] = self.epochs.get(ref.root_id, 0) + 1
        return token

    def is_fenced(self, root_id: str) -> bool:
        with self.fences.live_lock:
            return (
                self.closed or root_id in self.root_fences or root_id in self.recovery
            )

    def unfence(self, token):
        with self.fences.live_lock:
            self.root_fences = {
                key: value for key, value in self.root_fences.items() if value != token
            }

    def validate_ref(self, ref, *, access=False):
        if not isinstance(ref, DataRootRef):
            raise TypeError("exact root reference required")
        row = next(
            (
                row
                for row in self.coordinator.published_snapshot()["data_roots"]
                if row["root_id"] == ref.root_id
            ),
            None,
        )
        if row is None or root_ref(row) != ref:
            raise PermissionError("root_review_stale")
        if not row.get("custody") or row["custody"]["state"] != "present":
            raise PermissionError("root_custody_unavailable")
        if access and (row["deletion_fenced"] or self.is_fenced(ref.root_id)):
            raise PermissionError(self.recovery.get(ref.root_id, "root_access_fenced"))
        with open_root(self.coordinator.owner, row):
            pass
        return row

    def begin_grants(self, refs, installation_id, workspace_id, coverage):
        if coverage not in {"unknown", "qualified_none", "known"}:
            raise ValueError("invalid root coverage")
        if refs and coverage != "known" or coverage == "known" and not refs:
            raise ValueError("incomplete root grant set")
        if (
            type(refs) is not tuple
            or len(refs) > 256
            or len({r.root_id for r in refs}) != len(refs)
        ):
            raise ValueError("invalid root grant set")
        claim = uuid4().hex
        with self.fences.live_lock:
            if self.closed or any(self.is_fenced(ref.root_id) for ref in refs):
                raise PermissionError("root_access_fenced")
            self.pending[claim] = tuple(ref.root_id for ref in refs)
            versions = tuple(
                (ref.root_id, self.epochs.get(ref.root_id, 0)) for ref in refs
            )
        try:
            if (
                refs
                or coverage == "unknown"
                and self.coordinator.published_snapshot()["data_roots"]
            ):
                self.require_dirty()
            grants = []
            for ref in refs:
                row = self.validate_ref(ref, access=True)
                if row["custody"]["attached_installation_id"] != installation_id or row[
                    "workspace_id"
                ] not in {None, workspace_id}:
                    raise PermissionError("root_grant_scope_changed")
                grants.append(
                    {
                        "root_id": ref.root_id,
                        "root_generation": ref.generation,
                        "binding_digest": digest(row["custody"]["binding"]),
                        "access": "read_write",
                    }
                )
            return claim, versions, sorted(grants, key=lambda row: row["root_id"])
        except BaseException:
            with self.fences.live_lock:
                self.pending.pop(claim, None)
            raise

    def publish_grants(self, claim, versions, token, grants):
        with self.fences.live_lock:
            held = self.live_grants.get(token, ())
            captured = dict(versions)
            captured.update(
                (root, epoch)
                for root, epoch in self.grant_epochs.get(token, ())
                if root in held
            )
            # Preserve every durable claim even when publication is refused.
            # A failed initial reservation settles these through its owner;
            # later-grant failure must not hide access from process publication.
            self.live_grants[token] = tuple(
                sorted(set(held) | {root for root, _ in versions})
            )
            self.grant_epochs[token] = tuple(sorted(captured.items()))
            if self.closed or any(
                self.is_fenced(root_id) or self.epochs.get(root_id, 0) != epoch
                for root_id, epoch in self.grant_epochs[token]
            ):
                raise PermissionError("root_access_fenced")
            self.pending.pop(claim, None)

    def retain_cancel(self, owner_token: str, callback) -> None:
        """Retain the producer's exact host cancellation request, not a PID kill."""
        self.coordinator._require_worker()
        if not callable(callback) or owner_token not in self.live_grants:
            raise ValueError("root cancellation owner unavailable")
        with self.fences.live_lock:
            self.cancel_callbacks[owner_token] = callback

    def cancel_work(self, operation_id: str) -> None:
        self.coordinator._require_worker()
        operation = self.operations[operation_id]
        ids = {row["root_id"] for row in operation.review.targets}
        with self.fences.live_lock:
            callbacks = {
                owner: self.cancel_callbacks[owner]
                for owner, roots in self.live_grants.items()
                if ids.intersection(roots) and owner in self.cancel_callbacks
            }
            for record in self.fences.runs.values():
                if ids.intersection(self.live_grants.get(record.lease_token, ())):
                    callbacks.setdefault(record.lease_token, record.cancel)

        def done(task):
            operation.cleanup_tasks.discard(task)
            try:
                task.result()
            # retain host cancellation failures.
            except BaseException as error:  # noqa: BLE001
                operation.cleanup_errors.append(type(error).__name__)

        for callback in callbacks.values():
            try:
                result = callback()
                if inspect.isawaitable(result):
                    task = asyncio.ensure_future(result)
                    operation.cleanup_tasks.add(task)
                    task.add_done_callback(done)
            # retain host cancellation failures.
            except BaseException as error:  # noqa: BLE001
                operation.cleanup_errors.append(type(error).__name__)

    def check_publication(self, token):
        with self.fences.live_lock:
            versions = self.grant_epochs.get(token, ())
            if versions and (
                self.closed
                or any(
                    self.is_fenced(root_id) or self.epochs.get(root_id, 0) != epoch
                    for root_id, epoch in versions
                )
            ):
                raise PermissionError("root_late_publication_fenced")

    def abandon_grants(self, claim):
        with self.fences.live_lock:
            self.pending.pop(claim, None)

    def acquire(self, root: DataRootRef, owner_token: str) -> str:
        owner = self.coordinator.owner
        row = (
            owner._store()
            ._connection.execute(
                "SELECT * FROM processes WHERE token=?", (owner_token,)
            )
            .fetchone()
        )
        if (
            row is None
            or row["owner_session"] != owner.session_id
            or row["state"] not in {"pending", "published"}
        ):
            raise PermissionError("root_owner_unavailable")
        if row["root_coverage"] == "unknown":
            raise PermissionError("root_owner_coverage_unknown")
        claim, versions, grants = self.begin_grants(
            (root,), row["installation_id"], row["workspace_id"], "known"
        )
        try:
            with owner._store().transaction() as cursor:
                existing = self._validate_owner(cursor, row)
                all_grants = sorted(existing + grants, key=lambda item: item["root_id"])
                if (
                    len(all_grants) > 256
                    or len(canonical_json(all_grants).encode()) > 256 * 1024
                ):
                    raise ValueError("root grant limit")
                token = self._insert(cursor, owner_token, grants[0])
                cursor.execute(
                    "UPDATE processes SET root_coverage='known', root_grants_json=? WHERE token=?",
                    (canonical_json(all_grants), owner_token),
                )
            self.publish_grants(claim, versions, owner_token, all_grants)
            return token
        finally:
            self.abandon_grants(claim)

    @staticmethod
    def _insert(cursor, owner_token, grant):
        token = uuid4().hex
        cursor.execute(
            "INSERT INTO root_users VALUES (?, ?, ?, ?, ?, ?, 'held')",
            (
                token,
                grant["root_id"],
                grant["root_generation"],
                grant["binding_digest"],
                owner_token,
                grant["access"],
            ),
        )
        return token

    @staticmethod
    def _validate_owner(cursor, owner):
        raw = owner["root_grants_json"]
        if raw is not None and len(raw.encode()) > 256 * 1024:
            raise PermissionError("root_owner_coverage_incomplete")
        expected = json.loads(raw) if raw is not None else None
        if expected is not None and (type(expected) is not list or len(expected) > 256):
            raise PermissionError("root_owner_coverage_incomplete")
        actual = [
            dict(row)
            for row in cursor.execute(
                "SELECT root_id, root_generation, binding_digest, access FROM root_users WHERE owner_token=? ORDER BY root_id",
                (owner["token"],),
            )
        ]
        if (
            owner["root_coverage"] == "unknown"
            or expected != actual
            or (owner["root_coverage"] == "qualified_none") != (not actual)
        ):
            raise PermissionError("root_owner_coverage_incomplete")
        return actual

    def release(self, token: str, confirmed: bool) -> None:
        if type(confirmed) is not bool:
            raise ValueError("confirmation must be boolean")
        with self.coordinator.registry.transaction() as cursor:
            row = cursor.execute(
                "SELECT owner_token FROM root_users WHERE usage_token=?", (token,)
            ).fetchone()
            if row is None:
                raise ValueError("unknown root usage")
            owner = cursor.execute(
                "SELECT * FROM processes WHERE token=?", (row[0],)
            ).fetchone()
            self._validate_owner(cursor, owner)
            cursor.execute(
                "UPDATE root_users SET state=? WHERE usage_token=?",
                ("released" if confirmed else "unresolved", token),
            )
        # A process capable of future writes retains its grant until terminal;
        # per-grant true is appropriate only for a finite host handle lifetime.
        if confirmed:
            held = tuple(
                row[0]
                for row in self.coordinator.registry._connection.execute(
                    "SELECT root_id FROM root_users WHERE owner_token=? AND state!='released'",
                    (owner["token"],),
                )
            )
            with self.fences.live_lock:
                self.live_grants[owner["token"]] = held
                self.grant_epochs[owner["token"]] = tuple(
                    (root, epoch)
                    for root, epoch in self.grant_epochs.get(owner["token"], ())
                    if root in held
                )

    def blockers(self, roots, *, limit=50, offset=0):
        from .registry import validate_page

        validate_page(limit, offset)
        ids = {ref.root_id for ref in roots}
        owners = {ref.installation_id for ref in roots}
        owners.update(
            (row.get("custody") or {}).get("attached_installation_id")
            for row in self.coordinator.published_snapshot()["data_roots"]
            if row["root_id"] in ids
        )
        result = []
        with self.fences.live_lock:
            for claim, held in {**self.pending, **self.live_grants}.items():
                if ids.intersection(held):
                    result.append({"owner_token": claim, "reason": "live_root_user"})
            for root_id in ids:
                if root_id in self.recovery:
                    result.append(
                        {"root_id": root_id, "reason": self.recovery[root_id]}
                    )
        with self.coordinator.registry.transaction() as cursor:
            for owner in cursor.execute(
                "SELECT * FROM processes WHERE state!='settled'"
            ).fetchall():
                joins = cursor.execute(
                    "SELECT * FROM root_users WHERE owner_token=?", (owner["token"],)
                ).fetchall()
                relevant = owner["installation_id"] in owners or any(
                    row["root_id"] in ids for row in joins
                )
                if not relevant:
                    continue
                try:
                    self._validate_owner(cursor, owner)
                    blocked = any(
                        row["root_id"] in ids and row["state"] != "released"
                        for row in joins
                    )
                    reason = "root_user"
                except (ValueError, TypeError, PermissionError):
                    blocked, reason = True, "root_owner_coverage_incomplete"
                if blocked:
                    result.append(
                        {
                            "owner_token": owner["token"],
                            "workspace_id": owner["workspace_id"],
                            "reason": reason,
                        }
                    )
        return tuple(result[offset : offset + limit])

    def shutdown(self):
        self.coordinator._require_worker()
        with self.fences.live_lock:
            self.closed = True
            if (
                self.pending
                or any(self.live_grants.values())
                or any(
                    operation.cleanup_tasks for operation in self.operations.values()
                )
            ):
                raise PermissionError("root_users_not_drained")
        if self.coordinator._published is None:
            if self.coordinator.registry._connection.execute(
                "SELECT 1 FROM data_roots LIMIT 1"
            ).fetchone():
                raise PermissionError("root_authority_unavailable")
            return
        snapshot = self.coordinator.published_snapshot()
        refs = tuple(root_ref(row) for row in snapshot["data_roots"])
        if self.blockers(refs) or any(
            row.get("cleanup") for row in snapshot["data_roots"]
        ):
            raise PermissionError("root_cleanup_or_recovery_pending")
        try:
            self.coordinator.authority.save_runtime_checkpoint(
                self.coordinator.owner.session_id, "clean"
            )
        # third-party backend failure must preserve dirty.
        except Exception:  # noqa: BLE001
            if refs:
                try:
                    self.coordinator.authority.save_runtime_checkpoint(
                        self.coordinator.owner.session_id, "dirty"
                    )
                finally:
                    raise
            # No retained root has relied on this unsupported backend.


def issue_root_review(
    coordinator,
    targets,
    phase,
    *,
    group_id=None,
    target_digest=None,
    expires_at=None,
    action="delete",
    attachment=None,
):
    baseline = coordinator.published_snapshot()
    if (
        not 1 <= len(targets) <= 256
        or len({row["root_id"] for row in targets}) != len(targets)
        or len({row["installation_id"] for row in targets}) != 1
    ):
        raise ValueError("invalid exact root target set")
    targets = sorted(targets, key=lambda row: row["root_id"])
    serialized = canonical_json(targets)
    if len(serialized.encode()) > 256 * 1024:
        raise ValueError("root target byte limit")
    review = RootReview(
        targets[0]["installation_id"],
        coordinator.authority.load_marker(),
        canonical_json(baseline),
        uuid4().hex,
        expires_at if expires_at is not None else time.monotonic() + 900,
        serialized,
        phase,
        group_id or uuid4().hex,
        target_digest or digest([member(row) for row in targets]),
        action=action,
        attachment=attachment,
    )
    result = {
        "installation_id": review.installation_id,
        "kind": "root_data",
        "revision_digest": None,
        "result": "committed",
        "root_result": review.result,
    }
    operation_id = coordinator.authority.issue_operation_id(
        review.authority_marker.generation + 1,
        digest(
            {
                "marker": review.authority_marker.model_dump(),
                "authority": digest(baseline),
                "token": review.token,
                "targets": targets,
                "result": review.result,
                "action": action,
                "attachment": attachment,
            }
        ),
        result,
        uuid4().hex + uuid4().hex,
    )
    review = replace(review, operation_id=operation_id)
    coordinator._reviews[review.token] = review
    return review


def apply_root_review(coordinator, cursor, review):
    for row in review.targets:
        DataRoot.model_validate(row)
        cursor.execute(
            "INSERT INTO data_roots(root_id, installation_id, workspace_id, path, generation, deletion_fenced, custody_json, cleanup_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(root_id) DO UPDATE SET workspace_id=excluded.workspace_id, generation=excluded.generation, deletion_fenced=excluded.deletion_fenced, custody_json=excluded.custody_json, cleanup_json=excluded.cleanup_json",
            (
                row["root_id"],
                row["installation_id"],
                row["workspace_id"],
                row["path"],
                row["generation"],
                row["deletion_fenced"],
                (
                    canonical_json(row["custody"])
                    if row.get("custody") is not None
                    else None
                ),
                (
                    canonical_json(row["cleanup"])
                    if row.get("cleanup") is not None
                    else None
                ),
            ),
        )


def review_creation(coordinator, installation_id, workspace_id):
    baseline = coordinator.published_snapshot()
    if not any(
        row["installation_id"] == installation_id for row in baseline["installations"]
    ):
        raise ValueError("installation unavailable")
    root_id = uuid4().hex
    candidate = coordinator.owner.root / "data" / root_id
    if any(
        candidate == Path(row["path"])
        or candidate in Path(row["path"]).parents
        or Path(row["path"]) in candidate.parents
        for row in baseline["data_roots"]
    ):
        raise PermissionError("root_binding_conflict")
    row = {
        "root_id": root_id,
        "installation_id": installation_id,
        "workspace_id": workspace_id,
        "path": str(coordinator.owner.root / "data" / root_id),
        "generation": 0,
        "deletion_fenced": True,
        "custody": {
            "attached_installation_id": installation_id,
            "state": "creating",
            "binding": None,
        },
    }
    DataRoot.model_validate(row)
    review = issue_root_review(coordinator, [row], "creating", action="create")
    row["cleanup"] = {
        "group_id": review.group_id,
        "phase": "creating",
        "action": "create",
        "attachment": None,
        "target_count": 1,
        "target_digest": review.target_digest,
        "reviewed_generation": 0,
        "binding_digest": digest(None),
    }
    coordinator._reviews.pop(review.token)
    return issue_root_review(
        coordinator,
        [row],
        "creating",
        group_id=review.group_id,
        target_digest=review.target_digest,
        action="create",
    )


async def create_data(coordinator, review, operation_id):
    if (
        not isinstance(review, RootReview)
        or review.phase != "creating"
        or review.action != "create"
        or review.operation_id != operation_id
    ):
        raise ValueError("original root creation review required")
    usage = coordinator.root_usage
    usage.require_dirty()
    await commit_root_phase(coordinator, review, operation_id)
    row = review.targets[0]
    owner = coordinator.owner
    owner.require_owner(owner.root)
    # A failed creation stays authenticated/fenced; never adopt an existing leaf.
    fd = os.open(owner.root, DIR_FLAGS)
    try:
        owner.require_owner(owner.root)
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != owner._root_identity:
            raise PermissionError("root_anchor_changed")
        try:
            os.mkdir("data", mode=0o700, dir_fd=fd)
        except FileExistsError:
            pass
        data_fd = os.open("data", DIR_FLAGS, dir_fd=fd)
        try:
            os.mkdir(row["root_id"], mode=0o700, dir_fd=data_fd)
            os.fsync(data_fd)
        finally:
            os.close(data_fd)
        os.fsync(fd)
    finally:
        os.close(fd)
    with open_root(owner, row) as (_, binding):
        row["custody"] = dict(row["custody"], state="present", binding=binding)
    row.pop("cleanup")
    row["deletion_fenced"] = False
    final = issue_root_review(
        coordinator,
        [row],
        "created",
        group_id=review.group_id,
        target_digest=review.target_digest,
        expires_at=review.expires_at,
        action="create",
    )
    await commit_root_phase(coordinator, final, final.operation_id)
    return root_ref(row)


def review_deletion(coordinator, refs, *, action="delete", attachment=None):
    usage = coordinator.root_usage
    rows = [usage.validate_ref(ref) for ref in refs]
    for row in rows:
        if (
            row.get("cleanup")
            or row["deletion_fenced"]
            or usage.is_fenced(row["root_id"])
        ):
            raise PermissionError(
                usage.recovery.get(row["root_id"], "root_already_fenced")
            )
    if (
        action == "attach"
        and attachment is not None
        and not any(
            row["installation_id"] == attachment
            for row in coordinator.published_snapshot()["installations"]
        )
    ):
        raise ValueError("attachment installation unavailable")
    provisional = issue_root_review(
        coordinator, rows, "waiting", action=action, attachment=attachment
    )
    coordinator._reviews.pop(provisional.token)
    return issue_root_review(
        coordinator,
        phase_rows(provisional, "waiting"),
        "waiting",
        group_id=provisional.group_id,
        target_digest=provisional.target_digest,
        expires_at=provisional.expires_at,
        action=action,
        attachment=attachment,
    )


def phase_rows(review, phase):
    rows = review.targets
    for row in rows:
        row["deletion_fenced"] = True
        row["cleanup"] = {
            "group_id": review.group_id,
            "phase": phase,
            "action": review.action,
            "attachment": review.attachment,
            "target_count": len(rows),
            "target_digest": review.target_digest,
            "reviewed_generation": row["generation"],
            "binding_digest": digest(row["custody"]["binding"]),
        }
    return rows


def current_group(coordinator, original):
    rows = [
        row
        for row in coordinator.published_snapshot()["data_roots"]
        if (row.get("cleanup") or {}).get("group_id") == original.group_id
    ]
    if (
        len(rows) != len(original.targets)
        or digest([member(row) for row in rows]) != original.target_digest
    ):
        raise PermissionError("root_cleanup_group_changed")
    if any(
        row["cleanup"]["action"] != original.action
        or row["cleanup"]["attachment"] != original.attachment
        or row["cleanup"]["target_count"] != len(rows)
        or row["cleanup"]["target_digest"] != original.target_digest
        or row["generation"] != row["cleanup"]["reviewed_generation"]
        or digest(row["custody"]["binding"]) != row["cleanup"]["binding_digest"]
        or not row["deletion_fenced"]
        for row in rows
    ):
        raise PermissionError("root_cleanup_group_changed")
    return rows


def remove_root(coordinator, original, row):
    """Recheck complete group and exact ancestry before every descriptor unlink."""

    def check():
        current = current_group(coordinator, original)
        if any(item["cleanup"]["phase"] != "deleting" for item in current):
            raise PermissionError("root_deleting_intent_required")
        if coordinator.root_usage.blockers(tuple(root_ref(item) for item in current)):
            raise PermissionError("root_users_not_drained")
        with open_root(coordinator.owner, row):
            pass

    group = current_group(coordinator, original)
    if any(
        item["cleanup"]["phase"] != "deleting" for item in group
    ) or coordinator.root_usage.blockers(tuple(root_ref(item) for item in group)):
        raise PermissionError("root_deletion_not_ready")
    with open_root(coordinator.owner, row, absent=True) as (fds, binding):
        if binding is None:
            # Missing leaf only under the exact authenticated deleting ancestry.
            os.fsync(fds[-1])
            return
        root_fd = fds[-1]

        def descend(fd):
            for name in os.listdir(fd):
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    child_fd = os.open(name, DIR_FLAGS, dir_fd=fd)
                    try:
                        opened = os.fstat(child_fd)
                        if (info.st_dev, info.st_ino) != (opened.st_dev, opened.st_ino):
                            raise PermissionError("root_child_changed")
                        descend(child_fd)
                        check()
                        named = os.stat(name, dir_fd=fd, follow_symlinks=False)
                        if (named.st_dev, named.st_ino) != (
                            opened.st_dev,
                            opened.st_ino,
                        ):
                            raise PermissionError("root_child_changed")
                        os.rmdir(name, dir_fd=fd)
                        os.fsync(fd)
                    finally:
                        os.close(child_fd)
                else:
                    check()
                    current = os.stat(name, dir_fd=fd, follow_symlinks=False)
                    if (current.st_dev, current.st_ino, current.st_mode) != (
                        info.st_dev,
                        info.st_ino,
                        info.st_mode,
                    ):
                        raise PermissionError("root_child_changed")
                    os.unlink(name, dir_fd=fd)
                    os.fsync(fd)
                    coordinator._milestone("data_unlinked")

        descend(root_fd)
        check()
        os.rmdir(row["root_id"], dir_fd=fds[-2])
        os.fsync(fds[-2])
        coordinator._milestone("data_root_removed")


def begin_data_operation(coordinator, refs, operation_id):
    with coordinator.fences.live_lock:
        return _begin_data_operation(coordinator, refs, operation_id)


def _begin_data_operation(coordinator, refs, operation_id):
    usage = coordinator.root_usage
    original = next(
        (
            review
            for review in coordinator._reviews.values()
            if review.operation_id == operation_id and isinstance(review, RootReview)
        ),
        None,
    )
    if (
        original is None
        or original.phase not in {"waiting", "deleting"}
        or tuple(root_ref(row) for row in original.targets)
        != tuple(sorted(refs, key=lambda ref: ref.root_id))
    ):
        raise ValueError("original exact root review required")
    operation = usage.operations.get(operation_id)
    if operation is None:
        if time.monotonic() >= original.expires_at:
            raise ValueError("root_review_expired")
        existing = {
            usage.root_fences[ref.root_id]
            for ref in refs
            if ref.root_id in usage.root_fences
        }
        if existing:
            prior = next(
                (
                    item
                    for item in usage.operations.values()
                    if item.fence_token in existing
                    and item.review.group_id == original.group_id
                ),
                None,
            )
            if (
                len(existing) != 1
                or prior is None
                or prior.task is not None
                and not prior.task.done()
            ):
                raise PermissionError("root_already_fenced")
            token = prior.fence_token
        else:
            token = usage.fence(refs)
        operation = DataOperation(
            original, token, OperationReceipt(operation_id, "live_fenced", False)
        )
        usage.operations[operation_id] = operation
    return operation


async def delete_data(coordinator, refs, operation_id):
    operation = begin_data_operation(coordinator, refs, operation_id)
    usage = coordinator.root_usage
    original = operation.review
    if operation.task is None:

        async def run():
            usage.require_dirty()
            waiting = original
            try:
                operation.receipt = await commit_root_phase(
                    coordinator, waiting, waiting.operation_id
                )
                operation.receipt = replace(
                    operation.receipt, phase="waiting", cleanup_pending=True
                )
                coordinator._milestone("data_fenced")
                while usage.blockers(refs) or operation.cleanup_tasks:
                    if operation.cancel_requested:
                        break
                    await asyncio.sleep(0.01)
                rows = current_group(coordinator, original)
                if operation.cancel_requested and original.phase != "deleting":
                    for row in rows:
                        row.pop("cleanup")
                        row["deletion_fenced"] = False
                    phase = "cancelled"
                else:
                    if time.monotonic() >= original.expires_at:
                        raise ValueError("root_review_expired")
                    for row in rows:
                        with open_root(
                            coordinator.owner, row, absent=original.phase == "deleting"
                        ):
                            pass
                    if original.action == "attach":
                        for row in rows:
                            row.pop("cleanup")
                            row["deletion_fenced"] = False
                            row["generation"] += 1
                            row["custody"]["attached_installation_id"] = (
                                original.attachment
                            )
                        phase = "attached"
                    else:
                        deleting = issue_root_review(
                            coordinator,
                            phase_rows(original, "deleting"),
                            "deleting",
                            group_id=original.group_id,
                            target_digest=original.target_digest,
                            expires_at=original.expires_at,
                        )
                        operation.receipt = await commit_root_phase(
                            coordinator, deleting, deleting.operation_id
                        )
                        operation.receipt = replace(
                            operation.receipt, phase="deleting", cleanup_pending=True
                        )
                        coordinator._milestone("data_deleting")
                        if operation.cancel_requested:
                            return operation.receipt
                        for row in rows:
                            remove_root(coordinator, original, row)
                        for row in rows:
                            row.pop("cleanup")
                            row["deletion_fenced"] = False
                            row["generation"] += 1
                            row["custody"]["state"] = "cleaned_absent"
                        phase = "complete"
                final = issue_root_review(
                    coordinator,
                    rows,
                    phase,
                    group_id=original.group_id,
                    target_digest=original.target_digest,
                    expires_at=original.expires_at,
                )
                operation.receipt = await commit_root_phase(
                    coordinator, final, final.operation_id
                )
                operation.receipt = replace(
                    operation.receipt,
                    phase=phase,
                    runtime_stopped=phase != "cancelled",
                    cleanup_errors=tuple(operation.cleanup_errors),
                )
                usage.unfence(operation.fence_token)
                return operation.receipt
            except BaseException as error:
                operation.receipt = replace(
                    operation.receipt,
                    cleanup_pending=True,
                    cleanup_errors=(type(error).__name__,),
                )
                raise

        operation.task = asyncio.create_task(run())
        operation.task.add_done_callback(
            lambda task: None if task.cancelled() else task.exception()
        )
    return await asyncio.shield(operation.task)


def review_reconciliation(coordinator, refs, confirm_quiescence):
    """Review host-supplied whole-root terminal proof, never a PID/zero-row guess."""
    if not callable(confirm_quiescence) or confirm_quiescence(refs) is not True:
        raise PermissionError("root_quiescence_unproven")
    current = {
        row["root_id"]: row for row in coordinator.published_snapshot()["data_roots"]
    }
    rows = []
    for ref in refs:
        row = current.get(ref.root_id)
        if row is None or root_ref(row) != ref:
            raise PermissionError("root_review_stale")
        cleanup = row.get("cleanup")
        with open_root(
            coordinator.owner,
            row,
            absent=bool(cleanup and cleanup["phase"] == "deleting"),
            rebind=True,
        ) as (_, binding):
            old_binding = (row.get("custody") or {}).get("binding")
            if cleanup and binding is not None and old_binding != binding:
                raise PermissionError("root_pending_cleanup_cannot_rebind")
            if not cleanup:
                row["custody"] = dict(
                    row.get("custody")
                    or {
                        "attached_installation_id": row["installation_id"],
                        "state": "present",
                    },
                    binding=binding,
                )
                row["generation"] += 1
        rows.append(row)
    review = issue_root_review(coordinator, rows, "reconciled", action="reconcile")
    coordinator.root_usage.proofs[review.token] = (tuple(refs), confirm_quiescence)
    return review


async def reconcile_data(coordinator, review, operation_id):
    usage = coordinator.root_usage
    if (
        coordinator._reviews.get(review.token) != review
        or review.operation_id != operation_id
        or review.action != "reconcile"
    ):
        raise ValueError("original root reconciliation review required")
    refs, proof = usage.proofs[review.token]
    new_fences = tuple(ref for ref in refs if ref.root_id not in usage.root_fences)
    token = usage.fence(new_fences)
    usage.require_dirty()
    if proof(refs) is not True:
        raise PermissionError("root_quiescence_unproven")
    with usage.fences.live_lock:
        ids = {ref.root_id for ref in refs}
        if any(ids.intersection(roots) for roots in usage.pending.values()):
            raise PermissionError("root_pending_grants_not_drained")
    for row in review.targets:
        with open_root(
            coordinator.owner,
            row,
            absent=bool(row.get("cleanup") and row["cleanup"]["phase"] == "deleting"),
        ):
            pass
    # Whole-root host evidence can reconcile known lost joins. Preserve every
    # unreviewed root and any unknown owner whose possible scope is wider.
    snapshot = coordinator.published_snapshot()
    with coordinator.registry.transaction() as cursor:
        for owner in cursor.execute(
            "SELECT * FROM processes WHERE state!='settled'"
        ).fetchall():
            if owner["root_coverage"] == "unknown":
                applicable = {
                    row["root_id"]
                    for row in snapshot["data_roots"]
                    if row["installation_id"] == owner["installation_id"]
                    or (row.get("custody") or {}).get("attached_installation_id")
                    == owner["installation_id"]
                }
                if applicable and applicable <= ids:
                    cursor.execute(
                        "UPDATE root_users SET state='released' WHERE owner_token=?",
                        (owner["token"],),
                    )
                    cursor.execute(
                        "UPDATE processes SET state='settled' WHERE token=?",
                        (owner["token"],),
                    )
                continue
            expected = json.loads(owner["root_grants_json"])
            for grant in expected:
                if grant["root_id"] in ids:
                    cursor.execute(
                        "INSERT INTO root_users VALUES (?, ?, ?, ?, ?, ?, 'released') ON CONFLICT(owner_token, root_id, root_generation) DO UPDATE SET state='released'",
                        (
                            uuid4().hex,
                            grant["root_id"],
                            grant["root_generation"],
                            grant["binding_digest"],
                            owner["token"],
                            grant["access"],
                        ),
                    )
            if expected and {grant["root_id"] for grant in expected} <= ids:
                cursor.execute(
                    "UPDATE processes SET state='settled' WHERE token=?",
                    (owner["token"],),
                )
    with usage.fences.live_lock:
        for owner, roots in list(usage.live_grants.items()):
            usage.live_grants[owner] = tuple(root for root in roots if root not in ids)
            usage.grant_epochs[owner] = tuple(
                (root, epoch)
                for root, epoch in usage.grant_epochs.get(owner, ())
                if root not in ids
            )
    await commit_root_phase(coordinator, review, operation_id)
    with usage.fences.live_lock:
        for root_id in ids:
            usage.recovery.pop(root_id, None)
    usage.unfence(token)
    return tuple(root_ref(row) for row in review.targets)


def review_resume(coordinator, refs):
    rows = []
    for ref in refs:
        row = next(
            (
                row
                for row in coordinator.published_snapshot()["data_roots"]
                if row["root_id"] == ref.root_id
            ),
            None,
        )
        if row is None or root_ref(row) != ref or not row.get("cleanup"):
            raise PermissionError("root_pending_cleanup_unavailable")
        if ref.root_id in coordinator.root_usage.recovery:
            raise PermissionError(coordinator.root_usage.recovery[ref.root_id])
        rows.append(row)
    group_ids = {row["cleanup"]["group_id"] for row in rows}
    if len(group_ids) != 1 or len({row["cleanup"]["phase"] for row in rows}) != 1:
        raise PermissionError("root_cleanup_group_changed")
    phase = rows[0]["cleanup"]["phase"]
    if phase not in {"waiting", "deleting"}:
        raise PermissionError("root_creation_recovery_required")
    original = issue_root_review(
        coordinator,
        rows,
        phase,
        group_id=next(iter(group_ids)),
        target_digest=rows[0]["cleanup"]["target_digest"],
        action=rows[0]["cleanup"]["action"],
        attachment=rows[0]["cleanup"]["attachment"],
    )
    current_group(coordinator, original)
    return original


def applicable_roots(authority, installation_id, workspace_id):
    """Capture attached roots, preserving legacy unknown ownership explicitly."""
    return sorted(
        (
            row
            for row in authority["data_roots"]
            if (
                row["custody"]["attached_installation_id"]
                if row.get("custody")
                else row["installation_id"]
            )
            == installation_id
            and row["workspace_id"] in {None, workspace_id}
        ),
        key=lambda row: row["root_id"],
    )


async def commit_root_phase(coordinator, review, operation_id):
    """Enter the sole authority pipeline only from a checked root lifecycle phase."""
    coordinator.root_usage.committing.add(review.token)
    try:
        return await coordinator.commit(review, operation_id)
    finally:
        coordinator.root_usage.committing.discard(review.token)


async def cancel_data_deletion(coordinator, operation_id):
    usage = coordinator.root_usage
    operation = usage.operations[operation_id]
    operation.cancel_requested = True
    if operation.task is not None and not operation.task.done():
        return
    receipts = await coordinator.recover()
    if any(item.phase == "recovery_required" for item in receipts):
        return
    snapshot = coordinator.published_snapshot()
    rows = [
        row
        for row in snapshot["data_roots"]
        if (row.get("cleanup") or {}).get("group_id") == operation.review.group_id
    ]
    if rows:
        rows = current_group(coordinator, operation.review)
        if any(row["cleanup"]["phase"] == "deleting" for row in rows):
            return
        for row in rows:
            row.pop("cleanup")
            row["deletion_fenced"] = False
        final = issue_root_review(
            coordinator,
            rows,
            "cancelled",
            group_id=operation.review.group_id,
            target_digest=operation.review.target_digest,
            expires_at=operation.review.expires_at,
        )
        operation.receipt = await commit_root_phase(
            coordinator, final, final.operation_id
        )
    else:
        originals = {row["root_id"]: member(row) for row in operation.review.targets}
        current = {
            row["root_id"]: member(row)
            for row in snapshot["data_roots"]
            if row["root_id"] in originals
        }
        if current != originals:
            raise PermissionError("root_cleanup_group_changed")
    usage.unfence(operation.fence_token)
    operation.receipt = replace(
        operation.receipt, phase="cancelled", cleanup_pending=False
    )
