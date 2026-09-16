"""Durable encrypted plugin authority and exact-transition recovery evidence.

The coordinator alone calls ``certify_commit`` after its owned registry
transaction returns. This module cannot attest to a SQLite commit; prepared
material deliberately contains no commit certificate (ADR-162, R9).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from tldw_chatbook.runtime_policy.server_credentials import is_secure_keyring_backend
from tldw_chatbook.Skills_Interop.skill_trust_crypto import (
    canonical_json,
    decrypt_json_blob,
    derive_plugin_authority_keys,
    encrypt_json_blob,
)
from tldw_chatbook.Skills_Interop.skill_trust_store import (
    default_trust_store_dir,
    skill_trust_account_scope,
)
from tldw_chatbook.Utils.private_paths import (
    PrivateFileWritePrecondition,
    atomic_private_write_bytes,
    open_private_binary,
    secure_private_directory,
)

from .authority import (
    PluginMarker,
    authority_message,
    canonical_snapshot,
    empty_snapshot,
    snapshot_digest,
)

MARKER_SERVICE = "tldw_chatbook.plugin_trust"
MARKER_ACCOUNT = "managed-plugins:generation-marker:v1"
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024


def default_plugin_authority_dir(local_skills_store_dir: str | Path) -> Path:
    """Use the protected trust subtree, outside standalone snapshots/reset."""
    return default_trust_store_dir(local_skills_store_dir) / "plugins"


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate authority key")
        result[key] = value
    return result


def _decode(data: bytes) -> dict:
    value = json.loads(data, object_pairs_hook=_pairs)
    if type(value) is not dict:
        raise ValueError("authority object required")
    return value


def _read(path: Path) -> dict:
    with open_private_binary(path) as opened:
        if not opened.result.verified_private:
            raise ValueError("unqualified private authority read")
        data = opened.stream.read(MAX_ARTIFACT_BYTES + 1)
    if len(data) > MAX_ARTIFACT_BYTES:
        raise ValueError("authority artifact limit")
    return _decode(data)


def _directory(path: Path) -> None:
    # The private helper may create multiple ancestors. Synchronize each new
    # directory entry, up through its nearest pre-existing parent.
    directories = [path, path.parent]
    ancestor = path.parent
    while not os.path.lexists(ancestor):
        directories.append(ancestor.parent)
        ancestor = ancestor.parent
    result = secure_private_directory(path, create=True, application_owned=True)
    if not result.verified_private:
        raise ValueError("unqualified private authority directory")
    # Synchronize the directory entry as well as later file publications.
    for directory in dict.fromkeys(directories):
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _write(path: Path, payload: dict, *, precondition=None) -> None:
    encoded = canonical_json(payload)
    if len(encoded) > MAX_ARTIFACT_BYTES:
        raise ValueError("authority artifact limit")
    _directory(path.parent)
    result = atomic_private_write_bytes(
        path,
        encoded,
        target_precondition=precondition or PrivateFileWritePrecondition.missing(),
    )
    if not result.verified_private:
        raise ValueError("unqualified durable authority publication")


def _sync_existing(path: Path) -> None:
    """A prior replace may have failed its parent fsync; retry must qualify it."""
    with open_private_binary(path) as opened:
        if not opened.result.verified_private:
            raise ValueError("unqualified durable authority retry")
        os.fsync(opened.stream.fileno())
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            current = os.stat(path.name, dir_fd=fd, follow_symlinks=False)
            pinned = os.fstat(opened.stream.fileno())
            if (current.st_dev, current.st_ino) != (pinned.st_dev, pinned.st_ino):
                raise ValueError("authority retry identity changed")
            os.fsync(fd)
        finally:
            os.close(fd)


class KeyringPluginMarkerStore:
    """Separate scoped marker account; no passphrase or key cache is stored."""

    reduced_protection = False

    def __init__(self, store_dir: Path, *, keyring_backend=None):
        if keyring_backend is None:
            import keyring

            keyring_backend = keyring.get_keyring()
        if not is_secure_keyring_backend(keyring_backend):
            raise ValueError("secure plugin marker unavailable")
        self.backend = keyring_backend
        self.account = f"{MARKER_ACCOUNT}:{skill_trust_account_scope(store_dir)}"

    def load_marker(self) -> dict | None:
        payload = self.backend.get_password(MARKER_SERVICE, self.account)
        return (
            None
            if payload is None
            else PluginMarker.model_validate(_decode(payload.encode())).model_dump()
        )

    def save_marker(self, marker: dict) -> None:
        payload = PluginMarker.model_validate(marker).model_dump()
        self.backend.set_password(
            MARKER_SERVICE, self.account, canonical_json(payload).decode()
        )

    def clear(self) -> None:
        # Reviewed reset must also remove malformed marker content; backend
        # availability and exact account ownership remain mandatory.
        if self.backend.get_password(MARKER_SERVICE, self.account) is not None:
            self.backend.delete_password(MARKER_SERVICE, self.account)
        if self.backend.get_password(MARKER_SERVICE, self.account) is not None:
            raise ValueError("plugin marker reset incomplete")


class FilePluginMarkerStore:
    """Explicit reduced rollback protection; construction is not acceptance."""

    reduced_protection = True

    def __init__(self, store_dir: Path):
        self.path = Path(store_dir) / "generation_marker.json"

    def load_marker(self) -> dict | None:
        if not os.path.lexists(self.path):
            return None
        return PluginMarker.model_validate(_read(self.path)).model_dump()

    def save_marker(self, marker: dict) -> None:
        payload = PluginMarker.model_validate(marker).model_dump()
        condition = PrivateFileWritePrecondition.missing()
        if os.path.lexists(self.path):
            with open_private_binary(self.path) as opened:
                if not opened.result.verified_private:
                    raise ValueError("unqualified marker")
                condition = PrivateFileWritePrecondition.from_opened(opened)
        _write(self.path, payload, precondition=condition)

    def clear(self) -> None:
        if not os.path.lexists(self.path):
            return
        # Authenticate filesystem identity, not corrupt JSON being reset.
        with open_private_binary(self.path) as opened:
            if not opened.result.verified_private:
                raise ValueError("unqualified marker reset")
            fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                current = os.stat(self.path.name, dir_fd=fd, follow_symlinks=False)
                pinned = os.fstat(opened.stream.fileno())
                if (current.st_dev, current.st_ino) != (pinned.st_dev, pinned.st_ino):
                    raise ValueError("marker reset identity changed")
                os.unlink(self.path.name, dir_fd=fd)
                os.fsync(fd)
            finally:
                os.close(fd)


@dataclass(frozen=True)
class TransitionEvidence:
    """Authenticated evidence; ``committed=False`` never permits promotion."""

    old: PluginMarker
    new: PluginMarker
    snapshot: dict
    committed: bool


class PluginAuthorityStore:
    """Passphrase-per-start plugin trust, with an injected marker backend.

    Mutating callers must hold the existing plugin runtime owner. Reset requires
    explicit caller review. Reduced rollback protection must be accepted per
    session, and never converts unauthenticated data into reviewed package trust.
    """

    def __init__(
        self,
        store_dir: Path,
        marker_store: object,
        *,
        accept_reduced_protection: bool = False,
    ):
        self.store_dir = Path(store_dir).absolute()
        self.marker_store = marker_store
        self.accept_reduced_protection = accept_reduced_protection
        self._keys = None

    def _require_posture(self) -> None:
        if (
            getattr(self.marker_store, "reduced_protection", False)
            and not self.accept_reduced_protection
        ):
            raise ValueError("reduced rollback protection requires explicit acceptance")

    def _require_keys(self):
        self._require_posture()
        if self._keys is None:
            raise ValueError("plugin authority locked")
        return self._keys

    def load_marker(self) -> PluginMarker | None:
        """Read the exact marker; availability failures propagate without defaults."""
        self._require_posture()
        value = self.marker_store.load_marker()
        return None if value is None else PluginMarker.model_validate(value)

    def posture(self) -> str:
        """Report setup/locked/recovery/unavailable without exposing plaintext."""
        if (
            getattr(self.marker_store, "reduced_protection", False)
            and not self.accept_reduced_protection
        ):
            return "reduced_acceptance_required"
        try:
            marker = self.load_marker()
        # Third-party keyring errors report posture; they never grant authority.
        except Exception:  # noqa: BLE001
            return "unavailable"
        artifacts = os.path.lexists(self.store_dir) and any(self.store_dir.iterdir())
        if marker is None:
            return "recovery_required" if artifacts else "needs_setup"
        if not os.path.lexists(self.store_dir / "metadata.json"):
            return "recovery_required"
        if self._keys is None:
            return "locked"
        try:
            self.verify_current()
        except (ValueError, OSError, RuntimeError):
            return "recovery_required"
        return (
            "ready_reduced"
            if getattr(self.marker_store, "reduced_protection", False)
            else "ready"
        )

    def bootstrap(self, passphrase: str) -> None:
        """Explicitly initialize only empty generation-zero authority."""
        self._keys = None
        if self.load_marker() is not None or (
            os.path.lexists(self.store_dir) and any(self.store_dir.iterdir())
        ):
            raise ValueError("plugin authority already exists or setup incomplete")
        _directory(self.store_dir)
        salt = secrets.token_bytes(32)
        _write(
            self.store_dir / "metadata.json", {"schema_version": 1, "salt": salt.hex()}
        )
        self._keys = derive_plugin_authority_keys(passphrase, salt=salt)
        snapshot = empty_snapshot()
        marker = PluginMarker(
            generation=0,
            operation_id="bootstrap",
            recovery_snapshot_digest=snapshot_digest(snapshot),
        )
        try:
            self._save_snapshot(snapshot, marker)
            self.marker_store.save_marker(marker.model_dump())
            if self.load_marker() != marker:
                raise ValueError("bootstrap marker publication mismatch")
            self.verify_current()
        except BaseException:
            self._keys = None
            raise

    def unlock(self, passphrase: str) -> None:
        """Authenticate existing current authority; never bootstrap or bless rows."""
        self._keys = None
        self._require_posture()
        if not os.path.lexists(self.store_dir / "metadata.json"):
            raise ValueError("plugin authority setup required")
        metadata = _read(self.store_dir / "metadata.json")
        if (
            set(metadata) != {"schema_version", "salt"}
            or type(metadata["schema_version"]) is not int
            or metadata["schema_version"] != 1
            or type(metadata["salt"]) is not str
            or len(metadata["salt"]) != 64
        ):
            raise ValueError("invalid plugin authority metadata")
        self._keys = derive_plugin_authority_keys(
            passphrase, salt=bytes.fromhex(metadata["salt"])
        )
        try:
            self.verify_current()
        except BaseException:
            self._keys = None
            raise

    def lock(self) -> None:
        """Drop session key references."""
        self._keys = None

    def _snapshot_path(self, marker):
        return self.store_dir / "snapshots" / f"{marker.recovery_snapshot_digest}.json"

    def _operation_path(self, purpose, operation_id):
        leaf = hashlib.sha256(operation_id.encode()).hexdigest()
        return (
            self.store_dir
            / ("intents" if purpose == "prepared" else "certificates")
            / f"{leaf}.json"
        )

    def _save_snapshot(self, snapshot, marker):
        key = self._require_keys().snapshot_key
        path = self._snapshot_path(marker)
        if os.path.lexists(path):
            if self.verify_snapshot(marker) != snapshot:
                raise ValueError("conflicting immutable snapshot")
            _sync_existing(path)
            return
        header = {"schema_version": 1, "marker": marker.model_dump()}
        blob = encrypt_json_blob(
            snapshot, key, associated_data=authority_message("snapshot", header)
        )
        _write(path, {"header": header, "blob": blob})
        self.verify_snapshot(marker)

    def verify_snapshot(self, marker: PluginMarker) -> dict:
        """Authenticate marker-bound complete reconstruction authority."""
        key = self._require_keys().snapshot_key
        marker = PluginMarker.model_validate(marker)
        value = _read(self._snapshot_path(marker))
        header = {"schema_version": 1, "marker": marker.model_dump()}
        if (
            set(value) != {"header", "blob"}
            or value["header"] != header
            or type(value["blob"]) is not dict
            or set(value["blob"]) != {"alg", "nonce", "ciphertext", "tag"}
        ):
            raise ValueError("invalid authority envelope")
        if type(value["header"].get("schema_version")) is not int:
            raise ValueError("invalid snapshot header version")
        PluginMarker.model_validate(value["header"]["marker"])
        snapshot = canonical_snapshot(
            decrypt_json_blob(
                value["blob"],
                key,
                associated_data=authority_message("snapshot", header),
            )
        )
        if snapshot_digest(snapshot) != marker.recovery_snapshot_digest:
            raise ValueError("authority digest mismatch")
        result = snapshot["operation_result"]
        if marker.generation == 0:
            if snapshot != empty_snapshot():
                raise ValueError("nonempty bootstrap authority")
        elif result is None or result["operation_id"] != marker.operation_id:
            raise ValueError("operation identity mismatch")
        return snapshot

    def verify_current(self) -> dict:
        """Return only the snapshot named by the current exact marker."""
        self._require_keys()
        marker = self.load_marker()
        if marker is None:
            raise ValueError("plugin authority marker missing")
        return self.verify_snapshot(marker)

    @staticmethod
    def _transition(old, new):
        old, new = PluginMarker.model_validate(old), PluginMarker.model_validate(new)
        if new.generation != old.generation + 1 or new.operation_id in (
            old.operation_id,
            "bootstrap",
        ):
            raise ValueError("invalid authority transition")
        return {
            "schema_version": 1,
            "old": old.model_dump(),
            "new": new.model_dump(),
            "recovery_snapshot_digest": new.recovery_snapshot_digest,
        }

    def _read_evidence(self, purpose, operation_id):
        keys = self._require_keys()
        envelope = _read(self._operation_path(purpose, operation_id))
        if set(envelope) != {"payload", "mac"} or type(envelope["mac"]) is not str:
            raise ValueError("invalid authority evidence")
        payload = envelope["payload"]
        if type(payload) is not dict or set(payload) != {
            "schema_version",
            "old",
            "new",
            "recovery_snapshot_digest",
        }:
            raise ValueError("invalid transition payload")
        old, new = PluginMarker.model_validate(
            payload["old"]
        ), PluginMarker.model_validate(payload["new"])
        if (
            payload != self._transition(old, new)
            or type(payload["schema_version"]) is not int
            or new.operation_id != operation_id
        ):
            raise ValueError("transition identity mismatch")
        key = keys.prepared_key if purpose == "prepared" else keys.committed_key
        expected = hmac.new(
            key, authority_message(purpose, payload), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(envelope["mac"], expected):
            raise ValueError("authority evidence authentication failed")
        return payload

    def _save_evidence(self, purpose, payload):
        operation_id = payload["new"]["operation_id"]
        path = self._operation_path(purpose, operation_id)
        if os.path.lexists(path):
            if self._read_evidence(purpose, operation_id) != payload:
                raise ValueError("conflicting immutable evidence")
            _sync_existing(path)
            return
        keys = self._require_keys()
        key = keys.prepared_key if purpose == "prepared" else keys.committed_key
        tag = hmac.new(
            key, authority_message(purpose, payload), hashlib.sha256
        ).hexdigest()
        _write(path, {"payload": payload, "mac": tag})
        self._read_evidence(purpose, operation_id)

    def prepare(self, snapshot: dict, old: PluginMarker, new: PluginMarker) -> None:
        """Persist complete snapshot and intent, without any commit proof."""
        self._require_keys()
        payload = self._transition(old, new)
        if self.load_marker() != old:
            raise ValueError("stale authority marker")
        self.verify_snapshot(old)
        snapshot = canonical_snapshot(snapshot)
        if snapshot_digest(snapshot) != new.recovery_snapshot_digest:
            raise ValueError("prepared snapshot mismatch")
        result = snapshot["operation_result"]
        if result is None or result["operation_id"] != new.operation_id:
            raise ValueError("prepared operation identity mismatch")
        self._save_snapshot(snapshot, new)
        self._save_evidence("prepared", payload)

    def certify_commit(self, old: PluginMarker, new: PluginMarker) -> None:
        """Coordinator-internal: call ONLY after durable registry commit returns.

        No registry flag or prepared artifact authorizes this call. F4 owns the
        ordering boundary; active malicious app code is outside ADR-009's model.
        """
        payload = self._transition(old, new)
        if self.load_marker() not in (old, new):
            raise ValueError("stale authority marker")
        if self._read_evidence("prepared", new.operation_id) != payload:
            raise ValueError("prepared transition mismatch")
        self.verify_snapshot(new)
        self._save_evidence("committed", payload)

    def verify_transition(self, operation_id: str) -> TransitionEvidence:
        """Read authenticated exact intent, snapshot and optional certificate."""
        payload = self._read_evidence("prepared", operation_id)
        old, new = PluginMarker.model_validate(
            payload["old"]
        ), PluginMarker.model_validate(payload["new"])
        snapshot = self.verify_snapshot(new)
        committed = os.path.lexists(self._operation_path("committed", operation_id))
        if committed and self._read_evidence("committed", operation_id) != payload:
            raise ValueError("commit certificate mismatch")
        return TransitionEvidence(old, new, snapshot, committed)

    def advance_marker(self, old: PluginMarker, new: PluginMarker) -> None:
        """Advance only an exact certified transition, allowing exact retries."""
        self._transition(old, new)
        evidence = self.verify_transition(new.operation_id)
        if not evidence.committed or evidence.old != old or evidence.new != new:
            raise ValueError("committed transition required")
        current = self.load_marker()
        if current == new:
            self.verify_current()
            return
        if current != old:
            raise ValueError("stale authority marker")
        self.marker_store.save_marker(new.model_dump())
        if self.load_marker() != new:
            raise ValueError("marker publication mismatch")

    def reset(self) -> Path | None:
        """Caller-reviewed plugin-only reset; archive evidence rather than erase it.

        Failure is raised and evidence retained. The archive stays under the
        protected trust root. Neither standalone files nor its accounts change.
        """
        self.lock()
        self._require_posture()
        self.marker_store.clear()
        if self.load_marker() is not None:
            raise ValueError("plugin marker reset incomplete")
        if not os.path.lexists(self.store_dir):
            return None
        _directory(self.store_dir)
        archive = self.store_dir.with_name(f"plugins-reset-{secrets.token_hex(16)}")
        parent_fd = os.open(
            self.store_dir.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        try:
            os.rename(
                self.store_dir.name,
                archive.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return archive
