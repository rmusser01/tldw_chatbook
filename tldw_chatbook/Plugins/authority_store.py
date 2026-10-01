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
import re
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
    PrivatePathStatus,
    atomic_private_write_bytes,
    open_private_binary,
    secure_private_directory,
    verify_trusted_directory,
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
MAX_RESET_RECORDS = 1000
MAX_TRANSITIONS = 1000


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
        self._require_completed_resets()
        return self._keys

    @staticmethod
    def result_digest(result: dict, *, exclude_id: bool = False) -> str:
        from .authority import OperationResult

        value = OperationResult.model_validate(
            dict(result, operation_id=result.get("operation_id", "binding"))
        ).model_dump(mode="json")
        if exclude_id:
            value.pop("operation_id")
        return hashlib.sha256(canonical_json(value)).hexdigest()

    def issue_operation_id(
        self,
        expected_generation: int,
        binding_digest: str,
        result_without_id: dict,
        nonce: str,
    ) -> str:
        """Issue a fixed-width authenticated identity, with no commitment claim."""
        from .authority import IssuedOperationIdentity

        identity = IssuedOperationIdentity(
            generation=expected_generation, binding_digest=binding_digest, nonce=nonce
        )
        if "operation_id" in result_without_id:
            raise ValueError("issued result must exclude operation ID")
        payload = dict(
            identity.model_dump(),
            result_digest=self.result_digest(result_without_id, exclude_id=True),
        )
        tag = hmac.new(
            self._require_keys().prepared_key,
            authority_message("issued_operation", payload),
            hashlib.sha256,
        ).hexdigest()
        return f"pi1.{identity.generation:016x}.{identity.binding_digest}.{identity.nonce}.{tag}"

    def verify_operation_id(self, operation_id: str, result: dict):
        """Authenticate generation and exact result, never infer permission."""
        from .authority import IssuedOperationIdentity

        if (
            not isinstance(operation_id, str)
            or re.fullmatch(
                r"pi1\.[0-9a-f]{16}\.[0-9a-f]{64}\.[0-9a-f]{64}\.[0-9a-f]{64}",
                operation_id,
            )
            is None
            or result.get("operation_id") != operation_id
        ):
            raise ValueError("invalid issued operation identity")
        _, generation, binding, nonce, tag = operation_id.split(".")
        identity = IssuedOperationIdentity(
            generation=int(generation, 16), binding_digest=binding, nonce=nonce
        )
        payload = dict(
            identity.model_dump(),
            result_digest=self.result_digest(result, exclude_id=True),
        )
        expected = hmac.new(
            self._require_keys().prepared_key,
            authority_message("issued_operation", payload),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(tag, expected):
            raise ValueError("issued operation authentication failed")
        return identity

    def namespace_id(self) -> str:
        return hashlib.sha256(
            b"chatbook.plugins.archive-pin-v1\0"
            + bytes.fromhex(self.metadata()["salt"])
        ).hexdigest()

    def sign_archive_pin(self, payload: dict) -> str:
        if payload.get("namespace_id") != self.namespace_id():
            raise ValueError("managed pin namespace mismatch")
        self.verify_current()
        return hmac.new(
            self._require_keys().committed_key,
            authority_message("archive-pin-v1", payload),
            hashlib.sha256,
        ).hexdigest()

    def verify_archive_pin(self, payload: dict, mac: str) -> None:
        if not hmac.compare_digest(self.sign_archive_pin(payload), mac):
            raise ValueError("managed pin authentication failed")

    def metadata(self) -> dict:
        """Read versioned trust metadata without reinterpreting historical bytes."""
        value = _read(self.store_dir / "metadata.json")
        version = value.get("schema_version")
        if (
            type(version) is not int
            or version not in {1, 2}
            or set(value)
            != (
                {"schema_version", "salt"}
                if version == 1
                else {"schema_version", "salt", "cutover_digest"}
            )
            or not isinstance(value.get("salt"), str)
            or re.fullmatch(r"[0-9a-f]{64}", value["salt"]) is None
        ):
            raise ValueError("invalid plugin authority metadata")
        if version == 2 and (
            not isinstance(value.get("cutover_digest"), str)
            or re.fullmatch(r"[0-9a-f]{64}", value["cutover_digest"]) is None
        ):
            raise ValueError("invalid plugin cutover identity")
        return value

    def verify_legacy_cutover(self):
        """Authenticate the bounded compatibility map selected by metadata v2."""
        from .authority import LegacyCutover

        envelope = _read(self.store_dir / "legacy-cutover.json")
        if (
            len(canonical_json(envelope)) > 16 * 1024 * 1024
            or set(envelope) != {"payload", "mac"}
            or not isinstance(envelope["mac"], str)
        ):
            raise ValueError("invalid legacy cutover envelope")
        payload = LegacyCutover.model_validate(envelope["payload"])
        if any(
            len(canonical_json(entry.model_dump(mode="json"))) > 8192
            for entry in payload.entries
        ):
            raise ValueError("legacy cutover entry limit")
        tag = hmac.new(
            self._require_keys().committed_key,
            authority_message("legacy_cutover", envelope["payload"]),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(envelope["mac"], tag):
            raise ValueError("legacy cutover authentication failed")
        metadata = self.metadata()
        if (
            metadata["schema_version"] == 2
            and hashlib.sha256(canonical_json(envelope)).hexdigest()
            != metadata["cutover_digest"]
        ):
            raise ValueError("fixed legacy cutover mismatch")
        return payload

    def ensure_issued_metadata(self, lineage: tuple[TransitionEvidence, ...]) -> None:
        """Finalize one qualified v1 cutover; v2 maps never grow or shrink.

        Called only by the coordinator after full recovery under its OS owner.
        Every supplied transition is reauthenticated; pending map ancestry and
        entries must still have exact witnesses before any metadata replacement.
        """
        metadata = self.metadata()
        if metadata["schema_version"] == 2:
            self.verify_legacy_cutover()
            _sync_existing(self.store_dir / "legacy-cutover.json")
            _sync_existing(self.store_dir / "metadata.json")
            return
        marker = self.load_marker()
        current = self.verify_current()
        qualified = {}
        endpoints = {marker}
        by_new = {}
        for item in lineage:
            if self.verify_transition(item.new.operation_id) != item:
                raise ValueError("cutover lineage changed")
            by_new[item.new] = item
        ancestor = marker
        while ancestor in by_new:
            item = by_new[ancestor]
            result = item.snapshot["operation_result"]
            qualified[result["operation_id"]] = self.result_digest(result)
            ancestor = item.old
            endpoints.add(ancestor)
        if len(by_new) != len(endpoints) - 1:
            raise ValueError("disconnected cutover lineage")
        if current["operation_result"] is not None:
            result = current["operation_result"]
            qualified[result["operation_id"]] = self.result_digest(result)
        path = self.store_dir / "legacy-cutover.json"
        condition = PrivateFileWritePrecondition.missing()
        prior_envelope = None
        if os.path.lexists(path):
            prior = self.verify_legacy_cutover()
            if prior.marker not in endpoints:
                raise ValueError("pending cutover anchor not in exact lineage")
            prior_snapshot = self.verify_snapshot(prior.marker)
            prior_result = prior_snapshot["operation_result"]
            if prior_result is not None:
                qualified[prior_result["operation_id"]] = self.result_digest(
                    prior_result
                )
            if any(
                qualified.get(entry.operation_id) != entry.result_digest
                for entry in prior.entries
            ):
                raise ValueError("pending cutover entry lacks exact witness")
            with open_private_binary(path) as opened:
                if not opened.result.verified_private:
                    raise ValueError("unqualified pending cutover")
                condition = PrivateFileWritePrecondition.from_opened(opened)
                prior_envelope = _decode(opened.stream.read(MAX_ARTIFACT_BYTES + 1))
        from .authority import LegacyCutover

        payload = LegacyCutover.model_validate(
            {
                "schema_version": 1,
                "marker": marker.model_dump(),
                "entries": [
                    {"operation_id": identity, "result_digest": digest}
                    for identity, digest in sorted(qualified.items())
                ],
            }
        ).model_dump(mode="json")
        envelope = {
            "payload": payload,
            "mac": hmac.new(
                self._require_keys().committed_key,
                authority_message("legacy_cutover", payload),
                hashlib.sha256,
            ).hexdigest(),
        }
        if len(canonical_json(envelope)) > 16 * 1024 * 1024 or any(
            len(canonical_json(entry)) > 8192 for entry in payload["entries"]
        ):
            raise ValueError("legacy cutover byte limit")
        if prior_envelope == envelope:
            _sync_existing(path)
        else:
            _write(path, envelope, precondition=condition)
        if self.verify_legacy_cutover().model_dump(mode="json") != payload:
            raise ValueError("cutover replacement mismatch")
        if self.metadata() != metadata or self.load_marker() != marker:
            raise ValueError("cutover authority changed")
        # Requalify every retained artifact immediately before the durable switch.
        for item in lineage:
            if self.verify_transition(item.new.operation_id) != item:
                raise ValueError("cutover inventory changed")
        with open_private_binary(self.store_dir / "metadata.json") as opened:
            if (
                not opened.result.verified_private
                or _decode(opened.stream.read(MAX_ARTIFACT_BYTES + 1)) != metadata
            ):
                raise ValueError("cutover metadata changed")
            condition = PrivateFileWritePrecondition.from_opened(opened)
        _write(
            self.store_dir / "metadata.json",
            {
                "schema_version": 2,
                "salt": metadata["salt"],
                "cutover_digest": hashlib.sha256(canonical_json(envelope)).hexdigest(),
            },
            precondition=condition,
        )
        self.verify_legacy_cutover()
        _sync_existing(path)
        _sync_existing(self.store_dir / "metadata.json")

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
            self._require_completed_resets()
        except (ValueError, OSError):
            return "recovery_required"
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
        self._require_completed_resets()
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
            self.ensure_issued_metadata(())
        except BaseException:
            self._keys = None
            raise

    def unlock(self, passphrase: str) -> None:
        """Authenticate existing current authority; never bootstrap or bless rows."""
        self._keys = None
        self._require_posture()
        if not os.path.lexists(self.store_dir / "metadata.json"):
            raise ValueError("plugin authority setup required")
        metadata = self.metadata()
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
        names, _ = self._snapshot_inventory()
        if len(names) >= 2 * MAX_TRANSITIONS + 2:
            raise ValueError("snapshot inventory capacity exhausted")
        header = {"schema_version": 1, "marker": marker.model_dump()}
        blob = encrypt_json_blob(
            snapshot, key, associated_data=authority_message("snapshot", header)
        )
        _write(path, {"header": header, "blob": blob})
        self.verify_snapshot(marker)

    def verify_snapshot(self, marker: PluginMarker) -> dict:
        """Authenticate marker-bound complete reconstruction authority."""
        marker = PluginMarker.model_validate(marker)
        return self._verify_snapshot_value(marker, _read(self._snapshot_path(marker)))

    def _verify_snapshot_value(self, marker: PluginMarker, value: dict) -> dict:
        key = self._require_keys().snapshot_key
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
        if self.metadata()["schema_version"] == 2:
            self.verify_legacy_cutover()
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
        old, new = (
            PluginMarker.model_validate(payload["old"]),
            PluginMarker.model_validate(payload["new"]),
        )
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
        names = self._transition_inventory()
        if (
            self._operation_path("prepared", new.operation_id).name
            not in names["intents"]
            and len(names["intents"]) >= MAX_TRANSITIONS
        ):
            raise ValueError("transition inventory capacity reached")
        if self.load_marker() != old:
            raise ValueError("stale authority marker")
        self.verify_snapshot(old)
        snapshot = canonical_snapshot(snapshot)
        if snapshot_digest(snapshot) != new.recovery_snapshot_digest:
            raise ValueError("prepared snapshot mismatch")
        result = snapshot["operation_result"]
        if result is None or result["operation_id"] != new.operation_id:
            raise ValueError("prepared operation identity mismatch")
        if self.metadata()["schema_version"] != 2:
            raise ValueError("plugin metadata cutover required")
        self.verify_legacy_cutover()
        identity = self.verify_operation_id(new.operation_id, result)
        if identity.generation != new.generation:
            raise ValueError("issued generation mismatch")
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
        old, new = (
            PluginMarker.model_validate(payload["old"]),
            PluginMarker.model_validate(payload["new"]),
        )
        snapshot = self.verify_snapshot(new)
        committed = os.path.lexists(self._operation_path("committed", operation_id))
        if committed and self._read_evidence("committed", operation_id) != payload:
            raise ValueError("commit certificate mismatch")
        return TransitionEvidence(old, new, snapshot, committed)

    def _transition_inventory(self) -> dict[str, set[str]]:
        self._require_keys()
        names = {}
        for directory in ("intents", "certificates"):
            path = self.store_dir / directory
            entries = set()
            if os.path.lexists(path):
                qualification = verify_trusted_directory(
                    path, allow_shared_sticky=False
                )
                if qualification.status != PrivatePathStatus.TRUSTED_DIRECTORY:
                    raise ValueError("unqualified transition inventory")
                with os.scandir(path) as inventory:
                    for entry in inventory:
                        if (
                            len(entries) >= MAX_TRANSITIONS
                            or re.fullmatch(r"[0-9a-f]{64}\.json", entry.name) is None
                            or not entry.is_file(follow_symlinks=False)
                        ):
                            raise ValueError(
                                "invalid or excessive transition inventory"
                            )
                        entries.add(entry.name)
            names[directory] = entries
        if not names["certificates"] <= names["intents"]:
            raise ValueError("orphan commit certificate")
        return names

    def list_transitions(
        self, *, limit: int, offset: int
    ) -> tuple[TransitionEvidence, ...]:
        """Discover a bounded authenticated page without trusting registry hints.

        Inventory is capped at 1,000 retained transitions; overflow, unexpected
        entries and orphan certificates fail closed. Each returned transition is
        authenticated. Callers must consume every page before reconciliation.
        Nothing is pruned. Filenames locate evidence, never prove commitment.
        """
        from .registry import validate_page

        validate_page(limit, offset)
        names = self._transition_inventory()
        result = []
        for name in sorted(names["intents"])[offset : offset + limit]:
            path = self.store_dir / "intents" / name
            envelope = _read(path)
            try:
                operation_id = envelope["payload"]["new"]["operation_id"]
                if not isinstance(operation_id, str):
                    raise TypeError("invalid transition discovery identity")
            except (KeyError, TypeError) as error:
                raise ValueError("invalid transition discovery envelope") from error
            if self._operation_path("prepared", operation_id) != path:
                raise ValueError("transition filename mismatch")
            result.append(self.verify_transition(operation_id))
        return tuple(result)

    def _unlink_artifact(
        self,
        path: Path,
        *,
        expected_identity=None,
        parent_identity=None,
        expected_digest=None,
    ) -> None:
        """Remove an exact qualified private artifact with directory durability."""
        with open_private_binary(path) as opened:
            if not opened.result.verified_private:
                raise ValueError("unqualified retention artifact")
            if expected_digest is not None:
                data = opened.stream.read(MAX_ARTIFACT_BYTES + 1)
                if hashlib.sha256(data).hexdigest() != expected_digest:
                    raise ValueError("authenticated retention bytes changed")
            pinned = os.fstat(opened.stream.fileno())
            if (
                expected_identity is not None
                and (pinned.st_dev, pinned.st_ino) != expected_identity
            ):
                raise ValueError("authenticated retention artifact changed")
            fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                parent = os.fstat(fd)
                if (
                    parent_identity is not None
                    and (parent.st_dev, parent.st_ino) != parent_identity
                ):
                    raise ValueError("retention snapshot directory changed")
                observed = os.stat(path.name, dir_fd=fd, follow_symlinks=False)
                if (pinned.st_dev, pinned.st_ino) != (observed.st_dev, observed.st_ino):
                    raise ValueError("retention artifact changed")
                os.unlink(path.name, dir_fd=fd)
                os.fsync(fd)
            finally:
                os.close(fd)

    def _snapshot_inventory(self):
        """Qualify the entire bounded namespace before selecting any artifacts."""
        path = self.store_dir / "snapshots"
        if not os.path.lexists(path):
            return (), None
        qualification = verify_trusted_directory(path, allow_shared_sticky=False)
        if qualification.status != PrivatePathStatus.TRUSTED_DIRECTORY:
            raise ValueError("unqualified snapshot inventory")
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            parent = os.fstat(fd)
            names = []
            with os.scandir(fd) as entries:
                for entry in entries:
                    if (
                        len(names) >= 2 * MAX_TRANSITIONS + 2
                        or re.fullmatch(r"[0-9a-f]{64}\.json", entry.name) is None
                        or not entry.is_file(follow_symlinks=False)
                    ):
                        raise ValueError("invalid or excessive snapshot inventory")
                    names.append(entry.name)
            return tuple(sorted(names)), (parent.st_dev, parent.st_ino)
        finally:
            os.close(fd)

    def _snapshot_references(self):
        names = self._transition_inventory()
        references = set()
        for name in names["intents"]:
            path = self.store_dir / "intents" / name
            envelope = _read(path)
            operation_id = envelope["payload"]["new"]["operation_id"]
            if self._operation_path("prepared", operation_id) != path:
                raise ValueError("transition filename mismatch")
            payload = self._read_evidence("prepared", operation_id)
            if (
                name in names["certificates"]
                and self._read_evidence("committed", operation_id) != payload
            ):
                raise ValueError("commit certificate mismatch")
            references.add(payload["new"]["recovery_snapshot_digest"])
        return references

    def retire_orphan_snapshots(
        self, *, protected_operation_ids: frozenset[str] = frozenset()
    ) -> None:
        """Finish interrupted retirement, never derive authority from orphan bytes.

        The coordinator calls after reconciliation under its exclusive owner.
        Keep only bounded marker/file identities between sequential body reads.
        """
        if self.metadata()["schema_version"] != 2:
            raise ValueError("retention requires durable issued-ID cutover")
        legacy = self.verify_legacy_cutover()
        legacy_results = {
            entry.operation_id: entry.result_digest for entry in legacy.entries
        }
        current = self.load_marker()
        self.verify_current()
        references = self._snapshot_references()
        names, parent_identity = self._snapshot_inventory()
        candidates = []
        for name in names:
            path = self.store_dir / "snapshots" / name
            with open_private_binary(path) as opened:
                if not opened.result.verified_private:
                    raise ValueError("unqualified snapshot retirement")
                data = opened.stream.read(MAX_ARTIFACT_BYTES + 1)
                if len(data) > MAX_ARTIFACT_BYTES:
                    raise ValueError("authority artifact limit")
                value = _decode(data)
                if (
                    type(value.get("header")) is not dict
                    or "marker" not in value["header"]
                ):
                    raise ValueError("invalid snapshot header")
                marker = PluginMarker.model_validate(value["header"]["marker"])
                if self._snapshot_path(marker) != path:
                    raise ValueError("snapshot filename mismatch")
                snapshot = self._verify_snapshot_value(marker, value)
                pinned = os.fstat(opened.stream.fileno())
                identity = (pinned.st_dev, pinned.st_ino)
            if (
                marker.generation == 0
                or marker.operation_id in protected_operation_ids
                or marker.generation >= current.generation
                or marker.recovery_snapshot_digest in references
            ):
                continue
            result = snapshot["operation_result"]
            if marker.operation_id.startswith("pi1."):
                issued = self.verify_operation_id(marker.operation_id, result)
                if issued.generation != marker.generation:
                    raise ValueError("snapshot issued generation mismatch")
            elif legacy_results.get(marker.operation_id) != self.result_digest(result):
                raise ValueError("unknown legacy snapshot result")
            candidates.append((path, identity, hashlib.sha256(data).hexdigest()))
        # Authenticate the whole inventory before unlinking any prefix. A changed
        # marker/reference or replacement file never inherits this qualification.
        for path, identity, digest in candidates:
            if (
                self.load_marker() != current
                or self._snapshot_references() != references
            ):
                raise ValueError("snapshot retention authority changed")
            self._unlink_artifact(
                path,
                expected_identity=identity,
                parent_identity=parent_identity,
                expected_digest=digest,
            )

    def retire_transition(self, evidence: TransitionEvidence) -> None:
        """Remove a caller-qualified oldest prefix item after durable hint removal."""
        if self.metadata()["schema_version"] != 2:
            raise ValueError("retention requires durable issued-ID cutover")
        self.verify_legacy_cutover()
        if (
            evidence.new == self.load_marker()
            or self.verify_transition(evidence.new.operation_id) != evidence
        ):
            raise ValueError("protected or changed retention transition")
        for purpose in ("committed", "prepared"):
            path = self._operation_path(purpose, evidence.new.operation_id)
            if os.path.lexists(path):
                self._unlink_artifact(path)
        # Historical snapshots are not needed once their last intent is gone.
        names = self._transition_inventory()
        for name in names["intents"]:
            payload = _read(self.store_dir / "intents" / name)["payload"]
            if (
                payload["new"]["recovery_snapshot_digest"]
                == evidence.new.recovery_snapshot_digest
            ):
                return
        self._unlink_artifact(self._snapshot_path(evidence.new))

    def advance_marker(self, old: PluginMarker, new: PluginMarker) -> None:
        """Advance only an exact certified transition, allowing exact retries."""
        self._transition(old, new)
        evidence = self.verify_transition(new.operation_id)
        if not evidence.committed or evidence.old != old or evidence.new != new:
            raise ValueError("committed transition required")
        current = self.load_marker()
        if current == new:
            self.verify_current()
        elif current != old:
            raise ValueError("stale authority marker")
        # A previous save may have replaced visible bytes and then failed its
        # durability boundary. Exact retries must qualify the backend write too.
        self.marker_store.save_marker(new.model_dump())
        if self.load_marker() != new:
            raise ValueError("marker publication mismatch")

    @property
    def _reset_state_path(self) -> Path:
        return self.store_dir.with_name(f"{self.store_dir.name}-reset-state.json")

    def _load_reset_state(self) -> tuple[dict, PrivateFileWritePrecondition]:
        """Read bounded cleanup bookkeeping, never execution authority."""
        scope = skill_trust_account_scope(self.store_dir)
        if not os.path.lexists(self._reset_state_path):
            return {
                "schema_version": 1,
                "store_scope": scope,
                "resets": [],
            }, PrivateFileWritePrecondition.missing()
        with open_private_binary(self._reset_state_path) as opened:
            if not opened.result.verified_private:
                raise ValueError("unqualified reset receipt")
            data = opened.stream.read(MAX_ARTIFACT_BYTES + 1)
            condition = PrivateFileWritePrecondition.from_opened(opened)
        if len(data) > MAX_ARTIFACT_BYTES:
            raise ValueError("reset receipt size limit")
        try:
            state = _decode(data)
        except (RecursionError, UnicodeError) as exc:
            raise ValueError("invalid reset receipt encoding") from exc
        if (
            set(state) != {"schema_version", "store_scope", "resets"}
            or type(state["schema_version"]) is not int
            or state["schema_version"] != 1
            or state["store_scope"] != scope
            or type(state["resets"]) is not list
            or len(state["resets"]) > MAX_RESET_RECORDS
        ):
            raise ValueError("invalid reset receipt")
        ids, archives = set(), set()
        for record in state["resets"]:
            if type(record) is not dict or set(record) != {
                "operation_id",
                "archive",
                "directory_device",
                "directory_inode",
                "phase",
            }:
                raise ValueError("invalid reset record")
            self._validate_reset_id(record["operation_id"])
            if record["archive"] is None:
                if (
                    record["directory_device"] is not None
                    or record["directory_inode"] is not None
                    or record["phase"] != "completed"
                ):
                    raise ValueError("invalid empty reset outcome")
            else:
                if type(record["archive"]) is not str or not re.fullmatch(
                    r"plugins-reset-" + scope + r"-[0-9a-f]{32}", record["archive"]
                ):
                    raise ValueError("invalid reset archive")
                if any(
                    type(record[key]) is not int or record[key] < 0
                    for key in ("directory_device", "directory_inode")
                ) or record["phase"] not in ("pending", "completed"):
                    raise ValueError("invalid reset identity")
                if record["archive"] in archives:
                    raise ValueError("duplicate reset archive")
                archives.add(record["archive"])
            if record["operation_id"] in ids:
                raise ValueError("duplicate reset identity")
            ids.add(record["operation_id"])
        if sum(record["phase"] == "pending" for record in state["resets"]) > 1:
            raise ValueError("conflicting pending resets")
        return state, condition

    @staticmethod
    def _validate_reset_id(operation_id: str) -> None:
        if type(operation_id) is not str or not re.fullmatch(
            r"[^\s\x00-\x1f]{1,256}", operation_id
        ):
            raise ValueError("invalid reset operation ID")

    def _qualify_reset_archive(self, record: dict) -> Path | None:
        if record["archive"] is None:
            return None
        archive = self.store_dir.parent / record["archive"]
        result = secure_private_directory(archive, create=False, application_owned=True)
        if not result.verified_private:
            raise ValueError("unqualified reset archive")
        parent_fd = os.open(
            archive.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        try:
            archive_fd = os.open(
                archive.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            try:
                info = os.fstat(archive_fd)
                if (info.st_dev, info.st_ino) != (
                    record["directory_device"],
                    record["directory_inode"],
                ):
                    raise ValueError("reset archive identity mismatch")
                os.fsync(archive_fd)
                os.fsync(parent_fd)
            finally:
                os.close(archive_fd)
        finally:
            os.close(parent_fd)
        return archive

    def _require_completed_resets(self) -> None:
        state, _ = self._load_reset_state()
        if any(record["phase"] == "pending" for record in state["resets"]):
            raise ValueError("plugin reset recovery required")
        if state["resets"]:
            # A visible completed receipt may itself have failed parent fsync.
            _sync_existing(self._reset_state_path)
            for record in state["resets"]:
                self._qualify_reset_archive(record)

    def reset(self, *, operation_id: str) -> Path | None:
        """Perform or retry one caller-reviewed plugin-only reset.

        The caller retains the operation ID with its review. Same-ID retries
        return only that archive, even after another namespace was bootstrapped.
        A new reviewed reset uses a new ID; pending different IDs are refused.
        Protected cleanup receipts preserve exact archive identities and pending
        durability across fresh instances. They never authorize execution.
        """
        self._require_posture()
        self._validate_reset_id(operation_id)
        state, condition = self._load_reset_state()
        record = next(
            (item for item in state["resets"] if item["operation_id"] == operation_id),
            None,
        )
        if record is not None and record["phase"] == "completed":
            archive = self._qualify_reset_archive(record)
            _sync_existing(self._reset_state_path)
            return archive
        if any(
            item["phase"] == "pending" and item["operation_id"] != operation_id
            for item in state["resets"]
        ):
            raise ValueError("different plugin reset pending")
        if record is None:
            if len(state["resets"]) >= MAX_RESET_RECORDS:
                raise ValueError("reset receipt history limit")
            if not os.path.lexists(self.store_dir):
                if self.load_marker() is not None:
                    raise ValueError("orphaned plugin reset marker")
                state["resets"].append(
                    {
                        "operation_id": operation_id,
                        "archive": None,
                        "directory_device": None,
                        "directory_inode": None,
                        "phase": "completed",
                    }
                )
                _write(self._reset_state_path, state, precondition=condition)
                return None
            self.lock()
            _directory(self.store_dir)
            info = self.store_dir.stat(follow_symlinks=False)
            record = {
                "operation_id": operation_id,
                "archive": f"plugins-reset-{state['store_scope']}-{secrets.token_hex(16)}",
                "directory_device": info.st_dev,
                "directory_inode": info.st_ino,
                "phase": "pending",
            }
            state["resets"].append(record)
            _write(self._reset_state_path, state, precondition=condition)
            state, condition = self._load_reset_state()
            record = state["resets"][-1]
        self.lock()
        archive = self.store_dir.parent / record["archive"]
        if os.path.lexists(self.store_dir):
            _directory(self.store_dir)
            parent_fd = os.open(
                self.store_dir.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            )
            try:
                info = os.stat(
                    self.store_dir.name, dir_fd=parent_fd, follow_symlinks=False
                )
                if (info.st_dev, info.st_ino) != (
                    record["directory_device"],
                    record["directory_inode"],
                ) or os.path.lexists(archive):
                    raise ValueError("reset source/archive identity conflict")
                self.marker_store.clear()
                if self.load_marker() is not None:
                    raise ValueError("plugin marker reset incomplete")
                os.rename(
                    self.store_dir.name,
                    archive.name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
            finally:
                os.close(parent_fd)
        elif self.load_marker() is not None:
            raise ValueError("reset marker changed after archive")
        self._qualify_reset_archive(record)
        record["phase"] = "completed"
        _write(self._reset_state_path, state, precondition=condition)
        return archive
