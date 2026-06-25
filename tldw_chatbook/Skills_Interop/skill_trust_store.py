"""Persistence for local skill trust manifests, snapshots, markers, and keys."""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..runtime_policy.server_credentials import is_secure_keyring_backend
from .skill_trust_crypto import (
    SkillTrustKeys,
    canonical_json,
    decrypt_json_blob,
    encrypt_json_blob,
    manifest_mac,
    sha256_hex,
)


_MANIFEST_FILENAME = "skill_trust_manifest.json"
_SNAPSHOTS_DIRNAME = "snapshots"
_DEFAULT_MARKER_SERVICE_NAME = "tldw_chatbook.skill_trust"
_DEFAULT_KEY_CACHE_SERVICE_NAME = "tldw_chatbook.skill_trust.keys"
_MARKER_USERNAME = "local-skills:generation-marker:v1"
_KEY_CACHE_USERNAME = "local-skills:trust-root:v1"
_KEY_CACHE_FIELDS = (
    "manifest_mac_key",
    "snapshot_key",
    "audit_mac_key",
    "wrapped_root_key",
)


class SkillTrustMarkerUnavailable(RuntimeError):
    """Raised when rollback marker storage cannot provide full protection."""

    reason_code = "rollback_marker_unavailable"


class SkillTrustGenerationMarkerStore(Protocol):
    """Persistence boundary for the latest accepted trust manifest marker."""

    def load_marker(self) -> dict[str, Any] | None:
        """Return the persisted marker, or ``None`` when no marker exists."""
        ...

    def save_marker(self, *, generation: int, manifest_digest: str) -> None:
        """Persist the latest accepted manifest generation and canonical digest."""
        ...


@dataclass(slots=True)
class FileSkillTrustGenerationMarkerStore:
    """Reduced-protection marker store intended for tests and explicit recovery."""

    marker_path: Path

    def load_marker(self) -> dict[str, Any] | None:
        """Load the file-backed generation marker if it exists."""

        if not self.marker_path.exists():
            return None
        payload = json.loads(self.marker_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        return payload

    def save_marker(self, *, generation: int, manifest_digest: str) -> None:
        """Atomically save the file-backed generation marker."""

        self.marker_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generation": generation,
            "manifest_digest": manifest_digest,
        }
        _atomic_write_json(self.marker_path, payload)


class UnavailableSkillTrustGenerationMarkerStore:
    """Marker store used when no secure OS-backed marker backend is available."""

    def __init__(self, message: str) -> None:
        self.message = message

    def __repr__(self) -> str:
        return "UnavailableSkillTrustGenerationMarkerStore(message=<redacted>)"

    def _raise_unavailable(self) -> None:
        raise SkillTrustMarkerUnavailable(self.message)

    def load_marker(self) -> dict[str, Any] | None:
        """Raise because rollback marker storage is unavailable."""

        self._raise_unavailable()

    def save_marker(self, *, generation: int, manifest_digest: str) -> None:
        """Raise because rollback marker storage is unavailable."""

        self._raise_unavailable()


@dataclass(slots=True, repr=False)
class KeyringSkillTrustGenerationMarkerStore:
    """Secure OS-keyring-backed generation marker store."""

    service_name: str = _DEFAULT_MARKER_SERVICE_NAME
    keyring_backend: Any | None = None

    def __post_init__(self) -> None:
        keyring_backend = _resolve_keyring_backend(self.keyring_backend)
        if not is_secure_keyring_backend(keyring_backend):
            raise SkillTrustMarkerUnavailable(
                "No secure OS-backed generation marker store is available."
            )
        self.keyring_backend = keyring_backend

    def __repr__(self) -> str:
        return "KeyringSkillTrustGenerationMarkerStore(keyring_backend=<redacted>)"

    def load_marker(self) -> dict[str, Any] | None:
        """Load the marker from secure keyring storage."""

        payload = self.keyring_backend.get_password(self.service_name, _MARKER_USERNAME)
        if not payload:
            return None
        marker = json.loads(payload)
        if not isinstance(marker, dict):
            return None
        return marker

    def save_marker(self, *, generation: int, manifest_digest: str) -> None:
        """Save the marker to secure keyring storage."""

        payload = json.dumps(
            {
                "generation": generation,
                "manifest_digest": manifest_digest,
            },
            sort_keys=True,
        )
        self.keyring_backend.set_password(self.service_name, _MARKER_USERNAME, payload)


def build_default_skill_trust_marker_store(
    keyring_backend: Any | None = None,
) -> SkillTrustGenerationMarkerStore:
    """Return a secure default marker store or a fail-closed unavailable store."""

    try:
        return KeyringSkillTrustGenerationMarkerStore(keyring_backend=keyring_backend)
    except Exception as exc:
        return UnavailableSkillTrustGenerationMarkerStore(str(exc))


@dataclass(slots=True, repr=False)
class KeyringSkillTrustKeyCache:
    """Optional secure keyring cache for derived trust key material."""

    service_name: str = _DEFAULT_KEY_CACHE_SERVICE_NAME
    keyring_backend: Any | None = None

    def __post_init__(self) -> None:
        keyring_backend = _resolve_keyring_backend(self.keyring_backend)
        if not is_secure_keyring_backend(keyring_backend):
            raise SkillTrustMarkerUnavailable("No secure OS-backed key cache is available.")
        self.keyring_backend = keyring_backend

    def __repr__(self) -> str:
        return "KeyringSkillTrustKeyCache(keyring_backend=<redacted>)"

    def save_keys(self, keys: SkillTrustKeys, *, salt: bytes) -> None:
        """Store derived key material in a secure keyring, never the passphrase."""

        salt_digest = _salt_digest(salt)
        payload = {
            "version": 1,
            "salt_digest": salt_digest,
            "manifest_mac_key": _encode_bytes(keys.manifest_mac_key),
            "snapshot_key": _encode_bytes(keys.snapshot_key),
            "audit_mac_key": _encode_bytes(keys.audit_mac_key),
            "wrapped_root_key": _encode_bytes(keys.wrapped_root_key),
        }
        self.keyring_backend.set_password(
            self.service_name,
            _KEY_CACHE_USERNAME,
            json.dumps(payload, sort_keys=True),
        )

    def load_keys(self, *, expected_salt: bytes) -> SkillTrustKeys | None:
        """Load cached derived trust keys from the secure keyring."""

        expected_salt_digest = _salt_digest(expected_salt)
        payload = self.keyring_backend.get_password(self.service_name, _KEY_CACHE_USERNAME)
        if not payload:
            return None
        data = json.loads(payload)
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("skill trust key cache invalid")
        if data.get("salt_digest") != expected_salt_digest:
            return None
        keys = {field: _decode_32_byte_key(data, field) for field in _KEY_CACHE_FIELDS}
        return SkillTrustKeys(**keys)


def build_default_skill_trust_key_cache(
    keyring_backend: Any | None = None,
) -> KeyringSkillTrustKeyCache | None:
    """Return a secure key cache, or ``None`` when unavailable."""

    try:
        return KeyringSkillTrustKeyCache(keyring_backend=keyring_backend)
    except Exception:
        return None


@dataclass(slots=True)
class SkillTrustStore:
    """File persistence for authenticated trust manifests and snapshots."""

    store_dir: Path
    marker_store: SkillTrustGenerationMarkerStore

    @property
    def manifest_path(self) -> Path:
        """Path to the authenticated skill trust manifest payload."""

        return self.store_dir / _MANIFEST_FILENAME

    @property
    def snapshots_dir(self) -> Path:
        """Directory containing encrypted trusted skill snapshots."""

        return self.store_dir / _SNAPSHOTS_DIRNAME

    def has_manifest(self) -> bool:
        """Return whether a trust manifest payload exists on disk."""

        return self.manifest_path.exists()

    def manifest_digest(self, manifest: dict[str, Any]) -> str:
        """Return the canonical JSON SHA-256 digest for a manifest."""

        return sha256_hex(canonical_json(manifest))

    def load_salt(self) -> bytes:
        """Load and validate the 32-byte KDF salt stored with the manifest."""

        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        encoded = payload.get("kdf_salt")
        if not isinstance(encoded, str):
            raise ValueError("skill trust salt missing")
        try:
            salt = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise ValueError("skill trust salt invalid") from exc
        if len(salt) != 32:
            raise ValueError("skill trust salt invalid")
        return salt

    def save_manifest(
        self,
        manifest: dict[str, Any],
        keys: SkillTrustKeys,
        *,
        salt: bytes | None = None,
    ) -> None:
        """Persist an authenticated manifest and update its generation marker."""

        if salt is None:
            salt = self.load_salt()
        if not isinstance(salt, bytes) or len(salt) != 32:
            raise ValueError("skill trust salt invalid")

        manifest_payload = _json_safe_payload(manifest)
        if not isinstance(manifest_payload, dict):
            raise ValueError("skill trust manifest must be an object")

        payload = {
            "kdf_salt": _encode_bytes(salt),
            "manifest": manifest_payload,
            "mac": manifest_mac(manifest_payload, keys.manifest_mac_key),
        }
        self.store_dir.mkdir(parents=True, exist_ok=True)
        previous_manifest_bytes = self.manifest_path.read_bytes() if self.has_manifest() else None
        previous_marker = self.marker_store.load_marker() if previous_manifest_bytes is not None else None
        _atomic_write_json(self.manifest_path, payload, indent=2)
        try:
            self.marker_store.save_marker(
                generation=int(manifest_payload["generation"]),
                manifest_digest=self.manifest_digest(manifest_payload),
            )
        except Exception:
            if previous_manifest_bytes is None:
                self.manifest_path.unlink(missing_ok=True)
            else:
                _atomic_write_bytes(self.manifest_path, previous_manifest_bytes)
            if previous_marker is not None:
                self.marker_store.save_marker(
                    generation=int(previous_marker["generation"]),
                    manifest_digest=str(previous_marker["manifest_digest"]),
                )
            raise

    def load_manifest(self, keys: SkillTrustKeys) -> dict[str, Any]:
        """Load a manifest after verifying its HMAC and generation marker."""

        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("manifest authentication failed")
        manifest = payload.get("manifest")
        tag = payload.get("mac")
        if not isinstance(manifest, dict) or not isinstance(tag, str):
            raise ValueError("manifest authentication failed")
        if manifest_mac(manifest, keys.manifest_mac_key) != tag:
            raise ValueError("manifest authentication failed")

        marker = self.marker_store.load_marker()
        digest = self.manifest_digest(manifest)
        if marker is None:
            raise ValueError("manifest generation marker mismatch")
        if int(marker.get("generation", -1)) != int(manifest.get("generation", -2)):
            raise ValueError("manifest generation marker mismatch")
        if marker.get("manifest_digest") != digest:
            raise ValueError("manifest generation marker mismatch")
        return manifest

    def save_snapshot(
        self,
        snapshot_id: str,
        payload: dict[str, Any],
        keys: SkillTrustKeys,
        *,
        generation: int,
    ) -> None:
        """Encrypt and persist a trusted snapshot payload."""

        snapshot_payload = _json_safe_payload(payload)
        if not isinstance(snapshot_payload, dict):
            raise ValueError("skill trust snapshot payload must be an object")
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        encrypted = encrypt_json_blob(
            snapshot_payload,
            keys.snapshot_key,
            associated_data=_snapshot_associated_data(snapshot_id, generation),
        )
        _atomic_write_json(self._snapshot_path(snapshot_id), encrypted, indent=2)

    def load_snapshot(
        self,
        snapshot_id: str,
        keys: SkillTrustKeys,
        *,
        generation: int,
    ) -> dict[str, Any]:
        """Load and decrypt a trusted snapshot payload."""

        encrypted = json.loads(self._snapshot_path(snapshot_id).read_text(encoding="utf-8"))
        if not isinstance(encrypted, dict):
            raise ValueError("snapshot authentication failed")
        return decrypt_json_blob(
            encrypted,
            keys.snapshot_key,
            associated_data=_snapshot_associated_data(snapshot_id, generation),
        )

    def _snapshot_path(self, snapshot_id: str) -> Path:
        if not snapshot_id or snapshot_id in {".", ".."}:
            raise ValueError("skill trust snapshot id invalid")
        if "/" in snapshot_id or "\\" in snapshot_id:
            raise ValueError("skill trust snapshot id invalid")
        return self.snapshots_dir / f"{snapshot_id}.json"


def _resolve_keyring_backend(keyring_backend: Any | None) -> Any:
    if keyring_backend is None:
        import keyring

        keyring_backend = keyring.get_keyring()
    get_keyring = getattr(keyring_backend, "get_keyring", None)
    if callable(get_keyring):
        return get_keyring()
    return keyring_backend


def _snapshot_associated_data(snapshot_id: str, generation: int) -> bytes:
    return f"snapshot:{snapshot_id}:generation:{generation}".encode("utf-8")


def _atomic_write_json(path: Path, payload: Any, *, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    text = json.dumps(payload, indent=indent, sort_keys=True) + "\n"
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_bytes(payload)
    temp_path.replace(path)


def _encode_bytes(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _decode_32_byte_key(payload: dict[str, Any], field: str) -> bytes:
    encoded = payload.get(field)
    if not isinstance(encoded, str):
        raise ValueError("skill trust key cache invalid")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("skill trust key cache invalid") from exc
    if len(decoded) != 32:
        raise ValueError("skill trust key cache invalid")
    return decoded


def _salt_digest(salt: bytes) -> str:
    if not isinstance(salt, bytes) or len(salt) != 32:
        raise ValueError("skill trust salt invalid")
    return sha256_hex(salt)


def _json_safe_payload(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        result: dict[str, Any] = {}
        for key, value in payload.items():
            if not isinstance(key, str):
                raise ValueError("skill trust JSON mapping keys must be strings")
            result[key] = _json_safe_payload(value)
        return result
    if isinstance(payload, tuple):
        return [_json_safe_payload(value) for value in payload]
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return [_json_safe_payload(value) for value in payload]
    if isinstance(payload, float) and not math.isfinite(payload):
        raise ValueError("skill trust JSON numbers must be finite")
    return payload
