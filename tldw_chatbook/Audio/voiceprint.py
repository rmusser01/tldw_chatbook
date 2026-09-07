"""Encrypted self-voiceprint store (TASK-31826).

Owns the single self voiceprint (an ECAPA centroid) used to auto-name a
known speaker across meetings. Pure Python -- no torch, no numpy at module
scope -- so importing this module (or the `Audio` package) never pulls in
the diarization stack. See Docs/superpowers/specs/
2026-09-06-meeting-voiceprint-design.md section 3.1 for the design.

The record is encrypted at rest with the store's OWN random key (not the
config-encryption password, whose availability depends on the user
unlocking an encrypted config). The key lives in the OS keyring when a real
backend is available, else in an owner-only key file next to the record.
Keyring reads happen on a worker thread with a short timeout so a blocked
or prompting Keychain never delays a meeting Start -- see
`KeyProvider.get`. The voiceprint vector itself must never be logged.
"""
from __future__ import annotations

import json
import math
import os
import secrets
import stat
import threading
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from tldw_chatbook.Utils.config_encryption import ConfigEncryption

_FORMAT_VERSION = 1
_KEYRING_SERVICE = "tldw_chatbook"
_KEYRING_USERNAME = "meeting-voiceprint"


class ModelMismatch(Exception):
    """Raised by `VoiceprintStore.import_` when models differ and replace=False."""


@dataclass
class Voiceprint:
    model_id: str
    centroid: list[float]  # unit-normalised
    sample_count: float  # effective samples
    meetings_contributed: int
    created_at: str
    updated_at: str
    threshold_used: float
    last_best_similarity: float | None = None
    format_version: int = 1


@dataclass
class LoadResult:
    voiceprint: Voiceprint | None
    reason: str | None  # None | "no_voiceprint" | "needs_reenrollment" | "cannot_decrypt" | "keyring_locked"
    mode: str | None


class KeyProvider(Protocol):
    mode: str

    def get_or_create(self) -> str:
        """Return the store's key, minting and persisting one if absent.

        May raise. Only called from enrollment paths (`VoiceprintStore.save`).
        """
        ...

    def get(self, timeout_s: float) -> str | None:
        """Return the store's key, or None if missing/blocked. Never raises."""
        ...


def unit_normalise(v: Sequence[float]) -> list[float]:
    """Return `v` scaled to unit length (the zero vector maps to itself)."""
    values = [float(x) for x in v]
    magnitude = math.sqrt(sum(x * x for x in values))
    if magnitude == 0.0:
        return values
    return [x / magnitude for x in values]


def _atomic_write(path: Path, content: str) -> None:
    """Write `content` to `path` via temp-file + `os.replace` (crash-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(content)
    os.replace(tmp_path, path)


class KeyringKeyProvider:
    """Stores the voiceprint key in the OS keyring."""

    mode = "keyring"

    def get_or_create(self) -> str:
        import keyring

        existing = keyring.get_password(_KEYRING_SERVICE, _KEYRING_USERNAME)
        if existing:
            return existing
        new_key = secrets.token_urlsafe(32)
        keyring.set_password(_KEYRING_SERVICE, _KEYRING_USERNAME, new_key)
        return new_key

    def get(self, timeout_s: float) -> str | None:
        result: list[str | None] = [None]

        def _read() -> None:
            try:
                import keyring

                result[0] = keyring.get_password(_KEYRING_SERVICE, _KEYRING_USERNAME)
            except Exception:
                result[0] = None

        thread = threading.Thread(target=_read, daemon=True)
        thread.start()
        thread.join(timeout_s)
        if thread.is_alive():
            return None
        return result[0]


class KeyfileKeyProvider:
    """Stores the voiceprint key in an owner-only key file (keyring fallback)."""

    mode = "keyfile"

    def __init__(self, path: Path):
        self._path = Path(path)

    def get_or_create(self) -> str:
        if self._path.exists() and self._has_safe_permissions():
            return self._path.read_text().strip()
        key = secrets.token_urlsafe(32)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".tmp")
        tmp_path.write_text(key)
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, self._path)
        return key

    def get(self, timeout_s: float) -> str | None:
        try:
            if not self._path.exists() or not self._has_safe_permissions():
                return None
            return self._path.read_text().strip()
        except OSError:
            return None

    def _has_safe_permissions(self) -> bool:
        # Refuse a key file readable/writable by group or other, as ssh does.
        mode = stat.S_IMODE(self._path.stat().st_mode)
        return not (mode & 0o077)


class VoiceprintStore:
    """Owns the single encrypted self voiceprint at `path`."""

    def __init__(
        self,
        path: Path,
        key_provider: KeyProvider,
        *,
        clock=None,
        per_meeting_cap: float = 20.0,
    ) -> None:
        self._path = Path(path)
        self._key_provider = key_provider
        self._clock = clock or (lambda: datetime.now(timezone.utc).isoformat())
        self._per_meeting_cap = per_meeting_cap

    def load(self, expected_model_id: str | None = None, timeout_s: float = 1.5) -> LoadResult:
        if not self._path.exists():
            return LoadResult(voiceprint=None, reason="no_voiceprint", mode=None)

        try:
            envelope = json.loads(self._path.read_text())
        except (OSError, json.JSONDecodeError):
            return LoadResult(voiceprint=None, reason="cannot_decrypt", mode=None)

        mode = envelope.get("mode")
        key = self._key_provider.get(timeout_s)
        if key is None:
            return LoadResult(voiceprint=None, reason="keyring_locked", mode=mode)

        try:
            payload_json = ConfigEncryption().decrypt_value(envelope["payload"], key)
            record = Voiceprint(**json.loads(payload_json))
        except Exception:
            return LoadResult(voiceprint=None, reason="cannot_decrypt", mode=mode)

        if expected_model_id is not None and record.model_id != expected_model_id:
            return LoadResult(voiceprint=None, reason="needs_reenrollment", mode=mode)

        return LoadResult(voiceprint=record, reason=None, mode=mode)

    def save(self, record: Voiceprint) -> None:
        key = self._key_provider.get_or_create()
        payload_json = json.dumps(asdict(record))
        envelope = {
            "format_version": _FORMAT_VERSION,
            "mode": self._key_provider.mode,
            "payload": ConfigEncryption().encrypt_value(payload_json, key),
        }
        _atomic_write(self._path, json.dumps(envelope))

    def merge_sample(self, centroid: Sequence[float], weight: float, model_id: str) -> Voiceprint:
        result = self.load(expected_model_id=model_id)
        if result.voiceprint is None:
            raise ValueError(f"cannot merge sample: {result.reason or 'no_voiceprint'}")
        current = result.voiceprint

        w = min(weight, self._per_meeting_cap)
        sample = unit_normalise(centroid)
        n = current.sample_count
        total = n + w
        merged = unit_normalise(
            [(c * n + s * w) / total for c, s in zip(current.centroid, sample)]
        ) if total else list(current.centroid)

        updated = replace(
            current,
            centroid=merged,
            sample_count=total,
            meetings_contributed=current.meetings_contributed + 1,
            updated_at=self._clock(),
        )
        self.save(updated)
        return updated

    def delete(self) -> bool:
        if self._path.exists():
            self._path.unlink()
            return True
        return False

    def export(self, dest: Path, passphrase: str) -> None:
        if not passphrase:
            raise ValueError("export requires a non-empty passphrase")
        result = self.load()
        if result.voiceprint is None:
            raise ValueError(f"cannot export voiceprint: {result.reason or 'no_voiceprint'}")

        payload_json = json.dumps(asdict(result.voiceprint))
        envelope = {
            "format_version": _FORMAT_VERSION,
            "mode": "passphrase",
            "payload": ConfigEncryption().encrypt_value(payload_json, passphrase),
        }
        _atomic_write(Path(dest), json.dumps(envelope))

    def import_(self, src: Path, passphrase: str, *, replace: bool) -> Voiceprint:
        envelope = json.loads(Path(src).read_text())
        if envelope.get("mode") != "passphrase":
            raise ValueError("import source is not passphrase-encrypted")

        payload_json = ConfigEncryption().decrypt_value(envelope["payload"], passphrase)
        imported = Voiceprint(**json.loads(payload_json))

        current = self.load().voiceprint
        if current is None or current.model_id == imported.model_id:
            if current is None:
                self.save(imported)
                return imported
            weight = min(imported.sample_count, self._per_meeting_cap)
            return self.merge_sample(imported.centroid, weight=weight, model_id=imported.model_id)

        if not replace:
            raise ModelMismatch(
                f"stored model {current.model_id!r} does not match imported model {imported.model_id!r}"
            )
        self.save(imported)
        return imported


def _has_real_keyring_backend() -> bool:
    try:
        import keyring
        from keyring.backends.fail import Keyring as _FailKeyring
    except Exception:
        return False
    return not isinstance(keyring.get_keyring(), _FailKeyring)


def default_store(user_data_dir: Path | None = None) -> VoiceprintStore:
    """Return the app's real voiceprint store: keyring key, keyfile fallback."""
    if user_data_dir is None:
        from tldw_chatbook.config import get_user_data_dir

        user_data_dir = get_user_data_dir()
    user_data_dir = Path(user_data_dir)

    key_provider: KeyProvider
    if _has_real_keyring_backend():
        key_provider = KeyringKeyProvider()
    else:
        key_provider = KeyfileKeyProvider(user_data_dir / "voiceprint.key")

    return VoiceprintStore(user_data_dir / "voiceprint.json", key_provider)
