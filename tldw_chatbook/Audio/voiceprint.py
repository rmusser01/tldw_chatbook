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
import tempfile
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
    """Raised by `VoiceprintStore.import_` when models differ and replace=False,
    and by `merge_sample` for a centroid of a different DIMENSION -- same
    model id, different shape is still a model the stored vector cannot be
    averaged with (final review I2)."""


class WrongPassphrase(ValueError):
    """Raised by `VoiceprintStore.import_` when the passphrase does not open
    the file. A `ValueError` still, so existing callers are unaffected; named
    so the screen can say "Wrong passphrase" instead of the class name
    (final review Minor 4)."""


class ExportRefused(ValueError):
    """Raised by `VoiceprintStore.export` for a destination that IS the
    store's own record -- writing a passphrase envelope over it makes every
    later load report `cannot_decrypt`, unrecoverably (final review Minor 3)."""


class StoreUnavailable(Exception):
    """Raised when a store operation cannot safely proceed: the existing
    record can't be read (locked key, corrupt file) and the caller either
    didn't ask to replace it, or the failure mode is too ambiguous to
    replace safely (a locked key never justifies overwriting)."""


class _EnvelopeError(Exception):
    """Internal: envelope file missing, corrupt, or not a JSON object."""


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
        """Return the store's key, or None if missing/blocked.

        Raises `StoreUnavailable` only for a store the user must REPAIR (a
        key file with loose permissions) -- never for one that may simply
        unblock itself, which is what None means (task 6 review M8).
        """
        ...


def unit_normalise(v: Sequence[float]) -> list[float]:
    """Return `v` scaled to unit length (the zero vector maps to itself)."""
    values = [float(x) for x in v]
    magnitude = math.sqrt(sum(x * x for x in values))
    if magnitude == 0.0:
        return values
    return [x / magnitude for x in values]


def _atomic_write(path: Path, content: str) -> None:
    """Write `content` to `path` atomically, owner-only (0o600).

    The temp file is `tempfile.mkstemp`-unique, not `path.with_suffix(".tmp")`
    (final review Minor 1): `export()` writes to a path the USER typed, and a
    `<name>.tmp` of their own next to it was created and then replaced away.
    mkstemp also opens at 0o600, so neither the key file nor the record --
    both biometric-derived -- is ever briefly world-readable under a
    permissive umask (Minor 2), which is what the separate `_write_restricted`
    used to buy for the key file alone.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(content)
        os.replace(tmp_name, path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _read_envelope(path: Path) -> dict:
    """Read and JSON-parse an envelope file, raising `_EnvelopeError` for
    anything that isn't a readable JSON object (missing file, bad JSON, or
    valid JSON that isn't a dict -- e.g. a bare `null`)."""
    try:
        envelope = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise _EnvelopeError(str(exc)) from exc
    if not isinstance(envelope, dict):
        raise _EnvelopeError("envelope is not a JSON object")
    return envelope


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
        if self._path.exists():
            if not self._has_safe_permissions():
                raise StoreUnavailable(
                    f"key file {self._path} has unsafe permissions; chmod it "
                    "to 600 (or delete it to mint a new key) before continuing"
                )
            return self._path.read_text().strip()
        key = secrets.token_urlsafe(32)
        _atomic_write(self._path, key)
        return key

    def get(self, timeout_s: float) -> str | None:
        try:
            if not self._path.exists():
                return None
            if not self._has_safe_permissions():
                # RAISED, not None (task 6 review M8): None means "blocked
                # key", which the rail renders as "keyring locked" -- and a
                # key-file install has no keyring to unlock. This surfaces as
                # `store_unavailable`, whose repair is the chmod below.
                raise StoreUnavailable(
                    f"key file {self._path} has unsafe permissions; chmod it to 600"
                )
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

    @property
    def mode(self) -> str:
        """The key mode in force ("keyring" | "keyfile"), for Settings.

        Reads the provider's declared mode -- no key access, no prompt.
        """
        return self._key_provider.mode

    def exists(self) -> bool:
        """Whether a stored record is present, WITHOUT touching the key.

        A stat, so a rail that only needs "is there a voiceprint at all?"
        (the Meetings prepare pass, which runs at screen mount) never raises
        the Keychain prompt that `load` may -- that read belongs at meeting
        Start, on a thread, with a timeout.
        """
        return self._path.exists()

    def load(self, expected_model_id: str | None = None, timeout_s: float = 1.5) -> LoadResult:
        if not self._path.exists():
            return LoadResult(voiceprint=None, reason="no_voiceprint", mode=None)

        try:
            envelope = _read_envelope(self._path)
        except _EnvelopeError:
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
        if len(sample) != len(current.centroid):
            # `zip` below truncates silently, and the short vector it produced
            # was saved, reloaded and handed to the diarizer worker -- where
            # every comparison against a full-length embedding raised and took
            # live speaker labels down with it (final review I2). Same model
            # id, different dimension IS a model mismatch: the screen already
            # renders that as "Different model -- choose Replace".
            raise ModelMismatch(
                f"stored voiceprint has {len(current.centroid)} dimensions, "
                f"the sample has {len(sample)}"
            )
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
        dest = Path(dest)
        if dest.resolve() == self._path.resolve():
            # The one destination that destroys the thing being exported
            # (final review Minor 3): a passphrase envelope over the record
            # makes every later `load()` report `cannot_decrypt`, with the
            # store's own key no longer able to open it.
            raise ExportRefused("that file is the stored voiceprint itself")
        result = self.load()
        if result.voiceprint is None:
            raise ValueError(f"cannot export voiceprint: {result.reason or 'no_voiceprint'}")

        payload_json = json.dumps(asdict(result.voiceprint))
        envelope = {
            "format_version": _FORMAT_VERSION,
            "mode": "passphrase",
            "payload": ConfigEncryption().encrypt_value(payload_json, passphrase),
        }
        _atomic_write(dest, json.dumps(envelope))

    def import_(self, src: Path, passphrase: str, *, replace: bool) -> Voiceprint:
        try:
            envelope = _read_envelope(src)
        except _EnvelopeError as exc:
            raise ValueError(f"import source is not readable: {exc}") from exc
        if envelope.get("mode") != "passphrase":
            raise ValueError("import source is not passphrase-encrypted")

        try:
            payload_json = ConfigEncryption().decrypt_value(envelope["payload"], passphrase)
        except Exception as exc:  # noqa: BLE001 - a bare ValueError reached the
            # screen as "Import failed (ValueError)." (final review Minor 4).
            raise WrongPassphrase("the passphrase did not open this file") from exc
        imported = Voiceprint(**json.loads(payload_json))

        # The existing record's readability gates whether -- and how -- we
        # may proceed. "No record" is a genuinely empty store: always safe
        # to adopt the import. A locked key or a corrupt file means we
        # CANNOT SEE what's currently stored, which is not the same thing
        # as nothing being stored -- silently overwriting it there would
        # discard a different person's voiceprint with no gate at all. A
        # locked key can unblock itself (unlock the keyring) so it is never
        # a valid reason to replace; a corrupt file cannot recover on its
        # own, so an explicit `replace=True` may discard it.
        result = self.load()
        if result.reason == "keyring_locked":
            raise StoreUnavailable("cannot import: the existing voiceprint key is locked")
        if result.reason == "cannot_decrypt":
            if not replace:
                raise StoreUnavailable(
                    "cannot import: the existing voiceprint is unreadable "
                    "(pass replace=True to overwrite it)"
                )
            self.save(imported)
            return imported

        current = result.voiceprint
        if current is None:
            self.save(imported)
            return imported
        if current.model_id == imported.model_id:
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
