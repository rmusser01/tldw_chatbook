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
from collections.abc import Callable, Sequence
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


class InvalidVoiceprint(StoreUnavailable):
    """Raised by `VoiceprintStore.import_` for a file that decrypts but does
    not hold a usable voiceprint (Qodo review 5): a missing or wrong-typed
    field, an unknown `format_version`, an empty or non-finite centroid, a
    negative count. A `StoreUnavailable` so nothing that already handles the
    store's refusals has to change, but its own class so the screen can say
    "Import file is not a valid voiceprint" instead of "Store locked".

    The message is always static -- the payload came from a file the user was
    handed, and echoing it would put a stranger's data on screen and in logs.
    """


class _EnvelopeError(Exception):
    """Internal: envelope file missing, corrupt, or not a JSON object."""


@dataclass
class Voiceprint:
    """The stored self voiceprint: one speaker centroid and its provenance.

    Attributes:
        model_id: The embedding model that produced `centroid`
            (`diarizer_worker.MODEL_ID`). A record from another model is
            reported as `needs_reenrollment` rather than matched.
        centroid: The unit-normalised speaker vector. NEVER logged, notified,
            or written anywhere unencrypted.
        sample_count: Effective samples behind the centroid -- the weight in
            `VoiceprintStore.merge_sample`'s running mean, not a count of
            meetings.
        meetings_contributed: How many meetings have merged into this record.
        created_at: ISO timestamp of the first enrollment.
        updated_at: ISO timestamp of the last save.
        threshold_used: The cosine-distance threshold enrollment was made
            with, handed to the worker as the match gate.
        last_best_similarity: The best similarity seen for this speaker in
            the last meeting, for the Settings diagnostic; None until one
            has run.
        format_version: On-disk record shape; only `1` is readable.
    """

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
    """What `VoiceprintStore.load` found -- a record, or why there is none.

    Attributes:
        voiceprint: The record, or None. `reason` says which.
        reason: None when `voiceprint` is set, else one of "no_voiceprint",
            "needs_reenrollment" (a record from another model),
            "cannot_decrypt" (corrupt file, or a key that no longer opens
            it) or "keyring_locked" (the key is present but blocked, and may
            unblock itself). The Meetings rail maps each to static copy.
        mode: The envelope's declared key mode ("keyring" | "keyfile"), or
            None when no envelope could be read.
    """

    voiceprint: Voiceprint | None
    reason: str | None
    mode: str | None


class KeyProvider(Protocol):
    """Supplies the key the record is encrypted with.

    Two implementations ship: `KeyringKeyProvider` (the OS keyring) and
    `KeyfileKeyProvider` (an owner-only file beside the record).

    Attributes:
        mode: "keyring" | "keyfile" -- stamped into the envelope and shown in
            Settings. Reading it must never touch the key itself.
    """

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


def _is_finite(value: object) -> bool:
    """Whether `value` is a real, finite number (a bool is not a number here)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validated(payload: object) -> Voiceprint:
    """Return the imported record, or refuse it (Qodo review 5).

    The passphrase proves who wrote an import file, never what is in it: a
    hand-edited or truncated record used to be saved as-is and only failed
    later, in the diarizer worker, on every comparison of the meeting.

    Args:
        payload: The decrypted JSON body of an import file.

    Returns:
        The record, safe to store and to hand the worker.

    Raises:
        InvalidVoiceprint: Any field missing, wrongly typed, out of range, or
            non-finite, or a `format_version` this build does not read. The
            message is static -- never the payload.
    """
    try:
        record = Voiceprint(**payload)  # type: ignore[arg-type] - guarded below
    except TypeError as exc:  # not a dict, missing field, or an unknown one
        raise InvalidVoiceprint("the import file is not a valid voiceprint") from exc
    if (
        record.format_version != _FORMAT_VERSION
        or not isinstance(record.model_id, str) or not record.model_id
        or not isinstance(record.centroid, list) or not record.centroid
        or not all(_is_finite(x) for x in record.centroid)
        or not _is_finite(record.sample_count) or record.sample_count < 0
        or not isinstance(record.meetings_contributed, int)
        or isinstance(record.meetings_contributed, bool)
        or record.meetings_contributed < 0
        or not _is_finite(record.threshold_used)
        or not isinstance(record.created_at, str) or not isinstance(record.updated_at, str)
        or not (record.last_best_similarity is None or _is_finite(record.last_best_similarity))
    ):
        raise InvalidVoiceprint("the import file is not a valid voiceprint")
    return record


def unit_normalise(v: Sequence[float]) -> list[float]:
    """Scale a vector to unit length.

    Args:
        v: The vector to normalise.

    Returns:
        `v` as a list scaled to length 1 -- the zero vector maps to itself,
        since there is no direction to preserve.
    """
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
    """Stores the voiceprint key in the OS keyring.

    Chosen by `default_store` whenever a real keyring backend is installed.
    """

    mode = "keyring"

    def get_or_create(self) -> str:
        """Return the keyring's key, minting and storing one if absent.

        Returns:
            The key, as a URL-safe token.

        Raises:
            Exception: Whatever the keyring backend raises -- an unavailable
                or refused Keychain is the caller's (enrollment's) problem to
                report, since it is the only path that may prompt.
        """
        import keyring

        existing = keyring.get_password(_KEYRING_SERVICE, _KEYRING_USERNAME)
        if existing:
            return existing
        new_key = secrets.token_urlsafe(32)
        keyring.set_password(_KEYRING_SERVICE, _KEYRING_USERNAME, new_key)
        return new_key

    def get(self, timeout_s: float) -> str | None:
        """Return the keyring's key, or None if it is missing or blocked.

        The read runs on a daemon thread and is abandoned on timeout: a
        Keychain that prompts (or hangs) must never delay a meeting Start.

        Args:
            timeout_s: How long to wait for the backend before giving up.

        Returns:
            The key, or None -- which callers render as "keyring locked", a
            condition that can clear itself once the user unlocks it.
        """
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
    """Stores the voiceprint key in an owner-only key file (keyring fallback).

    Used when no real keyring backend is installed. The file is created at
    0o600 and refused if it is ever readable by group or other, as ssh does.
    """

    mode = "keyfile"

    def __init__(self, path: Path):
        """Args:
            path: The key file, created on first use beside the record.
        """
        self._path = Path(path)

    def get_or_create(self) -> str:
        """Return the key file's key, minting the file if it is absent.

        Returns:
            The key, as a URL-safe token.

        Raises:
            StoreUnavailable: An existing key file is readable by group or
                other. It is never silently re-minted -- that would discard
                the voiceprint it opens; the repair is a chmod to 600.
            OSError: The file could not be read or written.
        """
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
        """Return the key file's key, or None if there is no key file.

        Args:
            timeout_s: Accepted for `KeyProvider` symmetry; a local file read
                cannot block the way a Keychain can, so it is unused.

        Returns:
            The key, or None when no key file exists (or it cannot be read).

        Raises:
            StoreUnavailable: The key file is readable by group or other.
                Raised rather than returned as None, which callers render as
                "keyring locked" -- a key-file install has no keyring to
                unlock, and the repair is a chmod to 600 (task 6 review M8).
        """
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
    """Owns the single encrypted self voiceprint at `path`.

    Reads never raise for an absent, locked or corrupt record -- `load`
    reports a reason instead -- while writes raise, so a caller that thought
    it saved something always did.
    """

    def __init__(
        self,
        path: Path,
        key_provider: KeyProvider,
        *,
        clock: Callable[[], str] | None = None,
        per_meeting_cap: float = 20.0,
    ) -> None:
        """Args:
            path: The record file. Its parent is created on first save.
            key_provider: Supplies the encryption key (`KeyringKeyProvider`
                or `KeyfileKeyProvider`; `default_store` picks one).
            clock: Zero-argument callable returning the ISO timestamp
                `merge_sample` stamps as `updated_at`. Defaults to UTC now.
            per_meeting_cap: Ceiling on one merge's weight, so a single long
                meeting cannot dominate the running mean.
        """
        self._path = Path(path)
        self._key_provider = key_provider
        self._clock = clock or (lambda: datetime.now(timezone.utc).isoformat())
        self._per_meeting_cap = per_meeting_cap
        # Every WRITER holds this (Qodo review 9). `merge_sample` is a
        # load-average-save, and an accepted learning offer overlapping an
        # explicit enrollment saved a record computed from the vector the
        # enrollment had just replaced -- the new enrollment, silently gone.
        # Reentrant because `merge_sample`/`import_` write through `save`.
        # In-process only: two app PROCESSES are a different (unreachable --
        # enrollment needs exclusive mic capture) problem, and a cross-platform
        # file lock is not worth carrying for it.
        self._write_lock = threading.RLock()

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
        """Read the stored record, reporting failures instead of raising.

        May touch the key, so it may prompt: call it at meeting Start, on a
        thread, never from a screen's mount (use `exists`).

        Args:
            expected_model_id: When given, a record from another model is
                reported as `needs_reenrollment` rather than returned -- its
                vector is not comparable with this build's embeddings.
            timeout_s: How long the key provider may take before the record
                counts as blocked ("keyring_locked").

        Returns:
            A `LoadResult`; `voiceprint` is None exactly when `reason` is set.
            This never raises for a missing, locked or corrupt record -- a
            meeting must start either way, with matching simply off.
        """
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
        """Encrypt `record` and replace the stored one atomically.

        Mints the key on first use, so this is the path that may raise a
        Keychain prompt. Holds the writer lock (Qodo review 9).

        Args:
            record: The complete record to store. It replaces whatever is
                there -- `merge_sample` is the averaging path.

        Raises:
            StoreUnavailable: The key file exists with unsafe permissions.
            OSError: The record could not be written (the previous file is
                left intact -- the write is a temp file plus a rename).
            Exception: Whatever the keyring backend raises when it cannot
                mint or store a key.
        """
        with self._write_lock:
            key = self._key_provider.get_or_create()
            payload_json = json.dumps(asdict(record))
            envelope = {
                "format_version": _FORMAT_VERSION,
                "mode": self._key_provider.mode,
                "payload": ConfigEncryption().encrypt_value(payload_json, key),
            }
            _atomic_write(self._path, json.dumps(envelope))

    def merge_sample(self, centroid: Sequence[float], weight: float, model_id: str) -> Voiceprint:
        """Average one meeting's centroid into the stored record and save it.

        The whole load-average-save runs under the writer lock, so a learning
        merge cannot overwrite a concurrent enrollment (Qodo review 9).

        Args:
            centroid: The sample's speaker vector; normalised here.
            weight: The sample's weight in seconds, capped at
                `per_meeting_cap` so one meeting cannot dominate.
            model_id: The model that produced `centroid`. A stored record
                from another model is refused, not merged.

        Returns:
            The saved, updated record (`meetings_contributed` incremented).

        Raises:
            ValueError: There is nothing to merge into (the reason names
                which: no record, locked key, unreadable file), or `centroid`
                holds a non-finite value.
            ModelMismatch: The stored vector has a different dimension --
                `zip` would truncate it silently, and the short vector that
                produced broke every comparison in the worker (final review
                I2). The screen renders this as "Different model".
        """
        with self._write_lock:      # load-average-save: one writer at a time
            result = self.load(expected_model_id=model_id)
            if result.voiceprint is None:
                raise ValueError(f"cannot merge sample: {result.reason or 'no_voiceprint'}")
            current = result.voiceprint

            if not all(_is_finite(x) for x in centroid):
                # A NaN reaches the stored vector and every later cosine
                # distance is NaN -- matching then neither fails nor succeeds,
                # it just stops (Qodo review 5).
                raise ValueError("cannot merge sample: the centroid is not finite")
            w = min(weight, self._per_meeting_cap)
            sample = unit_normalise(centroid)
            if len(sample) != len(current.centroid):
                # `zip` below truncates silently, and the short vector it
                # produced was saved, reloaded and handed to the diarizer
                # worker -- where every comparison against a full-length
                # embedding raised and took live speaker labels down with it
                # (final review I2). Same model id, different dimension IS a
                # model mismatch: the screen already renders that as
                # "Different model -- choose Replace".
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
        """Remove the stored record. The key stays and then holds nothing.

        A meeting already running keeps its in-memory copy until it ends
        (spec §3.1).

        Returns:
            True if a record was removed, False if there was none.

        Raises:
            OSError: The file exists but could not be removed.
        """
        with self._write_lock:
            if self._path.exists():
                self._path.unlink()
                return True
            return False

    def export(self, dest: Path, passphrase: str) -> None:
        """Write the record to `dest`, re-encrypted under a typed passphrase.

        The file is owner-only (0o600) and written through a unique temp
        file, so a neighbouring `<name>.tmp` of the user's own is never
        clobbered (final review Minor 1/2).

        Args:
            dest: Destination file. Its parent is created if needed.
            passphrase: Non-empty; the only thing that opens the export.

        Raises:
            ValueError: The passphrase is empty, or there is nothing to
                export (the reason names why the record could not be read).
            ExportRefused: `dest` IS the store's own record -- a passphrase
                envelope over it makes every later `load` report
                `cannot_decrypt`, unrecoverably (final review Minor 3).
            OSError: The destination could not be written.
        """
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
        """Adopt a passphrase-encrypted export, merging or replacing.

        A same-model import is merged into the stored record; a different
        model needs `replace=True`. An existing record that cannot be READ is
        never overwritten silently: a locked key may unlock itself so it is
        never a reason to replace, while a corrupt file may be discarded with
        an explicit `replace=True`.

        Args:
            src: The export file to read.
            passphrase: The passphrase it was exported under.
            replace: Whether the user has confirmed discarding what is
                stored. Merging never needs it.

        Returns:
            The record now stored (the merged one on the merge path).

        Raises:
            ValueError: `src` is unreadable or is not a passphrase envelope.
            WrongPassphrase: The passphrase did not open it. A `ValueError`
                too, so existing callers are unaffected.
            InvalidVoiceprint: It opened but does not hold a usable record.
                Checked BEFORE anything is stored (Qodo review 5).
            StoreUnavailable: The existing record cannot be read (locked
                key, or a corrupt file without `replace=True`).
            ModelMismatch: A different model, or a same-model vector of a
                different dimension, without `replace=True`.
        """
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
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError as exc:
            raise InvalidVoiceprint("the import file is not a valid voiceprint") from exc
        # BEFORE the store is touched at all, on BOTH the merge and the
        # replace path (Qodo review 5): the record used to go straight to
        # `save()`, so a malformed-but-decryptable file replaced a good
        # voiceprint and only failed later, in the worker, all meeting long.
        imported = _validated(payload)

        with self._write_lock:
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
    """Return the app's real voiceprint store: keyring key, keyfile fallback.

    Args:
        user_data_dir: Where `voiceprint.json` (and, in keyfile mode,
            `voiceprint.key`) live. Defaults to the app's user data dir,
            imported lazily so this module stays config-free at import time.

    Returns:
        A store whose key mode is "keyring" when a real keyring backend is
        installed, else "keyfile".
    """
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
