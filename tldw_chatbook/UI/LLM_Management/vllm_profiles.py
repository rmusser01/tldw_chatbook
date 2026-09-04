"""Device-local, versioned vLLM launch profiles with fail-closed writes."""

from __future__ import annotations

import json
import math
import os
import re
import stat
import unicodedata
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from typing import BinaryIO, Literal
from uuid import UUID, uuid4

import portalocker

from tldw_chatbook.Utils.atomic_file_ops import atomic_write_json

from .vllm_setup import (
    VllmLaunchDraft,
    VllmMode,
    VllmModelSource,
    is_safe_local_model_path_shape,
    is_valid_hugging_face_repository_id,
)

PROFILE_DOCUMENT_VERSION = 1
MAX_VLLM_PROFILES = 32
MAX_PROFILE_NAME_CODEPOINTS = 120
MAX_PROFILE_DOCUMENT_BYTES = 2 * 1024 * 1024
DEFAULT_PROFILE_NAME = "Default vLLM"
_DEFAULT_PROFILE_ID = "00000000-0000-4000-8000-000000000001"
_DOCUMENT_KEYS = frozenset({"version", "revision", "selected_profile_id", "profiles"})
_PROFILE_KEYS = frozenset(
    {
        "profile_id",
        "name",
        "python_environment",
        "model_source",
        "model_value",
        "bind_address",
        "port",
        "dtype",
        "tensor_parallel_size",
        "maximum_model_length",
        "gpu_memory_utilization",
        "trust_remote_code",
    }
)
_DTYPES = frozenset({"auto", "half", "float16", "bfloat16", "float32"})
_UNSAFE_TEXT_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp"})
_BARE_PYTHON_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")
_REPOSITORY_LOCK = RLock()


class VllmProfileError(RuntimeError):
    """Base class for fail-closed vLLM profile errors."""


class VllmProfileValidationError(VllmProfileError, ValueError):
    """A proposed profile or mutation violates the V1 contract."""


class VllmProfileCorrupt(VllmProfileError):
    """The current-version document cannot be decoded exactly."""


class VllmProfileFutureVersion(VllmProfileError):
    """The document belongs to a newer reader and must not be replaced."""


class VllmProfileConflict(VllmProfileError):
    """The compare-and-swap revision no longer matches storage."""


class _DuplicateJsonKey(ValueError):
    """A JSON object repeated a key and cannot be decoded safely."""


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build one JSON object while rejecting duplicates without echoing keys."""

    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJsonKey("profile document contains a duplicate key")
        value[key] = item
    return value


def default_vllm_profile_path() -> Path:
    """Resolve the active profile's device-local vLLM profile document."""

    from tldw_chatbook.config import get_user_data_dir

    return get_user_data_dir() / "vllm_launch_profiles.json"


def _safe_text(value: object, field: str, *, maximum: int, allow_empty: bool) -> str:
    if type(value) is not str:
        raise VllmProfileValidationError(f"{field} must be a string")
    if any(
        unicodedata.category(character) in _UNSAFE_TEXT_CATEGORIES
        for character in value
    ):
        raise VllmProfileValidationError(f"{field} contains unsafe characters")
    normalized = unicodedata.normalize("NFKC", value)
    if not allow_empty and not normalized:
        raise VllmProfileValidationError(f"{field} must not be empty")
    if len(normalized) > maximum:
        raise VllmProfileValidationError(f"{field} is too long")
    return normalized


def _canonical_name(value: object) -> str:
    raw = _safe_text(
        value,
        "name",
        maximum=MAX_PROFILE_NAME_CODEPOINTS,
        allow_empty=False,
    )
    normalized = " ".join(raw.split())
    if not normalized or len(normalized) > MAX_PROFILE_NAME_CODEPOINTS:
        raise VllmProfileValidationError("name must contain 1 to 120 characters")
    return normalized


def _name_key(value: str) -> str:
    return unicodedata.normalize("NFKC", " ".join(value.split())).casefold()


def _profile_id(value: object) -> str:
    if type(value) is not str:
        raise VllmProfileValidationError("profile_id must be a string UUID")
    try:
        parsed = UUID(value)
    except (AttributeError, ValueError) as error:
        raise VllmProfileValidationError("profile_id must be a UUID") from error
    if str(parsed) != value:
        raise VllmProfileValidationError("profile_id must be a canonical UUID")
    return value


def _optional_positive_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 1:
        raise VllmProfileValidationError(f"{field} must be a positive integer or null")
    return value


def _python_environment(value: object) -> str:
    normalized = _safe_text(
        value,
        "python_environment",
        maximum=4096,
        allow_empty=False,
    )
    if (
        Path(normalized).is_absolute()
        or _WINDOWS_ABSOLUTE_PATH.match(normalized)
        or _BARE_PYTHON_NAME.fullmatch(normalized)
    ):
        return normalized
    raise VllmProfileValidationError(
        "python_environment must be an absolute path or bare executable name"
    )


def _model_value(source: VllmModelSource, value: object) -> str:
    normalized = _safe_text(
        value,
        "model_value",
        maximum=4096,
        allow_empty=False,
    )
    valid = (
        is_valid_hugging_face_repository_id(normalized)
        if source is VllmModelSource.HUGGING_FACE
        else is_safe_local_model_path_shape(normalized)
    )
    if not valid:
        raise VllmProfileValidationError("model_value is invalid for model_source")
    return normalized


@dataclass(frozen=True, slots=True)
class VllmLaunchProfileV1:
    """The exact non-secret field set persisted for one vLLM launch profile."""

    profile_id: str
    name: str
    python_environment: str
    model_source: VllmModelSource
    model_value: str
    bind_address: str
    port: int
    dtype: str
    tensor_parallel_size: int | None
    maximum_model_length: int | None
    gpu_memory_utilization: float | None
    trust_remote_code: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", _profile_id(self.profile_id))
        object.__setattr__(self, "name", _canonical_name(self.name))
        object.__setattr__(
            self,
            "python_environment",
            _python_environment(self.python_environment),
        )
        if type(self.model_source) is not VllmModelSource:
            raise VllmProfileValidationError("model_source must be a VllmModelSource")
        object.__setattr__(
            self,
            "model_value",
            _model_value(self.model_source, self.model_value),
        )
        object.__setattr__(
            self,
            "bind_address",
            _safe_text(
                self.bind_address, "bind_address", maximum=255, allow_empty=False
            ),
        )
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise VllmProfileValidationError("port must be an integer from 1 to 65535")
        if type(self.dtype) is not str or self.dtype not in _DTYPES:
            raise VllmProfileValidationError("dtype is not supported")
        object.__setattr__(
            self,
            "tensor_parallel_size",
            _optional_positive_int(self.tensor_parallel_size, "tensor_parallel_size"),
        )
        object.__setattr__(
            self,
            "maximum_model_length",
            _optional_positive_int(self.maximum_model_length, "maximum_model_length"),
        )
        utilization = self.gpu_memory_utilization
        if utilization is not None and (
            type(utilization) is not float
            or not math.isfinite(utilization)
            or not 0 < utilization <= 1
        ):
            raise VllmProfileValidationError(
                "gpu_memory_utilization must be a finite float in (0, 1] or null"
            )
        if type(self.trust_remote_code) is not bool:
            raise VllmProfileValidationError("trust_remote_code must be a boolean")


@dataclass(frozen=True, slots=True)
class VllmProfileDocumentV1:
    """The exact V1 device-local profile document."""

    version: Literal[1]
    revision: int
    selected_profile_id: str
    profiles: tuple[VllmLaunchProfileV1, ...]

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != PROFILE_DOCUMENT_VERSION:
            raise VllmProfileValidationError("version must be exactly 1")
        if type(self.revision) is not int or self.revision < 0:
            raise VllmProfileValidationError("revision must be a non-negative integer")
        if (
            type(self.profiles) is not tuple
            or not 1 <= len(self.profiles) <= MAX_VLLM_PROFILES
        ):
            raise VllmProfileValidationError("profiles must contain 1 to 32 values")
        if any(type(profile) is not VllmLaunchProfileV1 for profile in self.profiles):
            raise VllmProfileValidationError("profiles must contain exact V1 profiles")
        ids = [profile.profile_id for profile in self.profiles]
        if len(ids) != len(set(ids)):
            raise VllmProfileValidationError("profile_id values must be unique")
        names = [_name_key(profile.name) for profile in self.profiles]
        if len(names) != len(set(names)):
            raise VllmProfileValidationError("profile names must be unique")
        if (
            type(self.selected_profile_id) is not str
            or self.selected_profile_id not in ids
        ):
            raise VllmProfileValidationError(
                "selected_profile_id must identify a profile"
            )


@dataclass(frozen=True, slots=True)
class VllmProfileMutation:
    """One successfully persisted mutation and its selected/affected profile."""

    profile: VllmLaunchProfileV1
    document: VllmProfileDocumentV1


def default_vllm_profile() -> VllmLaunchProfileV1:
    """Return the deterministic recoverable initial profile."""

    return VllmLaunchProfileV1(
        profile_id=_DEFAULT_PROFILE_ID,
        name=DEFAULT_PROFILE_NAME,
        python_environment="python",
        model_source=VllmModelSource.HUGGING_FACE,
        model_value="Qwen/Qwen2.5-0.5B-Instruct",
        bind_address="127.0.0.1",
        port=8000,
        dtype="auto",
        tensor_parallel_size=None,
        maximum_model_length=None,
        gpu_memory_utilization=None,
        trust_remote_code=False,
    )


def profile_from_draft(
    name: str,
    draft: VllmLaunchDraft,
    *,
    profile_id: str | None = None,
) -> VllmLaunchProfileV1:
    """Project only approved structured local-launch fields from a draft."""

    if type(draft) is not VllmLaunchDraft or draft.mode is not VllmMode.LOCAL:
        raise VllmProfileValidationError("only local launch drafts can be profiled")
    return VllmLaunchProfileV1(
        profile_id=profile_id or str(uuid4()),
        name=name,
        python_environment=draft.python_environment,
        model_source=draft.model_source,
        model_value=draft.model_value,
        bind_address=draft.bind_address,
        port=draft.port,
        dtype=draft.dtype or "auto",
        tensor_parallel_size=draft.tensor_parallel_size,
        maximum_model_length=draft.maximum_model_length,
        gpu_memory_utilization=(
            float(draft.gpu_memory_utilization)
            if draft.gpu_memory_utilization is not None
            else None
        ),
        trust_remote_code=draft.trust_remote_code,
    )


def draft_from_profile(
    profile: VllmLaunchProfileV1, *, raw_arguments: str = ""
) -> VllmLaunchDraft:
    """Restore one profile while keeping raw arguments explicitly launch-only."""

    if type(profile) is not VllmLaunchProfileV1:
        raise VllmProfileValidationError("profile must be an exact V1 profile")
    return VllmLaunchDraft(
        mode=VllmMode.LOCAL,
        python_environment=profile.python_environment,
        model_source=profile.model_source,
        model_value=profile.model_value,
        bind_address=profile.bind_address,
        port=profile.port,
        dtype=profile.dtype,
        tensor_parallel_size=profile.tensor_parallel_size,
        maximum_model_length=profile.maximum_model_length,
        gpu_memory_utilization=profile.gpu_memory_utilization,
        trust_remote_code=profile.trust_remote_code,
        raw_arguments=raw_arguments,
    )


def _profile_payload(profile: VllmLaunchProfileV1) -> dict[str, object]:
    return {
        "profile_id": profile.profile_id,
        "name": profile.name,
        "python_environment": profile.python_environment,
        "model_source": profile.model_source.value,
        "model_value": profile.model_value,
        "bind_address": profile.bind_address,
        "port": profile.port,
        "dtype": profile.dtype,
        "tensor_parallel_size": profile.tensor_parallel_size,
        "maximum_model_length": profile.maximum_model_length,
        "gpu_memory_utilization": profile.gpu_memory_utilization,
        "trust_remote_code": profile.trust_remote_code,
    }


def _document_payload(document: VllmProfileDocumentV1) -> dict[str, object]:
    return {
        "version": document.version,
        "revision": document.revision,
        "selected_profile_id": document.selected_profile_id,
        "profiles": [_profile_payload(profile) for profile in document.profiles],
    }


def _decode_profile(value: object) -> VllmLaunchProfileV1:
    if type(value) is not dict or set(value) != _PROFILE_KEYS:
        raise VllmProfileValidationError("profile keys do not match V1")
    source = value["model_source"]
    if type(source) is not str:
        raise VllmProfileValidationError("model_source must be a string")
    try:
        model_source = VllmModelSource(source)
    except ValueError as error:
        raise VllmProfileValidationError("model_source is invalid") from error
    return VllmLaunchProfileV1(
        profile_id=value["profile_id"],
        name=value["name"],
        python_environment=value["python_environment"],
        model_source=model_source,
        model_value=value["model_value"],
        bind_address=value["bind_address"],
        port=value["port"],
        dtype=value["dtype"],
        tensor_parallel_size=value["tensor_parallel_size"],
        maximum_model_length=value["maximum_model_length"],
        gpu_memory_utilization=value["gpu_memory_utilization"],
        trust_remote_code=value["trust_remote_code"],
    )


def _decode_document(value: object) -> VllmProfileDocumentV1:
    if type(value) is not dict:
        raise VllmProfileValidationError("profile document must be an object")
    version = value.get("version")
    if type(version) is int and version > PROFILE_DOCUMENT_VERSION:
        raise VllmProfileFutureVersion("profile document version is newer than V1")
    if set(value) != _DOCUMENT_KEYS:
        raise VllmProfileValidationError("document keys do not match V1")
    profiles = value["profiles"]
    if type(profiles) is not list:
        raise VllmProfileValidationError("profiles must be an array")
    return VllmProfileDocumentV1(
        version=value["version"],
        revision=value["revision"],
        selected_profile_id=value["selected_profile_id"],
        profiles=tuple(_decode_profile(profile) for profile in profiles),
    )


def _revalidate_profile(profile: VllmLaunchProfileV1) -> VllmLaunchProfileV1:
    if type(profile) is not VllmLaunchProfileV1:
        raise VllmProfileValidationError("profile must be an exact V1 profile")
    return VllmLaunchProfileV1(
        profile_id=profile.profile_id,
        name=profile.name,
        python_environment=profile.python_environment,
        model_source=profile.model_source,
        model_value=profile.model_value,
        bind_address=profile.bind_address,
        port=profile.port,
        dtype=profile.dtype,
        tensor_parallel_size=profile.tensor_parallel_size,
        maximum_model_length=profile.maximum_model_length,
        gpu_memory_utilization=profile.gpu_memory_utilization,
        trust_remote_code=profile.trust_remote_code,
    )


def _effective_user_id() -> int:
    """Resolve an owner identity before I/O, or raise a cause-free safe error."""

    get_effective_uid = getattr(os, "geteuid", None)
    effective_uid: object = None
    if callable(get_effective_uid):
        try:
            effective_uid = get_effective_uid()
        except Exception:  # noqa: BLE001 - normalize an untrusted OS capability hook
            effective_uid = None
    if type(effective_uid) is not int or effective_uid < 0:
        raise VllmProfileCorrupt("vLLM profile storage is unavailable")
    return effective_uid


def _verify_open_regular_file(
    path: Path,
    descriptor: int,
    effective_uid: int,
) -> None:
    """Verify one private regular leaf against the preflighted owner."""

    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode):
        raise OSError("profile storage leaf is not a regular file")
    if stat.S_IMODE(opened.st_mode) & 0o077:
        raise OSError("profile storage leaf permissions are not private")
    if opened.st_uid != effective_uid:
        raise OSError("profile storage leaf has a different owner")
    named = path.lstat()
    if stat.S_ISLNK(named.st_mode) or (
        opened.st_dev,
        opened.st_ino,
    ) != (named.st_dev, named.st_ino):
        raise OSError("profile storage leaf changed during open")


def _open_existing_regular_file(path: Path, flags: int, effective_uid: int) -> int:
    descriptor = os.open(
        path,
        flags | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        _verify_open_regular_file(path, descriptor, effective_uid)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _create_regular_file(path: Path, effective_uid: int) -> int:
    descriptor = os.open(
        path,
        os.O_CREAT
        | os.O_EXCL
        | os.O_RDWR
        | os.O_NONBLOCK
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        _verify_open_regular_file(path, descriptor, effective_uid)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _reject_symlink_leaf(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode):
        raise VllmProfileCorrupt("vLLM profile storage is unavailable")


class VllmProfileRepository:
    """CAS repository for one device-local vLLM launch-profile document."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_vllm_profile_path()

    def load(self) -> VllmProfileDocumentV1:
        """Load exact V1 state, returning a virtual Default for a missing file."""

        with _REPOSITORY_LOCK:
            return self._load_locked(_effective_user_id())

    def _load_locked(self, effective_uid: int) -> VllmProfileDocumentV1:
        return self._load_locked_with_presence(effective_uid)[0]

    def _load_locked_with_presence(
        self,
        effective_uid: int,
    ) -> tuple[VllmProfileDocumentV1, bool]:
        try:
            descriptor = _open_existing_regular_file(
                self.path,
                os.O_RDONLY,
                effective_uid,
            )
        except FileNotFoundError:
            profile = default_vllm_profile()
            return VllmProfileDocumentV1(1, 0, profile.profile_id, (profile,)), False
        except OSError as error:
            raise VllmProfileCorrupt("vLLM profile document is unavailable") from error
        try:
            with os.fdopen(descriptor, "rb") as stream:
                if os.fstat(stream.fileno()).st_size > MAX_PROFILE_DOCUMENT_BYTES:
                    raise VllmProfileValidationError("profile document is too large")
                encoded = stream.read(MAX_PROFILE_DOCUMENT_BYTES + 1)
                if len(encoded) > MAX_PROFILE_DOCUMENT_BYTES:
                    raise VllmProfileValidationError("profile document is too large")
                value = json.loads(
                    encoded.decode("utf-8"),
                    object_pairs_hook=_reject_duplicate_json_keys,
                )
            return _decode_document(value), True
        except VllmProfileFutureVersion:
            raise
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            _DuplicateJsonKey,
            VllmProfileValidationError,
        ) as error:
            raise VllmProfileCorrupt("vLLM profile document is unavailable") from error

    @contextmanager
    def _exclusive_transaction(self) -> Iterator[tuple[int, Callable[[], None]]]:
        """Serialize read/CAS/replace across threads and separate app processes.

        The app's private user-data directory is the trusted parent boundary. If
        another principal can rename entries in that directory, no userspace leaf
        check can eliminate the final rename window; the shared atomic replace
        still replaces rather than follows a last-instant destination symlink.
        """

        effective_uid = _effective_user_id()
        _reject_symlink_leaf(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(f"{self.path.name}.lock")
        try:
            descriptor = _create_regular_file(lock_path, effective_uid)
        except FileExistsError:
            try:
                descriptor = _open_existing_regular_file(
                    lock_path,
                    os.O_RDWR,
                    effective_uid,
                )
            except OSError as error:
                raise VllmProfileCorrupt(
                    "vLLM profile storage is unavailable"
                ) from error
        except OSError as error:
            raise VllmProfileCorrupt("vLLM profile storage is unavailable") from error
        stream: BinaryIO = os.fdopen(descriptor, "a+b")
        locked = False
        try:
            portalocker.lock(stream, portalocker.LockFlags.EXCLUSIVE)
            locked = True

            def verify_held_lock() -> None:
                try:
                    _verify_open_regular_file(
                        lock_path,
                        stream.fileno(),
                        effective_uid,
                    )
                except OSError as error:
                    raise VllmProfileCorrupt(
                        "vLLM profile storage is unavailable"
                    ) from error

            yield effective_uid, verify_held_lock
        finally:
            if locked:
                portalocker.unlock(stream)
            stream.close()

    @staticmethod
    def _expected_revision(value: object) -> int:
        if type(value) is not int or value < 0:
            raise VllmProfileValidationError(
                "expected_revision must be a non-negative integer"
            )
        return value

    def _commit(
        self,
        current: VllmProfileDocumentV1,
        profiles: tuple[VllmLaunchProfileV1, ...],
        selected_profile_id: str,
        profile: VllmLaunchProfileV1,
        verify_held_lock: Callable[[], None],
    ) -> VllmProfileMutation:
        validated_profiles = tuple(
            _revalidate_profile(candidate) for candidate in profiles
        )
        validated_profile = next(
            candidate
            for candidate in validated_profiles
            if candidate.profile_id == profile.profile_id
        )
        document = VllmProfileDocumentV1(
            version=1,
            revision=current.revision + 1,
            selected_profile_id=selected_profile_id,
            profiles=validated_profiles,
        )
        payload = _document_payload(document)
        # Round-trip through the strict decoder before the shared writer is called.
        _decode_document(payload)
        _reject_symlink_leaf(self.path)
        verify_held_lock()
        atomic_write_json(
            self.path,
            payload,
            mode=0o600,
            indent=2,
            privacy_safe_log=True,
        )
        return VllmProfileMutation(validated_profile, document)

    def save(
        self, profile: VllmLaunchProfileV1, *, expected_revision: int
    ) -> VllmProfileMutation:
        """Create or replace one exact profile and select it."""

        if type(profile) is not VllmLaunchProfileV1:
            raise VllmProfileValidationError("profile must be an exact V1 profile")
        expected = self._expected_revision(expected_revision)
        with _REPOSITORY_LOCK, self._exclusive_transaction() as transaction:
            effective_uid, verify_held_lock = transaction
            verify_held_lock()
            current, existed = self._load_locked_with_presence(effective_uid)
            if current.revision != expected:
                raise VllmProfileConflict("profile revision changed")
            positions = {
                candidate.profile_id: index
                for index, candidate in enumerate(current.profiles)
            }
            profiles: tuple[VllmLaunchProfileV1, ...]
            if not existed and current.revision == 0:
                profiles = (profile,)
            elif profile.profile_id in positions:
                mutable = list(current.profiles)
                mutable[positions[profile.profile_id]] = profile
                profiles = tuple(mutable)
            else:
                if len(current.profiles) >= MAX_VLLM_PROFILES:
                    raise VllmProfileValidationError("profile store is capped at 32")
                profiles = current.profiles + (profile,)
            return self._commit(
                current,
                profiles,
                profile.profile_id,
                profile,
                verify_held_lock,
            )

    def select(self, profile_id: str, *, expected_revision: int) -> VllmProfileMutation:
        """Persist selection without changing launch or process state."""

        expected = self._expected_revision(expected_revision)
        with _REPOSITORY_LOCK, self._exclusive_transaction() as transaction:
            effective_uid, verify_held_lock = transaction
            verify_held_lock()
            current = self._load_locked(effective_uid)
            if current.revision != expected:
                raise VllmProfileConflict("profile revision changed")
            profile = next(
                (
                    candidate
                    for candidate in current.profiles
                    if candidate.profile_id == profile_id
                ),
                None,
            )
            if profile is None:
                raise VllmProfileValidationError("profile is unavailable")
            return self._commit(
                current,
                current.profiles,
                profile.profile_id,
                profile,
                verify_held_lock,
            )

    def rename(
        self, profile_id: str, name: str, *, expected_revision: int
    ) -> VllmProfileMutation:
        """Rename one profile under the canonical uniqueness boundary."""

        expected = self._expected_revision(expected_revision)
        with _REPOSITORY_LOCK, self._exclusive_transaction() as transaction:
            effective_uid, verify_held_lock = transaction
            verify_held_lock()
            current = self._load_locked(effective_uid)
            if current.revision != expected:
                raise VllmProfileConflict("profile revision changed")
            mutable = list(current.profiles)
            for index, candidate in enumerate(mutable):
                if candidate.profile_id == profile_id:
                    renamed = replace(candidate, name=name)
                    mutable[index] = renamed
                    return self._commit(
                        current,
                        tuple(mutable),
                        current.selected_profile_id,
                        renamed,
                        verify_held_lock,
                    )
            raise VllmProfileValidationError("profile is unavailable")

    def duplicate(
        self, profile_id: str, *, expected_revision: int
    ) -> VllmProfileMutation:
        """Duplicate one profile using the first deterministic free copy suffix."""

        expected = self._expected_revision(expected_revision)
        with _REPOSITORY_LOCK, self._exclusive_transaction() as transaction:
            effective_uid, verify_held_lock = transaction
            verify_held_lock()
            current = self._load_locked(effective_uid)
            if current.revision != expected:
                raise VllmProfileConflict("profile revision changed")
            if len(current.profiles) >= MAX_VLLM_PROFILES:
                raise VllmProfileValidationError("profile store is capped at 32")
            source = next(
                (
                    candidate
                    for candidate in current.profiles
                    if candidate.profile_id == profile_id
                ),
                None,
            )
            if source is None:
                raise VllmProfileValidationError("profile is unavailable")
            occupied = {_name_key(candidate.name) for candidate in current.profiles}
            suffix = " copy"
            number = 1
            while True:
                rendered_suffix = suffix if number == 1 else f"{suffix} {number}"
                base = source.name[
                    : MAX_PROFILE_NAME_CODEPOINTS - len(rendered_suffix)
                ].rstrip()
                name = f"{base}{rendered_suffix}"
                if _name_key(name) not in occupied:
                    break
                number += 1
            duplicate = replace(source, profile_id=str(uuid4()), name=name)
            return self._commit(
                current,
                current.profiles + (duplicate,),
                duplicate.profile_id,
                duplicate,
                verify_held_lock,
            )

    def delete(self, profile_id: str, *, expected_revision: int) -> VllmProfileMutation:
        """Delete one profile, recreating Default vLLM when it was the last."""

        expected = self._expected_revision(expected_revision)
        with _REPOSITORY_LOCK, self._exclusive_transaction() as transaction:
            effective_uid, verify_held_lock = transaction
            verify_held_lock()
            current = self._load_locked(effective_uid)
            if current.revision != expected:
                raise VllmProfileConflict("profile revision changed")
            profiles = tuple(
                candidate
                for candidate in current.profiles
                if candidate.profile_id != profile_id
            )
            if len(profiles) == len(current.profiles):
                raise VllmProfileValidationError("profile is unavailable")
            if not profiles:
                selected = default_vllm_profile()
                profiles = (selected,)
            elif current.selected_profile_id == profile_id:
                selected = profiles[0]
            else:
                selected = next(
                    candidate
                    for candidate in profiles
                    if candidate.profile_id == current.selected_profile_id
                )
            return self._commit(
                current,
                profiles,
                selected.profile_id,
                selected,
                verify_held_lock,
            )
