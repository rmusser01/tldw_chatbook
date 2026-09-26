"""Typed llama.cpp tuning and private, versioned device-local profiles."""

from __future__ import annotations

import json
import os
import stat
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from threading import RLock
from typing import BinaryIO, Literal
from uuid import UUID

import portalocker
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr, ValidationError

from tldw_chatbook.Utils.atomic_file_ops import atomic_write_json

PROFILE_DOCUMENT_VERSION = 1
MAX_LLAMACPP_PROFILES = 32
MAX_PROFILE_DOCUMENT_BYTES = 256 * 1024
MAX_PROFILE_NAME_CODEPOINTS = 120
_CACHE_TYPES = frozenset(
    {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"}
)
_UNSAFE_NAME_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp"})
_LOCK = RLock()


class LlamaCppProfileError(RuntimeError):
    """Base class for bounded profile failures."""


class LlamaCppProfileValidationError(LlamaCppProfileError, ValueError):
    """A proposed profile or launch combination violates the contract."""


class LlamaCppProfileCorrupt(LlamaCppProfileError):
    """The existing document or lock cannot be trusted."""


class LlamaCppProfileFutureVersion(LlamaCppProfileError):
    """A newer writer owns this document."""


class LlamaCppProfileConflict(LlamaCppProfileError):
    """The stored revision differs from the caller's expected revision."""


class _DuplicateKey(ValueError):
    pass


def _positive(value: object, field: str) -> None:
    if value is not None and (type(value) is not int or value < 1):
        raise LlamaCppProfileValidationError(
            f"{field} must be a positive integer or null"
        )


@dataclass(frozen=True, slots=True)
class LlamaCppTuning:
    """Validated tuning; null means the installed runtime's default."""

    context_size: int | None = None
    gpu_layers: int | None = None
    threads: int | None = None
    parallel: int | None = None
    flash_attention: Literal["on", "off", "auto"] | None = None
    cache_type_k: str | None = None
    cache_type_v: str | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None

    def __post_init__(self) -> None:
        for field in (
            "context_size",
            "threads",
            "parallel",
            "batch_size",
            "ubatch_size",
        ):
            _positive(getattr(self, field), field)
        if self.gpu_layers is not None and (
            type(self.gpu_layers) is not int or self.gpu_layers < -1
        ):
            raise LlamaCppProfileValidationError(
                "gpu_layers must be -1 or non-negative"
            )
        if self.flash_attention is not None and (
            type(self.flash_attention) is not str
            or self.flash_attention not in {"on", "off", "auto"}
        ):
            raise LlamaCppProfileValidationError(
                "flash_attention must be on, off, auto or null"
            )
        for field in ("cache_type_k", "cache_type_v"):
            value = getattr(self, field)
            if value is not None and (
                type(value) is not str or value not in _CACHE_TYPES
            ):
                raise LlamaCppProfileValidationError(f"{field} is unsupported")


_MANAGED_FLAGS: dict[str, tuple[str, ...]] = {
    "context_size": ("-c", "--ctx-size"),
    "gpu_layers": ("-ngl", "--gpu-layers", "--n-gpu-layers"),
    "threads": ("-t", "--threads"),
    "parallel": ("-np", "--parallel"),
    "flash_attention": ("-fa", "--flash-attn"),
    "cache_type_k": ("-ctk", "--cache-type-k"),
    "cache_type_v": ("-ctv", "--cache-type-v"),
    "batch_size": ("-b", "--batch-size"),
    "ubatch_size": ("-ub", "--ubatch-size"),
}
_OUTPUT_FLAGS = {
    "context_size": "--ctx-size",
    "gpu_layers": "--n-gpu-layers",
    "threads": "--threads",
    "parallel": "--parallel",
    "flash_attention": "--flash-attn",
    "cache_type_k": "--cache-type-k",
    "cache_type_v": "--cache-type-v",
    "batch_size": "--batch-size",
    "ubatch_size": "--ubatch-size",
}
_RAW_TO_FIELD = {
    alias: field for field, aliases in _MANAGED_FLAGS.items() for alias in aliases
}
_RESERVED_FLAGS = frozenset(
    {
        "-m",
        "--model",
        "-mu",
        "--model-url",
        "-hf",
        "-hfr",
        "--hf-repo",
        "--alias",
        "-a",
        "--host",
        "--port",
    }
)


def build_tuning_arguments(
    tuning: LlamaCppTuning, raw_args: tuple[str, ...]
) -> tuple[str, ...]:
    """Add configured tuning to tokenized expert argv without overlapping owners."""

    if type(tuning) is not LlamaCppTuning or type(raw_args) is not tuple:
        raise LlamaCppProfileValidationError("tuning and expert arguments are invalid")
    for token in raw_args:
        if type(token) is not str or "\x00" in token:
            raise LlamaCppProfileValidationError("expert arguments are invalid")
        option = token.partition("=")[0]
        if option in _RESERVED_FLAGS:
            raise LlamaCppProfileValidationError(
                "expert option is reserved by launch controls"
            )
        field = _RAW_TO_FIELD.get(option)
        if field is not None and getattr(tuning, field) is not None:
            raise LlamaCppProfileValidationError(
                "structured and expert options conflict"
            )
    generated: list[str] = []
    for field in fields(LlamaCppTuning):
        value = getattr(tuning, field.name)
        if value is not None:
            generated.extend((_OUTPUT_FLAGS[field.name], str(value)))
    return (*generated, *raw_args)


def _profile_id(value: object) -> str:
    if type(value) is not str:
        raise LlamaCppProfileValidationError("profile_id must be a canonical UUID")
    try:
        parsed = UUID(value)
    except ValueError:
        raise LlamaCppProfileValidationError(
            "profile_id must be a canonical UUID"
        ) from None
    if str(parsed) != value:
        raise LlamaCppProfileValidationError("profile_id must be a canonical UUID")
    return value


def _name(value: object) -> str:
    if type(value) is not str or any(
        unicodedata.category(character) in _UNSAFE_NAME_CATEGORIES
        for character in value
    ):
        raise LlamaCppProfileValidationError("profile name is invalid")
    name = " ".join(unicodedata.normalize("NFKC", value).split())
    if not 1 <= len(name) <= MAX_PROFILE_NAME_CODEPOINTS:
        raise LlamaCppProfileValidationError(
            "profile name must contain 1 to 120 characters"
        )
    return name


@dataclass(frozen=True, slots=True)
class LlamaCppLaunchProfileV1:
    """Stable identifier, display name and tuning only."""

    profile_id: str
    name: str
    tuning: LlamaCppTuning

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", _profile_id(self.profile_id))
        object.__setattr__(self, "name", _name(self.name))
        if type(self.tuning) is not LlamaCppTuning:
            raise LlamaCppProfileValidationError(
                "tuning must be typed llama.cpp tuning"
            )


@dataclass(frozen=True, slots=True)
class LlamaCppProfileDocumentV1:
    """Immutable V1 profile collection and compare-and-swap revision."""

    version: Literal[1]
    revision: int
    profiles: tuple[LlamaCppLaunchProfileV1, ...]

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != PROFILE_DOCUMENT_VERSION:
            raise LlamaCppProfileValidationError("profile version must be 1")
        if type(self.revision) is not int or self.revision < 0:
            raise LlamaCppProfileValidationError("revision must be non-negative")
        if (
            type(self.profiles) is not tuple
            or len(self.profiles) > MAX_LLAMACPP_PROFILES
        ):
            raise LlamaCppProfileValidationError("profile store is capped at 32")
        if any(
            type(profile) is not LlamaCppLaunchProfileV1 for profile in self.profiles
        ):
            raise LlamaCppProfileValidationError("profile collection is invalid")
        ids = [profile.profile_id for profile in self.profiles]
        names = [profile.name.casefold() for profile in self.profiles]
        if len(ids) != len(set(ids)) or len(names) != len(set(names)):
            raise LlamaCppProfileValidationError(
                "profile identifiers and names must be unique"
            )


class _TuningPayload(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, hide_input_in_errors=True
    )

    context_size: StrictInt | None
    gpu_layers: StrictInt | None
    threads: StrictInt | None
    parallel: StrictInt | None
    flash_attention: Literal["on", "off", "auto"] | None
    cache_type_k: StrictStr | None
    cache_type_v: StrictStr | None
    batch_size: StrictInt | None
    ubatch_size: StrictInt | None


class _ProfilePayload(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, hide_input_in_errors=True
    )

    profile_id: StrictStr
    name: StrictStr
    tuning: _TuningPayload


class _DocumentPayload(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, hide_input_in_errors=True
    )

    version: StrictInt
    revision: StrictInt
    profiles: list[_ProfilePayload]


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey("duplicate profile key")
        result[key] = value
    return result


def default_llamacpp_profile_path() -> Path:
    """Return the active user's device-local profile document path."""

    from tldw_chatbook.config import get_user_data_dir

    return get_user_data_dir() / "llamacpp_launch_profiles.json"


def _document_payload(document: LlamaCppProfileDocumentV1) -> dict[str, object]:
    return {
        "version": document.version,
        "revision": document.revision,
        "profiles": [
            {
                "profile_id": profile.profile_id,
                "name": profile.name,
                "tuning": asdict(profile.tuning),
            }
            for profile in document.profiles
        ],
    }


def _decode(value: object) -> LlamaCppProfileDocumentV1:
    if type(value) is not dict:
        raise LlamaCppProfileValidationError("profile document is invalid")
    version = value.get("version")
    if type(version) is int and version > PROFILE_DOCUMENT_VERSION:
        raise LlamaCppProfileFutureVersion(
            "profile document belongs to a newer version"
        )
    try:
        payload = _DocumentPayload.model_validate(value)
    except ValidationError:
        raise LlamaCppProfileValidationError("profile document is invalid") from None
    return LlamaCppProfileDocumentV1(
        payload.version,
        payload.revision,
        tuple(
            LlamaCppLaunchProfileV1(
                item.profile_id, item.name, LlamaCppTuning(**item.tuning.model_dump())
            )
            for item in payload.profiles
        ),
    )


def _effective_uid() -> int:
    get_uid = getattr(os, "geteuid", None)
    try:
        uid = get_uid() if callable(get_uid) else None
    except Exception:  # noqa: BLE001 - normalize OS capability failures
        uid = None
    if type(uid) is not int or uid < 0:
        raise LlamaCppProfileCorrupt("llama.cpp profile storage is unavailable")
    return uid


def _verify_leaf(path: Path, descriptor: int, uid: int) -> None:
    opened = os.fstat(descriptor)
    named = path.lstat()
    if (
        not stat.S_ISREG(opened.st_mode)
        or stat.S_IMODE(opened.st_mode) & 0o077
        or opened.st_uid != uid
        or stat.S_ISLNK(named.st_mode)
        or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
    ):
        raise OSError("profile storage leaf is unavailable")


def _open_leaf(path: Path, flags: int, uid: int, *, create: bool = False) -> int:
    options = flags | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
    if create:
        options |= os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, options, 0o600)
    try:
        _verify_leaf(path, descriptor, uid)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _reject_symlink(path: Path) -> None:
    try:
        if stat.S_ISLNK(path.lstat().st_mode):
            raise LlamaCppProfileCorrupt("llama.cpp profile storage is unavailable")
    except FileNotFoundError:
        return


class LlamaCppProfileRepository:
    """CAS repository with process locking and atomic private replacement."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_llamacpp_profile_path()

    def load(self) -> LlamaCppProfileDocumentV1:
        """Load the exact V1 document, or return an empty revision-zero view."""

        with _LOCK:
            return self._load(_effective_uid())

    def _load(self, uid: int) -> LlamaCppProfileDocumentV1:
        try:
            descriptor = _open_leaf(self.path, os.O_RDONLY, uid)
        except FileNotFoundError:
            return LlamaCppProfileDocumentV1(1, 0, ())
        except OSError:
            raise LlamaCppProfileCorrupt(
                "llama.cpp profile document is unavailable"
            ) from None
        try:
            with os.fdopen(descriptor, "rb") as stream:
                if os.fstat(stream.fileno()).st_size > MAX_PROFILE_DOCUMENT_BYTES:
                    raise LlamaCppProfileValidationError(
                        "profile document is too large"
                    )
                encoded = stream.read(MAX_PROFILE_DOCUMENT_BYTES + 1)
            if len(encoded) > MAX_PROFILE_DOCUMENT_BYTES:
                raise LlamaCppProfileValidationError("profile document is too large")
            value = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_pairs)
            return _decode(value)
        except LlamaCppProfileFutureVersion:
            raise
        except (
            OSError,
            UnicodeError,
            ValueError,
            RecursionError,
        ):
            raise LlamaCppProfileCorrupt(
                "llama.cpp profile document is unavailable"
            ) from None

    @contextmanager
    def _transaction(self) -> Iterator[tuple[int, BinaryIO]]:
        uid = _effective_uid()
        _reject_symlink(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(f"{self.path.name}.lock")
        try:
            try:
                descriptor = _open_leaf(lock_path, os.O_RDWR, uid, create=True)
            except FileExistsError:
                descriptor = _open_leaf(lock_path, os.O_RDWR, uid)
        except OSError:
            raise LlamaCppProfileCorrupt(
                "llama.cpp profile storage is unavailable"
            ) from None
        with os.fdopen(descriptor, "a+b") as stream:
            portalocker.lock(stream, portalocker.LockFlags.EXCLUSIVE)
            try:
                _verify_leaf(lock_path, stream.fileno(), uid)
                yield uid, stream
            except OSError:
                raise LlamaCppProfileCorrupt(
                    "llama.cpp profile storage is unavailable"
                ) from None
            finally:
                portalocker.unlock(stream)

    @staticmethod
    def _expected_revision(value: object) -> int:
        if type(value) is not int or value < 0:
            raise LlamaCppProfileValidationError(
                "expected_revision must be non-negative"
            )
        return value

    def _commit(
        self, document: LlamaCppProfileDocumentV1, stream: BinaryIO, uid: int
    ) -> LlamaCppProfileDocumentV1:
        payload = _document_payload(document)
        encoded = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        if len(encoded) > MAX_PROFILE_DOCUMENT_BYTES:
            raise LlamaCppProfileValidationError("profile document is too large")
        _reject_symlink(self.path)
        try:
            _verify_leaf(
                self.path.with_name(f"{self.path.name}.lock"), stream.fileno(), uid
            )
        except OSError:
            raise LlamaCppProfileCorrupt(
                "llama.cpp profile storage is unavailable"
            ) from None
        atomic_write_json(
            self.path, payload, mode=0o600, indent=2, privacy_safe_log=True
        )
        return document

    def save(
        self, profile: LlamaCppLaunchProfileV1, *, expected_revision: int
    ) -> LlamaCppProfileDocumentV1:
        """Create or replace one profile when the stored revision still matches."""

        if type(profile) is not LlamaCppLaunchProfileV1:
            raise LlamaCppProfileValidationError("profile must be an exact V1 profile")
        expected = self._expected_revision(expected_revision)
        with _LOCK, self._transaction() as (uid, stream):
            current = self._load(uid)
            if current.revision != expected:
                raise LlamaCppProfileConflict("profile revision changed")
            profiles = list(current.profiles)
            for index, candidate in enumerate(profiles):
                if candidate.profile_id == profile.profile_id:
                    profiles[index] = profile
                    break
            else:
                profiles.append(profile)
            return self._commit(
                LlamaCppProfileDocumentV1(1, current.revision + 1, tuple(profiles)),
                stream,
                uid,
            )

    def delete(
        self, profile_id: str, *, expected_revision: int
    ) -> LlamaCppProfileDocumentV1:
        """Delete one existing profile when the stored revision still matches."""

        target = _profile_id(profile_id)
        expected = self._expected_revision(expected_revision)
        with _LOCK, self._transaction() as (uid, stream):
            current = self._load(uid)
            if current.revision != expected:
                raise LlamaCppProfileConflict("profile revision changed")
            profiles = tuple(p for p in current.profiles if p.profile_id != target)
            if len(profiles) == len(current.profiles):
                raise LlamaCppProfileValidationError("profile is unavailable")
            return self._commit(
                LlamaCppProfileDocumentV1(1, current.revision + 1, profiles),
                stream,
                uid,
            )
