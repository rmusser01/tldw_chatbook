"""Profile-key custody backed by a secure keyring or a passphrase wrapper."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from tldw_chatbook.runtime_policy.server_credentials import is_secure_keyring_backend
from tldw_chatbook.Utils.private_paths import (
    PrivatePathError,
    PrivatePathStatus,
    open_private_binary,
)


_KEYRING_SERVICE = "tldw_chatbook.personal_context"
_KEYRING_PREFIX = "profile-key-v1:"
_PASSPHRASE_DOMAIN = b"tldw-chatbook:personal-context:profile-key:v1\x00"
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1


class ProfileLockedError(RuntimeError):
    """Report that protected profile keys cannot be loaded safely."""

    reason_code = "profile_locked"


@dataclass(frozen=True, slots=True)
class ProfileKeyMaterial:
    """Separate per-profile confidentiality and canonical-integrity keys."""

    encryption_key: bytes
    integrity_key: bytes
    key_version: int = 1

    def __post_init__(self) -> None:
        if len(self.encryption_key) != 32 or len(self.integrity_key) != 32:
            raise ValueError("profile keys must be exactly 32 bytes")
        if self.encryption_key == self.integrity_key:
            raise ValueError("encryption and integrity keys must be separate")
        if self.key_version < 1:
            raise ValueError("key_version must be positive")


class ProfileKeyProtector(Protocol):
    """Key-custody interface used by the local repository."""

    def load_or_create(self, profile_ref: str) -> ProfileKeyMaterial: ...

    def load(self, profile_ref: str) -> ProfileKeyMaterial: ...

    def delete(self, profile_ref: str) -> None: ...


def _new_material() -> ProfileKeyMaterial:
    return ProfileKeyMaterial(secrets.token_bytes(32), secrets.token_bytes(32))


def _serialize_material(material: ProfileKeyMaterial) -> bytes:
    return (
        bytes([material.key_version]) + material.encryption_key + material.integrity_key
    )


def _deserialize_material(payload: bytes) -> ProfileKeyMaterial:
    if len(payload) != 65 or payload[0] < 1:
        raise ProfileLockedError("Profile key material is invalid.")
    return ProfileKeyMaterial(payload[1:33], payload[33:65], payload[0])


def _normalized_ref(profile_ref: str) -> str:
    normalized = profile_ref.strip()
    if not normalized or len(normalized) > 256 or "\x00" in normalized:
        raise ValueError("profile_ref must be a bounded non-empty string")
    return normalized


class InMemoryProfileKeyProtector:
    """Explicit volatile protector for tests and memory-only operation."""

    def __init__(self) -> None:
        self._materials: dict[str, ProfileKeyMaterial] = {}

    @property
    def is_empty(self) -> bool:
        return not self._materials

    def load_or_create(self, profile_ref: str) -> ProfileKeyMaterial:
        profile_ref = _normalized_ref(profile_ref)
        return self._materials.setdefault(profile_ref, _new_material())

    def load(self, profile_ref: str) -> ProfileKeyMaterial:
        profile_ref = _normalized_ref(profile_ref)
        try:
            return self._materials[profile_ref]
        except KeyError:
            raise ProfileLockedError("Profile key material is unavailable.") from None

    def delete(self, profile_ref: str) -> None:
        self._materials.pop(_normalized_ref(profile_ref), None)

    def clear_without_authorization(self) -> None:
        """Simulate external key loss in tests without repository deletion."""

        self._materials.clear()


class KeyringProfileKeyProtector:
    """Store the key bundle only in a verified secure OS-backed keyring."""

    def __init__(self, keyring_backend: Any | None = None) -> None:
        if keyring_backend is None:
            import keyring

            keyring_backend = keyring.get_keyring()
        get_keyring = getattr(keyring_backend, "get_keyring", None)
        if callable(get_keyring):
            keyring_backend = get_keyring()
        if not is_secure_keyring_backend(keyring_backend):
            raise ProfileLockedError(
                "No secure OS-backed profile keyring is available."
            )
        self._keyring = keyring_backend

    def load_or_create(self, profile_ref: str) -> ProfileKeyMaterial:
        profile_ref = _normalized_ref(profile_ref)
        stored = self._read(profile_ref)
        if stored is not None:
            return stored
        material = _new_material()
        payload = _KEYRING_PREFIX + base64.b64encode(
            _serialize_material(material)
        ).decode("ascii")
        try:
            self._keyring.set_password(_KEYRING_SERVICE, profile_ref, payload)
        except Exception as exc:
            raise ProfileLockedError(
                "The secure profile keyring is unavailable."
            ) from exc
        return material

    def load(self, profile_ref: str) -> ProfileKeyMaterial:
        material = self._read(_normalized_ref(profile_ref))
        if material is None:
            raise ProfileLockedError("Profile key material is unavailable.")
        return material

    def delete(self, profile_ref: str) -> None:
        profile_ref = _normalized_ref(profile_ref)
        try:
            if self._keyring.get_password(_KEYRING_SERVICE, profile_ref) is not None:
                self._keyring.delete_password(_KEYRING_SERVICE, profile_ref)
        except Exception as exc:
            raise ProfileLockedError(
                "The secure profile keyring is unavailable."
            ) from exc

    def _read(self, profile_ref: str) -> ProfileKeyMaterial | None:
        try:
            payload = self._keyring.get_password(_KEYRING_SERVICE, profile_ref)
        except Exception as exc:
            raise ProfileLockedError(
                "The secure profile keyring is unavailable."
            ) from exc
        if payload is None:
            return None
        if not payload.startswith(_KEYRING_PREFIX):
            raise ProfileLockedError("Profile key material is invalid.")
        try:
            decoded = base64.b64decode(
                payload.removeprefix(_KEYRING_PREFIX), validate=True
            )
        except (ValueError, TypeError) as exc:
            raise ProfileLockedError("Profile key material is invalid.") from exc
        return _deserialize_material(decoded)


class PassphraseProfileKeyProtector:
    """Persist a versioned scrypt/AES-GCM-wrapped profile key bundle."""

    def __init__(
        self,
        bundle_path: str | os.PathLike[str],
        passphrase_provider: Callable[[], str | None],
    ) -> None:
        self._path = Path(bundle_path)
        self._passphrase_provider = passphrase_provider

    def load_or_create(self, profile_ref: str) -> ProfileKeyMaterial:
        profile_ref = _normalized_ref(profile_ref)
        if os.path.lexists(self._path):
            return self.load(profile_ref)
        passphrase = self._get_passphrase()
        material = _new_material()
        salt = secrets.token_bytes(32)
        nonce = secrets.token_bytes(12)
        aad = _PASSPHRASE_DOMAIN + profile_ref.encode("utf-8")
        key = self._derive_key(passphrase, salt, profile_ref)
        ciphertext = AESGCM(key).encrypt(nonce, _serialize_material(material), aad)
        payload = {
            "version": 1,
            "scrypt": {"n": _SCRYPT_N, "r": _SCRYPT_R, "p": _SCRYPT_P},
            "salt": base64.b64encode(salt).decode("ascii"),
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        }
        self._write_private(json.dumps(payload, separators=(",", ":")).encode())
        return material

    def load(self, profile_ref: str) -> ProfileKeyMaterial:
        profile_ref = _normalized_ref(profile_ref)
        try:
            encoded = self._read_private()
            passphrase = self._get_passphrase()
            payload = json.loads(encoded)
            if payload.get("version") != 1 or payload.get("scrypt") != {
                "n": _SCRYPT_N,
                "r": _SCRYPT_R,
                "p": _SCRYPT_P,
            }:
                raise ValueError("unsupported wrapper")
            salt = base64.b64decode(payload["salt"], validate=True)
            nonce = base64.b64decode(payload["nonce"], validate=True)
            ciphertext = base64.b64decode(payload["ciphertext"], validate=True)
            aad = _PASSPHRASE_DOMAIN + profile_ref.encode("utf-8")
            plaintext = AESGCM(self._derive_key(passphrase, salt, profile_ref)).decrypt(
                nonce, ciphertext, aad
            )
            return _deserialize_material(plaintext)
        except FileNotFoundError:
            raise ProfileLockedError("Profile key material is unavailable.") from None
        except ProfileLockedError:
            raise
        except PrivatePathError as exc:
            raise ProfileLockedError("Profile key material is not private.") from exc
        except (InvalidTag, KeyError, TypeError, ValueError, OSError) as exc:
            raise ProfileLockedError(
                "Profile key material could not be unlocked."
            ) from exc

    def _read_private(self) -> bytes:
        """Read the bundle from a pinned private regular file."""

        try:
            with open_private_binary(self._path) as opened:
                if opened.result.status is PrivatePathStatus.HARDENED_PRIVATE:
                    raise ProfileLockedError("Profile key material is not private.")
                return opened.stream.read()
        except PrivatePathError as exc:
            if exc.result.reason != "required_posix_guards_unavailable":
                raise

        nofollow = getattr(os, "O_NOFOLLOW", 0)
        if not nofollow:
            raise ProfileLockedError("Profile key material is not private.")
        descriptor = os.open(self._path, os.O_RDONLY | nofollow)
        try:
            lexical = self._path.lstat()
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(lexical.st_mode)
                or not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or stat.S_IMODE(opened.st_mode) & 0o077
                or (lexical.st_dev, lexical.st_ino) != (opened.st_dev, opened.st_ino)
            ):
                raise ProfileLockedError("Profile key material is not private.")
            with os.fdopen(descriptor, "rb", closefd=True) as stream:
                descriptor = -1
                return stream.read()
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def delete(self, profile_ref: str) -> None:
        _normalized_ref(profile_ref)
        try:
            self._path.unlink(missing_ok=True)
        except OSError as exc:
            raise ProfileLockedError(
                "Profile key material could not be deleted."
            ) from exc

    def _get_passphrase(self) -> str:
        passphrase = self._passphrase_provider()
        if passphrase is None:
            raise ProfileLockedError("Profile unlock was cancelled.")
        if not isinstance(passphrase, str) or not passphrase:
            raise ProfileLockedError("A non-empty profile passphrase is required.")
        return passphrase

    @staticmethod
    def _derive_key(passphrase: str, salt: bytes, profile_ref: str) -> bytes:
        domain_salt = hashlib.sha256(
            _PASSPHRASE_DOMAIN + profile_ref.encode("utf-8") + salt
        ).digest()
        try:
            return hashlib.scrypt(
                passphrase.encode("utf-8"),
                salt=domain_salt,
                n=_SCRYPT_N,
                r=_SCRYPT_R,
                p=_SCRYPT_P,
                dklen=32,
            )
        except ValueError as exc:
            raise ProfileLockedError("Profile key derivation is unavailable.") from exc

    def _write_private(self, payload: bytes) -> None:
        if not self._path.parent.is_dir():
            raise ProfileLockedError("Profile key destination is unavailable.")
        temporary = self._path.with_name(
            f".{self._path.name}.{secrets.token_hex(8)}.tmp"
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = -1
        try:
            descriptor = os.open(temporary, flags, 0o600)
            os.write(descriptor, payload)
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            os.replace(temporary, self._path)
        except OSError as exc:
            raise ProfileLockedError(
                "Profile key material could not be persisted."
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)


class InterviewPassphraseKeyProtector:
    """Store each interview-session key in an independent passphrase bundle."""

    def __init__(
        self,
        bundle_directory: str | os.PathLike[str],
        passphrase_provider: Callable[[], str | None],
    ) -> None:
        self._directory = Path(bundle_directory)
        self._passphrase_provider = passphrase_provider
        try:
            metadata = self._directory.lstat()
        except OSError as exc:
            raise ProfileLockedError(
                "Interview key destination is unavailable."
            ) from exc
        if not stat.S_ISDIR(metadata.st_mode):
            raise ProfileLockedError("Interview key destination is not private.")
        get_effective_uid = getattr(os, "geteuid", None)
        if callable(get_effective_uid) and (
            metadata.st_uid != get_effective_uid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise ProfileLockedError("Interview key destination is not private.")

    def _protector(self, profile_ref: str) -> PassphraseProfileKeyProtector:
        profile_ref = _normalized_ref(profile_ref)
        digest = hashlib.sha256(
            b"tldw-chatbook:interview-passphrase-bundle:v1\x00"
            + profile_ref.encode("utf-8")
        ).hexdigest()
        return PassphraseProfileKeyProtector(
            self._directory / f"draft-{digest}.keys",
            self._passphrase_provider,
        )

    def load_or_create(self, profile_ref: str) -> ProfileKeyMaterial:
        return self._protector(profile_ref).load_or_create(profile_ref)

    def load(self, profile_ref: str) -> ProfileKeyMaterial:
        return self._protector(profile_ref).load(profile_ref)

    def delete(self, profile_ref: str) -> None:
        self._protector(profile_ref).delete(profile_ref)
