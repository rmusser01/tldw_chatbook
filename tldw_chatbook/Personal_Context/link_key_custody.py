"""Secure device wrapping and staged integrity-key custody for first link."""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any, Protocol

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from tldw_chatbook.runtime_policy.server_credentials import is_secure_keyring_backend

from .key_protector import ProfileLockedError


_KEYRING_SERVICE = "tldw_chatbook.personal_context.link"
_WRAPPING_KEY_NAME = "device-wrapping-rsa-v1"
_WRAPPING_PREFIX = "rsa-private-pkcs8-v1:"
_STAGED_PREFIX = "staged-integrity-v1:"


class PersonalContextWrappingKeyProvider(Protocol):
    """Provide a registered RSA key without exposing its private material."""

    @property
    def public_key_pem(self) -> str: ...

    def unwrap_integrity_key(
        self, wrapped_key_blob: str, *, integrity_key_id: str
    ) -> bytes: ...


class PersonalContextLinkKeyCustodian(Protocol):
    """Hold an unactivated server integrity key under one exact link binding."""

    def stage(self, *, integrity_key: bytes, **binding: str) -> None: ...

    def load(self, **binding: str) -> bytes: ...

    def delete(self, **binding: str) -> None: ...


def _validate_integrity_key(value: bytes) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ValueError("wrapped_integrity_key_invalid")
    return value


def _unwrap(private_key: rsa.RSAPrivateKey, blob: str, key_id: str) -> bytes:
    if not isinstance(blob, str) or not blob.startswith("rsa-oaep-sha256:"):
        raise ValueError("wrapped_integrity_key_invalid")
    try:
        ciphertext = base64.urlsafe_b64decode(blob.split(":", 1)[1])
        plaintext = private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=f"personal-context:{key_id}".encode(),
            ),
        )
    except Exception as exc:
        raise ValueError("wrapped_integrity_key_invalid") from exc
    return _validate_integrity_key(plaintext)


class InMemoryPersonalContextWrappingKeyProvider:
    """Volatile wrapping-key provider for tests and memory-only operation."""

    def __init__(self) -> None:
        self._private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048
        )

    @property
    def public_key_pem(self) -> str:
        return self._private_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("ascii")

    def unwrap_integrity_key(
        self, wrapped_key_blob: str, *, integrity_key_id: str
    ) -> bytes:
        return _unwrap(self._private_key, wrapped_key_blob, integrity_key_id)


class KeyringPersonalContextWrappingKeyProvider:
    """Keep the device RSA private key only in a verified OS keyring."""

    def __init__(self, keyring_backend: Any | None = None) -> None:
        if keyring_backend is None:
            import keyring

            keyring_backend = keyring.get_keyring()
        get_keyring = getattr(keyring_backend, "get_keyring", None)
        if callable(get_keyring):
            keyring_backend = get_keyring()
        if not is_secure_keyring_backend(keyring_backend):
            raise ProfileLockedError("No secure device keyring is available.")
        self._keyring = keyring_backend

    def _private(self) -> rsa.RSAPrivateKey:
        try:
            stored = self._keyring.get_password(_KEYRING_SERVICE, _WRAPPING_KEY_NAME)
            if stored is None:
                private_key = rsa.generate_private_key(
                    public_exponent=65537, key_size=2048
                )
                encoded = private_key.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.PKCS8,
                    serialization.NoEncryption(),
                )
                self._keyring.set_password(
                    _KEYRING_SERVICE,
                    _WRAPPING_KEY_NAME,
                    _WRAPPING_PREFIX + base64.b64encode(encoded).decode("ascii"),
                )
                return private_key
            if not stored.startswith(_WRAPPING_PREFIX):
                raise ValueError("invalid wrapper")
            loaded = serialization.load_pem_private_key(
                base64.b64decode(stored.removeprefix(_WRAPPING_PREFIX), validate=True),
                password=None,
            )
            if not isinstance(loaded, rsa.RSAPrivateKey) or loaded.key_size < 2048:
                raise ValueError("invalid RSA key")
            return loaded
        except Exception as exc:
            raise ProfileLockedError("The secure device keyring is unavailable.") from exc

    @property
    def public_key_pem(self) -> str:
        return self._private().public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("ascii")

    def unwrap_integrity_key(
        self, wrapped_key_blob: str, *, integrity_key_id: str
    ) -> bytes:
        return _unwrap(self._private(), wrapped_key_blob, integrity_key_id)


def _binding_name(binding: dict[str, str]) -> str:
    required = {
        "server_profile_id",
        "dataset_id",
        "device_id",
        "profile_id",
        "integrity_key_id",
        "key_record_id",
    }
    if set(binding) != required or any(
        not isinstance(value, str) or not value for value in binding.values()
    ):
        raise ValueError("staged_integrity_key_binding_invalid")
    payload = json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
    return "staged:" + hashlib.sha256(payload).hexdigest()


class InMemoryPersonalContextLinkKeyCustodian:
    """Volatile staged custody used only by explicit tests."""

    def __init__(self) -> None:
        self._staged: dict[str, bytes] = {}

    def stage(self, *, integrity_key: bytes, **binding: str) -> None:
        self._staged[_binding_name(binding)] = _validate_integrity_key(integrity_key)

    def load(self, **binding: str) -> bytes:
        try:
            return self._staged[_binding_name(binding)]
        except KeyError:
            raise ValueError("staged_integrity_key_binding_mismatch") from None

    def delete(self, **binding: str) -> None:
        self._staged.pop(_binding_name(binding), None)


class KeyringPersonalContextLinkKeyCustodian:
    """Stage an integrity key only in a verified secure OS keyring."""

    def __init__(self, keyring_backend: Any | None = None) -> None:
        if keyring_backend is None:
            import keyring

            keyring_backend = keyring.get_keyring()
        get_keyring = getattr(keyring_backend, "get_keyring", None)
        if callable(get_keyring):
            keyring_backend = get_keyring()
        if not is_secure_keyring_backend(keyring_backend):
            raise ProfileLockedError("No secure link keyring is available.")
        self._keyring = keyring_backend

    def stage(self, *, integrity_key: bytes, **binding: str) -> None:
        name = _binding_name(binding)
        value = _STAGED_PREFIX + base64.b64encode(
            _validate_integrity_key(integrity_key)
        ).decode("ascii")
        try:
            self._keyring.set_password(_KEYRING_SERVICE, name, value)
        except Exception as exc:
            raise ProfileLockedError("The secure link keyring is unavailable.") from exc

    def load(self, **binding: str) -> bytes:
        try:
            stored = self._keyring.get_password(_KEYRING_SERVICE, _binding_name(binding))
            if not isinstance(stored, str) or not stored.startswith(_STAGED_PREFIX):
                raise ValueError("staged_integrity_key_binding_mismatch")
            return _validate_integrity_key(
                base64.b64decode(stored.removeprefix(_STAGED_PREFIX), validate=True)
            )
        except ValueError:
            raise
        except Exception as exc:
            raise ProfileLockedError("The secure link keyring is unavailable.") from exc

    def delete(self, **binding: str) -> None:
        name = _binding_name(binding)
        try:
            if self._keyring.get_password(_KEYRING_SERVICE, name) is not None:
                self._keyring.delete_password(_KEYRING_SERVICE, name)
        except Exception as exc:
            raise ProfileLockedError("The secure link keyring is unavailable.") from exc
