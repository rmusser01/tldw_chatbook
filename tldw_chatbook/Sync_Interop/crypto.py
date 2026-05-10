"""Client-side crypto helpers for local-first Sync v2 payloads."""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Mapping

from Cryptodome.Cipher import AES
from Cryptodome.Protocol.KDF import scrypt
from pydantic import BaseModel


DATASET_KEY_BYTES = 32
AES_GCM_NONCE_BYTES = 12
AES_GCM_TAG_BYTES = 16
SCRYPT_N = 16_384
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_SALT_BYTES = 32


class SyncEncryptedPayload(BaseModel):
    """Serialized encrypted sync payload with versioned metadata."""

    version: str = "sync_payload_v1"
    algorithm: str = "AES-256-GCM"
    nonce: str
    ciphertext: str
    tag: str


class SyncRecoveryBundle(BaseModel):
    """Client-generated wrapped dataset key material for server storage."""

    version: str = "sync_recovery_bundle_v1"
    algorithm: str = "AES-256-GCM"
    key_purpose: str = "dataset_recovery"
    wrapped_key_blob: str
    kdf_metadata: dict[str, Any]
    recovery_hint: str | None = None


def generate_dataset_key() -> bytes:
    """Return a new 256-bit dataset key."""

    return os.urandom(DATASET_KEY_BYTES)


def encrypt_sync_payload(payload: Mapping[str, Any], *, key: bytes) -> SyncEncryptedPayload:
    """Encrypt a JSON sync payload using AES-256-GCM."""

    _validate_dataset_key(key)
    nonce = os.urandom(AES_GCM_NONCE_BYTES)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    plaintext = _canonical_json_bytes(payload)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    return SyncEncryptedPayload(
        nonce=_b64encode(nonce),
        ciphertext=_b64encode(ciphertext),
        tag=_b64encode(tag),
    )


def decrypt_sync_payload(encrypted: SyncEncryptedPayload | Mapping[str, Any], *, key: bytes) -> dict[str, Any]:
    """Decrypt and authenticate a Sync v2 payload."""

    _validate_dataset_key(key)
    record = _coerce_encrypted_payload(encrypted)
    try:
        cipher = AES.new(key, AES.MODE_GCM, nonce=_b64decode(record.nonce))
        plaintext = cipher.decrypt_and_verify(
            _b64decode(record.ciphertext),
            _b64decode(record.tag),
        )
        value = json.loads(plaintext.decode("utf-8"))
    except Exception as exc:
        raise ValueError("Failed to decrypt sync payload") from exc
    if not isinstance(value, dict):
        raise ValueError("Failed to decrypt sync payload")
    return value


def wrap_dataset_key_for_recovery(
    dataset_key: bytes,
    *,
    recovery_secret: str | bytes,
    recovery_hint: str | None = None,
) -> SyncRecoveryBundle:
    """Wrap a dataset key with a user-held recovery secret."""

    _validate_dataset_key(dataset_key)
    salt = os.urandom(SCRYPT_SALT_BYTES)
    nonce = os.urandom(AES_GCM_NONCE_BYTES)
    wrapping_key = _derive_recovery_key(recovery_secret, salt=salt)
    cipher = AES.new(wrapping_key, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(dataset_key)
    return SyncRecoveryBundle(
        wrapped_key_blob=_b64encode(nonce + ciphertext + tag),
        kdf_metadata={
            "algorithm": "scrypt",
            "version": 1,
            "salt": _b64encode(salt),
            "n": SCRYPT_N,
            "r": SCRYPT_R,
            "p": SCRYPT_P,
            "key_len": DATASET_KEY_BYTES,
        },
        recovery_hint=recovery_hint,
    )


def unwrap_recovery_bundle(
    bundle: SyncRecoveryBundle | Mapping[str, Any],
    *,
    recovery_secret: str | bytes,
) -> bytes:
    """Recover a dataset key from a wrapped recovery bundle."""

    record = _coerce_recovery_bundle(bundle)
    try:
        metadata = record.kdf_metadata
        salt = _b64decode(str(metadata["salt"]))
        wrapped = _b64decode(record.wrapped_key_blob)
        min_len = AES_GCM_NONCE_BYTES + DATASET_KEY_BYTES + AES_GCM_TAG_BYTES
        if len(wrapped) < min_len:
            raise ValueError("invalid wrapped key length")
        nonce = wrapped[:AES_GCM_NONCE_BYTES]
        tag = wrapped[-AES_GCM_TAG_BYTES:]
        ciphertext = wrapped[AES_GCM_NONCE_BYTES:-AES_GCM_TAG_BYTES]
        wrapping_key = _derive_recovery_key(
            recovery_secret,
            salt=salt,
            n=int(metadata.get("n", SCRYPT_N)),
            r=int(metadata.get("r", SCRYPT_R)),
            p=int(metadata.get("p", SCRYPT_P)),
        )
        cipher = AES.new(wrapping_key, AES.MODE_GCM, nonce=nonce)
        dataset_key = cipher.decrypt_and_verify(ciphertext, tag)
    except Exception as exc:
        raise ValueError("Failed to unwrap recovery bundle") from exc
    _validate_dataset_key(dataset_key)
    return dataset_key


def _derive_recovery_key(
    recovery_secret: str | bytes,
    *,
    salt: bytes,
    n: int = SCRYPT_N,
    r: int = SCRYPT_R,
    p: int = SCRYPT_P,
) -> bytes:
    secret = (
        recovery_secret.encode("utf-8")
        if isinstance(recovery_secret, str)
        else recovery_secret
    )
    return scrypt(secret, salt, key_len=DATASET_KEY_BYTES, N=n, r=r, p=p)


def _validate_dataset_key(key: bytes) -> None:
    if not isinstance(key, bytes) or len(key) != DATASET_KEY_BYTES:
        raise ValueError("dataset key must be 32 bytes")


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _b64encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"), validate=True)


def _coerce_encrypted_payload(value: SyncEncryptedPayload | Mapping[str, Any]) -> SyncEncryptedPayload:
    if isinstance(value, SyncEncryptedPayload):
        return value
    return SyncEncryptedPayload.model_validate(value)


def _coerce_recovery_bundle(value: SyncRecoveryBundle | Mapping[str, Any]) -> SyncRecoveryBundle:
    if isinstance(value, SyncRecoveryBundle):
        return value
    return SyncRecoveryBundle.model_validate(value)
