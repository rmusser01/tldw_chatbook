"""Versioned authenticated envelopes for Personal Context objects."""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


ALGORITHM = "aes-256-gcm-v1"
KEY_BYTES = 32
NONCE_BYTES = 12


@dataclass(frozen=True, slots=True)
class EncryptedEnvelope:
    """One content key wrapped independently from its encrypted payload."""

    algorithm: str
    nonce: bytes
    wrap_nonce: bytes
    ciphertext: bytes
    wrapped_dek: bytes
    key_version: int


class EnvelopeCipher:
    """Encrypt each object under a random DEK wrapped by a profile key."""

    def __init__(self, profile_key: bytes, *, key_version: int = 1) -> None:
        if len(profile_key) != KEY_BYTES:
            raise ValueError("profile key must be exactly 32 bytes")
        if key_version < 1:
            raise ValueError("key_version must be positive")
        self._profile_key = profile_key
        self._key_version = key_version

    def encrypt(self, plaintext: bytes, aad: bytes) -> EncryptedEnvelope:
        """Return a fresh authenticated envelope bound to ``aad``."""

        dek = secrets.token_bytes(KEY_BYTES)
        nonce = secrets.token_bytes(NONCE_BYTES)
        wrap_nonce = secrets.token_bytes(NONCE_BYTES)
        return EncryptedEnvelope(
            algorithm=ALGORITHM,
            nonce=nonce,
            wrap_nonce=wrap_nonce,
            ciphertext=AESGCM(dek).encrypt(nonce, plaintext, aad),
            wrapped_dek=AESGCM(self._profile_key).encrypt(wrap_nonce, dek, aad),
            key_version=self._key_version,
        )

    def decrypt(self, envelope: EncryptedEnvelope, aad: bytes) -> bytes:
        """Decrypt an envelope only when its algorithm, key, and AAD match."""

        if envelope.algorithm != ALGORITHM:
            raise ValueError(f"Unsupported envelope algorithm: {envelope.algorithm}")
        if envelope.key_version != self._key_version:
            raise ValueError(
                f"Unsupported envelope key version: {envelope.key_version}"
            )
        if (
            len(envelope.nonce) != NONCE_BYTES
            or len(envelope.wrap_nonce) != NONCE_BYTES
        ):
            raise ValueError("Invalid AES-GCM nonce length")
        dek = AESGCM(self._profile_key).decrypt(
            envelope.wrap_nonce, envelope.wrapped_dek, aad
        )
        return AESGCM(dek).decrypt(envelope.nonce, envelope.ciphertext, aad)
