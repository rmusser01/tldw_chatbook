"""Cryptographic helpers for local skill trust state."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from typing import Any

from Cryptodome.Cipher import AES
from Cryptodome.Protocol.KDF import scrypt


SKILL_TRUST_KDF_N = 16384
SKILL_TRUST_KDF_R = 8
SKILL_TRUST_KDF_P = 1
SKILL_TRUST_KEY_SIZE = 32
SKILL_TRUST_NONCE_SIZE = 12


@dataclass(frozen=True, slots=True, repr=False)
class SkillTrustKeys:
    """Purpose-separated keys derived from the local skill trust passphrase."""

    manifest_mac_key: bytes
    snapshot_key: bytes
    audit_mac_key: bytes
    wrapped_root_key: bytes

    def __repr__(self) -> str:
        return (
            "SkillTrustKeys("
            "manifest_mac_key=<redacted>, "
            "snapshot_key=<redacted>, "
            "audit_mac_key=<redacted>, "
            "wrapped_root_key=<redacted>)"
        )


def canonical_json(payload: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes for authenticated payloads.

    Args:
        payload: JSON-serializable payload to canonicalize.

    Returns:
        UTF-8 encoded JSON bytes with stable key ordering and separators.
    """

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """Return the SHA-256 digest of bytes as lowercase hexadecimal.

    Args:
        data: Bytes to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """

    return hashlib.sha256(data).hexdigest()


def derive_skill_trust_keys(passphrase: str, *, salt: bytes) -> SkillTrustKeys:
    """Derive purpose-separated local skill trust keys from a passphrase.

    Args:
        passphrase: User-provided trust passphrase.
        salt: 32-byte salt bound to the local trust manifest.

    Returns:
        Purpose-separated keys for manifests, snapshots, audit entries, and key wrapping.

    Raises:
        ValueError: If ``salt`` is not exactly 32 bytes.
    """

    if not isinstance(salt, bytes) or len(salt) != 32:
        raise ValueError("skill trust salt must be 32 bytes")
    root = scrypt(
        passphrase.encode("utf-8"),
        salt,
        key_len=SKILL_TRUST_KEY_SIZE,
        N=SKILL_TRUST_KDF_N,
        r=SKILL_TRUST_KDF_R,
        p=SKILL_TRUST_KDF_P,
    )
    return SkillTrustKeys(
        manifest_mac_key=_derive_subkey(root, b"tldw-chatbook-skill-trust-manifest-v1"),
        snapshot_key=_derive_subkey(root, b"tldw-chatbook-skill-trust-snapshot-v1"),
        audit_mac_key=_derive_subkey(root, b"tldw-chatbook-skill-trust-audit-v1"),
        wrapped_root_key=_derive_subkey(root, b"tldw-chatbook-skill-trust-wrapped-root-v1"),
    )


def _derive_subkey(root: bytes, purpose: bytes) -> bytes:
    return hmac.new(root, purpose, hashlib.sha256).digest()


def _require_aes_256_key(key: bytes) -> None:
    if not isinstance(key, bytes) or len(key) != SKILL_TRUST_KEY_SIZE:
        raise ValueError("skill trust AES-256-GCM key must be 32 bytes")


def manifest_mac(manifest_payload: dict[str, Any], key: bytes) -> str:
    """Return an HMAC-SHA256 tag for a canonical manifest payload.

    Args:
        manifest_payload: Manifest payload to authenticate.
        key: HMAC key derived for manifest authentication.

    Returns:
        Lowercase hexadecimal HMAC-SHA256 tag.
    """

    return hmac.new(key, canonical_json(manifest_payload), hashlib.sha256).hexdigest()


def encrypt_json_blob(
    payload: dict[str, Any],
    key: bytes,
    *,
    associated_data: bytes,
) -> dict[str, str]:
    """Encrypt and authenticate a JSON payload with AES-256-GCM.

    Args:
        payload: JSON object to encrypt.
        key: 32-byte AES-256-GCM key.
        associated_data: Additional authenticated data bound to the ciphertext.

    Returns:
        JSON-safe encrypted blob containing algorithm, nonce, ciphertext, and tag.

    Raises:
        ValueError: If ``key`` is not exactly 32 bytes.
    """

    _require_aes_256_key(key)
    nonce = os.urandom(SKILL_TRUST_NONCE_SIZE)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    cipher.update(associated_data)
    ciphertext, tag = cipher.encrypt_and_digest(canonical_json(payload))
    return {
        "alg": "AES-256-GCM",
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        "tag": base64.b64encode(tag).decode("ascii"),
    }


def decrypt_json_blob(
    blob: dict[str, str],
    key: bytes,
    *,
    associated_data: bytes,
) -> dict[str, Any]:
    """Decrypt and authenticate a JSON object encrypted by :func:`encrypt_json_blob`.

    Args:
        blob: JSON-safe encrypted blob produced by ``encrypt_json_blob``.
        key: 32-byte AES-256-GCM key.
        associated_data: Additional authenticated data expected for the ciphertext.

    Returns:
        Decrypted JSON object.

    Raises:
        ValueError: If the key, algorithm, authentication tag, or plaintext shape is invalid.
    """

    _require_aes_256_key(key)
    try:
        if blob.get("alg") != "AES-256-GCM":
            raise ValueError("unsupported snapshot algorithm")
        nonce = base64.b64decode(blob["nonce"], validate=True)
        ciphertext = base64.b64decode(blob["ciphertext"], validate=True)
        tag = base64.b64decode(blob["tag"], validate=True)
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        cipher.update(associated_data)
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)
        payload = json.loads(plaintext.decode("utf-8"))
    except Exception as exc:
        raise ValueError("snapshot authentication failed") from exc
    if not isinstance(payload, dict):
        raise ValueError("snapshot payload must be an object")
    return payload
