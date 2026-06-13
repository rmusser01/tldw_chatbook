from __future__ import annotations

import base64

import pytest

from tldw_chatbook.Sync_Interop import crypto
from tldw_chatbook.Sync_Interop.crypto import (
    decrypt_sync_payload,
    encrypt_sync_payload,
    generate_dataset_key,
    unwrap_recovery_bundle,
    wrap_dataset_key_for_recovery,
)


def test_encrypt_private_payload_does_not_leak_plaintext() -> None:
    dataset_key = generate_dataset_key()

    encrypted = encrypt_sync_payload(
        {"body": "known private text", "metadata": {"tags": ["sync"]}},
        key=dataset_key,
    )
    serialized = encrypted.model_dump_json()

    assert "known private text" not in serialized
    assert decrypt_sync_payload(encrypted, key=dataset_key) == {
        "body": "known private text",
        "metadata": {"tags": ["sync"]},
    }
    assert encrypted.version == "sync_payload_v1"
    assert encrypted.algorithm == "AES-256-GCM"


def test_encrypting_same_payload_twice_uses_distinct_nonces_and_ciphertext() -> None:
    dataset_key = generate_dataset_key()

    first = encrypt_sync_payload({"body": "same text"}, key=dataset_key)
    second = encrypt_sync_payload({"body": "same text"}, key=dataset_key)

    assert first.nonce != second.nonce
    assert first.ciphertext != second.ciphertext
    assert decrypt_sync_payload(first, key=dataset_key) == {"body": "same text"}
    assert decrypt_sync_payload(second, key=dataset_key) == {"body": "same text"}


def test_wrong_dataset_key_fails_authentication() -> None:
    encrypted = encrypt_sync_payload({"body": "private"}, key=generate_dataset_key())

    with pytest.raises(ValueError, match="Failed to decrypt sync payload"):
        decrypt_sync_payload(encrypted, key=generate_dataset_key())


def test_decrypt_payload_preserves_unexpected_parser_errors(monkeypatch) -> None:
    dataset_key = generate_dataset_key()
    encrypted = encrypt_sync_payload({"body": "private"}, key=dataset_key)

    def raise_unexpected_error(_value):
        raise RuntimeError("unexpected parser bug")

    monkeypatch.setattr(crypto.json, "loads", raise_unexpected_error)

    with pytest.raises(RuntimeError, match="unexpected parser bug"):
        decrypt_sync_payload(encrypted, key=dataset_key)


def test_recovery_bundle_wraps_dataset_key_without_plaintext_material() -> None:
    dataset_key = generate_dataset_key()

    bundle = wrap_dataset_key_for_recovery(
        dataset_key,
        recovery_secret="correct horse battery staple",
        recovery_hint="personal laptop",
    )
    serialized = bundle.model_dump_json()

    assert base64.b64encode(dataset_key).decode("ascii") not in serialized
    assert dataset_key.hex() not in serialized
    assert bundle.version == "sync_recovery_bundle_v1"
    assert bundle.recovery_hint == "personal laptop"
    assert unwrap_recovery_bundle(bundle, recovery_secret="correct horse battery staple") == dataset_key


def test_wrong_recovery_secret_cannot_unwrap_bundle() -> None:
    bundle = wrap_dataset_key_for_recovery(
        generate_dataset_key(),
        recovery_secret="correct horse battery staple",
    )

    with pytest.raises(ValueError, match="Failed to unwrap recovery bundle"):
        unwrap_recovery_bundle(bundle, recovery_secret="wrong secret")


def test_unwrap_recovery_bundle_preserves_unexpected_key_derivation_errors(monkeypatch) -> None:
    bundle = wrap_dataset_key_for_recovery(
        generate_dataset_key(),
        recovery_secret="correct horse battery staple",
    )

    def raise_unexpected_error(*_args, **_kwargs):
        raise RuntimeError("unexpected kdf bug")

    monkeypatch.setattr(crypto, "_derive_recovery_key", raise_unexpected_error)

    with pytest.raises(RuntimeError, match="unexpected kdf bug"):
        unwrap_recovery_bundle(bundle, recovery_secret="correct horse battery staple")


def test_unwrap_recovery_bundle_rejects_unbounded_scrypt_metadata(monkeypatch) -> None:
    bundle = wrap_dataset_key_for_recovery(
        generate_dataset_key(),
        recovery_secret="correct horse battery staple",
    ).model_dump(mode="json")
    bundle["kdf_metadata"]["n"] = crypto.SCRYPT_N * 1024

    def fail_if_scrypt_runs(*_args, **_kwargs):
        raise AssertionError("scrypt should not run for invalid metadata")

    monkeypatch.setattr(crypto, "scrypt", fail_if_scrypt_runs)

    with pytest.raises(ValueError, match="unsupported scrypt n"):
        unwrap_recovery_bundle(bundle, recovery_secret="correct horse battery staple")
