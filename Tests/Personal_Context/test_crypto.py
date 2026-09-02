from __future__ import annotations

from dataclasses import replace

import pytest
from cryptography.exceptions import InvalidTag

from tldw_chatbook.Personal_Context.crypto import EnvelopeCipher


def test_envelope_round_trip_uses_unique_object_and_wrap_nonces() -> None:
    cipher = EnvelopeCipher(b"e" * 32)

    first = cipher.encrypt(b"private profile text", b"record:a:v1")
    second = cipher.encrypt(b"private profile text", b"record:a:v1")

    assert first.algorithm == "aes-256-gcm-v1"
    assert first.nonce != second.nonce
    assert first.wrap_nonce != second.wrap_nonce
    assert first.nonce != first.wrap_nonce
    assert cipher.decrypt(first, b"record:a:v1") == b"private profile text"


def test_envelope_rejects_changed_aad_and_wrapped_key() -> None:
    cipher = EnvelopeCipher(b"e" * 32)
    envelope = cipher.encrypt(b"private profile text", b"record:a:v1")

    with pytest.raises(InvalidTag):
        cipher.decrypt(envelope, b"record:b:v1")
    with pytest.raises(InvalidTag):
        cipher.decrypt(
            replace(
                envelope,
                wrapped_dek=envelope.wrapped_dek[:-1]
                + bytes([envelope.wrapped_dek[-1] ^ 1]),
            ),
            b"record:a:v1",
        )


def test_envelope_rejects_invalid_key_sizes_and_unknown_algorithms() -> None:
    with pytest.raises(ValueError, match="exactly 32 bytes"):
        EnvelopeCipher(b"short")

    cipher = EnvelopeCipher(b"e" * 32)
    envelope = cipher.encrypt(b"text", b"aad")
    with pytest.raises(ValueError, match="Unsupported envelope algorithm"):
        cipher.decrypt(replace(envelope, algorithm="aes-256-gcm-v2"), b"aad")
    with pytest.raises(ValueError, match="Unsupported envelope key version"):
        cipher.decrypt(replace(envelope, key_version=2), b"aad")
