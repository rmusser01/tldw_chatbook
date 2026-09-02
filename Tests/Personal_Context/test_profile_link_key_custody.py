from __future__ import annotations

import base64

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from tldw_chatbook.Personal_Context.key_protector import (
    InMemoryProfileKeyProtector,
    ProfileKeyMaterial,
)
from tldw_chatbook.Personal_Context.link_key_custody import (
    InMemoryPersonalContextLinkKeyCustodian,
    InMemoryPersonalContextWrappingKeyProvider,
)


def test_wrapping_provider_advertises_rsa_public_key_and_unwraps_exact_key() -> None:
    provider = InMemoryPersonalContextWrappingKeyProvider()
    integrity_key = b"i" * 32
    integrity_key_id = "integrity-key-1"
    public_key = serialization.load_pem_public_key(provider.public_key_pem.encode())
    ciphertext = public_key.encrypt(
        integrity_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=f"personal-context:{integrity_key_id}".encode(),
        ),
    )

    assert provider.unwrap_integrity_key(
        "rsa-oaep-sha256:" + base64.urlsafe_b64encode(ciphertext).decode(),
        integrity_key_id=integrity_key_id,
    ) == integrity_key

    with pytest.raises(ValueError, match="wrapped_integrity_key_invalid"):
        provider.unwrap_integrity_key(
            "rsa-oaep-sha256:" + base64.urlsafe_b64encode(ciphertext).decode(),
            integrity_key_id="different-key",
        )


def test_staged_key_is_exactly_bound_and_profile_bundle_can_be_replaced() -> None:
    custodian = InMemoryPersonalContextLinkKeyCustodian()
    binding = {
        "server_profile_id": "server-1",
        "dataset_id": "dataset-1",
        "device_id": "device-1",
        "profile_id": "profile-1",
        "integrity_key_id": "integrity-key-1",
        "key_record_id": "record-1",
    }
    custodian.stage(**binding, integrity_key=b"n" * 32)
    storage_key = custodian.load_or_create_storage_key(**binding)

    assert custodian.load(**binding) == b"n" * 32
    assert storage_key == custodian.load_or_create_storage_key(**binding)
    assert len(storage_key) == 32
    assert storage_key != b"n" * 32
    with pytest.raises(ValueError, match="staged_integrity_key_binding_mismatch"):
        custodian.load(**{**binding, "device_id": "device-2"})

    protector = InMemoryProfileKeyProtector()
    before = protector.load_or_create("profile-ref")
    replacement = ProfileKeyMaterial(
        encryption_key=before.encryption_key,
        integrity_key=b"n" * 32,
        key_version=before.key_version + 1,
    )
    protector.replace("profile-ref", replacement)
    assert protector.load("profile-ref") == replacement
