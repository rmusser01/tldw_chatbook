from __future__ import annotations

import json

import pytest

from tldw_chatbook.Personal_Context.key_protector import (
    KeyringProfileKeyProtector,
    PassphraseProfileKeyProtector,
    ProfileLockedError,
)


class SecureKeyring:
    __module__ = "keyring.backends.macOS"
    priority = 5

    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.values.get((service, username))

    def set_password(self, service: str, username: str, value: str) -> None:
        self.values[(service, username)] = value

    def delete_password(self, service: str, username: str) -> None:
        self.values.pop((service, username), None)


class PlaintextKeyring(SecureKeyring):
    __module__ = "keyring.backends.file"


def test_keyring_protector_round_trips_separate_keys_and_deletes() -> None:
    backend = SecureKeyring()
    protector = KeyringProfileKeyProtector(backend)

    created = protector.load_or_create("install-1")
    loaded = protector.load("install-1")

    assert len(created.encryption_key) == len(created.integrity_key) == 32
    assert created.encryption_key != created.integrity_key
    assert loaded == created
    assert b"\x00" not in next(iter(backend.values.values())).encode()

    protector.delete("install-1")
    with pytest.raises(ProfileLockedError, match="key material is unavailable"):
        protector.load("install-1")


def test_keyring_protector_rejects_unavailable_or_insecure_backend() -> None:
    with pytest.raises(ProfileLockedError, match="secure OS-backed"):
        KeyringProfileKeyProtector(PlaintextKeyring())


def test_passphrase_protector_persists_only_wrapped_bundle_and_reopens(
    tmp_path,
) -> None:
    bundle_path = tmp_path / "profile.keys"
    protector = PassphraseProfileKeyProtector(
        bundle_path, lambda: "correct horse battery staple"
    )

    created = protector.load_or_create("install-1")
    durable = bundle_path.read_bytes()

    assert created.encryption_key not in durable
    assert created.integrity_key not in durable
    assert b"correct horse battery staple" not in durable
    payload = json.loads(durable)
    assert payload["version"] == 1
    assert "ciphertext" in payload and "salt" in payload and "nonce" in payload
    assert (
        PassphraseProfileKeyProtector(
            bundle_path, lambda: "correct horse battery staple"
        ).load("install-1")
        == created
    )


def test_passphrase_cancellation_and_wrong_passphrase_lock_without_fallback(
    tmp_path,
) -> None:
    bundle_path = tmp_path / "profile.keys"
    with pytest.raises(ProfileLockedError, match="cancelled"):
        PassphraseProfileKeyProtector(bundle_path, lambda: None).load_or_create(
            "install-1"
        )
    assert not bundle_path.exists()

    PassphraseProfileKeyProtector(bundle_path, lambda: "right").load_or_create(
        "install-1"
    )
    original = bundle_path.read_bytes()
    with pytest.raises(ProfileLockedError, match="could not be unlocked"):
        PassphraseProfileKeyProtector(bundle_path, lambda: "wrong").load("install-1")
    assert bundle_path.read_bytes() == original

    with pytest.raises(ProfileLockedError, match="could not be unlocked"):
        PassphraseProfileKeyProtector(bundle_path, lambda: "right").load("install-2")
    assert bundle_path.read_bytes() == original


def test_missing_existing_passphrase_bundle_never_creates_replacement(tmp_path) -> None:
    bundle_path = tmp_path / "missing.keys"
    with pytest.raises(ProfileLockedError, match="key material is unavailable"):
        PassphraseProfileKeyProtector(bundle_path, lambda: "new-passphrase").load(
            "install-1"
        )
    assert not bundle_path.exists()
