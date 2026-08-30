from __future__ import annotations

import json

import pytest

from tldw_chatbook.Personal_Context import key_protector as key_protector_module
from tldw_chatbook.Personal_Context.key_protector import (
    InterviewPassphraseKeyProtector,
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


def test_interview_passphrase_protector_keeps_session_bundles_independent(
    tmp_path,
) -> None:
    key_directory = tmp_path / "interview-keys"
    key_directory.mkdir(mode=0o700)
    protector = InterviewPassphraseKeyProtector(
        key_directory, lambda: "correct horse battery staple"
    )

    first = protector.load_or_create("session-ref-1")
    second = protector.load_or_create("session-ref-2")

    assert first != second
    assert len(tuple(key_directory.iterdir())) == 2
    protector.delete("session-ref-1")
    with pytest.raises(ProfileLockedError, match="unavailable"):
        protector.load("session-ref-1")
    assert protector.load("session-ref-2") == second
    assert len(tuple(key_directory.iterdir())) == 1


def test_interview_passphrase_protector_rejects_unsafe_posix_directory(
    tmp_path,
) -> None:
    if not hasattr(key_protector_module.os, "geteuid"):
        pytest.skip("POSIX ownership checks are unavailable")
    key_directory = tmp_path / "interview-keys"
    key_directory.mkdir(mode=0o755)

    with pytest.raises(ProfileLockedError, match="not private"):
        InterviewPassphraseKeyProtector(key_directory, lambda: "passphrase")


def test_interview_passphrase_protector_without_posix_identity_api(
    tmp_path, monkeypatch
) -> None:
    key_directory = tmp_path / "interview-keys"
    key_directory.mkdir(mode=0o755)
    monkeypatch.delattr(key_protector_module.os, "geteuid", raising=False)

    protector = InterviewPassphraseKeyProtector(
        key_directory, lambda: "portable passphrase"
    )

    created = protector.load_or_create("session-ref")
    assert protector.load("session-ref") == created


def test_interview_passphrase_protector_rejects_directory_symlink(tmp_path) -> None:
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    key_directory = tmp_path / "interview-keys"
    key_directory.symlink_to(target, target_is_directory=True)

    with pytest.raises(ProfileLockedError, match="not private"):
        InterviewPassphraseKeyProtector(key_directory, lambda: "passphrase")
