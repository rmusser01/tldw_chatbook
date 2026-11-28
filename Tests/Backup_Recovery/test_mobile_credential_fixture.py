"""Isolated readable credential absence is distinct from an unavailable backend."""

from pathlib import Path

import pytest
from keyring.backends.null import Keyring

from tldw_chatbook.Backup_Recovery import credentials
from tldw_chatbook.Backup_Recovery.credential_policies import GENERATION_KEYRINGS
from tldw_chatbook.runtime_policy.server_credentials import (
    CredentialStoreUnavailable,
    KeyringServerCredentialStore,
    build_default_server_credential_store,
)


class EmptyIsolatedKeyring:
    def __init__(self):
        self.reads = []

    def get_password(self, service, username):
        self.reads.append((service, username))


def test_null_backend_remains_unavailable_for_default_native_store():
    with pytest.raises(CredentialStoreUnavailable):
        build_default_server_credential_store(keyring_backend=Keyring())


def test_readable_empty_fixture_checks_all_optional_generation_slots(monkeypatch):
    backend = EmptyIsolatedKeyring()
    store = KeyringServerCredentialStore(keyring_backend=backend)
    monkeypatch.setattr(credentials, "_credential_store", lambda: store)
    material, issues = [], []
    root = Path("/isolated-credential-test")
    credentials._capture_owned(
        "config", root / "config.toml", {}, root, material, issues
    )
    assert backend.reads == [
        (service, name) for _, service, names in GENERATION_KEYRINGS for name in names
    ]
    assert not material and not issues


@pytest.mark.parametrize("optional", [False, True])
def test_readable_absence_preserves_required_credential_refusal(monkeypatch, optional):
    backend = EmptyIsolatedKeyring()
    store = KeyringServerCredentialStore(keyring_backend=backend)
    monkeypatch.setattr(credentials, "_credential_store", lambda: store)
    material, issues = [], []
    credentials._capture_record(
        {
            "kind": "citation",
            "file": "chat.db",
            "service": "test-citation",
            "username": "test-key",
            "remappable": False,
        },
        material,
        issues,
        optional=optional,
    )
    assert backend.reads == [("test-citation", "test-key")]
    if optional:
        assert not material and not issues
    else:
        assert len(material) == len(issues) == 1
        assert material[0]["status"] == "missing"
        assert issues == ["credential_missing:" + material[0]["id"]]


@pytest.mark.parametrize("optional", [False, True])
def test_unreadable_fixture_never_becomes_optional_absence(monkeypatch, optional):
    class UnreadableKeyring:
        def get_password(self, service, username):
            raise RuntimeError("private-backend-error-must-not-escape")

    store = KeyringServerCredentialStore(keyring_backend=UnreadableKeyring())
    monkeypatch.setattr(credentials, "_credential_store", lambda: store)
    material, issues = [], []
    credentials._capture_record(
        {
            "kind": "generation",
            "file": "config.toml",
            "service": "test-generation",
            "username": "test-key",
            "remappable": False,
        },
        material,
        issues,
        optional=optional,
    )
    assert len(material) == len(issues) == 1
    assert material[0]["status"] == "unreadable"
    assert issues == ["credential_unreadable:" + material[0]["id"]]
    assert "private-backend-error" not in repr((material, issues))
