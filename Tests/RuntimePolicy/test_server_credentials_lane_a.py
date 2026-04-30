from __future__ import annotations

import json

from tldw_chatbook.runtime_policy.server_credentials import (
    DEFAULT_KEYRING_SERVICE_NAME,
    InMemoryServerCredentialStore,
    KeyringServerCredentialStore,
    SERVER_CREDENTIAL_ACCESS_TOKEN,
    SERVER_CREDENTIAL_REFRESH_TOKEN,
)


class FakeKeyring:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def set_password(self, service_name: str, username: str, password: str) -> None:
        self.values[(service_name, username)] = password

    def get_password(self, service_name: str, username: str) -> str | None:
        return self.values.get((service_name, username))

    def delete_password(self, service_name: str, username: str) -> None:
        self.values.pop((service_name, username), None)


def test_clear_all_removes_credentials_for_multiple_servers():
    store = InMemoryServerCredentialStore()
    store.set_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN, "a1")
    store.set_secret("server-b", SERVER_CREDENTIAL_REFRESH_TOKEN, "b2")

    store.clear_all()

    assert store.get_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN) is None
    assert store.get_secret("server-b", SERVER_CREDENTIAL_REFRESH_TOKEN) is None


def test_keyring_clear_all_removes_indexed_entries():
    fake = FakeKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    store.set_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN, "a1")
    store.set_secret("server-zombie", SERVER_CREDENTIAL_REFRESH_TOKEN, "z9")

    index_entries = [
        ["server-a", SERVER_CREDENTIAL_ACCESS_TOKEN],
        ["server-zombie", SERVER_CREDENTIAL_REFRESH_TOKEN],
    ]
    index_key = (DEFAULT_KEYRING_SERVICE_NAME, "__credential_refs__")
    indexed = json.loads(fake.values[index_key])
    assert [
        [entry["server_profile_id"], entry["credential_type"]]
        for entry in indexed
    ] == index_entries
    assert all(entry["version"] == 1 for entry in indexed)
    assert all(
        entry["username"].startswith(f"{DEFAULT_KEYRING_SERVICE_NAME}:v1|")
        for entry in indexed
    )
    assert "a1" not in fake.values[index_key]
    assert "z9" not in fake.values[index_key]

    store.clear_all()

    assert fake.values == {}


def test_keyring_clear_all_removes_legacy_list_index_entries():
    fake = FakeKeyring()
    index_key = (DEFAULT_KEYRING_SERVICE_NAME, "__credential_refs__")
    fake.values[index_key] = json.dumps([["server-a", SERVER_CREDENTIAL_ACCESS_TOKEN]])
    fake.values[(DEFAULT_KEYRING_SERVICE_NAME, "server-a:access_token")] = "legacy-secret"
    store = KeyringServerCredentialStore(keyring_backend=fake)

    store.clear_all()

    assert fake.values == {}


def test_keyring_clear_server_removes_all_indexed_entries_for_profile():
    fake = FakeKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    store.set_secret("server-a", "custom_token", "c1")

    store.clear_server("server-a")

    assert store.get_secret("server-a", "custom_token") is None
