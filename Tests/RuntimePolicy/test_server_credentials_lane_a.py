from __future__ import annotations

import json

from tldw_chatbook.runtime_policy.server_credentials import (
    DEFAULT_KEYRING_SERVICE_NAME,
    InMemoryServerCredentialStore,
    KeyringServerCredentialStore,
    SERVER_CREDENTIAL_ACCESS_TOKEN,
    SERVER_CREDENTIAL_REFRESH_TOKEN,
    ServerCredentialScope,
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


def test_keyring_scoped_principal_none_and_dash_do_not_collide():
    fake = FakeKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    null_principal_scope = ServerCredentialScope(
        server_profile_id="server-a",
        normalized_origin="https://server.example.com",
        credential_type=SERVER_CREDENTIAL_ACCESS_TOKEN,
        principal_id=None,
    )
    dash_principal_scope = ServerCredentialScope(
        server_profile_id="server-a",
        normalized_origin="https://server.example.com",
        credential_type=SERVER_CREDENTIAL_ACCESS_TOKEN,
        principal_id="-",
    )

    store.set_scoped_secret(null_principal_scope, "null-principal-secret")
    store.set_scoped_secret(dash_principal_scope, "dash-principal-secret")

    assert store.get_scoped_secret(null_principal_scope) == "null-principal-secret"
    assert store.get_scoped_secret(dash_principal_scope) == "dash-principal-secret"

    store.delete_scoped_secret(null_principal_scope)

    assert store.get_scoped_secret(null_principal_scope) is None
    assert store.get_scoped_secret(dash_principal_scope) == "dash-principal-secret"


def test_keyring_non_legacy_scoped_lookup_does_not_read_legacy_secret():
    fake = FakeKeyring()
    fake.values[(DEFAULT_KEYRING_SERVICE_NAME, "server-a:access_token")] = "legacy-secret"
    store = KeyringServerCredentialStore(keyring_backend=fake)
    non_legacy_scope = ServerCredentialScope(
        server_profile_id="server-a",
        normalized_origin="https://server.example.com",
        credential_type=SERVER_CREDENTIAL_ACCESS_TOKEN,
        principal_id="user-a",
    )

    assert store.get_scoped_secret(non_legacy_scope) is None


def test_keyring_non_legacy_scoped_delete_does_not_delete_legacy_secret():
    fake = FakeKeyring()
    legacy_key = (DEFAULT_KEYRING_SERVICE_NAME, "server-a:access_token")
    fake.values[legacy_key] = "legacy-secret"
    store = KeyringServerCredentialStore(keyring_backend=fake)
    non_legacy_scope = ServerCredentialScope(
        server_profile_id="server-a",
        normalized_origin="https://server.example.com",
        credential_type=SERVER_CREDENTIAL_ACCESS_TOKEN,
        principal_id="user-a",
    )

    store.delete_scoped_secret(non_legacy_scope)

    assert fake.values[legacy_key] == "legacy-secret"


def test_keyring_scoped_methods_normalize_and_validate_direct_scopes():
    fake = FakeKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    padded_scope = ServerCredentialScope(
        server_profile_id=" server-a ",
        normalized_origin=" https://server.example.com ",
        credential_type=f" {SERVER_CREDENTIAL_ACCESS_TOKEN} ",
        principal_id=" user-a ",
    )
    normalized_scope = ServerCredentialScope(
        server_profile_id="server-a",
        normalized_origin="https://server.example.com",
        credential_type=SERVER_CREDENTIAL_ACCESS_TOKEN,
        principal_id="user-a",
    )

    store.set_scoped_secret(padded_scope, "scoped-secret")

    assert store.get_scoped_secret(normalized_scope) == "scoped-secret"

    invalid_scope = ServerCredentialScope(
        server_profile_id="server-a",
        normalized_origin="   ",
        credential_type=SERVER_CREDENTIAL_ACCESS_TOKEN,
    )

    try:
        store.set_scoped_secret(invalid_scope, "invalid-secret")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid direct scope should raise ValueError")
    assert "invalid-secret" not in fake.values.values()


def test_keyring_clear_server_removes_all_indexed_entries_for_profile():
    fake = FakeKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    store.set_secret("server-a", "custom_token", "c1")

    store.clear_server("server-a")

    assert store.get_secret("server-a", "custom_token") is None


def test_keyring_clear_server_removes_unindexed_legacy_entries_for_profile():
    fake = FakeKeyring()
    legacy_key = (DEFAULT_KEYRING_SERVICE_NAME, "server-a:access_token")
    fake.values[legacy_key] = "legacy-secret"
    store = KeyringServerCredentialStore(keyring_backend=fake)

    assert store.get_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN) == "legacy-secret"

    store.clear_server("server-a")

    assert store.get_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN) is None
    assert legacy_key not in fake.values
