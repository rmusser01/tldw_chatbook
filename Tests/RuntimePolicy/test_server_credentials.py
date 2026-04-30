from __future__ import annotations

import pytest

from tldw_chatbook.runtime_policy import (
    DEFAULT_KEYRING_SERVICE_NAME,
    SERVER_CREDENTIAL_ACCESS_TOKEN,
    SERVER_CREDENTIAL_API_KEY,
    SERVER_CREDENTIAL_BEARER_TOKEN,
    SERVER_CREDENTIAL_REFRESH_TOKEN,
    InMemoryServerCredentialStore,
    KeyringServerCredentialStore,
    ServerCredentialRef,
    redact_secret,
)
from tldw_chatbook.runtime_policy.server_credentials import (
    CredentialStoreUnavailable,
    UnavailableServerCredentialStore,
    build_default_server_credential_store,
)


class FakeKeyring:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}
        self.deleted: list[tuple[str, str]] = []

    def set_password(self, service_name: str, username: str, password: str) -> None:
        self.values[(service_name, username)] = password

    def get_password(self, service_name: str, username: str) -> str | None:
        return self.values.get((service_name, username))

    def delete_password(self, service_name: str, username: str) -> None:
        self.deleted.append((service_name, username))
        self.values.pop((service_name, username), None)


class RaisingDeleteKeyring(FakeKeyring):
    def delete_password(self, service_name: str, username: str) -> None:
        self.deleted.append((service_name, username))
        raise RuntimeError("delete failed")


class PasswordDeleteErrorKeyring(FakeKeyring):
    class errors:
        class PasswordDeleteError(Exception):
            pass

    def delete_password(self, service_name: str, username: str) -> None:
        self.deleted.append((service_name, username))
        raise self.errors.PasswordDeleteError("delete failed")


class FakePlaintextKeyring:
    __module__ = "keyring.backends.file"
    priority = 1


class FakeFailKeyring:
    __module__ = "keyring.backends.fail"
    priority = 0


class FakeMacOSKeyring(FakeKeyring):
    __module__ = "keyring.backends.macOS"
    priority = 5


class FakeChainerKeyring:
    __module__ = "keyring.backends.chainer"

    def __init__(self, *backends):
        self.backends = list(backends)


def test_default_credential_store_rejects_plaintext_or_fail_backends():
    for backend in [FakePlaintextKeyring(), FakeFailKeyring()]:
        with pytest.raises(CredentialStoreUnavailable) as exc:
            build_default_server_credential_store(keyring_backend=backend)

        assert exc.value.reason_code == "credential_store_unavailable"


def test_unavailable_credential_store_disables_persistent_secret_operations():
    store = UnavailableServerCredentialStore("no secure store")

    with pytest.raises(CredentialStoreUnavailable) as exc:
        store.get_secret("https://server.example.com/api", SERVER_CREDENTIAL_ACCESS_TOKEN)

    assert exc.value.reason_code == "credential_store_unavailable"


def test_default_credential_store_inspects_wrapped_backends():
    with pytest.raises(CredentialStoreUnavailable):
        build_default_server_credential_store(keyring_backend=FakeChainerKeyring(FakePlaintextKeyring()))

    secure_child = FakeMacOSKeyring()
    store = build_default_server_credential_store(keyring_backend=FakeChainerKeyring(secure_child))

    assert isinstance(store, KeyringServerCredentialStore)


def test_in_memory_credentials_are_scoped_by_server_and_purpose():
    store = InMemoryServerCredentialStore()

    store.set_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN, "access-a")
    store.set_secret("server-b", SERVER_CREDENTIAL_ACCESS_TOKEN, "access-b")
    store.set_secret("server-a", SERVER_CREDENTIAL_REFRESH_TOKEN, "refresh-a")

    assert store.get_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN) == "access-a"
    assert store.get_secret("server-b", SERVER_CREDENTIAL_ACCESS_TOKEN) == "access-b"
    assert store.get_secret("server-a", SERVER_CREDENTIAL_REFRESH_TOKEN) == "refresh-a"


def test_in_memory_credentials_clear_one_server_without_touching_another():
    store = InMemoryServerCredentialStore()
    store.set_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN, "access-a")
    store.set_secret("server-b", SERVER_CREDENTIAL_ACCESS_TOKEN, "access-b")

    store.clear_server("server-a")

    assert store.get_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN) is None
    assert store.get_secret("server-b", SERVER_CREDENTIAL_ACCESS_TOKEN) == "access-b"


def test_in_memory_delete_one_purpose_leaves_other_purposes_for_server():
    store = InMemoryServerCredentialStore()
    store.set_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN, "access-a")
    store.set_secret("server-a", SERVER_CREDENTIAL_REFRESH_TOKEN, "refresh-a")

    store.delete_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN)

    assert store.get_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN) is None
    assert store.get_secret("server-a", SERVER_CREDENTIAL_REFRESH_TOKEN) == "refresh-a"


@pytest.mark.parametrize(
    ("server_id", "purpose"),
    [
        ("", SERVER_CREDENTIAL_ACCESS_TOKEN),
        ("   ", SERVER_CREDENTIAL_ACCESS_TOKEN),
        ("server-a", ""),
        ("server-a", "   "),
    ],
)
def test_empty_server_id_or_purpose_raises_value_error(server_id: str, purpose: str):
    store = InMemoryServerCredentialStore()

    with pytest.raises(ValueError):
        store.set_secret(server_id, purpose, "secret")

    with pytest.raises(ValueError):
        store.get_secret(server_id, purpose)

    with pytest.raises(ValueError):
        store.delete_secret(server_id, purpose)


def test_purpose_containing_colon_raises_value_error():
    store = InMemoryServerCredentialStore()

    with pytest.raises(ValueError):
        store.set_secret("server-a", "bad:purpose", "secret")

    with pytest.raises(ValueError):
        store.get_secret("server-a", "bad:purpose")

    with pytest.raises(ValueError):
        store.delete_secret("server-a", "bad:purpose")


def test_redact_secret_never_returns_original_non_empty_secret_and_handles_empty_values():
    assert redact_secret(None) == "<unset>"
    assert redact_secret("") == "<unset>"

    for secret in ["short", "12345678", "abcdef123456", "ab...3456", "ab...<redacted>...3456"]:
        redacted = redact_secret(secret)
        assert redacted != secret

    redacted_long = redact_secret("abcdef123456")
    assert redacted_long.startswith("ab")
    assert "3456" in redacted_long

    redacted_collision = redact_secret("ab...3456")
    assert redacted_collision.startswith("ab")
    assert "3456" in redacted_collision

    redacted_marker_collision = redact_secret("ab...<redacted>...3456")
    assert redacted_marker_collision.startswith("ab")
    assert "3456" in redacted_marker_collision


def test_server_credential_ref_username_uses_server_and_purpose():
    assert ServerCredentialRef("server-a", SERVER_CREDENTIAL_API_KEY).username == "server-a:api_key"


def test_keyring_store_uses_namespaced_username_and_supports_get_delete():
    fake = FakeKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)

    store.set_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN, "secret")

    stored_usernames = {username for service, username in fake.values if service == DEFAULT_KEYRING_SERVICE_NAME}
    assert any("tldw_chatbook.server_credentials" in username for username in stored_usernames)
    assert any("profile=server-a" in username for username in stored_usernames)
    assert any("type=access_token" in username for username in stored_usernames)
    assert store.get_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN) == "secret"

    store.delete_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN)

    assert store.get_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN) is None


def test_keyring_records_use_listable_chatbook_namespace():
    fake = FakeKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)

    store.set_secret("https://server.example.com/api", SERVER_CREDENTIAL_ACCESS_TOKEN, "secret")

    stored_usernames = {username for service, username in fake.values if service == DEFAULT_KEYRING_SERVICE_NAME}
    assert "__credential_refs__" in stored_usernames
    assert any("tldw_chatbook.server_credentials" in username for username in stored_usernames)
    assert any("profile=https%3A%2F%2Fserver.example.com%2Fapi" in username for username in stored_usernames)
    assert any("type=access_token" in username for username in stored_usernames)


def test_keyring_clear_all_enumerates_namespace_index_and_removes_orphans():
    fake = FakeKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    store.set_secret("orphan-profile", SERVER_CREDENTIAL_REFRESH_TOKEN, "zombie")

    store.clear_all()

    assert fake.values == {}


def test_keyring_delete_secret_tolerates_missing_values_without_calling_delete():
    fake = RaisingDeleteKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)

    store.delete_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN)

    assert fake.deleted == []


def test_keyring_delete_secret_propagates_existing_secret_runtime_delete_errors():
    fake = RaisingDeleteKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    store.set_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN, "secret")

    with pytest.raises(RuntimeError, match="delete failed"):
        store.delete_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN)


def test_keyring_delete_secret_propagates_existing_secret_password_delete_errors():
    fake = PasswordDeleteErrorKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    store.set_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN, "secret")

    with pytest.raises(fake.errors.PasswordDeleteError, match="delete failed"):
        store.delete_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN)


def test_keyring_clear_server_deletes_known_purpose_usernames_for_server():
    fake = FakeKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    for purpose in [
        SERVER_CREDENTIAL_ACCESS_TOKEN,
        SERVER_CREDENTIAL_REFRESH_TOKEN,
        SERVER_CREDENTIAL_API_KEY,
        SERVER_CREDENTIAL_BEARER_TOKEN,
    ]:
        store.set_secret("server-a", purpose, f"{purpose}-secret")

    store.clear_server("server-a")

    assert fake.values == {}
