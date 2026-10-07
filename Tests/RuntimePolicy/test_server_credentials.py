from __future__ import annotations

import json
from threading import Event, Thread, current_thread

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
    ServerCredentialScope,
    redact_secret,
)
from tldw_chatbook.runtime_policy.server_credentials import (
    CredentialStoreUnavailable,
    UnavailableServerCredentialStore,
    build_default_server_credential_store,
    is_secure_keyring_backend,
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


class BlobLimitedKeyring(FakeKeyring):
    """Bound only the native effect, matching Windows' UTF-16 credential blob."""

    def __init__(self) -> None:
        super().__init__()
        self.fail_root = False
        self.fail_part_after: int | None = None
        self.part_writes = 0
        self.fail_cleanup = False
        self.commit_root_then_raise = False
        self.commit_part_then_raise = False

    def set_password(self, service_name: str, username: str, password: str) -> None:
        if len(password.encode("utf-16-le")) > 2560:
            raise RuntimeError("credential_blob_limit")
        if username == "__credential_refs__" and self.fail_root:
            raise RuntimeError("index_write_failed")
        if username == "__credential_refs__" and self.commit_root_then_raise:
            super().set_password(service_name, username, password)
            raise RuntimeError("index_write_uncertain")
        if username.startswith("__credential_refs__:"):
            if self.part_writes == self.fail_part_after:
                if self.commit_part_then_raise:
                    super().set_password(service_name, username, password)
                raise RuntimeError("index_write_failed")
            self.part_writes += 1
        super().set_password(service_name, username, password)

    def delete_password(self, service_name: str, username: str) -> None:
        if self.fail_cleanup and username.startswith("__credential_refs__:"):
            raise RuntimeError("cleanup_failed")
        super().delete_password(service_name, username)


class MetadataCountingKeyring(BlobLimitedKeyring):
    def __init__(self) -> None:
        super().__init__()
        self.metadata_calls = {"get": 0, "set": 0, "delete": 0}
        self.root_writes = 0

    def get_password(self, service_name, username):
        if username.startswith("__credential_refs__"):
            self.metadata_calls["get"] += 1
        return super().get_password(service_name, username)

    def set_password(self, service_name, username, password):
        if username.startswith("__credential_refs__"):
            self.metadata_calls["set"] += 1
        if username == "__credential_refs__":
            self.root_writes += 1
        super().set_password(service_name, username, password)

    def delete_password(self, service_name, username):
        if username.startswith("__credential_refs__"):
            self.metadata_calls["delete"] += 1
        super().delete_password(service_name, username)


def _native_index_scopes(count: int = 24) -> list[ServerCredentialScope]:
    return [
        ServerCredentialScope(
            server_profile_id=f"C:/Users/runneradmin/AppData/Local/Temp/qualification/profile-{number % 2}/config.toml",
            normalized_origin=f"https://native-purpose-{number}.invalid",
            credential_type=SERVER_CREDENTIAL_API_KEY,
            principal_id="disposable-principal-\N{KEY}",
        )
        for number in range(count)
    ]


@pytest.mark.parametrize("operation", ["absent_server", "absent_origin", "empty_all"])
def test_bulk_clear_without_indexed_matches_skips_index_publication(operation):
    fake = MetadataCountingKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    if operation == "empty_all":
        fake.values[(DEFAULT_KEYRING_SERVICE_NAME, "__credential_refs__")] = "[]"
    else:
        store.set_secret("indexed-server", SERVER_CREDENTIAL_API_KEY, "disposable")
    previous = fake.values.copy()
    fake.fail_root = True
    fake.root_writes = 0

    if operation == "empty_all":
        store.clear_all()
    elif operation == "absent_origin":
        store.clear_server("indexed-server", normalized_origin="absent-origin")
    else:
        store.clear_server("absent-server")

    assert fake.values == previous
    assert fake.root_writes == 0


@pytest.mark.parametrize("failure", [False, True])
def test_unindexed_legacy_clear_skips_index_publication(failure):
    class FailingLegacyKeyring(MetadataCountingKeyring):
        def delete_password(self, service_name, username):
            if failure and username == "unindexed-server:api_key":
                raise RuntimeError("credential_delete_failed")
            super().delete_password(service_name, username)

    fake = FailingLegacyKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    store.set_secret("indexed-server", SERVER_CREDENTIAL_API_KEY, "disposable")
    expected = fake.values.copy()
    for purpose in (SERVER_CREDENTIAL_ACCESS_TOKEN, SERVER_CREDENTIAL_API_KEY):
        fake.values[(DEFAULT_KEYRING_SERVICE_NAME, f"unindexed-server:{purpose}")] = (
            "legacy-disposable"
        )
        assert store.get_secret("unindexed-server", purpose) == "legacy-disposable"
    fake.fail_root = True
    fake.root_writes = 0

    if failure:
        with pytest.raises(RuntimeError, match="credential_delete_failed") as caught:
            store.clear_server("unindexed-server")
        assert caught.value.__cause__ is None
        expected[(DEFAULT_KEYRING_SERVICE_NAME, "unindexed-server:api_key")] = (
            "legacy-disposable"
        )
    else:
        store.clear_server("unindexed-server")
        assert store.get_secret("unindexed-server", SERVER_CREDENTIAL_API_KEY) is None

    assert store.get_secret("unindexed-server", SERVER_CREDENTIAL_ACCESS_TOKEN) is None
    assert fake.values == expected
    assert fake.root_writes == 0


@pytest.mark.parametrize("operation", ["clear_all", "clear_server"])
@pytest.mark.parametrize("count", [16, 32])
def test_bulk_clear_metadata_work_is_linear(operation, count):
    fake = MetadataCountingKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    scopes = _native_index_scopes(count)
    for scope in scopes:
        store.set_scoped_secret(scope, "disposable")
    header = json.loads(
        fake.values[(DEFAULT_KEYRING_SERVICE_NAME, "__credential_refs__")]
    )
    previous_parts = header["parts"]
    fake.metadata_calls = {"get": 0, "set": 0, "delete": 0}
    fake.root_writes = 0

    if operation == "clear_all":
        store.clear_all()
    else:
        store.clear_server(scopes[0].server_profile_id)

    assert fake.metadata_calls["get"] <= 2 * previous_parts + 4
    assert fake.metadata_calls["set"] <= previous_parts + 1
    assert fake.metadata_calls["delete"] == previous_parts
    assert fake.root_writes <= 1
    fresh = KeyringServerCredentialStore(keyring_backend=fake)
    remaining = [] if operation == "clear_all" else scopes[1::2]
    assert set(fresh._load_index()) == set(remaining)
    assert [fresh.get_scoped_secret(scope) for scope in scopes] == [
        "disposable" if scope in remaining else None for scope in scopes
    ]


@pytest.mark.parametrize("operation", ["clear_all", "clear_server"])
@pytest.mark.parametrize("failure_location", ["scoped", "legacy"])
def test_bulk_clear_partial_failure_retains_failed_and_pending_scopes(
    operation, failure_location
):
    from tldw_chatbook.runtime_policy.server_credentials import _username_for_scope

    scopes = [
        ServerCredentialScope.legacy("server-a", purpose)
        for purpose in (
            SERVER_CREDENTIAL_ACCESS_TOKEN,
            SERVER_CREDENTIAL_API_KEY,
            SERVER_CREDENTIAL_BEARER_TOKEN,
            SERVER_CREDENTIAL_REFRESH_TOKEN,
        )
    ]
    other = ServerCredentialScope.legacy("server-b", SERVER_CREDENTIAL_API_KEY)

    class FailingDeleteKeyring(MetadataCountingKeyring):
        failure_username = None

        def delete_password(self, service_name, username):
            if username == self.failure_username:
                # Both earlier scopes must already be uncached during the batch.
                assert all(
                    store.get_scoped_secret(scope) is None for scope in scopes[:2]
                )
                raise RuntimeError("credential_delete_failed")
            super().delete_password(service_name, username)

    fake = FailingDeleteKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    for scope in [*scopes, other]:
        store.set_scoped_secret(scope, "disposable")
        fake.values[
            (
                DEFAULT_KEYRING_SERVICE_NAME,
                f"{scope.server_profile_id}:{scope.credential_type}",
            )
        ] = "legacy-disposable"
    # Exercise a legacy-only value as well as a scope with both records.
    fake.values.pop((DEFAULT_KEYRING_SERVICE_NAME, _username_for_scope(scopes[1])))
    for scope in [*scopes, other]:
        assert store.get_scoped_secret(scope) is not None
    fake.failure_username = (
        _username_for_scope(scopes[2])
        if failure_location == "scoped"
        else "server-a:bearer_token"
    )
    fake.root_writes = 0

    with pytest.raises(RuntimeError, match="credential_delete_failed"):
        if operation == "clear_all":
            store.clear_all()
        else:
            store.clear_server("server-a")

    assert fake.root_writes == 1
    assert store.get_scoped_secret(scopes[2]) == (
        "disposable" if failure_location == "scoped" else "legacy-disposable"
    )
    fresh = KeyringServerCredentialStore(keyring_backend=fake)
    assert set(fresh._load_index()) == {*scopes[2:], other}
    assert all(fresh.get_scoped_secret(scope) is None for scope in scopes[:2])
    assert all(
        fresh.get_scoped_secret(scope) is not None for scope in [*scopes[2:], other]
    )
    fake.failure_username = None
    fresh.clear_all()
    assert fake.values == {}


@pytest.mark.parametrize("operation", ["clear_all", "clear_server"])
def test_bulk_clear_preserves_value_error_when_remaining_index_publish_fails(operation):
    from tldw_chatbook.runtime_policy.server_credentials import _username_for_scope

    class FailingDeleteKeyring(BlobLimitedKeyring):
        failure_username = None

        def delete_password(self, service_name, username):
            if username == self.failure_username:
                raise RuntimeError("credential_delete_failed")
            super().delete_password(service_name, username)

    fake = FailingDeleteKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    for scope in _native_index_scopes(8):
        store.set_scoped_secret(scope, "disposable")
    scopes = store._load_index()
    fake.failure_username = _username_for_scope(scopes[1])
    fake.fail_root = True

    with pytest.raises(RuntimeError, match="credential_delete_failed") as caught:
        if operation == "clear_all":
            store.clear_all()
        else:
            store.clear_server(scopes[0].server_profile_id)

    assert str(caught.value.__cause__) == "index_write_failed"
    fresh = KeyringServerCredentialStore(keyring_backend=fake)
    # Failed publication retains the old index and all possibly pending references.
    assert fresh._load_index() == scopes
    assert fresh.get_scoped_secret(scopes[0]) is None
    assert all(fresh.get_scoped_secret(scope) is not None for scope in scopes[1:])


def test_native_index_survives_growth_fresh_store_and_scoped_clear():
    fake = BlobLimitedKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    scopes = _native_index_scopes()
    for number, scope in enumerate(scopes):
        store.set_scoped_secret(scope, f"disposable-{number}")

    fresh = KeyringServerCredentialStore(keyring_backend=fake)
    assert [fresh.get_scoped_secret(scope) for scope in scopes] == [
        f"disposable-{number}" for number in range(len(scopes))
    ]
    assert all(len(value.encode("utf-16-le")) <= 2560 for value in fake.values.values())
    fresh.clear_server(scopes[0].server_profile_id)
    assert [fresh.get_scoped_secret(scope) for scope in scopes] == [
        None if number % 2 == 0 else f"disposable-{number}"
        for number in range(len(scopes))
    ]
    KeyringServerCredentialStore(keyring_backend=fake).clear_all()
    assert fake.values == {}


@pytest.mark.parametrize("failure", ["part", "root"])
def test_failed_native_index_write_preserves_previous_index(failure):
    fake = BlobLimitedKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    scopes = _native_index_scopes(8)
    for number, scope in enumerate(scopes):
        store.set_scoped_secret(scope, f"disposable-{number}")
    index_key = (DEFAULT_KEYRING_SERVICE_NAME, "__credential_refs__")
    previous = fake.values[index_key]
    previous_parts = {
        key for key in fake.values if key[1].startswith("__credential_refs__:")
    }
    if failure == "part":
        fake.fail_part_after = fake.part_writes + 1
    else:
        fake.fail_root = True

    with pytest.raises(RuntimeError, match="index_write_failed"):
        store.set_scoped_secret(_native_index_scopes(9)[-1], "new-disposable")

    assert fake.values[index_key] == previous
    assert {
        key for key in fake.values if key[1].startswith("__credential_refs__:")
    } == previous_parts
    fake.fail_root = False
    fake.fail_part_after = None
    fresh = KeyringServerCredentialStore(keyring_backend=fake)
    fresh.clear_all()
    assert all(fresh.get_scoped_secret(scope) is None for scope in scopes)
    if failure == "part":
        assert not any(
            username.startswith("__credential_refs__") for _, username in fake.values
        )


def test_native_index_failed_initial_root_publication_removes_written_parts():
    fake = BlobLimitedKeyring()
    fake.fail_root = True

    with pytest.raises(RuntimeError, match="index_write_failed"):
        KeyringServerCredentialStore(keyring_backend=fake)._save_index(
            _native_index_scopes(8)
        )

    assert fake.values == {}


@pytest.mark.parametrize("existing", [False, True])
def test_native_index_part_commit_then_error_cleans_attempted_generation(existing):
    fake = BlobLimitedKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    scopes = _native_index_scopes(8)
    if existing:
        store._save_index(scopes[:-1])
    previous = fake.values.copy()
    fake.fail_part_after = fake.part_writes + 1
    fake.commit_part_then_raise = True

    with pytest.raises(RuntimeError, match="index_write_failed"):
        store._save_index(scopes)

    assert fake.values == previous


@pytest.mark.parametrize("committed", [False, True])
def test_native_index_unavailable_root_readback_retains_new_parts(committed):
    class UnavailableReadbackKeyring(BlobLimitedKeyring):
        root_write_failed = False
        unavailable_readback = True

        def set_password(self, service_name, username, password):
            try:
                super().set_password(service_name, username, password)
            except RuntimeError:
                if username == "__credential_refs__":
                    self.root_write_failed = True
                raise

        def get_password(self, service_name, username):
            if (
                username == "__credential_refs__"
                and self.root_write_failed
                and self.unavailable_readback
            ):
                raise RuntimeError("index_readback_unavailable")
            return super().get_password(service_name, username)

    fake = UnavailableReadbackKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    scopes = _native_index_scopes(9)
    for scope in scopes[:-1]:
        store.set_scoped_secret(scope, "disposable")
    previous_parts = {
        key for key in fake.values if key[1].startswith("__credential_refs__:")
    }
    fake.fail_root = not committed
    fake.commit_root_then_raise = committed

    with pytest.raises(
        RuntimeError,
        match="index_write_uncertain" if committed else "index_write_failed",
    ):
        store.set_scoped_secret(scopes[-1], "new-disposable")

    assert previous_parts < {
        key for key in fake.values if key[1].startswith("__credential_refs__:")
    }
    fake.unavailable_readback = False
    assert set(store._load_index()) == set(scopes if committed else scopes[:-1])


def test_native_index_root_commit_then_error_keeps_published_index_readable():
    fake = BlobLimitedKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    scopes = _native_index_scopes(9)
    for number, scope in enumerate(scopes[:-1]):
        store.set_scoped_secret(scope, f"disposable-{number}")
    fake.commit_root_then_raise = True

    with pytest.raises(RuntimeError, match="index_write_uncertain"):
        store.set_scoped_secret(scopes[-1], "new-disposable")

    fake.commit_root_then_raise = False
    fresh = KeyringServerCredentialStore(keyring_backend=fake)
    assert [fresh.get_scoped_secret(scope) for scope in scopes[:-1]] == [
        f"disposable-{number}" for number in range(len(scopes) - 1)
    ]
    fresh.clear_all()
    assert all(fresh.get_scoped_secret(scope) is None for scope in scopes)


def test_native_index_growth_keeps_legacy_list_entries_clearable():
    fake = BlobLimitedKeyring()
    fake.values[(DEFAULT_KEYRING_SERVICE_NAME, "__credential_refs__")] = json.dumps(
        [["legacy-server", SERVER_CREDENTIAL_ACCESS_TOKEN]]
    )
    fake.values[(DEFAULT_KEYRING_SERVICE_NAME, "legacy-server:access_token")] = (
        "legacy-disposable"
    )
    store = KeyringServerCredentialStore(keyring_backend=fake)
    for scope in _native_index_scopes(8):
        store.set_scoped_secret(scope, "scoped-disposable")

    fresh = KeyringServerCredentialStore(keyring_backend=fake)
    assert (
        fresh.get_secret("legacy-server", SERVER_CREDENTIAL_ACCESS_TOKEN)
        == "legacy-disposable"
    )
    fresh.clear_all()
    assert fake.values == {}


@pytest.mark.parametrize("damage", ["missing", "malformed", "unsupported_header"])
def test_damaged_native_index_part_refuses_clear_before_any_credential_delete(damage):
    fake = BlobLimitedKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    for scope in _native_index_scopes(8):
        store.set_scoped_secret(scope, "scoped-disposable")
    missing = next(
        key for key in fake.values if key[1].startswith("__credential_refs__:")
    )
    if damage == "unsupported_header":
        index_key = (DEFAULT_KEYRING_SERVICE_NAME, "__credential_refs__")
        header = json.loads(fake.values[index_key])
        header["version"] = 3
        fake.values[index_key] = json.dumps(header)
    elif damage == "missing":
        fake.values.pop(missing)
    else:
        fake.values[missing] = "invalid-json"
    previous = fake.values.copy()

    with pytest.raises(CredentialStoreUnavailable):
        KeyringServerCredentialStore(keyring_backend=fake).clear_all()

    assert fake.values == previous


@pytest.mark.parametrize("parts", [16_385, 1_000_001])
def test_native_index_oversized_header_refuses_work_before_part_reads(parts):
    fake = FakeKeyring()
    fake.values[(DEFAULT_KEYRING_SERVICE_NAME, "__credential_refs__")] = json.dumps(
        {"version": 2, "generation": "a" * 32, "parts": parts}
    )
    requested = []
    original_get = fake.get_password

    def get_password(service_name, username):
        requested.append(username)
        return original_get(service_name, username)

    fake.get_password = get_password

    with pytest.raises(CredentialStoreUnavailable):
        KeyringServerCredentialStore(keyring_backend=fake).clear_all()

    assert requested == ["__credential_refs__"]
    assert fake.deleted == []


@pytest.mark.parametrize(
    "chunk",
    [None, "", "x" * 1025, 1],
    ids=["missing", "empty", "oversized", "nonstring"],
)
def test_native_index_invalid_first_chunk_stops_before_later_part_reads(chunk):
    fake = FakeKeyring()
    fake.values[(DEFAULT_KEYRING_SERVICE_NAME, "__credential_refs__")] = json.dumps(
        {"version": 2, "generation": "a" * 32, "parts": 3}
    )
    first_part = "__credential_refs__:" + "a" * 32 + ":0"
    if chunk is not None:
        fake.values[(DEFAULT_KEYRING_SERVICE_NAME, first_part)] = chunk
    requested = []
    original_get = fake.get_password

    def get_password(service_name, username):
        requested.append(username)
        return original_get(service_name, username)

    fake.get_password = get_password

    with pytest.raises(CredentialStoreUnavailable):
        KeyringServerCredentialStore(keyring_backend=fake).clear_all()

    assert requested == ["__credential_refs__", first_part]
    assert fake.deleted == []


def test_native_index_oversized_save_refuses_before_part_writes():
    fake = BlobLimitedKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    scope = ServerCredentialScope(
        server_profile_id="p" * (9 * 1024**2),
        normalized_origin="https://disposable.invalid",
        credential_type=SERVER_CREDENTIAL_API_KEY,
    )

    with pytest.raises(CredentialStoreUnavailable):
        store._save_index([scope])

    assert fake.values == {}


@pytest.mark.parametrize("operation", ["add", "update"])
def test_scoped_set_oversized_index_refuses_before_credential_write(
    operation, monkeypatch
):
    from tldw_chatbook.runtime_policy import server_credentials

    fake = FakeKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    existing = ServerCredentialScope.legacy("server-a", SERVER_CREDENTIAL_API_KEY)
    store.set_scoped_secret(existing, "previous-disposable")
    index_key = (DEFAULT_KEYRING_SERVICE_NAME, "__credential_refs__")
    # The compact old-format index fits, but its normalized replacement does not.
    fake.values[index_key] = json.dumps([["server-a", SERVER_CREDENTIAL_API_KEY]])
    monkeypatch.setattr(
        server_credentials,
        "_KEYRING_INDEX_MAX_TOTAL_CHARACTERS",
        len(fake.values[index_key]),
    )
    previous = fake.values.copy()
    scope = (
        existing
        if operation == "update"
        else ServerCredentialScope.legacy("server-b", SERVER_CREDENTIAL_API_KEY)
    )

    with pytest.raises(CredentialStoreUnavailable):
        store.set_scoped_secret(scope, "new-disposable")

    assert fake.values == previous
    fresh = KeyringServerCredentialStore(keyring_backend=fake)
    assert fresh._load_index() == [existing]
    assert fresh.get_scoped_secret(existing) == "previous-disposable"
    if operation == "add":
        assert fresh.get_scoped_secret(scope) is None


def test_scoped_set_preflight_deduplicates_normalized_scope_before_publication(
    monkeypatch,
):
    from tldw_chatbook.runtime_policy import server_credentials

    scope = ServerCredentialScope.legacy("server-a", SERVER_CREDENTIAL_API_KEY)
    username = server_credentials._username_for_scope(scope)

    class OrderedKeyring(FakeKeyring):
        check_order = False
        checked_publication = False

        def set_password(self, service_name, record_username, password):
            if self.check_order and record_username == "__credential_refs__":
                assert self.values[(service_name, username)] == "updated-disposable"
                self.checked_publication = True
            super().set_password(service_name, record_username, password)

    fake = OrderedKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    store.set_scoped_secret(scope, "previous-disposable")
    index_key = (DEFAULT_KEYRING_SERVICE_NAME, "__credential_refs__")
    previous_index = fake.values[index_key]
    monkeypatch.setattr(
        server_credentials, "_KEYRING_INDEX_MAX_TOTAL_CHARACTERS", len(previous_index)
    )
    fake.check_order = True

    store.set_scoped_secret(
        ServerCredentialScope(" server-a ", " server-a ", " api_key "),
        "updated-disposable",
    )

    assert fake.checked_publication
    assert fake.values[index_key] == previous_index
    assert store.get_scoped_secret(scope) == "updated-disposable"


def test_native_index_oversized_legacy_payload_refuses_before_deletion():
    fake = FakeKeyring()
    fake.values[(DEFAULT_KEYRING_SERVICE_NAME, "__credential_refs__")] = "[]" + " " * (
        16 * 1024**2 - 1
    )

    with pytest.raises(CredentialStoreUnavailable):
        KeyringServerCredentialStore(keyring_backend=fake).clear_all()

    assert fake.values
    assert fake.deleted == []


@pytest.mark.parametrize("operation", ["delete", "clear_server", "clear_all"])
def test_native_index_destructive_transactions_wait_for_scoped_write(
    operation, monkeypatch
):
    from tldw_chatbook.runtime_policy import server_credentials

    paused = Event()
    resume = Event()
    contender_boundary = Event()
    contender_done = Event()
    contender_read = Event()
    contender_lock = Event()
    errors = []
    existing_lock = server_credentials._RECOVERY_SCOPE_LOCK

    class ObservedLock:
        def __enter__(self):
            if current_thread().name == "index-contender":
                contender_lock.set()
                contender_boundary.set()
            existing_lock.acquire()
            return self

        def __exit__(self, *args):
            existing_lock.release()

    class PausingKeyring(BlobLimitedKeyring):
        pause_owner = False

        def get_password(self, service_name, username):
            value = super().get_password(service_name, username)
            if username == "__credential_refs__":
                if current_thread().name == "index-owner" and self.pause_owner:
                    self.pause_owner = False
                    paused.set()
                    if not resume.wait(10):
                        raise RuntimeError("test_barrier_timeout")
                elif current_thread().name == "index-contender":
                    contender_read.set()
                    contender_boundary.set()
            return value

    monkeypatch.setattr(server_credentials, "_RECOVERY_SCOPE_LOCK", ObservedLock())
    fake = PausingKeyring()
    scopes = _native_index_scopes(9)
    store = KeyringServerCredentialStore(keyring_backend=fake)
    for scope in scopes[:-1]:
        store.set_scoped_secret(scope, "disposable")
    other_store = KeyringServerCredentialStore(keyring_backend=fake)
    fake.pause_owner = True

    def write():
        try:
            store.set_scoped_secret(scopes[-1], "new-disposable")
        except Exception as error:  # noqa: BLE001 - retain thread failures for the owner assertion.
            errors.append(type(error).__name__)

    def destroy():
        try:
            if operation == "delete":
                other_store.delete_scoped_secret(scopes[0])
            elif operation == "clear_server":
                other_store.clear_server(scopes[0].server_profile_id)
            else:
                other_store.clear_all()
        except Exception as error:  # noqa: BLE001 - retain thread failures for the owner assertion.
            errors.append(type(error).__name__)
        finally:
            contender_done.set()

    owner = Thread(target=write, name="index-owner")
    contender = Thread(target=destroy, name="index-contender")
    owner.start()
    try:
        assert paused.wait(10)
        contender.start()
        assert contender_boundary.wait(10)
        if not contender_lock.is_set():
            assert contender_done.wait(10)
        read_before_release = contender_read.is_set()
    finally:
        resume.set()
        owner.join(10)
        if contender.ident is not None:
            contender.join(10)

    assert not owner.is_alive() and not contender.is_alive()
    assert not read_before_release
    assert errors == []
    fresh = KeyringServerCredentialStore(keyring_backend=fake)
    remaining = (
        []
        if operation == "clear_all"
        else [
            scope
            for scope in scopes
            if (
                scope != scopes[0]
                if operation == "delete"
                else scope.server_profile_id != scopes[0].server_profile_id
            )
        ]
    )
    assert set(fresh._load_index()) == set(remaining)
    if operation == "clear_all":
        assert fake.values == {}


def test_native_index_cleanup_failure_keeps_committed_credentials_readable():
    fake = BlobLimitedKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    scopes = _native_index_scopes(8)
    for number, scope in enumerate(scopes[:-1]):
        store.set_scoped_secret(scope, f"disposable-{number}")
    fake.fail_cleanup = True

    store.set_scoped_secret(scopes[-1], "new-disposable")

    fresh = KeyringServerCredentialStore(keyring_backend=fake)
    assert [fresh.get_scoped_secret(scope) for scope in scopes] == [
        *(f"disposable-{number}" for number in range(len(scopes) - 1)),
        "new-disposable",
    ]
    fake.fail_cleanup = False
    fresh.clear_all()
    assert all(fresh.get_scoped_secret(scope) is None for scope in scopes)


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


def _fake_keyring_backend(module_name: str, *, priority: int = 1):
    return type("FakeBackend", (), {"__module__": module_name, "priority": priority})()


@pytest.mark.parametrize(
    ("module_name", "expected_secure"),
    [
        ("keyring.backends.SecretService", True),
        ("keyring.backends.libsecret", True),
        ("keyring.backends.kwallet", True),
        ("keyring.backends.null", False),
        ("keyring.backends.fail", False),
        ("keyring.backends.file", False),
        ("keyring.backends.unknown", False),
    ],
)
def test_secure_keyring_classifier_recognizes_only_secure_backend_modules(
    module_name: str,
    expected_secure: bool,
):
    assert (
        is_secure_keyring_backend(_fake_keyring_backend(module_name)) is expected_secure
    )


def test_default_credential_store_rejects_plaintext_or_fail_backends():
    for backend in [FakePlaintextKeyring(), FakeFailKeyring()]:
        with pytest.raises(CredentialStoreUnavailable) as exc:
            build_default_server_credential_store(keyring_backend=backend)

        assert exc.value.reason_code == "credential_store_unavailable"


def test_unavailable_credential_store_disables_persistent_secret_operations():
    store = UnavailableServerCredentialStore("no secure store")

    with pytest.raises(CredentialStoreUnavailable) as exc:
        store.get_secret(
            "https://server.example.com/api", SERVER_CREDENTIAL_ACCESS_TOKEN
        )

    assert exc.value.reason_code == "credential_store_unavailable"


def test_default_credential_store_inspects_wrapped_backends():
    with pytest.raises(CredentialStoreUnavailable):
        build_default_server_credential_store(
            keyring_backend=FakeChainerKeyring(FakePlaintextKeyring())
        )

    with pytest.raises(CredentialStoreUnavailable):
        build_default_server_credential_store(
            keyring_backend=FakeChainerKeyring(
                FakePlaintextKeyring(), FakeMacOSKeyring()
            )
        )

    secure_child = FakeMacOSKeyring()
    store = build_default_server_credential_store(
        keyring_backend=FakeChainerKeyring(secure_child)
    )

    assert isinstance(store, KeyringServerCredentialStore)
    assert store._keyring is secure_child


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


def test_in_memory_clear_server_with_origin_scopes_to_one_server_in_shared_profile():
    """task-31824: two servers sharing one `server_profile_id` (the
    TLDW_CONFIG_PATH scoped-mode shape) must stay independently clearable.
    `clear_server(profile)` alone clears every origin under that profile;
    passing `normalized_origin` narrows the clear to one server."""
    store = InMemoryServerCredentialStore()
    scope_a = ServerCredentialScope(
        server_profile_id="shared-profile",
        normalized_origin="server-a",
        credential_type=SERVER_CREDENTIAL_ACCESS_TOKEN,
    )
    scope_a_bearer = ServerCredentialScope(
        server_profile_id="shared-profile",
        normalized_origin="server-a",
        credential_type=SERVER_CREDENTIAL_BEARER_TOKEN,
    )
    scope_b = ServerCredentialScope(
        server_profile_id="shared-profile",
        normalized_origin="server-b",
        credential_type=SERVER_CREDENTIAL_ACCESS_TOKEN,
    )
    store.set_scoped_secret(scope_a, "a-secret")
    store.set_scoped_secret(scope_a_bearer, "a-bearer")
    store.set_scoped_secret(scope_b, "b-secret")

    store.clear_server("shared-profile", normalized_origin="server-a")

    assert store.get_scoped_secret(scope_a) is None
    assert store.get_scoped_secret(scope_a_bearer) is None
    assert store.get_scoped_secret(scope_b) == "b-secret"


def test_in_memory_delete_one_purpose_leaves_other_purposes_for_server():
    store = InMemoryServerCredentialStore()
    store.set_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN, "access-a")
    store.set_secret("server-a", SERVER_CREDENTIAL_REFRESH_TOKEN, "refresh-a")

    store.delete_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN)

    assert store.get_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN) is None
    assert store.get_secret("server-a", SERVER_CREDENTIAL_REFRESH_TOKEN) == "refresh-a"


def test_in_memory_plain_and_legacy_scoped_api_read_each_others_writes():
    """The plain API and `ServerCredentialScope.legacy` scope the same slot
    (task-31416): a caller using either sees what the other wrote."""
    store = InMemoryServerCredentialStore()
    store.set_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN, "access-a")

    scope = ServerCredentialScope.legacy("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN)
    assert store.get_scoped_secret(scope) == "access-a"

    store.set_scoped_secret(scope, "access-a-updated")
    assert store.get_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN) == (
        "access-a-updated"
    )


def test_in_memory_scoped_secrets_isolate_two_profiles_at_the_same_origin():
    """Two distinct `server_profile_id`s at the same `normalized_origin` do
    not share a secret (task-31416 AC#1)."""
    store = InMemoryServerCredentialStore()
    origin = "https://server.example.com/api"
    scope_a = ServerCredentialScope(
        server_profile_id="profile-a",
        normalized_origin=origin,
        credential_type=SERVER_CREDENTIAL_ACCESS_TOKEN,
    )
    scope_b = ServerCredentialScope(
        server_profile_id="profile-b",
        normalized_origin=origin,
        credential_type=SERVER_CREDENTIAL_ACCESS_TOKEN,
    )

    store.set_scoped_secret(scope_a, "token-a")
    store.set_scoped_secret(scope_b, "token-b")

    assert store.get_scoped_secret(scope_a) == "token-a"
    assert store.get_scoped_secret(scope_b) == "token-b"

    store.delete_scoped_secret(scope_a)

    assert store.get_scoped_secret(scope_a) is None
    assert store.get_scoped_secret(scope_b) == "token-b"


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

    for secret in [
        "short",
        "12345678",
        "abcdef123456",
        "ab...3456",
        "ab...<redacted>...3456",
    ]:
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
    assert (
        ServerCredentialRef("server-a", SERVER_CREDENTIAL_API_KEY).username
        == "server-a:api_key"
    )


def test_keyring_store_uses_namespaced_username_and_supports_get_delete():
    fake = FakeKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)

    store.set_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN, "secret")

    stored_usernames = {
        username
        for service, username in fake.values
        if service == DEFAULT_KEYRING_SERVICE_NAME
    }
    assert any(
        "tldw_chatbook.server_credentials" in username for username in stored_usernames
    )
    assert any("profile=server-a" in username for username in stored_usernames)
    assert any("type=access_token" in username for username in stored_usernames)
    assert store.get_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN) == "secret"

    store.delete_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN)

    assert store.get_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN) is None


def test_keyring_records_use_listable_chatbook_namespace():
    fake = FakeKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)

    store.set_secret(
        "https://server.example.com/api", SERVER_CREDENTIAL_ACCESS_TOKEN, "secret"
    )

    stored_usernames = {
        username
        for service, username in fake.values
        if service == DEFAULT_KEYRING_SERVICE_NAME
    }
    assert "__credential_refs__" in stored_usernames
    assert any(
        "tldw_chatbook.server_credentials" in username for username in stored_usernames
    )
    assert any(
        "profile=https%3A%2F%2Fserver.example.com%2Fapi" in username
        for username in stored_usernames
    )
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


class CountingKeyring(FakeKeyring):
    def __init__(self) -> None:
        super().__init__()
        self.reads = 0

    def get_password(self, service_name: str, username: str) -> str | None:
        self.reads += 1
        return super().get_password(service_name, username)


def test_repeated_secret_reads_share_one_keyring_round_trip() -> None:
    """TASK-32922: every server API call re-read the keyring on the UI loop.

    ``build_client`` resolves the token before its client cache can hit, so
    each call was a SecretService D-Bus round trip on Linux.
    """
    fake = CountingKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    scope = ServerCredentialScope(
        server_profile_id="p1",
        normalized_origin="https://server.example",
        credential_type=SERVER_CREDENTIAL_API_KEY,
    )
    store.set_scoped_secret(scope, "secret-1")
    fake.reads = 0

    for _ in range(10):
        assert store.get_scoped_secret(scope) == "secret-1"
    assert fake.reads == 1


def test_writes_and_deletes_are_visible_immediately() -> None:
    fake = CountingKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    scope = ServerCredentialScope(
        server_profile_id="p1",
        normalized_origin="https://server.example",
        credential_type=SERVER_CREDENTIAL_API_KEY,
    )
    store.set_scoped_secret(scope, "secret-1")
    assert store.get_scoped_secret(scope) == "secret-1"

    store.set_scoped_secret(scope, "secret-2")
    assert store.get_scoped_secret(scope) == "secret-2"

    store.delete_scoped_secret(scope)
    assert store.get_scoped_secret(scope) is None

    store.set_scoped_secret(scope, "secret-3")
    store.clear_all()
    assert store.get_scoped_secret(scope) is None


def test_slow_secret_read_is_cached_from_when_it_returned(monkeypatch) -> None:
    """Qodo review on #2820: a read slower than the TTL (unlock prompt) was
    stored already expired, so the next API call read the keyring again."""
    from tldw_chatbook.runtime_policy import server_credentials

    clock = [100.0]
    monkeypatch.setattr(server_credentials.time, "monotonic", lambda: clock[0])
    fake = CountingKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    scope = ServerCredentialScope(
        server_profile_id="p1",
        normalized_origin="https://server.example",
        credential_type=SERVER_CREDENTIAL_API_KEY,
    )
    store.set_scoped_secret(scope, "secret-1")
    real_get = fake.get_password

    def slow(service_name: str, username: str):
        clock[0] += 45  # an unlock prompt longer than the TTL
        return real_get(service_name, username)

    fake.get_password = slow
    fake.reads = 0
    assert store.get_scoped_secret(scope) == "secret-1"
    assert store.get_scoped_secret(scope) == "secret-1"
    assert fake.reads == 1


def test_externally_rotated_secret_is_picked_up_within_seconds(monkeypatch) -> None:
    """Qodo review on #2820: tldw_server rejects a rotated key immediately, so
    a key rotated by another process must reach this process quickly."""
    from tldw_chatbook.runtime_policy import server_credentials

    clock = [100.0]
    monkeypatch.setattr(server_credentials.time, "monotonic", lambda: clock[0])
    fake = CountingKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    scope = ServerCredentialScope(
        server_profile_id="p1",
        normalized_origin="https://server.example",
        credential_type=SERVER_CREDENTIAL_API_KEY,
    )
    store.set_scoped_secret(scope, "old-key")
    assert store.get_scoped_secret(scope) == "old-key"
    other_process = KeyringServerCredentialStore(keyring_backend=fake)
    other_process.set_scoped_secret(scope, "new-key")

    clock[0] += 5.1
    assert store.get_scoped_secret(scope) == "new-key"
