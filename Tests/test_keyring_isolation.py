"""The test suite never reaches the developer's real OS credential store (TASK-19570 A).

Background: `TldwCli.__init__` wires a server context provider unconditionally --
`app.py:_wire_server_context_provider` -> `build_default_server_credential_store()` ->
`keyring.get_keyring()` (`runtime_policy/server_credentials.py`). There is no test seam
on that path: `Tests/UI/app_factory.py` contains no `keyring` string at all, and
`PYTHON_KEYRING_BACKEND` appeared in zero `conftest.py` files -- its only occurrences
were subprocess env dicts in `Tests/Packaging/`, which isolate a *spawned* process and
not the test session that spawns it.

So every one of the ~620 files that mounts an app was reading the host's credential
subsystem. On macOS a first read can raise a Keychain consent dialog or block on a
locked keychain, and under `timeout_method="thread"` a blocked test kills the entire
run rather than just itself.

A guard that can be removed silently is no guard, so these tests pin the mechanism
(the ambient backend), the seam (what production resolves to), and the end state (what
a constructed app actually holds).
"""

from __future__ import annotations

import os

import pytest

from tldw_chatbook.runtime_policy.server_credentials import (
    CredentialStoreUnavailable,
    build_default_server_credential_store,
)

NULL_BACKEND = "keyring.backends.null.Keyring"


def test_the_ambient_keyring_backend_is_the_null_backend() -> None:
    """The mechanism: the conftest bootstrap pins the backend before any app import."""
    assert os.environ.get("PYTHON_KEYRING_BACKEND") == NULL_BACKEND

    import keyring

    backend = keyring.get_keyring()
    qualified = f"{type(backend).__module__}.{type(backend).__name__}"
    assert qualified == NULL_BACKEND, (
        f"tests are running against the {qualified!r} keyring backend; on a developer "
        "machine that is the real OS credential store"
    )


def test_the_production_credential_seam_resolves_to_no_os_store() -> None:
    """The seam: the exact call `TldwCli.__init__` makes finds nothing secure.

    `_resolve_secure_keyring_backend` admits only backends whose module is on its
    secure list, so the null backend is correctly refused rather than silently
    accepted as a place to read and write secrets.
    """
    with pytest.raises(CredentialStoreUnavailable):
        build_default_server_credential_store()


def test_a_constructed_app_records_the_store_as_unavailable() -> None:
    """The end state: app construction takes the documented fallback, not the OS.

    This is the assertion that would have caught the original gap -- it exercises
    `_wire_server_context_provider` itself rather than the env var it depends on.

    Since TASK-21111(b) the store is resolved lazily, so reading either attribute
    below is what triggers `build_default_server_credential_store()`. That is the
    point of the test, not a hole in it: the fallback decision still has to come
    out the same. `Tests/App/test_startup_init_hygiene.py` pins the complementary
    property -- that construction alone reaches no keyring at all.
    """
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()

    assert app.server_credential_store_unavailable_reason is not None, (
        "app construction found a usable OS credential store during tests -- the "
        "keyring sandbox is not in effect and tests can read/write real credentials"
    )
    assert type(app.server_credential_store).__name__ == "UnavailableServerCredentialStore"
