# Connection Auth Server Switching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate Chatbook's active-server profile, credential, client-construction, token lifecycle, capability snapshot, and server-switch invalidation behavior without creating a second server authority.

**Architecture:** Extend the existing runtime-policy and Unified MCP target seams. `RuntimePolicyContext` remains authoritative for active source/server state, `ActiveServerCapabilityService` remains the capability snapshot seam, and `ConfiguredServerTargetStore` becomes the shared server profile metadata registry through a wrapper/facade rather than a duplicated registry. Add a secure credential store plus a compatibility server client provider so existing `from_app_config` services can migrate in batches.

**Tech Stack:** Python 3.11+, dataclasses, protocols, `keyring`, existing `tldw_chatbook.tldw_api.TLDWAPIClient`, pytest, existing runtime-policy tests.

---

## Scope

Implement only Workstream 1 from `Docs/superpowers/specs/2026-04-28-remaining-server-parity-roadmap-design.md`.

In scope:

- Secure credential-store abstraction.
- Keyring-backed credential store and in-memory test store.
- Active-server context/provider facade over existing runtime-policy and target-store seams.
- Legacy `tldw_api` config compatibility path.
- Token lifecycle persistence for explicit login/refresh/logout flows.
- Server switching invalidation tests.
- Pilot migration for server runtime and auth/account services.
- Migration audit document for remaining server-backed services.

Out of scope:

- Sync/mirror implementation.
- Realtime/event observer implementation.
- UI redesign.
- Workflow routes.
- Big-bang migration of every server service.
- Plaintext secret persistence.

## File Structure

- Create `tldw_chatbook/runtime_policy/server_credentials.py`
  - Defines credential records, credential store protocol, in-memory store, keyring store, purpose constants, and redaction helpers.

- Create `tldw_chatbook/runtime_policy/server_context.py`
  - Defines `ActiveServerContext`, typed errors, `RuntimeServerContextProvider`, client construction from active server context, legacy config fallback, and profile/credential invalidation helpers.

- Modify `tldw_chatbook/runtime_policy/bootstrap.py`
  - Keep existing `build_runtime_api_client()` and `build_runtime_api_client_from_config()` stable.
  - Add compatibility hooks that allow app wiring to use `RuntimeServerContextProvider` without breaking existing service constructors.

- Modify `tldw_chatbook/MCP/server_target_store.py`
  - Add profile-oriented aliases or helpers only if needed.
  - Do not add secret fields to `ConfiguredServerTarget`.
  - Preserve existing MCP tests and behavior.

- Modify `tldw_chatbook/Auth_Account_Interop/auth_account_scope_service.py`
  - Add optional credential/context integration for login, refresh, and logout persistence.
  - Preserve existing method signatures where possible.

- Modify `tldw_chatbook/Auth_Account_Interop/server_auth_account_service.py`
  - Add a constructor/factory path that receives a client provider.
  - Keep existing `from_app_config()` compatibility.

- Modify `tldw_chatbook/Server_Runtime_Interop/server_runtime_service.py`
  - Add a constructor/factory path that receives a client provider.
  - Keep existing `from_app_config()` compatibility.

- Modify `tldw_chatbook/app.py`
  - Instantiate the credential store, runtime server context provider, and pilot migrated services.
  - Keep legacy server config and Unified MCP target store synchronization.

- Modify `tldw_chatbook/runtime_policy/__init__.py`
  - Export new credential and context provider types.

- Create `Tests/RuntimePolicy/test_server_credentials.py`
  - Covers secret isolation, no plaintext metadata leakage, keyring abstraction behavior through fakes, and clearing.

- Create `Tests/RuntimePolicy/test_server_context_provider.py`
  - Covers active-server resolution, legacy fallback, credential priority, client construction, server-switch invalidation, and typed errors.

- Modify `Tests/RuntimePolicy/test_runtime_policy_bootstrap.py`
  - Covers compatibility behavior and no regressions to legacy client creation.

- Modify `Tests/RuntimePolicy/test_active_server_capabilities.py`
  - Covers capability state remains tied to the authoritative runtime-policy active server.

- Modify `Tests/Auth_Account/test_auth_account_scope_service.py`
  - Covers storing login/refresh credentials and clearing on logout.

- Modify `Tests/Server_Runtime/test_server_runtime_service.py`
  - Covers provider-backed client construction.

- Modify `Tests/UI/test_screen_navigation.py`
  - Service-wiring only: assert app exposes the new context provider and credentials store.

- Create `Docs/Development/server-client-provider-migration-audit.md`
  - Lists server-backed service constructors using legacy config, whether migrated, and follow-up priority.

## Task 1: Credential Store

**Files:**

- Create: `tldw_chatbook/runtime_policy/server_credentials.py`
- Create: `Tests/RuntimePolicy/test_server_credentials.py`
- Modify: `tldw_chatbook/runtime_policy/__init__.py`

- [ ] **Step 1: Write failing tests for in-memory credential storage**

Add tests covering set/get/delete/clear by server ID and purpose:

```python
from tldw_chatbook.runtime_policy.server_credentials import (
    InMemoryServerCredentialStore,
    SERVER_CREDENTIAL_ACCESS_TOKEN,
    SERVER_CREDENTIAL_REFRESH_TOKEN,
)


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
```

- [ ] **Step 2: Write failing tests for redaction**

```python
from tldw_chatbook.runtime_policy.server_credentials import redact_secret


def test_redact_secret_never_returns_original_secret():
    assert redact_secret("abcdef123456") != "abcdef123456"
    assert redact_secret("abcdef123456").startswith("ab")
    assert "3456" in redact_secret("abcdef123456")


def test_redact_secret_handles_empty_values():
    assert redact_secret(None) == "<unset>"
    assert redact_secret("") == "<unset>"
```

- [ ] **Step 3: Write failing tests for keyring adapter using a fake keyring object**

```python
from tldw_chatbook.runtime_policy.server_credentials import KeyringServerCredentialStore


class FakeKeyring:
    def __init__(self):
        self.values = {}

    def set_password(self, service_name, username, password):
        self.values[(service_name, username)] = password

    def get_password(self, service_name, username):
        return self.values.get((service_name, username))

    def delete_password(self, service_name, username):
        self.values.pop((service_name, username), None)


def test_keyring_store_uses_server_and_purpose_as_username():
    fake = FakeKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)

    store.set_secret("server-a", "access_token", "secret")

    assert fake.values[("tldw_chatbook.server_credentials", "server-a:access_token")] == "secret"
    assert store.get_secret("server-a", "access_token") == "secret"
```

- [ ] **Step 4: Run tests and verify they fail**

Run:

```bash
python -m pytest Tests/RuntimePolicy/test_server_credentials.py -q
```

Expected: fails because `server_credentials.py` does not exist.

- [ ] **Step 5: Implement `server_credentials.py`**

Use this minimal structure:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

SERVER_CREDENTIAL_ACCESS_TOKEN = "access_token"
SERVER_CREDENTIAL_REFRESH_TOKEN = "refresh_token"
SERVER_CREDENTIAL_API_KEY = "api_key"
SERVER_CREDENTIAL_BEARER_TOKEN = "bearer_token"
DEFAULT_KEYRING_SERVICE_NAME = "tldw_chatbook.server_credentials"


class ServerCredentialStore(Protocol):
    def set_secret(self, server_id: str, purpose: str, secret: str) -> None: ...
    def get_secret(self, server_id: str, purpose: str) -> str | None: ...
    def delete_secret(self, server_id: str, purpose: str) -> None: ...
    def clear_server(self, server_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class ServerCredentialRef:
    server_id: str
    purpose: str

    @property
    def username(self) -> str:
        return f"{self.server_id}:{self.purpose}"


def redact_secret(secret: str | None) -> str:
    if not secret:
        return "<unset>"
    if len(secret) <= 8:
        return "<redacted>"
    return f"{secret[:2]}...{secret[-4:]}"


class InMemoryServerCredentialStore:
    def __init__(self) -> None:
        self._values: dict[tuple[str, str], str] = {}

    def set_secret(self, server_id: str, purpose: str, secret: str) -> None:
        self._values[(_normalize(server_id), _normalize(purpose))] = str(secret)

    def get_secret(self, server_id: str, purpose: str) -> str | None:
        return self._values.get((_normalize(server_id), _normalize(purpose)))

    def delete_secret(self, server_id: str, purpose: str) -> None:
        self._values.pop((_normalize(server_id), _normalize(purpose)), None)

    def clear_server(self, server_id: str) -> None:
        normalized_server_id = _normalize(server_id)
        for key in list(self._values):
            if key[0] == normalized_server_id:
                self._values.pop(key, None)


class KeyringServerCredentialStore:
    def __init__(self, *, service_name: str = DEFAULT_KEYRING_SERVICE_NAME, keyring_backend=None) -> None:
        if keyring_backend is None:
            import keyring as keyring_backend
            self._password_delete_error = keyring_backend.errors.PasswordDeleteError
        else:
            self._password_delete_error = getattr(
                getattr(keyring_backend, "errors", None),
                "PasswordDeleteError",
                None,
            )
        self.service_name = service_name
        self.keyring = keyring_backend

    def set_secret(self, server_id: str, purpose: str, secret: str) -> None:
        self.keyring.set_password(self.service_name, ServerCredentialRef(_normalize(server_id), _normalize(purpose)).username, str(secret))

    def get_secret(self, server_id: str, purpose: str) -> str | None:
        return self.keyring.get_password(self.service_name, ServerCredentialRef(_normalize(server_id), _normalize(purpose)).username)

    def delete_secret(self, server_id: str, purpose: str) -> None:
        try:
            self.keyring.delete_password(self.service_name, ServerCredentialRef(_normalize(server_id), _normalize(purpose)).username)
        except Exception as exc:
            if self._password_delete_error is not None and isinstance(exc, self._password_delete_error):
                return
            raise

    def clear_server(self, server_id: str) -> None:
        for purpose in (
            SERVER_CREDENTIAL_ACCESS_TOKEN,
            SERVER_CREDENTIAL_REFRESH_TOKEN,
            SERVER_CREDENTIAL_API_KEY,
            SERVER_CREDENTIAL_BEARER_TOKEN,
        ):
            self.delete_secret(server_id, purpose)


def _normalize(value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError("server credential identifiers cannot be empty")
    return normalized
```

- [ ] **Step 6: Export new types**

Modify `tldw_chatbook/runtime_policy/__init__.py` to export:

```python
from .server_credentials import (
    DEFAULT_KEYRING_SERVICE_NAME,
    InMemoryServerCredentialStore,
    KeyringServerCredentialStore,
    ServerCredentialStore,
    redact_secret,
)
```

- [ ] **Step 7: Run tests**

Run:

```bash
python -m pytest Tests/RuntimePolicy/test_server_credentials.py -q
```

Expected: pass.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/runtime_policy/server_credentials.py tldw_chatbook/runtime_policy/__init__.py Tests/RuntimePolicy/test_server_credentials.py
git commit -m "Add secure server credential store"
```

## Task 2: Active Server Context Provider

**Files:**

- Create: `tldw_chatbook/runtime_policy/server_context.py`
- Create: `Tests/RuntimePolicy/test_server_context_provider.py`
- Modify: `tldw_chatbook/runtime_policy/__init__.py`

- [ ] **Step 1: Write failing tests for active server resolution**

```python
from dataclasses import replace

import pytest

from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore
from tldw_chatbook.MCP.unified_control_models import ConfiguredServerTarget
from tldw_chatbook.runtime_policy.bootstrap import RuntimePolicyContext
from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.runtime_policy.server_context import RuntimeServerContextProvider, ServerContextUnavailable
from tldw_chatbook.runtime_policy.server_credentials import InMemoryServerCredentialStore, SERVER_CREDENTIAL_ACCESS_TOKEN


def _context(tmp_path, active_server_id="http://server.test"):
    store = RuntimeSourceStateStore(tmp_path / "runtime_policy.json")
    state = RuntimeSourceState(active_source="server", active_server_id=active_server_id, server_configured=True)
    return RuntimePolicyContext(state=state, store=store)


def test_context_provider_resolves_default_target_and_keyring_secret(tmp_path):
    targets = ConfiguredServerTargetStore(tmp_path / "targets.json")
    targets.save_targets([
        ConfiguredServerTarget(
            server_id="http://server.test",
            label="Test server",
            base_url="http://server.test",
            auth_mode="bearer",
            auth_reference="keyring:access_token",
            is_default=True,
        )
    ])
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret("http://server.test", SERVER_CREDENTIAL_ACCESS_TOKEN, "access-1")
    provider = RuntimeServerContextProvider(
        runtime_context=_context(tmp_path),
        target_store=targets,
        credential_store=credentials,
        app_config={},
    )

    resolved = provider.get_active_context()

    assert resolved.active_server_id == "http://server.test"
    assert resolved.base_url == "http://server.test"
    assert resolved.auth_method == "bearer"
    assert resolved.auth_token == "access-1"


def test_context_provider_rejects_server_mode_without_active_server(tmp_path):
    provider = RuntimeServerContextProvider(
        runtime_context=RuntimePolicyContext(
            state=RuntimeSourceState(active_source="server", active_server_id=None, server_configured=False),
            store=RuntimeSourceStateStore(tmp_path / "runtime_policy.json"),
        ),
        target_store=ConfiguredServerTargetStore(tmp_path / "targets.json"),
        credential_store=InMemoryServerCredentialStore(),
        app_config={},
    )

    with pytest.raises(ServerContextUnavailable):
        provider.get_active_context()
```

- [ ] **Step 2: Write failing tests for legacy fallback**

```python
def test_context_provider_can_build_from_legacy_config_when_no_target_exists(tmp_path):
    provider = RuntimeServerContextProvider(
        runtime_context=_context(tmp_path, active_server_id="http://legacy.test"),
        target_store=ConfiguredServerTargetStore(tmp_path / "targets.json"),
        credential_store=InMemoryServerCredentialStore(),
        app_config={"tldw_api": {"base_url": "http://legacy.test", "api_key": "legacy-key"}},
    )

    resolved = provider.get_active_context()

    assert resolved.active_server_id == "http://legacy.test"
    assert resolved.base_url == "http://legacy.test"
    assert resolved.auth_method == "api_key"
    assert resolved.auth_token == "legacy-key"
    assert resolved.credential_source == "legacy_config"
```

- [ ] **Step 3: Write failing tests for client construction**

```python
def test_context_provider_builds_runtime_api_client_from_active_context(tmp_path):
    provider = RuntimeServerContextProvider(
        runtime_context=_context(tmp_path, active_server_id="http://legacy.test"),
        target_store=ConfiguredServerTargetStore(tmp_path / "targets.json"),
        credential_store=InMemoryServerCredentialStore(),
        app_config={"tldw_api": {"base_url": "http://legacy.test", "bearer_token": "bearer-1", "auth_mode": "bearer"}},
    )

    client = provider.build_client()

    assert str(client.base_url).rstrip("/") == "http://legacy.test"
    assert client.bearer_token == "bearer-1"
```

- [ ] **Step 4: Run tests and verify they fail**

```bash
python -m pytest Tests/RuntimePolicy/test_server_context_provider.py -q
```

Expected: fails because provider does not exist.

- [ ] **Step 5: Implement `server_context.py`**

Implement these public types:

```python
@dataclass(frozen=True, slots=True)
class ActiveServerContext:
    active_server_id: str
    label: str | None
    base_url: str
    auth_method: str
    auth_token: str | None
    credential_source: str


class ServerContextUnavailable(RuntimeError):
    reason_code = "server_context_unavailable"


class ServerCredentialsUnavailable(RuntimeError):
    reason_code = "server_credentials_unavailable"
```

Implement `RuntimeServerContextProvider` with:

- `get_active_context() -> ActiveServerContext`
- `build_client() -> TLDWAPIClient`
- `clear_active_server_credentials() -> None`
- `clear_server_credentials(server_id: str) -> None`
- `resolve_target() -> ConfiguredServerTarget | None`
- private legacy fallback helper using `derive_configured_server_binding()` and existing `build_runtime_api_client()` logic.

Credential precedence:

1. Keyring/injected credential store for the active server and requested purpose.
2. Legacy config token if target `auth_reference` is `legacy:tldw_api` or no profile target exists.
3. No token.

Do not write credentials into `ConfiguredServerTargetStore`.

- [ ] **Step 6: Export provider types**

Modify `tldw_chatbook/runtime_policy/__init__.py` to export:

```python
from .server_context import (
    ActiveServerContext,
    RuntimeServerContextProvider,
    ServerContextUnavailable,
    ServerCredentialsUnavailable,
)
```

- [ ] **Step 7: Run tests**

```bash
python -m pytest Tests/RuntimePolicy/test_server_context_provider.py Tests/RuntimePolicy/test_server_credentials.py -q
```

Expected: pass.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/runtime_policy/server_context.py tldw_chatbook/runtime_policy/__init__.py Tests/RuntimePolicy/test_server_context_provider.py
git commit -m "Add active server context provider"
```

## Task 3: App Wiring And Capability Snapshot Invalidation

**Files:**

- Modify: `tldw_chatbook/app.py`
- Modify: `tldw_chatbook/runtime_policy/bootstrap.py`
- Modify: `Tests/UI/test_screen_navigation.py`
- Modify: `Tests/RuntimePolicy/test_active_server_capabilities.py`
- Modify: `Tests/RuntimePolicy/test_runtime_policy_bootstrap.py`

- [ ] **Step 1: Write failing app-wiring assertions**

In `Tests/UI/test_screen_navigation.py`, extend the existing service wiring test:

```python
from tldw_chatbook.runtime_policy import (
    KeyringServerCredentialStore,
    RuntimeServerContextProvider,
)


def test_app_wires_server_context_provider(app):
    assert isinstance(app.server_credential_store, KeyringServerCredentialStore)
    assert isinstance(app.server_context_provider, RuntimeServerContextProvider)
    assert app.server_context_provider.runtime_context is app.runtime_policy
    assert app.server_context_provider.target_store is app.unified_mcp_target_store
```

If the fixture does not expose `app` directly, add these assertions to the existing navigation/app wiring test that currently checks `active_server_capability_service`.

- [ ] **Step 2: Write failing bootstrap compatibility tests**

In `Tests/RuntimePolicy/test_runtime_policy_bootstrap.py`, add:

```python
def test_set_authoritative_runtime_source_clears_probe_state_on_server_identity_change(tmp_path):
    # Use existing helpers in this test file where possible.
    # Assert reachability/auth probe fields reset when app_config tldw_api base_url changes.
```

Use existing test patterns in the file. Expected behavior: no new active-server authority; `RuntimeSourceState.active_server_id` changes and probe fields clear.

- [ ] **Step 3: Wire provider in `app.py`**

In app initialization near current runtime-policy and Unified MCP wiring:

```python
self.server_credential_store = KeyringServerCredentialStore()
self.server_context_provider = RuntimeServerContextProvider(
    runtime_context=self.runtime_policy,
    target_store=self.unified_mcp_target_store,
    credential_store=self.server_credential_store,
    app_config=self.app_config,
)
```

Use `InMemoryServerCredentialStore` only in tests if dependency injection is already available. Do not store tokens in app config.

- [ ] **Step 4: Ensure target-store synchronization still happens once**

Keep existing `ConfiguredServerTargetStore.upsert_legacy_config_target(self.app_config)` behavior. The provider should consume that target store rather than adding another target store.

- [ ] **Step 5: Run focused tests**

```bash
python -m pytest Tests/UI/test_screen_navigation.py Tests/RuntimePolicy/test_runtime_policy_bootstrap.py Tests/RuntimePolicy/test_active_server_capabilities.py -q
```

Expected: pass. Ignore unrelated UI snapshot/layout failures if the file has broader UI tests; if so, run only the service-wiring test node.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/app.py tldw_chatbook/runtime_policy/bootstrap.py Tests/UI/test_screen_navigation.py Tests/RuntimePolicy/test_runtime_policy_bootstrap.py Tests/RuntimePolicy/test_active_server_capabilities.py
git commit -m "Wire active server context provider"
```

## Task 4: Token Lifecycle Integration

**Files:**

- Modify: `tldw_chatbook/Auth_Account_Interop/auth_account_scope_service.py`
- Modify: `tldw_chatbook/Auth_Account_Interop/server_auth_account_service.py`
- Modify: `Tests/Auth_Account/test_auth_account_scope_service.py`
- Modify: `Tests/tldw_api/test_auth_user_client.py` only if existing client behavior changes.

- [ ] **Step 1: Write failing tests for login credential persistence**

Add to `Tests/Auth_Account/test_auth_account_scope_service.py`:

```python
from tldw_chatbook.runtime_policy.server_credentials import (
    InMemoryServerCredentialStore,
    SERVER_CREDENTIAL_ACCESS_TOKEN,
    SERVER_CREDENTIAL_REFRESH_TOKEN,
)


class FakeServerContextProvider:
    active_server_id = "http://server.test"

    def __init__(self):
        self.credential_store = InMemoryServerCredentialStore()

    def store_auth_tokens(self, *, access_token=None, refresh_token=None):
        if access_token:
            self.credential_store.set_secret(self.active_server_id, SERVER_CREDENTIAL_ACCESS_TOKEN, access_token)
        if refresh_token:
            self.credential_store.set_secret(self.active_server_id, SERVER_CREDENTIAL_REFRESH_TOKEN, refresh_token)

    def clear_active_server_credentials(self):
        self.credential_store.clear_server(self.active_server_id)


async def test_login_persists_tokens_when_context_provider_is_available():
    class FakeTokenAuthAccountService(FakeAuthAccountService):
        async def login(self, **kwargs):
            self.calls.append(("login", kwargs))
            return {"access_token": "access-1", "refresh_token": "refresh-1", "token_type": "bearer"}

    server = FakeTokenAuthAccountService()
    provider = FakeServerContextProvider()
    scope = AuthAccountScopeService(server_service=server, server_context_provider=provider)

    await scope.login(username="ada@example.com", password="secret")

    assert provider.credential_store.get_secret("http://server.test", SERVER_CREDENTIAL_ACCESS_TOKEN) == "access-1"
    assert provider.credential_store.get_secret("http://server.test", SERVER_CREDENTIAL_REFRESH_TOKEN) == "refresh-1"
```

- [ ] **Step 2: Write failing tests for logout clearing**

```python
async def test_logout_clears_active_server_credentials_when_requested():
    class FakeLogoutAuthAccountService(FakeAuthAccountService):
        async def logout(self, **kwargs):
            self.calls.append(("logout", kwargs))
            return {"detail": "logged out"}

    server = FakeLogoutAuthAccountService()
    provider = FakeServerContextProvider()
    provider.store_auth_tokens(access_token="access-1", refresh_token="refresh-1")
    scope = AuthAccountScopeService(server_service=server, server_context_provider=provider)

    await scope.logout(clear_bearer_token=True)

    assert provider.credential_store.get_secret("http://server.test", SERVER_CREDENTIAL_ACCESS_TOKEN) is None
    assert provider.credential_store.get_secret("http://server.test", SERVER_CREDENTIAL_REFRESH_TOKEN) is None
```

- [ ] **Step 3: Implement context integration in `AuthAccountScopeService`**

Add optional constructor argument:

```python
def __init__(self, *, server_service: Any = None, policy_enforcer: Any = None, server_context_provider: Any = None):
    self.server_service = server_service
    self.policy_enforcer = policy_enforcer
    self.server_context_provider = server_context_provider
```

After successful `login()` and `refresh_auth_token()`, if result contains `access_token` or `refresh_token`, call:

```python
store = getattr(self.server_context_provider, "store_auth_tokens", None)
if callable(store):
    store(access_token=result.get("access_token"), refresh_token=result.get("refresh_token"))
```

After successful `logout(clear_bearer_token=True)`, call:

```python
clear = getattr(self.server_context_provider, "clear_active_server_credentials", None)
if clear_bearer_token and callable(clear):
    clear()
```

Do not persist username/password.

- [ ] **Step 4: Add provider helper methods**

In `RuntimeServerContextProvider`, add:

```python
def store_auth_tokens(self, *, access_token: str | None = None, refresh_token: str | None = None) -> None:
    context = self.get_active_context()
    if access_token:
        self.credential_store.set_secret(context.active_server_id, SERVER_CREDENTIAL_ACCESS_TOKEN, access_token)
    if refresh_token:
        self.credential_store.set_secret(context.active_server_id, SERVER_CREDENTIAL_REFRESH_TOKEN, refresh_token)
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest Tests/Auth_Account/test_auth_account_scope_service.py Tests/RuntimePolicy/test_server_context_provider.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Auth_Account_Interop/auth_account_scope_service.py tldw_chatbook/runtime_policy/server_context.py Tests/Auth_Account/test_auth_account_scope_service.py Tests/RuntimePolicy/test_server_context_provider.py
git commit -m "Persist active server auth tokens securely"
```

## Task 5: Provider-Backed Service Pilot

**Files:**

- Modify: `tldw_chatbook/Auth_Account_Interop/server_auth_account_service.py`
- Modify: `tldw_chatbook/Server_Runtime_Interop/server_runtime_service.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/Auth_Account/test_auth_account_scope_service.py`
- Modify: `Tests/Server_Runtime/test_server_runtime_service.py`
- Modify: `Tests/UI/test_screen_navigation.py`

- [ ] **Step 1: Write failing tests for provider-backed runtime service**

In `Tests/Server_Runtime/test_server_runtime_service.py`:

```python
class FakeProvider:
    def __init__(self, client):
        self.client = client
        self.calls = 0

    def build_client(self):
        self.calls += 1
        return self.client


async def test_server_runtime_service_can_use_context_provider_client():
    fake_client = FakeServerRuntimeClient()
    provider = FakeProvider(fake_client)
    service = ServerRuntimeService.from_server_context_provider(provider)

    await service.get_health()

    assert provider.calls == 1
    assert fake_client.calls[0] == ("get_server_health",)
```

- [ ] **Step 2: Add factory support**

For both `ServerRuntimeService` and `ServerAuthAccountService`, preserve `__init__(client=...)` and `from_app_config(...)`, then add:

```python
@classmethod
def from_server_context_provider(cls, provider: Any, *, policy_enforcer: Any = None) -> "ServerRuntimeService":
    return cls(client_provider=provider, policy_enforcer=policy_enforcer)
```

Update `_require_client()`:

```python
if self.client is not None:
    return self.client
if self.client_provider is not None:
    return self.client_provider.build_client()
raise ValueError("Server backend is unavailable.")
```

Use the class name's existing constructor style and do not break tests that pass fake clients directly.

- [ ] **Step 3: Wire pilot services through provider in `app.py`**

For server runtime and auth/account only:

```python
self.server_runtime_service = ServerRuntimeService.from_server_context_provider(
    self.server_context_provider,
    policy_enforcer=self.service_policy_enforcer,
)
self.server_auth_account_service = ServerAuthAccountService.from_server_context_provider(
    self.server_context_provider,
    policy_enforcer=self.service_policy_enforcer,
)
self.auth_account_scope_service = AuthAccountScopeService(
    server_service=self.server_auth_account_service,
    policy_enforcer=self.service_policy_enforcer,
    server_context_provider=self.server_context_provider,
)
```

Adjust exact names to match existing app wiring. Do not migrate every service in this task.

- [ ] **Step 4: Run focused tests**

```bash
python -m pytest Tests/Server_Runtime/test_server_runtime_service.py Tests/Auth_Account/test_auth_account_scope_service.py Tests/UI/test_screen_navigation.py -q
```

Expected: pass or, if UI tests are noisy, run the exact app-wiring node only.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Auth_Account_Interop/server_auth_account_service.py tldw_chatbook/Server_Runtime_Interop/server_runtime_service.py tldw_chatbook/app.py Tests/Auth_Account/test_auth_account_scope_service.py Tests/Server_Runtime/test_server_runtime_service.py Tests/UI/test_screen_navigation.py
git commit -m "Pilot server services on active context provider"
```

## Task 6: Migration Audit And Guardrails

**Files:**

- Create: `Docs/Development/server-client-provider-migration-audit.md`
- Modify: `Tests/RuntimePolicy/test_server_context_provider.py`

- [ ] **Step 1: Generate service audit list**

Run:

```bash
rg -n "build_runtime_api_client_from_config|build_runtime_api_client\\(" tldw_chatbook > /tmp/server-client-builder-audit.txt
```

Expected: list of current server-backed services that still use direct config construction.

- [ ] **Step 2: Create migration audit doc**

Create `Docs/Development/server-client-provider-migration-audit.md` with this structure:

```markdown
# Server Client Provider Migration Audit

Date: 2026-04-28

Purpose: Track migration from direct legacy `tldw_api` config client construction to `RuntimeServerContextProvider`.

## Authoritative Seams

- `RuntimePolicyContext` owns active source/server state.
- `ConfiguredServerTargetStore` owns server profile metadata.
- `RuntimeServerContextProvider` builds credential-bound clients.
- `ActiveServerCapabilityService` owns capability snapshots.

## Migrated In This Tranche

| Service | File | Status | Notes |
| --- | --- | --- | --- |
| ServerRuntimeService | `tldw_chatbook/Server_Runtime_Interop/server_runtime_service.py` | migrated | Pilot provider-backed service. |
| ServerAuthAccountService | `tldw_chatbook/Auth_Account_Interop/server_auth_account_service.py` | migrated | Required for token lifecycle. |

## Compatibility Mode Remaining

| Service | File | Status | Priority |
| --- | --- | --- | --- |
| ServerChatConversationService | `tldw_chatbook/Chat/server_chat_conversation_service.py` | legacy-compatible | high |
| ServerMediaReadingService | `tldw_chatbook/Media/server_media_reading_service.py` | legacy-compatible | high |
| ServerNotesWorkspaceService | `tldw_chatbook/Notes/server_notes_workspace_service.py` | legacy-compatible | high |
| ServerWritingService | `tldw_chatbook/Writing_Interop/server_writing_service.py` | legacy-compatible | high |
| ServerResearchService | `tldw_chatbook/Research_Interop/server_research_service.py` | legacy-compatible | high |
```

Add the rest of the `rg` output grouped as high/medium/low. This doc is allowed to be approximate but must not omit known direct builders.

- [ ] **Step 3: Add a provider no-secret-regression test**

In `Tests/RuntimePolicy/test_server_context_provider.py`, add:

```python
def test_provider_does_not_write_secret_material_to_target_store(tmp_path):
    # Store a token in credential_store, resolve/build client, reload target JSON, assert token value is absent.
```

Use `path.read_text()` and assert the literal secret is not present.

- [ ] **Step 4: Run verification**

```bash
python -m pytest Tests/RuntimePolicy/test_server_context_provider.py -q
git diff --check
```

Expected: pass and no whitespace errors.

- [ ] **Step 5: Commit**

```bash
git add Docs/Development/server-client-provider-migration-audit.md Tests/RuntimePolicy/test_server_context_provider.py
git commit -m "Document server client provider migration"
```

## Task 7: Final Verification

**Files:**

- No new files unless failures require fixes.

- [ ] **Step 1: Run focused runtime/auth/server tests**

```bash
python -m pytest \
  Tests/RuntimePolicy/test_server_credentials.py \
  Tests/RuntimePolicy/test_server_context_provider.py \
  Tests/RuntimePolicy/test_runtime_policy_bootstrap.py \
  Tests/RuntimePolicy/test_active_server_capabilities.py \
  Tests/Auth_Account/test_auth_account_scope_service.py \
  Tests/Server_Runtime/test_server_runtime_service.py \
  Tests/MCP/test_server_target_store.py \
  -q
```

Expected: pass.

- [ ] **Step 2: Run app wiring service test only**

Run the exact app-wiring test node from `Tests/UI/test_screen_navigation.py` that checks backend/service construction.

Expected: pass. Ignore broader UI tests if they fail due to pre-existing UI breakage.

- [ ] **Step 3: Compile touched packages**

```bash
python -m compileall \
  tldw_chatbook/runtime_policy \
  tldw_chatbook/Auth_Account_Interop \
  tldw_chatbook/Server_Runtime_Interop
```

Expected: no compile errors.

- [ ] **Step 4: Check diff hygiene**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intended files changed.

- [ ] **Step 5: Commit any final fixes**

```bash
git add <changed-files>
git commit -m "Harden active server connection foundation"
```

Skip this commit if no final fixes are needed.

## Implementation Notes

- Do not add a new active-server state file. Use existing runtime-policy state and target store.
- Do not add secret fields to `ConfiguredServerTarget`.
- Do not write secrets into docs, logs, test snapshots, app config, target-store JSON, or unsupported-capability reports.
- Keep legacy config client construction working until each service is migrated.
- Prefer small constructor additions over large refactors in server service classes.
- Keep UI changes to wiring assertions only.
- If keyring is unavailable at runtime, surface a typed credential-store unavailable error or use explicit legacy compatibility; do not silently downgrade to plaintext storage.

## Follow-Up Plans

After this plan lands, create separate plans for:

- Realtime/event observation and notification dedupe/cursors.
- Sync/mirror identity map and dry-run mode.
- Domain-by-domain migration from legacy config builders to `RuntimeServerContextProvider`.
- UI/UX handoff view-model contracts.
