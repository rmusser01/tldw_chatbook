# Connection Auth Foundation Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining Tranche 1 connection/auth contract gaps so Chatbook has one safe active-server, credential, provider, and capability foundation for later event/sync/domain work.

**Architecture:** Extend the existing runtime-policy seams instead of replacing them. `RuntimePolicyContext` remains the active-server authority, `ConfiguredServerTargetStore` remains the persisted profile metadata store, `RuntimeServerContextProvider` remains the credential-bound client provider, and `ActiveServerCapabilityService` remains the capability snapshot authority. This plan hardens credentials, typed failures, invalidation hooks, and UX-facing service contracts around those existing seams.

**Tech Stack:** Python 3.11+, dataclasses, protocols, `keyring`, existing `tldw_chatbook.tldw_api.TLDWAPIClient`, pytest, pytest-asyncio, existing runtime-policy tests.

---

## Scope

Implement only Tranche 1 hardening from `Docs/superpowers/specs/2026-04-29-backend-server-parity-handoff-roadmap-design.md`.

In scope:

- Cross-platform credential-store availability and no-plaintext fallback behavior.
- Stable, listable Chatbook credential namespace records for reliable global sign-out.
- Backward-compatible credential APIs preserved as provider-backed adapters.
- Typed, sanitized auth/context failures for UX and services.
- Active-server switch invalidation hooks for provider clients, capability snapshots, and future event/sync handles.
- Capability status alignment between runtime policy and target-store last-known status.
- Tranche 1 UX handoff contract schemas and service-level contract tests.
- Migration-audit guard that blocks new raw `tldw_api` client builders outside audited compatibility seams.

Out of scope:

- Realtime/event observer implementation.
- Sync/mirror identity maps or dry-run reports.
- Workflow implementation.
- UI redesign.
- UI tests as blockers unless they validate service wiring or contract behavior.
- Big-bang migration of every server-backed service.

## Existing Seams To Preserve

- `tldw_chatbook/runtime_policy/bootstrap.py`
  - Owns runtime-policy loading, active-source application, and legacy runtime client builders.

- `tldw_chatbook/runtime_policy/server_context.py`
  - Owns active server context resolution, credential lookup, provider-built client caching, and credential invalidation helpers.

- `tldw_chatbook/runtime_policy/server_credentials.py`
  - Owns credential-store protocol, in-memory fake, keyring-backed store, purpose constants, and redaction helpers.

- `tldw_chatbook/runtime_policy/server_capabilities.py`
  - Owns active-server capability snapshot refresh and runtime-policy reachability/auth state updates.

- `tldw_chatbook/MCP/server_target_store.py`
  - Owns persisted server target/profile metadata. It must not persist secrets.

- `tldw_chatbook/app.py`
  - Owns app startup wiring for the target store, credential store, context provider, and capability service.

## File Structure

- Modify `tldw_chatbook/runtime_policy/server_credentials.py`
  - Add `ServerCredentialScope`, richer credential refs, secure-backend availability errors, namespace-prefix enumeration behavior, and compatibility adapters for existing `server_id/purpose` calls.

- Modify `tldw_chatbook/runtime_policy/server_context.py`
  - Add typed unavailable/auth failure models, sanitized error metadata, active auth context fields, and explicit invalidation entrypoints.

- Modify `tldw_chatbook/runtime_policy/types.py`
  - Extend `ServerAuthState` only if needed for `credential_store_unavailable`, `unauthorized`, or `stale_authorization`; otherwise keep runtime state coarse and expose detailed reason codes through context/capability contracts.

- Modify `tldw_chatbook/runtime_policy/server_capabilities.py`
  - Persist capability refresh status back to `ConfiguredServerTargetStore` through an optional target-store dependency or explicit callback.

- Modify `tldw_chatbook/runtime_policy/__init__.py`
  - Export new credential/error/contract types.

- Modify `tldw_chatbook/MCP/server_target_store.py`
  - Add only status helper coverage needed by `ActiveServerCapabilityService`. Do not add secrets.

- Modify `tldw_chatbook/app.py`
  - Use a credential-store factory instead of direct `KeyringServerCredentialStore()` construction.
  - Route runtime backend/server switches through one invalidation hook.
  - Keep the app-wiring change localized to `_wire_server_context_provider()`, `handle_runtime_backend_changed()`, and cached-client shutdown.

- Create `tldw_chatbook/UX_Interop/server_connection_contracts.py`
  - Versioned typed contracts for active server status, auth state, credential-store unavailable, server switching invalidation, and capability status.

- Modify `tldw_chatbook/UX_Interop/server_parity_contracts.py`
  - Re-export or aggregate the connection/auth contracts for the UX handoff packet.

- Modify `Tests/RuntimePolicy/test_server_credentials.py`
  - Add namespace, backend-availability, no-plaintext fallback, orphan cleanup, and redaction tests.

- Modify `Tests/RuntimePolicy/test_server_credentials_lane_a.py`
  - Update lane-specific credential tests for listable namespace behavior.

- Modify `Tests/RuntimePolicy/test_server_context_provider.py`
  - Add typed failure, credential-store unavailable, active principal/scope, and invalidation tests.

- Modify `Tests/RuntimePolicy/test_active_server_capabilities.py`
  - Add target-store status alignment and sanitized failure tests.

- Modify `Tests/RuntimePolicy/test_runtime_policy_bootstrap.py`
  - Add app-wiring tests for credential-store factory and backend switch invalidation.

- Modify `Tests/RuntimePolicy/test_server_client_provider_migration_audit.py`
  - Add guard coverage for the updated allowed raw-client-builder baseline.

- Create `Tests/UX_Interop/test_server_connection_contracts.py`
  - Pin UX handoff contract schema payloads without depending on current UI screens.

- Modify `Docs/Development/server-client-provider-migration-audit.md`
  - Record the hardening work, remaining compatibility holdouts, and the exact semantic audit key used for raw builders.

## Task 1: Harden Credential Namespace And Compatibility APIs

**Files:**

- Modify: `tldw_chatbook/runtime_policy/server_credentials.py`
- Modify: `Tests/RuntimePolicy/test_server_credentials.py`
- Modify: `Tests/RuntimePolicy/test_server_credentials_lane_a.py`
- Modify: `tldw_chatbook/runtime_policy/__init__.py`

- [ ] **Step 1: Write failing namespace tests**

Add tests proving keyring usernames are listable, include a stable app namespace, and preserve existing public methods:

```python
def test_keyring_records_use_listable_chatbook_namespace():
    fake = FakeKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)

    store.set_secret("https://server.example.com/api", SERVER_CREDENTIAL_ACCESS_TOKEN, "secret")

    stored_usernames = {username for service, username in fake.values if service == DEFAULT_KEYRING_SERVICE_NAME}
    assert "__credential_refs__" in stored_usernames
    assert any("tldw_chatbook.server_credentials" in username for username in stored_usernames)
    assert any("profile=https%3A%2F%2Fserver.example.com%2Fapi" in username for username in stored_usernames)
    assert any("type=access_token" in username for username in stored_usernames)
```

Add a test that `clear_all()` deletes an indexed credential even when its server profile no longer exists:

```python
def test_keyring_clear_all_enumerates_namespace_index_and_removes_orphans():
    fake = FakeKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    store.set_secret("orphan-profile", SERVER_CREDENTIAL_REFRESH_TOKEN, "zombie")

    store.clear_all()

    assert fake.values == {}
```

Update the existing lane-A custom credential test so `clear_server()` now clears every indexed credential for the server, including non-standard credential types:

```python
def test_keyring_clear_server_removes_all_indexed_entries_for_profile():
    fake = FakeKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    store.set_secret("server-a", "custom_token", "c1")

    store.clear_server("server-a")

    assert store.get_secret("server-a", "custom_token") is None
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest Tests/RuntimePolicy/test_server_credentials.py Tests/RuntimePolicy/test_server_credentials_lane_a.py -q
```

Expected: failures show current usernames are still `server_id:purpose` and do not carry the richer namespace key.

- [ ] **Step 3: Add scoped credential refs while preserving old methods**

Implement this shape in `server_credentials.py`:

```python
@dataclass(frozen=True)
class ServerCredentialScope:
    server_profile_id: str
    normalized_origin: str
    credential_type: str
    principal_id: str | None = None

    @classmethod
    def legacy(cls, server_id: str, purpose: str) -> "ServerCredentialScope":
        normalized = _normalize_non_empty(server_id, "server_id")
        return cls(
            server_profile_id=normalized,
            normalized_origin=normalized,
            credential_type=_normalize_purpose(purpose),
            principal_id=None,
        )
```

Keep these existing protocol methods working:

```python
def set_secret(self, server_id: str, purpose: str, secret: str) -> None: ...
def get_secret(self, server_id: str, purpose: str) -> str | None: ...
def delete_secret(self, server_id: str, purpose: str) -> None: ...
def clear_server(self, server_id: str) -> None: ...
def clear_all(self) -> None: ...
```

Add richer internal methods if useful:

```python
def set_scoped_secret(self, scope: ServerCredentialScope, secret: str) -> None: ...
def get_scoped_secret(self, scope: ServerCredentialScope) -> str | None: ...
def delete_scoped_secret(self, scope: ServerCredentialScope) -> None: ...
```

Use URL quoting to build usernames:

```python
def _username_for_scope(scope: ServerCredentialScope) -> str:
    return (
        f"{DEFAULT_KEYRING_SERVICE_NAME}:v1|"
        f"profile={quote(scope.server_profile_id, safe='')}|"
        f"origin={quote(scope.normalized_origin, safe='')}|"
        f"principal={quote(scope.principal_id or '-', safe='')}|"
        f"type={quote(scope.credential_type, safe='')}"
    )
```

- [ ] **Step 4: Update keyring index format with backward-compatible reads**

Index entries should be dicts:

```python
{
    "version": 1,
    "server_profile_id": "...",
    "normalized_origin": "...",
    "principal_id": None,
    "credential_type": "access_token",
    "username": "tldw_chatbook.server_credentials:v1|...",
}
```

Keep `_load_index()` able to read old list entries like `["server-a", "access_token"]` and convert them into legacy scopes. This prevents breaking existing installs.

- [ ] **Step 5: Run credential tests**

Run:

```bash
python -m pytest Tests/RuntimePolicy/test_server_credentials.py Tests/RuntimePolicy/test_server_credentials_lane_a.py -q
```

Expected: all selected credential tests pass.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/runtime_policy/server_credentials.py tldw_chatbook/runtime_policy/__init__.py Tests/RuntimePolicy/test_server_credentials.py Tests/RuntimePolicy/test_server_credentials_lane_a.py
git commit -m "feat: harden server credential namespace"
```

## Task 2: Add Secure Credential Store Availability Gate

**Files:**

- Modify: `tldw_chatbook/runtime_policy/server_credentials.py`
- Modify: `tldw_chatbook/runtime_policy/server_context.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/RuntimePolicy/test_server_credentials.py`
- Modify: `Tests/RuntimePolicy/test_server_context_provider.py`
- Modify: `Tests/RuntimePolicy/test_runtime_policy_bootstrap.py`

- [ ] **Step 1: Write failing tests for unavailable/insecure backends**

Add fake backends:

```python
class FakePlaintextKeyring:
    __module__ = "keyring.backends.file"
    priority = 1

class FakeFailKeyring:
    __module__ = "keyring.backends.fail"
    priority = 0
```

Test the factory rejects them:

```python
def test_default_credential_store_rejects_plaintext_or_fail_backends():
    with pytest.raises(CredentialStoreUnavailable) as exc:
        build_default_server_credential_store(keyring_backend=FakePlaintextKeyring())

    assert exc.value.reason_code == "credential_store_unavailable"
```

Test no persistent fallback is created:

```python
def test_unavailable_credential_store_disables_persistent_secret_operations():
    store = UnavailableServerCredentialStore("no secure store")

    with pytest.raises(CredentialStoreUnavailable) as exc:
        store.get_secret("https://server.example.com/api", SERVER_CREDENTIAL_ACCESS_TOKEN)

    assert exc.value.reason_code == "credential_store_unavailable"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest Tests/RuntimePolicy/test_server_credentials.py::test_default_credential_store_rejects_plaintext_or_fail_backends Tests/RuntimePolicy/test_server_credentials.py::test_unavailable_credential_store_disables_persistent_secret_operations -q
```

Expected: imports fail because the factory/error types do not exist.

- [ ] **Step 3: Implement credential-store availability types**

Add:

```python
class CredentialStoreUnavailable(RuntimeError):
    reason_code = "credential_store_unavailable"


class UnavailableServerCredentialStore:
    def __init__(self, message: str) -> None:
        self.message = message

    def get_secret(self, server_id: str, purpose: str) -> str | None:
        raise CredentialStoreUnavailable(self.message)
```

Implement all protocol methods on `UnavailableServerCredentialStore` to raise `CredentialStoreUnavailable`.

- [ ] **Step 4: Implement secure backend classifier and factory**

Add:

```python
_SECURE_KEYRING_MODULE_PARTS = ("macos", "windows", "secretservice")
_INSECURE_KEYRING_MODULE_PARTS = ("fail", "null", "plaintext", "file")


def is_secure_keyring_backend(keyring_backend: Any) -> bool:
    module_name = str(getattr(keyring_backend.__class__, "__module__", "")).lower()
    priority = getattr(keyring_backend, "priority", None)
    if isinstance(priority, (int, float)) and priority <= 0:
        return False
    if any(part in module_name for part in _INSECURE_KEYRING_MODULE_PARTS):
        return False
    return any(part in module_name for part in _SECURE_KEYRING_MODULE_PARTS)


def build_default_server_credential_store(keyring_backend: Any | None = None) -> ServerCredentialStore:
    if keyring_backend is None:
        import keyring

        keyring_backend = keyring.get_keyring()
    if not is_secure_keyring_backend(keyring_backend):
        raise CredentialStoreUnavailable("No secure OS-backed credential store is available.")
    return KeyringServerCredentialStore(keyring_backend=keyring_backend)
```

If the implementation needs to accept a direct `keyring` module rather than a backend instance in tests, support both by normalizing with `get_keyring()` when present.
If the selected backend is a keyring chainer/wrapper, inspect the wrapped backend list and accept only when the resolved child backend is one of the allowed OS-backed stores.

- [ ] **Step 5: Wire app startup without plaintext fallback**

In `app.py`, replace direct construction:

```python
self.server_credential_store = KeyringServerCredentialStore()
```

with:

```python
try:
    self.server_credential_store = build_default_server_credential_store()
except CredentialStoreUnavailable as exc:
    self.server_credential_store = UnavailableServerCredentialStore(str(exc))
```

Do not write secrets to config or JSON if the store is unavailable.

- [ ] **Step 6: Run tests**

Run:

```bash
python -m pytest Tests/RuntimePolicy/test_server_credentials.py Tests/RuntimePolicy/test_server_context_provider.py Tests/RuntimePolicy/test_runtime_policy_bootstrap.py -q
```

Expected: selected runtime-policy credential tests pass.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/runtime_policy/server_credentials.py tldw_chatbook/runtime_policy/server_context.py tldw_chatbook/app.py Tests/RuntimePolicy/test_server_credentials.py Tests/RuntimePolicy/test_server_context_provider.py Tests/RuntimePolicy/test_runtime_policy_bootstrap.py
git commit -m "feat: gate server credentials on secure storage"
```

## Task 3: Add Typed Auth And Context Failure Contracts

**Files:**

- Modify: `tldw_chatbook/runtime_policy/server_context.py`
- Modify: `tldw_chatbook/runtime_policy/types.py`
- Modify: `tldw_chatbook/runtime_policy/__init__.py`
- Modify: `Tests/RuntimePolicy/test_server_context_provider.py`

- [ ] **Step 1: Write failing tests for reason-code-bearing errors**

Add tests:

```python
def test_context_unavailable_error_exposes_reason_code_and_safe_payload(tmp_path):
    provider = _provider(tmp_path, runtime_context=_runtime_context(active_server_id=None))

    with pytest.raises(ServerContextUnavailable) as exc:
        provider.get_active_context()

    assert exc.value.reason_code == "server_not_configured"
    assert exc.value.to_contract()["reason_code"] == "server_not_configured"
    assert "token" not in repr(exc.value.to_contract()).lower()
```

```python
def test_credential_store_unavailable_reason_is_preserved(tmp_path):
    provider = _provider(tmp_path, credential_store=UnavailableServerCredentialStore("secure store unavailable"))

    with pytest.raises(ServerCredentialsUnavailable) as exc:
        provider.get_active_context()

    assert exc.value.reason_code == "credential_store_unavailable"
    assert exc.value.to_contract()["recoverable"] is True
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest Tests/RuntimePolicy/test_server_context_provider.py::test_context_unavailable_error_exposes_reason_code_and_safe_payload Tests/RuntimePolicy/test_server_context_provider.py::test_credential_store_unavailable_reason_is_preserved -q
```

Expected: missing `to_contract()` and reason-code specificity.

- [ ] **Step 3: Introduce base typed error**

Add:

```python
@dataclass(frozen=True)
class ServerContextFailure:
    reason_code: str
    message: str
    recoverable: bool = True
    active_server_id: str | None = None


class ServerContextError(RuntimeError):
    reason_code = "server_context_unavailable"

    def __init__(self, message: str, *, reason_code: str | None = None, recoverable: bool = True, active_server_id: str | None = None) -> None:
        super().__init__(message)
        self.reason_code = reason_code or self.reason_code
        self.recoverable = recoverable
        self.active_server_id = active_server_id

    def to_contract(self) -> dict[str, object]:
        return {
            "reason_code": self.reason_code,
            "message": str(self),
            "recoverable": self.recoverable,
            "active_server_id": self.active_server_id,
        }
```

Make `ServerContextUnavailable` and `ServerCredentialsUnavailable` inherit from it. Preserve existing imports and exception names.

- [ ] **Step 4: Map current failures to stable reason codes**

Use at least:

- `server_not_configured`
- `server_profile_missing`
- `server_unavailable`
- `auth_required`
- `credential_store_unavailable`
- `server_credentials_unavailable`
- `stale_authorization`
- `profile_no_longer_authorized`

Do not include token values, auth headers, or raw config in messages or contract payloads.

- [ ] **Step 5: Run tests**

Run:

```bash
python -m pytest Tests/RuntimePolicy/test_server_context_provider.py -q
```

Expected: provider tests pass with old exception classes and new contract payloads.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/runtime_policy/server_context.py tldw_chatbook/runtime_policy/types.py tldw_chatbook/runtime_policy/__init__.py Tests/RuntimePolicy/test_server_context_provider.py
git commit -m "feat: add typed server auth failures"
```

## Task 4: Add Server Switch Invalidation Hook

**Files:**

- Modify: `tldw_chatbook/runtime_policy/server_context.py`
- Modify: `tldw_chatbook/runtime_policy/bootstrap.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/RuntimePolicy/test_server_context_provider.py`
- Modify: `Tests/RuntimePolicy/test_runtime_policy_bootstrap.py`

- [ ] **Step 1: Write failing tests for centralized invalidation**

Add provider test:

```python
@pytest.mark.asyncio
async def test_invalidate_for_server_switch_closes_cached_client_and_records_previous_context(tmp_path):
    credentials = InMemoryServerCredentialStore()
    credentials.set_secret("https://server.example.com/api", SERVER_CREDENTIAL_ACCESS_TOKEN, "access-1")
    provider = _provider(tmp_path, credential_store=credentials)
    first_client = provider.build_client()
    opened_http_client = await first_client._get_client()

    provider.invalidate_for_server_switch(previous_server_id="https://server.example.com/api", next_server_id="https://backup.example.com/api")
    await provider.close_cached_client()

    assert provider._cached_client is None
    assert opened_http_client.is_closed
```

Add app-wiring test that `handle_runtime_backend_changed("server")` calls the hook when active server changes.

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest Tests/RuntimePolicy/test_server_context_provider.py::test_invalidate_for_server_switch_closes_cached_client_and_records_previous_context Tests/RuntimePolicy/test_runtime_policy_bootstrap.py -q
```

Expected: missing invalidation method or app hook.

- [ ] **Step 3: Implement provider invalidation method**

Add:

```python
def invalidate_for_server_switch(self, *, previous_server_id: str | None, next_server_id: str | None) -> None:
    if previous_server_id != next_server_id:
        self._invalidate_cached_client()
```

Do not clear stored credentials here. Server switch invalidates active clients and handles, not persisted per-server credentials.
Do not remove entries from `_legacy_cleared_server_ids`; global sign-out and explicit credential clearing must continue to block legacy config reimport after switching away and back.

- [ ] **Step 4: Route runtime source updates through one app hook**

In `app.py`, capture previous state before `set_authoritative_runtime_source()` in `handle_runtime_backend_changed()` and call:

```python
self._invalidate_server_runtime_handles(previous_state, self.runtime_policy.state)
```

Implement `_invalidate_server_runtime_handles()` near `_close_server_context_provider_cached_client()`:

```python
def _invalidate_server_runtime_handles(self, previous_state: RuntimeSourceState | None, next_state: RuntimeSourceState | None) -> None:
    provider = getattr(self, "server_context_provider", None)
    invalidate = getattr(provider, "invalidate_for_server_switch", None)
    if callable(invalidate):
        invalidate(
            previous_server_id=getattr(previous_state, "active_server_id", None),
            next_server_id=getattr(next_state, "active_server_id", None),
        )
```

Keep future event/sync invalidation as no-op extension points, not implemented event/sync code.

- [ ] **Step 5: Run tests**

Run:

```bash
python -m pytest Tests/RuntimePolicy/test_server_context_provider.py Tests/RuntimePolicy/test_runtime_policy_bootstrap.py -q
```

Expected: selected runtime-policy tests pass.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/runtime_policy/server_context.py tldw_chatbook/runtime_policy/bootstrap.py tldw_chatbook/app.py Tests/RuntimePolicy/test_server_context_provider.py Tests/RuntimePolicy/test_runtime_policy_bootstrap.py
git commit -m "feat: centralize server switch invalidation"
```

## Task 5: Align Capability Snapshot Status With Target Store

**Files:**

- Modify: `tldw_chatbook/runtime_policy/server_capabilities.py`
- Modify: `tldw_chatbook/MCP/server_target_store.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/RuntimePolicy/test_active_server_capabilities.py`

- [ ] **Step 1: Write failing tests for target-store status persistence**

Add:

```python
@pytest.mark.asyncio
async def test_active_server_capabilities_updates_target_store_status(tmp_path):
    target_store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    target_store.save_targets([
        ConfiguredServerTarget(
            server_id="https://server.example.com/api",
            label="Primary",
            base_url="https://server.example.com/api",
            auth_mode="api_key",
            is_default=True,
        )
    ])
    context = _context(RuntimeSourceState(active_source="server", active_server_id="https://server.example.com/api", server_configured=True))
    service = ActiveServerCapabilityService(
        runtime_context=context,
        server_runtime_scope_service=FakeServerRuntimeScope(),
        target_store=target_store,
    )

    await service.refresh()

    target = target_store.get_target("https://server.example.com/api")
    assert target.last_known_reachability == "reachable"
    assert target.last_known_auth_state == "authenticated"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest Tests/RuntimePolicy/test_active_server_capabilities.py::test_active_server_capabilities_updates_target_store_status -q
```

Expected: constructor does not accept `target_store` or target status is unchanged.

- [ ] **Step 3: Add optional target-store dependency**

Update constructor:

```python
def __init__(self, *, runtime_context: Any, server_runtime_scope_service: Any, target_store: Any | None = None) -> None:
    self.runtime_context = runtime_context
    self.server_runtime_scope_service = server_runtime_scope_service
    self.target_store = target_store
```

After runtime state persistence, call `target_store.update_target_status(...)` when `state.active_server_id` is present. Swallow `KeyError` only for missing profiles and report an internal warning through the returned `errors` list if needed. Do not create a second capability authority.

- [ ] **Step 4: Wire app capability service with target store**

In `app.py`, pass:

```python
target_store=self.unified_mcp_target_store
```

to `ActiveServerCapabilityService`.

- [ ] **Step 5: Run tests**

Run:

```bash
python -m pytest Tests/RuntimePolicy/test_active_server_capabilities.py Tests/RuntimePolicy/test_server_context_provider.py -q
```

Expected: capability and provider tests pass.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/runtime_policy/server_capabilities.py tldw_chatbook/MCP/server_target_store.py tldw_chatbook/app.py Tests/RuntimePolicy/test_active_server_capabilities.py
git commit -m "feat: persist active server capability status"
```

## Task 6: Pin UX Handoff Contracts For Tranche 1

**Files:**

- Create: `tldw_chatbook/UX_Interop/server_connection_contracts.py`
- Modify: `tldw_chatbook/UX_Interop/server_parity_contracts.py`
- Create: `Tests/UX_Interop/test_server_connection_contracts.py`

- [ ] **Step 1: Write failing contract tests**

Create tests:

```python
from tldw_chatbook.UX_Interop.server_connection_contracts import (
    build_active_server_status_contract,
    build_auth_failure_contract,
)


def test_active_server_status_contract_is_versioned_and_secret_free():
    payload = build_active_server_status_contract(
        active_server_id="https://server.example.com/api",
        label="Primary",
        reachability="reachable",
        auth_state="authenticated",
        credential_source="credential_store:access_token",
    )

    assert payload["schema_version"] == 1
    assert payload["owner"] == "runtime_policy"
    assert payload["active_server_id"] == "https://server.example.com/api"
    assert "secret" not in repr(payload).lower()


def test_auth_failure_contract_uses_shared_reason_codes():
    payload = build_auth_failure_contract(
        reason_code="credential_store_unavailable",
        message="Secure credential storage is unavailable.",
        recoverable=True,
    )

    assert payload["schema_version"] == 1
    assert payload["reason_code"] == "credential_store_unavailable"
    assert payload["recoverable"] is True
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest Tests/UX_Interop/test_server_connection_contracts.py -q
```

Expected: module missing.

- [ ] **Step 3: Implement contract builders**

Use pure functions and dict payloads. Do not import Textual UI classes.

```python
def build_active_server_status_contract(*, active_server_id: str | None, label: str | None, reachability: str, auth_state: str, credential_source: str | None = None) -> dict[str, object]:
    return {
        "schema_version": 1,
        "owner": "runtime_policy",
        "stability": "tranche_1",
        "active_server_id": active_server_id,
        "label": label,
        "reachability": reachability,
        "auth_state": auth_state,
        "credential_source": credential_source,
    }
```

Add builders for:

- active server status
- auth failure
- credential-store unavailable
- server-switch invalidation
- capability status

- [ ] **Step 4: Aggregate exports**

Update `server_parity_contracts.py` to re-export these builders or include them in its registry so the UX handoff packet has one import surface.

- [ ] **Step 5: Run tests**

Run:

```bash
python -m pytest Tests/UX_Interop/test_server_connection_contracts.py -q
```

Expected: contract tests pass.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UX_Interop/server_connection_contracts.py tldw_chatbook/UX_Interop/server_parity_contracts.py Tests/UX_Interop/test_server_connection_contracts.py
git commit -m "feat: add server connection UX contracts"
```

## Task 7: Update Migration Audit Guard

**Files:**

- Modify: `Tests/RuntimePolicy/test_server_client_provider_migration_audit.py`
- Modify: `Docs/Development/server-client-provider-migration-audit.md`

- [ ] **Step 1: Inspect the current audit test**

Run:

```bash
python -m pytest Tests/RuntimePolicy/test_server_client_provider_migration_audit.py -q
```

Record current failures before editing. If it already passes, continue with semantic-key tightening.

- [ ] **Step 2: Make the audit key semantic, not line-based**

The audit should identify allowed raw builders by:

- file path
- matched builder signature or call expression
- per-file match count
- reason category

It must not rely on raw line numbers as the sole allowlist key.

- [ ] **Step 3: Add test for stable audit keys**

Add or update a test:

```python
def test_raw_client_builder_audit_uses_semantic_keys_not_line_numbers():
    entries = load_provider_migration_audit_entries()

    assert entries
    for entry in entries:
        assert "path" in entry
        assert "signature" in entry or "call_pattern" in entry
        assert "line" not in set(entry) or ("signature" in entry or "call_pattern" in entry)
```

- [ ] **Step 4: Update the audit document**

In `Docs/Development/server-client-provider-migration-audit.md`, add a short section:

```markdown
## Audit Key Contract

Allowed raw builder entries are keyed by path plus semantic match signature/call pattern and per-file match count. Line numbers are informational only and must not be the sole allowlist key.
```

- [ ] **Step 5: Run tests**

Run:

```bash
python -m pytest Tests/RuntimePolicy/test_server_client_provider_migration_audit.py -q
```

Expected: audit tests pass.

- [ ] **Step 6: Commit**

```bash
git add Tests/RuntimePolicy/test_server_client_provider_migration_audit.py Docs/Development/server-client-provider-migration-audit.md
git commit -m "test: harden provider migration audit keys"
```

## Final Verification

- [ ] **Step 1: Run focused runtime-policy and contract tests**

Run:

```bash
python -m pytest \
  Tests/RuntimePolicy/test_server_credentials.py \
  Tests/RuntimePolicy/test_server_credentials_lane_a.py \
  Tests/RuntimePolicy/test_server_context_provider.py \
  Tests/RuntimePolicy/test_active_server_capabilities.py \
  Tests/RuntimePolicy/test_runtime_policy_bootstrap.py \
  Tests/RuntimePolicy/test_server_client_provider_migration_audit.py \
  Tests/UX_Interop/test_server_connection_contracts.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 3: Confirm no secret literals are persisted in changed test fixtures**

Run:

```bash
rg -n "literal-provider-token-must-not-leak|access-1|refresh-1|legacy-bearer" tldw_chatbook/runtime_policy tldw_chatbook/MCP tldw_chatbook/UX_Interop
```

Expected: no matches in production code.

- [ ] **Step 4: Commit any final documentation updates**

If the final verification required doc or audit updates:

```bash
git add Docs/Development/server-client-provider-migration-audit.md Docs/superpowers/plans/2026-04-29-connection-auth-foundation-hardening.md
git commit -m "docs: update connection auth hardening plan"
```

## Handoff Notes For UX Developer

After this plan lands, UX should consume only the contracts in `tldw_chatbook/UX_Interop/server_connection_contracts.py` and the aggregated exports in `server_parity_contracts.py`.

Do not read current screen state directly for:

- active server status
- auth failure presentation
- credential-store unavailable presentation
- server-switch invalidation behavior
- capability status

The UX layer may assume local mode works without credential state, server mode never silently falls back to local writes, and unavailable server actions produce machine-readable reason codes.
