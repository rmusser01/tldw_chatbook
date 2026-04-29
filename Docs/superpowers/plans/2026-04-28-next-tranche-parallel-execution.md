# Next Tranche Parallel Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the next server-parity tranche by hardening connection/auth behavior first, then executing capability/auth normalization and the high-priority provider-migration backlog in parallel without creating new ownership or merge bottlenecks.

**Architecture:** Keep `RuntimePolicyContext`, `RuntimeServerContextProvider`, `ConfiguredServerTargetStore`, and `ActiveServerCapabilityService` as the only active-server seams. Implement Lane A as the merge gate for credential lifecycle and invalidation, then branch Lane B and Lane C from the reviewed Lane A state. Lane C is intentionally limited to the high-priority compatibility backlog plus a single migration-audit owner; medium/low-priority migration targets stay as follow-on work.

**Tech Stack:** Python 3.11+, pytest, dataclasses, `keyring`, existing `tldw_chatbook.tldw_api.TLDWAPIClient`, runtime-policy services, existing migration-audit tooling.

---

## Scope

In scope for this plan:

- `Lane A` connection/auth hardening
- `Lane B` capability/auth normalization
- `Lane C` high-priority provider-migration compatibility cleanup
- migration-audit owner workflow and audit-guard verification

Out of scope for this plan:

- remote event transport
- sync dry-run/read-only mirror expansion
- medium/low-priority provider-migration backlog from the audit document
- UI redesign
- workflows

## File Structure

### Lane A

- Modify: `tldw_chatbook/runtime_policy/server_credentials.py`
  - Add global-clear support, orphan-safe deletion semantics, and any non-secret credential-reference index needed for keyring-wide cleanup.
- Modify: `tldw_chatbook/runtime_policy/server_context.py`
  - Add active-profile-only legacy import behavior, global sign-out invalidation, and stronger server-switch invalidation coverage.
- Modify: `tldw_chatbook/Auth_Account_Interop/auth_account_scope_service.py`
  - Persist auth tokens through the provider seam and clear them through the same seam.
- Modify: `tldw_chatbook/app.py`
  - Restrict changes to `_wire_server_context_provider()`, `_close_server_context_provider_cached_client()`, and direct logout/server-switch blocks that clear credentials or provider clients.
- Test: `tests/RuntimePolicy/test_server_context_provider.py`
- Test: `tests/Auth_Account/test_auth_account_scope_service.py`
- Test: `tests/RuntimePolicy/test_runtime_policy_bootstrap.py`
- Create: `tests/RuntimePolicy/test_server_credentials_lane_a.py`

### Lane B

- Modify: `tldw_chatbook/runtime_policy/server_capabilities.py`
  - Normalize capability refresh and invalidation behavior.
- Modify: `tldw_chatbook/runtime_policy/source_state.py`
  - Normalize persisted runtime-policy freshness/state projection behavior.
- Modify: `tldw_chatbook/runtime_policy/types.py`
  - Only if status typing needs explicit expansion or tightening.
- Modify: `tldw_chatbook/MCP/server_target_store.py`
  - Only `update_target_status()` and related last-known status projection logic.
- Test: `tests/RuntimePolicy/test_active_server_capabilities.py`
- Create: `tests/RuntimePolicy/test_runtime_source_state_lane_b.py`
- Create: `tests/MCP/test_server_target_store_lane_b.py`

### Lane C

- Modify: `Docs/Development/server-client-provider-migration-audit.md`
  - Only through one migration-audit integration owner.
- Modify high-priority compatibility modules only:
  - `tldw_chatbook/Auth_Account_Interop/server_auth_account_service.py`
  - `tldw_chatbook/Server_Runtime_Interop/server_runtime_service.py`
  - `tldw_chatbook/Chat/server_chat_conversation_service.py`
  - `tldw_chatbook/Chat/server_chat_loop_service.py`
  - `tldw_chatbook/Character_Chat/server_character_persona_service.py`
  - `tldw_chatbook/Character_Chat/server_chat_dictionary_service.py`
  - `tldw_chatbook/Media/server_media_reading_service.py`
  - `tldw_chatbook/Notes/server_notes_workspace_service.py`
  - `tldw_chatbook/Prompt_Management/server_prompt_service.py`
  - `tldw_chatbook/Prompt_Management/prompt_scope_service.py`
  - `tldw_chatbook/Chatbooks/server_chatbook_service.py`
  - `tldw_chatbook/Prompt_Studio_Interop/server_prompt_studio_service.py`
- Test: `tests/RuntimePolicy/test_server_client_provider_migration_audit.py`
- Add lane-specific service tests only where a converted compatibility factory currently lacks direct provider-backed coverage.

## Task 1: Lane A Credential Store Global Clear

**Files:**
- Modify: `tldw_chatbook/runtime_policy/server_credentials.py`
- Test: `tests/RuntimePolicy/test_server_credentials_lane_a.py`

- [ ] **Step 1: Write the failing global-clear tests**

```python
from tldw_chatbook.runtime_policy.server_credentials import (
    InMemoryServerCredentialStore,
    SERVER_CREDENTIAL_ACCESS_TOKEN,
    SERVER_CREDENTIAL_REFRESH_TOKEN,
)


def test_clear_all_removes_credentials_for_multiple_servers():
    store = InMemoryServerCredentialStore()
    store.set_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN, "a1")
    store.set_secret("server-b", SERVER_CREDENTIAL_REFRESH_TOKEN, "b2")

    store.clear_all()

    assert store.get_secret("server-a", SERVER_CREDENTIAL_ACCESS_TOKEN) is None
    assert store.get_secret("server-b", SERVER_CREDENTIAL_REFRESH_TOKEN) is None
```

- [ ] **Step 2: Add a fake-keyring failing test for orphan-safe global clear**

```python
class FakeKeyring:
    def __init__(self):
        self.values = {}

    def set_password(self, service_name, username, password):
        self.values[(service_name, username)] = password

    def get_password(self, service_name, username):
        return self.values.get((service_name, username))

    def delete_password(self, service_name, username):
        self.values.pop((service_name, username), None)


def test_keyring_clear_all_removes_indexed_entries():
    fake = FakeKeyring()
    store = KeyringServerCredentialStore(keyring_backend=fake)
    store.set_secret("server-a", "access_token", "a1")
    store.set_secret("server-zombie", "refresh_token", "z9")

    store.clear_all()

    assert fake.values == {}
```

- [ ] **Step 3: Run the lane-specific credential tests and confirm failure**

Run:

```bash
python -m pytest tests/RuntimePolicy/test_server_credentials_lane_a.py -q
```

Expected: fail because `clear_all()` or supporting behavior does not exist yet.

- [ ] **Step 4: Implement `clear_all()` and any non-secret credential-reference index**

Implementation requirements:
- do not persist secrets in JSON
- if keyring enumeration is unavailable, persist only a non-secret index of `(server_id, purpose)` references
- preserve current per-server `clear_server()` behavior

- [ ] **Step 5: Re-run the lane-specific credential tests**

Run:

```bash
python -m pytest tests/RuntimePolicy/test_server_credentials_lane_a.py -q
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/runtime_policy/server_credentials.py tests/RuntimePolicy/test_server_credentials_lane_a.py
git commit -m "feat: add global server credential clearing"
```

## Task 2: Lane A Provider Invalidation And Legacy Import

**Files:**
- Modify: `tldw_chatbook/runtime_policy/server_context.py`
- Test: `tests/RuntimePolicy/test_server_context_provider.py`

- [ ] **Step 1: Add failing tests for active-profile-only legacy import**

```python
def test_legacy_config_token_imports_only_for_active_server(provider, credential_store):
    provider.runtime_context.state = replace(
        provider.runtime_context.state,
        active_source="server",
        active_server_id="server-a",
        server_configured=True,
    )

    context = provider.get_active_context()

    assert context.active_server_id == "server-a"
    assert credential_store.get_secret("server-a", "access_token") is not None
    assert credential_store.get_secret("server-b", "access_token") is None
```

- [ ] **Step 2: Add failing tests for global sign-out and server-switch invalidation**

```python
def test_clear_all_credentials_invalidates_cached_client(provider):
    first = provider.build_client()
    provider.clear_all_credentials()

    with pytest.raises(ServerCredentialsUnavailable):
        provider.get_active_context()

    assert provider._cached_client is None


def test_switching_active_server_rebuilds_client_with_new_profile(provider, runtime_context):
    client_a = provider.build_client()
    runtime_context.state = replace(runtime_context.state, active_server_id="server-b")
    client_b = provider.build_client()
    assert client_a is not client_b
```

- [ ] **Step 3: Run the focused provider tests and verify failure**

Run:

```bash
python -m pytest tests/RuntimePolicy/test_server_context_provider.py -q
```

Expected: fail on missing global-clear or import/invalidation behavior.

- [ ] **Step 4: Implement provider hardening**

Implementation requirements:
- add provider-level `clear_all_credentials()`
- import legacy credentials only for the active profile
- invalidate cached clients on global clear, per-profile clear, token store, and server switch
- preserve typed unavailable/credential errors

- [ ] **Step 5: Re-run provider tests**

Run:

```bash
python -m pytest tests/RuntimePolicy/test_server_context_provider.py -q
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/runtime_policy/server_context.py tests/RuntimePolicy/test_server_context_provider.py
git commit -m "feat: harden runtime server context invalidation"
```

## Task 3: Lane A Auth Scope And App Wiring

**Files:**
- Modify: `tldw_chatbook/Auth_Account_Interop/auth_account_scope_service.py`
- Modify: `tldw_chatbook/app.py`
- Test: `tests/Auth_Account/test_auth_account_scope_service.py`
- Test: `tests/RuntimePolicy/test_runtime_policy_bootstrap.py`

- [ ] **Step 1: Add failing tests for token persistence and logout clearing**

```python
def test_login_persists_tokens_through_server_context_provider(scope_service, credential_store):
    scope_service.store_login_tokens(access_token="access-1", refresh_token="refresh-1")
    assert credential_store.get_secret("server-a", "access_token") == "access-1"
    assert credential_store.get_secret("server-a", "refresh_token") == "refresh-1"


def test_logout_clears_tokens_through_server_context_provider(scope_service, credential_store):
    scope_service.store_login_tokens(access_token="access-1", refresh_token="refresh-1")
    scope_service.clear_login_tokens()
    assert credential_store.get_secret("server-a", "access_token") is None
    assert credential_store.get_secret("server-a", "refresh_token") is None
```

- [ ] **Step 2: Add a failing bootstrap/app wiring test**

```python
def test_app_wires_server_context_provider_once(app):
    assert app.server_context_provider is not None
    assert app.server_credential_store is not None
```

- [ ] **Step 3: Run the auth/bootstrap tests and verify failure**

Run:

```bash
python -m pytest tests/Auth_Account/test_auth_account_scope_service.py tests/RuntimePolicy/test_runtime_policy_bootstrap.py -q
```

Expected: fail on missing token lifecycle helpers or wiring behavior.

- [ ] **Step 4: Implement minimal auth/app changes**

Implementation requirements:
- route login token storage through `server_context_provider.store_auth_tokens(...)`
- route logout clearing through `server_context_provider.clear_active_server_auth_tokens()` or `clear_all_credentials()` where appropriate
- keep `app.py` edits restricted to `_wire_server_context_provider()`, `_close_server_context_provider_cached_client()`, and direct logout/server-switch blocks

- [ ] **Step 5: Re-run the auth/bootstrap tests**

Run:

```bash
python -m pytest tests/Auth_Account/test_auth_account_scope_service.py tests/RuntimePolicy/test_runtime_policy_bootstrap.py -q
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Auth_Account_Interop/auth_account_scope_service.py tldw_chatbook/app.py tests/Auth_Account/test_auth_account_scope_service.py tests/RuntimePolicy/test_runtime_policy_bootstrap.py
git commit -m "feat: wire auth token lifecycle through server context provider"
```

## Task 4: Lane B Capability/Auth Normalization

**Files:**
- Modify: `tldw_chatbook/runtime_policy/server_capabilities.py`
- Modify: `tldw_chatbook/runtime_policy/source_state.py`
- Modify: `tldw_chatbook/runtime_policy/types.py`
- Modify: `tldw_chatbook/MCP/server_target_store.py`
- Test: `tests/RuntimePolicy/test_active_server_capabilities.py`
- Create: `tests/RuntimePolicy/test_runtime_source_state_lane_b.py`
- Create: `tests/MCP/test_server_target_store_lane_b.py`

- [ ] **Step 1: Add failing tests for capability invalidation**

```python
async def test_capability_snapshot_becomes_unknown_after_server_switch(service, runtime_context):
    await service.refresh()
    runtime_context.state = replace(runtime_context.state, active_server_id="server-b")
    snapshot = await service.refresh()
    assert snapshot["active_server_id"] == "server-b"
```

- [ ] **Step 2: Add failing tests for freshness/status projection**

```python
def test_normalize_runtime_source_state_resets_stale_auth_and_reachability():
    stale = RuntimeSourceState(
        active_source="server",
        active_server_id="server-a",
        server_configured=True,
        server_reachability="reachable",
        server_auth_state="authenticated",
        server_reachability_checked_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        server_auth_checked_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    normalized = normalize_runtime_source_state(stale, now=datetime.now(timezone.utc), freshness_window=timedelta(minutes=5))
    assert normalized.server_reachability == "unknown"
    assert normalized.server_auth_state == "unknown"
```

- [ ] **Step 3: Add failing tests for target-store projection updates**

```python
def test_update_target_status_preserves_projection_fields_only(store, target):
    updated = store.update_target_status(
        target.server_id,
        last_known_reachability="reachable",
        last_known_auth_state="authenticated",
    )
    assert updated.last_known_reachability == "reachable"
    assert updated.last_known_auth_state == "authenticated"
```

- [ ] **Step 4: Run Lane B tests and verify failure**

Run:

```bash
python -m pytest tests/RuntimePolicy/test_active_server_capabilities.py tests/RuntimePolicy/test_runtime_source_state_lane_b.py tests/MCP/test_server_target_store_lane_b.py -q
```

Expected: fail on missing invalidation/projection behavior or missing new tests.

- [ ] **Step 5: Implement normalization changes**

Implementation requirements:
- keep runtime-policy state authoritative
- treat target-store last-known status as projection only
- do not add any new selected-server or auth-state authority

- [ ] **Step 6: Re-run Lane B tests**

Run:

```bash
python -m pytest tests/RuntimePolicy/test_active_server_capabilities.py tests/RuntimePolicy/test_runtime_source_state_lane_b.py tests/MCP/test_server_target_store_lane_b.py -q
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/runtime_policy/server_capabilities.py tldw_chatbook/runtime_policy/source_state.py tldw_chatbook/runtime_policy/types.py tldw_chatbook/MCP/server_target_store.py tests/RuntimePolicy/test_active_server_capabilities.py tests/RuntimePolicy/test_runtime_source_state_lane_b.py tests/MCP/test_server_target_store_lane_b.py
git commit -m "feat: normalize active server capability and status projections"
```

## Task 5: Lane C Migration Audit Owner Workflow

**Files:**
- Modify: `Docs/Development/server-client-provider-migration-audit.md`
- Modify: `tests/RuntimePolicy/test_server_client_provider_migration_audit.py`

- [ ] **Step 1: Add failing audit-guard tests for semantic matching**

```python
def test_audit_guard_rejects_new_unlisted_legacy_builder(audit_runner):
    result = audit_runner(code='build_runtime_api_client_from_config(app_config)')
    assert result.exit_code != 0


def test_audit_guard_uses_semantic_not_line_number_matching(audit_runner):
    result = audit_runner(code='\\n\\nbuild_runtime_api_client_from_config(app_config)')
    assert "line-number-only" not in result.output
```

- [ ] **Step 2: Update the audit doc to mark one Lane C integration owner workflow**

Required content:
- pending audit delta handoff format
- migrated modules
- remaining compatibility factories
- explicit holdouts

- [ ] **Step 3: Run the audit-guard tests**

Run:

```bash
python -m pytest tests/RuntimePolicy/test_server_client_provider_migration_audit.py -q
```

Expected: PASS after doc/test updates.

- [ ] **Step 4: Commit**

```bash
git add Docs/Development/server-client-provider-migration-audit.md tests/RuntimePolicy/test_server_client_provider_migration_audit.py
git commit -m "chore: formalize migration audit owner workflow"
```

## Task 6: Lane C High-Priority Batch A

**Files:**
- Modify: `tldw_chatbook/Auth_Account_Interop/server_auth_account_service.py`
- Modify: `tldw_chatbook/Server_Runtime_Interop/server_runtime_service.py`
- Test: `tests/Auth_Account/test_server_auth_account_service.py`
- Create or modify focused runtime-service tests in `tests/RuntimePolicy/` or the existing runtime service test file if present

- [ ] **Step 1: Write failing tests for provider-only compatibility paths**

```python
def test_server_auth_account_service_from_provider_uses_provider_client(provider):
    service = ServerAuthAccountService.from_server_context_provider(provider)
    assert service.client is provider.build_client()
```

- [ ] **Step 2: Run the batch-A tests and verify failure**

Run:

```bash
python -m pytest tests/Auth_Account/test_server_auth_account_service.py tests/RuntimePolicy/test_server_context_provider.py -q
```

Expected: fail if compatibility paths still bypass provider-backed behavior.

- [ ] **Step 3: Convert compatibility factories with minimal code changes**

Rules:
- keep `from_config()` only if it delegates through the provider seam or remains explicitly audited
- do not edit shared audit file directly from this sub-batch; produce pending audit delta notes

- [ ] **Step 4: Re-run the batch-A tests**

Run:

```bash
python -m pytest tests/Auth_Account/test_server_auth_account_service.py tests/RuntimePolicy/test_server_context_provider.py -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Auth_Account_Interop/server_auth_account_service.py tldw_chatbook/Server_Runtime_Interop/server_runtime_service.py tests/Auth_Account/test_server_auth_account_service.py
git commit -m "refactor: migrate auth and runtime compatibility paths"
```

## Task 7: Lane C High-Priority Batch B

**Files:**
- Modify: `tldw_chatbook/Chat/server_chat_conversation_service.py`
- Modify: `tldw_chatbook/Chat/server_chat_loop_service.py`
- Modify: `tldw_chatbook/Character_Chat/server_character_persona_service.py`
- Modify: `tldw_chatbook/Character_Chat/server_chat_dictionary_service.py`
- Add or modify lane-specific tests in the matching domain test modules

- [ ] **Step 1: Add failing provider-backed compatibility tests for chat/character services**
- [ ] **Step 2: Run only the chat/character migration tests and confirm failure**

Run:

```bash
python -m pytest tests -q -k "chat and server and provider"
```

Expected: at least one failure tied to compatibility construction paths.

- [ ] **Step 3: Convert only the batch-B compatibility factories**
- [ ] **Step 4: Re-run the targeted tests**
- [ ] **Step 5: Commit**

```bash
git commit -am "refactor: migrate chat and character compatibility paths"
```

## Task 8: Lane C High-Priority Batch C

**Files:**
- Modify: `tldw_chatbook/Media/server_media_reading_service.py`
- Modify: `tldw_chatbook/Notes/server_notes_workspace_service.py`
- Add or modify lane-specific media/notes migration tests

- [ ] **Step 1: Add failing provider-backed compatibility tests for media/notes**
- [ ] **Step 2: Run only the media/notes migration tests and confirm failure**

Run:

```bash
python -m pytest tests -q -k "media or notes"
```

Expected: failure tied to legacy compatibility construction.

- [ ] **Step 3: Convert only the batch-C compatibility factories**
- [ ] **Step 4: Re-run the targeted tests**
- [ ] **Step 5: Commit**

```bash
git commit -am "refactor: migrate media and notes compatibility paths"
```

## Task 9: Lane C High-Priority Batch D

**Files:**
- Modify: `tldw_chatbook/Prompt_Management/server_prompt_service.py`
- Modify: `tldw_chatbook/Prompt_Management/prompt_scope_service.py`
- Modify: `tldw_chatbook/Chatbooks/server_chatbook_service.py`
- Modify: `tldw_chatbook/Prompt_Studio_Interop/server_prompt_studio_service.py`
- Add or modify lane-specific prompt/chatbook migration tests

- [ ] **Step 1: Add failing provider-backed compatibility tests for prompt/chatbook services**
- [ ] **Step 2: Run only the prompt/chatbook migration tests and confirm failure**

Run:

```bash
python -m pytest tests -q -k "prompt or chatbook"
```

Expected: failure tied to compatibility construction or audited fallback paths.

- [ ] **Step 3: Convert only the batch-D compatibility factories**
- [ ] **Step 4: Re-run the targeted tests**
- [ ] **Step 5: Commit**

```bash
git commit -am "refactor: migrate prompt and chatbook compatibility paths"
```

## Task 10: Lane C Audit Integration And Final Verification

**Files:**
- Modify: `Docs/Development/server-client-provider-migration-audit.md`
- Modify: `tests/RuntimePolicy/test_server_client_provider_migration_audit.py`

- [ ] **Step 1: Apply pending audit deltas from batches A-D through the single audit owner**
- [ ] **Step 2: Update audit tests to match the new baseline**
- [ ] **Step 3: Run the focused full-tranche verification suite**

Run:

```bash
python -m pytest \
  tests/RuntimePolicy/test_server_credentials_lane_a.py \
  tests/RuntimePolicy/test_server_context_provider.py \
  tests/Auth_Account/test_auth_account_scope_service.py \
  tests/RuntimePolicy/test_active_server_capabilities.py \
  tests/RuntimePolicy/test_runtime_source_state_lane_b.py \
  tests/MCP/test_server_target_store_lane_b.py \
  tests/RuntimePolicy/test_server_client_provider_migration_audit.py -q
```

Expected: PASS

- [ ] **Step 4: Run repository hygiene checks**

Run:

```bash
git diff --check
python -m compileall tldw_chatbook
```

Expected:
- `git diff --check`: no output
- `compileall`: exit code `0`

- [ ] **Step 5: Commit**

```bash
git add Docs/Development/server-client-provider-migration-audit.md tests/RuntimePolicy/test_server_client_provider_migration_audit.py
git commit -m "chore: finalize next tranche migration audit updates"
```

## Follow-On Work Explicitly Deferred

Do not pull these into this tranche:

- medium-priority provider migration modules from `Docs/Development/server-client-provider-migration-audit.md`
- low-priority provider migration modules from `Docs/Development/server-client-provider-migration-audit.md`
- remote event transport
- sync dry-run/read-only mirror expansion
- workflow surfaces
