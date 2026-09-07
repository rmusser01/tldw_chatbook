# Exclusive Config Persistence and Runtime Snapshot Plan

**Goal:** Complete TASK-491 by making `config.py` the only owner of the
effective `config.toml` lifecycle and by publishing successful writes through
a generation-aware defensive runtime snapshot.

**Architecture:** Keep the existing process-wide config lock and effective-path
selection, but make the lock cover read/modify/private-atomic-write/cache
publication as one transaction. Add narrow serialized-text, raw replacement,
backup, and shutdown APIs in `config.py`; UI and application code call those
APIs instead of opening config files. Request-sensitive consumers resolve a
fresh defensive snapshot at the request boundary. Storage defaults remain
restart-bound and Console session values remain the highest-precedence runtime
override.

**Tech Stack:** Python 3.11+, stdlib threading/copy, TOML, existing
`private_paths`, pytest.

---

### Task 1: Lock the ownership contract with failing tests

**Files:**
- Create: `Tests/test_config_persistence_owner.py`
- Create: `Tests/test_config_runtime_snapshot.py`
- Modify: `Tests/test_config_app_config_encryption.py`
- Modify: `Tests/UI/test_settings_configuration_hub.py`

- [x] Prove every config mutation and display/export path affects only
  `TLDW_CONFIG_PATH` when set and never creates the default or historical
  fallback path.
- [x] Prove raw display/export returns the serialized encrypted representation,
  raw replacement cannot downgrade encryption, and backups are private.
- [x] Prove concurrent in-process readers observe either the prior or next
  complete generation, never a file/cache split.
- [x] Prove provider and credential/security reads see the next successful
  save while returned snapshots cannot mutate the cached generation.
- [x] Prove ADR-004 storage paths remain restart-bound and ADR-006 Console
  session overrides retain precedence.
- [x] Add a source guard for direct production `config.toml` writes, mutable
  `settings` imports, and request-sensitive module-scope snapshots.
- [x] Run the new tests and record the expected failures.

### Task 2: Complete the config-owned private transaction

**Files:**
- Modify: `tldw_chatbook/config.py`
- Test: `Tests/test_config_persistence_owner.py`
- Test: `Tests/test_config_runtime_snapshot.py`

- [x] Replace the generic pathname atomic writer with descriptor-anchored
  private atomic replacement for the effective config.
- [x] Track a monotonically increasing generation and publish deep defensive
  snapshots only after file replacement and cache reload succeed.
- [x] Return defensive copies from the new runtime snapshot API so consumers
  cannot mutate the cache.
- [x] Add config-owned serialized-text read, validated raw replacement,
  serialized backup/export, and shutdown persistence APIs under the same lock;
  keep existing setting-delete/reset paths on that owner.
- [x] Preserve an encrypted on-disk representation for display/export/backup
  and reject plaintext sensitive values when encryption is enabled.
- [x] Run the focused config transaction tests until green.

### Task 3: Route all config lifecycle owners through `config.py`

**Files:**
- Modify: `tldw_chatbook/app.py`
- Modify: `tldw_chatbook/UI/Screens/settings_config_adapter.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Test: `Tests/UI/test_settings_configuration_hub.py`

- [x] Replace both application startup creation blocks with the config
  bootstrap API and replace shutdown encryption writes with the config
  shutdown API.
- [x] Move advanced raw TOML display/save/backup/recovery into the Settings
  adapter backed by config-owned APIs.
- [x] Remove `DEFAULT_CONFIG_PATH` selection and direct read/write/replace
  operations from the Settings screen.
- [x] Ensure successful UI saves refresh `app_config` from the published
  generation and failures leave the prior file/cache generation intact.
- [x] Run application/settings ownership tests until green.

### Task 4: Make request-sensitive consumers live

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py`
- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls_Local.py`
- Modify: `tldw_chatbook/Widgets/Chat_Widgets/chat_right_sidebar.py`
- Modify only additional source-guard-identified request/credential consumers.
- Test: `Tests/test_config_runtime_snapshot.py`
- Test: existing provider and Console session suites.

- [x] Remove mutable `settings` imports.
- [x] Resolve provider endpoints/credentials from a defensive current snapshot
  at each request or view refresh boundary.
- [x] Keep environment and explicit call arguments at their existing
  precedence.
- [x] Preserve persisted storage defaults as next-launch values and Console
  session values above persisted defaults.
- [x] Run focused provider/security/Console tests until green.

### Task 5: Add the production ownership guard

**Files:**
- Modify: `Tests/test_config_persistence_owner.py`

- [x] Inventory production references to `DEFAULT_CONFIG_PATH`,
  `TLDW_CONFIG_PATH`, `config.toml`, and config write primitives.
- [x] Allow config-derived sibling artifact paths without treating them as
  config persistence owners.
- [x] Fail on direct effective-config opens/writes outside `config.py`.
- [x] Fail on production imports of mutable `settings` and reviewed
  request-sensitive module-scope credential/provider snapshots.
- [x] Run the guard and document intentional restart-bound constants.

### Task 6: Verify and close TASK-491

**Files:**
- Modify:
  `backlog/tasks/task-491 - Make-config-persistence-use-one-effective-path-and-live-runtime-boundary.md`

- [x] Run the focused persistence/runtime/settings/provider suites.
- [x] Run relevant broader config, Settings, provider, and Console suites.
- [x] Run changed-file Ruff, Python compilation, and `git diff --check`.
- [x] Run an encrypted sentinel probe covering override isolation, private
  atomic saves/backups, raw display/export, generation refresh, and downgrade
  rejection.
- [x] Self-review for plaintext downgrade, stale cache publication, direct
  ownership leaks, and accidental live application-state expansion.
- [x] Check all TASK-491 acceptance criteria, add implementation notes and
  evidence, set the task Done through Backlog, and commit only TASK-491 files.

## ADR Check

ADR required: yes

ADR path: `backlog/decisions/029-local-private-data-boundary.md`

Reason: TASK-491 implements ADR-029's accepted single config persistence owner
and live request-boundary snapshot. ADR-004 storage defaults and ADR-006 Console
session precedence remain explicitly unchanged.
