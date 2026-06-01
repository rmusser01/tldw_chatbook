# Settings Configuration Hub Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Settings the trustworthy configuration hub for Chatbook, with real load, validate, save, revert, recovery, and cross-screen verification paths instead of static summaries.

**Architecture:** Keep Settings as the global defaults and configuration surface, while runtime ownership stays with the owning destinations. Console owns active chat/run state, MCP owns server/tool management, ACP owns runtime/session setup, and Settings owns persisted defaults, validation, safety checks, and guided recovery entry points. Each category must either be fully wired or explicitly labeled read-only/WIP with a clear owner and recovery path.

**Tech Stack:** Python 3.11+, Textual, existing Settings screen contracts, existing config helpers, `chat_api_call()` provider registry, Console provider readiness helpers, Backlog.md, pytest, Textual-web/CDP screenshot QA.

---

## Scope

This plan covers the staged work tracked by `TASK-73` and its child tasks.

The highest-leverage path is:

1. Define the Settings ownership contract.
2. Make Providers & Models fully functional for Console use.
3. Expand Console defaults.
4. Harden Storage, Privacy, and Diagnostics.
5. Add server, sync, workspace, and handoff defaults without moving runtime workflows into Settings.
6. Add guided domain categories for the remaining product modules.
7. Polish Advanced Config and close out with full actual-use QA.

Do not start Sync v2 execution work from this plan. Sync may be represented as status/defaults inside Settings, but mutation replay, conflict review, and restore flows remain owned by Sync-specific tasks.

## Current Baseline

Settings on `origin/dev` already has:

- `tldw_chatbook/UI/Screens/settings_screen.py`
  - Destination-native three-column shell.
  - Category search and grouped navigation.
  - State banners and category-specific inspector guidance.
  - Provider/model defaults and endpoint editing.
  - Console large-paste setting.
  - Storage, privacy, diagnostics, and advanced TOML safety checks.
- `tldw_chatbook/UI/Screens/settings_config_adapter.py`
  - Config load/save adapter.
  - Raw TOML validation.
  - Secret redaction.
- `tldw_chatbook/UI/Screens/settings_config_models.py`
  - Category, validation, draft, and summary data models.
- `tldw_chatbook/UI/Screens/provider_model_resolution.py`
  - Shared effective provider/model resolution for Console-adjacent surfaces.
- `tldw_chatbook/Chat/provider_readiness.py`
  - Side-effect-free provider readiness.
- `tldw_chatbook/Chat/console_session_settings.py`
  - Console session settings, provider readiness, and summary contracts.
- `Tests/UI/test_settings_configuration_hub.py`
  - Mounted Settings regressions for shell, search, focus, provider defaults, save/revert, endpoint validation, privacy, diagnostics, and advanced config.

The main gap is completeness: Settings does not yet provide guided, end-to-end configuration for the whole app, and several categories still behave as summaries rather than real configuration surfaces.

## Configuration Ownership Decisions

These decisions prevent Settings categories from creating contradictory writes.

- Providers & Models owns provider identity, default model, provider endpoint/base URL, credential source, and provider+model default profiles. A model default profile may include model-scoped sampling and transport defaults because those values are part of choosing that specific model.
- Console Defaults owns global fallback sampling and transport defaults. It must normalize the existing `streaming` / `enable_streaming` compatibility seam before adding more controls.
- Provider credentials and endpoints live under `api_settings.<provider>`. Chat defaults live under `chat_defaults`. Console runtime/session overrides remain Console-owned.
- Provider+model default profiles live under the provider config, for example `api_settings.<provider>.model_defaults[<exact model id>]`. Persist the profile map as data under the provider section instead of relying on dotted section paths for model names, because model IDs may contain dots, slashes, colons, spaces, or provider-specific punctuation.
- Server, sync, workspace, and handoff Settings rows must be sourced from existing runtime-policy, sync, workspace, or destination state contracts. If no source contract exists, Settings renders explicit WIP/read-only ownership copy instead of inventing status.
- Domain configuration categories are added first as an ownership/status contract pass. A domain only gets save/revert controls after a separate PR-sized task identifies the concrete source of truth and destination owner.

## Console Default Resolution Hierarchy

Console settings must resolve in this order:

1. Active Console session override, when the user changed the value in the current tab.
2. Provider+model default profile from Settings, keyed by normalized provider and exact model id.
3. Provider fallback defaults from `api_settings.<provider>`.
4. Global Console defaults from `chat_defaults` or the Stage 3 Console Defaults canonical section.
5. Hardcoded safe defaults only when config is missing or invalid.

When a user changes the model in Console, the new model inherits its provider+model default profile. Existing session overrides must not be silently carried onto the new model unless the user explicitly keeps them. Console should expose recovery copy such as "Using model defaults" or "Modified in this session" plus a reset action back to model defaults.

## Non-Negotiable Gates

- Actual rendered screenshot QA is required for every visible Settings category change. Use Textual-web/CDP, not mockups or SVG/code renders.
- User approval is required after screenshots before opening each UI PR.
- Settings must never hide why a control is disabled.
- Secrets must never be printed in screenshots, test output, notifications, logs, or validation copy.
- Every save path must support validation, visible success/failure, and revert or recovery.
- Keyboard use must remain first-class: category search, category navigation, tab order, Enter activation, and footer shortcuts must be verified.
- Do not move MCP server/tool operations into Settings.
- Do not move ACP runtime/session setup into Settings.
- Do not make read-only or WIP categories look functional.

## Files And Responsibilities

- `tldw_chatbook/UI/Screens/settings_screen.py`
  - Settings destination shell, category rendering, user-facing actions, and mounted UI event handling.
- `tldw_chatbook/UI/Screens/settings_config_models.py`
  - Small state models for categories, drafts, validation, config ownership, and impact summaries.
- `tldw_chatbook/UI/Screens/settings_config_adapter.py`
  - Narrow persistence adapter over existing config helpers.
- `tldw_chatbook/UI/Screens/provider_model_resolution.py`
  - Shared provider/model source-of-truth resolution between Settings and Console.
- `tldw_chatbook/Chat/provider_readiness.py`
  - Side-effect-free provider readiness and credential source display.
- `tldw_chatbook/Chat/console_provider_support.py`
  - Provider identity mapping between Settings/readiness keys and `chat_api_call()` execution keys.
- `tldw_chatbook/Chat/console_session_settings.py`
  - Console defaults/session settings and readiness copy.
- `tldw_chatbook/config.py`
  - Default config constants and save/load helpers.
- `tldw_chatbook/css/components/_agentic_terminal.tcss`
  - Shared Settings styling, focus visibility, and three-column layout styling.
- `Tests/UI/test_settings_configuration_hub.py`
  - Primary mounted Settings coverage.
- `Tests/UI/test_console_session_settings.py`
  - Console settings modal/defaults interaction coverage.
- `Tests/UI/test_console_native_chat_flow.py`
  - Console generation path regressions after Settings defaults change.
- `Tests/Chat/test_provider_readiness.py`
  - Provider readiness coverage.
- `Tests/Chat/test_console_provider_support.py`
  - Provider identity and `chat_api_call()` registry coverage.
- `Docs/superpowers/qa/settings-configuration-hub/`
  - Actual CDP/Textual-web screenshot evidence for every visible slice.
- `backlog/tasks/task-73*.md`
  - Tracking tasks for this staged effort.

## Stage 1: Ownership Matrix And Settings Contract

Backlog: `TASK-73.1`

- [ ] **Step 1: Write failing documentation contract tests**

Add or extend a lightweight documentation/backlog regression, likely in `Tests/UI/test_settings_configuration_hub.py` or a new focused test file, to assert the Settings category list includes all expected ownership categories and that MCP/ACP runtime ownership copy remains explicit.

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py --tb=short
```

Expected: fail until the ownership rows and category copy exist.

- [ ] **Step 2: Add config ownership models**

Extend `settings_config_models.py` with a small `SettingsOwnershipRecord` or equivalent model that can describe:

- category id
- owns config sections
- reads runtime state from
- writes allowed yes/no
- runtime owner
- user-facing boundary copy
- WIP/read-only reason, if applicable

- [ ] **Step 3: Render ownership in Settings**

Update `settings_screen.py` so Overview and the inspector can show the ownership contract without duplicating long prose. The output should make clear that Settings owns global defaults and persisted configuration, not all runtime execution.

- [ ] **Step 4: Add docs**

Create `Docs/superpowers/specs/2026-05-29-settings-configuration-hub-design.md` or add an ownership matrix section if an existing Settings spec is present and more appropriate.

- [ ] **Step 5: Verify**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py --tb=short
git diff --check
```

- [ ] **Step 6: Capture screenshot**

Use Textual-web/CDP to capture the Overview or ownership screen after the change. Save under:

```text
Docs/superpowers/qa/settings-configuration-hub/stage-1-ownership.png
```

Do not proceed to PR until the user approves the actual screenshot.

## Stage 2: Providers & Models End-To-End

Backlog: `TASK-73.2`

- [ ] **Step 1: Add failing provider catalog and model-default tests**

Extend `Tests/UI/test_settings_configuration_hub.py`, `Tests/UI/test_console_session_settings.py`, and `Tests/Chat/test_console_provider_support.py` so Settings can list all providers supported by `chat_api_call()` while preserving custom entry paths, and so Console can inherit provider+model default profiles when a user switches models.

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_session_settings.py Tests/Chat/test_console_provider_support.py --tb=short
```

Expected: fail until Settings uses the provider registry/catalog instead of only freeform inputs and until Console has a model-default inheritance layer.

- [ ] **Step 2: Add guided provider controls**

Update `settings_screen.py` to expose provider and model selection from the existing provider catalog. Keep a custom/manual provider escape hatch for providers that exist in config but are not catalog-discovered.

Add an explicit model-default profile panel for the selected provider+model. The panel may edit model-scoped defaults such as `temperature`, `top_p`, `min_p`, `top_k`, `max_tokens`, and `streaming` because they apply only when that model is selected. Do not add global sampling or transport defaults in this stage; global fallbacks belong to Stage 3.

- [ ] **Step 3: Add credential source controls**

Add masked credential display and editing paths. A provider should show:

- `api_key` configured in config, masked
- `api_key_env_var`, with present/missing status
- key not required
- key missing with recovery copy

Never render raw secret values.

- [ ] **Step 4: Add endpoint/base URL policy**

Reuse `provider_readiness.py`, `console_provider_support.py`, and `console_session_settings.py` where possible. Endpoint editing must validate URLs before save and preserve existing provider-specific endpoint key names.

- [ ] **Step 5: Save to config and refresh runtime state**

Use `SettingsConfigAdapter` and existing config save helpers. Save provider identity, default model, endpoint/base URL, credential-source fields, and the selected provider+model default profile. Store model profiles as a nested mapping under the provider section, not as dotted section names derived from model IDs. After save, update `app_instance.app_config` and any relevant app reactive values so Console and Settings do not contradict each other.

- [ ] **Step 6: Console smoke coverage**

Add focused tests that change provider defaults in Settings, then assert Console readiness and blocked/ready feedback use the same effective provider/model. Add Console tests that switch from one model to another and prove the new model inherits its model-default profile while session edits are marked as overrides and can be reset to model defaults.

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_session_settings.py Tests/UI/test_console_native_chat_flow.py Tests/Chat/test_provider_readiness.py Tests/Chat/test_console_provider_support.py --tb=short
```

- [ ] **Step 7: CDP QA**

Capture and approve:

- provider category with missing key
- provider category with configured local endpoint
- provider category with a selected model-default profile
- Console after provider save
- Console after switching models and inheriting the new model defaults

Save under:

```text
Docs/superpowers/qa/settings-configuration-hub/stage-2-provider-*.png
```

## Stage 3: Console Defaults

Backlog: `TASK-73.3`

- [ ] **Step 1: Add failing Console default tests**

Assert Settings can load, edit, validate, save, and revert global Console defaults beyond paste collapse. These are fallbacks only; model-specific defaults remain in Providers & Models.

Candidate defaults:

- streaming
- temperature
- top_p
- max_tokens
- paste collapse enabled
- paste collapse threshold
- blocked-send display behavior, if already supported

- [ ] **Step 2: Define one defaults source of truth**

Before adding new controls, add a compatibility contract for existing `streaming` and `enable_streaming` reads. Pick the canonical stored key, document fallbacks, and add tests so Console and Settings read the same effective value.

- [ ] **Step 3: Separate global defaults from session overrides**

Settings copy must say global defaults. Console copy must say current session when a value is per-session.

- [ ] **Step 4: Implement guided controls**

Use simple Textual inputs/selects/buttons. Validate numeric ranges before save. Keep current paste collapse controls intact.

- [ ] **Step 5: Persist and refresh Console**

Save to the existing config sections. Verify Console uses the new default values after reload or direct runtime update.

- [ ] **Step 6: Verify and capture**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_session_settings.py --tb=short
git diff --check
```

Capture:

```text
Docs/superpowers/qa/settings-configuration-hub/stage-3-console-defaults.png
```

## Stage 4: Storage, Privacy, And Diagnostics

Backlog: `TASK-73.4`

- [ ] **Step 1: Add failing storage/privacy tests**

Assert Storage and Privacy categories have real checks, visible outcomes, and no raw secret leakage.

- [ ] **Step 2: Make Storage guided but conservative**

Storage can validate paths and show current locations. Any path mutation must be separately gated by validation and explicit confirmation. Do not move files in this stage unless there is already a safe app-level helper.

- [ ] **Step 3: Make Privacy guided but safe**

Expose redaction, key storage mode, encryption status, env-var status, and diagnostics policy. Only wire save controls for supported config paths.

- [ ] **Step 4: Harden Diagnostics**

Diagnostics should validate, reload, redact, and explain config source. It should not mutate raw TOML unless the user uses Advanced Config.

- [ ] **Step 5: Verify and capture**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py --tb=short
git diff --check
```

Capture:

```text
Docs/superpowers/qa/settings-configuration-hub/stage-4-storage-privacy-diagnostics.png
```

## Stage 5: Server, Sync, Workspace, And Handoff Source Contracts

Backlog: `TASK-73.5`

- [ ] **Step 1: Inventory source-of-truth contracts**

Before adding UI, document the concrete state providers Settings is allowed to read. Candidate seams include:

- `RuntimeServerContextProvider` and active-server capability/profile services for server status.
- `Sync_Interop.sync_promotion_state`, `Sync_Interop.sync_readiness`, and existing app `sync_scope_service` for sync safety.
- Existing Workspaces services and Console workspace context for active/default workspace status.
- Existing ACP screen/runtime/session state helpers for ACP task/run handoff readiness.

If any source is missing, Settings must render read-only/WIP copy naming the owning destination and must not invent synthetic status.

Concrete source contracts for this stage:

| Status/default | Source contract Settings may read | Owner boundary |
| --- | --- | --- |
| Server profile and authority | `runtime_policy.types.RuntimeSourceState` via `app_instance.runtime_policy.state`; `runtime_policy.server_context.RuntimeServerContextProvider` owns active context resolution | Settings renders active/default status only; server switching/auth remains runtime-policy/server surfaces. |
| Sync safety and dry-run/blocking copy | `Sync_Interop.sync_scope_service.SyncScopeService.list_write_sync_promotion_states`; `Sync_Interop.sync_promotion_state.SyncPromotionState`; `Sync_Interop.sync_readiness` fallback | Settings renders dry-run/recovery copy only; sync execution, conflict review, replay, and rollback stay sync-owned. |
| Workspace context/default | `Workspaces.LocalWorkspaceRegistryService.get_active_workspace`; `Chat.console_chat_store.ConsoleChatStore.workspace_context`; `Workspaces.display_state.LIBRARY_WORKSPACE_VISIBILITY_COPY` | Settings renders the current/default context only; Console/Home/Library own switching and staging. Library browse/search remains global. |
| Handoff policy | `Workspaces.models.WorkspaceTransferPolicy`; `Chat.chat_handoff_models.ChatHandoffPayload` | Settings exposes copy/reference/metadata-only policy language only; actual source staging and transfer stay destination-owned. |
| ACP handoff readiness | `ACP_Interop.runtime_session.ACPRuntimeSessionState` via `app_instance.get_acp_runtime_session_state()` | Settings renders runtime/session readiness only; ACP owns runtime launch, session setup, and Console follow payload creation. |

- [ ] **Step 2: Add failing cross-surface status tests**

Assert Settings, Home, Console, and Library use consistent labels for server profile, sync safety, workspace context, and handoff policy.

- [ ] **Step 3: Add Settings defaults/status only**

Settings may show global defaults and status for:

- active server profile
- local/server authority
- sync dry-run or blocked state
- workspace default
- handoff default policy

Do not add sync mutation replay or runtime execution from Settings.

- [ ] **Step 4: Verify owner boundaries**

Copy should link users to the correct destination:

- Sync execution/recovery stays in sync surfaces.
- Workspace switching/active context stays in Console/Home/Library flows.
- ACP task/run packages stay ACP-owned.

- [ ] **Step 5: Verify and capture**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py Tests/UI/test_home_screen.py Tests/UI/test_console_persistent_rails.py Tests/UI/test_destination_shells.py --tb=short
git diff --check
```

Capture:

```text
Docs/superpowers/qa/settings-configuration-hub/stage-5-server-sync-workspace.png
```

## Stage 6: Domain Configuration Category Contracts

Backlog: `TASK-73.6`

- [ ] **Step 1: Add category contract tests**

Each major module must either have a concrete source-of-truth contract for later guided settings or be labeled read-only/WIP with owner and recovery copy. This stage is not allowed to implement every domain setting in one PR.

Candidate categories:

- Library & RAG
- Artifacts
- Personas
- Skills
- Schedules
- Watchlists
- Workflows
- MCP defaults
- ACP defaults

- [ ] **Step 2: Add contract/status categories**

Add categories as ownership/status surfaces first. Only add save/revert controls for a domain if the source-of-truth config is already identified and the change remains PR-sized. Otherwise, record the follow-up task needed for that specific domain.

- [ ] **Step 3: Keep destination ownership intact**

Settings should not become a replacement for domain screens. It should configure defaults, not hide the actual workflow surfaces.

- [ ] **Step 4: Verify and capture**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py Tests/UI/test_destination_shells.py --tb=short
git diff --check
```

Capture the categories that changed under:

```text
Docs/superpowers/qa/settings-configuration-hub/stage-6-domain-categories-*.png
```

## Stage 7: Advanced Config And Closeout QA

Backlog: `TASK-73.7`

- [ ] **Step 1: Add failing Advanced Config polish tests**

Assert Advanced Config provides:

- validation-before-save
- stale-validation warning
- backup path on save
- redacted validation errors
- restore/recovery copy
- section search or jump, if implemented

- [ ] **Step 2: Add recovery affordances**

Keep Raw TOML available, but make it the expert escape hatch. Prefer guided category links when possible.

- [ ] **Step 3: Run full Settings walkthrough**

Actual use must verify:

- configure provider and observe Console readiness
- change Console defaults and observe Console behavior
- validate Storage
- validate Privacy with no secret leak
- run Diagnostics validation/reload
- use Advanced Config validation safely
- keyboard-only category navigation and action activation

- [ ] **Step 4: Run focused verification**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_session_settings.py Tests/UI/test_console_native_chat_flow.py Tests/Chat/test_provider_readiness.py Tests/Chat/test_console_provider_support.py --tb=short
git diff --check
```

- [ ] **Step 5: Capture closeout screenshots**

Capture approved final screenshots for:

- Overview
- Providers & Models
- Console Behavior
- Storage
- Privacy & Security
- Diagnostics
- Advanced Config
- any newly added domain categories

Save under:

```text
Docs/superpowers/qa/settings-configuration-hub/closeout-*.png
```

## PR Strategy

Use small PRs in dependency order:

1. `TASK-73.1`: Settings ownership matrix and contract.
2. `TASK-73.2`: Providers & Models functionalization.
3. `TASK-73.3`: Console defaults.
4. `TASK-73.4`: Storage, Privacy, Diagnostics hardening.
5. `TASK-73.5`: Server, sync, workspace, and handoff source contracts/status.
6. `TASK-73.6`: Domain configuration category contracts.
7. `TASK-73.7`: Advanced Config polish and full Settings closeout QA.

If a stage expands beyond one reviewable PR, split before implementation along user-visible category boundaries. Do not mix unrelated category rewrites.

## Final Acceptance Criteria

- Settings categories are either functional from a known source-of-truth contract or explicitly labeled read-only/WIP.
- Provider and model configuration supports all `chat_api_call()` provider identities that Console supports.
- Console readiness and Settings readiness cannot contradict each other.
- Provider/model settings do not save Console sampling or transport defaults.
- Save/revert/test actions are deterministic and covered by mounted tests.
- Storage/privacy/diagnostics expose status without leaking secrets.
- Advanced Config remains safe and guarded.
- Actual CDP/Textual-web screenshot QA exists for every changed Settings category and is user-approved.
- A full QA walkthrough verifies the app and functionality work, not merely that the screen renders or is clickable.
