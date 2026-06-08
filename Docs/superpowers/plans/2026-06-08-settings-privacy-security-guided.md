# Settings Privacy And Security Guided Posture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `Settings > Privacy & Security` into a structured, useful privacy posture and recovery panel without exposing raw secrets or adding unsafe credential/encryption mutation.

**Architecture:** Add a focused privacy posture helper that converts existing app config and environment status into redacted user-facing rows. Settings renders those rows in the existing three-column terminal workbench and adds recovery navigation to existing Settings categories.

**Tech Stack:** Python 3.11+, Textual, existing Settings screen contracts, pytest, Textual-web/CDP screenshot QA.

---

## Scope

This plan implements `TASK-82`.

In scope:

- `Settings > Privacy & Security` posture rows
- redacted provider credential source summary
- config encryption status summary
- sensitive field and provider config secret counts
- local/server data-boundary summary
- `Check Privacy`
- navigation to `Providers & Models`
- navigation to `Advanced Config`
- category-specific inspector guidance
- focused helper and mounted UI tests
- actual CDP/Textual-web screenshot QA

Out of scope:

- encryption enable/disable/change-password flows
- editing provider secrets directly from Privacy & Security
- revealing raw credential values
- sync, server handoff, or workspace execution behavior
- broad Settings layout restructuring

## ADR Check

ADR required: no

ADR path: N/A

Reason: This slice presents existing privacy/config state and adds recovery navigation while preserving the current credential/encryption service boundary. It does not introduce a storage schema, sync/conflict policy, data ownership change, provider/runtime boundary, security policy, dependency, or long-lived application structure.

## Files And Responsibilities

- `tldw_chatbook/UI/Screens/settings_privacy_security.py`
  - Pure privacy posture loading and redacted row-building helpers.
- `tldw_chatbook/UI/Screens/settings_screen.py`
  - Privacy category rendering, recovery navigation buttons, status updates, and inspector guidance.
- `Tests/UI/test_settings_privacy_security.py`
  - Pure helper coverage for posture calculation, redaction, malformed config, and environment status.
- `Tests/UI/test_settings_configuration_hub.py`
  - Mounted Settings coverage for Privacy rendering, privacy check, and recovery navigation.
- `Docs/superpowers/qa/product-maturity/screen-qa/settings/`
  - Actual Textual-web/CDP screenshot evidence for the visible Privacy slice.
- `backlog/tasks/task-82 - Functionalize-Settings-Privacy-and-Security-posture.md`
  - Task status, ADR check, implementation notes, and verification evidence.

## Task 1: Add Pure Privacy Posture Contract

- [x] **Step 1: Write failing helper tests**

Create `Tests/UI/test_settings_privacy_security.py` covering:

- empty config loads safe redacted defaults
- encryption status is derived from `encryption.enabled`
- sensitive config fields are counted, not displayed
- provider env-var status reports present/missing/configured counts
- provider config secrets are counted, not displayed
- malformed config values do not crash helper functions

Run:

```bash
python -m pytest -q Tests/UI/test_settings_privacy_security.py --tb=short
```

Expected: fail because the helper module does not exist yet.

- [x] **Step 2: Implement minimal helper module**

Create `settings_privacy_security.py` with:

- `SettingsPrivacyPosture`
- `build_settings_privacy_posture(app_config, environ=None)`
- `build_privacy_posture_rows(posture)`
- helpers that redact by construction and never return raw secret values

- [x] **Step 3: Verify helper tests pass**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_privacy_security.py --tb=short
```

Expected: pass.

## Task 2: Render Guided Privacy And Security Pane

- [x] **Step 1: Write failing mounted tests**

Extend `Tests/UI/test_settings_configuration_hub.py` to assert:

- Privacy & Security renders structured posture rows before a check is run
- no raw secret values appear in visible text
- existing `Check Privacy` action still updates the status result
- `Save` and `Revert` remain disabled/read-only for this category

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py --tb=short
```

Expected: fail until the new pane content is wired.

- [x] **Step 2: Wire privacy posture rendering**

Modify `SettingsScreen` to:

- build posture rows from the helper
- render `Privacy posture`, `Credential sources`, and `Data boundary` sections
- keep unsupported mutation copy explicit and password-gated
- keep existing privacy check worker behavior

- [x] **Step 3: Verify mounted rendering tests pass**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py --tb=short
```

Expected: pass.

## Task 3: Add Recovery Navigation And Inspector Guidance

- [x] **Step 1: Write failing navigation tests**

Extend mounted Settings tests to assert:

- pressing `Open Providers & Models` selects `SettingsCategoryId.PROVIDERS_MODELS`
- pressing `Open Advanced Config` selects `SettingsCategoryId.ADVANCED_CONFIG`
- the inspector explains Privacy & Security rows and recovery actions

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py --tb=short
```

Expected: fail until recovery controls and inspector copy are wired.

- [x] **Step 2: Implement recovery navigation and inspector copy**

Modify `SettingsScreen` button handlers and inspector summary so Privacy & Security has category-specific guidance.

- [x] **Step 3: Verify navigation tests pass**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py --tb=short
```

Expected: pass.

## Task 4: QA, Evidence, And Closeout

- [x] **Step 1: Run focused automated verification**

Run:

```bash
python -m pytest -q Tests/UI/test_settings_privacy_security.py Tests/UI/test_settings_configuration_hub.py --tb=short
git diff --check
```

- [x] **Step 2: Capture actual rendered screenshot**

Use Textual-web/CDP and capture `Settings > Privacy & Security`.

Screenshot must show:

- top-level Settings tabs
- three-column Settings layout
- structured privacy posture rows
- recovery navigation actions
- category-specific inspector guidance
- no raw secrets

- [x] **Step 3: Request user approval**

Do not create the PR until the user approves the actual rendered screenshot.

- [x] **Step 4: Update task notes**

Update `TASK-82` with:

- checked acceptance criteria
- implementation notes
- verification commands and screenshot evidence
- ADR check result

- [x] **Step 5: Commit and open PR against `dev`**

Create a small PR titled:

```text
Functionalize Settings Privacy and Security posture
```
