# Settings Storage Defaults Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Settings > Storage a real guided configuration category for persisted local database path defaults without live file migration or active database reconnection.

**Architecture:** Add a focused Storage defaults helper beside the existing Settings helper modules. SettingsScreen owns rendering, draft state, validation feedback, and background save orchestration; the helper owns config loading, strict validation, save payload construction, and path-check row generation. Storage services remain runtime owners until the app restarts.

**Tech Stack:** Python 3.11, Textual, existing SettingsConfigAdapter, pytest, Backlog.md, Textual-web/CDP screenshot QA.

---

## ADR Check

ADR required: yes
ADR path: `backlog/decisions/004-settings-storage-defaults-restart-boundary.md`
Reason: Storage path defaults define persisted database path ownership and reject live migration/reconnection in Settings. This is a storage/runtime boundary future contributors are likely to revisit.

## Files

- Create: `tldw_chatbook/UI/Screens/settings_storage_defaults.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `Tests/UI/test_settings_configuration_hub.py`
- Modify: `backlog/tasks/task-81 - Functionalize-Settings-Storage-defaults.md`
- Modify: `Docs/superpowers/qa/product-maturity/screen-qa/settings/notes.md`
- Add: `Docs/superpowers/qa/product-maturity/screen-qa/settings/storage-defaults-2026-06-07/*.jpg`

## Task 1: Storage Defaults Helper

- [x] **Step 1: Write failing pure helper tests**

Add tests in `Tests/UI/test_settings_configuration_hub.py` proving the helper:

- Loads all supported `[database]` path defaults.
- Normalizes string path values without expanding them in the saved payload.
- Rejects empty, path-traversal, directory-shaped database file paths, and unreadable parent directories when strict validation is requested.
- Builds a deep-merged save payload under `database`.

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py::test_settings_storage_defaults_load_validate_and_build_save_payload --tb=short
```

Expected: fail because `settings_storage_defaults.py` does not exist.

- [x] **Step 2: Implement minimal helper**

Create `settings_storage_defaults.py` with:

- `SettingsStorageDefaults` dataclass.
- `load_storage_defaults(app_config)`.
- `validate_storage_defaults(values)`.
- `build_storage_save_sections(app_config, values)`.
- `build_storage_check_rows(values)`.

Keep save payload values as user-entered strings. Do not create directories or touch files.

- [x] **Step 3: Verify helper test passes**

Run the focused helper test. Expected: pass.

## Task 2: Guided Storage UI

- [x] **Step 1: Write failing mounted UI tests**

Add mounted tests proving Storage renders editable fields, dirty state, restart-required copy, invalid recovery, Save/Revert controls, and readable focused input text.

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py::test_settings_storage_renders_guided_defaults_and_validates --tb=short
```

Expected: fail because Storage remains validation-only.

- [x] **Step 2: Add SettingsScreen storage draft state**

Add `_storage_loaded_defaults`, `_storage_draft`, `_storage_result`, `_syncing_storage_defaults`, and utility methods mirroring the Appearance/Library-RAG patterns.

- [x] **Step 3: Render Storage controls**

Replace the read-only Storage path rows with labeled inputs for:

- Base data directory.
- ChaChaNotes DB.
- Prompts DB.
- Media DB.
- Research DB.
- Writing DB.
- Library Collections DB.
- Workspaces DB.

Preserve the `Check Storage` action and update copy to say it validates draft paths without moving data.

- [x] **Step 4: Wire field handlers**

Handle `Input.Changed` for each Storage field, mark the category dirty, update validation classes, and keep typed text visible when focused.

- [x] **Step 5: Verify mounted UI test passes**

Run the focused mounted UI test. Expected: pass.

## Task 3: Save, Revert, And Storage Check

- [x] **Step 1: Write failing save/revert tests**

Add tests proving Save uses a background worker, invalid drafts block Save, saved payload updates `app_config["database"]`, and Revert restores loaded values.

Run:

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py::test_settings_storage_save_and_revert_defaults --tb=short
```

Expected: fail because Storage Save/Revert is not wired.

- [x] **Step 2: Wire category actions**

Extend `_guided_actions_enabled`, `_guided_action_state_text`, `_category_state_banner_text`, `handle_save_settings_category`, and `handle_revert_settings_category` for Storage.

- [x] **Step 3: Add background save worker**

Add `_settings_save_storage_worker(thread=True, exclusive=True)` and `_apply_storage_save_result(...)`, following Appearance and Library/RAG save patterns.

- [x] **Step 4: Make Check Storage use draft values**

When Storage is active, `Check Storage` and shortcut `t` should validate the current Storage draft and report draft path readiness. It must not create directories or files.

- [x] **Step 5: Verify save/revert tests pass**

Run focused save/revert tests. Expected: pass.

## Task 4: Verification And Screenshot QA

- [x] **Step 1: Run focused tests**

```bash
python -m pytest -q Tests/UI/test_settings_configuration_hub.py --tb=short
```

Expected: pass.

- [x] **Step 2: Run diff hygiene**

```bash
git diff --check
```

Expected: clean.

- [x] **Step 3: Capture actual Textual-web/CDP screenshots**

Required screenshots:

- `Docs/superpowers/qa/product-maturity/screen-qa/settings/storage-defaults-2026-06-07/settings-storage-baseline.jpg`
- `Docs/superpowers/qa/product-maturity/screen-qa/settings/storage-defaults-2026-06-07/settings-storage-focused-path-input.jpg`
- `Docs/superpowers/qa/product-maturity/screen-qa/settings/storage-defaults-2026-06-07/settings-storage-invalid-path-recovery.jpg`
- `Docs/superpowers/qa/product-maturity/screen-qa/settings/storage-defaults-2026-06-07/settings-storage-saved-restart-required.jpg`

Saved under `Docs/superpowers/qa/product-maturity/screen-qa/settings/storage-defaults-2026-06-07/`. The browser capture returned JPEG bytes, so evidence files use `.jpg`. Do not use SVGs, code renders, or mockups.

- [x] **Step 4: Get user approval**

Show the actual screenshots to the user. Do not open the PR until the user approves the rendered Storage screen.

- [x] **Step 5: Update task and notes**

Mark all acceptance criteria complete, add Implementation Notes, update Settings QA notes, and leave TASK-81 Done only after verification and screenshot approval are complete.
