# Sync Profile Status Surfacing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface the merged Sync v2 profile summary in a read-only running-app workflow.

**Architecture:** Add a pure display-state adapter for Sync v2 profile summaries, render it in the existing Library Collections panel, and have `LibraryScreen` load the summary from `SyncScopeService` when an active server scope exists. The UI remains observational: no sync, restore, push, or pull actions are introduced.

**Tech Stack:** Python, Textual widgets, pytest, existing `SyncScopeService` and Library Collections panel/state patterns.

---

### Task 1: Pure Sync Profile Status State

**Files:**
- Create: `tldw_chatbook/Sync_Interop/sync_profile_status_state.py`
- Test: `Tests/Sync_Interop/test_sync_profile_status_state.py`

- [ ] **Step 1: Write failing tests** for `not_configured`, `server_frontend`, `pending`, and `attention_required` summary mappings.
- [ ] **Step 2: Run tests** with `../../.venv/bin/python -m pytest Tests/Sync_Interop/test_sync_profile_status_state.py -q` and confirm import failure.
- [ ] **Step 3: Implement dataclass mapping** with stable label, detail, severity, pending count, conflict count, dataset/device labels, and read-only copy.
- [ ] **Step 4: Re-run focused tests** and confirm pass.

### Task 2: Render Status in Library Collections

**Files:**
- Modify: `tldw_chatbook/Library/library_collections_state.py`
- Modify: `tldw_chatbook/Widgets/Library/library_collections_panel.py`
- Test: `Tests/Widgets/test_library_collections_panel.py`

- [ ] **Step 1: Write failing widget test** proving the panel renders the Sync v2 profile label/detail before collection rows.
- [ ] **Step 2: Run the widget test** and confirm the expected selector is missing.
- [ ] **Step 3: Add optional profile status to panel state** and render it in `LibraryCollectionsPanel` as a read-only status banner.
- [ ] **Step 4: Re-run widget and Library state tests** and confirm pass.

### Task 3: Load Summary From App Scope

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Test: `Tests/UI/test_product_maturity_phase39_library_collections.py`

- [ ] **Step 1: Write failing mounted UI test** with a fake `sync_scope_service` that returns a summary and tracks that no push/pull methods are invoked.
- [ ] **Step 2: Run the mounted test** and confirm the profile status is absent.
- [ ] **Step 3: Load profile summary through `sync_scope_service.get_sync_v2_profile_summary`** using the active server scope, worker-isolated like other repository reads.
- [ ] **Step 4: Re-run focused UI tests** and confirm pass.

### Task 4: Verification And Task Closeout

**Files:**
- Modify: `backlog/tasks/task-69 - Surface-Chatbook-Sync-v2-profile-status.md`

- [ ] **Step 1: Run focused tests** for the new state, widget, and mounted UI coverage.
- [ ] **Step 2: Run broader sync/library slice** with `Tests/Sync_Interop Tests/Library Tests/Widgets/test_library_collections_panel.py Tests/UI/test_product_maturity_phase39_library_collections.py`.
- [ ] **Step 3: Run compile, diff, and Bandit checks** on touched production files.
- [ ] **Step 4: Update Backlog notes and commit.**
