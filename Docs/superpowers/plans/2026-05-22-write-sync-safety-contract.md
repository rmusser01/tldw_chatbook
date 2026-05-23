# Write Sync Safety Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make write-sync promotion visibly safe by exposing shared authority, dry-run, review, conflict, and rollback states before any mutation replay can run.

**Architecture:** Add a pure `Sync_Interop` display contract that converts existing readiness, mirror-report, conflict, and profile-state data into user-facing promotion state. Wire the same state into Library Collections, Console workspace context, and Settings so each surface says the same thing: what is authoritative, whether sync is dry-run only, why writes are blocked, and what review/rollback work is required. This slice must not enqueue, replay, dispatch, or approve mutations.

**Tech Stack:** Python 3.11+, Textual, pytest, existing `Sync_Interop`, `Workspaces`, Library, Console, and Settings screen patterns.

---

## File Structure

- Create `tldw_chatbook/Sync_Interop/sync_promotion_state.py`
  - Pure dataclasses and builders for write-sync promotion display state.
  - No Textual imports, no repository writes, no network calls, no mutation methods.
- Modify `tldw_chatbook/Sync_Interop/sync_scope_service.py`
  - Add a read-only helper that builds promotion states for a list of domains from the existing repository and readiness registry.
- Modify `tldw_chatbook/UI/Screens/library_screen.py`
  - Replace ad hoc Collection sync copy with shared promotion labels while keeping all Collection CRUD local.
- Modify `tldw_chatbook/Workspaces/display_state.py`
  - Use the shared promotion copy for the Console workspace rail sync label.
- Modify `tldw_chatbook/UI/Screens/settings_screen.py`
  - Add a visible Sync Safety section showing Library, Collections, and Workspaces authority labels and blocked write state.
- Test `Tests/Sync_Interop/test_sync_promotion_state.py`
  - Unit coverage for dry-run, conflict, rollback, unknown-domain, and write-enabled-clamped states.
- Modify `Tests/Sync_Interop/test_sync_scope_service.py`
  - Service-level coverage for read-only promotion summaries.
- Modify `Tests/UI/test_product_maturity_phase39_library_collections.py`
  - Library Collection mounted assertions for shared promotion copy.
- Modify `Tests/UI/test_console_workspace_context_rail.py`
  - Workspace rail assertion for shared sync safety label.
- Modify or add `Tests/UI/test_settings_screen.py`
  - Settings mounted assertion for visible Sync Safety labels.
- Modify `backlog/tasks/task-60.4.2 - Post-release-write-sync-promotion-tranche.md`
  - Mark in progress, then Done only after implementation, tests, QA evidence, and user-approved screenshots.
- Create `Docs/superpowers/qa/product-maturity/post-release-ux-hci/2026-05-22-write-sync-safety.md`
  - Record actual-use QA, screenshots, and non-destructive safety constraints.

## Non-Negotiable Constraints

- Do not enable mutation replay.
- Do not add an "Approve sync" button.
- Do not call `send_changes`, outbox drain, `sync_once`, or remote mutation methods from UI surfaces.
- Any `write_enabled=True` lower-level input must still render as blocked unless explicit review, conflict, and rollback gates are present.
- Keep Library/Notes globally visible; workspace state may affect Console context eligibility, not browse/search visibility.
- Capture actual rendered screenshots for changed visible surfaces before claiming completion.

---

### Task 1: Pure Write-Sync Promotion State

**Files:**
- Create: `tldw_chatbook/Sync_Interop/sync_promotion_state.py`
- Test: `Tests/Sync_Interop/test_sync_promotion_state.py`

- [x] **Step 1: Write failing unit tests for safety states**

Cover:
- Unknown domain renders `Sync: unavailable`, `Authority: local-first`, and `mutation_allowed=False`.
- Dry-run eligible domain renders `Sync: dry-run only`, `Review: required before writes`, and `mutation_allowed=False`.
- Conflict reports render `Sync: conflict review required` and block mutation.
- Rollback-required profile state renders `Rollback: required before writes`.
- A readiness report with `write_enabled=True` is still clamped to `mutation_allowed=False` in this slice.

Run:
```bash
python -m pytest -q Tests/Sync_Interop/test_sync_promotion_state.py --tb=short
```

Expected: fail because the module does not exist.

- [x] **Step 2: Implement minimal pure state module**

Implement:
- `SyncPromotionState`
- `SyncPromotionSurfaceSummary`
- `build_sync_promotion_state(...)`
- `build_sync_promotion_summary(...)`

The builder should accept:
- `domain`
- `surface_label`
- optional `readiness`
- optional `latest_mirror_report`
- optional `conflict_reports`
- optional `profile_state`
- optional `source_authority`
- optional `workspace_id`

Return labels such as:
- `authority_label`
- `sync_label`
- `review_label`
- `conflict_label`
- `rollback_label`
- `primary_recovery`

All states in this slice return `mutation_allowed=False`.

- [x] **Step 3: Run unit tests green**

Run:
```bash
python -m pytest -q Tests/Sync_Interop/test_sync_promotion_state.py --tb=short
```

Expected: pass.

---

### Task 2: Sync Scope Service Read-Only Promotion Helper

**Files:**
- Modify: `tldw_chatbook/Sync_Interop/sync_scope_service.py`
- Test: `Tests/Sync_Interop/test_sync_scope_service.py`

- [x] **Step 1: Write failing service tests**

Add tests proving:
- `list_write_sync_promotion_states(...)` returns per-domain promotion states without calling server mutation methods.
- Conflict reports and latest mirror reports from `SyncStateRepository` influence the returned labels.
- Unknown domains remain blocked.

Run:
```bash
python -m pytest -q Tests/Sync_Interop/test_sync_scope_service.py --tb=short
```

Expected: fail because the helper does not exist.

- [x] **Step 2: Implement read-only helper**

Add `SyncScopeService.list_write_sync_promotion_states(...)`.

The method may read:
- `state_repository.get_latest_mirror_report`
- `state_repository.list_conflict_reports`
- `state_repository.get_sync_v2_profile_state`
- `build_sync_readiness_report`

The method must not call:
- `send_changes`
- `get_changes`
- `sync_once`
- any outbox dispatch/drain method
- any server mutation method

- [x] **Step 3: Run service tests green**

Run:
```bash
python -m pytest -q Tests/Sync_Interop/test_sync_scope_service.py --tb=short
```

Expected: pass.

---

### Task 3: Library Collections Shared Sync Safety Copy

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_product_maturity_phase39_library_collections.py`

- [x] **Step 1: Write failing mounted assertion**

Update Library Collections tests to expect:
- `Authority: local`
- `Sync: dry-run only` or `Sync: unavailable`
- `Review required before writes`
- no enabled write-sync action
- no copy implying write sync is available

Run:
```bash
python -m pytest -q Tests/UI/test_product_maturity_phase39_library_collections.py --tb=short
```

Expected: fail on missing shared labels.

- [x] **Step 2: Wire shared promotion state into Collection decoration/rendering**

Use `build_sync_promotion_state` for `library_collections` records and render its labels in the Collection detail/inspector area.

Do not change Collection create/rename/delete behavior.

- [x] **Step 3: Run Library Collections tests green**

Run:
```bash
python -m pytest -q Tests/UI/test_product_maturity_phase39_library_collections.py --tb=short
```

Expected: pass.

---

### Task 4: Workspace Rail Sync Safety Label

**Files:**
- Modify: `tldw_chatbook/Workspaces/display_state.py`
- Modify: `Tests/UI/test_console_workspace_context_rail.py`

- [x] **Step 1: Write failing workspace display assertion**

Assert active workspace state includes a clear sync safety label such as:
- `Sync: dry-run only; writes require review`

Run:
```bash
python -m pytest -q Tests/UI/test_console_workspace_context_rail.py --tb=short
```

Expected: fail on current simpler workspace sync label.

- [x] **Step 2: Use shared promotion copy for active workspace state**

Build workspace sync label from `build_sync_promotion_state(domain="workspaces", ...)`.

Keep workspace switching disabled.

- [x] **Step 3: Run workspace rail tests green**

Run:
```bash
python -m pytest -q Tests/UI/test_console_workspace_context_rail.py --tb=short
```

Expected: pass.

---

### Task 5: Settings Sync Safety Section

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Test: `Tests/UI/test_settings_screen.py` or nearest existing Settings mounted test file

- [x] **Step 1: Write failing Settings mounted test**

Assert Settings renders:
- `Sync safety`
- `Library: dry-run only`
- `Collections: dry-run only` or `Collections: unavailable`
- `Workspaces: dry-run only`
- `Writes: blocked until review, conflict, and rollback gates are ready`

Run:
```bash
python -m pytest -q Tests/UI/test_settings_screen.py --tb=short
```

Expected: fail on missing section/test file.

- [x] **Step 2: Add Settings section**

Add a static, read-only Sync Safety section in the existing Settings detail or inspector pane.

No toggles, no enable buttons, no mutation controls.

- [x] **Step 3: Run Settings tests green**

Run:
```bash
python -m pytest -q Tests/UI/test_settings_screen.py --tb=short
```

Expected: pass.

---

### Task 6: QA Evidence, Backlog, And Focused Verification

**Files:**
- Modify: `Docs/superpowers/qa/product-maturity/post-release-ux-hci/README.md`
- Create: `Docs/superpowers/qa/product-maturity/post-release-ux-hci/2026-05-22-write-sync-safety.md`
- Modify: `backlog/tasks/task-60.4.2 - Post-release-write-sync-promotion-tranche.md`

- [x] **Step 1: Run focused verification**

Run:
```bash
python -m pytest -q \
  Tests/Sync_Interop/test_sync_promotion_state.py \
  Tests/Sync_Interop/test_sync_scope_service.py \
  Tests/UI/test_product_maturity_phase39_library_collections.py \
  Tests/UI/test_console_workspace_context_rail.py \
  Tests/UI/test_settings_screen.py \
  --tb=short
git diff --check
```

Expected: pass.

- [x] **Step 2: Capture actual rendered screenshots**

Use the project CDP/textual-web QA process to capture at least:
- Settings Sync Safety section.
- Library Collections sync safety labels.
- Console workspace rail sync safety label if visible in the changed state.

Do not use SVG/code mockups as approval evidence.

- [x] **Step 3: Ask user approval for visible surfaces**

Show the actual screenshots and wait for approval before PR creation.

- [x] **Step 4: Update QA and Backlog**

Record:
- What was verified.
- What remains intentionally blocked.
- Screenshots used for approval.
- Confirmation that no write mutation path was enabled.

Mark `TASK-60.4.2` Done only after all acceptance criteria and screenshot approval are complete.

- [ ] **Step 5: Commit and open PR**

Run:
```bash
git add \
  tldw_chatbook/Sync_Interop/sync_promotion_state.py \
  tldw_chatbook/Sync_Interop/sync_scope_service.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  tldw_chatbook/Workspaces/display_state.py \
  tldw_chatbook/UI/Screens/settings_screen.py \
  Tests/Sync_Interop/test_sync_promotion_state.py \
  Tests/Sync_Interop/test_sync_scope_service.py \
  Tests/UI/test_product_maturity_phase39_library_collections.py \
  Tests/UI/test_console_workspace_context_rail.py \
  Tests/UI/test_settings_screen.py \
  Docs/superpowers/qa/product-maturity/post-release-ux-hci/README.md \
  Docs/superpowers/qa/product-maturity/post-release-ux-hci/2026-05-22-write-sync-safety.md \
  "backlog/tasks/task-60.4.2 - Post-release-write-sync-promotion-tranche.md" \
  Docs/superpowers/plans/2026-05-22-write-sync-safety-contract.md
git commit -m "Add write sync safety promotion state"
git push -u origin codex/task6042-write-sync-safety
gh pr create --base dev --head codex/task6042-write-sync-safety --title "Add write sync safety promotion state" --body-file <generated-pr-body>
```

Expected: PR opened against `dev`.
