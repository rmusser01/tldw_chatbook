---
id: TASK-32190
title: Preserve Advanced Config drafts and guard draft replacement
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 23:04'
updated_date: '2026-09-09 23:38'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Raw TOML edits must survive navigation and remain visibly unsaved so expert search-backend setup cannot silently lose work. Recovery and save actions must respect the exact draft the user reviewed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Unsaved raw TOML, including invalid and empty text, survives category and Settings destination navigation with accurate dirty state.
- [x] #2 Validation applies only to the current draft; Save retains the existing syntax gate, atomic write, and backup behavior, and clears only the text actually saved.
- [x] #3 Revert and Load Backup protect unsaved edits with confirmation; stale worker results cannot overwrite newer edits or another category.
- [x] #4 Returning to or saving a preserved draft detects intervening config edits and does not silently overwrite them.
- [x] #5 Mounted keyboard/navigation and failure/race tests, compact layout evidence, independent review, and user documentation cover the final behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce lost raw edits and inaccurate dirty state through the mounted canonical Settings screen before code changes.
2. Keep raw editor text, validation revision and file baseline in in-memory Settings draft state across category/destination navigation, using a small dedicated controller.
3. Extend the existing config owner with an atomic file-snapshot comparison for raw save; preserve encryption, revision-owned sections and backup behavior. A stale file/profile must reject save rather than overwrite another edit.
4. Route Validate, Save, Load Backup and Revert through guarded workers. Confirm draft replacement, reject stale completions, and clear only the saved revision. Keep the expert validation gate and honest raw-editor controls.
5. Verify persistence, empty/invalid drafts, worker races, replacement confirmation, conflicts, encryption/backup contracts and compact rendered controls with targeted tests and independent review; update walkthrough and review ledger.

ADR required: yes
ADR path: backlog/decisions/033-settings-commit-models-three-honestly-labeled.md
Reason: extend the existing guarded-raw-TOML policy with in-memory draft ownership and atomic stale-file checks; ADR-031 supplies destructive-action confirmation. No additional storage or parallel config owner is introduced.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Advanced Config now retains exact raw text, including invalid and empty drafts, in a live session stored by the existing memory-only screen-state store. A small controller owns revisions and the exact file/profile baseline; the panel is a disposable view. Validation grants Save only to the current revision. Revert and Load Backup confirm before replacing unsaved work, and newer edits supersede delayed operations. Saving uses app-owned workers so navigation cannot cancel runtime/state completion after a thread writes the file.

The existing config owner now exposes an exact serialized snapshot and guarded replacement under its write lock, returning the actual saved representation from the same lock. Conflicts leave config and backup untouched. Atomic writes, encryption, protected revision-owned sections, safe diagnostics, and the Settings path-validation boundary are retained. Initial reads are asynchronous; pending editor document changes are captured before completion guards so status paints cannot erase queued keystrokes. Failed saves retain the draft with guidance to check disk before retrying.

The raw controls use a compact two-row grid, expandable guide and editor-owned scrolling. The r shortcut is advertised consistently by footer and F1 help and is inert during text entry. Keyboard focus returns after validation when it has not moved elsewhere. Mounted production-CSS captures cover clean/dirty states at 120x35 and 80x24, with all four controls and at least three editor content rows available.

Validation: the expanded targeted run covered 501 cases across raw draft/snapshot tests, config persistence/delete/encryption, the Settings hub, guided Web Search, footer/category navigation, screen-state storage and import provenance. It finished with 500 passed and one stale assertion expecting 26 categories; Web Search deliberately raises the count to 27. Updated that assertion and reran it plus the grouping guard: 2 passed. No product failure remains. Independent re-review separately passed all 34 raw-draft/snapshot tests and found no remaining actionable issues after four async/lifecycle findings were repaired. New modules/tests pass full Ruff and format checks; changed legacy files pass scoped E9/F63/F7/F82 checks and changed-range formatting; CSS build and git diff --check pass. One pre-existing Requests dependency warning remains. No full repository sweep or real provider requests were run.

ADR: extended backlog/decisions/033-settings-commit-models-three-honestly-labeled.md; ADR-031 supplies confirmation and shortcut conventions. The live-session/app-worker detail was refined during independent review to cover destination recreation during a write. Updated Docs/User_Guide/settings.md, the original critique status ledger, .impeccable/surfaces/settings-advanced-config.md, and the incident-based lesson in backlog/docs/lessons-textual.md. New evidence is in .impeccable/review/raw-config/. Changes remain uncommitted in the isolated codex/shared-search-backend-default worktree; no merge or push.

PR integration: moved onto dev 86a8054edb, preserving current TLS, profile/footer, privacy and config publication behavior. Final evidence and baseline limitations: Docs/superpowers/reviews/2026-09-09-search-settings-pr-integration.md (525 targeted cases passed across two runs, three live cases skipped).
<!-- SECTION:NOTES:END -->
