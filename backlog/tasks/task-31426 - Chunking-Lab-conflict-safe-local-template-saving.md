---
id: TASK-31426
title: Chunking Lab - conflict-safe local template saving
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-04 23:13'
updated_date: '2026-09-05 02:29'
labels:
  - chunking
  - chunking-lab
dependencies:
  - TASK-31421
  - TASK-31422
references:
  - backlog/decisions/118-chunking-lab-local-execution-and-recovery.md
documentation:
  - Docs/superpowers/specs/2026-09-04-chunking-lab-design.md
  - Docs/superpowers/plans/2026-09-04-chunking-lab.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Save Lab recipes through the existing canonical service with truthful validation, builtin protection, and atomic conflict detection. Covers spec section 9 and AC 3-4, 7, 16, 18, 20. Reuses the save semantics requested in existing TASK-24404 without adding a Settings form. ADR required: yes; ADR path: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md; reason: optimistic concurrency and Lab save service contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Lab create and update validate the final full flat body and record fields with parity validation plus the Lab capability gate, preserving advanced data and reporting structured errors without requiring a preview.
- [x] #2 Updates compare ID, UUID, version, builtin protection, and live state in the same transaction; intervening changes or deletion preserve the draft and offer Reload or Save as new without overwrite.
- [x] #3 Builtins default to Save as new, reserved auto names are refused, concurrent creates respect live-name uniqueness, and stored-invalid rows remain visible and repairable.
- [x] #4 Save A persists its pinned recipe and Save B its current valid draft; neither save changes Library content or defaults, and successful changes can trigger ingest-picker refresh.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md. Reason: implements approved canonical local save and atomic expected-version contract. 1. Read Task6 brief/context and real Media DB service/tests. 2. Write failing real SQLite stale-version and lossless-save tests. 3. Add headless shared-preflight Lab Save and atomic live/builtin/UUID/version checks without catalog schema changes. 4. Verify caller input preservation, reserved names, concurrent uniqueness, stored-invalid repair, advanced tags/metadata, and no source/default mutation. 5. Run targeted service/parity/catalog regressions and scoped static checks; self-review, independent review, and evidence notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the headless Lab save adapter over the canonical Media DB catalog and extended template updates with atomic ID/UUID/version/live/builtin predicates. Saves deep-detach authored bodies before canonical tag extraction, run the existing parity plus Lab capability gate, persist the authored body rather than normalized execution defaults, and return the refreshed record. Concurrent live-name collisions remain InputError, stale/deleted expected records become TemplateSaveConflict, and builtins remain BuiltinTemplateError. Added real SQLite coverage for stale updates, concurrent creates, stored-invalid repair, reserved auto variants, builtin copy/update behavior, resource ceilings, advanced metadata/tags, caller/default immutability, and refreshed identities. ADR required: yes; implemented ADR-118 and retained ADR-078's canonical store/flat body without schema changes. Targeted verification: 90 passed; scoped Ruff checks and formatting passed. One environment-level RequestsDependencyWarning remains unchanged.
<!-- SECTION:NOTES:END -->
