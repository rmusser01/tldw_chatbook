---
id: TASK-16200
title: Build one-time Database Notes import planner and preview model
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-14 03:35'
labels:
  - notes
  - folders
  - import
dependencies:
  - TASK-15706
references:
  - Docs/superpowers/specs/2026-08-12-notes-folder-import-sync-design.md
  - backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md
  - >-
    backlog/decisions/060-notes-sync-round-trip-and-interoperability-constraints.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Establish the read-only planning boundary for importing individual files or one recursive folder into Database Notes so users can review hierarchy, repeat matches, collisions, and per-item actions before any note or folder is persisted.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Planner accepts either one or more individual files or exactly one directory, scans directories recursively while including the selected root as the proposed top-level folder, and rejects mixed/unsafe/unbounded selections without mutation.
- [ ] #2 An immutable preview classifies every discovered source as new, unchanged repeat, changed repeat, uncertain match, unsupported, or failed with a bounded user-safe reason and the specified default action.
- [ ] #3 Every approved preview item carries its parsed note payloads and proposed manual folder memberships, preserving source hierarchy and assigning every structured multi-note result to its source parent folder.
- [ ] #4 Folder-label collisions require an explicit use-existing, unique-sibling, or renamed-root decision; no existing tree is merged silently and empty or unsupported-only branches propose no folder creation.
- [ ] #5 Per-item overrides support Skip and Create new, plus Update existing only for exact or user-confirmed matches; content replacement and folder-membership addition are represented as independent decisions.
- [ ] #6 Planning performs no note, folder, receipt, or configuration writes; persistent diagnostics exclude content, absolute paths, hashes, and exception text.
- [ ] #7 Focused planner, path-safety, hierarchy, classification, collision, structured-source, bounds, and no-mutation tests pass.
<!-- AC:END -->

## Implementation Plan

ADR required: no
ADR paths: `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`, `backlog/decisions/060-notes-sync-round-trip-and-interoperability-constraints.md`
Reason: This task implements the already-approved one-time import ownership, hierarchy, matching, and independent content/membership decision boundaries without changing storage, sync policy, or service contracts.

1. Define frozen preview-domain models for source selection, classifications, actions, parsed note payloads, proposed memberships, matches, collision decisions, bounds, and aggregate plans.
2. Write focused failing tests for selection rules, safe recursive discovery, root inclusion, hierarchy preservation, structured multi-note parsing, bounded failure reasons, match classifications, defaults, collision handling, and immutable overrides.
3. Implement the read-only filesystem planner with fail-closed link handling, configurable depth/file/byte limits, private in-memory fingerprints, and parsers that do not emit source content or sensitive diagnostics.
4. Implement pure collision-resolution, uncertain-match confirmation, and per-item override operations that preserve the separation between content replacement and folder-membership addition.
5. Verify the planner performs no repository, receipt, configuration, or filesystem writes and run the focused Notes/Library regression gate.
6. Self-review the completed diff against the approved design and ADRs, update acceptance criteria and implementation notes, and record any generalized lesson only if the work produces one.

## Definition of Done

- [ ] All acceptance criteria are checked and backed by automated evidence.
- [ ] The implementation plan was followed or deviations are documented in Implementation Notes.
- [ ] Focused unit and integration tests cover the new planner behavior and pass.
- [ ] Relevant static analysis and formatting checks pass.
- [ ] The approved design and ADR-059/ADR-060 remain the governing documentation; no new ADR is required.
- [ ] The final diff has been self-reviewed for privacy, path safety, bounded resource use, and regressions.
- [ ] Implementation Notes summarize the approach, decisions, tests, and modified files.
- [ ] Any reusable lesson discovered by the task is recorded with its incident evidence, or the notes state that none was needed.
- [ ] The Backlog task status is set to Done only after every item above is complete.
