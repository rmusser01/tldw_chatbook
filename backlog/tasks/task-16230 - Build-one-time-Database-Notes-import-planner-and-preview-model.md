---
id: TASK-16230
title: Build one-time Database Notes import planner and preview model
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 03:35'
updated_date: '2026-08-14 09:05'
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
    backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Establish the read-only planning boundary for importing individual files or one recursive folder into Database Notes so users can review hierarchy, repeat matches, collisions, and per-item actions before any note or folder is persisted.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Planner accepts either one or more individual files or exactly one directory, scans directories recursively while including the selected root as the proposed top-level folder, and rejects mixed/unsafe/unbounded selections without mutation.
- [x] #2 An immutable preview classifies every discovered source as new, unchanged repeat, changed repeat, uncertain match, unsupported, or failed with a bounded user-safe reason and the specified default action.
- [x] #3 Every approved preview item carries its parsed note payloads and proposed manual folder memberships, preserving source hierarchy and assigning every structured multi-note result to its source parent folder.
- [x] #4 Folder-label collisions require an explicit use-existing, unique-sibling, or renamed-root decision; no existing tree is merged silently and empty or unsupported-only branches propose no folder creation.
- [x] #5 Per-item overrides support Skip and Create new, plus Update existing only for exact or user-confirmed matches; content replacement and folder-membership addition are represented as independent decisions.
- [x] #6 Planning performs no note, folder, receipt, or configuration writes; persistent diagnostics exclude content, absolute paths, hashes, and exception text.
- [x] #7 Focused planner, path-safety, hierarchy, classification, collision, structured-source, bounds, and no-mutation tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR paths: `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`, `backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md`
Reason: This task implements the already-approved one-time import ownership, hierarchy, matching, and independent content/membership decision boundaries without changing storage, sync policy, or service contracts.

1. Define frozen preview-domain models for source selection, classifications, actions, parsed note payloads, proposed memberships, matches, collision decisions, bounds, and aggregate plans.
2. Write focused failing tests for selection rules, safe recursive discovery, root inclusion, hierarchy preservation, structured multi-note parsing, bounded failure reasons, match classifications, defaults, collision handling, and immutable overrides.
3. Implement the read-only filesystem planner with fail-closed link handling, configurable depth/file/byte limits, private in-memory fingerprints, and parsers that do not emit source content or sensitive diagnostics.
4. Implement pure collision-resolution, uncertain-match confirmation, and per-item override operations that preserve the separation between content replacement and folder-membership addition.
5. Verify the planner performs no repository, receipt, configuration, or filesystem writes and run the focused Notes/Library regression gate.
6. Self-review the completed diff against the approved design and ADRs, update acceptance criteria and implementation notes, and record any generalized lesson only if the work produces one.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a read-only, immutable one-time Database Notes import planner. The
planner accepts individual files or one recursive directory, uses bounded
non-following discovery and descriptor/handle revalidation, parses supported text
and structured formats, preserves the selected hierarchy, classifies repeat
observations with privacy-safe defaults, requires explicit normalized root-collision
decisions, and applies immutable per-item action/effect choices. Update authorization
is fail-closed: it requires one payload, an exact or user-confirmed match, and a
current optimistic note version. No note, folder, receipt, configuration, or source
filesystem writes were added.

The implementation plan was followed with one security-driven refinement: source
discovery, parsing, and the native Windows read-only adapter were separated so the
path-safety boundary remained reviewable. Native Windows source handles deny writer
and delete sharing, path-to-handle comparisons use only shared stable fields with a
nonzero inode, and handle-to-handle post-read checks remain strict. Three guarded
native-Windows tests could not execute on this macOS host and are recorded as skips,
not passing native evidence.

Verification on 2026-08-14:

- Planner plus Windows-adapter gate: 419 passed, 3 native-Windows-only skipped.
- Established Notes/Library regression gate: 267 passed, 593 deselected after
  rebasing onto the latest `origin/dev`.
- Broader `Tests/Notes` gate: 1,953 passed, 11 failed, 50 skipped; an untouched
  latest-`origin/dev` worktree reproduced the same 11 Git integration/environment
  failures with 1,552 passed and 47 skipped, showing the planner branch added 401
  passes and no new broader-suite failure.
- Ruff check and format check passed for all five production modules and both focused
  test modules; `git diff --check`, `compileall`, and Python 3.11 AST parsing passed.
  Mypy is configured but not installed in the workspace environment.
- Independent specification and quality reviews approved every implementation slice
  after Windows TOCTOU, surrogate-error privacy, optimistic-version, duplicate-target,
  empty-folder, and model-boundary findings were corrected with regression tests.
- PR review follow-up routes user selections through the shared lexical path validator
  without weakening the no-follow boundary, marks POSIX directory descriptors
  close-on-exec, preserves process interruptions raised during POSIX/Windows cleanup,
  and documents all public planner functions in Google style. Fourteen focused
  regressions cover these review corrections.
- Final independent review found that exact and user-confirmed items could acquire
  duplicate update authority for one note. Confirmation and override transforms now
  reject duplicate targets, while the aggregate plan model independently rejects two
  selected updates to the same note. Four additional regressions cover exact/uncertain,
  uncertain/uncertain, override, and direct-construction paths without exposing note
  identifiers in errors.

ADR required: no. ADR-059 and ADR-073 remain the governing ownership, privacy,
matching, hierarchy, and update-safety decisions. The Windows path/handle incident
and its native sharing/nonzero-identity follow-up were recorded in
`backlog/docs/lessons-testing-evidence.md`.

Modified files: the task plan and task record; five import planner/domain/discovery/
parser/Windows-adapter modules; two focused test modules; and the testing-evidence
lesson. No unrelated production files were changed.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 All acceptance criteria are checked and backed by automated evidence.
- [x] #2 The implementation plan was followed or deviations are documented in Implementation Notes.
- [x] #3 Focused unit and integration tests cover the new planner behavior and pass.
- [x] #4 Relevant static analysis and formatting checks pass.
- [x] #5 The approved design and ADR-059/ADR-073 remain the governing documentation; no new ADR is required.
- [x] #6 The final diff has been self-reviewed for privacy, path safety, bounded resource use, and regressions.
- [x] #7 Implementation Notes summarize the approach, decisions, tests, and modified files.
- [x] #8 Any reusable lesson discovered by the task is recorded with its incident evidence, or the notes state that none was needed.
- [x] #9 The Backlog task status is set to Done only after every item above is complete.
<!-- DOD:END -->
