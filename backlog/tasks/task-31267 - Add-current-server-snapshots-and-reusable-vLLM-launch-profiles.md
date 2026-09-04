---
id: TASK-31267
title: Add current-server snapshots and reusable vLLM launch profiles
status: Done
assignee:
  - '@codex'
created_date: '2026-09-03 22:34'
updated_date: '2026-09-04 10:21'
labels:
  - vllm
  - lab
  - profiles
dependencies:
  - TASK-31265
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make repeated vLLM operation efficient and honest by separating the immutable running configuration from editable restart intent and retaining reusable non-secret launch profiles.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The UI displays an immutable current-server snapshot separately from the next-launch draft.
- [x] #2 Edits made while running are labeled as next-restart changes and can be applied with one Restart with draft action.
- [x] #3 Users can create, select, rename, duplicate, and delete named vLLM profiles containing only approved non-secret launch fields.
- [x] #4 The last selected vLLM view and profile restore across screen recomposition and application restart.
- [x] #5 Storage, migration if required, privacy, and profile lifecycle tests cover invalid, stale, and recovery states.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add strict launch-profile schema validation and a versioned device-local repository with optimistic revisions, safe canonical names, a 32-profile cap, selected-profile restoration, and atomic writes.
2. Extend vLLM launch ownership with immutable current-server snapshots and restart sequencing that preserves generation fencing, claim ownership, and exact-process termination guarantees.
3. Add the profile/current-server/restart controls to the vLLM Lab, keeping editable next-restart configuration distinct from immutable current state and using thread workers for repository I/O.
4. Cover schema, migration/future-version behavior, atomicity, optimistic conflicts, name collisions, profile UX, snapshot privacy, and restart safety with focused tests.
5. Run focused verification, update the ADR-linked task notes and acceptance criteria, and record exact evidence in the implementation report.

Fix Round 1:
6. Add regression tests and strict source-specific model validation at construction, decode, and pre-write boundaries without disclosing rejected values.
7. Harden document and adjacent lock handling against symlinks using descriptor-based no-follow checks, and add a deterministic two-process same-revision race test.
8. Amend accepted ADR-117 to the approved Task 4 exact V1 schema and rerun all scoped verification gates.

Fix Round 2:
9. Add deterministic regressions for unavailable ownership verification and lock-path replacement after acquisition, proving bytes and lock targets remain untouched.
10. Fail closed when platform ownership cannot be verified and revalidate the held lock descriptor/path identity immediately before protected document access and atomic replacement.

Fix Round 3:
11. Add fresh-repository regressions for missing, non-callable, raising, and invalid-result effective-UID capabilities, asserting generic repository errors and zero filesystem mutation.
12. Preflight and normalize ownership capability before directory/lock creation, reuse the validated effective UID for descriptor checks, then rerun all scoped gates.

Fix Round 4:
13. Add RED assertions for a raising ownership probe that require both `__cause__` and `__context__` to be `None`, no canary anywhere in the reachable exception graph, a generic safe message, and zero filesystem mutation for absent and existing parents.
14. Normalize ownership-probe failures outside the active exception handler without changing UID validation, descriptor checks, symlink defenses, locking, or CAS behavior.
15. Run isolated GREEN, the focused Task 4 suites, whole vLLM workflow, and exact static gates; append evidence, restore Done, and commit the narrow fix.

Task 6 Fix Round 2:
16. Add sequential RED tests for byte-capped profile reads, duplicate keys at every JSON object level, byte preservation, and real success/failure log-sink privacy canaries.
17. Route profile writes through a privacy-safe atomic wrapper that preserves lock/CAS/no-follow/inode guarantees, and reject oversized or duplicate-key documents before schema construction.
18. Run focused GREEN, the complete profile/core matrix, and exact static/diff/privacy gates; append evidence before restoring Done.

ADR required: no new ADR
ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md
Reason: ADR-117 remains the accepted device-local profile/CAS boundary; this round enforces bounded strict reads and privacy-safe atomic outcomes without changing the architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ADR-117’s device-local vLLM profile and restart boundaries. Added exact strict V1 JSON profiles at get_user_data_dir()/vllm_launch_profiles.json with a 32-profile cap, canonical Unicode/casefold name uniqueness, deterministic duplicate suffixes, restrictive atomic writes, and cross-process portalocker CAS; future/corrupt documents and write failures remain byte-preserving and fail closed. Added selected-profile restoration through thread-worker I/O, launch-only raw-argument isolation, exact claim-bound immutable Current server snapshots, safe Next restart dirty labels, and two-generation restart sequencing that proves the old process dead and claim released before a new reservation. Termination failure retains the old snapshot and creates no second process. No DB/schema migration.

Fix Round 1 tightened the same ADR-117 boundary: model values now receive source-specific non-secret validation during construction, decode, and immediately before write; nonexistent safe absolute local paths remain repairable while repository-ID, option, traversal, control, and credential-URL violations fail generically. Profile document and adjacent lock leaves now use no-follow descriptor opens, regular/current-owner/private-mode and inode-identity checks, with no chmod of existing pathnames. A simultaneous two-process revision-0 barrier proves one successful CAS and one conflict. ADR-117 remains Accepted and is amended to the approved exact V1 key names, 120-code-point names, document selected_profile_id/revision, and intentional omission of updated_at. Focused profile/setup/connection/UI tests, the complete vLLM workflow, full Ruff on new files, scoped Ruff on all touched Python files, focused mypy, formatting of new files, py_compile, and diff checks pass. ADR: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md. Evidence: .superpowers/sdd/2026-09-03-vllm-lab-console-complete-redesign/task-4-report.md.

Fix Round 2 makes file ownership validation mandatory: platforms without a valid effective-UID API now fail closed before profile bytes are read or written. Each mutation revalidates the exact held lock descriptor against the named private regular lock immediately before document load/revision validation and again immediately before the shared atomic writer. Deterministic inode-swap tests cover both boundaries, preserve the document and both lock targets, and prove the moved held inode is unlocked/closed after failure. This relies on ADR-117’s existing private, user-owned data-directory boundary; if an untrusted principal can rename within that parent, userspace leaf checks cannot eliminate the last instruction-level rename window, while atomic replace still does not follow a destination symlink. No ADR, schema, database, UI, or process-lifecycle change. Exact RED/GREEN and static evidence is appended to .superpowers/sdd/2026-09-03-vllm-lab-console-complete-redesign/task-4-report.md.

Fix Round 3 moves ownership capability validation ahead of every filesystem access or mutation. Missing, non-callable, raising, and invalid-result effective-UID hooks now become cause-free generic VllmProfileCorrupt errors; no raw exception or sensitive payload reaches callers. One validated UID is reused transaction-locally for document/lock descriptor ownership and both held-lock identity guards. The 8-case absent/existing-directory matrix proves no document, lock, or data directory is created on failure and a pre-existing directory/sentinel remains unchanged. No ADR/schema/database/UI/lifecycle change. Exact evidence: .superpowers/sdd/2026-09-03-vllm-lab-console-complete-redesign/task-4-report.md.

Fix Round 4 corrects the exception-object contract within ADR-117’s fail-closed storage boundary. Ownership-probe exceptions are converted to an invalid sentinel inside the handler, then the generic VllmProfileCorrupt is raised after the handler exits, so callers receive neither `__cause__` nor `__context__` and no sensitive probe payload is reachable. The absent/existing-parent matrix now walks the complete exception graph while retaining its zero-filesystem-mutation assertions. Ownership validation, descriptor checks, symlink and lock-swap protections, CAS, schema, UI, and process lifecycle are unchanged. Exact RED/GREEN and static evidence: .superpowers/sdd/2026-09-03-vllm-lab-console-complete-redesign/task-4-report.md.

Task 6 Fix Round 2 caps profile documents at 2 MiB before JSON decoding and
rejects duplicate JSON keys recursively at every object level. Oversized,
duplicate-version, duplicate-profile-field, and duplicate-nested documents
fail closed and mutation attempts preserve their exact bytes. Those four tests
were RED before the reader guard and GREEN as `4 passed, 61 deselected` (the
three duplicate mutation canaries also pass after final strengthening). Profile
commits now opt into the shared atomic writer's privacy-safe log mode; real
Loguru success/failure sinks prove no path, raw exception, credential, raw
command, URL, or model value is emitted. The privacy tests were RED as `2
failed, 59 deselected` and GREEN with the preservation control as `3 passed, 58
deselected`. Atomic replace, restrictive mode, held-lock identity, no-follow,
inode revalidation, and CAS ordering are unchanged. No new ADR: ADR-117 already
owns this exact private profile repository boundary.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

This task previously held id TASK-31219. The Task 6 collision correction moved
the full dependent vLLM sequence into one collision-free monotonic block so no
task depends on a future/higher task id. It therefore moved to TASK-31267 after
TASK-31264 preflight, TASK-31265 readiness, and TASK-31266 Console handoff. The
record was originally added by
`ffc4f9d8f8343169097dcac40d3ba4ed0a2177c0`.
