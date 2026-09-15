---
id: TASK-32601
title: Restore the authoring-only Workflows editor on dev
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-15 04:09'
updated_date: '2026-09-15 19:09'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved portable workflow editor on current dev while keeping unfinished execution and SQLite ownership changes isolated.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Users can create, edit, save, import and export workflow definitions in the canonical Workflows screen under ADR-138's approved stable-file exchange contract.
- [x] #2 The library, step navigator, overview and collapsed continuous form remain usable with keyboard and at supported terminal sizes.
- [x] #3 Drafts survive navigation and restart, failed persistence stays recoverable, and opaque server fields survive round trips.
- [x] #4 Only existing SQLite safety utilities are used; new execution and process-ownership infrastructure remain absent.
- [x] #5 The authoring slice meets the user-approved no-new-static-debt gate: introduced findings are fixed and retained baseline failures are source-attributed and documented.
- [x] #6 Validation notices and retry actions reflect the selected workflow and actual retryable failures; step labels and issue summaries update during editing without replacing focused controls or resetting view state.
- [ ] #7 PR 2690 is rebased onto current dev, all Qodo findings have verified resolutions or evidence-backed replies, and required checks pass before the requested merge.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes — amendment to existing ADR-138, not a new ADR number.
ADR path: backlog/decisions/138-portable-workflow-definitions-and-local-execution.md; ADR-125 and ADR-150 also apply.
Reason: record the user-approved stable-file exchange contract (2026-09-15) for authoring on current dev; runtime and the unapproved helper-owned-lock proposal are excluded.

1. Preserve the original and integration branches; use the clean codex/workflows-authoring-dev worktree based on 77eb2601a6.
2. Follow Docs/superpowers/plans/2026-09-14-workflows-authoring-dev.md. Reuse the reviewed editor-only checkpoint b34eda3d64 and the lossless document/draft code.
3. Register workflow document storage through current private_sqlite utilities. Retain existing migrations for file compatibility, but omit runtime locks and execution APIs.
4. Wire the editor into the real app, including durable draft flushing before navigation/quit and explicit local JSON import/export. Preserve current Console follow behavior.
5. Check the current tldw_server dev definition contract; verify targeted storage, authoring, lifecycle, keyboard and production-CSS behavior. Review actual captures at 160x48, 110x36 and 60x20.
6. Record exact verification and independent review. Do not mark Done with unresolved failures or unwaived static debt.
7. Apply the separately user-approved TASK-32601 no-new-static-debt gate in ADR-138: compare source spans against base77eb2601a6, correct changed import blocks as needed, verify scoped tests/static checks, and obtain the existing reviewer's scoped gate disposition. No broad cleanup, suppression or merge.

### Approved UAT correction (2026-09-15)

ADR required: no new ADR.
ADR path: backlog/decisions/138-portable-workflow-definitions-and-local-execution.md; ADR-150 also applies.
Reason: bounded feedback/reconciliation fixes in the existing authoring UI; no storage, runtime, or architectural changes.

1. Reproduce stale step labels/section summaries, misleading Retry on validation, and stale errors after successful context changes with failing regression tests. Preserve real load/write retry coverage.
2. Correct feedback state and update derived labels in place, preserving focused controls, cursor, collapse state and scrolling.
3. Run targeted editor/lifecycle tests and scoped static checks; repeat the original live-app reproductions in the disposable UAT profile.
4. Record fresh evidence and review; complete AC2 and AC6 only after the corrections pass.

### Requested PR integration and Qodo remediation (2026-09-15)

ADR required: amend existing ADR-138 if admission bounds or collection API contracts change; no new storage/runtime architecture.
ADR path: backlog/decisions/138-portable-workflow-definitions-and-local-execution.md.
Reason: the user explicitly requested rebase, address every Qodo comment, then merge. Existing authoring ownership, preservation, and stable-file decisions remain in force.

1. Commit verified UAT fixes, retain a backup branch and local evidence, and replay only the feature range onto the fetched dev tip. Inspect range-diff and resolve conflicts without losing dev work.
2. Track all seven Qodo comments by discussion ID. Reproduce the two shutdown claims with real draft-owner tests; preserve cancellation shielding and allow rejected revision saves to settle before the final durable flush. Add the requested class documentation and concrete Console item annotations.
3. Profile raw validation and imported complexity before changing their paths. Add narrow regression tests for responsive/stale-safe validation and controlled complexity rejection; keep invalid drafts recoverable. Record precise limits/worker ownership in ADR-138 before implementation.
4. Bound workflow/revision/draft listing queries and provide reachable pages in existing selectors. Preserve selection and history identity; validate paging arguments and test multiple pages, search and save/recovery across page boundaries. Record the exact API contract in ADR-138 before implementation.
5. Run affected domain, storage, editor, lifecycle and production-CSS tests plus the repository preflight guards. Inspect introduced static findings, not only net counts. Push the rebased branch using an explicit lease tied to the reviewed remote head.
6. Reply in each Qodo inline thread with fix/test evidence or a demonstrated false-positive explanation. Recheck reviews on the final pushed head and address new findings. Use the app's follow-up mechanism if awaiting posted reviews or CI; never bypass required checks.
7. Merge PR2690 into dev only after review disposition and required checks are satisfied for the exact final head; verify GitHub reports MERGED. Keep disposable evidence and unrelated worktrees intact.

## ID provenance

The CLI offered TASK-32591; the all-ref object-path and 38-worktree scans already contained IDs through 32600. Only this newly created file/header was moved to the checked-free TASK-32601 before implementation. No existing task was renumbered.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Current qualification:** reopened for the separately requested PR integration
and Qodo remediation. The following UAT qualification describes the prior checkpoint.

PR integration code checkpoint4fe2c2b5c8: rebased onto dev94cc1200d5 with all
14 feature patches unchanged. Seven Qodo findings have tested code resolutions
or evidence-backed cancellation-shield disposition. Bounded paging, projection
reuse, complexity admission and shutdown fixes preserve authoring-only scope.
Independent review caught and verified corrections for two lineage/node-budget
boundary failures; no remaining actionable findings. Latest314 targeted tests,
prior102 editor/projection tests,11 CSS checks and77 new-base destination/editor
tests passed. Scoped lint/format/diff checks pass. ADR-138 hardening and detailed
evidence: Docs/UAT/2026-09-15-workflows-qodo-remediation.md. AC7 remains open
pending GitHub replies, final-head required checks and the requested merge.

The approved UAT corrections are verified. Ordinary
validation notices no longer hide draft/revision status or imply Retry. Successful
context transitions clear operation errors; Validate preserves unrelated errors.
Step headings, navigator prompts and required/execution summaries update in place,
preserving controls and view state. Review additionally caught a deleted raw-step
selection; it now normalizes to Overview before the next edit. No SQLite, runtime,
provider or dependency changes. ADR-138/ADR-150 remain applicable; no new ADR.

Final verification: 93 targeted editor/authoring/lifecycle/destination tests passed
(128 deselected), four changed Python files pass Ruff/format, and diff-check passes.
The independent reviewer closed both correction findings, passed 18 scoped tests
and repeated the timing-sensitive recovery case 3/3. Tests now await scheduled
scrolling and actual modal mounting instead of assuming one pause drains events.
The live app passed the original feedback scenarios, all three layouts, raw
selected-step removal followed by editing, and real foreign-writer refusal/Retry.
Ten checks of those captures and the saved SQLite content passed; quick_check is
ok. Both app processes exited 0, temporary writers were released and UAT sessions
closed. Existing startup/dependency warnings are disclosed, not suppressed.

Evidence and limits: `Docs/UAT/2026-09-15-workflows-feedback-corrections.md`.
Original UAT evidence remains in `.uat-workflows-9NUT5t/UAT-REPORT.md`; final raw
captures/profile are also retained there locally. Updated the user guide and the
testing-evidence lesson for structural fallback followed by incremental editing.
No full sweep, current-dev merge, commit or push in this correction turn.

**Prior qualification:** all then-existing ACs and the approved scoped DoD were satisfied.
The separately user-approved no-new-static-debt gate in ADR-138 passes:711
remaining lint findings and64formatter edits across5files map exactly to baseline;
zero unmatched. All25newPythonfiles and the rewritten screen are clean. Two changed
test import blocks were sorted. The existing reviewer accepted the gate with no
new Critical/Important findings. Fresh tests:50passed across two targeted selections.
Task marked Done via Backlog CLI; no merge/push or runtime authority.

Final evidence: `.superpowers/sdd/2026-09-14-workflows-authoring-dev/static-gate.md`,
machine-readable attribution in `static-gate-results.json`, and reviewer disposition
in `static-gate-review.md`. Code62ab9ebc04, evidence7c46af626c. Whole-file lint and
format still fail as documented; this is an approved no-new-debt qualification,
not a claim that baseline debt was removed. Capture cleanup remains nonblocking.

The following records the earlier implementation/review checkpoints; statements
that the static gate was unwaived or In Progress describe those historical states.

On 2026-09-15 the user approved retaining
file-picker exchange under the stable-file assumption now recorded in ADR-138,
the spec, plan and user guide. Existing metadata-only alias rejection remains;
post-validation substitution or a detached live inode can still disrupt SQLite
locking. This is an accepted operating limitation, not a technical race fix.
No extra helper or file-I/O subsystem is authorized or added. The original
reviewer closed the alias finding under that approved scope and completed the
whole-branch review through e7a54a992f. Technical authoring review passes, but
full DoD/merge approval does not: baseline static-analysis debt remains unwaived.
Functional ACs are verified; no Done transition.

Implemented the reviewed b34eda3d64 authoring slice: real library, navigator,
overview and independently collapsed continuous forms; immutable saved revisions,
durable recoverable drafts, explicit private JSON import/export and disabled Run.
The app owns lazy document/draft setup and drains across navigation/quit. Existing
Console follow is a separate stable region. No execution API, runtime model graph,
new helper protocol, schema v5, server/provider writes or model calls were added.

WorkflowsDB uses the existing `connect_private_sqlite` seam, ordinary transactions
and close; v0-v4 migration bytes are preserved. Added only the domain registration
and owner census row. Tests exercise real SQLite foreign writers, persistence,
cancelled setup/close/exchange, failed flush retry, real app navigation/quit and
actual file pickers. The user guide documents privacy and local/server validation
limits. ADR-138 (including its approved stable-file amendment), ADR-125 and
ADR-150 remain the applicable decisions; no new ADR number or shared boundary.

Final targeted verification: 355 passed (authoring/storage/editor/quit/token/
bundle/census) plus 19 passed, 276 deselected (affected Workflows destination and
Console cases). The two visual findings now have the retained reviewer's scoped
ship verdict: precise draft-versus-revision status and Prompt label/value/focus
painted together at 60x20. Final interaction fixes serialize widget reconciliation,
retain raw-editor focus visibility and settle test modal mounting. Full failing
test names, diagnoses, exact RED/GREEN commands and final evidence are in
`.superpowers/sdd/2026-09-14-workflows-authoring-dev/task-1-report.md`.

At that checkpoint the task remained **In Progress** pending the static-analysis gate and full DoD.
Existing whole-file Ruff/formatter debt is reported separately, not waived or
suppressed. New/rewritten authoring files pass Ruff and formatting. All work is
confined to the named authoring-dev worktree; preserved checkouts are untouched.

Code-review fix round 1 adds a feature-local, metadata-only exchange precheck for
database/sidecar aliases. Real DELETE/WAL foreign-writer tests first reproduced
lock loss after a refused import, then verified preserved exclusion. Import and
export reject aliases before generic file I/O; shared helpers, SQLite lifecycle,
migrations and UI are unchanged. The report records exact RED/GREEN evidence and
the concurrent-path-replacement limitation. The covering authoring/storage suite
passes 35 tests, including off-loop metadata checks for import and export; scoped
Ruff/format and diff-check are clean. Review subsequently completed as above.

Final assessment and scope-based closure are recorded in
`.superpowers/sdd/2026-09-14-workflows-authoring-dev/final-review.md`.
Fresh coordinator run after the documentation amendment: 35 passed, one existing
warning in 21.53s; dedicated new/rewritten Python files pass Ruff and format
(26 files). No production edits in that approval turn. At that checkpoint the
whole-file baseline was 713 Ruff findings and five formatter failures, without
a task-specific exception. The later approved qualification above resolves that
gate without SQLite work and reduces the retained findings to 711.

Nonblocking review follow-up: the standalone QA capture script needs an explicit
owned bootstrap-profile lifetime and factory cleanup in finally. Current success
captures remain evidence, but repeated runs can leave disposable profiles behind
and capture failures skip factory cleanup. Recorded separately from the authoring
ACs; no data deleted or cleanup code added in this turn.
<!-- SECTION:NOTES:END -->
