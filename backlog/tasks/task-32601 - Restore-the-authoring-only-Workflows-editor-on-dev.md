---
id: TASK-32601
title: Restore the authoring-only Workflows editor on dev
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-15 04:09'
updated_date: '2026-09-16 03:27'
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

### Approved CI correction and second Qodo pass (2026-09-15)

ADR required: no new ADR; directly apply existing ADR-097/ADR-150 and ADR-138.
ADR path: backlog/decisions/097-boot-budget-ratchets.md; backlog/decisions/138-portable-workflow-definitions-and-local-execution.md.
Reason: repair introduced boot-cost and authoring correctness regressions using
existing boundaries, with no new loading, storage, runtime or sync infrastructure.

1. Keep the existing failing boot-byte and selector-count guards unchanged. Use
   the existing screen-owned stylesheet splitter and app route loader for the
   Workflows-only tokens; narrow broad subjects and audit token ownership.
2. Test first navigation and Workflows-as-initial-route stylesheet loading,
   rendered form/dialog behavior and the unchanged performance limits. Rebuild
   generated sheets rather than editing them directly.
3. Verify second-pass Qodo findings before changes: discard/update/selection
   races, deleted draft identity, conditional media text and metadata types.
   Add regressions, preserve raw buffers, and keep the existing draft owner.
4. Add controller API documentation/precise options annotations and one debounce
   policy constant. Verify targeted suites/static checks, obtain scoped review,
   push, reply to each finding, and wait for exact-head checks before merge.

### Final-head Qodo follow-up (2026-09-16 UTC)

ADR required: amend existing ADR-138 only if the library read contract changes.
ADR path: backlog/decisions/138-portable-workflow-definitions-and-local-execution.md.
Reason: retain existing ownership and paging limits while eliminating unnecessary
full-document reads/projections for names; no schema or runtime infrastructure.

1. Verify final-head Qodo findings 4021531211/1225/1230/1239. Document/type the
   three specified widget modules and move the standard-library catalog import.
2. Measure library-name projection, reproduce excess parsing with real stored
   pages, then use bounded name/identity-only reads for both library surfaces.
   Preserve Unicode search, stable pagination, exact selection and raw documents.
3. Run targeted store/controller/paging/editor and import/CSS guards, review the
   changes, reply to each finding and refresh exact-head review/checks before merge.

### Search-boundary Qodo follow-up (2026-09-16 UTC)

ADR required: amend existing ADR-138 for the bounded search contract; no new ADR.
ADR path: backlog/decisions/138-portable-workflow-definitions-and-local-execution.md.
Reason: address findings 4021705887/4021705892 without new storage infrastructure.

1. Reproduce unbounded search acceptance before SQLite and repeated search scans
   with real query tracing; retain tests of Unicode, raw names and exact paging.
2. Validate both search entry points through one shared strict Pydantic model:
   512-character raw query limit plus existing page/offset bounds, without
   trimming or coercion. Preserve admitted lone-surrogate strings with a local
   plain field validator; ordinary Pydantic string validation rejects them.
3. Execute one ordered summary query per search and consume at most 100 rows per
   fetch, stopping once the requested matching page is full. No connection,
   schema, index or cache changes.
4. Verify targeted regressions, mounted search/error recovery, shared validation
   consumers and preflight; obtain review, push, reply and await final-head gates.

### Shared bound follow-up (2026-09-16 UTC)

ADR required: no; mechanical consolidation under existing ADR-138.
ADR path: backlog/decisions/138-portable-workflow-definitions-and-local-execution.md.
Reason: Qodo 4021826851 identifies the duplicated signed-SQLite-integer bound.

1. Define SQLITE_INTEGER_MAX in the existing input-validation module, use it in
   WorkflowSearchInput and alias the existing document-service MAX_GENERATION to
   it. Preserve all public values, bounds and behavior.
2. Verify existing generation/paging boundary and search regressions, scoped
   static attribution and preflight. Obtain scoped review, push, reply and wait
   for exact-head CI/Qodo before merge.

### New-base CSS integration (2026-09-16 UTC)

ADR required: no new ADR; apply existing ADR-097 and ADR-150.
ADR path: backlog/decisions/097-boot-budget-ratchets.md.
Reason: rebasing onto dev65a143 adds Notes styles and exposes a 490-byte
startup-CSS breach; AC7 already requires resolving introduced integration gates.

1. Retain the failing unchanged boot-budget guard (768,490 / 768,000 bytes).
2. Scope the six remaining bare editor/library/navigator geometry selectors to
   their existing unique Workflows IDs so the existing conservative splitter
   can defer them. Preserve declarations, route loader and limits; add no API.
3. Verify both entry paths, real rendered geometry, compact editor interactions,
   splitter/cascade/byte guards and preflight; obtain scoped independent review.
4. Record evidence, push with the exact remote-head lease, then await fresh
   Qodo and exact-head CI before merging. Keep AC7 open until verified merge.

### Creation and public-contract review (2026-09-16 UTC)

ADR required: amend existing ADR-138 for the creation-dialog input boundary.
ADR path: backlog/decisions/138-portable-workflow-definitions-and-local-execution.md.
Reason: address Qodo4022051681/1690/1696/1704/1712 without altering portable
definition admission or adding dependency/storage infrastructure.

1. Reproduce oversized/wrong-type creation names reaching setup and missing-SVG
   capture failing without installation guidance. Pin no-write/draft retention.
2. Validate creation names in the shared input module before setup: text only,
   at most256 raw Python characters, trim surrounding whitespace, reject empty;
   retain admitted Unicode. Apply the same helper to the standalone controller's
   creation fallback, retaining its InvalidDraft category. Keep errors concise
   enough for the existing status area; do not restrict imported/raw names.
3. Check the existing SVG availability helper before capture imports; raise an
   informative ImportError with the real svg extra and native-Cairo guidance.
4. Annotate the paging event and document the requested catalog/create/import/
   export APIs with actual arguments, returns and failure contracts.
5. Run affected boundary/authoring/exchange/paging/catalog/shared-consumer tests,
   source-attributed static checks and preflight; obtain scoped review, push,
   reply to all five threads and await exact-head review/checks before merge.

### Creation-name portability correction (2026-09-16 UTC)

ADR required: amend existing ADR-138's local creation boundary; no new ADR.
ADR path: backlog/decisions/138-portable-workflow-definitions-and-local-execution.md.
Reason: Qodo4022157035 identifies names admitted locally but unencodable by the
server's direct text binding. Preserve existing/imported JSON rather than rewriting it.

1. Verify the current server dev name binding and reproduce SQLite surrogate
   encoding failures alongside a valid supplementary-character control.
2. Replace the new-creation surrogate acceptance test with unopened-store
   rejection cases across the surrogate range. Preserve valid composed,
   decomposed and supplementary Unicode and the 256-character boundary.
3. Reject surrogates through strict UTF-8 encoding in the existing bounded
   validator; keep concise error guidance, controller category and lossless
   imported/raw-name contracts. No replacement or Unicode normalization.
4. Run affected creation/retention/import/search/UI checks, static attribution
   and preflight; obtain scoped review, push, reply and await exact-head gates.

## ID provenance

The CLI offered TASK-32591; the all-ref object-path and 38-worktree scans already contained IDs through 32600. Only this newly created file/header was moved to the checked-free TASK-32601 before implementation. No existing task was renumbered.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shared-bound follow-up: sole Qodo finding 4021826851 on f67867129b is addressed
by one SQLITE_INTEGER_MAX constant in existing shared validation, aliased as the
existing MAX_GENERATION in document_service. Values and behavior are unchanged;
no new ADR or infrastructure. Real SQLite maximum-bound characterization passed
before/after extraction; 244 service/summary/draft-owner/mounted-paging tests pass.
Independent review has no findings. Source-attributed static debt is unchanged,
all three files are formatted, and new-code Ruff/diff checks pass. Evidence is in
Docs/UAT/2026-09-15-workflows-qodo-remediation.md. Exact-head gates/merge pending.

Search-boundary follow-up: Qodo findings 4021705887/4021705892 on af041970e9
reproduced oversized-query admission and 10/11 repeated search statements over
1,000 heads. Both read APIs now validate through one shared strict Pydantic model
with a 512-character query cap, preserving admitted Unicode and existing paging
limits. Search performs one ordered query with bounded fetches and early stopping;
no schema/cache/connection changes. ADR-138, the plan and user guide are updated.
Final targeted verification: 520 tests plus 13 import/CSS checks pass, all seven
preflight guards pass, independent review has no findings. Eight shared-module
Ruff findings are exactly source-attributed to baseline; no new static debt or
suppressions. All four touched Python files are formatted. Detailed evidence:
Docs/UAT/2026-09-15-workflows-qodo-remediation.md. New-head review/CI and merge
remain pending; AC7 stays open.

Final-head Qodo follow-up (2026-09-16 UTC): four fresh findings on 2e76d7bdf2
held merge despite its passing required/UI checks. Library pages now read only
bounded display-name/identity tuples through the existing document service;
both main and compact mounted regressions prove zero full-document projections
after reproducing 20/21 per page. Exact heads, Unicode search, stale-result guards
and raw legacy content remain preserved. Added widget public contracts/types and
moved deepcopy to the standard-library import block. No new storage/runtime
infrastructure; ADR-138 documents the summary contract. Fresh targeted results:
420 domain/storage/UI tests, 13 import/CSS checks, all seven preflight guards,
plus scoped Ruff/format/diff checks pass. Separate 166- and 20-test selections
overlap and are not additional suite totals. Detailed evidence and qualifications:
Docs/UAT/2026-09-15-workflows-qodo-remediation.md. Independent review and new-head
GitHub review/checks were held for the independently reproduced escaped-surrogate
name failure. The local extraction/decoding correction preserves admitted names
and exact saved bytes; six new cases cover service and both mounted layouts.
The reviewer verified that correction and reported no remaining blockers. A
subsequent 425-pass/1-failure run exposed an incomplete-field test reading hit
coordinates during focus-scroll animation; instrumentation confirmed it, and
the test now awaits scheduled animation completion. Final corrected-tree targeted
rerun: 426 passed in 140.11s; all seven preflight guards and ten changed Python
files' Ruff/format checks pass. New-head GitHub gates remain pending; AC7 remains
open until verified merge.

Final integration checkpoint: dev advanced to 67bfde41d1 while review ran.
Rebased all 17 patches, preserving both independent lessons appends; all code
patches are identical by range-diff. Qodo's e5fde9a080 review resolved all
behavioral findings; its final draft-method documentation request is implemented.
A fresh integration run caught changed recovery-message wording; corrected the
instruction and strengthened three owner regressions. The final 145-test
owner/editor/file-picker/CSS/boot/authoring selection passes on the new base;
all 7 derived-artifact preflight checks pass. Independent review found no
Critical/Important issues and accepted the wording correction. AC7 stays open
until refreshed exact-head review/checks and verified merge; automatic follow-up
remains active. See the UAT record for exact evidence and baseline warnings.

Second Qodo/CI pass (2026-09-15): reused the existing screen-owned CSS splitter
and route loader, bringing boot CSS from 770,849 to 767,019 bytes and broad-subject
rules from 275 to 272 without raising limits. First-route, initial-route and
dialog behavior pass. Fixed verified discard races with the existing transition
lock/freshness checks, rejected deleted identity while retaining exact invalid
drafts, corrected server media output declarations, and added requested API
documentation/types plus one debounce constant. No new infrastructure or ADR;
ADR-097/138/150 apply. Coordinator 301 domain/storage tests, 150 UI/CSS tests,
20 boot-budget tests and all 7 preflight guards pass. Detailed RED/GREEN evidence
and environment qualifications: Docs/UAT/2026-09-15-workflows-qodo-remediation.md.
Independent review, new-head Qodo/CI disposition and merge remain pending; AC7
and task status remain open. User-approved automatic follow-up is active.

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

New-base integration: rebased 21 patches onto dev65a1437183, preserving both independent lesson appends; all code patches unchanged by range-diff. Qodo import-spacing comment4021886696 was verified false-positive and replied/resolved with Ruff evidence. Combined CSS exceeded the unchanged 768000-byte startup limit by490. Six Workflows geometry selectors now use existing unique pane IDs, allowing the existing splitter/loader to defer612bytes without changed declarations or infrastructure. Corrected targeted selection:182passed; startupCSS767878/768000, broadselectors271/274, all7preflight guards pass. Independent review and exact-head GitHub gates remain pending. ADR097/150 apply; detailed evidence in Docs/UAT/2026-09-15-workflows-qodo-remediation.md. AC7 remains open until verified merge.

Independent review of the new-base six-selector correction is complete: no Critical/Important/Minor findings; ready for push. Reviewer independently verified unchanged declarations, unique ID owners, route loading, generated partition, boot-byte count and mounted-fixture computed-style/geometry parity at160x48,110x36,60x20. Exact-head GitHub review/checks and verified merge still required.

Creation/API review round: Qodo4022051681/1690/1696/1704/1712 addressed. Shared Pydantic creation-name boundary (256 raw characters, strict text, trim/nonblank, preserved Unicode) runs before authoring setup and standalone-controller creation; imported/raw names remain lossless. Concise errors are painted and valid retry is tested in the actual app. Capture checks existing SVG availability with installation guidance. Paging annotations and catalog/create/import/export contracts completed. Initial regressions reproduced name/dependency failures; corrected verification:539 targeted tests,18 focused final-boundary cases,13 performance checks and all7preflight guards pass. Eight shared-module Ruff findings are exactly baseline-attributed, no new debt; eight touched Python files formatted. Independent review reports no issues. Existing ADR138 amended; full evidence in Docs/UAT/2026-09-15-workflows-qodo-remediation.md. Exact-head Qodo/CI and merge remain pending; AC7 open.

Creation portability follow-up: Qodo4022157035 verified against current tldw_server dev59049e094 (unchanged direct name TEXT binding) and a real SQLite encoding probe. Existing creation validator now rejects surrogate code points through bounded strict UTF-8 encoding, preserving valid Unicode, the raw256 limit, concise ValueError/InvalidDraft guidance, and imported/raw legacy names. RED7failed18passed; GREEN25focused and547broad targeted tests passed (4.85s/164.30s). All7preflight guards pass; two Python files formatted, tests Ruffclean and shared8baseline findings exactly source-attributed with zero new debt. Independent review found no issues. ADR138 amended; detailed evidence in Docs/UAT/2026-09-15-workflows-qodo-remediation.md. No new infrastructure, models or full-suite. Exact-head review/checks and merge remain pending; AC7 open.
<!-- SECTION:NOTES:END -->
