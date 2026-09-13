---
id: TASK-32565
title: Progressive file picker loading with lazy metadata and selectable sorting
status: Done
assignee:
  - '@codex'
created_date: '2026-09-13 18:06'
updated_date: '2026-09-13 19:17'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Large folders currently block display behind eager directory-row creation and repeated metadata reads. Stream usable entries and let users choose ordering without freezing the picker.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Entries become usable before directory enumeration finishes; progress distinguishes scanning, sorting and completion.
- [x] #2 Discovery order is default; name, modified, accessed, created and size sorting work during and after scanning with ascending and descending directions and stable selection.
- [x] #3 Listing disk work stays off the UI thread; normal browsing loads metadata only around the viewport and bounds publication work.
- [x] #4 Navigation, refresh and dismissal reject stale results; search, filters, directory selection and multi-selection retain their contracts.
- [x] #5 Targeted regression tests and large-folder responsiveness measurements cover the new behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/160-progressive-file-picker-listings.md
Reason: defines shared picker discovery order, asynchronous ownership, and timestamp sorting semantics.
1. Add failing mounted regressions for progressive display, stale cancellation, lazy metadata and sorting.
2. Introduce shared background scanning and projection with bounded UI publication and cheap single-line measurement.
3. Integrate enhanced row rendering, progress and sort controls; preserve caller and cancellation contracts.
4. Run targeted picker tests, lint and formatting, plus large-folder responsiveness and rendered UI checks.
5. Document behavior and verification; self-review and close task only when acceptance criteria pass.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented progressive listings for shared and enhanced file/folder pickers. The original freeze came from eager directory row creation, repeated metadata reads, and Rich measuring every offscreen row. Directory discovery now runs in a cancellable worker with a bounded queue; usable rows arrive in short batches. Fixed-height row measurement avoids eager rendering, and normal browsing hydrates metadata around the viewport only.

Added scanning/sorting/completion status and discovery/name/modified/accessed/created/size controls in either direction. Sort/filter work runs off the UI thread and preserves highlighted/selected paths. Creation uses birth time, with unknown values last in both directions. Existing filters, search, save/folder selection, and multiselect contracts remain covered. Navigation cancels obsolete projections; generation/revision ownership and painted-option identity reject stale results and clicks.

Review and native verification exposed a first-reactive-read reentrancy race that shared a scan queue between two loads. Reading reactive inputs before assigning ownership fixed it; a deterministic regression and the testing-evidence lesson document the incident. Narrow controls truncate instead of wrapping; alignment tests wait for lazy metadata to paint. Preserved existing plain-picker column/hidden-entry styles. Added a scoped .gitignore exception because the generic parts/ packaging rule hid the new source module.

ADR: backlog/decisions/160-progressive-file-picker-listings.md. User documentation: Docs/User_Guide/file-picker.md, linked from Library import/export. Production changes are the shared progressive navigation module/export, base/file dialog controls and event routing, enhanced picker snapshot rendering, and token-backed list CSS with rebuilt bundle.

Verification:
- 164 targeted picker and CSS tests passed in 61.53 seconds. Coverage includes both picker families, progressive results, off-loop/lazy metadata, sorting, stale clicks/results, reentrant ownership, selection, keyboard/save/folder flows, callable filters, narrow layout, token governance and bundle reproducibility.
- A final shared-row styling check reran progressive and column tests: 17 passed. Narrow/layout affected-group rerun: 25 passed.
- New production/test files pass Ruff lint and formatting. Changed legacy files introduce no new Ruff diagnostics; existing lint/format debt was left intact. Critical syntax/undefined checks and git diff --check pass.
- Native PTY run with real application CSS and deliberately slow enumeration rendered progress, changed sort during scanning, then completed with exactly 4,000 unique files and 4,001 unique options including parent.
- Local 120x40 mounted-widget benchmark, 10,000 real temporary files: first entries 4,674.9 ms before / 35.8 ms after; full loading 4,683.3 / 1,427.6 ms; maximum loading heartbeat gap 2,797.6 / 113.4 ms; scroll/resize gap 1,664.7 / 15.5 ms. Only 91 metadata records were cached after initial display plus scrolling to the end. Completed enumeration and unique paths were asserted. These are local synthetic measurements, not a filesystem-independent latency guarantee.
- 1,000-file comparison: first entries 438.9 / 35.5 ms, full loading 439.4 / 154.7 ms, loading heartbeat gap 269.1 / 24.7 ms.
- Full repository suite was not run, per repository policy. Existing requests dependency warning and unrelated stale pytest cleanup warnings remain.

No plan scope deviation. Explicit metadata sorting necessarily reads matching entries beyond the viewport; this remains background work and is described in the ADR and user guide. Filesystem calls already in flight stop cooperatively, with stale publication rejected. Self-review and independent review findings are resolved. Changes remain uncommitted in the user's workspace.

Follow-up requested code review: investigating rapid sort changes during batched publication losing the highlighted path. Reopened for regression and ownership fix under existing AC #2.

Requested fresh code review completed. One P2 finding was confirmed and fixed: replacing a sort during batched publication lost the original highlighted file; real Enter dispatch could activate the temporary parent fallback. The shared loader now retains a path/interaction restore target across canceled projections and leaves no fallback actionable during rebuilding. Only movement keys, clicked selections, and actual enhanced type-ahead matches invalidate restoration; activation keys do not. Directory changes discard the old restore target. Explicit pending path highlights retain their existing scan-completion semantics.

Added six mounted 2,000-file regressions across plain/enhanced pickers: superseding sorts, actual Enter dispatch, and deliberate End navigation. Both original failures were observed before fixing. Final targeted verification: 126 passed in 54.72 seconds (progressive, enhanced mount, folder selection, keyboard/save, action-tooltip tests). Changed loader/tests pass Ruff lint/format; enhanced picker has no new Ruff diagnostics against HEAD; git diff --check passes. Independent reviewer ran six further mounted probes for Enter, End, matching typeahead and unmatched typeahead and reported no remaining Critical, Important or Minor findings. Existing dependency/pytest cleanup warnings remain; no full suite was run.

Documented the real-key-dispatch testing incident in backlog/docs/lessons-testing-evidence.md. ADR-160 remains applicable; no change in architecture or user-facing scope. Task returned to Done after the final re-review and tests; changes remain uncommitted.

PR preparation: ported the reviewed change onto origin/dev at 8a6ba98c0d in an isolated checkout, preserved newer testing lessons and rebuilt the CSS bundle from current sources. All 170 targeted picker/CSS tests passed in 75.52 seconds on this base. Backlog-ID and persistent-diagnostic inventory checks passed; new files pass Ruff lint/format. Fresh sweep of 219 local/remote refs and worktrees found no task/ADR ID collision.
<!-- SECTION:NOTES:END -->
