---
id: TASK-19602
title: >-
  PR #1893 (library first-run/power-user workflows) re-broke the
  library-prompts-canvas suite
status: Done
assignee:
  - '@codex'
created_date: '2026-08-21'
updated_date: '2026-08-26 00:41'
labels:
  - ci
  - library
  - regression
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #1893 (feat(library): improve first-run and power-user workflows,
merged 2026-08-21 17:27) reintroduced failures in
`Tests/UI/test_library_prompts_canvas.py` that TASK-18611 AC#1 had just
fixed (PR #1849, merged 2026-08-20, verified 310 passing at dev@25500ad87).

Local reproduction on dev@2a15a72bb (clean worktree): **20 failed /
318 passed** in that one file, including one of the two
TASK-18611-trio tests. Distinct new failure signatures, all pointing at
#1893's workspace-registry work:

1. `WorkerFailed: AttributeError("'LibraryHarness' object has no attribute
   'app_config'")` — a worker in the production code path now reads
   `app_config` off the app (or a worker-context object) that the test
   harness app does not provide. Either production reads a missing
   attribute on a non-TldwCli host, or the harness needs the attribute.
2. New UI state leaking into tests: visible text now shows
   "Handoff · blocked until workspace registry is ready" — the prompts
   canvas waits on a workspace registry that the harness never readies,
   so `#library-prompts-delete-undo` never mounts (30s timeout, 1242
   polls) in `test_cancelled_prompt_import_retains_writer_ownership_until_commit`.
3. The remaining ~18 failures cluster around the same wait-for-settlement
   paths (receipt/undo/import-status), consistent with the same registry
   dependency stalling settlement in the harness.

PR #1893 merged with two red checks (GGUF-windows, Backlog Guard) while
its UI shards were still pending; the post-merge shard runs had not
concluded at filing time, so CI confirmation is pending — but the local
clean-worktree reproduction is deterministic.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `test_library_prompts_canvas.py` is green on a clean dev checkout (full file, single run).
- [x] #2 The `app_config` AttributeError is resolved at the correct seam (production guard or harness provision — not a blind `getattr` swallow).
- [x] #3 The workspace-registry readiness path either does not gate the prompts canvas in a host without a registry, or the harness readies it; the decision is recorded in the PR.
- [x] #4 The TASK-18611 trio stays green (no regression of PR #1849's fix).
- [x] #5 Prompt Recipe block-order tests reflect the accepted outcome-first essential/optional ordering.
- [x] #6 Collection-manager tests exercise the manager from populated or explicit empty-collection recovery states without contradicting the distilled empty-state contract.
- [x] #7 Prompt re-entry tests reflect the explicit Library landing/Continue receipt contract and still prove exact one-request scope restoration.
- [x] #8 Production-CSS pager geometry tests run in a deterministic single-Library-screen harness and remain green at both required terminal sizes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the focused Library Prompt baseline on latest dev and classify every failure as product defect, stale contract, or harness race.
2. Align stale Recipe, collection, Continue-landing, and production-CSS harness expectations with their accepted contracts; keep production behavior unchanged.
3. Run the focused failures, the complete affected test files, and static checks; document exact evidence.
4. Self-review the diff and update the task acceptance criteria and implementation notes.

ADR required: no
ADR path: N/A
Reason: This is a test-baseline repair implementing already accepted UI and persistence contracts; it introduces no architectural decision.
<!-- SECTION:PLAN:END -->

## Notes

<!-- SECTION:NOTES:BEGIN -->
Related: TASK-18611 (whose AC#1 fix this regresses), TASK-19601/19573
(guard red — separate). Bisect hint: the failures appear with #1893's
merge; `git log 2a15a72bb -- tldw_chatbook/UI/Screens/library_screen.py`
around the workspace-registry commits is the starting point.
<!-- SECTION:NOTES:END -->

## CI confirmation (2026-08-21 ~19:30Z, post-merge shard logs)

#1893's own (post-merge) shard runs confirm the local repro, with a third
signature beyond the two predicted:

1. `test_library_prompts_canvas.py` — "Prompt browse never settled"
   (result shows status=ready/error with matching scope but the wait
   predicate never satisfied) and "#library-prompt-row-25 never mounted
   within 30s (1078 polls)" with visible text "Loading page 1… Page is
   loading." — the new pagination path (page_size=20) is stalling in the
   harness.
2. `test_library_shell.py` — a cluster of Back/flush/autosave failures
   ("Back never triggered the flush save", "Back never completed after
   the in-flight autosave resolved", blank-note GC failure) — broader
   Library fallout from the same PR, NOT limited to the prompts canvas.
3. `_build_test_app() got an unexpected keyword argument
   'configured_default'` — #1893 (or a sibling same-day merge) CHANGED
   the shared test factory's signature, breaking callers across the
   suite. Also a `NoActiveAppError` in canvas kwargs separation.

Scope note: this is bigger than the prompts canvas — treat as the
#1893 Library regression umbrella; the local repro (20 failed) plus
these CI signatures are the evidence base.

## Fix progress (2026-08-21 evening, PR #1909 merged)

Two of the three signatures fixed and verified:

1. `_build_test_app` shadowing (e351a9c99) — the local wrapper in
   test_library_shell.py now forwards configured_default. CI TypeError
   signature eliminated.
2. Hidden Back control (1bda754fa, bisected) — wide-mode editor swaps
   #library-note-back for #library-notes-task-return (same guarded-exit
   seam; product contract intact). 14 old tests pressed the hidden
   button (silent no-op). Width-aware `_press_note_back` helper added;
   Back/flush/GC cluster 9/9 green. library_shell: 20 -> 7 failures.

Remaining open scope (all reproduce on pure dev; 30 total):
- test_library_shell.py x7: rail_preferences_prefers_app_config,
  conversation_canvas_sync_gates[size0/1], prompts_rail_row_exact_count,
  restore_state per-pane-filter attrs x2, note_recompose_fifty_cycles.
- test_library_prompts_canvas.py x20 (shared-editor/read-only-preview
  family, kwargs separation, settle/pagination waits).
- test_library_file_notes_workspace.py x3 (source choices keyboard
  switch, database-files switch retains canvas).

AC#2 and AC#3 are satisfied (the AttributeError and the workspace-registry
gate were the same two root causes). AC#1/AC#4 remain open on the 30
above. CI note: org-wide runner congestion prevented guard/test checks
from starting on PRs #1900/#1906/#1909; all merged on exact-expression
local verification.

## Fix progress tranche 2 (2026-08-21 night, PR #1912 merged)

Eleven more failures cleared (7 library_shell + 4 prompts/kwargs), incl.
ONE real product bug: _library_prompt_write_worker_is_active (355de75da)
crashed headless callers with NoActiveAppError because it touched the
screen-owned worker manager that only exists on a mounted screen; the
guard now keeps the app-owned half of the scan. The other ten were
test-side drift to current contracts (a24a2202f's raw-content previews,
3aef9bcd1's distilled empty states, c9ac43eff's scope-era restore,
11b2fa700's lifecycle reader). Methodology note recorded: fixing the
compiled-preview drift on the production side first BROKE 10 newer
tests -- the full-suite differential caught it and inverted the
direction. Deterministic totals: shell 7 -> 0; prompts+file_notes
23 -> 18.

Remaining open: prompts_canvas x17 + file_notes x1 deterministic, plus
the shell's 3 order-flakes (media pager size0 / initial-error /
page-error) and 2 file_notes order-flakes.

## Fix progress tranche 3 (2026-08-21 late, PR #1915 merged)

4 more cleared incl. ONE real product bug:
_read_library_prompt_editor_fields preferred the persisted detail over
the live legacy-lane TextAreas, so unsaved edits silently dropped out
of copy/save/export. Now live-if-non-empty wins; structured/foreign
artifacts (lanes mount empty, truth in STRUCTURE) keep the detail
compatibility fallback -- the naive live-always form blanked those
copies and the in-branch differential caught it pre-merge. Plus 3
label expectations aligned to a24a2202f's relabels.

Running totals: 41 of the original ~50 failures cleared across tranches
1-3; prompts+file_notes unique failing tests 19 -> 16.

Remaining-16 diagnostic map for the next tranche:
- copy x2 unsaved-lane (11492-class): the a24a2202f editor republishes
  block state on a system-lane change and reverts a programmatic
  user-lane .text write (block_state non-None even for legacy); needs
  the editor's real edit seam (_change_field / per-lane publish), not
  textarea assignment.
- double-notify x6: 'Library tools are now available.' (4aa59c20a
  lifecycle graduation toast, intended) precedes the expected toast in
  single-call assertions -- tests need the lifecycle toast filtered or
  the graduation settled before the action.
- NoMatches x3: #library-row-browse-prompts x2 (rail collapsed/renamed
  at these sizes?) and #library-prompts-export (button gated or moved).
- app_config on LibraryHarness WorkerFailed x1 + file_notes x3
  (source-choices keyboard, database-files return, False-is-True x2).

## Fix progress tranche 4 (2026-08-21/22 night, PR #1917 merged)

11 more cleared; unique failing tests across both suites 16 -> 5. The
unlock: the pytest sandbox factory builds a NEW Library profile
(lifecycle UNKNOWN -> 2-row landing rail) while bare-probe runs read
the real config's graduated lifecycle -- which is why the rail-row
strand never reproduced outside pytest. Legacy-profile
_build_test_app wrappers installed in test_library_prompts_canvas.py
and test_library_file_notes_workspace.py (mirroring
test_library_shell.py). Plus: export-refusal empty-CTA seeding, five
notify count-asserts relaxed to latest-call (4aa59c20a graduation
toast precedes legitimately), discard clean-state aligned (always
enabled under a24a2202f).

Remaining 5, with root-cause evidence:
- copy-lane x2: a24a2202f editor republishes block state on a
  system-lane change and reverts a programmatic user-lane .text write.
- db-files switch: workspace stuck in save_state=conflict AFTER the
  test's confirmed reload + successful flush (spy:
  _flush_active_file_notes sees conflict and vetoes every return) --
  needs the conflict-clear path after confirmed reload root-caused;
  possibly a real product bug.
- source-choices keyboard (sizes 1/2): keyboard switch path.
- cancelled_prompt_import: order-dependent with the trio fix's
  drain semantics (18611 notes).

## Fix progress tranche 5 (2026-08-22, PR #1921 merged)

2 more cleared (5 -> 3): both were the wide focused-task contract
(1bda754fa) reaching the file-notes suite -- the db-files switch test's
post-reload Database press never entered its handler (spy: zero handler
entries) because wide terminals hide the source strip behind
#library-notes-task-return; the test now presses the live control. The
source-choices keyboard test aligns its wide-size rail assert and
returns via task-return; the compact size keeps the strip flow.

Remaining 3, with evidence:
- cancelled_prompt_import retry Undo: post-settlement probe shows
  receipt set, mutation_in_flight False, mutation_status '' -- yet the
  receipt row (and Undo) is NOT composed; the canvas recompose after
  the import's browse projection drops the row despite
  delete_receipt in kwargs. Next: instrument
  _library_prompts_canvas_kwargs timing vs the recompose that follows
  import settlement; likely a state-object refresh ordering gap
  (possibly a real product bug).
- copy-lane x2: the a24a2202f editor republishes block state on a
  system-lane change and reverts a programmatic user-lane .text write
  (block_state non-None even for legacy prompts). Needs the editor's
  real edit seam (_change_field / per-lane publish), not textarea
  assignment.

## Final tranche (2026-08-22, PR #1944 merged) — tracked failures cleared

All three remaining fixed:
- cancelled-import retry Undo: the Library source-snapshot worker
  crashed (AttributeError: 'LibraryHarness' has no app_config) once the
  test swapped screen.app_instance to the harness; the crash aborted
  the post-import canvas sync so the receipt row never composed.
  LibraryHarness now shares the real app's app_config. Diagnostic
  lesson recorded: a spy on the snapshot coroutine PERTURBED the bug
  away by stripping @work (the coroutine was never awaited -> no
  worker -> no crash) -- instrumenting decorated methods requires
  re-wrapping through the same decorator.
- copy-lane x2: per-lane settle pacing (each programmatic lane write's
  publish/preview-sync must settle before the next write).

Six tranches total: ~50 deterministic failures -> 0 tracked, across
PRs #1909/#1912/#1915/#1917/#1921/#1944, with 3 real product bugs
fixed (headless kwargs NoActiveAppError, hidden-Back/control no-ops in
wide focused-task mode x2 surfaces, prompt-field reader dropping
unsaved edits) and every tranche differentially verified.

Follow-up (NEW, from today's dev movement at e52e66c82, fails on
pristine dev): test_library_prompts_restored_create_row_list_dispatches
_browse_once and test_library_prompts_fresh_reentry_refetches_the_
applied_scope_once -- dispatch/refetch counting; plus the known
cross-suite order flakes (combined three-file runs surface them; CI
shards files separately). File under TASK-18610's remainder or a new
task.

## Implementation Notes (2026-08-25 baseline closeout)

- Aligned the Recipe test with the accepted outcome-first ordering: essential user blocks precede optional Success criteria.
- Kept collection-manager coverage honest under the distilled empty-state contract by seeding the paging case and exercising the empty-collection “All prompts” recovery path before reopening the manager. The recovery wait now observes the applied browse scope, removing a requested/applied race.
- Reframed saved Prompt re-entry around the explicit Library landing/Continue receipt contract: non-resumable create state stays on the landing page, while an authoritative applied Prompt scope continues and refetches exactly once.
- Moved Prompt pager/focus integrations from the full `TldwCli` startup lifecycle to the single-Library production-CSS harness. This preserves the exact stylesheet and compositor assertions without allowing unrelated default-route/Console mounts to replace the screen under test.
- Verification: the nine formerly failing parametrized cases pass together; the original eight-file baseline reached 682/683 with only the subsequently removed full-app routing race; the complete `test_library_prompts_canvas.py` file then passed 338/338; the four changed pager/focus cases passed 4/4; Ruff and `git diff --check` passed.
- ADR check: no ADR required. The changes enforce existing Recipe, empty-state, Continue, and harness contracts and introduce no architecture or product behavior.
