---
id: TASK-19602
title: >-
  PR #1893 (library first-run/power-user workflows) re-broke the
  library-prompts-canvas suite
status: In Progress
assignee: []
created_date: '2026-08-21'
updated_date: '2026-08-21'
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
- [ ] #1 `test_library_prompts_canvas.py` is green on a clean dev checkout (full file, single run).
- [ ] #2 The `app_config` AttributeError is resolved at the correct seam (production guard or harness provision — not a blind `getattr` swallow).
- [ ] #3 The workspace-registry readiness path either does not gate the prompts canvas in a host without a registry, or the harness readies it; the decision is recorded in the PR.
- [ ] #4 The TASK-18611 trio stays green (no regression of PR #1849's fix).
<!-- AC:END -->

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