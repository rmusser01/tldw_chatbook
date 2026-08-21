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
