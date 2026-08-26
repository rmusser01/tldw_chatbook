---
id: TASK-22216
title: Move the actor-pack staging sweep out of TldwCli.__init__
status: Done
assignee:
  - '@claude'
created_date: '2026-08-24'
updated_date: '2026-08-25 21:47'
labels:
  - performance
  - startup
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22216).

PR #1998 (`ac1037732`) put synchronous filesystem work back into construct — the class
task-21106 removed: `app.py:7322` constructs `ActorPackImportService`, whose `__init__`
ends with `self.sweep_staging()` (`Actor_Packs/importer.py:216`). The sweep runs
`secure_private_directory(..., create=True)` — a per-component walk from `/` with
`os.open`+`fstat` owner/mode checks (`Utils/private_paths.py:995-1030`) — then `os.scandir`
over up to 32 staging candidates, each with lstat + two O_NOFOLLOW opens (`importer.py:
218-255`, `:1313`), every boot, before the event loop exists. Small-medium warm; medium on
network/FUSE homes or with residue. The boot-files guard cannot see it (it asserts six DB
filenames, and this is a directory).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `TldwCli.__init__` performs no staging filesystem I/O (probe or guard asserts the sweep is not reached from construct)
- [x] #2 The sweep runs on first import-feature use or a deferred worker; crash-recovery semantics preserved (staging residue still cleaned within the session)
- [x] #3 The boot-time guard is extended to cover this class (construct-time filesystem side effects), with its blind spots stated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Remove the eager sweep_staging() call from ActorPackImportService.__init__ (Actor_Packs/importer.py) — construction records paths only.\n2. Add a thread-safe once-gate ensure_staging_swept() on the service (lock + success-latch); gate the entry of inspect_archive (the only staging-creating operation) so a pre-deferred-worker import still sweeps first and the lock serializes against the deferred worker.\n3. Add app-level ensure_actor_pack_staging_sweep() (mirrors ensure_actor_pack_recovery) and kick it as a thread worker from _schedule_deferred_startup_work.\n4. Red-first probes: subprocess construct-time probe counting secure_private_directory/os.scandir traffic attributable to the staging sweep (nonzero today, zero after) + boot-guard extension asserting the staging directory is not created at construct, with blind spots documented.\n5. Update the two construct-time sweep tests in Tests/Actor_Packs/test_actor_pack_import.py to the new seam; add once-gate/serialization/failure-retry unit tests.\n6. Targeted suites + --collect-only sweep + preflight; mutation tests (re-add eager sweep -> probe reds; remove once-gate latch -> double-sweep test reds).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Moved the actor-pack staging sweep off the construction path (the task-21106 class, reintroduced by PR #1998).

**Approach.** `ActorPackImportService.__init__` is now pure — it records the staging/profile paths and initializes a threading.Lock + latch; the eager `self.sweep_staging()` is gone. A new `ensure_staging_swept()` runs the sweep lazily-once per service instance: the gate latches only on success (a failed attempt raises the same `actor_pack_import_cleanup_denied` and the next caller retries), and the lock serializes the two racers — the deferred startup worker and a first `inspect_archive` — so the sweep always completes before the session's first candidate exists and can never remove an in-flight one. App side: `ensure_actor_pack_staging_sweep()` (mirrors the task-21106 recovery seam; absorbs+logs failure with category tokens only per TASK-15103) kicked as its own thread worker from `_schedule_deferred_startup_work`; `inspect_archive` gates on the same once-lock, whichever fires first.

**Pre-sweep-use safety.** An import that starts before the sweep never depended on construct-time work: `inspect_archive` runs `_preflight_space` -> `secure_private_directory(create=True)` (raises unless verified-private) BEFORE `_create_candidate`, so directory creation + hardening are intrinsic to the operation path. All other staging readers (`read_portrait_preview`, `cleanup_review`, `revalidate_review`, activation) require a review minted by `inspect_archive` on the same instance, so the gate is already latched.

**Failure semantics.** The old sweep failure was NOT absorbed — it propagated out of `TldwCli.__init__` and aborted boot. New: deferred-worker failure is logged (gate stays open), first-use failure surfaces the categorized `ActorPackImportError` to the import caller — strictly gentler, deliberate, documented in both docstrings.

**Guard.** New `Tests/App/test_boot_construct_fs_side_effects.py`: subprocess boot with counters at the importer seam (secure_private_directory + staging-scandir) asserting zero construct-time staging I/O and no `actor_pack_imports/` directory; blind spots stated honestly (fixed-list tripwire, seam-scoped counters, name-matched scandir filter, construct-only, TLDW_TEST_MODE). Red-first: pre-fix counts were 1/1 + directory created; post-fix 0/0.

**Verification.** Targeted suites green (Actor_Packs 208; seams+guards 14; import review 5; runtime ownership 17; startup perf + probe-launch-wake 20). Mutations both caught: eager sweep re-added -> 3 probes red; once-gate removed -> 3 tests red incl. a later import sweeping away an in-flight candidate. Preflight all green (diagnostic inventory regenerated for the two reviewed, token-only warning calls). Full --collect-only: 59,426 collected; 28 pre-existing optional-deps errors (numpy/audio/TTS), none in touched areas.

**Files.** tldw_chatbook/Actor_Packs/importer.py, tldw_chatbook/app.py, Tests/App/test_boot_construct_fs_side_effects.py (new), Tests/UI/test_actor_pack_staging_sweep_seam.py (new), Tests/Actor_Packs/test_actor_pack_import.py, Docs/security/production-diagnostic-inventory.json.
<!-- SECTION:NOTES:END -->
