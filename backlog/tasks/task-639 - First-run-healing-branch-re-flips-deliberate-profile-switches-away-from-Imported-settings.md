---
id: TASK-639
title: >-
  First-run healing branch re-flips deliberate profile switches away from
  Imported settings
status: Done
assignee:
  - '@claude'
created_date: '2026-07-25 21:54'
updated_date: '2026-07-26 00:34'
labels:
  - followup
  - uat
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-641/635 review (Minor): ensure_imported_profile()'s self-healing branch (active_config.py:367-370) re-activates imported_settings on EVERY first RAG-touch-in-process where the active pointer differs from imported_settings and the profile already exists on disk -- it cannot distinguish a genuinely half-done first run (crashed before activating) from a user who deliberately switched away to a different profile afterward. A user who explicitly Set-active'd to a different profile, then restarted the app, would get silently switched back to Imported settings the next time anything touches get_shared_rag_service() for the first time in that new process. This fix should also account for cleaning up configs already damaged by the pre-635 always-import bug (a fresh user who got an unwanted imported_settings profile created and activated before this fix shipped).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The healing branch only re-activates imported_settings when the active pointer is still the default builtin (i.e. a genuine half-done first run), never when the user has since deliberately activated a different profile
- [x] #2 A migration/cleanup path exists (or is explicitly scoped out with rationale) for configs already carrying an unwanted imported_settings profile + pointer from before task-635 shipped
- [x] #3 Existing half-done-first-run healing regression coverage (test_ensure_imported_profile_heals_half_done_first_run) still passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce root cause in active_config.py:ensure_imported_profile() -- the
   healing branch fires whenever the existing imported_settings profile's
   pointer differs from imported_settings, with no way to tell a half-done
   first run apart from a deliberate later switch.
2. TDD: add a RED test proving a deliberate switch to a different (non-
   default) profile survives a later ensure_imported_profile() call
   unchanged (currently fails: gets flipped back).
3. Fix AC#1: gate the healing branch on `_active_profile_id() == DEFAULT_PROFILE`
   (pointer never successfully written by anyone) instead of `!= _IMPORTED_ID`.
   Verify the existing half-done-first-run regression test
   (test_ensure_imported_profile_heals_half_done_first_run) and the fresh-user
   no-op test stay green.
4. AC#2 cleanup: add a provable-damage-artifact detector (fingerprint-
   identical to the builtin default + no hand-set legacy search/reranking
   differences) for configs whose pointer is ALREADY imported_settings from
   the pre-635 always-import bug on a fresh install. When detected, delete
   the imported_settings profile and repoint to the default builtin (must
   delete, not just repoint, or the healing branch would recreate the
   flip-flop on the next call). Add positive (damage artifact -> healed) and
   negative (real hand-set difference / fingerprint mismatch -> left alone)
   regression tests.
5. Run Tests/RAG/test_first_run_import.py, then the full Tests/RAG/ suite;
   fix any regressions.
6. Update task ACs + Implementation Notes, mark Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: ensure_imported_profile()'s healing branch gated re-activation on
`_active_profile_id() != _IMPORTED_ID` -- true for BOTH a half-done first run
(pointer never written) and a user who deliberately activated a different
profile after import completed, so it silently re-flipped deliberate switches
back to "Imported settings" on every process's first RAG touch.

Fix (AC#1): narrowed the gate to `_active_profile_id() == DEFAULT_PROFILE`
(hybrid_basic) -- the one condition unique to "the pointer was never
successfully written by anyone yet." Any other pointer value (builtin or
user profile) is left completely alone, per the task's own AC#1 wording,
which explicitly frames "still the default builtin" as the definition of a
genuine half-done first run. No new state file was needed.

Damage cleanup (AC#2): added `_is_pre635_damage_artifact()` -- true only when
an existing "Imported settings" profile is index-fingerprint-identical to
the default builtin AND carries none of the non-fingerprint legacy
query-time/reranking fields (_LEGACY_SEARCH_KEYS + _LEGACY_PROCESSOR_KEYS,
reranking_config). When the pointer is already imported_settings and this
holds, the profile is DELETED (not just repointed -- leaving it on disk
would let the pointer==DEFAULT_PROFILE healing branch re-activate it on the
very next call, a flip-flop) and the pointer is reset to DEFAULT_PROFILE.
This only fires for configs provably damaged by the pre-635 always-import
bug on what would now be a fresh install (no legacy material, so the
snapshot is byte-for-byte the default); anything with a real hand-set
difference, a different fingerprint, or a fabricated reranking_config is
left untouched -- can't distinguish genuine choice from damage there, so it
doesn't guess (documented in the function's docstring).

Known accepted gap: a user who, after being on "Imported settings",
deliberately switches BACK to the default builtin profile is indistinguishable
from "pointer never written" using only the pointer -- they would get healed
back to Imported settings on the next first-touch. This is the same
conflation AC#1 itself defines ("pointer still default" == "genuine half-done
first run"), and the task explicitly scopes the never-flip guarantee to
"non-default profile." Avoiding it would require a separate persisted
first-run-complete marker, which the task said to avoid unless a hole was
found; this one is inherent to "no new state file" and is called out here
rather than silently accepted.

Tests: 6 new cases in Tests/RAG/test_first_run_import.py -- deliberate-switch
non-reflip (RED before the fix, GREEN after), half-done-first-run still heals
(explicit task-639 lock on the new gate condition), damage-artifact cleanup
(positive), and three negative cases (hand-set search-key difference,
fingerprint mismatch, fabricated reranking_config) proving the cleanup never
guesses. Tests/RAG/test_first_run_import.py: 22 passed (16 baseline + 6 new).
Tests/RAG/: 574 passed, 8 skipped (baseline 568/8 + 6 new).

Files: tldw_chatbook/RAG_Search/simplified/active_config.py,
Tests/RAG/test_first_run_import.py.
<!-- SECTION:NOTES:END -->
