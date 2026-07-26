---
id: TASK-639
title: >-
  First-run healing branch re-flips deliberate profile switches away from
  Imported settings
status: Done
assignee:
  - '@claude'
created_date: '2026-07-25 21:54'
updated_date: '2026-07-26 00:58'
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

--- Review round 2 (Critical, reviewer-reproduced) ---

Finding: _is_pre635_damage_artifact()'s "no hand-set difference" allow-list
(_LEGACY_SEARCH_KEYS + _LEGACY_PROCESSOR_KEYS) did not cover every field the
Settings screen editor (apply_defaults_to_profile,
settings_rag_profile_adapter.py:150-167) can hand-tune -- e.g.
search.hybrid_alpha (default 0.7), default_search_mode, citation_style,
snippet_max_chars, max_context_size, fts_top_k, vector_top_k,
embedding.batch_size. None of those are index-determining (fingerprint
still matched the default) and none were in the allow-list, so a profile
customized in ONLY one of those fields still looked "provably safe" and was
PERMANENTLY DELETED. Reproduced with a RED test
(test_ensure_imported_profile_never_deletes_settings_screen_customization)
confirmed failing against the round-1 code before applying the fix below.

Decision: replaced the content-comparison heuristic entirely with a durable
marker, [rag.service].first_run_import_done (_first_run_import_done /
_mark_first_run_import_done), written at BOTH places
ensure_imported_profile() ever deliberately activates imported_settings (the
fresh-import path and the half-done-first-run healing path). Once set, the
pointer is never re-evaluated again, regardless of what it names -- this
also closes the previously-disclosed "switch back to the default builtin"
gap from round 1, since the marker (not the pointer's current value) is now
the source of truth for "was this deliberate".

For a config whose imported_settings pointer/profile predates the marker
(marker absent, existing profile, pointer != DEFAULT_PROFILE): chose
reviewer's option (b) -- NEVER delete. The marker is simply set and both the
profile and pointer are left completely untouched. Rationale: no finite
content check can prove "no customization exists" (that's exactly what broke
in round 1), so a compare-and-maybe-delete strategy can always be defeated by
some field nobody thought to check; leaving it as-is is unconditionally safe
(zero risk of destroying real user data) and reaches the same practical
end-state (the flapping stops, the pointer is confirmed and never
reconsidered again). Only the true half-done case (existing profile, no
marker, pointer STILL the default builtin -- i.e. never successfully written
by anyone) completes the activation, matching AC#1's own framing of that
condition.

_LEGACY_MERGED_SEARCH_FIELDS and _is_pre635_damage_artifact() were removed
entirely (no longer needed -- nothing compares content anymore); the
`fingerprint_collection` import in active_config.py was removed along with
it (dead post-removal).

Tests: replaced the 3 round-1 "does_not_delete_*" negative tests and the
delete-based "heals_pre635_damage_artifact" test (now factually wrong) with:
test_ensure_imported_profile_never_deletes_settings_screen_customization
(the reviewer's repro, RED-then-GREEN),
test_ensure_imported_profile_adopts_preexisting_imported_pointer_without_deleting,
test_ensure_imported_profile_marker_present_never_touches_pointer_even_back_to_default,
test_fresh_user_leaves_marker_unset. Added marker assertions to the 3
existing tests that activate imported_settings
(test_first_run_creates_imported_profile_and_sets_active,
test_ensure_imported_profile_heals_half_done_first_run,
test_ensure_imported_profile_swallows_save_failure). Updated the shared
_wire() fixture's save_setting_to_cli_config fake to route by `key` (profile
vs first_run_import_done) into separate ptr["v"]/ptr["marker"] slots instead
of one shared value, so the two writes can no longer silently clobber each
other in tests.

Gates: Tests/RAG/test_first_run_import.py: 22 passed. Tests/RAG/: 574 passed,
8 skipped (same totals as round 1 -- test count is unchanged, only which
scenarios are covered).

Known accepted gap (unchanged from round 1, now narrower): a config that
predates the marker AND whose pointer happens to already read as the default
builtin is still indistinguishable from "never completed" -- this can only
occur once, on the first marker-aware run, since the marker is written at
every activation site from then on.

Files: tldw_chatbook/RAG_Search/simplified/active_config.py,
Tests/RAG/test_first_run_import.py.
<!-- SECTION:NOTES:END -->
