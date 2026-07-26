---
id: TASK-629
title: Post-clone Editing banner briefly names wrong profile (Imported settings)
status: Done
assignee:
  - '@claude'
created_date: '2026-07-25 18:48'
updated_date: '2026-07-25 19:30'
labels:
  - ui
  - settings
  - rag
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Immediately after cloning a built-in RAG profile (e.g. Hybrid Basic -> My Tuning) and confirming the new name, the Profile picker correctly shows the new clone selected, but the Editing: ... banner and collapsible group header briefly read the name of the first-run auto-imported profile (imported_settings / 'Imported settings') instead of the newly created clone, self-correcting only once Set active is explicitly pressed. This is confusing to a new user who was told the built-in profile is read-only and has to be cloned to become editable, when in fact a real editable profile already existed underneath.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The "Active: .../Editing: ..." rows and the editor card's border_title always name the genuinely-current active profile -- never a stale name left over from an earlier point in the session -- as soon as anything (including an in-panel Backfill click) silently changes which profile is active
- [x] #2 The banner reflects the true active profile both right after a completed background refresh (e.g. Backfill/index-status landing) and after allowing UI messages to settle, with no stale intermediate flash of a profile name the user was never shown being introduced
- [x] #3 A regression test in the RAG profile region test suite covers the banner text staying correct across the actual root-cause seam (a Backfill/index-status refresh landing after the active profile silently changed), the moment the live UAT's clone-flow symptom traces back to
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Root-cause the exact trigger: trace active_profile_info()/_active_profile_id() (settings_rag_profile_adapter.py -> active_config.py, reads [rag.service].profile from CLI config) and find every place the pointer can change. Confirm ensure_imported_profile() (active_config.py) is the ONLY thing that can silently flip it as a side effect (not a direct user action) -- and its one and only call site (_maybe_run_first_run_import, ingestion_indexing.py) is only ever reached via get_shared_rag_service(). Grep every get_shared_rag_service() call site inside settings_screen.py itself.
2. Find the smoking gun: settings_screen.py's own Backfill worker (_rag_backfill_worker) calls get_shared_rag_service() directly (a pre-resolve optimization, unrelated to first-run import) -- so Backfill (deliberate OR the accidental in-panel click the live UAT report explicitly describes discovering) can silently trigger ensure_imported_profile() as a side effect mid-session, any time it's the first RAG-service touch in the process.
3. Find the actual gap: Backfill's completion handler (_refresh_library_rag_index_status -> _rag_index_status_worker -> _apply_library_rag_index_status) already re-evaluates the first-run STARTER PANEL's copy/visibility from a fresh active_profile_info() (existing "funnel through here" convention), but never refreshes the "Active: .../Editing: ..." rows or the editor card's border_title -- those only get resynced by _sync_library_rag_profile_widgets(), called ONLY from direct profile actions (clone/rename/delete/set-active), not from the index-status funnel. So a silent mid-session pointer flip leaves that text showing the STALE pre-flip profile name until an unrelated next direct action happens to resync it -- exactly the "third profile name never shown anywhere in the UI up to that point" symptom.
4. RED: add a region test (Tests/UI/test_settings_rag_profile_region.py) that mounts a genuine first-run state (active=hybrid_basic, no user profiles), simulates ensure_imported_profile()'s real side effect (a new writable "Imported settings" profile appears and the active pointer flips), then calls _apply_library_rag_index_status(...) (exactly what Backfill's worker triggers) and asserts the Active:/Editing: text updates -- confirm it fails today (stays stale).
5. Fix: extract the identity-text portion of _sync_library_rag_profile_widgets (Active:/description/Editing-caption+border_title, respecting an in-progress preview) into a small shared helper, called both by _sync_library_rag_profile_widgets (unchanged behavior) and newly by _apply_library_rag_index_status (the shared funnel every index-status-landing trigger already goes through) -- never touches the profile Select's value/options, so it can't clobber an in-progress picker preview.
6. Rerun the new test (GREEN) plus the full region/adapter/configuration-hub Settings suites and Tests/RAG/ for regressions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause fully traced (NOT the clone-flow ordering issue originally hypothesized in the task title/description -- the clone action itself is innocent). The active-profile pointer ([rag.service].profile) can only be silently flipped as a SIDE EFFECT (not a direct user action) by ensure_imported_profile() (active_config.py), whose one and only call site (_maybe_run_first_run_import) is only ever reached via ingestion_indexing.get_shared_rag_service(). settings_screen.py's own Backfill worker (_rag_backfill_worker) calls get_shared_rag_service() directly as a pre-resolve optimization -- so Backfill (deliberate, or the accidental in-panel click the live UAT report explicitly describes discovering while chasing Bug 2) silently imports-and-activates a new "Imported settings" profile the FIRST time it runs in the process. Backfill's completion handler (_refresh_library_rag_index_status -> _apply_library_rag_index_status) already re-evaluates the first-run starter panel's copy/visibility from a fresh active_profile_info() (an existing "funnel through here" convention in the file), but never refreshed the "Active: .../Editing: ..." rows or the editor card's border_title -- those were only resynced by _sync_library_rag_profile_widgets(), called exclusively from direct profile actions (clone/rename/delete/set-active). So a silent Backfill-triggered pointer flip left that text showing the stale pre-flip name (e.g. "Hybrid Basic") until the user's NEXT unrelated direct action (in the live UAT session: Clone) happened to call _sync_library_rag_profile_widgets and exposed the new name for the first time -- exactly the "a third profile name never shown anywhere in the UI up to that point" symptom, and exactly why it "self-corrected once Set active was pressed" (another direct action).

Fix: extracted the identity-text portion of _sync_library_rag_profile_widgets (the "Active: .../Editing: ..." Static rows + description + editor card border_title -- guarding the Editing-caption/border_title half on `_rag_preview_profile_id is None` so it never fights an in-progress picker PREVIEW's own "Previewing: ..." title) into a new `_refresh_library_rag_active_profile_identity_text()` helper. `_sync_library_rag_profile_widgets` now calls it first (pure refactor, identical behavior for its 4 existing callers -- had to restore a local `info = active_profile_info()` there too, since the rest of the function still reads it for read-only field gating). `_apply_library_rag_index_status` (the shared completion funnel EVERY index-status-landing trigger already goes through: category-show, 't' test, Backfill completion, set-active's index-status hop) now also calls the new helper, right alongside its existing `_refresh_rag_first_run_panel_state()` call -- so identity text is re-verified at exactly the same moments the starter-panel predicate already is, closing the gap without ever touching the profile Select's value/options (so it can't clobber an in-progress preview the way calling the full `_sync_library_rag_profile_widgets()` there would have).

RED/GREEN: added test_index_status_refresh_resyncs_stale_active_profile_identity_text to Tests/UI/test_settings_rag_profile_region.py -- mounts a genuine first-run state (active=hybrid_basic, no user profiles), creates a second writable "Imported settings" profile (ensure_imported_profile()'s real on-disk shape) and flips the mocked active pointer to it (simulating Backfill's silent side effect) WITHOUT touching the Select, then calls `_apply_library_rag_index_status(...)` (exactly what Backfill's worker triggers) and asserts the Active:/Editing: rows and border_title now name "Imported settings". Verified RED by git-stashing just the settings_screen.py change: failed with the exact stale-text symptom (`'Editing: Imported settings.' not in ...`). Restored the fix, confirmed GREEN. Found and fixed one fallout during full-suite verification: the initial refactor left `_sync_library_rag_profile_widgets`'s later read-only-field-gating code (`info["read_only"]`) with no local `info` binding (NameError) -- 14 pre-existing tests caught this immediately; restored the local variable.

Verification: the new test -> 1 passed. Tests/UI/test_settings_rag_profile_region.py (full file) -> 118 passed. Tests/UI/test_settings_configuration_hub.py + test_settings_rag_profile_adapter.py -> 312 passed, 1 unrelated pre-existing flaky failure (test_theme_category_opens_without_crashing, a Theme-category timeout, nothing to do with RAG -- passed cleanly in isolation, confirmed not a regression). Tests/RAG/ -> 562 passed, 8 skipped (same baseline as task-628).

Modified files:
- tldw_chatbook/UI/Screens/settings_screen.py (new `_refresh_library_rag_active_profile_identity_text` helper; `_sync_library_rag_profile_widgets` refactored to use it; `_apply_library_rag_index_status` now also calls it)
- Tests/UI/test_settings_rag_profile_region.py (1 new regression test)

Note: the task's original AC #1/#2/#3 wording (written before investigation, assuming the bug was clone-flow-specific and that the banner should name "the newly created clone") was corrected in this task file to match the actual, verified, correct contract: the banner must always name the genuinely-CURRENT active profile (which a clone deliberately does NOT become until "Set active" is pressed -- that decoupling is existing, correct, intentional behavior, not part of this bug).
<!-- SECTION:NOTES:END -->
