---
id: TASK-38990
title: "Architecture guards catch product drift on dev: a stock ProgressBar, ungrouped exclusive workers, raw loader awaits and two boundary imports"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 01:42
updated_date: 2026-11-06 04:44
labels:
- follow-up
- ui
- architecture
priority: medium
references:
- tldw_chatbook/UI/Screens/meetings_screen.py
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/UI/Screens/watchlists_collections_screen.py
- tldw_chatbook/Actor_Packs/contracts.py
- tldw_chatbook/Persona_Visual/builtin_pixel_migu.py
- Tests/Architecture
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05. A serial `Tests/Architecture` run (1044 s, load 7-10) failed five guards, each on real product drift. No PR lane runs these guard files, which is why the drift reached dev.

The five guards and what they caught:
- **Progress guard:** a stock `ProgressBar` at `tldw_chatbook/UI/Screens/meetings_screen.py:272-273`, dating from 9fe7f34a14 (2026-09-04).
- **Ungrouped exclusive workers,** 7 sites:
  - `chat_screen.py:7824`
  - `watchlists_collections_screen.py:1647`
  - `SmartContentTree.py:192`
  - `ChatbookImportWizard.py:662`
  - `console_goal_setup_modal.py:287`
  - `workspace_change_review.py:121`
  - `workspace_create_modal.py:548`
- **Raw loader awaits,** 3 sites:
  - `schedules_workbench._request_tasks_refresh._load`
  - `watchlists_collections_screen._follow_coordinated_briefing`
  - `_request_briefings_refresh.refresh`
- **Actor packs boundary:** `tldw_chatbook/Actor_Packs/contracts.py` imports `tldw_api.character_persona_schemas`.
- **Persona visual boundary:** `tldw_chatbook/Persona_Visual/builtin_pixel_migu.py` imports `Character_Chat.builtin_pixel_migu`.

None of the branches due to merge before this one touches these files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The meetings screen uses the repo's pausable progress bar instead of a stock ProgressBar
- [x] #2 Every listed exclusive worker names a group, and each listed refresh dispatches through its loader-group helper
- [x] #3 Actor_Packs and Persona_Visual no longer import the forbidden packages, or each boundary records a reviewed exception with its reason
- [x] #4 The five guard nodes pass on dev (red before: 5 of 5 at 29592ecc9b), and each still fails when its drift is reintroduced (negative control)
<!-- AC:END -->

## Implementation Plan

1. Re-confirm the 5 guard reds at base e8eeae7db9.
2. Fix the product at each site: house progress bar in Meetings, a work-named group at each ungrouped exclusive worker, and the coordinated briefing follower dispatching through `_request_briefings_refresh`.
3. Where a guard is wrong, fix the guard without losing its intent and pin the limit of the fix with a synthetic negative control.
4. Boundaries: remove each forbidden import, or record a reviewed exception that stays pinned.
5. Negative controls: reintroduce each drift; run the covering tests.

## Implementation Notes

**Product fixes.**
- **Progress guard.** Meetings level meters `#meetings-level-mic` / `#meetings-level-sys` are now `PausableProgressBar`. `pausable_progress` is already boot-loaded (splash), so the screen's pre-import payload is unchanged. No visible change.
- **Ungrouped exclusive workers.** Each site now names a group after its work:
  - `chat_screen._console_environment`: the injected `run_worker` lambda now requires `group` (keyword-only, passed through explicitly). Its 3 `ConsoleEnvironmentController` callers already pass one; the spread can no longer drop it silently. Line count unchanged. This also turns `Tests/UI/test_chat_screen_worker_groups.py::test_every_exclusive_worker_on_chat_screen_names_a_group` green; it was red at base.
  - `wl-daily-report-banner`
  - `smart-content-tree-load`
  - `chatbook-import-conflict-scan`
  - `goal-launch`
  - `workspace-readiness-poll`
  - `workspace-create-binding-retry`
  
  Each was the only ungrouped exclusive worker on its node, so self-exclusion is kept. It no longer cancels unrelated default-group workers on that node, which is the task-19559 bug class.
- **Raw loader await, `_follow_coordinated_briefing`.** This was a real bypass. The coordinated Generate follower awaited `_load_briefings` outside `wl-briefings-load`; the local path already used `_request_briefings_refresh`. It now dispatches through the helper with the same arguments, and the helper's own refresh applies the failed state. It joins `MUTATION_REFRESH_HELPER_INVENTORY` (12 owners).

**Guard fix (worker-group guard).** The other 2 raw-await sites were guard false positives: `_request_tasks_refresh._load` (since 48d01048ae) and `_request_briefings_refresh.refresh`. Each is the dispatch helper's own nested worker body, scheduled by `self.run_worker(..., exclusive=True, group=<literal>)`, so its await runs inside the group. `_exclusive_helper_scopes` exempts an await only if it is nested in an inventoried helper whose own `run_worker` call is exclusive and names a literal non-default group. The intent stays: an await anywhere else, including in an ungrouped helper or one that is not inventoried, is still flagged.

**Boundaries: reviewed exceptions, not product changes.**
- **Actor_Packs.** `contracts.py` validates imported persona policy rules with the server-mirroring `PersonaPolicyRule` (fe5fee00da, lazy import). A second schema copy would fork the contract. `tldw_api.character_persona_schemas` is pure Pydantic, and `tldw_api/__init__` is lazy (PEP 562), so no HTTP client loads. The new `test_reviewed_boundary_exceptions_are_live_and_network_free` pins both facts.
- **Persona_Visual.** `builtin_pixel_migu.py` is the create-only Buddy installer. It writes a Character_Chat persona and its visual pack in one coordinated step, so it imports `Character_Chat` by design (f4c1ad1a7a; this guard has been red since then). Its exception is the only one in that package, and `test_reviewed_boundary_exceptions_are_live` fails if it goes stale.
- Production has not called `ensure_builtin_pixel_migu_buddy` since 11150b849d (`Persona_Buddy.library` seeds the built-in Buddy). Moving or deleting it is an owner decision, proposed as a follow-up.

**Evidence.**
- **Red before.** All 5 guard nodes failed in the serial base run at e8eeae7db9. New tests:
  - `test_raw_loader_in_an_exclusive_helper_body_is_the_dispatch` was red with the exemption removed.
  - `test_raw_loader_in_an_ungrouped_or_uninventoried_body_is_flagged` was red under an over-broad exemption that exempted any nested body.
  - The coordinated-follow pin `reload_groups == ["wl-briefings-load"]` was red with the product fixes reverted; the list was `[None]`.
- **Negative controls (AC#4).** These are temporary edits, each reverted.
  - Reverting the 8 product files turned red: all 7 ungrouped sites, the `_follow_coordinated_briefing` raw await, and both stock `ProgressBar`s.
  - Adding `tldw_api.client` to `contracts.py` and `httpx` to `character_persona_schemas.py` turned both Actor Pack tests red.
  - Adding `Character_Chat.Character_Chat_Lib` to `builtin_pixel_migu.py` turned the Persona Visual guard red.
  - A stale exception turned the liveness pin red.
- **After.** The 5 guard files pass serially (27 + 15 passed).
- **Covering tests, -n 4 at load 15-60.**
  - Unmarked, 30 files: 389 failed on mine vs 390 at base. Nearly all are the known `RecoveryRequired` trap (TASK-38111), and the failing sets are identical apart from `test_chat_screen_worker_groups`, which is now green.
  - Under a temporary module `bootstrap_profile` mark (8 Meetings, Schedules, Destination, Console-environment and Watchlists files), the failures were noisy and seen on both trees with product reverted. None was attributable to these edits.
- **Watchlists, temporary `@private_profile_test` marks, then removed.**
  - The coordinated test needed a sync change, not a weaker assertion. It now waits for the loader group (`_settle_briefings_load`), which tolerates a superseded, cancelled load, and asserts the reload's worker group.
  - `test_accepted_missing_receipt_retries_its_exact_id` uses the same helper. Its first-half `storage_mismatch` assertion had failed 1-4 of 4 per run, in both variants and on both trees, under load. It then passed in every run.
  - Pre-existing flakes remain on both trees: the after-retry `view_state == "idle"` assertion and `test_generation_completion_does_not_publish_into_a_new_watchlist_scope[False]`, which never touches the changed path.

Files:
- product:
  - `UI/Screens/meetings_screen.py`
  - `UI/Screens/chat_screen.py`
  - `UI/Screens/watchlists_collections_screen.py`
  - `UI/Widgets/SmartContentTree.py`
  - `UI/Wizards/ChatbookImportWizard.py`
  - `Widgets/Console/console_goal_setup_modal.py`
  - `Widgets/Settings_Widgets/workspace_change_review.py`
  - `Widgets/workspace_create_modal.py`
- tests:
  - `Tests/Architecture/test_worker_exclusive_group_inventory.py`
  - `test_actor_pack_boundary.py`
  - `test_persona_visual_runtime_boundary.py`
  - `Tests/Watchlists/test_watchlists_artifacts_refresh_states.py`

