---
id: TASK-41635
title: "Change Review filesystem initialization's two-thread pool can be held by unproven git cleanups"
status: Done
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- workspaces
- console
priority: medium
references:
- tldw_chatbook/Workspaces/change_review_consent.py
dependencies:
- TASK-41307
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** Change Review runs filesystem initialization on a fixed daemon pool (`Workspaces/change_review_consent.py`, `worker_count: int = 2`), so that a slow root never creates an unbounded thread population. A git cleanup whose return was never proven stays retained (ADR-134/285), and TASK-41307's Release frees only the tool-worker slot. Two such cleanups can hold both pool threads, and every later Change Review initialization, for any root, then waits behind them with nothing on screen to say why.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A retained unproven git cleanup cannot block Change Review initialization for other roots, or the UI states why initialization is waiting and what releases it
- [x] #2 A test holds two unproven cleanups and starts a third initialization, asserting the chosen behaviour
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new decision. Initialization custody joins ADR-134's TASK-40716 rule that custody holds a daemon worker, never a joined pool thread; recorded as a dated implementation-note amendment to ADR-134.

Chosen behaviour (revised after review, 2026-11-18): AC #1's second branch. A retained cleanup's owner and opaque claim stay held until restart (TASK-40716), and initialization only ever borrows an opaque turn's owner or takes its own opaque claim, so later roots cannot be admitted until restart; freeing the claim would contradict that decision. The UI therefore says why such a root waits and what releases it (a restart), and offers no retry. The first round's plan, which had the third root become ready, held only for a fake ticket that took no claim.

1. Test first with the real shapes (`native_initializer` fixture: real runtime, issuer, `_InitializationSource`, `_SnapshotTicket`, capacity and activity). An opaque turn's owner and claim are borrowed by two roots whose cleanups are retained; a third root is admitted outside the turn and a fourth inside it. Pin each root's readiness and copy, retry, and that the holders are not pool threads. Red on the previous tip and on dev.
2. Keep the daemon custody worker. Add a `RESTART_REQUIRED` readiness with restart copy; map refusals to readiness in one place (`_fail(work, error)`), used by both `_schedule` and the worker. `_InitializationSource.admit` raises `CleanupUnprovenRefused` when the claim refusing it belongs to a retained cleanup's owner.
3. Settings ▸ Workspaces: its own line for those roots, no retry (retry already skips non-failed roots). Panel test red first.
4. ADR-134 amendment, User Guide sentence, task notes; covering files, preflight, `Tests/Architecture` serial.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Behaviour (AC #1, second branch).** A root that a retained unproven cleanup holds now reads `RESTART_REQUIRED`: "Change history waits for an app restart: an earlier cleanup could not be proven, so its workspace stays held. This turn continues without it." The copy names no tool: the retained cleanup whose claim refuses a root can be native Git, Raw CLI, Undo or an agent's tool. The Console shows that as the turn's *change review skipped* note. Settings ▸ Workspaces shows "Change history for N folder(s) waits for an app restart: …" on its own line with no retry, and "Retry failed preparation" skips those roots.

**Why not the first branch.** In production, later roots cannot be admitted while the hold lasts. Initialization either borrows an opaque turn's owner (`ordinary_tree.borrow` requires an opaque footprint) or takes its own opaque claim, and opaque conflicts with every claim. After a retention, a root in the same turn is refused by its tool reservation (`CleanupUnprovenRefused`) until Release frees the retained operation; a still-live turn then admits work again. Any other root is refused by the turn's held opaque claim, which TASK-40716 keeps until restart. The first round's note that ordinary turns' claims are not opaque was wrong, and its unit test reached READY only because its fake ticket took no claim. That test is removed.

**What the daemon custody worker gains.** It is kept from the first round: native jobs run on `run_on_custody_thread`, so a retained cleanup holds a daemon worker, not one of the two pool threads. Before, the pool threads were leaked to the hold, so the retained roots and every later root sat in "Preparing change history" forever with nothing saying why (the dev run of the new test times out in exactly that state). Now each such root reaches its honest readiness at once.

**Changes.**
- `Workspaces/change_review_consent.py`:
  - new `RootReadinessState.RESTART_REQUIRED` and `_RESTART_REASON`;
  - `_fail(work, error)` maps refusals in one place: `CleanupUnprovenRefused` → restart; `Full`/`CapacityRefused` → busy; else → failed. `_schedule` and the worker both pass the error (`_schedule` used to compute "busy" itself, so a refused tool reservation read "busy; retry from Settings").
- `Workspaces/change_review_ownership.py` (`_InitializationSource.admit`): when the opaque prelaunch is refused and a conflicting claim belongs to a retained cleanup's owner, it raises `CleanupUnprovenRefused(label, retention)` instead of `initialization_activity_busy`.
- `Workspaces/worktree_activity.py`: `holds_conflicting_claim(owner, footprint)`, a membership predicate under the activity lock (it returns no owner handles, like its sibling `recovery_blockers`).
- `Widgets/Settings_Widgets/workspace_change_review.py`: a `restart` row.
- ADR-134 amendment rewritten. The thread bound comes from the retained owner's reservation refusal and its held opaque claim, not from the tool slot. Until Release only jobs admitted before the first retention can be retained; after a Release a still-live turn can admit, and possibly retain, more, so each further custody worker follows a user's Release. User Guide (`console/agent-runs-and-tools.md`, held-slots paragraph): one sentence on the new state.

**Tests (red first, mutation-checked).**
- `test_retained_cleanups_hold_later_roots_until_restart_and_say_so` (`Tests/Workspaces/test_native_change_review_initialization.py`). Real runtime, issuer, `_InitializationSource`, `_SnapshotTicket`, capacity and activity. An opaque turn owner holds a real opaque claim, and a stand-in tree's `borrow` validates that claim and returns the owner, as `OrdinaryTurn.borrow` does. Two roots borrow it and their initializers retain "native Git" and hold, as `run_native_git` does. A third root is bound outside the turn (own opaque prelaunch, refused by the held claim) and a fourth inside it (reservation refused).
  - Asserts: all four read `restart_required` with the restart copy; retry returns 0 and changes nothing; the two holders are daemon threads outside the pool; the queue drains. Teardown proves both retentions.
  - Red at the previous tip deb368fdd0: retained roots `failed`, later roots `busy; retry from Settings`. Red on dev d0486978dd: roots stay preparing and the wait times out.
  - Mutations, each failing: admit detection reverted (outside root → busy); `_schedule` passing a busy error (inside root → busy); custody worker reverted to `custody()` on the pool thread (wait times out).
- `test_a_claim_held_without_a_retained_cleanup_stays_retryable` (negative control, same file). A "Raw CLI" retention whose owner holds no claim sits beside an opaque claim held by a live owner that retained nothing. The refused root reads FAILED "busy; retry from Settings"; after that claim is released, retry returns 1 and the root reaches READY. With the membership check replaced by `if True:` it fails (the root reads `restart_required`).
- `test_change_review_restart_required_root_names_the_restart_without_retry` (`Tests/UI/test_settings_workspaces_category.py`): the row's copy, no "preparation failed", no retry button. Red with the previous tip's panel.
- `test_unavailable_cleanup_proof_retains_original_worker_in_isolated_process` (real Git wrapper) keeps the first round's update: the holder is not a pool thread and the queue drains. It fails on the unfixed consent file.

**Evidence** (2026-11-18, base d0486978dd).
- Delta review round 3 (restart copy without "Git", negative control, membership predicate, ADR and User Guide wording that the restriction holds until Release): the Change Review files plus `Tests/Audio/test_voice_process_core.py` at `-n 4`: 7 failed, 97 passed. The same 7 fail on a detached origin/dev worktree (5 init waits in `test_native_change_review_initialization.py`, 2 `test_exact_native_terminal_preserves_two_transcripts_and_seal_order`). `./scripts/preflight.sh` run at this head.
- Change Review files at head 94e0844dc0 (`test_change_review_consent`, `test_native_change_review_{initialization,git_custody,ownership}`, `test_settings_workspaces_category`, `test_settings_change_review_journeys`; `-n 4`, load ~40): 11 failed, 110 passed, both new tests included. All 11 also failed on base d0486978dd in this task's base runs:
  - `test_settings_change_review_journeys` ×4;
  - the five 3-5 s init waits;
  - `test_borrowed_tree_keeps_its_unsettled_claim_after_both_publications_fail` and `test_original_opaque_claim_is_borrowed_and_bounded_claim_refuses`.
- First round (still applies to the custody-worker change): the serial native init/custody runs and alternating head/base rounds showed no head-only failure. The other consent-touching Chat/UI files: head 62 failed, base 61; the extra passes alone on both.
- The isolated real-Git custody test passes at 94e0844dc0 (it is in the run above).
- `./scripts/preflight.sh` passes at 94e0844dc0.
- `Tests/Architecture` (serial, at 94e0844dc0): 3 failed, 1476 passed, 5 skipped. The same three fail on base d0486978dd: `test_reviewed_diagnostic_changes_are_metadata_only` and the two `test_timer_path_static_update_inventory.py` tests. Their sites are `project_index_binding_card.py`, `train_journey.py` and `console_interrupt_rounds.py`; the Change Review panel is not listed.

Files: `tldw_chatbook/Workspaces/{change_review_consent,change_review_ownership,worktree_activity}.py`, `tldw_chatbook/Widgets/Settings_Widgets/workspace_change_review.py`, `Tests/Workspaces/{test_native_change_review_initialization,test_native_change_review_git_custody}.py`, `Tests/UI/test_settings_workspaces_category.py`, `backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md`, `Docs/User_Guide/console/agent-runs-and-tools.md`.
<!-- SECTION:NOTES:END -->
