---
id: TASK-42111
title: "First turn after a restart may skip change review without a disclosure row"
status: Done
assignee:
- '@claude'
created_date: 2026-11-20 07:20
updated_date: '2026-11-25 14:39'
labels:
- follow-up
- console
- workspaces
priority: medium
references:
- tldw_chatbook/Workspaces/change_review_consent.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Seen once during TASK-42101's live check (2026-11-19, branch fu2/lfs-filters, scratch profile, fake llama.cpp server). Not reproduced and not traced.

**What happened.** Workspace 1 had Change Review enabled, and its only folder's preparation was refused. The app was restarted, and the first Console turn in that workspace ran and answered "Done.". It showed no Activity group and no "⚠ change review skipped …" row. The turn before the restart, and the turn after it, each showed the skipped row with the reason.

**Expected.** `ChangeReviewConsentService` admission lists every non-READY root as skipped with its reason; a PREPARING root's reason is "Preparing change history; this turn continues without it". So a missing row means either the admission observation was not current (`return ChangeReviewAdmission()` with nothing to disclose), or the root had no readiness entry for the current consent revision.

**Not every restart.** On a later relaunch the same workspace's first turn did show the row, so the gap is at least intermittent.

**Evidence.** The pane captures are in the session scratchpad. Turn evidence: 02 (no row) against 01 and 04 (rows).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The first turn after a restart in a workspace with Change Review enabled is reproduced and shown either to disclose its skipped root, or why it does not
- [x] #2 If a turn can skip change review silently, it now discloses the skip (test, red first)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace which consent read a stock Console turn's configuration takes (`capture_turn` vs `admit_turn`) and what it answers for a bound root with no readiness entry (a fresh service, as after a restart).
2. Write a deterministic red test of that window through the stock turn-configuration read.
3. Fix it in the shared `_admit` so every capture discloses an unprepared root as skipped, without making a capture schedule preparation.
4. Run every test file that reads Change Review admission; preflight and ruff gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Cause: the first stock turn after a start captured its Change Review roots through a read that never schedules preparation.** That read drops a bound root that has no readiness entry, and readiness lives only in memory, so a restart empties it. `ChangeReviewConsentService.capture_turn` is `_admit(schedule_missing=False)`. Its second loop skipped any binding whose entry was missing or held another consent revision (`continue`), so the root landed in neither `ready_roots` nor `skipped_roots`.

**Which route a turn takes.** A stock Console Send captures its configuration in `capture_console_turn_configuration`: `console_received_dispatch` → `capture_turn_configuration_snapshot` → `console_configuration_preparation` (line ~1834), with no `_captured_change_review`. That reads `capture_change_review_metadata` → `capture_turn`. The ordinary tree then calls `capture_change_review_admission` (`admit_turn`), which does schedule the root. For the stock service it keeps that answer only when `change_review_capture_pending` is set, and it is not, so the turn's row still comes from the capture.

**Why it was intermittent.** The Environment panel reads `admit_turn` at rail open, poll ticks and landings. That read creates the root's PREPARING (and later FAILED) entry. If it ran before the first Send, that turn showed the row; if the Send came first, the turn was silent. The live observation fits this: turn 02 was the first Send after the restart, before any panel read; turn 04 came after the turn's own `admit_turn` had recorded the failed preparation. The other suspect, a stale observation (`return ChangeReviewAdmission()`), needs a toggle in flight or a disposed service. Neither happens at start.

**Reproduced (AC#1).** An uncommitted mounted probe used the stock services (`_build_test_app`, a real registry and consent service). It bound a folder, enabled consent without any admission (a restarted app's state, `consent._readiness == {}`), mounted the Console and called `controller.capture_turn_configuration_snapshot`. It also recorded every consent read (`_original_consent_reads`). On dev 2959b48314:
- no consent read ran before the Send capture;
- the capture's only reads were `capture_turn` → `()`;
- the turn's configuration had `change_review_skipped_roots=()` and `workspace_roots=()`, so no row.

With the first fix (47f2d636ec), the same probe listed the root as skipped; since review M3 the capture's reason is "Change history is not prepared yet; this turn continues without it."

**Fix (AC#2).** One branch in the shared `_admit`: a bound root with no entry for the current consent revision is now listed as skipped instead of dropped. The capture path (`schedule_missing=False`) gives the new `_NOT_PREPARED_REASON`, "Change history is not prepared yet; this turn continues without it.", because a capture starts no preparation. An admission (`schedule_missing=True`) keeps `_PREPARING_REASON`, because a concurrent admission is re-scheduling the root. A capture still schedules nothing. Every capture caller goes through `_admit`, so all of them are covered.

**Tests, red first.** `test_a_turn_capture_discloses_a_root_with_no_current_readiness` in `Tests/Workspaces/test_change_review_consent.py` runs over a real registry and the stock turn-configuration read (`capture_change_review_metadata`). It has two cases: `after_a_restart`, a fresh service, and `after_a_consent_change`, where the root became READY and then a consent compare-and-set moved the revision while consent stayed on. Each asserts `service.status(...).roots == ()` right after the capture; scheduling records PREPARING readiness under the lock before the admission returns, so a scheduling capture fails this deterministically. Each then asserts the exact skipped row. Controls, each run on both cases:
- dev's `change_review_consent.py`: both fail with `assert ((), (), ()) == ((), (), (SkippedReviewRoot(...),))`. The root is dropped after a restart and after a consent change (review M2).
- the first fix (47f2d636ec, preparing reason on the capture path): both fail on the reason (review M3).
- the fix with `capture_turn` mutated to `schedule_missing=True`: both fail with `assert (RootReadiness(...),) == ()` (review M1).
- the fix: both pass.

**Review of 47f2d636ec (local).** Every finding is fixed or recorded:
- **M1, fixed.** The old `calls == []` check ran before any worker could call the initializer; the reviewer's `schedule_missing=True` mutation passed it 3 of 3. It is replaced by the `status()` check above, which fails that mutation.
- **M2, fixed.** The revision-mismatch half of the condition is covered by the `after_a_consent_change` case, red on dev.
- **M3, fixed.** "Preparing change history" overpromised on the capture path. On paths where `admit_ordinary_turn` returns early (`Chat/console_ordinary_tree.py:66-67`: no native process groups, as on Windows, or no hooks v2 runtime), nothing on the Send path calls `admit_turn`. The capture now gives the neutral reason, a named constant beside `_PREPARING_REASON`. A grep for "Preparing change history" found no other test or doc asserting it for the capture path. The other hits are admission paths (`test_change_review_consent.py`'s preparing tests), Settings (`workspace_change_review.py`, `test_settings_workspaces_category.py`) and hand-made reasons in Chat tests.
- **N1, fixed.** The comment on the branch now covers both cases: a capture after a restart or consent change, and an admission whose entry a concurrent one re-keyed to a newer revision.
- **N2, recorded as a known gap, not changed here.** `ChangeReviewConsentService.status()` (`tldw_chatbook/Workspaces/change_review_consent.py:371`, the filter at :385) still drops a root with no readiness entry for the current revision. After a restart, Settings ▸ Workspaces shows no readiness for a root that the turn row reports as not prepared, until something admits the workspace.

**Runs.**
- `Tests/Workspaces/test_change_review_consent.py` whole: 12 passed on dev. After review: 14 passed serially (`-n 0`) and 14 passed at `-n 4`.
- Every test file that reads Change Review admission, at 47f2d636ec, in one `-n 4` run at load 40-57 (16 files): 28 failed / 374 passed. Seven of those files on unmodified dev: 25 failed / 213 passed. 24 failure ids are shared, mostly `test_console_turn_execution_context.py` (`'_CustodyController' object has no attribute 'store'`) and `test_console_configuration_capture.py` (a fake consent service the stock path no longer reads). Three ids failed only on the branch: `test_review_runs_before_baseline_gate_and_tool_dispatch_waits`, `test_a_bound_workspace_lands_through_the_real_consent_service` and `test_consent_revoked_after_a_worker_re_check_drops_the_landing`. All three passed when rerun on the branch (3 passed, load 39); they assert ready roots or ordering, not skipped roots.
- `test_console_session_admitted_roots.py::test_the_three_readers_return_the_same_string_roots` fails on dev under load as well (3 of 3 `-n 4` file runs at load 47-57: `TimeoutError` in its 5 s `asyncio.timeout` around two roots' preparation). It passed alone. This is pre-existing and was not touched.
- Preflight passed. `ruff check tldw_chatbook --select F` passed, and `Tests/lint/ruff_gate.py` passed. Logger calls are unchanged.

No live run: the mounted probe reproduced the window deterministically on the real Send capture route. No lesson added, because TASK-41507's entry ("a launch test that starts on Library cannot see a cold-start defect") already covers it. Here every Change Review harness (`_console` in `test_console_session_admitted_roots.py`) called `admit_turn` and waited for readiness before mounting, so the cold capture was never exercised.

Files: `tldw_chatbook/Workspaces/change_review_consent.py`, `Tests/Workspaces/test_change_review_consent.py`.
<!-- SECTION:NOTES:END -->
