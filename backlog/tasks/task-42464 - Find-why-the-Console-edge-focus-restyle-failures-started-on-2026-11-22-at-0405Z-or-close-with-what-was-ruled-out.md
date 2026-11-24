---
id: TASK-42464
title: "Find why the Console edge-focus restyle failures started on 2026-11-22 at 04:05Z, or close with what was ruled out"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 06:38
updated_date: 2026-11-24 00:30
labels:
- follow-up
- ci
- console
- investigation
priority: low
references:
- Tests/UI/test_console_edge_focus_restyle_scope.py
- Tests/UI/test_console_shell_regions.py
- tldw_chatbook/Widgets/Console/console_setup_modal.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), by TASK-42416. Checked on the round-22 stack 336d15338b (dev 2f7790873a plus round 22's stacked branches); paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** TASK-42416 fixed the cause of `Tests/UI/test_console_edge_focus_restyle_scope.py`'s CI failures. The shell harness (`make_console_pilot` in `Tests/UI/test_console_shell_regions.py`) hid the first-run setup overlay once, after a fixed pause. Every later `ConsoleSetupModal.sync_card_state` (`Widgets/Console/console_setup_modal.py:243`) set it back to blocking, so a sync landing after the hide covered the Inspect handle and `pilot.click` returned False. TASK-42416's local spy counted 11-14 setup syncs before the hide, from the credential-readiness poll's first pass, `_refresh_console_hooks`, the identity-surface retry and the transcript ticks.

That race existed before 2026-11-22, yet in TASK-42416's scan of 140 Derived Artifacts PR runs (2026-11-20 00:00Z to about 21:50Z on 2026-11-22, every attempt) the PR fast-lane job that runs the file failed 9 times, all on this file, from 2026-11-22 04:05Z to 17:57Z, and never earlier. The first failure came about two hours after round 19's merges. Something raised the rate then: a change that added a later setup sync or moved one, or a runner change. If a product change made the overlay re-show later after startup, a real first run would see that too. Close the task if nothing is found.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The notes list the commits merged to dev from 2026-11-21 00:00Z to 2026-11-22 04:05Z that touch the setup overlay's sync sources or the shell harness, each marked with whether it changes when the overlay is re-shown after startup
- [x] #2 If one does, the notes name it with a local before/after measure (the setup syncs after the harness's hide, by count or timing); a user-visible change to the first-run overlay is fixed with a test shown red first, or recorded as a proposed follow-up
- [x] #3 If none does, the task closes with what was ruled out
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read-only CI evidence: list the Derived Artifacts PR runs around 2026-11-20..2026-11-23, find the fast-lane failures on `test_console_edge_focus_restyle_scope.py`, and pull the failing logs for the first ones (`gh run view --log-failed`).
2. `git log` on dev from 2026-11-21 00:00Z to 2026-11-22 04:05Z (first-parent merges and their commits); keep those touching the setup overlay's sync sources (`console_setup_modal.py`, readiness poll, `_refresh_console_hooks`, identity-surface retry, transcript ticks) or the shell harness (`Tests/UI/test_console_shell_regions.py`, the edge-focus test), and mark each for whether it changes when the overlay is re-shown after startup.
3. If one does, measure setup syncs after the harness's hide before/after it locally (a spy, through the wrapper); if a user-visible first-run change, fix with a red-first test or record a follow-up.
4. Check whether round 22's harness fix (TASK-42416, in this base) covers it: soak the test 20 times at base under load.
5. Notes, ACs, Done (or close with what was ruled out).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Verdict: no change on the overlay's sync path was found, what made the step faster at the onset is not established, and TASK-42416's harness fix (#3723, e8aca4955c, merged 2026-11-23 15:18Z) already covers them on this base. Closed with what was ruled out.**

**CI record** (read-only rescan of every Derived Artifacts run and attempt, 2026-11-19 00:49Z to 2026-11-23 22:19Z; "PR Fast Lane (admission-sensitive 2)" PR jobs that ran to success or failure):
- before 2026-11-22 04:05Z: 100 passed, 0 failed;
- 04:05Z to the fix: 67 passed, 15 failed, every failure on this file (`test_an_edge_focus_cue_restyles_only_its_region_and_control` and/or `test_no_rule_styles_another_node_by_an_edge_regions_focus`, last on 2026-11-23 10:35Z; one dev `workflow_dispatch` run at 09:59Z failed the same way);
- trees containing #3723: 17 passed, 0 failed.

**AC #1: commits merged to dev from 2026-11-21 00:00Z to 2026-11-22 04:05Z** that touch the setup overlay's sync sources (`chat_screen.py`'s guidance/setup-modal path, `Widgets/Console`, `UI/Console_Modules`, config readiness reads, the Textual shims in `tldw_chatbook/__init__.py`) or the shell harness (`test_console_shell_regions.py`, `test_destination_shells.py`, `ConsoleHarness`, `app_factory.py`, both conftests). None changes when the overlay is re-shown after startup.
- #3674 r18-providers (05:32Z 11-21): 9fde20af32, 036a6f50b6, 607a876ec5, 5d6f0ffeb1, load-time config migrations (/v1 bases, encrypted slot keys). **No**: they run at config load; no sync source or timer changes.
- #3675 r18-consoletests: bd343b444a deletes uncalled first-send methods; a91a9645c4 clears unseen Console receipts in bootstrap-profile teardown. **No.**
- #3676 r18-library: 6ed301cfd8, 912ae105d3, 2fd86a73a3, the recompose press keeper. **No**: it changes delivery of a queued press during a removal, not when guidance syncs run.
- #3680 r18-consolerun (17:49Z 11-21): cc3a15f9ac, 43def8c7e0, Console scratch directories in basetemp. **No.**
- #3685 r18-devred: 421fad45fc, 04bc1bcf40, config migration 2's shapes. **No.**
- #3690 r19-docs2 (02:15Z 11-22): fa64b8ffa7, Change Review copy. **No.**
- #3691 r19-testsa: 75daacd280, conftests remove their sandbox root at exit. **No.**
- #3693 r19-consolea: 70a5c26563, `_console_first_send_completed` caches only a True read. It is read inside each guidance sync; in the harness the flag is False before and after, so card state and sync count are unchanged. **No.**
- #3694 r19-providers2: b290f519ef, 932acd7937, config migrations kept in memory after a write. **No.**
- #3697 r19-recompose: 6afb465d5f, 6227182910, 6b55a47262, ae59544f40, the keeper moved to `App._prune`, plus hook/bench rows resolved by button. **No**; its own PR run passed.
- Also checked: #3654 (e5626ea043, 21:24Z 11-21, the dev base of the first failing runs) only refactors the Git environment into `scrubbed_git_environment()` with the same result. No dev merge landed from 02:16Z to 04:05Z on 11-22.

**AC #2:** none does, so there is no commit to name and no first-run fix or follow-up. The re-show is visible only because the harness hides an overlay the product shows on purpose until setup is complete; in a real first run each sync keeps it shown.

**AC #3, ruled out:**
- **The lane.** The admission-2 job and its step are identical at 230205800c and fe84affafc, and none of the step's 48 target files changed in the window.
- **The runner.** The same images before and after (ubuntu-26.04 20261116.202.1 on Azure, 20261116.88.1 on AWS); 59 Azure passes on that image before the onset. The same CPython 3.12.16 and pytest 9.2.0, and the same 89 installed packages in the last pre-onset Azure pass (job 137058698672, 03:32Z) and the first failure (137065646598, 04:05Z).
- **Content.** The first failures were PR runs of the stacked r19 branches, each testing dev e5626ea043 plus its stack prefix: testsb, backuprec and consoleb failed, while consolea, providers2, backupdef, recompose and filing, which carry the same content and more, passed. Each of those is a single run, so as evidence against a flaky cause they are weak: at the failing period's rate (15 of 82, about 18%) a branch carrying the cause passes one run about 82% of the time. The first failing tree (e5626ea043 + docs2/testsa/testsb) adds no change on the overlay path. The counts carry more weight: before the onset, trees containing the r18 merges passed 31 of 31 (about 0.2% likely at that rate), while trees containing e5626ea043 passed 12 of 12 (about 9% likely, so that commit is not excluded by the count; its product change is a behaviour-identical refactor).
- **Local measure.** A scratch spy on `ConsoleSetupModal.sync_card_state` replayed the old harness (composer, `pause(0.2)`, one hide, `pause()`, the click). 12 mounts at e5626ea043 and 12 at fe84affafc ran concurrently, `-n 4` each, at load 50-70, with the refusing Keychain probe. Both trees gave 11-13 syncs before the hide and 0 after; the last landed 0.13-2.2 s before it, and 24 of 24 clicks hit.

**What the data does show.** Within the failing period the failures were the faster runs: by the step's pytest time they ranked 5-56 of 82 (one-sided permutation p ≈ 0.005). From the onset the step's median also fell from 1373 s to 1206 s, with the same targets, Python, packages and image, while the project-chat UAT step in the same job did not get faster (149 s to 164 s). That fits TASK-42416's mechanism: a fixed pause after the composer, and timer-driven setup syncs that land after the hide when the mount is quick. Why the step got faster is not established: the UAT step slowing at the same time argues against plain faster runners, and no change on the overlay's sync path was found. The fast-run skew is also why a loaded local machine cannot reproduce the failure.

**Already fixed on this base (b31e3e9241).**
- **Soak.** 20 of 20 runs of the file at `-n 4` (40 of 40 tests) passed, at a 1-minute load of 54-109. On its own this cannot fail here, since the old harness passed 24 of 24 mounts locally too.
- **Forced check at base** (scratch test, removed). One `_sync_console_transcript_guidance()` was forced after the hide. The current `make_console_pilot` + `click_inspect_rail_open` still clicked through. The old once-only hide lost the click (`pilot.click` returned False). Both cases passed as written.

No repo files changed for this task.
<!-- SECTION:NOTES:END -->
