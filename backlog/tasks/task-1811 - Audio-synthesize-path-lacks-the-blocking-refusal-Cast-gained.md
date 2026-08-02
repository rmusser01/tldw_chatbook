---
id: TASK-1811
title: Audio synthesize path lacks the blocking refusal Cast gained
status: In Progress
assignee: []
created_date: '2026-08-01 18:25'
updated_date: '2026-08-02 08:57'
labels:
  - watchlists
  - briefings
  - audio
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed at close-out of the Watchlists briefings phase 4 plan (spec #2), per phase 4 task 1's own
recorded, deliberate deferral.

Phase 4 task 1 added an in-process claim registry so scheduled generation and a manual button
press can't race each other (`Subscriptions/briefing_service.py`'s `GenerationInFlightError` /
`_ACTIVE_*_CLAIMS`, mirrored in `briefing_cast.py` and `briefing_audio.py`). As part of that, Cast
gained a `blocking` refusal it never had before: a Cast press while a cast for the same briefing is
already in flight now refuses with a specific toast
(`UI/Screens/watchlists_collections_screen.py`'s `_sweep_and_guard_cast` +
the `blocking` check in `_cast_script`) instead of racing a second cast over the same briefing.

Audio's `_synthesize_audio` has the identical shape Cast had *before* that fix: a front-of-worker
sweep with no `blocking` check after it, so two presses of Synthesize could in principle start two
concurrent renders for the same script. This was investigated and explicitly deferred in task 1's
own report, with reasoning: nothing scheduled in phase 4 synthesizes audio (only `briefing_job`
generates text briefings; casting and audio synthesis stay button-driven), and the screen's own
`_audio_in_flight` reactive already blocks a second Synthesize press on the *same screen instance*
before either press reaches the service layer. The gap that remains is narrower than Cast's was
before its own fix (the claim-aware sweep change from the same task applies to audio's sweep too,
so a live claim still survives an Artifacts-load sweep either way) -- it is specifically the
missing *cross-instance* `blocking` refusal (e.g. two screen instances, or a future caller that
isn't gated by `_audio_in_flight`) that Cast now has and audio does not.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 EITHER `_synthesize_audio` gains a `blocking` refusal symmetric to Cast's (a Synthesize press for a script already claimed by `active_audio_claims()` refuses with a specific toast naming the running synthesis, instead of relying solely on `_audio_in_flight`) OR a decision not to add it is recorded here with reasoning, ratified by the project owner or a reviewer
- [x] #2 If implemented, a test proves a Synthesize press during a claimed audio synthesis is refused via the specific `blocking` toast, not a generic error toast (mirroring `test_a_cast_press_during_a_claimed_briefing_refuses_not_run_concurrently`)
- [x] #3 No change to the claim-aware sweep behavior already shipped in phase 4 task 1 (a live audio claim must continue to survive `fail_interrupted_audio` regardless of this task's outcome)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Mirror Cast's `blocking` refusal into `_synthesize_audio` in `watchlists_collections_screen.py`: after the front-of-worker sweep, check `active_audio_claims()` and refuse with a specific toast naming the running synthesis.
2. Test mirroring `test_a_cast_press_during_a_claimed_briefing_refuses_not_run_concurrently`; assert the specific toast, not a generic error.
3. Verify the claim-aware sweep behavior is untouched (existing live-claim-survives-sweep tests stay green).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Chose AC #1's implementation arm (symmetric blocking refusal), mirroring Cast's
phase-4 fix exactly: `_synthesize_audio` now runs a new `_sweep_and_guard_audio`
(sibling of `_sweep_and_guard_cast`, using `db.list_briefing_audio(script_id)` +
a new `_audio_row_label` helper) instead of a bare `fail_interrupted_audio` call,
and refuses with a specific `blocking` toast ("... is already being synthesized
for this script. Nothing else was started.", severity="warning", markup=False)
when a row survives the sweep because a live in-process claim holds it -- same
guard placement (after the sweep, before `generate_script_audio`), same
specific-toast convention as Cast's own fix. No `recovered`-branch refusal was
added (mirrors `_cast_script`'s own reasoning: `briefing_audio` has no
one-complete-row-per-script invariant, so a press that recovers a zombie must
still synthesize real audio in the same press).

Test (AC #2): `test_a_synthesis_press_during_a_claimed_script_refuses_not_run_
concurrently` in Tests/Watchlists/test_watchlists_artifacts_pane.py, mirroring
`test_a_cast_press_during_a_claimed_briefing_refuses_not_run_concurrently`:
claims the script's audio id directly via `briefing_audio._claim_audio`
(standing in for another in-process caller), presses the real Synthesize
button, and asserts the specific blocking toast (severity, markup, "already
being synthesized" substring) plus that `generate_script_audio` was never
called and the live claim's row was not falsified. Mutation-tested by
temporarily neutralizing the new `blocking` check (`if False and blocking:`) --
confirmed the test REDs -- then restored; `git status --short` clean afterward.

AC #3 verified, not re-implemented: the claim-aware sweep behavior from phase 4
task 1 is untouched (`fail_interrupted_audio` still receives `exclude`d claims
from `_sweep_and_guard_audio`, same as before from the inline call); existing
`test_a_zombie_generating_audio_row_is_recovered_on_a_plain_artifacts_load` and
`test_synthesizing_recovers_a_zombie_audio_row_via_its_own_sweep` stay green
unchanged.

Verified: the new test, the full `Tests/Watchlists/ -k audio` set (19 passed),
the whole `test_watchlists_artifacts_pane.py` file (112 passed), and
`Tests/Subscriptions/test_briefing_audio_db.py` +
`test_briefing_audio_pipeline.py` + `test_briefing_audio_synthesis.py` (60
passed, untouched module).

Modified files:
- tldw_chatbook/UI/Screens/watchlists_collections_screen.py
- Tests/Watchlists/test_watchlists_artifacts_pane.py
<!-- SECTION:NOTES:END -->
