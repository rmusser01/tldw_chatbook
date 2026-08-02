---
id: TASK-1811
title: Audio synthesize path lacks the blocking refusal Cast gained
status: To Do
assignee: []
created_date: '2026-08-01 18:25'
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
- [ ] #1 EITHER `_synthesize_audio` gains a `blocking` refusal symmetric to Cast's (a Synthesize press for a script already claimed by `active_audio_claims()` refuses with a specific toast naming the running synthesis, instead of relying solely on `_audio_in_flight`) OR a decision not to add it is recorded here with reasoning, ratified by the project owner or a reviewer
- [ ] #2 If implemented, a test proves a Synthesize press during a claimed audio synthesis is refused via the specific `blocking` toast, not a generic error toast (mirroring `test_a_cast_press_during_a_claimed_briefing_refuses_not_run_concurrently`)
- [ ] #3 No change to the claim-aware sweep behavior already shipped in phase 4 task 1 (a live audio claim must continue to survive `fail_interrupted_audio` regardless of this task's outcome)
<!-- AC:END -->
