---
id: TASK-1719
title: 'Briefings audio: test-coverage residuals from phase 2b review'
status: To Do
assignee: []
created_date: '2026-07-31 23:59'
labels:
  - watchlists
  - briefings
  - tts
  - testing
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 2b's audio pipeline (task-1630, `Tests/Subscriptions/test_briefing_audio_pipeline.py`) is
well-tested, but self-review at close-out surfaced three residuals in what the existing test
suite actually proves, versus what its names and comments claim:

1. **The named invariant is narrower than it reads.** `test_a_failed_synthesis_never_touches_the_
   script` is called out in the module docstring as "THE named invariant (spec §Error handling
   ethos)" -- but it only drives a `TurnSynthesisError` raised from inside the per-turn synthesis
   loop. `generate_script_audio` has a second, earlier failure path -- `resolve_roster_voices`
   raising `VoiceResolutionError`, handled by `_record_voice_resolution_failure` before any
   `briefing_audio` row's synthesis loop even starts -- and no test asserts the parent script row
   is untouched on *that* path. The two paths are different code, so passing on one says nothing
   about the other.
2. **A "no file left behind" test that cannot fail on its stated path.** `test_no_file_left_
   behind_when_something_fails_after_the_write` mocks `wav_duration_seconds` to raise after a
   real write succeeds -- a genuine post-write failure, correctly exercised. But the pipeline has
   a second, earlier write-adjacent failure: `atomic_private_write_bytes` itself raising (a real
   write failure, not a downstream duration-read failure). That branch's own cleanup behavior is
   delegated to `Utils/private_paths`'s atomicity guarantees and has no test at all in this
   module -- so "no file left behind" is proven for one of the two failure shapes near the write,
   not both.
3. **An egress-rationale test whose premise the fixture skips.** `test_generate_script_audio_
   logs_no_turn_content_on_failure`'s docstring justifies itself by this app's `diagnose=True`
   log sink dumping a failing frame's locals -- but the test's fake `synthesize` stub raises the
   `TurnSynthesisError` directly, so the frame that actually raises is the fake stub's own thin
   body, not a real `synthesize_turn` frame that would hold the turn text as a local variable.
   The assertion (`canary not in log_text`) still passes, but it does not exercise the risk the
   docstring names -- a real synthesis failure deep inside `synthesize_turn` would need its own
   check to prove the same claim.

None of these are believed to hide an actual bug -- the code's error-boundary shape (traced in
task-1630's delivery note) covers all three cases correctly. This task is about the tests'
claims matching what they exercise, not about pipeline behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The script-untouched invariant is either verified on the `VoiceResolutionError` path too
      (a real test), or the module docstring/named-invariant comment is narrowed to state it
      covers only the synthesis-failure path
- [ ] #2 The write-failure branch (`atomic_private_write_bytes` itself raising) either gets a
      real test proving no orphan file remains, or an in-code note names `private_paths`'
      own test suite as the place that carries that guarantee
- [ ] #3 The `diagnose=True` egress test either exercises a real `synthesize_turn` failure frame
      (not a fake stub raising directly), or its docstring is corrected to state what it actually
      proves
<!-- AC:END -->
