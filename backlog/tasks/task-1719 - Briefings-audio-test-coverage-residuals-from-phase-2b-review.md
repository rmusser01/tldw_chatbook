---
id: TASK-1719
title: 'Briefings audio: test-coverage residuals from phase 2b review'
status: In Progress
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
- [x] #1 The script-untouched invariant is either verified on the `VoiceResolutionError` path too
      (a real test), or the module docstring/named-invariant comment is narrowed to state it
      covers only the synthesis-failure path
- [x] #2 The write-failure branch (`atomic_private_write_bytes` itself raising) either gets a
      real test proving no orphan file remains, or an in-code note names `private_paths`'
      own test suite as the place that carries that guarantee
- [x] #3 The `diagnose=True` egress test either exercises a real `synthesize_turn` failure frame
      (not a fake stub raising directly), or its docstring is corrected to state what it actually
      proves
<!-- AC:END -->

## Implementation Notes

All three residuals closed in `Tests/Subscriptions/test_briefing_audio_pipeline.py`; no production
code changed (`tldw_chatbook/Subscriptions/briefing_audio.py` diff is empty).

- **#1 (real test).** Added `test_a_voice_resolution_failure_never_touches_the_script`, mirroring
  `test_voice_resolution_failure_for_a_deleted_profile_is_a_failed_row`'s setup plus the
  before/after `db.get_briefing_script` comparison the synthesis-path test already makes.
  Mutation-verified: temporarily added `db.update_briefing_script(script_id, error=...)` inside
  `_record_voice_resolution_failure` -> test went RED (`before != after`, diffing on `error`) ->
  reverted -> green, clean `git status`.
- **#2 (real test).** Added `test_no_orphan_file_when_the_atomic_write_itself_raises`: monkeypatches
  `briefing_audio.atomic_private_write_bytes` to raise `OSError`, asserts the row is `failed` and
  `briefing_audio_dir()` has no `.wav` files. Read `Utils/private_paths.atomic_private_write_bytes`
  first: it writes to a private temp file and only `os.rename`s onto the destination after the
  write/fsync/postcondition all succeed, with its own `finally` unlinking the temp file on every
  exit path -- so a raise from it can never leave anything at the destination, by construction.
  That mechanism (and its temp-file cleanup under a synthetic OS-level failure) is `private_paths`'
  own guarantee to prove against its real POSIX write path; I checked `Tests/Utils/
  test_private_paths.py` and `test_private_persistent_artifacts.py` and neither currently has a
  POSIX mid-write-failure residue test for `atomic_private_write_bytes` specifically (a real gap,
  but in a different module, out of this task's scope) -- so the test's docstring names the
  ownership honestly without citing a specific test that doesn't exist. What this test pins is
  `generate_script_audio`'s own contract at that call site. Mutation-verified: changed the
  `except Exception` around the `atomic_private_write_bytes` call to `except ValueError` -> the
  injected `OSError` propagated uncaught, test went RED -> reverted -> green, clean `git status`.
- **#3 (doc-narrowing, not a new test) — verified empirically, not just reasoned about.** Before
  writing anything, I ran the existing test with the log output printed: `generate_script_audio`'s
  `except Exception as exc: logger.warning(f"...: synthesis failed: {type(exc).__name__}")` never
  attaches the exception object or a traceback to the log record (no `logger.exception(...)`, no
  `.opt(exception=...)`) -- captured `log_text` was one plain line with zero traceback content.
  I then confirmed via a scratch loguru-only test (no tldw_chatbook import) that `diagnose=True`
  only annotates *lines a logged traceback contains* with the locals *that line references*, and
  never fires at all when no exception is attached -- so swapping the fake stub for a real,
  several-frames-deep `synthesize_turn` failure would produce byte-identical log output to today's
  shallow stub: no traceback either way. Arm (a) would therefore add a fake `tts_service` and
  contort the test for zero additional coverage. Took arm (b): corrected the test's docstring to
  state precisely what it proves (this log statement's own shape -- type name only, nothing
  attached -- keeps turn content out, independent of which frame raised), and added a direct
  `"Traceback" not in log_text` assertion pinning the mechanism itself rather than only its
  consequence. Mutation-verified: temporarily changed the log call to
  `logger.opt(exception=exc).warning(...)` -> RED, with the captured output showing loguru
  dumping `self.fail_exc`'s repr (canary and all) right at the stub's `raise self.fail_exc` line,
  confirming both the new `"Traceback"` assertion and the existing canary assertion catch a real
  regression -> reverted -> green, clean `git status`.

Verification: `test_briefing_audio_pipeline.py` (32 tests) + `test_briefing_audio_db.py` (13) +
`test_briefing_audio_synthesis.py` (24) = 69 passed. `--collect-only Tests/Subscriptions` collects
624 tests with no errors. `git status --short` on `tldw_chatbook/` is clean after every mutation
check.
