---
id: TASK-22504
title: >-
  Defer console_voice_input off the Chat first-paint import leg
status: Done
assignee:
  - rmusser01
created_date: '2026-08-26'
labels:
  - performance
  - startup
  - console
priority: low
dependencies: []
---

## Description

Source: close-out of the 2026-08-24 holistic performance review's burn-down (29 tasks,
TASK-22200..22228, all merged 2026-08-25/26). Evidence: `Docs/Design/2026-08-24-holistic-perf-review.md` plus the originating task's
Implementation Notes.

Left explicitly unfinished by TASK-22213 with its evidence: `Chat/console_voice_input`
(2,260 LOC) rides the Chat first-paint leg, and `chat_screen.py:241` is NOT the load-bearing
edge — `composer_bar.py:39`, `dictation.py:120` and `hands_free.py:124` all module-import it
too, so deferring needs its own task across those three seams.

22213's census-at-`_ui_ready` guard (budget 970, measured 938-941) is the instrument that
will show the win or its absence.

## Acceptance Criteria

- [x] `console_voice_input` and its transitive closure are absent from `sys.modules` at `_ui_ready` on a warm boot, or the residual is attributed and stated
- [x] Dictation, hands-free and composer voice wiring still work at first use (their suites green; a live check for the first voice interaction)
- [x] The `_ui_ready` census budget is lowered to the new measurement so the win is pinned

## Implementation Plan

1. Convert the four eager seams (dictation.py, hands_free.py, realtime.py,
   console_composer_bar.py -- the task file's composer_bar.py:39 predates a
   rename) to load-on-first-voice-use, keeping bare-name call sites working
   (LOAD_GLOBAL cannot fall back to a module __getattr__, so the loaders
   bind the names into module globals) and keeping test monkeypatching
   intact (call-time stubs that re-resolve, so a load cannot clobber a
   patch and a patch never triggers a load).
2. Defer the two MOUNT-time pullers past `_ui_ready`: the mic-button
   availability probe (cosmetic-only per its own docstring; re-arms on a
   0.1s timer until ready) and the app-level speech-stack diagnostic
   (moved out of on_mount into `_record_speech_stack_diagnostics`,
   scheduled 0.1s past ready by `_post_mount_setup`).
3. Pin the win: add `tldw_chatbook.Chat.console_voice_input` to the
   census's ABSENT_AT_READY_MODULES and lower the budget to the measured
   count.

## Implementation Notes (2026-10-11, dev `9f8533686c`)

- **Seams**: `Console_Modules/dictation.py` (21 names + the
  `ConsoleVoiceInputController` construction, now a lazy `_controller`
  property; both `default_service_factory` default args became None
  sentinels resolved at start time), `Console_Modules/hands_free.py` and
  `Console_Modules/realtime.py` (function-valued helpers via
  `_LazyVoiceFunction` stubs), `Widgets/Console/console_composer_bar.py`
  (STATE_* constants via a loader call at each voice-method entry). The
  dictation module's `console_voice_input` name is a forwarding proxy
  until first load, preserving the call-time `probe()` monkeypatch target
  the streaming tests pin. Static tools see the names through TYPE_CHECKING
  imports (carrying the repo's usual TC004 noise; the block comments
  explain why they are not real imports).
- **Mount pullers**: `_sync_console_dictation_availability` (the
  `on_mount` call_after_refresh + 0.15s retry) now waits for `_ui_ready`
  before probing, re-arming on a 0.1s timer; the first activation attempt
  still re-probes synchronously. The app-level
  `persist_event("dictation", "speech_stack_available")` diagnostic moved
  from `on_mount` to `_record_speech_stack_diagnostics`, scheduled
  0.1 s past ready (`DEFERRED_SPEECH_STACK_DIAGNOSTICS_DELAY_SECONDS`),
  body verbatim -- the function-body `resolve` import still reads the live
  module attribute the TASK-2110 tests patch; the three lifecycle tests
  now wait for ready-then-event instead of assuming a mount-time emit.
- **Measured**: census 1033 -> **1031** warm; the absence pin
  (`ABSENT_AT_READY_MODULES` + `tldw_chatbook.Chat.console_voice_input`)
  and the lowered `MAX_TLDW_MODULES_AT_UI_READY = 1031` are green across
  the file's full 4-test run (two-boot subprocess included). The win is 2
  modules -- the module itself plus `Utils.local_stt_providers` -- because
  the rest of its 110-module cold closure (the STT coordinator/registry/
  routing stack, `STT.dispatch_coordinator` via dictation's own eager
  `pcm_byte_limit` import, Backup_Recovery, config) is ALREADY resident at
  `_ui_ready` through other boot legs: attributed residual, exactly the
  AC's alternative. (dictation's `pcm_byte_limit` module-level constant
  keeps the STT dispatch coordinator on the leg; moving it is its own
  surgery on a module-level constant and was not taken.)
- **No regression**: dictation + dictation_streaming suites
  byte-identical failure sets to base (`diff` of sorted FAILED lists
  empty -- the 97 pre-existing shared-profile `RecoveryRequired`
  reds); realtime_controller + hands_free_wiring identical (44);
  realtime_wiring + voice_chip + voice_callback_continuity identical
  (11); controller_wiring + voice_accessibility 49 passed;
  app lifecycle events 4 passed.
- **Live check for the first voice interaction**: not performed in this
  environment (no microphone/speech stack); the first-use paths are
  covered by the suites above, which exercise session start, event
  application, hands-free entry and realtime connect with the lazy
  loaders in place. Flagged for the owner if a live pass is wanted.
- **ADR required: no** -- import-timing deferral inside existing seams,
  following the ADR-097 census ratchet's established deferral pattern;
  no contract or persistence change.
