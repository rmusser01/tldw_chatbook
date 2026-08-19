---
id: TASK-18609
title: >-
  Triage the 119 UI and core-suite failures exposed by the sharded Tests
  workflow
status: To Do
assignee: []
created_date: '2026-08-19 09:30'
labels:
  - ci
  - testing
  - triage
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-18608's phase 1 made the Tests workflow able to FINISH. The first
complete UI run in 100+ attempts then produced the suite's first full,
name-level failure inventory in that window: **13,558 collected / 13,432
passed / 119 failed across 57 files** (ubuntu, PR #1826 run 32268704382;
per-shard artifacts `ui-test-results-0..11`). These failures are not new --
they were invisible because every previous run was cancelled at ~11%.

This task is the triage: classify each failure cluster (environment-sensitive
test, test-double drift, or real product bug), fix or quarantine each, and
drive the workflow to green so it can gate merges again.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Every cluster below is classified: real product bug, test rot, or runner-environment sensitivity -- with one line of evidence each.
- [ ] #2 Real product bugs get fix tasks filed (or fixed here if small); test rot is fixed; environment-sensitive tests are made runner-robust or explicitly skipped-on-CI with a documented reason.
- [ ] #3 A complete `Tests` run on a PR to `dev` is green (both core legs and all UI shards).
- [ ] #4 The core-suite failures from the same run (macOS 55 + ubuntu's full list, see notes) are included in the same triage.
<!-- AC:END -->

## Notes

<!-- SECTION:NOTES:BEGIN -->
**UI inventory by cluster** (from per-shard artifacts; error signatures
grouped):

- **24x `test_library_prompts_canvas.py`** -- every case waits 15s for a
  prompt row (`#library-prompt-row-5`) that never mounts; visible text
  shows "Loading prompts…" stuck. Looks like a data-loading worker that
  never settles on a headless runner. Largest cluster; classify first.
- **~8x NoMatches** across settings/speech/audio files -- a selector finds
  no node (e.g. `test_settings_panel_scoped_updates`, handoff tests).
- **5x `test_console_transcript_pruning.py`**, 5x
  `test_settings_qwencloud_api_mode.py`, 4x each:
  `test_console_selection_menu`, `test_first_run_wizard_live_contract`,
  `test_console_transcript_markdown_widget`,
  `test_console_transcript_two_sided_window`.
- **4x `test_settings_configuration_hub.py`** -- the same four that fail on
  a local macOS run: staging a Console Behavior edit does not mark the
  draft dirty ("Save (s) -- no changes"). NOT environment-specific (fails
  on ubuntu CI, macOS CI, and local macOS) -- a real pre-existing failure
  that predates PR #1824.
- **3x AttributeError `'SimpleNamespace' object has no attribute
  'set_annotation_previews'`** (`test_console_turn_activity_line` via
  `chat_screen.py:15822`) -- production code calls a method a test double
  does not implement; classic test-double drift.
- Remaining ~45 failures are 1-3x each across ~40 files.

**Core suite:** macOS leg carried 55 failures (all pass on a local macOS
3.12 run of the same files -- runner-env-specific, concentrated in
subprocess/thread/SSH/fs-sensitive tests: Notes git-push services 17,
note-import-planner 9, TTS audio supervisor/admission 13, summarization
privacy 3, Architecture inventory 6, plus 7 singles). The ubuntu leg's
complete list lands with this run's artifacts (its predecessor died at 42%
with 43 unnameable failures visible); merge the two lists when triaging.

**Corrections to earlier attributions:** the four
`test_settings_configuration_hub` failures were initially (PR #1824
review) called "environment-specific to the reviewer's machine" -- the
sharded run disproves that; they reproduce on ubuntu CI. They were still
correctly excluded from PR #1824 (they fail on clean dev), but they belong
in THIS triage as real failures.

**Core inventories (complete, from PR #1826's run):** ubuntu finished for
the first time under the raised budget -- **20,658 total / 20,527 passed /
47 failed** (git-push-service 16, note-import-planner 9, Architecture
inventory 6, tts_request_admission 4, summarization privacy 3, git
integration 2, + 7 singles incl. Wizards first-run). macOS: 20,606 /
20,473 / **58 failed** -- 52 identical to the previous run, 3 flaky-fixed,
6 flaky-new (all subprocess/timing-sensitive; the two runs also sat on
different dev bases, so small deltas are expected). Union of both legs +
the 119 UI failures is the full triage scope.

<!-- SECTION:NOTES:END -->
