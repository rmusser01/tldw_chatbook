---
id: TASK-18610
title: Pass-2 triage — runner-environment failures and remaining suite red
status: To Do
assignee: []
created_date: '2026-08-19 11:30'
labels:
  - ci
  - testing
  - triage
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pass 1 of the TASK-18609 triage (PR: fix/task-18609-triage-pass-1) fixed the
clusters that fail everywhere and have local reproductions. This task is the
remainder, ordered by size, each with its evidence from the complete CI
inventories.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 git-push-service 16 (ubuntu) / 17 (macOS): read the canary output from pass-1's CI run (`test_git_network_pin_environment_contract` prints the full stat table naming the failing predicate + directory), then fix the harness or (if a real predicate gap) the validator.
- [ ] #2 UI 119 failures / 57 files — largest cluster first: 24 in `test_library_prompts_canvas` (every case waits 15s for `#library-prompt-row-5` that never mounts; "Loading prompts…" stuck; looks like a data-load worker that never settles headless). Then: 4 `test_settings_configuration_hub` (Console Behavior edit does not mark the draft dirty — reproducible locally, real), ~8 NoMatches selectors, `SimpleNamespace` lacks `set_annotation_previews` (test-double drift, chat_screen.py:15822), and ~45 1-3x stragglers.
- [ ] #3 TTS 8 (ubuntu): 4 `test_tts_request_admission` (publication timing — saved model_id stays Old), 1 migration atomic-move message, 1 audio_cpp guided_text flag, 1 app-lifecycle drain, 1 profile repository (macOS-only concurrency).
- [ ] #4 Wizards 1: `test_mounted_model_owner_timeout_fences_late_result` — retry button stays `hidden` after timeout.
- [ ] #5 git-integration 2: unborn-HEAD discovery returns `repository=None` where a branch is expected.
- [ ] #6 Local-only stragglers (not in CI's inventory; fail on a clean dev worktree macOS): `chat_screen.py` is 21,253 lines against TASK10's 20,943-line ratchet ceiling -- a real 310-line growth since the ceiling was last set (the ratchet did its job; the ceiling needs an owner decision: extract or re-baseline). Same file trips `test_console_wave6_inventory` (`assert 21253 <= 20943`). These two were failing before the sharding fix too -- they were among the failures the cancelled runs hid.
<!-- AC:END -->

## Notes

<!-- SECTION:NOTES:BEGIN -->
Evidence per cluster is in the TASK-18609 notes and the run artifacts
(ui-test-results-0..11, core-test-results-{ubuntu,macos}, run 32268704382).
<!-- SECTION:NOTES:END -->
