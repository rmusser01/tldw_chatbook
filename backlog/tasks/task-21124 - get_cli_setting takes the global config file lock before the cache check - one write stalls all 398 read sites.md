---
id: TASK-21124
title: >-
  get_cli_setting takes the global config file lock before the cache check - one write stalls all 398 read sites
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - config
priority: high
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21124).

Reads: `config.py:6057 -> :4976 -> :5107` acquires `_config_file_lock()` BEFORE the cache
short-circuit (which sits inside `_load_cli_config_bootstrap_unlocked:4872-4877`). Writes hold
that lock through 2 fsyncs (temp fd + parent dir, `Utils/private_paths.py:660,691`), 3 full
TOML parses, and a settings rebuild. With 398 `get_cli_setting` call sites - many on the event
loop - any concurrent config write (even correctly off-loop ones) stalls loop-side reads for
the whole write. Amplifiers verified: Logs filter chip = 2 rewrites/4 fsyncs per click
(UI/Logs_Window.py:273-276), theme switch (app.py:872), lab rail toggle, and a per-keystroke
writer (UI/Dictation_Window_Improved.py:602 -> dictation_service_lazy.py:1383).

## Acceptance Criteria

- [ ] Cache-hit reads never take the file lock (double-checked fast path on the existing _CONFIG_GENERATION; writers already bump it)
- [ ] The write path is coalesced to one parse (verify re-parse dropped or debug-gated); the Logs-chip and dictation writers are debounced
- [ ] A two-thread probe (writer loop vs timed reader) shows reader p95 unaffected by concurrent writes; before/after numbers in the task
