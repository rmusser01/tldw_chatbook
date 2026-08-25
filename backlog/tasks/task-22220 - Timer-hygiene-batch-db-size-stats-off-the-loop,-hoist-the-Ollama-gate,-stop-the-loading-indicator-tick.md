---
id: TASK-22220
title: >-
  Timer hygiene batch: db-size stats off the loop, hoist the Ollama gate, stop the loading-indicator tick
status: To Do
assignee: []
created_date: '2026-08-24'
labels:
  - performance
  - cleanup
priority: low
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22220).

Three small pre-existing recurring costs from the timer census (42-site table in the
evidence doc):
1. `Utils/db_status_manager.py:34-98` — every 120 s, ~15 stat/exists syscalls for
   DB+WAL+SHM sizes run synchronously ON the event loop, plus an unconditional
   `logger.info` triple; no gate, no change-skip.
2. `UI/LLM_Management_Window.py:611`, `:740-756` — the 3 s Ollama probe constructs and
   schedules a worker every tick even when the inside-the-coroutine `is_active` gate will
   immediately drop it; hoist the gate into `_schedule_ollama_api_state`.
3. `Widgets/loading_states.py:233-266` — `InlineLoadingIndicator` arms a 0.5 s tick on
   mount, discards the handle, and `set_success`/`set_error`/`reset` never stop it: a
   forever timer per mounted indicator.

## Acceptance Criteria

- [ ] DB-size collection runs off-loop (or is change-gated) and the periodic log line only fires on change
- [ ] The Ollama tick creates no worker when the screen is inactive (gate hoisted to the scheduler)
- [ ] The loading indicator's timer stops on terminal states and on reset-to-idle
- [ ] Each item verified by a probe; no behavior change beyond the mechanics
