---
id: TASK-22220
title: >-
  Timer hygiene batch: db-size stats off the loop, hoist the Ollama gate, stop the loading-indicator tick
status: Done
assignee:
  - '@claude'
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

## Implementation Plan

1. Red-first probes (written as the eventual targeted tests, run against the
   unmodified tree to prove each defect):
   - db-size: wrap `os.stat` recording thread idents scoped to the tmp DB dir —
     count loop-thread stat calls per `update_db_sizes` fire (nonzero today);
     loguru sink counting "DB sizes:" INFO lines across two unchanged fires
     (2 today).
   - Ollama tick: full-app harness (`_build_test_app` + `LLMScreen`), cover the
     screen with a bare `Screen`, shadow `window.run_worker` with a counter,
     call `_schedule_ollama_api_state` 5x — 5 constructions today. Guard test:
     active screen still schedules exactly one worker with
     `exclusive=True, group="ollama-api-state", exit_on_error=False`.
   - InlineLoader (`Widgets/loading_states.py:233-266`; the finding names it
     `InlineLoadingIndicator`, but that class — `UI/CCP_Modules/
     ccp_loading_indicators.py` — already stops its own `@work` loop; the
     cited lines are `InlineLoader`): after `set_success()`, count alive
     timers in the pump's `_timers` (1 today); reset() re-arm probe.
2. Implement:
   - `Utils/db_status_manager.py`: move the stat collection into
     `asyncio.to_thread` (a `_collect_db_sizes` helper); assign
     `app.db_sizes_status` on the loop after the await; change-gate the
     `logger.info` triple on the collected dict.
   - `UI/LLM_Management_Window.py`: hoist the `is_attached`/`screen.is_active`
     gate into `_schedule_ollama_api_state`; keep the in-coroutine pre- and
     post-await guards (scheduling→running and mid-probe races).
   - `Widgets/loading_states.py`: store the `set_interval` handle on
     `InlineLoader`; stop it in `set_success`/`set_error`; re-arm in `reset()`
     (guarded on `is_running`); discard stopped handles from `_timers` so
     reset cycles don't accumulate dead Timer objects.
3. Targeted tests + `--collect-only` sweep, tee'd; counts from tees.
4. `./scripts/preflight.sh`.
5. Mutation-test each mechanism (remove change-gate / unhoist gate / drop
   re-arm — expect the matching probe to red), then restore.
6. Teardown walk: db-size fire cancelled mid-collection (closing app);
   `set_success` after unmount.
7. Tick ACs, Implementation Notes, Done; commit with explicit paths; push.

## Acceptance Criteria

- [x] DB-size collection runs off-loop (or is change-gated) and the periodic log line only fires on change
- [x] The Ollama tick creates no worker when the screen is inactive (gate hoisted to the scheduler)
- [x] The loading indicator's timer stops on terminal states and on reset-to-idle
- [x] Each item verified by a probe; no behavior change beyond the mechanics

## Implementation Notes

Three minimal mechanism fixes, each pinned by a born-red probe and a kill-the-fix
mutation run.

1. **`Utils/db_status_manager.py`** — the ~15 stat/exists syscalls per 120 s fire
   moved into a `_collect_db_sizes()` helper run via `asyncio.to_thread`; only the
   `app.db_sizes_status` assignment happens back on the loop. The INFO triple is
   change-gated on the collected dict (`_last_logged_sizes`): first fire and real
   size changes still log, an unchanged fire updates the cache silently.
   Born-red probe: an `os.stat` wrapper scoped to the staged DB dir recorded
   **15 of 15 stats on the event-loop thread** pre-fix, 0 post-fix; the loguru
   sink recorded **2 "DB sizes:" lines across two unchanged fires** pre-fix, 1
   post-fix. Teardown: cancellation at the off-loop await propagates (not
   swallowed by the catch-all — `CancelledError` is a `BaseException`) and
   publishes nothing (`test_update_cancelled_mid_collection_writes_nothing`).
2. **`UI/LLM_Management_Window.py`** — the `is_attached`/`screen.is_active` gate
   is hoisted from `_update_ollama_api_state` into `_schedule_ollama_api_state`,
   so a 3 s tick on a hidden tab constructs neither the coroutine nor a worker.
   The coroutine keeps its own pre-await guard (scheduling→running race) and the
   task-15473 post-await re-check (mid-probe deactivation). Probe: **5 worker
   constructions over 5 ticks** with the screen covered pre-fix, 0 post-fix; a
   companion test pins the preserved active-case semantics (one worker,
   `exclusive=True`, `group="ollama-api-state"`, `exit_on_error=False`).
3. **`Widgets/loading_states.py`** — `InlineLoader` (the finding says
   `InlineLoadingIndicator`, but the cited lines 233-266 are `InlineLoader`; the
   CCP `InlineLoadingIndicator` runs a `@work` loop it already stops) now stores
   its `set_interval` handle; `set_success`/`set_error` stop it, `reset()`
   re-arms it (guarded on `is_running`), `on_mount` arms only in the loading
   state, and stopped handles are discarded from the pump's `_timers` set so
   repeated cycles don't accumulate dead Timer objects. Born-red: one alive
   timer survived `set_success()`/`set_error()` pre-fix; post-fix the re-armed
   timer demonstrably animates dots again. Teardown: terminal-state calls after
   unmount are inert.

Verification: 18 targeted tests green (3 files); neighbor suites 209 passed with
one failure (`test_llm_screen_lab_adoption.py::test_missing_vad_...` —
**pre-existing on base `70d28febc`**, identical `NoMatches('#model-install-cancel')`
with my three source files reverted to base); `--collect-only` sweep 59,427
collected, 28 errors all missing-optional-dep (numpy/playwright) families
untouched by this task; `./scripts/preflight.sh` green after a per-row-reviewed
inventory regen (the one flagged row was the `logger.info` moving inside the
change-gate — moved/re-indented only, statement text unchanged). Mutations:
removing the change-gate, unhoisting the screen gate, and dropping the re-arm
each redded exactly the probe built for them.

Modified: `tldw_chatbook/Utils/db_status_manager.py`,
`tldw_chatbook/UI/LLM_Management_Window.py`,
`tldw_chatbook/Widgets/loading_states.py`,
`Docs/security/production-diagnostic-inventory.json`,
`Tests/Utils/test_db_status_manager.py` (+3 tests); added
`Tests/UI/test_llm_ollama_tick_gate.py`,
`Tests/Widgets/test_inline_loader_timer.py`.
