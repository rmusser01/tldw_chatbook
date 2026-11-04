# ADR-354: Windows Textual timer waits use the event loop

Status: Accepted
Date: 2026-11-04
Related Task: [TASK-38462](../tasks/task-38462%20-%20Keep-Windows-timer-sleeps-from-starving-the-default-thread-pool.md)
Supersedes: N/A

## Decision

Select `asyncio.sleep` for Textual's shared `timer.sleep` slot on Windows in the existing `ThreadWorkerContextGuard.on_load`, before its installed-executor early return.

## Context

Textual 8.2.8 sends both Windows timer waits and cancellation signals through the default executor. Long timers compete with application thread workers. TASK-38206 documented eight occupied workers and queued timers; PR3345's Windows job126398414464 separately witnessed the 0.1-second DB timer2.75 seconds late, its executor Future still pending and its callback never entered. Five of six sampled workers were timer sleepers; two workers were omitted. The existing three-second startup check must remain enforced.

`textual.timer` is the only installed Textual consumer importing `_time.sleep`. Its existing Timer continues to own callbacks, intervals, repeats, pause/reset and stop. The application already centralizes Textual compatibility in `Utils/text_selection_crash_guard.py`. The existing FreshContextExecutor, job context isolation, worker capacity and replaced-pool shutdown remain unchanged. The default is process-global to Textual timers on Windows; macOS/Linux do not change it.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Increase workers or deadlines | Retains the contention and grows resource requirements with long-lived timers. |
| Separate timer pool | Adds another pool and shutdown contract while keeping one worker occupied per sleep. |
| Special-case DB startup | Leaves recomposition, animation and other thread workers exposed to the same timer contention. |
| Custom Windows wait-handle integration | Adds OS-handle and private event-loop lifecycle machinery already avoided by the standard library timer. |

## Consequences

`asyncio.sleep` schedules a cancellable event-loop handle and consumes no executor worker. Textual's Windows high-resolution sleeper and its small early-wakeup compensation are no longer used; Windows cadence may follow the standard event-loop clock resolution. Existing timer semantics and deadlines remain the qualification target. No new pool, dependency, module, storage authority, profile action or resource-limit increase is introduced.

A controlled eight-worker proxy must reproduce the queued-wait failure before the change and demonstrate real Textual callback delivery plus long-timer cancellation while all workers remain held afterward. Owned jobs must be released and awaited. Existing executor/context-isolation tests, original startup assertions and current-head hosted Windows qualification remain mandatory; proxy results do not establish Windows kernel timing. Revisit this shim when the pinned Textual implementation stops routing Windows sleeps through the default executor.

## Links

- [Shared served Console implementation plan](../../Docs/superpowers/plans/2026-10-25-shared-console-mobile-supervision-implementation.md)
- [Application shell architecture](../../Docs/Architecture/app-shell.md)
