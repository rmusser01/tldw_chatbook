# ADR-354: Windows Textual timer waits use the event loop

Status: Accepted
Date: 2026-11-04
Amended: 2026-11-05 (integration with dev TASK-38462)
Related Task: [TASK-38462](../tasks/task-38462%20-%20Keep-Windows-timer-sleeps-from-starving-the-default-thread-pool.md)
Supersedes: N/A

## Decision

Retain dev's `install_windows_timer_sleep()` in the existing
`ThreadWorkerContextGuard.on_load`, before its installed-executor early return.
Windows waits of at least 0.25 seconds use `asyncio.sleep`. Short waits preserve
Textual's high-resolution `time.sleep` behavior and early-wakeup allowances on
dev's existing fixed 32-worker `textual-timer` pool. Neither waits nor cancellation
use the application's default executor. Native Textual Timer lifecycle code stays
unchanged. macOS/Linux leave the timer slot untouched.

The initial PR #3345 decision selected `asyncio.sleep` for every wait. Dev had
independently qualified the bounded short/coarse implementation by the time this
branch rebased onto 3c39215749a96d5c1b8c05972fdd2aeb7588b002. The old five-line
override would silently discard dev's short-wait precision; it is removed.
This amendment adopts that existing implementation without raising its limits.

## Context

Textual 8.2.8 sends both Windows timer waits and cancellation signals through the default executor. Long timers compete with application thread workers. TASK-38206 documented eight occupied workers and queued timers; PR3345's Windows job126398414464 separately witnessed the 0.1-second DB timer2.75 seconds late, its executor Future still pending and its callback never entered. Five of six sampled workers were timer sleepers; two workers were omitted. The existing three-second startup check must remain enforced.

`textual.timer` is the only installed Textual consumer importing `_time.sleep`. Its existing Timer continues to own callbacks, intervals, repeats, pause/reset and stop. The application already centralizes Textual compatibility in `Utils/text_selection_crash_guard.py`. The existing FreshContextExecutor, job context isolation, worker capacity and replaced-pool shutdown remain unchanged. The default is process-global to Textual timers on Windows; macOS/Linux do not change it.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Increase workers or deadlines | Retains the contention and grows resource requirements with long-lived timers. |
| Separate pool for every timer | Long waits still occupy workers. Dev's accepted implementation uses its bounded pool only for sub-0.25-second precision waits. |
| Special-case DB startup | Leaves recomposition, animation and other thread workers exposed to the same timer contention. |
| Custom Windows wait-handle integration | Adds OS-handle and private event-loop lifecycle machinery already avoided by the standard library timer. |

## Consequences

Coarse waits use cancellable event-loop handles. A cancelled short wait returns
immediately; its already-issued physical sleep retires within the unchanged short
interval before its thread rejoins the existing pool. Cancellation submits no
extra job. The process-wide pool retains the standard executor shutdown contract;
fixture-created pools must be physically joined and the prior pool restored after
native Timers stop. The application default executor, capacity, fresh job contexts,
and replaced-pool shutdown are unchanged. No dependency, storage authority, profile
action, resource-cap increase or deadline increase accompanies this integration.

A controlled eight-worker proxy must reproduce the queued-wait failure before the change and demonstrate real Textual callback delivery plus long-timer cancellation while all workers remain held afterward. Owned jobs must be released and awaited. Existing executor/context-isolation tests, original startup assertions and current-head hosted Windows qualification remain mandatory; proxy results do not establish Windows kernel timing. Revisit this shim when the pinned Textual implementation stops routing Windows sleeps through the default executor.

## Links

- [Shared served Console implementation plan](../../Docs/superpowers/plans/2026-10-25-shared-console-mobile-supervision-implementation.md)
- [Application shell architecture](../../Docs/Architecture/app-shell.md)
