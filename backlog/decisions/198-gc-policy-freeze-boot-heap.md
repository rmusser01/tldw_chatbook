# ADR-198: Freeze the boot heap; keep CPython's GC thresholds

Date: 2026-09-29
Status: Proposed. The owner approved drafting it as decision D3 of the 2026-09-27 structural perf audit on 2026-09-29, on the claim that the freeze removes the 130–871 ms pauses. The measurement below shows it shortens them but does not remove them, so acceptance waits for owner review.
Task: [TASK-33270](../tasks/task-33270%20-%20PERF-11-GC-policy---gc.freeze-after-ready-and-pre-import-documented-thresholds.md) (PERF-11)
Satisfies: the ADR precondition in [TASK-31966](../tasks/task-31966%20-%20Investigate-and-reduce-Console-activation-GC-pauses.md) AC #2 for a global GC policy
Evidence: [2026-09-27 structural perf audit](../../qa/perf-structural-audit-2026-09-27/report.md) §1, §3 R3, §4 PERF-11

## Context

`tldw_chatbook` is a long-lived Textual process. Nothing in the package calls
`gc.freeze()` or tunes the collector, so every automatic generation-2 collection
walks the whole heap, including the ~0.57M objects that exist once boot finishes
and never become garbage. They include module globals, pydantic and dataclass
model definitions, screen classes brought in by the background pre-import pass,
and interned strings.

The 2026-09-27 audit measured, in isolated profiles:

- A full collection took 122–266 ms on the boot heap and 326–422 ms after an
  8-destination tour, when the heap had grown to 1.38–1.63M objects.
- Automatic gen-2 collections fired 12 times in 24 screen visits, at 272–871 ms
  each, and 7 times during boot, at 54–174 ms each.
- The pre-import pass alone caused two gen-2 pauses of 59 and 69 ms.
- `gc.freeze()` after boot cost 0.0–0.1 ms, and it removes the frozen objects
  from every later collection.

These pauses land on the event-loop thread, where the user feels them as
stalls on screen switches and Settings category changes.

TASK-31966 requires an ADR before any global GC policy change.

## Decision

1. **Freeze the heap at two boot points.**
   - When `_ui_ready` flips, from the event loop.
   - When each background screen pre-import pass finishes, from that pass's
     own thread. The routes it imports are the largest post-ready addition to
     the long-lived heap.

   Both call one helper, `freeze_long_lived_heap()` in
   `Utils/ui_responsiveness.py`. It runs `gc.collect(1)` and then `gc.freeze()`.
2. **Run only a young collection before freezing.** `gc.collect(1)` covers
   generations 0 and 1: cheap, and it reclaims the short-lived cycles boot just
   made. A full `gc.collect()` is not run there. It would put a one-time
   130–270 ms stall on the loop at exactly the moment the user starts
   interacting.
3. **Keep CPython's default thresholds** (`gc.get_threshold()`, 700/10/10 on
   3.12). Freezing is what shrinks each gen-2 pass, because the pass then
   scans only objects created after boot. The measured side effect is
   more-frequent full collections (see Measured result). The knob that would
   trade that back is `threshold[2]`: once the unfrozen old generation is small,
   it is what gates a full pass. It is not tuned here because nothing was
   measured with it. Any threshold change amends this ADR with measurements.
4. **Never `gc.disable()`.** Memory would grow without bound in a session that
   runs for days.
5. **Be observable.** The helper logs the freeze count at DEBUG, so a stall
   investigation can see what was frozen and when.

## Measured result

All runs used a scratch profile at 235x52: boot, then a 24-visit tour of 8
destinations. Every number is the sum or median of automatic gen-2 pauses,
timed with `gc.callbacks`. Two runs per arm, with PERF-05's leak fixes applied.

| | No freeze | Freeze |
|---|---|---|
| Boot window, total gen-2 time (includes pre-import) | 2.3–3.4 s | 1.1–1.3 s |
| Tour, gen-2 collections | 8 | 18–20 |
| Tour, median pause | 509–663 ms | 155–170 ms |
| Tour, max pause | 750–993 ms | 277–705 ms |
| Tour, total gen-2 time | 4.5–5.2 s | 3.0–4.3 s |

The freeze makes each pause about 3–4× shorter. Full collections, however, run
more than twice as often. After a freeze, CPython's rule that a full collection
waits until 25% of the long-lived heap is pending is measured against a much
smaller unfrozen old generation. Total collection time still falls, and the
worst stalls shrink.

The freeze does **not** eliminate gen-2 pauses. What remains scales with the
live heap the session builds after boot. At the end of the tour that heap was
about 0.6M unfrozen objects. Two things dominate it:

- **Textual's `Strip`.** Each strip eagerly allocates seven `FIFOCache`s, so
  ~50–60K retained render lines are ~0.4–0.5M GC-tracked objects.
- **Screens that outlive their visit.** Settings leaked until PERF-05 fixed it.
  Schedules and Workflows still leak, filed as a follow-up.

## Consequences

- **Pauses are proportional to post-boot allocation.** Gen-2 cost scales with
  objects created after boot (screens, widgets, transcripts), not the whole heap.
  Collections are more frequent, as measured above.
- **Boot-era gen-2 garbage becomes permanent.** Any cyclic garbage still in the
  old generations at each freeze is never reclaimed. That includes the widget
  tree of a screen that is mounted at a freeze and later discarded, along with
  anything reachable only through it. It is bounded: at most three freezes run
  per process (the initial-screen pre-import, `_ui_ready`, and the
  whole-registry pre-import). The tour measured +13K–54K retained objects
  (about 1–4%) against no freeze.
- **Leaks stay visible.** Freezing does not hide leaks: objects allocated after
  the last freeze are still collected normally.
  [PERF-05 / TASK-33264](../tasks/task-33264%20-%20PERF-05-Screen-leaks---Settings-signal-subscription-Personas-worker-pin-Home-recompose-pin.md)
  fixed the Settings and Personas leaks. Schedules and Workflows screens are
  still pinned through a retained `contextvars.Context` (follow-up task).
- **Freezing is cheap and idempotent.** A later freeze just moves more
  long-lived objects into the permanent generation.
- **Forked children are unaffected.** Frozen objects stay shared
  copy-on-write, which is CPython's documented motivation for `gc.freeze()`.

## Alternatives considered

- **Full `gc.collect()` before each freeze.** This reclaims boot garbage but
  adds a one-time 130–270 ms loop stall right after ready. Rejected; it can be
  revisited if measured permanent garbage turns out to matter.
- **Raise thresholds, or gate collections on idle.** Raising `threshold[2]` is
  the lever against the higher collection frequency, but nothing was measured
  with it (see Decision 3). Idle-time collection alone would not shrink a pass.
- **`gc.disable()` plus manual collection at idle.** This risks unbounded
  growth when nothing is ever idle, such as during a long agent run.

## Verification

- `Tests/Performance/test_gc_policy.py`:
  - pins the helper's collect-then-freeze behaviour;
  - boots the real app on a private profile and pins that the heap is frozen
    at `_ui_ready`;
  - records the automatic gen-2 pause durations of a destination tour as
    evidence in the task notes rather than as a timing assertion.
