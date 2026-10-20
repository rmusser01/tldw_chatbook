# ADR-198: Freeze the boot heap; keep CPython's GC thresholds

Date: 2026-09-29
Status: Accepted 2026-09-29 by the owner as a partial win (decision D3 of the 2026-09-27 structural perf audit). The owner first approved drafting it on the claim that the freeze removes the 130–871 ms pauses. The measurement below shows it shortens them without removing them, and the owner accepted it on that basis. The <50 ms target moved to TASK-33545.
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

## Amendment (2026-10-10, TASK-33620.15.3): re-freeze at idle, bound the frozen garbage

**Context.** Live profiling on 2026-10-10 (160x45, Anthropic haiku, 56
sends) found the residual this ADR predicted, on the Console's hottest path.
Generation-2 collection on the main thread took about 70 ms per send window
(Enter to the provider call), with single pauses of 23-65 ms at up to 16 turns
and 78-122 ms at 31-46 turns. That was 54% of the time spent in Textual screen
updates after a send was accepted. Each pass rescanned everything the Console
had built since the last boot freeze, and the pauses grew with every send
within a launch (36 ms on the first send, 121 ms by the fifth).

**Decision.**

1. **Re-freeze at idle moments the Console names.** The Console asks for a
   re-freeze when a run settles (the transcript poll's stop edge) and when
   its transcript shows a different conversation.
   `IdleHeapRefreezer` (`Utils/ui_responsiveness.py`, one per process) runs
   it only when two things hold:
   - the Console says it is not busy. It is busy from Enter until the run
     settles, and whenever it is not the top screen;
   - no user input has arrived for 2 s. A `TextualAppGuards` mixin stamps
     every input event.

   A busy Console drops the request and asks again when it next settles.
2. **Collect the unfrozen heap in full before each re-freeze**
   (`refreeze_long_lived_heap`). Frozen objects are not scanned, so this
   costs what the next automatic pass would have cost, paid at an idle moment
   instead of in a send window. Without it, cyclic garbage already in the old
   generation would be frozen for good. The boot freezes keep decision 2's
   young collection.
3. **Bound what re-freezing can leak.** `gc.freeze()` never collects a frozen
   object. Textual widgets are cyclic: their parent links, styles and message
   pumps form reference cycles. So a widget frozen while live and dropped
   later (a switched-away transcript, a recomposed rail) is garbage that
   never goes away. `Tests/Performance/test_gc_idle_refreeze.py` shows this
   with real widgets.

   Each re-freeze therefore measures the permanent generation. Once it has
   grown by 25% of its size after the last compaction (with a 50K floor), the
   next idle moment with 5 s of quiet compacts instead: unfreeze, collect the
   whole heap, freeze what survives. 25% is CPython's own rule for deferring a
   full collection (`long_lived_pending < long_lived_total / 4`), applied
   here to the frozen set.

   The measure is the generation's real size, not a count of what each
   re-freeze added. Live, a send's re-freeze froze 70-130K objects for
   about 20K of real growth: reference counting frees most of the rest out
   of the permanent generation, because the next paint replaces them.
   Counting additions instead triggered a compaction every third send.

   Sizing the generation means walking it. The generation is walked once
   after each re-freeze, at the next maintenance, so the budget reads the
   frozen set after the next run has freed what it replaced. The walk takes
   about 5 ms of CPU per 1.3M objects, and it runs only at idle. A first
   version walked only when a running total of what re-freezes counted
   reached the budget. That total is not a true upper bound, because other
   threads can allocate between a re-freeze's count and its freeze.

   No re-freeze runs while a compaction is due. The permanent generation
   therefore never outgrows its size after the last compaction by more than
   that budget plus one re-freeze, and a send's re-freeze is large. On
   2026-10-20 the frozen set peaked 396K objects above its post-compaction
   size against a 272K budget, because the re-freeze before the compaction
   froze at least 123K; the compaction then reclaimed 273K. The first
   version peaked the same way on 2026-10-19 (372K against 244K, at least
   128K in the last re-freeze). Both rounds show the bound holding, not a
   missed compaction. The next compaction reclaims whatever in it has
   become garbage, including boot-era garbage the earlier freezes kept.
   (Garbage can also form inside the post-compaction set -- a transcript
   frozen live, then switched away -- so the bound is on the frozen set's
   size, not on the garbage in the growth alone.)

   **Switched-away transcripts compact at once (2026-10-19).** The budget
   alone let them pile up. Live, across 24 chat switches and 12 runs, the
   frozen set's garbage reached 212-373K objects before a compaction ran,
   and each such compaction held the UI loop for 2.1-2.8 s of CPU (27-37 s
   of wall time at a load average near 95), against 329 ms for dev's worst
   generation-2 pass in the same run. The total collection CPU was the same
   on both arms (12.4 s against 12.5 s); the budget only gathered it into a
   few long stalls. The Console rebuilds its transcript for every
   conversation it shows, so it records the process's freeze epoch
   (`heap_freeze_epoch()`) with each conversation it paints. When it paints
   another one and the epoch has moved, a freeze caught the outgoing rows
   while they were shown, and it calls `IdleHeapRefreezer.note_stranded()`:
   the next idle maintenance compacts, whatever the budget says, and meets
   about one transcript of garbage. The note survives a request dropped by a
   send. Rows replaced before any freeze are ordinary garbage, which the
   re-freeze's own full collection reclaims, so rapid switching owes nothing.

   **A freeze follows only a collection that ran (2026-10-19).** CPython's
   `gc.collect()` returns 0 at once, collecting nothing, while another pass
   is in progress, and a worker thread's pass can hand the GIL to the UI
   loop from its finalizers or callbacks. In one live run, a compaction met
   such a pass. It froze the heap again, garbage included, and was booked
   as done, so that run's frozen garbage outlived it. A re-freeze or
   compaction now requires the generation-2 pass count to move (five tries,
   2 ms apart, so the other pass can finish). Otherwise the maintenance
   reports `busy`. A re-freeze then freezes nothing. A compaction freezes
   the heap again, so the automatic passes do not rescan it, and stays owed.
4. **Unchanged.** Thresholds stay at CPython's defaults. The collector is
   never disabled. The policy is off under pytest (`PYTEST_CURRENT_TEST`), as
   the screen pre-import is; `TLDW_IDLE_HEAP_REFREEZE` overrides it either way.

**Consequences.**

- A compaction scans the whole heap. That is the pass decision 2 kept off
  `_ui_ready`, and this ADR measured it at 130-420 ms. It runs only after 5 s
  without input and with no send in flight: after a switch away from a
  transcript a freeze caught, and otherwise about once per 25% growth of the
  frozen set. A user who switches chats often pays one compaction per
  switch at idle, where origin/dev paid a generation-2 pass per switch that
  could land anywhere, a send window included. Live on 2026-10-19 (about
  800K objects, load average 40-65), a switch's compaction took 263-375 ms of
  UI-loop CPU. Origin/dev's automatic generation-2 passes in the same round
  took a median of 192 ms and at most 264 ms. Dev's passes can land in a
  send window; the compactions ran only at idle.
- The "boot-era garbage becomes permanent" consequence above is now temporary:
  the first compaction reclaims it.
- Re-freezing ends CPython's habit, noted in the measured result above, of
  running full collections more often against a small unfrozen old
  generation. Each re-freeze resets it.

**Measured.** See TASK-33620.15.3's Implementation Notes: paired live sends
against origin/dev, and memory across conversation switches and run cycles.

## Python versions

The owner plans to move from 3.12 to 3.13 and then 3.14. Checked against the 3.14 documentation on 2026-09-29:

- **3.13, and 3.14.5 and later**, run the same generational collector as 3.12. This policy carries over unchanged, including `gc.collect(1)` meaning "collect the middle generation" and the `threshold[2]` note in Decision 3.
- **3.14.0 to 3.14.4** shipped an incremental collector with only two generations. There, `gc.collect(1)` runs one increment and `threshold[2]` is ignored. The freeze is still valid (`gc.freeze()` is unchanged), but these measurements do not transfer. 3.14.5 reverted to the 3.13 collector after production reports of memory pressure, so a 3.14 migration should target 3.14.5 or later, or re-measure first.
- **Free-threaded builds** (PEP 703; officially supported from 3.14 by PEP 779) use a different collector. Revisit this ADR before adopting one.

## Verification

- `Tests/Performance/test_gc_policy.py`:
  - pins the helper's collect-then-freeze behaviour;
  - boots the real app on a private profile and pins that the heap is frozen
    at `_ui_ready`;
  - records the automatic gen-2 pause durations of a destination tour as
    evidence in the task notes rather than as a timing assertion.
- `Tests/Performance/test_gc_idle_refreeze.py` (amendment): the re-freeze's
  full collect then freeze; scheduling (quiet period, busy owner, departed
  owner, stale timers, errors); the compaction budget; the input tap; and,
  with the real collector and real Textual widgets, that a frozen
  switched-away transcript survives a full collection (negative control),
  that a compaction reclaims it, and that the frozen set stays bounded over
  24 switches; a stranded note compacts at the next maintenance and frees
  the real Console transcript's switched-away rows (negative control: without
  it they survive), and survives a dropped request. Run while another
  collection is in progress, a re-freeze freezes nothing, and a compaction
  freezes the heap again and stays owed.
- `Tests/UI/test_console_idle_heap_refreeze.py` (amendment): on the mounted
  Console, a re-freeze is requested when a conversation is painted and when a
  run settles, and the busy check holds on every loop turn from Enter to the
  provider call; the paint policy notes stranded garbage only when a freeze
  ran while the outgoing conversation was shown.
