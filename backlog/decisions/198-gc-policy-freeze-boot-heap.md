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

## Amendment (2026-11-01, PR3336 / TASK-33620.15.50): first-send imports use runtime idle maintenance

The coordinator chose the existing idle owner before this repair. First-send
imports can overlap foreground navigation or early received-turn custody after
their pre-import park. Their per-step and closing young-collection boot freezes
can capture live foreground cycles or old-generation garbage and exempt it from
growth accounting, including before the first baseline measurement. Re-parking
or a post-import idle snapshot cannot prove those objects are boot-only.

Remove those first-send boot freezes. After source priming completes, queue one
UI-loop callback that resolves the current attached Console and requests its
existing idle maintenance. No attached view or shutdown means no request. The
existing busy predicate also observes navigation admission/FIFO ownership and
global runtime custody before controller creation. Busy requests may drop;
painted/settled events provide later opportunities. No new ownership state,
lock, retry scheduler or debt reset is introduced. Original boot freezes and
their count-without-permanent-walk optimization remain unchanged. ADR-097 and
ADR-230 continue to apply.

First-send additions now use runtime full collection, measured growth, input
quiet times and stranded-debt/CollectorBusy policy. This can restore automatic
generation-2 work during navigation; the historical +10.5 ms observation is a
risk signal, not a measurement of repaired source. Native navigation/provider
latency, memory-byte bounds and continuous-input progress remain unqualified.
TASK-33620.15.18 AC2, .49 and unmeasured .50 criteria remain open; .51's narrow
historical acceptance does not transfer. Functional real-GC controls will be
recorded separately from those performance limits.

**Measured 2026-11-05 (TASK-33620.15.51).** The historical acceptance above named
TASK-33620.15.18's reviewed head f1b164cc2f, and it still does not transfer: that head
was not re-measured. .51 re-scoped its criteria to the merged code, and this note
measures that. A first send's provider latency was measured live on dev 29592ecc9b,
against the same commit with #3336's runtime changes reverted. Setup: real Anthropic
haiku, Enter 4.5 s after the warm-up ended, 16 ABBA pairs over two rounds with the
profiles swapped. Enter to provider POST was not later on dev within noise: a paired
median of -60 ms (bootstrap 95% CI -176 to +18 ms), with dev slower in 5 of 16 pairs.
Two of those pairs ran on a loaded host. Without them, and balanced across the two
profiles, the estimate is -4 ms (CI -119 to +45). This design cannot resolve a 20 ms
effect in either direction.

The pass-end request usually re-froze 2.0 s after the warm-up ended. In 4 of 36 dev
launches its timer found a compaction already due instead, because the runtime
re-freezes count what they freeze as growth. In those launches the first send ran
with the warm-up's modules unfrozen, and the first reply's maintenance compacted for
241-298 ms on the UI loop (TASK-33620.15.57). With the old closing boot freeze, that
happened in 0 of 36 launches.

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
     every input event twice: when it is posted to the app, and when the app
     dispatches it. A key still waiting in the app's queue therefore counts
     as input (TASK-33620.15.10).

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
   A new chat gaining its conversation id on its first send (the painted key
   goes from (session, none) to (session, id)) is not a switch: the
   transcript matches rows by message id, so none are replaced, and the
   epoch recorded when the new chat was first shown carries over (review
   follow-up, 2026-10-20: live, a fresh session's first send owed a 316 ms
   compaction only because the screen pre-import froze the heap while the
   empty chat was shown).

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

   **The pass must be the caller's own (2026-10-27, TASK-33620.15.11).** The
   pass count also moves when a worker thread's pass ends between the two
   reads while the caller's call was skipped. A probe on 3.12.11 held a
   worker's generation-2 pass at its start: the UI thread's `gc.collect(2)`
   returned 0 with no callback on its own thread, and the count still went
   from 0 to 1 once the worker finished. So a try now counts only when a
   `gc.callbacks` watcher, installed for that one call, sees a
   generation-2 pass start and stop on the calling thread. The same probe
   on 3.13.13 and 3.14.6 gave the same result: the caller's own
   `gc.collect(2)` reports generation 2 start and stop on its thread, and a
   skipped call reports nothing there.

   **Boot freezes are not growth (2026-10-27, TASK-33620.15.12).** The boot
   points freeze through the refreezer. Once it has made its first
   measurement, it measures the frozen set before and after each boot
   freeze, adds the difference to the baseline, and re-measures at the next
   budget check, all under a lock that maintenance also holds. The screen
   and first-send pre-imports freeze from their own threads. Before the
   first measurement a boot freeze costs nothing extra, because that
   measurement includes it. After it, a boot freeze costs one more walk
   (about 2 ms per 700K objects on 3.12.11).

   **Boot freezes count, they do not walk (2026-10-30, TASK-33620.15.18).**
   Live, in a running app, one walk of the frozen set held the GIL 15-34 ms
   on an idle host and 160-290 ms at a load average of about 110, far above
   the micro-benchmark figure: the frozen set is scattered across the heap.
   Every boot point walked once before the first measurement (only for its
   DEBUG line) and twice after it, and the first-send warm-up's freeze runs
   while the user may already be typing: a key or click sent during it
   reached the app 21-23 ms late on an idle host, and a click's Library
   frame came 16-41 ms late. Now a boot freeze before
   the first measurement only freezes, and after it counts the unfrozen
   generations it is about to move (`gc.get_objects`, which touches only
   unfrozen objects) and adds that to the baseline. The next budget check
   still walks once. Objects another thread allocates between the count and
   the freeze go uncounted, the same tolerance the walks had. With a boot
   freeze typically under 1 ms, the first-send warm-up now freezes after each
   step that loads modules (54 freezes per launch with the closing one, a
   median 4.0 ms in all and at most 5.6 ms for one, over 60 launches; the
   closing freeze took a median 0.37 ms against 30.6 ms): left unfrozen until
   its last step, those modules made a generation-2 pass due, and a click's
   navigation ran it on the UI loop in 9 of 10 launches.

   So the baseline is now the frozen set's size after the last compaction
   plus what boot points froze since, and the bound above reads against
   it: the set never outgrows that baseline by more than the budget plus
   one re-freeze. The budget is a quarter of that baseline (50K at least),
   so a boot freeze of D objects raises it by up to D/4, as it always did
   for a boot freeze before the first measurement. The lock also keeps a boot freeze out of
   a compaction's window between its unfreeze and its collection, where it
   would freeze the garbage the compaction is about to reclaim.
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
- **A compaction cannot be made shorter than a full pass over the whole heap
  (TASK-33620.15.9, 2026-10-20).** The owner asked for its pause to be no
  longer than origin/dev's worst automatic generation-2 pass while keeping
  this amendment's send-window gain. CPython has no way to do that.
  `gc.unfreeze()` moves the whole permanent generation, boot heap included,
  into generation 2. Collections of generations 0 and 1 never reach it, and a
  generation-2 collection is one pass that cannot be interrupted. On 3.12.11,
  frozen cyclic garbage survived `gc.collect(0)` and `gc.collect(1)` after
  `gc.unfreeze()` and died only in `gc.collect(2)`. That pass took about as
  long with no garbage as with it, and its cost grew with the whole heap. The
  same holds on 3.13 and 3.14.5+ (see Python versions below). Live, in
  TASK-33620.15.3's round 9, all 27 compactions took at least 235 ms of
  UI-loop CPU. Fitting their CPU to the objects they freed gives 235 ms plus
  1.27 ms per 1K objects, and the boot heap alone is about 625K objects.
  Origin/dev's worst automatic generation-2 pass in the same round took 202
  ms. So any design that freezes the Console's heap pays at least one
  whole-heap pass to reclaim the garbage it strands. Spreading that pass
  across idle slices, compacting less often, or waiting longer before it
  cannot bring it under 202 ms. The bound can be met only by not freezing
  after boot. Collecting the unfrozen heap at idle without freezing it meets
  it, but keeps almost none of the gain. Dev's generation-2 passes follow
  allocation (97 of 139 in round 9 came exactly 11 generation-1 collections
  after the one before, CPython's minimum), and a send window allocates enough
  for two to five of them. Even a collection run exactly at Enter would have
  removed only 8 of the 45 passes in round 9's dev send windows: median
  generation-2 CPU per window 189 -> about 162 ms, against 189 -> 77 ms with
  the re-freeze. The owner ruled on 2026-10-22 (next consequence).
- **Owner decision (2026-10-22, TASK-33620.15.9): "Keep it as built".**
  The owner re-ruled once the 2026-10-20 bound proved impossible on CPython,
  and accepted this amendment's design unchanged: the re-freeze, the
  compaction after a switch away from a transcript a freeze caught, and the
  25% budget compaction, all after 5 s without input and with nothing in
  flight. The trade accepted is the measured one. Main-thread generation-2
  CPU per send window drops (median about 189 -> 77 ms in round 9), and in
  exchange an idle compaction costs at least about 235 ms of UI-loop CPU
  (worst seen about 490 ms) after a conversation switch, and about once per
  20 runs otherwise. A key pressed while one runs waits until it ends. The
  bound on a compaction's pause is waived, not met; the other rules of this
  amendment (never in a send window, never within the quiet period after
  input, frozen garbage reclaimed and bounded) still hold and stay pinned by
  the tests under Verification. Revisit if a later CPython adds an
  incremental or interruptible full collection (see Python versions).
- The "boot-era garbage becomes permanent" consequence above is now temporary:
  the first compaction reclaims it.
- Re-freezing ends CPython's habit, noted in the measured result above, of
  running full collections more often against a small unfrozen old
  generation. Each re-freeze resets it.

**Measured.** See TASK-33620.15.3's Implementation Notes: paired live sends
against origin/dev, and memory across conversation switches and run cycles.

## Amendment (2026-11-06, TASK-33620.15.57): the warm-up's end re-freezes; maintenance waits out the pre-import passes

**Context.** The 2026-11-01 amendment left the first-send warm-up's objects to
the Console's idle maintenance: the warm-up's completion requested it, and it
ran 2 s later at the earliest, after 2 s without input. On dev 29592ecc9b
(TASK-33620.15.51, 36 launches) the timer found a compaction already due in 4,
waited for 5 s of quiet, and the first send dropped the request. Those sends
ran with the warm-up's modules unfrozen, and the first reply's maintenance
compacted for 241-298 ms on the UI loop.

The refreezer's accounting, recorded live at every budget check (dev
2684e5453f, 36 launches per arm, TASK-33620.15.57's round b1), shows why a
compaction was due. The Console requests a re-freeze when it first paints a
conversation, near the start of the screen-route pre-import pass, and the
request's timer fires 2 s later. When the pass is still importing then, that
runtime re-freeze freezes the routes imported so far, and they count as
growth; the pass's own boot freeze then credits only what is left. The
warm-up's request is checked 2 s after the warm-up, and books the warm-up's
own objects (about 29K) as growth on top. The closer to the pass's end the
paint re-freeze runs, the more it books. On 29592ecc9b, in all 4 compacting
launches the paint re-freeze ran 0.03-0.25 s before the pass ended, and the
pass's boot freeze then moved only 0.4-18.5K objects; in the 3 dev launches
whose paint re-freeze ran 0.34-1.37 s before the end (24-66K left), none
compacted. On 2684e5453f a runtime maintenance ran inside the pass in 16 of
36 dev launches and 18 of 36 with #3336's runtime changes reverted. The dev
launch whose re-freeze ran 0.86 s before the end booked 110K against a 144K
budget; the reverted arm's two at 0.15-0.16 s came within 21-26K of theirs,
less than the warm-up's 29K that dev adds. No dev launch ran in that band in
this round, and none compacted.

**Decision.**

1. **No runtime maintenance inside a pre-import pass.** The Console's busy
   check (`_console_heap_busy`) also holds while the app's initial-screen or
   whole-registry pre-import thread runs, so a pass's modules are frozen by
   its boot freeze (credited to the baseline, as decision 1 intends) and
   never as growth. A request that meets the hold is dropped, as any busy
   request is; the warm-up's end, which ends the pass, takes its place.
2. **The warm-up's completion re-freezes at once.** It is dispatched behind
   the same park and 0.15 s input wait as each warm-up step, and on the UI
   loop, on an idle attached Console, it calls
   `IdleHeapRefreezer.refreeze_now`. That is a runtime re-freeze, not a boot
   point (the 2026-11-01 amendment's reason stands): a full collection, then
   a freeze, and what it freezes counts as growth. Before the refreezer's
   first look it counts the unfrozen generations instead of walking the
   frozen set (garbage the collection then reclaims included, so it can only
   overstate the growth the first look books); after it, it marks the
   measured size stale. It runs even while a compaction is due, which stays
   owed, so the frozen set can outgrow its baseline by the budget plus two
   re-freezes, once per process. A pending request is satisfied by it.
3. **No second re-freeze after it.** The completion requests idle
   maintenance only when it could not re-freeze (a covered or busy Console)
   or a compaction is owed for stranded garbage. A first version requested
   it as before; 2 s later that re-freeze collected a nearly empty unfrozen
   heap (0.2-0.9 ms). CPython runs an automatic full collection once a
   quarter of what the last full collection kept is pending, so one that
   keeps almost nothing brings the next ones forward: in that version's
   first two launches the first send ran two generation-2 passes on the UI
   loop (18.7 + 17.9 ms and 38.0 + 63.4 ms) where dev's ran one or none.
   The consequence above that "each re-freeze resets" CPython's habit of
   frequent full collections holds only for a re-freeze whose collection
   keeps a sizeable heap.
4. **Unchanged.** The compaction budget and its 50K floor (the owner's
   2026-10-22 ruling), the quiet times, the stranded-garbage policy, and the
   boot freezes and their count-without-walk.

The warm-up's own objects number about 29K live (unfrozen at its end, 19K of
them reachable from the modules it imported), below the 50K floor, so they
cannot make a compaction due on their own;
`Tests/UI/test_console_first_send_preimport.py` pins that with the real pass
and collector.

**Measured 2026-11-06.** Live, real Anthropic claude-haiku-4-5, 160x45, one
first send per launch 4.5 s after the warm-up ended, and a key sent the moment
the first reply's maintenance started. Round b1 had 36 blocks of one launch
per arm on one profile: dev 2684e5453f, the same commit with #3336's runtime
changes reverted, and the fix. Arm order cycled through all six permutations
over four seeded profiles. Launches were gated on other sessions' load
generators only; the host's idle CPU at the send ranged 0-95% (20 blocks at
15% or more in every arm).

- Warm-up modules unfrozen at Enter: 0 of 36 in each arm. Compactions after
  the provider POST: none in any arm. The first reply's maintenance was a
  re-freeze in all 108 launches (medians 20.9-22.0 ms on the UI loop), and
  the key sent into it paid 35-36 ms from injection to frame in every arm. So this
  round could not time a compaction, or a key sent during one.
- The fix never ran a runtime maintenance inside a pass (0 of 36, against
  16 and 18), and the first budget check after the warm-up booked
  29.2-29.4K of growth in all 36 launches (budgets 160-185K), so the warm-up
  cannot make a compaction due. The closing re-freeze held the UI loop
  3.1-11.3 ms (median 4.1).
- Enter to POST, paired per block: fix minus dev a median -72 ms (bootstrap
  95% CI -130 to +6); fix minus revert +10 ms (-130 to +38). Main-thread GC
  in that span: -1.8 ms (-9.1 to +2.4) and -3.5 ms (-11.4 to +4.4).
- Those medians hide a fatter tail. Counting every generation, main-thread
  GC wall time in that span below 15% idle (16 launches per arm): the fix's
  four largest totals were 154, 162, 169 and 344 ms, against maxima of
  134 ms (dev) and 139 (revert). All four exceed every dev and revert value
  in that bucket (about a 1% chance if the arms did not differ, by
  permutation). Paired, fix minus dev was +64 to +268 ms in 5 of 36 blocks,
  four of them at 0% idle. TASK-33620.15.60 takes this tail, not the
  median, as its acceptance test.
- Generation-2 passes on the UI loop in that span: two or more in 12 fix,
  14 dev and 15 revert launches, and 553, 516 and 572 ms of CPU in all. With
  the host at least 15% idle (20 launches per arm), the median total was
  6.7 ms (fix), 15.4 (dev) and 13.9 (revert). Below it (16 per arm, 0-14%
  idle at the send, mostly 0), the fix's four largest totals were 95-170 ms
  of wall time and 36-50 ms of CPU, two passes each. In the same blocks dev
  and the revert ran one pass, or two in one dev launch, for 9-28 ms of CPU.
  The fix's Enter to POST in those 16 blocks was still a paired median
  108 ms sooner than dev's (TASK-33620.15.60 follows it up). A likely cause, not
  measured (the probe does not count unfrozen objects): the fix leaves what
  the first 2 s after the warm-up allocate unfrozen until the first reply's
  maintenance, where dev's timer froze it.

#3336's input-latency wins during the warm-up, re-measured. Round m1 had 32
blocks of one launch per arm, 8 per cell, and the host was below 15% idle for
90 of its 96 inputs. Latency runs from injection to the frame.

- A key typed as the warm-up's manifest parse ended, just before its
  completion: medians of 31 ms (dev), 42 (fix) and 72 (revert). Paired,
  dev minus revert was -41 ms (dev faster in 8 of 8), fix minus revert -29
  (fix faster in 6 of 8), and fix minus dev +18 (fix slower in 6 of 8).
  Two fix keys took 78 and 112 ms against a dev maximum of 45. In both,
  the closing re-freeze (5.0 and 6.3 ms) had ended 14.9 and 41.2 ms before
  the key; the delay was a 20 ms Console poll timer in one and a late
  insert in the other. With 8 per cell, "keeps #3336's wins" holds only to
  this resolution on a loaded host.
- Follow-up round n1 (16 dev/fix pairs, 12 with both launches at least 15%
  idle) put fix minus dev at +0.1 ms (fix slower in 8 of 16; medians 12.8 and
  13.2 ms). In all 24 fix launches, the closing re-freeze ended 0.2-41 ms
  before the key was injected.
- A key 0.02-0.5 s into the warm-up: 24 ms (dev), 33 (fix), 38 (revert).
  Fix minus revert was -12 ms (faster in 5 of 8) and fix minus dev +5.5
  (slower in 4 of 8).
- A Library click into the warm-up: 699, 627 and 696 ms. At its closing: 731,
  716 and 849 ms. Fix minus dev was -48 and -4 ms.

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
  owner, stale timers, errors); the compaction budget; the input tap,
  including a key posted to the app but not yet dispatched (negative
  control: with the stamp at dispatch only, a maintenance runs); and,
  with the real collector and real Textual widgets, that a frozen
  switched-away transcript survives a full collection (negative control),
  that a compaction reclaims it, and that the frozen set stays bounded over
  24 switches; a stranded note compacts at the next maintenance and frees
  the real Console transcript's switched-away rows (negative control: without
  it they survive), and survives a dropped request. Run while another
  collection is in progress, a re-freeze freezes nothing, and a compaction
  freezes the heap again and stays owed. A boot freeze walks the frozen set
  neither before nor after the first measurement, and adds what it counted
  to the baseline (TASK-33620.15.18; negative control: the next budget check
  still walks).
- `Tests/UI/test_console_idle_heap_refreeze.py` (amendment): on the mounted
  Console, a re-freeze is requested when a conversation is painted and when a
  run settles, and the busy check holds on every loop turn from Enter to the
  provider call; the paint policy notes stranded garbage only when a freeze
  ran while the outgoing conversation was shown, and never when a new chat
  gains its conversation id.
- TASK-33620.15.57 (2026-11-06 amendment): `test_gc_idle_refreeze.py` pins
  `refreeze_now` (runs with a compaction owed and leaves it owed, never
  walks, books what it froze before the first look as growth, satisfies a
  pending request, freezes nothing while another collection runs, off under
  pytest by the default gate unless `TLDW_IDLE_HEAP_REFREEZE` overrides)
  and, with the real collector, that live foreground objects it freezes
  make a compaction due where a boot point's would not.
  `test_console_idle_heap_refreeze.py` pins the hold while either
  pre-import thread runs and the completion's re-freeze, fallback request
  and stranded request. `test_console_first_send_preimport.py` runs the real
  pass and completion with the real refreezer: every module the warm-up
  imported is frozen when the completion returns, with growth below the
  50K floor and no second request. `test_first_send_warmup_gc_overlap.py`
  runs the completion on the event loop after an overlapping navigation or
  custody cycle retired: the closing re-freeze's full collection reclaims
  it (the boot-freeze control retains it).
