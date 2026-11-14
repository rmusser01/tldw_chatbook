# Continued Console allocation and layout isolation

The approved lazy native-cache correction is committed at `50d3c9b055`.
Fresh scale build: 10,000 conversations / 250,004 messages / 10,000 search
documents; ready index, integrity OK, cleanup handles zero. Keyword300 passes:
P95 105.533ms, loop maximum 13.545792ms, no owned database descriptors or handles
after cleanup. Clean exact source fences pass. This is not UI qualification.

The original 70-case UI matrix remains NON-GREEN at this head: six activation
intervals fail, maximum 63.943958ms. All 60 exact searches, two preparations,
eight exact-target OPENED outcomes and all busy-paint checks pass. Limits remain
50ms loop / 100ms busy paint. Actual Chat handles are zero at native unmount and
at the separate post-Runner/default-executor-join inspection; no database files
remain open there. Five regular log/lock/lease descriptors are not whole-process
zero. The app exception is None. No limit, failure or resource warning is waived.

## Runtime isolation, not shipping changes or qualification

Passive callback clocks retain 161 long callbacks and 21 paired gen2 events,
18 on the main thread, without drops. A 54.062ms callback includes 23.339ms gen2
overlap; other paint callbacks reach 31.252ms without gen2 overlap. These are
inclusive clocks, not exclusive subtraction or proof of unique causation.

Separate widget clocks locate significant current-modal backdrop rendering and
ChatScreen layout work: 119 modal render-line calls total 612.779ms, and 88
ChatScreen layout calls total 502.354ms. The largest observed ChatScreen layout
is 30.969ms without gen2 overlap. The untimed final live-root census reads Strip
slot referents without allocating lazy caches: 185,915 unfrozen tracked objects,
12,937 Strips and 3,945 FIFOs, with 10,185 unfrozen Strips reachable from known
mounted line/cache roots. Hidden widgets own no cached lines. Counts include
unreclaimed cycles and are not a leak or exclusive-GC-cost proof.

All following separate original-driver controls remain NON-GREEN and are NOT
shipped: native tab paint batching (three failures), per-render native tint-style
memo (four), 1ms fair phase yields (three), append-only tab reuse (four), immutable
equal-copy suppression (three), native divide identity-cycle avoidance (five).
Tint memo avoids 52,875 repeated style transformations in that population but
does not clear latency. Identity-cycle avoidance changes private native cache
population/list reuse and is outside ADR226's preserved cache contract; it is
not authorized production work. No global primitive override, changed native
method, paint suppression, pixel freeze or extra collector action ships here.

## Bounded app-owned correction

The voice writer repeatedly removes/re-adds identical size classes and rewrites
empty Content. Native Textual8.2.8 also requests layout when clearing a sizing
rule with None even if no inline rule exists. Atomic classes alone therefore
do not stop the layout generation advancing. The adjacent Send-guidance writer
has that same redundant clear operation.

Keep the existing atomic owned-class pattern and current native visual for
identical voice copy. Compare Content plain text AND spans: native Content
equality ignores styling. Clear only existing inline size/margin rules. Keep
all width calculations, live-state/visibility updates, unrelated classes,
attachment restoration, timer/cancellation behavior and drift repair active.
No whole-state early return, new retained cache, native operation change,
collector policy, appearance or timing-limit change.

Native final-source RED: four failures in 2.20s. First correction retains visuals
but still fails four layout-generation assertions in 2.43s, exposing the native
None-clear behavior. Final projection selection: eight passed in 2.31s, including
changed copy, span repair, explicit inline width/height drift and focus retention.
New test Ruff/format clean; changed production regions format clean. Composer's
seven inherited Ruff findings are unchanged; no full-file lint-clean claim.

Broader caller selection: 263 passed / 15 failed / one FD warning in 610.83s.
The warning records +1,345 descriptors, start14/end1359, above the unchanged
limit200. This aggregate resource finding remains OPEN; the native Chat-zero
result is not a waiver or proof of whole-test-process retirement.

An exact-parent in-process replay reproduces 14 failures; the remaining wide
case varies with queued startup sync. Calibrate only the affected fixtures:
load the actual shipped bundle, test full voice copy where its shared draft
budget permits it, and assert real hidden-chip/draft retention below that budget.
The old 44-column truncation case cannot reach the current legible-chip branch;
measure the real row/reserved cells and pin the same 9..14-character branch at
108 columns. Keep the original literal-text/right-truncation assertions. Use
the existing production-ready helper to drain actual flags/Workers under its
unchanged bounds before manually projecting geometry. Seed both the canonical
store and its restored visible draft in two capture fixtures; load_draft is a
scope restore, not a persisted edit. The config-spy assertion now pins the
existing canonical explicit section/default arguments; no getter is changed.

Final combined affected selection: **50 passed in 34.03s**, no pytest warnings.
The other 263-pass population is not rerun or called aggregate green; its
warning remains retained. Initial fixture calibration failures (including an
unsupported tuple CSS_PATH and an out-of-root startup refusal) remain local,
not credited production REDs. No test is skipped, deadline raised, warning
suppressed or qualification assertion removed. The 52x20/120x50 driver is
unchanged. Touched Python lint additions zero: inherited counts composer7,
voice1, reason-width4, streaming3 and hands-free6; the new projection test is
clean. All eleven artifact guards passed on the unchanged production inputs.

Fresh clean-head scale/Keyword/original-matrix qualification is next. Allocation
and native projection results do not clear the remaining original UI failures.

All raw logs, frames and corpus copies remain LOCAL under
`/private/tmp/console-latency-continuation-fT3VrM`. Only audited outcomes and
hashes are published. Other sessions, shared dependencies, dirty main and the
original branch remain untouched. TASK31966 and broader qualification remain
open; no full repository sweep, push or merge is claimed.
