# CSS and Console presentation follow-up (TASK-35610)

Baseline: `ac0c13581a54327b1fd38a3b6ddcce45cdb416e0`, Python 3.12.10,
Textual 8.2.8, Windows 10, headless 170×48. This follows STORAGE_RESULTS.md.
Timing runs use `--clock-safe` (31.25 ms observers on this Windows loop), fresh
private profiles, original storage guards and no provider traffic. Timing runs
are serial and do not overlap targeted tests or CSS microbenchmarks.

## Root causes and changes

- CSS candidate selection rebuilt a set, checked ancestor requirements and sorted
  rules every time, even with identical selector inputs. A bounded per-stylesheet
  LRU reuses only ordered candidate rules. Live subject/ancestor names form the
  key, and a changed parsed rules-map identity discards the cache. No widgets or
  resolved style values are retained. Original Textual matching, pseudo-state,
  order, specificity, theme resolution and component processing still run.
- `_sync_console_rail_and_controls` built an inspector just to obtain a leading
  rail state. The subsequent control pass already builds the inspector and can
  supply it to the rail projection. Remove the leading build: the mounted
  synchronous pass builds two inspectors instead of three. Settings/fleet updates
  precede the control rail; hidden inspector publication and fresh post-await FULL
  work remain intact.
- The existing appearance cache required every requested conversation ID to be
  present in its returned map. Unknown/unsaved/deleted IDs are deliberately omitted
  by the service, so these forced repeated batched SELECTs throughout the display
  TTL. Track the last successfully queried IDs separately, including absent IDs.
  New IDs, expiry, a service replacement and explicit appearance writes still
  force original reads. A successful refresh removes requested entries no longer
  returned; failed reads do not mark IDs checked or renew expiry. Later queries
  do not extend the lifetime of earlier absent IDs.

No storage module, native admission, permission rule or execution authority
changes. Icon/color display caching retains its existing TTL. Existing ADR-126
and ADR-231 boundaries apply; no new architectural boundary or ADR is needed.

## CSS measurement

`plain-ready-presentation-css-observe.json` records four interleaved rounds on
the same mounted Console: 526 widgets, 523 distinct subject/ancestor shapes.
Each arm warms its own cache, then measures five complete `Stylesheet.update`
calls; the table contains the median for each round in milliseconds.

| Arm | Round 1 | Round 2 | Round 3 | Round 4 |
| --- | ---: | ---: | ---: | ---: |
| Original fast path | 256.98 | 246.60 | 261.15 | 250.96 |
| Initial 256-entry cache | 249.87 | 253.65 | 244.48 | 236.05 |
| Final 1024-entry cache | 196.39 | 192.37 | 196.59 | 194.31 |

Median-of-round-medians: **253.97 → 195.35 ms (23.1% lower)**. The initial bound
churned within one screen update; the final bound accommodates the measured
screen and its virtual components. All base and component styles matched in
every comparison (zero mismatches). This measures steady style application,
not startup parsing or a universal screen-switch improvement.

## Plain Windows return profiles

Each row contains six Home→Console returns. All drafts were retained, all attach
reconciliations completed, and all runs recorded zero network attempts and zero
real-profile guard events. The CSS diagnostic is excluded from this table.

| Run | Median input | Worst input | Median attach | Median per-return worst heartbeat overshoot |
| --- | ---: | ---: | ---: | ---: |
| Initial original | 1256 ms | 1785 ms | 1259 ms | 299 ms |
| Initial candidate, 256 entries | 1123 ms | 1576 ms | 1129 ms | 367 ms |
| Pinned original control | 1066 ms | 1617 ms | 1073 ms | 359 ms |
| Final candidate, 1024 entries | 1129 ms | 2523 ms | 1134 ms | 389 ms |

The final paired median input **worsened 5.9%** and the worst observed return
increased substantially. Do not claim an end-to-end or tail-latency improvement
from this change. The earlier pair improved 10.6%, illustrating run-to-run
variation. The repeatable evidence is narrower: cheaper full CSS application,
one fewer inspector build and successful reuse of absent appearance reads.

Overlapping main-thread sample counts, pinned control→final:

| Named frame | Control | Final |
| --- | ---: | ---: |
| Candidate selection | 16 | 8 |
| Inspector construction | 11 | 4 |
| Workspace browser presentation | 11 | 1 |
| Conversation appearance read | 10 | 0 |
| Home unread-notification count | 14 | 5 |

The outer CSS `apply` frame still leads overall (74→98 samples), mostly original
selector matching beneath the wrapper; these frames overlap and are not a
measurement of per-call time or invocation count. Original CSS matching, config
posture/presentation and Home notification reads remain investigation targets.
The safe targeted wins do not remove all event-loop stalls. Complete metrics:
`presentation-comparison.json`; named counts and final source hashes:
`presentation-hotspots.json`.

## Verification

- New CSS regression: original repeated selection hashes four fixture rules again;
  updated selection hashes none on the identical second request. Bounds, no widget
  retention, ancestor/subject changes, source replacement, disabled pseudo-state,
  theme changes, sibling order and recursive Input component styles are covered.
  Final four cases pass (`presentation-css-final.log`).
- Existing CSS file: seven cases pass when combined with the initial new tests,
  including real Console/Library/Settings resolved-style fidelity and upstream
  compatibility. Two existing cases fail only in native-worker teardown. Both
  reproduce unchanged with all three original modules pinned in their isolated
  children (`presentation-css-baseline.log`, source hashes in
  `presentation-baseline-receipts.jsonl`). Their own assertions pass. No storage
  check or teardown guard was weakened to make these green.
- Presentation/FULL targeted batch: 15 pass (`presentation-dedup-final.log`),
  including source/session/visit/maintenance changes, cancelled native-read
  retirement, appearance TTL/service/deletion behavior and mounted inspector
  counts. The original implementation fails the work-count and absent-ID controls
  (`presentation-rail-red.log`, `presentation-dedup-red.log`).
- Targeted Ruff checks pass for the CSS module, new tests and probe helpers.
  Existing screen/workspace lint debt is identical: 22 + 3 findings
  (`presentation-lint-baseline.json`). No full-suite sweep was run.

An additional existing workspace tick batch initially failed eight setup cases
with `raw_source_selection_changed`; all eight fail identically against the
pinned original (`presentation-tick-baseline.log`). Those harnesses import the
Console at collection but selected a different source per test. Their fixture
must retain the collection-time bootstrap profile, as the sibling FULL tests do;
the source-change refusal itself remains intact.

After that repair, eight cases passed and the zero-read case observed two reads,
including on a repeated candidate run; its pinned original control passed. This
was investigated rather than labelled a baseline failure. The corrected stack
observer (`presentation-tick-bindings-observed-corrected.log`) found both reads
in `_read_workspace_files_availability` under `console_preparation_reads.invoke`
on a worker thread. The counter wrappers select the original custom-reader
fallback, which calls both folder and runtime-binding readers. The counter now
distinguishes UI and background read/connection/SQL traffic while retaining real
native operations and positive controls. The first observer attempt raised from
`asyncio.current_task()` outside an event loop and is invalid evidence; its local
log is retained, and only the corrected observer is used here.

The nine workspace cases pass across targeted runs, not a single final nine-case
invocation: six unaffected cases passed in `presentation-tick-profile-fixture.log`;
the settled zero-UI-SQL and registry-mutation cases passed in
`presentation-tick-thread-counter.log`; the corrected positive control passed in
`presentation-tick-thread-control.log`. The new control initially assumed a clean
`ensure_default_workspace` must transact; it does not. It now explicitly executes
a real worker transaction and verifies transaction SQL is counted, preserving
that assertion. In total, **31 distinct targeted cases pass across the final
relevant runs**, with the two original CSS teardown failures separately audited.

## Reproduction

From the repository root, using native user-owned private fixture execution:

```powershell
.venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/probe.py --ready --resume-only --attach-poll --clock-safe --presentation-baseline --tag presentation-control
.venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/probe.py --ready --resume-only --attach-poll --clock-safe --tag presentation-final
.venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/probe.py --ready --resume-only --attach-poll --clock-safe --css-observe --tag presentation-css-observe
.venv/Scripts/python.exe Docs/Development/evidence/windows-performance-2026-10-18/summarize_full.py presentation
```

The baseline loader uses `git show` in memory; it does not replace working-tree
files. `--css-observe` is a separate diagnostic and its return timings are excluded
from paired timing comparisons. Samples count overlapping main-thread stacks,
not additive durations. Return-to-input measures a draft update, not paint or
completed Send/Stop. Small headless samples do not establish universal latency.
