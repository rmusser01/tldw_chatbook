# Settings and Library follow-up

Implemented over HEAD `69771c1811e734421f0bdf31ed90969f8c1f0c63` on native
Windows/Python 3.12.10/Textual 8.2.8. Changes remain uncommitted.

## What changed

- Settings rendering now reads a nullable prompt-count snapshot. An exclusive
  thread worker starts after the first refresh and runs the **original**
  `authoring.customized_count()` with its complete checked config lifetime.
  Config identity, revision, cancellation and attachment fence publication.
  Loading and failure are distinct from zero customized prompts; editor changes
  update existing rows and invalidate old results.
- Library construction reuses a ready Notes-sync owner or a preparing display.
  The original safe app reader runs after mount through the existing finite
  service-worker boundary. Direct Notes attention/location/hint readers use that
  boundary too; listener registration never invokes the lazy constructor.
  Failed preparation preserves known attention. Concurrent safe readers share
  failure publication as well as construction, preventing contradictory outcomes.
- No native storage/path checks, permission rules, prompt admission scope, sync
  conflict policy, or reviewed-write authorization were removed or relaxed.

ADR required: no. ADR path: N/A. Routine scheduling and concurrency corrections
preserve ADR-126 and TASK-21249's app-owned lifecycle and failure contract.
Tasks: TASK-35587 and TASK-35588.

## Measurements

All reported timing runs use the real application, a new private profile, network
guards, the same 170×48 headless viewport, and no cProfile. They do **not** measure
Windows Terminal drawing or real provider/network latency. No tests ran alongside
the two valid controlled Settings measurements or the final full tour.

### Controlled Settings comparison

`plain-ready-settings-original-valid.json` loads the exact original HEAD Settings
module in memory, retaining its original resource path. The working source file
is never replaced. `plain-ready-settings-fixed-controlled.json` runs the changed
module. Both prewarm the Settings import and one original checked count before
six alternating Home/Settings visits; neither patches the measured count getter.

| Request → after-refresh | Original | Fixed |
|---|---:|---:|
| Repeat-visit median (five visits) | 1,141 ms | 833 ms |
| Repeat-visit range | 905–1,532 ms | 657–1,121 ms |
| First visit | 928 ms | 1,316 ms |

Repeat median improves approximately **27%**. First visit is slower in this
comparison: do not claim a universal opening-time improvement. The main-thread
sampler observes **zero** customized-count stacks in both final fixed runs;
deterministic blocked-reader tests independently verify responsiveness.

An initial implementation started the worker during mount and removed the
main-thread count but did not improve the full-tour median. Starting it after
refresh avoids making the initial frame compete with that read. Its measurements
remain in `plain-ready-settings-fixed.*`; the subsequent focused experiment is
`plain-ready-settings-deferred.*`. The first original-source loader attempt
(`settings-original-controlled`) relocated module resources and timed out; it is
invalid timing evidence and was replaced by `settings-original-valid`.

### Final full tour

Compare the original retained `plain-ready.json` with
`plain-ready-settings-library-fixed.json`:

| Measurement | Original | Fixed |
|---|---:|---:|
| First Library screen arrival | 1,074 ms | 534 ms |
| First Library after-refresh | 2,519 ms | 2,230 ms |
| Repeat Library after-refresh | 110 / 120 ms | 261 / 123 ms |
| Settings after-refresh median (three visits) | 2,258 ms | 2,145 ms |

Library first arrival is about 50% faster and first refresh about 11% faster in
this **single comparison**, while repeat visits remain variable. The final
main-thread sample contains **zero** Notes-sync owner-construction stacks during
Library visits. These results establish removal of the measured blocking paths,
not a general startup budget or whole-app performance guarantee.

Both final runs complete normally with **zero network attempts and zero real
profile guard refusals**. All native storage checks remain active.

## Validation and limits

- `targeted-verified-tests.log`: **51 passed, one deselected**, covering Settings
  worker fencing, original prompt admission, Library construction/publication,
  sync roots, startup gates, concurrent failed readers and attention listeners.
- `native-worker-final-tests.log`: **22 passed**, including an additional test
  that runs the actual unmocked checked count through the Textual thread worker.
  Together these cover 52 distinct passing cases; the final batch repeats 21.
- Red controls are retained for the original Settings behavior, eager Library
  constructor, direct Notes attention getter, and concurrent failed-reader race.
- One older mounted focus test and seven older authoring fixture cases fail
  identically with the original HEAD modules. `compatibility-baseline.log` records
  the exact control: focus lands on `notes-sync-body`; authoring setup refuses
  `raw_source_selection_changed`. They were not weakened or counted as passes.
- Independent static review found two Library issues, both fixed and regression
  tested, then approved the updated scope. No full suite was run.
- New test/probe files pass Ruff. Per-file original/current lint comparison adds
  **zero diagnostics** (`lint-comparison.json`); existing screen/module lint debt
  remains. Modified regions and new tests pass formatting; whitespace checks pass.

## Console investigation

See `CONSOLE_INVESTIGATION.md` for the source paths and safety boundaries. Console
code is unchanged. The final tour still records a **1,167 ms worst key latency**
(median 3.1 ms, twelve keys). Samples during typing include synchronous local
skill-context admission, workspace registry reads, and token-estimator setup.
Thus the remaining Console problem is real even after the earlier fixes.

The next bounded target is local skill-context discovery: its async wrapper still
does synchronous native admission and scanning, and warm resumes queue it again.
Attention-mark reads and active-workspace reconciliation are the other measured
candidates. Coalesce work and prepare original checked reads off-loop, preserving
custom-service affinity and source/view fences. FULL reconciliation is an
authoritative path: ADR-231 explicitly forbids using display-only proofs there.

## Reproduction

Run `probe.py --ready --tag <unique-name>` for the full tour. Add
`--settings-only --warm-settings` for the focused six-visit tour. To reproduce the
original-source arm, export HEAD's `tldw_chatbook/UI/Screens/settings_screen.py`
to a temporary file and pass `--settings-source <file>` with those same flags.
The script compiles that source with the original module filename so relative
resources resolve correctly. The temporary exported source was removed after
measurement; all raw measurements remain.
All recorded disposable profiling homes were removed after their processes
exited. Final product/test file hashes are in `final-source-hashes.json`.

`baseline_tests.py <pytest arguments>` loads original HEAD UI modules in memory
for the compatibility control. Run native tests with
`python -m Tests.windows_private_fixture_runner <targeted tests>` and a private
TEMP/TMP directory directly under the user's home, as documented in REPORT.md.
