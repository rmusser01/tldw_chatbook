# Foreground and approval tracing recovery

`windows-foreground-approval-v2.zip` preserves the executed diagnostic;
`windows-foreground-approval-v3-997b33.zip` contains the separately corrected
approval probe, exercised by completed baseline `fgv3b1` and partial candidate `g32f`.
`windows-foreground-approval-v3-46ff4b.zip` is the separately reviewed candidate
bundle for `g34f`, SHA256
`167feb3cb3c189dfcd67eb363c24696da88e791337fbfe8b92c7c97e4bd4c610`.
It keeps the same baseline pins; 45 unique monitored candidate definitions
(48 labeled entries) were verified against q3. These are portable, source-only
**nonqualifying** bundles. Each
adjacent `.sha256` file pins its complete archive. They contain foreground
monitors, the respective approval monitors, installers, source-only
verification, a diagnostic summary helper, and the exact baseline/candidate
source pins. It contains no profile, interpreter, dependency, receipt, mutable
snapshot, user data or host path.

The baseline pins preserve the original local sources used by completed
`foreground-b1-20261117` (exit 0, original checks passed). This is the executed
approval **v2** probe, including its known ancestry false-negative: it reads
`_parent` directly, whereas pinned Textual stores a weakref in
`_MessagePump__parent`. That run recorded no original-label probe or qualified
approval paint. Preserve the diagnostic result; do not use v2 to claim paint
latency. The separate v3 baseline reached all five interactive approvals;
its actual region-cell delivery evidence is recorded below. Exact-label paint
remains unobserved.
The immutable v2 archive's candidate pins refer to `aab319544eb295d5b6ef78daab1bd8674743a839`.
The latest v3 candidate pins refer specifically to the product under q3:
`997b3339ffd37fab54dc28054ed4987fb933e478`. **Its full per-tool diagnostic has not
completed.** A later configuration fix or candidate is not covered. The immutable
archive describes source status at packaging time; the subsequent execution
record below does not change its source pins or claim a qualifying result.

## Native execution status

The original LOCAL baseline `fgv3b1` completed with **exit 0 / 1 passed** on
`b6b53f084a63058ca4b4dac99e074cff42192f7e`. Both original journeys and shutdown
passed: total 880.094 s, README 303.0486 s adjusted, approved edit 412.4056 s
adjusted. This is a new nonqualifying v3 run, not a correction to the v2 receipt.
All five approvals recorded qualified actual button-region cells with matching
live ancestry, visit and card generation; ready→cells was 0.033–0.210 s.
The 47 observed frames included five qualified region deliveries but zero
original-label probes or qualified fixed-label frames. Both observer summaries
closed with zero errors, source refusals, unfinished work and truncation.

Its fresh snapshot is `original-journey-win10-before-foreground-v3-20261118-a1`,
run `fgv3b1`, using the unchanged original QA and a private manifest/plugin/QA
copy and venv. Exhaustive verification checked 3,524 product sources and 6,333
private files before installation. The installed replay manifest SHA256 is
`b6fc22ba9d81981a8111fcc5b0df9c9f860f859237a5f4081b62415bc58e3580`.
The source-only archive stays unchanged; receipt and numeric details are in the
timing tracker. No selected-process per-resource retirement certificate is
inferred from this original local run.

On the frozen `997b3339ff` product, `g32f` installed the foreground and approval
v3 observers in both processes. It painted the original project disclosure at
Send +10.5304 s, then failed before the click or first provider request.
`ReadProjection` failed at application in
`ConsoleBackendReadiness.require_retired_context_change:861`; the original
backend exception was `RuntimeError`. This location alone does not identify
which checked predicate changed. The failed original total was 85.625 s and
the runner exited 1. Neither journey has a completed timing in this run.

All four observer summaries closed with zero errors, source refusals, unfinished
work and truncation. Both approval streams recorded revision 3 but **zero approval
stages**. The backend foreground observer recorded two step-write invocations
and two storage admissions, with no filesystem executor invocation. This proves
installation and bounded bookkeeping, not corrected ancestry or approval-paint
timing. There is no full candidate per-call comparison.

The separate `g33p` predicate diagnostic used the same product and original QA,
without these foreground/approval observers. README passed its original checks:
30.7859 s raw, 29.2408 s after the original measured waits, evidence verified at
Send +31.2873 s. Returning from Settings then failed at
`_require_selected_publication:75` through the decision timer, attach reconcile
and progress timer. The edit never reached Send; the failed original total was
120.968 s. Its predicate observer closed with zero failures in 5,780
`require_sources`, 5,780 `selection_is_current` and 10
`require_retired_context_change` calls. It did not reproduce `g32f`'s readiness
verifier failure and does not supply approval-paint evidence.

`g34f` uses the separate `46ff4b` bundle with foreground/approval observation and
the first-error observer. The optional source-predicate probe rejected changed
bytecode before the workload started and was omitted. That probe's rejection
  is not a product failure. The runner exited 1 after 66.984 s of original fixture
  time during Workspace setup, before either Send. ApplySourceRevision was
  refused at admission in `_require_rail_publication:294`. UI observer summaries
  closed cleanly with no tool/approval observations; backend closing summaries
  are absent. No per-tool comparison or retirement proof follows from this
  incomplete run. It must not inherit `997b33`'s compatibility pins.

See the [timing tracker](../windows-journey-timings.md) for source receipt names
and the distinct clean `q31c1` final-code-paint failure. Failed journey receipts
and completed observer bookkeeping are not physical-retirement certificates.

Compatibility was checked without importing product code: all original monitored
callables resolve; their bytecode, constants, names, closure and argument
signatures match the previously reviewed source, excluding source locations.
Fixed metadata arguments/free variables were checked explicitly. Whole-module
pins and callable coordinates were updated to the frozen candidate. The manifest
lists reference/candidate source lines and whether each target's module changed;
matching code and metadata contracts are required even when line positions move.

V3 reads only the exact `weakref.ReferenceType` from the original resident
`_MessagePump__parent` field. It does not call a parent property or arbitrary
callback. Source-only tests execute the pinned getter/setter AST inside its
original class name, covering Python name mangling, dead/wrong/malformed links,
overlapping foreground widgets, missing updates, and partial clipping. This
closes the demonstrated diagnostic error; it is not a native paint result.

V3 also reports `qualified_delivered_button_region_cells` separately. It requires
actual current ancestry/control eligibility, full clip/viewport containment and
region cells in an original supported compositor update observed at the original
display wrapper's return. It does **not** assert the fixed approval label was
painted. Installed/summary records carry `approval_probe_revision: 3`; original
label qualification and its missing observations remain independent.

## Restore sources and install

Verify the archive SHA256 before extracting it to a new local directory. Run:

```powershell
python -I -S verify_bundle.py
```

This checks every archived member hash and compiles Python syntax without
importing product code. The archive's `bundle-manifest.json` records its members,
input provenance and execution status when packaged.

Prerequisites remain the original frozen `before` overlay, the exact candidate
product overlay, reviewed installed dependencies, and fresh unused guarded
replays with their own functional virtual environments/manifests. The original
journey recovery README, original QA files, pair driver and existing
`guarded-windows-runner.zip` describe preparation. Historical host paths in that
older runner archive require explicit reviewed local configuration. No source or
environment is downloaded or reconstructed from hashes.

The installers use the existing reviewed incremental replay tool pack, including
its `replay.py` verification/atomic-write helpers. Copy the exact installer
sources from the extracted `foreground/` into a **new** local tool-pack copy.
Never overwrite a helper pinned by an active or historical run. Newly prepared
manifests must pin their actual reviewed helpers; no installer silently changes a
product/dependency pin or rewrites an existing runner pack.

For LOCAL baseline use `foreground/` and `approval/baseline/`. For the exact
process candidate use `foreground/` and `approval/candidate-ui/`; its backend
subset is included. Run the read-only preflight with locally configured paths:

```powershell
python -I -S install_tool_foreground.py --candidate-directory REPLAY --replay-tools TOOLS --observer-source FOREGROUND --approval-observer-source APPROVAL --check-only
```

Review that result, then repeat without `--check-only`. Route selection comes
from the original guarded manifest. The optional replacement installer accepts
only an unused already-instrumented replay and an exact previous approval source
hash. Neither installer edits a running snapshot. Original product, QA, network,
containment, ownership and retirement checks remain required.

Run the unchanged original fixture through its reviewed Windows runner with a
short fresh private label. Extra observation makes this run **ineligible for the
three qualifying pairs**; the normal pair-driver rules remain unchanged.

## Numeric analysis after the runner finishes

The adjacent tracked analysis reads one completed run and launches nothing:

```powershell
python -I -S analyze_diagnostics.py --run-directory FINISHED_RUN --output NEW_REPORT.json
python -I -S test_analyze_diagnostics.py
```

The output preserves that run's original total and both journey timings. It
reports approval stages, adjacent observed intervals, optional canonical
boundaries, actual foreground invocation entry/return locations, pre-executor
intervals, and storage durations grouped by exact monitored stack, phase and
thread. Actual executor-thread admissions are also clipped into before-executor,
request-build and after-executor intervals. Empty entry context may associate
only through its exact retained invocation and observed exit context, never a
nearby thread. Nested child intervals use their union; inclusive storage/thread totals
must not be added to critical-path wall time. Missing, reversed or scope-changing
boundaries remain null. No timestamp is subtracted across processes or streams.
The current adjacent analyzer additionally reports same-scope
`controls_ready_to_button_region_cells` and `button_region_cells_to_click`;
these do not populate the separate qualified fixed-label frame field. Historical
archives retain their packaged analyzer; use the tracked adjacent source for
these additional numeric fields.

`local_review_return` and `approval_request_return` follow user approval; they are
resolution completion, not policy work before paint. Original `painted_approval`
stays distinct from `qualified_rendered_fast_approve`. The latter requires the
actual active, enabled, mounted control, matching row/card generation and DOM
ancestry, full viewport/clip containment, original fixed-label match, and its
exact matched strip range inside the actual emitted chop/span. Partial clipping
or a label beyond a chop cannot qualify. The probe adds no renderer, layout,
query, crop or text-matcher call.

At the actual pilot observation/click, retained per-generation booleans explain
whether an eligible frame/cells/qualified label was seen. Absence alone does not
prove rendering was blocked. A Python return does not prove success, permission
or physical retirement. A complete diagnostic needs its closing summaries with
zero errors/refusals/unfinished work/truncation; diagnostic overhead remains part
of its measurements.

Additional source checks in the extracted bundle:

```powershell
python -I -S foreground/verify_static.py CANDIDATE ORIGINAL_BEFORE
python -I -S foreground/verify_offline.py
python -I -S approval/verify_static.py --variant VARIANT --source SOURCE --common COMMON_OBSERVER --dependency-root SITE_PACKAGES
```

These validate pins and synthetic bookkeeping/monitor plumbing, not the actual
journeys. `build_diagnostic_archive.py` is the source-only packaging recipe;
all machine-specific input locations are explicit CLI arguments and never enter
the archive. It refuses changed monitored definitions rather than inventing
compatibility. Use `--approval-revision 3` when packaging the v3 source; its
manifest records the source-only state at packaging time. Preserve that record;
append later native outcomes above rather than rebuilding historical archives.
The original source/environment prerequisites still apply.
