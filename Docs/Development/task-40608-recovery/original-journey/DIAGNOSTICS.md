# Foreground and approval tracing recovery

`windows-foreground-approval-v2.zip` preserves the executed diagnostic;
`windows-foreground-approval-v3-997b33.zip` contains the separately corrected, **unrun**
approval probe. Both are portable, source-only **nonqualifying** bundles. Each
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
latency. The separately identified v3 correction still needs a native run.
The immutable v2 archive's candidate pins refer to `aab319544eb295d5b6ef78daab1bd8674743a839`.
The latest v3 candidate pins refer specifically to the product under q3:
`997b3339ffd37fab54dc28054ed4987fb933e478`. **Its full per-tool diagnostic has not
run.** A later configuration fix or candidate is not covered.
The archive records no candidate timing result or executed overlay revision.

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
input provenance and unrun candidate status.

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
manifest explicitly records no baseline or candidate native run for that probe.
The original source/environment prerequisites still apply.
