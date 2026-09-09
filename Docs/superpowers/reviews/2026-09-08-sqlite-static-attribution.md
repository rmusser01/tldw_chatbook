# SQLite correction static attribution — 2026-09-08

Status: diagnostic comparison complete; static gates remain nonzero.
TASK-31942 stays In Progress and Canvas V2 stays disabled.

Baseline: `9bc73ffb35ccd6eb24629bfa8021b28063dc9112`.
Compared head: `d599fa037b0c5c7be6c53e8a316add84fd86cf4c`.
ADR required: no; this is the existing Task7 qualification analysis under
[ADR125](../../../backlog/decisions/125-lock-safe-private-sqlite-validation.md),
not a production, runtime, dependency or gate-policy change.

## Method and scope

The manifest is exactly `git diff --name-only BASE..HEAD -- '*.py'`: 57 files,
including the four Canvas test files added to the earlier 53-file scope.
Both lint arms use Ruff 0.16.6, current-checkout default discovery, explicit
`--target-version py312`, `--no-cache`, and immutable Git source supplied over
stdin with the real repository-relative filename. No source is imported or
modified. No existing file-mode changes occur in this comparison.

A baseline-only py311 arm identifies language-floor effects separately.
This is a controlled source comparison using today's tool and discovery context,
not a reconstruction of historical installed tooling. Ruff's verbose output
confirmed default settings and automatic py312 inference from requires-python.

A lint match requires identical code/message, columns, and complete diagnostic
span mapped through unchanged contiguous lines. Unmatched spans are candidates
for inspection, not automatic proof of newly introduced problems.

## Lint results

| Arm / classification | Diagnostics |
| --- | ---: |
| Baseline source, py311 control | 1405 |
| Baseline source, common py312 target | 1411 |
| Current source, common py312 target | 1373 |
| Current diagnostics matching unchanged baseline spans | 1360 |
| Current diagnostics requiring separate inspection | 13 |

The current count agrees with the saved 53-file report; the four later Canvas
test files contribute no lint diagnostics. The net decrease is not a claim
that every remaining issue is harmless or that every removed warning was a
deliberate fix.

The thirteen unmatched spans are accounted for as follows:

- **Ten modified import blocks already failed I001 at the same block starts
  in baseline.** They are in `test_chachanotes_connection_quiescence.py`,
  `test_core_sqlite_owner_privacy.py`, `test_private_sqlite_inventory.py`,
  `test_app_startup_performance.py`, `console_trace_maintenance.py`,
  `collections_legacy_recovery.py`, `TTS/__init__.py`,
  `profile_migration_publication.py`, `profile_migration_recovery.py`,
  and `app.py`. Focused baseline checks verified the actual I001 results.
  The blocks changed, so they are not labeled exact unchanged-span matches.
  This does not prove every newly added import is correctly ordered.
- **Two UP036 version guards** in `Packaging/windows/build_windows.py:26`
  and `run_all_tests_with_report.py:166` already failed on baseline; the
  diff updates their minimum from 3.11 to the approved 3.12 floor. Their
  diagnostic persists on the updated guard. Removing these explicit
  unsupported-interpreter checks is not authorized by this audit.
- **One relocated UP040 alias** at `TTS/profile_validation.py:118` is
  text-identical to baseline `TTS/profile_schema.py:142`:
  `RowLike: TypeAlias = sqlite3.Row | Mapping[str, object]`.
  It is a moved diagnostic, not an unchanged-location match.

Six diagnostics appear on unchanged baseline source when the lint target moves
from py311 to py312: two aliases in `profile_migration_candidate.py`, one generic
class and two generic functions in `profile_repository.py`, and the RowLike
alias above. These are language-target-sensitive recommendations, not six new
source edits. No annotation rewrite or lint suppression was made.

## Formatter results

Only the twelve formatter-dirty files named by the existing saved report were
compared. Their source has not changed since that report's 55f74aa009 checkpoint.
Both arms use Ruff's formatter on stdin under the same explicit py312 target.

Baseline has 309 edit groups; current has 307. **All 307 current formatter edit
groups exactly match baseline edit content plus the immediately adjacent source
lines.** None requires a new-debt classification within these twelve files.
This is stronger evidence than equal dirty-file counts; it is not a green
formatter result or a waiver.

The later served-flow test is outside those twelve files. Its pre-existing
formatter edits were separately verified unchanged during
[Task13](2026-09-08-canvas-startup-deadline-fix.md). No new whole-repository or
57-file formatter sweep is claimed here.

## Evidence and limits

Reproducible diagnostic scripts and compact JSON outputs are preserved under
`.superpowers/sdd/2026-09-07-sqlite-lock-safe-private-validation-implementation/`:

- `task-7-static-attribution.py` and `task-7-static-attribution.json`.
- `task-7-format-attribution.py` and `task-7-format-attribution.json`.
- `task-7-static-block-baselines.json` for the focused block checks.

Both final comparison scripts completed exit0; individual nonzero Ruff checks
are captured as diagnostics, not hidden. Diagnostic-harness attempts before
the final results included an unsupported missing-Git-path message variant and
oversized JSON/fix-context outputs truncated by the tool transport. Compact
outputs were subsequently read completely; truncated attempts are not the basis
of the classifications above.

No product/test code, dependencies, host resources, approvals or runtime budgets
changed. No pytest, browser, benchmark or semaphore allocation run occurred.
No prior review was repeated. The original failing static results remain on
record; this audit attributes them rather than making the static gate pass.

The eleven host-blocked concurrency cases still require a reported host-state
change and a successful isolated allocation control before retrying. Platform/
optional coverage remains separate. See
[final local qualification](2026-09-08-sqlite-final-local-qualification.md).

## Subsequent owner-approved gate

This report retains its historical scope and nonzero results. The subsequent
[Task15 correction and 58-file comparison](2026-09-08-sqlite-no-new-static-debt.md)
resolves the ten affected import blocks and satisfies the explicitly approved
correction-only no-new-static-debt gate. At `28a37eac8d`, 1363 lint findings
remain (1360 exact unchanged spans plus the three separately accepted findings);
all 319 current formatter groups match baseline over the expanded manifest.
Whole-file checks are still nonzero. The separately completed Task14 macOS
concurrency evidence and all remaining platform/admission limits are recorded
in the final qualification report; this earlier audit is not retroactively
expanded or relabeled as a clean lint run.
