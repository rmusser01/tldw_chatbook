# SQLite correction no-new-static-debt gate — 2026-09-08

Status: the owner-approved correction-specific static gate passes. Whole-file
lint and formatting remain nonzero. TASK-32160 remains In Progress; this is
Task15/AC16 completion, not Canvas V2 admission or whole-task completion.

Source-only change: `34891632589b123d2101e2551f0396fb97a679f0` →
`28a37eac8d2be04cdb13d8678ada9a48e9e8ca34`.
Correction attribution baseline: `9bc73ffb35ccd6eb24629bfa8021b28063dc9112`.

ADR required: yes; the owner-approved qualification-scope amendment is recorded
in [ADR125](../../../backlog/decisions/125-lock-safe-private-sqlite-validation.md).
No runtime, dependency, privacy, platform or other release gate changed.

## Scoped change and verification

Only the ten import blocks named by Task15 were reordered/grouped/formatted:
three DB test files, the startup performance test, console trace maintenance,
legacy collection recovery, the TTS package initializer, TTS migration
publication/recovery, and the runtime block starting at `app.py:114`.
No imports, comments, aliases, annotations, exports or functions were removed.
The preceding `install_stylesheet_fastpath()` call and later conditional/deferred
imports remain in place. There is no new framework, suppression or baseline file.

The implementer recorded actual Ruff I001 RED for all ten blocks before editing.
Afterward, whole-file I001 findings across those files dropped from 23 to 13;
all 13 residual blocks are outside the approved edit spans. Root independently
checked the final worktree against the pre-task commit before committing:

- All ten import-binding multisets match (430 bindings in total).
- Surrounding module AST, source outside the blocks (apart from boundary blank
  lines), and the multiset of all comments match.
- Every extracted block passes real Ruff I001 with its repository filename and
  explicit py312 target. Every exact import range passes Ruff format-check.
- `git diff --check` passes; the source commit contains only the ten named files.

The required covering selection ran once after the final source edits:

```text
../../.venv/bin/python -m pytest -q --tb=short \
  Tests/DB/test_chachanotes_connection_quiescence.py \
  Tests/DB/test_core_sqlite_owner_privacy.py \
  Tests/DB/test_private_sqlite_inventory.py \
  Tests/Performance/test_app_startup_performance.py \
  Tests/Chat/test_console_trace_compaction.py \
  Tests/Library/test_collections_legacy_recovery.py \
  Tests/TTS/test_profile_migration_publication.py \
  Tests/TTS/test_profile_migration_recovery.py
```

Result: **343 passed, 1 warning in 98.26s**, exit 0; no skips or deselections.
Repository pre-import isolation was used. The warning is the existing installed
Requests dependency-version warning; no dependency repair or suppression occurred.

## Whole-correction attribution

The comparison uses immutable Git sources for all **58 changed Python files**,
including the later Canvas tests and new Task14 CI contract test. Both arms use
Ruff 0.16.6, identical current-checkout configuration/discovery, `--no-cache`,
the real repository-relative stdin filename, and explicit `--target-version py312`.
The baseline-only py311 control measures target-sensitive recommendations, not
execution on an unsupported interpreter. No product module is imported.

| Lint classification | Count |
| --- | ---: |
| Baseline, py311 control | 1405 |
| Baseline, common py312 target | 1411 |
| Current source, common py312 target | 1363 |
| Current findings matching full unchanged baseline spans | 1360 |
| Remaining findings individually attributed below | 3 |

The ten corrected I001 findings are absent. Every other current lint diagnostic
has either an exact code/message/column and contiguous complete-source-span match,
or one of these specifically inspected attributions:

- UP036 at `Packaging/windows/build_windows.py:26` and
  `run_all_tests_with_report.py:166`: existing explicit unsupported-interpreter
  guards retargeted from 3.11 to the approved 3.12 floor. Root re-read both
  immutable versions; the guard behavior and diagnostic are retained deliberately.
- UP040 at `tldw_chatbook/TTS/profile_validation.py:118`: the exact
  `RowLike: TypeAlias = sqlite3.Row | Mapping[str, object]` line relocated from
  baseline `tldw_chatbook/TTS/profile_schema.py:142`. Root rechecked the exact text.

These are the owner's individually accepted exceptions, not a blanket allowance
for modified code. The aggregate decrease alone is not the basis for the verdict.

Formatting was compared over the same entire 58-file manifest, expanding the
earlier attribution report's narrower formatter selection. Baseline has 322 edit
groups; current has **319 edit groups in 12 files**. Every current group exactly
matches a baseline formatter edit's original/replacement content and immediately
adjacent source lines. There are **zero unmatched formatter groups**. The new
files have no formatting debt; the only lint debt in a new file is the attributed
relocated alias above. Both diagnostic scripts exit 0, meaning comparison success,
not that the underlying nonzero lint/format checks are clean.

The ten touched files alone still have 518 configured Ruff findings and two
whole-file formatter-dirty surfaces (console trace maintenance and app.py).
Their changed import spans are clean; remaining findings are covered by the
whole-correction attribution. Historical nonzero evidence is preserved in the
[earlier audit](2026-09-08-sqlite-static-attribution.md).

## Unchanged startup guards and independent review

Root ran the following exact guards together on source commit `28a37eac8d`:

```text
../../.venv/bin/python -m pytest -q --tb=short \
  Tests/Performance/test_app_import_weight.py::test_app_import_own_module_count_stays_at_the_post_diet_size \
  Tests/Performance/test_ui_ready_module_census.py::test_ui_ready_module_census_stays_at_the_pinned_size \
  Tests/Performance/test_screen_preimport_payload_budget.py::test_preimport_pass_payload_stays_within_budget
```

Result: **3 passed, 4 warnings in 13.67s**, exit 0. Counts remain **625/660**
imported Chatbook modules, **963/972** at UI readiness, and **499/500** preimport
modules. Payload remains 364325/378740 LOC; largest route remains
110163/123319 LOC. No budget or snapshot changed. The four warnings comprise
the inherited Requests warning and three deliberate headroom reports; preimport
has only one module of remaining headroom, not a new performance margin.

Independent task-scoped spec and quality review of the immutable source diff
approved the change with no Critical/Important findings. Its Minor is the
inherited Requests warning, retained here rather than repaired out of scope.
Its cross-task evidence checks are resolved by the root attribution/budget results
above and the linked ADR, plan and Backlog updates. No broad review was repeated.

## Evidence and limits

The existing ignored SDD directory for the SQLite implementation plan retains
`task-15-brief.md`, `task-15-report.md`, the immutable review package,
`task-15-{static,format}-attribution.py` and their JSON outputs,
`task-15-budget-guards.log`, and the progress ledger. The new diagnostic copies
take the source commit as their sole argument; the original Task7 evidence and
scripts are unchanged. No evidence directory was deleted.

Task14's fresh macOS concurrency evidence was not rerun. No local semaphore
control, broad test sweep, benchmark replay, host/dependency change, new CI/push,
PR/rebase/merge or V2 enablement occurred. Windows and unrelated platform/optional
coverage remain explicitly unverified under their existing contracts. Original
TASK-32160 AC1–7 remain unchecked pending final qualification reconciliation.
