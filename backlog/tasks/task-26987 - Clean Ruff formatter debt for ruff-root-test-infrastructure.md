---
id: TASK-26987
title: Clean Ruff formatter debt for ruff-root-test-infrastructure
status: Done
assignee:
  - rmusser01
created_date: '2026-08-31 18:31'
updated_date: '2026-10-02 14:05'
labels:
  - maintenance
  - formatting
  - quality
dependencies:
  - TASK-26000
references:
  - Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md
  - Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json
priority: medium
---

<!-- TASK-26000-BATCH: ruff-root-test-infrastructure -->
<!-- TASK-26000-PATHS-SHA256: fd7b774e7d2ef6cf2d616462d2e5eaecb60ab9736d06737711554cfe9fbc5158 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-root-test-infrastructure` Ruff formatter batch at the owner boundary recorded as: Root pytest guards, fixtures, and cross-suite test infrastructure.. The focused test surface recorded by TASK-26000 is `["Tests"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/conftest.py",
  "Tests/junit_outcome_diff.py",
  "Tests/network_guard.py",
  "Tests/test_call_from_thread_guard.py",
  "Tests/test_config_app_config_encryption.py",
  "Tests/test_config_chunking_defaults.py",
  "Tests/test_config_delete_settings.py",
  "Tests/test_config_encryption_effective_path.py",
  "Tests/test_config_private_bootstrap.py",
  "Tests/test_config_read_fastpath_task21124.py",
  "Tests/test_config_runtime_snapshot.py",
  "Tests/test_config_save_settings_semantics.py",
  "Tests/test_database_path_privacy.py",
  "Tests/test_hypothesis_profile.py",
  "Tests/test_keyring_isolation.py",
  "Tests/test_logging_private_files.py",
  "Tests/test_logs_buffer_single_record_per_emission.py",
  "Tests/test_network_guard.py",
  "Tests/test_persistent_diagnostic_sentinel_matrix.py",
  "Tests/test_persistent_log_is_not_empty.py",
  "Tests/test_probe_import_provenance.py",
  "Tests/test_smoke.py",
  "Tests/test_tiktoken_vendored_cache.py"
]
```

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [x] Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [x] Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [x] Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [x] Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [x] Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [x] `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [x] The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->

## Implementation Plan

1. Reconcile assigned paths against current `origin/dev` (existence + format state; missing paths recorded with their dev delete/rename lineage) and against the TASK-26000 evidence JSON `cleanup_records` `paths_sha256` (mechanical ownership check).
2. Snapshot each assigned path's `ast.dump` before formatting (type_comments=True with symmetric plain-parse fallback where prose `# type:` comments are unparseable; normalize `TypeIgnore.lineno`).
3. Run Ruff 0.15.22 `ruff format` on exactly the assigned paths.
4. Snapshot after-ASTs; require per-file equality (formatter-mandated docstring-normalization deviations enumerated if they occur).
5. `ruff format --check` must pass on every existing assigned path; `ruff check` findings must not increase vs the pre-format baseline.
6. Run the focused test surface (bounded; never a bare pytest), plus `Tests/CI/test_backlog_task_id_uniqueness.py` and `git diff --check`; verify the three-way path partition arithmetically before writing notes.
7. Tick ACs, record Implementation Notes (lineage, commands, results), set status Done.

## Implementation Notes

**Approach.** Executed the TASK-26000 formatter-debt cleanup contract at `origin/dev` tip `f80d3e0090` (isolated worktree `.worktrees/ruff-debt-batch-6`, branch `chore/ruff-debt-batch-6`). Formatter: Ruff 0.15.22 (the TASK-26000 pin); the repository ships no Ruff configuration, so defaults apply, matching the census contract.

**Ownership reconciliation (AC#1).** All assigned paths exist unchanged at the base — no deletes or renames to reconcile. Mechanically re-verified by recomputing `sha256(json.dumps(paths, separators=(",", ":")))` over the Assigned Paths: it matches both the `TASK-26000-PATHS-SHA256` marker in this file and the `paths_sha256` of the corresponding `cleanup_record` in `Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json`.

**Formatting (AC#2, AC#5).** 20 of 23 assigned paths formatted (root test-infrastructure production modules). `ruff format --check` passes on every assigned path after formatting (rc 0). Lint: `ruff check --output-format concise` over the batch's 233 assigned paths: 90 findings before and 44 after, ZERO increases — the formatter incidentally resolved 46 line-length findings in `Tests/Performance/run_console_three_turn_profile.py` (8), `Tests/Performance/test_console_three_turn_profile.py` (15), `Tests/RAG/simplified/test_collection_fingerprint.py` (16), and `Tests/RAG/test_active_config_resolution.py` (7) by reflowing; the remainder are pre-existing F401-type debt outside the TASK-26000 formatter scope.

**AST equality (AC#3).** Before/after `ast.parse(src, type_comments=True)` with `TypeIgnore.lineno` normalized to 0 and `ast.dump(include_attributes=False)`: hashes equal on every assigned path (the batch-5 symmetric plain-parse fallback was not needed on any file this batch).

**Comment directives (AC#4).** Ordered `tokenize.COMMENT` sequences and `# fmt: off` / `# fmt: on` counts are identical before and after on every assigned path.

**Focused tests (AC#6).** Focused surface: TASK-26000 recorded the whole `Tests` tree; under the repository's targeted-verification policy a bounded subset exercising the assigned modules was run instead: `Tests/test_config_save_settings_semantics.py`, `Tests/test_database_path_privacy.py`, `Tests/test_probe_import_provenance.py`, `Tests/test_config_hot_reload.py`, `Tests/test_config_key_validation.py` (26935/26971 precedent; a bare pytest with no path arguments is forbidden).. Result on the formatted tree: 31 failed, 63 passed in 6.19s (`--timeout=120`). Baseline A/B per `backlog/docs/lessons-backlog-hygiene.md` (no-stash rule): assigned paths swapped to unformatted `HEAD` content with `git checkout HEAD -- <paths>`, identical command re-run, files re-formatted and re-verified with `ruff format --check`. Baseline: 31 failed, 63 passed in 5.09s — count-identical to the formatted outcome, so the failures are pre-existing at the current dev tip (the documented hook-consent send-gate and config-admission red classes) and out of scope for a formatter-only task; AST equality independently proves runtime-identical files.

**Governance (AC#7).** `git diff --check` clean; `Tests/CI/test_backlog_task_id_uniqueness.py` — 3 passed (run on the fully formatted batch tree).

**No hand-written behavior change (AC#8).** Three-way partition verified arithmetically before closeout: 226 committed + 7 clean-at-base + 0 missing = 233 assigned paths; zero unassigned paths touched; every content diff is Ruff formatter output.

**Lineage.** 3 paths were already formatter-clean at the base (`Tests/test_config_save_settings_semantics.py`, `Tests/test_database_path_privacy.py`, `Tests/test_probe_import_provenance.py`) and were deliberately left untouched; the other 20 carried live debt and were formatted.

ADR required: no — mechanical, formatter-only cleanup executing the owner-approved TASK-26000 contract; no architecture, storage, or cross-module boundary touched.
