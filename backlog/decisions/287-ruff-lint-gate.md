# ADR-287: Ruff lint gate (zero-F policy + ratcheted baseline)

Status: Accepted 2026-11-02
Date: 2026-11-02
Related Task: [TASK-38600](../tasks/task-38600%20-%20Add-ruff-lint-gate-to-CI-and-clear-unused-import-debt.md)

## Context

The 2026-11-02 bug sweep audit (bug-sweep-2026-11-02-dev.md, section 4) ran
ruff 0.16.6 with `select = ["F", "B", "ASYNC", "PLE"]` over `tldw_chatbook`
and found ~1,210 findings — including 126 unused imports (F401), 8 undefined
names (F821), 7 redefinitions (F811), 5 unused locals (F841) and 1 pointless
f-string (F541). None of the repository's 24 CI workflows ran any linter, so
this debt accumulated invisibly.

Roughly 1,000 of the findings (B904 raise-from, B905 zip-strict, PLE1205
logging-format, B009/B010 getattr/setattr literals, ASYNC*) are spread across
live code paths where a mechanical mass-fix would be high-risk inside a
bug-fix PR, and several encode real review decisions per site (which
`raise ... from` cause, whether a `zip` should be `strict=`).

The F-class findings, by contrast, are mostly dead imports/locals with no
runtime effect once removed, and were fixed for real in TASK-38600 (146
findings → 0). Intentional re-exports that consumers rely on keep their import
under a reasoned `# noqa: F401` or a static `__all__` (e.g.
`Chat/console_glyphs.py` glyph re-exports per ADR-034;
`LLM_Calls/LLM_API_Calls.py` provider dispatch surface; three
`library_screen.py` names consumed by tests).

## Decision

1. **Ruff is the project linter**, configured in `pyproject.toml` under
   `[tool.ruff]` with `target-version = "py312"` and
   `[tool.ruff.lint] select = ["F", "B", "ASYNC", "PLE"]`. No generated
   directories exist in the package, so no `extend-exclude` entries are
   needed (`tldw_chatbook/Tools/remote_worker_bundle.py` is tracked source
   and stays in scope).

2. **Zero-F policy.** F-class findings (undefined names, unused
   imports/locals, redefinitions) are never baselined. The `gate-new` job in
   `.github/workflows/lint.yml` runs
   `ruff check tldw_chatbook --select F --output-format concise` and fails on
   ANY output. New F findings fail CI immediately.

3. **Ratcheted baseline for the rest.** Remaining per-rule debt is recorded in
   `Tests/lint/ruff_baseline.txt` (1064 findings across 26 rules at
   acceptance). The `gate-ratchet` job runs
   `python Tests/lint/ruff_gate.py`, which aggregates current per-rule counts
   and fails if ANY rule exceeds its baseline count; a rule absent from the
   baseline defaults to 0, so new rule codes also fail. Equal or lower passes,
   and improvements are printed so gains are visible. The baseline only
   shrinks deliberately: run `python Tests/lint/ruff_gate.py --update-baseline`
   with the pinned ruff version as part of a dedicated debt-reduction change.

4. **The ruff version is pinned in CI** (`ruff==0.16.6` when accepted) and
   recorded in the baseline header, because rule behavior and finding counts
   shift between ruff releases; an unpinned gate would ratchet against a
   moving target. Bumping the pin and regenerating the baseline happen
   together in one change.

5. The workflow runs on `pull_request` and `push` to `dev`, matching the
   branch-protection model (AGENTS.md: one PR per CI cycle under strict
   protection). Making the two jobs required checks is a repository-settings
   action taken when this gate is adopted, not part of the workflow file.

## Consequences

- New F, B, ASYNC or PLE findings fail CI; debt can only grow through a
  deliberate, reviewable baseline bump.
- Dead-import and undefined-name regressions (the classes that hide real bugs
  like the pre-fix F821s in `Tools/local_tool_impls.py` and
  `Agents/virtual_cli_provider.py`) are caught before merge.
- Contributors touching code with baselined debt are NOT forced to fix it
  (no `--fix` mandate, no per-file ratchets), but cannot make it worse.
- Keeping F at zero requires occasional adjudication: intentional re-exports
  need a static `__all__` or a `# noqa: F401` with a reason; every noqa in
  this codebase must carry its reason per TASK-38600.
- A ruff major/minor bump can shift counts in both directions; the pin makes
  each bump an explicit, baseline-accompanying decision.

## Alternatives considered

- **Fix everything now, gate clean.** Rejected: ~1,000 findings across B904/
  B905/PLE1205/B009/ASYNC need per-site judgment (which exception to chain
  from, whether a zip is length-matched); bulk-changing them inside the
  audit-fix PR would bury real bug fixes in mechanical churn and risk
  behavioral regressions in error paths.
- **Enable only the currently-clean rules.** Rejected: the dirty rules would
  stay invisible and the debt would keep growing; the ratchet makes the exact
  number a visible, enforced ceiling instead.
- **Pre-commit/local-only linting without CI.** Rejected: nothing in the 24
  existing workflows would enforce it, which is exactly how 1,210 findings
  accumulated; a gate that is not in CI is a suggestion.
- **Per-file or per-line `# noqa` instead of a baseline file.** Rejected for
  non-F rules: it would spread ~1,000 suppressions through the source, hiding
  the debt at the sites instead of centralizing it for deliberate paydown.
