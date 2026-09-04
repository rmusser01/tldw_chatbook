# SDD ledger — plan: Docs/superpowers/plans/2026-09-04-library-decomposition-wave4-skills.md

Spec (dev, corrected) + recipe §1-§18 binding. Branch: refactor/library-decomp-wave4-skills-ingest — NOTE branch name says skills-ingest but scope RULING below narrows to skills only (wave 5 = ingest).
Worktree: .worktrees/library-decomp-foundation (own venv, verified prior waves). Baseline: screen 43225/1311; skills cluster 133 methods (both prefix families, raw match) / 38 init-fields; existing skills controllers (import-coordinator, browse) untouched per plan.

## Pre-flight conflict scan
| Check | Finding |
|---|---|
| T1→T2→T3 same-shape series | Consistent with four prior series |
| Existing skills controllers vs new extraction | Plan pre-rules: untouched; delegating methods are exclusion candidates |
| Two prefix families + fourth test root (Tests/Skills) | Named in every task; the wave's likeliest novel trap |
| 133-method scale | Split-controller escape defined; sequential commits if split |
| Rubric conflicts | Same plan-mandated duplication as prior waves; carried in reviewer constraints |

Ruling: wave-4 scope = skills ONLY despite the branch name (measured 133/38 is the largest series yet; ingest 78/20 deferred to wave 5 rather than rushed). — Cost if wrong: branch name slightly misleading; content correct.
Standing rulings inherited: all of waves 1-3 (per-move pin lowering, no-red-ships, canon-docstring scope, born-governed, sweep protocol, monkeypatch/module-globals routing incl. the ingest-options trio which is INGEST-scope anyway, rev-parse hashes, verbatim comments, never-park, convergence loop at PR).

## Task log
