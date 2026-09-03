# SDD ledger — plan: Docs/superpowers/plans/2026-09-03-library-decomposition-wave3-search-rag.md

Spec: Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md (on dev; binding). Recipe: backlog/docs/library-decomposition-recipe.md (mechanics; carries all foundation+wave-2 lessons).
Branch: refactor/library-decomp-wave3-search-rag (cut from origin/dev @ 155ea1564 — the commit that merged wave 2; entire prior stack is mainline).
Worktree: .worktrees/library-decomp-foundation (reused; own venv verified in prior waves — re-verify at first test run).
Wave-2 census (task-31203 basis): search 14 methods / RAG ~39; entanglement 57.1%; tracked at .superpowers/sdd/2026-09-02-library-decomposition-wave2-cold-trio/task-8-report.md.

## Pre-flight conflict scan

| Pair / task | Check | Finding |
|---|---|---|
| T1 vs T2-4 | Governance guard lands before new controllers are born | Consistent — ordering is the point; governance must handle move-inflation (baselining at move landings) or the T2-4 moves trip it |
| T2-4 internal | Single vs split controller decided by ownership analysis, not assumed | Escape to BLOCKED defined in plan |
| @work methods | RAG worker paths cannot move (export-series lesson) | Plan pre-flags enumeration; screen-resident + named callables |
| Plan vs rubric | Same plan-mandated duplication (shims/delegators) as prior waves | Carried in reviewer constraints |
| Backlog ids | 31204/31205 filed on wave-2; dev max now ≥31205 | T1 and any filings sweep true max first |

Standing rulings inherited (all in force): stacked-branch discipline (though this wave is directly off dev); worktree-venv; sequential paired-baseline sweeps; per-move pin lowering; monkeypatch/module-globals routing; census-before-prune; rev-parse-only hashes; verbatim comments by copy-paste; RED commit = screen untouched + pins red at parent; never-park-silently in every dispatch; dev-race convergence loop for the eventual PR.

## Task log
