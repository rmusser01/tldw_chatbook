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

Task 1: implementer DONE (controller-file size governance, task-31203 AC#4) — chose
option (a) exact per-file `_BUDGETS` rows, new guard
`Tests/Architecture/test_library_modules_size_ratchet.py`, discovered by glob
(`UI/Library_Modules/*_controller.py`) so an unlisted controller fails loudly instead
of landing ungoverned (property `test_screen_size_ratchet.py`'s hand-kept dict lacks);
12 controllers pinned at exact measured line counts (699-2,023 lines, 11,130 total);
method count deliberately NOT tracked (justified: no reliable dominant-class-per-file
convention, e.g. `LibrarySkillImportCoordinator` in a `_controller.py` file, and
summing all classes would punish the canon's own helper-class pattern). Mutation-
tested all 4 directions (growth trip, anti-slack trip, unlisted-existing-file,
unlisted-new-file) plus the tolerance boundary (50 passes/51 fails); each reverted
cleanly, confirmed via empty `git diff`. Battery: new guard 25/25 passed; guard +
screen ratchet 28 passed/2 failed (both pre-existing chat_screen reds); Library
recompose census 6/6 passed; preflight clean (5/5 checks); full `Tests/Architecture`
527 passed/15 failed/1 skipped, all 15 confirmed pre-existing via `git stash`
re-run against the unmodified base. Recipe updated with new §17 (decision, reasoning,
re-pin-at-move flow, measured rows). task-31203 AC#4 checked; task stays To Do
(AC#1-3 out of scope). Full report: task-1-report.md.
