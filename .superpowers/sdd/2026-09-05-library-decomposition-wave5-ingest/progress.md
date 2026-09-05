# SDD ledger — plan: Docs/superpowers/plans/2026-09-05-library-decomposition-wave5-ingest.md

Spec (dev) + recipe §1-§19 binding. Branch: refactor/library-decomp-wave5-ingest (cut from origin/dev after the wave-4 merge). Worktree: .worktrees/library-decomp-foundation (own venv). Baseline: screen 41574/1302; ingest 78 methods/20 fields.

## Pre-flight conflict scan
| Check | Finding |
|---|---|
| T1→T2→T3→T4 same-shape series | Consistent with five prior series |
| Ingest-options trio | Named no-touch (module-globals, foundation-era discovery); verify location, never move |
| @work-heavy flows + worker groups | Early enumeration mandated |
| Form-persistence contract (task-2043) | Pin via characterization if uncovered, before any move |
| Rubric conflicts | Same plan-mandated duplication; carried in reviewer constraints |

Standing rulings inherited: all of waves 1-4.

## Task log
Task 1: dispatched (implementer sonnet), BASE=9e62dd8f7 — ingest state PR (series 1/3)
Task 1: INCIDENT — implementer killed by session usage limit mid-baseline; its in-place path-scoped checkout of the pre-task commit was left in the worktree (would have corrupted the next commit). Controller restored to HEAD; agent resumed with the isolated-baseline method mandated. Recipe candidate lesson for wave close: in-place baseline checkouts are interruption-unsafe — always use an isolated worktree/archive for baseline runs.
Task 1: DONE — a11220648 (RED: characterization + wiring pins) → 12ba4fb13 (GREEN: LibraryIngestState + shims, 20/20 fields, ratchet 41574/1302 → 41520/1302). Found and fixed a new bypass shape (`object.__new__(LibraryScreen)`/`.LibraryScreen.__new__` fixtures across 4 test files hand-setting flat `_library_ingest_*` names — 24 call sites patched with one `screen._ingest_state = LibraryIngestState()` line each, zero assertions touched) inside the same GREEN commit per no-red-ships. Full battery green: wiring RED→GREEN, characterization pre-change PASS (form-persistence contract already covered, no new pin needed), both ratchet rows, `-k "ingest and library"` (1298 passed/7 pre-existing), full sequential xdist paired-baseline sweep via isolated `git worktree` (356/3989 branch vs 370/3971 baseline, 8 branch-unique all resolved, zero real regressions), preflight all-green. Recipe §7 pre-existing-failure list and a new isolated-worktree-baseline lesson recorded. Report: .superpowers/sdd/2026-09-05-library-decomposition-wave5-ingest/task-1-report.md.
