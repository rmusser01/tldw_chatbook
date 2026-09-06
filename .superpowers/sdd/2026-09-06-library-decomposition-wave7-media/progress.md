# SDD ledger — plan: Docs/superpowers/plans/2026-09-06-library-decomposition-wave7-media.md
Wave-7 (media) SDD start. Branch refactor/library-decomp-wave7-media off origin/dev (post-wave-6 merge 761416317). Worktree: .worktrees/library-decomp-foundation (same venv).
Pre-flight conflict scan (4 tasks, waves 2-6 shape):
| pair | producer/consumer | finding |
| T1->T2 | LibraryMediaState + shims -> controller bindings | consistent; prompts template |
| T2->T3 | delegators/bindings -> retargets/prune | consistent; prune whitelist now 3-member (incl. on_<message>) |
| T1..T4 self-consistency | batteries name the guard set as of wave 6 (8 wiring suites at close; preimport suffix added in move commit) | consistent |
| plan vs rubric | no vacuous-test or duplication mandates | clean |
Wave-specific risks (plan Global Constraints): split decision (~244 methods — one controller would be ~8-9k lines; two-series or gate contingencies pinned); suspend/resume seam (TASK-31521 fields screen-owned, accessor-at-most); standing media_browse dev-red 410-vs-371 (absorb-only-if-touched rule); phase C explicitly fenced out. No conflicts requiring pre-execution rulings.
