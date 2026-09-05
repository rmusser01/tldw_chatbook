# SDD ledger — plan: Docs/superpowers/plans/2026-09-05-library-decomposition-wave6-prompts.md
Wave-6 (prompts) SDD start. Branch refactor/library-decomp-wave6-prompts off origin/dev (post-wave-5 merge 7aa048790). Worktree: .worktrees/library-decomp-foundation (same venv).
Pre-flight conflict scan (plan is 4 tasks, same shape as waves 2-5; table):
| pair | producer/consumer | finding |
| T1->T2 | LibraryPromptsState fields + shims -> controller bindings | consistent; T2 consumes T1's state accessor naming per ingest template |
| T2->T3 | delegators + bindings -> retargets/prune | consistent; prune only at zero consumers per census |
| T1..T4 self-consistency | each task names its own battery; close names all seven wiring suites | consistent with the guard set now on dev (preimport-closure + ui_ready census added per round-3) |
| plan vs rubric | no mandated vacuous tests, no verbatim-logic duplication mandates | clean |
Known wave-specific risks (from plan Global Constraints): born-lazy controller import (preimport-closure guard), library_prompts_state basename collision (package-qualified greps), two prefix families, 4 extra prompts test roots. No conflicts requiring pre-execution rulings.
