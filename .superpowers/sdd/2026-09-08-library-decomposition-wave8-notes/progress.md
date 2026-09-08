# SDD ledger — plan: Docs/superpowers/plans/2026-09-08-library-decomposition-wave8-notes.md
Wave-8 (notes) SDD start — THE FINAL EXTRACTION WAVE. Branch refactor/library-decomp-wave8-notes off origin/dev (post-wave-7 merge aecb14720). Worktree: .worktrees/library-decomp-foundation (same venv).
Pre-flight conflict scan (4 tasks, established shape):
| pair | producer/consumer | finding |
| T1->T2 | LibraryNotesState + shims -> controller bindings | consistent; media template |
| T2->T3 | delegators/bindings -> retargets/prune | consistent; three-member whitelist + five spellings |
| T2 canvas_sync branch -> T3 census | branch added in move commit, guard beside precedents; census counts it | consistent |
| T1..T4 self-consistency | batteries name the guard set as of wave 7 | consistent |
| plan vs rubric | no vacuous-test or duplication mandates | clean |
Wave-specific risks pinned: the three wave-7 inheritances (on_<message> enumeration with whitelist evidence; canvas_sync third branch + guard; shared-seam fixture census); four-member shell family stays whole; four prior-extracted wiring modules; sync/workspace seams; id-collision risk HIGH at filing time. No conflicts requiring pre-execution rulings.
