# Library Decomposition Wave 3 — Combined Search+RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the entangled search+RAG cluster from `LibraryScreen` as ONE combined series (task-31203, wave-2's entanglement-gate outcome), after first settling controller-file size governance (task-31203 AC#4).

**Architecture:** Identical mechanics to the export/collections series: state object(s) + controller(s) under the byte-for-byte canon; recipe `backlog/docs/library-decomposition-recipe.md` is the how; this plan pins boundaries, order, and wave-specific decisions. The combined cluster ≈ 14 search + ~39 RAG methods and ~2 search + ~19 RAG fields (2026-09-02 census in the tracked SDD dir `.superpowers/sdd/2026-09-02-library-decomposition-wave2-cold-trio/task-8-report.md`); every task re-derives.

**Spec:** `Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md` (on dev, as corrected). **Recipe:** the mechanics authority, incl. all wave-2 lessons (§ RED-commit criterion, dict.get→setattr census, three test roots UI/Library/Live, sequential sweeps, verbatim comments by copy-paste, per-move pin lowering, rev-parse-only hashes).

## Global Constraints

- Everything the wave-2 plan's Global Constraints said, verbatim, plus its execution lessons (now recipe-recorded).
- Every move PR lowers the library `_BUDGETS` row to its own fresh post-move measurement; ceiling AND slack green in both guard files at every task boundary.
- Backlog ids: sweep the true max across origin/dev + all local branches before filing anything (three collisions bit this program already; `lessons-backlog-hygiene.md`).
- dev races: at every push, expect the strict-up-to-date cycle (catch-up merge → re-measure pins → regenerate diagnostic inventory via its own protocol → re-push). Budget for it.
- The search+RAG boundary INSIDE the combined cluster is an ownership-analysis output, not an assumption: one controller unless the analysis shows a clean seam (e.g. rag-answer pipeline vs search/history surface); a split is two sequential move commits with their own wiring pins.

---

### Task 1: Controller-file size governance (task-31203 AC#4)

Decide and implement governance for `Library_Modules` controller files (`library_collections_controller.py` ~1,689 lines, `library_conversations_controller.py` ~1,738, `library_export_controller.py` ~1,300+; wave 3 adds more). Options the implementer weighs (decision recorded with reasoning in the recipe): (a) `_BUDGETS` rows per controller (exact-pin, like the screen); (b) a single aggregate Library_Modules budget; (c) a looser per-file ceiling with slack tolerance. Constraints: whatever ships must be mutation-tested both directions and must not punish the byte-for-byte canon (moved bodies inflate controllers by design — the governance targets NEW code creep, not the moves; the chosen mechanism must distinguish, e.g. by baselining at each move's landing). Files: `Tests/Architecture/test_screen_size_ratchet.py` (or a sibling), recipe update, task-31203 AC#4 ticked.

### Tasks 2–4: Combined search+RAG series (state → controller(s) → cleanup)

Recipe verbatim; conversations/export/collections series as templates; wave-2 SDD census as the starting cluster map. Wave-specific notes:
- Characterization first (genuinely-unpressed `@on` handlers; RAG answer/search flows likely have deep existing coverage — verify, don't assume).
- Ownership analysis decides: single `LibraryRagSearchController` vs a two-controller split; `_library_rag_searched_query` and `_library_search_history` land per consumer census; `_library_collections_saved_searches*` already collections-owned (wave-2 verdict, recorded).
- RAG has worker-heavy paths (`_start_library_rag_query`, streaming answer application) — expect `@work`-decorated methods that CANNOT move (the export-series `@work`/DOMNode lesson); they stay as screen methods routed through named callables; enumerate early.
- Dynamic-dispatch census incl. dict.get→variable→setattr flows before any move; three test roots swept.
- RED wiring commit (screen untouched, pins failing at parent) → move commit(s) → blame-ignore (rev-parse) → cleanup PR (retargets, shim deletion at zero consumers, delegator census pruning, dead imports, recipe tables, fresh pins).

### Task 5: Wave close

Recipe trajectory + per-subsystem table updates; stale-doc sweep of the new modules; full verification battery (all wiring suites, all characterization files, both ratchets + census + governance guard from Task 1, preflight, sequential paired-baseline xdist sweep, probe run vs the recipe-recorded band); final whole-branch review (most capable model) → fix wave → PR against dev.

## Self-review record
- Governance-first ordering means wave 3's own new controllers are born governed.
- The @work enumeration note prevents the one structural surprise RAG is most likely to spring.
- All mechanics by reference to the recipe; this plan only adds the wave's decisions.
