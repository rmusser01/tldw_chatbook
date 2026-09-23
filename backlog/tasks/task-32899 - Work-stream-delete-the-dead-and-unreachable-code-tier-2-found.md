---
id: TASK-32899
title: "Work stream: delete the dead and unreachable code tier 2 found"
status: To Do
assignee: []
created_date: '2026-09-21 23:05'
labels:
  - tier2-review
  - review-dead
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
About 24 deletion candidates in packages none of TASK-32807's six sub-tasks names, each with production
importer counts resolved by AST walk (relative levels resolved, function-body imports included) and
reachability resolved by **running** `resolve_screen_route()` rather than reading the registry.

Two entries are deliberately **not** deletions and must not be swept into one:
- the `Evals/` legacy run stack (~9,000 lines) is unreachable but carries a `subprocess` code-execution
  sandbox -- it needs a ruling (TASK-32904), not a cleanup PR;
- `Widgets/Tamagotchi/`'s widget half (2,181 lines) is a product decision (TASK-32905), because the
  storage half is wired into backup/recovery and the private-SQLite allowlist.

Watch for test-only lifelines: several candidates are kept green by tests that exist only to import them,
and `Local_Inference/mlx_lm_inference_local.py` has **30 tests asserting dead code's behaviour** -- a
future fixer will "repair" it by mistake. Delete the tests with the code.

Source: tier-2 code review 2026-09-21 -- `qa/tier2-code-review-2026-09-21/report.md` (26 slices, 890,356 lines: the surface tier 1 never reached). Per-slice evidence in `qa/tier2-code-review-2026-09-21/slices/`, reproductions in `phase4-verification.md`, and per-finding re-validation against `origin/dev d0face3ebe` in `validation/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each deletion states its production importer count and its route reachability
- [ ] #2 Test-only lifelines are deleted with the code they pin
- [ ] #3 `Prompt_Management/Prompt_Engineering.py`'s metaprompt is extracted for task-474 before deletion
- [ ] #4 Deletions land one PR per package family, not one PR for all of them
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented on `fix/tier2-dead` as **11 commits, one per package family**. Not pushed.
**13,454 lines / 28 files removed.** preflight GREEN; size ratchet node-id set identical to `origin/dev`;
`import tldw_chatbook.app` and `tldw_chatbook.cli` both succeed; zero dangling *imports* of any deleted
module (independently re-checked — the residual textual references are comments and docstrings only,
several of them deliberate breadcrumbs such as `screen_registry.py:275`).

**The most valuable output of this stream is the three deletions it refused.** Every importer count was
re-derived by AST walk rather than taken from the review's table, and three rows did not survive:

| row | review said | actually | evidence |
|---|---|---|---|
| 6 `*_Interop` modules (753 lines) | 0 importers, delete | **reachable — KEPT** | The review's **own** `slices/S24-interop-cluster.md:106-108` already said so: a naive walk reports 7 dead packages, but *"the lazy-aware walk reports **0 dead packages**. The naive answer is the trap here."* And `report.md`'s own Retired/contested section says *"All 31 are reachable from the composition root; residue is 753 lines (1.1%)."* The table contradicted both. |
| `swarmui_client.py` + `image_generation_service.py` (868 lines) | "0 live", delete | **live — KEPT** | `chat_screen.py` → `UI/Console_Modules/image.py` → `console_generate_image.py:436` imports `ImageGenerationService`, which imports `SwarmUIClient` at module scope (`:11`); also re-exported by `Media_Creation/__init__.py:5`. |
| `media_screen.py` | appears in a **delete** row *and* a **keep** row of the same table | delete | The keep reason (registry save_state/restore_state tests) is a test-only lifeline whose four tests were already red at baseline. |

`report.md`'s Legacy-reachability table has been struck through and annotated for all four affected rows,
and the pattern is recorded in `validation/PHASE-A-SUMMARY.md`: **every error found in this review so far is
in a summary artefact — a title, a headline count, or a roll-up table. Not one has been in an evidence
block.** The operational rule that held every time: read the evidence block, re-derive every count.

Two further corrections to figures: `Tests/LLM_Management/test_mlx_lm.py` collects **19** tests, not the 30
claimed; and a naive grep for `mlx_lm_inference_local` returns 2 hits that are the module's own header and
footer comments, so the review's "0 importers" was right and the grep was not.

Method: relative import levels resolved, function-body imports included, own file excluded, cross-checked
against the PEP 562 `_LAZY_EXPORTS`/`_EXPORT_MODULES`/`_SUBMODULE_BY_NAME`/`_SCREEN_EXPORTS` tables. Route
questions answered by **running** `resolve_screen_route()` over all 52 targets, never by reading the
registry. Each touched suite stash-verified against `origin/dev` on the failing-**name** set: family 10 went
208 red -> 202, the difference being exactly the 6 `MediaScreen` tests removed, all already red. Zero new reds.

Side effects worth knowing: deleting `XML_Ingestion.py` (dead — it raises `ImportError` on a name
`Client_Media_DB_v2` no longer exports) closes one entry in the unhardened-XML register, 7 -> 6. The
diagnostic regeneration dropped 9 legacy path-privacy candidates, including a pid, a full command line and
three raw file paths.

**Newly orphaned, deliberately left out of scope** — these became dead *because of* these deletions and
should be a follow-up rather than a same-PR widening: `app.chat_wrapper` +
`worker_events.chat_wrapper_function` (last caller was `MediaWindow_v2`; its census pin now asserts zero
approved callers, which is strictly stronger than before); `Widgets/Media/{media_viewer_panel,
media_navigation_panel,media_search_panel}.py` + `Event_Handlers/media_events.py`;
`Widgets/Coding_Widgets/repo_tree_widgets.py` (726 lines, 5 test files of its own).

Live defect found and **not** fixed here, filed as **TASK-32910**: Home's `review_read_later` deep link is
silently dropped — it emits `MEDIA_NAV_CONTEXT_BROWSE_SUBVIEW`, whose only consumer was the unreachable
`MediaScreen`. Pre-existing, and re-verified independently: the key now has zero readers repo-wide.
<!-- SECTION:NOTES:END -->
