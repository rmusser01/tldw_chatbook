---
id: TASK-16835
title: 'Wire or retire the multi-item review batch-analysis path (dead LLM branch, no event poster)'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-16'
updated_date: '2026-08-16'
labels:
  - dead-code
  - event-handlers
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-16194 (PR #1671) repaired `Event_Handlers/multi_item_review_events.py`'s four
nonexistent `app.run_in_thread` calls, but its review surfaced a pre-existing gap it
correctly left unfixed: **`app.llm_api_client` is never assigned anywhere in the live
app**. `grep -rn "llm_api_client" tldw_chatbook/` finds only the guards that read it
(`multi_item_review_events.py:85`, `:174`) and the call through it (`:194`
`app.llm_api_client.chat_with_model`), and `git log --all -S "llm_api_client"` shows the
attribute was never introduced in `app.py` at any point in history (review16194 §5). The
`hasattr` guard has therefore always been False in production: every "LLM analysis"
silently falls back to `generate_placeholder_analysis`.

Verified still true at dev `ee741cf10` — and the situation is one step worse than the
review recorded: the only production consumer of this module is `app.py:11727-11730`,
which dispatches `handle_batch_analysis_start` on a `BatchAnalysisStartEvent`, and
**nothing constructs or posts that event anywhere in `tldw_chatbook/`** — its poster was
`MultiItemReviewWindow`, deleted as dead code by TASK-1010 (PR #1019). So the whole
batch-analysis handler path is unreachable, and even if reached it would only produce
placeholders.

Decide: either wire the feature (assign a real LLM client — the codebase's actual
dispatcher is the sync `chat_api_call()` in `Chat/Chat_Functions.py:789`, so the existing
`asyncio.to_thread` hop at `:194` is the right shape — and give the event a real poster),
or retire the module the way TASK-16196 retired the legacy Study handlers. Do not leave a
third state where the code looks maintained but cannot execute.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 An explicit wire-or-retire decision is recorded (owner call if product-facing)
- [x] #2 If wired: `BatchAnalysisStartEvent` has a real production poster, `app.llm_api_client` (or a replacement seam) is genuinely assigned, and a test proves a batch analysis reaches a real (mockable) LLM dispatch instead of the placeholder (N/A — the retire branch was taken; see notes)
- [x] #3 If retired: the module, its app.py dispatch branch, and its tests are removed with the same per-symbol reachability evidence TASK-16196 used
- [x] #4 No silent placeholder fallback remains presented as an "LLM analysis" either way
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify reachability at this branch's HEAD (`e112798f1`) and against `origin/dev`
   (`391bc061e`): grep for any constructor of `BatchAnalysisStartEvent`, any external
   importer of `multi_item_review_events`, any assignment of `app.llm_api_client`, and
   any dynamic/reflective importer over `Event_Handlers`.
2. Check the Media hub's multi-select/review affordances at HEAD: trace the
   `media-nav-multi-item-review` button (`Widgets/Media/media_navigation_panel.py`)
   through `MediaWindow_v2.activate_media_type` / `_perform_search`, and establish
   whether any live surface can still reach (or was designed to reach) the handler.
3. Decision: retire (per the evidence — see notes). Follow the TASK-16196 playbook:
   per-symbol reachability table; delete
   `tldw_chatbook/Event_Handlers/multi_item_review_events.py` and
   `Tests/Event_Handlers/test_multi_item_review_events.py`; remove the
   `BatchAnalysisStartEvent` dispatch branch from `app.py`'s
   `on_collections_tag_message`; hand-remove the module's owner row from
   `Docs/security/production-diagnostic-inventory.json` and decrement the summary
   counts by exactly its contribution (16196 precedent — do NOT regenerate the whole
   inventory, which pulls in unrelated pre-existing drift).
4. Sweep for dangling refs (docs, configs, tests, inventory files) and record what is
   deliberately left (historical plan docs; the retired-`MediaScreen` chrome owned by
   task-2851's retirement).
5. Baseline then re-run: `Tests/Event_Handlers/`, the diagnostic-inventory
   architecture suite, `Tests/UI/test_screen_navigation.py`, and a full
   `pytest --collect-only Tests/` (zero errors, before/after counts compared).
6. ruff check/format on touched files; hand-edit ACs/notes/status; commit locally.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Decision: RETIRE.** Deleted `Event_Handlers/multi_item_review_events.py`, its test
file, and its `app.py` dispatch branch, per the TASK-16196 playbook. Not product-facing
in effect: the path has been unreachable end-to-end since TASK-1010 (PR #1019) deleted
its only poster, and even the nav chrome that once led toward it lives inside the
MediaScreen surface that task-2851 already made route-unreachable — retiring it removes
no behavior any user could reach.

**Reachability verdict (verified at branch base `e112798f1`, drift-checked against
`origin/dev` `391bc061e`): the path was triple-dead.**

1. `BatchAnalysisStartEvent` has zero constructors repo-wide — the only way into the
   module's entry handler cannot fire.
2. The Media hub's "Multi-Item Review" affordance never posts it: the
   `media-nav-multi-item-review` button (`Widgets/Media/media_navigation_panel.py:154`)
   posts `MediaTypeSelectedEvent("multi-item-review")` →
   `MediaWindow_v2.activate_media_type` → `_perform_search`, which special-cases the
   slug and returns without doing anything (`MediaWindow_v2.py:2391`, "Skipping browse
   query for special Media view"). And that whole surface (`MediaWindow_v2`, embedded
   only by `MediaScreen`) is itself unreachable from every route and alias — task-2851
   retired it, pinned by
   `Tests/UI/test_screen_navigation.py::test_no_route_reaches_the_retired_media_screen`.
   The live media surface (Library's `LIBRARY_ROW_BROWSE_MEDIA` canvas) has no
   batch-analysis affordance.
3. Even if reached, the "LLM analysis" was unreachable within the handler:
   `app.llm_api_client` is assigned nowhere (the `hasattr` guard at former `:85` was
   always False → the entry handler notified "LLM service not available" and returned;
   the `generate_single_analysis` guard at former `:174` always fell through to
   `generate_placeholder_analysis`). New finding beyond the filing: `app.llm_model_var`,
   `app.llm_temperature_var`, and `app.llm_context_size_var` (former `:186-188`) are
   ALSO phantoms — zero references outside this module — so even a wired client would
   have died in an `AttributeError` before the LLM call.

Consequence acknowledged: TASK-16194's thread repairs (PR #1671) were correct fixes to
a module nothing could execute — sunk groundwork on a corpse; its pinning tests are
deleted with the module.

**Per-symbol reachability table** (grep sweep over `tldw_chatbook/` + `Tests/` for
every reference outside the module itself):

| Symbol | Verdict | Evidence |
|---|---|---|
| `BatchAnalysisStartEvent` | dead | `grep -rn "BatchAnalysisStartEvent("` → only the class definition; historical poster `MultiItemReviewWindow` deleted by TASK-1010 (PR #1019), zero hits for it at HEAD |
| `handle_batch_analysis_start` | dead | sole caller was `app.py:11730`, inside a branch guarded by `event.__class__.__name__ == "BatchAnalysisStartEvent"` — an event with no constructor never arrives |
| `BatchAnalysisProgressEvent`, `BatchAnalysisCompleteEvent` | dead (doubly) | constructed only inside the unreachable `handle_batch_analysis_start`, AND no handler anywhere consumes them (no `@on`, no name-compare dispatch — `app.py`'s dispatcher only checked the Start event) |
| `generate_single_analysis`, `save_analysis_to_db`, `load_existing_analyses`, `generate_placeholder_analysis`, `_media_db_off_loop` | dead | referenced only within the module + its own test file (`Tests/Event_Handlers/test_multi_item_review_events.py`) |
| `app.llm_api_client` | phantom | only refs repo-wide were this module's two guards + one call; never assigned (filing verified `git log --all -S "llm_api_client"` shows it never existed in `app.py`) |
| `app.llm_model_var` / `llm_temperature_var` / `llm_context_size_var` | phantom | zero references anywhere outside the module |

No dynamic/reflective importers can defeat the static sweep: no
`pkgutil`/`iter_modules`/`walk_packages` use touches `Event_Handlers`, and every
`import_module` site (`screen_registry`, `UI/Screens/__init__`, `UI/Workbench`,
`UX_Interop`, `Tools/__init__`) resolves from its own fixed table, none referencing
`Event_Handlers`. Drift check `git diff e112798f1..origin/dev` over
`Event_Handlers/`, `app.py`, `Tests/Event_Handlers/`: 5 unrelated `app.py` lines — no
new poster or `llm_api_client` assignment appeared on dev.

**Dangling refs handled / deliberately left:**
- `Docs/security/production-diagnostic-inventory.json`: hand-removed the module's
  owner row (9 calls, TASK-494, digest `a28e2522873c4b164b74`) and decremented
  `summary.owner_files` 499→498 and `summary.task_494_calls` 6952→6943 by exactly its
  contribution — same approach as 16196; did NOT regenerate the file (regeneration
  pulls in the known, separately-tracked pre-existing drift).
- Left: the `media-nav-multi-item-review` button and `MediaWindow_v2`'s
  `["collections-tags", "multi-item-review"]` skip-list — chrome inside the
  task-2851-retired `MediaScreen` surface (kept alive only for its save/restore unit
  tests); it never referenced this module. Historical plan/spec docs under
  `Docs/superpowers/` mentioning "multi-item review" are records of merged PRs, left
  per 16196 precedent.
- Follow-up candidate flagged for the controller (not fixed here, no ID minted): the
  sibling `collections-tags` path looks equally posterless — `CollectionsTagWindow` is
  instantiated only in tests, never mounted in production, so the surviving
  `KeywordRename/Merge/Delete` branches of `app.py`'s `on_collections_tag_message`
  appear to have no live poster either.

**Tests** (venv python, PYTHONPATH pinned; outputs to scratchpad files):
- Baseline (pre-change): `Tests/Event_Handlers/` + `Tests/Architecture/
  test_persistent_diagnostic_inventory.py` + `Tests/UI/test_screen_navigation.py` →
  250 passed, 1 skipped, 2 failed — both failures are the known pre-existing
  inventory-drift reds (`test_production_diagnostic_inventory_and_sink_topology_are_
  unchanged`, `test_task_15743_final_rebase_diagnostics_are_metadata_only`; same
  recurring drift family 16196 documented).
- After: 246 passed (250 minus the 4 deleted tests), 1 skipped, same 2 reds for the
  same reasons — the failure output contains zero mentions of `multi_item_review`
  (the deleted module is not among the drift causes; row removal and module deletion
  are consistent, so the checker's regenerated view agrees).
- Collection: `pytest --collect-only -q Tests/` → 49510 collected/0 errors before,
  49506/0 after (exactly the 4 deleted tests).
- `ruff check` + `ruff format --check` clean on `tldw_chatbook/app.py`.

**Files changed:**
- Deleted `tldw_chatbook/Event_Handlers/multi_item_review_events.py`
- Deleted `Tests/Event_Handlers/test_multi_item_review_events.py`
- `tldw_chatbook/app.py` — removed the `BatchAnalysisStartEvent` dispatch branch from
  `on_collections_tag_message`
- `Docs/security/production-diagnostic-inventory.json` — removed the module's owner
  row + summary decrements
<!-- SECTION:NOTES:END -->
