---
id: TASK-38702
title: Re-green the Console fork-transition and semantic-mutation censuses
status: Done
created_date: '2026-11-03 12:55'
assignee:
- '@claude'
labels:
- ci
- tests
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two AST censuses guard the Console store's invariants: `Tests/Chat/test_console_fork_transition_census.py` (every public fork-field mutation is fenced by a fork transition, or classified as safe) and `Tests/Chat/test_console_semantic_mutation_inventory.py` (every live SQL sink and boundary call that writes messages or their sidecars is classified by what the model can see). Both are red on dev. The fork census fails on `publish_prompt_mixin_attempt`, added by the prompt mix-ins PR (#3346) without a classification. The semantic inventory fails on routes that are stale or have no classification. While they are red, neither census can catch a regression in the invariants it protects, including the generation-owner-scope lock order that PR #3352 just added to `console_chat_store.py`. No PR-gating CI lane runs them, which is why PRs merged green while they were red.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both census files pass on dev.
- [x] #2 Every newly classified route was reviewed on its merits and has a stated reason. A route that genuinely violates its invariant is fixed in production code, not allowlisted.
- [x] #3 A regression in either census fails a PR-gating CI lane.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run both files on the branch base and probe each failing scan directly, so every red test gets its exact unclassified/stale/unreviewed routes and the dev commit or PR that introduced each (git log -S).
2. Fork census: judge `publish_prompt_mixin_attempt` against the transition contract (owner of its fenced scope versus owner of each fork-field write); classify it as a direct transition owner or fix the store.
3. Semantic inventory: re-key the routes that only moved; classify each new route by what it can change in the provider-visible envelope; give every new dynamic SQL site a review whose targets are derived mechanically, adding a narrow evidence form plus regressions wherever no existing form fits; fix fail-closed scanner gaps that block an unrelated audit; keep the inventory document in sync.
4. Make both files cheap enough for a PR lane (measure; fix test-side costs such as per-test gc over cached ASTs) and add them to the PR Fast Lane with its pinned target list.
5. Verify: both files, Tests/CI lane contracts, preflight, ruff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Both censuses pass, and the PR Fast Lane now runs them. The fixes are classifications and two new evidence forms for the census test code. No production code changed: no route violated its invariant.

**Why each test was red, and which change made it red**
- Fork census, `test_public_fork_transition_inventory_is_bidirectional` and `test_every_public_direct_fork_field_assignment_is_fenced_or_classified`: #3346 (7970fbcada, merged 18482c1cf7 on 2026-11-02) added `publish_prompt_mixin_attempt` and did not list it. Both tests pass at 18482c1cf7^1 and fail at 18482c1cf7.
- Inventory, `test_live_sql_mutation_sites_are_classified`: `_direct_sql_routes()` raised on 19 unreviewed dynamic SQL sites:
  - `credentials._rewrite_database` (#2642);
  - project index (#3204);
  - goal ledger (#2563);
  - plugin coordinator and registry (#2946);
  - `keyword_store` (#3120).

  Behind those sites, two SQL routes had moved (#2432 importer, #2918 legacy trace batch) and two were new (#3346).
- The inventory first went red at #2432 (merged f32a16839d on 2026-09-05). Both route tests pass at f32a16839d^1 and fail at f32a16839d, because the importer's writes moved into `_import_conversations_with_database`. The later drifts accumulated while it was already red.
- `test_repository_scan_results_are_cached_and_immutable` has no defect of its own. It calls the same scan, and `functools.cache` does not store an exception, so every caller re-ran the package parse and raised.
- `test_public_mutation_boundary_calls_are_classified`:
  - Four routes had moved: #2432 importer ×2, #2642 persona delete, and #3023 native settle.
  - Three were new: the voice pair (#2504), the agent fork copy (#2643), and the mix-in attempt (#3346).
- `test_all_dynamic_sql_reviews_match_derived_production_targets`: #2862 (fd8e1ec4b9) added two getattr calls whose name is not a literal (a literal-dict subscript and a conditional). They made the `_commit_local_body` evidence fail closed. Behind that failure, the evidence fingerprint had also moved, because the same commit added `consumer_id` to both callers.

**Classifications (each with its reason in the census source)**
- `publish_prompt_mixin_attempt`: a direct transition owner. Its persisted and parent id writes run inside the assistant message's `_generation_owner_scope`, which publishes that session's fork marker. Two negative controls fail as they should: hoisting the write above the scope, and removing the scope.
- `_mark_assistant` and `write_fork_state`: presentation-only. They write only the `metadata_json` mix-in marker, the same column `update_message_metadata_local` writes. A retry reads the marker to choose the mix-in snapshot. That is request preparation, not a message revision.
- `commit_completed_voice_pair`, `copy_conversation_active_path` and `save_conversation_mixin_attempt`: model-visible, because each creates canonical message rows.
- The moved routes keep their old classes.
- The 17 project-index, goal, plugin and keyword sites are reviewed with targets derived from the code. Two new narrow evidence forms make this possible:
  - `_LoopTargetEvidence`: the interpolated target is the variable of an enclosing loop, over an inline literal tuple or over the `items()` of a local dict with literal keys. A site written through a local SQL name fails.
  - `_SpecFieldTargetEvidence`: the target is `self.spec.<field>`, and every direct `KeywordStore` construction passes a keyword-literal `KeywordStoreSpec`. Aliased imports, forwarding and literal `getattr` fail.

  Each form has regressions.
- `credentials._rewrite_database` is excluded like `_migrate_from_*`, as a backup archive staging rewrite into a fresh uuid-named file. A review was not an option: its targets include `messages`, which no review may exempt. Its two sites are pinned exactly, and each must execute on the connection opened on that fresh destination.
- A `getattr` name now resolves to the finite set of literals it can take, and each one is checked against the reviewed helpers. The `_commit_local_body` fingerprint is re-recorded; its literal tables are unchanged.

**Speed and the CI lane**
- The inventory file took 14 min serially. Tests/Chat's per-test `gc.collect()` walked the cached whole-package ASTs (~8.5M nodes, ~2.5 GB). The module now overrides that fixture with a no-op and frees its caches at module end.
- The fork census computes its mutating-route set once instead of once per route (one test: 50 s → under 1 s).
- Together: 159 passed in 114 s serially, on a machine at load average ~37. The slowest test took 85 s against the lane's 180 s per-test timeout.
- Both files are whole-file riders at the end of the PR Fast Lane's sandboxed step. They are in `.github/workflows/derived-artifacts.yml` and in `FAST_LANE_TARGETS` in `Tests/CI/test_ci_queue_pressure_contract.py`.

**Verification (branch claude/bd-38702 on dev 335f984bd2, 2026-11-03)**
- Base: 6 failed, 129 passed.
- After:
  - both files: 159 passed;
  - Tests/CI: 1177 passed, 1 skipped;
  - `./scripts/preflight.sh`: OK;
  - ruff: no new findings in the touched files (HEAD's existing drift left alone).

**Files**
- `Tests/Chat/test_console_fork_transition_census.py`
- `Tests/Chat/test_console_semantic_mutation_inventory.py`
- `Docs/Development/console-semantic-mutation-inventory.md`
- `.github/workflows/derived-artifacts.yml`
- `Tests/CI/test_ci_queue_pressure_contract.py`
- `backlog/docs/lessons-testing-evidence.md`
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
