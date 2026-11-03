---
id: TASK-38702
title: Re-green the Console fork-transition and semantic-mutation censuses
status: In Progress
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
- [ ] #1 Both census files pass on dev.
- [ ] #2 Every newly classified route was reviewed on its merits and has a stated reason. A route that genuinely violates its invariant is fixed in production code, not allowlisted.
- [ ] #3 A regression in either census fails a PR-gating CI lane.
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

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
