---
id: TASK-17600
title: result_reranking middleware ships enabled=true and is handled by a bare pass
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-17'
updated_date: '2026-08-18 00:13'
labels:
  - rag
  - config
dependencies: []
priority: medium
---

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: Tests/RAG_Search/test_pipeline_middleware_contract.py -- parse the shipped rag_pipelines.toml's pipeline before/after lists with tomllib and pipeline_loader.py's 'middleware_id ==' branches with ast, per phase, and fail on (a) declared-but-unimplemented, (b) a branch whose body is a bare pass, (c) implemented-but-undeclared, (d) a declared name with no [middleware.*] block. Must be RED now on the seven.
2. DELETE the promises (do NOT wire): result_reranking + the seven unimplemented names from the pipelines that declare them, and every [middleware.*] block left with no implementation and no consumer. Wiring result_reranking would switch reranking ON for every high_accuracy user and TASK-16965 measured it net-harmful; that reason goes in the diff. An emptied middleware list loses its key rather than shipping [].
3. F3: reranking_strategy reads nothing -- delete it from config.py's and rag_v2_example.toml's commented examples and correct RAG-DESIGN.md:2371 to name the mechanism that actually works (a saved/cloned profile's reranking_config.strategy, which round-trips through ProfileConfig.to_dict/from_dict).
4. GREEN; batteries counts READ; ruff; gate verbatim with the vacuity caveat.
<!-- SECTION:PLAN:END -->

## Description (the why)

Found during TASK-16965's user-visible sweep for `cross_encoder` surfaces.
`Config_Files/rag_pipelines.toml` declares a `result_reranking` middleware,
listed by the `high_accuracy` pipeline with **`enabled = true`** — and
`_apply_after_middleware` handles that name with a bare `pass`. It is a
shipped, switched-on stage that does nothing.

This is the same species this programme has repeatedly fixed (TASK-16174's
inert parent-inclusion knobs: shipped, user-switchable, read by nothing).
It is worse in one respect — the knobs were merely dead, whereas this one
appears in a pipeline a user selects *because* it promises higher accuracy.

TASK-16965 documented the fact where it was found but did not own the fix;
its own subject (the `cross_encoder` strategy) is a different mechanism —
that arc measured reranking as net-harmful on this corpus, which is
context a fix here should carry.

## Acceptance Criteria (the what)

- [ ] A decision is implemented, either arm acceptable: `result_reranking`
      is WIRED to the real reranking path, or it is REMOVED from
      `rag_pipelines.toml` and from any pipeline that lists it
- [ ] No middleware name remains that is declared `enabled = true` while its
      handler is a no-op — the sweep covers every name
      `_apply_after_middleware` (and its before-equivalent) accepts
- [ ] If wired: the TASK-16965 measurement is cited where a user chooses it,
      so a stage measured net-harmful on the eval corpus is not silently
      presented as an accuracy improvement
- [ ] A test fails if a declared-enabled middleware name has no
      implementation, so this class of gap cannot recur silently
- [ ] The same sweep covers `reranking_strategy` (TASK-16965 final review
      F3): it has ZERO readers repo-wide, yet RAG-DESIGN.md instructs users
      to select a strategy with it — a config key that reads nothing is the
      same species as a middleware that runs nothing
