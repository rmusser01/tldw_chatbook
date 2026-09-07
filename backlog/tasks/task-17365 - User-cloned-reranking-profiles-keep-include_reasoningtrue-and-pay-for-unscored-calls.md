---
id: TASK-17365
title: >-
  User-cloned reranking profiles keep include_reasoning=true and pay for
  unscored calls
status: Done
assignee:
  - '@claude'
created_date: '2026-08-17'
updated_date: '2026-08-18 00:13'
labels:
  - rag
  - settings
  - config
dependencies: []
priority: medium
---

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: Tests/RAG_Search/test_reranker_token_floor.py -- the effective max_tokens at the ONE consumption site (_call_llm_impl's chat_api_call kwarg) is >= the floor when include_reasoning is on; a deliberate 4000 is UNTOUCHED (floor, not assignment); no reasoning => 100 stays 100; plus an end-to-end truncation test (fake seam truncates to the budget) proving rows come back scored.
2. Implement a REASONING_TOKEN_FLOOR (400) applied at reranker.py's single max_tokens site. NO migration: a migration mutates a user's saved profile behind their back and must guess whether a large max_tokens was deliberate; a floor cannot guess wrong.
3. GREEN; Tests/RAG_Search/ counts READ; ruff.
<!-- SECTION:PLAN:END -->

## Description (the why)

TASK-17065 (AC#11) set `include_reasoning = False` on the two shipped
read-only profiles that carried it (`high_accuracy`, `research_papers`),
because with `max_tokens = 100` the free-form reasoning text leaves ~60
tokens for the JSON payload: truncation causes a parse failure, and the
call is **billed but unscored** (for listwise, the whole rerank fails).
Built-in profiles never persist to disk, so every install picks the fix up.

**A profile a user CLONED from either of those before the fix keeps
`include_reasoning = true` in their own saved profile**, and no migration
was written. Until TASK-17065, reranking called zero providers, so this
cost nothing; it now spends on every search for those users — the exact
failure mode AC#11 exists to prevent, just on the copies rather than the
originals.

## Acceptance Criteria (the what)

- [x] A decision is recorded, either arm acceptable: user profiles carrying
      `include_reasoning = true` alongside a small `max_tokens` are migrated
      on load, OR `max_tokens` is made strategy/reasoning-aware so the
      combination cannot truncate (the review's alternative: >= 400 tokens
      when reasoning is on)
- [x] Whichever arm ships, a saved profile with
      `include_reasoning = true, max_tokens = 100` no longer produces a
      billed-but-unscored call
- [x] A test drives the chosen path with a faked provider seam (no live
      calls) and asserts the reranked rows come back scored
- [x] If migration is the arm: it is idempotent, and a profile the user
      deliberately set to a large `max_tokens` with reasoning on is left
      alone

## Implementation Notes

**A floor, not a migration.** A migration would mutate a user's saved profile
behind their back and would have to guess whether a large `max_tokens` was
deliberate; a floor cannot guess wrong. When `include_reasoning` is on, the
effective budget is raised at the single consumption site to a value that
fits reasoning plus the JSON — a deliberate 4000 stays 4000, and a config
without reasoning is untouched, both pinned.

This reaches exactly the profiles TASK-16965's AC#11 could not: built-ins
never persist, so a profile a user CLONED from `high_accuracy` or
`research_papers` kept `include_reasoning = true` and would truncate its JSON
into a billed-but-unscored call.
