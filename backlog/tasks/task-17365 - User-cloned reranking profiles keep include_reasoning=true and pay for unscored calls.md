---
id: task-17365
title: >-
  User-cloned reranking profiles keep include_reasoning=true and pay for
  unscored calls
status: To Do
assignee: []
created_date: '2026-08-17'
labels: [rag, settings, config]
dependencies: []
priority: medium
---

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

- [ ] A decision is recorded, either arm acceptable: user profiles carrying
      `include_reasoning = true` alongside a small `max_tokens` are migrated
      on load, OR `max_tokens` is made strategy/reasoning-aware so the
      combination cannot truncate (the review's alternative: >= 400 tokens
      when reasoning is on)
- [ ] Whichever arm ships, a saved profile with
      `include_reasoning = true, max_tokens = 100` no longer produces a
      billed-but-unscored call
- [ ] A test drives the chosen path with a faked provider seam (no live
      calls) and asserts the reranked rows come back scored
- [ ] If migration is the arm: it is idempotent, and a profile the user
      deliberately set to a large `max_tokens` with reasoning on is left
      alone
