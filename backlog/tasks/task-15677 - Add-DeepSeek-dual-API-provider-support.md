---
id: TASK-15677
title: Add DeepSeek dual-API provider support
status: To Do
assignee: []
created_date: '2026-08-12 20:46'
labels: []
dependencies:
  - TASK-15675
  - TASK-15676
references:
  - Docs/superpowers/specs/2026-08-12-deepseek-dual-api-provider-design.md
  - >-
    Docs/superpowers/plans/2026-08-12-deepseek-dual-api-provider-implementation.md
  - backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md
  - backlog/decisions/064-deepseek-dual-api-provider-boundary.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Treat DeepSeek Chat Completions and the new Responses API as two strict wire modes of the existing `deepseek` provider while preserving default behavior, native Chatbook tools, durable conversation resume, and ordinary provider UX.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 DeepSeek remains one stable provider with explicit `api_mode`, defaults to `chat_completions`, and offers `responses` without changing unrelated providers.
- [ ] #2 Fresh configuration uses `deepseek-v4-flash` with `deepseek-v4-pro` available; explicit historical model selections remain user-owned.
- [ ] #3 Both modes share one frozen provider/model/base/key/retry resolution and exact provider-specific request allowlists; unsupported or malformed inputs fail before network I/O.
- [ ] #4 Existing Chatbook function tools work through the real Console runtime in both modes; DeepSeek web search, custom `apply_patch`, and other provider-hosted tools remain excluded.
- [ ] #5 Thinking defaults to provider behavior and Settings accepts only provider default, `low`, `high`, or `max`; unsupported sampler/tool-choice combinations and compatibility aliases without distinct behavior are omitted or rejected exactly as documented.
- [ ] #6 DeepSeek tool-associated reasoning and tool history use TASK-15675 checkpoints and replay on later same-provider turns while their owning visible turns remain in context.
- [ ] #7 Responses input translation, call/output adjacency, semantic SSE events, sequence/terminal rules, usage, errors, cancellation, and exactly-once closure are strict and bounded.
- [ ] #8 Chat Completions uses the hosted Chat wire boundary from TASK-15676 and preserves complete assistant `reasoning_content`/tool-call history required by DeepSeek.
- [ ] #9 Settings, readiness, dispatcher, Console pinning, model discovery, docs, and usage budgeting treat DeepSeek like the other first-class providers.
- [ ] #10 Unit, hostile parser, loopback HTTP, crash/restore, sync/export, joined native-tool, mutation, and doubly-gated isolated live tests pass without paid calls in default runs.
<!-- AC:END -->
