---
id: TASK-32852
title: Migrate deepseek and mistral chat handlers onto the hosted_chat engine
status: To Do
assignee: []
created_date: '2026-09-19 08:24'
labels:
  - core-review
  - review-cascade
dependencies:
  - TASK-19642.10
  - TASK-32851
parent_task_id: TASK-32850
references:
  - qa/cascade-review-2026-09-19/report.md
  - backlog/decisions/062-hosted-chat-completions-provider-boundary.md
  - backlog/decisions/064-deepseek-dual-api-provider-boundary.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Second ADR-062 migration wave, after groq/openrouter (TASK-32851) prove the pattern. `chat_with_deepseek` (`LLM_Calls/LLM_API_Calls.py:3147`, 254 LOC, payload 100% standard OpenAI keys) and `chat_with_mistral` (`:4987`, 239 LOC) are ~70-75% transport boilerplate. Same free fixes: deepseek's DONE-yield-in-`finally` (`:3310-3313`) and mistral's (`:5135-5138`), dropped streamed usage, and wrong metric labels (deepseek logs `mistral_api_*` `:3337-3367`; mistral logs `openrouter_api_*` `:5116-5127`).

Genuinely provider-specific and preserved: mistral's `random_seed` (not `seed`), `safe_prompt`, `has_system_in_input` dedup, `Accept` header, endpoint key `mistralai`, no stop/penalties.

Sequencing with TASK-15677 (DeepSeek dual-API, ADR-064): migrating first shrinks 15677's surface — the Responses wire mode lands as a mode on the migrated profile instead of on a hand-rolled handler. If 15677 starts first, contribute to it rather than racing (see lessons-backlog-hygiene on duplicate implementations). ADR required: no — executes ADR-062's direction; record the 064 interaction in the implementation notes.

Source: cascade review 2026-09-19 — `qa/cascade-review-2026-09-19/report.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `chat_with_deepseek` and `chat_with_mistral` route transport through `hosted_chat.py` as provider profiles; mistral's provider-specific payload behavior is preserved and pinned
- [ ] #2 The DeepSeek dual-API coordination with TASK-15677/ADR-064 is recorded: either this landed first (15677 builds on the profile) or the handoff is written down
- [ ] #3 Streaming Stop closes the transport (no yield-in-`finally`); streamed usage is captured for both
- [ ] #4 Metrics name the right provider; consumers of the old labels identified
- [ ] #5 Provider-neutral contract coverage for both; existing tests pass
<!-- AC:END -->
