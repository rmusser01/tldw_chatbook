---
id: TASK-25901
title: 'Agent loop: classified retry with bounded backoff on model errors'
status: To Do
assignee: []
created_date: '2026-08-31 15:07'
updated_date: '2026-08-31 15:11'
labels:
  - agents
  - reliability
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A single transient provider error currently discards an entire agent run along with every tool result in it. Verified on origin/dev: Agents/agent_runtime.py:1291-1302 turns any call_model exception into a STEP_MODEL_ERROR trace and re-raises, which agent_service catches as RUN_ERROR; a named grep for fallback_model, retry_model, max_retries and backoff across tldw_chatbook/Agents/ returns zero hits. Local-first users run flaky local servers and hit rate limits routinely, so this is the highest damage-per-occurrence gap found in the 2026-08-31 parity pass. Scope is retry only; cross-provider failover is separate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A transient model error (429, 5xx, connection reset, read timeout) is retried inside the loop rather than ending the run
- [ ] #2 Retries are bounded by a configurable attempt count and use jittered backoff; a Retry-After header is honored when present
- [ ] #3 Errors classified as terminal (auth failure, invalid request, content policy) are NOT retried and end the run immediately with their existing copy
- [ ] #4 Every retry emits a trace step so the attempt is visible in the run log rather than silent
- [ ] #5 Total retry time is bounded by the run's remaining wall budget and cannot extend a run past max_wall_seconds
- [ ] #6 Tests cover: transient error recovers, terminal error does not retry, budget exhaustion during retry ends the run honestly
<!-- AC:END -->
