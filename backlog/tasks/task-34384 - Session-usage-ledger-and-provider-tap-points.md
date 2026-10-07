---
id: TASK-34384
title: Session usage ledger and provider tap points
status: To Do
assignee: []
created_date: '2026-10-07 03:26'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Foundation for issue #365: a process-wide SessionUsageLedger (Chat/session_usage.py) recording exact provider usage (estimate fallback) exactly once per response, tapped at the parse boundaries in LLM_API_Calls (9 non-streaming sites, Anthropic SSE accumulator, OpenAI Responses completed_usage, chat-completions usage-chunk guard), the Console gateway, agent_service, Library RAG answers, and realtime. Existing histograms stay byte-identical. Spec: Docs/superpowers/specs/2026-09-22-quit-time-session-summary-design.md. Plan: Docs/superpowers/plans/2026-09-22-quit-time-session-summary.md (Tasks 1-3).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 SessionUsageLedger exists with unit tests (exact accumulation / estimate fallback / malformed payloads never raise / thread safety)
- [ ] #2 Nine non-streaming provider sites record exact usage (estimates when absent); existing usage histograms stay byte-identical
- [ ] #3 Streaming taps (Anthropic SSE accumulator; OpenAI Responses completed_usage; OpenAI chat-completions usage-chunk guard) record exactly once per response
- [ ] #4 Console gateway / agent_service / Library RAG / realtime parse boundaries record; from_provider_payload audit classifies boundary vs downstream re-read
- [ ] #5 Existing usage and streaming tests unchanged and passing
<!-- AC:END -->
