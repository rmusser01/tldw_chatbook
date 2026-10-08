---
id: TASK-34384
title: Session usage ledger and provider tap points
status: Done
assignee: []
created_date: '2026-10-07 03:26'
updated_date: '2026-10-07 04:03'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Foundation for issue #365: a process-wide SessionUsageLedger (Chat/session_usage.py) recording exact provider usage (estimate fallback) exactly once per response, tapped at the parse boundaries in LLM_API_Calls (9 non-streaming sites, Anthropic SSE accumulator, OpenAI Responses completed_usage, chat-completions usage-chunk guard), the Console gateway, agent_service, Library RAG answers, and realtime. Existing histograms stay byte-identical. Spec: Docs/superpowers/specs/2026-09-22-quit-time-session-summary-design.md. Plan: Docs/superpowers/plans/2026-09-22-quit-time-session-summary.md (Tasks 1-3).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SessionUsageLedger exists with unit tests (exact accumulation / estimate fallback / malformed payloads never raise / thread safety)
- [x] #2 Nine non-streaming provider sites record exact usage (estimates when absent); existing usage histograms stay byte-identical
- [x] #3 Streaming taps (Anthropic SSE accumulator; OpenAI Responses completed_usage; OpenAI chat-completions usage-chunk guard) record exactly once per response
- [x] #4 Console gateway / agent_service / Library RAG / realtime parse boundaries record; from_provider_payload audit classifies boundary vs downstream re-read
- [x] #5 Existing usage and streaming tests unchanged and passing
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-09-22-quit-time-session-summary.md Tasks 1-3 (ledger, non-streaming taps, streaming+service taps) in .worktrees/quit-summary-365 on dev; follow the plan's Dev-Tip Anchor Addendum for current line anchors
<!-- SECTION:PLAN:END -->

## Implementation Notes

Executed via Docs/superpowers/plans/2026-09-22-quit-time-session-summary.md Tasks 1-3 (worktree .worktrees/quit-summary-365, branch dev).

- Approach: process-wide thread-safe `SessionUsageLedger` (Chat/session_usage.py) recording exactly once per provider response at parse boundaries. Exact via ProviderUsage normalization; char-based estimate fallback for usage-less non-streaming responses.
- Tap points: 7 non-streaming sites in LLM_API_Calls (openai/anthropic/cohere/google/huggingface + moonshot/zai wrappers); streaming via Anthropic SSE accumulator seam (pre-yield, GeneratorExit-safe), OpenAI Responses completed_usage, OpenAI chat-completions substring-guard (include_usage already sent), google last-chunk-usage after normal loop end; 4 extracted LLM_Calls modules via their single-funnel `_log_usage_metrics`; `LegacyLineStream` exhaustion covers those modules' streaming; gateway-NATIVE httpx sites only (relay lines never re-record — double-count guard, enforced by review + tests); Library RAG; Console realtime reply usage (transcription duration is not tokens — excluded).
- Deviations from plan: provider set consolidated on dev (addendum in the plan); the review fix round removed an agent_service tap (fleets route through already-tapped chat_api_call) and relocated the gateway tap to native sites; final review caught the gateway taps passing full bodies where a bare usage dict is required — fixed with extraction + contract test.
- Tests: ledger unit suite + tap suite with exactly-once invariant (`calls == 1`), mocked-HTTP idiom, `bootstrap_profile` admission marker. Histograms byte-identical (regression-verified).
- Files: Chat/session_usage.py; LLM_Calls/LLM_API_Calls.py; LLM_Calls/{groq,deepseek,mistral,openrouter,legacy_line_stream}.py; Chat/console_provider_gateway.py; Library/library_rag_answer_service.py; UI/Console_Modules/realtime.py; Tests/Chat/test_session_usage{,_taps}.py. Commits 85a2cd4b41, cf9bfd29d3, c3ac82f593, efaa162469, 4f4cb17e32.
- ADR: none required (additive, ephemeral process-lifetime data; reasoning in the spec's ADR Check).
