---
id: TASK-34418
title: Provider HTTP session reuse and payload deepcopy removal ADR-214
status: Done
created_date: 2026-10-07 02:41
dependencies:
- TASK-34669
updated_date: 2026-10-07 07:38
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 3 / F8b: full message-history payload is deepcopied per POST attempt and every provider call opens a fresh requests Session paying TLS handshake per turn
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ADR-214 written before code,Per-thread session registry with same-key reuse,No cross-thread session sharing,Payload passed by reference with mutation probe test,TLS handshake count evidence recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 6 (T6)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Per-thread provider HTTP session registry (LLM_Calls/provider_sessions.py, threading.local, key provider:base_url, close_all_for_current_thread teardown) replacing fresh requests.Session-per-call at the hosted/qwencloud/summarization/LLM_API_Calls sites; ADR-222 documents lifecycle, thread-safety, and the OpenAI recovery-mode exception (recovery_review.openai_post owns+closes its session and sets trust_env=False — left unswapped by design, future non-owning seam noted). Payload deepcopy dropped from the hosted POST path (mutation-probe + golden request capture pin requests' non-mutation). Evidence: local HTTP server connection counter — 3 back-to-back calls 3 -> 1 connections (10 calls still 1); Tests/LLM_Calls 2025 passed with failure list identical to baseline; new test_provider_session_reuse.py 14/14. Live provider smoke honestly skipped (no keys configured). Full report: .superpowers/sdd/task-6-report.md
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
