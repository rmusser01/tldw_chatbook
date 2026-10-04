---
id: TASK-34368
title: Settle failed captured agent calls before admitting model retries
status: Done
created_date: 2026-10-04 18:06
assignee:
- '@codex'
updated_date: 2026-10-04 19:01
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Automatic model retries are treated as tool iterations while failed trace calls remain dispatch_started until assistant completion. This deterministically creates trace_tool_chain_unavailable and misleading recovery after a real provider error.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A captured agent call rejected with 429 or 500 retries successfully with distinct ordered durable call records.
- [x] #2 Retry after a tool response preserves prior results and the exact owner, source revision, policy and chain.
- [x] #3 Unknown, stale, foreign or changed-surface failed calls cannot authorize another captured dispatch.
- [x] #4 Real controller and mounted recovery coverage prove known failures do not become false unknown-delivery state.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce provider failure -> retry with real controller, gateway, store and SQLite. 2. Pin error-settlement ordering and same-chain retry eligibility, including negative controls. 3. Implement immediate trace-owned settlement for failed intermediate calls and exact unchanged-surface retry admission under existing TOOL_LOOP ownership. 4. Run targeted trace, recovery and agent regressions, then live DeepSeek UAT. ADR required: yes, amendment. ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md. Reason: clarify failed intermediate-call settlement and retry ownership without weakening unresolved-call gates or adding schema.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
No-response ERROR handoffs previously waited for final assistant persistence, while the bridge classified the automatic retry as TOOL_LOOP. The first call therefore remained DISPATCH_STARTED and rejected the retry with trace_tool_chain_unavailable. Seal failed attempts before retry admission; keep failed writes in owned settlement custody through cancellation. Normalize absent-response ERROR canonical IDs to preserve pending fingerprints and prevent linking later successful answers to failed calls. Admit only exact unchanged failed-attempt retries under existing owner/actor/chain/turn/source/policy/latest-call proofs. Compare immediately preceding rendered system content in both wire formats, including an atomic header check before dispatch capability promotion. Existing unknown/open, foreign/stale and changed-surface denials remain intact; no schema changes. Modified gateway, settlement, runtime and service; added real controller, both-wire ownership, fault/cancellation and mounted recovery regressions. Four genuine 429/500 x before/after-tool cases, 26 ownership/system cases, three custody cases and two mounted retry cases pass. Final 12-module targeted run: 216 passed, three baseline-only exclusions. Review's two Important findings were fixed; re-review found none remaining. Live captured DeepSeek conversation and durable links verified. ADR-097 amendment and lessons-testing-evidence incident updated. No full-suite claim.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
