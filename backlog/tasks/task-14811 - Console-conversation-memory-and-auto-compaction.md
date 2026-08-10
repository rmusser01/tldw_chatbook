---
id: TASK-14811
title: Console conversation memory and auto-compaction
status: Done
created_date: 2026-08-10 18:15
priority: high
updated_date: 2026-08-10 22:55
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give users explicit, durable control over how much conversation context the Console sends and whether earlier history is summarized, while keeping model limits, cost-bearing summarization, deterministic safety windowing, and preserved transcripts honest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The current Console conversation exposes an Automatic or custom token budget and a readable effective next-request estimate without confusing it with response max tokens.
- [x] #2 Users can choose Ask, Automatic, or Off compaction for the current conversation; once the conversation is durable, the choice persists across close, resume, and restart.
- [x] #3 Automatic compaction triggers against the projected post-transform request at a configurable high-water mark, reduces context toward a configurable target, and never deletes stored transcript content.
- [x] #4 Global Console Behavior settings own default budget, trigger, target, summary cap, failure policy, and carry-forward method; Internal Prompts remains the single owner of the editable summary prompt.
- [x] #5 Compaction is branch-safe and session-safe under parallel work, preserves whole tool-call and result units, discards stale results, and reports non-compactable overhead.
- [x] #6 Focused unit, persistence, provider-payload, mounted UI, geometry, accessibility, and live Console verification cover default, threshold, failure, model-switch, branch-switch, and narrow-terminal behavior.
- [x] #7 Prepared-request safety windowing prevents known-window or explicitly overridden-window overflow; unknown limits and user-supplied thresholds remain visibly unverified and never claim provider safety.
- [x] #8 Generated memory is local, prefix-valid, branch-safe, reviewable and resettable; a content-free auxiliary ledger accounts for successful, failed, cancelled, and stale summary attempts without storing transcript or summary bodies.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record durable ownership, provider serialization, persistence, safety-windowing, branch validity, auxiliary usage, and prompt boundaries in backlog/decisions/052-console-conversation-memory-and-compaction-policy.md.
2. Maintain the implementation-ready UX and technical design in Docs/superpowers/specs/2026-08-10-console-conversation-memory-compaction-design.md.
3. Maintain the dependency-ordered delivery plan in Docs/superpowers/plans/2026-08-10-console-conversation-memory-compaction-implementation.md.
4. Deliver through single-PR slices: TASK-14811.1 policy and persistence; TASK-14811.2 exact prepared requests and safety accounting; TASK-14811.2.1 bounded compaction, valid memory, and auxiliary usage; TASK-14811.3 current-conversation Console UX; TASK-14811.4 canonical Settings UX; TASK-14811.5 hardening and live verification.
5. Self-review every artifact for ambiguous ownership, provider wire-shape drift, unsafe fallbacks, iterative-memory loss, inaccessible UI, stale asynchronous results, missing billed-attempt accounting, and unverifiable acceptance criteria before implementation begins.

ADR required: yes
ADR path: backlog/decisions/052-console-conversation-memory-and-compaction-policy.md
Reason: The feature introduces durable per-conversation policy and summary provenance, a cost-bearing model-call service boundary, provider-specific prepared-request serialization, cross-module ownership between Console and Settings, and a long-lived context-injection contract.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Delivered the ADR-052 conversation-memory architecture through six completed slices: durable per-conversation policy and schema-v33 persistence; exact provider-prepared request accounting and safety windowing; bounded branch-safe compaction with local prefix-valid memory and a content-free auxiliary ledger; current-conversation Console controls; canonical Settings defaults, model-capacity repair, and Internal Prompts routing; and race/privacy/accounting hardening with isolated real-provider verification. The Console now distinguishes conversation budget from response tokens, supports Ask/Automatic/Off plus reset/review/manual compaction, preserves transcripts and whole tool units, fails visibly when limits are unknown or mandatory overhead is non-compactable, and discards stale asynchronous summaries across branch/model/policy/session changes. Final evidence: all child tasks are Done; the final relevant matrix passed 319 tests across policy, persistence, provider preparation, lifecycle/races, privacy logging, mounted UI, and narrow geometry; targeted Ruff and py_compile passed; whole-diff whitespace checking passed; isolated OpenAI gpt-4o verification exercised every policy/failure case, stored usage/pricing provenance without a transcript summary row, left the real config unchanged, and removed its scratch profile. ADR: backlog/decisions/052-console-conversation-memory-and-compaction-policy.md. Design and delivery plan are in Docs/superpowers/specs and Docs/superpowers/plans. The live-verification credential-probe/isolation incident is documented in backlog/docs/lessons-testing-evidence.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Console conversations now have explicit durable context budgets and Ask/Automatic/Off compaction, exact request accounting, branch-safe local memory, canonical Console/Settings controls, content-free auxiliary usage provenance, and verified failure/race/privacy behavior.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
