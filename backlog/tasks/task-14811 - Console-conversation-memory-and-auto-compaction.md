---
id: TASK-14811
title: Console conversation memory and auto-compaction
status: Done
assignee: []
created_date: '2026-08-10 18:15'
updated_date: '2026-08-11 01:16'
labels: []
dependencies: []
priority: high
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

<!-- SECTION:NOTES:BEGIN -->
Delivered ADR-052 through six completed slices: schema-v33 per-conversation policy and memory persistence; exact provider request accounting and safety windowing; bounded branch-safe compaction with a content-free auxiliary ledger; current-conversation Console controls; canonical Settings defaults and model-capacity repair; and race, privacy, accounting, and live-provider hardening. PR #1478 review follow-up wrapped repository reads in transactions, bounded pagination, completed API documentation, added content-free degraded-mode diagnostics, and retained the ADR-defined unknown-capacity contract with regression coverage. After rebasing onto dev 8d764c03b, CI exposed a PR-owned concurrency-test sequencing defect: the test proved the second client-creation thread was blocked but did not release the first before waiting for completion. The test now signals release explicitly; its red state was reproduced before the fix, then the isolated test and all 143 gateway tests passed. Post-rebase focused evidence totals 268 passing tests: 68 context-policy and lifecycle, 143 gateway, 23 Console and Settings UI/config, and 34 schema migration tests. Targeted Ruff and py_compile, diff checks, and isolated OpenAI gpt-4o verification were also completed. ADR: backlog/decisions/052-console-conversation-memory-and-compaction-policy.md. Design and delivery plan are in Docs/superpowers/specs and Docs/superpowers/plans. The credential-probe isolation incident is documented in backlog/docs/lessons-testing-evidence.md.

CI hardening follow-up TASK-14811.6 fixed an upstream-reproducible pytest-xdist crash in the OpenAI realtime fake-server harness: positive wire waits are load-tolerant, captured handler errors are traceback-free, and the full 36-test file passes both serially and with CI-equivalent xdist flags.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Console conversations now have explicit durable context budgets and Ask/Automatic/Off compaction, exact request accounting, branch-safe local memory, canonical Console/Settings controls, content-free auxiliary usage provenance, and verified failure/race/privacy behavior.
<!-- SECTION:FINAL_SUMMARY:END -->
