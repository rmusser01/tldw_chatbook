---
id: TASK-14804
title: Add the pure bounded Console prompt queue registry
status: Done
created_date: 2026-08-10 06:04
labels:
- console
- agents
- architecture
priority: high
references:
- backlog/decisions/046-visible-bounded-console-prompt-queue.md
documentation:
- Docs/superpowers/specs/2026-08-09-console-prompt-queue-design.md
- Docs/superpowers/plans/2026-08-09-console-prompt-queue.md
updated_date: 2026-08-10 15:13
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide a deterministic process-memory owner for per-session queued prompt state before controller scheduling or widgets depend on it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The registry owns immutable text-only entries and render-safe snapshots with stable IDs, FIFO order, per-session isolation, and a hard capacity of ten across waiting plus claimed entries.
- [x] #2 Revision-checked admission, edit, move, remove, clear-waiting, claim, settle, return-to-head, pause, resume, reservation, closing, shutdown, and session-removal transitions reject stale or invalid intents without partial mutation.
- [x] #3 Claimed or starting entries cannot be edited, moved, removed, or cleared as waiting work, and new entries append behind all older work while paused.
- [x] #4 Queue state records the active-chain context baseline needed for first-admission and later context-review decisions without storing provider payloads or authority.
- [x] #5 All transitions are synchronous and event-loop-thread confined; foreign-thread access is rejected or marshalled by callers rather than protected by widget locks.
- [x] #6 The registry has no Textual, provider, database, snapshot, prompt-history, diagnostics, or logging dependency, and queued prompt bodies are never serialized.
- [x] #7 Pure tests cover capacity, FIFO, revisions, lifecycle, every pause and claim transition, and mutation checks for the ten-entry and stale-revision guards.
- [x] #8 Admission and edit precompute a sanitized one-line preview against a fixed maximum cell budget independent of viewport width; body-free snapshots reuse unchanged previews without traversing full prompt bodies, prompt-bearing representations and errors are redacted, and widgets crop the safe preview further after resize.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add red pure tests for immutable IDs, FIFO/session isolation, capacity including claimed work, lifecycle transitions, revision checks, thread confinement, redacted previews/snapshots, and atomic final-empty admission routing. 2. Implement frozen queue models and a synchronous per-session ConsolePromptQueueRegistry with injected ID/clock producers, fixed-cell safe preview generation, revisioned body-free snapshots, context baseline, reservation, close/shutdown, and cleanup transitions. 3. Mutation-check the capacity and stale-revision guards; run focused, import-boundary, compile, and Ruff checks. 4. Self-review, document evidence, check all ACs, and mark Done. ADR required: yes. ADR path: backlog/decisions/046-visible-bounded-console-prompt-queue.md. Reason: ADR-046 already governs transient queue ownership and transition semantics; this task implements it without introducing a new decision.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented a synchronous owner-thread-confined ConsolePromptQueueRegistry with immutable redacted entries and claims, per-session FIFO state, ten-entry waiting-plus-claimed capacity, revision-checked lifecycle transitions, context baselines, reservations, closing/shutdown cleanup, exact final-empty/admission rerouting, selected-entry text reads, and cached body-free snapshots. Added fixed-cell ANSI/control/Rich-safe preview generation and 34 pure tests covering lifecycle, isolation, locks, all pause reasons, race orderings, thread confinement, redaction, Unicode previews, dependency boundaries, and non-traversal of prompt bodies. Mutation checks: raising the cap to 11 failed test_capacity_counts_waiting_plus_claimed; bypassing admission revision checking failed both stale-intent tests. Verification: focused 34 passed; Ruff check and format check passed; py_compile passed. Broader Chat fail-fast reached 208 passed/1 skipped before the unrelated absent pytest-mock mocker fixture; full collection found 35,811 tests and only two existing Confluence collection errors from absent optional Playwright. ADR required: yes; implemented existing ADR-046, with no new ADR. No persistent format, database, prompt-history, diagnostic, logging, provider, Textual, or screen dependency was added.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the pure process-memory queue state machine required by ADR-046. It now provides the deterministic, privacy-bounded foundation for TASK-14805's coordinator without making the queue visible or persistent.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] Acceptance criteria completed and implementation plan followed.
- [x] Automated pure-state coverage added and both required guards mutation-checked.
- [x] Ruff, Ruff format, and Python compilation checks pass for changed Python files.
- [x] Broader regression and full collection blockers are documented as missing optional test dependencies, not change-attributable failures.
- [x] ADR-046 reviewed and linked; no new ADR is required.
- [x] Self-review completed, including exact queue-empty/admission race handling and selected-entry-only body reads.
- [x] No new generalized lesson was needed; the task followed the existing mutation, collection, and backlog-hygiene lessons.
<!-- DOD:END -->
