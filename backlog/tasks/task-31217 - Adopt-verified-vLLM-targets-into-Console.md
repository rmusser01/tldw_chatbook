---
id: TASK-31217
title: Adopt verified vLLM targets into Console
status: Done
assignee: []
created_date: '2026-09-03 22:33'
updated_date: '2026-09-04 03:03'
labels:
  - vllm
  - lab
  - console
  - handoff
dependencies:
  - TASK-31215
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete the Lab workflow by applying a verified vLLM provider, canonical endpoint, and served model to Console with explicit session or durable scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Use in Console is enabled only for the current verified vLLM generation.
- [x] #2 Session adoption updates the active Console provider, endpoint, model, and readiness without writing durable configuration.
- [x] #3 The durable option delegates to the established Settings/provider persistence path and never silently replaces a different configured endpoint.
- [x] #4 Wildcard bind addresses are converted to an explicit usable client endpoint without weakening exposure warnings.
- [x] #5 Mounted Lab-to-Console and persistence regression tests cover session, durable, stale, and rollback paths.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused RED tests for exact secret-free vLLM handoff intents, detached-store validation, session-only Console adoption, Settings draft prefill, stale/detached rollback, pending replay, and endpoint preservation.
2. Implement immutable VllmConsoleIntent/VllmDefaultIntent values and strict detached HandoffChannel validation.
3. Wire verified Lab actions through normal navigation to Console active-session replacement and Settings draft staging, with generation-fenced claim acknowledgement/release and no direct config writes.
4. Run focused pytest, Ruff/mypy where supported, git diff --check, and self-review; capture config non-mutation evidence.

ADR required: yes
ADR path: backlog/decisions/115-vllm-lab-console-readiness-and-profiles.md
Reason: ADR-115 is the accepted cross-screen authority and persistence contract for verified vLLM adoption.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ADR-115 verified-target handoff with exact secret-free Console and Settings intents, strict detached-store reconstruction, and normal navigation from Lab.

Console performs one generation-fenced active-session replacement, preserves differing durable endpoints as Endpoint not saved, and releases stale, detached, failed, or inactive-session claims for replay. Settings stages provider/model/endpoint drafts only, shows endpoint-difference review copy, rolls back without changing config bytes, and leaves the existing Save action as the sole durable writer.

Modified the vLLM setup view, LLMScreen, pending handoff store, Console, canonical Settings screen, and focused Console/Lab/session tests; no app.py change was required because TldwCli already owns PendingHandoffStore and Task 2 already installs the app-scoped readiness owner.

Verification: focused Lab, Console/Settings/provider-persistence, pending-store, and upstream vLLM suites pass; Ruff, focused handoff/store mypy, py_compile, and git diff --check pass. The broad legacy screen mypy invocation still reports its existing baseline errors; no scoped seam errors remain. ADR required: yes. ADR path: backlog/decisions/115-vllm-lab-console-readiness-and-profiles.md.
<!-- SECTION:NOTES:END -->
