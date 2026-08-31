---
id: TASK-25888
title: >-
  Console trace-call ledger has no producer, so durable capture can never be
  enabled
status: To Do
assignee: []
created_date: '2026-08-31 20:05'
labels:
  - console
  - ux-review
  - follow-up
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-25814 stopped Console claiming Capture-On the runtime cannot honour, which restored the core loop. That fix is a truthful fallback, not a completion: the ledger's producer half does not exist. ConsoleTraceCallBoundary is constructed only in tests, no production code calls create_segment or attach_owner, and both callers of ensure_provider_gateway omit trace_call_boundary_factory. Until an owner and segment are established for a conversation and the call identity's turn, run, sequence and idempotency semantics are decided, supports_durable_capture is permanently False and Capture-On is unreachable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A production Console gateway is constructed with a working trace-call boundary factory
- [ ] #2 A conversation acquires a trace owner and segment through a defined, tested path
- [ ] #3 supports_durable_capture reports True in production and a Capture-On send records a durable call
- [ ] #4 A test builds the gateway the way production builds it and asserts the capability, so the gap cannot silently return
<!-- AC:END -->
