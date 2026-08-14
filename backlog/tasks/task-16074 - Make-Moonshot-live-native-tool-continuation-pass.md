---
id: TASK-16074
title: Make Moonshot live native-tool continuation pass
status: In Progress
assignee: []
created_date: '2026-08-14 02:20'
labels: []
dependencies:
  - TASK-15676
references:
  - backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md
  - Docs/superpowers/specs/2026-08-13-task-16074-moonshot-live-tool-uat-fix-design.md
  - Docs/superpowers/plans/2026-08-13-task-16074-moonshot-live-tool-uat-fix.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Correct the post-merge Moonshot Kimi K3 integration defect found by paid UAT so the real Console tool-call and continuation path completes successfully without weakening the provider contract or exposing credentials.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A doubly gated paid Moonshot Kimi K3 probe completes exactly one calculator call, continues with the tool result, and returns the required final marker.
- [ ] #2 The exact Moonshot SSE metadata field that triggered Chatbook's synthetic HTTP 502 error is accepted under a bounded allowlist and pinned by an automated regression at the real provider boundary.
- [ ] #3 Moonshot credentials and captured live/raw provider payloads remain absent from logs, tracebacks, fixtures, and committed files; regressions use only minimal synthetic SSE data.
- [ ] #4 Focused Moonshot, hosted Chat, AgentService, and Console continuation regressions remain green without changing unrelated provider behavior.
<!-- AC:END -->

## Implementation Plan

1. Pin Moonshot's bounded `system_fingerprint` streaming event in the neutral
   hosted parser and joined Console native-tool fixtures with strict RED tests.
2. Apply the minimal provider-neutral streaming allowlist/metadata validation
   correction and prove unknown or oversized metadata still fails closed.
3. Run only focused hosted/Moonshot/AgentService/Console/privacy regressions,
   then the doubly gated paid Moonshot UAT.
4. Close task evidence, rebase on `dev`, open the follow-up PR, address its
   checks/review comments, merge it, and clean up the branch/worktree.

ADR required: no

ADR path: backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md

Reason: this is a compatibility correction within ADR-063's existing hosted
provider wire and durable continuation boundaries; it does not introduce a new
storage, sync, security, dependency, or cross-module ownership decision.
