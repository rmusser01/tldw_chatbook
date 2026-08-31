---
id: TASK-26031
title: Outbound signed webhooks for run lifecycle events
status: To Do
assignee: []
created_date: '2026-08-31 15:46'
labels:
  - interop
  - ops
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Nothing outside the TUI can learn that a run finished. Verified on origin/dev: a named grep for webhook across tldw_chatbook excluding the *_Interop and tldw_api packages returns one hit, a capability-id string at runtime_policy/registry.py:1280 - there is no local emitter. A user running a long agent task has no way to have their own dashboard, phone or script notified. Hermes emits HMAC-signed fire-and-forget notifications from its hook manager. Distinct from inbound webhooks, which need an always-on listener and are deliberately not in scope here.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Run lifecycle events (at minimum: completed, failed, needs-approval) can POST to a user-configured endpoint
- [ ] #2 Requests are signed with a user-supplied secret using a documented scheme so the receiver can verify authenticity
- [ ] #3 Payloads carry identifiers and outcome category only - never message content, tool arguments or credentials
- [ ] #4 Delivery is fire-and-forget with a bounded timeout: a slow or dead endpoint never delays or fails the run
- [ ] #5 Delivery failures are visible somewhere the user can find them rather than silently dropped
- [ ] #6 The destination is subject to the existing SSRF egress policy at Utils/egress.py:1-11
- [ ] #7 Disabled by default; with no endpoint configured no request is ever made
<!-- AC:END -->
