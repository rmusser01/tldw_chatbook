---
id: TASK-26031
title: Outbound signed webhooks for run lifecycle events
status: In Progress
assignee: []
created_date: '2026-08-31 15:46'
updated_date: '2026-09-01 23:48'
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
- [x] #2 Requests are signed with a user-supplied secret using a documented scheme so the receiver can verify authenticity
- [x] #3 Payloads carry identifiers and outcome category only - never message content, tool arguments or credentials
- [x] #4 Delivery is fire-and-forget with a bounded timeout: a slow or dead endpoint never delays or fails the run
- [x] #5 Delivery failures are visible somewhere the user can find them rather than silently dropped
- [x] #6 The destination is subject to the existing SSRF egress policy at Utils/egress.py:1-11
- [x] #7 Disabled by default; with no endpoint configured no request is ever made
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pure emitter (payload redaction, HMAC-SHA256 signing, config gate, egress check)\n2. deliver_webhook async (bounded timeout, failures logged not raised)\n3. schedule_run_webhook fire-and-forget from the sync worker thread\n4. Wire completed/failed at AgentService._set_terminal_status choke point\n5. [webhooks] config section, default off\n6. needs-approval = follow-up (app-context approval bridge)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Signed run-lifecycle webhooks. completed+failed shipped and tested; needs-approval (the third 'at minimum' event) split to TASK-27020 because it fires from the app-context approval bridge, not a clean run-state seam.

Approach (tldw_chatbook/Agents/run_webhooks.py, new):
- build_webhook_payload: identifiers + outcome only, never content/args/creds (AC#3).
- sign_payload: X-Tldw-Signature: sha256=HMAC-SHA256(secret, raw body) (AC#2, documented).
- webhook_config_from_settings: reads [webhooks]; disabled by default (AC#7).
- deliver_webhook (async): config+event gate -> check_url_or_raise_async (SSRF egress, AC#6) -> bounded-timeout POST -> failures logged+counted, never raised (AC#4/#5).
- schedule_run_webhook: fire-and-forget on a daemon thread with its own loop, callable from the sync agent worker thread; cheap gate before spawning (AC#4).
- Wired at AgentService._set_terminal_status (the atomic terminal choke point): a FRESH persist of RUN_DONE->completed, RUN_ERROR/RUN_STUCK->failed schedules a webhook; cancelled/superseded do not notify. Best-effort, never breaks persistence.
- [webhooks] section added to config.py CONFIG_TOML_CONTENT (enabled=false, url/secret/events/timeout), documented.

AC#1 partial: completed+failed fire (the headline 'notify my dashboard when a long run finishes'); needs-approval deferred to TASK-27020.

Tests: Tests/Agents/test_run_webhooks.py (12: payload redaction, HMAC, config default-off, delivery gating/egress/success/failure, scheduler gating+delivery, and the AgentService terminal-seam mapping). 194 agent-service tests stay green.

Files: tldw_chatbook/Agents/run_webhooks.py (new), tldw_chatbook/Agents/agent_service.py, tldw_chatbook/config.py, Tests/Agents/test_run_webhooks.py.
<!-- SECTION:NOTES:END -->
