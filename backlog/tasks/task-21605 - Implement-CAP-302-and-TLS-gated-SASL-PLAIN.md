---
id: TASK-21605
title: Implement CAP 302 and TLS-gated SASL PLAIN
status: To Do
assignee: []
created_date: '2026-08-23 23:23'
updated_date: '2026-09-10 12:00'
labels:
  - irc
  - ircv3
  - sasl
  - security
dependencies:
  - TASK-21604
references:
  - backlog/decisions/148-network-chat-ircv3-and-tldw-pydle-boundary.md
  - backlog/decisions/149-network-chat-handoff-reliability-amendments.md
documentation:
  - Docs/superpowers/plans/2026-09-10-network-chat-pto-handoff.md
  - Docs/superpowers/plans/2026-08-23-tldw-pydle-first-release.md
  - Docs/superpowers/specs/2026-08-23-network-chat-ircv3-and-tldw-pydle-design.md
priority: high
type: task
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide deterministic modern registration and dependency-free authentication
so capabilities, readiness, and SASL state remain correct across multiline,
failure, timeout, and post-registration changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 CAP LS 302 accumulates every continuation row and preserves opaque case-sensitive capability names plus allowed values before computing requests.
- [ ] #2 Capability state distinguishes offered, supported, requested, pending, negotiated, and semantically supported, with deterministic encoded-line-safe CAP REQ chunks and correct ACK/NAK handling per request chunk; continuation syntax applies only to LS/LIST and values only to LS/NEW.
- [ ] #3 CAP END is sent exactly once during registration, CAP NEW/DEL update post-registration state without another registration END, and readiness transitions/emits exactly once on numeric `001` for all waiters only after required authentication succeeds; a legacy server without CAP remains usable when no mandatory capability is configured.
- [ ] #4 SASL PLAIN is implemented without `pure-sasl`, produces the correct base64 payload, uses 400-byte AUTHENTICATE chunks, and sends the required `+` terminator for empty or exact-multiple responses.
- [ ] #5 SASL PLAIN is refused unless the active transport completed certificate and hostname verification; required-SASL unavailability or rejection has a typed terminal outcome and never falls through as authenticated.
- [ ] #6 SASL timeouts are owned and cancellable, the upstream continuation-timer callback defect is regression-tested, and secret references are released after success, rejection, cancellation, timeout, or disconnect.
- [ ] #7 Multiline CAP, values, split requests, ACK/NAK, NEW/DEL, required/optional SASL, chunking, malformed challenge, timeout, immediate reply, and one-time readiness are covered by deterministic tests with content-free logs.
<!-- AC:END -->

## Handoff

Read the [PTO entry point](../../Docs/superpowers/plans/2026-09-10-network-chat-pto-handoff.md) before claiming this task. Implementation lives in the separate fork; this record stays in Chatbook. ADR-148 records the approved boundary; ADR-149 contains the proposed review corrections. Add an Implementation Plan only after moving this task to In Progress. No implementation or release evidence is claimed by this documentation PR.
