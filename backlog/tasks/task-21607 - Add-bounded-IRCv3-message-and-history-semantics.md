---
id: TASK-21607
title: Add bounded IRCv3 message and history semantics
status: To Do
assignee: []
created_date: '2026-08-23 23:23'
updated_date: '2026-09-10 12:00'
labels:
  - irc
  - ircv3
  - history
  - protocol
dependencies:
  - TASK-21605
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
Add the bounded modern message semantics needed by a useful IRCv3 client while
keeping draft history behavior capability-gated and keeping Chatbook's stable
events, paging, and retention policy outside the fork.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Message-tag parsing implements IRCv3 escaping and byte limits, retains bounded unknown tags without trusting them, and classifies malformed or oversized input without logging content.
- [ ] #2 `server-time` produces a validated timezone-aware timestamp, and echo-message preserves authoritative server echoes without collapsing identical legitimate messages or treating socket writes as delivery receipts; absent correlation identity yields an explicit unconfirmed local-send state.
- [ ] #3 BATCH state preserves ID, type, parameters, parent relation, ordered membership, and open/close lifecycle within explicit open-batch and member bounds, including unknown batch types and incomplete disconnect; historical events cannot mutate live membership/topic state, and unsupported event-playback capabilities are not requested.
- [ ] #4 Negotiated `draft/chathistory` requests preserve target, anchor/reference type, limit, batch identity, message ID, server time, sender/target, message type, and live-versus-history origin.
- [ ] #5 Empty history, standard-reply/numeric failure, unsupported references, malformed or mismatched batches, server over-return, cancellation, timeout, and disconnect all terminate the correlated history operation truthfully.
- [ ] #6 Fork models remain immutable adapter-facing views and do not take ownership of Chatbook display pagination, cross-page deduplication, transcript retention, unread state, or persistence.
- [ ] #7 Deterministic resource-bound and semantic tests cover escaped/unknown/oversized tags, live echo, nested and unknown batches, valid/empty/incomplete history, timestamp/msgid anchors, over-return, failure, cancellation, and disconnect.
<!-- AC:END -->

## Handoff

Read the [PTO entry point](../../Docs/superpowers/plans/2026-09-10-network-chat-pto-handoff.md) before claiming this task. Implementation lives in the separate fork; this record stays in Chatbook. ADR-148 records the approved boundary; ADR-149 contains the proposed review corrections. Add an Implementation Plan only after moving this task to In Progress. No implementation or release evidence is claimed by this documentation PR.
