---
id: TASK-21604
title: Implement ordered IRC protocol reduction and correlated replies
status: To Do
assignee: []
created_date: '2026-08-23 23:23'
updated_date: '2026-09-10 12:00'
labels:
  - irc
  - asyncio
  - reliability
  - architecture
dependencies:
  - TASK-21603
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
Make protocol state deterministic by applying inbound IRC frames in wire order,
separating state commits from application callbacks, and publishing query
correlation before a fast server can reply.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every complete inbound frame has monotonic receive order and valid frames mutate core protocol state in that order even when callbacks are delayed, fail, or issue additional IRC commands.
- [ ] #2 Protocol reducers perform only bounded state/correlation work and never await a future whose completion requires a later inbound frame; JOIN/WHOX and WHOIS flows cannot self-deadlock.
- [ ] #3 WHOIS and each other included query publish pending future/result state before their command can reach the writer, and immediate scripted replies resolve exactly once without being lost.
- [ ] #4 Correlated operations have distinct success, rejection, timeout, cancellation, malformed-response, and disconnect outcomes, and every pending operation terminates before connection state is discarded.
- [ ] #5 Application callbacks run after the corresponding state commit through owned bounded ordered delivery; callback failure is observable and cannot roll back, reorder, or directly mutate committed protocol state.
- [ ] #6 Presence/state events may coalesce only under the documented identity rule, message and query-result events are never silently dropped, and full callback queues trigger a classified slow-consumer close without making the reducer wait for callback capacity or query completion.
- [ ] #7 Unsupported ambiguous concurrent queries are rejected deterministically; cancelled/timed-out unlabelled queries retain bounded tombstones until their wire terminal or disconnect, preventing late replies from resolving a replacement query, using negotiated IRC casemapping for target identity.
- [ ] #8 Deterministic race, ordering, slow-callback, callback-failure, immediate-reply, query-cancellation, and shutdown tests prove the new reducer/correlation boundary.
<!-- AC:END -->

## Handoff

Read the [PTO entry point](../../Docs/superpowers/plans/2026-09-10-network-chat-pto-handoff.md) before claiming this task. Implementation lives in the separate fork; this record stays in Chatbook. ADR-148 records the approved boundary; ADR-149 contains the proposed review corrections. Add an Implementation Plan only after moving this task to In Progress. No implementation or release evidence is claimed by this documentation PR.
