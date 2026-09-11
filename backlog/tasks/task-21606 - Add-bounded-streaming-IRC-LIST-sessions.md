---
id: TASK-21606
title: Add bounded streaming IRC LIST sessions
status: To Do
assignee: []
created_date: '2026-08-23 23:23'
updated_date: '2026-09-10 12:00'
labels:
  - irc
  - channel-browser
  - streaming
  - reliability
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
Give channel browsers incremental, memory-bounded and cancellation-honest LIST
results without pretending a local consumer can stop an IRC server's reply
stream.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One LIST session per connection exposes an asynchronous stream of structured channel, user-count, and topic entries plus a typed terminal result.
- [ ] #2 Pending session state exists before LIST is sent, rows flow through a bounded queue, and caller delivery limits do not require accumulating the complete server list.
- [ ] #3 The terminal result distinguishes normal completion, empty completion, local cancellation, truncation, dropped rows, protocol error, timeout, and disconnect and reports received/delivered/dropped counts.
- [ ] #4 Local cancellation or delivery-limit completion stops caller delivery and drains/discards LIST numerics through `323` while unrelated protocol frames continue to be reduced.
- [ ] #5 A bounded drain deadline settles the caller when a server never sends `323`, retaining one bounded LIST tombstone until the terminal numeric or connection teardown; another LIST is rejected while ambiguous but unrelated chat remains usable.
- [ ] #6 Compatible advertised ELIST filters may be sent, but the API and documentation do not claim universal server-side pagination or cancellation.
- [ ] #7 Deterministic tests cover `321`/`322`/`323`, empty results, malformed counts, error numerics, TRYAGAIN, pressure, local limit, cancellation, disconnect, interleaved chat, and missing terminal numeric.
<!-- AC:END -->

## Handoff

Read the [PTO entry point](../../Docs/superpowers/plans/2026-09-10-network-chat-pto-handoff.md) before claiming this task. Implementation lives in the separate fork; this record stays in Chatbook. ADR-148 records the approved boundary; ADR-149 contains the proposed review corrections. Add an Implementation Plan only after moving this task to In Progress. No implementation or release evidence is claimed by this documentation PR.
