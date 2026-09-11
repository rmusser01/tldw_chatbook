---
id: TASK-21602
title: Remove IRC content from diagnostics and harden outbound transport
status: To Do
assignee: []
created_date: '2026-08-23 23:23'
updated_date: '2026-09-10 12:00'
labels:
  - irc
  - security
  - privacy
  - transport
dependencies:
  - TASK-21601
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
Make the fork safe to operate around private conversations and authentication
by replacing raw-frame diagnostics with content-free metadata and by giving
outbound IRC traffic one validated, byte-bounded, stall-bounded ordering seam.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Default and debug diagnostics never contain raw IRC frames, endpoints, nicknames, account/channel names, targets, message/topic bodies, PASS data, channel keys, tokens, SASL payloads, certificate material, or payload-bearing exception messages.
- [ ] #2 Diagnostic records use fixed content-free fields for opaque connection identity, command/capability name, numeric or standard-reply code, encoded size, timing, and reason category.
- [ ] #3 Synthetic plaintext and encoded secrets plus representative private chat content are absent from every captured log field, rendered record, exception representation, and default test artifact on send, receive, parse-error, timeout, and disconnect paths.
- [ ] #4 Outbound application operations preserve FIFO admission order through one writer; queue bytes and entries are bounded, protocol-control capacity prevents PONG starvation, and `write()`/`drain()` calls never interleave.
- [ ] #5 Outbound commands enforce encoded byte budgets and reject CR, LF, NUL, invalid targets, and unsafe unsplittable content without silently truncating credentials, targets, or message text.
- [ ] #6 A configurable bounded drain timeout produces one classified unexpected-disconnect outcome and does not leave a second writer, pending drain, or unobserved task.
- [ ] #7 Deterministic tests prove concurrent ordering, size/control validation, stalled-write behavior, and the complete content-free diagnostic contract.
<!-- AC:END -->

## Handoff

Read the [PTO entry point](../../Docs/superpowers/plans/2026-09-10-network-chat-pto-handoff.md) before claiming this task. Implementation lives in the separate fork; this record stays in Chatbook. ADR-148 records the approved boundary; ADR-149 contains the proposed review corrections. Add an Implementation Plan only after moving this task to In Progress. No implementation or release evidence is claimed by this documentation PR.
