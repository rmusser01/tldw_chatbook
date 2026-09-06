---
id: TASK-31738
title: llama.cpp opt-in automatic per-conversation prompt-cache reuse
status: To Do
assignee: []
created_date: '2026-09-05 19:54'
labels:
  - llamacpp
  - snapshots
  - follow-up
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow up on TASK-31552 and PR #2419: make prompt-cache persistence and reuse automatic for an explicitly opted-in conversation using a Chatbook-owned local llama.cpp server. Durable conversation history remains authoritative; the manual manager remains available. This task is deferred and does not authorize implementation now. Before implementation, review ADR-119 and record the conversation-routing, lifecycle, and ownership decisions in a linked ADR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Automation is opt-in and disabled by default; existing manual snapshots and ordinary chat remain usable without enabling it.
- [ ] #2 A conversation uses only cache state compatible with its own history and the active model, projector, runtime, and launch; switching, editing, or branching history cannot silently reuse another conversation or stale history.
- [ ] #3 Concurrent sends, conversation switches, cancellation, server restart, and uncertain Save or Restore outcomes cannot cross-bind slots or trigger unsafe retries; unavailable or incompatible caches fall back to an ordinary history send with honest status.
- [ ] #4 Automatic snapshots preserve private local ownership and a documented configurable retention policy defaulting to 10; cache loss never deletes conversation history or enters sync/export.
- [ ] #5 Targeted regressions and isolated real-server UAT demonstrate same-conversation reuse, cross-conversation isolation, restart behavior, and fallback through normal Chatbook send paths; documentation states the tested limits.
<!-- AC:END -->

## References

- [Completed manual manager](task-31552%20-%20llama.cpp-manual-prompt-cache-snapshot-manager.md)
- [Merged PR #2419](https://github.com/rmusser01/tldw_chatbook/pull/2419)
- [ADR-119: snapshot ownership](../decisions/119-llamacpp-prompt-cache-snapshot-ownership.md)
- [Live UAT and qualification limits](../../Docs/superpowers/reviews/2026-09-05-llamacpp-slot-snapshots-uat.md)
