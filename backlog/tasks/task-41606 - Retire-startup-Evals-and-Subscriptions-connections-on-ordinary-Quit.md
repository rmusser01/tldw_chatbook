---
id: TASK-41606
title: Retire startup Evals and Subscriptions connections on ordinary Quit
status: In Progress
created_date: 2026-11-18 02:37
assignee:
- '@codex'
labels:
- sqlite
- shutdown
references:
- https://github.com/rmusser01/tldw_chatbook/pull/3610
updated_date: 2026-11-18 03:02
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After a real ordinary Quit the application still holds the startup Evals and Subscriptions SQLite connections on the main thread.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Ordinary unmount retires both idle native startup connections and their leases on the owning thread.
- [x] #2 An active transaction retains its connection and foreign-thread handles retain their existing owners.
- [x] #3 The sealed macOS PTY finds both startup slots and their SQLite descriptors cleared when the app returns normally.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the current ordinary Quit and pin both real startup handles and active transactions in the existing unmount test.
2. Add the two cached app-owned stores to native main-thread retirement after workers settle. Reuse the existing bounded SQLite wait and leave worker-owned connections with their producers.
3. Verify native slot and lease retirement, then rerun the sealed macOS PTY through ordinary Quit.
ADR required: no
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md and backlog/decisions/353-sqlite-store-kernel.md
Reason: This adds omitted instances to the established retirement boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Ordinary unmount now calls each idle file-backed startup store's native close on the main thread after worker settlement. It keeps the existing checkpoint wait and skips active transactions. The native unmount tests cover lease retirement, closed handles, and active writes. The shutdown and connection suite passed 79 cases with one platform skip.

All three baseline PTY runs retained both main-thread slots, two leases, and seven SQLite descriptors at app return. All three candidates retired those same handles before the app returned and had zero private SQLite descriptors and zero target leases. A separate ordinary exit trace caught the existing Subscriptions interpreter hook closing its own transient handle. The PTY journeys had no active evaluation writer. Follow-up CI and qualification remain open.

See [the shutdown evidence](../../Docs/QA/console-library-responsiveness/README.md#cold-notes-sync-and-ordinary-quit), [ADR-126](../decisions/126-complete-local-backup-and-recovery.md), and [ADR-353](../decisions/353-sqlite-store-kernel.md).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
