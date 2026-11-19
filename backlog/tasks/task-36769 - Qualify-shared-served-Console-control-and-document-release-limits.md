---
id: TASK-36769
title: Qualify shared served Console control and document release limits
status: To Do
created_date: 2026-10-25 17:40
dependencies:
- TASK-36768
labels:
- console
- web-server
- mobile
- adr-244
priority: medium
documentation:
- Docs/superpowers/plans/2026-10-25-shared-console-mobile-supervision-implementation.md
- Docs/superpowers/specs/2026-10-25-shared-console-mobile-supervision-design.md
- backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Demonstrate the full phone-to-existing-Console workflow through real served processes and browsers and record the remaining physical phone/VPN qualification separately.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A real served child with isolated SQLite and a controlled agent/tool sequence keeps the same process, session and run across terminal closure, phone reconnection and terminal reattachment; mouse, multiline paste and resize work afterward.
- [ ] #2 Multi-chat and multi-subagent correction, background decisions, command acknowledgement loss, expiry, stale attachment races and projection overflow are verified through production routes with honest outcomes.
- [ ] #3 Migration crash/retry, populated conflicts, cross-config source claims and completed-result quarantine have current targeted evidence; retired mobile engine constructors are absent from the shared path.
- [ ] #4 Authentication, revocation, Canvas-disabled operation, same-origin protections and inert content hold in real browser journeys without granting authority from VPN access.
- [ ] #5 User and architecture documentation describe actual shared behavior, migration/recovery instructions and the native-attachment limit; automated/emulated evidence is distinct from an actual phone/VPN receipt and unavailable checks remain explicitly open.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
