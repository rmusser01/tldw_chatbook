---
id: TASK-36761
title: Retain served Console hosts and safely reattach terminal browsers
status: To Do
created_date: 2026-10-25 17:39
dependencies:
- TASK-36759
- TASK-36760
labels:
- console
- web-server
- mobile
- adr-244
priority: high
documentation:
- Docs/superpowers/plans/2026-10-25-shared-console-mobile-supervision-implementation.md
- Docs/superpowers/specs/2026-10-25-shared-console-mobile-supervision-design.md
- backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep accepted Console work alive across browser and VPN disconnects and reconnect a terminal to the same owning App without restarting its runtime.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Repeated authenticated phone or terminal connections reuse one compatible retained child per canonical storage identity, and child creation or saved-chat reservation races never create competing owners.
- [ ] #2 Browser closure, session expiry and device revocation detach only that client's authority; server shutdown or genuine parent/control loss performs bounded native teardown while retaining execution ownership through retirement.
- [ ] #3 Only the current terminal attachment can send input, resize or detach; replacement and stale cleanup cannot affect its successor.
- [ ] #4 Reattachment restores terminal modes and a full render at the new dimensions, including mouse, multiline paste and resize, without restarting the input thread or agent; detached output stays drained and bounded.
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
