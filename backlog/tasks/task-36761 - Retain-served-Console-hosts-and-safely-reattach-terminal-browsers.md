---
id: TASK-36761
title: Retain served Console hosts and safely reattach terminal browsers
status: Done
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
updated_date: 2026-10-25 20:58
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep accepted Console work alive across browser and VPN disconnects and reconnect a terminal to the same owning App without restarting its runtime.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Repeated authenticated phone or terminal connections reuse one compatible retained child per canonical storage identity, and child creation or saved-chat reservation races never create competing owners.
- [x] #2 Browser closure, session expiry and device revocation detach only that client's authority; server shutdown or genuine parent/control loss performs bounded native teardown while retaining execution ownership through retirement.
- [x] #3 Only the current terminal attachment can send input, resize or detach; replacement and stale cleanup cannot affect its successor.
- [x] #4 Reattachment restores terminal modes and a full render at the new dimensions, including mouse, multiline paste and resize, without restarting the input thread or agent; detached output stays drained and bounded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
Reason: Direct implementation of the accepted retained-host, attachment and physical lifecycle boundaries.

1. Add failing retained-host/current-attachment and driver-handshake tests against the installed Textual serve seams.
2. Retain one serialized AppService per admitted canonical identity, continuously drain bounded output, and generation-fence terminal input/resize/detach and saved-conversation reservations.
3. Add the repo-owned driver render handshake, restore application modes/full repaint without restarting input, and route actual parent/control loss through bounded native teardown with the existing execution fence.
4. Exercise actual App child/browser retention, takeover/revocation, click/paste/resize and shutdown. Resolve the reproduced left-rail initialization crash narrowly if it blocks required qualification.
5. Run the plan's targeted connected tests and scoped static/format checks, update living docs, commit explicit changes, then obtain independent spec and quality review before completion.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented retained per-storage Console hosts, bounded replaceable terminal attachments and a generation-tagged full-render handshake. Browser disconnect/takeover/revocation preserves native work; genuine control loss and server shutdown retain ownership through physical retirement. Added host reserve/register seam; remote saved-open/save integration remains Task9 per controller ruling. Fixed the reproduced partial-mount Console rail resize crash. Targeted actual-App Chromium checks cover one process/input thread, in-flight work across detach, transcript, mouse/paste/resize, takeover/revocation, TLS/proxy Canvas isolation and physical exit. ADR-244 applied; living architecture and verification lesson updated. Scoped default Ruff adds zero diagnostics versus exact BASE; formatting ratchet passes. Detailed report: .superpowers/sdd/2026-10-25-shared-console-mobile-supervision-implementation/task-3-report.md.
Independent Task3 spec and quality review completed after resize-ordering fix b6858fe490 (implementation b4c59028ab). Fix round1 addressed the only Important finding: delayed reattach begin now preserves a newer resize; RED2 failures then driver5 passes and actual-App reattach1 pass. Final actual-child retirement/reconnect4 passes, ownership17 passes, corrected spawn4 passes, connected runtime60 passes; direct TLS/trusted-proxy/Canvas isolation coverage recorded in task-3-report.md. Scoped default Ruff336 current=336 exact BASE, zero introduced; format ratchet, new-file formatting, syntax, JS and diff checks pass. Same PID/input thread, detached completion/full transcript, mouse/multiline paste/resize, takeover/revocation and physical control-loss retirement verified. Existing aiohttp deprecation and Textual upgrade risk remain explicit. ADR-244 applies. Task9 owns remote saved-open/save callback integration and canonical phone cutover per recorded scope ruling; Task3 verifies host reserve/register seam, not final shared-mobile release. No full sweep, external provider, user-data migration, push or merge.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
