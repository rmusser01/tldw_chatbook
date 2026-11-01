---
id: TASK-37602
title: >-
  Console: confirm whether Edit & resend still blocks later captured sends on
  provider-continuation chats
status: To Do
assignee: []
created_date: '2026-10-29 09:50'
labels:
  - console
  - trace
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by code reading in the TASK-33621.7 review (2026-10-29); not reproduced. The rewritten-tail admission (ADR-097 amendment 2026-10-27) is messages-only by design: a recorded surface whose replaced tail includes provider_continuations entries, or a request that carries them, stays refused with unsupported_surface_change. Chats on providers that record continuation groups (moonshot, zai and engine presets with continuation groups) may therefore still stop at the 'Trace capture blocked' card on every send after Edit & resend, which TASK-33621.7 fixed for the other providers. Reproduce with a continuation-recording provider before deciding on a fix.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A real-provider or real-adapter reproduction records whether a send after Edit & resend is refused on a continuation-recording provider
- [ ] #2 If it is refused, later captured sends after Edit & resend on that provider reach the provider with trace capture intact, or the card names the cause, under an ADR-097 amendment
- [ ] #3 A regression test through the real controller, store and trace runtime covers the decided behaviour
<!-- AC:END -->
