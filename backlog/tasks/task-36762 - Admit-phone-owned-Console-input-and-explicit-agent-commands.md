---
id: TASK-36762
title: Admit phone-owned Console input and explicit agent commands
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
Let phone users send text and correct exact existing main agents or subagents without borrowing the desktop composer or changing its active chat.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Phone Send uses immutable command-owned text through native per-session admission, checked preparation and queue custody while preserving desktop drafts, attachments, one-shot prefill and staged RAG.
- [ ] #2 Desktop typing, view changes or another device's edits alone do not reject or consume phone input; real session, settings, binding, permission and admission changes still refuse stale work.
- [ ] #3 Main Send, Steer, Redirect and Stop address exact session incarnations and applicable runs; existing active-tab wrappers use the same native implementation.
- [ ] #4 Subagent guidance and Stop remain scoped to the exact conversation/handle/run, distinguish queued from consumed guidance and refuse finished or replaced targets without relaunching them.
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
