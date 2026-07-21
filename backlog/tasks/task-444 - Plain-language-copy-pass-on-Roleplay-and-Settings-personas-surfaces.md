---
id: TASK-444
title: Plain-language copy pass on Roleplay and Settings personas surfaces
status: To Do
assignee: []
created_date: '2026-07-21 09:38'
labels:
  - roleplay
  - settings
  - copy
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). User-facing surfaces leak internal governance/architecture language: "Dictionaries (embedded copies)" / "World Books (embedded copies)" with no explanation of the embedded-copy semantics, "Authority: Local", the status line "Source: Local | Attachments: Console", and Settings > Domain Defaults > Personas showing "State: Read-only contract" / "destination ownership must be implemented before mutation". Rewrite for users; move contract language to docs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Embedded-copy semantics are explained in place or renamed to something self-evident
- [ ] #2 Settings personas category describes what the page is for in user language
- [ ] #3 Status-line terms are user-meaningful or removed
<!-- AC:END -->
