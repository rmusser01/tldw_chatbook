---
id: TASK-562
title: Sweep Select.BLANK to Select.NULL semantics in settings_screen
status: To Do
assignee: []
created_date: '2026-07-25 07:57'
labels:
  - settings
  - rag
  - tech-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Final 541 review confirmed Select.BLANK does not exist on Textual 8.2.7 (silently resolves to Widget.BLANK == False). Four sites in settings_screen.py compare/compose with it; the compose fallback (~:8405, SP3-era) would raise InvalidSelectValueError at mount if a hand-corrupted config pointer names a nonexistent profile. Others are dead comparisons degrading UX copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All Select.BLANK usages in settings_screen.py replaced with Select.NULL-correct logic,Corrupt active-profile pointer no longer crashes Settings mount (regression test),Blank selection + Set active yields a friendly notice instead of an adapter error
<!-- AC:END -->
