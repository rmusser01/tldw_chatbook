---
id: TASK-41942
title: "Collections capture wiring still compares its factory by `__code__`"
status: To Do
assignee: []
created_date: '2026-11-19 07:30'
labels:
- follow-up
- collections
- adr-356
priority: low
references:
- tldw_chatbook/app_collections_wiring.py
dependencies:
- TASK-41642
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** `app_collections_wiring.py` (~line 200) compares `factory.__code__` to decide the collections capture factory is stock, the identity check ADR-356 rule 4 retired on the Console paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The factory is accepted from the app's registration, not its code object
- [ ] #2 A wrapped factory is accepted and a foreign one refused (tests); the code-identity ratchet drops by the removed read
<!-- AC:END -->
