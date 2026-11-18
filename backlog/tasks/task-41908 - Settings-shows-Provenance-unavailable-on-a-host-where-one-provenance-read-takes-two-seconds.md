---
id: TASK-41908
title: "Settings shows Provenance unavailable on a host where one provenance read takes two seconds"
status: To Do
assignee: []
created_date: 2026-11-18 20:31
updated_date: 2026-11-18 20:31
labels:
- follow-up
- settings
- personal-context
priority: low
references:
- tldw_chatbook/Widgets/Settings_Widgets/personal_context_provenance.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41517 (round 16). The provenance lease deadline starts before the read (`personal_context_provenance.py` ~170) and is checked after it (~221), so a read that itself takes 2 s or more is always discarded and the panel shows "Provenance unavailable."; reads of 1-2 s alternate with "Refreshing recorded metadata…". It fails closed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A slow but successful provenance read is shown, with the lease bounding staleness rather than read time
- [ ] #2 A test with a forced slow read is red before the fix
<!-- AC:END -->
