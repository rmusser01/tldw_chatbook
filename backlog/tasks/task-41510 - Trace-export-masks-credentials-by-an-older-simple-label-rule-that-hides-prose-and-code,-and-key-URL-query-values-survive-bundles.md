---
id: TASK-41510
title: "Trace export masks credentials by an older simple-label rule that hides prose and code, and ?key= URL query values survive bundles"
status: To Do
assignee: []
created_date: 2026-11-17 05:27
labels:
- follow-up
- security
- trace
priority: high
references:
- tldw_chatbook/Chat/console_trace_export.py
dependencies:
- TASK-40301
- TASK-41133
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the trace stream (round 15) after TASK-40301 and TASK-41133. Two gaps: (1) the export's older simple-label rule masks ordinary prose and code such as `token = oauth.fetch_token(url,)`, so a bundle is unreadable where nothing secret is present; the credential-shaped value rule (as used by TASK-40301) should decide instead; (2) `?key=` URL query values that are not `AIza...` survive shared bundles while logs mask them (a TASK-40301 leftover). TASK-41133 covers `whsec_`/`gsk_` and compound labels, not these.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `token = oauth.fetch_token(url,)` and similar prose or code is not masked, and a real credential-shaped value is
- [ ] #2 A `?key=<value>` query value of any shape is masked in bundles as it is in logs
- [ ] #3 Tests pin both, each red before the fix
<!-- AC:END -->
