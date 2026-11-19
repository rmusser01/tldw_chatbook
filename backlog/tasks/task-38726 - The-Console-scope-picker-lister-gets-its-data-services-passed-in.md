---
id: TASK-38726
title: "The Console scope-picker lister gets its data services passed in"
status: To Do
created_date: 2026-11-04 02:52
labels:
- follow-up
- architecture-review-2026-10-25
- console
priority: low
references:
- tldw_chatbook/Chat/scope_picker_listers.py
dependencies:
- TASK-36944.5
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-36944.5 broke the Chat/Widgets cycle around the scope picker. The lister in Chat/scope_picker_listers.py still imports the DB directly (`fetch_keywords_for_media_batch`) and reaches into the app through getattr strings to find its services. The task description noted both as context; they were out of its scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 scope_picker_listers receives the database and app services it uses from whoever builds it, and does no getattr-by-string lookups on the app
- [ ] #2 The lister no longer imports tldw_chatbook.DB
- [ ] #3 The scope-picker tests show no new failures against dev
<!-- AC:END -->
