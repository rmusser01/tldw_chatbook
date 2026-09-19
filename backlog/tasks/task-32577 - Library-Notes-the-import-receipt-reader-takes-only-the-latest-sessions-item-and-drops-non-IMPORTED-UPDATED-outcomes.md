---
id: TASK-32577
title: >-
  Library Notes: the import receipt reader takes only the latest session's item
  and drops non-IMPORTED/UPDATED outcomes
status: To Do
assignee: []
created_date: '2026-09-14 22:45'
labels:
  - library
  - notes
  - rider
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Confirmed three times in wave 4 — by the task-32541 implementer, by its landing agent, and again in the ruling that sent it here. Repeat detection reads prior import receipts to recognise a source you already imported. The reader takes only the LATEST session's item for a source and keeps it only when the outcome is IMPORTED or UPDATED. So a re-review where everything was set to Skip overwrites the useful receipt with a skipped one and loses repeat detection for that source entirely — now including the CSV and YAML sources task-32541 just taught it to match. The effect is that the second import of a folder can present sources as New that the user has already imported.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A later Skip outcome does not destroy the repeat evidence an earlier Import or Update recorded for the same source
- [ ] #2 Repeat detection is driven by the newest receipt that actually carries evidence, not by the newest receipt of any kind
- [ ] #3 Pinned end to end: import a folder, re-review with everything on Skip, re-review again, and the sources still land under Unchanged repeat
- [ ] #4 Covers the multi-record sources (.csv, .yaml, .txt) that task-32541 brought into repeat matching
<!-- AC:END -->
