---
id: TASK-42256
title: "The Core Tests workflow job collects inconsistently across xdist workers and loses workers, so most nodes never run"
status: To Do
assignee: []
created_date: 2026-11-22 00:17
updated_date: 2026-11-22 00:17
labels:
- follow-up
- ci
- tests
priority: medium
references:
- .github/workflows/derived-artifacts.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the consoleb stream (round 19). The Core Tests workflow job reported "Different tests were collected" between xdist workers and "node down: Not properly terminated" on runs 44507416213 and 44549304009 (2026-11-19), so most nodes never ran. An existing-task search at e5626ea043 found nothing that covers it: TASK-14876 is a websockets serialization INTERNALERROR, TASK-35573 and TASK-21562 mention the strings in other contexts. The runs were not re-read for this filing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause of the inconsistent collection across workers is named from the two runs' logs
- [ ] #2 The cause of the lost workers is named, or shown to follow from the collection split
- [ ] #3 A Core Tests run after the fix collects the same set on every worker and loses none, with the run id recorded
<!-- AC:END -->
