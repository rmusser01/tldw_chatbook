---
id: TASK-34358
title: Learn and calibrate native response rule candidates
status: To Do
assignee:
  - '@codex'
created_date: '2026-10-04 05:54'
labels: []
dependencies:
  - TASK-34357
references:
  - Docs/superpowers/plans/2026-10-03-console-response-rules.md
documentation:
  - backlog/decisions/219-console-learned-response-rules.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable the approved Console response-rule behavior for plan task 5. Preserve existing work and make failure states honest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Learning discriminates the recorded and paraphrased violation from acceptable controls within three attempts and 120 seconds.
- [ ] #2 Synthetic responses cannot fabricate work success and unrelated action guidance remains inactive.
- [ ] #3 Editor tests retain the exact edited candidate and reuse validation only with matching evidence and protocol.
<!-- AC:END -->
