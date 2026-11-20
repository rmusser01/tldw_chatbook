---
id: TASK-36950
title: 'Work stream: agent loop decomposition'
status: Done
created_date: 2026-10-26 01:12
updated_date: '2026-11-20 15:10'
dependencies:
- TASK-36942.1
labels:
- architecture-review-2026-10-25
- agents
- architecture
priority: medium
documentation:
- Docs/QA/architecture-review-2026-10-25/report.md
- Docs/QA/architecture-review-2026-10-25/verification.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`AgentService._run_one` is a single 4,163-line function of 45 closures, and `run_agent_loop` is 2,451 lines whose 870-line per-call block branches 23 times on the tool name. Both sit on every agent run. The phases are already visible in the code; this stream makes them separate units, by pure moves only.

Source: architecture review 2026-10-25, `Docs/QA/architecture-review-2026-10-25/report.md` §F5; claim-by-claim verification in `Docs/QA/architecture-review-2026-10-25/verification.md` (origin/dev 7a5e2513c5).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every subtask of this stream is Done
- [x] #2 agent_service.py and agent_runtime.py have size ratchet rows that this stream lowers. **Result: NOT MET for agent_runtime.py, which has no row. The owner waived it on 2026-11-20.** agent_service.py's row was lowered by this stream.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Closed 2026-11-20 on dev `d85e90bc9e`. AC #1: all four subtasks (TASK-36950.1 to .4) are Done.

AC #2, waived by the owner on 2026-11-20:
- agent_service.py: TASK-36942.1 (`2e0adfee71`) added its row at 9,629. TASK-36950.3 (`a6c33f6976`) lowered it from 9,720 to 8,099, and TASK-36950.4 (`9bfa8eba5a`) from 8,099 to 7,677. It is 7,696 now.
- agent_runtime.py: no row, and none is required. It was 3,523 lines at `2e0adfee71`, under the 4,000-line row threshold (`_ROW_REQUIRED_AT_LINES`), so the criterion assumed a row that never existed. TASK-36950.2's pure move turned `run_agent_loop` from 2,190 lines into a 102-line driver over phase functions. Their extra `def` lines took the file from 3,716 to 3,954 lines. It is 3,961 now.
- Reason for the waiver: the stream's aim, separate units for the loop's phases, is met. A row added now could not show a reduction by this stream.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
