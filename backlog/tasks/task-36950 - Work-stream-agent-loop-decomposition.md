---
id: TASK-36950
title: 'Work stream: agent loop decomposition'
status: To Do
created_date: 2026-10-26 01:12
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
- [ ] #1 Every subtask of this stream is Done
- [ ] #2 agent_service.py and agent_runtime.py have size ratchet rows that this stream lowers
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
