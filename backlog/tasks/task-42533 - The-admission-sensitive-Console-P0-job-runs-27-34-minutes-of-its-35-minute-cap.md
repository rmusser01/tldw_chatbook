---
id: TASK-42533
title: "The admission-sensitive Console P0 job runs 27-34 minutes of its 35-minute cap"
status: To Do
assignee: []
created_date: 2026-11-24 09:14
updated_date: 2026-11-24 09:14
labels:
- follow-up
- ci
priority: medium
references:
- .github/workflows/derived-artifacts.yml
- Tests/CI/test_response_rules_critical_gate.py
- Tests/CI/test_console_p0_regression_gate.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Raised by the coordinator's review of TASK-42472 (2026-11-24).

**Why.** `console-p0-gate (admission-sensitive)` took 27-34 min on recent PR runs against its 35-min `timeout-minutes`: 45298294580 (27 min), 45278583267 (32), 45300571348 (33), 45280698401 (34). A slow runner or a few more entries will cancel it at the cap. TASK-42472 added one file there (`Tests/Chat/test_console_trace_model_retries.py`, 4 tests, 10.2 s locally) and put its other new lane file in the sandboxed step, after showing it ran green without its `bootstrap_profile` marker.

**Options.** Move files whose `bootstrap_profile` marker is not needed to the sandboxed step (5-6 min in CI), or split the admission-sensitive profile in two. Each file in that step runs in the shared bootstrap profile for TASK-32873's reason, so a moved file must pass in the per-test sandbox first.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Over the next 10 PR runs after the change, the admission-sensitive profile's slowest run ends at least 8 minutes under its cap
- [ ] #2 Each moved file is shown to pass in its new invocation, and whether it needs `bootstrap_profile` is recorded
- [ ] #3 `Tests/CI/test_response_rules_critical_gate.py` and `Tests/CI/test_console_p0_regression_gate.py` match the new lane lists
<!-- AC:END -->
