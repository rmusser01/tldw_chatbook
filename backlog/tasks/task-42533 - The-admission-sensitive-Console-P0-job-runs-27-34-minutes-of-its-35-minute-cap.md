---
id: TASK-42533
title: "The admission-sensitive Console P0 job runs 27-34 minutes of its 35-minute cap"
status: To Do
assignee: []
created_date: 2026-11-24 09:14
updated_date: 2026-11-24 10:23
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

**Why.** `console-p0-gate (admission-sensitive)` took 27-34 min on recent PR runs against its 35-min `timeout-minutes`: 45298294580 (27 min), 45278583267 (32), 45300571348 (33), 45280698401 (34). A slow runner or a few more entries will cancel it at the cap. TASK-42472 added one file there (`Tests/Chat/test_console_trace_model_retries.py`, 4 tests, 10.2 s locally). Its other new lane file, `Tests/Chat/test_console_failed_call_retry_ownership.py`, went to the Dreams tracking riders instead. That file reads config: a first attempt to move it to the sandboxed step without its marker passed only because a swallowed `RecoveryRequired` fell back to the default.

**Options.** Move files whose `bootstrap_profile` marker is not needed to the sandboxed step (5-6 min in CI), or split the admission-sensitive profile in two. Each file in that step runs in the shared bootstrap profile for TASK-32873's reason. A moved file must therefore be shown to read no config in the per-test sandbox. A green run is not proof: record `get_cli_setting` calls and `RecoveryRequired` raises, and require zero raises (see the lesson "Green without `bootstrap_profile` does not prove a file reads no config").
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Over the next 10 PR runs after the change, the admission-sensitive profile's slowest run ends at least 8 minutes under its cap
- [ ] #2 Each moved file passes in its new invocation, and a recorder shows no swallowed `RecoveryRequired` there (any file still needing `bootstrap_profile` stays out of sandboxed invocations)
- [ ] #3 `Tests/CI/test_response_rules_critical_gate.py` and `Tests/CI/test_console_p0_regression_gate.py` match the new lane lists
<!-- AC:END -->
