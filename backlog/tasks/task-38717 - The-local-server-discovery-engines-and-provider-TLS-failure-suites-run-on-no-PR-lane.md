---
id: TASK-38717
title: "The local-server discovery engines and provider TLS failure suites run on no PR lane"
status: To Do
assignee: []
created_date: 2026-11-04 00:59
labels:
- follow-up
- ci
- tests
priority: medium
references:
- Tests/Chat/test_local_server_discovery_engines.py
- Tests/Chat/test_provider_tls_failures.py
- .github/workflows/derived-artifacts.yml
dependencies:
- TASK-36532
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 6 of the 2026-10-23 follow-up burn-down (TASK-36532). Checked on dev c2cdb6b82e, 2026-11-03.

TASK-36532 put `Tests/Chat/test_local_server_discovery.py` on the "Run Dreams and Guardian boundary contracts" step of the `dreams-boundary` job (`derived-artifacts.yml:1090-1116`). Two sibling suites from the same provider-setup work (TASK-34100.6) still run on no PR lane:
- `Tests/Chat/test_local_server_discovery_engines.py` (10 test functions);
- `Tests/Chat/test_provider_tls_failures.py` (10 test functions).

Neither is in `scripts/ui_pr_gate_census.txt`, which takes only `Tests/UI`, or a target in any workflow. Only `test.yml`, which runs on a push to main or on dispatch, and the scheduled `nightly-deep.yml` collect them, as part of the whole `Tests/` directory. Both passed at the base in one `-n 4` run alongside other files.

A Tests/Chat file needs a workflow lane target. The owner allowed such a target in `derived-artifacts.yml` for TASK-36532.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both files run on a PR lane as whole-file targets, with their measured time in that lane recorded in the task notes
- [ ] #2 Each file passes 3 of 3 runs in that lane's configuration before it is added
<!-- AC:END -->
