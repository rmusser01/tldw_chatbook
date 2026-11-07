---
id: TASK-35567
title: test_shared_evidence_change_refreshes_the_console_once passes again
status: To Do
created_date: 2026-10-18 17:42
dependencies: []
labels:
- model-config-redesign
- follow-up
- testing
priority: low
references:
- Tests/UI/test_console_endpoint_discovery.py
- tldw_chatbook/UI/Screens/chat_screen.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during Phase 9 (2026-10-18).

`Tests/UI/test_console_endpoint_discovery.py::test_shared_evidence_change_refreshes_the_console_once` (TASK-33005.2) fails at Phase 9's head and at its base 111f8e48fc. Serially the private-profile child fails `assert 0 == 2`; under xdist it fails `assert 1 == 2`, on the shared-evidence version. TASK-35079's notes also recorded it failing on unmodified code. The test is in no PR lane (the census lists that file node by node for this reason), so CI does not see it. No task owns it.

It is not yet known whether the product no longer refreshes the Console once per shared-evidence change, which would be a product defect, or whether the test's setup no longer bumps the version, which would be a test defect. Settle that first.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause is found, with the commit that broke it, and recorded here as a product or a test defect
- [ ] #2 The test passes 10 of 10 serial runs and under xdist, and still fails if the Console's refresh on a shared-evidence change is removed
- [ ] #3 If it is a product defect, the fix restores one Console refresh per shared-evidence change
<!-- AC:END -->
