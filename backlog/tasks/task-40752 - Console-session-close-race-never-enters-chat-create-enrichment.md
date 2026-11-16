---
id: TASK-40752
title: Console session-close race never enters chat-create enrichment
status: To Do
type: bug
created_date: '2026-11-13 08:05'
labels:
- console
- tests
- ci
priority: medium
references:
- https://github.com/rmusser01/tldw_chatbook/actions/runs/41289667906/job/124798084440
- Tests/UI/test_console_session_tab_close.py
- tldw_chatbook/Chat/console_chat_controller.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The session-close race journey stops before reaching the enrichment boundary it intends to hold. Restore a reliable test of the ordering between chat-create enrichment and closing its session, and determine whether the missing boundary reflects fixture drift or a runtime issue.

### Reproduce

Use a clean Windows checkout of dev at `beab5c1e4e1128086ac518e525a9274dff4baf5f` with the repository's dev dependencies installed. The audit used Python 3.12.10, Textual 8.2.8 and pytest 8.3.5. From the repository root in PowerShell, activate the dev virtual environment and use a new private test root under a private parent such as `$env:USERPROFILE/.codex`.

```powershell
$testRoot = Join-Path "$env:USERPROFILE/.codex" ('ci-repro-' + [guid]::NewGuid().ToString('N'))
python -c "import sys; from tldw_chatbook.Utils.platform_files import os; os.mkdir(sys.argv[1], 0o700)" "$testRoot"
$env:TLDW_TEST_CONFIG_ROOT = $testRoot
$env:TMP = $testRoot
$env:TEMP = $testRoot
python -m pytest 'Tests/UI/test_console_session_tab_close.py::test_session_close_race_geometry_and_fleet_journeys' -q --tb=short --no-cov
```

### Findings

In the first scenario, `_verify_chat_create_enrichment_cannot_arm_after_its_session_closes` mounts Console, creates a target session, patches `_enrich_chat_create_confirm_payload` to set a threading event, and arms the pending `chat_create` round. At `test_console_session_tab_close.py:1179`, `assert await _settle(pilot, entered.is_set)` is false. The test never reaches the intended close-during-enrichment interleaving, so this result alone does not establish a production close race. TASK-35026 covers historical suite runtime; the earlier TASK-38544 receipt for this test failed later in the fleet retry scenario, at a different assertion.

The audit also tested a detached checkout of the same dev commit after reverse-applying the first-parent diff of merge commit `cf5fda70d3` for `tldw_chatbook/UI/Console_Modules/session.py` and `tldw_chatbook/UI/Console_Modules/wiring.py`, the two production files changed by #3306. The same assertion failed there.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The race scenario proves it reaches the enrichment boundary before closing the target session and verifies that the late enrichment cannot arm confirmation.
- [ ] #2 The race, 80x24 geometry, and fleet retry journey passes with the three scenarios still exercised.
<!-- AC:END -->
