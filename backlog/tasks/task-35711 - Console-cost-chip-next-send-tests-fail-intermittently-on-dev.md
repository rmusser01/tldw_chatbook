---
id: TASK-35711
title: Console cost-chip next-send tests fail intermittently on dev
status: To Do
created_date: 2026-10-20 08:32
labels:
- follow-up
- tests
- flaky
- console
priority: low
references:
- Tests/UI/test_console_cost_chip_screen.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while checking PR #3169's rebase onto dev 0fef489ece (2026-10-20). Four Console cost-chip cases fail intermittently, on plain dev as well as on that PR: `test_excluded_historical_media_keeps_next_send_priced[missing-data]`, `[nonvision]` and `[unaccepted]`, and `test_context_cost_refresh_does_not_materialize_attachment_payloads`, all in Tests/UI/test_console_cost_chip_screen.py. When they fail, the cost label reads "On next send —", where the tests expect a next-send price or "On next send unavailable". In one xdist run all four failed at the PR's head only. Serially, two runs per tree, these four failed 2 of 8 runs at dev 0fef489ece, 1 of 8 at the PR head 509f389f94 and 4 of 8 at the PR's previous head ee15f29f4a; `[assistant]` failed 2 of 2 there. No PR lane runs this file, so nothing gates on it, and a real next-send pricing regression could hide behind the flake. `test_console_settings_modal_save_disabled_during_active_run` failed intermittently on dev in the same runs (its settings button not visible); TASK-35444's notes already record it failing on dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause of the "On next send —" label in these cases is found and recorded
- [ ] #2 The four cases pass 20 consecutive serial runs on dev
- [ ] #3 The file runs in a PR lane, or the task records why it does not
<!-- AC:END -->
