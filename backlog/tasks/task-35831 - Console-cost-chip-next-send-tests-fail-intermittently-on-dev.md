---
id: TASK-35831
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
Found while checking PR #3169's rebase onto dev 0fef489ece (2026-10-20). Four Console cost-chip cases fail intermittently, on plain dev as well as on that PR: `test_excluded_historical_media_keeps_next_send_priced[missing-data]`, `[nonvision]` and `[unaccepted]`, and `test_context_cost_refresh_does_not_materialize_attachment_payloads`, all in Tests/UI/test_console_cost_chip_screen.py. When they fail, the cost label reads "On next send —", where the tests expect a next-send price or "On next send unavailable". In one xdist run all four failed at the PR's head only. Serially, two runs per tree, these four failed 2 of 8 runs at dev 0fef489ece, 1 of 8 at the PR head 509f389f94 and 4 of 8 at the PR's previous head ee15f29f4a; `[assistant]` failed 2 of 2 at ee15f29f4a. No PR lane runs this file, so nothing gates on it, and a real next-send pricing regression could hide behind the flake. `test_console_settings_modal_save_disabled_during_active_run` failed intermittently on dev in the same runs (its settings button not visible); TASK-35444's notes already record it failing on dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause of the "On next send —" label in these cases is found and recorded
- [ ] #2 The four cases pass 20 consecutive serial runs on dev
- [ ] #3 The file runs in a PR lane, or the task records why it does not
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-11-03 evidence (dev 9bbd5c2515, round-5 filing of the 2026-10-23 burn-down): three whole-file runs (`-n 4`, load 21-29) had 3, 5 and 7 failures.
- Three of those fail on every run. They are filed as TASK-38690.
- The "On next send —" label (this task's symptom) failed these, each in 1 of the 3 runs:
  - `test_followup_draft_does_not_change_completed_current_spend`;
  - `test_staged_evidence_changes_next_send_but_not_current_spend`;
  - `test_idle_draft_edit_burst_coalesces_one_cost_refresh`;
  - `test_context_cost_refresh_does_not_materialize_attachment_payloads`;
  - `test_system_prompt_revert_after_genuine_ttl_lapse_still_reports_expired` and `test_reverting_system_prompt_edit_with_ttl_remaining_returns_to_warm`, which assert an expired or warm state while the label reads "On next send —".
- Once each, with other symptoms:
  - `test_cost_tooltip_ends_its_info_with_the_providers_rate_limit`: the tooltip still read "Type a message to estimate the next-send input spend.";
  - `test_projected_delta_estimator_skipped_when_warm_without_break_reason`: estimator `call_count` 0, expected 1.
- The round-5 report put the burst test's unpriced label at 1 of 20 runs.
- `test_excluded_historical_media_keeps_next_send_priced` did not fail in these three runs.
<!-- SECTION:NOTES:END -->
