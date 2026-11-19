---
id: TASK-35059
title: 'Timer census rot: 13 unclassified repeating-clock .update() sites on dev'
status: To Do
created_date: 2026-10-14 20:03
labels:
- bug
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/Architecture/test_timer_path_static_update_inventory.py is deterministically red on pristine origin/dev (verified in a clean worktree): 13 .update() calls reachable from repeating set_interval/set_timer clocks do not pass layout= and have no CLASSIFIED_SITES entry - dev-side rot as the timer sites evolved (llamacpp_setup_view refresh_state, left_rail sync_model_recovery, console_runtime ensure_chat_controller, console_interrupt_rounds refresh_decision_clocks, MetricsLogger._get_labels, PersonaBuddyConsoleAdapter._owner, LocalSkillsService._summary_for_record, chat_screen _sync_console_cost_chip). Fix per the gate's own instructions: pass layout=False with a geometry-equivalence A/B where rendered size cannot change, or add CLASSIFIED_SITES entries with the reason. Until fixed, this required lane stays red for every PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every repeating-clock .update() site either passes layout=False with a geometry A/B proof or has a CLASSIFIED_SITES entry with its reason,test_timer_path_static_update_inventory.py green on dev,The classified-site rationales recorded for future reviewers
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

## Renumbering provenance

Renumbered from task-34992 to TASK-35027 to TASK-35059 (2026-10-15): dev minted the same id after this branch cut; per the owner rule (TASK-19601) the dev-side arrival keeps the id on merge.
