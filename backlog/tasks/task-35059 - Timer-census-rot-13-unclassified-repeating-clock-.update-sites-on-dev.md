---
id: TASK-35059
title: 'Timer census rot: 13 unclassified repeating-clock .update() sites on dev'
status: To Do
created_date: 2026-10-14 20:03
updated_date: 2026-11-10 18:16
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
Reconciled by TASK-39129 (2026-11-10, base ae9f9e8d21): none of the 13 sites this task lists is reported by `Tests/Architecture/test_timer_path_static_update_inventory.py` any more; later tasks classified or rewrote them. The file's reds on that base were TASK-39129's (Plugins, provider detection, composer raw-CLI status, the stale voice-chip row, five buddy-poll approval-card sites; classified on TASK-39129's branch) and TASK-36191's (`ProjectIndexBindingCard._poll`). With TASK-39129's rows and a temporary stand-in for TASK-36191's, the file passes 17/17. Status left To Do (corrected 2026-11-10): its criterion still applies and is not met. Its 13 sites are no longer reported, but "test_timer_path_static_update_inventory.py green on dev" holds only once TASK-39129's rows reach dev and TASK-36191 handles `ProjectIndexBindingCard`; until then the file stays red on dev. Close it when that file passes on dev.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

## Renumbering provenance

Renumbered from task-34992 to TASK-35027 to TASK-35059 (2026-10-15): dev minted the same id after this branch cut; per the owner rule (TASK-19601) the dev-side arrival keeps the id on merge.
