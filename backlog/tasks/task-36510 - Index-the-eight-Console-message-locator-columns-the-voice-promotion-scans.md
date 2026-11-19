---
id: TASK-36510
title: Index the eight Console message-locator columns the voice promotion scans
status: To Do
assignee: []
created_date: 2026-10-23 22:44
labels:
- follow-up
- performance
- database
priority: medium
references:
- Tests/Performance/test_speculative_voice_latency.py
- Packaging/speculative_voice_history_gate.py
- scripts/index_plan_pin_census.tsv
dependencies:
- TASK-36390
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36390 and its review (2026-10-23). The speculative-voice history gate promotes a completed pair by probing every message-locator column with `SELECT 1 ... WHERE <column> IN (?, ?)`. On a database with no `sqlite_stat1` (no DB module here runs `ANALYZE`, so this is every user's database), eight of those columns have no index that leads with them, and each probe scans its table:
- `console_hook_continuation_receipts`: `assistant_message_id` and `parent_assistant_message_id`
- `console_machine_followup_receipts`: `assistant_message_id` and `parent_assistant_message_id`
- `console_response_rule_drafts.message_id`, `console_response_rule_fixtures.message_id` and `console_response_rule_validations.message_id`
- `console_response_rule_assessments.message_id`. Its only index, `idx_response_rule_assessments_source`, is `(conversation_id, message_id)`.

The measured fresh-commit median paired delta is 5.1 ms, against the gate's 2.0 ms limit. A probe that indexed seven of the columns brought the gate's scan count to 0 and the delta to 0.06 ms.

The gate under-reports. It runs `ANALYZE` after seeding and leaves `conversation_id` all-NULL, so the planner skip-scans the assessments index, and the gate counts 14 scans (7 x 2) instead of 16.

TASK-36390 marks the history node `xfail(strict=True)` on this task.

The columns are `REFERENCES messages(id) ON DELETE CASCADE` children, so the indexes probably also help message deletes. Nobody has measured that.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 On a fresh schema with no sqlite_stat1, each of the eight columns is searched through an index rather than scanned, by a schema migration
- [ ] #2 Each new index has a query-plan pin captured with sqlite_stat1 absent
- [ ] #3 The voice history gate reads its query plans before any ANALYZE, so a locator column without a leading index fails it
- [ ] #4 The history node's strict xfail is removed and the node passes
<!-- AC:END -->
