---
id: TASK-42544
title: "Decide whether to drop the superseded ChaChaNotes index idx_console_trace_response_revision"
status: To Do
assignee: []
created_date: 2026-11-24 23:44
updated_date: 2026-11-24 23:44
labels:
- follow-up
- db
- performance
priority: low
references:
- tldw_chatbook/DB/migrations/chachanotes_v55_to_v56_console_semantic_trace.sql
- tldw_chatbook/DB/migrations/chachanotes_v71_to_v72_voice_trace_provenance.sql
- Tests/DB/test_chachanotes_v56_console_trace_query_plans.py
- scripts/index_plan_pin_census.tsv
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
v56 created the partial index `idx_console_trace_response_revision ON console_trace_response_links(semantic_revision_id) WHERE semantic_revision_id IS NOT NULL`. v72 (b7dd8e53f2) added the full `idx_console_trace_response_links_semantic_revision_exact` on the same column. On a current file with no `sqlite_stat1`, the full index wins every product query on that column. Probed in TASK-42517:
- `WHERE semantic_revision_id = ?`
- the policy join in `console_semantic_revision.py`
- the voice-promotion locator check

The partial index is chosen only for a `semantic_revision_id IS NOT NULL` range scan, which no product query issues.

TASK-42517 re-pinned the v56 plan test to the full index. Its census row stays `plan-pinned`, but the evidence is a genuine v71 file, where the partial index was still chosen. So it is a dead index with a green pin: every write to `console_trace_response_links` maintains it, and nothing reads through it.

Recommendation: drop it in the next ChaChaNotes schema step that is going ahead anyway, with the same requalification as v88:
- the recovery catalog and the restore sandbox admitting exactly that `DROP INDEX`;
- the census row changed or removed (`check_index_plan_pins.py` keeps a row for every name any migration still creates);
- `EXPECTED_CHACHANOTES_INDEXES`.

Keeping it costs one extra index write per response link.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A recorded decision: drop the index in a named schema step, or keep it with the reason
- [ ] #2 If dropped: no product query plan changes (each probed query still plans through the v72 exact index without `sqlite_stat1`), and the census, index literal and recovery catalog agree with the new schema
- [ ] #3 The census row no longer claims a live plan pin for an index no current query chooses
<!-- AC:END -->
