---
id: TASK-32274
title: Subscriptions DB refuses to open when schema_version holds both 1 and 2
status: Done
assignee:
  - '@rmusser'
created_date: '2026-09-10 19:10'
updated_date: '2026-09-10 20:53'
labels:
  - db
  - watchlists
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Startup crashes with 'Unsupported subscriptions schema version' when the schema_version table contains rows 1 and 2. The fresh-create path inserts 2 with INSERT OR IGNORE while the migration path deletes then inserts; a scratch profile reached the two-row state after an abnormal exit. The failure is a raw traceback with no recovery hint and blocks the whole app. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A schema_version table containing the current version alongside older rows opens and is normalised to the current version instead of raising.
- [x] #2 The sequence that produces the two-row state is reproduced by a test and can no longer occur.
- [x] #3 An unsupported version fails at boot with an actionable message naming the file, not a raw traceback.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read _initialize_schema's version check and _migrate_from_v1_to_v2 to confirm the migration is one atomic transaction.
2. Write failing tests: current+stale rows must normalize; unknown future version must raise a message naming the path and versions.
3. Attempt the two Step 4 trigger-reproduction sequences (reopen after migration commits; interrupt the migration's DELETE+INSERT swap) and record whether either yields [1, 2].
4. Fix _initialize_schema: check _CURRENT_SCHEMA_VERSION in versions first (delete stale rows if so), else versions == [1] migrates, else raise with the actionable message.
5. Run the targeted DB test files and close out the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed _initialize_schema (tldw_chatbook/DB/Subscriptions_DB.py ~L743-761): the version check now tests _CURRENT_SCHEMA_VERSION in versions first -- if present (alone or beside stale/unknown rows in either direction), it deletes every other row in a small transaction and continues instead of raising; only a bare [1] still takes the existing v1-to-v2 migration path; anything else (empty table, [0], [3], [1, 3], etc.) raises SubscriptionError naming the db_path_str and the found/supported versions. TDD: Tests/DB/test_subscriptions_db.py gained 8 tests -- normalization of [1, 2] to [2] (durable across reopen) and of [2, 3] to [2] (current beside a newer unknown row, not just an older one), raises for an unknown version ([3]), an unknown version beside v1 ([1, 3], so it does not silently take the v1 migration path), an empty schema_version table, and [0], plus two Step-4 trigger-reproduction attempts. Findings: because the entire v1-to-v2 migration (rename, rebuild, index creation, and the schema_version DELETE+INSERT) runs inside one SQLite transaction, neither reopening immediately after a committed migration nor interrupting the DELETE+INSERT swap before commit produces [1, 2] through this constructor's own code paths -- an uncommitted interruption rolls back to the prior state, and a completed migration always leaves exactly one row. The two-row state in the live evidence most plausibly came from a different build's unconditional INSERT OR IGNORE running beside an unmigrated v1 row (a cross-build scenario, not a race in this file); the normalization fix defends against that regardless of how it is reached. Verification (fix round, review finding): ran all eight files the brief's Step 5 glob names with the repo venv -- /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/DB/test_subscriptions_db.py Tests/DB/test_subscriptions_db_agent_read_only.py Tests/DB/test_subscriptions_db_briefing_provenance_migration.py Tests/DB/test_subscriptions_db_site_configs.py Tests/DB/test_subscriptions_db_watchlists_agent_search.py Tests/DB/test_subscriptions_db_watchlists_reader_snapshot.py Tests/DB/test_subscriptions_db_watchlists.py Tests/Subscriptions/test_subscriptions_db_connection_lifecycle.py -q -p no:cacheprovider -- 204 passed, 0 failed, no regressions.
<!-- SECTION:NOTES:END -->
