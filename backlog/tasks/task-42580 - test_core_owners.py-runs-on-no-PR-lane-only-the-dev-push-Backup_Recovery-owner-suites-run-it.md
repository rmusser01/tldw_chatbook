---
id: TASK-42580
title: "test_core_owners.py runs on no PR lane; only the dev-push Backup_Recovery owner suites run it"
status: To Do
assignee: []
created_date: 2026-11-26 15:58
updated_date: 2026-11-26 16:40
labels:
- follow-up
- tests
- ci-lanes
- backup-recovery
priority: low
references:
- Tests/Backup_Recovery/test_core_owners.py
- .github/workflows/derived-artifacts.yml
- .github/workflows/plugin-qualification.yml
- tldw_chatbook/DB/recovery_core.py
- tldw_chatbook/DB/recovery_core_schema.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Backup_Recovery/test_core_owners.py` pins the core stores' backup and restore schema policy. It includes `test_schema_policy_matches_installed_store`, which every ChaChaNotes schema step must update (the migration README's step 10 points at it). No PR lane runs it. Only the "Backup_Recovery owner suites (push to dev)" job runs it. That job runs only after a merge: on a dev push while `vars.CI_TIERED_ENABLED != 'true'`, and on the nightly dispatch. `plugin-qualification.yml` lists the file only as a path trigger, because `Tests/Plugins/test_recovery.py` imports its helpers; that workflow does not run it. Other laned tests already catch a schema step that misses its version bump, migration step or restore-sandbox admission: `test_prompt_mixin_restore.py` in architecture-guards and `test_chachanotes_v77_agent_chat_starts_migration.py` in dreams-tracking. Only the exact policy shape is pinned by this file alone: the versions tuple, the lineage's length and tail, and the step list. A drift there merges green on its PR and shows only on dev.

Green on dev 6e41919880 (2026-11-26): the file passed 66/66 at `-n 4`, run together with `test_publication_crashes.py` (115 passed).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The file runs on a PR lane, with no node listed twice across PR steps
- [ ] #2 The file is timed serially with that lane's flags, and the time and load are recorded in the task notes
- [ ] #3 Any node red on that lane is fixed (a product fix with red/green, or a test repair that keeps every assertion)
<!-- AC:END -->
