---
id: TASK-38724
title: "Participant module state is reached through an accessor API, not private names"
status: To Do
created_date: 2026-11-04 02:52
labels:
- follow-up
- architecture-review-2026-10-25
- backup-recovery
priority: low
references:
- Tests/Architecture/test_participants_public_names.py
- tldw_chatbook/Backup_Recovery/raw_participants.py
dependencies:
- TASK-36948.2
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-36948.2 gave every participants helper read from outside Backup_Recovery a public name. It left module state private and listed it in the guard's STATE_ALLOWLIST: raw_participants `_states`, `_local`, `_unlink` and `_CONSOLE_PERMISSION_LOAD_GUARDS`, mcp_source `_BINDINGS`, persona_visual `_repositories`, and the rag_definition singleton's `participant._condition`. Production code outside Backup_Recovery still reads these by private name. Tests also read `participants._repository_types` and `_installed_repositories`, which the guard does not scan.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Production code outside Backup_Recovery reads participant module state only through public accessors, and STATE_ALLOWLIST is empty and removed
- [ ] #2 Tests that read private participant state use the same accessors, or the task notes say why a test must reach in
- [ ] #3 The participants guard and the Backup_Recovery suites show no new failures against dev
<!-- AC:END -->
