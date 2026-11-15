---
id: TASK-41118
title: "Personal Context backup tests and test_activation_network_sync.py run on no PR lane"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- backup
- personal-context
- tests
- lanes
priority: medium
references:
- Tests/Backup_Recovery/test_activation_network_sync.py
- Tests/Backup_Recovery/test_portable_relocation.py
- Tests/Backup_Recovery/test_restore_data_groups.py
- Tests/Personal_Context/test_service.py
- .github/workflows/derived-artifacts.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** Round 14 (TASK-40133..40140, 40711) put the Personal Context backup, restore, re-encryption and relink behaviour under tests that no PR job runs, so a regression lands on dev unseen.

**Evidence.** None of these files is named in `.github/workflows/*.yml` or the UI census at the stack top: `Tests/Backup_Recovery/test_activation_network_sync.py`, `test_portable_relocation.py`, `test_restore_data_groups.py`, `test_credentials.py`, `test_data_groups.py`, `test_inert_extraction.py`, `Tests/Personal_Context/test_repository.py`, `test_runtime_policy.py`, `test_service.py`, `Tests/App/test_personal_context_wiring.py`. `test_detached_layout.py` runs whole only in the push-to-dev owner job; TASK-40138's notes list its new nodes as unlaned on PRs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Personal Context backup and restore tests added in round 14 run on a PR lane or the push-to-dev owner job (never the fast PR contract), chosen per file and recorded with each file's measured time
- [ ] #2 test_activation_network_sync.py runs on such a lane
- [ ] #3 The lane targets are pinned in Tests/CI
<!-- AC:END -->
