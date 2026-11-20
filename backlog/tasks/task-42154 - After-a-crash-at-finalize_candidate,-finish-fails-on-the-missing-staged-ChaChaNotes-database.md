---
id: TASK-42154
title: "After a crash at finalize_candidate, finish fails on the missing staged ChaChaNotes database"
status: To Do
assignee: []
created_date: 2026-11-20 22:23
updated_date: 2026-11-20 22:23
labels:
- follow-up
- backup
priority: medium
references:
- tldw_chatbook/Backup_Recovery/replacement.py
- tldw_chatbook/Backup_Recovery/publication.py
- tldw_chatbook/Backup_Recovery/journal.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the review of fu/r18-petsdef (round 18, rev3), and confirmed on the branch's code; the Backup_Recovery recovery code there matches dev.

The probe:
- A Replace on a separate config folder, with no pet files (the harness's `[pets-none]` shape without pets: groups conversations, settings and workspaces).
- The process is killed at `publication.finalize_candidate` (`os._exit`), after publication has moved the candidate's artifacts into place.
- A fresh process then runs `replacement.recover_replacement(..., action=...)`.

What happens:
- `rollback` succeeds (`rolled_back`), and `startup_permission` is `(True, 'startup_allowed')`.
- `finish` fails with `FileNotFoundError: 'tldw_chatbook_ChaChaNotes.db'`: `recover_replacement` → `_acquired_source` → `publication._descriptor` → `journal.observe_artifact`. The descriptor check expects every candidate artifact to still sit in the stage, but publication has already moved them. Startup then stays refused (`recovery_pending`) until the user picks `rollback`.

`finish` is offered as a recovery action but cannot complete after this crash point.

The probe script is `scratchpad/fu/pd-crash-probe.py` in the round's scratch folder; run it through `run.sh` with `finish` or `rollback`.

Priority medium: the app refuses to start until rollback is chosen, though rollback recovers it and nothing is lost.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test kills a Replace at finalize_candidate (after publication) on a separate config folder and on the default one, then runs finish and rollback in fresh processes; finish is red before the fix
- [ ] #2 finish completes after that crash, or the finish action is not offered when it cannot complete, and the notes say which and why
- [ ] #3 After each recovery the app starts (startup permission and admission), and rollback keeps working
- [ ] #4 The test runs on a PR or dev-push lane, with its time recorded
<!-- AC:END -->
