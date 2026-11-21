---
id: TASK-42154
title: "After a crash at finalize_candidate, finish fails on the missing staged ChaChaNotes database"
status: To Do
assignee: []
created_date: 2026-11-20 22:23
updated_date: 2026-11-20 22:34
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

Recipe (the round's scratch probe will be deleted; this is enough to write AC#1's test):
1. Build the case on `Tests/Backup_Recovery/test_created_owner_rollback.py`'s `_SCENARIO`:
   - `CASES={'control': {'groups': _PET_GROUPS, 'stores': (), 'extra': (), 'files': {}, 'delete': (), 'refusal': None}}`;
   - `BOARD_INSERT=_BOARD_INSERT`;
   - `LOCATION='separate'`, or `'default'` for the default-location variant.
2. Insert the crash point into the scenario's target role, just before `restored=reviewed(service,`:
   ```
   from tldw_chatbook.Backup_Recovery import publication as crash_publication
   crash_publication.finalize_candidate=lambda *args,**kwargs:os._exit(17)
   ```
   `replace()` imports `finalize_candidate` from `publication` inside the function, so the patch takes effect.
3. Run the source role, then the target role, each in a child process with `test_home_citation_retirement._run`'s environment: `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME` and `TLDW_CONFIG_PATH=<root>/config/config.toml` under the role's root, plus `TLDW_TEST_MODE=1`. The target exits with 17, after publication.
4. In a fresh child with the target's environment:
   ```
   pending,_=bootstrap._records(bootstrap.default_bootstrap_root())
   operation=pending[0]['operation_id']
   recover_replacement(operation,control_root=Path.home()/'recovery-control',action=ACTION,rollback_password=b'rollback-test-password',cancel=Event())
   ```
   (from `tldw_chatbook.Backup_Recovery.replacement`).
5. Check `bootstrap.startup_permission(selector, bootstrap.default_bootstrap_root())` afterwards. With `ACTION='finish'` this raises today and startup stays `(False, 'recovery_pending')`; with `ACTION='rollback'` it returns `rolled_back` and startup is `(True, 'startup_allowed')`.

Priority medium: the app refuses to start until rollback is chosen, though rollback recovers it and nothing is lost.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test kills a Replace at finalize_candidate (after publication) on a separate config folder and on the default one, then runs finish and rollback in fresh processes; finish is red before the fix
- [ ] #2 finish completes after that crash, or the finish action is not offered when it cannot complete, and the notes say which and why
- [ ] #3 After each recovery the app starts (startup permission and admission), and rollback keeps working
- [ ] #4 The test runs on a PR or dev-push lane, with its time recorded
<!-- AC:END -->
