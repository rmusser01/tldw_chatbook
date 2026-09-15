# Recovery restart preserves launching package — TASK-32562

Implemented and frozen for independent review. No commit or push.

The recovery restart used python -c with an inherited working directory. Python prepended that directory ahead of the trusted package root already supplied by _launch_environment. A repository checkout or other same-named package in that directory could therefore replace the recovery implementation and its asset/config-owner discovery. The existing isolated profile launcher already uses -P. This correction adds the same -P flag to recovery_restart.restart, before -c, on both platform branches. No environment filtering, working directory, path hint, process-release, or UI entry semantics changed. Existing ADR126 recovery restart scope is retained.

Native regression first ran the trusted actual recovery_restart from a fresh python -P interpreter, with a harmless competing tldw_chatbook package in cwd. Only the recovery UI entry was replaced by a fixed import-origin print; real restart environment selection and native exec/CreateProcess remained active. Before the product change the second interpreter imported the cwd shadow: RED 1 failed in 0.87s, /private/tmp/uat-recovery-restart-safe-path-red.log. Afterward the same test resolves the exact launching package. This case runs whichever native branch is present; macOS POSIX execution was verified locally, while native Windows acceptance remains for its runner.

All existing restart/handoff and Windows argument-contract cases plus the new real subprocess probe passed: 15 passed in 67.90s, /private/tmp/uat-recovery-restart-safe-path-green.log. Coverage includes real normal-app handoff, actual fresh recovery UI exec, unsuccessful shutdown, busy/canceled/unsupported handoff and pending confirmation. No deadlines/assertions were weakened. There was one existing requests dependency warning.

Ruff clean (empty JSON), git diff --check clean. Bandit: product zero findings; baseline and current both have two B108 and one B404 test findings; pytest B101 assertion findings increase from 17 to 19 for the two added real subprocess outcome assertions. No new non-assert finding. Native subprocess annotations identify the fixed interpreter, harmless fixed probe and separately passed fixture paths. Receipts: /private/tmp/uat-recovery-restart-safe-path-{ruff.json,bandit.json,bandit-baseline.json,hashes.json}.

Frozen files:

- tldw_chatbook/Backup_Recovery/recovery_restart.py — c07463a486e3f898c0c08ff038dd04115935af29cb335f26dc399eac0efe7907
- Tests/Backup_Recovery/test_recovery_restart.py — c0bf4e963b91570dc7c6c4a9f350838f0679a88dd4354b654a374ee95071fdfe
- Tests/Backup_Recovery/test_recovery_restart_windows.py — 13079954c1c98d00a7dc19521b3d0eb5c775b1f9ea95c91f26de0e4599d58cad

Independent review pending; parent owns Backlog evidence update and integration. No original keyboard fixture or installed artifact was modified.
