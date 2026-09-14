# Later rollback recovery-mode handoff

TASK-32562; bounded implementation against d58650f42ebc1ee6a0f3b3aa9220ce14e312d1e7. No commit/push or actual UAT profile mutation. Frozen six-file hashes: `/private/tmp/uat-later-handoff-hashes.json`.

## Behavior

Ordinary Chatbook now offers “Continue in recovery mode” in the later-rollback form and explains that Chatbook closes before selecting the copy and reviewing again. Review is disabled and its handler guarded; Start also refuses an otherwise valid stale plan while hosted by the ordinary application. Handoff clears passwords, the previous plan/availability/consequence confirmation, and credential review state, then uses existing guarded ordinary shutdown.

`RecoveryRestart` adds only optional strict-bool `recovery_copies=False`. The existing fixed `-P` fresh-process launch supplies exactly `inspect` or `copies`; child code validates that value. The existing request forwards only the nonsecret hint and target. Copies mode opens the existing copies view and prefills the current target. It transfers no copy selection, passwords, plan, acknowledgments, fingerprint, or authority. Fresh copy selection and review remain required. Default inspect behavior, filtered environment, native process behavior, guarded shutdown, and all service/planner/fingerprint/native admission checks remain unchanged.

Files:
- `tldw_chatbook/Backup_Recovery/recovery_restart.py`
- `tldw_chatbook/UI/Screens/backup_restore_screen.py`
- `tldw_chatbook/app.py` (only request signature and request construction)
- `Tests/Backup_Recovery/test_recovery_restart.py`
- `Tests/Backup_Recovery/test_recovery_restart_windows.py`
- `Tests/Backup_Recovery/test_later_rollback_handoff.py` (new)

## Verification

- RED: 9 failed in 2.03s, `/private/tmp/uat-later-handoff-red.log`. Review/start reached forbidden worker dispatch; button/strict bool absent; child ignored navigation/invalid-mode input.
- Initial combined GREEN: 25 passed in 76.10s, `/private/tmp/uat-later-handoff-green.log`. Includes real mounted ordinary-app shutdown for new later path, existing busy/cancel/unsupported/duplicate guards, original CLI successful/failed-unmount restart, default fresh inspection and safe import origin.
- Final expanded selection: 16 passed in 18.35s, `/private/tmp/uat-later-handoff-final.log`. Includes all 10 new cases, four Windows spawn/exit argument/error cases, and both actual normal-app→real POSIX exec→fresh RecoveryApp modes. New actual child tests prove no normal app/config imports and untouched invalid target bytes. Windows branch execution remains for native Windows CI; local subprocess is macOS.
- Native preserved-log regression uses actual `_Diagnostics`, private files, append, fingerprint, and `execute_rollback` entry. Same owner/path/ID/status/inode with new bytes refuses exact `target_changed` before control access. This intentionally minimal plan proves early preflight, not a complete authenticated rollback.
- Existing later UI lifecycle attempted separately: fixture import fails `raw_source_selection_changed` before UI or rollback execution. Exact HEAD overlays for all three changed product modules reproduce the same failure (0.85s current / 0.88s baseline). Logs `/private/tmp/uat-later-handoff-lifecycle{,-baseline}.log`; overlay `/private/tmp/uat-later-handoff-baseline-3xtu_nkr`. No fixture or guard weakened; no claim of completed later rollback from this test. Root will perform installed acceptance.
- Last product change after green was only root-requested user-facing wording; final static/hash refresh includes it.
- Ruff baseline/current: 490/490, zero new. Bandit baseline/current: 52/66; additions are exactly 14 assertions in the new test, no new product or non-assert finding. Reports `/private/tmp/uat-later-handoff-{baseline,current}-{ruff,bandit}.json`. `git diff --check` clean.

## Causal qualification

This repairs the demonstrated ordinary-writer flow asymmetry without weakening preserved-log validation. Original keyboard target_changed attribution remains qualified because the old in-memory fingerprint was not retained. Prior read-only evidence is `/private/tmp/uat-later-rollback-handoff-review.md`.

## Independent-review finding resolved and installed acceptance

The reviewer found the initial copies list was dropped: initial form events invalidated the revision captured by the on_mount list worker. The rendered-list RED reproduced in 1.10s (`/private/tmp/uat-later-list-red.log`). Suppressing only the initial target event was insufficient because the initial restore-mode Select event was also queued. Bounded ID/boolean observations in `/private/tmp/uat-later-list-posts.log` identified that event; no private input values were recorded.

Final on_mount suppresses only the programmatic initial target prefill and defers the existing `_open_copies` to `call_after_refresh`. No later input invalidation is suppressed. Tests require actual “No local entries.” rendering, then change the target and prove the revision increases and stale plan is cleared. Final focused **14 passed in 2.28s**, `/private/tmp/uat-later-list-final.log`.

The existing actual handoff test now uses the existing `native_package` fixture and `_run(installed_package=native_package)`. The test-only child driver loads only the existing network guard by exact filesystem path via runpy; it does not add the repository to child sys.path or weaken production environment filtering. Both source and child assert exact installed module origins; child checks package/restart/launcher/screen origins, rendered copies result, no selected copy or plan, no ordinary app/config imports, and unchanged target bytes. Deadlines remain 70 seconds per child.

**Final installed acceptance: 2 passed in 26.19s**, `/private/tmp/uat-later-installed-final.log`, including real ordinary app→guarded shutdown→native exec→fresh installed RecoveryApp for both inspect and copies. This supersedes the earlier source-tree-only exec evidence. Wheel SHA256 `b330d2bf9532a9f572f46a121e9fb8f82219cf0989fabd530ea8f831812f2908`; receipt `/private/tmp/uat-later-installed-final/f9-native-package0/native-package.json`. The native fixture verified all 2,866 installed files unchanged; all three frozen touched product hashes independently match the installed artifact. Native platform here is macOS, not Windows.

Final refreshed six-file hashes remain `/private/tmp/uat-later-handoff-hashes.json` (screen c132f53f, restart test fd81f1c6, new test 345239f0). Ruff/Bandit deltas unchanged after the final correction; diff check clean. No further edits planned absent a concrete review finding.
