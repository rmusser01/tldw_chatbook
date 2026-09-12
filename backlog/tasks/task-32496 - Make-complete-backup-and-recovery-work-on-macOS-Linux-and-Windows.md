---
id: TASK-32496
title: Make complete backup and recovery work on macOS Linux and Windows
status: In Progress
created_date: 2026-09-12 16:36
labels:
- backup-recovery
references:
- https://github.com/rmusser01/tldw_chatbook/pull/2642
updated_date: 2026-09-12 17:31
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Correct the backup implementation's platform-specific filesystem and release assumptions so the existing complete backup/restore/replacement/rollback behavior works on all three application platforms. User explicitly requires actual Linux testing and authorizes GitHub Actions Windows testing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Existing complete backup, isolated restore, replacement and retained-copy rollback have working platform operations on macOS, Linux and Windows without Go.
- [ ] #2 Actual installed application workflows create archives and restore/verify synthetic data on the supplied Linux host and native Windows GitHub runner; macOS covering regression tests pass. No component-only result is represented as product success.
- [ ] #3 Preserve private storage, no-overwrite publication, cooperative locking, authentication, cancellation and interruption recovery; update PR2642 with exact tests, limitations and source identities.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce actual Linux/Windows product failures and map platform dependencies.
2. Implement platform-native filesystem/locking/private-path support with targeted regressions, preserving existing recovery contracts.
3. Run complete installed backup/restore/replacement/rollback workflows on Linux over SSH, Windows GitHub Actions and macOS; fix observed failures.
4. Independent review, scoped security/static checks, update ADR126 correction and PR evidence. No completion claim until real product verification.
ADR: update existing backlog/decisions/126-complete-local-backup-and-recovery.md for cross-platform platform contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Linux product baseline at 98b5 failed actual installed F9 with missing fcntl.F_FULLFSYNC, not merely component refusal. Reviewed Linux fix commit5b7e60d96: 39 native tests passed on supplied host. First product run then found installed package directories0775 from remoteumask0002; corrected only private driver umask077 with failed receipts preserved. Actual mounted F9 archive creation/readback/resumed writes now passed23.94s; full F9/roundtrip/rollback still running. Windows Actions run34706507112 atce115673e executed25cases:20passed,5failed,0skips with two installed package hash receipts. Native WinError5 ACL hardening and WinError87 directory rename under correction, plus requiredSYSTEMROOT subprocess environment and missing adapter integration. Reviewer additionally found inherited-only public ACL and query-only directory barrier defects; both assigned to native implementer. No platform completion claim.
Linux actual product evidence for source5b7e60d96: installed F9 backup/restore/open all3 modes passed, combined replacement and later rollback passed, final two-profile capture/restore/open passed after removing group/other-write only from disposable codeload source directories. Raw fixtures/logs remain private on Linux host; automatic review rejected unnecessary bulk transfer. macOS facade regression:81native/crypto tests and3actual installed F9 modes passed. Private SQLite regression3failures reproduced unchanged on clean ce115673e baseline; current324passed. Windows run34706507112 actual25cases20pass5fail; native round1 fixes and remaining facade/path/SQLite reader integration now under review. Work remains In Progress; no completion based on component-only tests.
Integration review: corrected missed config-adapter descriptor APIs, facade callable/stat identity representation, all serialized relative archive/tree/credential paths including nested inventory recursion. New portability regression RED thenGREEN; capture/release/nativeplatform66passed4.17s. macOS actual combinedF9replacement+laterrollback1passed190.94s. Inventory suite98passed3old failures reproduced clean ce115673e; SQLite324passed3old failures likewise reproduced. Native Windows round1 source review resolved ACL exposure, real directory flush, rename buffer API size; actual execution pending integrated commit. Windows held-descriptor SQLite snapshot preserves fd binding and query-only using existing576MiB ceiling (transient up to~1152MiB), POSIX route unchanged; reviewer failurecleanup fixes underway. No new Ruff/Bandit findings in integrated scope relative to baseline.
Exact integrated7d101919099780c6f4236c05845e8d86513a87d9 verification complete on Linux/macOS: Linux47native0skip2.02s,3actualF9modes126.96s,2-profilecapture/restore/open66.04s, combinedreplacement/laterrollback281.09s; macOS5actualproductcases373.46s. Windows34707982570:39shared-fixture setup errors beforeanytestbody (untrusted ancestorowner), not native/productqualification. Nextdiagrun will isolate nativecases fromappconftest, preserveactualproductfixtures, record anonymous principal/ancestor roles and permissions (automaticreview rejected exactSID/path export; safer allowlistused). Necessary Windows recovery-generation lock path corrected,6focused+1existingindexer tests pass; native verification pending. PRbodyupdated honestly InProgress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
