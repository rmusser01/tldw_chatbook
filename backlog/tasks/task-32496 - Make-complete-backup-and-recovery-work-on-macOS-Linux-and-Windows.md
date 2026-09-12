---
id: TASK-32496
title: Make complete backup and recovery work on macOS Linux and Windows
status: In Progress
created_date: 2026-09-12 16:36
labels:
- backup-recovery
references:
- https://github.com/rmusser01/tldw_chatbook/pull/2642
updated_date: 2026-09-12 18:25
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
Windows065c5344e run34708741479 schema2 verified9artifacthashes: native26collected25pass1test-callTypeError (secure_private_directory missing requiredcreate/application_owned kwargs),0skip; native rename/rightsreopen/realflush/stat/locks passed. Productcollectionrefused beforetests dueDvolume unrecognized owner. AnonymousACL evidence C systemdriveTrustedInstaller0,UsersSYSTEM0,runnerhomeSYSTEM0,AppData/Local/TempAdmins0 with trusted permissions; testharness will relocate syntheticprivatefixture and byteverifiedtrackedsource to C without producttrustbroaden. CPython3.12.10 mkdir0700 usesOWNER_RIGHTS ACE; adaptermustresolve S-1-3-4 toactualGetSecurityInfoowner, notgloballytrust it. Nativeagentimplementing regression+fix. Linux newgenerationlocking scope7passed8.89s on065c5344e afterinstallingexistingoptionalNumPy2.5.3/ChromaDB1.5.8 onlyinprivatetestvenv; missingdependencyfailurespreserved.
Windows Actions 34709688754 at df281c240 reached actual test bodies: native32passed0skip; product15passed4failed0skip. 10SQLite descriptor and5projection lock cases pass. Two-profile seed failed atomic config overwrite because private_paths used os.rename where replacement intended; WindowsOS.replace already supports explicit replacement. Existing failure-injection regression observed RED after targeting replace, then private_paths switched to os.replace: local private paths66passed2existingplatformskips. Native34next cases include overwrite and high-level configwrite regression. Replacement/later tests refuse inventory; mountedreviewtimeout. Found fixture HOME changed without matching Windows USERPROFILE; fixing per-child both and adding safe inventory diagnostics. Artifact11/12 hashes matched; upload omitted a hidden sanitized app log, so fixing visible sanitized log names and double-backslash path redaction. Work remains In Progress.
Pushed ea917deb027af20150e665c192c264e4db596fc8 and dispatched Windows34710456481 (pending). Actual Linux private-path suite at exact public source archive SHA2561c125e2aa625f0b2019e86437020becbae31927c2e75a83d153abf7de0280dd2: 65passed1failed2skipped4.67s. Failure is existing final-symlink test assuming outside.write_text creates non0600 under driver umask077; refusal itself passed. Fixture now explicitly chmod0644 before attempt, local focused test1passed; next exact public revision will rerun Linux. Preparing test-only native Windows CREATE_NEW_CONSOLE wrapper for existing three F9 plaintext/encrypted/credential roundtrips; identical UI flow, no suspend/driver/qualification mock. Native console support itself remains unverified until Actions.
Windows34710456481/ea917deb: native34passed (overwritefixed), product15passed4failed;11/11 artifactSHA verified. Preciseinventoryblocking33 entries: chat.dictionaries unavailable, persona builtin root unavailable,31missing builtin children; no cross-owner ancestor pairs. Native directory stream API defect confirmed in Microsoft docs: FileStreamInfo returnsERROR_HANDLE_EOF38 when no streams exist; directories normally lack unnamedDATA. Minimaladapterfix accepts only38, stillpropagates5/87/234. Local26passed14native-onlyskips; native40expectednext. Windows seed60s observer expired whileactivelyseeding dictionaryrecords; Windows seedobserver120s anddumpstack55s added. Pytestouter600s accommodates preexistingcombinedchildbounds110+150+180. Application operation/UIdeadlines unchanged. Nativeconsolewrapper reviewed; same FLOW/OPEN and actualWindowspreflight/suspend. macOSplainF9 regression1passed34.45s. NextWindowsrun includes3F9modes.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
