---
id: TASK-32560
title: Test local backup and recovery on Linux over SSH
status: Done
created_date: 2026-09-12 14:57
labels:
- backup-recovery
references:
- https://github.com/rmusser01/tldw_chatbook/pull/2642
documentation:
- Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md
- backlog/decisions/126-complete-local-backup-and-recovery.md
updated_date: 2026-09-12 19:16
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User requested Linux SSH testing after opening PR2642 against dev. Exercise the existing backup implementation on a disposable Linux test installation and record actual results and limitations. Do not promote Linux release capabilities from component-only evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Record the actual Linux OS, architecture, Python/toolchain and filesystem test configuration with the exact source revision.
- [x] #2 Run applicable targeted helper, filesystem, archive/recovery and product checks using disposable data; distinguish passes, concrete failures and unavailable release paths.
- [x] #3 Record commands/results and any narrowly scoped corrections in the PR evidence without changing unrelated data or declaring unsupported Linux qualification.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 1 — Record the authorized Linux host, Python runtime, filesystem and exact public source: Complete.
Stage 2 — Preserve original component and native failures; correct platform operations under TASK32562: Complete.
Stage 3 — Run the full installed backup/restore/open/replacement/retained-copy rollback sequence on the actual SSH host: Complete at ebe86139753b56b7af9b363ea8e3b274723ed868.
ADR: existing backlog/decisions/126-complete-local-backup-and-recovery.md amended by the approved cross-platform correction. Windows verification remains tracked separately in TASK32562.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Read-only SSH access succeeded. Host reports Linux6.12.107+deb13-amd64 x86_64, Python3.13.5, ext4 home storage and tmpfs /tmp. Go is not on PATH. Private test tools/storage inspection in progress. Task ID checked against current dev and primary checkout task files.
Provisioned private Linux root /home/ml-user/Working/chatbook-backup-linux-20260912, Python3.12.8 from existing uv runtime, isolated venv and verified official Go1.26.2 archive SHA990e6b4bbba816dc3ee129eaeaf4b42f17c2800b88a2166c265ac1a200262282. Auto-review blocked local source upload; GitHub API verified repo PUBLIC and exact PR head2220f3bdcd5c8e9f43bd4acac10a07a12ef049d0, then accepted direct public codeload download on the host. No private local source sent. Confirmed unshare --user --map-current-user --net retains UID1000 and native TCP probe returns ENETUNREACH101. Serial existing Go tests/vet, helper protocol, crypto, native-files and archive-reader/writer batches started with private HOME/XDG/config and offline module caches. Linux native identity/publication is explicitly Darwin-only in current production source; these runs must not be mistaken for Linux release qualification.
User corrected the original design: Go should never have been used. Stop further Go qualification work; preserve completed Linux baseline while preparing a Python replacement within backup scope. Actual Go tests/vet passed; normal-UID helper protocol2/2 and crypto37/37 passed. Native-files33 total:5pass/28fail; archive73 total:8pass/65fail. Concrete causes include Darwin-only renameatx_np/F_FULLFSYNC, explicit qualification refusal, plus four unsafe_directory cases needing separate diagnosis. Initial all-setup failures were harness-only: unshare mapped host root UID0 to65534; verified normal UID mapping fixes helper setup without production changes. Normal-UID Python tests retained private HOME/XDG/config and Python network guard/offline Go caches, but no native network-namespace claim. Original and corrected receipts kept separately; source remains2220f3bdc. Go removal does not itself resolve Linux native-filesystem gaps.
Baseline report written Docs/Development/backup-linux-baseline-2026-09-12.md. All normal-UID JUnit/log records retrieved to /private/tmp/task32494-linux-results. No further Go tests after explicit user rejection; fixed prior synthetic age ciphertext preserved with hashes under Tests/Backup_Recovery/fixtures/age_v1 for Python compatibility. Native and archive failures remain concrete Linux gaps, not qualified support. Testing resumes against Python replacement underTASK32561; retain In Progress until revised results/remaining limitations are recorded.
Python correction test preparation: SSH reconnected; private venv remains Python3.12.8, Cryptodome3.23.0, pytest_timeout present. Installed existing test dependency uv0.5.27 beside private venv interpreter (was absent) and seeded isolated package dependencies under testroot/tools/python-uv-cache via testroot/tools/packaging-seed. Seed includes setuptools81.0.0,wheel0.48.0,pydantic2.12.5,loguru0.7.3,psutil7.2.2,pycryptodomex3.23.0. No Go executed or rebuilt. Python final source/tests still pending.
Prepared remote run-tests-python.py by adapting the existing private driver. It selects worker/crypto/packaging only, removes Go paths/settings, asserts Go is absent from test PATH, uses explicitly installed pytest plugins and the private uv offline cache, records worker/transport hashes, and preserves timestamped artifacts/runs. No test run yet; awaits reviewed/public Python PR source.
Python firstattempt98b5d0b24 source: worker63/crypto42/packaging11 allSETUPERROR, zero executedassertions/skips; artifacts-python-20260912-091533 retained. Rootprivatepaths guard refusedshared_writable_parent. statproved newlycreatedruns-python andtimestampgroup were0775, whereas case/tmpwere0700 (remoteumask0002 plus pathlibparents defaults). Correcteddrivercreates/chmods onlyownrunsroot0700 andnewtimestampgroup0700 beforecases; olddriverretained asrun-tests-python-before-private-parents.py. No applicationcode/qualificationoverride. Prerequisiteworkerbatch rerun next.
Final public revision 98b5d0b24c44def024f87ab54af61cb6b01b1511: worker 63, crypto 42, packaging 11 passed, no skips. Linux 6.12.107+deb13-amd64, x86_64, Python 3.12.8, Cryptodome 3.23.0, ext4; Go absent from PATH. Remote receipts artifacts-python-20260912-091812 and -091853 copied to /private/tmp/task32495-linux-success. Original Linux failures and first Python harness setup failure are retained. Only harness-owned parent permissions changed to 0700. PR2642 updated with final evidence and Linux limitations. See Docs/Development/backup-python-verification-2026-09-12.md and the linked baseline; lessons-testing-evidence.md records the private-parent incident.
Reopened after user correctly rejected component-only Linux completion. User requires actual full backup/restore and verification on Linux, macOS and Windows. Native Linux work and Windows Actions tests are now explicitly authorized. Previous 116-pass result covers only encryption/package components.
Latest actual SSH regression uses exact public ebe86139753b56b7af9b363ea8e3b274723ed868, codeloadSHA2563d4fad9d513510ddba9f81cc8577fc0aa9f4d02dc8ff06841bba8cdc6dd9e161. Linux Python3.12.8/ext4, private synthetic data, network guard/NullKeyring, Go absent PATH. Native49passed1.97s; three F9 modes passed129.70s; two-profile roundtrip passed66.65s. Combined replacement/later rollback still running. Artifacts retained under artifacts-platform-private-ebe86139753b56b7af9b363ea8e3b274723ed868-20260912-120739.
Latest combined replacement/retained-copy rollback completed successfully:1passed284.56s. All latest Linux batches completed with0failures/errors/skips:49native1.97s;3F9modes129.70s;two-profile1case66.65s;combinedrollback1case284.56s. Source ebe86139753b56b7af9b363ea8e3b274723ed868; existing private artifacts retained on host. Linux full product verification is now complete, while cross-platform TASK32562 remains In Progress for Windows.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed actual Linux SSH testing of full backup and recovery at ebe86139753b56b7af9b363ea8e3b274723ed868: 49 native tests, all three installed F9 create/restore/open modes, two-profile capture/restore/fresh application open, and combined replacement/retained-copy rollback all passed without skips or failures. Runtime: Python3.12.8, Linux6.12.107+deb13-amd64 x86_64, local ext4, Go absent from PATH. Private synthetic fixtures and raw logs remain on the authorized host. Exact source and results are in Docs/Development/backup-cross-platform-verification-2026-09-12.md and PR2642. This result supersedes the withdrawn component-only completion; Windows completion remains in TASK32562.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

## Renumbering provenance

Renumbered from TASK-32494 to TASK-32560 on 2026-09-13 to resolve the collision introduced when PR2642 merged dev in `d914a76a21b10cf687eeed5d2bee619bc6b4bb6e`. The already-landed “Cross-platform Console reply speech playback” task retains TASK-32494; dev assigned that ID in `97041ab370402ba22e75ba214bab75075be11dfb` before this backup record first appeared in `c19068652a38f2407ca885cc8c8c44d9777b4a18`. This follows the landed-task rule in `backlog/docs/lessons-backlog-hygiene.md`.

The fresh sweep covered all reachable task filenames across both local repositories' refs and 23 live worktrees (maximum 32559), then confirmed no TASK-32560/32561/32562 content references across 179 and 224 unique ref tips or live worktrees. Backup-owned semantic references follow the new IDs. Historical receipt paths retain their original names.
