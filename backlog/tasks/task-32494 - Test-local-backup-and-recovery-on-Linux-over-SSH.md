---
id: TASK-32494
title: Test local backup and recovery on Linux over SSH
status: In Progress
created_date: 2026-09-12 14:57
labels:
- backup-recovery
references:
- https://github.com/rmusser01/tldw_chatbook/pull/2642
documentation:
- Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md
- backlog/decisions/126-complete-local-backup-and-recovery.md
updated_date: 2026-09-12 16:36
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
Stage 1 — Private Linux host/tool/runtime inspection: Complete.
Stage 2 — Targeted original helper/filesystem/archive baseline: Complete, with concrete failures preserved.
Stage 3 — Final Python worker/transport/package tests and PR evidence: Complete. Full native recovery unavailable under existing Linux capability checks.
ADR required: no new ADR. Existing backlog/decisions/126-complete-local-backup-and-recovery.md applies; no native platform contract changed.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Read-only SSH access succeeded. Host reports Linux6.12.107+deb13-amd64 x86_64, Python3.13.5, ext4 home storage and tmpfs /tmp. Go is not on PATH. Private test tools/storage inspection in progress. Task ID checked against current dev and primary checkout task files.
Provisioned private Linux root /home/ml-user/Working/chatbook-backup-linux-20260912, Python3.12.8 from existing uv runtime, isolated venv and verified official Go1.26.2 archive SHA990e6b4bbba816dc3ee129eaeaf4b42f17c2800b88a2166c265ac1a200262282. Auto-review blocked local source upload; GitHub API verified repo PUBLIC and exact PR head2220f3bdcd5c8e9f43bd4acac10a07a12ef049d0, then accepted direct public codeload download on the host. No private local source sent. Confirmed unshare --user --map-current-user --net retains UID1000 and native TCP probe returns ENETUNREACH101. Serial existing Go tests/vet, helper protocol, crypto, native-files and archive-reader/writer batches started with private HOME/XDG/config and offline module caches. Linux native identity/publication is explicitly Darwin-only in current production source; these runs must not be mistaken for Linux release qualification.
User corrected the original design: Go should never have been used. Stop further Go qualification work; preserve completed Linux baseline while preparing a Python replacement within backup scope. Actual Go tests/vet passed; normal-UID helper protocol2/2 and crypto37/37 passed. Native-files33 total:5pass/28fail; archive73 total:8pass/65fail. Concrete causes include Darwin-only renameatx_np/F_FULLFSYNC, explicit qualification refusal, plus four unsafe_directory cases needing separate diagnosis. Initial all-setup failures were harness-only: unshare mapped host root UID0 to65534; verified normal UID mapping fixes helper setup without production changes. Normal-UID Python tests retained private HOME/XDG/config and Python network guard/offline Go caches, but no native network-namespace claim. Original and corrected receipts kept separately; source remains2220f3bdc. Go removal does not itself resolve Linux native-filesystem gaps.
Baseline report written Docs/Development/backup-linux-baseline-2026-09-12.md. All normal-UID JUnit/log records retrieved to /private/tmp/task32494-linux-results. No further Go tests after explicit user rejection; fixed prior synthetic age ciphertext preserved with hashes under Tests/Backup_Recovery/fixtures/age_v1 for Python compatibility. Native and archive failures remain concrete Linux gaps, not qualified support. Testing resumes against Python replacement underTASK32495; retain In Progress until revised results/remaining limitations are recorded.
Python correction test preparation: SSH reconnected; private venv remains Python3.12.8, Cryptodome3.23.0, pytest_timeout present. Installed existing test dependency uv0.5.27 beside private venv interpreter (was absent) and seeded isolated package dependencies under testroot/tools/python-uv-cache via testroot/tools/packaging-seed. Seed includes setuptools81.0.0,wheel0.48.0,pydantic2.12.5,loguru0.7.3,psutil7.2.2,pycryptodomex3.23.0. No Go executed or rebuilt. Python final source/tests still pending.
Prepared remote run-tests-python.py by adapting the existing private driver. It selects worker/crypto/packaging only, removes Go paths/settings, asserts Go is absent from test PATH, uses explicitly installed pytest plugins and the private uv offline cache, records worker/transport hashes, and preserves timestamped artifacts/runs. No test run yet; awaits reviewed/public Python PR source.
Python firstattempt98b5d0b24 source: worker63/crypto42/packaging11 allSETUPERROR, zero executedassertions/skips; artifacts-python-20260912-091533 retained. Rootprivatepaths guard refusedshared_writable_parent. statproved newlycreatedruns-python andtimestampgroup were0775, whereas case/tmpwere0700 (remoteumask0002 plus pathlibparents defaults). Correcteddrivercreates/chmods onlyownrunsroot0700 andnewtimestampgroup0700 beforecases; olddriverretained asrun-tests-python-before-private-parents.py. No applicationcode/qualificationoverride. Prerequisiteworkerbatch rerun next.
Final public revision 98b5d0b24c44def024f87ab54af61cb6b01b1511: worker 63, crypto 42, packaging 11 passed, no skips. Linux 6.12.107+deb13-amd64, x86_64, Python 3.12.8, Cryptodome 3.23.0, ext4; Go absent from PATH. Remote receipts artifacts-python-20260912-091812 and -091853 copied to /private/tmp/task32495-linux-success. Original Linux failures and first Python harness setup failure are retained. Only harness-owned parent permissions changed to 0700. PR2642 updated with final evidence and Linux limitations. See Docs/Development/backup-python-verification-2026-09-12.md and the linked baseline; lessons-testing-evidence.md records the private-parent incident.
Reopened after user correctly rejected component-only Linux completion. User requires actual full backup/restore and verification on Linux, macOS and Windows. Native Linux work and Windows Actions tests are now explicitly authorized. Previous 116-pass result covers only encryption/package components.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed requested SSH testing and updated PR2642. All 116 final Python component/package tests passed. Linux native backup/recovery remains unavailable due to existing Darwin-only filesystem operations and qualification; no platform support was added.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
