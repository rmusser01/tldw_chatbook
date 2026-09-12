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
updated_date: 2026-09-12 15:18
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User requested Linux SSH testing after opening PR2642 against dev. Exercise the existing backup implementation on a disposable Linux test installation and record actual results and limitations. Do not promote Linux release capabilities from component-only evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Record the actual Linux OS, architecture, Python/toolchain and filesystem test configuration with the exact source revision.
- [ ] #2 Run applicable targeted helper, filesystem, archive/recovery and product checks using disposable data; distinguish passes, concrete failures and unavailable release paths.
- [ ] #3 Record commands/results and any narrowly scoped corrections in the PR evidence without changing unrelated data or declaring unsupported Linux qualification.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage1 — Inspect host and provision a private test directory/venv and required pinned tools. In Progress.
Stage2 — Run existing helper and native filesystem checks, then applicable archive/recovery tests serially; diagnose concrete failures before changing tests or code. Not Started.
Stage3 — Exercise product paths permitted by actual Linux capabilities, record remaining unavailable paths, publish concise PR evidence. Not Started.
Use existing revision4/ADR126 contracts; targeted checks only. Source PR2642, production77b5fe940, tracking2220f3bdc. No global package changes or real-profile restores.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Read-only SSH access succeeded. Host reports Linux6.12.107+deb13-amd64 x86_64, Python3.13.5, ext4 home storage and tmpfs /tmp. Go is not on PATH. Private test tools/storage inspection in progress. Task ID checked against current dev and primary checkout task files.
Provisioned private Linux root /home/ml-user/Working/chatbook-backup-linux-20260912, Python3.12.8 from existing uv runtime, isolated venv and verified official Go1.26.2 archive SHA990e6b4bbba816dc3ee129eaeaf4b42f17c2800b88a2166c265ac1a200262282. Auto-review blocked local source upload; GitHub API verified repo PUBLIC and exact PR head2220f3bdcd5c8e9f43bd4acac10a07a12ef049d0, then accepted direct public codeload download on the host. No private local source sent. Confirmed unshare --user --map-current-user --net retains UID1000 and native TCP probe returns ENETUNREACH101. Serial existing Go tests/vet, helper protocol, crypto, native-files and archive-reader/writer batches started with private HOME/XDG/config and offline module caches. Linux native identity/publication is explicitly Darwin-only in current production source; these runs must not be mistaken for Linux release qualification.
User corrected the original design: Go should never have been used. Stop further Go qualification work; preserve completed Linux baseline while preparing a Python replacement within backup scope. Actual Go tests/vet passed; normal-UID helper protocol2/2 and crypto37/37 passed. Native-files33 total:5pass/28fail; archive73 total:8pass/65fail. Concrete causes include Darwin-only renameatx_np/F_FULLFSYNC, explicit qualification refusal, plus four unsafe_directory cases needing separate diagnosis. Initial all-setup failures were harness-only: unshare mapped host root UID0 to65534; verified normal UID mapping fixes helper setup without production changes. Normal-UID Python tests retained private HOME/XDG/config and Python network guard/offline Go caches, but no native network-namespace claim. Original and corrected receipts kept separately; source remains2220f3bdc. Go removal does not itself resolve Linux native-filesystem gaps.
Baseline report written Docs/Development/backup-linux-baseline-2026-09-12.md. All normal-UID JUnit/log records retrieved to /private/tmp/task32494-linux-results. No further Go tests after explicit user rejection; fixed prior synthetic age ciphertext preserved with hashes under Tests/Backup_Recovery/fixtures/age_v1 for Python compatibility. Native and archive failures remain concrete Linux gaps, not qualified support. Testing resumes against Python replacement underTASK32495; retain In Progress until revised results/remaining limitations are recorded.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
