---
id: TASK-40096
title: Remove repeated staging and verification work from isolated Eval restore
status: Done
created_date: 2026-11-09 19:37
labels:
- backup-recovery
- ci
- performance
references:
- backlog/decisions/125-lock-safe-private-sqlite-validation.md
- backlog/decisions/126-complete-local-backup-and-recovery.md
documentation:
- Docs/Architecture/backup-recovery.md
updated_date: 2026-11-09 21:12
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #3345 head 9332b96 timed out in the original Windows retained-Eval rebackup's first isolated restore child. Retained stacks show forward progress through payload extraction, metadata application, and installed SQLite validation. Preserve the prior TASK-38900 schema-policy correction and remove repeated work in the existing restore path so the original 45-second child gate can pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The existing isolated restore completes without repeated staging-volume resolution or duplicate post-worker verification, with the same per-chunk capacity and identity checks, physical durability, and Settings replacement safeguards.
- [x] #2 The original retained-Eval rebackup and relevant capacity, installed-drift, and durability controls pass within their original limits. Fresh original software UAT passes.
- [x] #3 Independent exact-commit review and the published head’s own hosted Windows and required checks verify the repair with the original 45-second restore child limit.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the exact 9332b96 Windows failure, digest-verified stacks and original 45-second child call. Compare connected source with the last passing Windows job and retain the original evidence.
2. Observe the original isolated restore's actual staging-volume resolutions and installed proof passes in a guarded local run. The control must reach real restore work and identify repeated calls before any product edit.
3. Keep one owned staging-volume resolution for the unchanged extraction loop. Retain every per-chunk named identity and free-space check, the existing buffer and cancellation cadence, digest validation, flushes and DACL operations.
4. Keep the final installed verification after private validation for all routes. Keep the preceding verification around real replacement Settings preservation. Remove only the pass that repeats the same proof with no intervening work on the isolated route.
5. Verify the same control, original retained Eval journey, relevant capacity, installed drift, and durability coverage, and the original browser and mounted UAT. Check exact-head review and actual hosted Windows at the original limits before closing the task.
ADR required: no. ADR path: N/A. Reason: removes redundant work within existing custody and validation boundaries governed by ADR-125 and ADR-126.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Windows Eval on 9332b96 passed 137 cases, retained one platform FIFO skip, and failed when the original isolated restore child reached its 45-second limit. The child stacks show forward progress through staging writes, metadata and installed SQLite validation. They identify the phases, but do not measure elapsed time within them.

Staging now retains the first extraction-root capacity binding for every member. Each chunk still checks the named identity and free space before writing. File creation, buffer sizes, cancellation, checksums, DACLs and flushes are unchanged. Installed validation now drops one repeated check on the isolated route. It keeps the checks before and after metadata, and all replacement Settings checks.

The instrumented original local restore child observed 70 staging-volume resolutions and five installed verification passes before the change, then one resolution and four passes after it. The original producer, isolated restore, ordinary reopen and rebackup completed with the repair. The uninstrumented local Eval selection passed all 139 cases. Hosted Windows Eval passed on e1facb0 in job 129417643009. The original rebackup and all 26 policy cases ran, with the original 45-second restore child limit.

The supplementary run passed 32 cases and retained one inherited setup failure. The same directory fixture fails at the Config manifest relation guard on the exact parent, before installed validation. The fixture is outside the bounded PR recovery rosters, and the stricter publication guard is unchanged.

Static checks passed, including the 207 size and globals cases, the diagnostic inventory and 34 boot checks. Fresh original software UAT passed all 15 browser cases and the mounted journey. Its audit confirmed all four processes closed, nine matching retained artifacts, ordinary reopen, runtime disposal, gateway shutdown and empty guards. The existing recovery design in ADR-125 and ADR-126 still applies.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
