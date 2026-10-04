# Console performance fixes implementation plan

> For agentic workers: use systematic debugging and test-driven development for each independent fix; integrate and request a whole-branch review.

**Goal:** Remove measured Console pauses and native refusals, verify complete captured conversations on all three platforms, and create one PR against dev.
**Architecture:** Preserve existing storage admission and owned-operation boundaries. Deduplicate layout, batch finite reads, reduce refresh fan-out with fenced snapshots, and qualify native Windows reuse with differential evidence.
**Tech stack:** Python 3.12+, Textual 8, SQLite, Windows NTFS facade, targeted pytest/native CI.
**Spec:** Docs/superpowers/specs/2026-10-04-console-performance-fixes-design.md
**ADR required:** yes for Windows reuse/owner changes and confirmed trace GC contract changes.
**ADR path:** backlog/decisions/126-complete-local-backup-and-recovery.md and backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md.
**Reason:** Evidence reuse and trace revision lifetimes are governed runtime/security contracts. UI batching/layout preserves existing contracts (no new ADR).

## Global constraints
- Keep private path ownership, native qualification, maintenance, epoch/provenance, scope revision, and cancellation lifetime checks.
- Targeted tests only; do not change unrelated dirty main checkout files, raise performance ceilings, or suppress normal timers to pass.
- Use token-backed existing UI geometry; verify actual rendered recovery-row geometry.
- Every task begins with a reproduced failure and ends with passing meaningful checks and concise notes.

## Review focus
- ACL/owner/path changes between reuse observation and counting must run full derivation and retain original refusal.
- Character/profile/session changes while worker reads finish must reject stale publication.
- Cancellation must not close a worker connection still in use.
- Startup GC must not delete the exact admitted revision before provider reservation.
- Repeated recovery state must avoid CSS work while external geometry mutations are corrected.

## TASK-34403: Control layout and complete native performance verification
Files: Widgets/Console/console_control_bar.py; Tests/UI/test_console_recovery_height_idempotence.py; Tests/Performance/test_console_native_pause_probe.py; .github/workflows/console-pause-native-evidence.yml; qa/console-pause-investigation-2026-10-04/.
- [ ] Count stylesheet work for unchanged real control state; run to confirm failure.
- [ ] Add class/inline-constraint equality guard; test recovery visible/hidden geometry and externally changed constraints.
- [ ] Integrate independent fixes, investigate remaining sampled stalls, and add ratchets for complete native sends without provider latency.
- [ ] Run targeted regressions and three-platform native receipts including ordinary timers/GC; record exact source, counts, timing limits, and ownership negatives.
- [ ] Verify live DeepSeek three-turn conversation on the completed source.
- [ ] Review all work, update task evidence, and create/attach the combined PR against dev.

## TASK-34404: Windows native admission and SQLite sidecar ownership
Files: Utils/windows_files.py; Backup_Recovery/storage_admission.py; DB/private_sqlite.py as necessary; focused native and differential tests; ADR-126.
- [ ] Reproduce elevated-runner owner/default-token behavior and NTFS change-time invalidations using native receipts.
- [ ] Write RED tests for safe Windows reuse and owner fix; retain rejection of foreign/shared objects and pause/provenance mutations.
- [ ] Implement minimal native fix and measured safe reuse; preserve exact full-derivation refusal/fallback semantics.
- [ ] Run targeted native oracle tests and report handle count reduction and remaining risks.

## TASK-34405: Character and availability finite read batching
Files: UI/Console_Modules/character_context.py; UI/Console_Modules/workspace.py; relevant focused tests.
- [ ] Reproduce owned connection/helper fan-out with real finite reads; write RED batching checks plus interleaving authority/revision negatives.
- [ ] Batch under one finite owned callback and remove redundant runtime-binding reads with unchanged retirement and freshness boundaries.
- [ ] Verify all existing character scope and workspace publication/cancellation tests, lint, and helper count reduction.

## TASK-34406: UI refresh fan-out and startup trace GC race
Files: UI/Screens/chat_screen.py and Console modules excluding character_context/workspace; Chat/console_trace_service.py and DB trace repositories as necessary; focused tests; ADR-097 for lifetime amendments.
- [ ] Reproduce refresh fan-out and deterministically confirm/rule out TASK-33621.47 startup GC race.
- [ ] Add failing checks for reduced reads and scope-fenced publication; fix confirmed GC revision race with appropriate lifetime protection.
- [ ] Implement minimal shared refresh deduplication and bounded worker reads, preserving fresh post-await snapshots and live status.
- [ ] Verify targeted capture/retry/refresh/GC/maintenance regressions and identify any remaining broader app performance root.

## Execution rulings
The user's explicit instruction is to fix the identified and broader defects and open the PR. Continue authorized implementation without a new design/execution permission loop. Independent subsystem work is delegated under dispatching-parallel-agents; integration, layout, native evidence and final PR remain with the primary agent. Targeted native CI is part of verification. Preserve main checkout and all other task edits.
