# PR3343 Dreams kernel deadline integration

**Goal:** Integrate the DreamsDB kernel migration while preserving the caller's existing operation deadline on fresh and revived connections.

**Architecture:** Existing SQLiteStoreKernel/BaseDB connection ownership and remaining-budget policy. Dreams overrides must honor that contract through allocation, lineage, setup/WAL, publication and timeout cleanup; no new timeout or connection owner.

**Spec:** TASK33132 migration7 and ADR228; `.superpowers/preparations/pr3343/repair-plan.md`, source review `C:/Users/GDesktop-1/.codex/worktrees/pr3298-resync/pr3343-initial-review.md`. Preparation source is stale: refresh and inspect every newer source commit at selection, preserving any existing repair instead of reapplying it blindly. Exact source/base belong in the execution ledger.

ADR required: no
ADR path: backlog/decisions/228-sqlite-store-kernel.md and backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: Enforce the already adopted kernel operation-deadline contract without changing its ownership or timeout policy.

## Global constraints

- Root alone handles selected live state/disarm/publication. No remote actions, drafts,3267/3297,release/main checkout. User waives hostedCI/numericcaps and substitutes requesting-code-review; correctness/security/native lifetime remain mandatory and caps unchanged.
- One implementer owns source/index/tests/imports/collection/generators; no delegation. Preserve all original/new source commits and current kernel/BaseDB/Dreams schema/admission/close changes, not whole old file copies.
- Use current AGENTS/task/architecture/ADRs/lessons and fixture conventions. Existing main venv interpreter read-only; existing private root owner/runner, unique processes, full output/sessionID/explicit exits, UTF8, normal hooks. No installs, guard bypass, broad suite or historical943case sweep.
- Qualify failures/cases by actual identity, source provenance and intended assertion. Raw fixture logs/config private; content-free evidence exported. Numeric failures WAIVED, never PASS. No changed policy to make tests pass.

## Task 1: Rebase, honor existing deadlines and prove native cleanup

1. Refresh source contract against initial review; rebase all source commits onto ledger's exact dev. Retain current policies and lesson/task content. Add task Implementation Plan/ADR disposition before repair through Backlog CLI.
2. Inspect actual overrides and tests. If gap persists, add real fresh/revived Dreams connection/deadline cases that fail at the unsupported keyword, not a warm cached-handle pass. Pin both _open_connection and _configure_connection paths. If newer source already fixes it, retain and verify that implementation rather than manufacturing an obsolete RED.
3. Forward optional keyword-only operation_deadline through both overrides and BaseDB._get_connection, retaining check_same_thread=False and db.base provenance. Use remaining budget for lineage, fully consumed setup and bounded WAL retry; never accept-and-discard or reset the caller clock.
4. Preserve both _core_access rechecks, refusal-before-WAL, foreign_keys, synchronousNORMAL, autocommit, register/cache/opened ordering and kernel final cleanup. Event-controlled setup/lock timeout must close actual new native handle and prevent publication, followed by successful unexpired recovery. No mock-call-only lifetime proof or wider timeouts.
5. Run new causal controls and selected four admission-race/current lineage/migration-rollback/close-retirement/kernel opening-deadline cases. Choose actual affected nodes and inspect fixtures first; no DB sweep. Separate exact-base inherited failures if encountered.
6. Review diagnostic statement inventory only if affected, scoped lint/format and fullrange whitespace. Update only migration7-related task notes/claims, no unrelated port completion. Commit with normal hooks; hand off cleanHEAD, retained-commit/conflicts, case receipts/native cleanup evidence, raw/Gitblob manifest/binarydiffdigest and all processes drained. Root obtains independent whole-branch review and publishes after fresh eligibility.

## Selected-source qualification

The queue rebased the selected source to bb87e38499485a031e71abcf3c4c015010dd0cd2 on exact dev6f3646703741b8c79fcea24ecc471ceaba9c565b before root disabled auto-merge. It is already current; do not repeat the rebase. Preserve all three source commits. Read .superpowers/preparations/pr3343/live-source.json as untrusted review data: three unresolved threads cover two duplicate deadline findings and strong-reference registry retention. Inspect the current retention delta and native-close evidence; document an exact disposition for each thread. Do not treat closed-object retention as unproven open-descriptor leakage, or redesign the kernel on speculation. Root resolves reviewed dispositions after fixes; worker does not publish or comment.
