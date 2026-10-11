# PR2563: native goal runs on current dev

The user authorizes review, fixes, rebase and protected sequential merge of PR2563. The existing accepted native goal design is binding; this integration adds no later milestones. Original source: 50eec6e8e4ec24095a404b1e83dcf757ff17623b. Initial dev: a3c47c6c72d885db2af0971515ddcdcb6cee78fa.

Spec: original Docs/superpowers/specs/2026-09-08-gnhf-inspired-goal-runs-design.md, also preserved in the wave2 evidence directory. Current-dev audit and exact source manifest are in that directory.

ADR required: yes, existing native goal decision with a current-dev integration amendment.
ADR path: backlog/decisions/227-native-console-goal-runs.md (verify global ID ownership before restoration; preserve original provenance if renumbering).
Reason: goal lifecycle/storage is already designed; current schema, backup, custody, allowance and filesystem boundaries require the original decision to describe the actual integration. Governing current ADRs include 069, 131, 134, 135, 150, 158, 208 and 219.

## Global constraints

- Work only in this managed worktree. Do not mutate the primary dirty checkout or any other worktree, stash or shared Git configuration. Never push or merge from the implementation task.
- Use current dev as authority for all shared modules. Carry only goal-specific changes between 77bc58dc171c3dcd2178f4433d19a6ceb59b1e7b and 20bb97cdded8f45cc77fd9848d232e5dc951b24c plus directly relevant initial goal governance. Exclude unrelated prerequisite and scheduling changes.
- No full test sweep or cloud/provider spending. Run targeted meaningful verification using primary .venv Python3.12 with PYTHONPATH set to this worktree and isolated test/profile directories. Historical source evidence remains historical.
- Read actual task files and relevant ADRs/lessons before implementation. Update Backlog status/plan before implementation, using the installed CLI. Preserve existing notes; do not mark unverified acceptance criteria Done. Coordinate any fresh task/ADR ID with root because later queued PRs contain overlapping IDs.
- Preserve validation, permission, resource, recovery, privacy, accounting, runtime custody and shutdown boundaries. No raised size/style/test ratchets or foreign resource closure.
- Every visible control must work through current Console owners and current token-backed design language. Read design-language.md and component-patterns.md before UI changes.

## Task 1: Integrate and qualify native goal runs

This is one cohesive integration task: its storage, runtime and UI contracts are coupled, so perform the following phases sequentially in this worktree, committing reviewable phases without creating competing runtime owners.

1. Inspect the original goal tasks/spec/ADR and current connected code. Restore only the relevant source governance; amend its current-dev integration details before code. Record current plans/ADR checks in the actual goal Backlog tasks. Report proposed fresh ID allocations to root before minting.
2. Adapt goal storage to the actual next AgentRuns schema (initially23 ->24) without relabeling predecessor migrations. Extend exact backup/recovery catalog and domain owners and installed SQL/audit-file policy. Preserve all existing data and supported predecessor catalogs. Test populated upgrade/reopen, archive/staging, rollback/cancellation and unknown/hybrid refusal without automatic replay.
3. Extend current automatic-work models/attempts and policy for goal iteration while retaining progress-only wakes, message acknowledgement, automatic chat starts, immutable allowance roots, conservative unknown usage and manual reserves. Integrate one runtime startup audit and current custody/capacity/decision owners.
4. Port goal service, evidence/verifier, prepared history, launch/provisioning and finite lifetime admission. Fix the confirmed unstarted Pause/Resume trap with safe no-accepted-attempt guards; retain strict checkpoint and uncertainty rules after acceptance. Keep selected source reads under current admitted-root/workspace executor boundaries and primary-only writes; sources never grant instructions or permissions.
5. Add goal setup/history/status, canonical Settings and composer /goal through current command/draft routes, using design tokens and current CSS/size governance. Preserve cancellation/refusal drafts and revalidation after waits.
6. Run goal-specific source suites and affected current migration/backup, automatic-work/chat-start, runtime custody, human decisions, hooks, provider, permission, admitted-root/executor and UI governance regressions. Add actual mounted Pause-before-first-capacity -> Resume -> first dispatch and real isolated CLI nonzero -> correction -> zero plus process-restart/no-duplicate evidence. Fix actual integration defects with tests that demonstrate behavior.
7. Self-review the complete resulting diff, scoped Ruff/format/diff checks. Update tasks/implementation notes with exact current receipts and explicit remaining external qualification; do not promote old failed live-model trials. Commit only task-owned files and return the report path, phase commits and targeted receipts. Root will perform independent review, latest-base rebase, GitHub checks and merge.

Acceptance: the original first-release behavior works on current owners; all identified review defects are addressed; current affected boundaries retain their behavior; new evidence validates actual integrated code. No source stack duplication or unverified completion claims.
