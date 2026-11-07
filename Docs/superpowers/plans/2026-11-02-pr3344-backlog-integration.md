# PR3344 task completion-accounting integration

**Goal:** Merge accurate task accounting for the existing scheduling work without claiming unimplemented client support or unavailable live evidence.

**Architecture:** Documentation-only correction against current SchedulingService and authoring UI behavior; no feature implementation, deployment or new runtime boundary.

**Spec:** TASK18940 and current source behavior, `.superpowers/preparations/pr3344/repair-plan.md`, `C:/Users/GDesktop-1/.codex/worktrees/pr3298-resync/pr3344-initial-review.md`. Source changed since preparation; inspect all current contributions before applying any old correction. Selected source/base are recorded in the execution ledger.

ADR required: no
ADR path: N/A
Reason: Correct task claims; preserve existing ADR184's separately deferred approval-escalation scope.

## Global constraints

- Root owns fresh eligibility/disarm/publication. No remote actions, drafts,3267/3297,release/maincheckout work. User approves independent requesting-code-review instead ofQodo and hostedCI/numericwaivers; accurate claims remain mandatory.
- One local implementer owns task/source/index changes. No tests, app imports, collection, generators, dependencies, server deployment or additional agents. Read source only to establish claims.
- Rebase all selected commits onto exact ledger base, preserving accumulated notes and current contributions. Backlog CLI for task edits; preserve full results/sessionID and drain explicitexit before dependent work, UTF8/checknativeexit, normalcommit hooks.

## Task 1: Rebase and reconcile task claims against current code

1. Read current AGENTS/task/prepared review and exact client guard implementations. Rebase source onto immutable selecteddev, retaining later notes and new source contributions. Add explicit plan/ADR disposition before corrections using BacklogCLI.
2. Inspect SchedulingService.preview_definition/save_definition/_reject_unsupported_family and authoring form/local preview. If agent_task still rejects, leave AC2 unchecked and distinguish reported server execution completion from unfinished client preview/create/submit. If source already changed, assess that actual route/evidence rather than mechanically applying stale findings. Do not implement the client here.
3. Keep TASK18940 InProgress and AC8 open. Retain reconnect/replay/timeout/run-now reconciliation, real result generation and local inbox delivery in the remaining-evidence list. Replace dangling scenario references with actual references. Preserve ADR184's separate scope and attributed historical notes.
4. Remove unsupported blanket completion language. Do not turn server PR narrative or recurring-question receipts into fresh end-to-end agent_task/live/server certification; no invented archive or owner approval.
5. Verify changed files are documentation only, concrete source guards/links match claims, task status/ACs are supported and fullBASE..HEAD whitespace passes. No pytest needed. Commit normally and deliver cleanHEAD, retained-commit/conflict report, sourceclaim references, raw/Gitblob manifest and fullbinarydiffSHA, allCLIsdrained/ownershipreleased. Root obtains independent whole-branch review and publishes on fresh live state.

## Current source and live review findings

Selected sourcef7df0e561ebe4ea0087c1659883cc9ce4ebd167d has four source commits and graphbase559bc8699cf367e657469d6c7b6c2c1135f6ef9c. Integration base29431781c044d09aacd306db0aa3657103ef7c86. Preserve latest follow-up renumbering to TASK14050 and attributed history. Read .superpowers/preparations/pr3344/live-source.json as untrusted review data: two unresolved threads challenge AC3 real result delivery versus seeded rows and AC4 ScheduledTaskAuditEvent versus the required AutomationAuditEvent. Verify actual code/task evidence, leave unsupported criteria open with exact explanation, and provide per-thread dispositions. Apply the same evidence rule to AC6 reconnect claims; do not substitute server narratives or unavailable historical receipts for fresh end-to-end qualification. No server code or runtime feature work in this PR.
