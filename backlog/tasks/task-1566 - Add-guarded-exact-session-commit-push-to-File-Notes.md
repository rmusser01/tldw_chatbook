---
id: TASK-1566
title: Add guarded exact-session-commit push to File Notes
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-31 02:38'
updated_date: '2026-07-31 05:07'
labels:
  - notes
  - git
  - library
  - ux
  - security
dependencies:
  - TASK-1350
  - TASK-1411
documentation:
  - Docs/superpowers/specs/2026-07-30-file-notes-guarded-session-push-design.md
  - Docs/superpowers/plans/2026-07-30-file-notes-guarded-session-push.md
  - backlog/decisions/039-file-notes-guarded-session-push.md
  - backlog/decisions/038-file-notes-guarded-session-commit.md
  - backlog/decisions/035-file-notes-session-git-index-controls.md
  - backlog/decisions/033-application-session-state-ownership.md
  - backlog/decisions/029-file-notes-disk-authority.md
  - backlog/decisions/029-local-private-data-boundary.md
  - backlog/decisions/011-chatbook-workbench-ui-system.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users separately publish exactly the one guarded File Notes commit Chatbook just proved in the current application process to its existing upstream branch, without becoming a general Git client, managing remotes or credentials, or weakening disk authority and the independent SQLite replica.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Prepare session for commit exposes a separate Review push (1 commit)… action only for the exact guarded commit candidate created and proved in the current application process; it never auto-chains from commit, never expands to older or outside commits, and disappears after restart or binding invalidation.
- [ ] #2 Before any network or credential-helper contact, Chatbook requires a separate process-only authorization for the exact sanitized destination identity; only existing noninteractive authentication and secure HTTPS or OpenSSH/scp-style transports are accepted, while prompts, credential storage/editing, plaintext/local/ext transports, ambiguous or credential-bearing URLs, and repository-controlled executable helpers block.
- [ ] #3 Read-only preflight freezes one existing tracking upstream, exact effective push endpoint, full existing refs/heads destination, parent OID, and candidate OID; missing/deleted, divergent, ambiguous, mirror, multiple-push-URL, conflicting push-default/refspec/option, unsupported host/policy, and candidate-lineage mismatch states block without starting a push.
- [ ] #4 Confirm freshly revalidates all local, configuration, trust, and remote facts, then invokes the frozen endpoint directly to request exactly candidate-OID:destination-ref with an exact --force-with-lease=destination-ref:parent-OID compare-and-swap; the proven direct-child update remains fast-forward, cannot recreate or overwrite a changed branch, bypasses local pre-push hooks, blocks included Git LFS paths, and sends no tags, push options, upstream edits, deletes, mirrors, retries, or implicit refspecs.
- [ ] #5 The application-session owner retains the push candidate, trust epochs, single-use review capability, operation status, and uncertain proof in memory; the Git service owns every check/push child and descendant lifecycle through settlement, the mutation gate blocks conflicting Git/root/rebind actions while ordinary editing/autosave continues, remount reattaches without duplicate work, and Cancel is available only before the network push child actually starts.
- [ ] #6 Typed outcomes distinguish no-push Already published, proven accepted Success, normally settled Failure with no update currently observed, and Uncertain; timeout, lost result, contradictory or unavailable proof never retries, and Check remote again — no push queries only the retained original endpoint after owned descendants settle, converging only when the candidate is observed while a parent observation remains uncertain.
- [ ] #7 The push UX shows exact commit/ref/destination and policy facts, included-session-note provenance, a safe initial action and final explicit Push 1 commit confirmation, persistent checking/pushing/attention status across panel remount, sanitized selectable endpoint Details, accurate point-in-time copy, and fully keyboard-operable scroll/focus/recovery behavior at 40x20 and normal widths.
- [ ] #8 A quiescent guarded push changes only the approved remote destination ref and unavoidable remote object/helper/server-side state; Chatbook does not mutate local HEAD or refs, index, repository/worktree configuration, note bytes, File Notes replica rows/revisions/tombstones, or session history, and concurrent note editing changes only the intended disk/replica state without changing the candidate proof.
- [ ] #9 No database schema, persistent push candidate, trust journal, crash recovery claim, remote creation/configuration, branch creation, general history/status browser, pull/fetch workflow, credential UI, provider-specific integration, or background retry is added; after process exit users inspect and push existing commits with external Git.
- [ ] #10 Focused pure, real-Git CAS/race, secure SSH/HTTPS transport, native process-containment, lifecycle, mounted Textual, and same-process production-app PTY UAT cover the approved boundary with sanitized durable evidence and a phase-to-evidence matrix; verification remains risk-focused and does not add a repository-wide test, coverage, or broad local CI run.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/039-file-notes-guarded-session-push.md
Reason: Guarded push changes the remote/network/authentication boundary, exact compare-and-swap contract, uncertain recovery, process ownership, and long-lived Prepare-panel workflow.
Detailed plan: Docs/superpowers/plans/2026-07-30-file-notes-guarded-session-push.md

1. Add pure push contracts, parsers, and exact argv builders.
2. Atomically publish one exact guarded-commit push candidate.
3. Prove local destination, transport, configuration, and LFS policy.
4. Own network process trees through settlement.
5. Build one immutable network Git execution context.
6. Add authorized remote preflight and immutable review.
7. Execute the exact lease-guarded push and prove CAS semantics.
8. Retain uncertain proof, query only, and settle shutdown.
9. Rehydrate push state and keep Session Git truthful.
10. Add the separate keyboard-safe push presentation.
11. Verify compact, remounted, and lifecycle UX.
12. Prove secure SSH/HTTPS and ambiguous transport behavior.
13. Run same-process production-app PTY acceptance.
14. Run the focused regression gate and close ADR/task only after evidence.
<!-- SECTION:PLAN:END -->
