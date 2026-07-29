---
id: TASK-1350
title: Add guarded session commit to File Notes
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-29 19:17'
updated_date: '2026-07-29 23:43'
labels:
  - notes
  - git
  - library
  - ux
  - security
dependencies:
  - TASK-1213
  - TASK-1235
documentation:
  - >-
    Docs/superpowers/specs/2026-07-29-file-notes-guarded-session-commit-design.md
  - Docs/superpowers/plans/2026-07-29-file-notes-guarded-session-commit.md
  - backlog/decisions/038-file-notes-guarded-session-commit.md
  - backlog/decisions/035-file-notes-session-git-index-controls.md
  - backlog/decisions/033-application-session-state-ownership.md
  - backlog/decisions/029-file-notes-disk-authority.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users turn the exact Chatbook-owned staged notes from the current File Notes session into one reviewed local Git commit without absorbing unrelated staged content, changing file/SQLite authority, or expanding into push and remote management.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Prepare session for commit exposes Commit staged (N), a required-subject/optional-body form, a read-only review, and an explicit Confirm commit step with unambiguous keyboard, focus, cancellation, a two-row footer at narrow widths, and 40x20 behavior.
- [ ] #2 Immediately before execution, Chatbook proves that the complete staged delta against the captured attached branch HEAD exactly equals current Chatbook staging ownership; unrelated staged content, newer unstaged edits on included groups, conflicts, special index states, detached or unborn HEAD, active Git sequencer operations, legacy grafts, and partial/promisor repositories block without starting a commit child or making a Chatbook-authored HEAD, index, note-worktree, SQLite, or remote mutation, subject to ADR-035's disclosed trusted-filter boundary. A dedicated internal proof path may transiently read repository-wide logical-index metadata for preflight, postflight, and recovery only; it never exposes or retains unrelated path identities as session status or ownership.
- [ ] #3 The review shows the exact normalized UTF-8 message, branch, old commit, included-note count/list, resolved author and committer, and the policy that hooks are bypassed and the commit is unsigned; missing identity or invalid message blocks.
- [ ] #4 Confirm revalidates the immutable review snapshot under the process owner mutation gate, keeps the editor read-only through terminal outcome, executes one direct-argv noninteractive normal commit with hooks, configured filesystem-monitor helpers, signing, and automatic maintenance disabled, and never abandons the retained child lifecycle.
- [ ] #5 Postflight classifies Cancelled, Blocked, Succeeded, Failed unchanged, or Uncertain from the child result plus the exact branch and replacement-free raw parent, full-tree, metadata, and logical-index facts; success additionally proves the branch tip is the exact reviewed commit and the logical index equals its tree with no staged delta. Chatbook never rolls back, deletes locks, or automatically retries, and retains immutable proof facts only as long as an uncertain attempt needs Check again; later exact proof may converge to Succeeded, while unchanged-state recovery requires both the captured old branch/index and a known normal unsuccessful child result.
- [ ] #6 A proven success retires only fully committed session groups, retains newer worktree edits, refreshes actual status, and reports the committed note count plus the approved promise that unrelated changes were untouched—defined as no unrelated staged content included and no unrelated worktree path selected by Chatbook; failure and uncertainty preserve the message draft and provide exact recovery.
- [ ] #7 Commit actions never rewrite note bytes or mutate the independent SQLite replica, revisions, tombstones, repository configuration, remotes, credentials, push/pull/fetch, lazy-fetch objects, amend, signing, or general branch management; the feature adds no durable trust state and preserves ADR-035 trust invalidation semantics.
- [ ] #8 Focused unit, disposable-repository integration, lifecycle, and mounted Textual tests include repository-wide logical-index proof without unrelated-path disclosure, ambient date/replacement/graft-state isolation, partial/promisor blocking without network access, no automatic-maintenance descendant, exact success-index proof, uncertain Check-again convergence, a representative session set in a 1,000-note repository, and a bounded Git process count; a representative two-note flow receives narrow/wide live UAT, with no unrelated full-suite or broad CI expansion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/038-file-notes-guarded-session-commit.md
Reason: ADR-038 defines the guarded local-commit service, security, recovery, process, and UX boundary; ADR-035 remains the staging authority.

Detailed plan: Docs/superpowers/plans/2026-07-29-file-notes-guarded-session-commit.md

1. Add pure commit contracts and exact owner authority.
2. Add retained child settlement and complete guarded review proof.
3. Execute once, prove typed outcomes, and converge uncertainty.
4. Cover the disposable-repository security matrix.
5. Add the Prepare-panel review and workspace lifecycle.
6. Run focused verification and wide/40x20 live UAT.
<!-- SECTION:PLAN:END -->
