---
id: TASK-1350
title: Add guarded session commit to File Notes
status: Done
assignee:
  - '@codex'
created_date: '2026-07-29 19:17'
updated_date: '2026-07-30 09:50'
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
- [x] #1 Prepare session for commit exposes Commit staged (N), a required-subject/optional-body form, a read-only review, and an explicit Confirm commit step with unambiguous keyboard, focus, cancellation, a two-row footer at narrow widths, and 40x20 behavior.
- [x] #2 Immediately before execution, Chatbook proves that the complete staged delta against the captured attached branch HEAD exactly equals current Chatbook staging ownership; unrelated staged content, newer unstaged edits on included groups, conflicts, special index states, detached or unborn HEAD, active Git sequencer operations, legacy grafts, and partial/promisor repositories block without starting a commit child or making a Chatbook-authored HEAD, index, note-worktree, SQLite, or remote mutation, subject to ADR-035's disclosed trusted-filter boundary. A dedicated internal proof path may transiently read repository-wide logical-index metadata for preflight, postflight, and recovery only; it never exposes or retains unrelated path identities as session status or ownership.
- [x] #3 The review shows the exact normalized UTF-8 message, branch, old commit, included-note count/list, resolved author and committer, and the policy that hooks are bypassed and the commit is unsigned; missing identity or invalid message blocks.
- [x] #4 Confirm revalidates the immutable review snapshot under the process owner mutation gate, keeps the editor read-only through terminal outcome, executes one direct-argv noninteractive normal commit with hooks, configured filesystem-monitor helpers, signing, and automatic maintenance disabled, and never abandons the retained child lifecycle.
- [x] #5 Postflight classifies Cancelled, Blocked, Succeeded, Failed unchanged, or Uncertain from the child result plus the exact branch and replacement-free raw parent, full-tree, metadata, and logical-index facts; success additionally proves the branch tip is the exact reviewed commit and the logical index equals its tree with no staged delta. Chatbook never rolls back, deletes locks, or automatically retries, and retains immutable proof facts only as long as an uncertain attempt needs Check again; later exact proof may converge to Succeeded, while unchanged-state recovery requires both the captured old branch/index and a known normal unsuccessful child result.
- [x] #6 A proven success retires only fully committed session groups, retains newer worktree edits, refreshes actual status, and reports the committed note count plus the approved promise that unrelated changes were untouched—defined as no unrelated staged content included and no unrelated worktree path selected by Chatbook; failure and uncertainty preserve the message draft and provide exact recovery.
- [x] #7 Commit actions never rewrite note bytes or mutate the independent SQLite replica, revisions, tombstones, repository configuration, remotes, credentials, push/pull/fetch, lazy-fetch objects, amend, signing, or general branch management; the feature adds no durable trust state and preserves ADR-035 trust invalidation semantics.
- [x] #8 Focused unit, disposable-repository integration, lifecycle, and mounted Textual tests include repository-wide logical-index proof without unrelated-path disclosure, ambient date/replacement/graft-state isolation, partial/promisor blocking without network access, no automatic-maintenance descendant, exact success-index proof, uncertain Check-again convergence, a representative session set in a 1,000-note repository, and a bounded Git process count; a representative two-note flow receives narrow/wide live UAT, with no unrelated full-suite or broad CI expansion.
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a guarded local-commit flow that converts exactly the
Chatbook-owned staged File Notes groups into one reviewed normal commit. The
session owner remains the sole binding/staging authority, the Git service owns
all direct-argv process and proof work, the workspace owns binding-scoped
draft/editor leases, and the panel remains presentation-only.

- Added normalized message and identity contracts, repository-wide
  logical-index proof with no unrelated-path disclosure, exact commit
  revalidation/postflight, retained child settlement, typed outcomes, and
  proof-only uncertain recovery.
- Added the form/review/confirm/result UX, quiet editor actions, persistent
  count-and-promise success copy, responsive `40x20` two-row actions, focus
  handling, cancellation boundaries, and draft/remount behavior.
- Core implementation is in `file_notes_git_commit.py`,
  `file_notes_git_service.py`, `file_notes_session_owner.py`,
  `library_file_notes_git_panel.py`, and `library_file_notes_workspace.py`,
  with focused unit, disposable-repository, lifecycle, and mounted-UI tests.
- Commit execution does not call note-write or SQLite mutation APIs, uses no
  shell, bypasses hooks/signing/maintenance, performs no remote operation, and
  introduces no durable trust or recovery state. ADR-038 governs the feature;
  ADR-035, ADR-033, and ADR-029 remain intact. No new ADR was required.
- Final review additionally closed effective repository include/worktree
  promisor detection and made retained proof-child draining safe under repeated
  explicit cancellation; linked worktrees without worktree config remain
  supported. Both final reviewers approved the fixes.
- Focused verification passed: 588 tests in 184.99 seconds, changed-file Ruff,
  compile checks, and `git diff --check`. Formatter check still flags nine
  changed files: eight already failed at the pre-feature baseline and the
  ninth is the new integration harness. They were not mechanically reformatted
  as part of this already-reviewed feature closure.
- Live UAT passed at `120x40` and `40x20` with real owner/service/SQLite and
  disposable Git repositories. Commit
  `0bc3cc655ee46bfd77fc23d2524bdf08627bd3f4` contained only `one.md` and
  `two.md`; unrelated unstaged work remained, the index was clean, cancel
  preserved the draft, unrelated staging blocked, and disk/replica logical
  bytes were unchanged. Evidence:
  `/tmp/task1350-live-uat.O2BJB7/evidence.json`.
- Manual UAT did not manufacture a genuinely ambiguous live commit child or
  duplicate every injected edge case. Newer-edit blocking, uncertainty,
  `Check again`, and exact convergence are covered by the focused automated
  boundary.

Post-merge full-app UAT correction (2026-07-30):

- The earlier `/tmp/task1350-live-uat.O2BJB7` run was a mounted Textual
  `WorkspaceHarness`, not a launch and traversal of the complete Chatbook
  application. Its Git, SQLite, autosave, staging, review, commit, and safety
  evidence remains valid automated end-to-end coverage.
- A subsequent real `python -m tldw_chatbook.app` PTY walkthrough proved the
  guarded two-note commit and unrelated-staged-content block, but found that
  the Files source choice is not rendered at normal widths and Prepare actions
  are unreachable at `40x20` with a realistic linked-root path. Full UX
  acceptance is therefore blocked pending TASK-1411.
- Durable evidence:
  `Docs/superpowers/qa/file-notes-full-app-uat-2026-07-30/README.md`.
<!-- SECTION:NOTES:END -->
