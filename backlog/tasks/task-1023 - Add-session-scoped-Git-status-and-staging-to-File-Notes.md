---
id: TASK-1023
title: Add session-scoped Git status and staging to File Notes
status: In Progress
assignee: []
created_date: '2026-07-27 20:33'
updated_date: '2026-07-27 21:23'
labels:
  - notes
  - git
  - library
  - ux
  - security
dependencies:
  - TASK-969
  - TASK-982
documentation:
  - Docs/superpowers/specs/2026-07-27-file-notes-session-git-staging-design.md
  - Docs/superpowers/plans/2026-07-27-file-notes-session-git-staging.md
  - backlog/decisions/034-file-notes-session-git-index-controls.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users inspect and safely stage or unstage only paths changed during the current Chatbook File Notes session while preserving disk authority and external Git index state observed by a fresh preflight. This first Git slice remains optional and excludes commit, push, remote, branch-mutation, and repository-wide status behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A selected File Notes root inside one supported Git worktree replaces the unbounded session summary with Session Git (N), where N is the coalesced current-session group count; after trust, the view shows the canonical repository, actual session-path Git states, Back to Files, and explicit Session paths only and complete-file-state staging labels.
- [ ] #2 User can Stage one eligible logical row or Stage All eligible rows after pending autosave settles; create, modify, delete, mode, and inseparable move-lineage changes are supported, while any literal endpoint whose Git mutation closure includes a non-session ancestor or descendant is blocked and unrelated index/worktree state remains unchanged except for disclosed trusted-filter side effects or the documented external-index race.
- [ ] #3 User can Unstage one or all still-owned groups by restoring exact saved pre-Stage stage-0 entries or absences without a path-oriented or worktree-restoring operation; Stage updates retain previously owned entries' original baselines, baseline insertion blocks on any unexpected current index ancestor/descendant and explicitly removes only matching owned conflicts, repository/HEAD/topology/post-Stage/semantic mismatch revokes eligibility, and a changed move topology requires a successful Stage update rather than no-op endpoint ownership.
- [ ] #4 Session rows, repository trust, Git service/session state, and valid staging ownership survive leaving and reopening the fresh Library screen within one application process; changing the selected root or restarting clears them, and a staged path then appears external.
- [ ] #5 Before the first worktree-aware command, the trust prompt shows the canonical repository and process-only scope, warns that configured filters may execute, initially focuses Cancel, and treats Escape/close as decline; worktree top-level, worktree Git directory, Git common directory, and their platform-stable filesystem identities are revalidated immediately after confirmation and before each worktree-aware status or mutation.
- [ ] #6 The documented row/action mapping, checking/stale/error states, status summaries, Back behavior, focus restoration, and Up/Down/Tab/Shift+Tab/Enter/Escape contract work in wide and narrow layouts; Git mutation controls are disabled without disabling editor input or Back while status/mutation work is pending.
- [ ] #7 Fresh preflight blocks observed pre-existing or partially staged same-path state, conflicts, ignored paths, nested repositories, active sparse checkout/index, directory or unsafe worktree types, nondefault semantic index flags, invalid identity, parsed status paths outside the requested session whitelist, and Stage or Unstage file/directory mutation closure outside exact session ownership; Stage is fail-fast even when user Git configuration enables add.ignoreErrors.
- [ ] #8 Stage/Unstage child execution, postflight, and ownership publication complete under a process-owned Git service across view recomposition or forced unmount; normal screen departure is vetoed by a separate mutation gate that also blocks root transitions and structural create/move/delete/restore/save-copy actions but leaves editing, autosave, replica synchronization, and in-screen Back usable, while `TldwCli` shuts the owner down even with no Library screen mounted using bounded child termination and no index-lock deletion.
- [ ] #9 Refresh performs no polling and starts no status work while the view is hidden, permits one status query plus at most one coalesced rerun, safely finishes an already-running query under the process owner, makes mutation wait for active status, rejects stale-generation results, and surfaces timeout, lock contention, or uncertain mutation results without claiming protection from the explicitly unsupported concurrent external-index race.
- [ ] #10 Focused unit, one disposable-repository integration matrix containing the primary `git diff`/`git diff --cached` acceptance flow, mounted Textual lifecycle/keyboard tests, and one lightweight 1,000-plus-unrelated-notes fixture cover the approved boundary without a duplicate acceptance layer, full-suite, network, pagination, or broad performance execution.
- [ ] #11 Chatbook selects no Git operation intended to modify the worktree and never writes the SQLite replica or File Notes session history during Git actions; commit, push, pull, fetch, remotes, credentials, branch mutation, hunk staging, repository initialization/repair, persistent ownership, full-repository status, sparse support, and nested-repository management remain absent.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/034-file-notes-session-git-index-controls.md
Reason: ADR-034 already defines Git index ownership, trust, lifecycle, and UX; ADR-033 governs the process-session owner.

1. Move sequenced File Notes session changes into a root-generation-bound process owner.
2. Add pure session grouping, porcelain parsing, closure, row-policy, and ownership models.
3. Add sanitized direct-argv repository discovery and trusted coalesced status.
4. Implement exact Stage and Stage update with saved original baselines.
5. Implement exact saved-baseline Unstage with index replacement-closure checks.
6. Add the retained Session Git navigator, trust prompt, focused controls, and selective mutation gate.
7. Inject the owner across fresh Library screens and settle it before Textual closes screens.
8. Run only the approved focused repository/UI/lifecycle/UAT matrix and reconcile TASK-1023.

Detailed plan: Docs/superpowers/plans/2026-07-27-file-notes-session-git-staging.md
<!-- SECTION:PLAN:END -->
