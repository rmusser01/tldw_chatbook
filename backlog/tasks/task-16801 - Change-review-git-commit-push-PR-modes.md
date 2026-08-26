---
id: TASK-16801
title: 'Change review: git commit/push/PR modes'
status: Done
assignee: []
created_date: '2026-08-15'
labels:
  - console
  - change-review
  - git
dependencies:
  - TASK-1972
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Change review (TASK-1972 and the turn file card that presents it) only
shows the working-tree diff a turn produced — there is no way to act on
it with git from inside the app, and when the workspace
happens to be a real git repository, a user who wants to commit, push, or
open a pull request for an agent's changes has to leave Console and drop
to a shell for everything past inspecting the diff.

This is arc B of the V2 turn-file-card design
(`Docs/superpowers/specs/2026-08-15-console-turn-file-review-design.md`,
"Out of scope" section). The V2 bucket originally named two additions;
the sidebar multi-file review view was SPLIT OUT to TASK-18060 (owner
ruling 2026-08-18, tackled individually) and is specced in
`Docs/superpowers/specs/2026-08-18-console-review-rail-design.md`. This
task is now scoped to the git half alone: contextual `current` /
`commit` / `push` / PR actions that only appear when the workspace is a
git repository and each action's own precondition is met (a configured
remote for push, a supported git host for PR creation, etc.). Note the
groundwork gap recorded during arc-A review: no active "is this
workspace a real git repo" detection exists today (`RuntimeBindingKind.
GIT_WORKTREE` is an unused placeholder; the shadow-repo service never
touches user git state) -- this arc builds that detection.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 When the workspace is a git repository, change review offers contextual actions for `current` (working tree), `commit`, `push`, and opening a PR
- [x] #2 Each action is offered only when its own precondition holds (e.g. push requires a configured remote, PR creation requires a supported git host) and explains why it is unavailable otherwise, rather than failing silently or with a raw error
- [x] #3 Commit and push never run without an explicit confirmation step, consistent with the no-silent-destructive-action precedent already established for revert (TASK-1845/TASK-1972/TASK-1974)
- [x] #4 A workspace that is not a git repository, or one where change tracking is degraded, shows none of the git-contextual modes rather than a broken or erroring control
- [x] #5 Commit and push are exercised end-to-end against a real local git repository in tests (a temp repo with a local bare remote), not only against mocked git calls
<!-- AC:END -->

## Implementation Plan

Executed via subagent-driven development from
`Docs/superpowers/plans/2026-08-20-console-review-git-modes.md`, arguing from
`Docs/superpowers/specs/2026-08-20-console-review-git-modes-design.md`:
9 tasks (engine runner + detection, working-tree status/diff/preview, commit
engine, push + PR URLs, provider seam + kill switch, current-mode screen,
commit UI, push/PR UI, opener wiring + User Guide), each with a fresh
implementer and an independent review, then a whole-branch review and one
final fix wave.

## Implementation Notes

Adds a `Working tree (current)` mode to Change Review that reads the REAL
repository (not the shadow tracker) and offers confirmed commit, push and
open-PR actions.

**Engine** (`Workspaces/git_workspace.py`, new): a third git runner posture,
disclosed in its docstring against the two existing ones — ambient environment
preserved (HOME, ssh agent, credential helpers, so the user's own auth works)
with only repo-TARGETING and pathspec-mode variables scrubbed, plus
`GIT_TERMINAL_PROMPT=0` so auth fails honestly instead of hanging a TUI.
Detection refuses a workspace root that is not the repository toplevel.
Commit uses a pathspec commit (`add -A -- <sel>` then `commit -m msg --
<sel>`), which leaves a user's unrelated pre-staged work staged and
uncommitted; the ADD pathspec is filtered by `os.path.lexists` (and skipped
when empty, since an empty `add -A --` stages the whole tree) so index-recorded
renames and staged deletions commit correctly. Push passes an explicit
fully-qualified refspec on both forms.

**UI** (`UI/Screens/change_review_screen.py`): a sentinel entry in the turn
selector loads working-tree state in its own exclusive worker with a
dispatch-scope guard; snapshot-only features (revert, comments) gate off in
current mode so the pseudo-row's `id=-1` can never reach the notes DB. Git
actions ride their own worker group behind a liveness guard; landings catch
Textual's teardown `RuntimeError` only, so a real bug arrives as a traceback.
Kill switch `[change_review] git_actions` (default on) removes the feature
entirely.

**Security — four repository-supplied argv vectors found and fixed during
review**, none of which put a dangerous flag in our own argv: an option-shaped
remote name, an option-shaped branch name via `.git/HEAD`, push config
(`remote.push` / `remote.mirror` / `push.default=matching`) turning an ordinary
push into a forced update or ref deletion, and pathspec magic in a filename
turning a one-file selection into a multi-file commit. Each was reproduced
destroying real data before being fixed, and each fix is pinned by a test that
asserts the other clone's commit, branch and tag survive. `diff.external` and
textconv could also make the review pane render a fabricated or blank diff for
a changed file; both `git diff` sites now pass the machine-safe flags. Lesson
recorded in `backlog/docs/lessons-testing-evidence.md`.

**Verification:** 629 tests across the arc's suites (two consecutive
foreground runs), `--collect-only Tests/` = 52,273 collected / 0 errors, ruff
clean. All engine and end-to-end tests drive real temp repositories with a
real local bare remote — no mocked git. Guards were mutation-tested throughout
rather than trusted: reviewers killed the index-hijack pin, both argv
refusals, the run_active refusals, the stale-land guards and the mode gating,
each confirming a test dies.

**Not done, filed:** TASK-19700 (agents can write `.git/` in a workspace root
— the upstream cause of the threat model above), TASK-19701 (`pushurl` /
`pushInsteadOf` redirects), TASK-19702 (Default workspace never sees Change
Review at all — predates this arc), TASK-19703 (three honesty/polish items).

**Files:** `tldw_chatbook/Workspaces/git_workspace.py` (new),
`tldw_chatbook/UI/Screens/change_review_screen.py`,
`tldw_chatbook/UI/Screens/chat_screen.py`,
`tldw_chatbook/css/components/_change_review.tcss` + regenerated bundle,
`Docs/User_Guide/console/agent-runs-and-tools.md`, and eight test files under
`Tests/Workspaces/` and `Tests/UI/`.
