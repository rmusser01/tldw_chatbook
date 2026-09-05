---
id: TASK-31450
title: >-
  Console Inspect rail Environment redesign: git/PR/CI/tasks/agents panel
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-04 20:30'
labels:
  - console
  - inspector
  - ux
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebuild the Console Inspect rail around a Codex-style Environment panel so a
user doing agentic development can answer "what changed, is CI green, which
task is this, what are my agents doing?" without leaving the app. Owner-approved
spec: `Docs/superpowers/specs/2026-09-04-console-inspector-environment-redesign-design.md`;
implementation plan: `Docs/superpowers/plans/2026-09-04-console-inspector-environment-redesign.md`.
ID leapfrogged to 31450 after a sweep (dev max 31383, branch-name max 26042).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Inspect rail shows an Environment section with live git working-tree change counts, branch (ahead/behind, worktree marker), and an execution-target row, bound to the active workspace root; non-git workspaces show one quiet empty row
- [x] #2 PR and CI-checks rows appear via the gh CLI when available and silently hide when gh is absent/unauthenticated/no-PR; failures keep last good data with a stale marker and back off after 3 consecutive errors — silent absence captured live; the populated-rows path verified against a REAL open PR out-of-app (see Live-evidence gaps)
- [x] #3 A Tasks section shows the branch-linked backlog task (with AC progress) or status counts, expanding to a scrollable In-Progress-first list; absent without a backlog/ dir
- [x] #4 The agent fleet section renders in the Inspect rail (moved from the left rail) with its existing behavior intact — shipped in task 10 with widget/wiring tests; NOT exercised live (needs a real sub-agent fleet, see Live-evidence gaps)
- [x] #5 Row actions work: Changes→Change Review, Commit-or-push→Change Review in working-tree mode, PR open-in-browser, PR/checks/task add-to-composer; all keyboard-reachable — Changes→Change Review and Tasks→Add-to-chat driven live; the PR/checks actions could not be reached live (no PR rows in an isolated run)
- [x] #6 Environment/Tasks collapse state persists per workspace; no new I/O on the 0.2s tick; zero new boot cost while the rail is collapsed — persistence verified live across an app restart. **AC wording correction:** the rail-preference key is `global` by default (`[console] rail_layout_scope = "global"`), so the state persists per *layout scope*, which is per-workspace only when that scope is selected
- [x] #7 Both-seams test coverage (projection + screen wiring) and a live 80x24 tmux verification with captures in Implementation Notes
<!-- AC:END -->

## Implementation Plan

See `Docs/superpowers/plans/2026-09-04-console-inspector-environment-redesign.md`
(14 TDD tasks: pure state → gatherers → persistence/composition → orchestration/wiring → docs+live verification).

## Implementation Notes

Thirteen TDD tasks built the panel bottom-up (pure state -> gatherers ->
persistence/composition -> orchestration/wiring); this task is the docs,
derived-artifact, and live-verification close-out. Branch
`feat/console-inspector-environment`, base `origin/dev @ 5f030c8a07`, head
`08ca0957b1` at verification time.

### Approach

- **Pure/impure split.** `Chat/console_environment_state.py` holds frozen
  dataclasses plus pure row projections (`project_environment_section`,
  `project_tasks_section`); `Workspaces/environment_status.py` holds the
  subprocess/filesystem gatherers (`gather_git_env`, `gather_pr_env`,
  `BacklogTaskScanner`). Every hide rule, label, compact-count format, and
  truncation lives on the pure side and is unit-tested without subprocesses.
- **Two exclusive worker tiers, one landing seam.** `ConsoleEnvironmentController`
  (`UI/Console_Modules/environment.py`) runs `console-environment-local`
  (git + backlog) and `console-environment-net` (gh) as separate exclusive
  groups so the 10 s local poll can never cancel an in-flight gh fetch, and
  lands both through one `_land()` with scope + per-tier dispatch-token
  guards. gh caches for 60 s keyed `(root, branch)`; **Refresh** busts it.
- **Rail composition.** `ConsoleInspectorRail` mounts Environment
  (`#console-environment-section`, view-all slot "Refresh"), Tasks
  (`#console-tasks-section`), and the relocated fleet section
  (`#console-agent-section-subagents`, retitled "Agents") ahead of the
  staged-context tray, each hidden when its projection has no rows.
- **Binding.** The panel reports on `_console_environment_root()` = the
  conversation's first change-review workspace root, so the panel and Change
  Review can never describe different trees.

### Deviations from the plan / spec

- **Fleet auto-open had to be re-added at the rail level.** Moving the fleet
  list into a rail that defaults closed made `_apply_fleet_agent_section_
  auto_open` a no-op; task 13 added `_apply_fleet_inspector_rail_auto_open`
  (sibling of the launch auto-open), keyed on fleet ROW presence, with a
  sticky per-window user dismissal. It only fires at >=150 columns, inherited
  from the existing launch-auto-open precedent (recorded ruling).
- **AC#6 wording:** collapse persists under the configured rail layout scope,
  `global` by default — not per workspace unless that scope is selected.
- **Unauthenticated `gh` maps to `ERROR`, not `MISSING_TOOL`.** `gh` exits 4
  when it has no credentials, which `gather_pr_env` classifies as `ERROR`
  (the spec's degradation table says `MISSING_TOOL`). User-visible behaviour
  is identical — the rows are absent either way — but the error path also
  increments the 3-strike backoff counter. Left as-is; noted as a concern.

### Live verification (2026-09-04, isolated scratch profile)

tmux at 80x24 and 200x50, `.venv/bin/python -m tldw_chatbook.app`, launched
with `HOME`/`XDG_*`/`TLDW_CONFIG_PATH` all under a scratch profile and
`TLDW_CHANGE_REVIEW_ENABLED=1`. The real `~/.config/tldw_cli/config.toml`
md5 was `bcb16a978c700ea6c32c2d74a9d353ae` before and after every run.

**(a) 80x24, no workspace bound — the quiet empty state**

```
│ Environment                 ▾
│   No git workspace
```

Tasks/PR/checks rows all absent, exactly as specified. To bind a workspace a
user needs a workspace with a folder binding AND Change Review consent
enabled for it; the panel reads the change-review-admitted roots.

**(a) 80x24, workspace bound to this dirty worktree — real counts**

```
│ …feat/console-inspector-enviro
│   Changes
│   +121 −24
```

`+121 −24` matches `git diff --numstat` exactly (29+92 adds, 19+5 dels).
Walking the section by Tab (each frame captured):

```
TAB 3   ┌Changes                    ┐   └+121 −24                   ┘
TAB 4   ┌Local                      ┐
TAB 5   ┌feat/console-inspector-env…┐   └↑18 ↓108 wt:console-inspec…┘
TAB 6   ┌Commit or push · 3 files   ┐
TAB 7              Refresh
TAB 8  Tasks   61 doing · 543 todo ▾    61 in progress · 543 to do
```

Ahead/behind, the `wt:` linked-worktree marker, the dirty-only Commit-or-push
row, and the Refresh slot all render with real data. No PR or checks row
appeared (see gaps below) — the documented silent absence.

**(b) 80x24 — nothing clips the rail's last child.** A 38-step Tab walk
reached the last focusable child with the scrollbar thumb at the bottom and
its content fully painted before wrapping to the top:

```
│ Resume state: local session,  ▁
│ not persisted yet             ▁
│
│ ▼ more sections — scroll
```

The outer `▼ more sections — scroll` hint is present throughout, as expected.

**(c) Collapse persists across a restart.** Collapsing both sections wrote
`environment_open = false` / `tasks_open = false` into the scratch profile's
`[console.rail_state."console_rail_state:global:shared-layout-v1"]`, and the
next launch came up collapsed (Environment header + Refresh only; Tasks
chevron `▸`). The scratch config still parsed as TOML after the app's own
rewrite.

**(d) gh rows.** `gh` is installed and authenticated for this user
(`gh auth status` -> `rmusser01`, keyring). Inside the isolated run gh is
unauthenticated (it reads `$XDG_CONFIG_HOME/gh` and resolves its keychain
token through the real `HOME`, both of which the scratch profile redirects),
so `gh pr view` exits 4 and the PR/checks rows were silently absent in every
live capture — the correct degradation, captured. The success path was then
verified out-of-app against a REAL open PR, with the same containment
(`TLDW_CONFIG_PATH` scratch, real config md5 unchanged): `gather_pr_env`
returned `OK`, `#2402`, `OPEN`, `+562 −41`, 13 checks (12 passed, 1 pending),
and `project_environment_section` over that snapshot produced

```
env-pr        | PR #2402 · Open
env-pr-title  | Screen-instance reuse mechanism via installed screen | +562 −41
env-pr-open   | Open in browser
env-pr-add    | Add to chat
env-checks    | 1 pending check
```

No PR was created and nothing was pushed to manufacture this.

**(4) 200x50 wide run.** Environment and Tasks render; Agents stays hidden
(no sub-agents). `Enter` on the Changes row expands in place:

```
│   Changes                        +121 −24
│   M Docs/User_Guide/console/agent-…   +29 −19
│   M Docs/User_Guide/console/contex…   +92 −5
│   A task31450-verify-scratch.txt      +0 −0
│   Review in Change Review
```

`Enter` on **Review in Change Review** pushed the Change Review screen. A
second workspace on a `feat/task-777-demo` branch with a `backlog/` dir
exercised the branch-linked Tasks variant and its actions:

```
│ Tasks      task-777 · In Progress ▾
│   task-777 · In Progress
│   2/4 ACs · Demo environment task
│   task-777 · Demo environment task   In Progress
│   task-778 · Another demo            To Do
│   Add task to chat
```

`Enter` on **Add task to chat** put the task title + path into the composer
(`Pasted text | 139 characters | Expand`). A third workspace with no
`backlog/` directory rendered no Tasks section at all.

### Concerns found live (no production code changed in this task)

1. **The Environment section header loses its title and its collapse chevron
   whenever the branch name is long.** The header is
   `Horizontal[title(1fr, min 0), summary(auto), toggle(3)]`; the summary is
   `<branch> +adds −dels`, which at the rail's ~31-cell content width consumes
   the whole row, so the section renders as a bare truncated branch name with
   no "Environment" label and no visible `▾`. This is **not** a narrow-terminal
   artifact — the rail is a fixed ~35 columns, so it reproduces identically at
   200x50. The toggle is still keyboard-reachable (2nd tab stop in the section)
   and still works, but it is invisible and unlabelled. Compare Tasks, whose
   shorter summary leaves `Tasks   61 doing · 543 todo ▾` intact.
2. **A muted "No git workspace" is indistinguishable from "Change Review has
   not finished preparing this root".** Root readiness is in-memory per boot,
   so every launch has a window where the panel honestly reports no workspace.
   Worse: killing the app during the first shadow-repo initialisation leaves a
   partial shadow repo (loose objects, no HEAD/refs) and every subsequent boot
   then fails readiness **silently** — the panel sat on "No git workspace" for
   60+ s with nothing in the app log. Deleting
   `<data_dir>/change_review/<hash>/` restored it immediately. The defect is in
   the Change Review layer upstream of this feature, but the panel inherits it.
3. **Untracked files report `+0 −0`.** A new 7-line file shows as
   `A <path>  +0 −0`, and does not contribute to the headline count, while it
   does count toward `Commit or push · N files`.
4. **`Commit or push · 1 files`** — no singular form.
5. **Expanding a row drops keyboard focus**, because the section is
   re-projected; the next Tab restarts from the section toggle.
6. **Environment/Tasks are not part of the rail's `n`/`p` named-section
   navigation** — `n` from the pinned summary jumps straight to "Sources —
   next send", and `p` does not walk back into them.
7. **Unauthenticated `gh` is classified `ERROR`, not `MISSING_TOOL`** (above).

### Live-evidence gaps (stated, not papered over)

- The **Agents** section was never rendered live: it needs a real reply that
  spawns sub-agents, which the scratch profile's placeholder credential cannot
  produce. Its move, ids, title, and hide-when-empty behaviour are covered by
  task 10's widget/wiring tests.
- **PR/checks rows** were never rendered inside the app (see (d)); the
  gatherer + projection were verified against real gh data instead.
- **Stale markers and the 3-failure backoff** are covered by controller unit
  tests only; no live failure was induced.
- **Commit-or-push -> Change Review working-tree mode**, **PR open-in-browser**,
  and **checks Fix** were not activated live.
- The **0.2 s tick does no new I/O** and **zero boot cost while collapsed** are
  design properties backed by the controller's rail-open gate and its tests,
  not measured in this pass.

### Derived artifacts

`python -m tldw_chatbook.css.build_css` produced no diff (no CSS changed in
this task). `./scripts/preflight.sh` is green on all six checks (generated
stylesheets, profile-owned path census, production diagnostic inventory,
duplicate backlog ids across 3218 files, chachanotes table allowlist, index
plan pins).

### Files changed in this task

- `Docs/User_Guide/console/context-and-rag.md` — new "The Environment panel
  (Environment, Tasks, Agents)" section (rows, expansions, actions, refresh
  triggers, the "vs last fetch" honesty note, the degradation table, `gh`
  optionality, collapse persistence), Inspect-rail layout-tour bullet updated,
  new Verified-against stamp.
- `Docs/User_Guide/console/agent-runs-and-tools.md` — fleet panel relocated to
  the Inspect rail's **Agents** section throughout (transcript pointer, left-rail
  bullet list, the three-state fleet panel, drill-in, survivor rows, Cancel all
  agents), plus the >=150-column rail auto-open, new Verified-against stamp.
- `backlog/docs/lessons-live-verification.md` — new entry on `gh` (and any
  keychain-backed CLI) losing its credentials under a scratch-profile launch.
- this task file.
