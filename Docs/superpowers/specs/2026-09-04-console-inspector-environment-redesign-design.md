# Console Inspect Rail — Environment Redesign

- **Date:** 2026-09-04
- **Status:** Draft for owner review
- **Surface:** `ConsoleInspectorRail` (`UI/Console_Modules/right_rail.py`), composed in `UI/Screens/chat_screen.py`
- **Baseline:** design targets `origin/dev` (verified against `f6896176c8`, 2026-09-04), **not** the checkout this spec was drafted in (`feat/task-3401-video-generation-foundation`). The rail differs between the two — verified on dev: Alt+I exists (`("Alt+I", "inspect")`), the commit/push/PR worker flow exists on Change Review (TASK-16801 arc B), `GitWorkspaceInfo` carries `ahead`/`behind`, and there is **no** Changed-files rail section (that widget is feature-branch-only). Implementation starts from a fresh `origin/dev` worktree; re-verify remaining file:line references there.

## Why

The Inspect rail describes the *conversation* (run state, sources, settings) but says nothing about the *working environment* the conversation is acting on: the git working tree, branch/worktree, PR and CI state, the backlog task being worked, and the agent fleet doing the work. Users doing agentic development flip to a terminal or GitHub to answer "what changed, is CI green, which task is this?" — questions the rail should answer at a glance, in the style of Codex's Environment panel (the reference screenshots).

## Owner decisions (recorded during brainstorm, 2026-09-04)

1. **Full rail redesign, nothing dropped.** All current inspector content survives, regrouped below the new sections.
2. **Environment binds to the active workspace root** (the Workspaces registry root the rail's "Workspace" row shows). Panel re-binds on workspace/conversation switch; non-git workspaces get a quiet empty state.
3. **Status + safe actions.** Rows show live status; actions route through existing surfaces (Change Review, staged-context tray, composer). No branch checkout, commit, or push executes from the rail in v1.
4. **PR/CI via the `gh` CLI.** Auth is gh's problem. Rows hide when gh is missing/unauthenticated or the remote isn't GitHub.
5. **Tasks row: branch task + full list.** Collapsed line shows the branch-linked backlog task when one matches, else in-progress/to-do counts; expansion shows the branch task's AC checklist above the full scrollable list.
   - *Post-implementation ruling (2026-09-04, final review):* shipped **count-only**. The collapsed line carries AC progress as `3/6 ACs · <title>`; the expansion shows the full task list, **not** the per-criterion checklist. Rendering the checklist needs the AC block parsed per criterion (text + tick state) and height-capped inside the rail's 34-column body, which is its own piece of work. Completion of this decision is **task-31625**.
6. **Agent fleet section MOVES from the left rail to the redesigned right rail** — single home, reusing `_console_agent_fleet_section_state()` (`UI/Console_Modules/agent.py`) wholesale.
7. **"Local" row = execution target.** Local means this instance; remote means a remote `tldw_server`. Remote is a **designed placeholder only** in v1 — the enum and row exist, no remote path is built.
8. **Architecture: one snapshot provider** (Approach A) — a single `EnvironmentSnapshot` assembled by exclusive grouped workers, two-phase landing (local sources fast, gh slow), following the repo's standard grouped-worker + `call_from_thread` landing convention. (The Changed-files worker precedent cited during brainstorm is feature-branch-only; dev's precedent is the general `group="console-*"` exclusive-worker pattern in `chat_screen.py`.)

## Non-goals (v1)

- Branch checkout / branch creation from the rail; inline commit or push execution.
- API-based PR creation or manipulation (already ruled out of scope in the git-modes arc close-out).
- Fetching CI logs (the Fix action stages check names + details URL only).
- Remote `tldw_server` execution target (placeholder row only).
- GitLab/Codeberg PR data (git rows work everywhere; PR/CI rows are GitHub-only via gh).
- Editing backlog tasks from the rail (read-only list; "Add to chat" stages the file).
- Setting `RuntimeBindingKind.GIT_WORKTREE` on workspace records (the rail *detects* linked worktrees directly; wiring the registry enum stays a follow-up).

## Layout / information architecture

Top-to-bottom inside the existing 34-column rail:

```
Inspect ──────────────────────→   header (collapse, unchanged)
· project-instruction status      pinned rows, unchanged
· send-authority summary
╭ Environment ─────────────────╮
│ ± Changes        +1,204 −86  │  git working tree vs HEAD (numstat)
│ ⌂ Local                    ▾ │  execution target (v1: static)
│ ⎇ feat/task-3401-video…    ▾ │  branch · ↑2↓1 · worktree marker
│ ● Commit or push             │  only when dirty or ahead>0
│ ⑂ PR #2281 · Open            │  gh; absent when N/A
│ ◐ 3 failing checks       Fix │  gh checks; absent when N/A
╰──────────────────────────────╯
╭ Tasks ───────────────────────╮
│ ▸ task-3401 · In Progress    │  branch task, else "3 doing · 12 todo"
╰──────────────────────────────╯
╭ Agents ──────────────────────╮
│ 2 running · 41k tok          │  fleet section moved from left rail
╰──────────────────────────────╯
── existing content, regrouped (dev inventory) ──
· Run state (run-inspector groups: Run, Source Readiness,
  Tools, Approvals, Artifacts, …, More… disclosure)
· Selected turn activity (cited sources + library activity,
  `ConsoleSelectedTurnActivity`)
· Context tray + library search builder + Retrieval scope
· Live work (anchor position preserved per task-400's pin)
· Session settings summary
· outer scroll-hint row (unchanged)
```

Key IA rulings:

- **Environment / Tasks / Agents are collapsible sections** built on the generic `ConsoleInspectorSection` primitive (`Widgets/Console/console_inspector_section.py`), which today serves only the left rail. This incidentally delivers per-section collapse on the right rail (the missing prerequisite TASK-24611 records).
- **Two "changes" concepts stay distinct.** The Environment *Changes* row reads the real git working tree (`working_tree_status()`); agent-turn shadow-repo diffs remain Change Review's job, reached via the existing `Review changes` action (there is **no** Changed-files rail section on dev — the earlier draft wrongly listed one, an artifact of mapping the feature branch). The rail-level changes surface is the git row alone; "what did the agent do" stays one action away in Change Review.
- **Compact counts.** Line totals format compactly once large (`+1.7M −278k` at ≥100k) — a seven-digit pair plus label does not fit the rail width.
- **Visual treatment follows the rail's existing quiet-frame/heading idiom** — the boxes in the sketch above are illustrative only. No new side borders or row indents: in a ~33-column content budget, padding is content (the burn-down measured one column of row indent wrapping two live-work rows and breaking a geometry pin).
- **Inline expansions, not popovers.** Codex's floating popovers become expand-in-place rows: keyboard-reachable, no overlay layer, TUI-native.
- **Absence is silent.** No gh → no PR/CI rows. No `backlog/` → no Tasks card. No repo → one muted "No git workspace" row. Empty never renders as error.
- Geometry pins and live-work anchoring from the 2026-08 burn-down stay untouched; the redesign must not change the collapsed handle, and any min-height edits must touch **both** halves of the open/collapsed widget pair (the half-landed-fix lesson).

## Architecture

### Pure state — `Chat/console_environment_state.py` (new)

Follows the `console_display_state.py` convention: frozen dataclasses + pure row projection, zero I/O.

```
EnvironmentSnapshot
├─ git: GitEnvState          adds, dels, per-file list, branch, detached,
│                            unborn, upstream, ahead, behind, dirty,
│                            worktree_name | None
├─ target: ExecTargetState   kind: LOCAL | REMOTE_TLDW_SERVER (placeholder)
├─ pr: PrEnvState            number, title, state, is_draft, url, adds,
│                            dels, checks: pass/fail/pending counts +
│                            failing names + details URLs
├─ tasks: TasksEnvState      branch_task (id, title, status, ac_done,
│                            ac_total) | None, counts by status, entries
└─ per-source: availability (OK | NOT_APPLICABLE | MISSING_TOOL | ERROR)
               + fetched-at stamp
```

Projection functions build `ConsoleDisplayRow`s from the snapshot. All hide-when-empty, truncation, and label logic lives here and is unit-tested without subprocesses.

### Impure gatherers — `Workspaces/environment_status.py` (new)

Beside and reusing `Workspaces/git_workspace.py`:

- **Git:** `detect_git_workspace()` + `working_tree_status()` (both exist; argv-list subprocess, no shell). The lock-safety requirement — `GIT_OPTIONAL_LOCKS=0`, so status polls never write `index.lock` under an agent's concurrent git — is **already satisfied**: `git_workspace.py`'s own `_user_git_env()` pins it for every call. Verified on dev; no new work, but any *new* git invocation added by this feature must go through `_run_user_git` to inherit it. Linked-worktree detection: `git rev-parse --git-dir` ≠ `--git-common-dir` → worktree, name = root basename. All new subprocess calls follow the git-modes hardening rules: branch names and paths are data, never shell-interpolated; read `backlog/docs/lessons-testing-evidence.md` before touching argv construction.
- **PR + CI, one call:** `gh pr view <branch> --json number,title,state,isDraft,url,additions,deletions,statusCheckRollup`, cwd pinned to the workspace root, branch passed explicitly. No second `gh pr checks` round-trip. Run non-interactively: `GH_PROMPT_DISABLED=1`, `GH_NO_UPDATE_NOTIFIER=1`, `NO_COLOR=1`. Exit codes / absent binary map to `NOT_APPLICABLE` / `MISSING_TOOL`.
- **Tasks:** list `<workspace>/backlog/tasks/`; id + title come from filenames; YAML frontmatter is parsed **only** for files whose (mtime, size) changed since the last scan — the cache is **instance-scoped on the gatherer, never module-global** (a module-global cache leaked between tests in the 2026-08-30 perf review). AC checkboxes are parsed only for the branch-linked task's file. Branch link: `task-(\d+(?:\.\d+)*)` matched against the branch name (subtask ids like `task-3401.6` count).

### Orchestration — ChatScreen (repo-standard grouped exclusive workers)

- **Two exclusive thread-worker groups, one landing seam:** `console-environment-local` (git + tasks) and `console-environment-net` (gh). A single group would let the 10s local poll cancel an in-flight gh fetch every cycle (exclusive workers kill their predecessor); splitting the tiers keeps Approach A's one-seam property — both land through the same `_land_console_environment(snapshot_part, scope_token)`.
- Dispatch triggers: workspace/conversation scope change, rail open, manual refresh, a poll **only while the rail is open**, plus event-driven nudges — an agent turn completing and app focus regained both schedule a local-tier refresh, so the data is fresh exactly when it just changed. Nothing runs while collapsed; zero boot cost.
- **Two-phase landing:** the local tier lands immediately; the net tier lands when it arrives. gh results cache with a 60s TTL keyed by root+branch; manual refresh busts the TTL.
- Landing: `call_from_thread` with the standard stale-scope guard, then fingerprinted `sync_state`-style pushes so unchanged data does no DOM work.
- Cadence: local tier ~10s while open; gh tier 60s TTL. Hard timeouts (git 5s, gh 5s — this workspace has shown 1,300+ uncommitted files, and a cold `-uall` status can be slow) flip the source to `ERROR`; last good data is kept with a quiet stale marker. **Backoff:** after 3 consecutive failures a source stops polling until manual refresh or scope change — no 10s flap loop.
- The 0.2s Console sync tick gains **no** I/O — it only pushes already-landed state.

## Interactions

**Keyboard.** New sections join the rail's boundary-focus navigation. Enter/Space toggles row expansion; every action is keyboard-reachable. Expansion state persists in the rail-preferences blob beside `inspector_more_open`. Focus decisions are made synchronously at the call site — never via a flag around `focus()` (`DescendantFocus` is delivered asynchronously) — and no interaction may move focus onto the control that undoes it.

> **Post-implementation ruling (2026-09-04, final review): boundary-focus navigation deferred.** v1 ships the Environment / Tasks / Agents sections as **tab-reachable rows only**. Including them in the rail's `n`/`p` boundary-focus navigation is high-risk surgery on shared machinery that every other rail section depends on, and the arc's own reviews found no user-visible loss from tab-reachability alone. Deferred to **task-31624**. The rest of this paragraph shipped as written: expansion is Enter/Space, the state persists in the rail-preferences blob, and the focus rule is enforced — expanding a row re-focuses that same row after the section's recompose (`_request_console_environment_row_focus` in `chat_screen.py`), because "do nothing about focus" turned out not to mean "leave focus alone": the recompose unmounts the focused row and Textual resets focus onto the section's collapse chevron, i.e. exactly the control that undoes the gesture.

| Row | Enter/expand | Actions |
|---|---|---|
| Changes | per-file ± list | **Review** → Change Review (reuses the existing `Review changes` inspector action) |
| Local | target list: `Local ✓`, greyed `Remote tldw_server — not configured` | none (v1) |
| Branch | upstream, ahead/behind, worktree path, full name | none (v1) |
| Commit or push | — (single line; label adapts: "Commit or push · 12 files" / "Push ↑2"; hidden when clean & synced) | jumps to the git-modes commit flow on Change Review |
| PR | title, state/draft, ±, checks rollup | **Open in browser**; **Add to chat** → inserts a PR summary into the composer (prefill) |
| Failing checks | failing check names | **Fix** → inserts check names + details URLs into the composer (prefill) |
| Tasks | AC progress as a count on the collapsed line, above the full In-Progress→To-Do list (per-criterion checklist deferred — task-31625) | **Add to chat** on the branch task inserts its file path + title into the composer; list is read-only |
| Agents | existing fleet rows | existing fleet actions (cancel, …) unchanged |

"Add to chat"/"Fix" deliberately use the **composer prefill path** (`console_composer_bar.py` on dev), not the staged-context tray: the tray is typed evidence-bundle references (source id, authority, freshness) with no free-text kind, and inventing one is v2 scope. Prefill appends below any existing draft and never overwrites user text.

**Refresh:** automatic on rail-open and scope change; the Environment header carries a refresh action that busts the gh TTL. Alt+I (dev) remains the rail toggle; no new global bindings.

**Truncation:** titles ellipsize to the 34-column budget; the full text is always one Enter away in the expansion.

> **Post-implementation ruling (2026-09-04, final review):** the *header summary* needed its own budget, not just the row titles. The section header is `title (1fr) + summary (auto) + toggle (3)` on one line, so an unbudgeted summary starved the 1fr title to a single column (measured) and pushed the collapse chevron off the header on any 33+ character branch name — at 200×50 as well as 80×24, because the rail's body width is fixed. `ENV_SUMMARY_BUDGET` (18 columns) now bounds it in the **projection**, keeping the ± counts whole and ellipsizing the branch fragment; truncation stays in the pure module, never in the widget.

## Degradation & error handling

| Condition | Behavior |
|---|---|
| Workspace not a git repo / none bound | Environment = one muted "No git workspace" row; Tasks/PR/CI absent |
| `gh` missing or unauthenticated | PR + checks rows absent (`MISSING_TOOL`) |
| gh timeout/error after prior success | last good data + quiet stale marker; no toast, no retry storm |
| Remote isn't GitHub | PR/CI `NOT_APPLICABLE`, rows absent |
| Detached HEAD / unborn branch | branch row: `detached @ abc1234` / "no commits yet"; PR lookup skipped |
| No `backlog/` dir | Tasks card absent |
| Source failed 3× consecutively | polling for that source pauses (backoff) until manual refresh or scope change |

**Honesty note:** ahead/behind counts compare against the *last-fetched* upstream ref — the rail never runs `git fetch` (network + remote-tracking mutation are out of scope for a status panel). The branch-row expansion labels the divergence "vs last fetch" so a stale remote never masquerades as "in sync".

Background failures log via loguru with context and only flip availability enums. Nothing raised in a refresh cycle reaches the worker's `exit_on_error` path; guards cover exactly the source that failed, never siblings (the ACP-snapshot blast-radius lesson: a guard that freezes unrelated rows trades one bug for a quieter one).

## Performance budget

- **Boot:** zero new work. Sections compose empty; first fetch on rail-open (collapsed is the shipping default).
- **Open steady state:** ~3 git subprocesses + 1 dir scan per 10s; 1 gh call per 60s.
- **Tick:** the 0.2s sync tick performs no subprocess, file, or network I/O for this feature.
- **CSS:** new rules go in `css/components/_agentic_terminal.tcss`, reusing existing section classes where possible; bundle regenerated with `css/build_css.py` (never hand-edited); ADR-097 boot-CSS ratchets checked before the PR.

## Testing

- **Pure projection:** unit tests for row building, hide rules, availability states, labels, compact number formatting (`+1.7M −278k`), and subtask-id branch matching; truncation asserted via the widget's own `render_line`, never whole-frame text (the vacuous-ellipsis lesson).
- **Orchestration:** a wiring test that a local-tier dispatch does **not** cancel an in-flight net-tier worker, and a backoff test (3 failures → polling pauses → manual refresh resumes).
- **Gatherers:** real temp git repos (init / commit / `worktree add`) for the git tier — no git mocks. gh is mocked only at the subprocess seam with recorded JSON fixtures; `MISSING_TOOL` covered via an unresolvable binary. Backlog scanner: mtime-cache behavior, malformed frontmatter, absent dir.
- **Both seams, always:** every behavior gets a projection test **and** a screen-wiring test (a projection-only test passes with the screen unwired — that is how a prior fix shipped broken).
- **Widget/UI:** Textual harness tests for compose, expansion toggle, focus paths, and action routing (assert the Change Review push and the staged-context add actually occur).
- **Live verification:** an 80×24 tmux run is a required verification step — the harness allots enough rows to hide the clipping failure mode that bit this rail before.
- **Fleet move:** before renaming or relocating any label/id, sweep test consumers for **both** quoting styles (`["Label"]` membership checks and `"Label: "` rendered-text assertions); keep old labels as aliases with their own widget ids where the ownership classifier would raise.
- Targeted test selection per `backlog/docs` guidance; a gate passes only on a read nonzero passed-count.

## Migration notes

- **Left rail:** the fleet section (`ConsoleInspectorSection(section_id=CONSOLE_AGENT_FLEET_SECTION_ID)` + `#console-agent-fleet-summary` in `UI/Console_Modules/left_rail.py`) unmounts; its sync path (`_request_console_agent_fleet_sync` → coalesced sync) retargets the right rail. State builders in `UI/Console_Modules/agent.py` are reused unchanged.
- **Ownership map:** NOT involved — the strict classifier (`console_inspector_ownership.py`) governs only `ConsoleInspectorState` row labels inside `ConsoleRunInspector`. The new Environment/Tasks/Agents sections are standalone `ConsoleInspectorSection` widgets (the fleet-section pattern), which sit outside it. The rule still binds anyone adding a *run-inspector row*: an unowned label crashes the rail under STRICT policy.
- **Docs:** the matching `Docs/User_Guide/` Console page updates in the same PR (repo rule for UI changes).

## Risks & open questions

1. **gh output shape drift** across gh versions (`statusCheckRollup` fields). Mitigation: parse defensively, fixture-test against recorded JSON, treat parse failure as `ERROR` (stale marker), never crash a row. Also: `gh pr view <branch>` may resolve a closed PR when no open one exists — surface the state honestly ("Merged 6d ago", like the reference screenshot) rather than filtering to open-only.
2. **Width budget.** Six Environment rows + two cards spend vertical space in a 24-row terminal; the burn-down proved the rail clips its *last* child when starved. Mitigation: sections collapse; live 80×24 verification is mandatory; bounded sections cap the Tasks list.
3. **Backlog scan cost** in repos with thousands of task files. Mitigation: mtime cache, filename-only ids/titles, frontmatter parsed on change only; measured before merge.
4. **Two sessions, one checkout.** Implementation happens in a dedicated worktree off `origin/dev` (standing rule); this spec commits on that branch, not the video-gen feature branch.
5. **First backlog scan is a full parse** (thousands of frontmatter reads) before the mtime cache warms. It runs in the worker, so the UI never blocks, but the Tasks card may land seconds after the git rows on the first open — the card renders a quiet "scanning…" placeholder rather than popping in late unexplained.
6. **Open:** whether the retired left-rail slot backfills with anything, or the left rail simply shrinks — owner call during implementation review.
