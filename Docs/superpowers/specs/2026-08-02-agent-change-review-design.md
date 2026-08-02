# Agent Change Review — PRD / design spec

**Date:** 2026-08-02
**Status:** Approved design, pending implementation
**Feature name:** Agent Change Review (working name; "Turn Review" in UI copy)

## Problem

After an agent turn that touches disk, the user has no in-app way to see what
actually changed. The transcript shows tool markers (`⚙ write_file → ok`), but
not the content of the change, not script side effects, and not a way to back
anything out. Staying "in the loop" today means leaving the app for `git diff`
— and only works if the workspace happens to be a repo the user manages.

Codex GUI's post-turn review (compact "Edited N files ±x/y" card → full
tree-plus-diff surface with undo) is the reference UX.

## Goal

After every agent turn that modified files in the active workspace's folder
roots, the user can — without leaving the app:

1. see that changes happened (count, +adds/−dels) at a glance in the transcript;
2. open a full review: changed files grouped by Added/Modified/Deleted/Renamed,
   with syntax-highlighted diffs;
3. revert any single file, or the whole turn, safely;
4. browse the change history of previous turns.

## Non-goals (v1)

- Per-hunk revert / partial apply (phase 3 candidate).
- "Mark reviewed" bookkeeping state.
- Commit/push of the user's own repos, PR creation (the reference UI's
  "Commit or push" button is explicitly out).
- Remote filesystems (no fsspec/universal_pathlib dependency).
- Gated review-then-apply mode (phase 2 — designed-for, see §10).
- Redaction inside diffs: the review shows the user their own files on their
  own screen; unlike logs/exports, no secret masking is applied.

## Decisions taken (with reasons)

| # | Decision | Why |
|---|---|---|
| 1 | **Apply-then-review** in v1; gated mode phased later | Matches reference UX; today's write path untouched; gated mode needs merge machinery |
| 2 | **Track everything on disk** (snapshot diff), not tool logs | Script/skill side effects are exactly what "fully in the loop" must catch |
| 3 | **App-managed shadow git repo** per root | Works for non-repo roots; the user's own `.git` (repo, index, HEAD, stashes) is never touched |
| 4 | Transcript card + **dedicated Review screen** | Diffs need real estate; the Console pane is already width-starved (TASK-1846) |
| 5 | v1 actions: **Undo-all + per-file revert** | Covers real needs; both are clean restores from snapshots |
| 6 | **Feature gates on the `git` binary** | One pipeline; a difflib twin would be a permanent silent-divergence liability |
| 7 | **No new dependencies** | browsr is an app, not a library; we render a changed-file *list*, not a filesystem browser, so universal-directorytree buys nothing |

## 1. Tracking substrate: the shadow repo

One shadow repo per **canonical root path** (symlinks resolved, `~` expanded),
shared across workspaces that include the same root, stored under the app data
dir:

```
<app data dir>/change_review/<sha256(root)[:16]>/git/   # via the app's data-dir helper
```

(A flat `[change_review]` config section, deliberately not nested under
`[workspaces]`: `get_cli_setting`'s dotted-section form has a recorded
history of silently dropping caller defaults in this repo.)

- All porcelain/diff parsing uses `-z` NUL-delimited output: paths with
  spaces, newlines, or arbitrary UTF-8 are data, and revert executes
  deletions from parsed paths.
- `GIT_DIR` lives there; `core.worktree` points at the root. Nothing named
  `.git` is ever created inside the user's tree.
- Every git invocation passes explicit `--git-dir`/`--work-tree`, never `cd`s,
  and runs with `GIT_*` environment variables scrubbed.
- **Config pinned at init** (each one is a real failure on real machines):
  - `user.name` / `user.email` → fixed app identity (commit fails without one);
  - `commit.gpgsign=false` (a global `gpgsign=true` would sign — or prompt on —
    every snapshot);
  - `core.hooksPath` → empty dir (global husky-style hooks must not fire on
    snapshots);
  - `gc.auto=0` (GC is ours to schedule, §6).
- The user's `.gitignore` files are respected for noise control — **with one
  carve-out that protects the core promise**: any path the run's recorded
  file tools touched is force-added (`git add -f`) at snapshot time, so a
  direct agent edit to an ignored file (`.env` is the canonical case)
  ALWAYS surfaces in the review. Script side effects into ignored
  directories remain a documented blind spot until phase 2. Forced excludes
  (`.git/`, common junk: `node_modules/`, `.venv/`, `__pycache__/`, build
  dirs) live in the shadow repo's `info/exclude`, plus **dynamic oversize
  excludes**: git cannot exclude by size, so a pre-scan appends paths larger
  than `max_file_bytes` to `info/exclude` (recorded, surfaced in the review as
  "N oversized files untracked"). The oversize scan also runs on NEW files
  at every snapshot (cheap — `status` already lists them), so a large
  artifact the agent downloads mid-turn is excluded and disclosed rather
  than committed into the shadow store.

### Nested repos (known hole, handled honestly)

`git add` records a child repo as a **gitlink**: uncommitted changes *inside*
a nested clone are invisible to the snapshot. A root that *is* a repo works
fine (its top-level `.git` is just excluded). A root *containing* repos — the
common `~/projects` shape — does not.

- **v1:** detect nested repos during root scan; the card and Review screen
  state plainly: "N nested repositories inside this root are not tracked."
- **Fast-follow (task-filed):** auto-register detected nested repos as their
  own tracked sub-roots (own shadow repos, excluded from the parent's).

## 2. Turn protocol

A "turn" is one agent run (`run_reply`). Around it:

1. **Run start:** `add -A` + snapshot commit → baseline **B** (skipped if the
   tree is clean relative to the previous snapshot — then B = previous tip).
   B is kicked **in parallel with the model request** and awaited at the
   tool-dispatch gate: it must complete before the first tool executes, not
   before the send, so the model's own first-token latency absorbs the
   snapshot cost. `core.untrackedCache=true` keeps the per-turn status scan
   cheap on large roots.
2. **Run end:** same → **E** — on EVERY terminal path, including failed and
   cancelled runs: a run that died halfway through editing is when review
   matters most.
3. `B == E` → no changes, no card.
4. The turn's changes are exactly `diff(B, E)`; summary via
   `diff --numstat -M`, per-file content via `diff -M` / `show`.

The **first snapshot of a root happens at root registration time** (background
worker), not on the first send — first-turn latency must not absorb the cost
of hashing a whole tree.

**Failure posture: tracking never blocks the agent.** A failed snapshot logs,
the run proceeds, and the card degrades to "change tracking failed (reason)".
No `git` binary (checked once per session via `shutil.which`) → the feature is
absent with honest Settings copy, and runs behave exactly as today.

### Concurrency

- Per-root lock (in-process lock + a portable atomic-`mkdir` lockdir for
  cross-process safety — `flock` does not exist on Windows and CI runs
  Windows lanes) around snapshot/revert; `index.lock` collisions retried
  with backoff. Two fleet sessions writing one workspace is a today-case, not
  a corner.
- Overlapping runs on one root share the timeline; each run's record still
  stores its own (B, E). Attribution of interleaved writes is imperfect and
  documented (§5).

## 3. Data model

New AgentRunsDB table (schema bump per repo discipline), coordinated with run
retention:

```
change_snapshots(
  run_id TEXT NOT NULL,          -- FK to runs
  root TEXT NOT NULL,            -- canonical root path
  baseline_sha TEXT NOT NULL,    -- B
  end_sha TEXT NOT NULL,         -- E
  files_changed INTEGER, adds INTEGER, dels INTEGER,
  reverted TEXT DEFAULT '',      -- '', 'all', or JSON list of reverted paths
  tracking_error TEXT DEFAULT ''
)
```

Multi-root workspaces: one row per (run, root); the UI aggregates.

## 4. Review UI

### Transcript surface

A per-turn change-summary row in the Console transcript (same display-only
family as TOOL markers, `markup=False` — the literal-backslash lesson):

```
✎ Edited 3 files  +92 −468 — review with `v`
```

Selected-row actions gain `review` (`v`); the run inspector's actionable
group gains a "Review changes" row. "Undo all" lives on the Review screen, not
the transcript row — a one-keystroke destructive action in the transcript
would repeat the mistake TASK-1845 fixed on the approval card.

### Review screen

`UI/Screens/change_review_screen.py`, reached via `push_screen`, `Esc` returns
to the Console. Layout:

- **Header:** turn selector ("Last turn ▾" — previous turns from
  `change_snapshots`), workspace + root labels, totals, and any honesty
  banners (nested repos untracked, oversized files untracked, tracking error).
- **Left — changed-file tree:** grouped Added / Modified / Deleted / Renamed,
  flat within groups, built on the repo's existing Tree widgets. Badges:
  per-file `+a −d`; `⚠ outside file tools` (§5); `binary`.
- **Right — diff viewer:** unified diff per file with syntax highlighting;
  windowed/lazy rendering (only the focused file's hunks are mounted — a 50k
  line generated file must not freeze the screen); per-file line cap with an
  explicit "diff truncated — N more lines" row; binary files render as
  `Binary (2.1 KB → 3.4 KB)`; renames render as `old → new` with content diff.
- **Keyboard-first:** `j/k` file next/prev, `Enter` focus diff, `u` revert
  file (confirm), `U` undo all (confirm), `Esc` back. All states legible in
  monochrome (PRODUCT.md rule).

### Empty/degraded states

- No changes this turn → no card (the screen, if opened from history, says
  "No file changes in this turn").
- Ephemeral (temporary) sessions: write tools are already refused, so review
  naturally shows nothing; no special casing.
- Workspace has no folder roots → feature dormant, Settings copy explains.

## 5. Attribution: honest limits + the badge

`diff(B, E)` attributes **everything that changed during the run window** to
the turn — including a user's own mid-run edits and any external writer.
Post-run async writes (a background process the agent started) land in the
next turn's baseline move and are invisible. Both limits are documented in
the review's help text; phase 2's gated mode is the real fix.

**Tool-log cross-reference badge:** the run's recorded steps (AgentRunsDB)
name the paths its file tools touched. A changed file that no recorded file
tool touched gets a `⚠ changed outside direct file tools` badge — turning the
attribution limit into signal (that badge is how a script side effect or an
external writer becomes visible *as such*). Badge absence is not proof of
tool provenance (script writes are the point); copy says "outside direct
file tools", never "not by the agent".

## 6. Cost bounds & retention

- `max_files` / `max_total_bytes` budget per root, checked during the
  registration scan: over budget → tracking disabled for that root with
  honest copy ("narrow the root or add excludes"), never a silent half-track.
- Dynamic oversize excludes per §1.
- History rows whose snapshots were pruned render as "pruned by retention"
  in the turn selector rather than erroring.
- **Retention:** turn snapshots pruned past `retention_days` (default 30):
  drop `change_snapshots` rows, then `reflog expire` + `git gc --prune` in the
  shadow repo, scheduled off the existing maintenance path. Orphaned shadow
  repos (root removed/renamed) are GC'd by age.

## 7. Revert semantics

- **Per-file revert** = restore that path to its B state:
  - modified → `checkout B -- path`;
  - deleted → same (restores);
  - **created → guarded `rm`** — `checkout B -- path` errors on a path absent
    from B; un-create is an explicit delete;
  - renamed → restore old path, remove new.
- **Undo all** = the above for every file in the turn's diff.
- **User-edited-since guard:** before any revert, each target file's disk
  state is compared to E. Files that differ (the user — or a later turn —
  changed them after this turn) are listed *by name* in the confirm dialog
  before anything is overwritten.
- Every revert is followed by a fresh snapshot commit, and the turn's
  `reverted` field is updated — history stays true.
- **Reverts refuse while any run is active on the root** ("finish or stop
  the run first"): the per-root lock serializes git operations, but the
  agent's own file tools do not take it — reverting under a writing agent
  would interleave clobbers.
- Reverts are app actions (not agent tool calls): they bypass the tool gate
  by design but run under the same per-root lock, and each file's outcome is
  reported individually — a partial failure is never silent.

## 8. Configuration

```toml
[change_review]
enabled = true            # global kill; per-workspace override in Settings
max_file_bytes = 5_000_000
max_files = 50_000
max_total_bytes = 2_000_000_000
retention_days = 30
diff_display_max_lines = 2000   # per file, truncation disclosed
```

Per-workspace toggle surfaces in Settings alongside folder roots. Env-var
overrides follow the repo's `TLDW_<SECTION>_<KEY>` convention.

## 9. Testing posture

- **Real git, no mocks:** tmp-dir shadow repos against tmp roots; every claim
  exercised against actual `git` output.
- **Revert round-trips** for each of create/modify/delete/rename, plus the
  user-edited-since guard (edit E-state before reverting → confirm lists it).
- **Hardening tests:** global `commit.gpgsign=true` and a global `hooksPath`
  simulated in a scratch `HOME` — snapshots must still succeed silently.
- **Nested repo test:** child repo inside root → gitlink change does NOT
  surface as content; the warning does.
- **Sabotage-verification** (project standing rule): each guard demonstrated
  to fail when its production half is removed.
- **UI tests load the shipped stylesheet** (bare-harness measurements are
  fiction — TASK-1846) and wait on conditions, not pause counts (TASK-1900).
- **Live tmux verification** before merge: real app, real root, real agent
  run with a file edit; review opened, diff read, revert performed, at 80×24
  and 212×64.

## 10. Phasing

- **v1 (this spec):** everything above.
- **Fast-follows (task-filed with v1):** nested-repo auto-sub-roots;
  tool-badge refinements.
- **Phase 2 — gated mode (designed-for, not built):** per-workspace opt-in;
  the agent's file-root resolution swaps to an app-managed `git worktree`
  checked out from the shadow repo's tip; Accept applies the diff to the real
  root, Reject drops the worktree. The by-root shadow repo and (B, E) records
  are exactly the substrate this needs — no v1 rework anticipated.
- **Phase 3 candidates:** per-hunk revert; "mark reviewed"; open-in-$EDITOR.
