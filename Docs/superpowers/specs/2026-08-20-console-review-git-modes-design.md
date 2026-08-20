# Change Review Git Modes — Design (TASK-16801 arc B)

**Date:** 2026-08-20
**Task:** TASK-16801 "Change review: git commit/push/PR modes"
**Predecessors:** TASK-1972 (change review), TASK-18060 arc A
(`2026-08-18-console-review-rail-design.md`), revert precedent
TASK-1845/1974.
**Owner rulings (2026-08-20):** commit is working-tree, file-picked
(the `current` mode shows real git status; commit stages exactly the
files the user selects, all pre-checked); PR = browser compare URL now
with an API-based follow-up task filed at close-out; branching =
commit to the checked-out branch with an optional "create branch
first" field, visible warnings (never blocks) for detached HEAD and
main/master, push `-u` when no upstream.

## 1. Problem and scope

Change review is turn-centric and read-only against SHADOW snapshots
(`Workspaces/change_tracking.py`): the tracker keeps its own `GIT_DIR`
under the app data dir and never touches the user's git state. When the
workspace root happens to be a real git repository, acting on an
agent's changes (commit, push, open a PR) means leaving Console for a
shell. This arc adds:

- **Detection groundwork**: an active "is this workspace root a real
  git repository" probe (today `RuntimeBindingKind.GIT_WORKTREE` is an
  unused placeholder; nothing detects real repos).
- **A `current` mode** on the Change Review screen: the REAL working
  tree's changed files and diffs, alongside the existing snapshot
  turns.
- **Contextual actions** in that mode: commit (file-picked, confirmed),
  push (confirmed), open-PR (compare URL in the browser).

Out of scope (recorded, not implied): API-based PR creation (follow-up
task filed at close-out); setting `RuntimeBindingKind.GIT_WORKTREE` on
registry rows (detection is read-only probing, not registry state);
force-push of any kind; commit/push against snapshot (non-`current`)
modes; multi-repo batch actions.

## 2. Code grounding (verified against dev `2a74a7b31`)

- `Workspaces/change_tracking.py` — shadow tracker; scrubs ALL `GIT_*`
  env, explicit `--git-dir`/`--work-tree`; `-z` NUL-delimited parsing
  discipline ("paths are data"). Never touches user git state.
- `Workspaces/change_revert.py` + `ChangeRevertConfirmModal`
  (`change_review_screen.py:1977`) — the confirmation precedent:
  summary + named-file list, `SafeModalDismissMixin`, per-path outcome
  honesty, `provider.run_active` probe refuses during live runs.
- `UI/Screens/change_review_screen.py` — `ChangeReviewScreen` with turn
  `Select`, tree of `(row, ChangedFile)` leaves, diff pane with line
  cursor + comment anchors (arc A), honesty banners, synchronous
  `_load_turn`, `_diff_text_for` memo keyed
  `(generation, id(row), path)`.
- `Chat/console_agent_bridge.py` — builds the provider; snapshot rows
  carry `root` (canonical workspace root), one row per root per window;
  roots originate from `turn_context.workspace_roots`
  (`console_chat_controller.py:10815`).
- `UI/Screens/chat_screen.py:18657-18671` — the opener wires
  `provider.run_active` to live controller state
  (`CONSOLE_ACTIVE_RUN_STATUSES`).
- `Tools/git_tool_impls.py` — read-only agent git tools. `run_git` has
  a sanitized env that STRIPS `HOME` (no `~/.gitconfig`, no credential
  helpers) — correct there, disqualifying here; its allowlist is
  read-only. `prepare_repository` refuses a repo root ABOVE the
  workspace root (confinement precedent).
- `Notes/file_notes_git_*` — the guarded-push machine for File Notes.
  Conceptual precedent only (explicit confirm, honest outcome states);
  not a library this arc lifts.
- `Utils/github_api_client.py` — `[github] api_token` /
  `GITHUB_API_TOKEN` auth plumbing exists (read-only browsing today);
  relevant to the API-PR follow-up, unused in this arc.
- Textual 8: `App.open_url` exists (installed `textual/app.py:4790`).

Empirically verified 2026-08-20 (scratch repo probes; each becomes a
pinned test in §9):

1. `git add -A -- <selected>` + `git commit -m <msg> -- <selected>`
   commits EXACTLY the selected files and leaves an unrelated
   pre-staged index entry staged and uncommitted. (A bare
   `git commit -m` would hijack the user's staged-but-unselected work
   into our commit — the arc's canonical would-have-shipped bug.)
2. `git status --porcelain=v1 -z` rename record is `R<XY>\0new\0old\0`
   — NEW path first; an untracked directory collapses to one
   `?? dir/` row without `-uall`.
3. `git check-ref-format --branch "-bad"` exits 128 with a clear
   message — branch-name validation and option-injection guard in one.

## 3. Detection groundwork — `Workspaces/git_workspace.py`

One new module owning BOTH detection and actions (one seam, one env
posture, one docstring disclosing why it differs from the two existing
runners).

```python
@dataclass(frozen=True)
class GitWorkspaceInfo:
    root: Path              # the workspace root probed
    repo_root: Path         # `rev-parse --show-toplevel`
    branch: str | None      # current branch, None when detached
    detached: bool
    upstream: str | None    # e.g. "origin/feat/x", None when unset
    remotes: tuple[tuple[str, str], ...]  # (name, push URL) pairs
    ahead: int              # vs upstream; 0 when no upstream
    behind: int

def detect_git_workspace(root: Path) -> "GitWorkspaceInfo | None": ...
```

Rules:

- **Root must equal the repo toplevel** (paths compared resolved). A
  workspace root INSIDE a repo returns a typed refusal
  (`GitWorkspaceRefusal(reason)`) surfaced as the mode's
  "why unavailable" copy — mirroring `prepare_repository`'s
  confinement rule. A nested repo under the root is irrelevant here
  (detection probes the root itself). Relaxable later; refusal copy:
  "workspace is inside a repository — git actions need the workspace
  root to be the repository root".
- Not a repo at all, or no git binary → `None` (mode absent, AC #4).
- Detection is read-only plumbing: `rev-parse --show-toplevel`,
  `rev-parse --abbrev-ref HEAD` (detached → `HEAD`),
  `rev-parse --abbrev-ref @{upstream}` (check=False),
  `remote -v` (push lines), and, when upstream exists,
  `rev-list --left-right --count @{upstream}...HEAD` for
  behind/ahead.
- Runs ONLY on screen open / explicit reload, inside the screen's
  load worker. Never on the Inspector rail's 0.2s tick (arc A hard
  invariant: no DB/git on the sync tick — the rail is untouched by
  this arc).

### 3.1 The third env posture (module docstring MUST disclose all three)

| Runner | Posture | Why |
|---|---|---|
| Shadow tracker | scrubs ALL `GIT_*`, explicit `--git-dir` | must never touch user git state |
| Read-only agent tools (`run_git`) | minimal env, `HOME` stripped | model-driven; no user identity wanted |
| **Git modes (this arc)** | ambient env preserved, repo-TARGETING vars scrubbed | acts AS the user in their own repo |

Git-modes runner (`_run_user_git(root, *args, timeout, check)`):

- Start from ambient `os.environ` — `HOME`, `SSH_AUTH_SOCK`,
  credential helpers, `GIT_SSH_COMMAND`, `GIT_ASKPASS`/`SSH_ASKPASS`
  all preserved (an https push may pop a GUI askpass — that is the
  user's own configuration working).
- Scrub (case-insensitive, Windows): `GIT_DIR`, `GIT_WORK_TREE`,
  `GIT_INDEX_FILE`, `GIT_OBJECT_DIRECTORY`,
  `GIT_ALTERNATE_OBJECT_DIRECTORIES`, `GIT_NAMESPACE`,
  `GIT_COMMON_DIR` — a stray targeting var must not redirect a commit
  into the wrong repo (the app itself sets none, but the app may be
  LAUNCHED from a hook or an env that does).
- Set `GIT_TERMINAL_PROMPT=0` (fail honestly, never hang a TUI on a
  hidden prompt), `GIT_OPTIONAL_LOCKS=0`, `GIT_PAGER=cat`.
- `subprocess.run` with argv lists (never shell), `cwd=root`
  (no `-C` needed), stdin `DEVNULL`, `capture_output=True`, timeouts:
  30s reads, 120s commit (user hooks run — see §5), 300s push
  (network). Output surfaced to the UI is excerpt-capped (stderr
  first 400 chars, matching the tracker's convention).
- User hooks and signing config RUN (their repo, their rules); a hook
  or gpg failure surfaces as the step's honest error, with the
  stderr excerpt.

## 4. The `current` mode (screen surface)

- The turn `Select` gains ONE pseudo-entry, value
  `CURRENT_MODE_SENTINEL = "__git_current__"` (module constant;
  run_ids are UUIDs so no collision), labeled
  `Working tree (current) — <branch>` (detached: `detached HEAD`).
  Present only when the kill switch is on AND detection succeeded for
  ≥1 candidate root. Candidate roots = distinct roots across the
  conversation's snapshot rows ∪ live workspace roots passed by the
  opener (`_open_change_review` gains a `workspace_roots` kwarg the
  chat screen fills from the controller's workspace roots — exact
  accessor pinned at plan time; `None` for every legacy caller). The
  entry appears FIRST in the list (it is "now"; turns stay
  newest-first below it) but the screen still OPENS on the latest
  turn exactly as today — the pseudo-entry is offered, never the
  default (byte-compatible open behavior).
- Selecting it dispatches ONE exclusive worker (`thread=True`) that
  runs, per detected root: detection (fresh), then
  `status --porcelain=v1 -z -uall`, parsed with the tracker's
  token-walk discipline (rename = NEW then OLD, §2 probe 2). The UI
  shows "Loading working tree…" until `call_from_thread` lands the
  result. This deliberately does NOT copy `_load_turn`'s synchronous
  posture — real-repo status/diff on a large cold repo can stall the
  UI thread; snapshot turns keep their existing sync path unchanged.
- Landed rows reuse the existing tree/leaf plumbing by synthesizing
  ONE pseudo-row dict per root:
  `{"root": str(root), "kind": "git_current", "id": -1}` — the
  `(row, ChangedFile)` leaf shape, per-root tree grouping, and the
  diff memo key `(generation, id(row), path)` all survive unchanged.
  `ChangedFile` is reused with porcelain-derived status letters
  (`??` → `A`-like with an `untracked=True` marker carried in a
  parallel set on the screen, not a ChangedFile field change).
- Diffs: tracked files via `git diff HEAD -- <path>` (worker-fetched
  through the SAME `_diff_text_for` memo — the provider method
  branches on the pseudo-row kind); untracked files synthesized in
  Python: bounded read (display-cap lines), NUL-byte sniff → "binary
  file" label — never `git diff --no-index` (exit-code-1 semantics,
  platform quirks) and never index tricks like `--intent-to-add`
  (mutating the user's index from a VIEW is forbidden).
- Header/banner in current mode: branch, upstream, ahead/behind
  (e.g. `feat/x ↑2 ↓0 → origin/feat/x`), and the totals line the
  turns already get.
- A CLEAN tree still enters the mode: empty tree shows "working tree
  clean", commit disabled (§6), push/PR still offered — unpushed
  commits are the point right after a commit.

### 4.1 Snapshot-only features gate on mode (the row-consumers table)

Every consumer of `(row, ChangedFile)` audited; in current mode:

| Path | Behavior in current mode |
|---|---|
| `action_revert_file` / `action_undo_all` | no-op + notify "revert works on recorded turns — select a turn" (revert is snapshot-anchored) |
| `action_comment_file` / `c` line-comment | no-op + notify "comments attach to recorded turns" (notes anchor to `change_snapshots` rows; pseudo-row `id=-1` must never reach the DB) |
| `_marked_diff_lines` / notes strip | skipped entirely (no snapshot id to query) |
| `_diff_text_for` memo | works as-is (pseudo-row identity keys it; generation bump on every current-mode reload) |
| j/k, line cursor, diff render | work unchanged |

The gate is one predicate `self._current_mode_active()` checked at the
TOP of each listed action.

## 5. Commit (file-picked, confirmed)

Button `Commit…` + binding, visible/enabled only in current mode with
≥1 listed file. Flow:

1. **Refusal first**: `provider.run_active()` → notify refusal
   (same copy pattern as revert's). Checked again inside the engine
   immediately before running (injected probe, `change_revert.py`
   precedent).
2. **Fresh preflight**: modal opens only after a FRESH
   `status --porcelain -z -uall` read (worker) — the visible list may
   be stale (the working tree moves under the view; the modal must
   list what commit will actually see).
3. **`ChangeGitCommitModal`** (`SafeModalDismissMixin`, escape=cancel):
   - file checklist, all pre-checked (unchecking excludes);
   - required message `Input` (stripped-nonempty validated);
   - current branch line; optional "create branch first" `Input`;
   - warnings (never blocks): detached HEAD ("commit will not be on
     any branch"), branch in `{main, master}` ("committing directly
     to <branch>");
   - a merge/rebase in progress (`rev-parse --verify MERGE_HEAD` /
     `REBASE_HEAD`, check=False) → commit REFUSED with reason
     ("finish or abort the merge/rebase first") — a pathspec commit
     is invalid mid-merge and the raw git error is worse copy.
4. **Engine** (`commit_selected(root, files, message, new_branch)`)
   runs per-step, stopping at the first failure, each step an
   outcome row:
   - optional: `git check-ref-format --branch <name>` (validation +
     option-injection guard, §2 probe 3) then
     `git checkout -b <name>`;
   - `git add -A -- <selected paths>` (handles deletions);
   - `git commit -m <message> -- <selected paths>` — the pathspec
     commit; message and paths are argv elements, paths after `--`.
     **Pinned semantics (§2 probe 1): commits exactly the selected
     paths; an unrelated pre-staged index entry survives staged and
     uncommitted.**
5. Outcome: notify (sha short + file count on success; step + stderr
   excerpt on failure), then reload current mode (the tree changed).
   Buttons disabled while the exclusive worker runs (no
   double-dispatch).

## 6. Push (confirmed) and PR

**Multi-root targeting** (applies to commit, push, and PR): each
action acts on ONE root's repo — the root of the focused leaf; with
no focused leaf (clean tree), the sole detected root when there is
exactly one, else the action's modal carries a root `Select`. The
confirm modal always NAMES the root it will act on. (One detected
root is the overwhelmingly common case; this paragraph exists so the
>1 case is defined rather than improvised.)

**Push** — enabled in current mode when detection found ≥1 remote
(AC #2: no remote → disabled with "no git remote configured").
NOT refused during active runs (push ships already-committed state
only — the working tree is untouched; the spec says this explicitly
because commit IS refused). Flow: confirm modal naming branch,
target remote, and upstream state:

- upstream set → `git push <remote-of-upstream>` (no refspec games);
- no upstream → `git push -u <remote> <branch>`; remote = the sole
  remote when exactly one, else a `Select` in the modal;
- detached HEAD → push disabled with reason ("no branch checked
  out").

Never `--force`/`--force-with-lease` — a non-fast-forward rejection
surfaces git's stderr excerpt honestly (no-silent-destructive
precedent, AC #3). A credential failure under `GIT_TERMINAL_PROMPT=0`
("could not read Username…", "Permission denied (publickey)") maps to
appended hint copy: "credentials were not available non-interactively
— push once from a terminal or configure a credential helper/ssh
agent". Runs on the exclusive worker (300s timeout); buttons disabled
while running; ahead/behind re-read after.

**Open PR** — enabled when the current branch HAS an upstream (which
a first `-u` push establishes); otherwise disabled with "push the
branch first". Builds the compare URL from the upstream's remote URL:

- Remote URL parsing: `https://host/owner/repo(.git)`,
  `ssh://git@host/owner/repo(.git)`, scp-like
  `git@host:owner/repo(.git)`; strip `.git`; anything else →
  unsupported.
- Hosts and templates (branch percent-encoded, `/` kept):
  - `github.com` → `https://github.com/{o}/{r}/compare/{branch}?expand=1`
  - `gitlab.com` → `https://gitlab.com/{o}/{r}/-/merge_requests/new?merge_request%5Bsource_branch%5D={branch}`
  - `bitbucket.org` → `https://bitbucket.org/{o}/{r}/pull-requests/new?source={branch}`
  - `codeberg.org` (Gitea family needs a base) → only when
    `refs/remotes/<remote>/HEAD` resolves locally:
    `https://codeberg.org/{o}/{r}/compare/{base}...{branch}`; else
    disabled with "can't determine the default branch — open the PR
    on codeberg.org".
  - Any other host → disabled with "PR links support github.com,
    gitlab.com, bitbucket.org, codeberg.org" (AC #2).
- Opened via `self.app.open_url(url)` — never `webbrowser.open`
  (stdout can corrupt the TUI). The URL is built from parsed
  components only; no user text is interpolated unencoded.

## 7. Engine/provider seam

- `Workspaces/git_workspace.py`: detection (§3), the runner (§3.1),
  `working_tree_status(root)`, `working_tree_diff(root, path)`,
  `untracked_preview(root, path, max_lines)`,
  `commit_selected(...)`, `push_current(...)`,
  `pr_compare_url(info) -> str | Refusal`. Pure of UI; outcomes are
  frozen dataclasses (`GitStepOutcome(step, ok, detail)` lists), all
  errors typed (`GitWorkspaceError`), stderr excerpt-capped.
- `AgentRunsChangeReviewProvider` grows thin wrappers
  (`detect_git`, `current_status`, `current_diff_text`,
  `commit_selected`, `push_current`, `pr_url`) so screen tests drive
  the REAL provider against real repos (the fixture-invented-shapes
  trap — five prior instances). The `run_active` probe is passed into
  `commit_selected` exactly as `revert_paths` receives it.
- Screen work runs via `run_worker(..., exclusive=True, thread=True)`
  landing with `call_from_thread` (repo pattern).

## 8. Config, gating, honesty

- Kill switch: `[change_review] git_actions = true` (flat section,
  same read pattern as `diff_display_max_lines`). Off → no
  detection, no pseudo-entry, zero behavior change.
- AC #4 "degraded" interpretation (pinned): the modes hide when git
  itself is unavailable or detection fails/refuses; a HISTORICAL
  per-turn `tracking_error` row does NOT hide them — detection is
  live truth about the real repo, not about the shadow tracker's
  past. (`ShadowRepoService.available` false ⇒ no git binary ⇒
  detection also fails ⇒ hidden — consistent automatically.)
- Every disabled action carries its reason as copy (tooltip/banner
  line), never a dead control (AC #2); every failure names the step
  and the excerpt, never a rolled-up "git failed" (revert's per-path
  honesty precedent).

## 9. Test plan (AC #5: real git, no mocks)

All engine/e2e tests drive REAL temp repos; screen tests use the real
CSS stack + real provider over a file-backed AgentRunsDB (arc A
rules).

- **e2e commit+push** (the AC's named case): tmp repo + `git init
  --bare` local remote; commit selected files → push `-u` → assert
  the BARE remote's ref moved and log contains the message; second
  push → "up to date" outcome; divergence made via a second clone →
  push fails honestly, no `--force` anywhere in argv (assert on
  captured argv).
- **Index-preservation pin** (§2 probe 1 as a regression test):
  pre-staged unrelated file survives a selected-files commit staged
  and uncommitted.
- **Detection**: non-repo → None; root inside a repo → refusal with
  copy; detached HEAD; no-remote; ahead/behind counts; no-git-binary
  → None (service-unavailable path).
- **Status parsing**: rename record NEW-then-OLD; `-uall` per-file
  untracked (directory case pinned); paths with spaces/UTF-8.
- **Untracked preview**: text bounded at cap; NUL byte → binary
  label; never spawns `--no-index` (argv assert).
- **Branch validation**: `-bad` refused pre-flight via
  check-ref-format; `checkout -b` existing branch → step failure
  surfaces, no commit attempted.
- **Merge-in-progress refusal**: MERGE_HEAD present → commit refused
  with the reason copy.
- **URL builder** (pure unit): all three remote shapes × four hosts;
  `.git` stripping; branch with `/` and unicode percent-encoding;
  unsupported host refusal; codeberg with/without
  `refs/remotes/origin/HEAD`.
- **Screen**: pseudo-entry present only when detection succeeds AND
  kill switch on (both directions, guard proven red pre-fix);
  entry absent for non-repo (AC #4); revert/comment no-op with
  notify in current mode (row-consumers table); commit modal opens
  from a FRESH status read (file created after view load appears in
  the modal); run_active refusal for commit; push NOT refused while
  run_active (both asserted); buttons disabled while the worker
  runs.
- **Env posture**: runner preserves `HOME`, scrubs `GIT_DIR` (env
  assert on a captured invocation); `GIT_TERMINAL_PROMPT=0` set.

## 10. Docs and close-out

- `Docs/User_Guide/console/agent-runs-and-tools.md`: new "Git actions
  in change review" subsection + "Verified against" stamp.
- Close-out files the API-PR follow-up task (GitHub API creation via
  the existing `[github] api_token` plumbing) — filed then, not
  referenced before it exists (backlog rule).
