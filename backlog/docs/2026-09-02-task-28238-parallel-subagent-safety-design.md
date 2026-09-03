# TASK-28238 — Parallel sub-agent safety: stale-write guard + worktree isolation (design)

Status: design approved 2026-09-02 (brainstorming). Two phases under one task,
phase 1 first. Not yet planned/implemented.

## Problem

`FleetCoordinator` runs concurrent sub-agent children, and today they all share
ONE working tree: every child's fs tools (`fs_write`/`fs_edit`/`fs_patch`)
resolve against the same `workspace_root`. So two children editing the same file
race — a later writer silently clobbers an earlier one's change, with no signal.
The fs tools already confine to a `workspace_root` and check a root-level
`root_identity`, but there is no per-file read-then-write guard, and no
per-child tree isolation. (Optimistic locking exists only for notes today,
`Tools/note_management_tools.py`; the fs `expected_sha256` is a different thing
— the approval-time rug-pull check, not a disk-changed-since-read guard.)

## Goals

- Phase 1: `fs_write`/`fs_edit`/`fs_patch` refuse when the target changed since
  this run last read it, naming the conflict. (AC#2)
- Phase 2: a fleet child can opt into an isolated git worktree whose changes
  merge back explicitly, never silently. (AC#1)
- Single-agent behavior unchanged by default. (AC#3)
- The read-then-write race is exercised by a test that races two writers. (AC#4)

## Non-goals

- Pessimistic/cross-process file locking. This is optimistic locking (mirrors
  the note precedent); the goal is catching sibling races, not transactions.
- Guarding blind create-create collisions (neither side read first) — out of
  scope for phase 1's "read-then-changed"; phase 2 isolation covers it.
- Real OS confinement of tool code.

## Phase 1 — stale-write guard (primary)

### Where it lives
The fs provider (`Agents/local_tool_provider.py`), which already owns
`workspace_root` and path confinement. Add a **read-ledger**.

### Ledger keying (correctness-critical)
Key the ledger by `(run_id, resolved_path) -> ReadStamp`, NOT by provider
instance. The provider is built once per run (`_compose_local_provider`), BUT
fleet children SHARE one registry/provider — that is exactly why
`RunToolPolicy` keys its caps by `(run_id, tool)`. A per-instance ledger would
be shared across children, so a sibling's successful write would update the
shared entry and MASK a peer's staleness (false negative — the race goes
undetected). `run_id` comes from `run_context.current_run_id`, the same source
`RunToolPolicy` uses; `""` (no run) is its own key.

`resolved_path` is the fully-resolved realpath (reuse the existing confinement
resolution) so read-via-symlink / write-via-`..` map to one entry.

`ReadStamp` = `(sha256, size)` or the sentinel `ABSENT`.

### Record (on read)
On a successful `fs_read` of a path, store `ReadStamp`:
- File present: `sha256` of the **whole file** captured at read time, plus size.
  `fs_read` supports offset/windowed reads; the ledger must hash the whole file,
  not the returned window, or a whole-file check at write can never match.
- File absent (read of a missing path): store `ABSENT`.

Only `fs_read` records — `fs_list`/`fs_glob`/`fs_grep` don't establish "I saw
the content I'm about to base a write on."

### Check (on write/edit/patch)
At EXECUTION time (after any approval card — the file can change between approval
and execution too), for the target path:
1. If no ledger entry for `(run_id, path)` → proceed (blind write / new file).
2. If entry is a hash and current on-disk whole-file `sha256` ≠ recorded →
   REFUSE (`StaleWrite`): name the path, "changed since you read it (was
   `<sha8>`/`<size>`, now `<sha8>`/`<size>`); re-read and retry." No stored-
   content diff (storing every read file's content is memory bloat; the model
   re-reads to see the diff).
3. If entry is `ABSENT` and the path now exists → REFUSE similarly
   (check-then-create race, symmetric).
4. Otherwise → proceed.

Runs alongside, and before, the existing approval rug-pull check
(`expected_sha256` proposal-match): staleness check → rug-pull check → apply.

### Update (on successful write)
On a successful `fs_write`/`fs_edit`/`fs_patch`, update `(run_id, path)` to the
just-written content's hash, so the agent's own read→write→write chain never
false-positives.

### Race flow
A `fs_read(x)` → ledger[(A,x)]=h1; B `fs_write(x)` → disk=h2, ledger[(B,x)]=h2;
A `fs_edit(x)` → exec hashes disk=h2 ≠ h1 → refuse; A re-reads (ledger[(A,x)]=h2),
re-decides, retries.

### Bounds / cost
The check is one stat + whole-file hash of the target at execution — cheap for
normal files; a full read+hash for very large files (acceptable; writes are not
a hot loop). No cross-process lock; the TOCTOU window between the staleness hash
and the write is tiny and acceptable for optimistic locking.

### AC coverage (phase 1)
- AC#2: the refuse-on-change path above, conflict named.
- AC#3: a lone agent read→write with nothing else touching the file → hash
  matches → proceeds unchanged; blind writes and creates proceed.
- AC#4: a test races two writers (A reads, B writes, A writes → A refused with
  the conflict named); plus blind-write-proceeds, absent-then-created-refuses,
  and no-false-positive-on-own-rewrite tests.

### Known limitations (state honestly)
- Blind create-create collisions (neither side read first) are not caught —
  phase 2 isolation covers them.
- A blind `fs_edit` (no prior `fs_read`) relies on find-must-match as its only
  safety net.

## Phase 2 — git-worktree isolation (later, sketch)

### Opt-in
`isolation="worktree"` on the sub-agent spawn, default off. Single-agent and
ordinary fleet children unchanged (AC#3).

### Mechanism
On spawning an isolated child: `git worktree add <scratch> -b agent/<run_id>
<base-ref>` off the workspace repo; compose that child's provider with
`workspace_root=<scratch>` via the existing `_compose_local_provider` path. The
child's fs tools are confined to its own tree; the phase-1 guard still applies
within it.

### Merge-back — explicit, never silent (AC#1)
A deliberate step, not automatic: surface the child's diff and require an
explicit apply (`merge_child_worktree(handle_id)` — parent-agent tool or UI
action) that attempts the merge into the base and, ON CONFLICT, refuses and
surfaces the conflict for manual resolution rather than overwriting. A child
never merged leaves its work in its branch, discardable.

### Git dependency, handled honestly
Worktrees require a git workspace. If isolation is requested but the workspace
is not a git repo (or git is unavailable), the spawn REFUSES with a clear reason
rather than silently sharing the tree.

### Lifecycle
Worktree dir + branch created at spawn, removed via `git worktree remove` after
merge-back or discard; a prune pass GCs worktrees left by crashed/abandoned
children.

### Cost
Each worktree is a real checkout dir + git overhead per add (~200-500ms), so
isolation stays opt-in for children that genuinely need write-isolation.

### Open questions for the phase-2 build (not resolved here)
- Base ref: HEAD vs the run-start commit.
- Whether uncommitted shared-tree changes carry into the worktree.
- How merge-back reconciles with the user's own concurrent edits.
- Per-child vs batched merge-back.

## How the phases compose
The guard protects the shared tree that all fleet children use today (phase 1,
immediate value, no git dependency). Worktrees add true isolation for children
that opt in (phase 2); within a worktree the guard is belt-and-suspenders. Both
key on `(run_id, resolved_path)` under whatever root the child's provider has,
so they are correct in both shared-tree and isolated modes.
