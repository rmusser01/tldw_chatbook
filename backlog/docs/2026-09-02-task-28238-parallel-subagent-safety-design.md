# TASK-28238 — Parallel sub-agent safety: stale-write guard + worktree isolation (design)

Status: design approved 2026-09-02 (brainstorming), then hardened by an
adversarial design review against the code (same day; corrections folded in
below). Phase 1 implemented on branch `feat/task-28238-stale-write-guard`
(2026-09-02); phase 2 (worktree isolation) not started.

## Problem

`FleetCoordinator` runs concurrent sub-agent children, and today they all share
ONE working tree: every child's fs tools (`fs_write`/`fs_edit`/`fs_patch`)
resolve against the same `workspace_root`. So two children editing the same file
race — a later writer silently clobbers an earlier one's change, with no signal.
The fs tools already confine to a `workspace_root` and check a root-level
`root_identity`, but there is no per-file read-then-write guard, and no
per-child tree isolation. (Optimistic locking exists for notes,
`Tools/note_management_tools.py`, and — review finding — `fs_write` ALREADY has
a model-opt-in atomic compare-and-swap: `write_file(expected_sha256=...)`
refuses under a per-target lock when the on-disk digest changed
(`Tools/local_tool_impls.py:468-476`). What's missing is (a) anything AUTOMATIC
— the model must volunteer the param today, and never does; (b) any guard at all
for `fs_edit`/`fs_patch`. A separate, promotion-only `expected_sha256`
proposal-match exists in the agent-lesson path; do not confuse the two.)

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

### Record (on read) — provider-side, never keyed off fs_read success
Review CRITICAL: an fs_read of a missing path is NOT a success — it raises
`LocalToolError`, and confinement/denylist refusals raise the SAME exception
type (`local_tool_impls.py:337` vs the `resolve_workspace_path` refusals). So
the ledger must never be stamped from fs_read's outcome. Instead, on an fs_read
dispatch the PROVIDER resolves the path itself via `resolve_workspace_path`:
- resolve raises (refused path) -> record NOTHING;
- resolved + `is_file()` false -> record `ABSENT`;
- resolved + present -> record `(sha256(whole file), size)`. `fs_read` windows
  with offset/limit but `_read_relative_file` already reads the whole file
  (`local_tool_impls.py:349`); the provider-side hash is a second read of the
  file (accepted cost — keeps `local_tool_impls` stateless; noted as avoidable
  later by recording inside the impl if large-file double-reads ever matter).
- A binary file: fs_read itself still refuses (error), but the provider-
  side observation records the file's DISK state (hash) anyway — the
  ledger tracks what was on disk, not what the model saw, and that is the
  correct base for a later staleness comparison.

Only `fs_read` records — `fs_list`/`fs_glob`/`fs_grep` don't establish "I saw
the content I'm about to base a write on."

### Ledger home, lock, cap (review: reuse existing patterns in this file)
An instance dict + `threading.Lock` on `LocalToolProvider`, mirroring
`_inline_bytes_by_run`/`_spill_lock` (`local_tool_provider.py:718-719`): sibling
fleet children run tools concurrently on daemon threads, so the read-modify-
write must be locked (precedent: `RunToolPolicy._lock`). Bound growth with a
per-run entry cap borrowing the `_MAX_PROMOTION_PROPOSALS_PER_RUN` pattern
(`:722,727,1261`) — necessary because `build_server_local_provider`
(`MCP/local_server_tools.py:366`) composes a LONG-LIVED provider where
`current_run_id()` is always `""` and entries would otherwise accumulate for
the server lifetime. A size cap, not an eviction subsystem.

### Check (on write/edit/patch)
At EXECUTION time (after any approval card — the file can change between
approval and execution too). Insertion point: `_invoke_allowed`, after the gate
resolves `allow`, before `selected_spec.handler(clean_args)`. (The promotion-
only proposal-match is a separate sub-path and is untouched.)

Decision per target path:
1. No ledger entry for `(run_id, path)` → proceed (blind write / new file).
2. Entry is a hash and current on-disk whole-file `sha256` ≠ recorded →
   REFUSE (`StaleWrite`): name the path, "changed since you read it (was
   `<sha8>`/`<size>`, now `<sha8>`/`<size>`); re-read and retry." No stored-
   content diff (storing every read file's content is memory bloat; the model
   re-reads to see the diff).
3. Entry is `ABSENT` and the path now exists → REFUSE similarly
   (check-then-create race, symmetric).
4. Otherwise → proceed.

Mechanism differs by tool (review finding — reuse the atomic CAS where it
exists):
- **fs_write**: when the model did not supply `expected_sha256` and the ledger
  has a hash entry, the provider INJECTS `expected_sha256=<ledger hash>` into
  the handler call, reusing `write_file`'s already-atomic, per-target-locked
  compare-and-swap (`local_tool_impls.py:468-476`) — no TOCTOU window, minimal
  new code. A model-supplied `expected_sha256` wins (explicit intent). The
  `ABSENT` case maps to the CAS's absent-precondition equivalent, or a provider
  pre-check if none exists.
- **fs_edit / fs_patch**: no CAS parameter exists, so the provider pre-hashes
  each target before dispatching the handler (the acknowledged tiny TOCTOU
  window is acceptable for optimistic locking).
- **fs_patch is MULTI-TARGET**: `parse_patch_targets` yields many files; check
  EVERY target and refuse the whole patch if ANY is stale (the provider
  preflight already loops the plans, `local_tool_provider.py:988-996`).

### Update (on successful write)
On a successful `fs_write`/`fs_edit`/`fs_patch`, update `(run_id, path)` to the
just-written content's hash, so the agent's own read→write→write chain never
false-positives. Review note: for `fs_edit`/`fs_patch` the handler itself
reads+modifies+writes, so the Update hash is a POST-HANDLER RE-READ of the file
(the provider cannot know the written bytes up front); for `fs_write` the hash
can be computed from the content argument directly. fs_patch updates every
patched target.

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
  and no-false-positive-on-own-rewrite tests. Review confirmed no thread
  harness is needed: bind identities sequentially with
  `run_context.use_run_id("A")` / `use_run_id("B")` against the ONE shared
  provider (precedent: `Tests/Agents/test_local_tool_provider.py:28,115`) —
  the ledger keying is what is under test.

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
- **Dispatch wiring (review-found hole): all children share ONE registry with
  ONE LocalToolProvider owning the fs tool names — a per-child
  `workspace_root` has nowhere to plug in today.** Needs a per-child registry,
  or a root that resolves per run_id. (The existing ROOT_CHANGED/root_identity
  guard is NOT the obstacle — a per-child provider pinned to its scratch never
  trips it.)
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
