# TASK-28238 — Parallel sub-agent safety: stale-write guard + worktree isolation (design)

Status: design approved 2026-09-02 (brainstorming), then hardened by an
adversarial design review against the code (same day; corrections folded in
below). Phases 1 and 2 implemented (phase 1 merged in PR #2341; phase 2 on
branch feat/task-28238-phase2-worktree-isolation).

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

Qodo round #2: an empty run_id is EXCLUDED from the ledger entirely, not just
capped — `build_server_local_provider` serves MANY independent MCP clients,
all with `current_run_id() == ""`; keying them into the shared `""` bucket
above would let one client's successful write refresh the stamp another
client read, masking that client's own stale overwrite (the exact failure
per-run keying exists to prevent, just reintroduced one level up). No run
identity therefore means no guard state at all — record, injection, pre-check,
and post-write update are all no-ops for `run_id == ""` — matching pre-feature
behavior for those callers. The per-run cap above now protects only real
fleet-run buckets, since the one bucket that could grow unboundedly for the
life of the process is never populated.

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

## Phase 2 — git-worktree isolation (resolved design, 2026-09-03)

Phase 1 merged (PR #2341). The sketch's open questions are now resolved; this
section is the binding phase-2 design (owner approved the shape and chose the
merge-back mechanism).

### Dispatch wiring (the sketch's blocker — RESOLVED, no per-child registry)
The provider already models multiple roots as `RunAdmittedWorkspaceRoot`
(each with its own `WorkspaceToolExecutor`, `allow_write`, and a revalidation
`guard`), selected in `_select_admitted_root`. Phase 2 adds a per-run agent
root map on `LocalToolProvider`:

- `admit_run_workspace_root(run_id: str, authority: RunAdmittedWorkspaceRoot)`
  and `retire_run_workspace_root(run_id: str)` — a lock-guarded
  `_agent_roots: dict[str, RunAdmittedWorkspaceRoot]`, SEPARATE from the
  constructor's alias map (no mixed-mode breakage).
- `_select_admitted_root` consults `_agent_roots.get(current_run_id())`
  FIRST: an isolated child's fs/git tool calls auto-route to its worktree
  authority — no `root_alias` argument for the model to remember. An explicit
  `root_alias` and unmapped runs behave exactly as today, so the parent,
  Console workspace bindings, and single-agent flows are untouched (phase-1
  AC#3 discipline).
- The phase-1 ledger needs no change: keys are `(run_id, resolved_path)` and
  the child's paths resolve inside its own tree.

### Worktree lifecycle
New module `Agents/agent_worktree.py`:
- `create_agent_worktree(repo_root, run_id)` →
  `git worktree add <scratch>/agent-<run8> -b agent/<run_id> HEAD`.
  **Base ref = HEAD at spawn** (decided). **Uncommitted shared-tree changes do
  NOT carry** into the worktree — a clean checkout; dirt belongs to the user
  (decided).
- Refuses cleanly (reason-coded) when the workspace root is not a git repo or
  git is unavailable — the spawn then fails honestly rather than silently
  sharing the tree (AC#1's "never silent").
- `remove_agent_worktree` (worktree remove; branch deleted on discard, kept
  after merge-back until discarded) and a GC sweep for worktrees left by
  crashed/abandoned children.

### Spawn surface
`spawn_subagent` gains `isolation: "worktree"` (default off / absent =
today's shared tree). On an isolated spawn, agent_service: create worktree →
build the authority (executor bound to the worktree; guard revalidates the
worktree still exists) → `admit_run_workspace_root(child_run_id, ...)` when
the child run id is bound → child runs entirely inside its tree →
`retire_run_workspace_root` at child finish (worktree itself survives until
merged or discarded).

### Merge-back — explicit, never silent (owner decision: BOTH modes)
A parent-invocable, `mutates`-tagged tool (floored to ask → approval card):
`merge_agent_worktree(handle_id, mode="apply"|"merge")`, default `apply`:
- `mode="apply"`: 3-way-apply the child branch's diff into the shared working
  tree, leaving everything UNCOMMITTED for user review/commit. The
  `agent/<run_id>` branch survives as backup until discarded.
- `mode="merge"`: a real `git merge --no-ff agent/<run_id>` into the current
  branch — an actual merge commit.
- BOTH refuse on conflict (including git's own dirty-tree overlap refusal for
  merge mode), naming the conflicting files; nothing is ever half-applied
  silently. Result includes a diffstat of what landed (or would land).
- `discard_agent_worktree(handle_id)` removes worktree + branch without
  applying anything. **Per-child** merge-back (decided); batching can compose
  later.

### Tests (phase 2)
- Wiring: an admitted run routes fs tools to the worktree root (write lands in
  the worktree, not the shared tree); unmapped runs unchanged; retire restores.
- Lifecycle: create/remove/GC round-trip on a real temp git repo; non-git root
  refuses with the reason.
- Merge-back: apply mode lands the diff uncommitted; merge mode creates the
  merge commit; conflicting change in the shared tree → refusal naming files,
  tree untouched; discard removes everything.
- Spawn integration: isolated child's writes are invisible in the shared tree
  until merge-back; explicit merge-back lands them (AC#1).

## How the phases compose
The guard protects the shared tree that all fleet children use today (phase 1,
immediate value, no git dependency). Worktrees add true isolation for children
that opt in (phase 2); within a worktree the guard is belt-and-suspenders. Both
key on `(run_id, resolved_path)` under whatever root the child's provider has,
so they are correct in both shared-tree and isolated modes.
