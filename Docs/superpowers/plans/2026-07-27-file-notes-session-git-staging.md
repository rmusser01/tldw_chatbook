# File Notes Session Git Staging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a session-only File Notes Git view that shows, stages, and safely unstages only paths changed by Chatbook while files remain authoritative on disk and SQLite remains an independent replica.

**Architecture:** Move the raw File Notes change log into one process-memory owner shared by fresh Library screens. The same owner provides one atomic lease coordinator: Git mutation and root/path/source/screen-transition admissions are mutually exclusive; status admitted first may finish while mutation waits, but mutation admission blocks every later status start; editing/autosave/replica work never takes a lease. Attach one optional, standard-library Git service to the owner for hardened repository discovery, status, exact Stage/Unstage ownership, and retained child-process lifecycle; the workspace remains a presentation and autosave coordinator. Add a third retained Navigator mode for Session Git without introducing commit/push behavior, a VCS abstraction, a database migration, or a new dependency.

**Tech Stack:** Python 3.11+, Textual 8, `asyncio.create_subprocess_exec`, Git porcelain v2/index plumbing, frozen dataclasses, SQLite boundary spies, pytest/pytest-asyncio.

**Backlog:** [TASK-1213](../../../backlog/tasks/task-1213%20-%20Add-session-scoped-Git-status-and-staging-to-File-Notes.md)

**Specification:** [File Notes Session Git Staging Design](../specs/2026-07-27-file-notes-session-git-staging-design.md)

**Depends on:** TASK-969, TASK-982

**ADR required:** yes

**ADR path:** `backlog/decisions/035-file-notes-session-git-index-controls.md`

**Reason:** ADR-035 already fixes the Git index, trust, lifecycle, and UX boundary; ADR-033 governs the app-session owner. No additional ADR is needed.

---

## Execution Environment and Scope

Run every command from this worktree:

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/file-notes-move-tombstone
```

The worktree intentionally has no local `.venv`. Use the main checkout's
verified environment explicitly:

```bash
../../.venv/bin/python -c "import pathlib, tldw_chatbook; print(pathlib.Path(tldw_chatbook.__file__).resolve())"
../../.venv/bin/python -c "import textual, pytest; print(textual.__version__, pytest.__version__)"
../../.venv/bin/python -m ruff --version
git --version
```

Expected: the package path resolves inside this worktree; the current verified
versions are Python 3.12.11, Textual 8.2.7, pytest 8.4.2, Ruff 0.15.22, and Git
2.39.5.

This task deliberately uses focused verification. Do not run full CI, the full
pytest suite, network/remotes, a broad performance harness, or a hung-process
test. New test modules that import the application stack must install the
existing `parakeet_mlx` module stub before importing Chatbook.

## File Structure

- Create `tldw_chatbook/Notes/file_notes_session_owner.py`: immutable session/Git state models, root-generation binding, thread-safe sequenced change log, trust/status/ownership publication, atomic transition/status/mutation leases, and attached-service shutdown.
- Create `tldw_chatbook/Notes/file_notes_git_service.py`: session grouping, Git parsers and policy, sanitized direct-argv runner, repository identity, status coalescing, exact Stage/Unstage orchestration, and the production owner builder.
- Modify `tldw_chatbook/Notes/file_notes_service.py`: preserve the public `SessionChange` import while publishing successful Chatbook mutations to an injected owner binding instead of a service-local list.
- Create `tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py`: Session Git rows/actions and the safe-focus trust modal; it contains presentation only.
- Modify `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`: inject/bind the owner, retain Files/search/Git navigator modes, schedule visible-only refresh, flush before Stage, and enforce the separate structural/leave gate.
- Modify `tldw_chatbook/UI/Screens/library_screen.py`: inject the app owner through the default zero-argument workspace factory while preserving test factories.
- Modify `tldw_chatbook/app.py`: construct the owner beside the existing process owners and shut it down before Textual closes screens, with an idempotent unmount fallback.
- Create `Tests/Notes/test_file_notes_session_owner.py`: owner generation, concurrency, publication, trust/ownership reset, mutation gate, and shutdown tests.
- Create `Tests/Notes/test_file_notes_git_service.py`: pure grouping/parser/policy/command/ownership tests plus delayed-runner concurrency tests.
- Create `Tests/Notes/test_file_notes_git_integration.py`: compact disposable-repository and configured-filter tests.
- Create `Tests/UI/test_library_file_notes_git.py`: mounted Session Git mode, trust, keyboard, actions, narrow-layout, pending-work, and scale tests.
- Modify `Tests/Notes/test_file_notes_service.py`: injected-recorder and existing session-log regressions.
- Modify `Tests/UI/test_library_file_notes_workspace.py`: replace raw-summary assertions and preserve existing File Notes/editor/replica behavior.
- Modify `Tests/UI/test_screen_navigation.py`: fresh Library screen owner continuity and navigation-veto coverage.
- Create `Tests/ProductionApp/test_file_notes_session_owner_lifecycle.py`: real app teardown ordering with and without a mounted Library.

Do not modify `FileNotesReplica`, its schema, Git configuration, repository
remotes, or note bytes from the Git service.

## Task 1: Move Session Changes Behind the Process Owner

**Files:**

- Create: `tldw_chatbook/Notes/file_notes_session_owner.py`
- Modify: `tldw_chatbook/Notes/file_notes_service.py:88-176, 616-618, 723, 1149-1160`
- Modify: `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py:148-224, 489-533, 706-722, 807-867`
- Create: `Tests/Notes/test_file_notes_session_owner.py`
- Modify: `Tests/Notes/test_file_notes_service.py`
- Modify: `Tests/UI/test_library_file_notes_workspace.py`

- [ ] **Step 1: Write failing owner and recorder tests**

Cover these exact invariants:

```python
def test_same_root_keeps_session_and_different_root_resets_it(tmp_path: Path) -> None:
    owner = FileNotesSessionOwner()
    first = owner.select_root(tmp_path / "a")
    owner.record_change(first, SessionChange("modified", "one.md"))

    assert owner.select_root(tmp_path / "a") == first
    assert [item.change.relative_path for item in owner.snapshot(first).changes] == [
        "one.md"
    ]

    second = owner.select_root(tmp_path / "b")
    assert second.generation == first.generation + 1
    assert owner.snapshot(second).changes == ()
    assert owner.record_change(first, SessionChange("modified", "late.md")) is False


def test_recorder_assigns_one_monotonic_sequence_under_threads(tmp_path: Path) -> None:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")
    with ThreadPoolExecutor(max_workers=4) as pool:
        accepted = list(
            pool.map(
                lambda number: owner.record_change(
                    binding,
                    SessionChange("modified", f"{number}.md"),
                ),
                range(40),
            )
        )

    snapshot = owner.snapshot(binding)
    assert all(accepted)
    assert [item.sequence for item in snapshot.changes] == list(range(1, 41))
    assert len({item.change.relative_path for item in snapshot.changes}) == 40
```

Also test that a stale binding cannot publish a session change, same-root
workspace reconstruction keeps the sequenced log, shutdown is idempotent, and
no owner state is persisted. Task 2 adds the corresponding trust/status/index
ownership reset assertions once those typed records exist.

Add an owner-only admission test:

```python
transition = owner.try_acquire_transition(binding, "root")
assert transition is not None
assert owner.try_acquire_mutation(binding) is None
transition.release()

mutation = owner.try_acquire_mutation(binding)
assert mutation is not None
assert owner.try_acquire_transition(binding, "screen") is None
assert owner.try_acquire_status(binding) is None
mutation.release()

status = owner.try_acquire_status(binding)
assert status is not None
waiting_mutation = owner.try_acquire_mutation(binding)
assert waiting_mutation is not None
assert owner.try_acquire_status(binding) is None
status.release()
waiting_mutation.release()
```

Also prove stale bindings cannot acquire either lease, releases are idempotent,
and root selection cannot accidentally clear a live transition token before
its holder releases it.

Extend the service tests to inject one owner/binding and assert:

- create, modify, move, delete, and restore record only after disk success;
- move records one source/destination change;
- `reconcile()` never records external changes;
- a late result using the old generation is ignored;
- the existing `service.session_changes` compatibility property still returns
  unsequenced `SessionChange` values.

- [ ] **Step 2: Run the tests and verify the owner is absent**

```bash
../../.venv/bin/python -m pytest -q Tests/Notes/test_file_notes_session_owner.py Tests/Notes/test_file_notes_service.py
```

Expected: FAIL during import because `file_notes_session_owner.py` does not
exist.

- [ ] **Step 3: Implement the minimal owner and binding**

Use immutable public snapshots and a private `RLock` because
`FileNotesService` records from `asyncio.to_thread`:

```python
@dataclass(frozen=True, slots=True)
class SessionChange:
    action: Literal["created", "modified", "moved", "deleted", "restored"]
    relative_path: str
    destination_path: str | None = None


@dataclass(frozen=True, slots=True)
class SessionBinding:
    root_key: str
    generation: int


@dataclass(frozen=True, slots=True)
class SequencedSessionChange:
    sequence: int
    change: SessionChange


FileNotesSessionOwner.select_root(root: str | Path) -> SessionBinding
FileNotesSessionOwner.record_change(
    binding: SessionBinding,
    change: SessionChange,
) -> bool
FileNotesSessionOwner.snapshot(
    binding: SessionBinding,
) -> FileNotesSessionSnapshot
FileNotesSessionOwner.try_acquire_transition(
    binding: SessionBinding,
    kind: Literal["root", "path", "source", "screen"],
) -> SessionTransitionLease | None
FileNotesSessionOwner.try_acquire_mutation(
    binding: SessionBinding,
) -> GitMutationLease | None
FileNotesSessionOwner.try_acquire_status(
    binding: SessionBinding,
) -> GitStatusLease | None
FileNotesSessionOwner.shutdown() -> None
```

Canonicalize the root once in `select_root()`. Increment the generation and
clear all root-scoped state only when that canonical root changes. Keep no
filesystem watcher and do not accept external reconcile records.

Define a narrow lifecycle `Protocol` in the owner module so an attached Git
service can be shut down without importing the concrete service and creating a
circular dependency. The owner remains the sole holder of its attached
service. `select_root()`, snapshots, and publications use the same lock as
`record_change()` because initial runtime construction and filesystem
operations already run in worker threads; only the async mutation lock and
child-task coordination are event-loop-affine.

All lease admissions must be non-awaiting checked operations under that same
owner lock. A transition lease blocks mutation admission; a mutation lease
blocks transition and new status admission. A status lease acquired first does
not block mutation admission: the mutation lease is granted immediately, then
its retained task waits for the already-admitted status task while all later
status admissions fail. Lease objects have one idempotent `release()` and
remain valid for release even if a root transition advances the selected-root
generation. Shutdown seals every admission. Do not build this from a check
followed by a later async lock acquisition.

- [ ] **Step 4: Delegate `FileNotesService` publication**

Import and re-export `SessionChange` from the owner module. Extend the service
constructor with optional `session_owner` and `session_binding`. When omitted,
create a private owner bound to `self.root` so direct callers keep the existing
session property behavior.

Replace each `_session_changes.append(...)` with one private helper:

```python
def _record_session_change(self, change: SessionChange) -> None:
    self._session_owner.record_change(self._session_binding, change)
```

Keep the helper at the same successful publication points. Do not move it
before a disk write or into replica reconciliation.

- [ ] **Step 5: Bind the workspace without changing replica ownership**

Add `session_owner: FileNotesSessionOwner | None = None` to the workspace. A
directly constructed workspace owns a private owner; a production-injected
workspace does not. `_build_runtime()` selects/binds the canonical root and
passes that binding to every replacement `FileNotesService`. A root generation
held by an old service must not be able to append after `set_root()`. Workspace
shutdown awaits only a private owner; it must never close an injected app owner.

For this task, replace `_refresh_session_changes()` with a bounded
`Session Git (N)` count using the temporary count of raw sequenced changes.
Task 2 will switch `N` to coalesced groups and Task 6 will make the entry
interactive. Do not render the old semicolon-separated list.

- [ ] **Step 6: Run focused owner, service, and existing workspace tests**

```bash
../../.venv/bin/python -m pytest -q Tests/Notes/test_file_notes_session_owner.py Tests/Notes/test_file_notes_service.py Tests/UI/test_library_file_notes_workspace.py
```

Expected: PASS. Existing disk/replica behavior and the single retained editor
remain unchanged.

- [ ] **Step 7: Commit the session owner**

```bash
git add tldw_chatbook/Notes/file_notes_session_owner.py tldw_chatbook/Notes/file_notes_service.py tldw_chatbook/Widgets/Library/library_file_notes_workspace.py Tests/Notes/test_file_notes_session_owner.py Tests/Notes/test_file_notes_service.py Tests/UI/test_library_file_notes_workspace.py
git commit -m "refactor(notes): retain file session changes in app owner [TASK-1213]"
```

## Task 2: Define Session Groups, Git Records, and Eligibility

**Files:**

- Modify: `tldw_chatbook/Notes/file_notes_session_owner.py`
- Create: `tldw_chatbook/Notes/file_notes_git_service.py`
- Create: `Tests/Notes/test_file_notes_git_service.py`

- [ ] **Step 1: Write failing pure grouping and parser tests**

Parameterize:

- repeated edits to one path keep the earliest sequence as group ID;
- create/delete and delete/restore stay visible even when later clean;
- move source/destination are inseparable;
- a chained move retains all endpoints and displays original source/final
  destination;
- a later destination edit stays in that move group;
- reuse of an old move source creates a new group rather than joining the old
  lineage;
- only active-path mapping is used to extend a group.

Add byte-oriented porcelain-v2 cases for ordinary `1`, rename `2`, unmerged
`u`, untracked `?`, and ignored `!` records, including spaces, leading dashes,
tabs/newlines, non-UTF-8 filesystem bytes, and pathspec characters. Parsed
paths outside the supplied session whitelist must fail the complete result.
Chatbook move grouping must ignore Git rename pairing because status will run
with rename detection disabled.

- [ ] **Step 2: Write failing row-policy and closure tests**

Use frozen inputs to cover the specification's entire row/action table:
unstaged, Chatbook-owned, owned with newer unstaged edits, owned with changed
topology, externally staged, partially externally staged, clean, ignored,
conflict, unsupported semantic flags, nested repository, unavailable, and
error.

Test pure helpers for:

- tracked ancestor/descendant Stage closure;
- current-index file/directory replacement closure before Unstage;
- transient move endpoints omitted from mutation pathspecs;
- a closure path outside the same session lineage blocking the whole group;
- `skip-worktree`, `assume-unchanged`, and zero-object intent-to-add blocking;
- exact mode/object/stage/flag signatures;
- detached and explicit unborn `HEAD` identities.

- [ ] **Step 3: Run the pure tests and verify the module is absent**

```bash
../../.venv/bin/python -m pytest -q Tests/Notes/test_file_notes_git_service.py
```

Expected: FAIL during import because `file_notes_git_service.py` does not
exist.

- [ ] **Step 4: Add the immutable Git state stored by the owner**

Keep the source of truth in `file_notes_session_owner.py`. Add frozen, slotted
models for:

```python
FileSystemIdentity
RepositoryIdentity
SessionChangeGroup
IndexEntry
IndexBaseline
StagingOwnership
SessionGitRow
SessionGitStatus
```

`StagingOwnership` must contain repository identity, `HEAD`/unborn identity,
approved endpoint topology, original per-entry baselines, and exact post-Stage
entries/semantic flags. `SessionGitStatus` must carry binding generation and a
monotonic status generation so stale results can be rejected.

Add checked owner methods to publish/clear trust, status, and ownership. They
must accept the current `SessionBinding` and reject a stale generation under
the owner's lock. Do not expose mutable dictionaries.

Extend `Tests/Notes/test_file_notes_session_owner.py` here to prove a root
change atomically clears trust, status, and staging ownership; a stale binding
cannot republish any of them; and same-root fresh workspaces retain them.

- [ ] **Step 5: Implement the pure grouping/parser/policy layer**

In `file_notes_git_service.py`, implement:

```python
coalesce_session_changes(
    changes: Sequence[SequencedSessionChange],
) -> tuple[SessionChangeGroup, ...]
parse_porcelain_v2_z(
    payload: bytes,
    *,
    allowed_paths: frozenset[str],
) -> tuple[PorcelainRecord, ...]
compute_stage_closure(
    endpoints: Collection[str],
    index_entries: Mapping[str, IndexEntry],
) -> frozenset[str]
classify_session_rows(
    groups: Sequence[SessionChangeGroup],
    status_records: Sequence[PorcelainRecord],
    index_entries: Mapping[str, IndexEntry],
    ownership: Mapping[int, StagingOwnership],
) -> tuple[SessionGitRow, ...]
```

Treat repository paths as canonical POSIX strings internally, encode command
arguments with `os.fsencode`, decode Git path bytes with `os.fsdecode`, and
sanitize control characters only when building display text. Do not reject
valid filenames merely because they contain shell metacharacters; the runner
will never use a shell.

- [ ] **Step 6: Run pure tests and update the workspace count**

```bash
../../.venv/bin/python -m pytest -q Tests/Notes/test_file_notes_git_service.py Tests/Notes/test_file_notes_session_owner.py
```

Expected: PASS.

Change the workspace's compact count to
`len(coalesce_session_changes(owner.snapshot(binding).changes))`.

- [ ] **Step 7: Commit grouping and policy**

```bash
git add tldw_chatbook/Notes/file_notes_session_owner.py tldw_chatbook/Notes/file_notes_git_service.py tldw_chatbook/Widgets/Library/library_file_notes_workspace.py Tests/Notes/test_file_notes_git_service.py
git commit -m "feat(notes): model session Git rows and eligibility [TASK-1213]"
```

## Task 3: Add Hardened Repository Discovery and Coalesced Status

**Files:**

- Modify: `tldw_chatbook/Notes/file_notes_git_service.py`
- Modify: `tldw_chatbook/Notes/file_notes_session_owner.py`
- Create: `Tests/Notes/test_file_notes_git_integration.py`
- Modify: `Tests/Notes/test_file_notes_git_service.py`

- [ ] **Step 1: Write failing runner, identity, and command tests**

With a fake subprocess runner, assert:

- direct argv only and `shell` is never accepted;
- stdin/stdout/stderr remain bytes;
- redirecting/injection variables are removed, including fixed variables
  (`GIT_DIR`, `GIT_WORK_TREE`, `GIT_COMMON_DIR`, `GIT_INDEX_FILE`,
  `GIT_OBJECT_DIRECTORY`, `GIT_ALTERNATE_OBJECT_DIRECTORIES`, `GIT_NAMESPACE`,
  `GIT_CEILING_DIRECTORIES`, `GIT_DISCOVERY_ACROSS_FILESYSTEM`,
  `GIT_SHALLOW_FILE`, `GIT_GRAFT_FILE`, `GIT_REPLACE_REF_BASE`,
  `GIT_NO_REPLACE_OBJECTS`, `GIT_EXEC_PATH`, `GIT_PREFIX`,
  `GIT_CONFIG_SYSTEM`, `GIT_CONFIG_GLOBAL`, `GIT_CONFIG_NOSYSTEM`,
  `GIT_CONFIG_PARAMETERS`, `GIT_GLOB_PATHSPECS`, `GIT_NOGLOB_PATHSPECS`,
  `GIT_LITERAL_PATHSPECS`, `GIT_ICASE_PATHSPECS`) and dynamic
  `GIT_CONFIG_COUNT`/`GIT_CONFIG_KEY_*`/`GIT_CONFIG_VALUE_*` families;
- ordinary environment/configuration remains available for attributes and
  filters;
- `GIT_TERMINAL_PROMPT=0` is forced;
- status also sets `GIT_OPTIONAL_LOCKS=0`, disables fsmonitor hooks and rename
  detection, and requests porcelain v2, NUL output, all untracked files, and
  matching ignored files;
- every path command uses `--literal-pathspecs` and an explicit `--`;
- stderr is bounded and control-character sanitized;
- a status timeout terminates, performs a bounded wait, kills if required,
  performs a second bounded wait, and returns stale/error rather than looping.

- [ ] **Step 2: Write failing disposable-repository discovery/status tests**

Create one local helper inside
`Tests/Notes/test_file_notes_git_integration.py` using `tmp_path`,
`shutil.which("git")`, and direct `subprocess.run`. Give each fixture a private
HOME/global-config path and local author identity; never read or mutate the
developer's Git configuration.

Cover repository root equal to and above the notes root, a linked worktree with
distinct worktree Git dir/common dir, detached and unborn `HEAD`, non-repository
roots, a replaced Git directory identity, and supported filenames. Assert the
status result contains only requested session paths.

- [ ] **Step 3: Run focused tests and verify status is unimplemented**

```bash
../../.venv/bin/python -m pytest -q Tests/Notes/test_file_notes_git_service.py -k "runner or environment or identity or status or coalesc"
../../.venv/bin/python -m pytest -q Tests/Notes/test_file_notes_git_integration.py -k "discover or status or linked or filename"
```

Expected: FAIL on the new runner/service APIs.

- [ ] **Step 4: Implement discovery and identity revalidation**

Use `asyncio.create_subprocess_exec` through one injectable runner. Discovery
may use `rev-parse` commands that do not inspect worktree content. Capture
canonical worktree top, worktree-specific Git directory, common directory, and
`stat(..., follow_symlinks=False)` device/inode identity for each.

The selected notes root must be equal to or contained by the discovered
worktree. Revalidate all three paths immediately after trust and before each
worktree-aware status or mutation. An identity change clears the owner's trust,
status, and ownership for that binding.

Derive each command path by joining the validated File Notes relative path to
the selected root, proving containment, and then expressing it relative to the
canonical worktree top. This prefix conversion is required when the repository
root is above the notes root. A notes root that is itself a submodule worktree
uses that worktree; a path crossing into a nested worktree beneath the selected
root is blocked.

Read the branch display with machine-safe symbolic-ref/`HEAD` commands and
represent normal, detached, and unborn states explicitly; do not parse
localized `git status` headers.

Fail closed on:

- missing/unsupported Git;
- unsafe or non-directory root;
- active sparse checkout or sparse index;
- a session endpoint crossing a nested `.git` file/directory boundary;
- symlink, directory, or non-regular endpoint types;
- repository identity replacement.

Do not initialize/repair a repository or change `safe.directory`.
Capability-check the required Git features/commands and report unsupported Git
instead of parsing a human-readable fallback.

- [ ] **Step 5: Implement trust-aware, retained status scheduling**

Expose a narrow service contract:

```python
FileNotesGitService.discover(
    binding: SessionBinding,
) -> DiscoveryResult
FileNotesGitService.start_status(
    binding: SessionBinding,
    changes: tuple[SequencedSessionChange, ...],
) -> asyncio.Task[SessionGitStatus]
FileNotesGitService.shutdown() -> None


def build_file_notes_session_owner() -> FileNotesSessionOwner:
    owner = FileNotesSessionOwner()
    owner.attach_git_service(FileNotesGitService(owner))
    return owner
```

`start_status()` is synchronous. It must refuse worktree-aware status without a
matching owner trust grant, acquire a status lease before creating a child
task, and raise a typed `mutation_active` admission result without creating a
task when a mutation lease already exists. Release the status lease
immediately if retained-task creation fails. Retain the service-owned task and
await it through `asyncio.shield`; cancellation of one mounted UI waiter must
not cancel the query.

Allow one active status plus at most one latest-snapshot rerun. Multiple
triggers while status is active replace that one snapshot and return the same
retained cycle task. Before spawning the rerun child, release the completed
query's status lease and acquire a fresh one. If mutation was admitted in the
meantime, the new lease fails: clear the service rerun, publish stale, and
start no child. The mutation task awaits the current status cycle, which now
finishes after the already-running query instead of beginning its rerun.

Do not store a service-level deferred request received during mutation, add a
polling timer, or add an automatic retry loop. The visible workspace owns one
post-mutation refresh bit and will call `start_status()` with the latest owner
snapshot only after postflight. Publish only when binding and status
generations remain current.

Read the complete index once with a NUL-safe `ls-files` form so semantic flags
and ancestor/descendant closure can be evaluated in Python. Reject a porcelain
record outside the requested session/approved-closure whitelist.

- [ ] **Step 6: Add delayed-runner scheduling and shutdown tests**

Prove:

- ten triggers during one blocked status produce that query plus at most one
  rerun with the latest snapshot;
- status admitted first may finish after mutation admission, but a pending
  rerun obtains no second lease/child and the mutation proceeds afterward;
- status requested after mutation admission returns `mutation_active`, creates
  no task/child, and stores no service rerun;
- cancelled UI waiters do not cancel service completion/publication;
- `shutdown()` seals new admission and uses finite terminate/kill waits;
- inability to confirm child termination publishes uncertainty and no
  ownership;
- no code deletes `.git/index.lock`.

- [ ] **Step 7: Run the status foundation tests**

```bash
../../.venv/bin/python -m pytest -q Tests/Notes/test_file_notes_git_service.py
../../.venv/bin/python -m pytest -q Tests/Notes/test_file_notes_git_integration.py -k "discover or status or linked or filename or sparse or nested"
```

Expected: PASS.

- [ ] **Step 8: Commit discovery and status**

```bash
git add tldw_chatbook/Notes/file_notes_session_owner.py tldw_chatbook/Notes/file_notes_git_service.py Tests/Notes/test_file_notes_git_service.py Tests/Notes/test_file_notes_git_integration.py
git commit -m "feat(notes): add trusted session-path Git status [TASK-1213]"
```

## Task 4: Implement Exact Stage and Stage Update Ownership

**Files:**

- Modify: `tldw_chatbook/Notes/file_notes_git_service.py`
- Modify: `tldw_chatbook/Notes/file_notes_session_owner.py`
- Modify: `Tests/Notes/test_file_notes_git_service.py`
- Modify: `Tests/Notes/test_file_notes_git_integration.py`

- [ ] **Step 1: Write failing Stage preflight and ownership tests**

Cover per-group and bulk Stage for modify, create, delete, restore, executable
mode, grouped move, and chained move. Assert:

- pre-existing/partially staged same-path state, conflict, ignored, nested,
  sparse, unsafe type, semantic flag, or an out-of-lineage closure blocks the
  whole group before mutation;
- clean groups are reported but omitted;
- transient move endpoints absent from `HEAD`, index, and worktree are not
  passed as unmatched pathspecs;
- one bulk action includes only eligible effective paths;
- the command is exactly one path-scoped `git add --all` operation with
  `-c add.ignoreErrors=false`, literal pathspecs, and `--`;
- a nonzero/uncertain result claims no new ownership;
- unrelated worktree and staged/index state remains byte-for-byte unchanged.

- [ ] **Step 2: Write failing Stage update and race tests**

Start with a Chatbook-owned group, make newer unstaged edits, and assert Stage
update:

- requires every saved post-Stage entry still to match;
- retains every existing entry's earliest original baseline;
- captures a baseline only for newly affected entries that are clean at fresh
  preflight;
- expands move topology only after the update succeeds;
- never invents ownership for a no-op endpoint;
- loses/revokes ownership when repository identity, `HEAD`, index entry,
  topology precondition, or semantic flag changes;
- publishes no ownership if the postflight result is uncertain.

Use a delayed fake runner to alter the observed index between preflight and
postflight and assert fail-closed behavior without claiming cross-process
atomicity. Also block one active status query and prove Stage waits for it
rather than running concurrently while its already-acquired mutation lease
prevents a root/path/source/screen transition. Add the inverse race: hold a
transition lease before Stage admission and assert no retained mutation task or
Git child starts.

- [ ] **Step 3: Run tests and verify Stage is absent**

```bash
../../.venv/bin/python -m pytest -q Tests/Notes/test_file_notes_git_service.py -k "stage and not unstage"
../../.venv/bin/python -m pytest -q Tests/Notes/test_file_notes_git_integration.py -k "stage and not unstage"
```

Expected: FAIL on the new Stage API.

- [ ] **Step 4: Implement the service-owned Stage lifecycle**

Add a synchronous admission API:

```python
FileNotesGitService.start_stage(
    binding: SessionBinding,
    group_ids: Collection[int],
) -> asyncio.Task[GitActionResult]
```

The workspace will flush first, but `start_stage()` must still:

1. acquire the owner's mutation lease for the exact binding before returning
   or scheduling any other callback;
2. transfer that lease immediately to the retained service task, releasing it
   in the caller only if task creation fails;
3. wait for an active status while still holding the mutation lease;
4. snapshot current groups and freshly revalidate repository identity;
5. run fresh status/index/semantic/closure preflight;
6. save baselines for entries the action can actually change;
7. run one exact Stage command;
8. re-read `HEAD`, index entries, flags, identity, and topology;
9. atomically publish ownership only for fully verified results and mark the
   prior status stale for the latest binding;
10. release the lease after postflight in the retained task's `finally`.

Retain the operation task on the service and shield UI awaiters. A slow clean
filter may update elapsed UI state but must not be killed during normal app
operation. The service must never perform a preliminary “gate is free” check
and then acquire it after awaiting status. If admission fails,
`start_stage()` raises a typed busy/stale-binding result before creating a
task. The workspace catches that synchronously; otherwise it gives the returned
task to a Textual worker that awaits it through `asyncio.shield`, allowing the
button handler to return so editor and Back messages continue to run.

- [ ] **Step 5: Add real-repository Stage acceptance cases**

For the primary fixture, inspect:

```bash
git diff -- <session paths>
git diff --cached -- <session paths>
```

before and after Stage. Include root-equal/root-above, grouped moves, unrelated
staged files, file/directory collision blocking, path characters, and local
`add.ignoreErrors=true` overridden by Chatbook's command.

Spy on the replica and owner recorder during each Git action: no replica method
and no `record_change()` call may occur.

- [ ] **Step 6: Run Stage tests**

```bash
../../.venv/bin/python -m pytest -q Tests/Notes/test_file_notes_git_service.py -k "stage and not unstage"
../../.venv/bin/python -m pytest -q Tests/Notes/test_file_notes_git_integration.py -k "stage and not unstage"
```

Expected: PASS.

- [ ] **Step 7: Commit Stage**

```bash
git add tldw_chatbook/Notes/file_notes_session_owner.py tldw_chatbook/Notes/file_notes_git_service.py Tests/Notes/test_file_notes_git_service.py Tests/Notes/test_file_notes_git_integration.py
git commit -m "feat(notes): stage exact session path groups [TASK-1213]"
```

## Task 5: Implement Exact Saved-Baseline Unstage

**Files:**

- Modify: `tldw_chatbook/Notes/file_notes_git_service.py`
- Modify: `tldw_chatbook/Notes/file_notes_session_owner.py`
- Modify: `Tests/Notes/test_file_notes_git_service.py`
- Modify: `Tests/Notes/test_file_notes_git_integration.py`

- [ ] **Step 1: Write failing signature and index-info tests**

Assert Unstage is eligible only when repository identity, `HEAD`/unborn marker,
approved topology, every exact post-Stage index entry, and semantic flags still
match. A topology change must expose Stage update and disable Unstage.

Test the exact NUL stream builder:

```python
payload = build_update_index_payload(ownership, current_index)
assert payload == (
    b"0 " + ZERO_OID + b"\\towned/conflict\\0"
    b"100644 " + ORIGINAL_OID + b" 0\\ttracked.md\\0"
    b"0 " + ZERO_OID + b"\\tcreated.md\\0"
)
```

Expected owned file/directory conflicts must be removed first. An unexpected or
external index ancestor/descendant must block before stdin is written.

- [ ] **Step 2: Write failing real Unstage cases**

Cover:

- normal, detached, and unborn `HEAD`;
- modify/create/delete/restore/mode and grouped moves;
- owned file-to-directory and directory-to-file replacement reversal;
- unexpected external replacement closure remaining unchanged;
- external index or `HEAD` change revoking ownership;
- newer unstaged edits remaining in the worktree;
- Stage update followed by Unstage restoring the original pre-first-Stage
  baseline;
- selected and bulk Unstage including only valid owned groups.

Assert Unstage never invokes checkout, restore, reset, read-tree, or any other
worktree-oriented/path-broadening command.

- [ ] **Step 3: Run tests and verify Unstage is absent**

```bash
../../.venv/bin/python -m pytest -q Tests/Notes/test_file_notes_git_service.py -k unstage
../../.venv/bin/python -m pytest -q Tests/Notes/test_file_notes_git_integration.py -k unstage
```

Expected: FAIL on the new Unstage API.

- [ ] **Step 4: Implement exact Unstage**

Add the synchronous admission twin:

```python
FileNotesGitService.start_unstage(
    binding: SessionBinding,
    group_ids: Collection[int],
) -> asyncio.Task[GitActionResult]
```

Use the same synchronous lease admission and retained operation lifecycle as
Stage. Freshly compare the complete saved signature, derive current index
replacement closure, and build one `git update-index -z --index-info` stdin
payload. Emit removals before baseline insertions; represent an original
absence with a mode-zero removal. Never reconstruct from live `HEAD`.

After the command, re-read exact index state. Consume ownership only when the
baseline is verified, revoke it on an observable mismatch, and report
uncertainty without claiming success after an ambiguous external race.

- [ ] **Step 5: Run all core Git tests**

```bash
../../.venv/bin/python -m pytest -q Tests/Notes/test_file_notes_git_service.py Tests/Notes/test_file_notes_git_integration.py
```

Expected: PASS.

- [ ] **Step 6: Commit Unstage**

```bash
git add tldw_chatbook/Notes/file_notes_session_owner.py tldw_chatbook/Notes/file_notes_git_service.py Tests/Notes/test_file_notes_git_service.py Tests/Notes/test_file_notes_git_integration.py
git commit -m "feat(notes): unstage owned baselines exactly [TASK-1213]"
```

## Task 6: Build the Session Git Navigator UX

**Files:**

- Create: `tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py`
- Modify: `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py:296-361, 618-771, 1064-1169, 1207-1514`
- Create: `Tests/UI/test_library_file_notes_git.py`
- Modify: `Tests/UI/test_library_file_notes_workspace.py`

- [ ] **Step 1: Write failing mounted navigator and row-action tests**

Mount the real workspace with an injected owner/fake Git service. Verify:

- the old unbounded summary is gone;
- `Session Git (N)` is focusable and `N` is the coalesced group count;
- opening it hides rather than unmounts Files/search widgets and shows a third
  retained navigator surface;
- `Back to Files` and Escape restore the prior Files/search mode and focus the
  Session Git entry;
- repository/branch, `Session paths only`, and complete-file-state labels are
  visible;
- every state in the approved action table has the correct selected/bulk
  action and disabled reason;
- selection follows stable earliest-sequence group IDs across refresh and move
  expansion;
- Up/Down select rows, Tab/Shift+Tab move focus, and Enter activates only the
  focused control;
- stale/error retains rows, disables mutations, and leaves Refresh enabled;
- untrusted shows only `Trust and check status`;
- checking/mutating disables Git mutation controls but not editor input or Back.
- session mutation while the Git view is hidden only marks status stale and
  invokes no Git service call; reopening starts one refresh.
- while Stage/Unstage is active, repeated manual/debounced/session-change
  refresh triggers start no status task and collapse to one postflight refresh
  using the latest owner snapshot if the Git view is still visible; hiding or
  unmounting before postflight starts none and leaves status stale.

- [ ] **Step 2: Write failing trust and narrow-layout tests**

The trust modal must show canonical repository path, process-only scope, and
the configured-filter warning. Assert initial focus is `#cancel-button`;
Escape, close, and Cancel all decline without a status/filter command; an
explicit retry can reopen it.

In narrow layout, switch Navigator/Editor before and after Session Git and
assert the same retained `TextArea`, body, cursor, selection, search query, tree
expansion, and Git row selection survive.

- [ ] **Step 3: Run UI tests and verify the panel is absent**

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_library_file_notes_git.py
```

Expected: FAIL during import because the panel does not exist.

- [ ] **Step 4: Implement the presentation-only panel**

Use a focusable `ListView` (or the repository's equivalent Textual list
pattern), buttons, and typed Textual messages:

```python
LibraryFileNotesGitPanel.BackRequested
LibraryFileNotesGitPanel.RefreshRequested
LibraryFileNotesGitPanel.TrustRequested
LibraryFileNotesGitPanel.StageRequested(group_ids: tuple[int, ...])
LibraryFileNotesGitPanel.UnstageRequested(group_ids: tuple[int, ...])
LibraryFileNotesGitPanel.render_status(status: SessionGitStatus) -> None
```

The panel must not execute Git, hold trust/ownership, or read the filesystem.
Subclass `ConfirmationDialog` for the trust prompt and focus
`#cancel-button` in `on_mount()`.

- [ ] **Step 5: Integrate three retained navigator modes**

Replace `#file-notes-session-changes` with the compact entry and mount the Git
panel once in `compose()`. Track the mode before opening Git so Back restores
Files versus search results exactly. Keep both existing trees and the editor
mounted.

The workspace owns:

- discover/prompt/accept/revalidate/open flow;
- selected row ID;
- visible-only immediate/manual/debounced refresh requests;
- one `_git_refresh_after_mutation` boolean, never a queued snapshot;
- status/action text;
- a single post-action refresh only while visible, otherwise stale marking.

Every refresh entry first checks the owner mutation lease. While mutation is
active, set the boolean and do not call `start_status()`. On postflight, clear
the boolean and read the latest owner change snapshot; if the panel is still
visible, synchronously call `start_status()` once and give its retained task to
a render worker. If hidden/unmounted, publish stale only. Do not attach Git
refresh to `_start_poll()`.

- [ ] **Step 6: Wire selected and bulk actions through the existing flush**

Stage handlers must:

1. await `flush_pending_work()`;
2. reject dirty/saving/conflict/error;
3. recheck root/path transition and the exact owner binding;
4. synchronously call `start_stage()`/`start_unstage()`, which either refuses
   admission without a task or returns an already-retained service task;
5. return the button handler after giving that task to a Textual render worker;
6. render the action counts/result when the shielded task settles.

Unstage uses the same binding/action path but does not require another editor
flush unless an autosave is pending. Both actions remain owned by the service
if the widget worker is cancelled.

After each successful File Notes session mutation, mark owner status stale.
Schedule one debounced refresh only if the Git panel is visible. If a Git
mutation lease is active when that debounce fires, it only sets the one
post-mutation refresh bit. The action render worker performs the same
visible/latest-snapshot check in `finally`, so its cancellation on unmount
cannot start hidden work and the service-owned mutation still settles.

- [ ] **Step 7: Add the selective mutation gate**

While `owner.mutation_active(binding)`:

- root switching and every `_hold_path_transition()` path operation return a
  concise busy status; this covers open/reload plus structural
  create/move/delete/restore/save-copy while the mutation uses its exact
  binding/snapshot;
- Library/workspace `flush_pending_work()` ultimately returns `False` so
  normal screen/source departure is vetoed;
- editor input, autosave, replica reconcile, protect/unprotect, and in-screen
  `Back to Files` remain usable;
- `_update_controls()` disables structural and Git mutation buttons without
  making the editor read-only.

Recheck the gate after any modal/await before beginning a structural action.
Do not reuse `_root_transitioning`, `_path_transitioning`, `_service_lock`, or
`_refresh_lock` as the Git gate.

Replace those check-only transition entries with the shared lease admission:

- `set_root()` obtains a `"root"` transition lease after its draft flush and
  before publishing `_root_transitioning`;
- `_hold_path_transition()` obtains a `"path"` transition lease before
  publishing `_path_transitioning`;
- each lease is held through the complete transition and released in `finally`;
- if mutation already owns the coordinator, acquisition fails with the busy
  status;
- if transition admission wins first, Stage/Unstage receives no mutation lease
  and starts no retained task.

All of these calls are non-awaiting owner operations. On the Textual event loop,
the final local-state recheck and owner lease acquisition therefore have no
yield point between them. Task 7 applies the same owner coordinator to source
and whole-screen departure.

- [ ] **Step 8: Run mounted UI and File Notes regressions**

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_library_file_notes_git.py Tests/UI/test_library_file_notes_workspace.py
```

Expected: PASS.

- [ ] **Step 9: Commit the Session Git UI**

```bash
git add tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py tldw_chatbook/Widgets/Library/library_file_notes_workspace.py Tests/UI/test_library_file_notes_git.py Tests/UI/test_library_file_notes_workspace.py
git commit -m "feat(library): add session Git navigator actions [TASK-1213]"
```

## Task 7: Wire Fresh Screens and Pre-Screen App Shutdown

**Files:**

- Modify: `tldw_chatbook/app.py:3594-3617, 7849-ongoing cleanup`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:1041-1055, 1526-1549, 1725-1773, 6780-6804`
- Modify: `Tests/UI/test_screen_navigation.py`
- Create: `Tests/ProductionApp/test_file_notes_session_owner_lifecycle.py`

- [ ] **Step 1: Write failing fresh-screen continuity tests**

Using the production `TldwCli` navigation path, assert:

- two separately constructed Library screens inject the same app owner but
  distinct workspace/editor/replica instances;
- leaving/reopening preserves rows, trust, status, and still-valid Unstage
  ownership;
- changing selected root clears them;
- constructing a new app treats a previously staged path as external;
- a running mutation makes existing `flush_pending_work()` navigation/source
  paths veto departure, and navigation succeeds after postflight.
- if navigation/source transition admission wins immediately after flush,
  Stage gets no mutation lease and starts no child; if Stage admission wins
  during the flush, navigation/source acquisition fails and leaves the current
  screen/source mounted.

- [ ] **Step 2: Write failing real teardown-order tests**

Use an actual `TldwCli.run_test()` lifecycle with an instrumented owner and
replica. Test both a mounted Library and no Library screen ever mounted:

```python
assert events.index("git-owner-settled") < events.index("replica-closed")
```

Also assert one shutdown call despite the fallback, bounded terminate then
kill for a retained child, no duplicate operation after forced workspace
unmount, no ownership on uncertain termination, and no index-lock deletion.

- [ ] **Step 3: Run lifecycle tests and verify the app has no owner**

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_screen_navigation.py -k "file_notes and (owner or git or mutation)"
../../.venv/bin/python -m pytest -q Tests/ProductionApp/test_file_notes_session_owner_lifecycle.py
```

Expected: FAIL because the owner is not constructed/injected/shut down.

- [ ] **Step 4: Construct and inject the process owner**

In `TldwCli.__init__`, beside `screen_state_store` and `pending_handoffs`:

```python
self.file_notes_session_owner = build_file_notes_session_owner()
```

Do not special-case only `_create_navigation_screen()`: startup has a separate
construction path. Instead, have `LibraryScreen`'s default workspace factory
capture `app_instance.file_notes_session_owner` and pass it to
`LibraryFileNotesWorkspace`. Preserve an explicitly supplied zero-argument
test factory.

`LibraryScreen.on_unmount()` continues shutting down its workspace/owned
replica, but the workspace must not shut down an injected app owner.

After the existing pending-work flush in `handle_screen_navigation()`, call an
optional synchronous screen hook with no intervening `await`:

```python
release_navigation = None
acquire_navigation = getattr(
    current_screen,
    "acquire_navigation_transition",
    None,
)
if callable(acquire_navigation):
    admission = acquire_navigation()
    if admission is False:
        return
    release_navigation = admission
```

`LibraryScreen.acquire_navigation_transition()` returns `None` when File Notes
is inactive, `False` when the owner refuses a `"screen"` lease, or the acquired
lease's bound idempotent `release` method. It performs no await. Hold that lease
by wrapping the existing save/construct/restore/context/switch block in
`try/finally`; release it after `switch_screen()`/outgoing unmount and on every
failure path.

Use the same pattern around the in-screen Files-to-Database/source switch:
after `_flush_active_file_notes()` and before changing the source, acquire a
`"source"` lease from the active workspace; hold it through recompose and
release in `finally`. This closes the reciprocal race without adding a general
application-state owner.

- [ ] **Step 5: Shut down before Textual closes screens**

Textual 8.2.7's `App._shutdown()` calls `_close_all()` before dispatching the
app Unmount event. Therefore `on_unmount()` alone cannot establish the required
Git-before-replica ordering.

Add one narrow, version-pinned override:

```python
async def _shutdown_file_notes_session_owner(self) -> None:
    owner = getattr(self, "file_notes_session_owner", None)
    if owner is not None:
        await owner.shutdown()


async def _shutdown(self) -> None:
    await self._shutdown_file_notes_session_owner()
    await super()._shutdown()
```

Also call the same idempotent helper from `on_unmount()` as a fallback for
partial/manual lifecycles. Do not rely only on `action_quit()` or
`on_shutdown_request()`; those miss programmatic `exit()`/test teardown or run
too late.

Owner shutdown seals admission, joins retained status/mutation/postflight,
then performs separate finite terminate and kill waits. It never removes a Git
lock file.

- [ ] **Step 6: Run navigation and teardown tests**

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_screen_navigation.py -k "file_notes and (owner or git or mutation)"
../../.venv/bin/python -m pytest -q Tests/ProductionApp/test_file_notes_session_owner_lifecycle.py
```

Expected: PASS.

- [ ] **Step 7: Commit app lifecycle wiring**

```bash
git add tldw_chatbook/app.py tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_screen_navigation.py Tests/ProductionApp/test_file_notes_session_owner_lifecycle.py
git commit -m "feat(app): own File Notes Git lifecycle across screens [TASK-1213]"
```

## Task 8: Close the Approved Focused Matrix and Reconcile TASK-1213

**Files:**

- Modify as required within the files listed above; do not add a second acceptance layer.
- Modify: `Tests/Notes/test_file_notes_git_integration.py`
- Modify: `Tests/UI/test_library_file_notes_git.py`
- Modify: `backlog/tasks/task-1213 - Add-session-scoped-Git-status-and-staging-to-File-Notes.md`

- [ ] **Step 1: Add the controlled configured-filter trust test**

Use a repository-local executable filter writing only to a `tmp_path` sentinel.
Assert:

- discovery and declining/closing/Escape do not run the filter;
- accepting trust permits worktree-aware status and Stage to reach it;
- trust is process/root/full-identity scoped;
- Chatbook requests no worktree mutation even though arbitrary filter side
  effects are explicitly disclosed.

- [ ] **Step 2: Complete the compact repository matrix**

Fill only cases not already added beside Tasks 3-5:

- ignored and conflict;
- linked worktree common-dir identity;
- external index/`HEAD` changes;
- sparse checkout/index blocked;
- nested repository path blocked;
- repository replacement;
- spaces, leading dashes, pathspec characters, and filesystem-byte names;
- lock contention and uncertain mutation result;
- primary `git diff`/`git diff --cached` Stage/Unstage flow;
- no worktree, replica, or session-history mutation by Git actions.

Keep this one compact fixture/matrix. Do not download submodules, touch a
remote, simulate an actually hung process, or add platform combinations.

- [ ] **Step 3: Add the lightweight 1,000-note UI fixture**

Create 1,005 unrelated Markdown/text files and only three session changes.
Assert:

- exactly the coalesced session groups become rows/pathspecs;
- unrelated note paths are absent from Git command argv;
- Files/search state survives entering/leaving Session Git;
- no pagination or timing threshold is introduced.

- [ ] **Step 4: Run the complete focused verification**

```bash
../../.venv/bin/python -m pytest -q Tests/Notes/test_file_notes_session_owner.py Tests/Notes/test_file_notes_git_service.py Tests/Notes/test_file_notes_git_integration.py Tests/Notes/test_file_notes_service.py
../../.venv/bin/python -m pytest -q Tests/UI/test_library_file_notes_git.py Tests/UI/test_library_file_notes_workspace.py
../../.venv/bin/python -m pytest -q Tests/UI/test_screen_navigation.py -k "file_notes"
../../.venv/bin/python -m pytest -q Tests/ProductionApp/test_file_notes_session_owner_lifecycle.py
../../.venv/bin/python -m compileall -q tldw_chatbook/Notes/file_notes_session_owner.py tldw_chatbook/Notes/file_notes_git_service.py tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py
../../.venv/bin/python -m ruff check tldw_chatbook/Notes/file_notes_session_owner.py tldw_chatbook/Notes/file_notes_git_service.py tldw_chatbook/Notes/file_notes_service.py tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py tldw_chatbook/Widgets/Library/library_file_notes_workspace.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/app.py Tests/Notes/test_file_notes_session_owner.py Tests/Notes/test_file_notes_git_service.py Tests/Notes/test_file_notes_git_integration.py Tests/Notes/test_file_notes_service.py Tests/UI/test_library_file_notes_git.py Tests/UI/test_library_file_notes_workspace.py Tests/UI/test_screen_navigation.py Tests/ProductionApp/test_file_notes_session_owner_lifecycle.py
git diff --check
```

Expected: every command exits 0. If an existing large file has a verified
pre-existing Ruff diagnostic, record the exact baseline and run Ruff on the
new/changed lines or new files; do not create a broad cleanup diff.

- [ ] **Step 5: Perform focused user acceptance testing**

In one disposable local repository through the mounted File Notes UI:

1. edit, create, move, and delete notes;
2. confirm only those coalesced rows appear;
3. Stage selected and Stage All;
4. inspect CLI `git diff` and `git diff --cached`;
5. type a newer edit while the staged version remains owned;
6. Stage update, then Unstage selected/All;
7. confirm original index baselines return and worktree edits remain;
8. leave/reopen Library and confirm continuity;
9. restart the app and confirm the staged path is external;
10. confirm commit/push/remotes and full-repository status are absent.

Record concise observed results in the task Implementation Notes. Do not add a
duplicate automated acceptance suite.

- [ ] **Step 6: Self-review against all eleven acceptance criteria**

Review the final diff for:

- disk/SQLite/session-history authority violations;
- shell invocation or unsanitized Git redirection;
- path broadening or missing D/F closure checks;
- trust/status work starting while hidden;
- duplicate trust/ownership state outside the owner;
- mutation lifecycle owned by a widget;
- editor/Back incorrectly disabled;
- any commit/push/remote/full-status scope creep.

- [ ] **Step 7: Reconcile and close the Backlog task**

Check all eleven acceptance criteria, add concise `## Implementation Notes`
covering the actual files, decisions, focused test/UAT evidence, and links to
ADR-035/ADR-033, then:

```bash
backlog task edit 1023 -s Done
backlog task 1023 --plain
```

Expected: TASK-1213 is Done, all acceptance criteria are checked, the
implementation plan and notes are present, and both ADR links remain.

- [ ] **Step 8: Commit task reconciliation**

```bash
git add "backlog/tasks/task-1213 - Add-session-scoped-Git-status-and-staging-to-File-Notes.md"
git commit -m "docs(notes): complete session Git staging task [TASK-1213]"
```
