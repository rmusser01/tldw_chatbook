# File Notes Guarded Session Commit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Use superpowers:test-driven-development for each behavior change and superpowers:verification-before-completion before every completion claim. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user review and create one local Git commit containing exactly the currently Chatbook-owned staged File Notes groups, with truthful recovery when the result cannot be proved.

**Architecture:** Extend the existing process-owned `FileNotesSessionOwner`, `FileNotesGitService`, Prepare panel, and File Notes workspace. The owner remains the sole binding/trust/staging authority; the service remains the sole Git/process authority; the workspace owns the binding-scoped draft and tokenized editor lease; the panel remains presentation-only. Add one small pure module for commit message/identity/display models and byte parsers so the already-large Git service stays an I/O orchestrator, not a second policy implementation.

**Tech Stack:** Python 3.11+, asyncio subprocesses, Git 2.x plumbing/porcelain, immutable dataclasses, Textual 3.3+, pytest/pytest-asyncio, disposable local Git repositories.

**Backlog:** [TASK-1350](../../../backlog/tasks/task-1350%20-%20Add-guarded-session-commit-to-File-Notes.md)

**Specification:** [File Notes Guarded Session Commit Design](../specs/2026-07-29-file-notes-guarded-session-commit-design.md)

**Decision:** [ADR-038](../../../backlog/decisions/038-file-notes-guarded-session-commit.md)

**Depends on:** TASK-1213, TASK-1235

**ADR required:** yes

**ADR path:** `backlog/decisions/038-file-notes-guarded-session-commit.md`

**Reason:** ADR-038 defines the guarded local-commit service, security, recovery, process, and UX boundary; ADR-035 remains the staging authority.

---

## Non-Negotiable Boundary

- Implement local normal commit only. Do not add push, remotes, credentials,
  history browsing, branch management, amend, signing, or general repository
  status.
- Keep Markdown/text files authoritative and SQLite an independent replica.
  After autosave settlement, the commit path calls no note filesystem or
  replica mutation API.
- Never expose, persist, log, or add to ownership an unrelated repository path.
  Repository-wide logical-index path identities may exist only inside one
  proof calculation and must be discarded before returning a result.
- Do not weaken the existing session-path whitelist used by normal File Notes
  status.
- Do not add a second app/session owner, a Git facade hierarchy, a commit queue,
  a durable recovery journal, or a dependency.
- Do not add row virtualization unless the focused 1,000-note measurement
  demonstrates that the existing `ListView` is inadequate.
- Do not run the full suite, broad CI, network tests, or an unrelated
  performance soak. The focused baseline for the existing Git service and
  mounted File Notes Git UI is 220 passing tests in about 32 seconds.

## File Responsibilities

- Create `tldw_chatbook/Notes/file_notes_git_commit.py`: immutable public
  commit form/review/outcome models plus pure message, identity, raw-delta, and
  raw-commit parsing/validation. It owns no process, repository I/O, trust,
  session state, or widget state.
- Modify `tldw_chatbook/Notes/file_notes_session_owner.py`: authority
  generation, exact captured sequence membership, one atomic lease-bound
  commit publication, quarantine, and recovery admission.
- Modify `tldw_chatbook/Notes/file_notes_git_service.py`: exact argv and
  environment builders, local repository blockers, complete-index proof,
  retained child settlement, review/confirm/check-again cycles, postflight,
  and shutdown.
- Modify `tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py`:
  presentation-only form/review/progress/result surfaces and typed intents.
- Modify `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`:
  draft, autosave settlement, editor lease, operation IDs, lifecycle
  coordination, focus, and typed outcome projection.
- Do not modify `tldw_chatbook/app.py` or
  `tldw_chatbook/UI/Screens/library_screen.py` unless a focused lifecycle test
  proves an existing owner-injection or owner-first-shutdown seam is
  insufficient. If that happens, document the deviation in this plan before
  editing either file and include the changed file in Task 10's focused
  static/compile checks.
- Create `Tests/Notes/test_file_notes_git_commit.py`: pure contract, fake-runner,
  service-cycle, and recovery tests.
- Create `Tests/Notes/test_file_notes_git_commit_integration.py`: disposable
  real-repository commit matrix.
- Modify `Tests/Notes/test_file_notes_session_owner.py`: exact owner
  publication/quarantine tests.
- Modify `Tests/UI/test_library_file_notes_git.py`: mounted panel/workspace,
  focus, responsive, and representative-session tests.
- Modify
  `Tests/ProductionApp/test_file_notes_session_owner_lifecycle.py` only for
  retained commit/shutdown settlement.
- Modify `Tests/UI/test_screen_navigation.py` only if fresh-workspace
  rehydration needs a production navigation regression.

## Public and Private Contract

The exact names may be adjusted once to match local naming conventions, but
the boundaries may not move:

- `file_notes_git_commit.py` exposes frozen `GitIdentity`,
  `CommitIncludedNote`, `CommitReviewProjection`, opaque
  `CommitReviewHandle`, `CommitReviewResult`, `CommitOutcome`, and
  `CommitRecoveryProjection` values. Widgets receive projections/outcomes,
  never raw Git output or the private proof snapshot.
- `FileNotesGitService.start_commit_review(...)` returns one retained,
  cancellation-safe review settlement. On success it stores one private,
  single-use `_CommitReviewSnapshot` and returns an opaque handle plus a
  sanitized projection.
- `FileNotesGitService.start_commit(...)` consumes that handle, revalidates,
  runs at most one branch-mutating child, performs postflight, and publishes
  one typed outcome even if the UI waiter disappears.
- `FileNotesGitService.cancel_commit(...)` can cancel owned read-only
  preflight/revalidation only. It returns false and performs no cancellation
  once the branch-mutating child has started.
- `FileNotesGitService.check_commit_again(...)` operates only on the exact
  retained uncertain evidence. It never creates a new commit child.
- `FileNotesSessionOwner.publish_commit_outcome(...)` performs one atomic,
  exact-token state transition. UI operation IDs may suppress stale rendering
  only; they never suppress service postflight or owner publication.

## Exact Git Contract

Every proof and commit command uses direct argv, a sanitized environment, and
`GIT_NO_LAZY_FETCH=1`. Commit proof disables replacement refs, rename
detection, external diff execution, text conversion, and configured
filesystem monitors. The branch child is exactly equivalent to:

```text
git --no-replace-objects
  -c core.hooksPath=<verified-private-empty-directory>
  -c core.fsmonitor=false
  -c maintenance.auto=false
  -c gc.auto=0
  -c commit.gpgSign=false
  -c i18n.commitEncoding=UTF-8
  commit --no-gpg-sign --cleanup=verbatim -F -
```

Supply the exact normalized UTF-8 message on stdin. Strip Git
repository/index/config redirection, terminal/editor/prompt variables, and
ambient `GIT_AUTHOR_DATE`/`GIT_COMMITTER_DATE`; bind only the reviewed author
and committer names/emails. Do not use a shell or `--no-verify`.

The complete proof uses a bounded command count independent of included-note
count:

- local config/filesystem blocker checks before object resolution;
- complete semantic index: `ls-files -z --stage -v`;
- complete cached delta against captured old `HEAD`:
  `diff-index --cached --raw -z --no-renames --no-ext-diff --no-textconv`;
- expected tree: `write-tree`;
- identity: `git var GIT_AUTHOR_IDENT` and `git var GIT_COMMITTER_IDENT`;
- replacement-free raw commit object: `cat-file commit <oid>`; and
- replacement-free branch/index postflight using the same complete proof.

`write-tree` may write unreachable tree objects, but preflight must not change
`HEAD`, a ref, the logical index, any worktree path, SQLite, configuration, or
a remote.

---

## Task 1: Add Pure Commit Contracts and Byte Parsers

**Files:**

- Create: `tldw_chatbook/Notes/file_notes_git_commit.py`
- Create: `Tests/Notes/test_file_notes_git_commit.py`

- [ ] Write `test_commit_message_*` tests first for:
  subject trimming; required 1–512-character subject; single-line subject;
  CRLF/CR normalization; surrounding blank-body-line removal; preservation of
  internal body whitespace/newlines; exact `subject\n` or
  `subject\n\nbody\n`; UTF-8/64-KiB bounds; emoji and ordinary RTL acceptance;
  and rejection of NUL, unsafe C0/C1 controls, lone surrogates, and
  directional-override/isolate controls.
- [ ] Write `test_git_identity_*` tests first for parsing Git's effective
  `<name> <email> <timestamp> <offset>` ident from the right, missing/empty
  name/email, author/committer display collapse, hostile terminal controls,
  and markup-looking but otherwise printable text.
- [ ] Write `test_raw_staged_delta_*` and `test_raw_commit_object_*` tests
  first for NUL-delimited additions/deletions/mode changes, malformed/truncated
  records, one exact parent, tree, author, committer, multiline headers,
  message bytes, and signature-header detection. Preserve filename bytes only
  long enough for proof comparison; diagnostics remain generic.
- [ ] Run:

```bash
python -m pytest Tests/Notes/test_file_notes_git_commit.py -q -k "commit_message or git_identity or raw_staged_delta or raw_commit_object"
```

Expected: FAIL because the new module/contracts do not exist.

- [ ] Implement only the frozen models and pure functions needed by these
  tests. Use `markup=False`-safe display strings and bounded generic
  diagnostics; do not import Textual or perform Git I/O.
- [ ] Re-run the command. Expected: PASS.
- [ ] Run:

```bash
python -m ruff check tldw_chatbook/Notes/file_notes_git_commit.py Tests/Notes/test_file_notes_git_commit.py
python -m ruff format --check tldw_chatbook/Notes/file_notes_git_commit.py Tests/Notes/test_file_notes_git_commit.py
git diff --check
```

- [ ] Commit:

```bash
git add tldw_chatbook/Notes/file_notes_git_commit.py Tests/Notes/test_file_notes_git_commit.py
git commit -m "feat(notes): add guarded commit contracts [TASK-1350]"
```

## Task 2: Add Exact Owner Authority, Publication, and Quarantine

**Files:**

- Modify: `tldw_chatbook/Notes/file_notes_session_owner.py`
- Modify: `tldw_chatbook/Notes/file_notes_git_service.py`
- Modify: `Tests/Notes/test_file_notes_session_owner.py`
- Modify: `Tests/Notes/test_file_notes_git_service.py`

- [ ] Write `test_commit_authority_*` owner tests first for:
  a monotonic Git-authority generation; invalidation on binding, trust,
  staging ownership, relevant session-change, commit publication, and root
  transitions; and rejection of an ABA review after state changes away and
  back to equivalent values. Do not increment merely because an equivalent
  presentation/status object was republished.
- [ ] Extend `SessionChangeGroup`/coalescing tests so every group records the
  exact ordered session sequence IDs it contains while preserving its stable
  earliest-sequence `group_id`.
- [ ] Write atomic-publication tests first:
  - success consumes all old-HEAD staging ownership, retires only explicitly
    proven captured sequence IDs, and preserves later sequences plus groups
    with newer postcommit worktree divergence;
  - failed-unchanged preserves active ownership and draft-independent owner
    facts;
  - uncertainty invalidates status and moves captured ownership to quarantine;
  - ordinary Stage/Unstage/commit admission returns `recovery_required` while
    quarantine exists;
  - exact recovery can restore only the captured ownership under the original
    repository/branch/index proof and only when active ownership is empty;
  - root rebinding/process exit discards review/recovery evidence without
    deriving or restoring ownership; and
  - an already-admitted exact token may publish during owner-first shutdown,
    while all new admissions remain sealed.
- [ ] Run:

```bash
python -m pytest Tests/Notes/test_file_notes_session_owner.py Tests/Notes/test_file_notes_git_service.py -q -k "commit_authority or commit_publication or commit_quarantine or coalesce_session_changes"
```

Expected: FAIL on missing generation, sequence membership, publication, and
recovery admission.

- [ ] Add frozen owner-side capture/publication/recovery values. Validate the
  exact active mutation lease token, binding, authority generation,
  repository identity, attached branch, ownership, and captured sequence
  membership in one lock acquisition.
- [ ] Add a single `publish_commit_outcome(...)` transition instead of
  composing existing `clear_status`, `clear_ownership`, and list edits. Keep
  `StagingOwnership` session-path-only and expose at most a sanitized
  recovery-pending projection in `FileNotesSessionSnapshot`.
- [ ] Extend coalescing with sequence membership without moving ownership or
  Git policy into the owner.
- [ ] Re-run the focused command. Expected: PASS.
- [ ] Commit:

```bash
git add tldw_chatbook/Notes/file_notes_session_owner.py tldw_chatbook/Notes/file_notes_git_service.py Tests/Notes/test_file_notes_session_owner.py Tests/Notes/test_file_notes_git_service.py
git commit -m "feat(notes): add atomic commit authority [TASK-1350]"
```

## Task 3: Retain and Settle the Exact Commit Child

**Files:**

- Modify: `tldw_chatbook/Notes/file_notes_git_service.py`
- Modify: `Tests/Notes/test_file_notes_git_commit.py`
- Modify: `Tests/Notes/test_file_notes_git_service.py`

- [ ] Add `test_retained_commit_child_*` tests first with controlled fake
  subprocesses for:
  normal zero/nonzero exit; timeout with a still-live child; later natural
  nonzero exit; later zero exit; terminate/kill requested by shutdown;
  caller cancellation; repeated settlement; loop affinity; and bounded
  stderr. The exact retained token must never settle another child.
- [ ] Run:

```bash
python -m pytest Tests/Notes/test_file_notes_git_commit.py Tests/Notes/test_file_notes_git_service.py -q -k "retained_commit_child"
```

Expected: FAIL because the runner has only global shutdown settlement.

- [ ] Replace the runner's sticky-only uncertainty with operation-scoped
  retained-child records. Return an opaque retained-child token only when
  termination is uncertain; expose a non-sealing settlement/read API that can
  distinguish alive, known natural return code, Chatbook stop requested, and
  still-uncertain. Keep existing callers source-compatible.
- [ ] Preserve runner-owned shielding. Never abandon `communicate()`, lose a
  final return code that is still knowable, or treat a force-stopped child as a
  known normal unsuccessful commit.
- [ ] Keep global shutdown bounded and idempotent; it must settle all retained
  tokens and report uncertainty if termination still cannot be proved.
- [ ] Re-run the focused command. Expected: PASS.
- [ ] Commit:

```bash
git add tldw_chatbook/Notes/file_notes_git_service.py Tests/Notes/test_file_notes_git_commit.py Tests/Notes/test_file_notes_git_service.py
git commit -m "feat(notes): retain exact commit child outcomes [TASK-1350]"
```

## Task 4: Build the Guarded Review Preflight

**Files:**

- Modify: `tldw_chatbook/Notes/file_notes_git_commit.py`
- Modify: `tldw_chatbook/Notes/file_notes_git_service.py`
- Modify: `Tests/Notes/test_file_notes_git_commit.py`
- Create: `Tests/Notes/test_file_notes_git_commit_integration.py`

- [ ] Add pure argv/environment tests first for:
  `--no-replace-objects`; `core.fsmonitor=false`; no rename/external-diff/
  textconv; `GIT_NO_LAZY_FETCH=1`; repository/index/config redirection
  removal; prompt/editor suppression; ambient date removal; identity binding;
  exact commit argv; and exact stdin.
- [ ] Add fake-runner review tests first for this order:
  root/repository identity → local special-state/lock/graft/partial/promisor
  blockers → attached `refs/heads/*`/old-HEAD read → complete index/delta/tree
  → owned-worktree freshness → identities. Assert no object-resolving command
  runs after a local blocker and no commit child runs during review.
- [ ] Add proof tests for exact union equality between the complete staged
  delta and current `StagingOwnership`, including additions, deletions, modes,
  move topology, intent-to-add, conflicts, gitlinks, sparse/semantic flags,
  unrelated staged state, stale ownership, and an included group with newer
  saved unstaged edits. Unrelated unstaged paths remain allowed. Ownership
  whose post-Stage entries equal its saved baseline contributes no staged
  delta, is not counted/included, and cannot authorize an empty commit.
- [ ] Add hostile unrelated-path tests proving the path is absent from every
  public result, diagnostic, log capture, owner snapshot, and serialized
  representation after the proof call returns.
- [ ] Add initial disposable-repository review tests for attached, detached,
  unborn, missing identity, unrelated staged, newer included edit, active
  operation, lock, graft, replacement ref, and partial/promisor cases. A
  promisor block must execute no fetch/network command.
- [ ] Run:

```bash
python -m pytest Tests/Notes/test_file_notes_git_commit.py Tests/Notes/test_file_notes_git_commit_integration.py -q -k "commit_argv or commit_environment or commit_review or complete_commit_proof"
```

Expected: FAIL because no review API or complete proof path exists.

- [ ] Implement exact commit builders and a commit-specific sanitized
  environment without changing ordinary Stage/Unstage environment behavior.
- [ ] Implement local blocker inspection before object resolution. Check both
  worktree/common Git operation markers and relevant locks; common
  `info/grafts`; repository-format partial-clone extension; promisor remote
  configuration; local `.promisor` object markers; sparse state; and
  unsupported index entries. Do not delete or repair anything.
- [ ] Implement one bounded complete-index proof path separate from the normal
  session status parser. Compare raw logical records to owned post-Stage
  entries/topology, compute only signatures/object IDs needed after the call,
  and discard unrelated raw path identities before returning.
- [ ] Resolve both effective identities with `git var`, normalize/validate the
  message, create one private immutable single-use snapshot, and return only
  an opaque handle plus sanitized projection. Release the owner mutation gate
  for human review.
- [ ] Re-run the focused command. Expected: PASS.
- [ ] Commit:

```bash
git add tldw_chatbook/Notes/file_notes_git_commit.py tldw_chatbook/Notes/file_notes_git_service.py Tests/Notes/test_file_notes_git_commit.py Tests/Notes/test_file_notes_git_commit_integration.py
git commit -m "feat(notes): add guarded commit review proof [TASK-1350]"
```

## Task 5: Execute Once and Prove the Immediate Outcome

**Files:**

- Modify: `tldw_chatbook/Notes/file_notes_git_commit.py`
- Modify: `tldw_chatbook/Notes/file_notes_git_service.py`
- Modify: `Tests/Notes/test_file_notes_git_commit.py`
- Modify: `Tests/Notes/test_file_notes_git_commit_integration.py`

- [ ] Add confirmation tests first for one-shot handle consumption, full
  revalidation under the exact mutation lease, generation/repository/branch/
  index/worktree/message/identity drift, cancellation before child start, and
  refusal of cancellation after child start.
- [ ] Add hooks-directory tests first: `mkdtemp` outside the repository,
  owner-only mode, same filesystem identity, verified empty, alive for the
  entire child, `rmdir` only after certain termination, retained while a child
  may be alive, and no recursive deletion.
- [ ] Add outcome tests first:
  - `Succeeded` requires normal zero exit or later exact recovery plus exact
    attached branch, raw single parent, complete tree, message, reviewed
    names/emails, no signature header, logical index equal to the new tree,
    and no staged delta;
  - `Failed unchanged` requires a known normal nonzero exit plus exact old
    branch/index and no unexpected lock/special operation;
  - every contradiction, incomplete postflight, unexpected branch/index
    movement, extra/missing content, or uncertain child is `Uncertain`;
  - no outcome performs rollback, lock deletion, retry, worktree mutation, or
    replica mutation.
- [ ] Add real-repository happy-path and definite-failure tests verifying
  parent/tree/message/identities/unsigned object/ref/index, hook sentinel not
  run, configured signing overridden, configured fsmonitor not invoked,
  automatic maintenance disabled, and unrelated unstaged bytes unchanged.
- [ ] Run:

```bash
python -m pytest Tests/Notes/test_file_notes_git_commit.py Tests/Notes/test_file_notes_git_commit_integration.py -q -k "commit_confirmation or hooks_directory or commit_outcome or guarded_commit_success or guarded_commit_failure"
```

Expected: FAIL because confirmation/execution/postflight are not implemented.

- [ ] Implement final revalidation and atomically consume the confirmation
  capability before execution. Use the exact direct argv/stdin/environment
  contract and bind reviewed names/emails while allowing Git to select current
  execution timestamps.
- [ ] Keep the owner mutation lease through child completion, postflight, and
  the single atomic owner publication. Publish success by retiring only
  postflight-proven captured sequences, retaining newer worktree groups,
  clearing old-HEAD ownership, and refreshing actual session status.
- [ ] Emit the exact success summary and adjacent qualification:
  `Committed N session notes as <short-oid>; unrelated changes untouched.` and
  `No unrelated staged content was committed; Chatbook selected no unrelated
  worktree paths.`
- [ ] Re-run the focused command. Expected: PASS.
- [ ] Commit:

```bash
git add tldw_chatbook/Notes/file_notes_git_commit.py tldw_chatbook/Notes/file_notes_git_service.py Tests/Notes/test_file_notes_git_commit.py Tests/Notes/test_file_notes_git_commit_integration.py
git commit -m "feat(notes): execute and prove guarded commit [TASK-1350]"
```

## Task 6: Converge Uncertainty and Settle Shutdown

**Files:**

- Modify: `tldw_chatbook/Notes/file_notes_git_commit.py`
- Modify: `tldw_chatbook/Notes/file_notes_git_service.py`
- Modify: `Tests/Notes/test_file_notes_git_commit.py`
- Modify: `Tests/Notes/test_file_notes_git_commit_integration.py`
- Modify:
  `Tests/ProductionApp/test_file_notes_session_owner_lifecycle.py`

- [ ] Add recovery tests first for:
  child still alive; relevant lock/special state; later exact success; later
  exact unchanged state after known natural nonzero exit; later unchanged
  state without a known normal failure; repository differing from both states;
  repeated `Check again`; rebinding; and process exit. Assert `Check again`
  never starts a commit child.
- [ ] Add lifecycle tests first for caller/UI waiter cancellation, panel
  unmount, owner-first application shutdown, bounded terminate/kill, exact
  owner publication before service cleanup, and hooks-directory retention or
  cleanup according to child certainty.
- [ ] Run:

```bash
python -m pytest Tests/Notes/test_file_notes_git_commit.py Tests/Notes/test_file_notes_git_commit_integration.py Tests/ProductionApp/test_file_notes_session_owner_lifecycle.py -q -k "commit_recovery or commit_check_again or retained_commit_shutdown"
```

Expected: FAIL because uncertain evidence cannot yet converge and shutdown does
not yet settle a commit cycle.

- [ ] Quarantine captured staging ownership before releasing an uncertain
  mutation lease. Clear cached status, preserve only immutable old-HEAD/tree/
  message/identity/index proof plus the exact retained child token, and block
  all ordinary Git mutations.
- [ ] Make `Check again` unavailable while the exact child may be alive or a
  relevant lock/special operation remains. After certain termination:
  publish normal success on exact reviewed-child/index proof; restore captured
  ownership only on exact old-state proof plus known natural nonzero result;
  otherwise remain uncertain without deriving ownership from fresh status.
- [ ] Extend service shutdown to settle review, confirmation, recovery, runner,
  and hooks-directory lifecycles idempotently. Do not clear owner state before
  the retained operation publishes its exact outcome.
- [ ] Re-run the focused command. Expected: PASS.
- [ ] Commit:

```bash
git add tldw_chatbook/Notes/file_notes_git_commit.py tldw_chatbook/Notes/file_notes_git_service.py Tests/Notes/test_file_notes_git_commit.py Tests/Notes/test_file_notes_git_commit_integration.py Tests/ProductionApp/test_file_notes_session_owner_lifecycle.py
git commit -m "feat(notes): recover uncertain guarded commits [TASK-1350]"
```

## Task 7: Complete the Disposable-Repository Security Matrix

**Files:**

- Modify: `Tests/Notes/test_file_notes_git_commit_integration.py`
- Modify only if a new test fails:
  `tldw_chatbook/Notes/file_notes_git_commit.py`
- Modify only if a new test fails:
  `tldw_chatbook/Notes/file_notes_git_service.py`
- Modify only if a new test fails:
  `tldw_chatbook/Notes/file_notes_session_owner.py`

- [ ] Add the remaining real-repository cases one at a time:
  create/modify/delete/mode/grouped move/chained move; exact complete tree;
  unrelated staged block with unchanged branch/index and no path disclosure;
  trusted clean-filter freshness; conflict/intent-to-add/gitlink/semantic
  index states; sequencer/bisect markers; index/ref locks; replacement refs;
  grafts; partial/promisor repositories with no network; ambient dates;
  postflight `HEAD`/index drift with no rollback; retained newer worktree
  edits; and unchanged note bytes plus SQLite replica/revision/tombstone rows.
- [ ] Add one 1,000-note repository with a representative session set, and
  instrument the runner to assert the review/confirm/postflight Git process
  count is bounded by a constant and identical for the small and large
  repository. Do not enumerate 1,000 rows in the review UI.
- [ ] Before any implementation adjustment, run:

```bash
python -m pytest Tests/Notes/test_file_notes_git_commit_integration.py -q
```

Expected: the new matrix either passes against Tasks 4–6 or identifies one
specific contract defect. Fix only a demonstrated defect; do not add a new
abstraction or broader repository feature.

- [ ] Re-run until PASS. Also run:

```bash
python -m pytest Tests/Notes/test_file_notes_git_commit.py Tests/Notes/test_file_notes_session_owner.py -q
git diff --check
```

Expected: PASS.

- [ ] Commit:

```bash
git add Tests/Notes/test_file_notes_git_commit_integration.py tldw_chatbook/Notes/file_notes_git_commit.py tldw_chatbook/Notes/file_notes_git_service.py tldw_chatbook/Notes/file_notes_session_owner.py
git commit -m "test(notes): cover guarded commit repository matrix [TASK-1350]"
```

Omit unchanged production files from `git add`.

## Task 8: Add the Prepare-Panel Commit Form and Review

**Files:**

- Modify:
  `tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py`
- Modify: `Tests/UI/test_library_file_notes_git.py`

- [ ] Add mounted panel tests first for:
  `Commit staged (0/N)` and its zero explanation; form entry and inline
  validation; current branch/count; typed intents; binding draft projection;
  exact `markup=False` message/identity/path preview; included-note
  disclosure; complete focused path; change types; the exact hook/unsigned,
  no-unrelated-staged/worktree-selection, and complete-staged-file policy
  copy; Edit-first focus; Enter activating only the focused button without
  transferring the Review Enter to Confirm; and state-aware Escape. The
  displayed count comes from service/owner-authorized groups that contribute
  an actual staged delta, never a panel guess from row count.
- [ ] Add geometry tests first for one mounted panel at normal width and
  `40x20`. At 40 columns the fixed review footer must render Edit and Cancel
  on row one and full-width Confirm on row two, in keyboard order
  disclosure → Edit → Cancel → Confirm. Confirm is last and never initially
  focused.
- [ ] Add result tests first proving running, success, failed, blocked, and
  uncertain copy wraps/scrolls and is never passed through
  `_fit_two_line_copy()`/`_fit_fixed_regions()`. Uncertainty exposes only safe
  inspection and `Check again`. Pin these required strings exactly:
  `Checking commit...`; `Committing N session notes...`;
  `Git is updating the branch; cancellation is unavailable.`; and
  `Commit may have succeeded. Git actions are disabled until the repository
  is checked. Run git status and git log -1, then choose Check again.`
- [ ] Run:

```bash
python -m pytest Tests/UI/test_library_file_notes_git.py -q -k "commit_panel or commit_form or commit_review or commit_result or commit_footer"
```

Expected: FAIL because the panel has only staged-note list actions.

- [ ] Keep one mounted panel with a list surface and a hidden commit workflow
  surface; do not add a modal, screen, or recompose-driven second panel. Use a
  scrollable workflow body plus fixed footer and the existing `ListView` for
  included notes.
- [ ] Add presentation phases `list`, `form`, `checking`, `review`,
  `confirming`, `executing`, and `result`, with render methods receiving only
  sanitized immutable projections. The panel emits typed intents and contains
  no Git command, proof, ownership, or service policy.
- [ ] Make `Back to navigator` exist only at list level. Form/checking Cancel
  returns to the staged-note list; Review Escape/Edit returns to the form;
  Review `Cancel commit` returns to the staged-note list; executing ignores
  cancellation/navigation; uncertainty has no implicit mutation.
- [ ] Re-run the focused command. Expected: PASS.
- [ ] Commit:

```bash
git add tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py Tests/UI/test_library_file_notes_git.py
git commit -m "feat(notes): add guarded commit review UI [TASK-1350]"
```

## Task 9: Wire Draft, Editor Lease, Retained Operations, and Focus

**Files:**

- Modify:
  `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`
- Modify:
  `tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py`
- Modify: `Tests/UI/test_library_file_notes_git.py`
- Modify only if required by a failing fresh-screen test:
  `Tests/UI/test_screen_navigation.py`

- [ ] Add mounted workspace tests first for a draft keyed by exact repository
  binding: it survives Edit, Cancel, blocked/stale review, definite failure,
  `Uncertain`, repeated `Check again`, later unchanged convergence, and panel
  replacement; proven success clears it, including success reached through
  `Check again`; root/repository rebinding clears it with visible explanation.
- [ ] Add tokenized editor-read-only tests first. Acquire before autosave
  settlement for the exact `(editor identity, binding)`, block review on
  dirty/saving/conflict/save-error, preserve Stage's existing writable-editor
  behavior, and release only this flow's reason on cancellation/block/terminal
  outcome. Other read-only reasons remain effective.
- [ ] Add retained-operation tests first for review cancellation, confirm
  revalidation cancellation, no cancellation/navigation after child start,
  operation-ID stale projection rejection, panel/workspace unmount, remount
  rehydration, and process-owned postflight/publication/lease finalization.
  With a deliberately delayed fake service operation over 100 ms, assert the
  Textual pilot remains responsive, can move focus, and can invoke allowed
  pre-child cancellation; Git work must not block the Textual event loop.
- [ ] Add focus tests first:
  Subject after entering form; relevant field/action after validation/block;
  Edit first on review; prior focus after cancellation; first remaining row or
  Back after success; and no delayed `_focus_session_git_panel()` callback
  stealing focus from Subject, Edit, disclosure, or result actions.
- [ ] Add a full keyboard flow at normal width and `40x20`, plus a
  representative session set in a 1,000-note repository. Assert editor action
  toolbars stay visually quiet and narrow Navigator/Editor remain alternate
  views. Assert the incumbent Textual status/notification mechanism announces
  checking, committing, success, failure, and uncertainty.
- [ ] Run:

```bash
python -m pytest Tests/UI/test_library_file_notes_git.py Tests/UI/test_screen_navigation.py -q -k "guarded_commit or commit_editor_lease or commit_operation or commit_focus or commit_40x20 or commit_1000"
```

Expected: FAIL because the workspace protocol/state/lifecycle are not wired.

- [ ] Extend `_SessionGitService` with the typed review/commit/recovery API.
  Store draft, operation ID, opaque handle, sanitized projection, retained
  settlement, and prior focus per exact binding. Every async renderer checks
  binding plus operation ID; the service still publishes stale/unmounted
  operation outcomes.
- [ ] Replace direct editor `read_only` assignments with
  `_sync_editor_read_only()` and tokenized reason leases. Acquire the commit
  lease before `flush_pending_work()`. Attach final release to the retained
  service settlement, not to a disposable Textual waiter/render callback.
- [ ] Separate confirm into cancelable read-only revalidation and
  non-cancelable child-start phases. Rehydrate retained phase/outcome from the
  process service/owner when the panel remounts.
- [ ] Restrict `_focus_session_git_panel()` retry behavior to the list state
  and its captured list/entry targets; it must never redirect focus from a
  commit-workflow descendant.
- [ ] Route checking, committing, success, failure, and uncertainty through
  the incumbent Textual status/notification mechanism using the same
  operation-ID stale-projection guard as visible rendering.
- [ ] Re-run the focused command. Expected: PASS.
- [ ] Commit:

```bash
git add tldw_chatbook/Widgets/Library/library_file_notes_workspace.py tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py Tests/UI/test_library_file_notes_git.py Tests/UI/test_screen_navigation.py
git commit -m "feat(notes): wire guarded commit workspace flow [TASK-1350]"
```

Omit `Tests/UI/test_screen_navigation.py` if unchanged.

## Task 10: Focused Verification, Live UAT, and Task Closure

**Files:**

- Modify:
  `backlog/tasks/task-1350 - Add-guarded-session-commit-to-File-Notes.md`
- Modify this plan only to check completed steps or document an approved
  deviation.

- [ ] Run the complete focused automated boundary once:

```bash
python -m pytest Tests/Notes/test_file_notes_git_commit.py Tests/Notes/test_file_notes_git_commit_integration.py Tests/Notes/test_file_notes_session_owner.py Tests/Notes/test_file_notes_git_service.py Tests/UI/test_library_file_notes_git.py Tests/ProductionApp/test_file_notes_session_owner_lifecycle.py -q
```

Expected: PASS. If `Tests/UI/test_screen_navigation.py` changed, append that
file to this command.

- [ ] Run static checks only on changed Python files:

```bash
python -m ruff check tldw_chatbook/Notes/file_notes_git_commit.py tldw_chatbook/Notes/file_notes_session_owner.py tldw_chatbook/Notes/file_notes_git_service.py tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py tldw_chatbook/Widgets/Library/library_file_notes_workspace.py Tests/Notes/test_file_notes_git_commit.py Tests/Notes/test_file_notes_git_commit_integration.py Tests/Notes/test_file_notes_session_owner.py Tests/Notes/test_file_notes_git_service.py Tests/UI/test_library_file_notes_git.py Tests/ProductionApp/test_file_notes_session_owner_lifecycle.py
python -m ruff format --check tldw_chatbook/Notes/file_notes_git_commit.py tldw_chatbook/Notes/file_notes_session_owner.py tldw_chatbook/Notes/file_notes_git_service.py tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py tldw_chatbook/Widgets/Library/library_file_notes_workspace.py Tests/Notes/test_file_notes_git_commit.py Tests/Notes/test_file_notes_git_commit_integration.py Tests/Notes/test_file_notes_session_owner.py Tests/Notes/test_file_notes_git_service.py Tests/UI/test_library_file_notes_git.py Tests/ProductionApp/test_file_notes_session_owner_lifecycle.py
python -m compileall -q tldw_chatbook/Notes/file_notes_git_commit.py tldw_chatbook/Notes/file_notes_session_owner.py tldw_chatbook/Notes/file_notes_git_service.py tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
git diff --check
```

Expected: all commands exit 0. Append `Tests/UI/test_screen_navigation.py` only
if changed. If the documented lifecycle deviation changed
`tldw_chatbook/app.py` or `tldw_chatbook/UI/Screens/library_screen.py`, append
the changed path to the Ruff check, Ruff format check, and `compileall`
commands. Do not substitute a full-suite or broad-CI run.

- [ ] Perform focused live UAT in a disposable real notes repository using the
  real TUI:
  1. At a normal wide terminal, edit and stage two session notes; leave one
     unrelated worktree change unstaged; enter a subject and multiline body;
     verify review branch/old commit/identity/count/list/policies; confirm; and
     verify exact Git commit plus unchanged unrelated/note bytes.
  2. Exercise Cancel, an unrelated-staged block, a newer-included-edit
     `Stage update` block, Edit, Review `Cancel commit`, uncertainty draft
     preservation, `Check again`, and fresh Review recovery. Verify checking,
     committing, success, failure, and uncertainty announcements.
  3. Resize/relaunch at `40x20` and repeat form/review keyboard order,
     two-row footer, Confirm-last focus, running state, result, and recovery
     readability, including the complete `git status` / `git log -1`
     uncertainty instruction.
  4. Record the disposable repository commands, observed commit OID, terminal
     sizes, outcomes, and any deviations in TASK-1350 Implementation Notes.
- [ ] Review the final diff for unrelated scope, unrelated-path disclosure,
  note/SQLite mutation calls, shell command construction, durable state, and
  per-note Git loops.
- [ ] Re-read TASK-1350 and ADR-038. Check all eight acceptance criteria only
  when their evidence is present. Add concise Implementation Notes listing the
  approach, security boundaries, changed files, focused commands/counts/
  durations, UAT evidence, ADR links, and deviations.
- [ ] Mark TASK-1350 Done only after all Definition of Done requirements are
  satisfied:

```bash
backlog task edit 1350 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 --check-ac 6 --check-ac 7 --check-ac 8 -s Done --plain
```

- [ ] Commit task closure:

```bash
git add 'backlog/tasks/task-1350 - Add-guarded-session-commit-to-File-Notes.md' Docs/superpowers/plans/2026-07-29-file-notes-guarded-session-commit.md
git commit -m "docs(backlog): close guarded session commit [TASK-1350]"
```

## Completion Evidence

The implementation is complete only when:

- every exact proof and typed outcome is covered by focused tests;
- every branch-mutating child is retained through certain settlement or
  quarantined uncertainty;
- success proves the reviewed raw commit and a clean logical index;
- failed-unchanged has a known normal nonzero result and exact old state;
- unrelated repository paths never cross the internal proof boundary;
- the editor, draft, focus, narrow footer, and recovery behavior pass mounted
  tests and live UAT;
- note bytes and SQLite replica/revision/tombstone state remain unchanged; and
- TASK-1350 contains checked acceptance criteria and implementation evidence.
