# File Notes Guarded Session Commit Design

Date: 2026-07-29
Task: [TASK-1350](../../../backlog/tasks/task-1350%20-%20Add-guarded-session-commit-to-File-Notes.md)
Decision: [ADR-038](../../../backlog/decisions/038-file-notes-guarded-session-commit.md)
Amends: [ADR-035](../../../backlog/decisions/035-file-notes-session-git-index-controls.md)
Conforms to:
[ADR-033](../../../backlog/decisions/033-application-session-state-ownership.md),
[ADR-029](../../../backlog/decisions/029-file-notes-disk-authority.md), and
[ADR-011](../../../backlog/decisions/011-chatbook-workbench-ui-system.md)

## Summary

Extend File Notes `Prepare session for commit` with one guarded local commit of
all currently staged session-note groups whose exact staging ownership
Chatbook can still prove. The user supplies a required subject and optional
body, reviews the exact branch, identities, message, note set, hook policy, and
signing policy, then explicitly confirms.

The operation uses a normal whole-index Git commit only after proving that the
complete staged delta contains no content outside current Chatbook ownership.
It bypasses hooks, disables signing, retains the child lifecycle, and verifies
the exact commit afterward. It does not add push, remotes, credentials, amend,
repository repair, full-repository status, or general branch management.

Ordinary Markdown/text files remain authoritative. The independent SQLite
replica, protected revisions, and deletion tombstones are unaffected.

## User outcome

A user can finish a work or study session, stage selected or all eligible
session notes, and create one reviewed local commit without leaving Chatbook.
Afterward, ordinary `git status`, `git log`, `git push`, and the user's existing
remote workflow continue to work unchanged.

The user receives a truthful result:

- `Succeeded` only when Chatbook proves the exact reviewed commit;
- `Failed unchanged` only when it proves the branch and logical index did not
  change; or
- `Uncertain` when neither statement can be established.

Chatbook never guesses, rolls back shared Git state, or automatically retries.

## Approved product choices

- Commit all currently staged session groups that Chatbook still owns.
- Block when any unrelated staged delta exists.
- Leave all unstaged session notes and unrelated unstaged repository changes
  untouched.
- Require an existing attached branch. Detached and unborn `HEAD` are
  unsupported.
- Block merge, rebase, cherry-pick, revert, bisect, sequencer, conflict, and
  unsupported index states.
- Block an included staged note that has newer saved unstaged edits; require
  `Stage update`.
- Use existing Git-config/environment identity and block when it is missing.
- Create unsigned commits and bypass every repository commit hook.
- Require a subject and allow an optional multiline body.
- Use a read-only review followed by explicit confirmation.
- Make the Chatbook editor read-only from review preparation until review
  cancellation or a terminal commit result.
- Keep the complete interaction inside `Prepare session for commit`.
- Track only Chatbook session changes. No full-repository management appears.
- Leave push as a separate future work item.

## Scope

### Included

- Local normal commit of current Chatbook-owned staged session groups
- Deterministic message construction and validation
- Git-resolved author/committer confirmation
- Complete-index and owned-worktree preflight
- Immutable review snapshot and confirmation revalidation
- Hook bypass and signing disablement
- Retained process lifecycle and exact postflight
- Prepare-panel form, review, progress, outcomes, and recovery
- Focused automated and live acceptance coverage

### Excluded

- Push, pull, fetch, remotes, credentials, upstream selection, or network work
- Amend, fixup, squash, merge commits, initial commits, empty commits,
  sign-offs, trailers, templates, or signed commits
- Branch create, switch, delete, rename, repair, reset, checkout, restore, or
  arbitrary ref manipulation
- Hunk staging, full-repository status, unrelated-path disclosure, or general
  Git history browsing
- Hook configuration or execution
- Persistent commit queue, review snapshot, staging ownership, or crash journal
- Sparse checkout/index, nested repository management, submodule mutation, or
  unsupported semantic index normalization
- Any note-worktree or SQLite mutation beyond autosave settlement; required Git
  metadata writes and the secure temporary hooks-directory lifecycle remain
  part of the commit operation

## Chosen Git approach

Use a normal whole-index `git commit` after proving that every logical staged
difference belongs to current Chatbook staging ownership.

This is the smallest approach that preserves normal Git commit object, ref, and
reflog behavior. It deliberately refuses unrelated staged content rather than
building a private index.

The following alternatives are rejected:

- `git commit --only -- <paths>` can derive content from live worktree bytes
  instead of the reviewed staged entries and weakens deletion, mode, move, and
  later-edit guarantees.
- A trusted alternate index requires a new temporary-index security and
  reconciliation lifecycle.
- `commit-tree` plus `update-ref` would make Chatbook reproduce porcelain
  identity, reflog, and failure behavior.
- Normal hooks cannot coexist with the promise that the commit contains only
  the reviewed owned tree because a hook may stage another path.

## Ownership and component boundaries

### `FileNotesGitService`

The Git service remains the only layer that:

- discovers and inspects the repository;
- resolves author and committer identity;
- constructs and invokes Git commands;
- owns the child process through certain termination or uncertainty;
- parses Git output; and
- performs postflight proof.

No widget or workspace constructs Git commands or interprets raw Git output.

### `FileNotesSessionOwner`

The process-memory owner remains the sole authority for:

- selected-root and repository binding;
- process trust;
- session changes;
- staging ownership;
- authority generations;
- the exclusive Git-mutation gate; and
- atomic publication or invalidation after the operation.

The owner issues the generation/binding portion of the review snapshot. It
holds the mutation gate during snapshot construction and again from Confirm
revalidation through child completion, postflight, and state publication.

### Workspace

The workspace coordinates:

- the message draft scoped to the current repository binding;
- editor read-only lease and autosave settlement;
- operation workers and operation IDs;
- navigation/root invalidation;
- the opaque review snapshot; and
- rendering typed outcomes.

It never mutates the review snapshot or publishes Git ownership based on a
stale result. A superseded panel may ignore only stale UI projection by
operation ID; the process owner always consumes a running commit result,
performs postflight, publishes owner state, and releases the mutation gate and
editor lease.

### Prepare panel

The panel receives only sanitized immutable display models. It renders form,
review, progress, and result states and emits typed intents:

- `Commit staged`
- `Review commit`
- `Edit message`
- `Cancel commit`
- `Confirm commit`
- `Check again`

It contains no repository policy or Git authority.

## Review snapshot

Before review:

1. Acquire a tokenized editor read-only lease for the exact editor and binding.
2. Settle pending debounced autosave. A dirty, saving, conflict, or save-error
   state blocks and releases the lease.
3. Acquire the owner mutation gate for the same root/session generation.
4. Run complete preflight and create one opaque immutable snapshot.
5. Release the mutation gate while retaining the editor lease.

The snapshot contains:

- unique operation ID and owner generation;
- selected-root binding and complete repository identity;
- exact attached branch ref and old `HEAD` object ID;
- complete logical staged-delta/index signature;
- every included staging-ownership and topology signature;
- owned endpoint/closure worktree freshness;
- included session group identities and display facts;
- resolved author and committer names/emails;
- exact normalized message bytes; and
- the facts required to reproduce the review count and policy copy.

It contains object identifiers, modes, raw path identities, and signatures, but
not note bodies or blob contents. It is process-memory only and single-use.

The workspace stores it opaquely. The panel receives a separately sanitized
projection. Repository path, note path, identity, message, and diagnostic
display cannot inject Textual markup or invisible terminal controls.

No lock is held during human review. Confirm reacquires the mutation gate and
reruns the full preflight. Every security-relevant value must equal the
snapshot. Drift invalidates the review and returns to the form with the draft
preserved.

## Commit message

The form contains:

- required single-line Subject; and
- optional multiline Body.

Before review:

- trim surrounding subject whitespace;
- require 1-512 subject characters after trimming;
- normalize CRLF/CR to LF;
- remove surrounding blank lines from Body while preserving internal
  whitespace and line breaks;
- produce `subject\n` when Body is empty;
- otherwise produce `subject\n\nbody\n`;
- encode as UTF-8; and
- require the final encoded message to be at most 64 KiB.

Printable Unicode, emoji, and ordinary RTL text are valid. NUL, invalid Unicode,
unsafe terminal control characters, and directional-override controls are
rejected because Chatbook cannot provide a safe exact preview.

No 50/72-column style rule is enforced. No template, comment stripping,
sign-off, or trailer insertion occurs. The read-only review displays the exact
normalized message that will be sent to Git.

## Identity

Resolve both identities through Git under the exact sanitized repository
environment before review. Git's effective author and committer resolution,
not a direct read of only `user.name` and `user.email`, is authoritative.

Missing author or committer name/email blocks with a command-line recovery
instruction. Chatbook offers no identity editor and writes no configuration.

The review shows one `Identity` line when author and committer are equal and
separate `Author`/`Committer` lines otherwise. Confirm binds the reviewed
name/email values into the child environment so later configuration drift
cannot change them. Git chooses execution timestamps.

## Complete preflight

Preflight fails closed unless all of the following are freshly true:

- selected root and complete repository identity match current process trust;
- `HEAD` resolves to an existing commit through an attached `refs/heads/*`
  branch;
- branch ref and old `HEAD` match the snapshot at Confirm;
- no merge, rebase, cherry-pick, revert, bisect, sequencer, or equivalent
  worktree-specific operation is active;
- no relevant Git lock is already present;
- no unmerged or intent-to-add entry exists, and no staged/owned entry has
  gitlink mode, sparse state, an unsupported semantic index flag, or another
  unsupported state;
- the complete staged delta against old `HEAD`, including object IDs, modes,
  additions, deletions, and file/directory topology, exactly equals the union
  of current Chatbook-owned post-Stage entries/absences;
- every included group's repository, `HEAD`, topology, original baseline, and
  post-Stage ownership signatures remain valid;
- every included endpoint and required mutation closure has no newer saved
  worktree divergence from its staged state under trusted Git
  attributes/filters;
- no unrelated staged delta exists anywhere in the repository;
- message bytes and resolved identities match review; and
- no root/path transition or other File Notes mutation is active.

Unrelated unstaged paths are allowed and remain untouched.

Preflight compares worktree freshness through trusted Git semantics rather than
requiring raw disk bytes to equal staged blob bytes; clean filters can
legitimately transform content.

## Git execution contract

After exact Confirm revalidation, the process-owned service runs the equivalent
of:

```text
git \
  -c core.hooksPath=<private-empty-directory> \
  -c commit.gpgSign=false \
  -c i18n.commitEncoding=UTF-8 \
  commit \
  --no-gpg-sign \
  --cleanup=verbatim \
  -F -
```

The actual invocation is one direct argument vector with no shell. Exact
message bytes are supplied on stdin. The sanitized environment removes
repository/index/config redirection, disables terminal prompting and editor
invocation, and binds confirmed author/committer names and emails.

The private hooks directory:

- is created with owner-only access outside the notes repository;
- is verified empty before invocation;
- remains alive for the complete retained child lifetime;
- is removed only after certain child termination; and
- is left in place after uncertain termination rather than being removed under
  a potentially live child.

No repository or global configuration is modified. `--no-verify` is not treated
as sufficient because it does not bypass every commit hook; the empty
`core.hooksPath` does.

No option permits amend, empty/initial commit, signing, sign-off, trailer,
template editing, path-limited live-worktree commit, or remote access.

## Lifecycle and cancellation

The editor lease starts before autosave settlement and remains through review.
Releasing it removes only this flow's read-only reason; it cannot make an
editor writable when another condition still requires read-only state.

The Prepare, Checking, and Review states may be cancelled:

- form/review Cancel returns to the staged-note list;
- Escape in Review returns to message editing;
- read-only preflight cancellation waits for owned read-only child cleanup,
  releases the gate/lease, and restores prior focus.

After the branch-mutating child begins:

- Cancel and navigation controls are disabled;
- the operation remains process-owned across panel replacement/unmount;
- the mutation gate and editor lease remain until terminal publication;
- no caller cancellation can abandon the child; and
- application shutdown performs the existing bounded graceful/forced retained
  child procedure and publishes uncertainty whenever termination cannot be
  proved.

A force-killed Chatbook process loses its in-memory snapshot. On restart,
ordinary fresh status is available, but Chatbook cannot certify the prior
attempt.

## Postflight

Immediately after certain child termination, fresh Git inspection verifies:

- the same attached branch ref;
- branch movement from the exact old `HEAD` to one new commit;
- exactly one parent, equal to old `HEAD`;
- complete new commit tree equal to the expected reviewed index tree;
- exact normalized message bytes;
- reviewed author/committer names and emails;
- absence of a commit-signature header; and
- current logical index and worktree facts.

External worktree changes that appeared after final preflight do not invalidate
an otherwise exact commit. They are classified separately during the status
refresh.

## Typed outcomes

### `Cancelled`

The user stopped before branch mutation began. No commit child ran. Restore the
editor lease and prior focus and preserve the binding-scoped draft.

### `Blocked`

Preflight found a condition requiring user action. No commit child ran. Return
to the message form, restore editing, preserve the draft, focus the relevant
field/recovery, and give one exact next action.

Examples:

- `Stage update` for newer note edits;
- manage unrelated staged content outside Chatbook, then Refresh;
- configure Git identity outside Chatbook, then Review again;
- finish/abort the active Git operation outside Chatbook, then Check again; or
- switch/create a normal branch outside Chatbook, then Refresh.

### `Succeeded`

The child terminated normally with success and every postflight proof matched.

Atomically:

- advance owner facts to the new `HEAD`;
- retire included groups that are fully represented by the commit;
- keep any group with newer worktree edits;
- clear consumed staging ownership;
- refresh actual session status;
- clear the message draft; and
- report:
  `Committed N session notes as <short-oid>; unrelated changes untouched.`

### `Failed unchanged`

The child terminated normally with failure while postflight proved:

- branch still equals old `HEAD`;
- complete logical index entries/staged delta remain equivalent; and
- no unexpected Git operation or relevant lock appeared.

Index stat-cache changes and unreachable object writes do not count as
user-visible index mutation. Refresh worktree facts separately. Preserve the
draft and valid staging ownership, but consume the review snapshot. `Review
again` always generates a new snapshot.

### `Uncertain`

Use uncertainty for:

- timeout with uncertain child termination;
- exit/postflight contradiction;
- unexpected branch or `HEAD` movement;
- index change outside the approved result;
- missing or additional committed content;
- a commit whose tree or metadata does not match review;
- postflight failure that prevents proof; or
- any repository fact that matches neither exact success nor unchanged
  failure.

Never reset `HEAD`, index, or worktree; delete a Git lock; remove a possibly
in-use hooks directory; or automatically retry.

Clear cached status and staging ownership, preserve the message draft, and
disable Git mutations. Show:

```text
Commit may have succeeded. Git actions are disabled until the repository is
checked. Run git status and git log -1, then choose Check again.
```

`Check again` is unavailable while the retained child might still be alive or
a relevant lock/special operation remains. After certain termination, it
performs fresh repository discovery, identity verification, branch inspection,
and complete status. It reports whether the exact reviewed commit is present,
the old branch/index state remains, or the repository differs from both.

Fresh safe status may restore ordinary Stage/Unstage eligibility. Cleared
staging ownership is never recreated implicitly; affected groups must be
staged again before another commit review.

Repository trust remains only when complete repository identity is freshly
re-established.

## External concurrency

Normal porcelain commit has no cross-process compare-and-swap for Chatbook's
semantic precondition. External index or `HEAD` mutation between final preflight
and Git's own locks is unsupported.

Chatbook minimizes the window, bypasses hooks, and performs immediate
postflight. Postflight can detect an incorrect irreversible commit but cannot
safely undo it. Recovery copy and ADR-038 state that limitation directly.

## Prepare-panel interaction

### Staged-note list

- Always render `Commit staged (N)`.
- At zero, disable it with `Stage at least one session note to commit`.
- Activating it replaces the list/actions with the message form and focuses
  Subject.
- `Back to navigator` exists only at the list level.

### Message form

- Required Subject and optional Body
- Current branch and staged-note count
- `Cancel commit` and `Review commit`
- Escape cancels to the staged-note list
- Inline validation with focus returned to Subject or Body

The draft survives Edit, Cancel, blocked preflight, stale review, and definite
failure while the repository binding remains the same. Proven success clears
it. Repository rebinding clears it with an explicit explanation.

### Checking

- Show `Checking commit...`.
- Keep `Cancel commit` available because no branch mutation has begun.
- Restore editor, toolbars, and prior focus after cancellation/block.

### Review

Show:

- branch and old short commit ID;
- exact normalized message preview;
- resolved identity/identities;
- `N session notes will be committed; unrelated changes untouched`;
- change-type counts;
- included notes as New, Modified, Deleted, or `old -> new`;
- `Commit policy: Git hooks will not run · Commit will be unsigned`; and
- `Edit message`, `Cancel commit`, and `Confirm commit`.

Focus initially lands on `Edit message`, never Confirm. Enter activates only the
focused control. Escape returns to message editing.

The included-note disclosure is expanded only when count and available height
make it useful. Its large form is virtualized/incremental, and the focused row
exposes the complete sanitized path. Elision never hides the change type.

### Execution

Show:

```text
Committing N session notes...
Git is updating the branch; cancellation is unavailable.
```

Disable navigation and commit controls after the child begins.

### Result and recovery

The result/recovery region wraps and scrolls when required. It is not projected
through the incumbent two-line status elision that could hide the result,
promise, or next action.

Success returns to the refreshed staged-note list and focuses the first
remaining session change or `Back to navigator`. Blocked/failure recovery
focuses the relevant field or action. Uncertainty exposes only safe inspection
and `Check again`.

## Responsive and accessibility contract

- Keep form/review inside the approved Prepare panel; do not add a modal or
  separate screen.
- Keep the editor mounted. Existing editor action toolbars remain visually
  quiet in Prepare mode.
- Use a scrollable form/review body and a fixed visible action footer.
- At narrow sizes, Navigator and Editor remain alternate views rather than
  squeezed columns.
- At `40x20`, all action labels, policy disclosure, current outcome, and
  recovery action remain readable.
- Use visible text for ready, blocked, warning, running, success, failure, and
  uncertainty; color is supplemental.
- Use context-specific keyboard guidance and deterministic focus restoration.
- Never transfer the Enter used for Review into Confirm.
- Announce checking, committing, success, failure, and uncertainty through the
  incumbent Textual status/notification mechanism.

## Disk and SQLite boundary

The commit feature invokes no command intended to update worktree bytes and
calls no File Notes filesystem or replica mutation API after autosave
settlement.

Commit does not:

- rewrite frontmatter or body bytes;
- create, move, delete, restore, or protect notes;
- add replica rows or revisions;
- change tombstones;
- change session history except retiring/retaining in-memory session groups
  after proven Git facts; or
- make SQLite an authority.

Disk remains authoritative and SQLite remains an independent replica/recovery
store under ADR-029.

## Performance

- Run no Git subprocess per note. The preflight/postflight command count is
  bounded independently of included-note count.
- Use Git's NUL-delimited bulk machine formats and existing exact index models.
- Keep Git work over 100 ms off the Textual event loop.
- Retain signatures and object IDs, not note bodies.
- Virtualize or incrementally mount large review lists.
- Support a repository with at least 1,000 notes without broad repository UI
  enumeration.

## Focused verification

This work adds no multi-hour full-suite, network, remote, optional-dependency,
or broad performance gate.

### Unit and service tests

- deterministic message normalization, validation, control rejection, and
  UTF-8 bounds;
- author/committer resolution, missing identity, display collapse, and
  confirmation binding;
- complete staged-delta equality, modes, deletions, topology, unrelated staged
  blocking, and owned worktree freshness;
- attached/detached/unborn and special-operation detection;
- immutable/single-use review snapshot and generation drift;
- direct argv, exact stdin, sanitized environment, hook directory, signing and
  encoding overrides, and no shell/editor/prompt;
- success, unchanged failure, uncertainty, and later Check-again state;
- gate, retained child, shutdown, editor lease, and owner publication;
- bounded/sanitized hostile diagnostics and paths; and
- no per-note subprocess growth.

### Disposable-repository integration

Cover:

- modify, create, delete, mode, and grouped/chained move commits;
- exact parent, complete tree, message, identity, unsigned commit, and branch
  ref;
- a hook sentinel proving no hook runs;
- configured signing proving Chatbook still creates an unsigned commit;
- unrelated unstaged state remaining untouched;
- unrelated staged state blocking with unchanged branch/index;
- staged notes with newer worktree edits requiring Stage update;
- missing identity, detached/unborn branch, conflicts, special operations,
  index locks, unsupported index states, and repository replacement;
- trusted clean-filter worktree freshness rather than raw-byte comparison;
- definite failure versus simulated timeout/termination uncertainty;
- postflight `HEAD`/index drift and no rollback;
- success group retirement and retained newer worktree edits; and
- unchanged note bytes, SQLite replica, revisions, and tombstones.

### Mounted Textual tests

Cover:

- `Commit staged (0/N)` availability and explanation;
- message validation, binding-scoped draft, and exact preview;
- unambiguous Cancel/Edit/Back controls and Escape behavior;
- review focus never landing on Confirm;
- editor read-only lease and exact restoration;
- cancellation before child start and no cancellation afterward;
- wrapped non-elided recovery and fixed action footer;
- one-note and 1,000-note included-note presentation;
- operation-ID stale-result rejection; and
- wide plus `40x20` geometry and keyboard flow.

### Focused live UAT

Using a disposable real notes repository and the real TUI:

1. Edit and stage two session notes.
2. Leave one unrelated worktree change unstaged.
3. Enter a subject and multiline body.
4. Review branch, old commit, identity, count/list, hook policy, and unsigned
   policy.
5. Confirm and verify the exact commit with Git.
6. Confirm unrelated worktree state and note bytes remain untouched.
7. Exercise Cancel, unrelated-staged block, newer-edit block, and fresh-review
   recovery.
8. Repeat the interaction checks at a normal wide size and `40x20`.

## ADR check

ADR required: yes

ADR path:
`backlog/decisions/038-file-notes-guarded-session-commit.md`

Reason: local commit changes the Git/runtime security boundary, branch mutation,
identity handling, hook/signing policy, process lifecycle, outcome contract,
and long-lived Prepare-panel workflow. ADR-038 amends only ADR-035's explicit
commit exclusion and leaves its session-only staging and no-push boundaries in
force.
