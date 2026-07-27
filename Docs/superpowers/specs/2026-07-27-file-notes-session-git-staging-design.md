# File Notes Session Git Staging Design

Date: 2026-07-27
Task: [TASK-985](../../../backlog/tasks/task-985%20-%20Add-session-scoped-Git-status-and-staging-to-File-Notes.md)
Decision: [ADR-034](../../../backlog/decisions/034-file-notes-session-git-index-controls.md)

## Goal

Let users inspect and stage or unstage the paths changed during the current
Chatbook File Notes session without changing disk authority, taking ownership
of pre-existing index state, or expanding into commit and push workflows.

## Existing boundary

ADR-029 makes ordinary files beneath one selected root authoritative and keeps
Git external. File Notes already records process-lifetime `SessionChange`
entries for successful Chatbook create, modify, move, delete, and restore
operations. It does not record external filesystem changes as Chatbook session
changes.

This design amends only the Git boundary. The filesystem service and
`file_notes.sqlite` remain unaware of Git. No note bytes, recovery rows,
revisions, or session history are changed by a Git action.

## User experience

The Navigator keeps its current Files and search-result modes. A compact
`Session Git (N)` entry opens a third navigator mode instead of permanently
squeezing a potentially long Git list beneath the folder tree. Returning to
Files restores the previous tree/search state. On narrow terminals this remains
inside Navigator and does not remount or resize the editor.

The Session Git view contains:

- a header naming the repository and branch state;
- persistent `Session paths only` and `Stages complete file contents` labels;
- one selectable row per coalesced session change;
- a Git-state badge and disabled reason on each row;
- contextual `Stage selected`, `Stage update`, and `Unstage selected` actions;
- `Stage All` and `Unstage All` actions for eligible rows;
- one concise action/result status line.

The view never claims to show full repository status. Other repository changes
and staged paths are intentionally neither listed nor counted.

Rows use these user-facing states:

- `Unstaged`;
- `Staged by Chatbook`;
- `Staged by Chatbook · newer unstaged edits`;
- `Staged externally`;
- `Partially staged externally`;
- `Clean · currently matches HEAD`;
- `Ignored`;
- `Git conflict`;
- `Nested repository unsupported`;
- `Git unavailable` or `Error`.

A clean row remains visible because this surface is also the current session's
activity record. A created-then-deleted path or a file edited back to `HEAD`
therefore does not disappear.

## Components and ownership

### `FileNotesGitService`

A small UI-independent service owns repository discovery, session-change
coalescing, porcelain parsing, eligibility, index signatures, and Git command
execution. It is bound to one selected File Notes root and is optional.

The service discovers the nearest containing Git worktree. The repository root
may equal or contain the notes root. A notes root that itself belongs to a
submodule worktree is supported because that submodule is its containing
worktree. A session path that crosses into another nested worktree or submodule
beneath the selected root is visible but not actionable.

Repository identity includes the canonical worktree top-level and canonical Git
directory identities. Discovery is repeated before an index mutation. A moved,
replaced, or differently resolved repository invalidates the service and
requires the user to reopen Session Git.

### Workspace integration

`LibraryFileNotesWorkspace` owns only presentation, selected-row state, the
session-only repository trust confirmation, refresh scheduling, and worker
lifecycle. It asks the existing File Notes leave/flush path to settle pending
autosave before staging. A remaining dirty, saving, conflict, or error state
blocks the action.

The editor is not locked for the duration of Git. Once the initial flush
finishes, later typing and atomic autosaves may continue. If disk content
changes after `git add` reads it, the refreshed row truthfully becomes staged
with newer unstaged edits.

Git ownership and trust are process-lifetime memory only. Root changes and
application restart discard them. A path left staged after restart is reported
as externally staged and cannot be unstaged by Chatbook.

## Session change coalescing

Git rows describe effective path groups rather than every autosave event:

- repeated changes to one path form one row;
- a move groups its source and destination as one inseparable action;
- later changes to the move destination stay in that move group;
- chained moves retain every touched path as a literal pathspec while displaying
  the original source and final destination;
- create/delete, delete/restore, and edit/revert sequences remain visible even
  when their current Git result is clean.

Coalescing never broadens beyond paths present in Chatbook's in-memory session
changes. External changes to another repository path cannot enter a bulk action.

Ownership is tracked per literal endpoint under one repository/`HEAD`
generation, then projected onto the current coalesced groups. If a staged file
is later moved, or a staged move gains another chained endpoint, the expanded
group remains eligible for Unstage only when every saved staged endpoint still
matches and each new endpoint is freshly verified to have no staged delta. The
new endpoint's exact `HEAD`-equivalent index state—an entry when `HEAD` contains
the path or absence when it does not, including an unborn branch—is saved as a
no-op Unstage precondition; Chatbook does not claim it staged that endpoint. A
successful Stage update replaces the staged and no-op preconditions with the
expanded group's fresh owned signatures. Any unmatched endpoint blocks the
group and revokes aggregate Unstage eligibility instead of guessing how
histories should merge.

## Git command contract

Git is invoked directly with argument arrays, never through a shell. All
pathspecs are repository-relative, literal, and follow an explicit `--`
boundary. Machine output is consumed as bytes using NUL-delimited porcelain v2
records; filenames are decoded through the platform filesystem rules and
sanitized only for display.

The runner removes ambient variables that can redirect the Git directory,
worktree, common directory, index, object database, namespace, discovery, or
injected configuration. It retains the ordinary environment and Git
configuration needed for normal attributes and clean filters. Terminal prompts
are disabled.

Read-only status runs with optional locks and filesystem-monitor hooks disabled.
It has a short hard timeout because it does not own an index mutation. The
service capability-checks the installed Git version/commands instead of parsing
human-readable fallback output.

Index mutations are single-flight per service. They are not force-killed merely
because they are slow: `git add` may legitimately execute a clean filter. The
UI reports elapsed slow state while File Notes remains usable. Application
shutdown first requests graceful child termination and never deletes a
remaining Git lock file.

Git's normal clean filters and attributes, including Git LFS, remain active for
staging so the index matches ordinary command-line Git semantics. Because a
repository-local clean filter may execute a program, the first Stage action for
each selected root in a process requires a concise trust confirmation. Status
does not require that confirmation.

## Status and eligibility

Status is requested only for the union of literal paths in the coalesced
session groups. Ignored and unmerged records are retained so the UI can explain
why they are blocked.

Before staging, each group receives a fresh preflight:

- any pre-existing staged delta observed on an endpoint blocks the entire
  group;
- mixed staged/unstaged state not owned by Chatbook is `Partially staged
  externally`;
- unmerged entries, ignored paths, nested repository boundaries, an invalid
  repository identity, or unavailable Git are blocked;
- clean groups remain visible but have no Stage action;
- an already Chatbook-owned group may be staged again only while its saved
  ownership signature still matches.

Staging is path scoped, not hunk or provenance scoped. It stages the complete
current content of every eligible path, including edits another tool made to
that same path. The persistent whole-file label makes this limitation explicit.

## Stage flow

1. Flush the current File Notes autosave through the existing workspace guard.
2. Snapshot the coalesced group set and run fresh Git status/index preflight.
3. Obtain the process-lifetime repository trust confirmation if this is the
   first Stage action.
4. Revalidate repository identity.
5. Run one literal, path-scoped Git command for the selected group or all
   eligible groups. A move always supplies every endpoint.
6. Read `HEAD` identity and exact index entries again.
7. Claim ownership only when Git exited successfully, repository and `HEAD`
   identities remained stable, and the fresh entries are valid for every
   endpoint in the group.
8. Refresh all displayed session rows from actual Git state.

`Stage All` skips ineligible rows and reports the exact staged, already staged,
clean, and blocked counts. A nonzero or uncertain command result creates no new
ownership claim, even if a later refresh finds an index change.

## Unstage flow

An owned group's Unstage signature contains:

- repository identity;
- the `HEAD` object ID, or an explicit unborn-branch marker;
- each Chatbook-staged endpoint's exact index mode, object ID, and stage number;
- explicit absence for deleted source paths;
- any later move endpoint's verified no-op `HEAD`-equivalent entry or absence.

Before Unstage, Chatbook compares the complete current signature with the saved
one. A different `HEAD`, conflict stage, entry, or absence revokes ownership and
changes the row to externally staged. Chatbook does not attempt a merge.

When the signature still matches, Git restores only those index paths to their
known precondition: each path's `HEAD` entry, or absence when `HEAD` does not
contain the path or the branch is unborn. Later move endpoints are included in
the inseparable group but already equal that target, so they remain no-ops;
only entries Chatbook staged are reversed. The worktree is never restored or
modified. `Unstage All` includes only currently valid Chatbook-owned groups. It
then refreshes actual index state before clearing ownership.

## Refresh and concurrency

Status refresh is:

- immediate when Session Git opens;
- debounced after a File Notes session mutation;
- immediate after a Git action;
- available through manual Refresh;
- polled slowly only while Session Git is visible.

Every query runs in a worker and uses a generation token so stale results cannot
replace newer state. Git mutations are serialized with one service lock.

External Git can still race between Chatbook's final preflight and Git's own
index lock. Git prevents index-file corruption, but no porcelain Git CLI
compare-and-swap exists for Chatbook's semantic precondition. The supported
contract therefore excludes concurrent external index mutation during one
Chatbook Stage or Unstage action. Chatbook minimizes that window with an
immediate preflight, reports lock contention, rereads actual state after every
command, and revokes ownership whenever the result is uncertain. It does not
claim that post-command inspection can prove an external writer did not finish
inside the remaining race window.

## Error behavior

All failures remain local to Session Git:

- missing/unsupported Git and non-repository roots disable Git controls only;
- `safe.directory` and ownership errors are shown without changing global or
  local Git configuration;
- an existing index lock reports `Git index busy; retry`; Chatbook never removes
  it;
- status timeout leaves the prior view marked stale;
- bounded, control-character-sanitized stderr supplies the row or action error;
- repository disappearance or identity change invalidates the service;
- filter/command failure is followed by a fresh status read and no unproven
  ownership claim.

Chatbook never initializes or repairs a repository, changes configuration,
invokes remotes or credentials, or edits the worktree through Git.

If users run another index-mutating Git command concurrently with a Chatbook
action, ordinary Git last-writer behavior may affect the same session path.
That unsupported race is disclosed in the Session Git help text; it is not
misrepresented as an atomic cross-process ownership guarantee.

## Focused verification

No full-suite, network, remote, submodule-download, hung-process, or
combinatorial platform gate is part of this task.

Fast unit tests cover:

- NUL-delimited porcelain parsing;
- session lineage/coalescing;
- row-state and eligibility policy;
- exact ownership signature comparison;
- command construction and sanitized environment boundaries;
- simulated status timeout and preflight/result races.

A compact disposable-repository matrix covers:

- repository root equal to and above the notes root;
- modify, create, delete, restore, and grouped/chained moves;
- per-group and bulk stage/unstage;
- unrelated worktree and index entries remaining untouched;
- pre-existing and partially staged same-path blocking;
- external index and `HEAD` changes revoking ownership;
- staged content followed by newer unstaged edits;
- ignored, conflict, detached, unborn, nested, and replaced-repository states;
- supported names containing spaces, leading dashes, and pathspec characters;
- redirecting Git environment variables being unable to change the target;
- exact worktree bytes, SQLite replica state, and File Notes session history
  remaining unchanged by Stage and Unstage.

One controlled filter fixture proves status executes no configured filter and
Stage reaches the filter only after trust confirmation.

Mounted Textual tests cover Session Git navigator switching, retained drafts,
selected and bulk actions, blocked rows, action summaries, nonfatal Git errors,
worker refresh, and narrow layout. Acceptance testing repeats the primary flow
against one disposable real repository and inspects both `git diff` and
`git diff --cached`.

## Explicit non-goals

Commit, push, pull, fetch, remotes, credentials, branch mutation, hunk staging,
full-repository status, repository initialization/repair, persistent staging
ownership, nested-repository management, and Git-based worktree file restore.
