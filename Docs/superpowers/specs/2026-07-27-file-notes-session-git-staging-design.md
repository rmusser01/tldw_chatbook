# File Notes Session Git Staging Design

Date: 2026-07-27
Task: [TASK-1213](../../../backlog/tasks/task-1213%20-%20Add-session-scoped-Git-status-and-staging-to-File-Notes.md)
Decision: [ADR-035](../../../backlog/decisions/035-file-notes-session-git-index-controls.md)
Conforms to: [ADR-033](../../../backlog/decisions/033-application-session-state-ownership.md)

## Goal

Let users inspect and stage or unstage the paths changed during the current
Chatbook File Notes session without changing disk authority, taking ownership
of pre-existing index state, or expanding into commit and push workflows.

## Existing boundary

ADR-029 makes ordinary files beneath one selected root authoritative and keeps
Git external. File Notes currently records `SessionChange` entries for
successful Chatbook create, modify, move, delete, and restore operations in a
service instance. Because Library navigation creates a fresh screen and
workspace, this feature moves that log into a narrow application-process owner
consistent with ADR-033. External filesystem changes remain excluded from
Chatbook session changes.

This design amends only the Git boundary. The filesystem service and
`file_notes.sqlite` remain unaware of Git. The Git service does not write note
bytes, recovery rows, revisions, or session history, and it never selects a Git
operation intended to update the worktree. Worktree-aware Git commands may run
user-configured clean/process filters after explicit trust; arbitrary side
effects from those programs are disclosed and outside Chatbook's guarantee.

## User experience

The Navigator keeps its current Files and search-result modes. The existing
unbounded `Session changes: ...` text is replaced by a compact
`Session Git (N)` entry, where `N` is the number of coalesced session groups.
The entry opens a third navigator mode instead of permanently squeezing a
potentially long Git list beneath the folder tree. Returning to Files restores
the previous tree/search state. On narrow terminals this remains inside
Navigator and does not remount or resize the editor.

The Session Git view contains:

- a visible `Back to Files` control and a header naming the canonical
  repository path and branch state;
- persistent `Session paths only` and `Stages complete file state (content,
  deletion, and mode)` labels;
- one selectable row per coalesced session change;
- a Git-state badge and disabled reason on each row;
- contextual `Stage selected`, `Stage update`, and `Unstage selected` actions;
- `Stage All` and `Unstage All` actions for eligible rows;
- one concise action/result status line.

The view never claims to show full repository status. Other repository changes
and staged paths are intentionally neither listed nor counted.

Before the first worktree-aware status for a selected root in a process, the
view shows the canonical repository path, explains that Git status and staging
may execute configured filters, states that trust lasts only for this
application process, and asks for trust. Initial focus is `Cancel`; `Escape`,
closing the prompt, or declining all mean no trust and no worktree-aware Git
command. File Notes remains fully usable and offers an explicit `Trust and
check status` retry.

Rows use this action contract:

| State | Selected-row actions | Stage All | Unstage All |
| --- | --- | --- | --- |
| `Unstaged` | `Stage selected` | Included | Excluded |
| `Staged by Chatbook` | `Unstage selected` | Excluded | Included |
| `Staged by Chatbook · newer unstaged edits` | `Stage update`, `Unstage selected` | Included as update | Included |
| `Staged by Chatbook · path lineage changed` | `Stage update`; Unstage disabled until update | Included as update | Excluded |
| `Staged externally` | Disabled: external index state | Excluded | Excluded |
| `Partially staged externally` | Disabled: external index state | Excluded | Excluded |
| `Clean · currently matches HEAD` | No action | Excluded | Excluded |
| `Ignored` | Disabled: ignored | Excluded | Excluded |
| `Git conflict` | Disabled: conflict | Excluded | Excluded |
| `Unsupported Git index state` | Disabled with flag reason | Excluded | Excluded |
| `Nested repository unsupported` | Disabled: nested repository | Excluded | Excluded |
| `Git unavailable` or `Error` | Disabled with reason | Excluded | Excluded |

Each coalesced group has a stable process-local identity based on its earliest
session-change sequence. Refresh retains the selected row while that group
still exists, including when a later move expands its endpoint lineage.

A clean row remains visible because this surface is also the current session's
activity record. A created-then-deleted path or a file edited back to `HEAD`
therefore does not disappear.

The view has one explicit keyboard contract. `Up` and `Down` move row
selection, `Tab` and `Shift+Tab` move among visible controls, and `Enter`
activates only the focused control. `Escape` or `Back to Files` returns to the
prior Files/search view, restores focus to the `Session Git (N)` entry, and
retains the mounted editor, content, cursor, and selection when switching
between Navigator and Editor on a narrow terminal.

While status is checking or an index mutation is running, Git mutation controls
are disabled; the editor and `Back to Files` remain usable. A stale or failed
status labels the retained rows as stale, disables Stage/Unstage controls, and
leaves `Refresh` enabled for a trusted repository. A successful refresh clears
the stale state. `Trust and check status` is reserved for the untrusted state.
The status line reports the latest action and its result rather than silently
changing button eligibility.

## Components and ownership

### `FileNotesSessionOwner`

A narrow process-memory owner, coordinated by `TldwCli` and injected into each
fresh Library workspace, owns the selected-root session-change log, repository
trust grants, staging ownership records, Git-mutation gate, and the optional
Git service lifecycle. It is the sole authority for those values; the Git
service publishes checked results through it and no screen or service keeps a
second ownership copy. It is not a general application-state container and is
not persisted or mirrored onto the screen. This follows ADR-033's rule that
application-session values live with the smallest owner sharing their lifecycle.

Leaving and reopening Library therefore preserves the current File Notes
session rows, trust, and valid Unstage authority. Selecting another notes root
clears the prior root's File Notes session state; application shutdown discards
all of it. A path left staged after restart is reported as externally staged
and cannot be unstaged by Chatbook.

### `FileNotesGitService`

A small UI-independent service computes repository discovery, session-change
coalescing, porcelain parsing, eligibility, and index signatures and owns Git
command execution. It is bound to the session owner's one selected File Notes
root and is optional. It owns each child process and postflight through
completion, then atomically publishes its checked result through the session
owner; it does not independently own trust, staging records, or the mutation
gate. A mounted Textual worker may await and render a result but never owns that
lifecycle. Screen unmount cannot leave an unobserved Git mutation or start a
duplicate replacement.

The service discovers the nearest containing Git worktree. The repository root
may equal or contain the notes root. A notes root that itself belongs to a
submodule worktree is supported because that submodule is its containing
worktree. A session path that crosses into another nested worktree or submodule
beneath the selected root is visible but not actionable.

Repository identity includes the canonical worktree top-level, canonical
worktree-specific Git directory, and canonical Git common directory, plus the
platform-stable filesystem identity of each resolved location
(`st_dev`/`st_ino` where meaningful). Identity is revalidated after trust
confirmation and before every worktree-aware status or index mutation. A
moved, replaced, or differently resolved repository invalidates the service and
its trust grant, requiring the user to reopen Session Git and trust the new
identity.

### Workspace integration

`LibraryFileNotesWorkspace` owns only presentation, selected-row state, trust
prompt presentation, and refresh requests. It asks the existing File Notes
leave/flush path to settle pending autosave before staging. A remaining dirty,
saving, conflict, or error state blocks the action. Trust is required before
the first worktree-aware status or index mutation for the selected root, not
merely before Stage.

The editor is not locked for the duration of Git. Once the initial flush
finishes, later typing and atomic autosaves may continue. If disk content
changes after `git add` reads it, the refreshed row truthfully becomes staged
with newer unstaged edits.

After the flush, staging rechecks that no root/path transition is active and
atomically acquires the session owner's Git-mutation gate. That gate blocks
root switching, leaving the Library screen, and create, move, delete, restore,
and save-copy actions until postflight finishes. It does not block editor input,
autosave, replica synchronization, or returning from Session Git to Files
inside the same workspace. Git status/mutation coordination uses locks
independent of the existing filesystem-service and reconciliation locks.

At application shutdown, `TldwCli` shuts down the process owner even when no
Library screen is mounted. It requests graceful child termination, waits a
bounded interval, then force-stops remaining owned child work and publishes an
uncertain/no-ownership result rather than waiting indefinitely. The Git
service's postflight lifecycle settles before File Notes replica cleanup, and
Chatbook never deletes a Git lock file. Trust remains process-memory only and
is keyed by the selected root plus the complete repository identity.

## Session change coalescing

Git rows describe effective path groups rather than every autosave event:

- repeated changes to one path form one row;
- a move groups its source and destination as one inseparable action;
- later changes to the move destination stay in that move group;
- chained moves retain every touched path as a lineage endpoint while
  displaying the original source and final destination;
- create/delete, delete/restore, and edit/revert sequences remain visible even
  when their current Git result is clean.

Coalescing never broadens beyond paths present in Chatbook's in-memory session
changes. External changes to another repository path cannot enter a bulk action.
Every lineage endpoint participates in boundary, eligibility, and ownership
preflight. Before Stage, Chatbook computes the tracked ancestor/descendant
closure that Git could add or remove for each literal endpoint. This covers
file-to-directory and directory-to-file replacements where a literal pathspec
can still affect a tracked path above or below the named endpoint. A group is
blocked if any path in that mutation closure is outside its session lineage.
Mutation pathspecs are then derived only from closure paths whose index state
the action can change. A transient move-chain path absent from `HEAD`, the
index, and worktree remains visible in the lineage but is never passed as an
unmatched mutation pathspec.

Ownership is tracked per literal endpoint under one repository/`HEAD`
generation, then projected onto the current coalesced groups. If a staged file
is later moved, or a staged move gains another chained endpoint, the expanded
group immediately loses Unstage eligibility because its endpoint topology no
longer matches the topology approved by the last Stage. `Stage update` remains
eligible only when every previously owned post-Stage signature still matches,
each added endpoint has no external staged delta, and the expanded mutation
closure is safe. A successful Stage update records the expanded topology plus
fresh post-Stage ownership for entries actually changed by Chatbook. It retains
the original pre-Stage baseline for every previously owned entry and captures a
new baseline only for newly affected entries whose fresh preflight was clean.
No no-op endpoint ownership is invented. Any mismatch blocks the group instead
of guessing how histories should merge.

## Git command contract

Git is invoked directly with argument arrays, never through a shell. All
pathspecs are repository-relative, literal, and follow an explicit `--`
boundary. Machine output is consumed as bytes using NUL-delimited porcelain v2
records; filenames are decoded through the platform filesystem rules and
sanitized only for display.

Stage uses path-scoped `git add --all` semantics so creations, modifications,
and tracked deletions are handled together. A selected or bulk action is one
logical operation over eligible groups, but only effective mutation pathspecs
whose complete ancestor/descendant mutation closure passed preflight are
supplied to Git; lineage-only endpoints are omitted. Every Stage invocation
forces `add.ignoreErrors=false`, so Git cannot report partial path success under
a user configuration that asks `git add` to continue after an indexing error.

The runner removes ambient variables that can redirect the Git directory,
worktree, common directory, index, object database, namespace, discovery, or
injected configuration. It retains the ordinary environment and Git
configuration needed for normal attributes and clean filters. Terminal prompts
are disabled.

After trust, read-only status runs with optional locks and filesystem-monitor
hooks disabled, explicit `--untracked-files=all`, matching ignored records, and
rename detection disabled. Move relationships come only from Chatbook's
session lineage. Parsed status paths must belong to the requested session
whitelist; an out-of-whitelist record fails closed rather than entering a row
or mutation. Status has a short hard timeout because it does not own an index
mutation. The service capability-checks the installed Git version/commands
instead of parsing human-readable fallback output.

An active sparse-checkout or sparse-index repository is unsupported in this
slice and disables Git mutation controls with an explicit reason. In
particular, an untracked file has no index `skip-worktree` flag from which
Chatbook could reliably infer whether ordinary `git add` will accept it under
the active sparse definition. Supporting sparse cones is deferred instead of
guessing.

Index mutations are single-flight per process-owned service. They are not
force-killed merely because they are slow: `git add` may legitimately execute a
clean filter. The UI reports elapsed slow state while File Notes remains usable.
The service, rather than a mounted widget, owns child termination, postflight,
and the final result.

Git's normal clean filters and attributes, including Git LFS, remain active so
status and staging match ordinary command-line Git semantics. Worktree-aware
status can invoke the same configured clean/process filters while comparing
worktree content with the index. The first such status or Stage action for each
selected root/repository identity in a process therefore requires the concise
trust confirmation.

## Status and eligibility

Status is requested only for the union of literal paths in the coalesced
session groups. Ignored and unmerged records are retained so the UI can explain
why they are blocked.

Before staging, each group receives a fresh preflight:

- any pre-existing staged delta observed on an endpoint blocks the entire
  group;
- mixed staged/unstaged state not owned by Chatbook is `Partially staged
  externally`;
- unmerged entries, ignored paths, nested repository boundaries, active sparse
  checkout/index state, an invalid repository identity, or unavailable Git are
  blocked;
- an endpoint present as a directory, symlink, or other non-regular worktree
  type is blocked before its path can reach a mutation command;
- an ancestor/descendant mutation closure containing a path outside the
  selected session lineage is blocked;
- endpoints with nondefault semantic index state, including `skip-worktree`,
  `assume-unchanged`, or intent-to-add, are blocked rather than normalized;
- clean groups remain visible but have no Stage action;
- an already Chatbook-owned group may be staged again only while its saved
  ownership signature still matches.

Staging is path scoped, not hunk or provenance scoped. It stages the complete
current file state of every eligible path—including content, deletion, and
executable-mode changes—and includes edits another tool made to that same path.
The persistent complete-file-state label makes this limitation explicit.

## Stage flow

1. Flush the current File Notes autosave through the existing workspace guard.
2. Discover and revalidate repository identity using commands that do not
   inspect worktree content; identity change clears the prior trust grant.
3. Obtain process-lifetime trust for the current root/repository identity if
   no trusted status has run for it.
4. Immediately revalidate the complete worktree/Git-directory/common-directory
   identity after the prompt. A change discards the grant and returns to the
   untrusted state before any worktree-aware command runs.
5. Recheck that no root/path transition is active and atomically acquire the
   Git-mutation gate for the same root and session generation.
6. Snapshot the coalesced group set; run fresh status/index preflight; compute
   each literal endpoint's ancestor/descendant mutation closure; retain every
   still-owned entry's original baseline; and save the exact current index
   entry or absence as baseline only for newly affected entries the action can
   change.
7. Revalidate repository identity again immediately before mutation; a change
   aborts without running a worktree-aware command for the new identity.
8. Run one literal, path-scoped `git add --all` operation with
   `add.ignoreErrors=false` over the effective mutation pathspecs for the
   selected group or all eligible groups. Every move-lineage endpoint is
   preflighted, but an absent transient endpoint is omitted.
9. Read `HEAD` identity and exact index entries again.
10. Claim ownership only when Git exited successfully, repository and `HEAD`
    identities remained stable, and the fresh entries are valid for every
    endpoint in the group. Save the approved group topology, merged original/new
    baselines, and exact post-Stage signature together.
11. Finish postflight and release the mutation gate even if the originating
    screen unmounted. Refresh displayed rows if Session Git remains visible;
    otherwise mark the view stale for its next open.

`Stage All` skips ineligible rows and reports the exact staged, already staged,
clean, and blocked counts. A nonzero or uncertain command result creates no new
ownership claim, even if a later refresh finds an index change.

## Unstage flow

An owned group's Unstage signature contains:

- repository identity;
- the `HEAD` object ID, or an explicit unborn-branch marker;
- the exact group-endpoint topology approved by its last successful Stage;
- each Chatbook-changed index entry's exact post-Stage mode, object ID, stage,
  and semantic flags;
- the same entry's exact pre-Stage stage-0 mode/object ID, or explicit absence.

Before Unstage, Chatbook compares the complete current signature with the saved
one and freshly preflights every current endpoint. A changed topology disables
Unstage until a successful Stage update. A different `HEAD`, conflict stage,
owned post-Stage entry, semantic flag, or absence revokes ownership; an
external staged delta on any other endpoint blocks the inseparable group. A
fresh status result then derives the row's actual state; Chatbook does not
force an external-staged label or attempt a merge.

Before reinserting any saved baseline entry, Chatbook also computes its current
index file/directory replacement closure. `update-index` can otherwise remove
an index ancestor or descendant while inserting a conflicting path even though
no pathspec is used. Every such conflicting current entry must exactly match an
owned post-Stage entry in the same group. An external or unexpected conflict
blocks Unstage.

When the signature still matches, one NUL-delimited
`git update-index -z --index-info` operation restores only Chatbook's saved
pre-Stage entries. The NUL stream first contains explicit mode-zero removals
for every expected owned file/directory conflict, then saved stage-0
mode/object records for existing baselines and mode-zero removals for baselines
that were absent. This exact index primitive does not reconstruct state from
live `HEAD`, does not broaden through pathspec matching, and selects no
worktree-restoring operation. It works for normal, detached, and unborn `HEAD`
states because the baseline was captured before Stage. `Unstage All` includes
only currently valid Chatbook-owned groups. Postflight rereads actual index
state, reports uncertainty, and consumes or revokes the ownership record as
appropriate.

## Refresh and concurrency

Status refresh is:

- immediate after trust when Session Git opens;
- debounced after a File Notes session mutation while Session Git is visible;
- immediate after a Git action only if Session Git remains visible;
- available through manual Refresh.

This first slice does not poll Git status. At most one status query runs for the
selected root. Triggers arriving during a query coalesce into at most one rerun
using the latest session snapshot. No status query starts while an index
mutation is active, and a mutation waits for the active status query before
running its own fresh preflight. When a mutation finishes, it schedules one
refresh only if the view remains visible; otherwise it marks the view stale.
Session mutations while the Git view is hidden do the same. Reopening performs
the refresh. Every query uses a generation token so stale results cannot replace
newer state. The process-owned Git service serializes mutations and coordinates
status with Git-specific locks that are independent of File Notes filesystem
and replica locks.

Leaving and re-entering Library reuses the same process session owner. A
pending status can finish into that owner while no Session Git view is mounted;
its result is rendered only when still current. A mutation prevents screen
navigation until its service-owned postflight completes, while `Back to Files`
inside Library remains available. Workspace unmount or worker cancellation
cannot cancel the underlying child lifecycle or publish a partial ownership
result.

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

- declined or missing trust runs no worktree-aware status or index mutation;
- missing/unsupported Git and non-repository roots disable Git controls only;
- `safe.directory` and ownership errors are shown without changing global or
  local Git configuration;
- an existing index lock reports `Git index busy; retry`; Chatbook never removes
  it;
- status timeout leaves the prior view marked stale;
- bounded, control-character-sanitized stderr supplies the row or action error;
- repository disappearance or identity change invalidates the service;
- status or filter failure leaves the prior view stale without an automatic
  retry loop;
- mutation failure creates no ownership claim and schedules at most one normal
  post-action refresh when visible, otherwise marking the view stale.

Chatbook never initializes or repairs a repository, changes configuration,
invokes remotes or credentials, or selects a Git operation intended to edit the
worktree. The trust prompt discloses that a user-configured filter is an
arbitrary program whose own side effects cannot be constrained by Chatbook.

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
- application-session owner root/lifecycle behavior;
- exact ownership, topology, and saved-baseline comparison;
- Stage pathspec and Unstage index ancestor/descendant mutation-closure
  rejection;
- semantic index-flag blocking and ownership invalidation;
- exact `update-index --index-info` input construction;
- command construction, forced fail-fast add configuration, status path
  whitelisting, and sanitized environment boundaries;
- simulated status timeout and preflight/result races.

A compact disposable-repository matrix covers:

- repository root equal to and above the notes root;
- linked-worktree identity including its Git common directory;
- modify, create, delete, restore, and grouped/chained moves;
- transient move-chain endpoints remaining in lineage while being omitted from
  effective mutation pathspecs;
- per-group and bulk stage/unstage;
- unrelated worktree and index entries remaining untouched;
- file/directory replacement collisions being blocked when Git would affect a
  tracked ancestor or descendant outside the session lineage;
- owned file/directory replacements unstaging through explicit conflict
  removals while an unexpected external index ancestor/descendant blocks
  Unstage unchanged;
- pre-existing and partially staged same-path blocking;
- external index and `HEAD` changes revoking ownership;
- move topology changes requiring a successful Stage update before Unstage;
- staged content followed by newer unstaged edits, including Stage update then
  Unstage restoring the original baseline;
- ignored, conflict, detached, unborn, nested, sparse-checkout, sparse-index,
  and replaced-repository states;
- supported names containing spaces, leading dashes, and pathspec characters;
- user-configured `add.ignoreErrors=true` being overridden by fail-fast Stage;
- redirecting Git environment variables being unable to change the target;
- the primary end-to-end flow inspecting both `git diff` and
  `git diff --cached`;
- Chatbook's command paths requesting no worktree, SQLite replica, or File Notes
  session-history mutation.

One controlled filter fixture proves declining trust runs neither status nor
the filter, accepting trust allows status and Stage to reach the configured
filter, and Chatbook itself requests no worktree update.

Mounted Textual tests cover Session Git navigator switching, retained drafts,
the state/action table, safe trust-prompt focus, keyboard/Back behavior, stable
row selection, selected and bulk actions, checking/stale/error controls, blocked
rows, action summaries, nonfatal Git errors, and narrow layout. Leaving and
re-entering Library proves session rows, trust, and valid ownership survive
fresh screen construction.

A delayed-runner fixture proves refresh triggers coalesce to one in-flight
query plus at most one rerun. It also proves editor input and `Back to Files`
complete while Git is pending; structural file actions and screen/root
transitions are blocked only during mutation; unmount cannot duplicate a Git
operation; and application shutdown settles the service-owned child/postflight
lifecycle with bounded termination and no ownership claim before replica
cleanup, including when no Library screen is mounted.

One focused scale fixture uses at least 1,000 unrelated notes and a small
session set. It asserts only session rows and pathspecs are produced and the
Files/search state survives Session Git navigation; this slice adds no
pagination or broad performance harness.

## Explicit non-goals

Commit, push, pull, fetch, remotes, credentials, branch mutation, hunk staging,
full-repository status, repository initialization/repair, persistent staging
ownership, nested-repository management, sparse-checkout/index support, and
Git-based worktree file restore.
