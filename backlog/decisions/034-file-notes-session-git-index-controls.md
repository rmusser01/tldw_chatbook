# ADR-034: File Notes Session Git Index Controls

Status: Accepted
Date: 2026-07-27
Related Task: [TASK-1023](../tasks/task-1023%20-%20Add-session-scoped-Git-status-and-staging-to-File-Notes.md)
Amends: [ADR-029 File Notes disk authority](029-file-notes-disk-authority.md)
Conforms to: [ADR-033 Application Session State Ownership](033-application-session-state-ownership.md)

## Context

ADR-029 keeps Markdown/text files authoritative, stores an independent SQLite
search/recovery replica, and leaves Git external. File Notes now provides a
usable disk-backed work session and records only the paths Chatbook changed.
That log currently belongs to one workspace/service instance even though
Library navigation creates fresh screens, so it does not yet satisfy the
intended application-session lifetime.

Users want the next reversible Git step available without turning Chatbook into
a general Git client. The Git index can already contain work staged through
other tools, and staging a path includes all current content for that path
rather than only edits attributable to Chatbook. Unstage is particularly unsafe
unless Chatbook can prove it is reversing its own index action.

## Decision

- Add a narrow process-memory File Notes session owner, coordinated by
  `TldwCli` and injected into fresh Library workspaces, for the selected-root
  session-change log, repository trust, staging ownership, mutation gate, and
  optional Git service lifecycle. It is the sole authority for those values,
  not a general root state object. Root change or process exit clears it;
  screen navigation does not.
- Bind the optional Git service to that owner's one selected notes root and its
  nearest containing Git worktree. The service, not a mounted widget, owns
  child execution and postflight through completion and atomically publishes
  checked results through the session owner; it keeps no second trust or
  ownership authority. `TldwCli` shuts the owner down even without a mounted
  Library screen, using bounded graceful/forced child termination, publishing
  uncertainty/no ownership, settling postflight before replica cleanup, and
  never removing a Git lock file.
- Show status only for paths present in current-process Chatbook session
  changes. Never describe it as full repository status.
- Coalesce repeated changes and treat every move's source/destination lineage as
  one inseparable Git group. Preflight every lineage endpoint and the tracked
  ancestor/descendant closure Git could mutate. Block the group if that closure
  contains a path outside its session lineage; pass only effective safe
  endpoints to Stage.
- Offer selected-group and bulk Stage/Unstage actions. Staging covers complete
  file state—content, deletion, and mode—and is labelled as such.
- Freshly preflight and refuse groups with observed pre-existing or partially
  staged same-path state, unmerged entries, ignored paths, or nested repository
  boundaries.
- Refuse nondefault semantic index states such as `skip-worktree`,
  `assume-unchanged`, and intent-to-add. Include semantic flags in saved
  ownership signatures so later flag changes revoke Unstage authority.
- Treat active sparse-checkout or sparse-index repositories as unsupported in
  this slice rather than infer cone membership for untracked paths.
- Before Stage, save each affected entry's exact stage-0 index baseline or
  absence. A Stage update retains every previously owned entry's original
  baseline and captures baselines only for newly affected clean entries. After
  success, atomically record the approved group topology plus repository,
  `HEAD`, and exact post-Stage signatures. Unstage only while those values
  still match.
- Restore owned baselines with one NUL-delimited
  `git update-index -z --index-info` operation using saved stage-0 entries and
  mode-zero removals. Before inserting a baseline, compute the current index
  file/directory replacement closure and block unless every entry it could
  displace exactly matches the group's owned post-Stage set; explicitly remove
  those expected owned conflicts before baseline additions. Do not reconstruct
  from live `HEAD`, broaden through an Unstage pathspec, or select a
  worktree-restoring operation.
- If a later move changes an owned group's endpoint topology, disable Unstage
  until a successful Stage update safely establishes the expanded topology.
  Do not create ownership records for no-op endpoints.
- Keep Git ownership and repository trust in process memory. Restart, root
  change, or external index/`HEAD` change revokes Chatbook's Unstage authority.
- Define repository identity as canonical worktree top-level,
  worktree-specific Git directory, and Git common directory plus each
  location's platform-stable filesystem identity (`st_dev`/`st_ino` where
  meaningful). Revalidate immediately after trust and before every
  worktree-aware status or mutation; identity change clears trust.
- Invoke Git through direct argument arrays and literal pathspecs with
  repository/index/config redirecting environment variables removed. Disable
  status rename detection, require NUL-delimited complete untracked output,
  reject parsed paths outside the session whitelist, and force
  `add.ignoreErrors=false` for Stage.
- Run read-only status without optional index writes or filesystem-monitor
  hooks. Preserve normal Git attributes and clean filters for worktree-aware
  status and Stage, preceded by one process-lifetime trust confirmation per
  selected root/repository identity. Show the canonical repository path and
  process-only scope; Cancel has initial focus and Escape/close decline.
- Replace the unbounded session-change summary with a dedicated Session Git
  navigator mode whose state/action mapping, loading/stale/error controls,
  keyboard/Back behavior, and selection remain stable across refresh.
- Do not poll Git status in this slice. Permit one query at a time and coalesce
  concurrent refresh triggers into at most one rerun. Hidden views become stale
  instead of launching Git, and mutations wait for an active status query.
- Before mutation, flush autosave, recheck root/path transition state, and
  acquire a separate Git-mutation gate. Block screen/root transitions and
  structural file actions through postflight while editor input, autosave,
  replica synchronization, and in-screen `Back to Files` remain usable. Git
  coordination must not reuse filesystem or replica locks.
- Keep File Notes usable when Git is absent, unsupported, locked, unsafe,
  replaced, untrusted, or failing. Chatbook selects no Git command intended to
  update note bytes or SQLite state; trusted filter side effects are disclosed
  as outside its guarantee.
- Do not add commit, push, remote, credential, branch, hunk, repository
  initialization, repair, or full-status behavior.

## Consequences

- Users can prepare the exact set of paths touched during a Chatbook work
  session while retaining their existing command-line commit/push workflow.
- Leaving and reopening Library preserves session rows, trust, and still-valid
  Unstage authority without caching a screen.
- Existing staged content on the same path is deliberately managed outside
  Chatbook; this prevents index-state loss at the cost of sometimes disabling a
  row.
- Chatbook can safely reverse only exact saved baselines it still proves it
  owns. A restart, topology change without Stage update, or external Git change
  may require command-line unstage.
- File/directory replacement closure can disable a row whose literal path would
  cause Git to mutate a non-session ancestor or descendant.
- Sparse repositories are read as unsupported rather than receiving unreliable
  staging behavior in this slice.
- Worktree-aware status and Stage may execute trusted clean/process filters,
  matching normal Git semantics. Declining trust runs neither operation.
- A Git mutation temporarily blocks structural File Notes actions and screen
  departure, but not editing, autosave, replica work, or in-screen navigation.
- Git remains an optional projection/action surface. Disk remains the note
  authority and SQLite remains an independent recovery replica.
- Concurrent external index mutations cannot be made transactionally atomic
  with Chatbook's semantic preflight. Concurrent external index mutation during
  one Chatbook action is unsupported. Git protects index-file integrity;
  Chatbook minimizes the race, detects observable uncertainty afterward, and
  drops ownership, but does not claim a cross-process compare-and-swap.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Keep all Git actions external | Preserves the old boundary but omits the requested reversible staging workflow. |
| Add commit and push in the same slice | Introduces author identity, hooks, credentials, remotes, upstream selection, and irreversible shared-state effects before index ownership is proven. |
| Keep session state on `LibraryFileNotesWorkspace` | Fresh-screen navigation would discard session rows, trust, and ownership before the application process ends. |
| Use GitPython | Adds a dependency without removing the need for explicit path, index, filter, and concurrency policy. |
| Trust literal pathspecs as exact mutation targets | File/directory replacements can make `git add` mutate tracked ancestors or descendants outside the named path. |
| Support sparse checkout immediately | Correct handling requires active sparse-definition/cone membership policy, including untracked paths with no index flag; blocking it is smaller and safer for this slice. |
| Unstage with a path-oriented HEAD restore | It can broaden through path matching and makes unborn/detached behavior depend on reconstructing a live baseline rather than restoring the saved one. |
| Stage only inferred Chatbook hunks | Chatbook cannot reliably attribute same-file edits from external tools, and hunk staging is a separate interaction model. |
| Unstage any current-session path | Can erase index state staged before or outside Chatbook. |
| Build a general VCS abstraction | Adds speculative structure for an explicitly Git-only requirement. |

## Links

- [Design specification](../../Docs/superpowers/specs/2026-07-27-file-notes-session-git-staging-design.md)
- [ADR-029](029-file-notes-disk-authority.md)
- [ADR-033](033-application-session-state-ownership.md)
- [TASK-1023](../tasks/task-1023%20-%20Add-session-scoped-Git-status-and-staging-to-File-Notes.md)
