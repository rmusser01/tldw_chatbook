# ADR-034: File Notes Session Git Index Controls

Status: Accepted
Date: 2026-07-27
Related Task: [TASK-985](../tasks/task-985%20-%20Add-session-scoped-Git-status-and-staging-to-File-Notes.md)
Amends: [ADR-029 File Notes disk authority](029-file-notes-disk-authority.md)

## Context

ADR-029 keeps Markdown/text files authoritative, stores an independent SQLite
search/recovery replica, and leaves Git external. File Notes now provides a
usable disk-backed work session and records only the paths Chatbook changed
during that process lifetime.

Users want the next reversible Git step available without turning Chatbook into
a general Git client. The Git index can already contain work staged through
other tools, and staging a path includes all current content for that path
rather than only edits attributable to Chatbook. Unstage is particularly unsafe
unless Chatbook can prove it is reversing its own index action.

## Decision

- Add an optional File Notes Git service bound to the one selected notes root
  and its nearest containing Git worktree.
- Show status only for paths present in current-process Chatbook session
  changes. Never describe it as full repository status.
- Coalesce repeated changes and treat every move's source/destination lineage as
  one inseparable Git group.
- Offer selected-group and bulk Stage/Unstage actions. Staging is whole-file,
  path-scoped behavior and is labelled as such.
- Refuse groups with pre-existing or partially staged same-path state, unmerged
  entries, ignored paths, or nested repository boundaries.
- Record exact repository, `HEAD`, and index-entry signatures after a successful
  Chatbook stage. Unstage only while that complete signature still matches.
- Keep Git ownership and repository trust in process memory. Restart or external
  index/`HEAD` change revokes Chatbook's Unstage authority.
- Revalidate canonical worktree and Git-directory identity before mutation.
- Invoke Git through direct argument arrays and literal pathspecs with
  repository/index/config redirecting environment variables removed.
- Run read-only status without optional index writes or filesystem-monitor
  hooks. Preserve normal Git attributes and clean filters for Stage, preceded by
  one process-lifetime trust confirmation per selected root.
- Keep File Notes usable when Git is absent, unsupported, locked, unsafe,
  replaced, or failing. Git errors never mutate note bytes or SQLite state.
- Do not add commit, push, remote, credential, branch, hunk, repository
  initialization, repair, or full-status behavior.

## Consequences

- Users can prepare the exact set of paths touched during a Chatbook work
  session while retaining their existing command-line commit/push workflow.
- Existing staged content on the same path is deliberately managed outside
  Chatbook; this prevents index-state loss at the cost of sometimes disabling a
  row.
- Chatbook can safely reverse only staging it still proves it owns. A restart or
  external Git change may require command-line unstage.
- Stage may execute trusted repository clean filters, matching normal Git
  semantics. Merely opening Session Git does not execute them.
- Git remains an optional projection/action surface. Disk remains the note
  authority and SQLite remains an independent recovery replica.
- Concurrent external index mutations cannot be made transactionally atomic
  with Chatbook's semantic preflight. Git protects index-file integrity;
  Chatbook detects uncertainty afterward and drops ownership.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Keep all Git actions external | Preserves the old boundary but omits the requested reversible staging workflow. |
| Add commit and push in the same slice | Introduces author identity, hooks, credentials, remotes, upstream selection, and irreversible shared-state effects before index ownership is proven. |
| Use GitPython | Adds a dependency without removing the need for explicit path, index, filter, and concurrency policy. |
| Stage only inferred Chatbook hunks | Chatbook cannot reliably attribute same-file edits from external tools, and hunk staging is a separate interaction model. |
| Unstage any current-session path | Can erase index state staged before or outside Chatbook. |
| Build a general VCS abstraction | Adds speculative structure for an explicitly Git-only requirement. |

## Links

- [Design specification](../../Docs/superpowers/specs/2026-07-27-file-notes-session-git-staging-design.md)
- [ADR-029](029-file-notes-disk-authority.md)
- [TASK-985](../tasks/task-985%20-%20Add-session-scoped-Git-status-and-staging-to-File-Notes.md)
