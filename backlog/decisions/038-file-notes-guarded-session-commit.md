# ADR-038: File Notes Guarded Session Commit

Status: Accepted
Date: 2026-07-29
Related Task: [TASK-1350](../tasks/task-1350%20-%20Add-guarded-session-commit-to-File-Notes.md)
Amends: [ADR-035 File Notes Session Git Index Controls](035-file-notes-session-git-index-controls.md)
Conforms to:
[ADR-033 Application Session State Ownership](033-application-session-state-ownership.md),
[ADR-029 File Notes disk authority](029-file-notes-disk-authority.md), and
[ADR-011 Chatbook Workbench UI System](011-chatbook-workbench-ui-system.md)

## Context

ADR-035 lets Chatbook inspect, stage, update, and unstage only the paths
recorded in the current File Notes application session. It deliberately leaves
commit and push outside Chatbook because commit introduces identity, hooks,
signing, branch mutation, and an irreversible shared-state effect.

That staging boundary is now shipped and accepted. Users still have to leave
the otherwise complete `Prepare session for commit` workflow to create the
local commit. The requested next slice is one guarded, reviewed local commit of
the staged session-note state Chatbook can still prove it owns. It is not a
request for a general Git client.

A normal Git commit consumes the complete current index. A path-limited
porcelain command can derive content from the live worktree rather than from
Chatbook's approved staged object IDs, and normal hooks can add unrelated
content after preflight. Concurrent external index or `HEAD` mutation cannot
be made transactionally atomic with a porcelain commit. The design must make
those constraints explicit rather than imply stronger isolation.

## Decision

- Add `Commit staged (N)` inside the existing File Notes
  `Prepare session for commit` panel. One operation commits every currently
  staged session group whose exact staging ownership remains valid. Unstaged
  session changes and unrelated unstaged repository changes remain untouched.
- Use a normal whole-index `git commit`, but only after a complete-index
  preflight proves that the entire logical staged delta against the captured
  old `HEAD` exactly equals the union of current Chatbook-owned post-Stage
  entries and absences. Any unrelated staged delta, any unmerged or
  intent-to-add entry, or any staged/owned entry with an unsupported semantic
  flag, gitlink mode, or other unsupported state blocks the commit.
- Treat the complete-index proof path as one narrow, internal, read-only
  exception to ADR-035's session-path status parser. It may transiently inspect
  repository-wide logical-index metadata, including unchanged entries, only
  for preflight equality, postflight classification, and `Check again`
  recovery. It disables rename detection, external diffs, text conversion, and
  configured filesystem monitors; publishes only a generic unrelated-state
  category; and never exposes, persists, or adds unrelated paths to session
  status or ownership.
- Support only an existing attached `refs/heads/*` branch with a resolved old
  commit. Detached and unborn `HEAD` are blocked. Merge, rebase, cherry-pick,
  revert, bisect, sequencer, or equivalent in-progress repository operations
  are blocked using worktree-specific Git state.
- Treat partial-clone/promisor repositories and a present common-Git-dir
  `info/grafts` file as unsupported. Detect them from local config/filesystem
  metadata before any object-resolving command. Set no-lazy-fetch semantics as
  defense in depth, and block when a required object is absent locally; the
  operation never contacts a promisor remote.
- Require every included owned group to have no newer saved worktree state
  relative to its staged state. The user must use `Stage update` before
  reviewing a commit that would otherwise omit newer note edits. Unrelated
  unstaged paths do not block.
- Before creating a review, make the Chatbook editor read-only, settle pending
  autosave, and briefly acquire the File Notes Git-mutation gate. The service
  and session owner produce one immutable, in-memory review snapshot containing
  the binding and repository identities, exact branch and old `HEAD`, complete
  staged-delta and ownership signatures, owned-path worktree freshness,
  included groups, resolved identities, and exact message bytes.
- Hold no mutation gate while the user reads the review. Confirmation
  reacquires the gate and holds it through complete revalidation, commit child
  execution, postflight, and atomic owner-state publication. Any snapshot drift
  invalidates the review instead of committing stale intent.
- Keep the review snapshot process-memory only. Its confirmation capability is
  single-use. If an attempt becomes uncertain, retain only the immutable proof
  evidence needed by `Check again` until a definite result, repository
  rebinding, or process exit. The panel receives a separate sanitized display
  projection and cannot construct or alter security-relevant fields. No
  database schema, commit queue, or persistent in-flight recovery journal is
  added.
- Require a single-line subject and optional multiline body. Normalize the
  reviewed UTF-8 message deterministically to `subject\n` or
  `subject\n\nbody\n`, with LF line endings, trimmed surrounding subject
  whitespace, surrounding body blank lines removed, and internal body
  whitespace preserved. Reject input that Chatbook cannot safely and exactly
  preview, and bound the final UTF-8 message to 64 KiB.
- Let Git resolve author and committer from the existing sanitized process
  environment and repository configuration. Missing identity blocks.
  Display both identities, collapsing them when equal, and bind their confirmed
  names and emails into the commit child. Remove ambient author/committer date
  overrides from resolution and execution. Chatbook does not invent, edit, or
  persist identity; Git chooses execution timestamps.
- Invoke Git through a direct argument vector with message bytes on stdin and
  no shell. Use a command-scoped, private empty hooks directory so no
  pre-commit, prepare-commit-msg, commit-msg, or post-commit hook runs. Disable
  commit signing through command-scoped configuration and `--no-gpg-sign`.
  Disable configured filesystem monitors and automatic maintenance/legacy
  auto-GC; force replacement-ref-free raw object semantics, UTF-8 commit
  encoding, and verbatim cleanup; and disable editor and terminal prompting.
- Create only a normal non-empty, non-initial commit. Do not amend, add
  sign-offs or trailers, initialize or repair the repository, edit Git
  configuration, or access remotes.
- Retain ownership of the commit child independent of the mounted panel.
  `Cancel` is available during form, review, and read-only preflight, but not
  after the branch-mutating child begins. The Git-mutation gate and editor
  read-only lease remain owned until a terminal result.
- Classify each attempt as `Cancelled`, `Blocked`, `Succeeded`,
  `Failed unchanged`, or `Uncertain`. Immediate success requires normal
  successful child termination plus proof that the same branch advanced from
  the captured old `HEAD` to exactly one unsigned commit whose sole parent,
  complete tree, message, author, and committer match the confirmed review, and
  that the complete logical index equals that new tree with no staged delta.
- Classify a normal nonzero result as `Failed unchanged` only when the branch
  and complete logical index state still match their captured values. Git
  index stat-cache changes and unreachable object writes are not user-visible
  index mutation. Refresh worktree state independently.
- Treat timeout with uncertain termination, contradictory child/postflight
  results, unexpected branch or index changes, additional or missing committed
  content, or inability to verify repository facts as `Uncertain`. Never reset
  `HEAD`, index, or worktree; never delete Git locks; and never retry
  automatically. Clear cached status, quarantine captured staging ownership so
  it cannot authorize an action, and disable further Git mutations until the
  retained child is certainly gone and fresh repository discovery/status
  succeeds. Retain that quarantined ownership only while process-memory
  recovery evidence remains.
- `Check again` converges when later proof is exact. If the same branch tip is
  the matching reviewed child of old `HEAD` and the complete logical index
  equals its tree with no staged delta, publish the normal `Succeeded` result.
  If the branch and complete logical index are still the captured old state and
  the retained child is now known to have terminated normally without success,
  publish `Failed unchanged` and reactivate the still-valid quarantined
  ownership. Any other repository state leaves the prior attempt `Uncertain`;
  it never invents ownership from fresh status alone.
- On proven success, retire only session groups fully represented by the
  commit. Preserve any group with newer post-commit worktree edits and refresh
  actual status. SQLite note content, revisions, protected snapshots, and
  tombstones remain unchanged.
- Keep the form, review, progress, and result states inside the existing
  Prepare panel. Use explicit controls, non-color status labels, deterministic
  focus, a scrollable review body, a fixed action footer that stacks to two
  rows at narrow widths, and the existing scrollable included-note list so the
  workflow remains keyboard-operable at `40x20`. Add incremental mounting only
  if focused measurement of session-row volume requires it.
- This ADR supersedes ADR-035's slice-level exclusion of commit and permits only
  the internal repository-wide logical-index proof path defined
  above. ADR-035's user-visible session-only status/staging, trust, exact index
  ownership, mutation gate, configured-filter disclosure,
  external-index-race limitation, and no
  push/remote/general-branch-management boundaries remain in force.

## Consequences

- Users can complete the local save-to-Git step without leaving Chatbook while
  retaining ordinary files as the note authority and SQLite as an independent
  replica/recovery store.
- Blocking unrelated staged state is intentionally stricter than command-line
  Git. Users must manage that state outside Chatbook before confirming a
  session-only commit.
- Hooks and signing are deliberately bypassed so repository commit hooks cannot
  run, broaden, or block the reviewed commit. ADR-035's separate trusted Git
  attribute/clean-filter execution and side-effect disclosure remains in
  force. Configured filesystem monitors and automatic Git maintenance are also
  disabled for the operation so no unowned helper or detached maintenance
  process escapes the retained-child lifecycle. The review discloses the
  hook/signing policy before confirmation.
- The editor is temporarily read-only from review preparation through review
  completion. Cancel, block, or terminal completion releases only this flow's
  read-only lease.
- A forced process kill after confirmation loses the in-memory review facts.
  On restart, Chatbook can show fresh Git state but cannot certify the prior
  attempt; the user must inspect `git status` and `git log -1`.
- External mutation after final preflight can still produce an irreversible
  unexpected commit before postflight detects it. This race is unsupported,
  clearly disclosed, and never "repaired" automatically.
- The approved `unrelated changes untouched` result means the exact commit
  included no unrelated staged content and Chatbook selected no unrelated
  worktree path. It does not claim that concurrent external tools could not
  change repository files. Review and success make that definition visible,
  and review also reminds users that included notes use their complete staged
  file state, not only edits attributable to Chatbook.
- Push, pull, fetch, upstream selection, remotes, credentials, signed commits,
  amend, and general branch management remain future, separately designed
  work.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Keep commit entirely external | Preserves ADR-035 unchanged but leaves the requested local completion step outside the otherwise complete Prepare workflow. |
| `git commit --only -- <paths>` | It can derive the commit from live worktree content rather than Chatbook's exact staged object IDs and complicates deletions, modes, moves, and message stdin. |
| Use a private alternate index | It could isolate unrelated real-index entries but introduces trusted `GIT_INDEX_FILE`, secure temporary-index construction, reconciliation, and cleanup complexity not needed when unrelated staged state may block. |
| Build the tree with `commit-tree` and update the ref directly | It offers a stronger custom compare-and-swap design but requires Chatbook to reproduce porcelain identity, reflog, signing, and error semantics. |
| Run normal repository hooks | A hook can stage unrelated paths after Chatbook's final preflight, invalidating the exact reviewed-content promise. |
| Preserve configured commit signing | Signing helpers may prompt or hang and would make reviewed completion dependent on external agent state. The approved slice explicitly creates unsigned commits. |
| Persist an in-flight recovery journal | It improves crash-after-confirm recovery but adds a new durable operational-state contract. This first commit slice stays process-memory only. |
| Add commit and push together | Push introduces remotes, credentials, upstream selection, network failure, and external effects beyond one reviewed local commit. |

## Links

- [Design specification](../../Docs/superpowers/specs/2026-07-29-file-notes-guarded-session-commit-design.md)
- [ADR-035](035-file-notes-session-git-index-controls.md)
- [ADR-033](033-application-session-state-ownership.md)
- [ADR-029](029-file-notes-disk-authority.md)
- [ADR-011](011-chatbook-workbench-ui-system.md)
- [TASK-1350](../tasks/task-1350%20-%20Add-guarded-session-commit-to-File-Notes.md)
