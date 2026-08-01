# ADR-039: File Notes Guarded Session Push

Status: Accepted
Date: 2026-07-30
Related Task: [TASK-1711](../tasks/task-1711%20-%20Add-guarded-exact-session-commit-push-to-File-Notes.md)
Amends: [ADR-038 File Notes Guarded Session Commit](038-file-notes-guarded-session-commit.md)
Conforms to:
[ADR-035 File Notes Session Git Index Controls](035-file-notes-session-git-index-controls.md),
[ADR-033 Application Session State Ownership](033-application-session-state-ownership.md),
[ADR-029 File Notes disk authority](029-file-notes-disk-authority.md),
[ADR-011 Chatbook Workbench UI System](011-chatbook-workbench-ui-system.md), and
[ADR-029 Local Private Data Boundary](029-local-private-data-boundary.md)

## Context

ADR-038 permits Chatbook to create one reviewed local commit only after proving
that the complete staged delta is exactly the current File Notes session state
Chatbook owns. It intentionally leaves push external because a push introduces
remote selection, authentication, network ambiguity, server-side mutation, and
effects that cannot be recovered by repairing local repository state.

That guarded commit is now a sufficiently narrow authority source for an
equally narrow publish action. The requested next slice is not "push the
current branch" or "push everything ahead of upstream." It is to offer, in the
same application process, a separate reviewed action that can publish exactly
the one guarded commit Chatbook just created and proved. Commits that predate
that operation, commits created outside Chatbook, and commits reconstructed
from repository history after restart are outside Chatbook's authority.

Normal `git push` defaults are too broad for this promise. A remote name may
resolve through changing push configuration, an implicit refspec may select
more than the reviewed ref, a missing destination may be created, and ordinary
failure output cannot prove whether a server accepted an update before the
connection was lost. Local pre-push hooks may launch unrelated work, while
Git LFS relies on such a hook to upload objects before the ref update. A
credible design must bind authorization to one effective destination, perform
an exact server-side compare-and-swap, own its child lifecycle, and preserve an
honest uncertain state when the result cannot be proved.

## Decision

### Exact candidate authority

- Add a separate `Review push (1 commit)…` action to the existing File Notes
  `Prepare session for commit` workflow. A successful guarded commit does not
  automatically open, confirm, or start a push.
- The application-session File Notes owner atomically publishes at most one
  private push candidate when ADR-038's guarded commit succeeds. The candidate
  binds the exact session/root and repository identities, attached local
  branch, parent and new commit OIDs, included session-note provenance, and
  monotonic binding, repository-trust, and candidate generations. The panel
  receives a separate sanitized projection and cannot construct or alter
  authority fields.
- Candidate authority is process-memory only and belongs to the exact proven
  guarded-commit result. Restart, root or repository rebinding, trust
  invalidation, or lineage drift removes availability. Chatbook never infers a
  candidate from `HEAD`, reflog, ahead counts, commit messages, or repository
  history after the fact.
- Candidate generation is independent of the session-change generation.
  Continued note editing and autosave may create newer disk and SQLite replica
  state without changing the already-created commit or invalidating its push
  candidate.
- A later guarded commit may replace an unpushed candidate, but it does not
  broaden the new candidate to include its predecessor. If the existing
  upstream is not exactly the new candidate's parent, guarded push is
  unavailable. Stale callbacks and results may retire only the exact candidate
  token they started with and cannot clear or overwrite a newer candidate or
  owner state.

### Existing destination and separate authorization

- Support only the one existing tracking upstream of the candidate's attached
  local branch. The upstream must resolve unambiguously to one configured
  remote, one effective push endpoint, and one existing full
  `refs/heads/*` destination. Chatbook never creates, recreates, selects, edits,
  or repairs a remote, tracking relationship, or branch.
- Resolve the effective push endpoint and relevant Git policy without network
  or credential-helper contact. Block a missing or deleted destination,
  multiple push URLs, mirror configuration, configured push refspecs or push
  options, a conflicting `branch.*.pushRemote` or `remote.pushDefault`, a
  custom receive-pack, ambiguous URL rewriting, or other configuration that
  could change the destination or broaden the requested operation.
- Before the first network query, HTTPS credential-helper contact, or SSH-agent
  contact, require a
  separate process-only `Authorize configured destination` decision. It binds
  a monotonic authorization epoch to the exact sanitized effective endpoint,
  destination ref, transport, repository identity, and relevant configuration
  authentication-helper policy, and SSH host-trust fingerprints. It is not
  persisted, does not assert that Chatbook trusts the remote's content or
  operator, and does not itself contact the network.
- Permit only secure HTTPS with normal certificate verification and standard
  OpenSSH/scp-style SSH with batch operation and existing host-key
  verification. Plain HTTP, `git://`, local paths, `file://`, external/custom
  transport helpers, ambiguous URL forms, embedded secret material, unknown
  SSH hosts, disabled certificate checks, and repository-controlled executable
  credential or SSH helpers block.
- Use only existing noninteractive authentication: a pinned SSH agent for SSH,
  or an installed credential helper allowed by policy for HTTPS. Chatbook never
  prompts for, reads, displays, edits, stores, or configures a password, token,
  private key, credential helper, or certificate exception. For SSH it reads
  only the standard public host-key trust files through bounded local
  descriptor reads, retains their exact bytes in process authority, and copies
  them only into the owner-only temporary network context. The authorization UI
  discloses that the existing SSH agent or an approved credential helper may be
  contacted after authorization and that terminal prompts are disabled.
- For SSH, bind the exact authorized URL host, username, and port in a
  Chatbook-owned immutable OpenSSH invocation. Do not reread live user SSH
  routing/command configuration after authorization. Capture, safely read, and
  fingerprint the ordered standard user and system `known_hosts` sources during
  local destination proof. Missing sources contribute an explicit missing fact
  and yield an empty strict-trust snapshot when all are absent; an unsafe,
  unreadable, unstable, symlinked, hard-linked, or oversized present source
  blocks before network contact. Confirm recaptures the sources and revokes the
  review on drift. The private context stores one owner-read-only snapshot,
  pins `UserKnownHostsFile` to it, and sets `GlobalKnownHostsFile=none`.
- SSH authentication is agent-only. The immutable invocation pins the proved
  agent socket, sets `IdentityFile=none` so no default or configured private-key
  file is read, and keeps `IdentitiesOnly=no` so identities already exposed by
  that agent remain usable. No pinned agent means SSH guarded push blocks before
  network contact. Host aliases, ProxyCommand/ProxyJump, custom IdentityFile
  selection, and other behavior requiring live user SSH config remain
  external-Git-only in this slice.
- Any change to the bound endpoint, ref, transport, repository identity,
  relevant configuration, helper policy, host-trust source fingerprint, or
  pinned-agent identity revokes the authorization even if values later change
  back. Monotonic epochs prevent value-level ABA from reviving stale authority.

### Remote proof and exact ref update

- After authorization, perform a read-only remote preflight against the frozen
  endpoint and exact destination ref. If the ref reports the candidate's
  parent, build a review. If it already reports the candidate, report
  `Already published` and do not start a push. A missing ref, another OID,
  ambiguous response, inaccessible destination, or inability to prove the
  exact state blocks review.
- The immutable review binds the exact root, repository/trust, candidate,
  destination-policy/configuration, authorization, operation, and
  remote-preflight facts, including the applicable SSH trust snapshot and
  pinned-agent policy. It deliberately excludes session-change,
  Git-authority, status, staging, index, and worktree generations that may
  advance through later note edits without changing the commit. Confirmation
  consumes one single-use capability and freshly revalidates only the
  push-relevant lineage, configuration/authorization epochs, and remote ref
  before any push child starts. Push-relevant drift invalidates the review
  instead of silently changing its target.
- Invoke the frozen effective endpoint directly rather than invoking a remote
  name. Request exactly
  `<candidate-oid>:<destination-refs/heads/ref>` guarded by
  `--force-with-lease=<destination-ref>:<parent-oid>`. The candidate is already
  proven to be the direct child of that parent, so the requested update remains
  a fast-forward; the explicitly named lease is a compare-and-swap that
  prevents a concurrent deletion, recreation, or divergent advance from being
  overwritten.
- Pass no implicit refspec, tags, push options, delete, mirror, force-without-an
  exact expected OID, upstream-setting option, submodule recursion, or
  automatic retry. Use direct argument vectors without a shell, disable
  terminal prompting and stdin, and bypass local pre-push hooks with
  `--no-verify`.
- Do not let a network child reread live source-repository, worktree, global,
  or system Git configuration after authorization. Create one owner-only
  temporary bare network-execution Git directory with no refs/remotes/hooks,
  disable external config sources, copy only exact approved noninteractive
  helper values from the authorized snapshot, omit all URL rewrites and
  broadening settings, and pass the frozen endpoint only as an argument. Give
  that isolated Git directory read-only object access through a
  Chatbook-controlled alternate rooted at the verified common object
  directory.
- Prove and pin the source repository object format locally before creating
  the network context. Confirm freshly re-proves it, context-use seams require
  the exact format-bound source authorization and object-directory identity,
  the private bare repository uses the same format, and only matching-width
  OIDs are accepted. A missing, unsupported, changed, or mismatched format
  blocks before network/helper contact. SHA-1 and SHA-256 repositories remain
  separate authorities; Chatbook never guesses or translates formats.
- Retain the immutable execution context, including any ephemeral SSH
  host-trust snapshot, through review, children, postflight, and query-only
  recovery. Remove it only after every owned descendant is terminal and no
  recovery needs it. A crash-left owner-only temporary directory may contain
  public host-key trust metadata but contains no credentials, private keys, or
  note content; it is never discovered or reused after restart and is not a
  durable push journal.
- Because Git LFS depends on the bypassed pre-push hook to publish required
  objects, block the operation when any path included by the candidate is
  governed by Git LFS. Chatbook does not attempt to emulate LFS upload or run
  another repository hook.
- Invoking the frozen URL is also a local-mutation boundary: Chatbook does not
  ask Git to update a local remote-tracking ref. Read-only checks disable
  optional index refresh and filesystem monitors. The operation does not
  select commands intended to mutate local `HEAD` or refs, the index,
  repository/worktree configuration, note bytes, File Notes replica rows,
  revisions or tombstones, or session history.

### Platform and private-artifact boundary

- Guarded push is admitted only on POSIX platforms where Chatbook can prove
  the owner/mode properties used by the isolated network context. On Windows,
  guarded push is unavailable and fails closed before context creation,
  credential-helper contact, SSH launch, or any network child. Existing
  Windows Job Object process-containment support does not establish an
  owner-only discretionary ACL for these private artifacts and therefore does
  not admit this workflow.
- Windows guarded push requires separately approved native owner-only ACL
  design and implementation work. This ADR does not approximate that boundary
  with `chmod`, broaden the current task into Windows ACL management, or claim
  partial guarded-push support on Windows.
- The context root and Git directories are owner-only. After construction,
  child-visible `HOME`, `XDG_CONFIG_HOME`, and `TMP`/`TEMP`/`TMPDIR` directories
  are read/execute-only to the owner and their exact modes are pinned through
  cleanup. An SSH context's combined host-trust snapshot is owner-read-only and
  its identity, size, mode, and digest are pinned through cleanup. Git, OpenSSH,
  and approved helpers must not require scratch writes there. The documented
  local threat boundary continues to trust processes running as the same
  effective UID and root; mode bits do not isolate one same-UID process from
  another.
- Public context and lease objects carry no reachable authority, lifecycle,
  or release-token fields. Exact-instance weak registries bind them to frozen
  authority facts and inaccessible mutable lifecycle bookkeeping, so copying,
  aliasing, or attribute mutation cannot transfer or redirect capability.

### Operation ownership and recovery

- The existing application-session Git service owns remote checks, the push,
  postflight, and recovery checks independently of any mounted panel. Each
  external operation uses a bounded, noninteractive child lifecycle in an
  isolated POSIX process group and retains descendants through terminate,
  force-kill, and output drain. The shared runner may contain other Windows
  children with a Job Object, but guarded push itself fails closed on Windows
  under the platform boundary above. If owned descendant termination cannot be
  proved, the outcome remains uncertain.
- Pre-existing SSH agents, credential services, connection masters, server
  processes, remote hooks, CI, and mirrors are outside Chatbook's child
  ownership. The UI and diagnostics do not claim that terminating a local
  child cancels work that a server may already have accepted.
- Use a minimal allowlisted child environment that preserves only required
  noninteractive Git/authentication behavior, including only the frozen SSH
  agent socket for SSH, and excludes provider tokens,
  ambient Git repository/index/config redirects, ambient prompt controls,
  author overrides, and other unrelated secrets. Install Chatbook's own
  noninteractive prompt controls after removing ambient overrides. Bound
  captured output and translate it into typed, sanitized categories; raw Git,
  SSH, and credential-helper output is not shown or written to persistent
  logs.
- Reuse ADR-035's Git-mutation gate for candidate/repository revalidation,
  remote operation, and settlement. Conflicting Git actions and root, source,
  or screen rebinding wait while an operation is active or unresolved.
  Ordinary editor input, debounced autosave, and replica synchronization
  continue, and later note edits remain local.
- External local repository mutation after the push child starts cannot alter
  the immutable remote request or erase a remote result that can still be
  proved. It does invalidate fresh local availability. Publication is guarded
  by the operation and candidate tokens so a stale result cannot overwrite
  newer owner or status state.
- `Cancel` is available while preparing, authorizing, and checking and may
  terminate a retained read-only check. The cancellation boundary is the
  service's actual push-child-start event. Once the network push child starts,
  no cancel control is offered and the service owns the operation through a
  terminal or uncertain result.
- Panel removal does not cancel or duplicate work. The application-session
  owner retains the operation identity and public status; a remounted panel
  reattaches to it. Shutdown uses bounded settlement and preserves uncertainty
  when ownership or proof cannot be completed.

### Outcomes

- Use distinct typed outcomes:
  - `Already published` means a pre-push check observed the exact candidate and
    Chatbook started no push.
  - `Succeeded` initially requires Git to report that the exact update was
    accepted and a postflight check to observe the candidate at the frozen
    destination.
  - `Failed with no update currently observed` requires a normal unsuccessful
    Git result, every owned child and descendant known terminal, and a
    postflight check that still observes the exact parent.
  - `Uncertain` covers timeout, unknown child/helper termination, transport or
    response loss, contradictory Git and remote evidence, a missing or
    different postflight ref, or inability to query the frozen destination.
- All outcome claims are point-in-time observations, not proof of causation or
  absence of later external ABA. In particular, observing the parent after a
  timeout or lost response does not prove failure because accepted server work
  may still be pending.
- Never retry automatically. A definite failure may create a new review only
  after fresh local and remote preflight. An uncertain result destroys its
  reusable review and push-mutation authorization. It retains only immutable
  original-destination evidence and a narrowly query-only recovery authority
  while the bound trust policy remains unchanged.
- Offer `Check remote again — no push` only after every owned descendant is
  terminal. It queries only the retained original endpoint and ref, never
  follows changed Git configuration and never sends another update. If trust
  policy changed, querying the original destination requires a fresh
  authorization for that same frozen identity.
- A recovery check that observes the candidate may converge the desired remote
  state to success without claiming Chatbook caused it. Observing the parent
  after a previously uncertain attempt remains uncertain; another OID, a
  missing ref, or a failed query remains `needs attention`.
- Process exit discards candidates, authorizations, review capabilities,
  operation attribution, and recovery evidence. On restart Chatbook shows no
  guarded push candidate and makes no claim about the previous attempt; users
  inspect and push existing commits with external Git. No durable push journal,
  background reconciler, or retry queue is added.

### User interface, mutation, and privacy boundaries

- Keep push as a visually separate list-level action below the commit actions
  in the existing Prepare panel. The destination authorization, immutable
  review, progress, result, and recovery states use visible controls, safe
  initial focus, deterministic focus repair, a scrollable body, and a fixed
  phase-specific footer that remains keyboard-operable at `40x20`.
- The authorization dialog starts on `Cancel`; its affirmative action is
  `Authorize and check`. The final review places `Back` before the explicit
  `Push 1 commit` action. Buffered input from an earlier asynchronous phase
  cannot confirm a later phase.
- Review shows the exact commit subject/OID and parent-to-candidate transition,
  local branch, full destination ref, sanitized selectable endpoint details,
  included session-note provenance, exact lease, secure-transport/authentication
  policy, local pre-push-hook bypass, and the fact that remote hooks, CI, or
  mirroring may run. SSH copy discloses strict snapshotted host trust and
  existing-agent-only authentication with identity files disabled. It explains
  that later note edits remain local and that Git publishes a commit and its
  required objects rather than independently transmitting a UI list of notes.
- A persistent Session Git indicator distinguishes destination checking,
  pushing, and push-needs-attention while the user edits or leaves the Prepare
  panel. Checking the candidate, checking before push, and checking an uncertain
  result use distinct labels. Progress permits `Back to Files — push
  continues`; failure offers `Review again`, never `Retry`.
- Endpoint details show the sanitized scheme, punycode host, port, repository
  path, and full destination ref in a keyboard-focusable/selectable surface.
  Credential-bearing endpoints block rather than being partially displayed.
  Authentication failures remain non-secret and direct users to external Git
  configuration; Chatbook adds no credential editor.
- A quiescent operation requests mutation of only the approved remote
  destination ref. The remote necessarily receives Git objects required to
  make that commit reachable. Credential helpers may update their own secure
  state, and remote hooks, CI, mirrors, object stores, reflogs, and server-side
  policy may cause effects beyond Chatbook's one-ref request. Those effects are
  disclosed and are not described as Chatbook-owned or reversible.
- Concurrent note editing may intentionally change authoritative disk bytes
  and their SQLite recovery/search replica while the fixed commit is
  publishing; it cannot change the candidate or broaden the remote request.
  No generic note-content or secret scanner is added. The user remains
  responsible for whether the already-configured destination is appropriate
  for the reviewed commit.
- No database schema, persistent trust record, persistent candidate, remote or
  credential configuration, provider-specific hosting integration, general
  repository status/history browser, branch manager, fetch, pull, or
  repository repair workflow is added.
  The ephemeral owner-only SSH host-trust snapshot is process-context material,
  not an application-managed durable trust record.

This ADR supersedes ADR-038's exclusion of push only for the exact,
same-process guarded-commit candidate and operation defined here. ADR-038's
complete-index proof, commit, uncertainty, and no-repair boundaries remain in
force. ADR-035's session-only visible status/staging, exact index ownership,
repository trust, mutation-gate, and no-general-Git-client boundaries also
remain in force.

## Consequences

- Users can publish the exact guarded commit without leaving Chatbook, while
  ordinary files remain authoritative and SQLite remains an independent
  replica/recovery store.
- The workflow is intentionally less capable than command-line Git. A valid
  local branch that is two commits ahead, lacks a configured upstream, targets
  a missing branch, relies on LFS, uses unsupported push policy, or contains a
  commit not proved by the current Chatbook process must be pushed externally.
- SSH guarded push also requires a safe standard host-trust snapshot and an
  existing pinned SSH agent. Users who rely on default private-key files,
  custom SSH routing, or interactive authentication continue with external
  Git.
- Separate destination authorization and final confirmation add friction, but
  they prevent destination resolution, authentication contact, and external
  mutation from being collapsed into one ambiguous action.
- Exact lease comparison protects the reviewed parent-to-child update from the
  final validation race without granting permission to force an unrelated
  history update or recreate a deleted branch.
- Bypassing local pre-push hooks prevents repository code from running or
  broadening the local operation. It also deliberately excludes LFS-backed
  candidate paths. Remote hooks and services remain controlled by the
  destination.
- Network completion cannot always be known. Preserving `Uncertain`, forbidding
  automatic retries, and making recovery query-only may require the user to
  finish with external Git, but avoids duplicate or unintended updates.
- Direct frozen-endpoint invocation prevents post-confirmation remote-name
  redirection. The isolated immutable network-execution context additionally
  prevents a child from reapplying changed live Git configuration and avoids
  updating local remote-tracking refs. Local tracking state may therefore
  remain stale until an external Git operation refreshes it.
- A forced process exit loses all guarded-push attribution. This is a deliberate
  consequence of retaining no private operational journal or credentials.
- Persistent diagnostics stay payload-free. UI, logs, errors, and durable QA
  evidence use sanitized structured facts rather than raw remote/helper
  output, credentials, or note content.
- Windows users continue to use external Git for this workflow until a
  separately approved native ACL boundary exists. Chatbook makes no claim that
  POSIX mode checks or Windows Job Objects provide that missing privacy proof.
- SHA-256 repositories use a matching SHA-256 private bare context and
  64-hex OIDs; SHA-1 repositories use the ordinary SHA-1 context and 40-hex
  OIDs. Format drift or cross-format OIDs fail locally.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Keep push entirely external | Preserves ADR-038 unchanged but leaves the requested final publish step outside the otherwise complete File Notes session workflow. |
| Push every commit by which the branch is ahead | Ahead counts can include older or externally created commits that Chatbook did not prove and is not authorized to publish. |
| Wrap ordinary `git push` defaults | A remote name, implicit refspec, push policy, hooks, and missing destination can redirect or broaden the effect beyond the reviewed candidate. |
| Invoke the configured remote by name | Re-resolution after confirmation can follow changed configuration and may update a local remote-tracking ref; the approved operation freezes and invokes the effective endpoint directly. |
| Push without an exact lease | A branch can advance or be deleted between preflight and update, allowing overwrite, accidental recreation, or an ambiguous rejection policy. |
| Use `--force` or unspecified `--force-with-lease` | Neither expresses the one approved expected parent as an explicit server-side compare-and-swap. |
| Create or repair the upstream branch | Remote and branch administration is general Git-client behavior and could publish to an unintended namespace. |
| Run local pre-push hooks | Hooks can execute repository code, launch unowned work, broaden behavior, or make the reviewed outcome depend on arbitrary local policy. |
| Support LFS by running its hook separately | Reproducing the hook ordering, authentication, object-transfer, and uncertain-recovery contract is a separate feature. |
| Prompt for or store credentials in Chatbook | It creates a credential-management and secure-persistence boundary unrelated to guarded File Notes ownership. |
| Let OpenSSH reread live standard `known_hosts` files | A source can drift between Review and Confirm or resolve from the OS account home instead of Chatbook's isolated process home; an authorized private snapshot keeps trust exact and testable. |
| Let OpenSSH fall back to default identity files | It silently reads live private-key paths outside the frozen context. Agent-only authentication keeps private key material outside Chatbook while preserving existing noninteractive SSH. |
| Permit HTTP, local, file, `git://`, or custom helper transports | They either lack the approved confidentiality/integrity properties or execute a transport outside the bounded standard-client policy. |
| Combine commit and push into one confirmation | Users must be able to review the local commit and external destination as separate effects, and a successful commit must not imply consent to network publication. |
| Automatically retry an uncertain push | The first request may have succeeded despite a lost response; retry could repeat side effects or race a changed destination. |
| Fetch to reconcile uncertain state | Fetch mutates local refs and expands the network/repository boundary when an exact query of the retained destination is sufficient. |
| Persist candidate and recovery state | Durable recovery would add private operational storage, migration, stale-credential/trust semantics, and restart attribution beyond this slice. |
| Add provider-specific Git hosting APIs | It would add provider credentials and divergent service contracts without improving the exact standard-Git ref-update guarantee. |
| Treat guarded push as portable through `chmod` or Job Objects | POSIX mode bits and child containment do not prove an owner-only Windows ACL; Windows remains fail-closed pending separately approved native ACL work. |
| Assume every source repository uses SHA-1 | Git supports SHA-256 repositories; an unmatched private bare repository cannot safely resolve or publish those objects through an alternate. |

## Links

- [Design specification](../../Docs/superpowers/specs/2026-07-30-file-notes-guarded-session-push-design.md)
- [ADR-038](038-file-notes-guarded-session-commit.md)
- [ADR-035](035-file-notes-session-git-index-controls.md)
- [ADR-033](033-application-session-state-ownership.md)
- [ADR-029 File Notes](029-file-notes-disk-authority.md)
- [ADR-029 Local Private Data](029-local-private-data-boundary.md)
- [ADR-011](011-chatbook-workbench-ui-system.md)
- [TASK-1711](../tasks/task-1711%20-%20Add-guarded-exact-session-commit-push-to-File-Notes.md)
