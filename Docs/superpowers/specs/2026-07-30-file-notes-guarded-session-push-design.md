# File Notes Guarded Session Push Design

Date: 2026-07-30
Task: [TASK-1566](../../../backlog/tasks/task-1566%20-%20Add-guarded-exact-session-commit-push-to-File-Notes.md)
Decision: [ADR-039](../../../backlog/decisions/039-file-notes-guarded-session-push.md)
Amends: [ADR-038](../../../backlog/decisions/038-file-notes-guarded-session-commit.md)
Conforms to:
[ADR-035](../../../backlog/decisions/035-file-notes-session-git-index-controls.md),
[ADR-033](../../../backlog/decisions/033-application-session-state-ownership.md),
[ADR-029 File Notes](../../../backlog/decisions/029-file-notes-disk-authority.md),
[ADR-011](../../../backlog/decisions/011-chatbook-workbench-ui-system.md), and
[ADR-029 Local Private Data](../../../backlog/decisions/029-local-private-data-boundary.md)

## Summary

Extend File Notes with a separate reviewed push for exactly the one guarded
commit Chatbook just created and proved in the current application process.
The action targets only that commit's existing tracking upstream and only while
the remote branch is exactly at the commit's parent.

Chatbook resolves the effective destination locally, asks for a separate
process-only authorization before any network or credential-helper contact,
checks the exact remote ref, and shows an immutable review. Confirm revalidates
the same facts and requests one explicit parent-to-child ref update guarded by
an exact server-side lease.

This is not a branch-ahead push, remote manager, credential manager, retry
queue, or general Git client. Ordinary Markdown/text files remain authoritative
and SQLite remains an independent search/recovery replica.

## User outcome

After Chatbook proves a guarded local commit, the user may choose
`Review push (1 commit)…`, authorize the already configured destination, review
the exact commit and remote branch, and explicitly choose `Push 1 commit`.

The successful path publishes exactly that commit to the one existing upstream
branch. A remote deletion, divergence, configuration change, unknown
destination state, unsupported authentication policy, or loss of a conclusive
network result does not broaden the operation.

The result remains truthful:

- `Already published` means the destination already reported the candidate and
  Chatbook sent no push;
- `Succeeded` means Git reported the exact ref update accepted and a final
  check observed the candidate;
- `Failed with no update currently observed` means Git completed normally with
  failure and a final check still observed the parent; and
- `Uncertain` means Chatbook cannot prove either a completed desired state or a
  definite unsuccessful exchange.

Chatbook never automatically retries an uncertain push.

## Approved product choices

- Push only the exact guarded commit created by Chatbook in this application
  process.
- Never infer push authority from `HEAD`, history, reflog, ahead counts, commit
  messages, or a prior process.
- Use only the candidate branch's one existing tracking upstream.
- Never create, recreate, select, edit, or repair remotes, upstream tracking,
  or branches.
- Keep push separate from commit. A successful commit never opens or starts a
  push automatically.
- Require a separate process-only destination authorization before any network
  or authentication-helper contact.
- Use existing noninteractive Git authentication only. Chatbook never prompts
  for or stores credentials.
- Support only secure HTTPS with normal certificate verification and standard
  OpenSSH/scp-style SSH with existing host-key verification.
- Invoke the frozen effective endpoint, not the configured remote name.
- Require the remote destination to report the candidate's exact parent before
  review and immediately before push.
- Use an exact
  `--force-with-lease=<destination-ref>:<parent-oid>` compare-and-swap. The
  candidate is already proved to be the direct child, so the requested update
  is still fast-forward.
- Always bypass local pre-push hooks and disclose that policy.
- Block when any path included by the candidate is governed by Git LFS.
- Permit cancellation only until the actual network push child starts.
- Keep the candidate and recovery evidence in process memory only.
- Keep ordinary note editing, debounced autosave, and replica updates usable
  while a fixed commit is being checked or pushed.
- Keep the complete workflow keyboard-operable at `40x20`.
- Run focused risk-based verification and real full-application PTY UAT, not a
  repository-wide local CI or coverage run.

## Scope

### Included

- One exact same-process guarded-commit candidate
- Atomic candidate publication with guarded-commit success
- Immutable included-session-note provenance
- Existing-upstream and effective-push-endpoint resolution
- Separate process-only destination authorization
- HTTPS and OpenSSH/scp-style transport policy
- Existing noninteractive SSH-agent or credential-helper authentication
- Exact remote-ref preflight and Confirm revalidation
- Exact explicit refspec and exact parent lease
- Local pre-push-hook bypass and LFS blocking
- Retained network child and descendant lifecycle
- Typed outcomes and query-only uncertain recovery
- Prepare-panel availability, authorization, review, progress, result, and
  recovery
- Persistent Session Git operation indicator across panel removal/remount
- Focused pure, repository, transport, lifecycle, UI, and full-app acceptance
  verification

### Excluded

- Pushing a range, every commit ahead, an externally created commit, an older
  Chatbook commit, or any commit reconstructed after restart
- Automatic commit-to-push chaining or one combined confirmation
- Remote add/remove/rename, upstream setup, branch create/recreate, or remote
  repair
- Pull, fetch, merge, rebase, history browsing, remote status browsing, or
  branch management
- Prompting for, displaying, editing, storing, or configuring passwords,
  tokens, private keys, certificates, host keys, or credential helpers
- Plain HTTP, `git://`, local/file transports, custom remote helpers, or
  ambiguous transport forms
- Multiple push URLs, mirror mode, implicit/configured push refspecs, tags,
  push options, deletes, custom receive-pack, or submodule recursion
- Local pre-push hook execution or an LFS-upload replacement
- Automatic retry, background reconciliation, durable operation journal, or
  restart attribution
- Provider-specific Git hosting APIs
- Generic note-content secret scanning
- Claims that Chatbook owns or can cancel remote hooks, CI, mirrors, credential
  services, pre-existing SSH agents/control masters, or already-running server
  work
- Database schema changes or any change to disk/SQLite authority

## Chosen approach

Use a strict guarded ref update.

The candidate is one exact commit `C` with one exact parent `P`. Chatbook
authorizes and checks an existing destination ref `R`. Review is available
only when `R` currently reports `P`. Confirm repeats the check, then requests:

```text
source:     C
target:     R
lease:      R must still equal P
```

The effective endpoint URL and full target ref are explicit arguments. No
remote name or implicit refspec is passed to the mutating child.

This approach is intentionally narrower than these rejected alternatives:

- **Push the branch's ahead range.** This can include older or externally
  created commits that Chatbook never proved.
- **Wrap ordinary `git push`.** Git defaults, remote names, refspecs, hooks,
  follow-tags, and missing-destination behavior can broaden or redirect the
  request.
- **Use an unspecified lease or force.** Neither binds the update to the exact
  reviewed parent.

## Ownership and component boundaries

### `file_notes_git_push.py`

Add a pure sibling to `file_notes_git_commit.py` for push-only contracts:

- availability and destination projections;
- sanitized endpoint identity;
- included-note provenance;
- opaque destination-authorization and review handles;
- immutable review projection;
- typed outcome and recovery projection;
- URL/ref/transport policy results; and
- bounded, display-safe error categories.

It contains no owner state, subprocess execution, network I/O, Textual widget,
or SQLite behavior. Commit types are not reused for push outcomes merely
because both workflows have a review and result.

### `FileNotesSessionOwner`

The existing application-session owner remains the sole authority for:

- the current `SessionBinding` and `RepositoryIdentity`;
- the private push candidate;
- the independent monotonic candidate generation and opaque candidate token;
- process-only destination authorization epochs;
- the single-use push review capability;
- public push operation status;
- uncertain recovery authority;
- the existing Git-mutation gate; and
- compare-and-token publication that prevents stale completion from clearing a
  newer candidate.

Candidate publication occurs in the same owner-locked success transition that
publishes an immediate or recovered guarded-commit success. It is not a second
best-effort call after `publish_commit_outcome()`.

That atomic transition:

1. validates the exact guarded-commit capture;
2. retires or retains the appropriate session groups;
3. advances the local guarded-commit facts;
4. copies immutable push provenance before the commit review/session rows are
   discarded;
5. creates or replaces the exact push candidate; and
6. increments the candidate generation.

Ordinary `record_change()` continues to advance the existing Git-authority
generation for status/staging/commit, but it does not advance the push
candidate generation. Later note edits therefore stale Session Git rows while
leaving the immutable candidate available.

Push authorization binds only push-relevant authority:

- the selected-root `SessionBinding` generation;
- repository identity and repository-trust generation;
- candidate token and candidate generation;
- destination-policy/configuration fingerprint and its observed generation;
- destination-authorization epoch; and
- exact push operation/review IDs.

It deliberately does **not** bind the session-change sequence,
`git_authority_generation`, status generation, staging-ownership generation, or
current index/worktree signature after the guarded commit succeeds. Those
values may advance through ordinary editing, autosave, Stage, or status
refresh without changing the immutable commit. Root/repository/branch/candidate
drift still invalidates push authority.

Root/repository rebinding, repository-trust invalidation, local branch/HEAD
lineage drift, process exit, or a newer guarded commit can revoke or replace
the candidate. A newer guarded commit never accumulates its predecessor into a
range. If its parent is not the current upstream tip, guarded push is
unavailable.

### `FileNotesGitService`

The existing process-owned service remains the only layer that:

- resolves upstream and effective destination policy;
- performs local candidate/configuration proof;
- asks the owner to capture destination authorization;
- constructs and invokes network Git commands;
- owns remote checks, push, postflight, and recovery tasks;
- owns every launched child and owned descendant through settlement;
- classifies raw Git/SSH/helper results into typed sanitized outcomes; and
- publishes checked results through the owner.

The service adds push-specific lifecycle state, such as a retained push
operation and uncertain push evidence. It does not overload
`RetainedCommitOperation`, `CommitOutcome`, or commit recovery contracts.

The existing `GitProcessRunner`/`AsyncGitProcessRunner` boundary is extended so
network commands run in an isolated POSIX process group or Windows Job Object.
The service receives the actual direct-child-spawn signal from the runner.
Intent to spawn is not the cancellation boundary.

### Workspace

`LibraryFileNotesWorkspace` coordinates presentation only:

- push operation IDs and stale-callback guards;
- panel-to-service typed intents;
- Back-to-Files and remount reattachment;
- persistent Session Git checking/pushing/attention indicators; and
- focus repair keyed by exact phase and operation ID.

Push does not acquire the guarded-commit editor read-only lease. The candidate
is already immutable, and later editing must remain available.

Push rehydration depends only on the owner/service candidate and retained
operation. It does not require a workspace-local commit draft.

### Prepare panel

`LibraryFileNotesGitPanel` receives sanitized immutable projections and emits
typed intents:

- `Review push`;
- `Authorize and check`;
- `Cancel push check`;
- `Back from push review`;
- `Push 1 commit`;
- `Back to Files — push continues`;
- `Review again`; and
- `Check remote again — no push`.

The panel never parses a URL, chooses a ref, constructs a Git argument, stores
authority, interprets raw process output, or decides an outcome.

## Push candidate contract

The private candidate contains:

- exact `SessionBinding`;
- complete `RepositoryIdentity`;
- local attached `refs/heads/*` branch;
- exact parent OID and candidate OID;
- proof that the candidate has exactly one parent equal to the captured parent;
- guarded-commit proof/capture identity;
- opaque candidate token and monotonic candidate generation;
- exact selected-root binding generation, repository identity/trust
  generation, and candidate generation current at publication;
- commit subject and safe short-OID display facts;
- committed session-note count, change-type counts, and sanitized included-note
  labels copied from the commit review; and
- facts needed to prove that the candidate is still the local branch tip.

It contains no note bodies, blob bytes, credentials, remote identity, or
durable recovery data.

Candidate availability is independent of later note edits. It still requires
fresh proof that:

- the selected root and repository identity match;
- the same attached local branch is still at the candidate;
- the raw candidate object still has the expected sole parent;
- no newer guarded commit has replaced its token; and
- no active or uncertain conflicting Git operation owns the gate.

Success or `Already published` clears only the matching candidate token. A
stale completion cannot erase a newer candidate. Local `HEAD` drift observed
after a push starts may revoke future candidate availability, but it cannot
erase a remote result proved for the retained operation or overwrite newer
owner/status state.

After restart, the UI does not infer a candidate from the local branch. It
states:

```text
No Chatbook push candidate this session.
Inspect or push existing commits with external Git.
```

## Destination resolution without network access

Activating `Review push (1 commit)…` first performs local-only proof. No remote
query, credential helper, SSH process, proxy, or authentication mechanism may
run before destination authorization.

The service resolves:

- the candidate's exact local branch;
- exactly one `branch.<name>.remote`;
- exactly one `branch.<name>.merge` that is a full `refs/heads/*`;
- the configured remote's one effective push URL, falling back to its one URL
  only when no push URL exists;
- the exact full destination ref;
- effective URL rewriting into one unambiguous endpoint;
- relevant configuration origins and values;
- transport and authentication-helper policy; and
- a fingerprint covering every fact that can select or redirect the
  destination.

Resolution blocks when:

- the branch lacks tracking configuration;
- the tracking remote is `.` or otherwise local;
- merge configuration is missing, plural, ambiguous, or not a full
  `refs/heads/*`;
- a different `branch.<name>.pushRemote` or `remote.pushDefault` applies;
- the remote has multiple push URLs;
- remote mirror mode is active;
- a configured remote push refspec, push option, custom receive-pack, or other
  broadening policy is present;
- URL rewriting is ambiguous or yields more than one effective endpoint;
- repository/worktree configuration supplies an executable credential or SSH
  helper;
- effective transport policy is unsupported; or
- any required local fact cannot be read and revalidated without side effects.

Git configuration is read with redirecting environment variables removed and
without invoking a hook, filter, pager, editor, or credential helper. The
private effective endpoint is frozen after policy validation. The display
projection contains only its sanitized identity.

## Destination authorization

Destination authorization is separate from ADR-035 repository/filter trust.
It is process-only and exact.

Before the first network/helper contact, show
`Authorize configured destination` with:

- sanitized endpoint summary;
- local branch and full destination ref;
- transport;
- process-only scope;
- the statement that configured SSH or credential helpers may run after
  authorization;
- the statement that terminal prompts are disabled; and
- the statement that authorization checks the destination and does not push.

`Cancel` has initial focus. Escape and dialog close decline. The affirmative
action is `Authorize and check`, not `Trust remote`.

Authorization binds:

- candidate token and repository identity;
- exact private effective endpoint and sanitized identity;
- destination ref;
- transport policy;
- relevant configuration and origin fingerprint;
- authentication/SSH-helper policy fingerprint; and
- a monotonic authorization epoch.

An unchanged exact authorization may be reused within the same process.
Changing any bound fact revokes it. Changing a value away and back does not
revive a stale authorization because the epoch advances.

Authorization is permission to contact one configured destination. It is not a
claim that Chatbook endorses the remote's operator, content, branch policy, or
server-side behavior.

## Transport and authentication policy

### HTTPS

Allow only an unambiguous `https://` endpoint with:

- no embedded password, token, query, or fragment;
- normal certificate and hostname verification;
- the system trust policy;
- no configuration that disables verification or substitutes repository-owned
  certificate/key material; and
- an existing allowed noninteractive credential helper when credentials are
  required.

Plain HTTP and custom HTTP transport helpers block.

### OpenSSH and scp-style SSH

Allow only unambiguous `ssh://` or standard scp-style endpoints with:

- a valid host and repository path;
- the exact URL host, username, and port used as connection-routing values;
- batch/noninteractive operation;
- existing host-key verification;
- no automatic unknown-host acceptance;
- no password, askpass, or terminal prompt;
- no repository/worktree-controlled SSH command/helper; and
- no agent forwarding added by Chatbook.

The network execution context supplies a Chatbook-owned OpenSSH invocation that
does not reread live user SSH routing/command configuration after
authorization. It binds the literal authorized host/user/port, batch and
host-key policy, disables forwarding and interactive authentication, and may
use the user's existing SSH agent, standard identity-file locations, and
standard known-hosts files. Host aliases, ProxyCommand/ProxyJump, custom
IdentityFile selection, or other behavior that depends on live user SSH config
is unsupported in this slice and remains available through external Git.

Unknown or changed host keys fail without offering an override. Pre-existing
agents and credential services remain external authentication infrastructure;
Chatbook does not claim to own or terminate those pre-existing processes.

### Rejected transports

Block:

- `http://`;
- `git://`;
- `file://`;
- local or relative filesystem paths;
- drive-letter/UNC/local-path forms;
- `ext::` and custom `remote-<helper>` transports;
- ambiguous scp/URL forms; and
- credential-bearing endpoints.

Sanitized display uses the explicit scheme, IDNA/punycode host, port when
present, normalized SSH username when applicable, literal authorized host,
repository path, and full destination ref. Credential-bearing URLs block
instead of being partially masked and accepted.

### Noninteractive child environment

Network Git commands receive a new minimal OS-specific allowlist rather than a
copy of the ambient application environment.

It preserves only values required for direct Git execution and the approved
existing noninteractive authentication path. It excludes:

- provider/API tokens and unrelated secrets;
- Git repository, worktree, index, object, config, namespace, and replace-ref
  redirects;
- ambient author/committer identity/date overrides;
- ambient askpass/editor/pager/prompt overrides;
- proxy or transport-command overrides not admitted by policy; and
- unrelated application state.

Chatbook then installs its own no-prompt controls, closes stdin, disables
terminal prompting, and bounds stdout/stderr. Raw helper/Git output is used
only inside the classifier and is never shown or persisted.

## Frozen network execution context

Passing a frozen URL is not sufficient by itself because an ordinary Git child
can reread live system, global, repository, and worktree configuration and
reapply URL rewrites or transport/helper settings after Chatbook's final
validation.

After destination authorization and before the first network command, the
service creates one private immutable `NetworkGitExecutionContext`:

- an owner-only temporary bare Git directory outside the notes root and source
  repository;
- no worktree, refs, remotes, tracking configuration, hooks, or URL-rewrite
  rules;
- a minimal owner-only config containing only the exact approved
  noninteractive credential-helper and transport-neutral values copied from
  the authorized configuration snapshot;
- a Chatbook-owned immutable OpenSSH invocation specification for SSH
  endpoints, with the exact authorized host/user/port and no live user SSH
  routing/command config;
- system and global Git configuration redirected to owner-only empty files or
  otherwise disabled for every child;
- source-object access only through a Chatbook-supplied read-only alternate
  rooted at the already verified common object directory; and
- exact command-scoped narrowing overrides for prompt, hook, tags, submodules,
  maintenance, and filesystem-monitor behavior.

The allowlist never copies:

- `url.*.insteadOf` or `url.*.pushInsteadOf`;
- a remote name, URL, refspec, mirror, receive-pack, or push option;
- repository/worktree executable helpers or SSH commands;
- HTTP extra headers, embedded credentials, certificate exceptions, proxy
  commands, or transport-command overrides; or
- unrelated source-repository configuration.

Allowed credential-helper configuration is copied by exact key/value and
configuration origin after policy validation; credential values are not.
The endpoint is already fully resolved and is passed only as an argument.

Every remote preflight, Confirm revalidation, push, postflight, and recovery
query runs through this same context. Network children use the temporary bare
Git directory, not the source worktree or its live config. The push child sees
the candidate object through the controlled alternate but cannot update the
source repository's refs, index, or config.

The source configuration fingerprint includes relevant content plus source
identity/change metadata. Confirm rejects a changed fingerprint, but a change
after child spawn cannot redirect that child because the immutable execution
context is already detached from the source configuration.

The context is retained through review, active children, postflight, and
uncertain query-only recovery. Cleanup occurs only after all owned descendants
are terminal and no retained recovery needs it. A crash may leave an
owner-only temporary directory containing no credentials or user content; it
is never discovered or reused after restart and remains eligible only for
ordinary operating-system temporary cleanup.

## LFS policy

Because the mutating command always bypasses `pre-push`, it cannot safely rely
on Git LFS's upload hook.

Before authorization/review and again at Confirm, inspect the candidate tree's
attributes for every included path. If any included path resolves to the LFS
filter, or if exact candidate-tree attribute evaluation is unavailable or
ambiguous in an LFS-configured case, block:

```text
This commit includes Git LFS-managed content.
Push it with your existing external Git/LFS workflow.
```

Chatbook does not execute the LFS hook separately, upload LFS objects, or claim
semantic LFS support merely because Git LFS is installed.

## Remote preflight

After authorization, the service performs a retained read-only query against
the frozen endpoint and exact destination ref.

The query must return exactly one unambiguous full-ref result:

- destination equals parent OID: produce a review;
- destination equals candidate OID: publish `Already published`, send no push,
  and clear only the matching candidate token;
- destination missing/deleted: block and do not recreate it;
- destination is another OID: block as diverged/unsupported;
- response is plural, malformed, inaccessible, or unprovable: block.

The query uses the frozen URL directly. It does not fetch, update a local ref,
enumerate unrelated remote refs, or expose raw remote output.

Remote checks are retained external operations just like the push. Cancellation
settles their owned process group/job and output before releasing the mutation
gate. A removed panel cannot abandon or duplicate a check.

## Immutable push review

The private review snapshot binds:

- candidate token/generation, selected-root binding generation, repository
  identity/trust generation, destination-policy/configuration generation,
  destination-authorization epoch, and exact operation/review IDs;
- root, repository, local branch, parent OID, and candidate OID;
- sole-parent/direct-child proof;
- commit subject and included-session-note provenance;
- private effective endpoint and sanitized destination identity;
- full destination ref;
- transport and authentication-helper policy fingerprint;
- destination-authorization handle/epoch;
- exact remote-preflight observation of the parent; and
- immutable network-execution-context identity; and
- exact command policy, including lease, hook bypass, LFS result, and timeout.

The workspace stores only an opaque single-use handle. The panel receives a
separate immutable sanitized projection.

No lock is held while the user reads the review. Confirm consumes the handle,
reacquires the Git-mutation gate, and freshly repeats the exact
root/repository/branch/candidate, configuration, authorization, LFS, and
remote-ref proofs. It does not compare session-change, status, staging, index,
or worktree generations that ordinary later editing may legitimately advance.
Any push-relevant drift invalidates the review and returns to
availability/recovery; it never substitutes a new destination or candidate.

## Exact push execution

After Confirm revalidation, invoke one direct argument vector equivalent to:

```text
git \
  --git-dir=<private-network-execution-git-dir> \
  --no-replace-objects \
  -c core.fsmonitor=false \
  -c maintenance.auto=false \
  -c gc.auto=0 \
  push \
  --porcelain \
  --no-verify \
  --no-follow-tags \
  --recurse-submodules=no \
  --force-with-lease=<destination-ref>:<parent-oid> \
  -- \
  <frozen-effective-endpoint> \
  <candidate-oid>:<destination-ref>
```

The exact implementation may add command-scoped disabling overrides only when
they narrow behavior and are covered by argument-vector tests. It must not use
a shell or remote name, and it must not read live source-repository, worktree,
global, or system Git configuration.

The request contains:

- one exact source OID;
- one exact existing `refs/heads/*` destination;
- one exact expected old OID; and
- no other refspec or push option.

It does not set upstream, push tags, follow tags, recurse into submodules,
delete/recreate a ref, mirror, use an unspecified force/lease, or retry.

The child-start callback fires only after the OS process actually exists. A
pre-spawn validation or `create_subprocess_exec`/platform-launch failure remains
a non-mutating blocked/failure result and does not enter uncertain recovery.

## Lifecycle, cancellation, and containment

### Before the push child starts

Cancellation is available during:

- local candidate/configuration proof;
- destination authorization;
- first remote check;
- review;
- Confirm's local and remote revalidation; and
- any retained read-only child cleanup.

Cancellation waits for owned child/descendant termination and output drain.
It sends no push.

### After the push child starts

The actual runner spawn signal atomically changes the UI and service boundary:

- Cancel disappears;
- the service owns settlement independent of the panel/workspace;
- the mutation gate remains held;
- `Back to Files — push continues` may leave the panel without cancelling;
- root/source changes and owner-destroying transitions remain blocked;
- ordinary editing, debounced autosave, and replica synchronization continue;
  and
- reopening Prepare reattaches to the exact retained operation without
  launching another child.

Every network command runs in a new POSIX process group or Windows Job Object.
Bounded settlement performs graceful terminate, force-kill, and pipe drain.
Recovery remains disabled until every owned descendant is known terminal.

This ownership does not extend to pre-existing agents/control masters,
credential-service daemons, or work already running on the server. Killing the
local group/job cannot prove that a server stopped.

### Time bounds

- Read-only network checks: 30 seconds.
- Push: 60 seconds.
- Terminate/kill/drain stages: short bounded constants covered with injected
  clocks and deterministic process barriers.

Tests never wait the production timeout values.

### Uncertain operation and transitions

An uncertain push retains the mutation lease or an equivalent transition gate
with the recovery evidence. Conflicting Git actions and root/source rebinding
remain blocked so proof cannot be orphaned. The user may continue editing the
current root or exit Chatbook and use external Git; process exit deliberately
discards attribution.

Panel remount is not rebinding. It may observe or reattach to the retained
operation through the owner/service identity.

## Postflight and typed outcomes

Postflight queries only the frozen endpoint and destination ref. It never
follows current remote configuration.

### `Cancelled`

The user cancelled before an actual push child started. Every owned read-only
child/descendant is settled. The candidate remains available if its authority
still matches.

### `Blocked`

Local policy, authorization, LFS, or remote proof prevented review or push.
No push child ran. Preserve the matching candidate and give one safe next
action, normally external Git configuration/inspection followed by
`Review push` again.

### `Already published`

The pre-push query observed the candidate at the exact destination. Chatbook
started no push. Display:

```text
Destination currently reports <short-oid>.
Chatbook sent no push.
```

This is a point-in-time desired-state observation, not a claim that Chatbook
caused the remote update. Clear only the matching candidate token.

### `Succeeded`

Immediate success requires:

- the push child and every owned descendant terminated normally;
- Git's machine-readable result identifies one accepted update for the exact
  destination ref; and
- postflight observes the candidate at that exact destination.

Display:

```text
Git reported one accepted branch update.
Final check found <destination-ref> at <short-oid>.
```

Also show that the commit was created from `N` session notes and that Chatbook
requested no other ref. Do not claim that the remote has no server hooks,
mirrors, CI, or later external mutation.

Clear only the matching candidate token. Local `HEAD`, refs, index, and
tracking refs remain unchanged.

### `Failed with no update currently observed`

This outcome requires:

- a natural, normally settled unsuccessful Git result;
- every owned child and descendant known terminal;
- a protocol/result category that is not an ambiguous lost response; and
- postflight currently observing the parent at the exact destination.

Display:

```text
Git reported the push failed.
Final check currently finds the destination at the reviewed parent.
```

This does not claim the server never performed work or that later external ABA
cannot occur. Preserve the matching candidate. Offer `Review again`, which
performs a new authorization/preflight/review cycle; never label it `Retry`.

### `Uncertain`

Use uncertainty for:

- timeout after transmission could have begun;
- lost response or transport disconnect;
- unknown owned-child/descendant settlement;
- inability to query the destination;
- contradiction between Git result and remote state;
- postflight observing a missing or different OID; or
- any state that cannot satisfy the exact success, already, or definite-failure
  proof.

Display:

```text
The push may or may not have advanced the destination.
Chatbook will not retry it.
```

Destroy the consumed review and any reusable mutation authorization. Retain
only the original frozen endpoint/ref, candidate/parent facts, candidate token,
sanitized identity, process/trust epochs, owned-child settlement, and facts
needed for a query-only recovery.

## Query-only uncertain recovery

`Check remote again — no push` becomes available only after every owned local
descendant is terminal. While settlement remains open, keep it disabled and
show the reason.

The action:

- sends no refspec or push;
- queries only the retained original endpoint and exact ref;
- never follows current remote configuration;
- never restores or reuses the consumed Confirm capability;
- never starts automatically; and
- may require a new authorization for the same frozen identity if the original
  process-only destination trust changed.

Recovery classification:

- candidate observed: converge to desired-state success without claiming
  Chatbook caused it;
- parent observed: remain uncertain because server work may still complete
  later;
- another OID or missing ref: remain `needs attention`;
- inaccessible/unprovable: remain uncertain.

No number of parent observations converts an ambiguous prior transmission into
definite failure.

If Git configuration now points to endpoint B while the uncertain evidence
belongs to endpoint A, recovery either reauthorizes and queries retained A or
blocks. It never contacts B.

## External concurrency and ABA limits

The exact lease gives the remote ref update a compare-and-swap, but it does not
make the full client/server workflow transactional.

- A concurrent remote deletion cannot be silently recreated.
- A concurrent divergent advance cannot be overwritten.
- A delete-and-recreate ABA that ends at the same OID cannot be distinguished
  from continuous identity by ordinary Git ref protocol.
- A server may accept work before the client receives a result.
- Server hooks, CI, mirrors, and later actors can mutate external systems after
  Chatbook's point-in-time proof.
- A local external `HEAD`/config change after push spawn cannot redirect the
  frozen command. It may invalidate future availability but cannot erase a
  remote outcome proved for the retained operation.

UI and documentation describe point-in-time observations, not historical
causation or absence of external side effects.

## Prepare-panel interaction

### Availability

Keep push separate from the commit workflow. At the staged-note/list level,
render `Review push (1 commit)…` beneath commit actions only when the current
owner projection has an exact eligible candidate.

Availability explanations distinguish:

- no guarded commit candidate in this process;
- candidate local lineage changed;
- destination configuration unsupported;
- destination authorization required;
- another Git operation is active;
- push checking;
- pushing; and
- push needs attention.

Later note edits may stale the ordinary Session Git rows but do not hide an
eligible push candidate.

### Authorization

Use the separate `Authorize configured destination` dialog described above.
It must be keyboard-operable and show complete sanitized details through a
focusable/selectable `Endpoint Details` surface.

### Checking

Use distinct visible labels:

- `Checking push candidate…` for local proof;
- `Checking remote before push…` for authorized remote proof/revalidation; and
- `Checking uncertain outcome…` for query-only recovery.

Cancel remains available until the actual push child starts.

### Review

Lead with:

```text
Push 1 commit created from N session notes to <remote>/<branch>.
```

Show:

- exact commit subject and short/full OID access;
- parent-to-candidate transition;
- local branch;
- configured remote label for orientation;
- full destination `refs/heads/*`;
- sanitized effective endpoint and keyboard-accessible Details;
- included-session-note count/change types and optional disclosure list;
- exact expected-parent lease;
- secure transport and existing noninteractive authentication policy;
- `Local pre-push hooks will not run`;
- `Remote hooks, branch policy, CI, or mirrors may run`;
- `Later note edits remain local and are not added to this commit`; and
- `Git publishes the reviewed commit and required Git objects; this list is
  provenance, not a separate note-transfer selection`.

The included-note list uses the retained commit provenance. It never rebuilds
from current session rows or displays note bodies.

Footer order:

1. `Back`
2. `Push 1 commit`

Initial focus is `Back`, never Push. Push remains the final focusable action.
Escape returns without pushing. Buffered Enter from authorization/checking
cannot cross a phase/operation ID and activate Push.

### Confirm revalidation

After `Push 1 commit`, return to `Checking remote before push…`. Keep Cancel
available until the service reports the actual push-child-start boundary.

### Pushing

Show:

```text
Pushing 1 reviewed commit…
Cancellation is unavailable after the network push starts.
```

Offer `Back to Files — push continues` when safe. The Session Git entry/status
continues to show `Pushing` while the user edits.

### Result and recovery

- `Already published`: `Back to session`.
- `Succeeded`: `Back to session`.
- definite failure: `Review again`.
- uncertain: `Check remote again — no push`, disabled with a visible reason
  until owned descendants settle.

The result region is non-elided and scrollable. It must not rely on the editor
action-status area that is hidden in narrow Navigator/Prepare layouts.

## Responsive, keyboard, and accessibility contract

- Preserve the approved Files navigator/Editor switch at narrow widths.
- Use a scrollable body and fixed phase-specific footer.
- Keep every action label and current outcome reachable at `40x20`.
- Keep focused controls inside the viewport.
- Provide a keyboard `Details` path for complete endpoint/ref text; do not rely
  on tooltip-only or truncated identity.
- Use visible non-color state labels for checking, ready, pushing, success,
  failure, and uncertainty.
- Use real Textual focus movement and phase/operation IDs to reject stale focus
  repair.
- Do not transfer Enter across asynchronous phase changes.
- Announce checking, pushing, success, failure, and uncertainty through the
  incumbent accessible status/notification boundary.
- Leaving and reopening Prepare reattaches to the existing operation and never
  launches a second request.
- A result that settles while the panel is hidden appears when the user
  returns.

## Local mutation, disk, and SQLite boundary

The guarded push invokes the frozen URL rather than a remote name. Therefore
Chatbook does not ask Git to update local remote-tracking refs.

For a quiescent push, Chatbook selects no operation intended to modify:

- symbolic `HEAD` or any local ref;
- the index;
- repository or worktree configuration;
- note bytes or modes;
- File Notes replica rows, revisions, or tombstones; or
- File Notes session history.

Git may create ephemeral local process/cache/object-enumeration state that does
not alter those authorities. Authentication helpers may update their own
secure state.

The remote necessarily receives objects required to make the commit reachable
and may update its reflog/object store. Remote hooks, branch policy, CI, mirrors,
and hosting services may cause additional server-side effects outside
Chatbook's one-ref request.

In a concurrent-edit scenario, the user's intended autosave may change note
bytes and the corresponding SQLite replica/revision state while push is active.
Those changes remain local and do not alter the already-created candidate,
refspec, lease, or remote proof.

No generic content-secret scanner is added. The user remains responsible for
whether the already configured destination is appropriate for the reviewed
commit.

## Error and diagnostic handling

Convert raw process/protocol details into bounded typed categories:

- destination configuration;
- unsupported transport or authentication policy;
- authorization revoked;
- host-key verification;
- TLS verification;
- authentication unavailable;
- remote branch missing;
- remote branch diverged;
- LFS-managed content;
- candidate/local lineage drift;
- server rejection;
- timeout/termination;
- uncertain transmission; and
- internal proof failure.

UI copy contains one recovery action and no raw Git, SSH, server-hook, or
credential-helper output.

Persistent diagnostics may record only approved operational metadata such as
operation category, phase, duration, and typed status. They do not record:

- raw or sanitized endpoint paths unless separately approved metadata policy
  permits them;
- usernames, credentials, helper output, note paths, or note bodies;
- raw Git/SSH arguments containing private repository identity; or
- provider/environment secret values.

Display sanitization rejects or escapes terminal controls, bidi overrides,
invalid Unicode, malformed URL encoding, and hostile server text. Redaction
canaries must be absent from service objects, exceptions, Loguru capture, UI
text/accessibility announcements, PTY captures, and QA evidence.

## Performance

- Run no subprocess per note.
- Resolve candidate provenance from the retained snapshot, not fresh per-note
  Git calls.
- Query only one exact remote ref.
- Bound remote checks and push independently of repository note count.
- Keep network/process work off the Textual event loop.
- Retain object IDs, signatures, and sanitized provenance, not note bodies or
  Git pack data.
- Keep captured process output bounded.
- Use the existing included-note disclosure/list pattern; add incremental
  mounting only if focused measurement demonstrates a need.

## Focused verification and UAT

This feature does not add a repository-wide pytest, coverage, optional
dependency, broad performance, or full local CI run. Verification covers the
new boundary and the adjacent File Notes owner/service/staging/commit/UI
lifecycle that it changes.

Use compact parameterized matrices and shared fixtures. Timeouts use injected
clocks and deterministic barriers, never real 30/60-second waits.

### Pure contract tests

Cover:

- candidate token/generation, value-level ABA, replacement, and stale-result
  compare-and-clear;
- later note edits not invalidating the candidate;
- atomic immediate/recovered commit-success candidate publication;
- immutable included-note provenance after session-group retirement;
- second guarded commit replacing rather than accumulating a candidate;
- full ref and URL parsing, credential-bearing URL rejection, IDNA display, and
  ambiguous scp/local forms;
- destination/config/helper policy and authorization fingerprint/epoch;
- exact refspec and exact lease construction;
- no tags/options/delete/upstream/submodule/implicit refspec;
- LFS candidate-path detection and indeterminate blocking;
- allowlisted environment and prompt suppression;
- typed outcome and point-in-time copy;
- bounded redaction/control-character handling; and
- query-only recovery state transitions.

### Disposable real-Git CAS integration

Use a local bare repository only through a test-only injected transport
admission seam. Production policy continues to reject local/file transport.
This lane proves Git ref and lease semantics, not SSH/HTTPS security.

Cover:

- one candidate advancing one existing destination branch;
- source OID, destination ref, and exact expected-parent lease;
- all remote refs/tags before and after, with only the approved destination ref
  changing;
- all local refs before and after remaining unchanged because the frozen URL,
  not a remote name, is invoked from the isolated network execution context;
- local symbolic HEAD, `ls-files --stage -v -z`, config bytes, selected
  worktree bytes/modes, and logical File Notes replica state;
- destination deletion after final revalidation: no recreation;
- divergent advance after final revalidation: no overwrite;
- configuration/authorization away-and-back ABA: no stale review;
- source/local/global Git configuration mutation after final validation cannot
  redirect a retained child or change its helper/transport policy;
- fetch URL plus different push URL, multiple push URLs, mirror, refspec,
  push-option, receive-pack, pushRemote/default, `remote = .`, rewrite, and
  unsupported transport cases using real Git resolution where semantics
  matter;
- second guarded commit while remote remains at the older parent: block rather
  than range-push;
- later note editing while candidate remains fixed;
- a receive-side count proving exactly one update request and no automatic
  retry; and
- no assertion over the remote object-store files, because receiving objects is
  required.

Use barriers between final revalidation and push. Do not use timing sleeps to
manufacture races.

### Secure transport integration

#### OpenSSH

Use an ephemeral loopback OpenSSH fixture with:

- generated disposable client/server keys;
- temporary isolated `HOME`, Git global config, SSH directory, and
  `known_hosts`;
- strict host-key checking, batch mode, identities-only fixture policy, no
  agent forwarding, no askpass, and closed stdin;
- no access to the user's real SSH config, agent, keys, or credential state;
- bounded server connection and receive-side counters; and
- prompt-invocation sentinels.

Prove:

- zero network connections and zero helper invocations before destination
  authorization;
- one read-only connection after authorization;
- no mutating request before final Confirm;
- a live user SSH HostName/ProxyCommand/IdentityFile routing override is not
  consulted by the frozen child;
- unknown/wrong host key fails without prompting;
- missing/wrong authentication fails promptly without prompting; and
- no provider-secret canary reaches the SSH environment.

#### HTTPS

Use one hermetic smart-Git HTTPS fixture with an ephemeral CA installed only in
the fixture's isolated test trust store and a fake credential helper. The
test-only runner dependency may inject that trust store after production
transport admission; production continues to permit only ordinary verified
HTTPS policy.

Prove certificate/hostname verification, noninteractive helper behavior,
prompt suppression, environment isolation, and credential/output redaction.
This lane does not claim compatibility with every public hosting provider or
credential manager.

#### Ambiguous transport result

Use a deterministic loopback proxy/wrapper:

- accepted update followed by dropped client response: outcome is Uncertain,
  no second push occurs, and query-only recovery later observes the candidate;
- disconnect before server acceptance: outcome remains Uncertain while the
  destination reports the parent; repeated parent observations never become
  definite failure.

Server invocation counters prove no automatic retry.

### Native process-containment integration

Use a helper executable that launches a stubborn grandchild, reports owned
PIDs/heartbeats, and can ignore graceful termination.

Assert:

- POSIX process-group termination locally;
- Windows Job Object behavior only on a Windows test runner;
- bounded terminate/force-kill/drain;
- no surviving owned descendants before recovery enables;
- mutation/transition gate retention while settlement is unproved;
- quarantine when containment cannot be proved; and
- temporary-resource cleanup after certain settlement.

Pre-existing agents, control masters, credential daemons, and server processes
are explicitly excluded from this proof.

### Service and lifecycle tests

Cover:

- no network/helper contact before authorization;
- immutable network-execution-context construction, owner-only lifecycle,
  allowlisted copied configuration, and no live source-config reads;
- exact parent/candidate/missing/divergent preflight;
- Confirm revalidation of candidate, config, trust, LFS, and remote ref;
- actual child-spawn cancellation boundary and pre-spawn launch failure;
- one push invocation versus legitimate preflight/postflight queries;
- panel removal/remount without duplicate task or child;
- result settling while hidden;
- root/source/repository transition block during active/uncertain work;
- ordinary editing/autosave during push;
- uncertain recovery retaining endpoint A after config changes to endpoint B;
- candidate-token-scoped publication under newer candidate/local-state drift;
- shutdown ordering before replica teardown;
- restart constructing a new owner with no candidate or recovery attribution,
  while any orphaned temporary context is never discovered or reused; and
- 30/60-second policies through injected clocks.

### Mounted Textual tests

Avoid a phase-by-viewport Cartesian product:

- every phase, outcome, disabled reason, recovery action, focus order, scroll,
  and Details path at `40x20`;
- one representative happy path at `120x40`;
- one retained-operation leave/reopen flow at both sizes; and
- retain the existing `160x45` Files-source-entry regression without repeating
  every push state there.

Use real Pilot keyboard events for focus/navigation assertions rather than
calling private handlers or button methods as acceptance evidence. Prove:

- Confirm is last and never initially focused;
- the focused action remains in the viewport;
- body scroll and fixed footer coexist;
- buffered Enter cannot cross phases;
- full endpoint details open and close by keyboard;
- Back to Files exposes checking/pushing/attention state;
- reopening reattaches without a second service request; and
- non-elided result/recovery copy remains visible in compact Navigator mode.

Mounted tests are automated UI evidence, not full-app PTY acceptance.

### Production full-app PTY UAT

Launch the unmodified production application with:

```text
python -m tldw_chatbook.app
```

Use an isolated synthetic config, notes root, SQLite data, Git repository, and
ephemeral production-compatible OpenSSH destination. Probe fixture capability
before the run. If an isolated standard-transport fixture is unavailable, do
not substitute an injected transport and call it network end-to-end UAT; record
the missing lane and run it in a suitable environment before acceptance.

Within one application process and using real terminal keyboard input:

1. Navigate through Library -> Notes -> Files.
2. Select the synthetic notes root.
3. Edit and autosave notes with exact frontmatter preservation.
4. Stage the intended session notes.
5. create and prove the guarded local commit;
6. open `Review push`;
7. cancel destination authorization and externally prove zero connections;
8. reopen, authorize, and observe the first read-only connection;
9. review the exact commit/ref/endpoint/lease/policies;
10. confirm the push and externally prove one exact ref transition;
11. leave/reopen during a retained delayed operation and prove no duplicate
    request;
12. continue editing while the candidate/push remains fixed; and
13. verify the critical keyboard/scroll/Details/result flow at `40x20`.

A second focused scenario exercises a divergent or missing destination block.
Uncontrolled process killing and every uncertain edge do not belong in
hands-on UAT; deterministic automated lanes cover them.

The PTY run must use Tab/Shift+Tab/Enter/Escape/scroll keys and visible UI. It
must not seed a candidate, call widget `.press()`, invoke private handlers, or
substitute `App.run_test()`/Pilot for the full application.

### Phase-to-evidence matrix

| Claimed phase/behavior | Full-app PTY | Mounted UI | Service/integration |
| --- | --- | --- | --- |
| Candidate availability after same-process commit | Required | Required | Required |
| Authorization Cancel and zero network | Required | Required | Required with external counter |
| Authorized remote checking | Required | Required | Required |
| Review readability/focus/Details | Required | Required | Projection facts |
| Confirm revalidation and child-start boundary | Visible transition | Required | Authoritative proof |
| Pushing and leave/reattach | Required | Required | Request/task counts |
| Exact successful remote ref update | Required | Result rendering | Authoritative ref proof |
| Already published | Optional representative | Required | Authoritative no-push proof |
| Definite failure/no update currently observed | Optional representative | Required | Authoritative proof |
| Uncertain and query-only Check again | Automated-only unless safely reproducible | Required | Authoritative transport proof |
| Process tree termination | Not hands-on | Projection only | Native OS integration |
| HTTPS policy/authentication | Not required in SSH UAT | Error rendering | Hermetic HTTPS integration |

The final QA record must not claim that a phase received full-app acceptance
when it was exercised only through mounted or service tests.

### Durable QA evidence

Store a small sanitized bundle under:

`Docs/superpowers/qa/file-notes-guarded-push-uat-2026-07-30/`

Include:

- `README.md` with exact source commit, production launch command, platform,
  Git/OpenSSH versions, terminal dimensions, operator (`agent-operated`),
  steps, verdict, and automated-only gaps;
- sanitized raw keystroke/action and terminal transcripts;
- key phase/viewport captures;
- `evidence.json` with parent/candidate OIDs, sanitized endpoint/ref identity,
  pre/post local and remote ref maps, connection/push counts, prompt/helper
  counts, logical note/replica assertions, and test-lane labels;
- process-tree settlement evidence from the native integration lane;
- canary-scan result; and
- SHA-256 manifest for every retained artifact.

Never retain private keys, credentials, raw helper/server diagnostics, private
note content, real user paths, or unsanitized absolute fixture paths.

### Focused verification boundary

Before completion run:

- new push contract/service/integration/UI tests;
- existing File Notes session-owner, Git-service, staging, guarded-commit,
  workspace/panel, and production-owner-shutdown regressions affected by the
  shared lifecycle;
- targeted Ruff and compile checks for changed production/test files;
- documentation/JSON checks;
- `git diff --check`; and
- the production-app PTY UAT above.

Do not run the repository-wide suite, repository-wide coverage, or a broad
local CI reproduction as a manual completion gate.

## ADR check

ADR required: yes

ADR path:
`backlog/decisions/039-file-notes-guarded-session-push.md`

Reason: guarded push changes the remote/network/authentication security
boundary, adds an exact external ref-update service contract, introduces
destination authorization and uncertain network recovery, extends
application-session ownership and process containment, and adds a long-lived
Prepare-panel workflow. ADR-039 amends only ADR-038's no-push exclusion for the
exact same-process guarded-commit candidate.
