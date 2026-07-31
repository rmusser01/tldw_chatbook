# File Notes Guarded Session Push Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Use superpowers:test-driven-development for each behavior change and superpowers:verification-before-completion before every completion claim. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user separately review and publish exactly the one guarded File Notes commit Chatbook just proved in the current application process, using its existing upstream branch without turning Chatbook into a general Git or credential client.

**Architecture:** Extend the existing process-owned `FileNotesSessionOwner`, `FileNotesGitService`, Prepare panel, and File Notes workspace. The owner atomically creates and retains one exact push candidate; the service alone resolves and contacts the destination through an immutable isolated Git context; the process runner owns network process trees through settlement; the workspace coordinates retained presentation; and the panel remains projection-only. Markdown/text files remain authoritative and SQLite remains an independent replica.

**Tech Stack:** Python 3.11+, asyncio subprocesses, Git 2.x plumbing/porcelain and smart transports, stdlib `ctypes` for Windows Job Objects, OpenSSH, immutable dataclasses, Textual 3.3+, pytest/pytest-asyncio, disposable Git repositories, and hermetic loopback SSH/HTTPS fixtures.

**Backlog:** [TASK-1566](../../../backlog/tasks/task-1566%20-%20Add-guarded-exact-session-commit-push-to-File-Notes.md)

**Specification:** [File Notes Guarded Session Push Design](../specs/2026-07-30-file-notes-guarded-session-push-design.md)

**Decision:** [ADR-039](../../../backlog/decisions/039-file-notes-guarded-session-push.md)

**Depends on:** TASK-1350, TASK-1411

**ADR required:** yes

**ADR path:** `backlog/decisions/039-file-notes-guarded-session-push.md`

**Reason:** Guarded push changes the remote/network/authentication security boundary, adds an exact external compare-and-swap contract and uncertain network recovery, extends process/session ownership, and adds a long-lived Prepare-panel workflow.

---

## Non-Negotiable Boundary

- Publish only the exact same-process guarded-commit candidate. Do not infer
  authority from `HEAD`, history, reflog, ahead counts, commit messages, or a
  prior process, and do not push a range.
- Keep commit and push separate. Commit success may reveal
  `Review push (1 commit)…`; it must never open, authorize, confirm, or start a
  push automatically.
- Use only one existing attached-branch tracking upstream and one existing full
  `refs/heads/*` destination. Do not add remote, branch, history, fetch, pull,
  retry, credential, or repair features.
- Require separate process-only destination authorization before the first
  network connection or credential-helper invocation.
- Production supports only verified HTTPS and literal-host OpenSSH/scp
  destinations with existing noninteractive authentication. Production
  continues to reject local/file/plaintext/custom-helper transports.
- Every network Git child runs from one owner-only immutable temporary bare
  context, with live source/local/global/system configuration disabled and
  source objects exposed only through a controlled read-only alternate.
- Request exactly
  `<candidate-oid>:<destination-ref>` with
  `--force-with-lease=<destination-ref>:<parent-oid>`. Send no implicit
  refspec, tags, options, deletes, mirror, upstream edit, submodule recursion,
  hook, or retry.
- Block candidate paths governed by Git LFS because local pre-push hooks are
  deliberately bypassed.
- Use typed, bounded diagnostics. Raw Git, SSH, server, and credential-helper
  output must not reach UI, logs, exceptions, or durable QA evidence.
- Keep ordinary note editing, debounced autosave, and replica synchronization
  available while the fixed candidate is checked or pushed. Do not acquire the
  guarded-commit editor read-only lease for push.
- An uncertain push never retries. `Check remote again — no push` may query
  only the retained original endpoint after every owned descendant settles.
- Add no database schema, durable candidate/trust/operation journal,
  dependency, provider-specific API, or app-wide state owner.
- Do not run the repository-wide suite, coverage, a six-hour soak, or broad
  local CI. Run only the risk-focused commands named in this plan.

## File Responsibilities

### Create

- `tldw_chatbook/Notes/file_notes_git_push.py`
  - Frozen push candidate, destination, authorization, review, outcome, and
    recovery contracts.
  - Pure endpoint/ref/config-policy validation, exact command construction,
    remote-result parsing, safe copy, and redaction/classification.
  - No owner state, subprocesses, network I/O, Textual, or SQLite.
- `tldw_chatbook/Notes/file_notes_git_network.py`
  - Private `NetworkGitExecutionContext`, minimal allowlisted network
    environment, immutable OpenSSH invocation, owner-only temporary bare Git
    context, controlled object alternate, and exact cleanup.
  - No authority, retained-task ownership, or outcome classification.
- `tldw_chatbook/Notes/git_process_containment.py`
  - One small platform adapter for POSIX process groups and Windows Job
    Objects, used by `AsyncGitProcessRunner`. Windows admission must be
    race-free: the child is created suspended, assigned to the Job Object, and
    only then resumed.
  - No Git policy or push state.
- `Tests/Notes/test_file_notes_git_push.py`
  - Pure contracts, policy, parsers, argv, safe-copy, and redaction tests.
- `Tests/Notes/test_file_notes_git_push_service.py`
  - Controlled-runner/context/clock service state-machine tests.
- `Tests/Notes/test_file_notes_git_push_integration.py`
  - Disposable real-Git compare-and-swap and local-invariance tests through a
    test-only local transport admission seam.
- `Tests/Notes/test_file_notes_git_push_transport.py`
  - Capability-gated hermetic OpenSSH and HTTPS integration.
- `Tests/Notes/test_git_process_containment.py`
  - Real owned-child/grandchild lifecycle tests per supported platform.
- `Tests/UI/test_library_file_notes_git_push.py`
  - Mounted push-only panel/workspace/focus/responsive tests, avoiding further
    growth of the existing large commit/staging UI test file.
- `Docs/superpowers/qa/file-notes-guarded-push-uat-2026-07-30/`
  - Sanitized production-app PTY evidence only.

### Modify

- `tldw_chatbook/Notes/file_notes_session_owner.py`
  - Atomic commit-success candidate publication, push-specific generations and
    opaque tokens, process-only authorization, single-use review authority,
    public retained-operation status, uncertain recovery, and token-scoped
    compare-and-clear.
- `tldw_chatbook/Notes/file_notes_git_service.py`
  - Candidate seed construction, local destination/LFS proof, retained remote
    checks, review/confirm, exact push, postflight, recovery, child-start
    boundary, and shutdown coordination.
- `tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py`
  - Separate push phases, authorization/details modal, immutable review,
    progress/result/recovery surfaces, typed intents, and phase-plus-operation
    focus guards.
- `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`
  - Push service protocol, operation IDs, observers, independent candidate
    availability, persistent Session Git indicator, hide/reattach, and focus
    repair. It does not acquire `_EditorReadOnlyLease`.
- `Tests/Notes/test_file_notes_session_owner.py`
  - Candidate publication/generation/ABA/restart/transition tests.
- `Tests/Notes/test_file_notes_git_commit_integration.py`
  - Immediate and recovered guarded-commit candidate publication with retained
    provenance.
- `Tests/ProductionApp/test_file_notes_session_owner_lifecycle.py`
  - Owner-first push/check/recovery settlement and no restart attribution.
- `backlog/tasks/task-1566 - Add-guarded-exact-session-commit-push-to-File-Notes.md`
  - Plan link now; checked acceptance criteria and implementation notes only
    after implementation and acceptance evidence.
- `backlog/decisions/039-file-notes-guarded-session-push.md`
  - Change `Proposed` to `Accepted` only at final closeout when code and UAT
    demonstrate the decision.

Do not modify `tldw_chatbook/app.py`,
`tldw_chatbook/UI/Screens/library_screen.py`, the File Notes SQLite schema, or
global configuration unless a focused test proves an existing injection or
owner-first-shutdown seam is insufficient. If such a seam fails, document the
specific deviation in this plan before editing it.

## Public and Private Contract

Exact private names may be adjusted once to match repository conventions, but
the boundaries must not move:

- `file_notes_git_push.py` exposes frozen sanitized values such as
  `PushCandidateProjection`, `PushDestinationIdentity`,
  `DestinationAuthorizationHandle`, `PushReviewHandle`,
  `PushReviewProjection`, `PushOutcome`, and `PushRecoveryProjection`.
  Widgets never receive the private effective endpoint, raw output, a
  candidate token, or a capability they can synthesize.
- `FileNotesSessionOwner.publish_commit_outcome(...)` creates/replaces the
  private push candidate inside the same owner-lock transition that publishes
  immediate or recovered guarded-commit success. There is no second
  best-effort publication call.
- Candidate generation is independent from
  `git_authority_generation`. `record_change()` may stale ordinary Session Git
  status without invalidating the immutable candidate.
- Repository trust, candidate, destination policy, destination authorization,
  push review, operation, and uncertain recovery each use exact monotonic
  epochs/tokens so away-and-back values cannot revive stale authority.
- `FileNotesGitService.start_push_review(...)`,
  `authorize_and_check_push(...)`, `start_push(...)`, `cancel_push(...)`, and
  `check_push_again(...)` return or retain typed push-specific values. They do
  not reuse `CommitOutcome`, `RetainedCommitOperation`, or commit recovery
  authority.
- `RetainedPushOperation` remains service-owned after panel/workspace removal.
  The actual push-child-spawn callback, not intent to spawn, removes Cancel.
- `NetworkGitExecutionContext` is created only after authorization and is
  reused unchanged for preflight, Confirm revalidation, push, postflight, and
  query-only recovery. Cleanup waits for terminal descendants and the end of
  recovery authority.
- Test-only `TransportAdmission`, `NetworkContextFactory`,
  `before_push_spawn`, clock/deadline, process-controller, and connection
  counter seams may be injected. Production construction remains strict and
  exposes no user-facing bypass.

## Exact Git and Process Contract

Local destination resolution runs without network/helper contact and with Git
repository/index/config redirect variables removed. It freezes one candidate
branch, one existing tracking upstream, one effective push endpoint, one full
destination ref, relevant configuration origins/values, and a policy
fingerprint.

Every remote query is equivalent to:

```text
git
  --git-dir=<private-network-git-dir>
  --no-replace-objects
  -c core.fsmonitor=false
  -c maintenance.auto=false
  -c gc.auto=0
  ls-remote --refs -- <frozen-endpoint> <full-destination-ref>
```

The mutating child is equivalent to:

```text
git
  --git-dir=<private-network-git-dir>
  --no-replace-objects
  -c core.fsmonitor=false
  -c maintenance.auto=false
  -c gc.auto=0
  push
  --porcelain
  --no-verify
  --no-follow-tags
  --recurse-submodules=no
  --force-with-lease=<destination-ref>:<parent-oid>
  --
  <frozen-endpoint>
  <candidate-oid>:<destination-ref>
```

The exact implementation may add only command-scoped narrowing overrides
covered by argument-vector tests. It must not use a shell, remote name, live
source configuration, stdin, interactive prompt, implicit refspec, or raw
diagnostic display.

---

## Task 1: Add Pure Push Contracts, Parsers, and Exact Command Builders

**Files:**

- Create: `tldw_chatbook/Notes/file_notes_git_push.py`
- Create: `Tests/Notes/test_file_notes_git_push.py`

- [ ] Write failing frozen-contract tests for sanitized candidate,
  destination, authorization, review, outcome, and recovery projections.
  Assert private endpoint/capability/token fields cannot appear in public
  projections.
- [ ] Write failing parameterized policy tests for:
  - exact `refs/heads/*` acceptance and malformed/relative/refname rejection;
  - verified `https://`, literal-host `ssh://`, and standard scp-style forms;
  - credential-bearing, query/fragment, plaintext, `git://`, file/local,
    drive/UNC, `ext::`, custom helper, ambiguous scp, and hostile Unicode
    rejection;
  - IDNA/punycode display, normalized SSH user/port/path, and safe selectable
    endpoint details; and
  - C0/C1, bidi-control, invalid encoding, markup, URL-secret, and path canary
    removal.
- [ ] Write failing parser/classifier tests for exactly one matching
  `ls-remote --refs` record, parent/candidate/missing/divergent/malformed
  states, one exact push-porcelain destination result, and bounded closed
  diagnostic categories that discard raw bytes.
- [ ] Write failing argv tests for the exact query and push vectors, full
  destination ref, candidate-to-ref refspec, exact parent lease, direct frozen
  endpoint, hook bypass, and explicit exclusion of remote names, tags, push
  options, delete/mirror/upstream/submodule/retry/implicit refspec behavior.
- [ ] Write failing pure outcome-copy tests for `Already published`,
  `Succeeded`, `Failed with no update currently observed`, `Uncertain`, and
  query-only recovery. Assert the copy is point-in-time and never claims no
  server work occurred.
- [ ] Run:

```bash
python3 -m pytest Tests/Notes/test_file_notes_git_push.py -q
```

Expected: FAIL because the push module and contracts do not exist.

- [ ] Implement only immutable models and pure validation/parser/builder
  functions. Do not import Textual, call Git, read configuration, or perform
  network I/O.
- [ ] Re-run the same command. Expected: PASS.
- [ ] Run:

```bash
python3 -m ruff check tldw_chatbook/Notes/file_notes_git_push.py Tests/Notes/test_file_notes_git_push.py
python3 -m ruff format --check tldw_chatbook/Notes/file_notes_git_push.py Tests/Notes/test_file_notes_git_push.py
python3 -m compileall -q tldw_chatbook/Notes/file_notes_git_push.py
git diff --check
```

Expected: all commands exit 0.

- [ ] Commit:

```bash
git add tldw_chatbook/Notes/file_notes_git_push.py Tests/Notes/test_file_notes_git_push.py
git commit -m "feat(notes): add guarded push contracts [TASK-1566]"
```

## Task 2: Atomically Publish One Exact Push Candidate

**Files:**

- Modify: `tldw_chatbook/Notes/file_notes_session_owner.py`
- Modify: `tldw_chatbook/Notes/file_notes_git_service.py`
- Modify: `Tests/Notes/test_file_notes_session_owner.py`
- Modify: `Tests/Notes/test_file_notes_git_commit_integration.py`

- [ ] Write failing owner tests for:
  - one private candidate token and independent monotonic candidate generation;
  - atomic creation inside successful `publish_commit_outcome(...)`;
  - no candidate for failed or uncertain local commit;
  - exact parent/candidate OIDs, attached branch, repository/root/trust
    generations, subject, count/change types, and immutable included-note
    labels copied before session groups retire;
  - immediate and recovered commit success producing the same candidate shape;
  - later `record_change()`, autosave, status, Stage, Unstage, and index/worktree
    churn advancing ordinary Git authority without advancing candidate
    generation;
  - root/repository/trust/branch/lineage drift, shutdown, and restart revoking
    availability;
  - a newer guarded commit replacing rather than accumulating the older
    candidate; and
  - success/Already/stale completion clearing only the exact candidate token.
- [ ] Extend guarded-commit integration tests so `_CommitReviewSnapshot` and
  `_CommitRecoveryProof` retain one immutable candidate seed, including
  provenance, and both success paths enter the same owner-locked publication
  seam.
- [ ] Run:

```bash
python3 -m pytest Tests/Notes/test_file_notes_session_owner.py Tests/Notes/test_file_notes_git_commit_integration.py -q -k "push_candidate or candidate_publication or guarded_commit_retains_newer or check_again_converges"
```

Expected: FAIL on missing candidate authority/publication.

- [ ] Extend `CommitPublication` with the immutable candidate seed. Construct
  it when the commit review's included-note projection is created, retain it
  through uncertain local-commit recovery, and create/replace the owner
  candidate inside the already-locked successful publication transition.
- [ ] Add a sanitized candidate availability/status projection to
  `FileNotesSessionSnapshot`. Keep the effective endpoint, private capability,
  token, and note bodies out of the snapshot.
- [ ] Re-run the focused command. Expected: PASS.
- [ ] Run:

```bash
python3 -m pytest Tests/Notes/test_file_notes_session_owner.py -q -k "commit_publication or record_change or active_mutation or shutdown"
python3 -m ruff check tldw_chatbook/Notes/file_notes_session_owner.py tldw_chatbook/Notes/file_notes_git_service.py Tests/Notes/test_file_notes_session_owner.py Tests/Notes/test_file_notes_git_commit_integration.py
python3 -m compileall -q tldw_chatbook/Notes/file_notes_session_owner.py tldw_chatbook/Notes/file_notes_git_service.py
git diff --check
```

Expected: all commands exit 0.

- [ ] Commit:

```bash
git add tldw_chatbook/Notes/file_notes_session_owner.py tldw_chatbook/Notes/file_notes_git_service.py Tests/Notes/test_file_notes_session_owner.py Tests/Notes/test_file_notes_git_commit_integration.py
git commit -m "feat(notes): publish exact guarded push candidates [TASK-1566]"
```

## Task 3: Prove Local Destination, Transport, Configuration, and LFS Policy

**Files:**

- Modify: `tldw_chatbook/Notes/file_notes_git_push.py`
- Modify: `tldw_chatbook/Notes/file_notes_git_service.py`
- Modify: `tldw_chatbook/Notes/file_notes_session_owner.py`
- Modify: `Tests/Notes/test_file_notes_git_push.py`
- Create: `Tests/Notes/test_file_notes_git_push_service.py`
- Modify: `Tests/Notes/test_file_notes_session_owner.py`

- [ ] Write failing controlled-runner tests proving `Review push` first performs
  local-only resolution and launches no remote query, SSH process, proxy, or
  credential helper.
- [ ] Write failing compact policy matrices for:
  - exactly one `branch.<name>.remote`, one full
    `branch.<name>.merge`, and one effective push URL;
  - fallback from absent push URL to one fetch URL;
  - missing/plural tracking, `remote = .`, multiple push URLs, mirror,
    refspec, push option, receive-pack, a `pushRemote` or
    `remote.pushDefault` that selects a different remote, ambiguous rewrite,
    and unsupported helper/transport blocking;
  - `pushRemote` or `remote.pushDefault` resolving to the same tracking remote
    remaining admissible without changing the frozen destination;
  - relevant configuration origin/value plus source-identity/change-metadata
    fingerprint and away-and-back ABA;
  - repository/worktree executable credential or SSH helper rejection;
  - secure HTTPS and literal-host SSH policy; and
  - LFS-filtered included paths or indeterminate exact candidate-tree LFS
    evaluation blocking before review and again at Confirm; and
  - one-path versus 1,000-path candidate-tree LFS/config proof using a bounded
    command count, never one subprocess per note.
- [ ] Write failing owner tests for an independent repository-trust generation,
  destination-policy generation, authorization epoch, exact candidate/config
  binding, revocation, and value-level ABA.
- [ ] Run:

```bash
python3 -m pytest Tests/Notes/test_file_notes_git_push.py Tests/Notes/test_file_notes_git_push_service.py Tests/Notes/test_file_notes_session_owner.py -q -k "destination or transport or configuration or authorization_epoch or lfs or no_network"
```

Expected: FAIL on missing local resolver, policy fingerprint, epochs, and LFS
proof.

- [ ] Add bounded local Git configuration/object proof to
  `FileNotesGitService`. Strip redirecting environment variables, disable
  hooks/filters/pagers/editors/prompts, and never call the normal broad
  `build_git_environment()` for network policy.
- [ ] Add strict production `TransportAdmission` and a local-only injected
  test implementation. Do not add a production config switch.
- [ ] Add owner capture/revoke methods for exact destination authorization.
  Authorization grants permission to contact one frozen destination; it does
  not claim the remote operator/content is trusted.
- [ ] Re-run the focused command. Expected: PASS.
- [ ] Run:

```bash
python3 -m ruff check tldw_chatbook/Notes/file_notes_git_push.py tldw_chatbook/Notes/file_notes_git_service.py tldw_chatbook/Notes/file_notes_session_owner.py Tests/Notes/test_file_notes_git_push.py Tests/Notes/test_file_notes_git_push_service.py Tests/Notes/test_file_notes_session_owner.py
python3 -m compileall -q tldw_chatbook/Notes/file_notes_git_push.py tldw_chatbook/Notes/file_notes_git_service.py tldw_chatbook/Notes/file_notes_session_owner.py
git diff --check
```

Expected: all commands exit 0.

- [ ] Commit:

```bash
git add tldw_chatbook/Notes/file_notes_git_push.py tldw_chatbook/Notes/file_notes_git_service.py tldw_chatbook/Notes/file_notes_session_owner.py Tests/Notes/test_file_notes_git_push.py Tests/Notes/test_file_notes_git_push_service.py Tests/Notes/test_file_notes_session_owner.py
git commit -m "feat(notes): prove guarded push destination policy [TASK-1566]"
```

## Task 4: Own Network Process Trees Through Settlement

**Files:**

- Create: `tldw_chatbook/Notes/git_process_containment.py`
- Modify: `tldw_chatbook/Notes/file_notes_git_service.py`
- Create: `Tests/Notes/test_git_process_containment.py`
- Modify: `Tests/Notes/test_file_notes_git_push_service.py`

- [ ] Write failing fake-controller unit tests for actual-spawn notification,
  graceful terminate, forced kill, bounded wait/drain, settlement proof, and
  an unproved-descendant result that keeps recovery disabled.
- [ ] Add a test helper executable that launches a stubborn grandchild,
  reports owned PIDs/heartbeats, and may ignore graceful termination.
- [ ] Write the real POSIX test for a new session/process group and group-wide
  terminate/kill/drain. Add a Windows-only Job Object test guarded by
  `skipif(os.name != "nt")`; its helper must spawn a grandchild immediately
  when resumed and prove that no descendant can escape between child creation
  and Job Object assignment. Do not claim Windows proof on a non-Windows run.
- [ ] Run:

```bash
python3 -m pytest Tests/Notes/test_git_process_containment.py Tests/Notes/test_file_notes_git_push_service.py -q -k "process_tree or child_spawn or containment or descendant"
```

Expected: FAIL because the runner only owns its direct child.

- [ ] Implement a small `ProcessTreeController` platform adapter:
  `start_new_session=True` plus retained PGID on POSIX; on Windows, use a
  stdlib `ctypes` launcher that creates the child suspended, assigns it to a
  kill-on-close Job Object, and resumes it only after assignment succeeds.
  Launch/assignment/resume failure must fail closed and settle the suspended
  child. Add no dependency and no platform shell command.
- [ ] Extend `GitProcessRunner.run(...)`/`AsyncGitProcessRunner` with an
  explicit owned-process-tree option and an actual direct-child-spawn callback.
  Existing non-network callers retain their current behavior unless opted in.
- [ ] Ensure timeout/cancellation/shutdown never releases a retained token or
  mutation gate until owned descendants and pipes are terminal, or publishes
  an explicitly uncertain containment result.
- [ ] Re-run the focused command. Expected: PASS on the current platform with
  only the other platform's native test skipped.
- [ ] Run:

```bash
python3 -m ruff check tldw_chatbook/Notes/git_process_containment.py tldw_chatbook/Notes/file_notes_git_service.py Tests/Notes/test_git_process_containment.py Tests/Notes/test_file_notes_git_push_service.py
python3 -m compileall -q tldw_chatbook/Notes/git_process_containment.py tldw_chatbook/Notes/file_notes_git_service.py
git diff --check
```

Expected: all commands exit 0.

- [ ] Commit:

```bash
git add tldw_chatbook/Notes/git_process_containment.py tldw_chatbook/Notes/file_notes_git_service.py Tests/Notes/test_git_process_containment.py Tests/Notes/test_file_notes_git_push_service.py
git commit -m "feat(notes): contain retained Git process trees [TASK-1566]"
```

## Task 5: Build One Immutable Network Git Execution Context

**Files:**

- Create: `tldw_chatbook/Notes/file_notes_git_network.py`
- Modify: `tldw_chatbook/Notes/file_notes_git_push.py`
- Modify: `tldw_chatbook/Notes/file_notes_git_service.py`
- Modify: `Tests/Notes/test_file_notes_git_push.py`
- Modify: `Tests/Notes/test_file_notes_git_push_service.py`

- [ ] Write failing tests for owner-only temporary directory/file modes,
  private bare Git layout with its required empty `refs/` directory but no ref
  files/remotes/hooks, empty/disabled system and global config, controlled
  source-object alternate, and cleanup only after all retained work/recovery
  ends.
- [ ] Write failing allowlist tests starting from an empty environment. Preserve
  only required OS/Git/noninteractive-auth values; reject repository/index/
  object/config/namespace/replace redirects, identity/date overrides, askpass,
  editor/pager/prompt, proxy/transport overrides, provider tokens, and unrelated
  application state. Install Chatbook no-prompt controls and close stdin.
- [ ] Write failing config-copy tests allowing only exact validated
  credential-helper/transport-neutral key/value/origin facts. Reject rewrites,
  remote URL/refspec/mirror/options/receive-pack, extra headers, embedded
  credentials, TLS exceptions, proxies, and repository/worktree executable
  helpers.
- [ ] Write failing OpenSSH invocation tests for literal authorized host/user/
  port, batch mode, strict host-key verification, no forwarding/password/
  askpass/live user routing config, existing agent/default identity locations/
  standard known-hosts only, and no `ProxyCommand`, `ProxyJump`, host alias, or
  custom `IdentityFile`.
- [ ] Run:

```bash
python3 -m pytest Tests/Notes/test_file_notes_git_push.py Tests/Notes/test_file_notes_git_push_service.py -q -k "network_context or network_environment or openssh or cleanup"
```

Expected: FAIL because the immutable context does not exist.

- [ ] Implement `NetworkGitExecutionContext` and `NetworkContextFactory` in
  the new module. Use exact known-shape cleanup and never discover/reuse
  crash-left directories after restart.
- [ ] Make every future query/push command require the context identity and
  run against its bare Git directory. The frozen endpoint remains an argv
  argument, never a remote configuration entry.
- [ ] Re-run the focused command. Expected: PASS.
- [ ] Run:

```bash
python3 -m ruff check tldw_chatbook/Notes/file_notes_git_network.py tldw_chatbook/Notes/file_notes_git_push.py tldw_chatbook/Notes/file_notes_git_service.py Tests/Notes/test_file_notes_git_push.py Tests/Notes/test_file_notes_git_push_service.py
python3 -m compileall -q tldw_chatbook/Notes/file_notes_git_network.py tldw_chatbook/Notes/file_notes_git_push.py tldw_chatbook/Notes/file_notes_git_service.py
git diff --check
```

Expected: all commands exit 0.

- [ ] Commit:

```bash
git add Docs/superpowers/plans/2026-07-30-file-notes-guarded-session-push.md tldw_chatbook/Notes/file_notes_git_network.py tldw_chatbook/Notes/file_notes_git_push.py tldw_chatbook/Notes/file_notes_git_service.py Tests/Notes/test_file_notes_git_push.py Tests/Notes/test_file_notes_git_push_service.py
git commit -m "feat(notes): isolate guarded push execution [TASK-1566]"
```

## Task 6: Add Authorized Remote Preflight and Immutable Review

**Files:**

- Modify: `tldw_chatbook/Notes/file_notes_session_owner.py`
- Modify: `tldw_chatbook/Notes/file_notes_git_service.py`
- Modify: `Tests/Notes/test_file_notes_session_owner.py`
- Modify: `Tests/Notes/test_file_notes_git_push_service.py`

- [ ] Write failing service/owner tests for:
  - local candidate checking before authorization with zero network/helper
    calls;
  - Cancel/decline preserving the matching candidate;
  - exact authorization creating one immutable context and permitting one
    retained exact-ref query;
  - parent observation producing one immutable, opaque, single-use review;
  - candidate observation producing `Already published`, no push child, and
    exact-token candidate clear;
  - missing/deleted/divergent/plural/malformed/inaccessible observations
    blocking without review or push;
  - review binding candidate/root/repository/trust/config/authorization/
    context/operation facts while deliberately excluding session-change,
    status, staging, index, and worktree generations;
  - reauthorization after any bound drift and away-and-back ABA rejection; and
  - cancellation/shutdown/removal retaining the read-only child through
    process-tree settlement.
- [ ] Run:

```bash
python3 -m pytest Tests/Notes/test_file_notes_session_owner.py Tests/Notes/test_file_notes_git_push_service.py -q -k "push_authorization or push_preflight or push_review or already_published"
```

Expected: FAIL on missing retained push review cycle/capabilities.

- [ ] Add push-specific retained operation and review snapshots. Do not reuse
  `RetainedCommitOperation`, `CommitOutcome`, or commit recovery types.
- [ ] Implement start-local-proof, authorize-and-check, retained preflight,
  exact owner capture, single-use review issuance, and `Already published`.
  Use only the frozen endpoint/ref and immutable execution context.
- [ ] Preserve the candidate for Cancel/Blocked; clear only the matching token
  for `Already published`.
- [ ] Re-run the focused command. Expected: PASS.
- [ ] Run:

```bash
python3 -m ruff check tldw_chatbook/Notes/file_notes_session_owner.py tldw_chatbook/Notes/file_notes_git_service.py Tests/Notes/test_file_notes_session_owner.py Tests/Notes/test_file_notes_git_push_service.py
python3 -m compileall -q tldw_chatbook/Notes/file_notes_session_owner.py tldw_chatbook/Notes/file_notes_git_service.py
git diff --check
```

Expected: all commands exit 0.

- [ ] Commit:

```bash
git add tldw_chatbook/Notes/file_notes_session_owner.py tldw_chatbook/Notes/file_notes_git_service.py Tests/Notes/test_file_notes_session_owner.py Tests/Notes/test_file_notes_git_push_service.py
git commit -m "feat(notes): add guarded push review authority [TASK-1566]"
```

## Task 7: Execute the Exact Lease-Guarded Push and Prove CAS Semantics

**Files:**

- Modify: `tldw_chatbook/Notes/file_notes_git_service.py`
- Modify: `tldw_chatbook/Notes/file_notes_session_owner.py`
- Modify: `Tests/Notes/test_file_notes_git_push_service.py`
- Create: `Tests/Notes/test_file_notes_git_push_integration.py`

- [ ] Write failing controlled-service tests for:
  - Confirm consuming the review once;
  - fresh root/repository/branch/candidate/config/trust/authorization/LFS and
    exact remote-parent proof;
  - later note/status/staging/index/worktree changes not invalidating the fixed
    candidate;
  - Cancel accepted before actual push child spawn and rejected immediately
    after the runner's real spawn signal;
  - pre-spawn validation/launch failure remaining a non-mutating Blocked/
    Failure result rather than uncertain;
  - one exact push invocation followed by one exact postflight query;
  - exact accepted-result plus candidate postflight producing `Succeeded`;
  - naturally settled nonzero/unambiguous result plus parent postflight
    producing `Failed with no update currently observed`; and
  - timeout, lost result, contradiction, query failure, missing/other
    postflight, or unproved descendants producing `Uncertain`.
- [ ] Build disposable source and bare destination fixtures using a test-only
  local transport admission. Write failing real-Git tests that snapshot:
  - every remote ref/tag;
  - every local ref and symbolic `HEAD`;
  - logical index bytes via `ls-files --stage -v -z`;
  - config bytes;
  - selected worktree bytes/modes; and
  - logical File Notes replica rows/revisions/tombstones.
- [ ] Add barrier-driven failing races for destination deletion and divergent
  advance after final revalidation. Assert the exact lease neither recreates
  nor overwrites and only one update request is received.
- [ ] Add real-Git tests for one direct-child candidate update, no other remote
  ref/tag change, no local tracking-ref mutation, a second guarded commit while
  the remote remains at the older parent blocking rather than range-pushing,
  and a concurrent note edit changing only intended disk/replica state. Do not
  assert that the destination object-store files remain unchanged because
  receiving required objects is inherent to push.
- [ ] Add a barrier-driven test that changes source repository, worktree,
  global, and system rewrite/helper/transport configuration after final
  validation but before child spawn. Prove the retained child still uses only
  the frozen context, endpoint, helper policy, and transport policy.
- [ ] Add token-publication tests proving Success clears and definite Failure
  preserves only the matching candidate; local `HEAD` drift after push spawn
  may revoke later availability but cannot erase a remote result proved for
  the retained operation or overwrite newer owner state.
- [ ] Run:

```bash
python3 -m pytest Tests/Notes/test_file_notes_git_push_service.py Tests/Notes/test_file_notes_git_push_integration.py -q -k "confirm or exact_push or compare_and_swap or destination_delete or divergent or local_invariance or concurrent_edit"
```

Expected: FAIL on missing confirm/push/postflight implementation.

- [ ] Implement the exact direct-argv push, injected deterministic
  `before_push_spawn` barrier, 30/60-second deadline policy with injected clock,
  actual-spawn boundary, machine-result classifier, exact postflight, and
  token-scoped owner publication.
- [ ] Keep the mutation/transition gate through all child/descendant settlement
  while allowing `record_change()` and ordinary autosave. Do not acquire an
  editor read-only lease.
- [ ] Re-run the focused command. Expected: PASS.
- [ ] Run:

```bash
python3 -m pytest Tests/Notes/test_file_notes_git_push_integration.py -q
python3 -m ruff check tldw_chatbook/Notes/file_notes_git_service.py tldw_chatbook/Notes/file_notes_session_owner.py Tests/Notes/test_file_notes_git_push_service.py Tests/Notes/test_file_notes_git_push_integration.py
python3 -m compileall -q tldw_chatbook/Notes/file_notes_git_service.py tldw_chatbook/Notes/file_notes_session_owner.py
git diff --check
```

Expected: all commands exit 0.

- [ ] Commit:

```bash
git add tldw_chatbook/Notes/file_notes_git_service.py tldw_chatbook/Notes/file_notes_session_owner.py Tests/Notes/test_file_notes_git_push_service.py Tests/Notes/test_file_notes_git_push_integration.py
git commit -m "feat(notes): execute exact guarded session push [TASK-1566]"
```

## Task 8: Retain Uncertain Proof, Query Only, and Settle Shutdown

**Files:**

- Modify: `tldw_chatbook/Notes/file_notes_session_owner.py`
- Modify: `tldw_chatbook/Notes/file_notes_git_service.py`
- Modify: `Tests/Notes/test_file_notes_session_owner.py`
- Modify: `Tests/Notes/test_file_notes_git_push_service.py`
- Modify: `Tests/Notes/test_file_notes_git_push_integration.py`
- Modify: `Tests/ProductionApp/test_file_notes_session_owner_lifecycle.py`

- [ ] Write failing tests for uncertainty retaining only original endpoint/ref,
  parent/candidate/token, sanitized identity, trust epochs, context, and
  descendant settlement—not a reusable review or push capability.
- [ ] Write failing `Check remote again — no push` tests:
  - unavailable until every owned descendant is terminal;
  - one exact query to retained endpoint A even if live config now names B;
  - candidate observation converging to desired-state success without claiming
    causation;
  - parent observation remaining uncertain forever;
  - missing/other/query failure remaining needs-attention; and
  - zero push/refspec invocation and no automatic retry.
- [ ] Add the trust-change case: if the retained destination authorization is
  revoked, recovery may contact only the same frozen endpoint after a fresh
  authorization for that exact identity; it must never follow live config to a
  replacement endpoint.
- [ ] Write failing transition tests proving active/uncertain push blocks other
  Git actions and root/source/screen rebinding, while same-root note editing,
  autosave, and replica synchronization remain usable.
- [ ] Extend production lifecycle tests for owner-first settlement of active
  check, active push, and query-only recovery before replica teardown; context
  cleanup after certain settlement; candidate/recovery attribution discarded
  on restart; and crash-left temp context never discovered or reused.
- [ ] Run:

```bash
python3 -m pytest Tests/Notes/test_file_notes_session_owner.py Tests/Notes/test_file_notes_git_push_service.py Tests/Notes/test_file_notes_git_push_integration.py Tests/ProductionApp/test_file_notes_session_owner_lifecycle.py -q -k "push_recovery or uncertain_push or retained_push or push_shutdown or restart"
```

Expected: FAIL on missing recovery and shutdown state.

- [ ] Implement one retained uncertain push evidence object and owner
  query-only recovery admission. Destroy consumed push authority, keep the
  mutation/transition gate, and never reconstruct an update capability.
- [ ] Extend service shutdown's explicit retained task/context list. Owner
  discards process-only candidate/authorization/review/status/recovery state
  only after service settlement; process exit persists none of it.
- [ ] Re-run the focused command. Expected: PASS.
- [ ] Run:

```bash
python3 -m ruff check tldw_chatbook/Notes/file_notes_session_owner.py tldw_chatbook/Notes/file_notes_git_service.py Tests/Notes/test_file_notes_session_owner.py Tests/Notes/test_file_notes_git_push_service.py Tests/Notes/test_file_notes_git_push_integration.py Tests/ProductionApp/test_file_notes_session_owner_lifecycle.py
python3 -m compileall -q tldw_chatbook/Notes/file_notes_session_owner.py tldw_chatbook/Notes/file_notes_git_service.py
git diff --check
```

Expected: all commands exit 0.

- [ ] Commit:

```bash
git add tldw_chatbook/Notes/file_notes_session_owner.py tldw_chatbook/Notes/file_notes_git_service.py Tests/Notes/test_file_notes_session_owner.py Tests/Notes/test_file_notes_git_push_service.py Tests/Notes/test_file_notes_git_push_integration.py Tests/ProductionApp/test_file_notes_session_owner_lifecycle.py
git commit -m "feat(notes): retain uncertain guarded push proof [TASK-1566]"
```

## Task 9: Rehydrate Push State and Keep Session Git Truthful

**Files:**

- Modify: `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`
- Create: `Tests/UI/test_library_file_notes_git_push.py`
- Modify: `Tests/Notes/test_file_notes_git_push_service.py`

- [ ] Add a push-capable fake service/retained-operation harness modeled on the
  existing workspace harness, with call counts, actual-child-start signal,
  owner-public operation status, and deterministic settlement.
- [ ] Write failing workspace tests for:
  - push availability derived independently from commit draft/status rows;
  - later edits staling Session Git rows without hiding the candidate;
  - commit success resyncing but never auto-opening/starting push;
  - distinct binding/candidate/operation IDs suppressing stale callbacks;
  - checking/pushing/needs-attention copy in the persistent Session Git entry;
  - panel removal and `Back to Files — push continues` not canceling work or
    suppressing service/owner publication;
  - result settlement while hidden and exact rehydration on reopen;
  - no duplicate task/query/push on remount; and
  - no `_EditorReadOnlyLease` during any push phase.
- [ ] Run:

```bash
python3 -m pytest Tests/UI/test_library_file_notes_git_push.py Tests/Notes/test_file_notes_git_push_service.py -q -k "workspace or rehydrate or session_git_indicator or back_to_files or editable"
```

Expected: FAIL because the workspace protocol/state has no push path.

- [ ] Extend `_SessionGitService` and add push-specific workspace
  operation/state/observer helpers. Keep them separate from
  `_CommitBindingKey`, commit draft invalidation, and
  `_commit_operation_is_current()`.
- [ ] Rehydrate push before ordinary status rendering from only the
  owner/service candidate and retained operation. Observer settlement must not
  require `_navigator_mode == "git"`; render only when visible, but always
  publish/cache the terminal state.
- [ ] Centralize Session Git label composition so note changes and every push
  transition preserve `Push checking`, `Pushing`, or `Push needs attention`.
- [ ] Re-run the focused command. Expected: PASS.
- [ ] Run:

```bash
python3 -m ruff check tldw_chatbook/Widgets/Library/library_file_notes_workspace.py Tests/UI/test_library_file_notes_git_push.py Tests/Notes/test_file_notes_git_push_service.py
python3 -m compileall -q tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
git diff --check
```

Expected: all commands exit 0.

- [ ] Commit:

```bash
git add tldw_chatbook/Widgets/Library/library_file_notes_workspace.py Tests/UI/test_library_file_notes_git_push.py Tests/Notes/test_file_notes_git_push_service.py
git commit -m "feat(notes): rehydrate guarded push workflow [TASK-1566]"
```

## Task 10: Add the Separate Keyboard-Safe Push Presentation

**Files:**

- Modify: `tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py`
- Modify: `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`
- Modify: `Tests/UI/test_library_file_notes_git_push.py`

- [ ] Write failing mounted panel tests for independent `PushPanelPhase`,
  projection-only rendering, typed intents, and
  `Review push (1 commit)…` beneath—not inside—the commit actions.
- [ ] Write failing authorization modal tests showing exact sanitized endpoint,
  branch/ref, transport, process-only scope, helper/no-prompt disclosure, and a
  focusable/selectable `Endpoint Details` surface. Assert `Cancel` initially
  focused; Escape/window close declines; `Authorize and check` is affirmative.
- [ ] Write failing immutable-review tests for exact subject/OID/parent/
  candidate, branch, configured remote label, full ref, endpoint Details,
  included-note provenance/count/change types, exact lease, secure transport,
  hook bypass, remote-side-effects disclosure, and later-edits-local copy.
- [ ] Write failing action/focus tests for:
  - footer order `Back`, then `Push 1 commit`;
  - `Back` initially focused and Push last;
  - phase-plus-operation-ID focus repair;
  - buffered Enter from authorization/checking not crossing into review;
  - Cancel until actual child spawn only;
  - `Back to Files — push continues`;
  - `Review again`, never `Retry`;
  - `Check remote again — no push`; and
  - non-elided, selectable result/recovery copy.
- [ ] Assert the distinct visible/accessibility labels
  `Checking push candidate…`, `Checking remote before push…`,
  `Checking uncertain outcome…`, and `Pushing 1 reviewed commit…`; no hidden
  editor-status surface may be the sole announcement.
- [ ] Run:

```bash
python3 -m pytest Tests/UI/test_library_file_notes_git_push.py -q -k "panel or authorization_dialog or push_review or focus or buffered_enter or endpoint_details"
```

Expected: FAIL because the push panel/modal/intents do not exist.

- [ ] Add a separate push workflow body and fixed phase-specific footer inside
  `LibraryFileNotesGitPanel.DEFAULT_CSS`; do not overload
  `CommitPanelPhase`, commit projections, or `SessionGitTrustDialog`.
- [ ] Wire only typed panel intents through the workspace to service methods.
  The panel must not parse endpoints, build argv, store authority, classify raw
  results, or infer an outcome.
- [ ] Re-run the focused command. Expected: PASS.
- [ ] Run:

```bash
python3 -m ruff check tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py tldw_chatbook/Widgets/Library/library_file_notes_workspace.py Tests/UI/test_library_file_notes_git_push.py
python3 -m compileall -q tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
git diff --check
```

Expected: all commands exit 0.

- [ ] Commit:

```bash
git add tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py tldw_chatbook/Widgets/Library/library_file_notes_workspace.py Tests/UI/test_library_file_notes_git_push.py
git commit -m "feat(notes): add guarded push review UI [TASK-1566]"
```

## Task 11: Verify Compact, Remounted, and Lifecycle UX

**Files:**

- Modify: `Tests/UI/test_library_file_notes_git_push.py`
- Modify: `Tests/ProductionApp/test_file_notes_session_owner_lifecycle.py`

- [ ] Add a compact mounted matrix at `40x20` covering every push phase,
  outcome, disabled reason, recovery action, focus order, body scroll, fixed
  footer, Details open/close, accessible status, and focused-control
  visibility. Use real Pilot Tab/Shift+Tab/Enter/Escape/scroll events.
- [ ] Add one representative happy path at `120x40`, and retained-operation
  leave/reopen tests at both `40x20` and `120x40`. Do not create a
  phase-by-viewport Cartesian suite or repeat every state at `160x45`.
- [ ] Preserve the existing narrow Navigator/Editor switching and the existing
  `160x45` Files-source-entry regression. Assert results do not depend on the
  editor action-status surface hidden in compact Navigator/Prepare mode.
- [ ] Add production-owner lifecycle assertions for one active push and one
  uncertain query-only recovery, without launching the full app UAT in pytest.
- [ ] Run:

```bash
python3 -m pytest Tests/UI/test_library_file_notes_git_push.py Tests/ProductionApp/test_file_notes_session_owner_lifecycle.py -q
```

Expected: FAIL on the newly added compact/remount/lifecycle assertions until
focus/geometry/lifecycle defects are corrected.

- [ ] Make only focused presentation/lifecycle corrections required by these
  tests. Do not redesign unrelated File Notes controls or change app/screen
  ownership.
- [ ] Re-run the same command. Expected: PASS.
- [ ] Run:

```bash
python3 -m ruff check Tests/UI/test_library_file_notes_git_push.py Tests/ProductionApp/test_file_notes_session_owner_lifecycle.py
git diff --check
```

Expected: all commands exit 0.

- [ ] Commit:

```bash
git add Tests/UI/test_library_file_notes_git_push.py Tests/ProductionApp/test_file_notes_session_owner_lifecycle.py tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py tldw_chatbook/Widgets/Library/library_file_notes_workspace.py tldw_chatbook/Notes/file_notes_git_service.py tldw_chatbook/Notes/file_notes_session_owner.py
git commit -m "test(notes): harden guarded push lifecycle UX [TASK-1566]"
```

Only stage production files in this commit if the focused failing tests required
corrections to them.

## Task 12: Prove Secure SSH/HTTPS and Ambiguous Transport Behavior

**Files:**

- Create: `Tests/Notes/test_file_notes_git_push_transport.py`
- Modify: `Tests/Notes/test_file_notes_git_push_service.py`
- Modify: `Tests/Notes/test_file_notes_git_push_integration.py`
- Modify production push/network/containment files only if these tests expose a
  contract defect.

- [ ] Add a capability-probed ephemeral loopback OpenSSH fixture with generated
  client/server keys, isolated `HOME`/Git config/`.ssh`/`known_hosts`, literal
  loopback host, strict host checking, batch mode, fixture-only identities,
  prompt sentinels, and independent connection/receive counters. It must never
  access the user's SSH config, agent, keys, or credential state.
- [ ] Write SSH tests proving zero connections/helper calls before
  authorization, one read-only connection after authorization, no mutating
  request before final Confirm, one exact push, hostile live user SSH
  HostName/ProxyCommand/IdentityFile config ignored, unknown/wrong host key
  fails without prompting, bad authentication fails promptly, and provider
  canaries do not enter the child environment.
- [ ] Add a hermetic smart-Git HTTPS fixture using stdlib TLS/HTTP around
  `git http-backend`, a capability-probed ephemeral CA/leaf, isolated fixture
  trust store injected only through `NetworkContextFactory`, and a fake
  credential helper.
- [ ] Write HTTPS tests for certificate/hostname verification,
  noninteractive-helper behavior, prompt suppression, environment isolation,
  and raw credential/helper/server-output redaction. Do not claim support for
  every public hosting provider or credential manager.
- [ ] Add one end-to-end redaction-canary journey that injects unique canaries
  into endpoint userinfo/query rejection, environment, copied helper
  configuration, raw stdout/stderr, and hostile server text. Assert every
  canary is absent from classifier exceptions, Loguru capture, retained
  service/owner objects, public projections, UI text, accessibility
  announcements, and serialized evidence projections.
- [ ] Add a deterministic loopback response-drop fixture for:
  - server accepts the exact update then the client loses the result:
    `Uncertain`, zero retry, query-only recovery observes candidate; and
  - disconnect before acceptance: `Uncertain`, destination remains parent,
    repeated parent observations never become definite failure.
- [ ] Run:

```bash
python3 -m pytest Tests/Notes/test_file_notes_git_push_transport.py Tests/Notes/test_file_notes_git_push_service.py Tests/Notes/test_file_notes_git_push_integration.py -q -k "ssh or https or authorization or prompt or redaction or dropped_response or no_retry"
```

Expected: FAIL until fixture-backed transport, redaction, and ambiguity
contracts hold. Capability-unavailable lanes must be explicit skips, not false
passes or substituted local-transport claims.

- [ ] Fix only defects within the approved production policy. Do not add custom
  CA, credential, SSH-routing, or local-transport user settings.
- [ ] Re-run the focused command. Expected: PASS with only honestly
  capability-gated skips.
- [ ] Run:

```bash
python3 -m ruff check Tests/Notes/test_file_notes_git_push_transport.py Tests/Notes/test_file_notes_git_push_service.py Tests/Notes/test_file_notes_git_push_integration.py tldw_chatbook/Notes/file_notes_git_push.py tldw_chatbook/Notes/file_notes_git_network.py tldw_chatbook/Notes/git_process_containment.py tldw_chatbook/Notes/file_notes_git_service.py
git diff --check
```

Expected: all commands exit 0.

- [ ] Commit:

```bash
git add Tests/Notes/test_file_notes_git_push_transport.py Tests/Notes/test_file_notes_git_push_service.py Tests/Notes/test_file_notes_git_push_integration.py tldw_chatbook/Notes/file_notes_git_push.py tldw_chatbook/Notes/file_notes_git_network.py tldw_chatbook/Notes/git_process_containment.py tldw_chatbook/Notes/file_notes_git_service.py
git commit -m "test(notes): verify guarded push transports [TASK-1566]"
```

Only stage production files if the transport tests required contract fixes.

## Task 13: Run Same-Process Production-App PTY Acceptance

**Files:**

- Create:
  `Docs/superpowers/qa/file-notes-guarded-push-uat-2026-07-30/README.md`
- Create sanitized transcript/capture/evidence/manifest files beneath that
  directory.
- Modify production/test files only if UAT finds a reproducible approved-scope
  defect; if so, add a failing focused regression test before the fix and
  repeat affected evidence.

- [ ] Probe loopback OpenSSH, tmux/PTY, Git, and terminal-capture capability.
  If an isolated production-compatible OpenSSH destination is unavailable,
  record the missing lane, do not relabel a local/injected transport run as
  end-to-end network UAT, and leave TASK-1566 In Progress. The PTY lane must be
  run in a suitable environment before Done/ADR acceptance.
- [ ] Launch the unmodified production application with:

```bash
python -m tldw_chatbook.app
```

Use isolated synthetic config/data/HOME/tmp, notes root, SQLite data, Git
repository, and OpenSSH destination. Do not use `App.run_test()`, Pilot, seeded
candidate state, widget `.press()`, or private handlers.

- [ ] In one application process, use actual terminal keyboard input to:
  1. navigate Library -> Notes -> Files;
  2. select the synthetic notes root;
  3. edit/autosave with exact frontmatter preservation;
  4. stage the intended session notes;
  5. create and prove the guarded commit;
  6. open `Review push`;
  7. cancel authorization and externally prove zero connections;
  8. reopen, authorize, and observe one read-only connection;
  9. review the exact commit/ref/endpoint/lease/policy facts;
  10. confirm and externally prove one exact ref transition;
  11. leave/reopen during a retained delayed operation with no duplicate;
  12. continue editing while the fixed candidate/push remains unchanged; and
  13. verify Tab/Shift+Tab/Enter/Escape/scroll/Details/result at `40x20`.
- [ ] Run a second focused missing or divergent destination scenario and prove
  no push/recreation/overwrite.
- [ ] Capture a truthful phase-to-evidence matrix matching the design. Do not
  claim full-app acceptance for uncertainty, process-tree, or HTTPS behavior
  covered only by deterministic automated lanes.
- [ ] Store only sanitized:
  - `README.md` with source commit, launch command, platform, Git/OpenSSH
    versions, dimensions, `agent-operated`, exact steps, verdict, and gaps;
  - keyboard/action and terminal transcripts;
  - key phase/viewport captures;
  - `evidence.json` with parent/candidate OIDs, sanitized endpoint/ref,
    pre/post local and remote ref maps, connection/push/prompt/helper counts,
    logical note/replica assertions, and test-lane labels;
  - process-tree settlement evidence from the native automated lane;
  - redaction-canary scan result; and
  - SHA-256 manifest for every retained artifact.
- [ ] Assert the bundle contains no private key, credential, raw helper/server
  diagnostic, note body, real user path, unsanitized fixture absolute path, or
  redaction canary. Scan every nested retained artifact, including terminal
  captures and accessibility/action transcripts.
- [ ] Create `SHA256SUMS` from a sorted recursive list of every retained file
  except `SHA256SUMS` itself. Store paths relative to the QA directory.
- [ ] Validate:

```bash
python3 -m json.tool Docs/superpowers/qa/file-notes-guarded-push-uat-2026-07-30/evidence.json >/dev/null
python3 -c 'from pathlib import Path; r=Path("Docs/superpowers/qa/file-notes-guarded-push-uat-2026-07-30"); m=r/"SHA256SUMS"; recorded={line.split("  ",1)[1] for line in m.read_text().splitlines() if line}; actual={str(p.relative_to(r)) for p in r.rglob("*") if p.is_file() and p != m}; assert recorded == actual, (sorted(recorded-actual), sorted(actual-recorded))'
(cd Docs/superpowers/qa/file-notes-guarded-push-uat-2026-07-30 && shasum -a 256 -c SHA256SUMS)
! rg -n --hidden 'GUARDED_PUSH_(SECRET|PATH)_CANARY|/Users/|/home/' Docs/superpowers/qa/file-notes-guarded-push-uat-2026-07-30
git diff --check
```

Expected: JSON parses, manifest coverage exactly matches all nested retained
artifacts except the manifest itself, every recorded hash verifies, the
private-path/canary scan finds no match, and diff check passes.

- [ ] Commit:

```bash
git add Docs/superpowers/qa/file-notes-guarded-push-uat-2026-07-30
git commit -m "test(notes): record guarded push acceptance [TASK-1566]"
```

## Task 14: Focused Regression Gate and Backlog/ADR Closeout

**Files:**

- Modify:
  `backlog/tasks/task-1566 - Add-guarded-exact-session-commit-push-to-File-Notes.md`
- Modify: `backlog/decisions/039-file-notes-guarded-session-push.md`
- Modify: relevant implementation/spec documentation only if behavior changed
  within the approved boundary.

- [ ] Run the complete new boundary:

```bash
python3 -m pytest Tests/Notes/test_file_notes_git_push.py Tests/Notes/test_file_notes_git_push_service.py Tests/Notes/test_file_notes_git_push_integration.py Tests/Notes/test_file_notes_git_push_transport.py Tests/Notes/test_git_process_containment.py Tests/UI/test_library_file_notes_git_push.py Tests/ProductionApp/test_file_notes_session_owner_lifecycle.py -q
```

Expected: PASS, with only explicitly documented capability/platform skips.
The production-compatible OpenSSH full-app PTY lane is not skippable for task
completion; if it has not run successfully in a suitable environment, stop
before ADR/task closeout.

- [ ] Run the affected adjacent File Notes regression boundary:

```bash
python3 -m pytest Tests/Notes/test_file_notes_session_owner.py Tests/Notes/test_file_notes_git_service.py Tests/Notes/test_file_notes_git_integration.py Tests/Notes/test_file_notes_git_commit.py Tests/Notes/test_file_notes_git_commit_integration.py Tests/UI/test_library_file_notes_git.py Tests/UI/test_library_file_notes_workspace.py -q
```

Expected: PASS. This is the largest local test command in the plan; do not
expand it into repository-wide pytest, coverage, or broad CI.

- [ ] Run targeted static/compile/document checks on the actual changed-file
  list:

```bash
python3 -m ruff check tldw_chatbook/Notes/file_notes_git_push.py tldw_chatbook/Notes/file_notes_git_network.py tldw_chatbook/Notes/git_process_containment.py tldw_chatbook/Notes/file_notes_session_owner.py tldw_chatbook/Notes/file_notes_git_service.py tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py tldw_chatbook/Widgets/Library/library_file_notes_workspace.py Tests/Notes/test_file_notes_git_push.py Tests/Notes/test_file_notes_git_push_service.py Tests/Notes/test_file_notes_git_push_integration.py Tests/Notes/test_file_notes_git_push_transport.py Tests/Notes/test_git_process_containment.py Tests/Notes/test_file_notes_session_owner.py Tests/Notes/test_file_notes_git_commit_integration.py Tests/UI/test_library_file_notes_git_push.py Tests/ProductionApp/test_file_notes_session_owner_lifecycle.py
python3 -m compileall -q tldw_chatbook/Notes/file_notes_git_push.py tldw_chatbook/Notes/file_notes_git_network.py tldw_chatbook/Notes/git_process_containment.py tldw_chatbook/Notes/file_notes_session_owner.py tldw_chatbook/Notes/file_notes_git_service.py tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
python3 -m json.tool Docs/superpowers/qa/file-notes-guarded-push-uat-2026-07-30/evidence.json >/dev/null
git diff --check
```

Expected: all commands exit 0.

- [ ] Perform a self-review against all ten TASK-1566 acceptance criteria and
  the phase-to-evidence matrix. Verify in particular:
  - no pre-authorization network/helper contact;
  - no candidate range broadening;
  - no live-config TOCTOU redirection;
  - exact lease/refspec and no retry;
  - no local/ref/index/config/note/replica mutation during quiescent push;
  - truthful uncertainty and query-only recovery;
  - no raw secret/private output;
  - editable notes during retained push; and
  - no database/durable push state/general Git feature.
- [ ] Request focused code review. Resolve technically valid findings with
  failing regression tests first and repeat only affected focused commands.
- [ ] After implementation, tests, review, and UAT are all green:
  - confirm the required production-compatible OpenSSH same-process PTY lane
    completed successfully, with zero-preauthorization-contact and exact
    one-ref evidence;
  - change ADR-039 from `Proposed` to `Accepted`;
  - check every TASK-1566 acceptance criterion;
  - add concise `## Implementation Notes` covering approach, tradeoffs, files,
    focused verification, UAT evidence, and ADR-039;
  - set TASK-1566 to Done through Backlog CLI.
- [ ] Run:

```bash
backlog task 1566 --plain
git status --short
git diff --check
```

Expected: TASK-1566 is Done with every AC checked, implementation notes and ADR
link present, and only intended closeout files are uncommitted.

- [ ] Commit:

```bash
git add "backlog/tasks/task-1566 - Add-guarded-exact-session-commit-push-to-File-Notes.md" backlog/decisions/039-file-notes-guarded-session-push.md Docs/superpowers/specs/2026-07-30-file-notes-guarded-session-push-design.md Docs/superpowers/plans/2026-07-30-file-notes-guarded-session-push.md
git commit -m "docs(notes): close guarded session push [TASK-1566]"
```

Do not mark the task Done or ADR Accepted merely because code exists. Both
changes occur only after focused automated verification, independent review,
and the truthful production-app PTY evidence are complete.
