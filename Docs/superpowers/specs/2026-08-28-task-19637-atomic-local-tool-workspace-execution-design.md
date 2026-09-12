# Atomic Local-Tool Workspace Execution Design

**Date:** 2026-08-28
**Status:** Review requested
**Related task:** TASK-19637
**Prerequisite for:** TASK-19504
**ADR required:** yes
**ADR path:** `backlog/decisions/101-one-shot-pinned-workspace-tool-execution.md`
**Reason:** This establishes a cross-platform helper-process security boundary
and defines its authority, lifecycle, failure, compatibility, and performance
contracts.

## 1. Summary

Workspace-confined local file and Git tools currently validate a `Path` and then
use that pathname later. A concurrent rename or replacement can retarget the
operation after the check. Call-time binding identity checks reduce the window
but cannot close it.

Each affected operation will instead run in a fresh, one-shot worker. The parent
revalidates the run-admitted workspace authority and admits the worker to a
retained process group or Job Object. The worker independently opens and verifies
the exact root, pins it for the operation, executes one closed operation, emits a
bounded result, and exits. No helper remains running between calls.

This is a root-identity execution lease, not a general OS sandbox. It protects
the exact failure TASK-19637 records while preserving existing permission,
sensitive-path, in-root symlink, linked-worktree, and result behavior when no
drift occurs.

## 2. Current gap

The affected path is:

```text
provider root guard
  -> resolve_workspace_path(...).resolve()
  -> ordinary Path or git -C pathname use
```

`LocalToolProvider.invoke()` runs `root_guard` before dispatch and may hold an
`authority_scope`, but the handlers captured by `_default_specs()` still receive
an ordinary `Path`. `local_tool_impls.py` and `git_tool_impls.py` resolve that
path again and reopen it later. The same cores are called directly by
`VirtualCliRegistry`, so fixing only `LocalToolProvider` would leave a bypass.

The race is externally reproducible:

1. admit root identity A at locator L;
2. finish the current pathname confinement check;
3. rename A away and place attacker-controlled root B at L;
4. allow the existing implementation to reopen L;
5. observe a read from or write into B.

On POSIX, a directory descriptor continues to refer to A after its name moves.
On Windows, a helper can set its own process current directory after identity
verification; Windows documents that current directory as locked against
movement, deletion, and rename while the process executes. Neither mechanism is
safe to establish by changing the multithreaded Chatbook process's current
directory or by running arbitrary Python in `preexec_fn`.

## 3. Goals and non-goals

### 3.1 Goals

1. Bind each affected call to the exact run-admitted root identity across root
   rename, replacement, symlink, junction, and reparse substitution attempts.
2. Prevent mutating local tools from writing into a replacement root.
3. Prevent read-only local and Git tools from returning replacement-root data.
4. Work or fail closed on macOS, Linux, and Windows without `preexec_fn`.
5. Preserve ordinary tool schemas, permission behavior, result text, sensitive
   path rules, in-root symlink behavior, and linked Git worktrees.
6. Keep helper requests, file contents, patches, and absolute roots out of argv,
   environment, generic diagnostics, and process listings.
7. Prove bounded cleanup for the helper and every Git descendant.
8. Measure the cost of one-shot startup before closeout.

### 3.2 Non-goals

- A general filesystem sandbox against privileged or arbitrary hostile code.
- Raw CLI or persistent terminal confinement.
- Mutating Git commands.
- A persistent helper pool or daemon.
- Changing permission-store, approval, or Change Review semantics.
- Redesigning sensitive-path policy.
- Eliminating every possible concurrent mutation of arbitrary descendants. The
  task owns retargeting of the admitted root; existing descendant validation and
  in-root symlink behavior remain in force.
- Making a multi-file patch transactionally atomic. It remains one root-pinned
  operation with the existing partial-failure contract.
- Consolidating every process-tree controller already present in the repository.

## 4. Scope

The pinned executor covers every production frontend that reaches the local path
cores:

- `fs_list`, `fs_read`, `fs_write`, `fs_edit`, `fs_patch`, `fs_glob`, and
  `fs_grep`;
- `stat_path`, currently exposed through Virtual CLI;
- `git_status`, `git_diff`, `git_log`, `git_blame`, and `git_branches`;
- Virtual CLI `ls`, `cat`, `grep`, `find`, `stat`, and read-only Git commands;
- external MCP serving when it composes the same Local Tool provider.

The separate built-in scratch/multi-root file tools remain under their existing
ADR-028 and ADR-069 authority. TASK-19504 deliberately preserves that family
while changing the Local Tool provider's root selection. Raw CLI remains under
ADR-094 and is explicitly not workspace-confined.

## 5. Authority and linearization

### 5.1 Immutable admitted authority

Provider construction receives an immutable authority snapshot containing:

- workspace ID;
- stable binding ID or configured-root authority kind;
- canonical-locator fingerprint;
- access mode;
- canonical root locator, derived under the existing ancestor-identity checks
  and kept private;
- canonical root identity and available ancestor identities; and
- authority generation or equivalent invalidation token when the owning
  registry supplies one.

TASK-19504 will expand this snapshot to selected and run-admitted bindings. This
task supports both that future form and today's configured-workspace root without
making a registry lookup an authorization source inside the helper.

### 5.2 Admission sequence

One operation is admitted in this order:

1. Resolve the exact provider owner and permission verdict as today.
2. Re-read and compare binding membership, locator fingerprint, access mode, and
   filesystem identity through the provider's root guard.
3. For a mutation, require current `rw` authority.
4. Normalize the already-validated operation into an immutable bounded request.
5. Spawn the fixed helper in a root-ineligible protocol-wait state and admit its
   process tree.
6. Send the request over the private stdin channel.
7. The helper opens and verifies the root, pins it, and publishes a content-free
   `admitted` frame.
8. The helper performs the operation and publishes exactly one terminal frame.

Successful root pinning is the filesystem execution linearization point. A root
or registry change before it refuses the operation. A registry downgrade after
it does not retarget or asynchronously cancel the in-flight call; it blocks all
later calls. This is the only enforceable meaning of an immediate downgrade that
does not introduce unsafe mid-write cancellation.

## 6. One-shot worker protocol

### 6.1 Launch

The parent launches the helper with an absolute `sys.executable` and a fixed
module entry point. `shell=False` is mandatory. Argv contains only fixed
code-owned switches; it contains no root, file path, patch, file content, model
argument, or executable selected by the caller. A workspace originally selected
through a symlink alias is canonicalized during run admission; the worker gets
that admitted canonical locator and identity, not an alias it would resolve
again at operation time.

The environment starts from a small Python/runtime allowlist. Provider API keys,
tokens, proxy credentials, tracing state, Python injection variables, and
unrelated ambient values are absent. The helper has `stdin`, `stdout`, and
`stderr` pipes and no interactive input.

The existing spawned-worker containment pattern in
`STT/executor_process_tree.py` is reused rather than adding a third Windows Job
Object implementation. The helper starts in a fixed protocol-wait path that
cannot read a workspace request or spawn descendants before containment is
acknowledged:

- POSIX: the fixed helper enters or is launched into a new session/process group
  using a safe subprocess/spawn facility, never `preexec_fn`;
- Windows: the parent obtains the fixed helper PID and the helper waits until the
  parent assigns it to a kill-on-close Job Object;
- only after the parent acknowledges containment does the helper read and admit
  the workspace request.

The integration may extract those generic primitives to a shared module only if
needed to avoid a domain-layer import. That extraction must preserve the STT API
and tests and must not redesign the separate asynchronous Notes Git controller.

### 6.2 Request

The request is a versioned JSON object with a hard byte cap and exact keys:

- protocol version;
- opaque operation ID;
- closed operation name;
- read or write intent;
- private canonical root locator captured at run admission;
- expected root and available ancestor identities;
- normalized, bounded operation arguments; and
- effective timeout and output ceilings.

Unknown keys, unknown versions, duplicate JSON keys, non-finite numbers, wrong
types, NUL, oversized strings/collections, or unsupported operations fail before
root access. File content and patch text are permitted only for their exact
mutating operations and receive explicit request-size ceilings.

The worker never accepts shell text, Python source, module names, arbitrary argv,
environment overrides, or caller-selected executable paths.

### 6.3 Response

Stdout carries framed protocol messages only:

- one optional content-free `admitted` frame; and
- exactly one terminal success or failure frame.

The terminal frame includes the operation ID, stable outcome code, bounded
result/error text, elapsed time, and truncation/cleanup flags. Absolute root text
is redacted before serialization. Stderr is bounded and reserved for
content-free helper diagnostics; it is never copied directly into model output.

Missing, malformed, duplicate-terminal, oversized, wrong-ID, or post-terminal
frames are protocol failures. A protocol failure never triggers an in-process
fallback.

## 7. Platform root pinning

### 7.1 POSIX

The helper receives the admitted canonical locator rather than re-resolving any
user-visible alias. It then:

1. opens the lexical root with directory, close-on-exec, and no-follow-final
   flags supported by the platform;
2. `fstat`s the descriptor and compares its directory identity with the admitted
   identity;
3. rejects an unsafe or mismatched descriptor;
4. calls `fchdir(root_fd)` inside the already-single-threaded helper;
5. verifies `stat(".")` against the same descriptor identity; and
6. keeps `root_fd` open until the terminal result is committed.

Filesystem operations receive normalized workspace-relative paths and use
descriptor-relative standard-library primitives where supported. They never
reconstruct `root_locator / relative_path` after pinning. Enumeration starts
from the descriptor (`scandir`/`listdir`, `glob`/`fwalk` equivalents as
appropriate) and retains the current result caps and filtering.

The parent-side validation still resolves current in-root symlinks and converts
an admitted exact target to a stable root-relative target. The helper is not a
license to accept `..`, absolute operation paths, or a target that parent policy
did not admit.

### 7.2 Windows

The helper receives the admitted canonical locator rather than re-resolving any
user-visible alias. It uses the existing identity representation where
sufficient and a small native wrapper where Python metadata cannot prove the
contract. It:

1. opens the root as a directory without following a reparse point;
2. rejects a symlink, junction, or other reparse point at that canonical
   locator;
3. compares volume/file identity with the admitted identity;
4. calls `SetCurrentDirectoryW` only inside the helper;
5. opens and verifies `.` against the expected identity; and
6. retains both the current-directory state and verification handle until the
   operation is terminal.

Windows documents the process current directory as locked against movement,
deletion, and rename. If native identity, reparse attributes, current-directory
locking, or post-set verification cannot be proven, the operation fails closed.

Operation paths remain relative to the pinned helper current directory. The
multithreaded Chatbook process never calls `SetCurrentDirectory`.

## 8. Filesystem operation behavior

The implementation separates policy normalization from root-relative execution:

```text
model args
  -> existing schema and sensitive-path validation
  -> immutable WorkspaceToolRequest
  -> pinned worker
  -> root-relative operation core
  -> existing bounded result text
```

The existing public sync functions may remain as compatibility/test adapters,
but production Local Tool and Virtual CLI invocation must use the pinned
executor. No frontend may silently call the old pathname implementation after a
lease failure.

`fs_patch` parses and validates every target before helper admission, then sends
one request. The worker retains one root pin across the whole patch. Existing
patch ordering, failure, newline, and partial-write behavior remains unchanged.

Enumeration retains current distinctions:

- list and glob may report an escaping symlink's name where current policy
  allows name-only disclosure;
- grep and reads do not return content resolved outside the admitted root;
- sensitive entries remain filtered before content access; and
- result ordering and truncation caps remain unchanged.

## 9. Git behavior

The helper resolves Git through the existing sanitized environment and uses an
absolute executable path. Git receives fixed allowlisted subcommands and
validated arguments exactly as today, except:

- the workspace locator is never passed through `git -C`;
- Git inherits the helper's pinned current directory;
- repository discovery and model-supplied path scopes are workspace-relative;
- the Git child does not use `start_new_session=True` or otherwise leave the
  helper's admitted process group/Job Object; and
- parent cancellation/timeout owns the entire helper/Git tree.

The current Git runner's internal output and command timeouts remain useful, but
they cannot create a second process-group owner. One layer has authoritative
tree cleanup and terminal reporting.

Linked worktrees remain supported. A worktree-local `.git` file may reference
Git administrative metadata outside the workspace root; this is Git's existing
repository metadata contract, not general filesystem authority. The allowlisted
read-only Git commands, sanitized environment, sensitive-path pathspecs, and
repository-root-within-workspace checks remain unchanged.

## 10. Failure and lifecycle

Terminal outcomes distinguish:

- authority changed before spawn;
- process containment unavailable;
- helper spawn or admission failure;
- root identity mismatch;
- unsafe root link/reparse metadata;
- unsupported platform capability;
- invalid operation request;
- tool-domain failure;
- timeout or cancellation;
- malformed/oversized helper response; and
- cleanup unproven.

Model-facing text is stable, bounded, and recovery-oriented without absolute
paths. Generic logs contain operation kind, stable reason code, platform, timing,
and cleanup state only.

Timeout and cancellation first request ordinary helper termination where safe,
then force-terminate the retained process group/Job Object and wait boundedly for
settlement. Closing a Windows kill-on-close Job handle is part of the fail-safe
cleanup path. A helper cannot be reported complete while a Git descendant is
still unaccounted for.

No error path executes the operation in Chatbook's process. No error path retries
against a newly resolved root.

## 11. Compatibility

The following remain stable when no drift occurs:

- model-facing schemas and tool names;
- permission identities, Ask/Allow/Off behavior, and approval copy;
- `LocalToolError`-style result text;
- file read numbering, size caps, binary refusal, edit/patch semantics, and
  enumeration ordering;
- sensitive-path and `.git` metadata write refusal;
- read-only Git output and linked-worktree support;
- configured-root external MCP behavior; and
- selected-binding behavior introduced by TASK-19504.

Persistent approvals may change only if a schema or definition hash must change;
the implementation should avoid such churn because the execution boundary is
internal.

## 12. Verification

### 12.1 Deterministic root-race tests

Test-only barriers pause the helper at two points: immediately before root pin
and after successful pin but before operation use.

For each relevant platform and operation class:

1. create authorized root A and replacement root B with distinguishable
   sentinels;
2. capture A's admitted identity;
3. pause at the chosen barrier;
4. rename, replace, symlink, junction, or reparse-substitute the locator where
   the platform permits;
5. resume; and
6. assert the operation uses A or refuses safely, never B.

Mutating cases prove B and every external sentinel remain byte-exact. Read cases
prove no B content appears. Windows tests accept a documented sharing-violation
refusal when the pinned current directory prevents the attempted rename.

### 12.2 Operation and compatibility tests

- Every structured local path tool routes through the executor.
- Every Virtual CLI filesystem/Git command routes through the executor.
- `fs_patch` keeps one pin across all targets.
- `rw`/`ro`, root guard, kill switch, approvals, and result redaction remain
  unchanged.
- In-root file symlink, escaping symlink name-only glob behavior, sensitive-path
  filtering, and linked Git worktree coverage remain green.
- Git never receives `-C <workspace-locator>` and never starts a new contained
  session inside the helper.

### 12.3 Lifecycle and protocol tests

- worker waits for process-containment acknowledgement;
- invalid/oversized requests fail before root access;
- wrong identity, unsafe reparse metadata, malformed response, timeout,
  cancellation, crash, and forced kill fail closed;
- POSIX grandchildren die with the retained process group;
- Windows grandchildren die with the Job Object;
- descriptors, handles, pipes, and temporary resources close on every terminal
  path; and
- no request body, result body, absolute root, patch, or file content reaches
  argv, environment, or generic logs.

### 12.4 Performance evidence

Record cold and warm median/p95 wall time for representative `stat`, small read,
small write, directory list, Git status, and Git diff operations against the
current direct implementation. Record helper startup separately from operation
time. The task does not introduce a flaky CI timing threshold; evidence must be
published in implementation notes, and a severe regression must be optimized or
explicitly brought back for design review rather than hidden by pooling.

### 12.5 Test scope

Development uses focused suites for the execution protocol, local cores, Local
Tool provider, Virtual CLI, Git, project-instruction authority, and platform
containment. Per repository policy, a full test sweep is run only with explicit
user approval before final merge.

## 13. Delivery and migration

1. Land this design and ADR before implementation.
2. Add the one-shot protocol and platform root pin adapters behind focused red
   race tests.
3. Add root-relative filesystem/Git execution without changing public behavior.
4. Route Local Tool and Virtual CLI production calls through the executor.
5. Run compatibility, lifecycle, privacy, and performance evidence.
6. Complete TASK-19637 and merge it before starting TASK-19504's new root
   selection behavior.

There is no database, config, or user-data migration. A platform that cannot
prove the required boundary reports local path tools unavailable for that call;
it never falls back to the old pathname-only execution.

## 14. Open implementation constraints

These are constraints to resolve in the implementation plan, not product choices
left open by the design:

- choose the smallest shared import surface for the existing worker process-tree
  primitives without duplicating Windows Job Object code;
- define exact request/output byte caps from current tool and run budgets;
- keep direct sync core APIs only where tests or non-production compatibility
  require them, while proving every production frontend is leased; and
- re-check ADR-101 against `origin/dev` and open PRs immediately before merge,
  renumbering the file, header, README row, task, spec, and plan references if a
  collision appears.
