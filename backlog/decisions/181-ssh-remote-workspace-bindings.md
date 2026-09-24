# ADR-181: SSH remote workspace bindings

Status: Proposed
Date: 2026-09-24
Related Task: [TASK-32926](../tasks/task-32926%20-%20SSH-remote-workspace-bindings.md)
Design: [SSH Remote Workspace Bindings — Design](../../Docs/superpowers/specs/2026-09-24-ssh-remote-workspace-bindings-design.md)

## Decision

- **`SSH_FILESYSTEM` binding kind.** Named workspaces gain SSH folder
  bindings (`RuntimeBindingKind.SSH_FILESYSTEM`) whose tool access reuses
  the existing one-shot pinned-worker architecture (ADR-101) over SSH stdio
  instead of a local `Popen`. From the worker's perspective the remote root
  *is* a local filesystem, so pinning (`O_DIRECTORY|O_NOFOLLOW`, `fchdir`,
  `(st_dev, st_ino)` identity), confinement, atomic writes, and CAS identity
  checks keep their current semantics and run server-side; `fs_*` parity and
  the ro/rw toggle are unchanged.
- **Transient stdlib-only worker over SSH stdio.** Per tool call, chatbook
  spawns `ssh [opts] [-p <port>] [-l <user>] -- <host> <interpreter> -I -c
  '<bootstrap>'` — argv built from the parsed locator components, never a
  shell string — writes a zlib-compressed, stdlib-only worker bundle
  (size-prefixed) followed by the request to the subprocess's stdin, and
  reads the magic-prefixed response from stdout. The size and magic prefixes
  are SSH-transport additions, not protocol changes. Nothing is installed or
  persisted on the server: the worker's code arrives over the session's
  stdin and vanishes with it. Authentication is fully delegated to the
  user's ssh-agent / `~/.ssh/config` (BatchMode; never prompt); chatbook
  stores no credentials.
- **Admitted-marker bucketing + two-tier watchdog.** Op failures are
  classified by the protocol's existing `admitted` marker, not exit-code
  guessing: failures before admission (unreachable, auth-failing,
  handshake-timing-out) mark the binding `BLOCKED`; failures after
  admission — including timeouts — never change status, and a timed-out op
  is a typed tool error whose root is admitted again on the next send. The
  worker-side watchdog is two-tier: a graceful Timer that unlinks the
  worker's registered temp files, plus an OS-level `signal.alarm` backstop
  (exit 75) that cannot be starved by GIL-holding C code — no orphaned
  processes, and no orphaned temp files on the graceful tier.
- **Status cache: optimistic cold start, transport-only learning, debounced
  recovery.** Availability is the existing per-binding `READY` / `BLOCKED` /
  `MISSING` model driven by a cheap `ping` op, cached in memory (never read
  from the stored status column at admission), optimistically admitted at
  cold start, learned only from transport-classified op outcomes, and
  recovered automatically by a debounced background probe once the host
  returns or a recreated root's identity is re-captured. Run composition
  reads cache only — the dispatch hot path never touches the network or
  spawns a subprocess — and degradation excludes the binding from the run
  exactly like a missing local folder.
- **`LocalRoot | RemoteRoot` type split.** Remote roots get their own type
  because roughly a dozen call sites today do laptop-disk work off
  `RunAdmittedWorkspaceRoot.root: Path` (request building, CAS hashing,
  exclusions, admission checks). A distinct type makes every un-migrated
  site fail loudly instead of quietly reading the laptop's copy of a path
  like `/tmp/x`. Prerequisite Phase 0 makes the worker's import closure
  stdlib-only while the parent side keeps pydantic request/response
  acceptance (ADR-175); the bundle ships a stdlib decoder whose
  accept/reject behavior is provably identical (shared conformance corpus).
- **Executor-owned ControlMaster.** Warm connections come from an
  explicitly-managed OpenSSH ControlMaster (`ssh -MNf` in its own session;
  per-call clients with `ControlMaster=no`; per-host lock;
  failure-triggered restart) so a timed-out call's process-group kill cannot
  take down the shared connection. ControlMaster sockets live on the laptop,
  not the server; Windows-as-local-host degrades to per-call connections via
  the same kill switch.

## Context

The Console's named Workspaces can bind local folders only
(`RuntimeBindingKind.LOCAL_FILESYSTEM` is the single implemented kind). A
user running chatbook on a laptop who wants their workspace root — plus
AGENTS.md project instructions — to live on a remote Linux server has no
supported path today: mounting (sshfs/FUSE) is fragile on macOS 26 and a
dead server can hang a stale mount, and nothing in the app speaks SSH.

The requirement is a first-class remote binding: chatbook stays on the
laptop, connects via SSH, exposes a folder on the remote server as a
workspace binding alongside ordinary local folder bindings, and degrades
gracefully — an unreachable remote never blocks use of the workspace's
local bindings or scratch. Availability is guaranteed in both directions:
an unreachable, auth-failing, or handshake-timing-out remote is excluded
from the run's admitted roots while sends still compose, and an operation
that runs past its deadline is a typed tool error that never changes the
binding's status, with automatic debounced recovery. Nothing is installed,
persisted, or left behind on the server, and no SSH credentials are stored.
The remote binding can be the Console working folder, so remote AGENTS.md /
AGENTS.override.md load under all existing ADR-068/069 rules (byte caps,
untrusted content, activation ledger, first-use consent) and never grant
tool permission. Server prerequisites: sshd with exec, and `python3` ≥ 3.10
(overridable per binding via an interpreter setting).

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Native SFTP client (asyncssh/paramiko) | Reimplements the filesystem: `fs_grep` means downloading candidates, atomic replace/flock/CAS need weaker analogues, largest scope, new dependency, worst performance. |
| Laptop-side mount (sshfs / macFUSE / rclone mount) | macOS 26 FUSE friction (system-extension approvals), slow remote grep/glob, no first-class status — and a stale mount on a dead server can hang the app, the exact opposite of the availability requirement. |
| Install chatbook (or a persistent agent) on the server | Explicitly excluded by the user's requirement: nothing runs or persists on the server. |
| rsync/git mirror + sync engine | Not live (remote changes invisible until sync), needs conflict resolution, mutations need push — surprising semantics for a workspace root. |
| Reuse `REMOTE_RUNTIME` for the kind | Reserved for agent-runtime-remote semantics; conflates filesystem bindings with runtime bindings. |
| Per-call `ssh -G` canonicalization at dispatch | Subprocess spawn in the send path violates the hot-path rule; resolved identity is stored at add/edit instead. |
| Transparent mid-run read retry | The executor's failure-triggered master restart covers stale masters between calls; retrying the failed op itself is still excluded (write idempotency unknowable; long ops double the wait). Revisit only with measurements. |
| Change-review row capture in v1 | `binding_added` schedules per-root review setup; with no finalization actions for remote rows, captured rows are unactionable noise. |

## Links

- [Design spec](../../Docs/superpowers/specs/2026-09-24-ssh-remote-workspace-bindings-design.md)
- [ADR-005](005-console-workspace-server-readiness.md)
- [ADR-028](028-settings-workspaces-category-and-folder-roots.md)
- [ADR-032](032-local-agent-tool-permission-boundary.md)
- [ADR-069](069-console-project-instruction-local-state-and-preflight.md)
- [ADR-101](101-one-shot-pinned-workspace-tool-execution.md)
- [ADR-102](102-console-run-admitted-local-path-authority.md)
- [ADR-174](174-workspace-binding-exclusions.md)
- [ADR-175](175-one-strict-json-acceptance-contract.md)
