# ADR-181: SSH remote workspace bindings

Status: Proposed
Date: 2026-09-24
Related Task: [TASK-33009](../tasks/task-33009%20-%20SSH-remote-workspace-bindings.md)
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

  **Limit, accepted:** `(st_dev, st_ino)` names a directory, not its
  history. A filesystem that hands a freed inode number straight back
  (observed on ext4 on the 2026-09-27 live host: `rm -rf root && mkdir
  root` recreated the same pair) makes a deleted-and-recreated root
  indistinguishable from the original, so it is served without a
  `STALE_IDENTITY` stop. The pin still defeats what it exists for — a
  symlink swap, a root replaced by a *different* live directory, and
  escapes out of the root — and this is the same limit local bindings have.
  A stronger identity (ext4's `i_generation` via an ioctl) is
  filesystem-specific and not stdlib-portable, so it is not used.
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
- [Session worker and bundle cache design (2026-09-27 amendment)](../../Docs/superpowers/specs/2026-09-27-ssh-session-worker-and-bundle-cache-design.md)
- [ADR-005](005-console-workspace-server-readiness.md)
- [ADR-028](028-settings-workspaces-category-and-folder-roots.md)
- [ADR-032](032-local-agent-tool-permission-boundary.md)
- [ADR-069](069-console-project-instruction-local-state-and-preflight.md)
- [ADR-101](101-one-shot-pinned-workspace-tool-execution.md)
- [ADR-102](102-console-run-admitted-local-path-authority.md)
- [ADR-174](174-workspace-binding-exclusions.md)
- [ADR-175](175-one-strict-json-acceptance-contract.md)

## Amendment 2026-09-27: session worker and bundle cache

Related Task: [TASK-33202](../tasks/task-33202%20-%20SSH-session-worker-and-host-bundle-cache.md)
Design: [SSH Session Worker and Bundle Cache — Design](../../Docs/superpowers/specs/2026-09-27-ssh-session-worker-and-bundle-cache-design.md).
Motivation: a warm `fs_*` call cost ~0.7 s on a LAN host, almost all of it
round trips (a new ssh channel per call plus shipping the bundle). The
target is a warm call of about one round trip.

- **One fork-server session per binding per Console run.** The first remote
  call of a run starts one long-lived `ssh … <python> -I -c '<bootstrap>'`
  channel over the existing ControlMaster; the host side is a
  single-threaded parent (`serve`) that forks one child per request. The
  per-operation rule of ADR-101 still holds: **each operation still runs in
  a fresh child process** with its own root pin, remote-home denylist,
  serialized exclusions, and two-tier watchdog (Timer exit 75 +
  `signal.alarm`), armed from that request's own remaining budget. Only the
  spawner changes, from `sshd` to the per-session parent. Children get
  stdin from `/dev/null`, stdout on a private pipe back to the parent, and
  keep fd 2 (the channel's stderr, where the watchdog marker goes); every
  other inherited descriptor is closed, so a lingering child can neither
  hold the channel's stdin/stdout open nor write unframed bytes into the
  framed stream. The run's tool executor and its AGENTS.md reader share the
  one session; sessions are never shared across bindings, runs, or hosts.
- **What the parent holds.** The drift-guarded bundle and the run's request
  frames. It never executes request logic, caps its live children, and
  exits on stdin EOF or after `session_idle_s` idle (no queued requests, no
  live children), killing any remaining children. It drops its reference
  to each request after forking, but **residual request bytes in parent
  memory may be inherited by later children of the same session**. That
  stays inside one binding/run trust scope (same root, same permission
  set), so it is not a cross-boundary leak; this ADR does not claim the
  memory is scrubbed.
- **"Nothing persisted on the server" becomes "nothing outside the user's
  own runtime directory".** A ~1 KB stage-1 loader replaces the bundle on
  the session's stdin and may cache the compressed bundle at
  `$XDG_RUNTIME_DIR/tldw-worker/<sha256>` (tmpfs, cleared at logout). The
  entry is used only when `$XDG_RUNTIME_DIR` is set, both it and
  `tldw-worker/` are directories owned by the current uid with no group or
  other access, the entry is a regular file (opened `O_NOFOLLOW`) owned by
  the current uid with mode 0600, and its sha256 matches its name; anything
  else is a miss. Cache writes are best-effort: a failed write still runs
  the verified bundle from memory.
  On a miss the laptop sends the bundle (capped at 8 MiB before reading),
  the loader verifies its sha256, writes it atomically (temp file +
  `os.replace`, 0600), and deletes other entries in that directory. With no
  usable private runtime directory (e.g. macOS hosts) nothing is cached —
  never a fallback to `/tmp` or `~/.cache`. The loader then reports the
  bundle's own stamp (`READY <stamp>`); the **laptop** compares it against
  its expected stamp and refuses a mismatch (a protocol-class start
  failure).
- **Cache threat model, stated honestly.** The checks protect against
  corruption, partial writes, and stale versions. They do **not** defend
  against another process running as the same user: such a process can
  already run code as that user, so a same-uid attacker is out of scope.
- **Destination check before every session start.** The `ssh -G`
  re-resolution runs before each session start; a mismatch starts no
  session and records BLOCKED (`destination_changed`). A live session's
  connection is fixed to the verified host, so re-capturing identity inside
  it after `STALE_IDENTITY` stays safe.
- **Failure classification (same taxonomy, same status-cache rules).** A
  session reports the same result shape as a one-shot call, so admitted →
  typed op error with status unchanged still holds. Session-specific rows:
  - A request whose deadline passes with no admitted marker while the
    session is still alive is `OP_TIMEOUT` (status-preserving): a live
    channel proves reachability, so a slow host queue never flips the
    binding to BLOCKED.
  - A session the laptop ended (stuck-parent kill, decode-error kill, run
    close) fails its unadmitted calls as `REMOTE_OP_FAILED` (status-preserving).
    Only a natural death (EOF, ssh exited on its own) classifies unadmitted
    calls through the transport taxonomy, using the session's real ssh exit code.
  - A call whose session the laptop closed while it was healthy (idle reap,
    run end, app exit) **before the call's request was sent** is not failed:
    nothing ran, so it asks the registry again and gets a fresh session, or
    the one-shot path once the run or the app has ended. This covers a close
    that lands before the call registered (TASK-33401) and one that lands
    after it registered but before any REQUEST byte was written
    (TASK-33421); a request that was written is never retried.
  - A natural death with ssh exit code 0 (host idle-exit, clean EOF) is a
    benign session end, never a transport failure: its unadmitted calls get
    `REMOTE_OP_FAILED` and the next call starts a fresh session without
    spending the run's single restart. The laptop reaps an idle session at
    `session_idle_s`; the host idles out later (`session_idle_s` + grace),
    so the laptop retires a session before the host would.
  - Session start failing transport-class (no marker; 255, connect timeout,
    127, 76) is the call's result — recorded as today, no one-shot retry.
    Protocol-class start failures (loader crash, hash or stamp mismatch,
    bad handshake) switch this run's binding to the one-shot path. A second
    mid-run death does the same.
  - A root pin failure is `STALE_IDENTITY` as before; the session stays up.
- **Run scoping.** The session key is unique per Console run invocation
  (message id + a fresh uuid), so regenerate and recovery re-runs never
  share a session. Run end closes the key's sessions and **tombstones** the
  key (last 1024 kept): a later call under a closed key — a surviving
  sub-agent, a Stop straggler — uses one-shot calls rather than reopening
  an unowned session. App exit closes every session.
- **Kill switch.** `[console_ssh] session_worker = false` restores pure
  one-shot behaviour exactly; `session_idle_s` (default 60) sets the idle
  close; `bundle_cache = false` disables the host cache (the bundle is then
  sent on every session start).
- **Measured (2026-09-27, LAN Wi-Fi host, Python 3.13, two runs).** Warm
  `fs_read` median 13.3 ms / p90 28.5 ms (n=420; same-window ICMP ping
  median 9.4 ms) and 14.8 ms / p90 127.3 ms (n=317; ping median 32.4 ms),
  against a target of spike echo floor 7.51 ms + 15 ms. A cache-hit session
  start took 110–422 ms (medians 192 ms and 367 ms over 5 starts per run),
  dominated by opening the ssh channel over the ControlMaster; a cache miss
  took 498 ms and 591 ms. Both are paid once per run per binding.
- **Follow-ups (2026-09-28, TASK-33400–33408).**
  - Callers queued behind a session start that fails transport-class all
    get that failure; nobody starts again against the same dead host. A
    later call still tries a fresh session.
  - A session handshake is part of its call: it gives up at the call's
    remaining budget + grace, capped at 30 s. Both Console callers build
    their executors with the default 300 s tool budget, so in practice the
    30 s cap is the deadline. Before the host answers, a silent host is
    classified like a one-shot handshake (transport-class). Once the host
    has answered (NEED or READY), the deadline is never transport-class
    (TASK-33420). At the 30 s cap the start is protocol-class: the binding
    status is unchanged and the run falls back to one-shot calls. That
    catches a stuck loader, and equally a cache-miss bundle upload still
    making progress slower than about 2.5 KB/s (74,233 B against the 30 s):
    the write deadline is absolute, not an inactivity window. Only when the
    call's budget + grace is under the 30 s cap (a short budget, not the
    default) does running out of it (mid-upload, or waiting for a READY
    that never comes) fail just that call as `OP_TIMEOUT`
    (status-preserving); that failure is not shared with callers queued
    behind the start, and the next call tries a session again. An ssh exit
    after the answer is still classified by its exit code, like any natural
    death. Trade-off (the budget case only; at the cap the run is one-shot
    and waiters go one-shot too): callers queued behind a start that ran
    out of budget each run their own start in turn with their full budget,
    so the k-th waits up to about (k+1) × (budget + grace), and that queue
    is bounded by `max_concurrent_calls`; handing a waiter only its
    remaining budget would push a nearly spent one into a pre-answer stall
    (UNREACHABLE, so BLOCKED), which is worse.
  - A mux failure (`MUX_ERROR`) at session start runs that call one-shot
    over the restarted master and fails nothing; the next call tries a
    session again. A second one in the same run switches the binding to
    one-shot for the run.
  - A session's death is classified from the **last** 64 KiB of its ssh
    stderr, where the reason is.
  - A request the host cannot fork or pipe for gets `STATUS(71)`
    (`EX_OSERR`) and fails alone as `REMOTE_OP_FAILED` (status-preserving:
    the live session proves reachability); the session carries on.
  - Idle reaping closes sessions on a background thread, never on the
    calling tool call; run end and app exit close a run's (or every)
    session in parallel and wait at most 5 s.
  - After app exit the master manager starts no new ControlMaster and runs
    no health check; a straggler call connects directly.
  - A fast operation's admitted marker, result and STATUS leave the host
    in one write; the parent holds a running child's output at most 10 ms
    before writing it. Measured: live host over Wi-Fi, 2026-09-28: warm median 11.5-13.8 ms coalesced vs 16.6-17.2 ms without (3.3, 4.7, 5.1 ms lower in three interleaved pairs).
