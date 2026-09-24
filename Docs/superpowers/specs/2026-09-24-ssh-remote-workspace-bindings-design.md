# SSH Remote Workspace Bindings — Design

Date: 2026-09-24
Status: Draft (pending review)
Related ADR: ADR-181 (to be created with this design; confirm the next free
number at planning — 179 and 180 are claimed). Amends no ADR; interacts with
ADR-005, ADR-028, ADR-032, ADR-069, ADR-101/102, ADR-173.

## Problem

The Console's named Workspaces can bind **local** folders only
(`RuntimeBindingKind.LOCAL_FILESYSTEM` is the single implemented kind).
A user running chatbook on a laptop who wants their workspace root — plus
AGENTS.md project instructions — to live on a remote Linux server has no
supported path today: mounting (sshfs/FUSE) is fragile on macOS 26 and a
dead server can hang a stale mount, and nothing in the app speaks SSH.

The requirement is a first-class remote binding: chatbook stays on the
laptop, connects via SSH, exposes a folder on the remote server as a
workspace binding alongside ordinary local folder bindings, and degrades
gracefully — an unreachable remote never blocks use of the workspace's
local bindings or scratch.

## Requirements

- A user can add an SSH folder binding to a **named** workspace (never the
  default workspace) from Settings → Workspaces: SSH target, absolute remote
  path, read-only/read-write.
- All `fs_*` tools (read/write/edit/list/glob/grep) work against the remote
  binding with the same semantics as local bindings, including the ro/rw
  toggle and CAS read-before-write stamps. Tools execute **server-side**.
- The remote binding can be selected as the Console working folder, so
  AGENTS.md / AGENTS.override.md load from it under all existing ADR-068/069
  rules (byte caps, untrusted content, activation ledger, first-use consent).
- **Availability guarantee** (the core requirement, with a named regression
  test): an unreachable, auth-failing, or timing-out remote binding is
  excluded from the run's admitted roots; sends still compose; local
  bindings and scratch keep working; the context note says the remote root
  was excluded and why.
- Nothing is installed, persisted, or left behind on the server: each tool
  call runs a transient python3 process whose code arrives over the SSH
  session's stdin and vanishes with it. (ControlMaster sockets live on the
  laptop, not the server.)
- Chatbook stores no SSH credentials. Authentication is fully delegated to
  the user's ssh-agent / `~/.ssh/config` (BatchMode; never prompt).
- Server prerequisites: sshd with exec, and `python3` ≥ 3.10 (overridable
  per binding via an interpreter setting for servers with versioned
  interpreters).

## Non-goals (v1)

- Native SFTP transport (no remote execution), FUSE/sshfs mount management,
  or rsync/mirror-sync semantics.
- `git_*` tools on remote bindings (call-time typed error; stay available
  for local aliases on mixed runs).
- Change-review **finalization actions** (revert/keep/apply) on remote rows
  — rows may be captured (see Decision), but finalization is deferred.
- RAG indexing of remote roots; background status polling; remote-side
  bundle caching; credential storage or management of any kind.
- Accepting `ssh://…` URIs as path arguments (no second path grammar —
  ADR-069 rejected that ambiguity class for `fs_glob` already).
- Multi-hop reachability beyond what the user's ssh config already
  expresses (ProxyJump etc. work for free; chatbook manages no tunnels).
- ControlMaster on Windows-as-local-host (ssh.exe lacks it; degrades to
  per-call connections via the same kill switch).

## Decision summary

Add a `SSH_FILESYSTEM` runtime binding kind whose tool access reuses the
existing one-shot pinned-worker architecture over SSH stdio instead of a
local fork. Per tool call, chatbook spawns
`ssh <target> [mux options] -- python3 -I -c '<bootstrap>'`, writes a
**zlib-compressed, stdlib-only worker bundle** followed by the standard
length-prefixed `WorkspaceToolRequest` frame to the subprocess's stdin, and
reads the framed `WorkspaceToolResponse` from stdout (magic-prefixed to
survive noisy remote shell rc files). The worker pins the remote root
(`O_DIRECTORY|O_NOFOLLOW`, `fchdir`, `(st_dev, st_ino)` identity) exactly as
the local worker does — from the worker's perspective the remote root *is* a
local filesystem, so pinning, confinement, atomic writes, and CAS identity
checks all keep their current semantics and run server-side.

Warm connections come from OpenSSH ControlMaster multiplexing that chatbook
manages on the laptop. Availability is the existing per-binding status
model: a cheap `ping` op maps to `READY` / `BLOCKED` (unreachable) /
`MISSING` (reachable, root path gone), status is cached and refreshed
lazily, and **run composition reads cache only — the dispatch hot path
never touches the network or spawns a subprocess**. Degradation excludes the
binding from the run exactly like a missing local folder.

## Binding model & data layer

- New `RuntimeBindingKind.SSH_FILESYSTEM = "ssh-filesystem"`. The
  `workspace_runtime_bindings` table already stores `binding_kind` as text —
  **no schema migration**. An SSH binding is a row with a different kind,
  locator, and metadata. Bindings remain device-local (the workspaces DB is
  local), and `save_runtime_binding`'s default-workspace rejection applies
  unchanged.
- **Locator**: `ssh://[user@]host[:port]/absolute/path`. The host may be an
  `~/.ssh/config` alias; user/port/jumps resolve through the user's config.
- **Canonical locator** (fingerprint source): computed at add/edit time via
  `ssh -G <target>` — the config-**expanded symbolic** identity (HostName as
  written in config, port, user, host lowercased; never DNS-resolved to an
  IP) plus the normalized path. Stored with the binding; dispatch compares
  stored fingerprint vs. binding row as a pure local string compare.
  Benefits: cosmetic alias edits don't trip re-consent; a config change that
  actually moves the destination does (ADR-069 retarget semantics);
  same-host nesting/duplicate checks become correct. Fresh `ssh -G`
  re-resolution happens only on add, edit, manual refresh, and workspace
  open (async) — never in the send path.
- **Registry**: `add_ssh_binding(workspace_id, target, path, *,
  allow_write=False)` mirroring `add_folder_binding` — absolute POSIX path
  required (no `~`), nesting/overlap checks only between SSH bindings on
  the same resolved host, never across local/remote. `metadata` carries
  `{"access": "rw"|"ro", "python": "python3"}`. **Charset validation** on
  user/host/port/interpreter (conservative `[A-Za-z0-9_./-]`-style classes;
  no spaces, quotes, or shell metacharacters) because these values cross a
  remote shell command line. `scrub_secret_metadata` discipline unchanged:
  no credentials in metadata, ever.
- **Status semantics** (recomputed lazily like local bindings; never
  persisted as durable truth — reason strings live in the ephemeral layer):

  | Probe (ping op) result | Status | Effect |
  |---|---|---|
  | Root stat returned | `READY` | admitted as a run root |
  | Connection/auth/timeout, python missing, pin refused | `BLOCKED` | excluded from run roots; reason surfaced ("unreachable", "host lacks python3", "root rejected: symlink") |
  | Connects, root path absent | `MISSING` | excluded; "missing on host" |

- **Selected working folder + failures**: fingerprint mismatch or `MISSING`
  → explicit re-selection (ADR-069 consistency). `BLOCKED` (transient
  network) → the dispatch degrades to scratch + local bindings with a
  visible warning and an easy retry; **no forced re-selection**, because a
  flaky link is not a retarget.
- **Status cache learning**: op outcomes feed the cache — a
  transport-classified failure flips the binding to `BLOCKED` immediately;
  any successful op or probe flips it back. Degradation compounds
  send-to-send without polling.

## Transport & executor

- `RemoteWorkspaceToolExecutor` implements the same
  `execute(tool, args, intent)` surface as `WorkspaceToolExecutor`, slotting
  into `RunAdmittedWorkspaceRoot.workspace_executor` so provider dispatch is
  transport-agnostic.
- **Spawn** (per call):
  `ssh <resolved-target> -o BatchMode=yes -o ConnectTimeout=3 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -o ControlMaster=auto -o ControlPath=<app-state>/ssh-cm/%C -o ControlPersist=<cfg> -- <python> -I -c '<bootstrap>'`
  — then stdin carries: N bytes of zlib-compressed bundle, then the framed
  request. The bootstrap is
  `import sys,zlib;exec(compile(zlib.decompress(sys.stdin.buffer.read(N)),"b","exec"))`.
  N is embedded as a literal by the executor.
- **Shell safety**: the bootstrap charset is restricted (letters, digits,
  `. , ( ) " ; _`) so it stays literal under sh/bash/zsh/fish/csh quoting;
  enforced by test. Target/interpreter charsets validated at add time.
  Bundle bytes travel via stdin — invisible to `ps` on the server and never
  shell-parsed. `ServerAlive*` options make a laptop sleep/wake stale master
  self-reap in ~30s; the per-call hard deadline is the backstop;
  `ControlMaster=auto` recovers with a fresh master.
- **Stdout contamination**: the worker's response begins with a fixed magic
  byte sequence; the executor skips leading stdout bytes until the magic
  (garbage cap ~4KB → typed "remote shell emits stdout noise" error with a
  hint to guard `.bashrc`). Debian/Ubuntu source `.bashrc` for
  `ssh host cmd`; unguarded rc files otherwise corrupt framing.
- **Bundle**: committed, build-time-generated single file
  (`Tools/remote_worker_bundle.py`) assembled from the existing protocol
  framing, root pinning, `local_tool_impls` core, dispatch, and
  sensitive-path modules by `Tools/build_remote_worker_bundle.py`.
  Constraints: **stdlib-only**, python ≥ 3.10 floor, reads frames only from
  the same buffered stdin stream the bootstrap used (build script emits the
  IO entry point to take the stream as an argument). Compression ~4:1 puts
  per-call overhead at ~40KB. A **drift-guard test** rebuilds and diffs, and
  scans for 3.11+ stdlib imports (`tomllib`, `typing.Self`,
  `asyncio.TaskGroup`, …) so the floor is enforced, not documented. Each
  call logs the bundle hash for audit.
- **New `ping` op**: returns root stat `(st_dev, st_ino, st_mode)`,
  canonical remote path, remote python version, and bundle-hash echo — one
  multiplexed round trip powering both the status probe and authority
  capture.
- **Timeouts**: `ConnectTimeout` bounds TCP only; every call and probe also
  has a hard process-side deadline with process-group kill (the hooks
  system's established pattern). Config `call_timeout_s` default 60s.
- **Concurrency**: semaphore keyed by **resolved host / ControlPath
  identity** (not per binding) — sshd's `MaxSessions` is per connection and
  multiple bindings on one host share one master. Default cap 8, config
  `max_concurrent_calls`; the probe counts against the same cap.
- **Failure taxonomy** (feeds BLOCKED reasons): ssh exit 255 → unreachable
  or auth; remote exit 127 → interpreter missing; deadline kill → timeout;
  mux-protocol error → stale master; magic-cap exceeded → noisy shell.
  Worker-side exceptions return framed typed errors rather than stream EOF.
- **BatchMode limitations, documented**: agent-confirmed keys (`ssh-add -c`)
  fail rather than prompt; unknown host keys fail closed (correct default);
  channel integrity for the bundle inherits the user's
  `StrictHostKeyChecking` policy — chatbook does not override it.
- Startup checks for a local `ssh` binary; the feature disables with a
  clear message if absent.
- **Future hook (explicitly not v1)**: a content-hash bundle cache in the
  remote `$TMPDIR` would cut per-call payload to ~50 bytes; the protocol
  shape (size-prefixed stdin payload) makes it a drop-in later.

## Run composition & tool parity

- `capture_run_admitted_workspace_roots` admits SSH bindings as
  `RunAdmittedWorkspaceRoot`s: own `root_alias`, `allow_write` from
  `metadata["access"]`, `RemoteWorkspaceToolExecutor` in the executor slot.
  Client-side path normalization is lexical (reject `..` escapes, hidden
  paths); the worker re-validates fully server-side — same defense-in-depth
  as today.
- **Alias uniqueness** across all admitted roots of a run is enforced at
  composition with deterministic disambiguation (`proj`, `proj-2`);
  ambiguous alias routing is impossible by construction.
- **Split authority** (same fail-closed posture, no added latency):
  client-side per call — registry row re-read, fingerprint match, access
  match, status not BLOCKED (catches retarget/permission changes); server
  side per call — the worker's root pin re-validates `(st_dev, st_ino)`
  before executing anything, catching symlink/mount drift inside the same
  round trip.
- **Parity matrix**:

  | Capability | SSH binding in v1 |
  |---|---|
  | fs_list / fs_read / fs_glob / fs_grep | full parity, server-side |
  | fs_write / fs_edit / fs_patch | full parity incl. CAS stamps (ledger stores worker-reported stat tuples as opaque identity; re-check runs in-worker) |
  | Read-only binding | mutating specs not advertised (existing `any_write` preflight) |
  | Result spill to local scratch | unchanged |
  | git_* | call-time typed error "git tools not yet supported on SSH bindings"; still advertised for local aliases on mixed runs |
  | Change review | rows captured for remote writes (diffs ride the write response if planning confirms `test_write_file_diff_capture.py`'s path; metadata-only rows otherwise); finalization actions on remote rows deferred |
  | Sensitive-path denylist + ADR-173 user exclusions | static denylist ships in the bundle; per-binding serialized `sensitive_exclusions` ride the request unchanged |

- **Model-facing surface**: remote roots appear only as
  `ssh://alias/path` URIs in the context note alongside an explicit
  alias→URI mapping (`proj → ssh://devbox/home/me/proj, remote, rw`), with
  a note that remote roots are reachable via `fs_*` tools only. Bare URIs
  passed as path arguments are rejected with a message that teaches the
  correct convention (`use root_alias "proj" with a relative path`) —
  self-correcting in one turn, no second grammar.
- **Family B** (`read_file`/`write_file`/`list_directory`/`edit_file`):
  local-only in v1; URIs don't parse as local paths so `validate_path`
  rejects them cleanly — a remote path can never alias a local file.
- **AGENTS.md from remote**: `_validate_project_instruction_binding`
  accepts `ssh-filesystem` + `READY`; the resolver's reads route through the
  executor via a bounded-read op (bytes + stat in one call); nested
  discovery's directory chain uses worker stats for `BindingRootIdentity`.
  Everything above the IO layer — caps, untrusted-content rules, ledger,
  lazy activation, first-use consent — unchanged. Instruction-loading
  failures on an unreachable remote follow ADR-069's prep-failure posture:
  content-free warning, proceed.
- **Degraded semantics**: at composition, BLOCKED/MISSING bindings are
  excluded and the note says so; mid-run drops return a typed
  transport-error tool result and the run continues on local roots.
  **Retry rule**: one transparent retry, **read ops only**, only for
  connection-level fast failures classified before any response bytes
  (dead master, connection refused). Never for writes (idempotency
  unknowable) and never after a deadline kill (retrying a 60s grep doubles
  the wait).
- **Hot-path rule (load-bearing)**: run composition reads cached status
  only. Stale-READY → mid-run typed error handles it; stale-BLOCKED →
  binding stays excluded (which is the availability requirement). Probes
  refresh asynchronously (workspace open, binding list, manual refresh,
  op-outcome learning). The only synchronous probe is add-binding, where
  the user is already waiting — and even that is advisory: a failing probe
  lets you save the binding with the reason shown, so setup works offline.

## UX surfaces & configuration

- **Settings → Workspaces**: "Add SSH folder…" inline form (target, path,
  ro/rw, interpreter override), advisory add-time probe, per-binding status
  column gains `unreachable` / `missing on host` states refreshed async
  (never blocking UI on a probe).
- **Console working-folder picker** and the **Alt+W workspace switcher**
  list SSH bindings with status chips; selection follows the existing
  ADR-069 consent flow unchanged.
- **Context note**: alias→URI mapping, `[ssh]` tags, explicit
  "remote binding unreachable — excluded this run" degradation line.
- **Config** (`[console_ssh]` in config.toml): `control_persist` (`"10m"`),
  `enable_multiplexing` (`true`; doubles as the Windows kill switch),
  `connect_timeout_s` (3), `call_timeout_s` (60), `max_concurrent_calls`
  (8). No master enable flag — the feature is dormant unless an SSH binding
  exists.

## Security posture (ADR-181 content)

- Auth fully delegated (BatchMode; agent/config only; no secrets in app,
  keyring, or metadata).
- The only code executed remotely is chatbook-shipped: stdlib-only,
  commit-time-built, drift-guarded, hash-logged per call, shell-charset
  validated end to end.
- Remote AGENTS.md is untrusted repo content under every ADR-068/069 rule;
  it never grants permission. Permission gates, hooks (deny-only
  PreToolUse), and approval effects apply to remote ops identically —
  `MUTATES_LOCAL` semantically means "mutates workspace storage" (rename
  deferred).
- Sensitive-path denylist enforced server-side in the bundle; ADR-173 user
  exclusions ride the serialized request.
- Remote file content transits the user's own SSH channel — same trust as
  their manual ssh. Host-key policy is the user's ssh config decision;
  BatchMode's unknown-host fail-closed is the default posture.
- The remote host itself is user-trusted infrastructure (their server,
  their ssh config); the worker runs as their SSH user with root-pin
  confinement equivalent to the local worker's.

## Testing

1. **Unit**: locator parse/canonicalize (aliases, ports, IPv6 brackets),
  charset validators, status mapping, bundle drift-guard + 3.10-compat
  scan, bootstrap shell-safety charset, compression round-trip.
2. **Loopback integration (workhorse)**: execute the bundle via
  `python3 -I -c` + stdin frames against a temp dir, no ssh — framing,
  magic-prefix skip, pinning, all ops, CAS stamps, single-stream
  discipline.
3. **Fake-ssh**: stub `ssh` script simulating unreachable / exit-127 /
  stdout noise / hang-past-deadline / unknown-host — proves the failure
  taxonomy and deadline kills.
4. **Composition & degradation**, including the **named availability
  regression test**: *workspace with remote BLOCKED + local READY → send
  composes, local tools advertised, remote excluded, note mentions it*.
5. **Opt-in marked test** with `ssh localhost` where key auth exists;
  plus a genuine live run against a real server before any task is marked
  Done (`lessons-live-verification.md`).
6. Existing suites stay green: `test_workspace_file_roots`,
  `test_local_tool_provider`, `test_project_instruction_resolver`,
  `test_console_project_instructions`, `test_workspace_tool_executor`,
  `test_workspace_tool_protocol`, `test_local_tool_impls`, and neighbors.

## Rollout (four independently testable phases)

1. **Foundation**: binding kind, registry methods, locator
  canonicalization + charset validation, bundle build + drift guard,
  loopback executor — zero network code.
2. **Transport**: ssh spawn, mux management, magic/deadline/retry,
  probe + status cache (with op-outcome learning), failure taxonomy.
3. **Run integration**: admitted roots with remote executors, alias
  uniqueness, degraded composition, AGENTS.md remote reads, change-review
   row capture.
4. **Surface**: Settings form, picker/switcher entries, context note,
  `[console_ssh]` config, user-guide docs, repo AGENTS.md note.

ADR-181 is authored and committed **before Phase 1 code** (repo rule), and
linked from the backlog task, plan, and implementation notes.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Native SFTP client (asyncssh/paramiko) | Reimplements the filesystem: `fs_grep` means downloading candidates, atomic replace/flock/CAS need weaker analogues, largest scope, new dependency, worst performance. |
| Laptop-side mount (sshfs / macFUSE / rclone mount) | macOS 26 FUSE friction (system-extension approvals), slow remote grep/glob, no first-class status — and a stale mount on a dead server can hang the app, the exact opposite of the availability requirement. |
| Install chatbook (or a persistent agent) on the server | Explicitly excluded by the user's requirement: nothing runs or persists on the server. |
| rsync/git mirror + sync engine | Not live (remote changes invisible until sync), needs conflict resolution, mutations need push — surprising semantics for a workspace root. |
| Per-call `ssh -G` canonicalization at dispatch | Subprocess spawn in the send path violates the hot-path rule; resolved identity is stored at add/edit instead. |
