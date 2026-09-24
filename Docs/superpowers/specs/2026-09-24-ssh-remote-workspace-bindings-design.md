# SSH Remote Workspace Bindings — Design

Date: 2026-09-24
Status: Draft (revision 3 after second code-level review)
Related ADR: ADR-181 (to be created with this design). Numbering caution: ADR
files 005/028/032/068/069/102 each have duplicates across branches and 178 is
duplicated on dev; 179/180 exist only on this branch. **Re-check 181 against
origin/dev at merge time.** Interacts with (by filename):
`005-console-workspace-server-readiness`,
`028-settings-workspaces-category-and-folder-roots`,
`032-local-agent-tool-permission-boundary`,
`069-console-project-instruction-local-state-and-preflight`,
`101/102` (pinned-worker seam), `174` (workspace binding exclusions — not
173, which is the UTC-timestamp decision on dev).

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
  toggle and CAS read-before-write stamps. Tools execute **server-side** —
  meaning, concretely, that request building, path validation, hashing, and
  exclusion/sensitive-path checks for a remote root never consult the
  laptop's filesystem (see "Remote roots never touch the laptop's disk").
- The remote binding can be selected as the Console working folder, so
  AGENTS.md / AGENTS.override.md load from it under all existing ADR-068/069
  rules (byte caps, untrusted content, activation ledger, first-use consent).
- **Availability guarantee, both directions** (the core requirement, with
  named regression tests): (1) an unreachable, auth-failing, or
  handshake-timing-out remote binding is excluded from the run's admitted
  roots; sends still compose; local bindings and scratch keep working; the
  context note says the remote root was excluded and why. (2) An operation
  that runs past its deadline is a typed tool error that **never changes
  the binding's status** — the next send admits the root again.
- Nothing is installed, persisted, or left behind on the server — including
  **no orphaned processes after a timed-out call** (worker-side watchdog) and
  **no orphaned temp files after an aborted atomic write** (the watchdog
  unlinks the worker's registered temp files before it exits — see
  "Worker-side watchdog").
  Each tool call runs a transient python3 process whose code arrives over
  the SSH session's stdin and vanishes with it. (ControlMaster sockets live
  on the laptop, not the server.)
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
- **Change review on remote bindings — rows included, not just
  finalization**: `add_folder_binding`-style `binding_added` events schedule
  per-root change-review setup work; since no finalization actions exist for
  remote rows in v1, capturing rows nobody can act on is noise. SSH bindings
  fire no change-review setup and capture nothing in v1.
- RAG indexing of remote roots; background status polling; remote-side
  bundle caching; credential storage or management of any kind.
- Accepting `ssh://…` URIs as path arguments (no second path grammar —
  ADR-069 rejected that ambiguity class for `fs_glob` already).
- Multi-hop reachability beyond what the user's ssh config already
  expresses (ProxyJump etc. work for free; chatbook manages no tunnels).
- ControlMaster on Windows-as-local-host (ssh.exe lacks it; degrades to
  per-call connections via the same kill switch).
- Human-readable root aliases (alias stays binding_id-derived — see
  "Model-facing surface").
- Transparent mid-run read retries (executor-owned master check-and-restart
  plus typed errors suffice; retrying the failed op itself stays excluded —
  revisit only with measurements).

## Decision summary

Add a `SSH_FILESYSTEM` runtime binding kind whose tool access reuses the
existing one-shot pinned-worker architecture over SSH stdio instead of a
local `Popen`. Per tool call, chatbook spawns
`ssh [opts] [-p <port>] [-l <user>] -- <host> <interpreter> -I -c '<bootstrap>'`
(argv built from the **parsed locator components** — see "Shell safety"),
writes a
**zlib-compressed, stdlib-only worker bundle** followed by the standard
length-prefixed `WorkspaceToolRequest` frame to the subprocess's stdin, and
reads the framed, magic-prefixed `WorkspaceToolResponse` from stdout. The
worker pins the remote root (`O_DIRECTORY|O_NOFOLLOW`, `fchdir`,
`(st_dev, st_ino)` identity) exactly as the local worker does — from the
worker's perspective the remote root *is* a local filesystem, so pinning,
confinement, atomic writes, and CAS identity checks all keep their current
semantics and run server-side.

Two prerequisites make that true rather than aspirational, and each is a
budgeted phase, not a detail:

1. **The worker's import closure is stdlib-only** (today
   `workspace_tool_protocol.py` imports pydantic; `sensitive_paths.py` and
   `path_validation.py` import loguru, config, Skills_Interop, RAG_Search,
   Metrics). Phase 0 fixes this at the *local* worker so the bundle really
   is the same code — **without weakening the parent side**: the laptop
   keeps pydantic models for request/response validation (the repo's
   one-strict-JSON acceptance contract — TASK-32855's shared payload
   validators — is a parent-side property), while the
   bundle ships a stdlib decoder whose **accept/reject behavior is
   provably identical** — a conformance test feeds both decoders the same
   corpus of valid and malformed frames and requires identical outcomes
   (see Testing).
2. **Remote roots get their own type** (`LocalRoot | RemoteRoot`), because
   roughly a dozen call sites today do laptop-disk work off
   `RunAdmittedWorkspaceRoot.root: Path` — request building, CAS hashing,
   exclusions, admission checks. A distinct type makes every un-migrated
   site fail loudly instead of quietly reading the laptop's copy of a path
   like `/tmp/x` or `/var/www/site`.

Warm connections come from an **explicitly-managed** OpenSSH ControlMaster
(`ssh -MNf` in its own session; per-call clients with `ControlMaster=no`)
so a timed-out call's process-group kill cannot take down the shared
connection. Availability is the existing per-binding status model driven by
a cheap `ping` op → `READY` / `BLOCKED` / `MISSING`, cached in memory (never
read from the stored status column at admission), refreshed lazily, and
learned from **transport-classified** op outcomes only (a slow op that times
out is a typed error, never a status flip). **Run composition reads cache
only — the dispatch hot path never touches the network or spawns a
subprocess.** Degradation
excludes the binding from the run exactly like a missing local folder.

## Binding model & data layer

- New `RuntimeBindingKind.SSH_FILESYSTEM = "ssh-filesystem"`. A new kind
  rather than the already-reserved `REMOTE_RUNTIME` because that value is
  reserved for "the agent *runtime* executes remotely" semantics (future
  VM/container runtimes); an SSH binding is a *filesystem* binding whose
  agent-side semantics are identical to `LOCAL_FILESYSTEM` (same tools,
  same admission, same consent flows) — reusing it would conflate the two.
  The `workspace_runtime_bindings` table already stores `binding_kind` as
  text — **no schema migration**. Bindings remain device-local, and
  `save_runtime_binding`'s default-workspace rejection applies unchanged.
- **Locator**: `ssh://[user@]host[:port]/absolute/path`. The host may be an
  `~/.ssh/config` alias; user/port/jumps resolve through the user's config.
- **Canonical locator** (fingerprint source): computed at add/edit time via
  `ssh -G` (argv built from the same parsed locator components as call
  spawns — `-p <port> -l <user> -- <host>`, brackets stripped) — the config-**expanded symbolic** identity (HostName
  as written in config, port, user, host lowercased; never DNS-resolved to
  an IP) plus the normalized path. Stored with the binding; dispatch
  compares stored fingerprint vs. binding row as a pure local string
  compare. Cosmetic alias edits don't trip re-consent; a config change that
  actually moves the destination does (ADR-069 retarget semantics); same-
  host nesting/duplicate checks become correct. Fresh `ssh -G` resolution
  happens only on add, edit, manual refresh, and workspace open (async) —
  never in the send path.
- **Registry**: `add_ssh_binding(workspace_id, target, path, *,
  allow_write=False)` mirroring `add_folder_binding` — absolute POSIX path
  required (no `~`), nesting/overlap checks only between SSH bindings on
  the same resolved host, never across local/remote. `metadata` carries
  `{"access": "rw"|"ro", "python": "python3"}`. No credentials in metadata,
  ever.
- **Charset validation, three distinct sets** (these values cross a remote
  shell command line, and OpenSSH parses options appearing after the host
  too, so position alone is not protection):
  - **Locator components, validated separately — `@` appears in no
    component ever.** The URI grammar is `ssh://[user@]host[:port]/path`,
    but the add-time validator parses it into user / host / port first and
    validates each: user `[A-Za-z0-9._-]+`; host `[A-Za-z0-9._-]+` or a
    bracketed IPv6 literal `[A-Za-z0-9:]+` (brackets are **stripped before
    argv construction**); port `[0-9]+`. Each component **must not start
    with `-`**; all other shell metacharacters and whitespace rejected.
    Because argv is rebuilt from these parts (`-p <port> -l <user> --
    <host>`), no single argv token ever mixes metacharacters.
  - Interpreter command: `[A-Za-z0-9_./-]+`, **must not start with `-`**,
    no spaces.
  - Bootstrap payload: authored without spaces — charset is exactly
    letters, digits, `. , ( ) " ; _` — enforced by test against the exact
    bootstrap string below.
- **Status semantics** (recomputed lazily like local bindings; reason
  strings live in the ephemeral layer):

  | Probe (ping op) result | Status | Effect |
  |---|---|---|
  | Root stat returned | `READY` | admitted as a run root |
  | Connection/auth/connect-timeout (transport), python missing, pin refused | `BLOCKED` | excluded from run roots; reason surfaced ("unreachable", "host lacks python3", "root rejected: symlink"). **Operation timeouts are not here** — a ping that connects and runs past its deadline is a typed error, not a status change |
  | Connects, root path absent | `MISSING` | excluded; "missing on host" |

  **Admission reads the in-memory status cache, never the stored `status`
  column** (which `add_*_binding` writes as READY and which stays
  informational for SSH bindings). **Cold start is optimistic**: an empty
  cache admits the binding; a dead host costs one typed mid-run error and
  flips the cache to BLOCKED for subsequent sends. This matches the
  availability-first posture and the op-outcome learning below.
- **Selected working folder + failures**: fingerprint mismatch or `MISSING`
  → explicit re-selection (ADR-069 consistency). `BLOCKED` (transient
  network) → the dispatch degrades to scratch + local bindings with a
  visible warning and an easy retry; **no forced re-selection**, because a
  flaky link is not a retarget.
- **Status cache learning — transport failures only**: op outcomes feed the
  cache, but only **transport-classified** failures flip the binding to
  `BLOCKED` immediately: ssh exit 255 with no remote command having run
  (unreachable / auth), connect or handshake timeout (`ConnectTimeout`
  expiry, banner/kex stall), interpreter missing (remote exit 127), root
  pin refused. An **operation that ran past its deadline** (worker watchdog
  exit 75, or the laptop-side kill with no watchdog report) is a **typed
  tool error — "operation timed out" — and leaves the cache unchanged**: a
  slow `fs_grep` on an otherwise healthy host must not take the root out
  of the next send. A mux-protocol error likewise does not flip the cache:
  it triggers master check-and-restart (below) and, if the restart's own
  connect fails with 255/timeout, *that* is the transport failure that
  flips it. Any successful op or probe flips the cache back to READY.
  Degradation compounds send-to-send without polling.

## Remote roots never touch the laptop's disk

`RunAdmittedWorkspaceRoot.root` becomes a union — `LocalRoot` (wraps `Path`)
or `RemoteRoot` (a pure-Python descriptor: alias, canonical locator,
`PurePosixPath` root) — so every existing `root: Path` consumer fails
loudly at the type boundary until migrated. The migrations:

- **Request building** (`WorkspaceToolExecutor._build_request`, currently
  `capture_directory_chain(root)` + `resolve_workspace_path` +
  `_parent_read_exclusions` on the laptop): the remote request builder does
  lexical normalization only (reject `..` escapes, hidden-path rules) and
  carries **no laptop-captured identity**; identity values in the request
  come from worker ping/stat responses.
- **CAS read-before-write** (four laptop-hashing sites:
  `_record_fs_read_observation`, `_fs_write_guard_injection`,
  `_stale_targets_for`, and `_update_ledger_after_write`
  (`local_tool_provider.py` ~2417 — re-stamps edit/patch targets after a
  write by resolving and hashing on the laptop via `_hash_file`; a fourth
  CAS site beyond the original three) — these hash files laptop-side via
  `_hash_file(resolved)` today, and the ledger stamps are the **full
  sha256 + size** (`sha256[:8]` is only how the refusal message displays
  the hash), not stat tuples): hashing moves into worker responses —
  `fs_read` returns sha256 + size alongside bytes (the worker already
  honours `expected_sha256`/`expected_absent`), and ledger stamps use
  worker-reported values only. **`_stale_write_refusal`** also hashes the
  laptop file to show the "now" value in the refusal message — its remote
  variant takes the worker-reported hash. **Wrong-file hazard**: if the
  remote path also exists on the laptop (`/tmp/x`, `/var/www/site`), the
  laptop copy must never be read, hashed, or stamped — covered by a named
  test.
- **Workspace exclusions** (`_exclusion_paths_provider`,
  `_project_instruction_excluded_dirs` — today `(root / rel).resolve()` on
  the laptop; on macOS `/var`, `/tmp`, `/etc` are symlinks into
  `/private`, so a remote root like `/var/www/site` would resolve every
  exclusion outside the root and silently drop it): exclusions ride the
  request as **serialized relative path strings**, matched **in the
  worker**; nothing resolves them against the laptop's filesystem.
  `add_binding_exclusion` (which currently refuses non-local bindings and
  stats the laptop's disk) learns SSH bindings: the exclusion UI stores raw
  relative paths for them, validated worker-side at apply time.
- **`_call_context`** (`path.is_file()` on the laptop): remote variant
  asks the worker for a stat.
- **Admission** (`_validate_project_instruction_binding` — local
  `resolve(strict=True)` + stat, serving as both admission check and
  per-call guard): for remote roots, admission = registry read + cached
  status (+ one synchronous ping at first selection, where the user is
  already waiting); the per-call guard is the worker's root pin.

## Transport & executor

- `RemoteWorkspaceToolExecutor` implements the same
  `execute(tool, args, intent)` surface as `WorkspaceToolExecutor`, slotting
  into `RunAdmittedWorkspaceRoot.workspace_executor` so provider dispatch is
  transport-agnostic.
- **Spawn (per call)**, options first, argv rebuilt from the parsed
  locator components (brackets stripped from IPv6 literals):
  `ssh -o BatchMode=yes -o ConnectTimeout=3 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -o ControlMaster=no -o ControlPath=<cm-dir>/%C [-p <port>] [-l <user>] -- <host> <python> -I -c '<bootstrap>'`
  — then stdin carries: N bytes of zlib-compressed bundle, then the framed
  request. The bootstrap is **exactly** (space-free; every character in the
  bootstrap charset above, and the charset test asserts this exact string):
  `exec(compile(__import__("zlib").decompress(__import__("sys").stdin.buffer.read(N)),"b","exec"))`
  (N embedded as a literal by the executor). `ssh -G` canonicalization
  builds its argv the same way from the same parsed parts.
- **ControlMaster lifecycle — explicit, and the executor owns its health**:
  the first need for a host starts `ssh -MNf [opts] -o
  ControlPersist=<control_persist> -- <host>` (same parsed-parts argv) in
  its own session (`start_new_session=True`), outside any call's process
  group, **under a per-host master lock** — two concurrent first calls
  would otherwise race to create the same socket; the loser of the lock
  finds the master already present. `ControlPersist` rides the `-MNf`
  command itself so a master orphaned by an app crash (no clean
  `ssh -O exit`) self-expires instead of living until the network drops.
  Per-call clients use `ControlMaster=no` + the shared `ControlPath` —
  which means **the executor, not OpenSSH, must detect and replace a dead
  master** (with `auto`, a client would transparently start a replacement;
  with `no`, it cannot): before first use per send batch, and on any
  mux-protocol error from a call, the executor runs `ssh -O check` (a
  failed mux connection counts as the same signal) and starts a fresh
  master under the same lock if the check fails. The failed *call* still
  returns a typed transport error — the restart is for subsequent calls,
  not a retry of the failed one. A deadline kill of a call therefore
  cannot take the shared connection or other in-flight calls with it, and
  cleanup checks don't see a leftover master as an orphan (masters close
  via `ssh -O exit` on app quit; ControlPersist bounds idle lifetime as
  the crash backstop). The mux directory is a **short `0700` dir** chosen
  so the full socket path stays under macOS's 104-byte `sun_path` limit
  (`%C` alone is 40 characters).
- **Shell safety**: bootstrap charset per above (no spaces by construction);
  target/interpreter charsets validated at add time with leading-dash
  rejection. Bundle bytes travel via stdin — invisible to `ps` on the
  server and never shell-parsed.
- **Stdout contamination**: the worker's response begins with a fixed magic
  byte sequence; the executor skips leading stdout bytes until the magic
  (garbage cap ~4KB → typed "remote shell emits stdout noise" error with a
  hint to guard `.bashrc`). Debian/Ubuntu source `.bashrc` for
  `ssh host cmd`; unguarded rc files otherwise corrupt framing.
- **Bundle**: committed, build-time-generated single file
  (`Tools/remote_worker_bundle.py`) assembled from the existing protocol
  framing, root pinning, `local_tool_impls` core, dispatch, and sensitive
  -path modules by `Tools/build_remote_worker_bundle.py` — possible only
  after Phase 0 makes that closure stdlib-only. Python ≥ 3.10 floor,
  enforced by **compiling the bundle under a real 3.10 in CI**
  (`uv python install 3.10`); an import denylist alone cannot catch
  3.11/3.12 syntax. The bundle reads frames only from the same buffered
  stdin stream the bootstrap used (the build script emits the IO entry
  point to take the stream as an argument). Compression ~4:1 puts per-call
  overhead at ~40KB. A drift-guard test rebuilds and diffs. Each call logs
  the bundle hash for audit.
- **Remote denylist, separate from the laptop's**: the sensitive-path
  denylist shipped in the bundle is **remote-home-relative** (`~/.ssh`,
  `~/.aws`, `~/.gnupg`, `~/.config/gcloud`, …), not the laptop's
  app-config/data directories (which mean nothing on the server). ADR-174
  user exclusions ride the serialized request as relative paths and are
  matched worker-side.
- **Worker-side watchdog**: the request already carries `timeout_seconds`
  (today `WORKSPACE_HELPER_TIMEOUT_SECONDS = 300`); the bundle arms
  `threading.Timer(request.timeout_seconds, _watchdog)` at request start,
  where `_watchdog` **best-effort unlinks the worker's registered temp
  files** (every atomic write registers its temp path in a module-level
  list on creation and deregisters on success — `os._exit` skips `finally`
  blocks, so the timer callback itself must do the cleanup) and then calls
  `os._exit(75)`. Exit code **75** (`EX_TEMPFAIL`) is fixed and reserved
  to the watchdog: ssh passes remote exit codes through, and 75 is neither
  255 (which reads as "unreachable/auth" — a transport failure) nor 127
  (interpreter missing); no other worker path may exit 75 (framed typed
  errors exit 0). The executor maps exactly "remote exit 75" to the typed
  tool error **"operation timed out"**, and — per the status-learning rule
  — leaves the binding's status unchanged. The watchdog deadline is
  strictly **earlier** than the laptop-side kill (same
  `timeout_seconds` constant, laptop kill at `timeout_seconds + grace`,
  grace ≈ 5s), so the outcome is decided by the watchdog's distinct exit
  code rather than by an ambiguous local kill; the local ssh process-group
  kill is the last-resort backstop (also a typed "timed out", also
  status-unchanged — true network death is reported earlier by
  `ServerAliveCountMax` expiry as ssh exit 255). This guarantees "nothing
  left behind": no orphaned process, and no orphaned temp file from an
  aborted atomic write. No separate `call_timeout_s` config — the existing
  constant is the single deadline source.
- **New `ping` op**: returns root stat `(st_dev, st_ino, st_mode)`,
  canonical remote path, remote python version, and bundle-hash echo — one
  multiplexed round trip powering both the status probe and authority
  capture.
- **Timeouts, two classes by design**: `ConnectTimeout` bounds TCP only;
  connect/handshake timeouts (with the handshake/banner stall bounded by
  the same deadline machinery) are **transport failures** and feed the
  BLOCKED learning. An operation that connected and ran past its deadline
  is **not** a transport failure — watchdog exit 75 or the laptop-side
  kill both produce the typed "operation timed out" tool error with status
  unchanged (see the watchdog and status-learning rules). Every call and
  probe has the hard process-side deadline with process-group kill (the
  hooks system's established pattern).
- **Concurrency**: semaphore keyed by **resolved host / ControlPath
  identity** (not per binding) — sshd's `MaxSessions` is per connection and
  multiple bindings on one host share one master. Default cap 8, config
  `max_concurrent_calls`; the probe counts against the same cap.
- **Failure taxonomy** (feeds BLOCKED reasons and typed errors, with the
  transport/non-transport split explicit): ssh exit 255 with no remote
  exit → unreachable/auth → **BLOCKED**; connect/handshake timeout →
  **BLOCKED**; remote exit 127 → interpreter missing → **BLOCKED** ("host
  lacks python3"); root pin refused → **BLOCKED**; remote exit **75** →
  watchdog fired → typed **"operation timed out"**, status **unchanged**;
  laptop deadline kill without a watchdog report → typed "timed out",
  status unchanged; mux-protocol error → stale master → typed transport
  error + executor check-and-restart of the master (cache untouched —
  only the restart's own 255/timeout connect failure flips it); magic-cap
  exceeded → typed "remote shell emits stdout noise". Worker-side
  exceptions return framed typed errors rather than stream EOF.
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
  `metadata["access"]`, `RemoteWorkspaceToolExecutor` in the executor slot,
  and the `RemoteRoot` descriptor in the `root` slot. Aliases stay
  **binding_id-derived**, hence unique by construction — no disambiguation
  machinery (`proj-2`) is added.
- **Split authority** (same fail-closed posture, no added latency):
  client-side per call — registry row re-read, fingerprint match, access
  match, cache not BLOCKED (catches retarget/permission changes); server
  side per call — the worker's root pin re-validates `(st_dev, st_ino)`
  before executing anything, catching symlink/mount drift inside the same
  round trip.
- **Parity matrix**:

  | Capability | SSH binding in v1 |
  |---|---|
  | fs_list / fs_read / fs_glob / fs_grep | full parity, server-side |
  | fs_write / fs_edit / fs_patch | full parity incl. CAS stamps (sha256+size from worker responses; re-check runs in-worker) |
  | Read-only binding | mutating specs not advertised (existing `any_write` preflight) |
  | Result spill to local scratch | unchanged |
  | git_* | call-time typed error "git tools not yet supported on SSH bindings"; still advertised for local aliases on mixed runs |
  | Change review | none in v1 — no `binding_added` change-review setup, no row capture (finalization doesn't exist for remote rows; rows nobody can act on are noise) |
  | Sensitive-path denylist + ADR-174 user exclusions | remote-home-relative denylist ships in the bundle; per-binding exclusions ride the request as relative paths, matched worker-side |

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
  accepts `ssh-filesystem` + cache-READY; the resolver's reads route through
  the executor via a bounded-read op (bytes + stat in one call); nested
  discovery's directory chain uses worker stats for `BindingRootIdentity`.
  Everything above the IO layer — caps, untrusted-content rules, ledger,
  lazy activation, first-use consent — unchanged. Instruction-loading
  failures on an unreachable remote follow ADR-069's prep-failure posture:
  content-free warning, proceed.
- **Degraded semantics**: at composition, BLOCKED/MISSING bindings are
  excluded and the note says so; mid-run drops return a typed
  transport-error tool result and the run continues on local roots. No
  transparent retry of the failed op: with `ControlMaster=no` the
  **executor owns master health** (check-and-restart after a mux failure,
  so the *next* call is healthy — see the lifecycle rule), but the failed
  call itself still returns its typed error; write idempotency is
  unknowable; retrying a long op doubles the wait.
- **Hot-path rule (load-bearing)**: run composition reads cached status
  only. Stale-READY → mid-run typed error handles it; stale-BLOCKED →
  binding stays excluded (which is the availability requirement). Probes
  refresh asynchronously (workspace open, binding list, manual refresh,
  op-outcome learning). The **two** synchronous probe points are
  add-binding and first-selection — both places where the user is already
  waiting — and add-time is advisory: a failing probe lets you save the
  binding with the reason shown, so setup works offline. No other probe
  is ever synchronous.

## Local-folder-binding touch points (enumerate before coding)

Sites that today filter or special-case `local-filesystem` / do laptop-disk
work per binding, to be migrated or explicitly extended (the implementation
plan completes this list with a `grep -rn "local-filesystem\|LOCAL_FILESYSTEM"`
pass; the OmniVoice spec's UI touch-point list is the format). Entries were
verified against a mix of this branch and dev — line numbers drift and one
site (`_call_context`) does not exist under that name on this branch at all
— so the planning pass re-locates every entry on the implementation branch:

- `Chat/console_chat_controller.py` — `_validate_project_instruction_binding`
  (admission), `_exclusion_paths_provider`,
  `_project_instruction_excluded_dirs` (laptop-resolved exclusions),
  `_workspace_binding_authority_is_current`.
- `Agents/local_tool_provider.py` — admission preflight (any_write),
  CAS machinery (`_record_fs_read_observation`, `_fs_write_guard_injection`,
  `_stale_targets_for`, `_update_ledger_after_write`, `_hash_file`), and
  **result redaction**: `redaction_root = authority.root`
  (~`local_tool_provider.py:1824`) needs a remote-path variant; the
  sibling `_result_redaction_root` in `virtual_cli_provider.py` (~212) is
  checked for the same treatment — if the virtual CLI can never hold a
  remote root, record that conclusion at the site instead.
- `Tools/workspace_tool_executor.py` — `_build_request`
  (`capture_directory_chain`, `resolve_workspace_path`,
  `_parent_read_exclusions`) and `_call_context` (~148, `path.is_file()`
  on the laptop — the remote variant asks the worker for a stat).
- `Tools/workspace_file_roots.py` — `_iter_valid_folder_bindings`,
  `allowed_file_roots`, `_binding_matches_frozen_authority`
  (per-component laptop `lstat`).
- `Workspaces/registry_service.py` — `list_folder_bindings` kind filter (×3
  local-filesystem literals), `_filesystem_binding_missing`,
  `add_folder_binding` validators.
- `Workspaces/display_state.py` — ~5 local-filesystem branches.
- `UI/Console_Modules/wiring.py`, `UI/Console_Modules/workspace.py`,
  Console file inspector — binding listing/labeling/status surfaces.
- `UI/Screens/settings_screen.py` — `_render_workspace_folder_bindings`
  and add/toggle/remove handlers.
- `Workspaces/change_review` setup scheduled off `binding_added`.

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
  `connect_timeout_s` (3), `max_concurrent_calls` (8). Deadlines reuse
  `WORKSPACE_HELPER_TIMEOUT_SECONDS`; no separate call-timeout knob. No
  master enable flag — the feature is dormant unless an SSH binding exists.

## Security posture (ADR-181 content)

- Auth fully delegated (BatchMode; agent/config only; no secrets in app,
  keyring, or metadata).
- The only code executed remotely is chatbook-shipped: stdlib-only,
  commit-time-built, drift-guarded, hash-logged per call, shell-charset
  validated end to end, with leading-dash option injection rejected at
  validation.
- Remote AGENTS.md is untrusted repo content under every ADR-068/069 rule;
  it never grants permission. Permission gates, hooks (deny-only
  PreToolUse), and approval effects apply to remote ops identically —
  `MUTATES_LOCAL` semantically means "mutates workspace storage" (rename
  deferred).
- Sensitive-path denylist enforced server-side in the bundle against a
  remote-home-relative list; ADR-174 user exclusions ride the serialized
  request and match worker-side.
- Remote file content transits the user's own SSH channel — same trust as
  their manual ssh. Host-key policy is the user's ssh config decision;
  BatchMode's unknown-host fail-closed is the default posture.
- The remote host itself is user-trusted infrastructure (their server,
  their ssh config); the worker runs as their SSH user with root-pin
  confinement equivalent to the local worker's. The worker watchdog
  bounds its lifetime to the request deadline — no orphaned processes.

## Testing

1. **Unit**: locator parse/canonicalize into user/host/port components
  (aliases, ports, IPv6 brackets — and **argv construction from the parsed
  parts**: `-p <port> -l <user> -- <host>` with brackets stripped, `ssh -G`
  argv identical), component charset validators (per-component; `@`
  rejected everywhere; leading-dash rejection), status mapping, bundle
  drift-guard, bootstrap shell-safety charset **asserted against the exact
  bootstrap string from the transport section**, compression round-trip,
  watchdog exit code (75) and temp-file registry/unlink behavior.
2. **Phase 0 gate**: import-closure test proving the *local* worker's
   import set is stdlib-only, **plus the decoder conformance test**: a
   shared corpus of valid and malformed frames (truncated, bad magic, bad
   length prefix, wrong checksum, unknown discriminator, depth/size
   bombs) fed to both the parent's pydantic acceptance and the worker's
   stdlib decoder, requiring **identical accept/reject outcomes** — the
   one-strict-JSON acceptance contract (TASK-32855) must not weaken on
   either side.
3. **Loopback integration (workhorse)**: execute the bundle via
  `python3 -I -c` + stdin frames against a temp dir, no ssh — framing,
  magic-prefix skip, pinning, all ops, CAS stamps (sha256+size),
  single-stream discipline, watchdog firing.
4. **Fake-ssh**: stub `ssh` script simulating unreachable / exit-127 /
  stdout noise / hang-past-deadline (asserting the typed "operation
  timed out" result, the **watchdog's exit 75 observed before the laptop
  kill**, **status cache unchanged after the timeout**), unknown-host —
  proves the failure taxonomy, deadline ordering, the transport/non-
  transport learning split, and **that the remote child is gone after
  timeout**. A master-restart scenario (pre-existing dead master socket →
  next call triggers check-and-restart and succeeds) and a concurrent
  first-call race (two callers, one per-host lock, one master) are
  covered here too.
5. **Wrong-file hazard (named test)**: a remote root whose path also
  exists on the laptop with different contents — the laptop copy is never
  read, hashed, or stamped (all four CAS sites plus `_stale_write_refusal`
  and result redaction).
6. **Composition & degradation**, including two **named availability
  regression tests**: *workspace with remote BLOCKED + local READY → send
  composes, local tools advertised, remote excluded, note mentions it*;
  and the inverse direction — *a healthy remote whose op times out (75)
  stays READY and is admitted on the next send*; plus cold-start-optimistic
  admission.
7. **3.10 floor**: CI compiles the bundle under a real 3.10
  (`uv python install 3.10`).
8. **Opt-in marked test** with `ssh localhost` where key auth exists; plus
  a genuine live run against a real server before any task is marked Done
  (`lessons-live-verification.md`).
9. Existing suites stay green: `test_workspace_file_roots`,
  `test_local_tool_provider`, `test_project_instruction_resolver`,
  `test_console_project_instructions`, `test_workspace_tool_executor`,
  `test_workspace_tool_protocol`, `test_local_tool_impls`, and neighbors.

## Rollout (six independently testable phases)

0. **Stdlib-only worker**: split protocol serde — parent keeps pydantic,
   bundle ships a stdlib decoder — strip pydantic/loguru/config/interop
   imports from the worker's closure (protocol serde, sensitive-path
   core); gated by the import-closure test **and** the decoder
   conformance corpus test (identical accept/reject on both sides). No
   behavior change to local tools.
1. **Binding foundation**: binding kind, registry methods, locator
   canonicalization + charset validation, remote denylist definition,
   bundle build + drift guard + 3.10 CI compile, loopback executor — zero
   network code.
2. **Transport**: explicit `ssh -MNf` master lifecycle (per-host lock,
   `ControlPersist` on the master command, executor-owned
   check-and-restart), hardened spawn (parsed-parts argv, leading-dash
   rejection), magic prefix, deadline + watchdog (exit 75, strictly before
   the laptop kill, temp-file cleanup), probe + status cache (optimistic
   cold start, transport-only op-outcome learning), failure taxonomy.
3. **Server-side authority relocation**: `LocalRoot | RemoteRoot` type
   split; migrate request building, CAS hashing (worker-reported
   sha256+size), exclusions (worker-matched, serialized relative paths),
   `_call_context`, admission — the phase that makes "tools execute
   server-side" literally true.
4. **Run integration**: admitted roots with remote executors, degraded
   composition, AGENTS.md remote reads, context-note surfaces.
5. **Surface**: Settings form, picker/switcher entries, `[console_ssh]`
   config, user-guide docs, repo AGENTS.md note.

ADR-181 is authored and committed **before Phase 0 code** (repo rule), and
linked from the backlog task, plan, and implementation notes.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Native SFTP client (asyncssh/paramiko) | Reimplements the filesystem: `fs_grep` means downloading candidates, atomic replace/flock/CAS need weaker analogues, largest scope, new dependency, worst performance. |
| Laptop-side mount (sshfs / macFUSE / rclone mount) | macOS 26 FUSE friction (system-extension approvals), slow remote grep/glob, no first-class status — and a stale mount on a dead server can hang the app, the exact opposite of the availability requirement. |
| Install chatbook (or a persistent agent) on the server | Explicitly excluded by the user's requirement: nothing runs or persists on the server. |
| rsync/git mirror + sync engine | Not live (remote changes invisible until sync), needs conflict resolution, mutations need push — surprising semantics for a workspace root. |
| Reuse `REMOTE_RUNTIME` for the kind | Reserved for agent-runtime-remote semantics; conflates filesystem bindings with runtime bindings. |
| Per-call `ssh -G` canonicalization at dispatch | Subprocess spawn in the send path violates the hot-path rule; resolved identity is stored at add/edit instead. |
| Transparent mid-run read retry | The executor's check-and-restart covers stale masters between calls; retrying the failed op itself is still excluded (write idempotency unknowable; long ops double the wait). Revisit only with measurements. |
| Change-review row capture in v1 | `binding_added` schedules per-root review setup; with no finalization actions for remote rows, captured rows are unactionable noise. |
