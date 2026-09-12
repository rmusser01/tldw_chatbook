# ADR-099: Persistent Terminal Session Runtime Boundary

Status: Accepted
Date: 2026-08-28
Related Task: [TASK-22512 - Persistent interactive PTY and ConPTY terminal sessions](../tasks/task-22512%20-%20Persistent-interactive-PTY-and-ConPTY-terminal-sessions.md)
Design: [Persistent terminal sessions design](../../Docs/superpowers/specs/2026-08-28-persistent-terminal-sessions-design.md)
Extends: [ADR-094 - Raw and Virtual CLI Execution Boundaries](094-raw-and-virtual-cli-execution-boundaries.md), only by governing its separately deferred persistent-terminal phase

## Qualification outcome

The retained Task 1 evidence qualified `pyte==0.8.2` and the POSIX environment boundary. The evaluated `pywinpty==3.0.5` native Windows boundary failed mandatory alternate-buffer isolation and post-exit EOF/output-integrity rows. In accordance with this decision, pywinpty is not admitted as a project dependency and Windows Terminal remains unavailable and fail closed. A different Windows dependency or API boundary requires a new or superseding ADR and passing native qualification before implementation.

## Decision

Chatbook will add a separate, user-controlled persistent Terminal inside the
Console. It does not change the one-shot user `!`, model `shell_exec`, or
read-only `virtual_cli` contracts established by ADR-094.

Terminal shares ADR-094's false-by-default persistent host-access unlock but
has an independent process-memory-only arm that resets every Chatbook launch.
Arming Terminal neither arms one-shot raw CLI nor registers a model tool.
Terminal sessions, input, output, and display state are user-only and receive
no provider, catalog, permission-store, conversation, AgentRuns, run-log,
export, or model-context projection.

One app-global `TerminalSessionManager` owns at most four process-memory session
records across all conversations and screens. Widgets are projections only;
navigation, recomposition, and remounting cannot restart, duplicate, or close a
session. Lifecycle, terminal reason, exit code, stream closure, and output
completeness are separate fields. Session state is not persisted or
reconnectable across app restarts.

POSIX sessions use an admission-gated controlling PTY. A launcher establishes a
new session and reports identity before parent admission; only after admission
may it acquire the controlling terminal and become the interactive shell. The
parent retains the PTY master and one authoritative reaper. Cleanup accounts
for shell job control creating process groups beyond the launcher's initial
group. Same-session processes and tracked descendants are identified by PID plus
birth time and revalidated before signalling. Group signalling additionally
requires a same-birth leader and complete exclusively owned membership; otherwise
only individually validated processes are signalled. POSIX death proof requires
the exact shell reaped, PTY EOF, and two stable zero-descendant scans; uncertainty
becomes `cleanup_unproven`. One `killpg` is not sufficient proof.

Windows sessions are not admitted in the current delivery. The evaluated
candidate was an admitted Python worker assigned to a kill-on-close Job Object
before low-level `pywinpty==3.0.5` ConPTY creation, but its native qualification
failed mandatory alternate-buffer isolation and post-exit EOF/output-integrity
rows. Chatbook therefore ships no pywinpty dependency or Windows terminal
backend under this ADR. Missing support is a content-free refusal, never a
legacy winpty, high-level `PtyProcess`, or ordinary-pipe fallback. A future
Windows boundary must establish its own supported platform floor and pass
native backend identity, ownership, bounded-I/O, concurrency, output-integrity,
EOF, and cleanup evidence under a new or superseding ADR before admission.

`pyte==0.8.2` is the v1 VT-style parser qualification target. Chatbook advertises
`TERM=linux`, incrementally decodes UTF-8, and renders only safe parsed cells.
Raw output control sequences never reach the host terminal or unrelated UI.
Clipboard, host-title, hyperlink, notification, arbitrary OSC, and unsupported
control operations are ignored. Fixed device replies are allowlisted and
bounded. A pre-parser gate caps every incomplete escape class, including CSI
bytes, parameter count/digits/value, intermediates, and string controls. The
screen adapter caps cells and cursor savepoints, and qualification inventories
every mutable pyte collection. Pyte must pass the design's shell, full-screen,
Unicode, resize, alternate-screen, paste, terminal-query, and hostile-sequence
qualification matrix before UI integration continues.

The app starts the normal interactive account shell from a dedicated
terminal-specific scrubbed environment rather than reusing the one-shot raw CLI
builder unchanged. It admits only validated path, account/home, temporary,
locale, and required platform-system values; terminal qualification records the
exact per-platform set and proves shell profile and command/module discovery.
Ambient provider, proxy, tracing, Python-injection, credential-agent, and
unrelated values remain excluded. Normal startup files may reload credentials,
environment values, and arbitrary behavior; the danger disclosure states this.
The active local workspace, or real home when none is selected, is only the
starting directory and is never described as confinement.

Persistent-terminal session names use the core `regex` package's UAX #29
extended-grapheme boundary (`\X`) after NFC normalization. This keeps the
1-64 display-character contract correct for regional-indicator flags, combining
sequences, and scripts whose displayed graphemes span several code points. A
separate 1,024-code-point input ceiling bounds normalization and segmentation,
and the grapheme iterator stops after the 65th cluster.

The runtime enforces explicit bounds, including four retained session records,
a 300x120 active viewport, 5,000 lines and 4 MiB of normal-screen scrollback per
session, 512 KiB pending input and output per session, and a 256 KiB atomic paste
limit. A paste containing prohibited terminal controls is refused as a whole
with a content-free local reason; it is never silently rewritten. Stateful
terminal output applies operating-system backpressure instead of being
discarded. Close, Disarm, and Shutdown use an out-of-band idempotent signal that
cannot wait behind saturated input. Four full sessions are limited to 256 MiB
incremental managed-runtime RSS across the Chatbook parent, app-owned
workers/helpers, and IPC, excluding user shell/program RSS.

Closing uses one five-second monotonic deadline: the initial hangup/settlement
window ends no later than T+0.75 seconds, terminate no later than T+2.25,
force-kill no later than T+3.75, and the final 1.25 seconds are reserved for
drain settlement and platform death proof. Healthy drain continues concurrently
without delaying priority cleanup. Disarm and Shutdown run sessions concurrently
against one shared deadline and cannot extend an attempt already in progress.
Exact shell exit starts the same bounded settlement rather than resetting stage
timers; only an explicit user Retry starts a fresh attempt against revalidated
cleanup authority. `exited` proves the root reaped and zero owned processes.
Stream closure requires EOF; output completeness additionally requires every
admitted byte to pass the healthy parser path. After process death, a failed
parser uses a bounded raw drain that discards content without projection or
persistence so EOF can still be proven while output remains incomplete. Cleanup
uncertainty is retained visibly, keeps its cleanup authority where possible,
continues occupying a session slot, and remains actionable while locked or
unarmed. App failure relies on ordinary PTY-master closure on POSIX. Windows
refuses before creating a terminal process or cleanup handle in the current
delivery. POSIX cleanup is an operational mechanism, not a sandbox or universal
guarantee against deliberately detached host-authority processes.

While terminal input is focused, terminal-convention keys are forwarded except
for Chatbook's reserved globals and Ctrl+]. Ctrl+] enters a local keyboard-
accessible scrollback mode whose line, page, oldest, Jump-live, focus-return,
and Tab navigation actions are never forwarded to the shell. Nested-program
mouse reporting remains deferred.

## Context

ADR-094 deliberately made raw execution one-shot and non-interactive: no PTY,
no stdin, no retained current directory or environment, and no terminal screen.
It deferred persistent PTY/ConPTY work to TASK-22512 because interactive shells
introduce materially different process, state, UX, and cleanup boundaries.

The approved product need is a real terminal for users rather than another
model tool. Users need interactive programs, retained shell state, resize, and
several named sessions, while the safe virtual CLI and command-visible model
approval boundaries must remain unchanged.

Chatbook already has useful lower-level seams: the raw executor's admission
gate and POSIX/Windows ownership evidence, an app-global runtime pattern, and
Textual Console composition. It does not have a terminal emulator, ConPTY
dependency, terminal-session owner, or safe nested-control renderer.

The `textual-terminal` project demonstrates a small pyte-to-Rich viewport and
terminal input mapping, but its widget directly owns a POSIX `pty.fork`
process, uses older Textual APIs, and documents unresolved descriptor cleanup.
It is therefore reference material rather than the runtime boundary. Any code
adapted from it must be narrowly audited and retain required attribution and
license notices.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Fork `textual-terminal` as the complete implementation | Its process starts inside the widget, is POSIX-only, lacks admission-before-exec, Job Object/ConPTY support, bounded queues/history, app-global ownership, and the required cleanup proof. Replacing those parts would leave a misleading fork around a Chatbook-owned runtime. |
| Build a VT/xterm parser from scratch | Creates a large protocol and security surface unrelated to Chatbook's core value. A qualified pyte adapter is smaller and independently testable. |
| Launch the user's external terminal application | Cannot deliver the approved Console workspace, session list, bounded scrollback, navigation survival, or Chatbook-owned cleanup. |
| Reuse raw `RawShellExecutor` unchanged | Its one-shot `stdin=DEVNULL`, profile suppression, output sanitizer, and process-group lifecycle intentionally cannot represent an interactive controlling terminal or shell job control. |
| Count names with Python `len()` or Rich cell spans | Code-point and terminal-cell helpers do not implement UAX #29 extended grapheme boundaries; they over-count valid displayed names such as regional-indicator flags and Indic conjuncts. |
| Put PTY ownership in the Textual widget | Widget remount/recompose would become a process-lifecycle operation, making navigation destructive and cleanup races difficult to reason about. |
| Persist or reconnect terminal sessions | Requires a durable authenticated supervisor or daemon, PID-reuse-safe recovery, and a new data/security boundary. Process-lifetime sessions satisfy the approved scope. |
| Give models terminal read/input tools now | Violates the approved user-only privacy boundary and couples terminal authority to model permissions. TASK-24462 records separately governed bounded read proposals. |

## Consequences

### Benefits

- Users receive a real stateful terminal without weakening the safe virtual CLI
  or command-visible model shell approval policy.
- Runtime ownership survives Console navigation and is independently testable
  from rendering.
- Admission-before-shell-start and platform-native cleanup remain explicit.
- Raw control sequences terminate at a reviewed parser boundary rather than
  reaching the host terminal.
- Fixed caps and backpressure prevent ordinary output volume from growing
  Python memory without bound.
- Terminal content stays out of Chatbook persistence and model context by
  construction.

### Costs and accepted risks

- `pyte` becomes a reviewed runtime dependency; the evaluated Windows-only `pywinpty` candidate is not admitted.
- `regex==2026.4.4` becomes a reviewed core runtime dependency for UAX #29
  session-name validation.
- Their supported wheel matrix, concurrency behavior, versions, licenses, and
  required notices must be recorded in
  `Docs/superpowers/reviews/evidence/task-22512/dependency-qualification.md`
  before lockfile admission.
- `TERM=linux` is a compatibility boundary, not complete xterm emulation; some
  advanced terminal applications may degrade or remain unsupported.
- Normal interactive startup files can restore secrets and arbitrary behavior
  that environment scrubbing removed.
- Deliberately omitted ambient values can make credential agents, proxies, or
  custom module paths unavailable until a startup profile restores them;
  standard profile execution is promised, not complete external-terminal
  environment parity.
- Chatbook's global Ctrl+P, Ctrl+Q, F1, and F6 plus the Ctrl+] release chord
  remain unavailable to nested terminal programs in v1.
- Python object overhead may exceed logical scrollback bytes; measured four-
  session memory evidence is required.
- A host-authority process can deliberately detach beyond ordinary PTY/session
  and Job Object cleanup. The app reports uncertainty rather than claiming
  containment.
- Mouse-aware nested programs remain keyboard-only until TASK-23114.

### Binding tripwires

- Persisting the Terminal arm, sessions, terminal content, or reconnect state
  requires a new ADR.
- Any terminal model tool or automatic model-context projection requires
  TASK-24462's separate design and ADR.
- Changing the pinned pyte, pywinpty, or regex version, the parser/low-level
  ConPTY API boundary, or the UAX #29 grapheme-boundary behavior requires
  rerunning the named qualification artifact and a new or superseding ADR
  decision before lockfile change.
- Nested-program mouse reporting requires TASK-23114's ADR check and real-
  terminal event evidence.
- Arbitrary launch commands, caller-provided environment overrides, or a claim
  of sandbox/workspace confinement require a scoped design and ADR review.
- Process cleanup may be strengthened, but it may not be described as security
  containment without enforceable OS-level evidence.
