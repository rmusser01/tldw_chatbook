# Persistent Terminal Sessions Design

**Date:** 2026-08-28
**Status:** Approved by the owner on 2026-08-28 after independent review and
final audit corrections
**Task:** TASK-22512
**Related tasks:** TASK-18926, TASK-22509, TASK-22510, TASK-24462,
TASK-23114

**Qualification outcome:** `pyte==0.8.2` passed. The evaluated
`pywinpty==3.0.5` native Windows boundary failed mandatory alternate-buffer
isolation and post-exit EOF/output-integrity checks, so the current delivery
plan is POSIX-only and Windows Terminal remains unavailable and fail
closed pending a new or superseding ADR and passing qualification.
**Related design:**
`Docs/superpowers/specs/2026-08-26-raw-and-virtual-cli-design.md`

**ADR required:** yes
**ADR path:**
`backlog/decisions/099-persistent-terminal-session-runtime-boundary.md`
**Reason:** Persistent interactive shells introduce long-lived PTY/ConPTY
ownership, terminal-emulation and dependency boundaries, app-global volatile
state, host-authority UX, and cross-platform shutdown behavior not governed by
the one-shot executor in ADR-094.

## 1. Summary

Chatbook will add a deliberately separate, user-controlled persistent Terminal
inside the Console. Up to four app-global sessions may remain alive while the
user changes conversations or screens. Each running session retains its shell
process, working directory, environment, active terminal screen, and bounded
scrollback until the shell exits, the user closes it, or Chatbook exits. An
ordinary shell exit retains the final terminal state until the user closes the
record.

Terminal is not a model tool and does not change the existing contracts for
user-authored `!` commands, model `shell_exec`, or `virtual_cli`. It shares the
persistent dangerous-host-access unlock used by raw CLI, but has an independent
process-memory-only arm that resets on every Chatbook launch. Arming Terminal
does not arm one-shot raw CLI or register model `shell_exec`.

The implementation owns process lifecycle rather than delegating it to a
Textual widget. POSIX uses an admission-gated controlling PTY. The evaluated
pywinpty ConPTY candidate did not pass mandatory native qualification, so the
current delivery has no Windows terminal backend or dependency. `pyte`
interprets terminal output into a bounded cell model before Textual renders it,
so raw control sequences never reach Chatbook's host terminal or unrelated UI.

## 2. Goals and non-goals

### Goals

1. Give users a normal interactive shell for stateful workflows that cannot fit
   one-shot raw commands.
2. Keep sessions alive across Console and application navigation for one
   Chatbook process.
3. Support POSIX PTY input, output, resize, Unicode, alternate-screen programs,
   and bounded scrollback while Windows remains unavailable and fail closed.
4. Make full OS-user authority, normal shell-profile side effects, and the
   absence of workspace confinement unmistakable.
5. Reuse Chatbook's proven admission and cleanup concepts while extending
   POSIX ownership for interactive job-control process groups.
6. Keep terminal bytes, keystrokes, and display state out of model context and
   Chatbook persistence.
7. Fail closed and visibly when authorization, backend admission, terminal
   parsing, or cleanup cannot be proven.

### Non-goals

- Persistent or reconnectable sessions across Chatbook restarts.
- A background terminal daemon.
- Model-readable or model-controllable terminal sessions.
- Terminal output in conversation history, AgentRuns, exports, or run logs.
- A sandbox, workspace confinement, network policy, or filesystem restriction.
- Arbitrary executable or launch-command entry in the session dialog.
- Nested-program mouse reporting in v1; TASK-23114 owns that follow-up.
- User-selected terminal encodings other than UTF-8.
- Perfect xterm compatibility or support for every private control sequence.

## 3. Approved product decisions

The following decisions were approved during design review:

- The Terminal replaces the Console center workspace while retaining Console
  navigation and rails (layout A).
- Sessions are app-global rather than conversation- or workspace-owned.
- The existing persistent raw-host-access unlock is shared, while Terminal has
  its own per-launch arm.
- Full disclosure occurs once when Terminal is armed for the launch. Persistent
  red host-access state remains visible; new sessions do not repeat the modal.
- New sessions start in the active local workspace binding when one is selected
  and otherwise in the user's real home directory. This is convenience, not
  confinement.
- A fixed maximum of four session records is enforced.
- Terminal data is user-only in v1. TASK-24462 owns any future explicit sharing
  with a model.
- Sessions launch the normal interactive user shell and its ordinary startup
  files from a scrubbed parent environment.
- Sessions live only for the current Chatbook process.
- `Ctrl+]` releases terminal input focus to Chatbook controls.
- Chatbook owns the runtime and may adapt only small, audited viewport/input
  ideas from `textual-terminal`; it will not adopt that project's process
  launcher or lifecycle implementation.

## 4. Trust and authorization model

### 4.1 App-level states

Terminal authority has three states:

```text
Locked -> Unlocked / not armed -> Armed for this launch
```

`[console] raw_cli_permitted = false` remains the persistent app-wide unlock.
The new `terminal_armed` value is process memory only. It always starts false
and is never written to config, workspace state, conversation state, snapshots,
sync, crash recovery, or restart metadata.

Opening Terminal while locked shows the unlock requirement and route to the
canonical Privacy & Security settings. Opening it while unlocked but unarmed
shows the Terminal arm action and disclosure. Neither state changes
automatically.

### 4.2 Disclosure

The Terminal arm disclosure states that:

- the shell and every program run with the same OS permissions as Chatbook;
- programs may read, modify, or delete any accessible data, access the network,
  invoke credentialed clients, or exhaust machine resources;
- the starting workspace or home directory is not confinement;
- Chatbook starts from a scrubbed environment, but normal shell profiles may
  reload credentials, aliases, environment variables, and arbitrary commands;
- omitted ambient integrations such as credential-agent sockets, proxies, or
  custom module paths may remain unavailable unless startup files restore them;
- shells and programs may persist their own history, files, logs, caches, and
  side effects even though Chatbook does not persist terminal content;
- closing or disarming attempts bounded cleanup, but intentionally detached
  processes may survive;
- Terminal content is user-only and is not sent to a model in v1.

### 4.3 Revocation

Disarm is immediate authority revocation for new terminal actions. If sessions
are live, the confirmation identifies how many will be terminated. Confirming
begins cleanup for every session in parallel. Saving the persistent unlock as
false has the same effect.

Cleanup uncertainty does not disappear with the surface. A session whose death
cannot be proven remains as an in-memory warning receipt, occupies one of the
four slots, and offers Retry cleanup. Re-arming never discards that receipt or
reuses its slot. Receipts and Retry cleanup remain visible and actionable while
Terminal is locked or unarmed; cleanup never requires restoring host authority.

## 5. Architecture

```text
Console TerminalWorkspace
        |
        v
TerminalSessionManager (app-global, process-memory-only)
        |
        +--> TerminalScreenModel --> safe styled-cell projection
        |
        +--> POSIX PTY backend --> admitted launcher becomes shell
        |
        `--> Windows --> unavailable / fail closed
```

### 5.1 TerminalSessionManager

One app-global `TerminalSessionManager` owns:

- the fixed four-record admission limit;
- opaque session IDs and validated display names;
- lifecycle state and focused-session selection;
- platform backend handles and the sole authoritative reaper for each launch;
- ordered input/resize routing plus priority close/disarm/shutdown signals;
- terminal screen models and bounded scrollback;
- cleanup receipts and shutdown coordination;
- view subscriptions using immutable projections and monotonically increasing
  view-generation tokens.

Widgets never own shell processes, PTY handles, parser state, reaper tasks, or
session truth. Console recomposition, navigation, and remounting may replace a
view without restarting, duplicating, or closing a session. Delayed resize,
focus, or repaint callbacks carrying a stale view-generation token are ignored.

### 5.2 Session record and states

Each record contains only process-memory state:

- random opaque session ID;
- user-visible name, shell identity, and starting directory;
- lifecycle state, terminal reason, stream-closed flag, output-complete flag,
  and timestamps;
- backend owner and process/session identity needed for safe cleanup;
- last applied rows/columns;
- terminal model, scrollback accounting, and live-bottom position;
- byte counters and content-free failure category.

Lifecycle and terminal reason are separate fields. Allowed lifecycle
transitions are:

```text
reserved -> creating -> admitting -> running
reserved/creating/admitting -> closed (launch failure; slot released)
running -> draining (shell exit)
running/draining -> closing (close, disarm, shutdown, runtime failure,
                              or owned descendants after shell exit)
draining -> exited (platform death proof; output completeness separate)
exited -> closing -> closed
closing -> closed | cleanup_unproven
cleanup_unproven -> closing -> closed | cleanup_unproven (Retry cleanup)
```

Admission reserves a slot atomically before any launch. Pre-launch failure
releases it. Running, exited, closing, and cleanup-unproven records retain it.
Closed records are removed. `exited` proves the exact shell was reaped and no
owned process remains; it does not by itself claim stream closure or complete
output. `stream_closed` is true only after PTY EOF, not merely because a parent
handle was forced closed. `output_complete` additionally requires that
all admitted bytes through EOF passed the healthy decoder/parser path without a
cleanup-only discard; bounded scrollback eviction does not change that flag.
POSIX proof requires PTY EOF. A nonzero shell exit is an ordinary exit code,
not infrastructure failure.

### 5.3 POSIX backend

The parent creates a PTY pair and starts a gated launcher. The launcher enters a
new POSIX session and reports immutable identity before it can execute user
startup code. The parent validates the identity and admits the generation. Only
after admission may the launcher acquire the slave as controlling terminal,
duplicate it onto standard streams, close unrelated descriptors, and `exec`
the selected interactive shell. The launcher PID remains the shell PID.

The app retains the PTY master and performs non-blocking reads, writes, and
window-size updates. One owner reaps the shell PID. Interactive shells may put
jobs in process groups distinct from the shell's initial group; terminal cleanup
therefore extends the raw executor's process-group policy with controlling-PTY
hangup, validated descendant/session enumeration, and bounded signal escalation.
A single `killpg(shell_pid)` is not accepted as proof of terminal-session death.

The admitted POSIX generation records shell PID, process birth time, session ID,
and initial process-group ID. Cleanup enumerates same-session processes and
tracked descendants through the existing psutil dependency, identifies each by
PID plus birth time, and revalidates identity immediately before signalling.
A process group is signalled as a group only while its leader's PID and birth
time still match and a complete current enumeration proves every member belongs
to the admitted generation. If the leader has exited, membership is mixed, or
enumeration is incomplete, the owner signals only individually revalidated
processes and reports uncertainty rather than using the numeric group ID.

POSIX death proof requires the exact shell to be reaped, PTY EOF, and zero
verified same-session or tracked descendants on two scans at least 50 ms apart
within the cleanup deadline. Access denial, identity mismatch, or incomplete
enumeration makes proof unavailable and results in `cleanup_unproven`. A process
that deliberately creates a new session may escape this ordinary ownership
model and is covered by the explicit detached-process limitation.

Closing the parent master on app failure should hang up ordinary controlling-PTY
programs. This is cleanup behavior, not a security guarantee. Processes that
deliberately detach, change session, reparent, or otherwise escape ownership may
survive.

### 5.4 Windows boundary

The evaluated v1 candidate was `pywinpty==3.0.5` through low-level
`winpty.PTY(backend=winpty.Backend.ConPTY)`, owned by an admitted worker in a
parent-held kill-on-close Job Object. Native qualification passed package,
ConPTY identity, Job admission and cleanup, bounded I/O, profiles, app-crash
cleanup, and managed RSS, but failed mandatory alternate-buffer isolation and
post-exit EOF/output integrity.

The candidate is therefore rejected for product admission. Chatbook ships no
pywinpty dependency and no Windows terminal backend in this delivery. Windows
requests receive a content-free unavailable/fail-closed result; high-level
`PtyProcess`, legacy winpty, and ordinary pipes are not fallbacks. A replacement
dependency or API boundary requires a new or superseding ADR and passing native
qualification before lockfile or backend work begins.

### 5.5 Shell resolution and startup

Shell selection is fixed-family rather than arbitrary command input:

- `Default` on POSIX resolves the OS account shell, with a validated executable
  fallback to Bash and then POSIX `sh`.
- Explicit POSIX choices list discovered Bash and Zsh.
- `Default` on Windows resolves `pwsh`, Windows PowerShell, then CMD.
- Explicit Windows choices list discovered PowerShell and CMD variants.
- An account's configured shell outside the explicit families may run through
  `Default`; it does not create an arbitrary picker entry.

All launch argv is code-owned. POSIX shells receive a TTY and no command string;
their normal interactive startup behavior applies. PowerShell remains
interactive and does not receive `-NoProfile` or `-NonInteractive`. CMD remains
interactive rather than using `/C`.

Terminal has its own scrubbed-environment contract. It keeps the raw CLI's
reviewed deny posture but does not reuse that one-shot environment builder
unchanged because interactive shells and startup profiles have different
usability requirements. The allowed initial values are:

- on POSIX: validated `PATH`, account-derived `HOME`, `USER`, `LOGNAME`, and
  `SHELL`, temporary-directory values, and locale categories;
- on Windows: validated `PATH`, account-derived `USERPROFILE`, `HOMEDRIVE`,
  `HOMEPATH`, and `USERNAME`; OS-derived `APPDATA`, `LOCALAPPDATA`,
  `PROGRAMDATA`, `ProgramFiles`, optional `ProgramFiles(x86)`/`ProgramW6432`,
  `SYSTEMROOT`, `WINDIR`, `COMSPEC`, and `PATHEXT`; temporary-directory and
  locale values.

Account and OS-derived values come from the current OS account and platform
APIs, not arbitrary caller overrides. Missing optional values are omitted
rather than invented. Chatbook sets `TERM=linux` and does not set `COLORTERM` or
other xterm-identifying markers; rows and columns come from PTY/ConPTY size
rather than inherited environment values. Ambient `PSModulePath` is not copied;
PowerShell reconstructs its documented defaults and startup profiles may extend
them. Provider keys, proxy credentials, tracing values, Python injection values,
credential-agent sockets, and unrelated ambient variables are never inherited.
The dependency qualification artifact records the exact per-platform variable
set and proves normal Bash/Zsh, PowerShell, and CMD profile and command/module
discovery behavior from that environment. Here, normal startup means the
standard account startup files execute; it does not promise parity with every
ambient variable available in an external terminal. Startup files may
repopulate any value and are covered by the disclosure.

The starting directory is validated as an existing directory immediately
before launch. Chatbook records and displays only the starting directory. A
later `cd` naturally remains in the shell but is not tracked, trusted, or
described as confined.

### 5.6 Terminal model and renderer

`pyte==0.8.2` is the v1 in-memory VT-style parser qualification target and
becomes a reviewed core dependency only after passing the gate below. The
integration uses incremental UTF-8 decoding and a terminal-specific screen
adapter. The active cell grid, cursor, mode flags, and normal-screen scrollback
are held by `TerminalScreenModel`; Textual receives only immutable safe styled
cells or run-compressed lines.

The parser selection has a mandatory qualification gate before UI integration:

- the resolved default account shell plus Bash/Zsh and PowerShell/CMD prompts;
- Vim/Nano, Less, and top/htop-class full-screen behavior where available;
- Unicode, combining characters, and double-width characters;
- alternate-screen enter/exit;
- resize and cursor preservation;
- bracketed paste;
- required device-status and capability responses;
- malformed, unsupported, and hostile control sequences.

If pyte cannot satisfy the matrix without a small bounded adapter, parser choice
must be revisited before implementation continues. Chatbook will not claim
xterm compatibility while advertising `TERM=linux`.

The existing `textual-terminal` project is reference material only. Small
rendering, style-mapping, or key-encoding portions may be adapted after review,
with required license notices and attribution. Its process launcher, direct
widget ownership, cancellation, descriptor cleanup, and POSIX-only lifecycle
are not imported.

## 6. Console experience

### 6.1 Entry and layout

Console exposes Terminal through a visible rail action and a command-palette
entry. V1 adds no global shortcut. The selected terminal replaces the center
workspace while ordinary Console navigation and rails remain available.

The surface always shows:

- persistent red `HOST TERMINAL - FULL USER ACCESS` state while armed;
- selected session, lifecycle state, shell, and starting directory;
- session list and New, Rename, Focus, Close, Retry cleanup, and Jump live
  actions when applicable;
- current terminal dimensions and visible clamping when the host allocation
  exceeds 300 columns or 120 rows;
- `Ctrl+] Release input` while the viewport owns keyboard input, or the
  scrollback-navigation keys and `Enter Focus terminal` while input is released.

### 6.2 Session creation

New Session collects:

- optional name, defaulting to `Terminal N`;
- one discovered shell choice, defaulting to `Default`;
- starting directory, defaulting to active local workspace or real home.

Names are 1-64 display characters after trimming and NFC normalization, reject
controls and markup, and are unique by normalized Unicode casefold within live
records. Names are never passed to a shell. Creating, renaming, focusing, and
closing are direct user actions that bypass the provider prompt queue and remain
available while a model turn is running.

### 6.3 Focus and navigation

A focused viewport receives terminal-convention keys except Chatbook's existing
global Ctrl+P, Ctrl+Q, F1, and F6 bindings. Ctrl+] is consumed locally and
releases input to Console controls; it cannot be sent to a nested program in
v1. These unavailable nested-program keys are a documented v1 limitation.

Sessions continue reading, parsing, and applying terminal state while hidden.
Only a visible selected session repaints. Hidden sessions retain their last
dimensions. Selecting or remounting one sends one debounced resize to the new
visible allocation without resetting screen state.

While terminal input is focused, normal PageUp/PageDown remain terminal input.
`Ctrl+]` consumes the chord and switches the viewport into scrollback-navigation
mode without changing the selected session. In that mode no navigation key is
forwarded: Up/Down move one retained line, PageUp/PageDown move one visible page,
Home moves to the oldest retained line, End invokes Jump live, Enter returns
terminal input focus without sending a newline, and Tab moves through the
visible session and workspace actions. The visible Focus action provides the
same return path. Local wheel scrolling also enters or continues navigation
mode. Leaving the live bottom freezes the viewed position while output
continues and exposes a new-output count plus Jump live. Alternate-screen
content never enters scrollback; local scroll keys are harmless no-ops there
and the visible status explains that no normal-screen history is available.

Nested-program mouse reporting is excluded from v1. TASK-23114 owns mode-aware
mouse forwarding and the real-terminal event-shape work it requires.

### 6.4 Close and disarm

Closing a running session confirms that its shell and programs will be
terminated. Closing an exited session does not confirm. Close stops new input,
sets an out-of-band cleanup signal, and starts output draining concurrently with
the platform cleanup ladder under the same monotonic deadline. The record
disappears only after proven closure; uncertain cleanup becomes a visible
retained receipt.

Disarm confirms once for all live sessions and runs cleanup in parallel. App
shutdown does not display interactive confirmations and has one bounded global
cleanup deadline.

## 7. Input, output, and resource bounds

Unless a limit explicitly says characters, lines, rows, columns, or cells, byte
limits use encoded UTF-8 transport bytes after invalid-sequence replacement.

### 7.1 Ordered input actor

Each session has one input actor that orders key bytes, complete paste events,
fixed terminal replies, and coalesced resize changes. Close, Disarm, and Shutdown
use a separate idempotent priority signal and cannot wait behind queued input.

Bounds per session:

- total pending input: 512 KiB;
- one paste: 256 KiB, admitted atomically or refused;
- one fixed terminal reply: 256 bytes;
- aggregate terminal replies: 4 KiB per second;
- latest-only pending resize, debounced for one event-loop turn and at most
  50 ms;
- viewport: 5-300 columns and 2-120 rows.

Paste validation is atomic. If pasted text contains NUL, ESC, DEL, or a C0/C1
control other than tab, carriage return, or line feed, Chatbook refuses the
entire paste before applying bracketed-paste markers or queueing any bytes. The
Terminal surface shows a content-free reason that a prohibited control class
was present; it neither logs nor echoes the rejected content. This prevents
pasted content from forging the bracketed-paste terminator or injecting
terminal-protocol bytes without silently altering what the user intended to
send. Keyboard events remain byte-accurate for supported keys.

When pending input is full, paste is refused as a whole and key input reports
`input_backpressure`; nothing is silently truncated or claimed as delivered.

### 7.2 Output ingestion

Output uses incremental UTF-8 decoding with replacement for invalid sequences.
POSIX reads and Windows worker events are capped at 64 KiB per chunk. At most
512 KiB of decoded-but-unparsed output may be pending per session. Once full,
the owner pauses the next PTY/ConPTY read so operating-system backpressure
applies. Stateful output bytes are never discarded to make room.

One parser turn processes no more than 256 KiB or 8 ms before yielding. Dirty
line changes are coalesced into at most one visible refresh per Textual frame.
Hidden sessions parse without repainting.

### 7.3 Screen and scrollback

The active screen is a bounded cell grid. Normal-screen scrollback stores
run-compressed safe text and styles, evicting oldest lines when either limit is
reached:

- 5,000 logical lines; or
- 4 MiB of deterministic character-plus-style accounting.

The logical accounting formula is UTF-8 text bytes plus 32 bytes per style run
and 16 bytes per retained line. Each active cell is limited to 32 Unicode scalar
values and 256 UTF-8 bytes so repeated combining or joiner characters cannot
grow one cell without bound; overflow is replaced and counted without logging
content. Alternate-screen lines are excluded.

Four sessions at maximum viewport and scrollback must add no more than 256 MiB
of total managed-runtime RSS over an empty-manager baseline after a five-second
quiescence. The total includes the Chatbook parent delta, app-owned terminal
workers and helper processes, and IPC buffers. It excludes the selected user
shell and user-launched program RSS. The benchmark records OS, architecture,
Python, Textual, and dependency versions. A ten-second synthetic ANSI output
flood runs an event-loop sentinel every 100 ms; its response-latency p95 must
remain below 100 ms on each supported-platform qualification host.

### 7.4 Control-sequence boundary

The adapter accepts only screen, cursor, color/style, supported mode, and
allowlisted query operations needed by the qualification matrix. It ignores
clipboard writes, host title/icon changes, hyperlinks, notifications, arbitrary
OSC payloads, and unknown controls. Raw payloads are never logged.

A bounded pre-parser gate runs before pyte and constrains every incomplete
escape class across chunks:

- every complete or incomplete control sequence: 4 KiB encoded bytes;
- CSI: at most 32 parameters, four decimal digits per parameter, values clamped
  to 9,999, and 16 private-marker/intermediate bytes;
- non-CSI ESC/charset/sharp sequences: 16 encoded bytes;
- OSC, DCS, PM, APC, and equivalent string controls: 4 KiB encoded bytes.

Crossing a limit discards content through the next valid final byte, string
terminator, CAN/SUB, or parser reset without retaining the payload. The screen
adapter caps pyte's cursor-save stack at 16 entries; a seventeenth save evicts
the oldest entry deterministically. It also caps cells as described above.
Qualification inventories every other mutable pyte parser/screen
collection and proves it is either static, viewport-bounded, or explicitly
capped; an unclassified mutable collection fails the parser gate. Fixed
terminal replies are code-owned constants, reveal no host data, and remain
subject to the input actor's per-reply and rate bounds.

Unknown or unsupported sequences are ignored without terminating the shell. An
internal parser invariant failure stops rendering, reports
`terminal_protocol_failed`, and closes the owned session; Chatbook never leaves
an invisible interactive process running behind a failed renderer.

## 8. Process cleanup and crash semantics

### 8.1 Normal close ladder

Cleanup is idempotent and identity-checked. On POSIX it combines controlling-
PTY closure/hangup, foreground/session-aware signalling, and validated recursive
descendant observation. Windows refuses before process creation, so it owns no
terminal process, stream, or cleanup handle under the current boundary.

Each ordinary close has one five-second monotonic deadline. The stages use
absolute no-later-than boundaries from its start so slow early stages cannot
consume the proof reserve:

1. stop input, request normal terminal hangup, and allow an initial healthy
   drain/settlement window until no later than T+0.75 seconds;
2. terminate remaining owned processes until no later than T+2.25 seconds;
3. force-kill remaining owned processes until no later than T+3.75 seconds;
4. reserve the remaining 1.25 seconds for bounded final drain, handle/reaper
settlement, and POSIX death proof, reporting `closed` only when proof
   succeeds.

Any stage advances immediately when its condition is satisfied; it never waits
merely to reach a boundary. Healthy output draining continues concurrently
through later stages and cannot delay the priority cleanup actor. POSIX's two
stable scans and EOF observation must fit in the final proof reserve. Within one
cleanup attempt, no internal I/O retry, read, scan, or backend wait resets the deadline.
An explicit user Retry action starts a fresh five-second attempt against the
retained, revalidated cleanup authority.

Proven closure releases all handles and removes the record. If proof fails, the
record becomes `cleanup_unproven`. The POSIX receipt retains the immutable
root/session identity, validated descendant birth identities, and PTY master so
Retry can still observe EOF. New input and repaint stop; bounded output parsing
may continue when healthy. While owned processes remain, a failed parser pauses
reads and relies on OS backpressure. After process death is otherwise proven, a
cleanup-only raw drain reads at most 64 KiB per call, yields under the parser-
turn budget, and discards bytes without decoding, rendering, logging,
persisting, or model exposure. Observed EOF sets `stream_closed=true` while
`output_complete` remains false. Missing EOF by the deadline remains
`cleanup_unproven`. Retry re-enumerates and revalidates before signalling.
Receipt actions remain available while locked or unarmed.

All sessions clean up concurrently during disarm/shutdown and share one
five-second monotonic global deadline rather than receiving five seconds each.
Their stage boundaries are calculated from the same global T0. Joining an
already-running close, settlement, or cleanup attempt is idempotent and uses the
earlier of its existing deadline and the new global deadline; Disarm or Shutdown
never extends an attempt. At the final app-shutdown deadline, the parent closes
remaining POSIX handles; there is no cross-restart cleanup claim.

### 8.2 Shell exit and output draining

The exact shell exit does not by itself prove terminal completion because a
descendant may retain the PTY slave. Shell exit disables further user
input, enters `draining`, and starts a five-second monotonic settlement deadline.
Healthy output drain and descendant validation run concurrently. If death and
stream settlement are not proven by T+0.75 seconds, remaining owned processes
enter `closing` and use the terminate, force-kill, and proof boundaries above
without resetting the deadline. When process death is proven, the record becomes
`exited`. `stream_closed` is true only after EOF, and `output_complete` also
requires the healthy parser path to have consumed all admitted bytes. Missing
PTY EOF prevents the stronger POSIX death proof and produces
`cleanup_unproven`.

### 8.3 App failure

On POSIX, process exit closes the PTY master, which should hang up ordinary
controlling-terminal processes. Focused real-platform tests must prove ordinary
cleanup for app-process failure. Windows refuses before process creation and
therefore has no terminal crash-cleanup mechanism in the current delivery.

No universal adversarial containment claim is made. A process running with the
user's authority may deliberately detach or escape the observable tree/session.
The design records cleanup certainty honestly and does not persist PID records
for speculative kill-on-next-launch behavior, which would introduce PID-reuse
hazards without a durable authenticated supervisor.

## 9. Error semantics

Stable terminal-reason categories are:

- `locked`, `unarmed`, `session_limit`;
- `invalid_name`, `invalid_start_directory`, `shell_unavailable`;
- `backend_unavailable`, `admission_failed`, `spawn_failed`;
- `input_backpressure`, `terminal_protocol_failed`, `io_failed`,
  `worker_failed`, `output_incomplete`;
- `cleanup_unproven`.

`reserved`, `creating`, `admitting`, `running`, `draining`, `exited`, `closing`,
and `closed` are lifecycle values rather than error categories. Exit code,
terminal reason, stream closure, and output completeness remain independent.

Backend or parser failure never falls back to ordinary pipes, one-shot raw CLI,
or a model tool. User-facing errors use stable safe copy and may show a user-
selected starting path only in the local Terminal surface. Generic diagnostics
record opaque session ID, lifecycle state, timing, byte counts, and content-free
failure category. They exclude terminal input/output, environment values, shell
profile data, session names, and private paths.

## 10. Privacy and model boundary

Terminal sessions register no tool provider, catalog schema, permission-store
principal, transcript marker, conversation message, AgentRuns row, run log, or
export record. Provider-history and context builders receive no session object
or terminal projection. App diagnostics exclude content as described above.

This means Chatbook does not persist terminal content. It does not mean the
shell is ephemeral or side-effect free: shell history, invoked programs, the OS,
and external services may persist any behavior they normally would.

TASK-24462 is the only recorded follow-up for deliberate user-to-model terminal
sharing. It requires its own design and ADR because even bounded read access
changes this privacy boundary.

## 11. Verification

Before dependency or UI implementation proceeds, commit
`Docs/superpowers/reviews/evidence/task-22512/dependency-qualification.md`.
It records exact pyte/pywinpty versions and hashes, supported OS/Python/
architecture and wheel matrices, API and backend identity, mutable-state audit,
I/O and EOF results, licenses and required notices, and any adapted
`textual-terminal` source attribution. A failed mandatory row stops that backend
or parser from shipping.

### 11.1 Unit and contract tests

- locked/unarmed/armed truth table and launch-reset behavior;
- independent raw CLI and Terminal arms;
- atomic four-record reservation and release races;
- name, shell, directory, terminal-specific environment, and argv validation,
  including proof that the one-shot raw environment builder is not reused
  unchanged;
- lifecycle/reason/output-complete transition matrix, cleanup retry cycles, and
  idempotent priority cleanup;
- absolute cleanup-stage boundaries, reserved proof time, a fresh deadline only
  for explicit user Retry, and one shared deadline for parallel shutdown;
- stream-closed versus output-complete semantics, including parser-failure raw
  cleanup drain with zero content projection or persistence;
- input, paste, reply, output, viewport, scrollback, and memory bounds;
- incremental UTF-8, wide/combining characters, malformed bytes, and hostile
  control sequences;
- unterminated and oversized CSI/control strings, excessive CSI parameters and
  digits, repeated cursor saves, and cross-chunk limit handling;
- bracketed paste and atomic paste-control refusal with a content-free notice;
- alternate-screen and scrollback separation;
- parser failure closes the session;
- view projections survive widget destruction and remounting;
- no provider, persistence, diagnostics, or export projection.

### 11.2 Real POSIX evidence

- controlling PTY and admission-before-exec proof;
- normal account shell startup and profile behavior;
- retained `cd` and environment state;
- interactive input, Unicode, resize/SIGWINCH, alternate screen, and the pyte
  qualification matrix;
- foreground/background job-control process groups;
- terminal-specific environment and ordinary Bash/Zsh profile behavior without
  ambient credential, proxy, tracing, injection, or agent-socket inheritance;
- exited group leaders, numeric PGID reuse, mixed/unrelated group membership,
  enumeration failure, and individually validated signalling fallback;
- exact death proof with two stable descendant scans and PTY EOF;
- exact-shell-exit plus descendant-held-slave drain behavior;
- close, disarm, shutdown, and app-process-failure cleanup;
- detached-descendant limitation documented rather than misclassified as
  containment.

### 11.3 Real Windows evidence

The retained native Windows qualification records the rejected candidate's
passing package, identity, Job, bounded-I/O, profile, crash-cleanup, and managed
RSS rows plus its mandatory alternate-buffer and EOF/output-integrity failures.
Current delivery evidence must prove that Windows remains unavailable, pywinpty
is absent from project metadata, and no high-level, legacy-winpty, or
ordinary-pipe fallback is reachable. No native Windows support claim is made.

### 11.4 Mounted and live Console evidence

- Terminal rail action and command-palette entry;
- unlock/arm/disarm states and persistent red danger copy;
- create, rename, list, focus, close, Retry cleanup, and Jump live;
- Ctrl+] release, keyboard-only scrollback navigation/focus return, harmless
  alternate-screen local scrolling, and preserved Chatbook globals;
- continued sessions and output across conversation/screen navigation;
- no process restart or duplication across recompose/remount;
- stale view-generation resize/focus/repaint callbacks are ignored;
- cleanup receipts and Retry remain available while locked or unarmed;
- model turn continues independently while Terminal is used;
- visible viewport clamping and scrollback behavior;
- real-terminal focus and input checks on POSIX, plus Windows fail-closed checks.

Pilot-only evidence does not qualify for PTY, ConPTY, real-terminal key/focus,
or process cleanup behavior. Focused suites are the default under repository
policy; a full suite is run only when explicitly requested.

## 12. Documentation

Update:

- Console user guide: entry, session controls, shells, starting directory,
  Ctrl+], scrollback, limits, reserved keys, and v1 non-goals;
- Privacy & Security guide: shared persistent unlock, independent launch arms,
  full host authority, profiles/history, and cleanup limitations;
- Tools guide: Terminal is user-only and distinct from raw `!`, model
  `shell_exec`, and `virtual_cli`;
- configuration reference: no persisted `terminal_armed` field;
- dependency/setup guidance: Windows is unsupported and fail closed pending a
  newly qualified ConPTY boundary;
- ADR-094 metadata/index links to ADR-099 without changing its one-shot
  contracts.

## 13. Alternatives considered

| Option | Decision |
| --- | --- |
| Chatbook-owned manager with pyte plus a native POSIX PTY; Windows fail closed | Selected: it preserves the required admission, privacy, resource, and cleanup boundaries while deferring ConPTY behind a new or superseding ADR and passing native qualification. |
| Fork `textual-terminal` as the complete runtime | Rejected: its widget owns the process directly, uses older Textual APIs, and lacks the required admission, bounds, and cleanup contracts. Small audited viewport ideas may still be adapted. |
| Implement a terminal parser from scratch | Rejected: terminal protocol complexity is not the product differentiator and would create a larger security and compatibility surface. |
| Launch an external OS terminal | Rejected: cannot provide the approved Console surface, app-global session list, bounded scrollback, or app-owned cleanup. |
| Reconnect sessions across restart | Rejected for v1: requires a durable authenticated supervisor/daemon and a separate lifecycle ADR. |
| Give models terminal read or input access | Rejected for v1: violates the approved user-only boundary; TASK-24462 records the separately governed read-only sharing follow-up. |

## 14. Delivery boundaries and tripwires

- Persisting `terminal_armed`, session state, terminal output, or reconnect
  metadata requires a new ADR.
- Registering any terminal model tool or placing output in model context requires
  TASK-24462's separate design and ADR.
- Nested-program mouse reporting belongs to TASK-23114 and requires an ADR check
  plus real-terminal event evidence.
- Changing the pinned pyte or pywinpty version or their parser/low-level ConPTY
  API boundary requires rerunning the named qualification artifact and a new or
  superseding ADR decision before lockfile change.
- Arbitrary launch commands, non-shell programs, custom environment overrides,
  or login-shell modes require a scoped design amendment.
- Any claim of sandboxing or workspace confinement requires real enforcement and
  a new ADR.
