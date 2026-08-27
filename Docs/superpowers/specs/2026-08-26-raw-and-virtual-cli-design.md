# Raw and Virtual CLI Design

**Date:** 2026-08-26
**Status:** Approved
**Related tasks:** TASK-18926, TASK-22509, TASK-22510, TASK-22512
**ADR required:** yes
**ADR path:** `backlog/decisions/093-raw-and-virtual-cli-execution-boundaries.md`
**Reason:** This reverses the raw-shell portion of ADR-033 and establishes new
security, runtime, permission, persistence, and cross-platform process
boundaries.

## 1. Summary

Chatbook will offer two command capabilities with intentionally different trust
models:

- **Virtual CLI:** a safe, read-only, model-only facade over existing local file
  and Git tool cores. It is available by default, invokes no host shell, remains
  workspace-confined, and gates every virtual command independently.
- **Raw CLI:** a dangerous, one-shot real shell available to users through the
  Console's `! ` prefix and to models through one `shell_exec` tool. It is off by
  default, requires persistent unlock plus re-arming on every launch, and has
  the full authority of the OS user running Chatbook.

The design is honest rather than euphemistic: selecting an initial working
directory is not confinement, a scrubbed environment is not a sandbox, and
process-tree cleanup cannot contain a command that deliberately escapes the
owned group/job.

## 2. Goals and non-goals

### Goals

1. Let a user run a real Bash, PowerShell, or CMD command directly from the
   Console without a provider call or token spend.
2. Let a model request the same one-shot executor only after explicit app and
   human authorization.
3. Stream bounded output live, support cancellation and timeout, and preserve an
   honest durable local record.
4. Expose a read-only virtual CLI to models by default with independent
   per-command permissions and no host-shell subprocess.
5. Reuse Chatbook's existing catalog, permission, approval, process-ownership,
   transcript, and run-log seams.
6. Work on POSIX and Windows with platform-native process ownership evidence.

### Non-goals

- A sandboxed raw shell.
- A persistent terminal, PTY/ConPTY, interactive stdin, terminal resize, or
  retained `cd`/environment state.
- Mutating virtual commands.
- Shell pipelines, redirects, expansion, or command substitution in the virtual
  CLI.
- Persistently allowing a model to run raw shell commands without approval.
- Automatically adding user-run command output to model context.
- A new terminal or tools screen.

## 3. Product and trust model

### 3.1 Raw CLI states

Raw CLI has two independent gates:

1. `[console] raw_cli_permitted = false` is the persistent app-wide unlock. It
   defaults false and participates in Settings' normal save/revert model.
2. `raw_cli_armed` is process-memory-only. Every Chatbook process starts false.
   It is never serialized to config, workspace state, conversation state,
   snapshots, sync, or restart recovery.

The visible states are:

- **Locked:** persistent unlock is false.
- **Unlocked, not armed:** config permits raw CLI, but this process cannot run it.
- **Armed for this launch:** raw CLI is active and a persistent red danger state
  is visible.

Unlock must be saved before Arm becomes available. Arming and “Disarm now” are
immediate runtime actions, not draft settings. Disarming or saving the persistent
unlock as false denies pending raw approvals and begins bounded cleanup of active
owned command processes.

### 3.2 Authority disclosure

The unlock and arm confirmations state all of the following:

- Commands run with the same OS permissions as Chatbook.
- Commands can read, modify, or delete any accessible file, including Chatbook's
  config and permission store.
- Commands can access the network, invoke credentialed clients, launch background
  processes, and exhaust machine resources.
- The environment is scrubbed, but commands can still read credential files and
  other user data.
- Cancellation attempts to terminate the owned process group/job; deliberately
  detached descendants may survive.
- Command text and bounded output may persist in local run logs.

No workspace, scratch, or selected-folder copy may use “confined,” “sandboxed,”
or equivalent language for raw CLI.

### 3.3 Caller policy

- **User `! ` command:** runs immediately once armed because the user authored
  it. It bypasses the provider and prompt queue, including while a model run is
  active.
- **Model `shell_exec`:** requires raw CLI permitted + armed, local tools enabled,
  the global tool kill switch off, and the raw-shell tool state not Off. It then
  asks unless the current Console session holds an in-memory shell grant.
- **Model `virtual_cli`:** requires local tools enabled, the global tool kill
  switch off, and the selected virtual command's independent permission.

The global “Block all tool calls” switch does not block user-authored `! `
commands because they are direct local actions, not model calls.

## 4. Architecture

```text
User ! command -----> Raw user adapter -----+
                                            +--> RawShellExecutor --> live events
Model shell_exec ---> Raw model adapter ----+

Model virtual_cli --> command validation --> virtual command gate
                  --> VirtualCliRegistry --> existing fs_*/git_* core
```

The raw and virtual paths share only a small result/event vocabulary and
tool-style transcript presentation. They never share execution logic, and the
virtual path has no fallback to the raw executor.

### 4.1 Raw command request

Each raw invocation creates an immutable request with:

- unique invocation id;
- caller (`user` or `model`);
- command string, non-empty and at most 16 KiB;
- shell selector (`auto`, `bash`, `powershell`, or `cmd`);
- initial working directory;
- timeout, at most 300 seconds;
- Console session id and current transcript anchor;
- cancellation token.

The initial directory defaults to the active local folder binding. With no
selected binding it defaults to the Chat's private scratch. An explicit absolute
directory is accepted when it exists and is a directory; it is displayed in the
model approval card. This is convenience, not authority enforcement.

`shell_exec.timeout_seconds` may lower the 300-second ceiling but cannot raise
it. User commands use the same ceiling in v1. The execution clock starts only
after approval and process-containment admission; time spent waiting for a human
decision does not consume it, following ADR-067.

### 4.2 Shell resolution and launch argv

The shell executable and flags are selected from fixed code-owned argv; only the
command body is dynamic. `shell=False` is used for the outer Python process
launch.

- `auto` on POSIX resolves Bash first and POSIX `sh` second.
- `auto` on Windows resolves PowerShell (`pwsh`, then Windows PowerShell) first
  and CMD second.
- Direct user `! ` commands use `auto` in v1. Explicit shell selection belongs
  to the shared executor contract and the model `shell_exec` schema; v1 adds no
  second composer prefix grammar or persisted user-shell preference.
- Explicit selectors fail clearly when unavailable. Windows Bash is supported
  only when an ordinary `bash` executable is discoverable; v1 does not add WSL
  path translation.
- Bash: `bash --noprofile --norc -c <command>`.
- POSIX fallback: `sh -c <command>`; non-login `sh -c` loads no user profile.
- PowerShell: fixed `-NoLogo -NoProfile -NonInteractive -Command <command>`.
- CMD: fixed `/D /S /C <command>` to disable AutoRun and use deterministic
  command-string handling.

Commands reject NUL. Multi-line commands are allowed within the input cap and
are rendered safely in approvals.

### 4.3 Scrubbed environment

The environment starts empty. It copies only the platform-appropriate subset of
`PATH`, `HOME`, `USERPROFILE`, `TMPDIR`, `TEMP`, `TMP`, locale variables,
`SYSTEMROOT`, `WINDIR`, `COMSPEC`, and `PATHEXT`. Chatbook/provider API keys,
tokens, proxy variables, tracing credentials, Python injection variables, and
all unrelated ambient values are absent. V1 has no caller-supplied environment
override.

The real home/profile locator remains available for ordinary command behavior.
The disclosure therefore says environment scrubbing does not stop a command from
reading files beneath the user's profile.

### 4.4 Admitted worker and process ownership

The executor does not launch the shell directly from the Textual/app process.
It uses a small spawned worker following `STT/executor_process_tree.py`:

1. The worker enters its own POSIX session/process group, or reports its PID and
   waits without running untrusted work on Windows.
2. The parent validates the POSIX identity or assigns the Windows worker to a
   kill-on-close Job Object.
3. Only after parent admission does the worker launch the shell. Shell children
   inherit the worker's process group or Job Object.
4. The worker drains shell stdout/stderr and forwards bounded coalesced events to
   the parent.

This closes the Windows race where a directly launched shell can spawn a child
before Job assignment. It does not prevent a command with host authority from
using platform mechanisms to detach where the OS permits that.

### 4.5 Lifecycle state

State transitions are monotonic and keyed by the invocation id:

```text
created -> awaiting_approval -> admitting -> running
running -> exited | timed_out | cancelled | cleanup_unproven
created/awaiting_approval/admitting -> refused | spawn_failed | shell_unavailable
```

Cancellation is idempotent. A late cancel or disarm cannot rewrite an already
terminal result or signal a process identity belonging to another invocation.
After a short graceful termination interval, cleanup force-terminates the owned
group/job and waits for a bounded proof. App shutdown uses the same path and does
not hang indefinitely when cleanup remains unproven.

## 5. Streaming, output, and persistence

### 5.1 Stream handling

Shell stdin is `DEVNULL`; this is explicitly non-interactive. Stdout and stderr
remain separate byte streams and are drained concurrently. Ordering is preserved
within each stream; cross-stream order is the order observed by the forwarding
queue and is not claimed to reconstruct kernel write chronology.

The worker uses a bounded coalescing buffer. It emits chunk snapshots on a byte
or short time threshold so a byte-at-a-time process cannot produce an unbounded
queue or repaint the TUI per byte. Invalid UTF-8 decodes with replacement.
ANSI/OSC and unsafe control sequences are removed before display or persistence.

The transcript keeps at most 32 KiB of preview text. A secure bounded spool holds
up to the configured agent run-log record limit. Once full, readers continue
draining pipes but discard additional bytes and set `truncated=true`; output
volume alone does not deadlock the process or grow memory/disk without bound.

### 5.2 Results

Every invocation reports:

- invocation id, caller, shell, and initial directory;
- elapsed execution time;
- stdout/stderr previews;
- exit code when the shell started and exited;
- terminal state;
- `truncated` and `cleanup_proven` as orthogonal flags.

A nonzero shell exit is a completed command rather than an infrastructure crash.
The adapter maps it to an unsuccessful tool result for model outcome reporting,
while retaining the exit code and both streams. Truncation never replaces the
real terminal state.

### 5.3 Durable user-command rows

User `! ` commands append live TOOL-style markers but never append provider-role
tool messages. Durability uses `AgentRunsDB` without a schema migration:

- create a row with `agent_kind="local_command"` and a generic task label that
  does not duplicate the command into metadata;
- write bounded command/result steps and the local run log under its run id;
- use the existing `assistant_message_id` locator as the persisted transcript
  anchor where the current leaf has one;
- teach transcript marker restoration to include `local_command` rows while
  agent/sub-agent counts, fleet state, costs, and agent rails explicitly exclude
  them.

The exact command and bounded sanitized output live in the local run record and
may be reconstructed into the TOOL marker on resume. Provider-history builders
continue to consume only conversation tree nodes, so these markers never enter
the next model request. User-command run logs use a dedicated app-private root
that is not registered with model-facing run-log search, slice, or statistics
tools; `local_command` records are model-excluded even when an agent can search
its own ordinary run logs.

Generic diagnostic logs record only invocation id, shell, timing, byte counts,
and outcome. They never record command text or output.

## 6. Raw CLI UI

### 6.1 Privacy & Security

The canonical Settings surface gains one “Raw CLI — unsafe” card. It displays
the three trust states from §3.1, the full authority disclosure, a persistent
unlock control, and an immediate Arm/Disarm action. Armed state uses a durable
red indicator rather than relying only on transient toast copy.

First-time persistent unlock and every-launch arm are separate confirmations.
Arm is disabled until an unlock change has been saved. Turning the saved unlock
off disarms immediately after save; Disarm remains available without changing a
Settings draft.

### 6.2 Composer

Only exact `! ` at the start of the resolved draft selects raw mode. `\! ` sends
a normal user message with the escape removed. A paste token cannot create raw
mode by itself. After the user physically enters `! ` and the composer shows the
red `RAW CLI - host access` state, pasted command body text is permitted.

Raw submit occurs before slash-command parsing and bypasses the model prompt
queue. It remains available while a model run is active. Locked/unarmed submit
returns an inline refusal with the path to Privacy & Security; it never opens,
changes, or arms Settings automatically.

### 6.3 Transcript row

The row shows caller, shell, initial directory, elapsed time, live output, exit
code, truncation, cleanup certainty, and a Stop action while running. User/model
strings render with markup disabled. “Stop” means bounded cleanup attempt, not
security containment.

### 6.4 Model approval

The raw approval row displays the complete safely-rendered command, shell,
directory, timeout, and host-authority warning. Its only decisions are:

- Run once;
- Allow all raw shell commands for this Console session;
- Deny.

The session grant is conversation-session-local and is cleared by disarming or
process restart. Approval copy must say that the session decision covers future
commands, not merely repetitions of the displayed command.

## 7. Permission model

### 7.1 Raw shell

The Tools destination always presents raw-shell capability and its current
locked/unarmed/armed availability, even when the model schema is absent. Its
permission row cycles only Ask and Off.

The raw resolver applies these rules at invocation time, not only catalog-build
time:

1. persistent unlock false or runtime arm false -> refuse;
2. local tools disabled or global tool kill switch on -> refuse model calls;
3. explicit/inherited Off -> refuse;
4. process-local session grant -> allow;
5. every other state, including a stored explicit Allow -> Ask.

This custom resolver prevents hand-edited permission JSON from creating a
persistent silent raw-shell grant.

### 7.2 Virtual CLI

Virtual commands live under the new reserved principal
`local:__virtual_cli__`. The exact `__virtual_cli__` external profile id and
associated projected records are rejected/filtered at save, load, and catalog
projection, matching ADR-032's `__local__` reservation discipline.

Each command owns a stable descriptor, definition hash, and independent
Allow / Ask / Off state. Missing state resolves to Ask. The outer
`virtual_cli` model schema is a dispatcher and receives no separate permission;
the runtime validates `command` first and gates `local:__virtual_cli__::<command>`.
Approval/session/persistent decisions are recorded against that command only.

## 8. Virtual CLI

### 8.1 Model schema

The model sees one structured tool:

```json
{
  "command": "ls",
  "argv": ["."]
}
```

`command` is a fixed enum and `argv` is an array of strings with aggregate and
per-item limits. The registry applies a command-specific parser that accepts
only its documented positional arguments and flags. It never reparses a shell
string and never recognizes pipes, redirects, substitutions, or expansion.

### 8.2 V1 command registry

| Virtual command | Core |
| --- | --- |
| `ls` | `fs_list` |
| `cat` | `fs_read` |
| `grep` | `fs_grep` |
| `find` | bounded `fs_glob` semantics |
| `stat` | a small field-allowlisted `Path.stat` core through `resolve_workspace_path` |
| `git_status` | `git_status` |
| `git_diff` | `git_diff` |
| `git_log` | `git_log` |
| `git_blame` | `git_blame` |
| `git_branches` | `git_branches` |

All commands are read-only. The existing filesystem path resolver, sensitive
path denylist, Git pathspec exclusions, scan/result caps, and model-actionable
errors remain authoritative after independent permission approval. There is no
raw-shell fallback for an unknown command or unsupported argument.

### 8.3 Catalog presentation

The virtual tool is registered by default when local tools are enabled. It is
available through normal direct/progressive tool disclosure; “available” does
not guarantee direct injection into every model prompt. Existing `fs_*` and
`git_*` tools remain discoverable. UI copy calls the virtual CLI a compact
alternative facade and explicitly warns that its independent permissions may
authorize `cat` while `fs_read` is Off.

## 9. Error semantics

Stable result categories include:

- `locked`, `unarmed`, `permission_off`, `approval_denied`;
- `invalid_command`, `invalid_arguments`, `invalid_directory`;
- `shell_unavailable`, `spawn_failed`, `containment_unavailable`;
- `exited`, `timed_out`, `cancelled`, `cleanup_unproven`.

Permission resolution fails closed and distinguishes resolver failure from a
configured Off state. UI and tool results never interpolate unsanitized exception
messages, command output, or private absolute paths into generic diagnostics.

## 10. Verification

### 10.1 Raw executor

- Fixed shell argv and profile-suppression tests for Bash, PowerShell, and CMD.
- Environment truth-table test proving only the allowlist survives.
- Real stdout/stderr, invalid UTF-8, nonzero exit, timeout, cancellation, and
  output-flood tests.
- Bounded buffer/spool tests that mutation-fail if output is captured fully.
- Real POSIX grandchild cleanup evidence and real Windows Job Object admission/
  cleanup evidence; mocked platform tests alone do not qualify.
- Admission race test proving the worker cannot launch the shell before process
  ownership is established.
- Idempotent cancel/disarm and terminal-state race tests.
- Detached-descendant limitation documented and never asserted as containment.

### 10.2 Permissions and security

- Locked/unarmed/armed matrix for user and model callers.
- Runtime recheck after schema creation and after approval.
- Stored raw Allow coerces to Ask; raw approval never offers Always Allow.
- Session grant clears on disarm and restart.
- Reserved `__virtual_cli__` profile rejected/filtered at every ADR-032 seam.
- Per-command matrix proving `ls` approval cannot authorize `cat`.
- Unknown/mutating virtual command and unsupported flag fail before dispatch.
- Existing path confinement and sensitive-path tests run through virtual aliases.
- Command/output ANSI, OSC, Rich markup, control character, and diagnostic-log
  privacy tests.

### 10.3 Console and persistence

- `! `, `\! `, explicit-prefix paste, locked refusal, and active-model-run bypass.
- Provider spy proving user raw commands issue zero model requests and spend zero
  tokens.
- History projection proving user commands/results never enter provider context.
- Live streaming, batching, Stop, timeout, exit, truncation, and cleanup-uncertain
  transcript rows.
- Saved-conversation resume reconstructs the local command marker at its anchor.
- `local_command` rows do not affect agent counts, fleet state, run costs, or
  assistant-turn grouping.
- Settings draft/save behavior and process-restart re-arm reset.
- Real mounted TUI checks on supported POSIX and Windows environments.

Per repository policy, implementation phases run focused suites. A full suite is
run only when the user explicitly requests it.

## 11. Delivery plan and backlog boundaries

### Phase 1: Revise TASK-18926 - raw executor and user `! ` vertical slice

Correct TASK-18926's false workspace-confinement statement. Deliver the admitted
worker, one-shot executor, streaming/result contracts, raw Settings states,
composer routing, durable local-command records, and focused cross-platform/UI
tests. This phase is independently valuable without model raw-shell access.

### Phase 2: Read-only virtual CLI and Tools permissions

Deliver the registry, structured model tool, synthetic principal reservation,
independent per-command permissions, default-Ask catalog/UI presentation, and
virtual alias security tests.

### Phase 3: Model `shell_exec`

Reuse Phase 1's executor. Add conditional model schema registration, forced
Ask/Off resolver, session-only approval, warning-rich approval rows, global tool
kill-switch integration, and disarm/approval race coverage.

### Phase 4: Persistent terminal backlog task only

Create a separate task, dependent on Phase 1, for PTY/ConPTY, interactive input,
terminal resize, retained cwd/environment, reconnect and navigation behavior,
session naming, concurrent terminal limits, and shutdown/recovery semantics. It
requires its own ADR check and is not implemented by Phases 1-3.

## 12. Documentation

Update:

- Console user guide: `! ` syntax, escape, local-only output, timeout, and Stop.
- Privacy & Security guide: unlock versus launch arm and full authority warning.
- Tools guide: virtual command list, independent permissions, and raw Ask/Off.
- Configuration reference: false-by-default `raw_cli_permitted`.
- ADR-033 links/notes so readers see ADR-093's partial supersession.
- TASK-18926 plan and acceptance criteria before implementation begins.
