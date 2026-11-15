# ADR-093: Raw and Virtual CLI Execution Boundaries

Status: Accepted
Date: 2026-08-26
Related Tasks: [TASK-18926 - Raw CLI executor and Console user command](../tasks/task-18926%20-%20Raw-CLI-executor-and-Console-user-command.md), TASK-22509, TASK-22510, TASK-22512
Design: [Raw and virtual CLI design](../../Docs/superpowers/specs/2026-08-26-raw-and-virtual-cli-design.md)
Partially supersedes: [ADR-302](302-local-agent-process-execution-boundary.md), only its rejection of raw shell execution

## Decision

Chatbook will expose two deliberately different command capabilities.

1. A read-only **virtual CLI** is available to models by default. It never
   invokes a host shell. One structured `virtual_cli` model tool dispatches an
   allowlisted command to existing policy-checked `fs_*` and read-only `git_*`
   cores. Virtual commands have independent Allow / Ask / Off permissions under
   the reserved synthetic principal `local:__virtual_cli__`; missing state
   resolves to Ask. The virtual CLI remains workspace-confined and subject to
   the sensitive-path denylist.
2. An explicitly dangerous **raw CLI** invokes a real host shell. It is disabled
   by default, requires both a persistent app-wide unlock and process-memory-only
   re-arming on every Chatbook launch, and is never described as sandboxed or
   workspace-confined. A raw command has the full filesystem, process, and
   network authority of the OS user running Chatbook.

Raw execution has two adapters over one executor:

- a user-authored `! ` Console command that makes no provider call and does not
  enter model context; and
- a model-facing `shell_exec` tool registered only while raw CLI is unlocked,
  armed, and the local-tool catalog is enabled.

Model raw-shell permission is permanently constrained to Ask or Off. Approval
offers Run once, Allow for this Console session, or Deny. No persistent Allow is
valid: a stored Allow is coerced to Ask at runtime. User-authored `! ` commands
run immediately once raw CLI is armed because the user supplied the command.
The global tool kill switch blocks the model tools but does not block direct
user commands.

Raw commands are one-shot and non-interactive in this phase: one shell process,
`stdin=DEVNULL`, no PTY, no retained working directory, and no retained
environment. Shell startup profiles are disabled. The environment is built from
a small allowlist of shell-essential variables rather than inherited wholesale.
Output streams live into a bounded transcript preview and a bounded local run
record. Timeout, cancellation, disarming, and shutdown terminate the owned POSIX
process group or Windows Job Object with bounded waits.

Process ownership is operational cleanup, not a security sandbox. A command
with host-user authority may deliberately detach descendants; Chatbook must
state when cleanup is unproven and must never promise that cancellation can
contain adversarial code.

User command records are durable but model-excluded. They use additive
`agent_kind="local_command"` rows in the unconstrained `AgentRunsDB.agent_kind`
column, anchored to the current persisted transcript leaf where available.
Their tool-style transcript markers are rebuilt on resume. Existing agent
counts and agent rails exclude this kind. Their run logs live under a dedicated
app-private root that model-facing run-log search/slice/statistics providers do
not register. This requires no database migration.

The future persistent PTY/ConPTY terminal is a separate backlog task. It is not
part of this decision's first implementation phase.

## Context

ADR-302 chose a virtual CLI and rejected raw shell execution because a claim of
workspace confinement based only on `cwd` would be false, and a portable OS
sandbox is a separate project. That reasoning remains correct. The product
decision has changed: Chatbook should offer an advanced, unmistakably dangerous
escape hatch for users and models who intentionally want the host shell, while
keeping the virtual CLI as the safe default.

The repository already owns most of the required seams:

- `Agents/tool_catalog.py` and `Agents/local_tool_provider.py` provide model-tool
  registration and invocation;
- ADR-300's permission store and approval cards provide fail-closed consent;
- `STT/executor_process_tree.py` provides proven POSIX process-group and Windows
  Job Object admission patterns;
- Console TOOL markers and AgentRunsDB provide durable local operational traces;
- existing `fs_*` and read-only `git_*` cores provide the virtual command
  implementations and their path/privacy boundaries.

The raw executor therefore extends these seams rather than creating a terminal
subsystem or a second permission store.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Preserve ADR-302's virtual-only decision | Does not meet the approved advanced-user requirement for a real host shell. |
| Claim raw commands are confined because they start in the workspace | False: a shell can use absolute paths, change directory, access the network, and launch arbitrary processes. |
| Require an OS sandbox for raw CLI v1 | Produces a different capability from the intentionally requested host-authority escape hatch and creates a large cross-platform sandbox project. |
| Separate `bash_exec`, `powershell_exec`, and `cmd_exec` tools | Duplicates schemas, permissions, lifecycle code, and UI while one fixed shell selector is sufficient. |
| Build a PTY terminal first | Adds retained state, interactive input, resize, reconnect, and terminal-emulation complexity before one-shot execution proves useful. |
| Give the virtual CLI the underlying `fs_*` / `git_*` permissions | Rejected by product decision; virtual commands require independent Allow / Ask / Off state. |
| Give raw shell persistent Allow | Leaves silent host-authority execution enabled beyond an explicit live session and conflicts with the approved re-arm-every-launch trust model. |

## Consequences

### Benefits

- Safe model command access remains available and fail-closed by default.
- Advanced users can run real cross-platform shell commands without an LLM call.
- Models can receive raw-shell authority only through explicit launch arming and
  a command-visible approval.
- User and model raw execution cannot drift because both use one executor.
- No new terminal screen, permission store, or database migration is required.

### Costs and accepted risks

- Raw CLI can read or destroy any OS-user-accessible data, modify Chatbook's own
  permission/configuration files, access the network, and exhaust resources.
- Environment scrubbing reduces accidental credential injection but does not
  stop a command from reading credential files or invoking configured clients.
- POSIX process groups and Windows Job Objects do not justify a universal claim
  that intentionally detached descendants are contained.
- Virtual CLI permissions may intentionally contradict the permissions of the
  equivalent structured tool; the Tools UI must make that independence visible.
- Raw command text and bounded output may contain secrets and persist in local
  run logs. Generic diagnostics exclude both, but the danger disclosure must
  explain local persistence.

### Binding tripwires

- A raw-shell path that claims workspace confinement requires a new ADR and real
  OS-level enforcement evidence.
- Persistent model Allow, startup-restored arming, shell profile loading, or
  ambient environment inheritance require this ADR to be revisited.
- Any mutating virtual command requires a separate design amendment, risk tags,
  and mutating-command acceptance tests.
- The persistent terminal phase requires its own ADR check because PTY ownership
  and retained session state are new runtime boundaries.

## Unproven-cleanup amendment (TASK-40716, 2026-11-13)

Unproven Raw CLI cleanup still keeps its process-group/scratch custody. Disposal no longer waits for it.

**Owner decision, 2026-11-13 ([TASK-40716](../tasks/task-40716%20-%20An-unproven-tool-cleanup-parks-a-worker-forever-refusing-later-tool-calls-and-holding-a-tool-slot.md)).**
- When a tool operation's cleanup cannot be proven, its run ends at once with a reason that names the unproven cleanup. This applies to every run, not only goal runs. The resource stays held and is never reused.
- The run has already ended by then. An explicit user abandon releases only the tool-worker slot. The resource, its storage leases and any activity claim stay held until restart (ADR-155).
- The abandon action sits on the run's terminal card: the run's reply, which the explanation naming the unproven cleanup follows.
- At app exit, disposal records retained items and stops waiting for them.

**Implementation note (not an owner decision).**
- `OwnedOperation.retain_unproven` records the retention on the runtime's capacity.
- A worker that still holds the original scopes waits on that record on a daemon thread until restart (no proof path exists yet; `UnprovenRetention.prove` would end the wait early, but no production code calls it). `UnprovenRetention.hold` never unwinds custody it has entered: on a joined thread it logs an error and still holds.
- Raw CLI user commands, Undo, @git reference expansion and the Console's startup SSH instruction read run their custody on a daemon worker, so no joined pool thread ever holds. Undo also marks its owner unproven.
- In an ordinary Console turn, @git references and the startup SSH instruction read borrow the turn's owner, so the turn's next tool reservation is refused as cleanup-unproven. That ends the agent run at its first tool call ("the cleanup of native Git could not be proven"), or, when the turn reserves a tool slot while composing its agent request (MCP), stops the turn before the run starts.
- A retention that arrives after a timeout, cancel or per-call abandon ends the run at its next tool call.
- A custody-bearing (SSH or Raw CLI) call never holds a run thread: with no timeout it still runs on a daemon tool worker, and a definitive-after-start one is refused before it starts.
- A workflow run held only by unproven cleanups (each of its tool workers is retained or has returned) does not stop exit. `ConsoleRuntime.dispose` closes every other owner and then reports `workflow_original_shutdown_unresolved`, and the app's exit runs its remaining owners before re-raising, logging any further failure it does not raise. A cancellation requested meanwhile is still re-raised, chained to the retained runs. A run whose custody is still live keeps stopping disposal before capacity closes.
- A goal iteration that ends on it is uncertain, so it needs recovery.
- The Release row follows the run's retained records, not how the run ended, and releases every slot the run still holds. A closed and reopened conversation finds it again by the card's persisted id.
