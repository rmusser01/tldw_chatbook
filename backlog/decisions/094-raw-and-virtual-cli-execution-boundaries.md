# ADR-094: Raw and Virtual CLI Execution Boundaries

Status: Accepted
Date: 2026-08-26
Related Tasks: [TASK-18926 - Raw CLI executor and Console user command](../tasks/task-18926%20-%20Raw-CLI-executor-and-Console-user-command.md), TASK-22509, TASK-22510, TASK-22512
Design: [Raw and virtual CLI design](../../Docs/superpowers/specs/2026-08-26-raw-and-virtual-cli-design.md)
Persistent terminal phase: [ADR-099](099-persistent-terminal-session-runtime-boundary.md)
Partially supersedes: [ADR-033](033-local-agent-process-execution-boundary.md), only its rejection of raw shell execution

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

ADR-033 chose a virtual CLI and rejected raw shell execution because a claim of
workspace confinement based only on `cwd` would be false, and a portable OS
sandbox is a separate project. That reasoning remains correct. The product
decision has changed: Chatbook should offer an advanced, unmistakably dangerous
escape hatch for users and models who intentionally want the host shell, while
keeping the virtual CLI as the safe default.

The repository already owns most of the required seams:

- `Agents/tool_catalog.py` and `Agents/local_tool_provider.py` provide model-tool
  registration and invocation;
- ADR-032's permission store and approval cards provide fail-closed consent;
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
| Preserve ADR-033's virtual-only decision | Does not meet the approved advanced-user requirement for a real host shell. |
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
