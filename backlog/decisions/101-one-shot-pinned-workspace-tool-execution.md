# ADR-101: Use one-shot pinned workers for local workspace tool execution

Status: Proposed
Date: 2026-08-28
Related Task: [TASK-19637](../tasks/task-19637%20-%20Atomically-pin-local-tool-workspace-execution.md)
Design: [Atomic local-tool workspace execution](../../Docs/superpowers/specs/2026-08-28-task-19637-atomic-local-tool-workspace-execution-design.md)
Supersedes: N/A

## Decision

Chatbook will execute workspace-confined local filesystem, patch, read-only Git,
and equivalent Virtual CLI operations inside a fresh one-shot worker. The worker
pins the exact run-admitted workspace directory before performing one operation,
returns a bounded structured result, and exits. No worker remains alive between
tool calls.

The application validates the workspace binding, access mode, locator
fingerprint, and captured filesystem identity immediately before starting the
worker. Process-containment admission and successful root pinning form the
operation's linearization boundary. A registry change made after that boundary
does not retarget the admitted call; it applies to every later call.

The worker receives its bounded request over standard input rather than argv or
environment. It accepts a closed operation enum and cannot execute arbitrary
Python, shell, or caller-supplied executables. The request uses the canonical
root locator and identity captured at run admission, so a stable user-selected
symlink alias remains compatible without becoming the later execution lookup.
Absolute root locators, file content, patches, and model arguments are never
placed in process listings or generic diagnostics.

On POSIX, the worker opens the root without following the final component,
compares the descriptor identity with the admitted identity, changes directory
through the descriptor, and performs path work relative to that pinned
descriptor. Git inherits the pinned current directory and does not receive
`git -C <workspace-path>`.

On Windows, the worker rejects a symlink/junction/reparse point at the admitted
canonical locator, validates the directory identity, sets and re-verifies its
own current directory, and retains that directory for the operation. Windows
locks a process current directory against movement, deletion, and rename. The
operation fails closed when the required identity or reparse metadata is
unavailable.

The worker and any Git descendant remain inside one retained POSIX process group
or Windows kill-on-close Job Object. The implementation reuses the repository's
existing worker-containment primitives. A Git child launched by the worker must
not create a new POSIX session or otherwise escape that containment.

This decision covers replacement of the authorized workspace root between
confinement checking and use. It does not claim to create a general OS sandbox,
contain raw CLI commands, or solve every possible concurrent mutation of an
arbitrary descendant. Existing in-root symlink behavior, sensitive-path rules,
linked-Git-worktree behavior, permission checks, and output contracts remain
compatible when the admitted root does not drift.

## Context

The local `fs_*`, patch, and read-only `git_*` cores currently accept a `Path`
workspace root. They resolve a model path, verify lexical/resolved containment,
and later reopen ordinary pathnames. The local provider adds call-time root
identity checks, but an external rename or replacement between that check and
the actual open can redirect the operation.

TASK-19504 will replace Console's hidden CWD/config-root fallback with exact
run-admitted workspace bindings. Shipping that stronger selection contract while
retaining a pathname check/use race would overstate its security boundary.
TASK-19637 therefore precedes TASK-19504.

A retained helper daemon was considered and rejected. It would reduce startup
cost but add idle processes, cross-call mutable state, recovery, invalidation,
and shutdown ownership. Per-call workers keep authority and failure scoped to
one operation while still providing the fresh single-threaded context required
for descriptor-relative `chdir` and Windows process-current-directory pinning.

The read-only Virtual CLI dispatches directly to the same local filesystem and
Git cores. It is part of this boundary even though it has independent permission
state; otherwise it would become an alternate unpinned execution path.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Repeat identity checks immediately before and after pathname operations | A check before use still leaves a race, while a check after use detects a violation only after data was read or written. |
| Keep one helper alive per active workspace authority | Adds idle processes, cross-call state, invalidation, crash recovery, and shutdown complexity before performance evidence shows it is necessary. |
| Run POSIX descriptor operations in process and use a helper only on Windows/Git | Reduces some process startup but splits one security boundary across divergent invocation paths and leaves more ways for providers or Virtual CLI to bypass it. |
| Change the multithreaded app process current directory on Windows | Windows current directory is process-wide, so this could redirect unrelated threads and libraries. |
| Reject every symlink beneath an admitted root | Broadens this root-race fix into a behavioral change and breaks existing compatible in-root symlink semantics. |
| Require a full OS sandbox | A portable sandbox is a distinct project. This task binds operations to one admitted root; it does not execute arbitrary code or claim hostile-code containment. |

## Consequences

### Benefits

- Root rename, replacement, symlink, junction, and reparse substitution cannot
  redirect an admitted local filesystem or Git operation.
- The same execution boundary covers structured local tools and Virtual CLI.
- No unsafe `preexec_fn` runs from Chatbook's multithreaded process.
- Every helper has one authority, one request, one result, and bounded cleanup.
- Root locators and tool payloads stay out of process argv and generic logs.

### Costs and accepted risks

- Every local path operation pays one Python worker startup. The implementation
  must record cold and warm median/p95 overhead before closeout.
- Filesystem implementations require a root-relative execution seam rather than
  reopening absolute paths produced by `Path.resolve()`.
- Windows needs native identity/reparse and Job Object evidence. Unsupported
  capabilities refuse calls rather than falling back to pathname-only checks.
- A call admitted before an access downgrade may finish. The downgrade blocks
  subsequent admission; it does not promise unsafe asynchronous cancellation of
  an operation already inside the pinned boundary.
- This boundary prevents root retargeting but is not a general defense against
  malicious code, raw shell access, privileged mount manipulation, or every
  possible race within descendant content.

### Binding tripwires

- Pooling or retaining workers across calls requires a new ADR or an amendment
  with lifecycle, invalidation, and performance evidence.
- Adding mutating Git, shell, caller-selected executables, or arbitrary Python to
  the worker requires a separate security decision.
- Any fallback that executes after failed identity/reparse admission violates
  this ADR.
- New local path frontends must route through the pinned executor rather than
  call the path cores directly.

## 2026-11-21 note (TASK-40608, owner-approved): one idle worker is started ahead of the call

Status: **approved by the owner on 2026-11-21** ("Keep one spare").

**Why.** Every local tool call started its own `python -I -m` worker. Alone that is about 0.15 s on Windows; inside a Console turn each call's worker took 0.33-0.52 s from launch to output, two or three times per turn.

**What changed.** After a call takes or starts its worker, `workspace_tool_executor` starts the next one in the background with the same command and environment and keeps it idle. The worker reads its whole request from stdin before it does anything, so the idle process is the same process, started earlier. Nothing else moves:

- Each process still serves exactly one request and exits. The one-shot rule, the pinned root, the job object or process group, the cleanup supervisor and the cleanup proof are unchanged and are applied to the taken process exactly as to a new one.
- A spare is taken only when it was started with exactly this call's command and environment and is still running. Anything else starts a worker as before.
- No spare is started before a first valid request for that environment has been dispatched, so a refused request still reaches no worker and starts none.
- The spare is ended at interpreter exit, when storage pauses for a backup or restore (`storage_admission._pause_listeners`), and when a call needs another environment. If the app dies, the spare reads end-of-file and exits.
- A test that replaces `subprocess.Popen` sees every spawn: no spare is started or taken through a replacement. `_SPARE_WORKERS = False` selects the per-call start.
- The worker now times its exchange from the request's arrival, not from process start.

**Measured** (Windows 10, 2026-11-21, three interleaved journey pairs): launch to output per call 0.33-0.52 s before, 0.30-0.39 s after in two pairs (0.37-0.66 s in the third, on a loaded machine). The gain is the interpreter start, about 0.1 s per call after the first.

## Links

- [TASK-19637](../tasks/task-19637%20-%20Atomically-pin-local-tool-workspace-execution.md)
- [Approved design after review](../../Docs/superpowers/specs/2026-08-28-task-19637-atomic-local-tool-workspace-execution-design.md)
- [ADR-317](317-console-project-instruction-local-state-and-preflight.md)
- [ADR-094](094-raw-and-virtual-cli-execution-boundaries.md)
- [Microsoft SetCurrentDirectory](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setcurrentdirectory)
- [Microsoft process inheritance](https://learn.microsoft.com/en-us/windows/win32/procthread/inheritance)
