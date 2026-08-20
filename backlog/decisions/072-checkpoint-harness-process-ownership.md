# ADR-072: Bound checkpoint-harness process ownership to retained signals

Status: Accepted
Date: 2026-08-14
Related Task: [TASK-19052](../tasks/task-19052%20-%20Restore-latest-dev-test-suite-health.md)
Related Spec: [Latest-dev Test-Suite Health Design](../../Docs/superpowers/specs/2026-08-13-task-19052-dev-test-suite-health-design.md)
Supersedes: N/A

## Decision

TASK-19052's unprivileged checkpoint harness owns test subprocesses that retain at
least one observable ownership signal: live ancestry, the task-scoped environment
tag, the harness process group, or the private inheritable sentinel descriptor.
Tests must not deliberately remove every ownership signal while leaving work
running. Such adversarial evasion is outside this diagnostic harness's contract.

On Darwin the harness supplements ancestry, task-tag, and process-group tracking
with one attempt-scoped regular-file sentinel. It creates the file without following
links in a private directory, passes the descriptor only to the pytest root, and uses
Darwin `libproc` census to find surviving holders. For each candidate the harness
obtains `(pid, pidversion)`, verifies an ownership signal for that PID, obtains the
identity again, and proceeds only when both identities match. It signals only that
bound audit token. Identity change or `ESRCH` restarts the bounded census; exhaustion
or other uncertainty is `process_containment_unavailable`. It never signals by bare
PID or PGID. After the pytest root is reaped, completion requires two successful,
non-truncated full censuses whose union of ancestry, tag, process-group, and sentinel
candidates is empty.

Darwin's `libproc` interfaces used here are private and ABI-unstable. Before launching
pytest, the harness must run a real capability self-test of its exact census calls,
`PROC_PIDUNIQIDENTIFIERINFO` flavor 17, and `proc_signal_with_audittoken`. A private
probe holds the sentinel; uncapped census must find and later lose it; flavor 17 must
return the pinned 56-byte structure and stable pidversion; a valid audit token must
signal it; and a token with mutated pidversion must return `ESRCH` without signaling
it. Missing symbols, wrong sizes, truncation, permission failure, stale-token
acceptance, identity mismatch, or signaling failure produce one attempt-scoped red
`process_containment_unavailable` outcome before pytest, with no complete marker.
They never fall back to bare `kill(pid)` or authorize green. Environment tags and
sentinel descriptors are corroborating ownership signals rather than authentication
boundaries.

## Context

Negative testing found that a late `atexit` fork could call `setsid`, replace its
environment during `execve`, and outlive a normally exiting pytest root. Darwin's
`NOTE_TRACK`, `NOTE_TRACKERR`, and `NOTE_CHILD` kqueue flags have been unsupported
since macOS 10.5, so recursive kqueue ownership is not available. Public,
unprivileged macOS APIs also provide no universal containment domain that survives
fork, `setsid`, environment replacement, descriptor closure, and exec.

The task is a repository test-health repair, not a privileged process-supervisor
project. The user approved the narrower cooperative-subprocess contract rather than
an elevated helper, exclusive execution UID/audit session, Endpoint Security client,
or disposable VM boundary.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Darwin `NOTE_TRACK`/`NOTE_CHILD` | The installed SDK documents them as unsupported and the live kernel returns `ENOTSUP`. |
| Process group or launchd job alone | A descendant can call `setsid()` and leave the original group. |
| Environment tag alone | An explicit minimal environment removes the tag at exec. |
| Bare PID plus birth-time recheck | Leaves a check-to-signal PID-reuse race. |
| Elevated UID/audit-session helper or VM | Could provide a stronger containment domain but materially expands privilege, tooling, and task scope. |
| Claim universal containment with the sentinel | Incorrect: a deliberately evasive child can close all inherited descriptors and strip every other signal. |

## Consequences

- Common detached and explicit-environment subprocesses remain detectable through
  overlapping ownership signals, including the sentinel for direct fork/exec.
- Cleanup signaling is bound to the observed process execution rather than a reusable
  numeric PID.
- Unsupported Darwin hosts fail closed before pytest starts.
- The harness does not claim security containment against adversarial tests. A future
  need for that guarantee requires a separate ADR and an external privileged or VM
  boundary.
- No application runtime, dependency, or production process policy changes.
