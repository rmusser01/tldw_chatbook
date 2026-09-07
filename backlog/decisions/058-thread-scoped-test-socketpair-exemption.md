# ADR-058: Thread-Scoped Test Socketpair Exemption

**Status:** Accepted
**Date:** 2026-08-12
**Related task:** [TASK-15458](../tasks/task-15458%20-%20Library-media-viewer---in-place-match-navigation-instead-of-full-document-re-parse.md)

## Allocation

The allocation checked `origin/dev` and every open pull-request head visible on
2026-08-12. ADR-057 was the highest allocated number and no ADR-058 was found.

## Context

The test suite installs a process-wide, default-deny network guard when
`Tests.conftest` is imported. On Windows with Python 3.12, asynchronous event
loop bootstrap calls the standard library's fallback `socket.socketpair()`.
That fallback creates an internal loopback TCP connection, which the guard
cannot distinguish from application egress and therefore blocks before pytest
fixtures or markers can run.

TASK-15100 documented a temporary command-line workaround that removes
`AF_INET` and `AF_INET6` from the guard's protected families. That makes pytest
run, but it also disables the property the guard exists to enforce. TASK-15458
review requires its literal focused pytest command to work without weakening
ordinary or concurrent-thread network denial.

## Decision

The network guard will capture the real `socket.socketpair` during its
idempotent installation and replace it with a wrapper. The wrapper maintains a
nested depth counter in `threading.local()`, calls the captured implementation,
and restores the prior depth in `finally`.

The guard permits protected-family connections only while the current thread's
socketpair depth is positive. The exception is therefore limited to the
dynamic extent of a standard-library socketpair call on that same thread. It
does not toggle the process-global explicit-test allow flag and does not alter
the protected-family set.

The following boundaries are required:

- Import-time installation and default denial remain in force.
- Direct `connect`, `connect_ex`, `sendto`, and `create_connection` attempts
  remain blocked and recorded.
- The exception applies only to the current thread and only during the captured
  socketpair call.
- Nested calls and all exceptional exits restore the previous depth.
- A concurrent thread remains denied while another thread is in socketpair.
- The existing explicit global allow remains the sole general test opt-in.
- `AF_UNIX` and other unprotected address families remain unaffected.
- Tests cover real Windows socketpair behavior, cross-thread isolation,
  exceptional restoration, ordinary denial, and literal async pytest startup.

## Alternatives Considered

| Alternative | Rejected because |
| --- | --- |
| Temporarily enable the global allow flag | Creates a process-wide race in which concurrent application egress can escape the guard. |
| Allow loopback or ephemeral ports generally | Reintroduces the original class of localhost-network escapes and cannot reliably identify socketpair traffic. |
| Inspect the Python call stack | Is brittle across Python versions and alternative runtimes, and is easier for tested code to imitate. |
| Install the guard after event-loop setup | Weakens import-time coverage and leaves early application imports unguarded. |
| Keep mutating the protected-family set in test commands | Makes successful runs cease to be evidence that the network guard works. |

## Consequences

Literal Windows async pytest commands can initialize their event loops while
the suite retains meaningful default-deny evidence. The implementation gains a
small thread-local wrapper and Windows-specific regression tests. Future Python
runtime changes to `socket.socketpair` remain observable through focused tests,
and any broader exception requires a separate decision.

## References

- [Windows test network guard socketpair design](../../Docs/superpowers/specs/2026-08-12-windows-test-network-guard-socketpair-design.md)
- [Testing evidence lessons](../docs/lessons-testing-evidence.md)
