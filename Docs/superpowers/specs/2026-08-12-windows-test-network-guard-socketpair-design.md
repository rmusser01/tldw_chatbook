# Windows Test Network Guard Socketpair Design

**Task:** TASK-15458 review expansion
**Date:** 2026-08-12
**Status:** Approved
**Decision:** [ADR-058](../../../backlog/decisions/058-thread-scoped-test-socketpair-exemption.md)

## Purpose

Make literal asynchronous pytest commands work on Windows while preserving the
test suite's import-time, default-deny network guard. This removes the temporary
TASK-15100 workaround that clears the guard's protected internet address
families before pytest creates a Windows Proactor event loop.

## Root Cause Evidence

`Tests/conftest.py` installs `Tests.network_guard` at import time. The guard
patches socket connection entry points and rejects `AF_INET` and `AF_INET6`
unless a test explicitly opts into the existing global allow mechanism.

On Windows with Python 3.12, `socket.socketpair()` is implemented by
`socket._fallback_socketpair()`. That fallback creates a loopback listener and
connects the client socket to it. Pytest's asynchronous setup reaches this
internal loopback connect before an autouse fixture or marker can grant an
exception, so the guard rejects event-loop bootstrap rather than application
egress. `backlog/docs/lessons-testing-evidence.md` records the original
TASK-15100 incident and the temporary guarded-family mutation workaround.

## Scope

In scope:

- Allow only connections made dynamically inside the real
  `socket.socketpair()` implementation on the calling thread.
- Preserve all current network-denial and recording behavior outside that
  dynamic extent.
- Prove the exception cannot leak across concurrent threads or after failure.
- Restore literal Windows pytest execution for the TASK-15458 focused tests.
- Add the two small status-formatting assertions requested in Task 1 review.

Out of scope:

- General localhost access.
- A new public network-test opt-in mechanism.
- Changes to application networking or production code.
- Changes to non-Windows socket behavior beyond routing calls through a
  behavior-preserving wrapper.

## Design

### Preserve default denial

The guard remains installed during `Tests.conftest` import. Direct
`connect`, `connect_ex`, `sendto`, and `create_connection` calls for protected
families remain blocked and recorded unless a test uses the existing explicit
global allow mechanism. `AF_UNIX` behavior remains unaffected.

### Use a thread-scoped dynamic exemption

At install time, capture the real `socket.socketpair` before replacing it with
a guard-owned wrapper. Maintain an integer nesting depth in `threading.local()`.
The wrapper increments that depth, calls the captured real implementation, and
restores the prior depth in `finally`.

The guard's blocking predicate may allow a protected-family connection only
when the current thread's socketpair depth is positive. It must not mutate the
guard's global allow flag or the set of protected address families. A direct
connection made by any other thread while socketpair is active therefore
continues to be denied and recorded.

### Preserve failure safety and idempotence

The wrapper supports nested calls through a depth counter rather than a
boolean. Restoration occurs in `finally`, including when the captured
socketpair implementation raises. Repeated guard installation keeps the
original captured functions and does not stack wrappers.

## Verification Strategy

Add tests that fail against the current guard before implementing the repair:

1. A Windows-only synchronous test calls the real guarded
   `socket.socketpair()`, exchanges a byte, and confirms no blocked-attempt
   record is created.
2. A concurrency test holds the socketpair wrapper open in one thread while a
   second thread attempts a direct loopback connection; the second operation
   remains blocked and recorded.
3. A failure-path test substitutes a raising socketpair implementation and
   proves the dynamic exemption is restored afterward.
4. Existing network-guard tests continue to prove default denial, recording,
   explicit opt-in, and unaffected local-family behavior.
5. The literal Task 1 command runs without changing the guarded family set:
   `python -m pytest Tests/Library/test_library_media_content.py -q`.
6. The media-content tests directly cover the `No matches` status and wrapped
   one-based match-index formatting.

Mutation checks must demonstrate that removing the thread-local exemption or
replacing it with a process-global allow causes the new tests to fail.

## Documentation

- Add the literal-command security property to TASK-15458's acceptance
  criteria.
- Amend the implementation plan before code changes begin.
- Replace the TASK-15100 workaround note with a follow-up incident entry once
  the literal command is verified.

## ADR Check

ADR required: yes
ADR path: `backlog/decisions/058-thread-scoped-test-socketpair-exemption.md`
Reason: the repair changes a repository-wide test security boundary and the
runtime interception contract used to enforce it.
