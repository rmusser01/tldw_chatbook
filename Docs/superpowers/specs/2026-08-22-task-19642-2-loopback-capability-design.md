# TASK-19642.2 Provider-Gateway Loopback Capability Design

**Status:** Approved by user on 2026-08-22; independent review issues resolved.

## Problem

Two provider-gateway regressions intentionally use a real HTTP/1.1 server on
`127.0.0.1:0` because `httpx.MockTransport` does not create httpcore's
loop-bound connection-pool primitives. On restricted hosts, the shared server
fixture fails during `socket.bind` with `PermissionError` before either
event-loop ownership contract runs.

The tests currently opt into unrestricted sockets with `allow_network`, even
though every request is to a listener owned by the test. That permission is
wider than the scenario requires and does not express what should happen when
the host cannot bind loopback.

## Decision

Keep the real HTTP server and both existing concurrency scenarios. Narrow the
two tests from unrestricted networking to the repository's existing
`loopback_network` policy. At construction of their shared numeric-loopback
listener, translate only `PermissionError` into an explicit pytest capability
skip. Listener construction is the capability boundary because the stdlib
server constructor owns socket creation, bind, and listen as one operation.
The canonical skip reason is exactly:

`loopback listener unavailable: permission denied`

Do not catch other `OSError` subclasses. Address exhaustion, malformed server
configuration, handler failures, timeouts, and gateway regressions remain test
failures rather than being mislabeled as unsupported-host behavior.

## Test Contract

Add focused fixture-contract regressions that replace only the server
constructor. One injects `PermissionError` and proves fixture setup skips with
the exact canonical reason. The other injects a non-permission `OSError` and
proves it propagates instead of being reclassified as an unsupported-host
capability. These regressions do not need a network marker because they open no
socket.

Verification has two modes:

1. Under the restricted test host, the two assigned nodes skip with
   `loopback listener unavailable: permission denied` rather than erroring at
   setup.
2. On a host granted numeric loopback bind capability, both assigned nodes run
   their existing real-socket assertions: agent-bridge-style loop swapping and
   concurrent client swapping remain non-vacuous.

The network-policy owner tests must continue to prove that `loopback_network`
permits numeric loopback only and blocks remote destinations.

## Scope

Expected implementation scope is one test module:
`Tests/Chat/test_console_provider_gateway.py`. Production gateway code,
process-wide network-guard behavior, marker registration, and external network
permissions do not change.

## Alternatives Rejected

- **Replace the server with `httpx.MockTransport`:** vacuous for these
  regressions because it bypasses the real httpcore connection pool and its
  event-loop binding.
- **Run a separate bind preflight:** adds a second socket and a time-of-check /
  time-of-use race without improving the fixture's direct error boundary.
- **Catch every `OSError`:** hides genuine server and resource failures as
  capability skips.
- **Keep `allow_network`:** grants external networking that neither test needs.

## ADR Check

ADR required: no.

ADR path: N/A.

Reason: this is a test-fixture correction that reuses the established
TASK-15111 `loopback_network` contract. It changes no production boundary,
storage, security policy, service interface, or long-lived architecture.
