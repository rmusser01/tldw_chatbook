---
id: TASK-25712
title: >-
  Console send is blocked before provider dispatch when trace capture cannot be
  saved
status: To Do
assignee: []
created_date: '2026-08-31 05:07'
updated_date: '2026-08-31 05:22'
labels:
  - console
  - ux-review
  - p0
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A Console send that cannot record a trace is refused entirely: the interrupt card states "The provider was not contacted". Reproduced on a clean first-run profile and on a repaired existing profile, against a local provider that answers the same request in 0.8s over curl. This makes the product's core loop unusable rather than degraded, and a diagnostics feature the user never enabled becomes a hard dependency of chatting.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A send whose trace capture fails still reaches the provider
- [ ] #2 Trace-capture failure is surfaced as a non-blocking warning, not a card that halts dispatch
- [ ] #3 A clean first-run profile with a reachable provider completes a send end to end without any interrupt card
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ROOT CAUSE FOUND (live instrumentation, not inference).

Patched TraceCallPersistenceError.__init__ to dump a stack, ran the real app,
sent a message. Single captured frame:

  console_agent_bridge.py:2895   _consume -> gateway.stream_chat(...)
  console_provider_gateway.py:3902  stream_chat -> self._reserve_trace_call(...)
  console_provider_gateway.py:2224  raise TraceCallPersistenceError(reservation_status="not_established")

Captured "active exc: None" -- nothing threw. Nothing failed to write. This is a
guard firing on a permanent condition:

  # console_provider_gateway.py:2223
  if self._trace_call_boundary_factory is None:
      raise TraceCallPersistenceError(reservation_status="not_established")

The chain:
1. ConsoleTurnPreparation.capture_mode defaults to CAPTURE_ON
   (console_turn_preparation.py:118).
2. stream_chat routes every CAPTURE_ON send through _reserve_trace_call
   (console_provider_gateway.py:3900-3906).
3. _reserve_trace_call hard-fails when _trace_call_boundary_factory is None.
4. NO PRODUCTION CODE EVER SUPPLIES THAT FACTORY. Both callers of
   ensure_provider_gateway omit it -- chat_screen.py:6392 and
   console_launch_wake.py:212. Only six Tests/ files inject one, and
   ConsoleTraceCallBoundary is constructed exclusively in tests.
=> every production Console send raises, and the controller converts it into
   the "Trace capture blocked" pause. The provider is never contacted.

The gateway's own docstring calls this seam "Optional ... when supplied"
(console_provider_gateway.py:2088), so the call site violates the documented
contract of the parameter it depends on.

WHY TESTS ARE GREEN: every test injects a factory, so the suite exercises only
the wired path and never the production one.

REGRESSION WINDOW: introduced by c78f641ad "feat(console): implement
reference-backed semantic trace ledger" (2026-08-30, 30 commits before the
reviewed tip 0ef6f3fd4) -- the commit added the consumer without wiring the
producer. Same-day regression on dev, not long-standing.

TWO FIX SHAPES, product decision required:
(a) Wire a real ConsoleTraceCallBoundary factory at both production call sites
    -- makes the shipped feature actually work.
(b) Treat a missing factory as "capture unavailable" and take the existing
    _capture_off_admission path instead of raising -- honours the documented
    "optional" contract and guarantees a missing seam can never kill the core
    loop.
Recommend BOTH: (a) restores intent, (b) is the defense-in-depth that stops
this class of half-wiring from ever being fatal again.

REGRESSION TEST GAP: needs a test that builds the gateway WITHOUT a factory
(i.e. exactly as production does) and asserts a CAPTURE_ON send still reaches
the adapter.
<!-- SECTION:NOTES:END -->
