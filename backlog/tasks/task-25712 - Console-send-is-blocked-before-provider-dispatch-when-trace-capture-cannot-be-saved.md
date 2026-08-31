---
id: TASK-25712
title: >-
  Console send is blocked before provider dispatch when trace capture cannot be
  saved
status: To Do
assignee: []
created_date: '2026-08-31 05:07'
updated_date: '2026-08-31 05:34'
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
ROOT CAUSE (verified by live instrumentation + isolated repro):

  console_agent_bridge.py:2895   _consume -> gateway.stream_chat(...)
  console_provider_gateway.py:3902  stream_chat -> self._reserve_trace_call(...)
  console_provider_gateway.py:2224  raise TraceCallPersistenceError(reservation_status="not_established")

Captured "active exc: None" -- nothing threw, nothing failed to write. The guard
fires on a permanent condition:

  if self._trace_call_boundary_factory is None:
      raise TraceCallPersistenceError(reservation_status="not_established")

Chain: ConsoleTurnPreparation.capture_mode defaults to CAPTURE_ON
(console_turn_preparation.py:118) -> stream_chat routes every CAPTURE_ON send
through _reserve_trace_call -> that raises when the factory is None -> NO
PRODUCTION CODE EVER SUPPLIES THAT FACTORY. Both callers of
ensure_provider_gateway omit it (chat_screen.py:6392,
console_launch_wake.py:212); ConsoleTraceCallBoundary is constructed only in
Tests/. Introduced by c78f641ad (2026-08-30) -- consumer landed without producer.

WHY CI IS GREEN: every test injects a factory, and the shared helper
_capture_on_prepared_request SILENTLY INSTALLS ONE when absent
(Tests/Chat/test_console_provider_gateway.py:141-160). The suite therefore
cannot observe the shipped configuration.

*** THE DEGRADE FIX (b) IS OFF THE TABLE -- DO NOT ATTEMPT IT. ***
I implemented it TDD-first and it broke a deliberate pinning test:
Tests/Chat/test_console_provider_gateway.py::
test_capture_on_without_durable_boundary_cannot_enter_adapter
That test is the exact inverse of the degrade: same setup (factory set to None,
CAPTURE_ON), asserting `pytest.raises(TraceCallPersistenceError)` AND
`adapter_called is False`. It passes on clean code. So refusing to dispatch a
Capture-On turn with no durable boundary is an INTENTIONAL SAFETY INVARIANT --
a provider call that leaves no auditable trace record must not happen. The
guard is correct; the wiring is missing.

Two degrade variants were tried and both are wrong:
  1. Downgrade capture_mode to CAPTURE_OFF at the admission seam ->
     TraceProvenanceAlignmentError "Capture Off cannot dispatch a capture-on
     prepared request" (the PREPARED REQUEST carries capture-on provenance,
     so dispatch-time downgrade is incoherent by construction).
  2. Return None from _reserve_trace_call + supply the paired
     capture_off_admission at both call sites (fresh route and
     _authorize_llamacpp_fallback). This DOES work mechanically -- both new
     tests passed -- but it defeats the invariant above.

=> THE ONLY CORRECT FIX IS (a): wire a real ConsoleTraceCallBoundary factory
   at both production call sites. The feature was shipped half-wired.

NOTE for whoever does it: _trace_dispatch_admission (:5050) documents the
pairing rule -- a null boundary is only accepted alongside a capture_off
admission -- and BOTH _reserve_trace_call call sites need the real factory
(:3902 fresh route, :4237 llama.cpp fallback; llama_cpp is the default local
provider and an empty streaming response is what drives that retry).

REGRESSION TEST TO ADD WITH THE FIX: assert that a gateway built the way
production builds it -- runtime.ensure_provider_gateway() as called from
chat_screen.py -- comes back with a non-None _trace_call_boundary_factory.
No current test covers the production wiring path.

UNRELATED PRE-EXISTING FAILURE observed at dev 0ef6f3fd4 (NOT caused by this):
Tests/Chat/test_console_trace_settlement.py::
test_cold_restart_recovers_open_calls_monotonically_and_idempotently
<!-- SECTION:NOTES:END -->
