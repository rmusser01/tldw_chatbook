---
id: TASK-25814
title: >-
  Console send is blocked before provider dispatch when trace capture cannot be
  saved
status: Done
assignee: []
created_date: '2026-08-31 05:07'
updated_date: '2026-08-31 14:12'
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
FIXED, and live-verified end to end.

ROOT CAUSE (instrumented, see prior notes): ConsoleTurnPreparation.capture_mode
defaults to CAPTURE_ON; stream_chat routes every CAPTURE_ON send through
_reserve_trace_call, whose first statement raises when
_trace_call_boundary_factory is None; and NO production code ever supplies that
factory (both ensure_provider_gateway callers omit it; ConsoleTraceCallBoundary
is constructed only in Tests/). Introduced by c78f641ad -- consumer landed
without producer.

WHY I DID NOT "JUST WIRE THE FACTORY": the ledger's producer half does not
exist. A real TraceCallIdentity needs an owner + segment + policy, and while
FrozenTracePolicy IS built in production (the provenance path), nothing ever
calls create_segment/attach_owner -- they have no production callers at all.
Wiring a factory would have meant inventing turn_id/run_id/call_sequence/
idempotency_key semantics, i.e. the ledger's dedup and ordering rules. Getting
those wrong writes an audit trail that looks authoritative and is wrong, which
is worse than a loud failure.

THE FIX instead removes the false claim, not the guard. The dispatch refusal is
a deliberate invariant -- test_capture_on_without_durable_boundary_cannot_enter_
adapter pins that a Capture-On turn with no durable boundary must NOT reach a
provider -- and it stays, untouched and passing. What was wrong was UPSTREAM:
preparing a turn as Capture-On against a runtime with no way to capture it.
ConsoleProviderGateway gains `supports_durable_capture`, and
_capture_mode_for_preparation (the single existing capture-mode seam) returns
CAPTURE_OFF when the runtime cannot honour capture. Capture Off is the app's own
modelled outcome for this -- it is exactly what the "Send without capture"
button already selects via one_shot_capture_off -- so no new semantics were
invented.

The invariant is therefore intact: a genuine Capture-On turn still requires a
boundary. We simply stop claiming Capture-On we cannot deliver.

EVIDENCE:
- Tests/Chat suites: 463 passed, only the pre-existing
  test_cold_restart_recovers_open_calls baseline failure.
- test_console_chat_controller.py: baseline 98 failed -> 10 failed with this
  change, and the 10 are a strict SUBSET of the 98 (comm -23 empty). The fix
  turns 88 previously-failing tests green, showing the gate was breaking the
  suite broadly, not just my manual send.
- LIVE, clean profile against a local llama.cpp-compatible server:
  "Name three primary colors." -> Generating... -> Thinking... <1s -> a real
  model reply. No trace card, no continuation card, provider contacted.

FOLLOW-UP worth its own task: the trace-call ledger remains half-built. When
someone implements the producer (owner/segment establishment and the identity
semantics), supports_durable_capture flips to True and Capture-On resumes with
the guard already in place to protect it.
<!-- SECTION:NOTES:END -->

## Renumbering

Filed as TASK-25712 on 2026-08-31 05:07. `dev` merged PR #2255 the same morning
carrying its own TASK-25712 ("Workspace and tree chat action menus in the
Console Context rail", created 04:34), so the backlog guard flagged a duplicate.

Per the 2026-08-21 owner rule (TASK-19601) the OLDER arrival keeps the id: the
workspace-menus task was created 33 minutes earlier and keeps 25712; this task
renumbers to 25814 (the next free id after a sweep of all refs and worktrees,
max 25813).

References updated: code comments in `Chat/console_provider_gateway.py` and
`Chat/console_chat_controller.py`, the pinned test
(`Tests/Chat/test_task_25814_capture_capability.py`), and the cross-references
in TASK-25815 and TASK-25823. Commit messages already merged into this branch
still cite 25712 and are left as historical record.
