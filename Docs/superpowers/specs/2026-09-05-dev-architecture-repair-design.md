# Dev Architecture test repair

Approved scope: repair bounded failures, then decompose oversized screens under
their existing ratchets and resume the comprehensive test sweep. Rebased onto
origin/dev `93388ba69b` before implementation.

## Bounded repairs (TASK-31715)

- Reuse `egress.create_default_session` for the TLS requests factory, preserving
  explicit verification overrides and adding the existing timeout protection.
- Track the moved Library conversation diagnostic at its current controller.
- Scan maintained Python source for profile ownership, excluding generated build
  output while retaining helper scripts and tests in the census.
- Preserve the existing layout classifications at renamed timer callbacks;
  classify auto-sized and change-gated updates with concrete source evidence.
  Fixed-size value updates use `layout=False`. Pin the two new repeating clocks.
- Give the daily-report banner its own exclusive worker group. Dispatch the
  coordinated briefing completion refresh through the existing loader helper.
  The inventory recognizes raw-loader awaits inside verified group dispatchers,
  but continues rejecting them in mutation owners and incorrectly grouped helpers.

## Screen decomposition

Use the existing Console and Library controller patterns and constructor seams.
Move cohesive responsibilities rather than compressing source or raising caps.
Inventory callers and tests before selecting each extraction. Preserve Textual
event/worker ownership on the screen where required. Each extraction receives a
task-specific plan and focused behavior verification before proceeding.

## Verification

Reproduce on the rebased tree, run affected behavior suites and Architecture,
then resume the full repository sweep. Record every skipped or environment-bound
test separately from successful tests. A sweep that stops at a failure is not
evidence for the tests after it.

ADR required: no for the bounded repairs.
ADR path: backlog/decisions/079-network-tls-trust-policy.md (existing TLS policy).
Reason: these repairs restore established contracts without new architecture.
Each controller extraction will check the existing decomposition decisions before
implementation and record any genuinely new boundary decision separately.
