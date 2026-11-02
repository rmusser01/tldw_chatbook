---
id: TASK-38611
title: MCP client settlement hardening batch from the 2026-11-02 audit
status: Done
created_date: 2026-11-02 14:30
labels:
- hardening
- mcp
priority: low
updated_date: 2026-11-02 14:52
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Settlement edge cases in MCP client: stop_and_settle can wait forever on an unreapable child violating the never-hangs contract, and failures of requests that were never sent tear down the whole connection. See bug-sweep-2026-11-02-dev.md MCP findings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Settlement wait is bounded and returns the original failure after a grace period
- [x] #2 Failures of unsent requests no longer tear down the connection
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Approach (a): added `_SETTLE_EXIT_GRACE_SECONDS = 2.0` (same scale as
  close()'s `_TERMINATE_TIMEOUT_SECONDS` reap waits) and bounded the three
  unbounded `while returncode is None` exit-waits: the inner
  `stop_and_settle` finally (settlement task returns to the OS reaper with
  a logged warning), the outer `_settle_failed_request` loop (breaks once
  the settlement finished but the child never reported exit -- the caller
  gets its ORIGINAL timeout/transport error), and
  `_retain_shared_request`'s custody wait (breaks to its original
  `RuntimeError("MCP transport unavailable")`). No kill-loop was added:
  close() already terminate+kills; the grace only stops waiting after it.
  Maintenance/drain machinery still requires positive exit
  (`_maintenance_session_settled`), so an unreaped child stays fenced for
  backup admission -- only the per-caller hang is fixed.
- Approach (b): `request()`'s `except (OSError, RuntimeError, ValueError)`
  now settles only when the frame actually reached the wire (mirrors the
  existing `sent` flag from the timeout branch). Pre-send failures
  (`profile.params` validation, the embedded-newline guard, "Connection is
  closed") raise without touching `_reader_unavailable`, so the session
  survives for the next request.
- Tests (`Tests/MCP/test_stdio_settlement_hardening.py`, stub unreapable
  process; grace constants monkeypatched to 0.05 s): settle returns within
  `wait_for` 2 s with terminate+kill attempted and returncode still None;
  `_retain_shared_request` raises its transport RuntimeError within the
  bound; a pre-send `ValueError` leaves the reader available with no
  `_request_settlement`, and the same session then completes a well-formed
  request. Pre-fix evidence: a standalone settle probe on an unreapable
  child was killed at 8 s (unbounded wait), and the unsent-failure test
  pre-fix entered settlement and hung the run (settle loop swallows
  cancellation by design).
- Files: `tldw_chatbook/MCP/client.py`,
  `Tests/MCP/test_stdio_settlement_hardening.py`.
- ADR required: no -- hardening inside the existing settlement contract;
  no boundary or policy change.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
