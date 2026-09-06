---
id: TASK-31801
title: Fence Console conversation binding publication
status: Done
assignee:
  - '@codex'
created_date: '2026-09-06'
updated_date: '2026-09-06 02:13'
labels:
  - tests
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The public first-persistence and explicit conversation-rebinding routes mutate
fork-owned identity without owning a fork transition. Protect their entire
publication lifetime even when called outside an already guarded parent route.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both binding publishers reject fork eligibility and fence issuance while publishing the source identity; independent sessions remain forkable.
- [x] #2 Successful, failed and invalid-input exits release ownership without changing existing binding and revision semantics.
- [x] #3 Behavioral regressions fail before the repair and pass afterward; complete affected files, scoped static checks and independent review qualify the change without exempting other census routes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add parameterized real-store regressions to the complete settings-publication test file. Observe fork eligibility and fence issuance at the existing session lookup seam before identity publication; retain the actual lookup and exercise successful and injected-failure exits. Confirm failures before production edits.
2. Reuse the existing session-transition decorator on both public publishers. Add only these two direct owners to the census; leave the pending-work carrier and delegated name-setter classifications unresolved in this slice.
3. Run complete publication, settings apply, first-send and fork files plus the census. Preserve and explicitly report remaining census classification failures. Run scoped Ruff/format and diagnostic delta, obtain independent review, and save the draft PR checkpoint.

ADR required: no
ADR path: backlog/decisions/092-console-chat-fork-copy-and-authority-boundary.md
Reason: Restore the existing immutable fork snapshot contract using the existing transition owner; no new storage or authority boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Wrapped both public conversation-binding publishers in the existing session transition, preserving validation and binding revision semantics. Four corrected behavioral regressions failed before production edits because real fork eligibility remained true; all eight publication tests now pass, including source/other-session isolation, successful and injected-error exits, invalid input and missing-session cleanup. Six complete affected files returned 329 passed and one known census classification failure in 49.39 seconds, with two existing dependency warnings; evidence: /private/tmp/tldw-binding-publication-final.xml. Only the two repaired census owners were registered; four separate carrier/delegation classifications remain open. Scoped Ruff/format/diff checks and independent review pass. All 81 store diagnostic statements unchanged. Updated the checkpoint with remaining classification evidence. ADR check: existing ADR-092 applies; no new ADR or authority boundary. No new general lesson beyond the documented probe timing.
<!-- SECTION:NOTES:END -->
