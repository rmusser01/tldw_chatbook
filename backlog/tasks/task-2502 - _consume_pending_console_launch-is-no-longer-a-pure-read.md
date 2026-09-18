---
id: TASK-2502
title: _consume_pending_console_launch is no longer a pure read
status: Done
assignee:
  - '@codex'
created_date: '2026-08-04 21:52'
updated_date: '2026-09-17 01:26'
labels:
  - console
  - rag
  - cleanup
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Current reproduction (2026-09-16): reusable Console screens now make the historical risk below reachable. A second CONSOLE_LIVE_WORK launch remains pending on return to the warm screen, and a first warm launch also requires explicit surface refresh. Repair both through the established staging boundary.

Historical finding:

`ChatScreen._consume_pending_console_launch` (`tldw_chatbook/UI/Screens/chat_screen.py`, ~lines 3655-3709) has two branches. The resident-launch branch (added by PR-T1's C1 fix, `fe9b7f89f`) is a genuine seam: it calls `_supersede_resident_console_launch_from_store()`, which routes any newer store entry through `_stage_console_library_rag_launch` so every mounted surface (strip, tray, rail, control bar, workspace context, settings estimate, Inspector) stays in sync before the value is claimed.

The other branch — the non-resident path at ~lines 3691-3709 — does not go through that staging seam. It claims directly from `store.claim(HandoffChannel.CONSOLE_LIVE_WORK)` and assigns `self._pending_console_launch_context = claim.value` itself, with no call to `_stage_console_library_rag_launch` and therefore no surface sync. It makes a launch live (the method's name and most callers still treat it as a read: `compose_content()` calls it on its first line expecting a value back, not a side effect), but does so through a different, narrower code path than every other place in the module that claims a handoff.

Today this is not reachable as a defect: every production "Use in Console" stager (Library search/RAG, media, notes, conversations) runs on a screen other than Console and navigates there afterward, so by the time this method's non-resident branch would run, the screen is still mid-`compose_content()` and the value it returns *does* get rendered by the normal compose path that follows it — the missing surface sync is masked by the fact that compose is about to build the DOM from the return value anyway. This is a call-ordering coincidence, not a structural guarantee: if a future stager ever posts to `HandoffChannel.CONSOLE_LIVE_WORK` while a Console screen instance is already mounted and not re-composing (e.g. a background/async stager, or a future entry point that doesn't navigate), the non-resident branch would claim and assign the launch with no surface refresh — reopening the same "invisible claim" shape PR-T1's C1 fix (task-2372's follow-up) closed for the resident branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The non-resident branch of `_consume_pending_console_launch` routes its claim through the same staging seam (`_stage_console_library_rag_launch` or equivalent) used by the resident branch, so a claim from either branch always syncs every mounted surface
- [x] #2 `_consume_pending_console_launch`'s docstring/name reflects that the method can have a side effect (staging a launch), not only that it returns one, or the side-effecting part is factored out so the method itself is a pure read
- [x] #3 A regression test drives the non-resident branch while a Console screen instance is already composed and mounted (not mid-`compose_content()`), asserting the claimed launch is reflected on the strip/tray/rail without requiring a subsequent compose to paint it
- [x] #4 The existing behavior for every currently-shipping stager (Library search/RAG, media, notes, conversations — all of which navigate to Console after staging) is unchanged
- [x] #5 Returning to an existing Console consumes the exact pending live-work revision and visibly stages it, with or without a resident launch, preserving the current draft.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/005-console-workspace-server-readiness.md; backlog/decisions/147-conversation-archive-and-exact-resume.md (existing)
Reason: restore existing explicit handoff and reused-screen contracts; no new authority, storage or lifecycle boundary.
1. Reproduce both warm-screen launch cases with settled navigation, exact revision checks and mounted strip/card assertions.
2. Route first and replacement claims through the existing staging surface sync, and register the live-work consumer on ordinary resume with existing cancellable timers.
3. Verify targeted cold/warm handoff, ownership, suspend and resume tests; run a private native warm-stage check and independent persistence/lifecycle review.
4. Update task notes and source-handoff QA, then review and commit the bounded repair.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Warm Console returns now consume CONSOLE_LIVE_WORK and refresh first/replacement evidence through one shared staging path. Claims settle before fallible presentation, preserving owned evidence after repaint errors; empty channels leave resident evidence and notices unchanged. The existing cancellable resume timer supplies warm replay without changing explicit saved-chat ordering.

Validation: 130 targeted tests pass (155.21s), including two warm paths that failed with pending revisions before repair and two presentation-failure ownership cases. A private native four-cell dark/light, 170x48/80x24 matrix verifies eight public live-work launches plus eight conversation-source handoffs with the same draft/session, normal exit 0 and unchanged source storage. Ten databases are healthy. The expanded neighbor run has eight pre-existing suspend/roleplay-resume failures, reproduced by restoring only the production screen to original HEAD; these paths remain unqualified.

Files: chat_screen.py, live-work handoff tests, resume-consumer census, user guide, QA helpers/receipts and testing lessons. Existing ADR-005 and ADR-147 govern; no new ADR, schema, dependency, authority or CSS change. Changed tests/helpers pass Ruff and formatting; production retains 212 baseline Ruff diagnostics with zero additions and formatted changed ranges. Independent implementation review has no blocking findings.

Evidence: Docs/superpowers/qa/2026-09-16-conversation-source/README.md and verification.json. Full suite, provider generation, push and integration were not performed.
<!-- SECTION:NOTES:END -->
