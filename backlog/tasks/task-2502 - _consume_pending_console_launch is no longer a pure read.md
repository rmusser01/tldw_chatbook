---
id: TASK-2502
title: _consume_pending_console_launch is no longer a pure read
status: To Do
assignee: []
created_date: '2026-08-04 21:52'
labels:
  - console
  - rag
  - cleanup
dependencies: []
priority: low
---

## Description

`ChatScreen._consume_pending_console_launch` (`tldw_chatbook/UI/Screens/chat_screen.py`, ~lines 3655-3709) has two branches. The resident-launch branch (added by PR-T1's C1 fix, `fe9b7f89f`) is a genuine seam: it calls `_supersede_resident_console_launch_from_store()`, which routes any newer store entry through `_stage_console_library_rag_launch` so every mounted surface (strip, tray, rail, control bar, workspace context, settings estimate, Inspector) stays in sync before the value is claimed.

The other branch — the non-resident path at ~lines 3691-3709 — does not go through that staging seam. It claims directly from `store.claim(HandoffChannel.CONSOLE_LIVE_WORK)` and assigns `self._pending_console_launch_context = claim.value` itself, with no call to `_stage_console_library_rag_launch` and therefore no surface sync. It makes a launch live (the method's name and most callers still treat it as a read: `compose_content()` calls it on its first line expecting a value back, not a side effect), but does so through a different, narrower code path than every other place in the module that claims a handoff.

Today this is not reachable as a defect: every production "Use in Console" stager (Library search/RAG, media, notes, conversations) runs on a screen other than Console and navigates there afterward, so by the time this method's non-resident branch would run, the screen is still mid-`compose_content()` and the value it returns *does* get rendered by the normal compose path that follows it — the missing surface sync is masked by the fact that compose is about to build the DOM from the return value anyway. This is a call-ordering coincidence, not a structural guarantee: if a future stager ever posts to `HandoffChannel.CONSOLE_LIVE_WORK` while a Console screen instance is already mounted and not re-composing (e.g. a background/async stager, or a future entry point that doesn't navigate), the non-resident branch would claim and assign the launch with no surface refresh — reopening the same "invisible claim" shape PR-T1's C1 fix (task-2372's follow-up) closed for the resident branch.

## Acceptance Criteria

- [ ] The non-resident branch of `_consume_pending_console_launch` routes its claim through the same staging seam (`_stage_console_library_rag_launch` or equivalent) used by the resident branch, so a claim from either branch always syncs every mounted surface
- [ ] `_consume_pending_console_launch`'s docstring/name reflects that the method can have a side effect (staging a launch), not only that it returns one, or the side-effecting part is factored out so the method itself is a pure read
- [ ] A regression test drives the non-resident branch while a Console screen instance is already composed and mounted (not mid-`compose_content()`), asserting the claimed launch is reflected on the strip/tray/rail without requiring a subsequent compose to paint it
- [ ] The existing behavior for every currently-shipping stager (Library search/RAG, media, notes, conversations — all of which navigate to Console after staging) is unchanged
