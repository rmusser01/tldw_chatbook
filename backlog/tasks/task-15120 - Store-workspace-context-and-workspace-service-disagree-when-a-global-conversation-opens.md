---
id: TASK-15120
title: >-
  Store workspace context and workspace service disagree when a global conversation opens
status: Done
assignee: []
created_date: '2026-08-11 05:00'
labels:
  - console
  - workspaces
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Exposed by task-14920. `test_console_browser_selecting_global_persisted_row_preserves_active_workspace` had been failing with `textual.pilot.OutOfBounds` — the click never landed, so the test never reached its own assertions. Once the click was repaired, the test failed on what it was actually written to pin:

```
assert after.workspace_id == "ws-a"                              # PASSES (service)
assert store.workspace_context.active_workspace_id == "ws-a"     # FAILS: 'global'
```

With `ws-a` active, opening a **global-scoped** persisted conversation leaves the workspace *service* reporting `ws-a` while the *store's* workspace context flips to `global`. Two sources of truth for "which workspace am I in" now disagree, deterministically (3/3 in isolation).

Which one is right needs a product ruling rather than a guess, and the test itself asserts both halves of the tension: it wants the store context to stay `ws-a` **and** the session's own `workspace_id` to be the global sentinel. Either the store context should track the active workspace (and opening a global conversation must not move it), or it should track the open conversation's scope (and the test's expectation is stale) — but it cannot be both, and today the two objects answer differently.

The test is marked `xfail(strict=True)` pointing here, so the divergence stays visible and a fix flips it loudly rather than passing unnoticed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 A ruling is recorded on what `store.workspace_context.active_workspace_id` means when the open conversation is globally scoped
- [x] #2 The store and the workspace service cannot report different active workspaces for the same state
- [x] #3 The `xfail(strict=True)` on `test_console_browser_selecting_global_persisted_row_preserves_active_workspace` is removed and the test asserts the ruled behaviour
<!-- AC:END -->

## Implementation Plan

1. Record the owner ruling; identify which layer violates it
2. Find why the follow-the-conversation seam skips global conversations
3. Map "global" onto the registry's own representation; mutation-check

## Implementation Notes

**Owner ruling (2026-08-13): the workspace context FOLLOWS the conversation.**
A user keeps conversations open across multiple workspaces at once; selecting
one switches the context to that conversation's workspace. So the STORE's
behaviour (context flipping to "global") was right and the SERVICE staying on
ws-a was the bug -- the pinning test's "preserves active workspace" premise
was wrong per the ruling, and it is rewritten, not just unmarked.

The seam already followed the ruling for real workspaces
(`_set_active_workspace_for_console_session`); it simply early-returned for
global conversations, leaving the previous workspace -- and its capabilities
-- active under a global conversation.

**One design correction along the way**: the first fix represented global as
"no active workspace" (registry -> None). That fights a deliberate design --
`_current_console_workspace_context` floors every read through
`ensure_default_workspace()`, which resurrects the built-in Default whenever
no workspace is active, precisely so a concrete-but-capability-less workspace
always exists. So the registry's stable representation of "no explicit
workspace" IS the Default workspace, and a global conversation now lands
there (clear + ensure), while the store's context reads "global". The two
layers agree by design rather than by accident: store="global" <->
registry=Default-with-no-capabilities.

Added `LocalWorkspaceRegistryService.clear_active_workspace()` as the
primitive. Mutation-checked: restoring the early-return turns the rewritten
test red. Both directions verified: global->Default (rewritten test) and
ws->ws (`test_activate_native_console_session_realigns_active_workspace`,
unchanged). 309 passed in the native module -- its last xfail is gone; 305
passed across session-controller/Workspaces/workbench modules.

Modified: `tldw_chatbook/Workspaces/registry_service.py`,
`tldw_chatbook/UI/Console_Modules/workspace.py`,
`Tests/UI/test_console_native_chat_flow.py`.
