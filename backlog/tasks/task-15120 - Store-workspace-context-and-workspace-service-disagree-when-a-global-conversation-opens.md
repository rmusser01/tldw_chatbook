---
id: TASK-15120
title: >-
  Store workspace context and workspace service disagree when a global conversation opens
status: To Do
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
- [ ] #1 A ruling is recorded on what `store.workspace_context.active_workspace_id` means when the open conversation is globally scoped
- [ ] #2 The store and the workspace service cannot report different active workspaces for the same state
- [ ] #3 The `xfail(strict=True)` on `test_console_browser_selecting_global_persisted_row_preserves_active_workspace` is removed and the test asserts the ruled behaviour
<!-- AC:END -->
