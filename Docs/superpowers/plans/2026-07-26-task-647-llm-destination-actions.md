# LLM Destination Action Ownership Implementation Plan (TASK-647)

**Status:** Implemented and reconciled with the production Lab frame on current
`dev`.

**Goal:** Make every visible Models action truthful and destination-owned,
remove the unsupported custom Transformers launch UI, and delete the dead
`TldwCli` button dispatcher and root LLM view state.

**Architecture:** The registered production `LLMScreen` owns the Lab Models
rail. Its deferred, real `LLMManagementWindow` body owns an allowlisted action
registry and the active body view. Rail presses set the mounted body's
`active_view`. UI lookup stays inside the mounted body; app-owned process
claims and handles cross recomposition safely.

**Backlog:** [TASK-647](../../../backlog/tasks/task-647%20-%20Restore-LLM-destination-actions-and-retire-the-dead-app-button-dispatcher.md)

**Specification:** [TldwCli Reactive State Decomposition Design](../specs/2026-07-26-tldwcli-reactive-state-decomposition-design.md)

**ADR required:** yes

**ADR path:** `backlog/decisions/033-application-session-state-ownership.md`;
`backlog/decisions/011-chatbook-workbench-ui-system.md`

**Reason:** The accepted ADRs assign view state and actions to the mounted
destination and process lifecycle to the application.

## Implementation

1. Resolve the optional macOS speech default with a side-effect-free installed
   package probe so importing a headless production app never imports native
   runtimes.
2. Register supported provider/model actions on `LLMManagementWindow` with the
   normalized `(window, app, event)` contract; unknown actions are ignored and
   failures produce bounded metadata-only recovery.
3. Keep Models navigation in `LLMScreen`'s production Lab rail and bind it to
   the deferred mounted body's `active_view`.
4. Remove the unsupported custom Transformers server-launch controls while
   retaining supported model directory, list, download, and output behavior.
5. Remove `TldwCli.llm_active_view`, its watcher, `button_handler_map`,
   `_build_handler_map`, the obsolete `llm_nav_events` module, root log
   callbacks, and the duplicate `ServerWorkerHandler`.
6. Keep six provider process lifecycles app-owned with identity claims,
   generation-safe publish/clear, bounded off-loop stop/reap, and truthful
   remount controls.
7. Keep command arguments, credentials, raw subprocess/API output, and
   exception payloads out of persistent diagnostics and bounded failure UI.

## Verification

- Test mounted behavior only with a normal production `TldwCli()`,
  `app.run_test()`, the registered `LLMScreen`, its Lab rail, and its real
  deferred `LLMManagementWindow`.
- Wait for the deferred body after first mount and every production screen
  recomposition before asserting ownership or controls.
- Use direct function tests for process termination, command construction,
  sanitization, and lifecycle claim operations.
- Cover the action census, removed controls, destination-local lookup,
  generation ownership, stale completion rejection, duplicate starts,
  bounded stop behavior, modal/recomposition ownership, and private diagnostic
  containment.
- Run `Tests/ProductionApp/test_llm_destination_actions.py` and the application
  ownership guard as part of the integrated state gate.

Tests must not use a custom or simplified application, surrogate screen,
`MagicMock` app, `SimpleNamespace` app, unbound `TldwCli` method, or
`object.__new__(TldwCli)`.
