---
id: TASK-24303
title: >-
  Nine modules crossed onto the first-paint leg and the ui-ready ratchet is breached
status: Done
assignee: []
created_date: '2026-08-28 23:30'
labels:
  - performance
  - guards
  - dev-red
  - boot
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Performance/test_ui_ready_module_census.py` measures 972 resident `tldw_chatbook` modules
at `_ui_ready` against a ratchet pin of 970 on pristine dev `3a3383123e`. It is also non-deterministic:
three of four consecutive runs are red, because boot work lands concurrently with `_ui_ready` and
the count moves between runs.

The nine additions since the pinned snapshot cluster into three deferrable features:

  Agents.raw_shell_tool_provider, Agents.virtual_cli_provider,
  Tools.git_tool_impls, Tools.local_tool_impls, Tools.virtual_cli_impls
      -- pulled by module-scope imports in Chat/console_chat_controller.py:277,282,
         and only needed once an agent run actually executes a tool

  TTS.legacy_request_builder, TTS.text_processing
      -- pulled through Subscriptions/briefing_audio.py:218,220, itself imported at module
         scope by UI/Screens/watchlists_collections_screen.py:41

  Workspaces.change_review_consent, Workspaces.change_review_finalization

Per ADR-097 the ratchet constant does not rise; the cost is deferred off the path or shed
elsewhere in the same change. The non-determinism is a second, separable defect: a guard that
passes one run in four will be dismissed as flaky the first time it blocks somebody.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 The ui-ready module census passes on an otherwise-unmodified dev checkout, with the ratchet constant unchanged
- [x] #2 The census returns the same count on repeated runs of an unchanged tree, so a breach is distinguishable from a race
- [x] #3 Shell, virtual-CLI and git tool providers are imported on first tool use rather than at Chat first paint
- [x] #4 The TTS request builder and text processing modules are absent from the first-paint closure
- [x] #5 Each deferral is re-measured after it lands, because the import-parent tracer records only the first importer and its attribution is an upper bound
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Defer the shell/virtual-CLI providers in `console_chat_controller`.
2. Re-measure; defer in the second importer too if the cost only relocated.
3. Defer the TTS chain at `briefing_audio`, not at the screen.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Deferred three import chains off the first-paint leg. All annotations in the
affected modules are PEP 563 strings, so only genuine runtime uses needed
moving; the rest became `TYPE_CHECKING` imports.

**The instructive part: the first attempt made a different guard WORSE.**
Deferring in `Chat/console_chat_controller` alone took the pre-import payload
from 379,497 to 381,696 LOC and turned that ratchet red. Bisecting my own edit
rather than theorising showed why: the modules stopped being resident before
the registry walk and started being CHARGED to it. The cost had relocated, not
gone. `UI/MCP_Modules/mcp_workbench` imports the same two providers at module
scope, so both importers had to defer before the cost actually left.

The TTS pair was deferred inside `Subscriptions/briefing_audio` itself rather
than at `watchlists_collections_screen`, which uses five names from it -- one
file instead of many, and it helps every importer.

**Measured, on pristine dev vs the branch:**
- ui-ready census **972/970 BREACHED (3 of 4 runs) -> 964/970, headroom 6**, and
  deterministic: 4 consecutive runs, same result.
- pre-import payload 379,497/380,000 (headroom 503) -> **378,930 (headroom 1,070)**,
  489 modules vs 491.
- The ratchet constants were NOT raised (ADR-097).

Files: `Chat/console_chat_controller.py`, `UI/MCP_Modules/mcp_workbench.py`,
`Subscriptions/briefing_audio.py`.
<!-- SECTION:NOTES:END -->
