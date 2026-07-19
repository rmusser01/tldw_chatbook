---
id: TASK-195
title: >-
  Fix pre-existing failure: console workspace-context active-conversation marker
  test
status: Done
assignee: []
created_date: '2026-07-12 12:44'
updated_date: '2026-07-16 16:45'
labels:
  - tests
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_console_workspace_context_rail.py::test_console_workspace_context_syncs_active_conversation_marker fails on dev (asserts '> Planning thread' marker in the rail text; reproduced at 75e6987c and 89de6554 during the 2026-07 UAT remediation, masked by the intentionally-cancelled CI). Root-cause whether the marker sync or the test expectation drifted.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The test passes on dev for the right reason (behavior fixed or expectation honestly updated with rationale),No other workspace-context rail behavior regresses
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the failure and capture the actual rendered text vs expected marker.
2. git log -S on the marker string and GLYPH_ACTIVE to find when/why the marker character changed.
3. Determine whether the dual active-row artifact is a real reachable regression or test-drift, using existing native-session/browser-row tests as ground truth.
4. Fix the test (glyph + realistic active-session simulation) or the product code, whichever the evidence supports.
5. Run the full rail file + persistent-rails file to confirm no regressions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: two independent, later, unrelated commits invalidated the test's expectation (no product regression). (1) 0cb02fa5 replaced the literal "> " active-row marker with GLYPH_ACTIVE ("▸ ") from the new shared Console glyph language, without updating this test. (2) a667ffbd (grouped conversation browser, added long after this test existed) introduced a second, independently-selected native-session placeholder row alongside the workspace-membership row; the old test drove the "active conversation changed" signal via a synthetic ChatSessionData passed to sync_shell_bar_from_session_data that isn't bound to any real native session, so the still-active placeholder session kept its own active glyph while the membership row also lit up -- two "active" rows for one conversation. Confirmed this dual-marker state is not reachable via real user action: sync_shell_bar_from_session_data's only production caller depends on ChatTabContainer, which Console (ChatScreen) never mounts (only the legacy Chat_Window does). Verified via Tests/UI/test_console_native_chat_flow.py's "shared-open-chat" case that native rows must select purely by session.id == active_session_id (not by conversation-id match, since two sessions can share one persisted_conversation_id across workspaces) -- prototyping a fix in _native_console_browser_rows broke that invariant, confirming this is test drift, not a product bug.

Fix: rewrote the test to (a) assert GLYPH_ACTIVE instead of the stale "> " literal, and (b) bind the active native session via store.restore_persisted_session(...) -- the same call the real "resume saved conversation" flow uses -- before syncing, so native and membership rows merge into one, exactly as production does. Assertion now checks exactly one rendered conversation row carries the active glyph and it's the expected row, guarding against reintroduction of the dual-marker artifact.

Verification: Tests/UI/test_console_workspace_context_rail.py (35 tests) all green including the previously-failing test; Tests/UI/test_console_persistent_rails.py (36 tests) all green; the shared-open-chat invariant test in test_console_native_chat_flow.py still passes unmodified (no production code touched). Full report: .superpowers/sdd/task-195-report.md.
<!-- SECTION:NOTES:END -->
