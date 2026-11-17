---
id: TASK-41311
title: "A direct store.close_session while a fleet child runs leaves the turn's claim retained: does the Console's close protocol avoid it?"
status: In Progress
created_date: 2026-11-16 19:55
labels:
- follow-up
- console
- agents
priority: low
references:
- tldw_chatbook/Chat/console_chat_store.py
- tldw_chatbook/Chat/console_runtime.py
- tldw_chatbook/Chat/console_chat_controller.py
- tldw_chatbook/Agents/ordinary_tree.py
- https://github.com/rmusser01/tldw_chatbook/pull/3548
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the 2026-11-13..16 burn-down. TASK-40715 found it while testing (PR #3548 lists it under "Follow-ups to be filed"). Checked on dev 5e3c8b2bd4 (2026-11-16) by reading the code; line numbers are that tree's, and paths are under `tldw_chatbook/`.

**Why.** While testing TASK-40715, closing a session with a direct `store.close_session` while a fleet child was running left the turn's claim retained, so the turn never retired. A retained claim that has widened to OPAQUE blocks every other chat's Send ("Workspace activity is busy") until restart. The user-facing close does not call the store directly, so this may only be a test-harness shape, but nobody has checked.

**Evidence.**
- The user-facing close goes through `UI/Console_Modules/session.py:3328`, then `ConsoleRuntime.close_session` (`Chat/console_runtime.py:5886`, which drains bounded publication and turns within a grace period), then the controller's ticketed close (`Chat/console_chat_controller.py:14900-14915`). That close refuses while a native preparation or an ordinary save still owns the session, and only then calls `store.close_session`.
- `ConsoleChatStore.close_session` (`Chat/console_chat_store.py:5081`) itself does none of that.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test closes a session through the Console's own close path while a fleet child runs, and records whether the turn's claim is released once the child returns
- [ ] #2 If the claim stays retained on that path, it is released once the child returns, pinned by that test (red on dev); if not, the task notes say which step of the close protocol prevents it, and any test that closes through the store directly is changed to use the real close path
<!-- AC:END -->

## Implementation Plan

ADR required: no. It extends the existing disposal carve-out for a retained tree's storage check to a closed session; no new decision.

1. Add a test that closes a chat through the Console's own close path (the tab's impact, then `ConsoleRuntime.close_session`) while a fleet child runs, both when the child observes cancellation and when it outlives the close grace, and records whether the claim is released once the child returns.
2. If it stays retained, find the step that keeps it and fix that step; show the test red on base and the mutation check.
