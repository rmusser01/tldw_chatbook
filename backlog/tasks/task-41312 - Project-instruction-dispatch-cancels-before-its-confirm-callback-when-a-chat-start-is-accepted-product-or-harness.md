---
id: TASK-41312
title: "Project-instruction dispatch cancels before its confirm callback when a chat start is accepted: product or harness?"
status: In Progress
created_date: 2026-11-16 19:55
labels:
- follow-up
- console
- agents
- tests
priority: low
references:
- tldw_chatbook/Chat/console_project_dispatch.py
- tldw_chatbook/Chat/console_chat_start.py
- Tests/Chat/test_console_ordinary_tree.py
- https://github.com/rmusser01/tldw_chatbook/pull/3548
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the 2026-11-13..16 burn-down. TASK-40715's notes list it as "found, not fixed". Checked on dev 5e3c8b2bd4 (2026-11-16) by reading the code; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** In the ordinary-tree test host, agent runs are cancelled before they reach the model. When project instructions need a confirmation and the session's chat start has been accepted, the dispatch returns "cancel" without ever asking. If a real Console can reach that state, an agent run there is cancelled with no prompt shown. If only the test host can reach it, the host is hiding everything after that point in the tests that use it. Either way, part of TASK-41028's red `Tests/Chat/test_console_ordinary_tree.py` nodes may come from this.

**Evidence.**
- `Chat/console_project_dispatch.py:111-123` (`confirm_project_dispatch`): when the first commit returns "prompt", `:114` checks `controller._chat_start.is_accepted(session_id)` and returns "cancel" at `:115`. The confirm callback (`controller._confirm_project_instruction_dispatch`, `:116-122`) is never reached.
- `Chat/console_chat_start.py:104-107`: `is_accepted` is true while the runtime still owns an accepted target start for that session.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The task notes say whether a real Console session can reach this branch with a pending project-instruction prompt, with the evidence (a mounted reproduction, or the code path that rules it out)
- [ ] #2 If it can, the user is asked (or told why the run stopped) instead of the run being silently cancelled, pinned by a test that is red on dev; if it cannot, the ordinary-tree test host is fixed so its agent runs reach the model
<!-- AC:END -->

## Implementation Plan

ADR required: no. A test-harness fix, plus findings recorded in these notes; no product behaviour changes.

1. Trace why agent runs in the ordinary-tree host never reach the model (probe the dispatch callback on the base tree).
2. Find whether a real Console session can reach `is_accepted` with a pending consent prompt (agent chat start path), and record the code path and pinning test.
3. If unreachable, fix the shared test double so host agent runs reach the model; show red-on-base nodes go green, and the mutation check.
