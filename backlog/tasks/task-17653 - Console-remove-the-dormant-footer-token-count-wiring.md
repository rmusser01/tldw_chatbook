---
id: TASK-17653
title: 'Console: remove the dormant footer token-count wiring'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-17'
labels:
  - console
  - cleanup
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`AppFooterStatus` mounts `#footer-token-count` on Console with `show_token_count=True`, but the widget is dormant: its only writer (`update_token_count_display` -> `update_chat_token_counter`) gates the footer branch on `not app._use_screen_navigation`, which is never true in screen-navigation mode. It is one flag flip away from rendering a full token readout one row below the cost chip's compact "2.7k tok" — a silent future duplicate.

Owner decision (2026-08-17): the cost chip is the single token/cost surface on Console; delete the never-taken path rather than leaving the latent duplicate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 The dormant footer token-count path for Console is removed (dead writer branch and/or the Console mount flag), with no visible change to the current default footer
- [x] #2 The cost chip remains the only token/cost readout on Console, and a test pins that the footer token counter cannot appear there
- [x] #3 Non-Console footer features (word count, DB size indicator, key hints, responsive reflow) are unaffected
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: mounted-Console pin — no write may reveal `#footer-token-count` (including the db_status "Token count error" path); word count unaffected.
2. Gate `AppFooterStatus.update_token_count` on `show_token_count` (the single choke point — the method used to reveal unconditionally); no screen arms the flag any more (base_app_screen chat -> False).
3. Delete the dead legacy tab-mode footer writes and the reader-less `app.current_token_count`/`token_count_pending` stashes in chat_token_events (screen navigation is hard-set True); remove the orphaned ScreenStackError import (test-grep confirmed not load-bearing).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The audit's "one flag flip away" was optimistic: `update_token_count` revealed the widget UNCONDITIONALLY (`display = bool(text)`); the `show_token_count` flag only gated responsive reflow, so db_status_manager's error path could surface "Token count error" on any screen. The gate now lives in the method itself, `base_app_screen` no longer arms the chat screen, and both dead legacy write branches (plus the stashes with zero readers, verified by repo+test grep) are deleted from `chat_token_events.py`. New pin suite `Tests/UI/test_footer_token_counter_retired.py` (RED-first on the reveal); 181 token-related tests + the two dedicated suites (test_token_display_limit, test_footer_token_dirty_gate) green; ruff clean.

Files: `tldw_chatbook/Widgets/AppFooterStatus.py`, `tldw_chatbook/UI/Navigation/base_app_screen.py`, `tldw_chatbook/Event_Handlers/Chat_Events/chat_token_events.py`, `Tests/UI/test_footer_token_counter_retired.py` (new).
<!-- SECTION:NOTES:END -->
