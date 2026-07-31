---
id: task-1580
title: 'Settings: footer s/r hints gated to draft-model categories'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-31'
labels:
  - settings
  - ux
  - rescore-p1
dependencies: []
priority: high
---

## Description (the why)

The 2026-07-31 Settings critique rescore (29 → 30/40) found the footer
advertising "s save category | r revert category" on every category,
including the six read-only view pages (Writes allowed: No), the autosaving
Splash Screen, and immediate-apply Workspaces — where both keys answer with
an informational toast. The footer is the keyboard contract; advertising
inert keys breaks it. task-1564 already gates the `t` hint the same way.

## Acceptance Criteria (the what)

- [x] Categories outside the guided draft model (read-only pages, Splash,
      Workspaces, Theme) do not advertise s/r in the footer
- [x] Every guided-mutation category still advertises s/r
- [x] Testable non-draft categories (e.g. Privacy & Security) keep `t`
      without gaining s/r
- [x] Existing footer behaviors (Esc-prefix under input focus, RAG
      accelerators) unchanged
- [x] Live verification at the real surface confirms the gated footer

## Implementation Plan (the how)

1. RED tests: non-draft categories drop s/r; guided categories keep them;
   Privacy & Security shows t only.
2. Gate `_footer_shortcut_entries` on GUIDED_SETTINGS_MUTATION_CATEGORIES,
   mirroring the TESTABLE_SETTINGS_CATEGORIES gate for `t`.
3. Update the two tests that pinned the old always-on contract.
4. Live tmux verification.

## Implementation Notes

Added a s/r gate in `_footer_shortcut_entries` keyed on
`GUIDED_SETTINGS_MUTATION_CATEGORIES` — the exact set where
`action_settings_save_category` acts instead of toasting. On categories with
no active settings keys (e.g. Theme, Overview) the settings registration is
empty and the footer falls back to the app default hints, which is honest.
Updated `test_settings_registration_updates_the_screens_own_footer` (now
switches to a guided category first, re-querying the footer after the
recompose) and the Overview assertion in
`test_settings_rag_profile_region.py`. Live-verified in tmux: Theme shows
default hints, Storage shows s|r|t, Console Behavior shows s|r.
Files: `tldw_chatbook/UI/Screens/settings_screen.py`,
`Tests/UI/test_settings_configuration_hub.py`,
`Tests/UI/test_screen_footer_hints.py`,
`Tests/UI/test_settings_rag_profile_region.py`.
