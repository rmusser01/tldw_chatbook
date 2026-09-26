---
id: TASK-33001
title: 'Model config P1: root-cause fixes, no layout change'
status: To Do
assignee: []
created_date: '2026-09-26 11:47'
labels:
  - model-config-redesign
  - phase-1
  - console
  - settings
  - readiness
dependencies: []
references:
  - 'backlog/docs/spec-2026-09-26-model-config-redesign.md'
  - 'qa/model-config-ux-review-2026-09-26/judge-synthesis.md'
  - 'qa/model-config-ux-review-2026-09-26/verified-claims.md'
  - 'qa/model-config-ux-review-2026-09-26/backlog-adr-check.md'
  - 'qa/model-config-ux-review-2026-09-26/report.md'
  - 'backlog/decisions/095-conversation-owned-console-generation-settings.md'
  - 'backlog/decisions/006-provider-aware-generation-settings.md'
  - 'backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md'
  - 'Tests/Architecture/test_screen_size_ratchet.py'
  - 'Tests/Architecture/test_module_size_ratchet.py'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 1 of the model-configuration redesign 'Switchboard with field truth' (backlog/docs/spec-2026-09-26-model-config-redesign.md §8; qa/model-config-ux-review-2026-09-26/judge-synthesis.md §4). Ships as one PR. It fixes the verified root causes under the model-configuration findings before any surface is redrawn, so later phases build on correct data and behaviour.

What it closes (verified at c4225b5d38, re-checked at this worktree's HEAD):
- C7(a): a provider switch borrows the global default model. resolve_effective_chat_configuration ranks chat_defaults.model above the target provider's own model (Chat/console_session_settings.py:1269-1275).
- C8(1), data only: three copies of field support treat every sampler as universal (Chat/console_chat_controller.py:774-802, Chat/console_settings_defaults.py:344, UI/Screens/settings_screen.py:12596), so Anthropic shows four samplers the request silently drops (Chat/Chat_Functions.py:281-300, :1431-1435).
- C3, double append: Provider Test bakes stored evidence into the result (settings_screen.py:15143-15150) and appends fresh evidence again (:15458-15469).
- C8(3): F6 on Settings only toasts, because SettingsScreen (settings_screen.py:2731) has no pane handler and the app falls back to a notice (app.py:20175-20186).
- C1(b) and the C1 first-run gap: the task-177 refresh never converges an untouched chat whose provider already reads Ready (UI/Console_Modules/session.py:3786-3791, :3814), and only the 'Start chatting' exit stages the first-chat handoff (UI/Wizards/FirstRunSetupWizard.py:9936). Owner decision D1, recorded in the ADR-095 amendment of 2026-09-26 (drafted in this worktree with the spec), removes those readiness gates. The spec moved this here from the Chat settings layout phase because it is a root fix with no layout change.
- Palette commands switch provider by a title-cased raw key and never show the model (app.py:1493-1603).
- The review's minor defects on these surfaces that need no layout change (report.md 'Minor observations').

Absorbs TASK-14812: its AC#6 ('cannot retain a model from the previous provider') has regressed, and its In Progress status is stale because every AC is checked.

No CSS, token or widget geometry changes in this phase. Constraints: chat_screen.py has 32 lines of headroom (Tests/Architecture/test_screen_size_ratchet.py:85), console_settings_modal.py has none (Tests/Architecture/test_module_size_ratchet.py:68), and ADR-097 ratchets never rise. Every test and live check that touches configuration uses a scratch TLDW_CONFIG_PATH, never the real profile.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Switching provider on any Console surface never leaves another provider's model in the draft
- [ ] #2 The Console draft rebase, the model-default writer and Settings model defaults all give the same answer on which generation fields apply, and that answer matches what the request actually forwards
- [ ] #3 Re-running Provider Test on an unchanged draft shows each endpoint fact once
- [ ] #4 F6 and Shift+F6 cycle focus through the Settings panes
- [ ] #5 An untouched open Console chat follows newly saved defaults after a Settings save or any completing first-run exit, whatever its readiness; chats with messages, edits or user work keep their settings
- [ ] #6 No command-palette entry switches provider without its model
- [ ] #7 The review's minor model-config defects that need no layout change are fixed: picker count copy, the silent result cap, the URL display break in native terminals, credential keys written for keyless providers, and values hidden on focus
- [ ] #8 The phase changes no CSS, design token or widget geometry
- [ ] #9 chat_screen.py stays within its screen-size budget, console_settings_modal.py does not grow, and no ADR-097 ratchet value rises
- [ ] #10 TASK-14812 is closed as Done with a note naming this phase's fix
- [ ] #11 Docs/User_Guide pages updated (settings.md, console.md), including their Verified-against stamps
- [ ] #12 ./scripts/preflight.sh passes
<!-- AC:END -->
