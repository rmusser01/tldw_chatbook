---
id: TASK-25709
title: Console conversation action menu — Escape and click-outside dismissal
status: Done
assignee:
  - '@Robert'
created_date: '2026-08-31 02:36'
updated_date: '2026-08-31 02:56'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Context rail conversation action menu (TASK-23200) can only be closed by choosing an action or pressing Escape while focus is still inside it. One click anywhere else strands it on screen. Extend the ADR-068 dismiss contract: Escape dismisses it even when focus has left the menu, and any click outside the menu folds it with no side effects.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Click outside the open menu (composer, transcript, rail) removes it without dispatching its actions
- [x] #2 Click inside the menu (buttons, border, padding) does not dismiss it
- [x] #3 Escape with focus outside the menu (e.g. composer) dismisses it and leaves focus where it is
- [x] #4 Escape with focus inside the menu still steps back from a submenu before closing
- [x] #5 Opening another row's menu still replaces the open one
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Tests first in Tests/UI/test_console_conversation_action_menu.py: click-outside dismisses, click-inside survives, stranded-Escape dismisses, regressions for submenu step-back and replace-on-second-asterisk.\n2. Menu widget (console_conversation_action_menu.py): WeakSet registry + conversation_action_menus_on_screen(); dismiss_menu(restore_focus=True) threading a flag onto ConversationActionMenuDismissed; registry-based dismiss_conversation_action_menus(screen, *, restore_focus=True).\n3. Screen (chat_screen.py): ancestor-walk guard by DOM id for in-menu clicks; outside-click path dismisses conversation menus with restore_focus=False (lazy import, ADR-097 boot ratchet); Escape actions (composer-home + collapsed-composer) gain the guard after the slash-popup check, restore_focus=False; replace-on-open passes restore_focus=False.\n4. Targeted pytest runs; self-review; task notes + AC check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added the two missing ADR-068 dismissal paths to the Context rail's conversation
action menu (TASK-23200), plus a latent crash fix its tests exposed.

- **Click-outside**: the screen's per-press dismissal pass
  (`_dismiss_console_selection_menus_outside_transcript`) now also folds conversation
  menus — guarded by a DOM-id ancestor-walk entry so clicks on the menu itself
  (buttons, border, padding) never dismiss it mid-press. The menu list comes from a
  new `WeakSet` registry (`conversation_action_menus_on_screen`) instead of a DOM
  query, keeping the TASK-21119 per-press rule; the module import stays lazy so the
  ADR-097 `_ui_ready` ratchet is untouched (census re-run green).
- **Escape with focus elsewhere**: both Escape consumers
  (`action_focus_console_composer_home`, `action_expand_collapsed_console_composer`)
  and the two loop-scoped Escape fallbacks (hands-free, realtime) gained the guard
  after the existing slash-popup check, mirroring `_dismiss_console_command_popup`.
  In-menu Escape behavior is unchanged (submenu step-back first).
- **Focus semantics**: Textual 8.2.8 focuses the clicked widget before the press
  bubbles (`Screen._forward_event`), so outside-click and stranded-Escape dismissals
  skip the opener focus-restore via a `restore_focus` flag on
  `ConversationActionMenuDismissed`; the menu's own Escape still restores the opener.
  Recorded in `backlog/docs/lessons-textual.md`.
- **Crash fix (pre-existing)**: re-pressing a row's asterisk while its menu was open
  crashed the app with `DuplicateIds` — `remove()` only schedules the prune, and the
  remount raced it. The open path is now async and awaits each open menu's detachment
  through a memoized single-shot removal awaitable (`await_detachment`/`_detach`).
  Same trap as the transcript selection menu's documented lesson.
- Tests: 4 new cases in `Tests/UI/test_console_conversation_action_menu.py` (outside
  click, chrome click, stranded Escape, replace-not-stack); 12/12 pass there, 162
  pass across the seven seam-adjacent suites (routing, rail, actions model,
  hands-free, composer draft, selection-dismissal perf). Ruff clean. The two
  Architecture ratchet failures (`wave6_closeout_inventory`,
  `review_selection_controller_boundary`) are pre-existing on clean origin/dev —
  verified by stashing this change and re-running.
- ADR check: no new ADR — direct application of ADR-068's dismiss contract; ADR-097
  (boot ratchet) and ADR-068 linked here. Files: `console_conversation_action_menu.py`,
  `chat_screen.py`, the test file, and the lessons entry.
<!-- SECTION:NOTES:END -->
