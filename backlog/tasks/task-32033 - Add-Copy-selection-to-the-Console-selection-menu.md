---
id: TASK-32033
title: Add Copy selection to the Console selection menu
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 05:16'
updated_date: '2026-09-08 05:37'
labels:
  - console
  - clipboard
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users selecting part of a Console message need an explicit way to copy that text without relying on the terminal copy shortcut.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Copy selection is available first in the selection menu for all selectable row types regardless of run state.
- [x] #2 Mouse and keyboard activation copy the exact highlighted text through the existing app clipboard path without quote prefixes or the 4000-character quote cap.
- [x] #3 Copying closes the menu, clears the text highlight, restores focus, and preserves the composer draft and message selection.
- [x] #4 An empty or vanished selection leaves the clipboard unchanged and dismisses safely.
- [x] #5 Targeted selection tests and static checks pass, and the user guide explains the action and terminal clipboard limitation.
- [x] #6 In short transcripts the menu stays within its owner and every action remains reachable by scrolling or keyboard navigation, including ANSI color mode.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing coverage for mouse and keyboard Copy selection, full text from plain/Markdown/diff selections, and empty or removed selections. Update existing menu-order and compact-layout expectations.
2. Add the menu action through the existing menu-to-transcript message path. Allow row selection readers to return uncapped text for copying while retaining their existing capped defaults for quoting and previews.
3. Reuse the app clipboard operation and menu dismissal/focus lifecycle; preserve the draft and whole-message selection. Document the action and terminal clipboard requirement.
4. Keep the additional menu row inside short transcript bounds with scrolling when compact actions cannot all fit; verify keyboard reachability and ANSI rendering.
5. Run the targeted selection suites, scoped lint/format checks, and a production Console rendering/interaction check; self-review the diff and record evidence.

ADR required: no

ADR path: backlog/decisions/068-console-text-selection-and-annotations.md

Reason: extends the existing row-selection and menu action design without changing ownership, persistence, clipboard transport, or keybinding policy. ADR-031 continues to apply.

Task ID hygiene: the CLI assigned 32032, which was already held by another worktree. Renumbered this newly created task to 32033 after checking local tasks, 176 remote refs, and 11 worktrees.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added Copy selection as the first menu action. Mouse and keyboard activation use the existing app clipboard operation with the exact selected text; plain, Markdown-source, and diff-line selections can now opt out of the quote cap while existing quote/preview callers retain capped defaults. Copy clears the text highlight/menu, restores focus, and preserves drafts and whole-message selection. Empty or removed selections leave the clipboard untouched.

The extra action exposed a seven-row transcript overflow. The menu now scrolls within the owner after its compacting pass, with every action reachable in normal and ANSI modes. The independent review confirmed that fix and found no remaining actionable issues.

Files: `Widgets/Console/console_selection_menu.py`, `Widgets/Console/console_transcript.py`, the new `Tests/UI/test_console_selection_copy.py`, existing selection-menu tests, `Docs/User_Guide/console/text-selection-and-feedback.md`, and `backlog/docs/lessons-textual.md`.

Verification:
- Nine expected failures before implementation demonstrated the missing action and keyboard copy behavior.
- Final targeted run: **174 passed** across `test_console_selection_copy.py`, `test_console_selection_menu.py`, `test_console_keyboard_selection.py`, `test_console_selection_rows.py`, `test_console_selection_app_smoke.py`, and `test_console_selection_end_to_end.py` (140.24 seconds). The existing requests dependency-version warning remains.
- Production ChatScreen render/interaction probes: two additional passing cases, with user and assistant menus contained in the transcript and Copy selection focused. SVG/PNG captures are in `/private/tmp/chatbook-copy-selection-32033/`.
- Ruff passes for the menu and both changed test files. All edited Python regions pass formatting checks. The transcript's 16 pre-existing lint findings and unrelated formatting drift are unchanged; a before/after diagnostic comparison found no added findings. Scoped whitespace checks pass.
- Clipboard assertions use the real Textual app clipboard in the headless harness. Linux desktop clipboard delivery was not exercised; the user guide explains the existing OSC 52 requirement and terminal-native selection fallback.

ADR check: direct extension of [ADR-068](../decisions/068-console-text-selection-and-annotations.md); no new ADR, storage change, clipboard backend, or shortcut. ADR-031 remains applicable. No full-suite run was performed, per repository policy.
<!-- SECTION:NOTES:END -->
