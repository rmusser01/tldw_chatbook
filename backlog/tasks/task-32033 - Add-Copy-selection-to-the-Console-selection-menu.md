---
id: TASK-32033
title: Add Copy selection to the Console selection menu
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-08 05:16'
updated_date: '2026-09-08 06:47'
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
- [x] #5 Targeted selection tests and static checks demonstrate no new failures relative to dev, and the user guide explains the action and terminal clipboard limitation.
- [x] #6 In short transcripts the menu stays within its owner and every action remains reachable by scrolling or keyboard navigation, including ANSI color mode.
- [x] #7 The previously failing keyboard-anchor and two feedback-persistence tests pass while retaining real menu-containment and SQLite-write coverage.
- [x] #8 An open menu adapts to terminal and transcript resizing, restores its full action list and feedback hint when space returns, and preserves focus.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing coverage for mouse and keyboard Copy selection, full text from plain/Markdown/diff selections, and empty or removed selections. Update existing menu-order and compact-layout expectations.
2. Add the menu action through the existing menu-to-transcript message path. Allow row selection readers to return uncapped text for copying while retaining their existing capped defaults for quoting and previews.
3. Reuse the app clipboard operation and menu dismissal/focus lifecycle; preserve the draft and whole-message selection. Document the action and terminal clipboard requirement.
4. Keep the additional menu row inside short transcript bounds with scrolling when compact actions cannot all fit; verify keyboard reachability and ANSI rendering.
5. Regenerate the bundled widget stylesheets and use the production stylesheet harness on current dev. Run the targeted selection suites, scoped lint/format checks, and a production Console rendering/interaction check; self-review the diff and record evidence. Compare any existing test failures against pristine dev.
6. Per the requested follow-up, trace and repair the three inherited failures before rebasing or merging. Use the existing production-style keyboard harness, sample geometry after selection entry, and wire the real workspace registry into the two real-database stores. Preserve or strengthen the behavioral assertions and rerun the complete selection slice. This is test-harness repair within existing runtime and persistence contracts; no new ADR is required.
7. Address Qodo's two findings: annotate/document the public compose override, and remeasure the menu when its owner's bounds change. Exercise shrink/grow cycles from terminal and transcript changes in normal and ANSI modes; restore normal sizing without losing focus. Reuse Textual's layout notification and the existing clamp; no new ADR is required.
8. Document the empty-selection and capped/full-text return contracts on all three modified selection getters, as requested by Qodo's follow-up review. This is documentation only; no new ADR or behavior test is required. Rebase onto the updated dev and repeat the targeted integration and preflight checks.

ADR required: no

ADR path: backlog/decisions/068-console-text-selection-and-annotations.md

Reason: extends the existing row-selection and menu action design without changing ownership, persistence, clipboard transport, or keybinding policy. ADR-031 continues to apply.

Task ID hygiene: the CLI assigned 32032, which was already held by another worktree. Renumbered this newly created task to 32033 after checking local tasks, 176 remote refs, and 11 worktrees.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added Copy selection as the first menu action. Mouse and keyboard activation use the existing app clipboard operation with the exact selected text; plain, Markdown-source, and diff-line selections can now opt out of the quote cap while existing quote/preview callers retain capped defaults. Copy clears the text highlight/menu, restores focus, and preserves drafts and whole-message selection. Empty or removed selections leave the clipboard untouched.

The extra action exposed a seven-row transcript overflow. The menu now scrolls within the owner after its compacting pass, with every action reachable in normal and ANSI modes. The independent review confirmed that fix and found no remaining actionable issues.

The consolidated widget stylesheets are regenerated from the menu source. The new Copy tests and the existing feedback and keyboard transcript harnesses load those production styles.

The three inherited failures were repaired at their test setup boundaries. The keyboard harness lacked the consolidated CSS and sampled the row before selection entry added its highlight and reflowed the layout. It now samples the selected row and additionally verifies that the menu stays inside its owner without shrinking it. The two real-database feedback tests now supply the same real workspace registry as ConsoleRuntime; their SQLite event and annotation assertions remain intact. Independent review found no masked production defect or weakened coverage.

Addressed Qodo's two findings: compose now declares ComposeResult and documents its yielded widgets; the menu observes screen layout changes, clears temporary compact styling and height limits when owner bounds change, and remeasures. Offset corrections use the measured position and request layout only when the offset changes. The layout subscription is removed on unmount. Eight resize cases cover terminal and transcript growth/shrinkage in both color modes, restoration of the feedback hint and actions, preserved focus and selection, and clicking Copy after repositioning. Independent review found no remaining actionable issues.

Qodo's follow-up requested explicit Returns sections on the three selection getters. Those docstrings now describe each row type's selected text, the capped/full-text option, and the empty-string result. The executable Python AST is unchanged by this documentation correction, with no new Ruff diagnostics and all edited regions formatted.

Files: `Widgets/Console/console_selection_menu.py`, `Widgets/Console/console_transcript.py`, both generated `css/widget_defaults_*.tcss` files, the new `Tests/UI/test_console_selection_copy.py`, existing selection-menu, keyboard-selection, and end-to-end tests, `Docs/User_Guide/console/text-selection-and-feedback.md`, and `backlog/docs/lessons-textual.md`.

Verification:
- All **183 cases pass** across the six targeted selection suites after rebasing onto dev at `c37d61136` (157.15 seconds), including the three inherited failures and all eight resize cases. Each inherited failure was reproduced before its repair and verified passing afterward. The existing requests dependency-version warning remains.
- All **55 selection-menu cases pass** after the Qodo follow-up (22.75 seconds). The four compact-to-full resize cases each failed before the owner-bound fix and passed afterward; four additional cases verify settled movement and click handling while the full menu continues to fit.
- All eight new Copy cases pass, including mouse and keyboard journeys with production styles, uncapped plain/Markdown/diff text, and stale selections. Earlier production render probes also verified user and assistant menu containment and first-action focus; SVG/PNG captures are in `/private/tmp/chatbook-copy-selection-32033/`.
- All six `scripts/preflight.sh` derived-artifact checks pass on the rebased branch, including stylesheet reproduction and task ID hygiene. The rebase changed none of the three patches, confirmed with git range-diff.
- The new Copy test file passes Ruff and formatting. Before/after checks find no new Ruff diagnostics in the five existing Python files; all edited Python regions and scoped whitespace checks pass. Existing lint findings and unrelated formatting drift remain.
- Clipboard assertions use the real Textual app clipboard in the headless harness. Linux desktop clipboard delivery was not exercised; the user guide explains the existing OSC 52 requirement and terminal-native selection fallback.

ADR check: direct extension of [ADR-068](../decisions/068-console-text-selection-and-annotations.md); no new ADR, storage change, clipboard backend, or shortcut. ADR-031 remains applicable. No full-suite run was performed, per repository policy.
<!-- SECTION:NOTES:END -->
