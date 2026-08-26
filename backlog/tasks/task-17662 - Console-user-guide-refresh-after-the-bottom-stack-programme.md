---
id: TASK-17662
title: 'Console user-guide refresh after the bottom-stack programme'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-18'
labels:
  - docs
  - console
dependencies:
  - task-17661
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The bottom-stack de-clutter programme (tasks 17650-17661, eight merged PRs) changed the Console's geometry substantially; the main guide page was patched incrementally per PR, but the child pages still describe the old layout (status chips "below the composer", staged strip "directly above the composer", draft cap "up to four rows"), the chat-basics page claims prompt-history recall does not exist (TASK-1364 shipped fish-style ghost text and Up/Down recall), the overview screenshot shows the pre-programme layout, and the main page's stamp block has accumulated seven per-task entries that read as noise. One coherent docs pass, drift-checked against the merged build.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Every geometry claim across console.md and its child pages matches the merged build (chips above the composer by default with the placement setting; strips at the top of the deck; 1-8 row composer with gaps; frame closes at the grid; footer token counter retired)
- [x] #2 chat-basics documents the prompt-history recall and ghost-text keys accurately (verified against the composer's key handling) and drops the false "no input history recall" quirk
- [x] #3 The overview screenshot is regenerated from the merged build and shows the new layout
- [x] #4 The main page's accumulated per-task stamps consolidate into one programme stamp; every touched child page gets a stamp
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Drift-grep all seven console pages for geometry claims; verify each against the merged build (and the composer's actual key handling for the recall/ghost keys).
2. Fix: chips position (context-and-rag ×2, agent-runs ×2), staged-strip position (context-and-rag ×2 + layout-tour bullet), draft cap four → eight + windowing note (chat-basics), composer visual description (console.md + chat-basics), queue-shelf position (chat-basics), the FALSE "no input history recall" quirk → accurate recall/ghost documentation with keys (Up/Down boundary-row recall, Right ghost-accept).
3. Consolidate the seven accumulated per-task stamps on console.md into one programme stamp; stamp every touched child page.
4. Regenerate overview.svg, action-row.svg (selected message + painted action row), and attachment-staged.svg (visible staged strip in its new top-of-deck slot) via export_screenshot on the merged build; verify image links resolve across all seven pages.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Docs-only pass. Headline correction beyond geometry: chat-basics' "No input history recall — pressing up/down never cycles through your past inputs" was FALSE — TASK-1364 shipped fish-style ghost text and boundary-row Up/Down recall; the keys documented were read from `handle_console_key` (Up on the first visual row / Down on the last recalls; Right at the draft's end accepts the ghost suggestion), not from memory. The three screenshots whose subjects changed were regenerated from the merged build (overview hero, the action-row shot, and attachment-staged — whose entire subject, the staged strip, moved above the status row); the remaining four (rewind/context modals, approval card, tabs coachmark) illustrate subjects that did not move and were left, noted here rather than silently skipped. console.md's stamp block: the seven per-task 2026-08-17/18 entries consolidated into one programme stamp; prior history untouched.

Files: `Docs/User_Guide/console.md`, `console/chat-basics.md`, `console/context-and-rag.md`, `console/agent-runs-and-tools.md`, `images/console/{overview,action-row,attachment-staged}.svg`.
<!-- SECTION:NOTES:END -->
