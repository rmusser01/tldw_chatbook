---
id: TASK-32326
title: >-
  Console rail terminology pass resolves Context naming collisions
status: Done
assignee:
  - '@zcode'
created_date: '2026-09-10 12:00'
labels:
  - console
  - ux
  - rail-ux-review
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX review B2. The left rail is branded 'Context' but no longer contains the context-staging UI (Sources tray moved to the Inspector in task-400); 'Chat Context' viewer (Ctrl+Shift+P) is an unrelated surface one word away; 'Sources' has four senses; docs say both 'Inspector' and 'run inspector'. Decide and apply one vocabulary: user-facing rail brand, staging concept, viewer name. Code ids (console-context-rail-*) must NOT be renamed -- this is a copy/docs pass, not a refactor.

Filed from the 2026-09-10 Console rail UX review (review item B2).
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Decide the vocabulary (rail brand keeps Context; viewer becomes Current Context). 2. Rename the two modal header sites + pin; rewrite doc occurrences. 3. Add the Terminology glossary section. 4. Run modal suites; grep for stragglers.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A single decision record (ADR or task note) fixes the vocabulary: what the left rail is called, what the staging tray is called, what the Ctrl+Shift+P viewer is called
- [x] #2 User-facing copy no longer uses 'Context' for two unrelated concepts; the F1 help panel and user-guide docs use the agreed terms consistently
- [x] #3 All shell ids and config keys remain unchanged (verified by grep: no test churn from renames)
- [x] #4 Docs glossary or terminology section added covering rail, handle, staged sources, scope, Inspector
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Close-out (2026-09-10)

**Approach — the vocabulary decision.** The LEFT rail keeps its established
brand ("Context", full name "Console context" rail; control-bar button
"Context rail", Alt+C). Renaming the rail's UI brand was considered and
rejected: it would churn every doc surface for a name users already
learned, and the review's actual harm was COLLISION, not the word itself.
The collisions resolved:

1. **"Chat Context" viewer -> "Current Context" tab.** The Conversation
   Inspector (Ctrl+Shift+P) header read "Chat Context" — one word off the
   rail's brand, describing neither surface. Renamed at both header sites
   (initial compose + sync) with a tombstone comment; test pin updated;
   7 doc occurrences rewritten (history mentions preserved as history).
2. **Glossary.** console.md gains a Terminology section defining: Context
   rail, Inspector, handle, staged sources, retrieval scope (incl. the
   Sources-kinds vs Scope-items distinction), Conversation Inspector
   (Current Context / Next Send), run inspector.
3. **Ids untouched** (AC#3): console-context-rail-*, rail_state keys, all
   DOM ids unchanged — verified by grep, no test churn from renames
   (the only test changes are the two deliberate label pins).

**ADR check.** A decision record exists: this task file + the glossary;
no architectural boundary moved, so no ADR (per CLAUDE.md's
not-required list: copy/docs pass).

**Modified.**
`tldw_chatbook/Widgets/Console/console_conversation_inspector.py`,
`Tests/UI/test_console_context_modal.py`, `Docs/User_Guide/console.md`,
`Docs/User_Guide/console/context-and-rag.md`. Verified: context-modal +
conversation-inspector suites — 78 passed.

### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

VERIFIED LIVE/Code: left header 'Context <', right '> Inspect', handle constants 'Context v'/'<- Inspector' (rail_state.py:79-80), Ctrl+Shift+P opens 'Conversation Inspector' modal whose tab header still says 'Chat Context' (console_conversation_inspector.py:1777-1781). Three names, one concept space.
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
