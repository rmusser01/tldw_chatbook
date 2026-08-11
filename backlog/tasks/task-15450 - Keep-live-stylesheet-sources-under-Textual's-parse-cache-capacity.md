---
id: TASK-15450
title: Keep live stylesheet sources under Textual's parse-cache capacity
status: To Do
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - perf
  - ui-platform
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the 2026-08-11 input-latency audit: a live headless tour of all 13 hotkey destinations ends at 93 stylesheet sources, past Textual 8.2.8's parse cache (`LRUCache(64)` in `css/stylesheet.py`). Past the cliff, every `stylesheet.parse()` runs fully cold — measured 125-127 ms per call on fast hardware, repeated back-to-back with zero cache benefit — and Textual re-runs that parse whenever a widget class not seen this session first mounts (screen switches, modals, deferred mounts). The cliff is crossed at the 8th destination; Personas alone adds 30 sources. The repo carries 183 widget `DEFAULT_CSS` declarations; each distinct mounted class adds one source. Additionally six modal classes declare class-level `CSS` (ConversationSelectionDialog, EmojiPickerScreen, VoiceBlendDialog, FileExtractionDialog, DeleteConfirmationModal, NoteSelectionDialog, plus ScraperBuilderWindow), each triggering a full cold reparse plus a whole-app restyle on first open.

Stability constraint (owner preference): fix by consolidating widget `DEFAULT_CSS` into the built bundle (`css/build_css.py` is the seam) rather than patching/monkeypatching Textual's cache size — a cache-size override is fragile across Textual upgrades. Screens must NOT gain `CSS_PATH` (the task-262 no-split verdict, Docs/Design/2026-07-17-css-split-investigation.md: per-screen CSS files re-trigger the first-push reparse this task exists to avoid). Related open umbrella: task-2902. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a full 13-destination tour the live stylesheet source count is under 64, measured by a repeatable Pilot probe (probe method recorded in the task)
- [ ] #2 Repeated stylesheet.parse() after a full tour is cache-warm (single-digit ms), measured before/after
- [ ] #3 The six modal class-level CSS declarations no longer trigger an app-wide cold reparse on first open
- [ ] #4 No visual regressions: representative screens compared before/after (pixel A/B or rendered-CSS diff), including specificity-sensitive widgets whose DEFAULT_CSS moved
<!-- AC:END -->
