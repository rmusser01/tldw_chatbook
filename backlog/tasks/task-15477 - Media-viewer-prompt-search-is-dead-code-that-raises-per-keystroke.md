---
id: TASK-15477
title: Media viewer prompt search is dead code that raises per keystroke
status: To Do
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - bug
  - media
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the latency audit: `Widgets/Media/media_viewer_panel.py:1606` imports `get_prompts_db` from `DB/Prompts_DB` — a symbol that does not exist anywhere in the repo — so every keystroke in the prompt-search box raises ImportError, swallowed at `:1643`; the feature has silently never worked. The handler also calls `call_from_thread` from the UI thread (`:1641`, illegal in Textual) and, as designed, would run its sqlite search inline per keystroke with no debounce/worker (`:1624-1631`).

Decide: wire it to the real Prompts DB seam (threaded + debounced per the task-15476 shape) or remove the affordance. Either way the per-keystroke exception churn stops. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Prompt search either works against the live prompts store, off-loop and debounced, or the affordance is removed
- [ ] #2 No exception per keystroke (log evidence)
- [ ] #3 A regression test covers the chosen path
<!-- AC:END -->
