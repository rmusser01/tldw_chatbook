---
id: TASK-35022
title: Ctrl+S saves a Settings page from inside its fields
status: To Do
created_date: 2026-10-14 17:10
labels:
- settings
- keyboard
- adr-031
- ux-review-2026-10-14
dependencies:
- TASK-33007
- TASK-35021
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Settings save key is the bare letter s. A focused text field types it as a letter, and an integer field drops it silently, so users have to remember to press Esc first. The footer says so, but users reach for Ctrl+S by reflex. The owner's 2026-10-03 refinement of ADR-031 makes Ctrl+S the save key in every editor. Under it, Ctrl+S performs the editor's visible commit action, flushes instant-apply edits, and reports a refused save without moving focus. Outside an editor it does nothing. TASK-34000.43 inventories this across the app. This task delivers it for the Settings pages that offer s, including the in-page editors on those pages that have their own commit button. Edits that a field commits or rolls back when it is left must take effect before the save, so Ctrl+S never saves a value the field itself would reject.

Plan: `backlog/docs/plan-2026-10-14-settings-keyboard-search.md`. Findings closed: D8 (Settings UX review, 2026-10-14).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With focus on any control in a Settings page whose footer offers s, Ctrl+S performs that page's Save (s) with the same result and messages, and focus stays on that control
- [ ] #2 Edits that a field commits or rolls back when it is left (an invalid Custom ID model, the session summary duration) take effect before the save, so Ctrl+S never saves a value that field would reject or drop
- [ ] #3 In an in-page editor that has its own commit button, such as a custom endpoint's Save name or Save endpoint, Ctrl+S performs that button's action and never the page's Save
- [ ] #4 Typing s, r or t in a field still types the letter. Ctrl+S does nothing on the rail, in the category filter, in the inspector or with nothing focused. On Settings surfaces whose footer does not offer s (Advanced Config, the Theme editor, My Profile, Internal Prompts and the like), this task's binding performs no save; Ctrl+S in those editors is left to TASK-34000.43
- [ ] #5 While a text field in the page has focus, the footer shows 'ctrl+s' with the action it performs in place of 'Esc, s save category', and the other keys keep their 'Esc,' form
- [ ] #6 The Settings User Guide's keyboard and troubleshooting sections teach Ctrl+S
<!-- AC:END -->
