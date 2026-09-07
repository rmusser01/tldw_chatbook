---
id: TASK-438
title: Alternate greeting selector in Roleplay preview
status: Done
assignee: []
created_date: '2026-07-21 09:38'
updated_date: '2026-07-24 01:41'
labels:
  - roleplay
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). The card view shows "Alternate greetings: N" with their text, but the preview always seeds the primary first_mes; there is no way to start a session from an alternate greeting anywhere in the app.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When a character has alternate greetings, the preview offers a way to pick which greeting seeds the conversation
- [x] #2 Reset returns to the chosen greeting, not silently to the primary one
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The Roleplay preview now offers a dropdown to choose which greeting (primary or an alternate) seeds the conversation, and Reset returns to the chosen greeting.

**AC#1 — selector:** a `Select` (`#personas-preview-greeting-select`) in a `#personas-preview-greeting-row`, shown only when a character has ≥1 alternate greeting. The controller owns the placeholder-processed greetings list `[first_message, *alternate_greetings]` (top-level `list[str]` on the record, available from the async `handle_character_loaded`) and a `_current_greeting_index`. Picking an option → `PreviewGreetingSelected(index)` → `handle_greeting_selected` → `pane.seed_greeting(chosen)`.

**AC#2 — Reset returns to the chosen greeting:** `seed_greeting` sets `pane._greeting`, and Reset already re-renders from `_greeting` — no new Reset logic. Verified end-to-end.

**Notable subtleties resolved during review (the `Select` widget is a footgun):**
- *AC#2 on same-character reload:* editing+saving the card after picking an alternate re-fires `handle_character_loaded` on the preserve path; the greeting-list rebuild must **keep** the chosen index (`keep_index`/clamp), else Reset silently reverts to the primary.
- *Critical — transcript wipe:* `Select.set_options()` internally snaps `value→0`, which (when a real alternate was selected) fired a spurious `Select.Changed(0)` that the controller misread as a user reselection and used to `invalidate()` the in-progress transcript. Fixed by wrapping the programmatic population in `with self.prevent(Select.Changed)` (suppresses the event at Textual's `check_message_enabled`). A test that drives the **real Select widget** now guards this (the earlier test called the controller directly and missed it).
- `EmptySelectError` in this Textual (8.2.7) for `Select([], allow_blank=False)` → constructed with an inert placeholder overwritten before the row is shown.

**Scope:** preview only (not the card view / Console). Files: `personas_pane_messages.py`, `personas_preview_pane.py`, `personas_preview_controller.py`, `personas_screen.py`, `Tests/UI/test_personas_preview.py`, `test_personas_workbench.py`, `test_personas_preview_restore.py`.
<!-- SECTION:NOTES:END -->
