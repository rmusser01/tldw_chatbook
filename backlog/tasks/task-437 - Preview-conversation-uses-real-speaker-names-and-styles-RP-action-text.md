---
id: TASK-437
title: Preview conversation uses real speaker names and styles RP action text
status: Done
assignee: []
created_date: '2026-07-21 09:38'
updated_date: '2026-07-23 21:32'
labels:
  - roleplay
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). Observed live: preview transcript labels speakers literally as "character:" and "you:" instead of the card name and persona/user name, and RP *action* asterisks render as raw text. Small changes with outsized effect on how in-genre the surface feels.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Preview messages are labelled with the character's actual name (and the persona/user name once available)
- [x] #2 Single-asterisk action/emphasis spans render styled rather than as literal asterisks
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The Roleplay preview transcript now labels replies with the character's real name and renders single-asterisk `*action*` spans as italics.

**AC#1 — real speaker names:** `PersonasPreviewPane` gained `_character_label`/`_user_label` fields (defaults `"character"`/`"you"`) read by the five label sites, plus `set_speakers(*, character=None, user=None)` (empty/None keeps the current label). The screen's `_select_character` calls `set_speakers(character=entity_name)` once, before seeding (covers normal + TASK-434 restore); `handle_character_loaded` deliberately does NOT re-set the label (a same-character reload that preserves the transcript would otherwise leave existing lines under the old prefix — mixed/stale prefixes, per Qodo review). The user label stays `"you"` — the active persona/user name is **TASK-442**'s remit. Verified end-to-end: the workbench suite selecting the "Detective Sam" stub now labels lines `"Detective Sam: …"` (23 assertions updated), and the `open_in_console` handoff body carries the real name.

**AC#2 — styled `*action*`:** a `_styled_line(line) -> rich.text.Text` helper escapes Rich markup then italicizes single-asterisk spans (`Text.from_markup(escape(line) → [i]…[/i])`), used at the four render sites. `self._lines`/`transcript_text()` stay **plain** (raw text for copy/handoff/restore); only the mounted `Static` content is the styled `Text`. Passing a `Text` object (not a markup string) keeps the repo's `Static.renderable` shim returning plain text and makes `MarkupError` impossible (escape-first + only balanced `[i]` pairs). The Markdown widget was rejected — it over-interprets chat text (lists/headings). Known accepted limitation: spaced math (`5 * 3 * 2`) italicizes, mirroring markdown; RP actions are the target.

**Review fix (regression):** `_character_label` was set on selection but never reset, so character → persona → Test Reply showed the stale name; added `reset_speakers()` called from `_apply_mode` (a mode switch clears the character context; a later selection re-sets). 

**Design/review catches:** the `"character:"/"you:"` prefix is display-only (no parser consumes `transcript_text()`); the helper is `_styled_line` not `_render_markup` (which shadows a Textual internal); a Rich `Text` (not a markup string) is required by the `renderable` shim.

**Files:** `personas_preview_pane.py`, `personas_preview_controller.py`, `personas_screen.py`, `Tests/UI/test_personas_preview.py`, `Tests/UI/test_personas_workbench.py`.
<!-- SECTION:NOTES:END -->
