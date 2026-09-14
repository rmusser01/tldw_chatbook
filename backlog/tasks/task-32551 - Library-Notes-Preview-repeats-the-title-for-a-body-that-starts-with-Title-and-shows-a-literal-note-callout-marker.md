---
id: TASK-32551
title: >-
  Library Notes: Preview repeats the title for a body that starts with "#
  Title", and shows a literal "[note]" callout marker
status: In Progress
assignee: []
created_date: '2026-09-13 06:47'
updated_date: '2026-09-14 18:36'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), both assessors, persona researcher. Residual of task-32142 (title added above the Preview body); #2's P3 "the preview leaks the Obsidian callout marker" was recorded and never filed.

**What happened.** An imported note whose body starts with `# Library ▸ Notes review` shows the title as the title line and again as the rendered H1 (A 39; B 34). Alex's seeded note shows a literal "[note]" at the top of Preview — the callout is not rendered (A 50). Captures: A 39, 50; B 34.

**Cause.** INFERRED.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When the body's first line is an H1 equal to the note title, Preview shows it once
- [x] #2 Obsidian callouts (> [!note] …) render as a styled block in Preview, or the marker line is hidden
- [x] #3 A test pins both renderings
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce (done: 'Markdown showcase' paints as the preview title AND again as the rendered H1; the '> [!warning]' callout ALREADY renders as a quoted 'Warning' block, so AC#2 is satisfied at this dev).
2. RED pin for the H1 duplicate, plus a pin for the seeded callout form.
3. render_preview_source(body, *, title='') drops a leading H1 equal to the title; both call sites pass the snapshot title.
4. GREEN + live.
<!-- SECTION:PLAN:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
Reproduced live at dev 2f97a42c9a on the seeded "Markdown showcase" note
(235x52): its body opens `# Markdown showcase`, so Preview painted the title
as the pane's title line and again as the rendered H1
(`editor-00-32551-preview-title-twice-callout-ok-235x52`).

**AC#1 fix.** `render_preview_source(body, *, title="")` drops the body's
first non-blank line when it is exactly `# ` + the note's title (stripped,
case-sensitive). Both call sites — the compose that mounts the Markdown
widget and the in-place sync whose staleness check compares against what
compose produced — pass `snapshot.title`, so the two cannot drift. Only the
OPENING heading and only an exact repeat: a different H1, or a later one,
is the author's own and stays (two negative-control pins).

**AC#2 does not reproduce.** The same capture shows the seeded `> [!warning]`
callout rendering as "▌ Warning The preview and the editor disagree about
callouts…" — task-32249 shipped the rewrite before critique #3 was run, and
its regex already covers every Obsidian form I could construct (`>[!note]`
with no space, `[!note]-`/`+` folds, nested `> > [!tip]`, a capitalised
`[!TODO]`, a title on the header line), while leaving a fenced example alone.
So the "literal [note]" A saw at capture 50 is either a stale observation or
a form neither the seed nor Obsidian produces; I did not invent a regex
change to justify the ticket. Pinned instead, end to end through
`render_preview_source`:
`Tests/UI/test_library_notes_w3_layout.py::test_the_seeded_callout_form_
renders_as_a_labelled_quote` uses the seed's exact text and is RED on
detached origin/dev (for the H1 half).

**Tests.** `::test_preview_shows_a_leading_h1_equal_to_the_title_once` reads
the mounted Markdown widget's own `source`; RED on dev, GREEN here. Live at
235x52 (`editor-10-32551-preview-title-once-callout-235x52`).

Modified: `tldw_chatbook/Widgets/Library/library_notes_canvas.py`, the pin
file, `Tests/UI/test_library_notes_w3_layout.py`,
`Docs/User_Guide/library/notes.md`.
<!-- SECTION:NOTES:END -->
