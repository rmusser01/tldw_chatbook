---
id: TASK-32620
title: >-
  Library Notes: Preview shows two titles for a frontmatter-titled note and runs
  a callout's type into its body
status: Done
assignee: []
created_date: '2026-09-15 06:42'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor A P3 and A's docs table, persona Jordan, Obsidian workflow. Residual of task-32551 (PR #2683), which covered the body-starts-with-a-heading case.

What happened. A note whose title came from Obsidian frontmatter renders in Preview as 'Q3 planning — library review' left-aligned (the note title) directly above a CENTRED 'Library review' (the body's own heading) -- two title-shaped lines, the second one centred like a page banner (A cap 25). The de-duplication rule only fires when the heading matches the title exactly, which a frontmatter-titled Obsidian note never does.

Same capture, cosmetic: a warning callout renders as one run of text with its type and body joined and no colon or break, where the guide says the type becomes a rendered label (A cap 25; B saw the same shape at cap 27).

Cause PROVEN by capture.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A rendered heading is left-aligned, not centred
- [x] #2 When a note's first heading differs from its title and the title came from frontmatter, the heading renders as an ordinary heading rather than a second title
- [x] #3 A callout's type and body are visually separated as the guide describes
- [x] #4 Both cases are pinned so the frontmatter path is covered alongside the filename path
<!-- AC:END -->

## Implementation Plan

1. Reproduce both halves under production CSS.
2. Left-align the Preview body's H1 the way the Library Reader's already is.
3. Break a callout's type from its body.

## Implementation Notes

**AC#1/AC#2 -- one rule, already precedented.** Textual's `MarkdownH1`
defaults to `content-align: center middle`, which is why the body's first
heading floated to the middle of the pane directly under the left-aligned
title line. The Library Reader overrode exactly this in task-31635;
`#library-note-preview-body MarkdownH1` now does the same. Left-aligned, a
body H1 reads as the body's first heading rather than a second title, which is
what AC#2 asks for -- task-32551's de-duplication cannot help here, because it
fires only on an exact title match and a frontmatter-titled Obsidian note
never produces one. Authored in `css/components/_agentic_terminal.tcss` (the
SOURCE); `screen_agentic_library.tcss` is generated, and the bundle check is
green.

**AC#3 -- the run-on.** The lines under a callout header are a lazy paragraph
continuation, so `**Warning**` and the first body line rendered as one run.
`render_obsidian_callouts` now emits CommonMark's hard break (two trailing
spaces) after the label. At the end of a block those are stripped, so a
bodyless callout is byte-identical to before. Verified in the mounted
`Markdown` widget, not only in the string.

**AC#4 -- both pinned.** The frontmatter path (title differs from the first
heading) is pinned alongside the filename path task-32551 already covers; the
alignment is measured against another BODY line rather than the title Static,
because the Markdown widget carries its own one-cell padding and that is not
what the pin is about.

`Tests/UI/test_library_notes_w3_layout.py`'s three exact-equality callout
assertions gained the two trailing spaces, with a pointer to the new test.

**Files.** `Utils/markdown_parsing.py`,
`css/components/_agentic_terminal.tcss` (+ generated
`css/screen_agentic_library.tcss`), `Tests/UI/test_library_notes_w3_layout.py`,
`Tests/UI/test_library_notes_w5_import_preview.py`,
`Docs/User_Guide/library/notes.md`.
