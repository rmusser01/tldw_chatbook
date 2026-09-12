---
id: TASK-32234
title: >-
  Library Media reader: 'No Markdown formatting to render' is false for
  non-allowlisted types (document, article, pdf…)
status: Done
assignee: []
created_date: '2026-09-10 14:52'
updated_date: '2026-09-10 17:40'
labels:
  - library
  - media
  - copy
  - critique-9
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`_MARKDOWN_MEDIA_TYPES` in `library_media_viewer_state.py` allowlists plaintext/markdown/obsidian_note/video/audio and `_is_markdown_media()` returns False for every other type before the content sniff runs, so a `document` whose stored text starts with `# Roadmap sync` shows literal Markdown and Info explains it with a sentence the user can disprove by looking at the screen. The nearest test parametrises only plaintext and video. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The rendered-view decision sniffs content for every media type, or the Info sentence names the real rule ('Rendered view is available for markdown, transcripts and plain-text items — this is a document')
- [x] #2 A test covers a `document` item with real Markdown
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing test: a 'document' item with real Markdown renders + drops the false note.
2. Delete the _MARKDOWN_MEDIA_TYPES gate; the bounded content sniff decides alone.
3. Update the two stale comments that reference the allowlist.
4. Docs + live verify.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Deleted `_MARKDOWN_MEDIA_TYPES` and the gate in front of the content sniff: `_is_markdown_media` is now `return looks_like_markdown_content(content)`. The allowlist ran BEFORE the sniff, so every type outside it (document, article, pdf) returned False without ever reading the text -- a `document` starting '# Roadmap sync' painted its hashes literally while Info explained it with 'No Markdown formatting to render — showing the stored text', a sentence the screen underneath disproved. The allowlist never carried information anyway: local_file_ingestion maps .md, .txt, .rst, .csv and .log all onto 'plaintext'. The sniff is already bounded (MAX_MARKDOWN_SNIFF_CHARS/LINES), so cost is unchanged, and RENDERED_VIEW_NOTE keeps its exact wording -- it is now true whenever it is shown. `media_type` stays in the signature for the caller's own type-aware copy.

Trade-off accepted: a body line that merely looks like a heading (a hashtag with a following space) now opens on Rendered. That is the same exposure `plaintext` already carried, the Raw toggle is one press away, and nothing is hidden or altered -- only the default view.

One pin asserted the removed behaviour: Tests/Library/test_library_media_viewer_state.py::test_build_state_non_markdown_type_never_flagged_even_with_heading_syntax. Rewritten to the new contract (renamed test_build_state_flags_markdown_by_content_whatever_the_type) rather than deleted, keeping a negative control so coverage is not lost. The two comment-only references in test_library_media_render_fixes.py and test_library_media_reader_scroller_resolution.py were updated; both files' failing-name sets are unchanged from origin/dev.

Live: the seeded 'Meeting notes 2026-09-01 — roadmap sync' (type document) now shows the Rendered|Raw toggle, paints a rendered heading, bullets and a numbered list, and its Info tab carries no rendered-view note.

Files: tldw_chatbook/Library/library_media_viewer_state.py, Tests/UI/test_library_crit9_media_reader.py, Tests/Library/test_library_media_viewer_state.py, Tests/UI/test_library_media_render_fixes.py, Tests/UI/test_library_media_reader_scroller_resolution.py, Docs/User_Guide/library/media-and-conversations.md.
<!-- SECTION:NOTES:END -->
