---
id: TASK-32234
title: >-
  Library Media reader: 'No Markdown formatting to render' is false for
  non-allowlisted types (document, article, pdf…)
status: To Do
assignee: []
created_date: '2026-09-10 14:52'
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
- [ ] #1 The rendered-view decision sniffs content for every media type, or the Info sentence names the real rule ('Rendered view is available for markdown, transcripts and plain-text items — this is a document')
- [ ] #2 A test covers a `document` item with real Markdown
<!-- AC:END -->
