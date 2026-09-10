---
id: TASK-32222
title: >-
  Library Media viewer: three-way mismatch on the Escape footer chip vs two
  guide claims
status: Done
assignee: []
created_date: '2026-09-10 14:55'
updated_date: '2026-09-10 17:40'
labels:
  - library
  - media
  - docs
  - critique-9
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The viewer's Escape chip reads 'esc focus Library' while the guide claims two different targets in two places. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 19.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One chip text, one guide sentence, matching the real target
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read _library_media_escape_label per viewer state; compare against the pins and the live chip.
2. Fix the two contradicting guide sentences to one sentence quoting the chip.
3. Pin the chip text against the guide string in a test.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Docs-only: the chip is pinned by test (Tests/UI/test_library_media_reader_flow.py) and the guide was not, so the guide moved. `_library_media_escape_label` was read state by state and verified live -- More open reads 'esc close', the plain viewer reads 'esc focus Items', the Items row reads 'esc focus Library', and the rail row drops the chip. The guide previously said Escape 'shows the list again' in the Sets section and 'never leaves the Reader at all' in Keyboard & commands, while the critique's live capture caught a third string; the two claims were describing focus and document from different angles without saying so.

Both sites now carry one rule -- 'Escape closes transient Reader state first — the Find bar, then the More strip — and then steps out of the Reader to the Items list; the footer chip always names the next step it will take' -- and the guide QUOTES the chip's own labels. The three-pane nuance is kept but reframed honestly: stepping out moves focus, not the document. No code change was needed: the live chip matched `_library_media_escape_label` in every state.

A new test asserts the three quoted strings equal what the code returns and that neither retired claim survives in the page, so the two halves cannot drift apart again.

Files: Docs/User_Guide/library/media-and-conversations.md, Tests/UI/test_library_crit9_media_reader.py.
<!-- SECTION:NOTES:END -->
