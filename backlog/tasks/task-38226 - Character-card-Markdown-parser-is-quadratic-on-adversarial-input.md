---
id: TASK-38226
title: 'Character card Markdown parser is quadratic on adversarial input'
status: To Do
assignee: []
created_date: '2026-10-30 18:43'
labels:
  - character-chat
  - performance
dependencies: []
references:
  - tldw_chatbook/Character_Chat/Character_Chat_Lib.py
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
load_character_card_from_string_content's Markdown branch (its frontmatter and fenced-JSON regexes) takes quadratic time on adversarial input, so a 10 MB crafted `.md` card would hold the card import's worker for a long time. It exists on dev today. Roleplay frame B2b's sniff never runs that parser, but a file the sniff routes to cards still reaches it; B2b does not fix the parser. Found during Roleplay frame B2b (TASK-33910.4, plan Docs/superpowers/plans/2026-10-26-roleplay-b2b-toolbar-new-import.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The parser reads a 10 MB adversarial Markdown file in linear time, pinned by a test with a time-independent bound (operation count or input-size scaling)
- [ ] #2 Every Markdown card that imports on dev still imports the same
<!-- AC:END -->
