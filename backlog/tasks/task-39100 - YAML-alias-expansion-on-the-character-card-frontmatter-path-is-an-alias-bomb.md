---
id: TASK-39100
title: 'YAML alias expansion on the character-card frontmatter path is an alias bomb'
status: To Do
assignee: []
created_date: '2026-11-06 09:00'
labels:
  - character-chat
  - security
dependencies: []
references:
  - tldw_chatbook/Character_Chat/Character_Chat_Lib.py
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The character-card frontmatter path parses YAML without limiting alias expansion, so a 217-byte frontmatter expands to 5.8 MB of JSON, and each alias level multiplies that by about 10 (an alias bomb, a denial of service from a crafted card). It exists on dev today; this task's PR does not fix it. Found by the Roleplay B2b follow-up burn-down (2026-11-05), measured on dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A crafted card whose frontmatter uses nested YAML aliases is refused or bounded before it expands, with a size or depth cap and a clear error
- [ ] #2 A test pins the 217-byte to 5.8 MB case and a normal card still imports
<!-- AC:END -->
