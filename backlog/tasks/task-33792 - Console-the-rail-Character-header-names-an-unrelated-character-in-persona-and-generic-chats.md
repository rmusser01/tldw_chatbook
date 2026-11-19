---
id: TASK-33792
title: >-
  Console: the rail Character header names an unrelated character in persona and
  generic chats
status: To Do
assignee: []
created_date: '2026-10-02 04:57'
labels:
  - console
  - ux-review-2026-10-01
  - bug
dependencies: []
references:
  - Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/report.md
  - Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/findings.json
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the 2026-10-01 Roleplay NN/g + HCI review: finding RP-084 (P2, severity 2, effort S). Filed as a Console task because the defect is in the Console, which the Roleplay review depends on. Report: `Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/report.md` (section 5 and Appendix A), with the machine-readable entry in `findings.json` beside it. The review ran on origin/dev @ 84247cb843 (worktree `.worktrees/roleplay-library-ux`).

Line refs below were re-checked on origin/dev @ ab4df99959. Paths are under `tldw_chatbook/` (`personas_screen.py` is `UI/Screens/personas_screen.py`; the `personas_*` and `persona_*` widgets are in `Widgets/Persona_Widgets/`). Capture names refer to the review's capture set; those copied into the QA folder are marked `evidence/`.

**What happens.** When no character is current, the Console left rail's Character section header falls back to the first character group. Right after a persona Chat now from Roleplay it reads "Character · Dungeon Ma…" directly above "No current character", while the status strip says "Persona: Cozy Storyteller". Generic chats show the same fallback ("Character · Detective").

**Who it hurts.** Anyone who lands in Console from a persona Chat now, or uses a generic chat: the header claims a character is in play when none is. The body row and the status strip are correct, which limits the damage.

**Evidence:**
- `tldw_chatbook/UI/Console_Modules/left_rail.py:552-560`: `_character_context_title` falls back to the first group when no group is current; it is used at `:546` and `:2263`, and no test pins it. The fallback arrived with 96f24f0efa (feat(console): add character conversation context); no branch carries a fix.
- Captures: `review-rv-gap1-persona-cozy-after-160x45` rows 34-35; `review-rv-gap1-3-persona-send-with-leaked-source-160x45` row 34; `review-rv-gap1-3-inspector-draft-sent-160x45` row 33.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With no current character, the Console rail's Character section header names no character (for example: Character · none).
- [ ] #2 In a persona-bound session, the header names no character: it names the persona, or the Character section is hidden.
- [ ] #3 In a character-bound session, the header still names that character and its chat count, as today.
- [ ] #4 A regression test opens a persona Chat-now session and a generic session and asserts the header contains no character label; it fails on the current code.
<!-- AC:END -->
