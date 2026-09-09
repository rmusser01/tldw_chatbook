---
id: TASK-32053
title: >-
  Library Search/RAG: evidence cards unreachable by keyboard; `o`/`u` inert; Tab
  lands on the only source toggle
status: Done
assignee: []
created_date: '2026-09-08 18:22'
updated_date: '2026-09-08 19:59'
labels:
  - library
  - search-rag
  - ux
  - keyboard
  - critique-8
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After a search, 8–14 Tabs never focus an evidence card, `o` (open) and `u` (use in Console) do nothing while the footer advertises them, and Tab from the query box lands on the sole enabled source toggle where Enter empties the results while the footer still says 'enter select evidence'. The Run button shows no focus either. The documented keyboard flow for evidence does not exist. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 4.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each evidence card can be reached by keyboard and shows a visible cursor (shape, not colour alone)
- [x] #2 Enter selects the focused card, `o` opens it and `u` stages it, matching the footer
- [x] #3 The footer names the focused control's Enter action (for example 'enter toggle Notes' on a source toggle, 'enter run search' in the query box)
- [x] #4 The Run button has a visible focus state
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live (Tab to evidence card, o/u/enter)\n2. Failing test\n3. Shape focus cue on cards, Tab from query -> first card, footer names focused control's enter action, Run focus ring\n4. Green, live-verify, docs
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Correction to the finding first: the evidence cards were ALREADY keyboard-reachable (five Tabs from the query box: Run, the enabled source toggles, then the cards), and Enter/`o`/`u` already worked -- verified headlessly and live. What was missing was any way to SEE it, which is how two live reviewers read the cards as unreachable and the keys as inert.

- Focus cue: `.library-rag-result-card:focus` swapped border COLOUR only, and even that was not the rule doing the work -- the generic `*:focus { outline: solid }` fallback was painting its own edges OVER the card's border, so the rule was invisible. The focus rule now suppresses the outline and gives the left edge the house `thick` block bar (same non-colour cue as the media/notes list rows), in the cell `solid` already reserved, so nothing shifts on focus/blur.
- Run had no focus state at all (identical painted rows). It gets the heavy side rails the compact canvas actions use, on its own padding cells.
- Footer: the Search/RAG set advertised 'enter select evidence' on every control, including the query box (Enter runs the search) and the source toggles (Enter empties the results). The 'enter' chip now names the focused control's own action -- run search / toggle <Source> / switch mode / select evidence -- through one resolver shared with the New-note canvas, and the footer's focus gate follows that label instead of only the typing flag.

NOT done: the brief's optional 'Tab from the query box jumps straight to the first evidence card'. Reachability is already five Tabs and the ACs do not ask for it; reordering the natural Tab path past the Run button and the source toggles would surprise more than it helps.

Files: tldw_chatbook/UI/Screens/library_screen.py, tldw_chatbook/css/components/_agentic_terminal.tcss (+ regenerated bundle), Tests/UI/test_library_crit8_keyboard.py, Tests/UI/test_library_footer_focus.py, Docs/User_Guide/library/search-and-rag.md.
<!-- SECTION:NOTES:END -->
