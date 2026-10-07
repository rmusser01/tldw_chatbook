---
id: TASK-33624
title: >-
  Console unification: one attention and glyph registry with a shared controls
  grammar
status: To Do
assignee: []
created_date: '2026-09-30 03:04'
labels:
  - console
  - ux-review-2026-09-29
  - unification
dependencies: []
references:
  - qa/console-ux-review-2026-09-29/report.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Why: the Console draws conversation state with several separate vocabularies: CONSOLE_RUN_MARKER_GLYPHS for tabs, ATTENTION_PRESENTATIONS for rail rows, STATUS_GLYPHS for sections, the Ctrl+K word map, and two ASCII fallback tables. Approval alone is drawn with eight different combinations of glyph, colour and word. A tab cannot show 'blocked'. In ASCII mode, done and failed differ only by letter case. ASCII mode is applied to only some surfaces. Three 'unseen result' stores clear on different events, so a green check can sit on the chat being read. Controls grammar has drifted the same way: the triangle glyph means collapsed, submenu, row cursor, rail edge and 'more tabs', and the row-menu opener doubles as the status glyph.

What unifying means here: one registry of conversation attention states. Each kind maps to a Unicode glyph, a distinct ASCII token (distinct even ignoring case), a colour token, short and long labels, legend text and a CSS class. Tabs, rail rows, the workspace tree, Ctrl+K, the cross-tab fleet line, status chips, overflow hints, the nav badge and a generated F1 legend all read from it. One acknowledgement event clears every 'unseen' store together. A controls-grammar table fixes one meaning per disclosure, submenu, rail-edge and cursor glyph, and a lint checks shipped labels against it. This builds on ADR-034 (shared rail disclosure glyphs). The theme has no P0/P1 findings, so this umbrella owns the unification work directly and parents any child tasks filed for it.

Evidence: qa/console-ux-review-2026-09-29/report.md (theme 'State markers and disclosure glyphs') and qa/console-ux-review-2026-09-29/findings.md / findings.json.

Related P2/P3 in the ledger: G4-52, GAP1-11, GAP3-05, GAP3-06, GAP3-07, GAP1-08, GAP1-06, GAP1-10, GAP1-12, G4-53, G2-40, G2-39, G4-62, GAP1-07, G2-23, GAP4-15, GAP2-17, GAP3-09, G1-25, G4-66.

Source: Console UX review 2026-09-29 — qa/console-ux-review-2026-09-29/report.md (themes) and findings.md (ledger).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every surface that shows conversation attention state (tabs, rail rows, workspace tree, Ctrl+K, fleet line, status chips, overflow hints, nav badge, F1 legend) renders it from one registry, and a test fails if a surface defines its own glyph, colour or word for a state
- [ ] #2 Each attention state, including blocked and waiting-for-a-question, has a Unicode glyph and an ASCII token that stay distinct from every other state even when compared case-insensitively
- [ ] #3 With ASCII glyphs enabled, no Console surface renders a non-ASCII state or disclosure glyph
- [ ] #4 Visiting a chat clears its 'unseen result' mark on every surface at the same moment
- [ ] #5 Each disclosure, submenu, rail-edge and row-cursor glyph has exactly one meaning, and a lint on shipped Console labels fails on a glyph used outside its assigned meaning
- [ ] #6 All child tasks filed under this umbrella are Done (the theme has no P0/P1 findings, so its children are the unification work itself)
<!-- AC:END -->
