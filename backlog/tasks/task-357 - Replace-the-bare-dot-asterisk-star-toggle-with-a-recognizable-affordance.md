---
id: TASK-357
title: Replace the bare dot-asterisk star toggle with a recognizable affordance
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every rail conversation row ends in a single '.' character on a button-colored cell. Nothing identifies it; during testing a stray click on it starred 'Quick question re keybindings' — no toast, no visible response at click time (the rail had also just mode-swapped) — and the star was only discovered later when the row appeared under Starred with a '*' glyph. A one-cell '.' vs '*' distinction is nearly invisible at a glance in a character grid.

**Repro:** Look at any rail conversation row's right edge ('.'), click it: the conversation is starred/unstarred silently; compare Starred section before/after.

**Verifier note:** Code-verified: star button label is literally '*' if row.starred else '.' (console_workspace_context.py:950), and the press handler (chat_screen.py:12435-12483) notifies only on failure — success is silent except the eventual rail resync. The '.'/'*' pair is outside the settled console_glyphs vocabulary (glyph-language decision predates this browser and doesn't cover it); the starring feature itself post-dates the ledger. Accidental silent toggle observed in testing supports P2 (unrecognizable affordance + unconfirmed state change on every row).

**Source:** Console UX expert review 2026-07-20 (finding j2-star-toggle-renders-as-dot; P2, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J2 returning power user journey. Evidence: `j2-57-mangle-trialB.png`, `j2-02-initial.png`, `j2-44-dot-click.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A recognizable glyph pair (☆/★ or [ ]/[*]) plus confirmation feedback ('Starred <title>') when toggled
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two changes. The rail star toggle now renders a recognizable filled/hollow star
pair (★ starred / ☆ not) instead of the near-invisible one-cell '*'/'.'
(`console_workspace_context.py`). And the press handler, which previously
notified only on failure (a successful star/unstar was silent — the review saw
an accidental star go unnoticed), now confirms the toggle with `Starred "<title>"`
/ `Unstarred "<title>"`; the title is carried on the star button as
`conversation_title` so the handler has it. RED→GREEN tests in
`test_console_workspace_context_rail.py` (glyph pair; press confirms the toggle);
44 rail tests green.
<!-- SECTION:NOTES:END -->
