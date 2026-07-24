---
id: TASK-390
title: Explain or hide the footer memory-stat abbreviations
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The status bar of the very first screen shows single-letter memory/storage stats (P:, C/N:, M:) with no legend, tooltip access, or context; a new user cannot decode them and they compete with the key hints for footer space.

**Repro:** Boot bare home -> read the right side of the bottom status row.

**Verifier note:** Accurate and uncovered. The string is built in db_status_manager.py:67 ('P: {prompts} | C/N: {chachanotes} | M: {media}' — database file sizes) and rendered into the per-screen footer next to 'Tokens: --' (AppFooterStatus.py:85) with no legend or tooltip; confirmed present on the first-run gate in j1-01. No ledger item covers it (open-rail-model-line-and-footer-nits lists other footer nits; per-screen-footer-hints/task-264 covers shortcut hints only) and no backlog task exists for it. Caveat for the filer: a 2026-07-17 shell-chrome critique session flagged AppStatusLine dead-chrome cleanup as in-flight uncommitted work in the main checkout, so check for an in-flight fix before filing a duplicate. P3: cosmetic diagnostics noise competing with key hints.

**Source:** Console UX expert review 2026-07-20 (finding j1-footer-cryptic-stats; P3, verdict NEW, medium confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J1 new-user cold start journey. Evidence: `j1-01-initial-console.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spell out or tooltip the abbreviations, or hide diagnostics behind a debug toggle on first-run
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Kept the compact footer stats (they compete with key hints, so widening them is
the wrong trade) and made them decodable on hover: the DB-size Static now carries
a static legend tooltip -- 'Local database file sizes / P: Prompts   C/N:
Conversations & Notes   M: Media'. The abbreviations were never wrong, only
undocumented; a hover legend spells every letter out without stealing footer
width. RED->GREEN test asserts the tooltip contains the spelled-out terms and the
word 'size'. AppFooterStatus.py only.
<!-- SECTION:NOTES:END -->
