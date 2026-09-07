---
id: TASK-2750
title: Renumber 34 duplicate task IDs resident on dev (repo-wide dup sweep)
status: Done
assignee:
  - '@claude'
created_date: '2026-08-06 23:50'
labels:
  - hygiene
  - backlog
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The post-merge duplicate scan after PR #1379 (2026-08-06) found **34 task IDs each carried by two distinct files on origin/dev**: 400-403, 506-519, 1333, 1338-1347, 1360-1361, 1373-1375, 2231. Spot-checks confirm they are real (e.g. two `task-400` files: "Fix MCP navigation crash…" vs "Move Console Context staged sources…"; two `task-1373`: Settings j/k navigation vs blocking-IO guard; two `task-2231`: Library ingest round 7 vs Roleplay redesign R2).

This is the recurring silent-pileup failure: the backlog-guard workflow's post-merge check runs on dev pushes whose CI is intentionally cancelled, and per-PR checks cannot see sibling branches, so duplicates accumulate until a dedicated sweep (precedent: PR #760 cleaned five in 2026-07).

Resolution rules that held previously: the **Done/older/load-bearing side keeps the number** (close-out commits, PR titles and code comments reference it); the To Do/newer side renumbers to fresh IDs claimed with headroom past every open branch's claims; cross-references (`dependencies:`, doc/code citations of the moved ids) are fixed in the same pass; frontmatter `id:` and filename must both move. Re-run the two-namespace scan (filename prefix + frontmatter id, `os.listdir` + regex — not `git ls-tree | sed`, which chokes on quoted paths) on the merged view immediately before AND after the cleanup PR merges.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The repo-wide two-namespace duplicate scan on origin/dev reports zero duplicate filename IDs and zero duplicate frontmatter ids
- [x] #2 Every renumbered task keeps its content, status and history; the keeper side of each pair is chosen by the Done/older/load-bearing rule and the choice is recorded in the cleanup PR body
- [x] #3 All cross-references to moved IDs (task dependencies, docs, code comments) resolve to the new numbers
- [x] #4 The scan is re-run on the merged view after the cleanup PR lands (concurrent sessions mint IDs by the hour)
<!-- AC:END -->

## Implementation Plan (the how)

1. Build the pair table (status, created_date, per-side reference attribution from code/doc citation contexts).
2. Check EVERY mover for an already-renumbered twin by title marker — earlier dedups (task-542/544/554/561/869) had already moved many of these; a duplicate with a living twin is a resurrected ghost and gets DELETED, not renumbered.
3. Renumber the genuine movers to a fresh headroom block (2800+ after an all-branches+worktrees sweep), rewriting frontmatter ids and intra-family cross-references.
4. Attribute and rewrite external references per side (code comments, specs, plans, ADR links incl. %20 filename links, sibling task files); regenerate the CSS bundle for the one tcss comment.
5. Two-namespace scan to zero; re-scan post-merge.

## Implementation Notes

35 duplicated IDs → 47 colliding files. **25 were resurrected ghosts** of tasks already renumbered by earlier dedup sessions (400→542, 401-epic+9 children→553.x, 402→561, 506-518 STT batch→593-605 incl. the retitled 517→604, 519→869) — deleted, references re-pointed at the living twins (the 60x20 plan's TASK-400 gates now cite 542). **22 genuinely un-deduped files renumbered to 2803-2837**: the README rewrite (403→2803), GGUF import (510→2808, only 506-518 member with no twin), Library-Notes 60x20 (1333→2818, In Progress, plan doc updated), local-agent phases 1/2/3a/3b-i/3b-ii/3c/4 + snippet bug (1338-1345→2819-2828, ADR-033 links updated), settings movers (1343/1344/1345/1347/1374→2825/2827/2829/2831/2835, incl. settings_screen + tcss source + bundle + test comments), watchlists tab-strip (1346→2830, spec ruling line updated), web-tools caching/robots (1360/1361→2832/2833), blocking-IO guard (1373→2834, ChatbookExport/Wizard comments updated), wizard below-fold (1375→2836), library round-7 P2 (2231→2837, library-side citations updated; personas side keeps 2231).

Keeper rule held throughout: Done/older/load-bearing side keeps its number; references attributed per side by citation context, never rewritten wholesale. Post-sweep two-namespace scan: ZERO duplicates across 1,494 files. Touched code compiles; affected suites green (library ingest state 133, settings tooltip test). Lesson recorded in lessons-backlog-hygiene.md (ghost-resurrection check before renumbering).
