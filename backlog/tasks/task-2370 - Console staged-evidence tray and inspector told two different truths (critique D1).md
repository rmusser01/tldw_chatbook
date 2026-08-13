---
id: TASK-2370
title: Console staged-evidence tray and inspector told two different truths (critique D1)
status: Done
assignee:
  - '@claude'
created_date: '2026-08-04 20:07'
labels:
  - console
  - rag
  - honesty
dependencies: []
priority: high
---

## Description

The 2026-08-04 RAG re-score critique (D1) found the Console staged-evidence tray counted rendered display ROWS, not distinct sources, producing self-contradictory chips such as "Sources 18" beside "1 staged". Separately, the Inspector rendered a bundle's status as "not staged" both during and immediately after a send that was actually carrying five staged sources — the strip and the Inspector disagreed about the same evidence. A third sub-claim (D1c) held that Library RAG staging never bundles evidence at all.

This directly contradicts the honesty theme the staged-evidence feature exists to serve: the UI must report what is actually staged, not a proxy that happens to correlate loosely with it.

## Acceptance Criteria

- [x] The staged-evidence tray displays a distinct-source count, not a display-row count
- [x] The Inspector reflects the strip's evidence state for the current send, including immediately after the send completes
- [x] The D1c sub-claim (Library RAG staging is bundleless) is independently verified true or false, with the finding recorded here regardless of outcome

## Implementation Notes

Fixed in PR-T1 (`feat/rag-truth-staged-evidence`) Task 1, commits `26ea5b160` (tray counts sources via `console_staged_source_count`, not rows; Inspector gains a one-send memory of the strip's evidence) and `44f2d9d6d` (review-directed hardening: the `source_count` sentinel changed from `-1` to `None` so an unset value can't silently re-admit the row-count lie for a future caller; `empty()` now passes `source_count=0` explicitly instead of riding the fallback).

D1c was investigated separately in Task 2 (commit `7950ed164`, review-confirmed on all four refutation legs) and is **REFUTED**: Library RAG staging has bundled evidence since May 2026 (blame `9512763ad9`). The live "1 staged" observation that motivated D1c was correct (one selected result) and is fully explained by the tray's row-count lie fixed above, not by a bundling gap. Task 2's review did surface a real, different content-loss defect while checking this claim — tracked separately, see task-2374.
