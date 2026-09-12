---
id: TASK-16472
title: embedding_template_selector calls nonexistent Grid.clear() at two sites
status: Done
assignee:
  - '@zcode'
created_date: '2026-08-14'
labels:
  - bug
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`tldw_chatbook/Widgets/embedding_template_selector.py:144` and `:163` call `grid.clear()` on a `Grid` — the same bug class TASK-15992 fixed in the selection dialogs (`hasattr(Grid, 'clear')` is False; `remove_children()` is the idiom), so exercising either path would raise AttributeError. Found by the TASK-15992 review's AST sweep of the whole package (assignments from `query_one(..., <container type>)` followed by `var.clear()`); these were the only two remaining hits (scratchpad `review15992.md`, section S1).

Reachability finding, recorded per the review: nothing imports `EmbeddingTemplateSelector` outside its own module, so the code is currently unreachable — the mechanical fix is one line per site, but the right disposition may be retirement per the repo's dead-code ruling rather than fixing an orphan.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No `.clear()` calls on Textual containers remain repo-wide; if cheap, extend the review's AST sweep into a small guard test so the bug class cannot return
- [x] #2 Fix-vs-retire is decided with reachability evidence recorded (who imports/mounts the widget, or proof nobody does)
<!-- AC:END -->

## Implementation Plan

1. Re-verify reachability on current dev: importer grep for Utils.embedding_templates (only the selector + Tests/UI/Embeddings), for EmbeddingTemplateSelector (only the same test file).
2. Land TASK-16472's retire branch and TASK-19643's deletion together (the test file imports both, so neither can land green alone): delete the widget, the module, the test file, and the runner's dead suites/coverage line.
3. Extend the TASK-15992 review's .clear() AST sweep into a standing guard test (16472 AC#1) and prove it catches the selector's two sites before the deletion.

ADR required: no
ADR path: N/A
Reason: Dead-code retirement already governed by ADR-078's parity spec (which filed these as separate units); no new architecture decision.

## Implementation Notes

RETIRE branch chosen for TASK-16472, jointly with TASK-19643's deletion. Reachability re-verified at implementation time on current dev: ``Utils/embedding_templates.py`` was imported only by ``Widgets/embedding_template_selector.py`` and ``Tests/UI/Embeddings/test_embedding_templates.py``; ``EmbeddingTemplateSelector`` was imported by nothing outside its own module except that same test file. The selector was unreachable production code whose only exercised path (``grid.clear()`` on a ``Grid`` that has no ``clear``) would raise AttributeError, so fixing it had no user to serve.

Deleted together: both production modules, ``Tests/UI/Embeddings/test_embedding_templates.py``, the runner's dead ``windows``/``templates`` suites and ``--cov=tldw_chatbook.Utils.embedding_templates`` line, and the stale file reference in ``Tests/Watchlists/test_watchlist_dialogs_escape.py``'s docstring (kept as a historical note). The ``.chunk-preview-*`` rules in ``css/features/_embeddings.tcss`` belong to the separate live ``Widgets/chunk_preview.py`` (DEFAULT_CLASSES) and were left alone.

TASK-16472 AC#1: the review's AST sweep is now a standing guard, ``Tests/Architecture/test_container_clear_guard.py`` -- it flags ``.clear()`` on names assigned from ``query_one(..., <container>)`` for every Textual layout container (Grid/Container/Vertical/Horizontal/both Scrolls/Center/Middle) across ``tldw_chatbook/``, while deliberately not flagging RichLog/Tree/DataTable/ListView which do have ``clear()``. Evidence it discriminates: run BEFORE the deletion it flagged exactly the selector's two sites (lines 144 and 163); after deletion it is green.

Verification: Tests/UI/Embeddings 41 passed; guard + watchlists-escape 11 passed; ``import tldw_chatbook.app`` clean; repo-wide grep leaves zero ``embedding_template`` references outside the historical docstring note. Ruff clean (guard file formatted).

Deleted: ``tldw_chatbook/Widgets/embedding_template_selector.py``, ``tldw_chatbook/Utils/embedding_templates.py``, ``Tests/UI/Embeddings/test_embedding_templates.py``. Modified: ``Tests/UI/Embeddings/run_tests.py``, ``Tests/Watchlists/test_watchlist_dialogs_escape.py``. Added: ``Tests/Architecture/test_container_clear_guard.py``.
