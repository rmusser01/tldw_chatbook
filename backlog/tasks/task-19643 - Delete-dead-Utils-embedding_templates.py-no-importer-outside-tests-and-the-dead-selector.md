---
id: TASK-19643
title: Delete dead Utils/embedding_templates.py (no importer outside tests and the dead selector)
status: Done
assignee:
  - '@zcode'
created_date: '2026-08-21'
labels:
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Utils/embedding_templates.py` has no importer outside tests (`Tests/UI/Embeddings/`) and `Widgets/embedding_template_selector.py` — and the selector is itself dead (nothing imports `EmbeddingTemplateSelector` outside its own module; TASK-16472 AC #2 already owns that widget's fix-vs-retire decision and records the reachability evidence).

Filed from the chunking template parity design spec §11 item 3a (`Docs/superpowers/specs/2026-08-21-chunking-template-parity-design.md`; ADR-078). Re-verified live 2026-08-21: importer grep in a worktree at/after the spec's pin, file untouched on `origin/dev` since. The spec files the module and the selector as separate units of work; this task covers the module only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Importer graph re-recorded at implementation time; deletion lands only after (or together with) TASK-16472's retire branch, so no live import breaks
- [x] #2 The module, its tests (`Tests/UI/Embeddings/test_embedding_templates.py` and the runner), and any packaging/CSS references are removed in one change with the targeted suites green
- [x] #3 If TASK-16472 instead revives the selector, this task records why the module survives (its revived consumer) and closes — a decision, not a silent drop
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
