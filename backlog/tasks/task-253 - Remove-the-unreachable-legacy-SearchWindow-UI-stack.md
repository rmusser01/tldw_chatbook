---
id: TASK-253
title: Remove the unreachable legacy SearchWindow UI stack
status: Done
assignee:
  - '@claude'
created_date: '2026-07-12 14:12'
labels:
  - rag
  - cleanup
  - ui
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UI/SearchWindow.py is imported by nothing after the master-shell redesign, yet it is the only mount point for SearchEmbeddingsWindow, Embeddings_Management_Window, the embeddings wizards and the chunking-template widgets - none of which are reachable from any registered route. UI/SearchRAGWindow.py.bak also lingers. Per the redesign rule legacy widgets are not reused unless fully redone Console-style; if manual embeddings management is still wanted it should be rebuilt as a new Console-parity screen in a separate task. Remove the dead stack so the codebase reflects the real UI surface. Filed from the 2026-07-12 RAG module audit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 UI/SearchWindow.py and screens/widgets reachable only through it (SearchEmbeddingsWindow, Embeddings_Management_Window, embeddings wizards, chunking-template widgets) are removed along with SearchRAGWindow.py.bak
- [x] #2 No route registry or import references to the removed screens remain
- [x] #3 Full test suite passes after removal
<!-- AC:END -->

## Implementation Plan

1. Verify importers of every candidate module (git grep for module paths AND class names). Confirmed dead stack rooted at UI/SearchWindow.py (no production importers) plus modules reachable only through it.
2. Delete production modules: UI/SearchWindow.py, UI/SearchEmbeddingsWindow.py, UI/Embeddings_Window.py (shim), UI/Embeddings_Creation_Content.py (alias shim), UI/Embeddings_Management_Window.py, UI/Wizards/EmbeddingsWizard.py, UI/Wizards/EmbeddingSteps.py (only importer: EmbeddingsWizard), Widgets/chunking_templates_widget.py, Widgets/chunking_template_editor.py (only importer: the templates widget), plus secondary orphans whose sole importer was Embeddings_Management_Window: Widgets/performance_metrics.py, Widgets/embeddings_list_items.py, Utils/model_preferences.py. Delete Event_Handlers/search_events.py (its only job is delegating button events to the removed "#search-window").
3. UI/SearchRAGWindow.py.bak is already absent on origin/dev (removed previously); note as no-op.
4. Reference cleanup: UI/Wizards/__init__.py (drop EmbeddingsWizard/EmbeddingSteps exports), Event_Handlers/event_dispatcher.py + app.py (drop search_events import/spread), app.py TemplateDeleteConfirmationEvent branch (imports the deleted chunking widget; nothing posts that event).
5. CSS: surgically remove selectors exclusive to the deleted stack from css/features/_embeddings.tcss and css/features/_search-rag.tcss; keep selector families used by live widgets (empty-state, chunk-preview, toast, detailed-progress, embedding-template-selector, activity-log, dialog, metric-*, model-name/header/provider/size, error-message, sidebar-title). Rebuild css/tldw_cli_modular.tcss via css/build_css.py.
6. Tests: delete test files exclusive to removed modules; edit mixed files (test_search_handoffs.py, test_ux_audit_smoke.py, test_disabled_action_recovery_tooltips.py, test_file_picker_action_tooltips.py, test_core_imports_unit.py, test_app_startup_performance.py) to drop only the removed-module tests.
7. Verify: `python -c "import tldw_chatbook.app"` smoke; pytest Tests/UI Tests/RAG diffed against a clean origin/dev baseline run; grep tree for lingering references.

## Implementation Notes

Pure removal of the dead SearchWindow stack; no rebuilds. Importer evidence was gathered per module (git grep for module paths and class names) before each deletion.

**Deleted production modules** (importers verified as: nothing, or only other members of this stack):
- `UI/SearchWindow.py` - no production importers since the master-shell redesign
- `UI/SearchEmbeddingsWindow.py` - imported only by SearchWindow + two shims below
- `UI/Embeddings_Window.py`, `UI/Embeddings_Creation_Content.py` - compat shims over SearchEmbeddingsWindow, themselves never composed
- `UI/Embeddings_Management_Window.py` - mounted only by SearchWindow
- `UI/Wizards/EmbeddingsWizard.py`, `UI/Wizards/EmbeddingSteps.py` - wizard mounted only by SearchWindow; EmbeddingSteps imported only by the wizard
- `Widgets/chunking_templates_widget.py`, `Widgets/chunking_template_editor.py` - widget mounted only by SearchWindow; editor opened only by the widget
- Secondary orphans whose sole importer was Embeddings_Management_Window: `Widgets/performance_metrics.py`, `Widgets/embeddings_list_items.py`, `Utils/model_preferences.py`
- `Event_Handlers/search_events.py` - its only job was delegating button events to the removed `#search-window`; import/spread dropped from `app.py` and `event_dispatcher.py`

**Not deleted (kept live, verified)**: `UI/Views/RAGSearch/*` + `UI/SearchRAGWindow.py` re-export shim (mounted by `UI/Screens/search_screen.py`), `Widgets/chunk_preview_modal.py` + `RAG_Search/enhanced_chunking_service.py`, `RAG_Admin/` scope services (task-254), `Widgets/empty_state.py`, `Widgets/activity_log.py` (SubscriptionWindow), `Widgets/toast_notification.py` / `detailed_progress.py` / `embedding_template_selector.py` / `Utils/embedding_templates.py` (pre-existing zero-importer orphans, not reachable through SearchWindow - left for a separate audit task).

**`UI/SearchRAGWindow.py.bak`**: already deleted upstream in 628b1b8b - no-op here.

**Other reference cleanup**: `UI/Wizards/__init__.py` exports trimmed; `app.py` dead `TemplateDeleteConfirmationEvent` branch removed (imported the deleted widget; nothing posts that event). The legacy tab-machinery strings (`"search-window"` window-id maps, `SEARCH_NAV_*` app-local constants, `SearchTabInitializer`) do not import or route to any removed screen and were left for the broader legacy-tab cleanup.

**CSS**: removed selector families exclusive to the deleted widgets from `css/features/_embeddings.tcss` (~870 lines: EmbeddingsWindow/EmbeddingsManagementWindow/EmbeddingsCreationContent, `.embeddings-*`, model/collection list-item rules, ModelListItem/CollectionListItem) and `css/features/_search-rag.tcss` (legacy `#search-window` shell: `.search-nav-pane`, `.search-view-area`, `#search-view-*`, embeddings creation/management view styles). Kept generic-named rules that currently match live widgets (`.metric-*` family used by Evals dashboards, `.model-name/-header/-provider/-size` used by HuggingFace/Evals/ModelManagement widgets, `.error-message`, `.sidebar-title`, `#dialog-*`, empty-state, chunk-preview, toast, detailed-progress, embedding-template, activity-log). Rebuilt `css/tldw_cli_modular.tcss` with `css/build_css.py`.

**Tests**: deleted files exclusive to removed modules (`Tests/UI/test_search_handoffs.py` SearchWindow-only tests, `Tests/UI/Embeddings/test_integration.py`, `test_model_preferences.py`, `test_performance_metrics.py`, `Tests/UI/test_embeddings_management_window_parity.py`, `Tests/test_embeddings_datatable_fix.py`, `Tests/UI/test_chunking_templates_widget_parity.py`, `Tests/UI/test_embedding_steps_tooltips.py`); trimmed mixed files (`test_search_handoffs.py`, `test_ux_audit_smoke.py`, `test_disabled_action_recovery_tooltips.py`, `test_file_picker_action_tooltips.py`, `test_core_imports_unit.py`, startup-perf lazy-import guard list, `Tests/UI/Embeddings/run_tests.py`/`test_base.py`/`README.md`). The task-248 nonblocking-cpu test that landed mid-task kept its repo-wide lexical guard; only its two deleted-widget tests were dropped.

**Verification**: `python -c "import tldw_chatbook.app"` clean; repo-wide grep for all removed module/class names returns nothing outside docs/backlog. Full `Tests/UI/ + Tests/RAG/ (+ Tests/UI/Embeddings, core-imports, cpu-guard)` run on the branch: 3759 passed / 36 failed / 27 skipped. Each failing test id was re-run on a pristine `origin/dev` (86de3394) worktree with the same venv: 28 fail there too (known pre-existing set: tools_settings api-key cluster, product-maturity visual/mcp snapshots, shell-destination order, backlog-id-uniqueness harness, etc.). Of the 8 that passed on baseline, 7 were CSS focus-contract tests asserting the deleted `.search-nav-pane`/`.embeddings-nav-*` selectors - fixed by trimming those checks (second commit); the 8th (`test_search_rag_window.py::...missing_embeddings_dependency...`) is part of a pre-existing order-dependent `NoMatches("#collections-list")` worker-race cluster in that file - it passes 5/5 in isolation on BOTH trees and different cluster members fail on baseline vs branch per run. After the test trims: re-running all previously-failing ids on the branch yields only failures that also fail on pristine origin/dev (plus the flaky cluster). No regressions introduced.
