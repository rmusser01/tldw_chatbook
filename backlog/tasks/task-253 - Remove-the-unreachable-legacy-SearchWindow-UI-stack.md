---
id: TASK-253
title: Remove the unreachable legacy SearchWindow UI stack
status: In Progress
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
- [ ] #1 UI/SearchWindow.py and screens/widgets reachable only through it (SearchEmbeddingsWindow, Embeddings_Management_Window, embeddings wizards, chunking-template widgets) are removed along with SearchRAGWindow.py.bak
- [ ] #2 No route registry or import references to the removed screens remain
- [ ] #3 Full test suite passes after removal
<!-- AC:END -->

## Implementation Plan

1. Verify importers of every candidate module (git grep for module paths AND class names). Confirmed dead stack rooted at UI/SearchWindow.py (no production importers) plus modules reachable only through it.
2. Delete production modules: UI/SearchWindow.py, UI/SearchEmbeddingsWindow.py, UI/Embeddings_Window.py (shim), UI/Embeddings_Creation_Content.py (alias shim), UI/Embeddings_Management_Window.py, UI/Wizards/EmbeddingsWizard.py, UI/Wizards/EmbeddingSteps.py (only importer: EmbeddingsWizard), Widgets/chunking_templates_widget.py, Widgets/chunking_template_editor.py (only importer: the templates widget), plus secondary orphans whose sole importer was Embeddings_Management_Window: Widgets/performance_metrics.py, Widgets/embeddings_list_items.py, Utils/model_preferences.py. Delete Event_Handlers/search_events.py (its only job is delegating button events to the removed "#search-window").
3. UI/SearchRAGWindow.py.bak is already absent on origin/dev (removed previously); note as no-op.
4. Reference cleanup: UI/Wizards/__init__.py (drop EmbeddingsWizard/EmbeddingSteps exports), Event_Handlers/event_dispatcher.py + app.py (drop search_events import/spread), app.py TemplateDeleteConfirmationEvent branch (imports the deleted chunking widget; nothing posts that event).
5. CSS: surgically remove selectors exclusive to the deleted stack from css/features/_embeddings.tcss and css/features/_search-rag.tcss; keep selector families used by live widgets (empty-state, chunk-preview, toast, detailed-progress, embedding-template-selector, activity-log, dialog, metric-*, model-name/header/provider/size, error-message, sidebar-title). Rebuild css/tldw_cli_modular.tcss via css/build_css.py.
6. Tests: delete test files exclusive to removed modules; edit mixed files (test_search_handoffs.py, test_ux_audit_smoke.py, test_disabled_action_recovery_tooltips.py, test_file_picker_action_tooltips.py, test_core_imports_unit.py, test_app_startup_performance.py) to drop only the removed-module tests.
7. Verify: `python -c "import tldw_chatbook.app"` smoke; pytest Tests/UI Tests/RAG diffed against a clean origin/dev baseline run; grep tree for lingering references.
