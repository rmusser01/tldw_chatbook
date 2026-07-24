---
id: TASK-503
title: >-
  RAG settings + profiles SP3: RAG settings screen (profile manager + full editor)
status: To Do
assignee: []
created_date: '2026-07-23 21:00'
updated_date: '2026-07-23 21:00'
labels:
  - rag
  - profiles
  - settings
dependencies:
  - task-483
  - task-502
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Final sub-project of the RAG settings screen + profiles program (overview: Docs/superpowers/specs/2026-07-21-rag-settings-profiles-overview.md; SP3 spec: Docs/superpowers/specs/2026-07-21-rag-settings-screen-design.md; plan: Docs/superpowers/plans/2026-07-23-rag-settings-screen-sp3.md). Makes the profile system user-facing: the "Library & RAG" settings category becomes "RAG", editing the ACTIVE PROFILE (the engine's real config source after SP2b) instead of the dead AppRAGSearchConfig.rag.* keys. Depends on SP2a (task-483, PR #780) and SP2b (task-502, PR #795), both merged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] The RAG category loads from and saves to the active profile file (via ConfigProfileManager); no AppRAGSearchConfig.rag.* writes remain (dead builders retired).
- [x] Profile-manager region: builtins-vs-user grouped list with active marker; set-active (off-thread, dirty-draft Save/Discard prompt), clone, rename, delete; builtin active → read-only editor + "Clone to edit".
- [x] Editor covers every ProfileConfig section in collapsible groups (search + embedding + chunking + vector-store + reranking, ~21 fields); index-determining fields marked; rerank toggle controls reranking_config presence and provably reaches the service flag (integration test).
- [x] RAGConfig.validate() wired into category validation (first caller) plus rerank checks.
- [x] Index status readout (built/empty/absent + provenance) with Backfill action; "new empty index — Backfill" warning on BOTH set-active and index-field save.
- [x] QA captures produced (pilot-rig with real bundle CSS); user screen approval gates the PR.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
SDD-executed (5 tasks, fix waves per review, final opus whole-branch review "with fixes" → all fixed + re-review clean). New headless seam `UI/Screens/settings_rag_profile_adapter.py` (load/apply/save + builtin refusal + profile CRUD wrappers + severity-split validation `hard_config_errors`/`soft_config_warnings` = `RAGConfig.validate()`'s FIRST live caller + `index_change_pending`/`fetch_index_status`). Screen: Profiles region (set-active/clone/rename/delete; dirty-draft Save/Discard/Cancel via new modals; delete-active falls back to hybrid_basic), Collapsible editor groups (~21 fields, ⚠ re-index markers), rerank toggle = `reranking_config` PRESENCE (the factory's real flag, integration-proven; top-k default corrected 5→20), index status row + thread-isolated Backfill (`asyncio.run` on a thread worker, pre-resolved service) + honest re-index warnings (absent/empty only; distinct unknown notice).

Notable catches: pending-activate leak past validation early-returns; ValueError-only wrappers vs worker exit_on_error app-crash; validator tests escaping the sandbox to the real `~/.local` (get_user_data_dir import-time freeze — root-cause follow-up task-519); QA captures caught TWO real visual bugs — repo CSS styled a DEAD `.collapsible--header` class (Textual 8.2.7 titles are CollapsibleTitle widgets → focused titles invisible; fix split: focus-visibility global, decorative restyle scoped to the RAG card with a scoped :focus restatement per Textual's pseudo-class-in-class-tier specificity) and a 12-col label clip making "Reranker model"/"Reranker top-k" identical.

Retired: `build_library_rag_save_sections`/`_rag_section` + obsolete tests (incl. the pre-existing-failing AppRAGSearchConfig read-path test). Follow-ups: mutate-then-persist on failed save, memory-store always-warns on set-active, dead chat-rag-panel css rules, editor long-tail fields (cache/parent/api-key). Files: settings_rag_profile_adapter.py (new), settings_screen.py, settings_library_rag_defaults.py, css/_widgets.tcss + _agentic_terminal.tcss + bundle, Tests/UI/test_settings_rag_profile_{adapter,region}.py, QA captures + rig in Docs/superpowers/qa/rag-settings-sp3-2026-07/.
<!-- SECTION:NOTES:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-07-23-rag-settings-screen-sp3.md — 5 TDD tasks: (1) adapter seam load/save→active profile + retitle; (2) profile-manager region; (3) extended editor groups + validation + rerank presence; (4) index status/backfill/warnings; (5) dead-code retirement + regression + QA captures. New headless module settings_rag_profile_adapter.py. SDD-executed, user-gated PR.
<!-- SECTION:PLAN:END -->
