---
id: TASK-38723
title: "Delete the architecture-review compatibility shims once nothing imports them"
status: To Do
created_date: 2026-11-04 02:52
labels:
- follow-up
- architecture-review-2026-10-25
priority: low
references:
- tldw_chatbook/UI/Screens
- tldw_chatbook/UI/character_display_text.py
- tldw_chatbook/config.py
- tldw_chatbook/Provider_Contract
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 2 of the 2026-10-25 architecture review (merged 2026-11-03, dev d204ddce70) moved modules and names but kept their old paths working so that branches written against the old paths would not break. Each shim is a second name for one object, and patches or reads through the wrong name are easy to get wrong. Once the in-repo importers are gone, the shims are only cost.

Shims left by Phase 2:
- UI/Screens alias modules for the nine modules TASK-36944.8 moved: settings_speech_tts, settings_config_models, settings_advanced_config, settings_config_adapter, settings_image_gen_defaults, settings_video_gen_defaults, settings_web_search, model_browser_state and provider_model_resolution.
- UI/character_display_text.py, a star re-export (TASK-36944.6).
- config.py's lazy `__getattr__` re-exports of the 27 Canvas and Console capture-policy names (TASK-36945.2). The Backup_Recovery and startup-deferral subprocess tests use them on purpose as compat coverage.
- Chat.* sys.modules aliases of the Provider_Contract modules (TASK-36944.1).
- Re-exports in console_scope_picker_modal (TASK-36944.5), chat_approval_card (TASK-36944.3), console_inspector_section (TASK-36944.4), tool_catalog for PathAwareToolProvider and ToolPathTarget, and console_project_instructions for EPHEMERAL_ORIGIN_KEY (TASK-36944.10).
- Private `_x = x` aliases in the Backup_Recovery participants modules (TASK-36948.2), agent_service and run_log (TASK-36944.2).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every shim listed in the description is deleted, or the task notes name the in-repo importer or patch target that still needs it
- [ ] #2 No test or production module imports, patches or reads a moved name through a deleted path
- [ ] #3 Tests/Architecture (serial) and preflight show no new failures against dev
<!-- AC:END -->
