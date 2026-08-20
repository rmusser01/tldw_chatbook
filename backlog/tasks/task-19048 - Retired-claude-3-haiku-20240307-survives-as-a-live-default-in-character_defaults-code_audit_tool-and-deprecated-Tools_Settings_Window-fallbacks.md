---
id: TASK-19048
title: >-
  Retired claude-3-haiku-20240307 survives as a live default in
  character_defaults, code_audit_tool, and deprecated Tools_Settings_Window
  fallbacks
status: To Do
assignee: []
created_date: '2026-08-20 15:46'
labels:
  - llm
  - bug
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-19020 replaced the retired claude-3-haiku-20240307 fallback in summarize_with_anthropic (the API now answers 404 not_found_error for that id, probe req_011CeEDXZ8iS29MZCgyySwQa) and swept the repo for the remaining occurrences. Three sites still use the retired id as an actual request model and were left for this follow-up: (1) config.py:3841 -- the shipped CONFIG_TOML template's [character_defaults] model default, so a fresh install's persona/character calls target a 404 model (the rp-ux QA report already observed there is no UI to change the character provider); (2) Tools/code_audit_tool.py:141 -- hardcoded model for the audit tool's live analysis call; (3) UI/Tools_Settings_Window.py:1044/2456/4116/4851/5647 -- fallback and reset values in the DEPRECATED (TASK-1346), nav-unreachable ToolsSettingsWindow (lowest priority; may be resolved by deleting the window instead). Metadata-only occurrences need no change here: capability/context-window maps (model_capabilities.py:104, config.py:4133, Chat/console_session_settings.py:89, Utils/token_counter.py:344) merely describe the id, and the providers dropdown lists (config.py:2805/3178) are already tracked by task-3600. Pick replacements consistent with TASK-19020's choice (claude-haiku-4-5 as the served successor in the haiku lineage) unless a site's intent argues otherwise.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The shipped [character_defaults] template default names a currently-served Anthropic model
- [ ] #2 code_audit_tool's hardcoded analysis model is a currently-served model
- [ ] #3 The deprecated Tools_Settings_Window fallbacks are updated or the surface's deletion is confirmed as covering them
<!-- AC:END -->
