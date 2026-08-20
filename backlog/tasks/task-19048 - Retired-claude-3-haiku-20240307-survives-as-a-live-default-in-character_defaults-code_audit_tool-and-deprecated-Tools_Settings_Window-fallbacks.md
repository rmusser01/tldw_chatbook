---
id: TASK-19048
title: >-
  Retired claude-3-haiku-20240307 survives as a live default in
  character_defaults, code_audit_tool, and deprecated Tools_Settings_Window
  fallbacks
status: In Progress
assignee:
  - '@claude'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify the metadata-only exclusion list with a repo-wide grep on the branch base (f79aa9082); confirm no new request-path site appeared since the 19020 sweep.
2. Decide the Tools_Settings_Window question with evidence: check task-1346's state and the board for a deletion task. (Found: 1346 is Done and explicitly scoped the window with a deprecation banner INSTEAD of deleting — deletion was "not trivially safe"; no deletion task exists. So update the 5 literals.)
3. Red-first pins: template parse pin in Tests/test_config_model_catalog_defaults.py ([character_defaults] model is claude-haiku-4-5 and present in the template's own [providers].Anthropic list, per the adjacent comment); code_audit wire-model pin in Tests/Tools/test_code_audit_repoint.py (captured model == claude-haiku-4-5). Run red, READ counts.
4. Fix: config.py:3841 and code_audit_tool.py:141 -> claude-haiku-4-5 (haiku-lineage successor per 19020 precedent; both sites' cheap-fast intent matches). Tools_Settings_Window character sites (1044/2456/4116/5647) -> claude-haiku-4-5 (they mirror [character_defaults]); site 4851 -> claude-sonnet-5 (it resets the [api_settings.anthropic] form, whose shipped template default is claude-sonnet-5 — the haiku lineage would diverge from the canonical default this reset restores). Update the stale hardcoded-model mention in Docs/Development/Agent-Tools/Claude_Code_File_Audit_System.md.
5. Green + targeted gate with READ counts: the two touched test files + Tests/UI/test_tools_settings_window.py + Tests/test_probe_import_provenance.py. No live-API leg: the 404 is on record (req_011CeEDXZ8iS29MZCgyySwQa) and claude-haiku-4-5/claude-sonnet-5 servedness was live-verified by 19020's runs (rows 1, 3, 4).
6. Hygiene: tick ACs, Implementation Notes, Done; report; PR against dev.
<!-- SECTION:PLAN:END -->
