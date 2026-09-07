---
id: TASK-229
title: Repair dotted-section get_cli_setting("chat.images") readers
status: Done
assignee:
  - '@claude'
created_date: '2026-07-14 10:30'
updated_date: '2026-07-16 14:34'
labels:
  - config
  - bug
  - chat
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-222's final review proved that `get_cli_setting(section, key, default)` performs a FLAT `config.get(section)` lookup and never resolves nested TOML tables, so every existing call using the dotted-section shape `get_cli_setting("chat.images", key, default)` silently returns its default. Known broken callers on dev (pre-existing, from #621/#626 — probe-verified): `Chat_Window_Enhanced.py:268` and `chat_session.py:107` plus both settings sidebars (`show_attach_button` — the attach button visibility toggle has never worked from config), and `chat_screen.py` `save_location` (Save Image always falls back to ~/Downloads regardless of config). TASK-222 fixed its own reads via `attachment_core._chat_images_setting` (fetch the section via `get_cli_setting("chat", "images", None)`, resolve the key locally) and pinned the real path with an unmocked integration test. This task repairs the remaining callers — either by migrating them to the same nested-read helper, or by teaching `get_cli_setting` itself to fall back to nested navigation for dotted sections (which needs its own review of every dotted caller repo-wide, since behavior changes from always-default to config-honoring). Audit repo-wide for other dotted-section calls while at it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 show_attach_button and save_location honor the live config values end-to-end (verified through the real loader, no mocked accessors)
- [x] #2 No production caller passes a dotted section to a flat get_cli_setting lookup (repo-wide audit; either callers migrate or the accessor gains nested resolution with all affected callers reviewed)
- [x] #3 At least one unmocked integration test per repaired reader pins the real config path (scratch TOML through the real loader)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: unmocked real-loader tests (TLDW_CONFIG_PATH + force_reload, zero accessor mocks) for every broken caller tuple: (chat.images, show_attach_button/save_location), (chat.voice, show_mic_button), (mcp.hub_state, advanced_open), three-level prompts.document_generation.* dict form; + working-shape regressions (traditional, 1-arg single-dot, dotted+default) and flat-key-shadowing precedence\n2. Fix get_cli_setting: nested-path fallback after the flat lookup misses (dotted section OR dotted key navigates the real TOML tree); flat hit always wins\n3. Consumer-path test: DocumentGenerator picks up a configured [prompts.document_generation.timeline] prompt through the real loader (in-memory DB)\n4. Behavior-flip review of all repaired callers (defaults match template values → no visible change on default configs)\n5. Sweep + repo audit re-run; PR
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed the ACCESSOR rather than migrating callers: get_cli_setting now falls back to nested-tree navigation when the flat lookup misses and the section/key carries dots (flat hit always wins — bit-for-bit preservation of every working shape, incl. the 1-arg single-dot and dotted+non-str-default forms). All 11 broken callers repaired at once: show_attach_button (Chat_Window_Enhanced, chat_session, both sidebars), show_mic_button (Chat_Window_Enhanced, settings_sidebar), save_location (chat_screen Save Image), [mcp.hub_state] advanced_open (mcp_inspector), and document_generator's three-level prompts.document_generation.{timeline,study_guide,briefing} dict-default form (audit found these additionally broken: the first-dot split left a dotted KEY that flat-missed). Behavior flips all intent-restoring; template/live defaults equal code defaults so default configs see no change. RED-first unmocked tests (Tests/Utils/test_config_nested_settings.py, 12) drive the REAL loader (TLDW_CONFIG_PATH + force_reload, zero accessor mocks — the T222 C1 lesson) over every repaired tuple + working-shape regressions + flat-shadow precedence + a DocumentGenerator consumer path. Rider fix discovered by the sweep: production reset_dependency_checks() restored DEPENDENCIES_AVAILABLE from a stale duplicated literal, silently dropping newer keys (svg_rendering from T222) — single-sourced via pristine module-level copy + cached-probe reset + test-pollution hygiene (snapshot/finally in test_initialize_dependency_checks; one sanctioned existing-test edit). Sweep: 1167 passed / 95 skipped / 0 failed across Utils+Chat+Event_Handlers+DB+unit. attachment_core._chat_images_setting left as-is (works either way; consolidation optional later). Files: config.py, Utils/optional_deps.py, Tests/Utils/test_config_nested_settings.py, Tests/Utils/test_optional_deps.py.
<!-- SECTION:NOTES:END -->
