---
id: TASK-229
title: Repair dotted-section get_cli_setting("chat.images") readers
status: In Progress
assignee:
  - '@claude'
created_date: '2026-07-14 10:30'
updated_date: '2026-07-16 14:11'
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
- [ ] #1 show_attach_button and save_location honor the live config values end-to-end (verified through the real loader, no mocked accessors)
- [ ] #2 No production caller passes a dotted section to a flat get_cli_setting lookup (repo-wide audit; either callers migrate or the accessor gains nested resolution with all affected callers reviewed)
- [ ] #3 At least one unmocked integration test per repaired reader pins the real config path (scratch TOML through the real loader)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: unmocked real-loader tests (TLDW_CONFIG_PATH + force_reload, zero accessor mocks) for every broken caller tuple: (chat.images, show_attach_button/save_location), (chat.voice, show_mic_button), (mcp.hub_state, advanced_open), three-level prompts.document_generation.* dict form; + working-shape regressions (traditional, 1-arg single-dot, dotted+default) and flat-key-shadowing precedence\n2. Fix get_cli_setting: nested-path fallback after the flat lookup misses (dotted section OR dotted key navigates the real TOML tree); flat hit always wins\n3. Consumer-path test: DocumentGenerator picks up a configured [prompts.document_generation.timeline] prompt through the real loader (in-memory DB)\n4. Behavior-flip review of all repaired callers (defaults match template values → no visible change on default configs)\n5. Sweep + repo audit re-run; PR
<!-- SECTION:PLAN:END -->
