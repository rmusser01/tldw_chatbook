---
id: TASK-865
title: >-
  Sweep hardcoded ~/.config/tldw_cli and ~/.local/share/tldw_cli call sites onto
  the real accessors
status: In Progress
assignee:
  - '@claude'
created_date: '2026-07-27 04:35'
updated_date: '2026-08-01 09:37'
labels:
  - security
  - config
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Around 30 call sites build config/state paths from Path.home()/".config"/"tldw_cli"/... or Path.home()/".local"/"share"/"tldw_cli"/... directly instead of using _get_effective_config_path().parent or get_user_data_dir(). The config-dir group (UI/Screens/chat_screen.py:15394,15425 for ui_state.toml; Event_Handlers/notes_events.py:144 and note_ingest_events.py:350 for note_templates.json; Subscriptions/website_monitor.py:72 for feed_cache/, plus ~25 lower-value sites) silently ignores TLDW_CONFIG_PATH -- these files land in the real ~/.config/tldw_cli regardless of which profile is active.

The data-dir group (~18 sites, including Chatbooks/chatbook_importer.py:77-79, Chatbooks/local_chatbook_service.py:102-107, Character_Chat/Character_Chat_Lib.py:1274,2790,3856, Event_Handlers/conv_char_events.py:4152,4213,4264) additionally omits the <user_folder> segment that get_user_data_dir() appends, so multi-user profiles collide into the same directory. The chatbook importer is the highest-value fix in this group because the drift is already live in production, not latent: a reproduction showed chatbook_importer.py:77's literal ~/.local/share/tldw_cli/temp/imports already exists on disk, while the derived, correct .../default_user/temp/imports (matching chatbook_creator.py:97's sibling path) does not -- meaning imports have been extracting to a path outside the per-user tree that any other local user account, or a future multi-profile user, would also read and write.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every swept hardcoded profile-owned config-dir occurrence in ADR-040's normative inventory derives its parent directory from get_cli_config_path().parent instead of a Path.home()/'.config'/'tldw_cli' literal
- [ ] #2 Every swept hardcoded active-user data-dir occurrence in ADR-040's normative inventory derives its path from get_user_data_dir() instead of a Path.home()/'.local'/'share'/'tldw_cli' literal that omits the user folder segment
- [x] #3 The chatbook importer's extraction root matches the chatbook creator's temp root (both under get_user_data_dir()/temp/...) with a test asserting the two derive to the same parent
- [x] #4 A test with TLDW_CONFIG_PATH pointed at a profile confirms at least one swept config-dir site (e.g. ui_state.toml) writes under that profile's directory, not the real ~/.config/tldw_cli
- [ ] #5 Every remaining executable literal is classified by an exact sentinel exception as inert configuration data, a canonical resolver/default seed, a compatibility constant, a shared artifact, or a read-only legacy probe
- [ ] #6 The rejected transcription-history store/viewer, unmounted legacy Dictation window, and unused legacy user-database path helper are retired rather than allowlisted
- [ ] #7 No existing global file is copied, moved, imported, or deleted by the completion tranche
- [ ] #8 Regression tests use production functions or the full TldwCli application; no reduced test application is introduced
- [ ] #9 Swept profile-owned state writers use ADR-029 private atomic replacement and preserve the previous file when serialization or replacement fails
- [ ] #10 Generated diagnostics/SQLite inventories, affected legacy-window tests/source censuses, stale feature documentation, release notes, and installed-wheel coverage agree with the retired modules and symbols
<!-- AC:END -->

## ADR Check

ADR required: yes

ADR path: [ADR-040: Profile-Owned State and Shared Asset Paths](../decisions/040-profile-owned-state-and-shared-asset-paths.md)

Reason: The completion tranche classifies persistent data ownership, profile
isolation, shared artifacts, legacy probes, and migration behavior across
multiple modules.

Design: [TASK-865 Profile-Owned Path Completion Design](../../Docs/superpowers/specs/2026-08-01-profile-owned-path-completion-design.md)

Completion-scope note: existing consumers of `_get_effective_config_path()`
already resolve the active config correctly and are not part of this
hardcoded-literal sweep. New or changed consumers use the public
`get_cli_config_path()` wrapper. This tranche does not modify Notes Sync.

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Use ADR-040 to classify every remaining executable literal as profile-owned config state, active-user data, shared artifact, inert default, read-only legacy probe, or unreachable code.
2. Write failing function/full-application path-isolation tests and an executable-token/source sentinel that detects embedded, multiline, indirect-join, duplicate, and stale cases with exact counted exceptions.
3. Retire every rejected transcript-history implementation, including the unmounted legacy Dictation window, plus the unused legacy user-database path helper; reconcile every importing test/source census, generated/curated inventory, compatibility comment, current architecture document, and release note without rewriting historical Backlog records.
4. Apply the design's normative disposition inventory: move each swept profile-owned config/data occurrence onto get_cli_config_path().parent or get_user_data_dir() at the call boundary without migrating existing files, preserve classified exceptions, and route swept private state writers through ADR-029 atomic replacement.
5. Run targeted ownership/privacy/inventory/installed-wheel suites, the full suite, and static checks; then reconcile the acceptance criteria, implementation notes, and task status.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed, with tests, every site the task named with an explicit file:line reference: config-dir group -- UI/Screens/chat_screen.py (both ui_state.toml sites), Event_Handlers/notes_events.py and note_ingest_events.py (note_templates.json), Subscriptions/website_monitor.py (feed_cache/); data-dir group -- Chatbooks/chatbook_importer.py (the highest-value fix: temp_dir now derives get_user_data_dir()/'temp'/'imports', matching chatbook_creator.py's sibling get_user_data_dir()/'temp'/'chatbooks'), Chatbooks/local_chatbook_service.py's _default_registry_path() fallback, Character_Chat/Character_Chat_Lib.py (all 3 base_directory sites), Event_Handlers/conv_char_events.py (all 3 export_dir sites). All now derive from _get_effective_config_path().parent or get_user_data_dir() via local imports, matching the codebase's established per-call-site import convention.

As a best-effort addition beyond the explicitly named sites, also fixed: Widgets/emoji_picker.py (converted the eager RECENT_EMOJIS_FILE module constant to a lazy _recent_emojis_path() -- an eager get_user_data_dir()-based constant would itself go stale across a profile switch within one process, the same latent-staleness class TASK-855 closed for the MCP stores), Widgets/settings_theme_editor.py (custom_themes_path), Notes/sync_service.py (sync_profiles.json default), Config_Files/create_custom_template.py (a standalone, non-imported dev helper script), RAG_Search/pipeline_loader.py and pipeline_builder_simple.py (rag_pipelines.toml user-override lookup).

NOT checking AC #1/#2 as fully satisfied: the task also references '~25 lower-value' config-dir sites and '~18' data-dir sites only in aggregate, without file:line references, and a full sweep was not completed given the size of that remainder. A broader grep during this task surfaced a SEPARATE, adjacent, and larger finding worth its own follow-up task: UI/ChatbookCreationWindow.py, UI/ChatbookExportManagementWindow.py, UI/Wizards/ChatbookCreationWizard.py and UI/Wizards/ChatbookImportWizard.py each build their OWN ad-hoc db_paths dict straight from self.app.config_data.get('database', {}) with hardcoded, WRONG, non-user-folder fallback literals (e.g. '~/.local/share/tldw_cli/tldw_prompts_db.db') instead of calling get_prompts_db_path()/get_media_db_path() -- these do NOT go through the get_*_db_path() accessors at all, unlike everything reconciled in TASK-858/899. Deliberately left out of this task's scope (a distinct defect class, not a Path.home()-literal sweep site) and recommend filing separately. Also deliberately left OUT as a scope decision, not an oversight: TTS/UI model-weight and voice-cache paths under ~/.config/tldw_cli/models/... and .../*_voices (STTS_Window.py, Dictation_Window.py, TTS/backends/kokoro.py, TTS/kokoro_pytorch.py, TTS/utils/download_models.py) -- these are large, shared binary caches/exports, not per-profile config/state; making them profile-relative would force re-downloading multi-hundred-MB models on every profile switch, which is very unlikely to be the intended behavior and is exactly the kind of live-file-relocation this task's hard constraints told me to stop and report on rather than silently do.

AC #3 (chatbook importer/creator parity) and AC #4 (a profile-retargeted config-dir site) are both satisfied with concrete tests: Tests/Chatbooks/test_chatbook_importer.py (temp_dir == get_user_data_dir()/'temp'/'imports', and shares a parent with ChatbookCreator's temp_dir), Tests/UI/test_chat_screen_ui_state_path.py (TLDW_CONFIG_PATH retargeted to a scratch profile -> _save_sidebar_state() writes ui_state.toml under THAT profile's directory, two different profiles do not collide). Files: tldw_chatbook/UI/Screens/chat_screen.py, Event_Handlers/{notes_events,note_ingest_events,conv_char_events}.py, Subscriptions/website_monitor.py, Chatbooks/{chatbook_importer,local_chatbook_service}.py, Character_Chat/Character_Chat_Lib.py, Widgets/{emoji_picker,settings_theme_editor}.py, Notes/sync_service.py, Config_Files/create_custom_template.py, RAG_Search/{pipeline_loader,pipeline_builder_simple}.py; new tests in Tests/Chatbooks/test_chatbook_importer.py and Tests/UI/test_chat_screen_ui_state_path.py.

The revised completion design supersedes the earlier partial-note decision to retain `UI/Dictation_Window.py`: TASK-1331 rejected transcript persistence and current production has no importer, so the completion tranche retires that entire legacy window while preserving `ImprovedDictationWindow` coverage. These partial notes remain historical progress notes and will be replaced by final implementation notes only after the remaining acceptance criteria are verified.
<!-- SECTION:NOTES:END -->
