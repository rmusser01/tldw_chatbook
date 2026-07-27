---
id: TASK-861
title: >-
  Sweep hardcoded ~/.config/tldw_cli and ~/.local/share/tldw_cli call sites onto
  the real accessors
status: To Do
assignee: []
created_date: '2026-07-27 04:35'
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
- [ ] #1 Every listed config-dir call site derives its parent directory from _get_effective_config_path().parent instead of a Path.home()/'.config'/'tldw_cli' literal
- [ ] #2 Every listed data-dir call site derives its path from get_user_data_dir() instead of a Path.home()/'.local'/'share'/'tldw_cli' literal that omits the user folder segment
- [ ] #3 The chatbook importer's extraction root matches the chatbook creator's temp root (both under get_user_data_dir()/temp/...) with a test asserting the two derive to the same parent
- [ ] #4 A test with TLDW_CONFIG_PATH pointed at a profile confirms at least one swept config-dir site (e.g. ui_state.toml) writes under that profile's directory, not the real ~/.config/tldw_cli
<!-- AC:END -->
