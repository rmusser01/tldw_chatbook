---
id: TASK-28006
title: Library media viewer - Generate analysis via the configured provider
status: Done
assignee: []
created_date: '2026-09-02 04:10'
updated_date: '2026-09-02 06:28'
labels:
  - library
  - media-ux
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-verified 2026-09-02 live on dev tip (worktree media-ux-fixes @ b7e89b6de, tmux scratch-profile run). Reframed: the "media" route is a live MediaScreen again (screen_registry.py:116) and its MediaWindow_v2 mounts the legacy generation workbench (MediaViewerPanel with #generate-analysis-btn), reachable from the Reader via More > Open manager. So analysis generation is NOT absent from the product - it is absent from the Library reading flow where sequential review happens: the Reader's Analysis tab is explicitly edit-only ("Analysis (re)generation via an LLM is explicitly out of scope", library_media_viewer.py 551-558; "Add analysis" opens an empty TextArea). Task: add a Generate action to the Reader's Analysis tab, resolving the provider through the existing resolve_ingest_analysis_provider seam (Library/ingest_analysis.py:135) and persisting through save_analysis_version, with the same promise/receipt honesty (an unready provider shows the reason). Leaving the reading context for the manager per item is not an acceptable review-flow answer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An item without an analysis offers a Generate action when an analysis provider is ready
- [x] #2 Generation persists the result as an analysis version visible in the viewer
- [x] #3 With no ready provider, the action communicates the same reason language as the ingest hint instead of silently failing
- [x] #4 Generation runs off the UI thread with visible progress
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Adds a Generate/Regenerate action to the Reader's Analysis tab (Widgets/Library/library_media_viewer.py _compose_analysis, wrapped with Edit in a ds-toolbar Horizontal). Handler handle_library_media_analysis_generate resolves the provider via the canonical resolve_ingest_analysis_provider seam (same as ingest, so promise/receipt cannot disagree): not-ready surfaces resolution.hint (the exact ingest language) via notify and does NOT dispatch; ready sets _library_media_generating_analysis, syncs (shows 'Generating analysis...'), and runs a worker. _dispatch_library_media_analysis calls chat_api_call off-thread (asyncio.to_thread) with the resolution's dispatch_name/api_key/model/sampling + api_key_resolved=True, and reuses extract_response_content (Chat_Functions) for the reply. Persists via the existing _save_library_media_analysis path (save_analysis_version + detail refresh). Design note: the button is always OFFERED (discoverable) and readiness is checked at press time with an honest reason (AC#3), rather than hidden when unready. Empty-content items get a 'no content to analyze' notice. Tests: test_library_media_generate_analysis_dispatches_and_persists, test_library_media_generate_analysis_without_provider_notifies_and_skips (patch resolve_ingest_analysis_provider + chat_api_call in the screen module). Files: library_media_viewer.py, library_screen.py, Tests/UI/test_library_shell.py.
<!-- SECTION:NOTES:END -->
