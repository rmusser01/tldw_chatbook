---
id: TASK-28006
title: Library media viewer - Generate analysis via the configured provider
status: To Do
assignee: []
created_date: '2026-09-02 04:10'
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
- [ ] #1 An item without an analysis offers a Generate action when an analysis provider is ready
- [ ] #2 Generation persists the result as an analysis version visible in the viewer
- [ ] #3 With no ready provider, the action communicates the same reason language as the ingest hint instead of silently failing
- [ ] #4 Generation runs off the UI thread with visible progress
<!-- AC:END -->
