---
id: TASK-3403
title: MIME-driven generated-video file extensions
status: Done
assignee: []
created_date: '2026-08-09 04:39'
updated_date: '2026-08-12 05:18'
labels:
  - video
  - generation
  - storage
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Correct the generated-video storage boundary so validated result MIME/container data determines the stored filename extension across providers. This task is independent of workflow packaging and image-generation work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Supported generated-video MIME/container values map to a single safe canonical filename extension.
- [x] #2 Video storage derives the filename extension from validated result metadata instead of assuming MP4.
- [x] #3 Unknown, contradictory, or unsupported MIME/container results fail before bytes are persisted.
- [x] #4 Existing message-name resolution, retention, eviction, tombstone, and save-copy behavior remains correct for every supported extension.
- [x] #5 Focused validation and VideoStore tests cover MP4 and at least one non-MP4 supported container.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add the immutable MP4/WebM mapping, update both current adapters, and TDD the worker plus outer no-persistence boundary in one runnable commit.
2. Persist canonical container metadata and thread explicit extensions through Console generation, pending recovery, regeneration, and external save while VideoStore compatibility defaults remain temporarily.
3. Migrate every production card/playback/save-copy/reload/remount reader to metadata-derived extensions and cover the ProductionApp composition path.
4. Remove VideoStore MP4 defaults only after caller migration; update all focused direct consumers and preserve complete retention/capacity inventory.
5. Amend ADR-044 as revision 3, run only exact touched-file tests/static/privacy/artifact gates, independently review, update notes, and close the task.

ADR required: no new ADR
ADR path: backlog/decisions/044-ephemeral-generated-video-storage-playback-and-streaming.md
Reason: ADR-044 already owns generated-video provider, ephemeral-storage, and metadata boundaries; this task amends its MP4-specific wording and metadata inventory without introducing a new architectural boundary.

Detailed plan: Docs/superpowers/plans/2026-08-12-mime-driven-generated-video-extensions.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Implemented one immutable exact MP4/WebM container, MIME, and extension vocabulary; adapters report observed containers and the worker rejects unknown or contradictory request/container/MIME facts before persistence or temporary staging. Container metadata now drives managed save, pending recovery, regeneration, playback, tombstones, reload/remount resolution, and save-copy suffixes. VideoStore requires explicit canonical extensions, preserves complete unknown-file retention/capacity accounting, and serializes cross-format slug publication.
- Tradeoffs: intentionally no generalized format registry, MIME guessing, probing, remuxing, transcoding, provider expansion, schema migration, or format UI. Historical metadata with no container remains MP4-compatible; present invalid metadata fails closed. ADR-044 revision 3 records the validated-extension and durable-container decisions.
- Production paths: tldw_chatbook/Video_Generation/video_formats.py; adapters/base.py; worker.py; adapters/minimax_video_adapter.py; adapters/comfyui_video_adapter.py; video_metadata.py; video_store.py; tldw_chatbook/Chat/console_generate_video.py; tldw_chatbook/UI/Screens/chat_screen.py. Test paths: Tests/Video_Generation/test_video_formats.py, test_contracts.py, test_worker.py, test_minimax_adapter.py, test_comfyui_adapter.py, test_video_metadata.py, test_video_store.py; Tests/Chat/test_console_generate_video.py, test_console_video_capacity.py, test_console_video_actions.py, test_console_video_message.py; Tests/ProductionApp/test_chat_composition_retirement.py.
- Final focused evidence: exact 12-file gate 454 passed, 2 Windows-only skips, 1 existing RequestsDependencyWarning. Exact changed-file Ruff gate and chat_screen.py E9/F63/F7/F82 gate passed. Temporary-output py_compile and git diff --check passed. Privacy, committed/uncommitted media-build artifact, and Image_Generation production-change searches were clean with expected no-match results. Per approved plan, no live provider call, full suite, or unrelated RuntimePolicy gate was run.
- Implementation commits: c569aed906bd42c6edc0b53673eac970a8c82adb; fe4cfa2aa01283c99989206f7620a2114f43c4ec; 6cb1911528066797e49a7d874361bbf05e3c26d5; 5ad532dc4c6f52e9d5e0dd3fb373c2e7851c70e4; a1bc91f6a1ee3baa878ca7aa130f5e085d6e17ec; 1f3de9c4cee3ccf0c280248b3558d4b377d24aa4; 8cc4d844743de7689dd70d21d3448cf2a96f0290; f50817494bcafe7d8229d113d7159b5e262821af; 0eaa44a5335341200251f516abc2098d306c6c60. Independent whole code/spec and quality reviews approved the final fix with no findings.
- Plan deviation: the four planned implementation phases landed as nine focused commits because review added hostile-result hardening, WebM recovery coverage, transactional cross-format publication, and rejected-adoption rewind fixes; scope and architecture were unchanged. The allocation/publication race incident is already covered by the existing check-and-commit linearizability lesson, so no new lesson edit was needed.
<!-- SECTION:NOTES:END -->
