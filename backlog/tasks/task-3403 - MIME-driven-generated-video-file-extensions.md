---
id: TASK-3403
title: MIME-driven generated-video file extensions
status: In Progress
assignee: []
created_date: '2026-08-09 04:39'
updated_date: '2026-08-12 02:41'
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
- [ ] #1 Supported generated-video MIME/container values map to a single safe canonical filename extension.
- [ ] #2 Video storage derives the filename extension from validated result metadata instead of assuming MP4.
- [ ] #3 Unknown, contradictory, or unsupported MIME/container results fail before bytes are persisted.
- [ ] #4 Existing message-name resolution, retention, eviction, tombstone, and save-copy behavior remains correct for every supported extension.
- [ ] #5 Focused validation and VideoStore tests cover MP4 and at least one non-MP4 supported container.
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
