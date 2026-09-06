---
id: TASK-31745
title: 'Meetings: surface the speaker-rename legend in LibraryMediaViewer'
status: Done
assignee: []
created_date: '2026-09-06 07:40'
labels:
  - audio
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After-the-fact speaker rename on a finished meeting is DOM-correct and tested, but its legend lives in LibraryMediaCanvas's preview sub-pane which is display:none app-wide (since commit d99fb4a9c 'mount collapsible media reader shell'); the live media-reading surface is the separate LibraryMediaViewer widget (library_media_reader_shell.py). Port the meeting speaker-rename legend (can_rename_meeting_speakers + rename_meeting_speaker, already shipped) into LibraryMediaViewer so a user can actually rename speakers on a finished meeting item. Deferred from the phase-2 diarization SDD run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A user can rename meeting speakers from the live Library media reader (LibraryMediaViewer), not only the hidden canvas preview
<!-- AC:END -->

## Implementation Notes

The rename functions moved to `Library/meeting_speaker_rename.py` (re-exported from the canvas) and the tested speaker legend is mounted in `LibraryMediaViewer` (height-auto, bounded to 12 rows), with an item-scoped `SpeakerRenamed` refresh that clears the viewer-state memo. The content guard accepts both app-produced transcript shapes (plain and Markdown, including the pre-escape legacy Markdown) and rewrites in the matching shape; ingest-produced content is refused with a one-line explanation, as decided. A failed rename restores the prior `meeting.json` so retries succeed. Landed in PR #2471 (commits 823d961cf, d25190898, 0ec964a4f, 8f01a6bd3, 2491787bc, d43e7ccc1, ddbcd1b3d).
