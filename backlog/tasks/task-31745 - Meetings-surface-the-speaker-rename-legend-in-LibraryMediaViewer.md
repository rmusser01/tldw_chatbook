---
id: TASK-31745
title: 'Meetings: surface the speaker-rename legend in LibraryMediaViewer'
status: To Do
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
- [ ] #1 A user can rename meeting speakers from the live Library media reader (LibraryMediaViewer), not only the hidden canvas preview
<!-- AC:END -->
