---
id: TASK-702
title: Library ingest reports a YouTube URL as an unsupported file
status: To Do
assignee: []
created_date: '2026-07-26 14:12'
labels:
  - library
  - ingest
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pasting a video URL into the Library ingest canvas reports it as an unsupported file, even though the ingest pipeline classifies the same URL as video and can ingest it. The canvas decides a source's type by file extension alone, so any URL without a recognisable extension falls through to the unsupported bucket. Video links are one of the most common things a user wants to import, so this makes the canvas look broken for a mainstream case.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A video-hosting URL is recognised as audio/video by the ingest canvas,A URL without a file extension is not reported as an unsupported file when the pipeline can ingest it,The canvas's classification of a source agrees with what the ingest pipeline will actually do
<!-- AC:END -->
